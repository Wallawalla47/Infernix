// ExpertStream's gated chunks with copy-ahead (runtime/expert_offload/expert_stream.h, candidates BX-09):
// plan_ahead fills the start of a layer's half from a guess, plan_layer keeps those slots and copies only
// the rest after them, mask_ahead hides the copied-ahead experts from the CPU split, and every slot the
// table names holds that expert's record bytes. The speed effect is a measurement (the copy-ahead ABBA).

#include "core/arena.h"
#include "core/device.h"
#include "runtime/expert_offload/expert_stream.h"

#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using infernix::runtime::expert_offload::ExpertStream;

int failures = 0;

void check(bool condition, const std::string& message) {
    if (condition) { return; }
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

constexpr std::uint32_t kLayers = 4, kExperts = 16, kHalf = 8;
constexpr std::uint64_t kStride = 256;

// Every record's bytes name its layer and expert.
std::uint8_t record_byte(std::uint32_t layer, std::uint32_t expert, std::uint64_t i) {
    return static_cast<std::uint8_t>(layer * 31 + expert * 7 + i);
}

struct Rig {
    infernix::PinnedHostBuffer records{kLayers * kExperts * kStride};
    std::vector<std::int32_t> residency = std::vector<std::int32_t>(kLayers * kExperts, -1);
    infernix::DeviceBuffer ring;
    cudaStream_t compute = nullptr;
    ExpertStream stream;

    Rig()
        : stream(kLayers, kExperts, kStride,
                 [this](std::uint32_t l, std::uint32_t e) {
                     return static_cast<const std::uint8_t*>(records.data()) + (static_cast<std::size_t>(l) * kExperts + e) * kStride;
                 },
                 /*one_allocation=*/true) {
        auto* bytes = static_cast<std::uint8_t*>(records.data());
        for (std::uint32_t l = 0; l < kLayers; ++l) {
            for (std::uint32_t e = 0; e < kExperts; ++e) {
                for (std::uint64_t i = 0; i < kStride; ++i) { bytes[(static_cast<std::size_t>(l) * kExperts + e) * kStride + i] = record_byte(l, e, i); }
                residency[static_cast<std::size_t>(l) * kExperts + e] = e < 4 ? static_cast<std::int32_t>(e) : -1; // 0..3 resident
            }
        }
        // The slot tables (rounded to 256 bytes) and two halves of kHalf records.
        const std::size_t tables = (kLayers * kExperts * sizeof(std::int32_t) + 255) / 256 * 256;
        ring = infernix::DeviceBuffer(tables + 2 * kHalf * kStride);
        CUDA_CHECK(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
    }
    ~Rig() {
        if (compute != nullptr) {
            (void)cudaStreamSynchronize(compute);
            (void)cudaStreamDestroy(compute);
        }
    }

    infernix::DeviceSpan span() const { return {ring.p, ring.bytes}; }

    // The layer's slot table and, for every expert it names, whether the slot holds its record.
    std::vector<std::int32_t> table(std::uint32_t layer, bool& bytes_ok) {
        CUDA_CHECK(cudaStreamSynchronize(compute));
        std::vector<std::int32_t> row(kExperts);
        CUDA_CHECK(cudaMemcpy(row.data(), stream.slots(layer), kExperts * sizeof(std::int32_t), cudaMemcpyDeviceToHost));
        bytes_ok = true;
        std::vector<std::uint8_t> slot(kStride);
        for (std::uint32_t e = 0; e < kExperts; ++e) {
            if (row[e] < 0) { continue; }
            CUDA_CHECK(cudaMemcpy(slot.data(), stream.base() + static_cast<std::size_t>(row[e]) * stream.pitch(), kStride,
                                  cudaMemcpyDeviceToHost));
            for (std::uint64_t i = 0; i < kStride; ++i) {
                if (slot[i] != record_byte(layer, e, i)) {
                    bytes_ok = false;
                    break;
                }
            }
        }
        return row;
    }
};

std::vector<std::int32_t> columns_of(std::initializer_list<std::pair<std::uint32_t, std::int32_t>> routed) {
    std::vector<std::int32_t> columns(kExperts, 0);
    for (const auto& [e, n] : routed) { columns[e] = n; }
    return columns;
}

void gated_chunk() {
    Rig rig;
    ExpertStream& s = rig.stream;
    s.begin(rig.span(), rig.residency.data(), rig.compute, /*gated=*/true);
    std::vector<std::uint8_t> no_cpu(kExperts, 0);

    // Layer 0: no guess, the plain gated plan (routed, non-resident: 4, 5, 9).
    const auto c0 = columns_of({{1, 30}, {4, 12}, {5, 1}, {9, 3}});
    check(s.plan_layer(0, c0, no_cpu) == 3, "layer 0 copies its three routed non-resident experts");
    s.before_experts(0, rig.compute);
    bool ok = false;
    auto row = rig.table(0, ok);
    check(ok, "layer 0's slots hold their records");
    check(row[1] == -1 && row[4] == 0 && row[5] == 1 && row[9] == 2 && row[6] == -1, "layer 0's table");
    s.after_experts(0, rig.compute);

    // Layer 1's guess: 5 (10), 6 (3), 12 (5) pass rows > 2; 7 (1) does not; 2 is resident.
    const auto g1 = columns_of({{2, 9}, {5, 10}, {6, 3}, {7, 1}, {12, 5}});
    check(s.plan_ahead(1, g1, /*rows=*/2, /*most=*/0) == 3, "layer 1 copies three guessed experts ahead");
    try {
        (void)s.plan_ahead(1, g1, 2, 0);
        check(false, "a second plan_ahead of one layer was accepted");
    } catch (const std::logic_error&) {}
    // Its true routing: 5 (7) and 6 (3) were guessed; 12 was not routed; 8 (4) is late; 9 (1) goes to the CPU.
    auto c1 = columns_of({{5, 7}, {6, 3}, {8, 4}, {9, 1}});
    auto priced = c1;
    s.mask_ahead(1, priced);
    check(priced[5] == 0 && priced[6] == 0 && priced[8] == 4 && priced[9] == 1, "mask_ahead hides only the experts copied ahead");
    std::vector<std::uint8_t> cpu1(kExperts, 0);
    cpu1[9] = 1;
    check(s.plan_layer(1, c1, cpu1) == 1, "layer 1 copies only its one late expert");
    s.before_experts(1, rig.compute);
    row = rig.table(1, ok);
    check(ok, "layer 1's slots (ahead and late) hold their records");
    // Half 1 starts at slot 8: the ahead set in ascending ids, then the late copy.
    check(row[5] == 8 && row[6] == 9 && row[12] == 10 && row[8] == 11 && row[9] == -1 && row[7] == -1 && row[2] == -1,
          "layer 1's table keeps the ahead slots and puts the late copy after them");
    s.after_experts(1, rig.compute);

    // Layer 2 reuses half 0 (after layer 0's experts): at most two ahead, the heaviest guesses.
    const auto g2 = columns_of({{4, 1}, {10, 9}, {11, 5}, {13, 7}});
    check(s.plan_ahead(2, g2, /*rows=*/0, /*most=*/2) == 2, "layer 2 copies its two heaviest guesses ahead");
    const auto c2 = columns_of({{10, 2}, {11, 6}});
    check(s.plan_layer(2, c2, no_cpu) == 1, "layer 2 copies its unguessed routed expert late");
    s.before_experts(2, rig.compute);
    row = rig.table(2, ok);
    check(ok, "layer 2's slots hold their records");
    check(row[10] == 0 && row[13] == 1 && row[11] == 2 && row[4] == -1, "layer 2's table (cap of two)");
    s.after_experts(2, rig.compute);

    // Layer 3 has no guess: its row is planned whole, whatever an earlier chunk left in it.
    const auto c3 = columns_of({{15, 2}});
    check(s.plan_layer(3, c3, no_cpu) == 1, "layer 3 copies its routed expert");
    s.before_experts(3, rig.compute);
    row = rig.table(3, ok);
    check(ok && row[15] == 8 && row[5] == -1, "layer 3's table");
    s.after_experts(3, rig.compute);
    s.end();

    // Layer 1 (rows > 2): ahead 5, 6, 12; routed 5, 6; heavy 5, 6 (ahead) and 8 (late). Layer 2 (rows > 0):
    // ahead 10, 13; routed 10; heavy 10 (ahead) and 11 (late).
    const auto& st = s.stats();
    check(st.ahead == 5, "five records copied ahead");
    check(st.ahead_used == 3, "three of them routed");
    check(st.heavy == 5 && st.heavy_ahead == 3, "five heavy experts in the guessed layers, three of them ahead");
    check(st.late == 2, "two late copies in the guessed layers");
    check(st.streamed == 3 + 3 + 1 + 2 + 1 + 1, "every copied record is counted once");
}

void ungated_refuses() {
    Rig rig;
    rig.stream.begin(rig.span(), rig.residency.data(), rig.compute, /*gated=*/false);
    try {
        (void)rig.stream.plan_ahead(1, std::vector<std::int32_t>(kExperts, 9), 2, 0);
        check(false, "plan_ahead was accepted in an ungated chunk");
    } catch (const std::logic_error&) {}
    std::vector<std::int32_t> columns(kExperts, 5);
    rig.stream.mask_ahead(1, columns); // a no-op outside a gated chunk
    check(columns[7] == 5, "mask_ahead changed an ungated chunk's columns");
    for (std::uint32_t l = 0; l < kLayers; ++l) {
        rig.stream.before_experts(l, rig.compute);
        rig.stream.after_experts(l, rig.compute);
    }
    rig.stream.end();
}

} // namespace

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    CUDA_CHECK(cudaSetDevice(0));
    gated_chunk();
    ungated_refuses();
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << (failures == 0 ? "expert stream checks passed\n" : "expert stream checks failed\n");
    return failures == 0 ? 0 : 1;
}
