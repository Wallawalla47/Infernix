// The SSD tier's in-place expert records on the real artifact (design §19.3.7 RT3/RT4): the expert
// banks bound Streamed give an ExpertStore whose every record is one or two 4 KiB-aligned file
// segments; records read through DirectReadQueue from those segments equal the bytes the pinned
// load reads, and so do the multipliers read from the scale tails, for the first and last layers and
// every layer holding a record that straddles part files. Skips (77) without NINFER_QWEN4_ARTIFACT.
#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "core/device.h"
#include "core/direct_read_queue.h"
#include "models/qwen4_exp/expert_store.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <new>
#include <span>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using models::qwen4_exp::ExpertStore;

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::cerr << "FAIL: " << what << '\n';
    }
}

std::string bank_name(std::uint32_t layer) { return "text/layers/" + std::to_string(layer) + "/moe/experts"; }

artifact::ParameterReference bind_bank(artifact::Binder& binder, const artifact::Reader& reader, std::uint32_t layer,
                                       artifact::Residency residency) {
    const auto& binding = reader.directory().bindings.at(bank_name(layer));
    const auto shape    = reader.geometry(binding.parts.front().object).shape;
    return binder.parameter(bank_name(layer), shape, residency, QType::NVFP4_MUL);
}

} // namespace

int main() {
    const char* path = std::getenv("NINFER_QWEN4_ARTIFACT");
    if (path == nullptr || !std::filesystem::exists(path)) {
        std::cout << "SKIP: NINFER_QWEN4_ARTIFACT is not set\n";
        return 77;
    }
    try {
        DeviceContext device(0);
        artifact::Reader reader(path);
        std::uint32_t layers = 0;
        while (reader.directory().bindings.contains(bank_name(layers))) { ++layers; }
        check(layers == 48, "48 MoE layers");

        // Every bank streamed: the store over all layers, with the scale tails read here.
        artifact::Binder streamed_binder(reader);
        std::vector<artifact::ObjectHandle> objects;
        std::vector<std::vector<float>> multipliers;
        std::uint64_t stride = 0;
        std::uint32_t experts = 0;
        for (std::uint32_t l = 0; l < layers; ++l) {
            const auto ref       = bind_bank(streamed_binder, reader, l, artifact::Residency::Streamed);
            const auto object    = ref.binding.parts.front().object;
            const auto& geometry = reader.geometry(object);
            stride  = geometry.record_stride;
            experts = static_cast<std::uint32_t>(geometry.shape[0]);
            std::vector<float> words(geometry.scale_bytes / sizeof(float));
            reader.read_into(artifact::object_offset(reader.directory().object(object)) + geometry.scale_offset,
                             std::as_writable_bytes(std::span(words)));
            objects.push_back(object);
            multipliers.push_back(std::move(words));
        }
        auto streamed_plan = std::move(streamed_binder).finish();
        check(streamed_plan.pinned_objects.empty() && streamed_plan.streamed_objects.size() == layers,
              "streamed banks are located, not placed");
        const auto streamed = artifact::materialize(reader, std::move(streamed_plan), device);
        check(streamed.stats().pinned_bytes == 0, "nothing pinned for streamed banks");
        const auto t0 = std::chrono::steady_clock::now();
        ExpertStore store(streamed.stream_source(), objects, stride, experts, multipliers);
        std::cout << "store: " << store.layers() << " layers x " << store.experts() << " records of " << store.record_bytes()
                  << " B in " << store.files().size() << " files, " << store.straddling() << " straddling, built in "
                  << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() << " ms\n";
        std::vector<std::uint32_t> sample{0, layers - 1};
        for (std::uint32_t l = 0; l < layers; ++l) {
            for (std::uint32_t e = 0; e < experts; ++e) {
                if (store.segments(l, e).size() == 2) {
                    std::cout << "straddling record L" << l << " E" << e << '\n';
                    if (std::find(sample.begin(), sample.end(), l) == sample.end()) { sample.push_back(l); }
                }
            }
        }

        // The sampled layers pinned, as the full-RAM load reads them.
        artifact::Binder pinned_binder(reader);
        std::vector<artifact::ParameterReference> refs;
        for (const auto l : sample) { refs.push_back(bind_bank(pinned_binder, reader, l, artifact::Residency::HostPinned)); }
        auto pinned_plan = std::move(pinned_binder).finish();
        const auto pinned = artifact::materialize(reader, std::move(pinned_plan), device);

        DirectReadQueue queue;
        std::vector<std::uint32_t> handles;
        for (const auto& file : store.files()) { handles.push_back(queue.open(file)); }
        const std::size_t bytes = store.record_bytes();
        auto* buffer = static_cast<std::byte*>(::operator new(bytes * 16, std::align_val_t{4096}));
        std::uint64_t compared = 0;
        double read_seconds    = 0;
        for (std::size_t i = 0; i < sample.size(); ++i) {
            const std::uint32_t l = sample[i];
            const auto& parent    = pinned.pinned_parent(refs[i].binding.parts.front().object);
            check(std::memcmp(store.multipliers(l), parent.data + parent.geometry.scale_offset, parent.geometry.scale_bytes) == 0,
                  "layer " + std::to_string(l) + " multipliers equal the pinned scale tail");
            for (std::uint32_t e0 = 0; e0 < experts; e0 += 16) {
                const auto start = std::chrono::steady_clock::now();
                std::uint64_t tag = 0;
                for (std::uint32_t k = 0; k < 16; ++k) {
                    for (const auto& s : store.segments(l, e0 + k)) {
                        queue.submit({handles[s.file], s.offset, s.bytes, buffer + k * bytes + s.at,
                                      DirectReadQueue::Priority::Demand, tag++});
                    }
                }
                std::vector<DirectReadQueue::Completion> done;
                while (done.size() < tag) { queue.poll(done, std::chrono::milliseconds(100)); }
                read_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                for (const auto& c : done) { check(c.status == DirectReadQueue::Status::Ok, "record read"); }
                for (std::uint32_t k = 0; k < 16; ++k) {
                    const auto e = e0 + k;
                    if (std::memcmp(buffer + k * bytes, parent.data + e * parent.geometry.record_stride, bytes) != 0) {
                        check(false, "record L" + std::to_string(l) + " E" + std::to_string(e) + " equals the pinned bytes");
                    }
                    ++compared;
                }
            }
        }
        ::operator delete(buffer, std::align_val_t{4096});
        std::cout << "compared " << compared << " records (" << static_cast<double>(compared * bytes) / 1e9 << " GB) at "
                  << static_cast<double>(compared * bytes) / 1e9 / read_seconds << " GB/s, queue depth 8\n";
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "expert store: all checks passed\n";
    return 0;
}
