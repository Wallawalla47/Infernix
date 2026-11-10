// CSF host records (ops/offloaded_sparse_moe/csf_record.h, csf_expand.h, cuda/csf_expand.cuh): every record compresses
// to a well-formed table and expands back to its bytes exactly, on the host (out of place and in place), on the GPU
// chunk by chunk from a host or device copy of the compressed record (the stage kernel's expansion), and in place in
// device memory (copy-engine landings), at both registered geometries, for scales as NVIDIA's exports have them (a
// row's codes within a few steps), with outlying codes (escapes), with no two equal codes in a row (every group raw),
// with constant rows and with raw and compressed groups mixed; the realistic records compress below 95 % of the
// expanded bytes.
#include "ops/op_family.h"
#include "ops/offloaded_sparse_moe/csf_expand.h"
#include "ops/offloaded_sparse_moe/csf_record.h"
#include "ops/offloaded_sparse_moe/cuda/csf_expand.cuh"

#include "ops/op_tester.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

using namespace infernix;
using namespace infernix::test;
namespace csf = infernix::ops::offloaded_moe::csf;
using infernix::ops::offloaded_moe::ExpertShape;
using infernix::ops::offloaded_moe::kUnitBytes;

namespace {

constexpr int kThreads = 256;

struct ChunkShared {
    csf::ChunkBuffer buffer;
    csf::ChunkScratch scratch;
    std::uint32_t words[csf::kMaxChunkRgs + 1];
};

// The stage kernel's expansion: chunk by chunk (grid-strided) from `src` into `dst`.
__global__ void expand_chunks_kernel(csf::Geometry g, const std::uint8_t* src, const std::uint32_t* table, std::uint8_t* dst) {
    __shared__ ChunkShared s;
    for (int c = static_cast<int>(blockIdx.x); c < g.chunks(); c += static_cast<int>(gridDim.x)) {
        const int r0 = g.chunk_first_rg(c), n = g.chunk_rgs(c);
        if (static_cast<int>(threadIdx.x) <= n) { s.words[threadIdx.x] = table[r0 + static_cast<int>(threadIdx.x)]; }
        __syncthreads();
        csf::load_chunk_async<kThreads>(src + s.words[0], s.words[n] - s.words[0], s.buffer.data);
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        __syncthreads();
        csf::expand_chunk<kThreads>(g, c, s.words, s.buffer.data, dst, s.scratch);
    }
}

enum class Scales { Realistic, Outliers, Distinct, Constant, Mixed };

std::vector<std::uint8_t> make_record(const ExpertShape& shape, Scales kind, std::uint32_t seed) {
    const csf::Geometry g = csf::Geometry::of(shape);
    std::vector<std::uint8_t> record(shape.record_bytes());
    std::mt19937 rng(seed);
    for (auto& b : record) { b = static_cast<std::uint8_t>(rng()); }
    for (int rg = 0; rg < g.rgs(); ++rg) {
        int centre[16];
        for (auto& c : centre) { c = 20 + static_cast<int>(rng() % 70); } // E4M3 codes of small positive scales
        for (int b = 0; b < g.blocks(rg); ++b) {
            std::uint8_t* unit = record.data() + g.expanded_offset(rg) + static_cast<std::size_t>(b) * kUnitBytes;
            for (int r = 0; r < 16; ++r) {
                const int realistic = centre[r] + static_cast<int>(rng() % 9) - 4;
                const int distinct  = (b * 37 + r * 11 + rg * 5) % 127;
                int code            = 0;
                switch (kind) {
                case Scales::Realistic: code = realistic; break;
                case Scales::Outliers: code = realistic + (rng() % 50 == 0 ? 30 : 0); break;
                case Scales::Distinct: code = distinct; break;
                case Scales::Constant: code = centre[r]; break;
                case Scales::Mixed: code = rg % 3 == 1 ? distinct : realistic; break;
                }
                unit[csf::kCodeBytes + r] = static_cast<std::uint8_t>(code);
            }
        }
    }
    return record;
}

int check(const ExpertShape& shape, Scales kind, const std::string& label) {
    int failures          = 0;
    const csf::Geometry g = csf::Geometry::of(shape);
    const auto record     = make_record(shape, kind, 1234U + static_cast<std::uint32_t>(kind));
    std::vector<std::uint8_t> compressed(record.size(), 0xAB);
    std::vector<std::uint32_t> table(static_cast<std::size_t>(g.table_words()));
    const std::size_t bytes = csf::compress_record(shape, record.data(), compressed.data(), table.data());
    if (bytes != table.back() || bytes > record.size() || !csf::valid_table(shape, table.data(), record.size())) {
        std::cerr << label << ": malformed table (" << bytes << " bytes)\n";
        return failures + 1;
    }
    std::vector<std::uint8_t> back(record.size(), 0xCD);
    csf::expand_record(shape, compressed.data(), table.data(), back.data());
    if (back != record) {
        std::cerr << label << ": host round trip differs\n";
        ++failures;
    }
    // In place: the expanded record's own buffer.
    std::vector<std::uint8_t> in_place = record;
    std::vector<std::uint32_t> table2(table.size());
    const std::size_t again = csf::compress_record(shape, in_place.data(), in_place.data(), table2.data());
    if (again != bytes || table2 != table || std::memcmp(in_place.data(), compressed.data(), bytes) != 0) {
        std::cerr << label << ": in-place compression differs\n";
        ++failures;
    }
    int raw = 0;
    for (int rg = 0; rg < g.rgs(); ++rg) { raw += table[rg + 1] - table[rg] == g.blocks(rg) * kUnitBytes ? 1 : 0; }
    // GPU, chunk by chunk from a device copy and from pinned host memory (the stage kernel's misses).
    const std::vector<std::uint8_t> head(compressed.begin(), compressed.begin() + static_cast<std::ptrdiff_t>(bytes));
    DeviceBuffer src = to_device(head);
    DeviceBuffer dev_table = to_device(table);
    std::uint8_t* host_src = nullptr;
    std::uint32_t* host_table = nullptr;
    cuda_check(cudaMallocHost(&host_src, bytes), "cudaMallocHost");
    cuda_check(cudaMallocHost(&host_table, table.size() * sizeof(std::uint32_t)), "cudaMallocHost");
    std::memcpy(host_src, head.data(), bytes);
    std::memcpy(host_table, table.data(), table.size() * sizeof(std::uint32_t));
    for (const bool mapped : {false, true}) {
        GuardedDeviceBuffer dst(record.size());
        dst.fill(0xEE);
        expand_chunks_kernel<<<64, kThreads>>>(g, mapped ? host_src : static_cast<const std::uint8_t*>(src.p),
                                               mapped ? host_table : static_cast<const std::uint32_t*>(dev_table.p),
                                               static_cast<std::uint8_t*>(dst.data()));
        cuda_synchronize();
        const std::string where = label + (mapped ? " GPU chunks (mapped host)" : " GPU chunks (device)");
        if (from_device<std::uint8_t>(dst.data(), record.size()) != record) {
            std::cerr << where << ": expansion differs\n";
            ++failures;
        }
        failures += dst.verify_guards(where);
    }
    // GPU in place: the compressed bytes at the start of a record-sized buffer, the table in mapped host memory.
    {
        std::vector<std::uint8_t> landing(record.size(), 0x5A);
        std::copy(head.begin(), head.end(), landing.begin());
        GuardedDeviceBuffer frame(record.size());
        cuda_check(cudaMemcpy(frame.data(), landing.data(), landing.size(), cudaMemcpyHostToDevice), "cudaMemcpy");
        csf::InPlaceJobs jobs;
        jobs.count  = 1;
        jobs.job[0] = {static_cast<std::uint8_t*>(frame.data()), host_table};
        csf::expand_in_place(shape, jobs, nullptr);
        cuda_synchronize();
        if (from_device<std::uint8_t>(frame.data(), record.size()) != record) {
            std::cerr << label << ": GPU in-place expansion differs\n";
            ++failures;
        }
        failures += frame.verify_guards(label + " GPU in place");
    }
    cuda_check(cudaFreeHost(host_src), "cudaFreeHost");
    cuda_check(cudaFreeHost(host_table), "cudaFreeHost");
    const double ratio = static_cast<double>(bytes) / static_cast<double>(record.size());
    std::cout << label << ": " << bytes << " of " << record.size() << " bytes (" << ratio << "), " << raw << " of "
              << g.rgs() << " groups raw\n";
    if (kind == Scales::Realistic && (ratio > 0.95 || raw != 0)) {
        std::cerr << label << ": realistic scales compress to only " << ratio << "\n";
        ++failures;
    }
    if (kind == Scales::Distinct && raw != g.rgs()) {
        std::cerr << label << ": escape-heavy groups were not stored raw\n";
        ++failures;
    }
    if (kind == Scales::Mixed && (raw == 0 || raw == g.rgs())) {
        std::cerr << label << ": the mixed record does not mix raw and compressed groups\n";
        ++failures;
    }
    return failures;
}

// Several records expanded in place in one launch, each CTA its own record.
int check_batch(const ExpertShape& shape) {
    const csf::Geometry g = csf::Geometry::of(shape);
    constexpr int kRecords = 5;
    int failures = 0;
    std::vector<std::vector<std::uint8_t>> records;
    std::vector<std::uint32_t*> tables;
    std::vector<GuardedDeviceBuffer> frames;
    csf::InPlaceJobs jobs;
    for (int i = 0; i < kRecords; ++i) {
        records.push_back(make_record(shape, i % 2 == 0 ? Scales::Realistic : Scales::Mixed, 99U + static_cast<std::uint32_t>(i)));
        std::uint32_t* table = nullptr;
        cuda_check(cudaMallocHost(&table, static_cast<std::size_t>(g.table_words()) * sizeof(std::uint32_t)), "cudaMallocHost");
        tables.push_back(table);
        std::vector<std::uint8_t> landing = records.back();
        (void)csf::compress_record(shape, landing.data(), landing.data(), table);
        frames.emplace_back(landing.size());
        cuda_check(cudaMemcpy(frames.back().data(), landing.data(), landing.size(), cudaMemcpyHostToDevice), "cudaMemcpy");
    }
    for (int i = 0; i < kRecords; ++i) { jobs.job[jobs.count++] = {static_cast<std::uint8_t*>(frames[static_cast<std::size_t>(i)].data()), tables[static_cast<std::size_t>(i)]}; }
    csf::expand_in_place(shape, jobs, nullptr);
    cuda_synchronize();
    for (int i = 0; i < kRecords; ++i) {
        if (from_device<std::uint8_t>(frames[static_cast<std::size_t>(i)].data(), records[static_cast<std::size_t>(i)].size()) !=
            records[static_cast<std::size_t>(i)]) {
            std::cerr << "batch record " << i << ": GPU in-place expansion differs\n";
            ++failures;
        }
        failures += frames[static_cast<std::size_t>(i)].verify_guards("batch record " + std::to_string(i));
        cuda_check(cudaFreeHost(tables[static_cast<std::size_t>(i)]), "cudaFreeHost");
    }
    return failures;
}

} // namespace

INFERNIX_OP_TEST(offloaded_moe_csf) {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    try {
        for (const auto& [shape, name] : {std::pair{infernix::ops::offloaded_moe::kFlashNextShape, "flash-next"},
                                          std::pair{infernix::ops::offloaded_moe::kQwen3_5_122BShape, "122b"}}) {
            failures += check(shape, Scales::Realistic, std::string(name) + " realistic");
            failures += check(shape, Scales::Outliers, std::string(name) + " outliers");
            failures += check(shape, Scales::Distinct, std::string(name) + " distinct");
            failures += check(shape, Scales::Constant, std::string(name) + " constant");
            failures += check(shape, Scales::Mixed, std::string(name) + " mixed");
            failures += check_batch(shape);
        }
    } catch (const std::exception& error) {
        std::cerr << "csf test threw: " << error.what() << '\n';
        return 1;
    }
    std::cout << (failures == 0 ? "OK" : "FAIL") << " offloaded_moe_csf\n";
    return failures == 0 ? 0 : 1;
}
