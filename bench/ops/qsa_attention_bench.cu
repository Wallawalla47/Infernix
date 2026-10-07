// Public-Op benchmark for QSA attention (Qwen4Exp geometry: 24/2 heads, D 256, Di 128, R 4, B 2048).
// One row of W columns at the end of a C-token context, KV in the requested storage. Each case
// times one public qsa_attention() call (selection, attention, merge) and the selection alone
// (detail::qsa_select); their difference is the attention and merge. Every measurement is a
// captured graph launched after an L2 flush: in a forward, a layer's KV is never L2-resident.
// Plane contents are fixed finite fixture values; the timing does not depend on them.

#include "infernix/ops/qsa.h"

#include "core/device.h"
#include "core/paged_kv_storage.h"
#include "infernix_bench_common.h"
#include "ops/qsa/qsa_select.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace infernix;

namespace {

constexpr int kD = 256, kHeads = 24, kKvHeads = 2, kDi = 128, kIndexHeads = 4, kR = 4, kBudget = 2048;
constexpr int kPage           = 64;
constexpr std::size_t kFlush  = std::size_t{256} << 20;
const ops::QsaGeometry kGeometry{.heads          = kHeads,
                                 .kv_heads       = kKvHeads,
                                 .head_dim       = kD,
                                 .index_heads    = kIndexHeads,
                                 .index_head_dim = kDi,
                                 .rotary_dim     = 64,
                                 .budget         = kBudget,
                                 .ratio          = kR,
                                 .theta          = 1.0e7F,
                                 .eps            = 1.0e-6F};

struct Options {
    std::vector<KvCacheStorage> storages{KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64};
    std::vector<int> contexts{8192, 131072};
    std::vector<int> widths{1, 5};
    int warmup = 5;
    int repeat = 50;
};

const char* storage_name(KvCacheStorage s) {
    switch (s) {
    case KvCacheStorage::BFloat16: return "bf16";
    case KvCacheStorage::Int8Group64: return "int8";
    case KvCacheStorage::Fp8E4M3Row256: return "fp8";
    case KvCacheStorage::Nvfp4Group16: return "nvfp4";
    case KvCacheStorage::Fp8KeyNvfp4Value: return "k8v4";
    case KvCacheStorage::Vq2: return "vq2";
    case KvCacheStorage::Q4KeyVq2Value: return "k4v2";
    }
    return "?";
}

KvCacheStorage parse_storage(std::string_view s) {
    for (const auto k : {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64, KvCacheStorage::Fp8E4M3Row256,
                         KvCacheStorage::Nvfp4Group16, KvCacheStorage::Fp8KeyNvfp4Value, KvCacheStorage::Vq2,
                         KvCacheStorage::Q4KeyVq2Value}) {
        if (s == storage_name(k)) { return k; }
    }
    throw std::invalid_argument("unknown storage " + std::string(s));
}

std::vector<std::string_view> split(std::string_view s) {
    std::vector<std::string_view> out;
    while (!s.empty()) {
        const auto comma = s.find(',');
        out.push_back(s.substr(0, comma));
        s = comma == std::string_view::npos ? std::string_view{} : s.substr(comma + 1);
    }
    return out;
}

Options parse_options(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        const auto value           = [&]() -> std::string_view {
            if (i + 1 >= argc) { throw std::invalid_argument("missing value for " + std::string(arg)); }
            return argv[++i];
        };
        const auto ints = [&](std::string_view v) {
            std::vector<int> out;
            for (const auto part : split(v)) { out.push_back(std::atoi(std::string(part).c_str())); }
            return out;
        };
        if (arg == "--storage") {
            o.storages.clear();
            for (const auto part : split(value())) { o.storages.push_back(parse_storage(part)); }
        } else if (arg == "--contexts") {
            o.contexts = ints(value());
        } else if (arg == "--widths") {
            o.widths = ints(value());
        } else if (arg == "--warmup") {
            o.warmup = std::atoi(argv[++i]);
        } else if (arg == "--repeat") {
            o.repeat = std::atoi(argv[++i]);
        } else if (arg == "--help") {
            std::printf("infernix_qsa_attention_bench [--storage bf16,int8,fp8,nvfp4,k8v4,vq2,k4v2] "
                        "[--contexts 8192,131072] [--widths 1,5] [--warmup N] [--repeat N]\n");
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown option " + std::string(arg));
        }
    }
    return o;
}

DeviceBuffer filled(std::size_t bytes, std::uint64_t seed) {
    // Fixture BF16 values in [0.002, 0.01]: finite under every storage's interpretation.
    return bench::make_bf16((bytes + 1) / 2, seed, 0.002F, 0.01F);
}

Tensor plane(const DeviceBuffer& b, DType dtype, int leading, int pages) {
    return Tensor(b.p, dtype, {leading, kPage, kKvHeads, pages});
}

struct Case {
    int width, context, pages;
    DeviceBuffer k, v, ks, vs, pooled, q, iq, tables, positions, rows, slots, workspace, out;
    DeviceBuffer call_k, call_v, window_k, window_v, window_ks, window_vs, window_tags; // vq2, k4v2
    std::size_t workspace_bytes = 0;
    ops::QsaKVLayer layer;
    ops::QsaBatch batch;
    ops::QsaAppend append;
    bool vq = false;

    Case(KvCacheStorage storage, int width_, int context_) : width(width_), context(context_) {
        pages                = (context + kPage - 1) / kPage;
        const auto layout    = paged_kv_storage_layout(storage, kD);
        const auto bytes     = [&](DType t, int leading) {
            return static_cast<std::size_t>(leading) * kPage * kKvHeads * pages * dtype_size(t);
        };
        k = filled(bytes(layout.key.data_dtype, layout.key.data_leading_extent), 11);
        v = filled(bytes(layout.value.data_dtype, layout.value.data_leading_extent), 12);
        layer.kv.storage      = storage;
        layer.kv.head_dim     = kD;
        layer.kv.num_kv_heads = kKvHeads;
        layer.kv.k_pages      = plane(k, layout.key.data_dtype, layout.key.data_leading_extent, pages);
        layer.kv.v_pages      = plane(v, layout.value.data_dtype, layout.value.data_leading_extent, pages);
        if (layout.key.has_scale()) {
            ks = filled(bytes(layout.key.scale_dtype, layout.key.scale_leading_extent), 13);
            layer.kv.k_scale_pages = plane(ks, layout.key.scale_dtype, layout.key.scale_leading_extent, pages);
        }
        if (layout.value.has_scale()) {
            vs = filled(bytes(layout.value.scale_dtype, layout.value.scale_leading_extent), 14);
            layer.kv.v_scale_pages =
                plane(vs, layout.value.scale_dtype, layout.value.scale_leading_extent, pages);
        }
        pooled             = bench::make_bf16(static_cast<std::size_t>(kDi / kR) * kPage * pages, 15);
        layer.pooled_pages = Tensor(pooled.p, DType::BF16, {kDi / kR, kPage, 1, pages});
        q                  = bench::make_bf16(static_cast<std::size_t>(kD) * kHeads * width, 16);
        iq                 = bench::make_bf16(static_cast<std::size_t>(kDi) * kIndexHeads * width, 17);
        std::vector<std::int32_t> table(pages), position(width), zero{0};
        for (int p = 0; p < pages; ++p) { table[p] = (p * 7919) % pages; } // scattered pages
        for (int j = 0; j < width; ++j) { position[j] = context - width + j; }
        const auto upload = [](const std::vector<std::int32_t>& h) {
            DeviceBuffer d(h.size() * sizeof(std::int32_t));
            CUDA_CHECK(cudaMemcpy(d.p, h.data(), d.bytes, cudaMemcpyHostToDevice));
            return d;
        };
        tables    = upload(table);
        positions = upload(position);
        rows      = upload(zero);
        slots     = upload(zero);
        batch     = ops::QsaBatch{.block_tables = Tensor(tables.p, DType::I32, {pages, 1}),
                                  .table_rows   = Tensor(rows.p, DType::I32, {1}),
                                  .positions    = Tensor(positions.p, DType::I32, {width}),
                                  .tail_slots   = Tensor(slots.p, DType::I32, {1}),
                                  .batch        = 1,
                                  .width        = width};
        // The vector-quantized storages: the call's K/V, which the Op appends, and an exact window
        // (zeroed: its tags never match, so exact keys also decode their codes after the tag check).
        vq = storage == KvCacheStorage::Vq2 || storage == KvCacheStorage::Q4KeyVq2Value;
        if (vq) {
            call_k = bench::make_bf16(static_cast<std::size_t>(kD) * kKvHeads * width, 18);
            call_v = bench::make_bf16(static_cast<std::size_t>(kD) * kKvHeads * width, 19);
            append = {.k = Tensor(call_k.p, DType::BF16, {kD, kKvHeads, width, 1}),
                      .v = Tensor(call_v.p, DType::BF16, {kD, kKvHeads, width, 1})};
            const int slots_n = kKVWindowSlots;
            window_k    = bench::make_zeros(static_cast<std::size_t>(kD) * slots_n * kKvHeads);
            window_v    = bench::make_zeros(static_cast<std::size_t>(kD) * slots_n * kKvHeads);
            window_ks   = bench::make_zeros(sizeof(std::uint16_t) * 4 * slots_n * kKvHeads);
            window_vs   = bench::make_zeros(sizeof(std::uint16_t) * 4 * slots_n * kKvHeads);
            window_tags = bench::make_zeros(sizeof(std::int32_t) * 2 * slots_n * kKvHeads);
            layer.kv.window = PagedKVWindowView{.k_codes  = Tensor(window_k.p, DType::I8, {kD, slots_n, kKvHeads, 1}),
                                                .v_codes  = Tensor(window_v.p, DType::I8, {kD, slots_n, kKvHeads, 1}),
                                                .k_scales = Tensor(window_ks.p, DType::FP16, {4, slots_n, kKvHeads, 1}),
                                                .v_scales = Tensor(window_vs.p, DType::FP16, {4, slots_n, kKvHeads, 1}),
                                                .tags     = Tensor(window_tags.p, DType::I32, {2, slots_n, kKvHeads, 1}),
                                                .slots    = Tensor(rows.p, DType::I32, {1})};
        }
        workspace_bytes = ops::qsa_attention_workspace_bytes(kGeometry, width, pages * kPage, storage);
        workspace       = DeviceBuffer(workspace_bytes);
        out             = DeviceBuffer(static_cast<std::size_t>(kD) * kHeads * width * 2);
    }

    void attention(cudaStream_t s) {
        Tensor o(out.p, DType::BF16, {kD, kHeads, width});
        ops::qsa_attention(Tensor(q.p, DType::BF16, {kD, kHeads, width}),
                           Tensor(iq.p, DType::BF16, {kDi, kIndexHeads, width}), layer, batch, kGeometry,
                           1.0F / 16.0F, pages * kPage, workspace.p, workspace_bytes, o, s, vq ? &append : nullptr);
    }

    // The selection qsa_attention runs first, into the head of the same workspace layout.
    void selection(cudaStream_t s) {
        const std::size_t select_bytes = ops::detail::qsa_select_scratch_bytes(kGeometry, width, pages * kPage);
        auto* base                     = static_cast<unsigned char*>(workspace.p);
        const std::size_t list_bytes   = (sizeof(std::int32_t) * (kBudget / kR) * width + 255) / 256 * 256;
        const std::size_t count_bytes  = (sizeof(std::int32_t) * width + 255) / 256 * 256;
        ops::detail::qsa_select(Tensor(iq.p, DType::BF16, {kDi, kIndexHeads, width}), layer.pooled_pages, batch,
                                kGeometry, pages * kPage, base + list_bytes + count_bytes, select_bytes,
                                {reinterpret_cast<std::int32_t*>(base),
                                 reinterpret_cast<std::int32_t*>(base + list_bytes)},
                                s, layer.spaces);
    }
};

bench::ColdTiming time(bench::L2FlushBuffer& flush, cudaStream_t stream, int warmup, int repeat,
                       const std::function<void(cudaStream_t)>& body) {
    body(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    bench::TimedGraph graph;
    graph.capture(stream, body);
    return bench::measure_cold_graph(graph, flush, stream, warmup, repeat);
}

} // namespace

int main(int argc, char** argv) {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
            std::printf("SKIP: no usable CUDA device\n");
            return 0;
        }
        const Options options = parse_options(argc, argv);
        cudaStream_t stream   = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        bench::L2FlushBuffer flush(kFlush);
        std::printf("storage context width   total_us  select_us  attention_us  (median of %d, cold L2, graph)\n",
                    options.repeat);
        for (const auto storage : options.storages) {
            for (const int context : options.contexts) {
                for (const int width : options.widths) {
                    Case data(storage, width, context);
                    try {
                        const auto total = time(flush, stream, options.warmup, options.repeat,
                                                [&](cudaStream_t s) { data.attention(s); });
                        const auto select = time(flush, stream, options.warmup, options.repeat,
                                                 [&](cudaStream_t s) { data.selection(s); });
                        std::printf("%-7s %7d %5d %10.2f %10.2f %13.2f\n", storage_name(storage), context, width,
                                    total.median_us, select.median_us, total.median_us - select.median_us);
                    } catch (const std::invalid_argument& e) {
                        std::printf("%-7s %7d %5d unsupported: %s\n", storage_name(storage), context, width,
                                    e.what());
                    }
                    std::fflush(stdout);
                }
            }
        }
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "infernix_qsa_attention_bench: %s\n", error.what());
        return 1;
    }
}
