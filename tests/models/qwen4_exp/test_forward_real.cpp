// Qwen4Exp forward on the real converted artifact, against the independent FP64 reference
// (tools/flash_next/reference.py). Skips unless NINFER_QWEN4_ARTIFACT names a Qwen4Exp artifact.
//
//   NINFER_QWEN4_ARTIFACT=out.ninfer [NINFER_QWEN4_NGRAM=out.ninfer.ngram]
//   ninfer_qwen4_exp_forward_real_test TOKENS [--kv KV] [--logits OUT.bin] [--residuals OUT.bin]
//       [--routes OUT.bin] [--blocks OUT.bin]
//   ninfer_qwen4_exp_forward_real_test TOKENS --dump-logits OUT.bin [--chunk N] [--kv KV]
//       [--dump-from P] [TOKENS --dump-logits OUT.bin [--chunk N] [--kv KV] [--dump-from P]]...
//   ninfer_qwen4_exp_forward_real_test TOKENS --cpu-columns [--kv KV]
//
// KV is a KV-cache storage: bf16 (default), int8, fp8, nvfp4 or k8v4.
//
// With --cpu-columns the test checks that calls the Program serves with the CPU expert service
// (decode widths, and prefill calls up to 255 columns with the assist) give the GPU route's logits
// bit for bit (check_cpu_columns).
// TOKENS is a comma-separated id list, or @FILE holding ids separated by commas or whitespace; the
// options after a TOKENS apply to it. With --dump-logits the test scores the text teacher-forced
// instead: it prefills it in chunks of N (default 256) with FP32 logits at every position, writes
// them in Strata's --dump-logits layout (int32 vocabulary, int32 rows, then one FP32 row per
// position) and prints the perplexity. With --dump-from P only positions from P on are written and
// scored (long texts: the logits of every position would not fit on disk). Several such texts are
// scored in order with one model load. --rope-yarn-factor F, anywhere, loads the model with that text
// YaRN factor (contexts up to the native positions times F).
// Otherwise the test prefills all but the last token as one chunk,
// then decodes the last token twice in one batch (two sequences with identical histories), and
// checks that both decode rows equal each other bit for bit and agree with a single prefill of
// every token at the last position. It writes the prefill logits of the last position (FP32) for
// comparison with the reference.

#include "artifact/reader.h"
#include "kv_cache_storage.h"
#include "core/device.h"
#include "core/host_memory.h"
#include "core/layout.h"
#include "core/linear_attention_state.h"
#include "core/paged_kv_cache.h"
#include "core/paged_kv_storage.h"
#include "models/qwen4_exp/execution/forward.h"
#include "models/qwen4_exp/execution/parameters.h"
#include "models/qwen4_exp/frontend/ngram_hash.h"
#include "models/qwen4_exp/load.h"
#include "models/qwen4_exp/program/prefix/state_image.h"
#include "models/qwen4_exp/program/ngram_volume.h"
#include "models/qwen4_exp/program/rope_positions.h"
#include "ops/offloaded_sparse_moe/cpu/miss_service.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#endif

using namespace ninfer;
namespace q4 = ninfer::models::qwen4_exp;

namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e)); }
}

std::vector<std::int32_t> parse_tokens(const std::string& argument) {
    std::string text = argument;
    if (!text.empty() && text.front() == '@') {
        std::ifstream file(text.substr(1));
        if (!file) { throw std::invalid_argument("cannot read " + text.substr(1)); }
        text.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    }
    for (char& ch : text) {
        if (ch == ',' || ch == '\n' || ch == '\r' || ch == '\t') { ch = ' '; }
    }
    std::vector<std::int32_t> out;
    std::stringstream stream(text);
    std::int32_t id = 0;
    while (stream >> id) { out.push_back(id); }
    return out;
}

// Host memory after each stage: the pinned weights are the bulk; the rest should stay small.
void print_memory(const char* stage) {
    constexpr double gib = 1073741824.0;
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX counters{};
    if (K32GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),
                                sizeof(counters))) {
        std::printf("[memory] %s: private %.2f GiB, working set %.2f GiB, system available %.2f GiB\n", stage,
                    counters.PrivateUsage / gib, counters.WorkingSetSize / gib, available_host_memory_bytes() / gib);
        return;
    }
#endif
    std::printf("[memory] %s: system available %.2f GiB\n", stage, available_host_memory_bytes() / gib);
}

double seconds_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

// Every slot's recurrent state, KV pages for `rows` sequences of up to `context` tokens, and no
// resident experts (every expert is read zero-copy from the pinned bank).
struct Harness {
    Harness(const q4::execution::Parameters& parameters, DeviceContext& device, std::int32_t context, std::int32_t columns,
            std::int32_t rows, KvCacheStorage storage)
        : config(parameters.model.config().text) {
        const auto& c = config;
        LayoutBuilder builder;
        const LinearAttentionStatePoolSpec gdn_spec{
            .layers         = c.gdn_layers,
            .conv_channels  = static_cast<std::int32_t>(c.gdn.conv_channels()),
            .conv_width     = static_cast<std::int32_t>(c.gdn.conv_kernel - 1),
            .value_heads    = static_cast<std::int32_t>(c.gdn.value_heads),
            .value_head_dim = static_cast<std::int32_t>(c.gdn.value_head_dim),
            .key_head_dim   = static_cast<std::int32_t>(c.gdn.key_head_dim),
            .slot_count     = rows,
        };
        const auto gdn_layout = plan_linear_attention_state_pool(builder, gdn_spec);
        state_bytes           = builder.finish(256);
        state_backing         = DeviceBuffer(state_bytes);
        state_backing.fill(0);
        gdn = std::make_unique<LinearAttentionStatePool>(DeviceSpan{state_backing.p, state_backing.bytes},
                                                         gdn_layout);

        const std::int32_t width = static_cast<std::int32_t>(c.residual_width());
        const std::int32_t span  = static_cast<std::int32_t>(c.ple.conv_span());
        ple_backing = DeviceBuffer(static_cast<std::size_t>(width) * span * rows * 2);
        ple_backing.fill(0);
        const std::int32_t di = static_cast<std::int32_t>(c.qsa.index_head_dim);
        const std::int32_t r  = static_cast<std::int32_t>(c.qsa.compress_ratio);
        tails_backing = DeviceBuffer(static_cast<std::size_t>(c.attention_layers) * di * (r - 1) * rows * 2);
        tails_backing.fill(0);

        const auto layout = paged_kv_storage_layout(storage, static_cast<std::int32_t>(c.attention.head_dim));
        KVPageGeometry geometry;
        for (std::uint32_t l = 0; l < c.attention_layers; ++l) {
            const auto kv_heads = static_cast<std::int32_t>(c.attention.kv_heads);
            geometry.planes.push_back({layout.key.data_dtype, layout.key.data_leading_extent, kv_heads});
            if (layout.key.has_scale()) {
                geometry.planes.push_back({layout.key.scale_dtype, layout.key.scale_leading_extent, kv_heads});
            }
            geometry.planes.push_back({layout.value.data_dtype, layout.value.data_leading_extent, kv_heads});
            if (layout.value.has_scale()) {
                geometry.planes.push_back({layout.value.scale_dtype, layout.value.scale_leading_extent, kv_heads});
            }
            geometry.planes.push_back({DType::BF16, di / r, 1});
        }
        pages_per_row = (context + kPagedKVPageSize - 1) / kPagedKVPageSize;
        LayoutBuilder kv_builder;
        const auto pool_layout = plan_device_kv_page_pool(
            kv_builder, {.page_group_count = static_cast<std::uint32_t>(pages_per_row * rows), .geometry = geometry});
        const auto table_layout = plan_kv_execution_tables(
            kv_builder, {.logical_page_capacity = static_cast<std::uint32_t>(pages_per_row), .table_rows = rows});
        kv_backing = DeviceBuffer(kv_builder.finish(256));
        kv_backing.fill(0);
        pool   = std::make_unique<DeviceKVPagePool>(DeviceSpan{kv_backing.p, kv_backing.bytes}, pool_layout);
        tables = std::make_unique<KVExecutionTablePool>(DeviceSpan{kv_backing.p, kv_backing.bytes}, table_layout,
                                                        *pool);
        for (std::int32_t row = 0; row < rows; ++row) {
            auto reservation = pool->reserve(static_cast<std::uint32_t>(pages_per_row));
            if (!reservation) { throw std::runtime_error("KV pool too small"); }
            std::vector<DeviceKVPageLease> pages;
            pages.reserve(static_cast<std::size_t>(pages_per_row));
            pool->materialize(*reservation, static_cast<std::uint32_t>(pages_per_row), pages);
            row_leases.push_back(tables->acquire(row));
            tables->publish(row_leases.back().handle(), 0, pages, device.stream);
            page_leases.push_back(std::move(pages));
            reservations.push_back(std::move(*reservation));
        }
        device.synchronize();

        frames_backing = DeviceBuffer(sizeof(std::int32_t) * c.moe.experts * c.num_hidden_layers);
        frames_backing.fill(0xFF); // -1: every expert served from the pinned host bank

        q4::execution::ForwardState state;
        state.gdn      = gdn.get();
        state.ple_conv = Tensor(ple_backing.p, DType::BF16, {width, span, rows});
        for (std::uint32_t l = 0; l < c.attention_layers; ++l) {
            state.qsa_tails.push_back(Tensor(static_cast<std::byte*>(tails_backing.p) +
                                                 static_cast<std::size_t>(l) * di * (r - 1) * rows * 2,
                                             DType::BF16, {di, r - 1, rows}));
        }
        // vq2/k4v2: the exact window of every attention layer and row, zeroed (no tag matches).
        window_geometry = q4::prefix::kv_window_geometry(c, storage, false, static_cast<std::uint32_t>(rows));
        if (window_geometry.present()) {
            window_backing = DeviceBuffer(window_geometry.bytes());
            window_backing.fill(0);
        }
        q4::execution::ForwardKV kv;
        kv.block_tables = tables->matrix();
        std::size_t plane   = 0;
        const auto kv_heads = static_cast<std::int32_t>(c.attention.kv_heads);
        for (std::uint32_t l = 0; l < c.attention_layers; ++l) {
            ops::QsaKVLayer layer;
            layer.kv.storage      = storage;
            layer.kv.head_dim     = static_cast<std::int32_t>(c.attention.head_dim);
            layer.kv.num_kv_heads = kv_heads;
            layer.kv.k_pages      = pool->plane(plane++);
            if (layout.key.has_scale()) { layer.kv.k_scale_pages = pool->plane(plane++); }
            layer.kv.v_pages = pool->plane(plane++);
            if (layout.value.has_scale()) { layer.kv.v_scale_pages = pool->plane(plane++); }
            layer.pooled_pages = pool->plane(plane++);
            if (window_geometry.present()) {
                layer.kv.window = q4::prefix::kv_window_view(window_geometry, window_backing.p, l);
            }
            kv.layers.push_back(layer);
        }
        q4::execution::ForwardExperts experts;
        for (std::uint32_t l = 0; l < c.num_hidden_layers; ++l) {
            experts.frames.push_back(static_cast<const std::int32_t*>(frames_backing.p) + l * c.moe.experts);
        }
        work_capacity = q4::execution::Forward::workspace_bytes(c, columns, context, storage);
        work          = std::make_unique<WorkspaceArena>(work_capacity);
        state_view    = state;
        kv_view       = kv;
        experts_view  = experts;
        forward = std::make_unique<q4::execution::Forward>(parameters, device, *work, std::move(state), std::move(kv),
                                                           std::move(experts), context);
    }

    // Another Forward over the same state, KV and workspace, with other expert sources.
    std::unique_ptr<q4::execution::Forward>
    forward_with(const q4::execution::Parameters& parameters, DeviceContext& device,
                 q4::execution::ForwardExperts sources, std::int32_t context) const {
        return std::make_unique<q4::execution::Forward>(parameters, device, *work, state_view,
                                                        kv_view, std::move(sources), context);
    }

    const q4::TextConfig& config;
    q4::execution::ForwardState state_view;
    q4::execution::ForwardKV kv_view;
    q4::execution::ForwardExperts experts_view;
    std::size_t state_bytes = 0;
    DeviceBuffer state_backing, ple_backing, tails_backing, kv_backing, frames_backing;
    DeviceBuffer window_backing;
    q4::prefix::KvWindowGeometry window_geometry;
    std::unique_ptr<LinearAttentionStatePool> gdn;
    std::unique_ptr<DeviceKVPagePool> pool;
    std::unique_ptr<KVExecutionTablePool> tables;
    std::vector<DeviceKVPageReservation> reservations;
    std::vector<std::vector<DeviceKVPageLease>> page_leases;
    std::vector<KVExecutionRowLease> row_leases;
    std::int32_t pages_per_row = 0;
    std::size_t work_capacity  = 0;
    std::unique_ptr<WorkspaceArena> work;
    std::unique_ptr<q4::execution::Forward> forward;
};

// Device copies of one call's inputs.
struct Call {
    DeviceBuffer buffer;
    q4::execution::ForwardBatch batch;
    std::vector<std::int32_t> host_slots, host_rows;
};

Call make_call(const q4::TextConfig& c, const q4::NgramVolume& volume, const std::vector<std::vector<std::int32_t>>& histories,
               std::int32_t first, std::int32_t width, std::vector<std::int32_t> slots, std::vector<std::int32_t> rows,
               bool every_column = false) {
    const auto batch = static_cast<std::int32_t>(histories.size());
    const std::int32_t columns = batch * width;
    const q4::NgramHash hash(c.ple.ngram, 0);
    const std::size_t row_bytes = c.ple.table.row_bytes;
    const std::size_t heads     = hash.heads();
    std::vector<std::int32_t> ids, positions, last;
    // Text prompts: RoPE positions equal the index on all three axes.
    std::vector<std::int32_t> rope(3 * static_cast<std::size_t>(columns)), block_rope(3 * static_cast<std::size_t>(batch));
    const q4::LaneRope text;
    std::vector<std::byte> ngram(columns * heads * row_bytes);
    for (std::int32_t b = 0; b < batch; ++b) {
        const auto& history = histories[b];
        std::vector<std::int32_t> padded(c.ple.ngram.ngram_size - 1, c.eos_token_id);
        padded.insert(padded.end(), history.begin(), history.begin() + first + width);
        std::vector<std::uint32_t> row_ids(static_cast<std::size_t>(width) * heads);
        hash.row_ids(padded, static_cast<std::size_t>(width), row_ids.data());
        volume.read_rows(row_ids, std::span<std::byte>(ngram).subspan(static_cast<std::size_t>(b) * width * heads * row_bytes,
                                                                    static_cast<std::size_t>(width) * heads * row_bytes));
        for (std::int32_t i = 0; i < width; ++i) {
            ids.push_back(history[first + i]);
            positions.push_back(first + i);
        }
        q4::stage_rope(text, static_cast<std::uint32_t>(first), width, rope, columns, b * width);
        q4::stage_rope_value(q4::block_start_rope(text, static_cast<std::uint32_t>(first), c.qsa.compress_ratio),
                             block_rope, batch, b);
        if (every_column) {
            for (std::int32_t i = 0; i < width; ++i) { last.push_back(b * width + i); }
        } else {
            last.push_back(b * width + width - 1);
        }
    }
    Call call;
    const std::size_t i32 = sizeof(std::int32_t);
    const auto padded_bytes = [](std::size_t n) { return (n + 255) / 256 * 256; };
    const std::size_t bytes = padded_bytes(ids.size() * i32) + padded_bytes(positions.size() * i32) +
                              padded_bytes(rope.size() * i32) + padded_bytes(block_rope.size() * i32) +
                              padded_bytes(slots.size() * i32) + padded_bytes(rows.size() * i32) +
                              padded_bytes(last.size() * i32) + padded_bytes(ngram.size());
    call.buffer = DeviceBuffer(bytes);
    auto* base = static_cast<std::byte*>(call.buffer.p);
    std::size_t offset = 0;
    auto put = [&](const void* data, std::size_t n) {
        call.buffer.copy_from_host(data, n, offset);
        void* p = base + offset;
        offset += (n + 255) / 256 * 256;
        return p;
    };
    call.batch.ids        = Tensor(put(ids.data(), ids.size() * i32), DType::I32, {columns});
    call.batch.positions  = Tensor(put(positions.data(), positions.size() * i32), DType::I32, {columns});
    call.batch.rope_positions   = Tensor(put(rope.data(), rope.size() * i32), DType::I32, {columns, 3});
    call.batch.block_start_rope = Tensor(put(block_rope.data(), block_rope.size() * i32), DType::I32, {batch, 3});
    call.batch.slots      = Tensor(put(slots.data(), slots.size() * i32), DType::I32, {batch});
    call.batch.table_rows = Tensor(put(rows.data(), rows.size() * i32), DType::I32, {batch});
    call.batch.logit_columns =
        Tensor(put(last.data(), last.size() * i32), DType::I32, {static_cast<std::int32_t>(last.size())});
    call.batch.ngram_rows = Tensor(put(ngram.data(), ngram.size()), DType::U8,
                                   {static_cast<std::int32_t>(row_bytes), static_cast<std::int32_t>(heads), columns});
    call.host_slots       = std::move(slots);
    call.host_rows        = std::move(rows);
    call.batch.host_slots = call.host_slots;
    call.batch.host_table_rows = call.host_rows;
    call.batch.batch = batch;
    call.batch.width = width;
    return call;
}

std::vector<float> to_float(const DeviceBuffer& logits, std::size_t count) {
    std::vector<float> out(count);
    logits.copy_to_host(out.data(), count * sizeof(float));
    return out;
}

std::vector<int> top(const std::vector<float>& v, std::size_t offset, std::size_t n, int k) {
    std::vector<int> index(n);
    std::iota(index.begin(), index.end(), 0);
    std::partial_sort(index.begin(), index.begin() + k, index.end(),
                      [&](int a, int b) { return v[offset + a] > v[offset + b]; });
    index.resize(k);
    return index;
}

// The Program CPU-serves the misses of every call of at most kMaxCpuColumns columns (short prompts,
// chunk remainders, forced tokens) and, with the prefill assist (design §19.3.1 P7), the thinnest
// misses of prefill calls up to 255 columns. A CPU-served expert has at most kMaxCpuColumns columns,
// so its output is the same bits on the CPU and the GPU (design §16.2), and so are the call's
// logits. Two sequences run the same prefix, then the last tokens as calls of 1, 2, 5, 8, 24 and 100
// columns, through the Program's MoE configuration (64 staging slots, the prefill overlap stream;
// no resident experts here), one of them with a CPU expert service configured as the Program's
// (decode cap 32, assist cap 256 from 9 columns) as well.
bool check_cpu_columns(const q4::execution::Parameters& parameters, DeviceContext& device,
                       const q4::NgramVolume& volume, const std::vector<std::int32_t>& tokens,
                       std::int32_t context, KvCacheStorage kv) {
    namespace moe               = ninfer::ops::offloaded_moe;
    const auto& c               = parameters.model.config().text;
    const auto n                = static_cast<std::int32_t>(tokens.size());
    const std::int32_t widths[] = {1, 2, 5, moe::kMaxCpuColumns, 24, 100};
    const std::int32_t tail     = 1 + 2 + 5 + moe::kMaxCpuColumns + 24 + 100;
    if (n <= tail + moe::kMaxCpuColumns) {
        throw std::invalid_argument("--cpu-columns needs more than 148 tokens");
    }
    const std::int32_t prefix = n - tail;
    Harness harness(parameters, device, context, prefix, 2, kv);
    std::uint64_t record_stride = 0;
    std::vector<moe::CpuMissService::Layer> layers;
    for (const auto& layer : parameters.layers) {
        record_stride = layer.moe.bank->planes.record_stride;
        layers.push_back(
            {.records       = reinterpret_cast<const std::uint8_t*>(layer.moe.bank->planes.records),
             .record_stride = record_stride,
             .scales        = layer.moe.bank->scales.data()});
    }
    constexpr std::int32_t kStagingSlots = 64;
    DeviceBuffer staging(static_cast<std::size_t>(kStagingSlots) * record_stride);
    cudaStream_t overlap = nullptr;
    std::array<cudaEvent_t, 5> events{};
    check(cudaStreamCreateWithFlags(&overlap, cudaStreamNonBlocking), "cudaStreamCreate");
    for (auto& event : events) {
        check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming), "cudaEventCreate");
    }
    bool ok = true;
    {
        constexpr std::int32_t kAssistColumns = 255;
        moe::CpuMissService service(layers, {.workers      = 6,
                                             .max_jobs     = 32,
                                             .max_columns  = kAssistColumns,
                                             .pcie_divisor = 3,
                                             .wide_from    = moe::kMaxCpuColumns + 1,
                                             .wide_jobs    = moe::kMaxCpuJobs,
                                             .cpus         = {}});
        auto sources           = harness.experts_view;
        sources.staging_base   = static_cast<std::uint8_t*>(staging.p);
        sources.staging_slots  = kStagingSlots;
        sources.overlap_stream = overlap;
        sources.overlap_events = events;
        const auto gpu         = harness.forward_with(parameters, device, sources, context);
        for (std::uint32_t l = 0; l < c.num_hidden_layers; ++l) {
            sources.cpu.push_back(service.channel(static_cast<int>(l)));
        }
        const auto cpu   = harness.forward_with(parameters, device, sources, context);
        const auto vocab = static_cast<std::int32_t>(c.vocab_size);
        DeviceBuffer logits(static_cast<std::size_t>(vocab) * moe::kMaxCpuColumns * sizeof(float));
        const auto run = [&](q4::execution::Forward& forward, std::int32_t slot, std::int32_t first,
                             std::int32_t width) {
            const bool every = width <= moe::kMaxCpuColumns;
            auto call        = make_call(c, volume, {tokens}, first, width, {slot}, {slot}, every);
            Tensor out(logits.p, DType::FP32, {vocab, every ? width : 1});
            forward.run(call.batch, out);
            device.synchronize();
            return to_float(logits, static_cast<std::size_t>(vocab) * (every ? width : 1));
        };
        (void)run(*gpu, 0, 0, prefix);
        (void)run(*cpu, 1, 0, prefix);
        if (prefix > kAssistColumns && service.served_experts() != 0) {
            std::printf("FAIL: a %d-column call reached the CPU service\n", prefix);
            ok = false;
        }
        std::int32_t first = prefix;
        for (const std::int32_t width : widths) {
            const auto before   = service.served_experts();
            const auto expected = run(*gpu, 0, first, width);
            const auto served   = run(*cpu, 1, first, width);
            const auto experts  = service.served_experts() - before;
            const bool same     = expected == served;
            std::printf("%d-column call at position %d: %llu experts CPU-served, logits %s the GPU "
                        "route's\n",
                        width, first, static_cast<unsigned long long>(experts),
                        same ? "equal" : "DIFFER FROM");
            ok = ok && same && experts > 0;
            first += width;
        }
    }
    for (auto event : events) { cudaEventDestroy(event); }
    cudaStreamDestroy(overlap);
    std::printf(ok ? "CPU-served calls equal the GPU route bit for bit\n"
                   : "FAIL: CPU-served calls differ from the GPU route\n");
    return ok;
}

} // namespace

int main(int argc, char** argv) {
    const char* artifact_env = std::getenv("NINFER_QWEN4_ARTIFACT");
    if (artifact_env == nullptr || argc < 2) {
        std::printf("SKIP: set NINFER_QWEN4_ARTIFACT and pass a token list\n");
        return 77;
    }
    // Each token list takes the options that follow it.
    struct Job {
        std::vector<std::int32_t> tokens;
        std::string logits_path, residuals_path, routes_path, blocks_path, dump_path;
        std::int32_t chunk     = 256;
        std::int32_t dump_from = 0;
        KvCacheStorage kv      = KvCacheStorage::BFloat16;
        bool cpu_columns   = false;
    };
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    try {
        std::vector<Job> jobs;
        models::LoadOptions load; // --rope-yarn-factor F (anywhere): the model's YaRN factor
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--rope-yarn-factor") {
                if (i + 1 >= argc) { throw std::invalid_argument("--rope-yarn-factor needs a value"); }
                load.rope_yarn_factor = std::stof(argv[++i]);
                continue;
            }
            if (arg.rfind("--", 0) != 0) {
                jobs.push_back(Job{parse_tokens(arg)});
                if (jobs.back().tokens.size() < 2) { throw std::invalid_argument("pass at least two tokens"); }
                continue;
            }
            if (arg == "--cpu-columns") {
                if (jobs.empty()) { throw std::invalid_argument("--cpu-columns needs a token list before it"); }
                jobs.back().cpu_columns = true;
                continue;
            }
            if (jobs.empty() || i + 1 >= argc) { throw std::invalid_argument(arg + " needs a token list before it and a value"); }
            const std::string value = argv[++i];
            Job& job = jobs.back();
            if (arg == "--logits") {
                job.logits_path = value;
            } else if (arg == "--residuals") {
                job.residuals_path = value;
            } else if (arg == "--routes") {
                job.routes_path = value;
            } else if (arg == "--blocks") {
                job.blocks_path = value;
            } else if (arg == "--dump-logits") {
                job.dump_path = value;
            } else if (arg == "--chunk") {
                job.chunk = std::stoi(value);
                if (job.chunk <= 0) { throw std::invalid_argument("--chunk must be positive"); }
            } else if (arg == "--dump-from") {
                job.dump_from = std::stoi(value);
                if (job.dump_from < 0) { throw std::invalid_argument("--dump-from must not be negative"); }
            } else if (arg == "--kv") {
                job.kv = ninfer::test::parse_kv_cache_storage(value);
            } else {
                throw std::invalid_argument("unknown option " + arg);
            }
        }
        const bool scoring = !jobs.front().dump_path.empty();
        for (const Job& job : jobs) {
            if (job.dump_path.empty() != !scoring || (jobs.size() > 1 && job.dump_path.empty())) {
                throw std::invalid_argument("several token lists are scored only, each with its own --dump-logits");
            }
        }
        print_memory("start");
        DeviceContext device(0);
        print_memory("device context");
        auto t0 = std::chrono::steady_clock::now();
        artifact::Reader reader(artifact_env);
        if (!q4::is_qwen4_exp(reader)) { throw std::invalid_argument("artifact is not Qwen4Exp"); }
        auto plan  = q4::plan_load(reader, load);
        print_memory("load plan");
        auto model = q4::materialize_model(std::move(plan), device);
        device.synchronize();
        const auto& stats = model->storage_stats();
        std::printf("loaded in %.1f s: %.2f GiB device, %.2f GiB pinned host\n", seconds_since(t0),
                    stats.device_capacity_bytes / 1073741824.0, stats.pinned_bytes / 1073741824.0);
        print_memory("materialized");
        const q4::execution::Parameters parameters(*model);
        print_memory("parameters");
        const auto& c = model->config().text;
        const char* ngram_env = std::getenv("NINFER_QWEN4_NGRAM");
        const q4::NgramVolume volume(ngram_env != nullptr ? std::filesystem::path(ngram_env)
                                                          : q4::default_ngram_volume(artifact_env),
                                     c.ple.table);
        const std::size_t vocab = c.vocab_size;
        if (scoring) {
            // Teacher-forced scoring: chunked prefill of each whole text in slot 0 of its own harness.
            for (const Job& job : jobs) {
                const auto& tokens          = job.tokens;
                const auto n                = static_cast<std::int32_t>(tokens.size());
                const std::int32_t context  = std::max<std::int32_t>(256, (n + 63) / 64 * 64);
                const std::int32_t chunk    = std::min(job.chunk, n);
                Harness harness(parameters, device, context, chunk, 1, job.kv);
                DeviceBuffer logits(vocab * chunk * sizeof(float));
                std::ofstream out(job.dump_path, std::ios::binary);
                const std::int32_t from      = std::min(job.dump_from, n);
                const std::int32_t header[2] = {static_cast<std::int32_t>(vocab), n - from};
                out.write(reinterpret_cast<const char*>(header), sizeof(header));
                double nll = 0.0;
                std::int32_t scored = 0, same_top1 = 0;
                t0 = std::chrono::steady_clock::now();
                for (std::int32_t first = 0; first < n; first += chunk) {
                    const std::int32_t width = std::min(chunk, n - first);
                    auto call = make_call(c, volume, {tokens}, first, width, {0}, {0}, true);
                    Tensor chunk_logits(logits.p, DType::FP32, {static_cast<std::int32_t>(vocab), width});
                    harness.forward->run(call.batch, chunk_logits);
                    device.synchronize();
                    if (first + width <= from) { continue; }
                    const auto rows       = to_float(logits, vocab * width);
                    const std::int32_t lo = std::max(0, from - first);
                    out.write(reinterpret_cast<const char*>(rows.data() + static_cast<std::size_t>(lo) * vocab),
                              static_cast<std::streamsize>((rows.size() - static_cast<std::size_t>(lo) * vocab) *
                                                           sizeof(float)));
                    for (std::int32_t i = lo; i < width && first + i + 1 < n; ++i) {
                        const float* row = rows.data() + static_cast<std::size_t>(i) * vocab;
                        const float peak = *std::max_element(row, row + vocab);
                        double sum = 0.0;
                        for (std::size_t v = 0; v < vocab; ++v) { sum += std::exp(double(row[v]) - peak); }
                        const std::int32_t next = tokens[first + i + 1];
                        nll -= double(row[next]) - peak - std::log(sum);
                        same_top1 += std::max_element(row, row + vocab) - row == next;
                        ++scored;
                    }
                }
                if (!out) { throw std::runtime_error("cannot write " + job.dump_path); }
                std::printf("%s: scored %d positions in %.1f s: mean NLL %.5f nats, perplexity %.4f, top-1 = next %.1f%%\n",
                            job.dump_path.c_str(), scored, seconds_since(t0), nll / scored, std::exp(nll / scored),
                            100.0 * same_top1 / scored);
            }
            return 0;
        }
        const Job& job = jobs.front();
        const auto& tokens = job.tokens;
        const auto n = static_cast<std::int32_t>(tokens.size());
        const std::int32_t context = std::max<std::int32_t>(256, (n + 63) / 64 * 64);
        const std::string& logits_path    = job.logits_path;
        const std::string& residuals_path = job.residuals_path;
        const std::string& routes_path    = job.routes_path;
        const std::string& blocks_path    = job.blocks_path;
        if (job.cpu_columns) {
            return check_cpu_columns(parameters, device, volume, tokens, context, job.kv) ? 0 : 1;
        }
        Harness harness(parameters, device, context, n, 3, job.kv);
        DeviceBuffer logits(vocab * 2 * sizeof(float));
        print_memory("harness");

        // 1. All tokens as one prefill chunk in slot/row 0: logits of the last position, and
        //    optionally every block's residual and routed experts.
        auto full = make_call(c, volume, {tokens}, 0, n, {0}, {0});
        Tensor full_logits(logits.p, DType::FP32, {static_cast<std::int32_t>(vocab), 1});
        const auto blocks = static_cast<std::size_t>(c.num_hidden_layers);
        const auto W = static_cast<std::int32_t>(c.residual_width());
        const auto K = static_cast<std::int32_t>(c.moe.top_k);
        const std::size_t residual_bytes = static_cast<std::size_t>(W) * n * 2;
        const std::size_t route_bytes    = static_cast<std::size_t>(K) * n * 4;
        const auto H = static_cast<std::int32_t>(c.hidden_size);
        const std::size_t io_bytes = static_cast<std::size_t>(H) * n * 2;
        DeviceBuffer taps(blocks * (residual_bytes + route_bytes + 4 * io_bytes));
        std::vector<Tensor> residual_taps, route_taps, io_taps[4];
        auto* base = static_cast<std::byte*>(taps.p);
        const std::size_t io_offset = blocks * (residual_bytes + route_bytes);
        for (std::size_t b = 0; b < blocks; ++b) {
            residual_taps.emplace_back(base + b * residual_bytes, DType::BF16, std::initializer_list<std::int32_t>{W, n});
            route_taps.emplace_back(base + blocks * residual_bytes + b * route_bytes, DType::I32,
                                    std::initializer_list<std::int32_t>{K, n});
            for (std::size_t kind = 0; kind < 4; ++kind) {
                io_taps[kind].emplace_back(base + io_offset + (b * 4 + kind) * io_bytes, DType::BF16,
                                           std::initializer_list<std::int32_t>{H, n});
            }
        }
        q4::execution::ForwardTap tap{&residual_taps, &route_taps, &io_taps[0], &io_taps[1], &io_taps[2], &io_taps[3]};
        t0 = std::chrono::steady_clock::now();
        harness.forward->run(full.batch, full_logits, &tap);
        device.synchronize();
        const auto dump = [&](const std::string& path, std::size_t offset, std::size_t bytes) {
            if (path.empty()) { return; }
            std::vector<std::byte> host(bytes);
            taps.copy_to_host(host.data(), bytes, offset);
            std::ofstream file(path, std::ios::binary);
            file.write(reinterpret_cast<const char*>(host.data()), static_cast<std::streamsize>(bytes));
        };
        dump(residuals_path, 0, blocks * residual_bytes);
        dump(routes_path, blocks * residual_bytes, blocks * route_bytes);
        dump(blocks_path, io_offset, blocks * 4 * io_bytes); // [blocks][mixer in, mixer out, moe in, moe out][T][H]
        std::printf("prefill %d tokens: %.1f ms\n", n, seconds_since(t0) * 1e3);
        print_memory("prefill");
        const auto prefill = to_float(logits, vocab);
        const auto best = top(prefill, 0, vocab, 5);
        std::printf("prefill top-5 at position %d:", n - 1);
        for (int id : best) { std::printf(" %d (%.3f)", id, prefill[id]); }
        std::printf("\n");
        if (!logits_path.empty()) {
            std::ofstream file(logits_path, std::ios::binary);
            file.write(reinterpret_cast<const char*>(prefill.data()), static_cast<std::streamsize>(vocab * sizeof(float)));
        }

        // 2. All but the last token in slots/rows 1 and 2, then one batched decode step of both.
        for (std::int32_t slot : {1, 2}) {
            auto head = make_call(c, volume, {tokens}, 0, n - 1, {slot}, {slot});
            Tensor head_logits(logits.p, DType::FP32, {static_cast<std::int32_t>(vocab), 1});
            harness.forward->run(head.batch, head_logits);
        }
        auto step = make_call(c, volume, {tokens, tokens}, n - 1, 1, {1, 2}, {1, 2});
        Tensor step_logits(logits.p, DType::FP32, {static_cast<std::int32_t>(vocab), 2});
        t0 = std::chrono::steady_clock::now();
        harness.forward->run(step.batch, step_logits);
        device.synchronize();
        std::printf("batched decode of 2 sequences: %.1f ms\n", seconds_since(t0) * 1e3);
        const auto decode = to_float(logits, 2 * vocab);
        std::size_t rows_differ = 0;
        double max_diff = 0;
        for (std::size_t i = 0; i < vocab; ++i) {
            rows_differ += decode[i] != decode[vocab + i];
            max_diff = std::max(max_diff, std::abs(double(decode[i]) - prefill[i]));
        }
        const auto decode_best = top(decode, 0, vocab, 5);
        std::printf("decode top-5:");
        for (int id : decode_best) { std::printf(" %d (%.3f)", id, decode[id]); }
        std::printf("\ndecode rows differing between identical sequences: %zu; max |decode - prefill| = %.4f\n",
                    rows_differ, max_diff);
        const bool ok = rows_differ == 0 && decode_best.front() == best.front();
        std::printf(ok ? "forward checks passed\n" : "FAIL: decode disagrees with prefill\n");
        return ok ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
