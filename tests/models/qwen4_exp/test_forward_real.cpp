// Qwen4Exp forward on the real converted artifact, against the independent FP64 reference
// (tools/flash_next/reference.py). Skips unless NINFER_QWEN4_ARTIFACT names a Qwen4Exp artifact.
//
//   NINFER_QWEN4_ARTIFACT=out.ninfer [NINFER_QWEN4_NGRAM=out.ninfer.ngram]
//   ninfer_qwen4_exp_forward_real_test TOKENS [--logits OUT.bin] [--residuals OUT.bin]
//
// TOKENS is a comma-separated id list. The test prefills all but the last token as one chunk,
// then decodes the last token twice in one batch (two sequences with identical histories), and
// checks that both decode rows equal each other bit for bit and agree with a single prefill of
// every token at the last position. It writes the prefill logits of the last position (FP32) for
// comparison with the reference.

#include "artifact/reader.h"
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
#include "models/qwen4_exp/program/ngram_volume.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
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

std::vector<std::int32_t> parse_tokens(const std::string& text) {
    std::vector<std::int32_t> out;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (!item.empty()) { out.push_back(std::stoi(item)); }
    }
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
    Harness(const q4::execution::Parameters& parameters, DeviceContext& device, std::int32_t context,
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
            kv.layers.push_back(layer);
        }
        q4::execution::ForwardExperts experts;
        for (std::uint32_t l = 0; l < c.num_hidden_layers; ++l) {
            experts.frames.push_back(static_cast<const std::int32_t*>(frames_backing.p) + l * c.moe.experts);
        }
        work_capacity = q4::execution::Forward::workspace_bytes(c, context, context);
        work          = std::make_unique<WorkspaceArena>(work_capacity);
        forward = std::make_unique<q4::execution::Forward>(parameters, device, *work, std::move(state), std::move(kv),
                                                           std::move(experts), context);
    }

    const q4::TextConfig& config;
    std::size_t state_bytes = 0;
    DeviceBuffer state_backing, ple_backing, tails_backing, kv_backing, frames_backing;
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
               std::int32_t first, std::int32_t width, std::vector<std::int32_t> slots, std::vector<std::int32_t> rows) {
    const auto batch = static_cast<std::int32_t>(histories.size());
    const std::int32_t columns = batch * width;
    const q4::NgramHash hash(c.ple.ngram, 0);
    const std::size_t row_bytes = c.ple.table.row_bytes;
    const std::size_t heads     = hash.heads();
    std::vector<std::int32_t> ids, positions, last;
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
        last.push_back(b * width + width - 1);
    }
    Call call;
    const std::size_t i32 = sizeof(std::int32_t);
    const auto padded_bytes = [](std::size_t n) { return (n + 255) / 256 * 256; };
    const std::size_t bytes = padded_bytes(ids.size() * i32) + padded_bytes(positions.size() * i32) +
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
    call.batch.slots      = Tensor(put(slots.data(), slots.size() * i32), DType::I32, {batch});
    call.batch.table_rows = Tensor(put(rows.data(), rows.size() * i32), DType::I32, {batch});
    call.batch.last_columns = Tensor(put(last.data(), last.size() * i32), DType::I32, {batch});
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
    std::vector<__nv_bfloat16> raw(count);
    logits.copy_to_host(raw.data(), count * sizeof(__nv_bfloat16));
    std::vector<float> out(count);
    for (std::size_t i = 0; i < count; ++i) { out[i] = __bfloat162float(raw[i]); }
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

} // namespace

int main(int argc, char** argv) {
    const char* artifact_env = std::getenv("NINFER_QWEN4_ARTIFACT");
    if (artifact_env == nullptr || argc < 2) {
        std::printf("SKIP: set NINFER_QWEN4_ARTIFACT and pass a token list\n");
        return 77;
    }
    std::string logits_path, residuals_path, routes_path, blocks_path;
    for (int i = 2; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == "--logits") { logits_path = argv[i + 1]; }
        if (std::string(argv[i]) == "--residuals") { residuals_path = argv[i + 1]; }
        if (std::string(argv[i]) == "--routes") { routes_path = argv[i + 1]; }
        if (std::string(argv[i]) == "--blocks") { blocks_path = argv[i + 1]; }
    }
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    try {
        const auto tokens = parse_tokens(argv[1]);
        if (tokens.size() < 2) { throw std::invalid_argument("pass at least two tokens"); }
        print_memory("start");
        DeviceContext device(0);
        print_memory("device context");
        auto t0 = std::chrono::steady_clock::now();
        artifact::Reader reader(artifact_env);
        if (!q4::is_qwen4_exp(reader)) { throw std::invalid_argument("artifact is not Qwen4Exp"); }
        auto plan  = q4::plan_load(reader, models::LoadOptions{});
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
        const auto n = static_cast<std::int32_t>(tokens.size());
        const std::int32_t context = std::max<std::int32_t>(256, (n + 63) / 64 * 64);
        Harness harness(parameters, device, context, 3, KvCacheStorage::BFloat16);
        const std::size_t vocab = c.vocab_size;
        DeviceBuffer logits(vocab * 2 * sizeof(__nv_bfloat16));
        print_memory("harness");

        // 1. All tokens as one prefill chunk in slot/row 0: logits of the last position, and
        //    optionally every block's residual and routed experts.
        auto full = make_call(c, volume, {tokens}, 0, n, {0}, {0});
        Tensor full_logits(logits.p, DType::BF16, {static_cast<std::int32_t>(vocab), 1});
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
            Tensor head_logits(logits.p, DType::BF16, {static_cast<std::int32_t>(vocab), 1});
            harness.forward->run(head.batch, head_logits);
        }
        auto step = make_call(c, volume, {tokens, tokens}, n - 1, 1, {1, 2}, {1, 2});
        Tensor step_logits(logits.p, DType::BF16, {static_cast<std::int32_t>(vocab), 2});
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
