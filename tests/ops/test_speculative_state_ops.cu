// State Ops of Qwen3.8-Flash-Next speculative verification (design §11): a verification call
// leaves every state unchanged and a commit later applies an accepted prefix. Each Op is checked
// bit for bit against the committing form of the same arithmetic over the committed columns:
//   - causal_conv1d_silu_from_states: outputs equal the snapshot form's, and no state changes;
//   - ple_conv_inject without destination slots: residual updates equal the committing call's,
//     and no state changes; ple_conv_commit of n columns equals ple_conv_inject over those n;
//   - qsa_pool_keys without tail updates: pooled keys equal the committing call's, and the tails
//     do not change; qsa_commit_tails of n columns equals qsa_pool_keys over those n.
#include "ninfer/ops/causal_conv1d_silu.h"
#include "ninfer/ops/ple.h"
#include "ninfer/ops/qsa.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <utility>
#include <vector>

using ninfer::DType;
using ninfer::Tensor;
using ninfer::test::cuda_check;

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++g_failures;
    }
}

template <class T> struct Buffer {
    T* p = nullptr;
    std::size_t n = 0;
    explicit Buffer(std::size_t count) : n(count) { cuda_check(cudaMalloc(&p, sizeof(T) * count), "cudaMalloc"); }
    Buffer(const std::vector<T>& v) : Buffer(v.size()) { upload(v); }
    ~Buffer() { cudaFree(p); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void upload(const std::vector<T>& v) {
        cuda_check(cudaMemcpy(p, v.data(), sizeof(T) * v.size(), cudaMemcpyHostToDevice), "cudaMemcpy");
    }
    std::vector<T> download() const {
        std::vector<T> v(n);
        cuda_check(cudaMemcpy(v.data(), p, sizeof(T) * n, cudaMemcpyDeviceToHost), "cudaMemcpy");
        return v;
    }
};

std::vector<std::uint16_t> random_bf16(std::mt19937& rng, std::size_t n, float scale) {
    std::normal_distribution<float> d(0.0F, scale);
    std::vector<std::uint16_t> v(n);
    for (auto& x : v) {
        float f = d(rng);
        std::uint32_t bits;
        std::memcpy(&bits, &f, 4);
        x = static_cast<std::uint16_t>((bits + 0x7FFFU + ((bits >> 16) & 1U)) >> 16);
    }
    return v;
}

// ----------------------------------------------------------------------------- convolution

void test_conv(std::int32_t C, std::int32_t W, std::int32_t B) {
    std::mt19937 rng(static_cast<std::uint32_t>(C * 131 + W * 7 + B));
    const std::int32_t slots = W * B + B;
    const auto x       = random_bf16(rng, static_cast<std::size_t>(C) * W * B, 1.0F);
    const auto weight  = random_bf16(rng, static_cast<std::size_t>(C) * 4, 0.5F);
    const auto states  = random_bf16(rng, static_cast<std::size_t>(C) * 3 * slots, 1.0F);
    std::vector<std::int32_t> initial(B), bases(B);
    for (std::int32_t b = 0; b < B; ++b) {
        initial[b] = W * B + b; // live slots after the snapshot reservations
        bases[b]   = b * W;
    }
    Buffer<std::uint16_t> dx(x), dw(weight), ds_ro(states), ds_snap(states), out_ro(x.size()), out_snap(x.size());
    Buffer<std::int32_t> dinit(initial), dbase(bases);
    Tensor tx(dx.p, DType::BF16, {C, W, B}), tw(dw.p, DType::BF16, {C, 4});
    Tensor tro(ds_ro.p, DType::BF16, {C, 3, slots}), tsnap(ds_snap.p, DType::BF16, {C, 3, slots});
    Tensor to_ro(out_ro.p, DType::BF16, {C, W, B}), to_snap(out_snap.p, DType::BF16, {C, W, B});
    Tensor tinit(dinit.p, DType::I32, {B}), tbase(dbase.p, DType::I32, {B});
    ninfer::ops::causal_conv1d_silu_from_states(tx, tw, tro, tinit, to_ro, nullptr);
    ninfer::ops::causal_conv1d_silu_snapshot(tx, tw, tsnap, Tensor{}, tinit, tbase, to_snap, nullptr);
    cuda_check(cudaDeviceSynchronize(), "conv");
    const std::string tag = "conv C=" + std::to_string(C) + " W=" + std::to_string(W) + " B=" + std::to_string(B);
    check(out_ro.download() == out_snap.download(), tag + ": from-states outputs equal the snapshot form's");
    check(ds_ro.download() == states, tag + ": from-states leaves every state unchanged");
}

// --------------------------------------------------------------------------------------- PLE

void test_ple(std::int32_t C, std::int32_t W, std::int32_t B, std::int32_t taps, std::int32_t dilation) {
    std::mt19937 rng(static_cast<std::uint32_t>(C + W * 17 + B * 101));
    const std::int32_t span = (taps - 1) * dilation, slots = B + 1, T = W * B;
    const auto gated      = random_bf16(rng, static_cast<std::size_t>(C) * T, 0.5F);
    const auto normalized = random_bf16(rng, static_cast<std::size_t>(C) * T, 1.0F);
    const auto weight     = random_bf16(rng, static_cast<std::size_t>(C) * taps, 0.5F);
    const auto states     = random_bf16(rng, static_cast<std::size_t>(C) * span * slots, 1.0F);
    const auto residual   = random_bf16(rng, static_cast<std::size_t>(C) * T, 1.0F);
    std::vector<std::int32_t> slot_ids(B), commit(B);
    for (std::int32_t b = 0; b < B; ++b) {
        slot_ids[b] = B - b; // not the identity, slot 0 unused
        commit[b]   = b % (W + 1);
    }
    commit[0] = W;
    const std::string tag = "ple C=" + std::to_string(C) + " W=" + std::to_string(W) + " B=" + std::to_string(B);
    Buffer<std::uint16_t> dg(gated), dn(normalized), dw(weight);
    Buffer<std::int32_t> dslots(slot_ids), dcommit(commit);
    Tensor tg(dg.p, DType::BF16, {C, T}), tn(dn.p, DType::BF16, {C, T}), tw(dw.p, DType::BF16, {C, taps});
    Tensor tslots(dslots.p, DType::I32, {B});
    // Verification form against the committing form: same residual, states untouched.
    Buffer<std::uint16_t> s_verify(states), s_commit(states), r_verify(residual), r_commit(residual);
    Tensor tsv(s_verify.p, DType::BF16, {C, span, slots}), tsc(s_commit.p, DType::BF16, {C, span, slots});
    Tensor trv(r_verify.p, DType::BF16, {C, T}), trc(r_commit.p, DType::BF16, {C, T});
    ninfer::ops::ple_conv_inject(tg, tn, tw, dilation, tsv, tslots, Tensor{}, trv, nullptr);
    ninfer::ops::ple_conv_inject(tg, tn, tw, dilation, tsc, tslots, tslots, trc, nullptr);
    cuda_check(cudaDeviceSynchronize(), "ple");
    check(r_verify.download() == r_commit.download(), tag + ": verification residual equals the committing call's");
    check(s_verify.download() == states, tag + ": verification leaves every history unchanged");
    // Commit of n columns against ple_conv_inject over those n columns of each row.
    ninfer::ops::ple_conv_commit(tn.view({C, W, B}), Tensor(dcommit.p, DType::I32, {B}), tsv, tslots, nullptr);
    cuda_check(cudaDeviceSynchronize(), "ple commit");
    Buffer<std::uint16_t> s_ref(states);
    Tensor tsr(s_ref.p, DType::BF16, {C, span, slots});
    for (std::int32_t b = 0; b < B; ++b) {
        if (commit[b] == 0) { continue; }
        Buffer<std::uint16_t> scratch(static_cast<std::size_t>(C) * commit[b]);
        Tensor tscratch(scratch.p, DType::BF16, {C, commit[b]});
        const Tensor gb = tg.slice(1, b * W, commit[b]), nb = tn.slice(1, b * W, commit[b]);
        Tensor slot = tslots.slice(0, b, 1);
        ninfer::ops::ple_conv_inject(gb, nb, tw, dilation, tsr, slot, slot, tscratch, nullptr);
    }
    cuda_check(cudaDeviceSynchronize(), "ple reference");
    check(s_verify.download() == s_ref.download(), tag + ": commit equals the committing call over the prefix");
}

// --------------------------------------------------------------------------------------- QSA

void test_qsa(std::int32_t W, std::int32_t B, std::int32_t first_position) {
    constexpr std::int32_t Di = 128, R = 4, kPage = 64;
    const ninfer::ops::QsaGeometry geometry{.heads = 16, .kv_heads = 2, .head_dim = 256, .index_heads = 4,
                                            .index_head_dim = Di, .rotary_dim = 64, .budget = 2048, .ratio = R,
                                            .theta = 1.0e7F, .eps = 1.0e-6F};
    std::mt19937 rng(static_cast<std::uint32_t>(W * 13 + B * 7 + first_position));
    const std::int32_t T = W * B, slots = B + 1, pages_per_row = 2, layers = 2;
    const auto raw   = random_bf16(rng, static_cast<std::size_t>(Di) * T * layers, 1.0F);
    const auto norm  = random_bf16(rng, Di, 0.2F);
    const auto tails = random_bf16(rng, static_cast<std::size_t>(Di) * (R - 1) * slots * layers, 1.0F);
    std::vector<std::int32_t> positions(T), table_rows(B), tail_slots(B), tables(pages_per_row * B), commit(B);
    for (std::int32_t b = 0; b < B; ++b) {
        for (std::int32_t j = 0; j < W; ++j) { positions[b * W + j] = first_position + 3 * b + j; }
        table_rows[b] = b;
        tail_slots[b] = (b + 1) % slots;
        for (std::int32_t p = 0; p < pages_per_row; ++p) { tables[b * pages_per_row + p] = b * pages_per_row + p; }
        commit[b] = (b * 3) % (W + 1);
    }
    commit[0] = W;
    const std::string tag = "qsa W=" + std::to_string(W) + " B=" + std::to_string(B) + " p0=" + std::to_string(first_position);
    const std::size_t pooled_count = static_cast<std::size_t>(Di / R) * kPage * pages_per_row * B;
    Buffer<std::uint16_t> draw(raw), dnorm(norm), tails_verify(tails), tails_commit(tails);
    Buffer<std::uint16_t> pooled_verify(std::vector<std::uint16_t>(pooled_count, 0)),
        pooled_commit(std::vector<std::uint16_t>(pooled_count, 0));
    Buffer<std::int32_t> dpos(positions), drows(table_rows), dtail(tail_slots), dtables(tables), dcommit(commit);
    const auto batch_of = [&](bool update) {
        return ninfer::ops::QsaBatch{.block_tables = Tensor(dtables.p, DType::I32, {pages_per_row, B}),
                                     .table_rows   = Tensor(drows.p, DType::I32, {B}),
                                     .positions    = Tensor(dpos.p, DType::I32, {T}),
                                     .tail_slots   = Tensor(dtail.p, DType::I32, {B}),
                                     .batch        = B,
                                     .width        = W,
                                     .update_tails = update};
    };
    const auto layer_of = [&](Buffer<std::uint16_t>& pooled) {
        ninfer::ops::QsaKVLayer layer;
        layer.pooled_pages = Tensor(pooled.p, DType::BF16, {Di / R, kPage, 1, pages_per_row * B});
        return layer;
    };
    const std::size_t layer_tail = static_cast<std::size_t>(Di) * (R - 1) * slots;
    // Layer 0 only for the pooled comparison: verification and committing calls.
    {
        Tensor keys(draw.p, DType::BF16, {Di, T});
        Tensor tv(tails_verify.p, DType::BF16, {Di, R - 1, slots}), tc(tails_commit.p, DType::BF16, {Di, R - 1, slots});
        ninfer::ops::qsa_pool_keys(keys, Tensor(dnorm.p, DType::BF16, {Di}), tv, layer_of(pooled_verify), batch_of(false),
                                   geometry, nullptr);
        ninfer::ops::qsa_pool_keys(keys, Tensor(dnorm.p, DType::BF16, {Di}), tc, layer_of(pooled_commit), batch_of(true),
                                   geometry, nullptr);
        cuda_check(cudaDeviceSynchronize(), "pool");
        check(pooled_verify.download() == pooled_commit.download(), tag + ": verification pools the same keys");
        check(tails_verify.download() == tails, tag + ": verification leaves the tails unchanged");
    }
    // Commit of n columns on both layers against qsa_pool_keys over each row's first n columns.
    Tensor all_tails(tails_verify.p, DType::BF16, {Di, R - 1, slots, layers});
    ninfer::ops::qsa_commit_tails(Tensor(draw.p, DType::BF16, {Di, W, B, layers}), Tensor(dpos.p, DType::I32, {W, B}),
                                  Tensor(dcommit.p, DType::I32, {B}), all_tails, Tensor(dtail.p, DType::I32, {B}),
                                  geometry, nullptr);
    Buffer<std::uint16_t> tails_ref(tails), pooled_ref(std::vector<std::uint16_t>(pooled_count, 0));
    for (std::int32_t l = 0; l < layers; ++l) {
        for (std::int32_t b = 0; b < B; ++b) {
            if (commit[b] == 0) { continue; }
            Tensor keys(draw.p, DType::BF16, {Di, T});
            Tensor tref(tails_ref.p + l * layer_tail, DType::BF16, {Di, R - 1, slots});
            ninfer::ops::QsaBatch one{.block_tables = Tensor(dtables.p, DType::I32, {pages_per_row, B}),
                                      .table_rows   = Tensor(drows.p, DType::I32, {B}).slice(0, b, 1),
                                      .positions    = Tensor(dpos.p, DType::I32, {T}).slice(0, b * W, commit[b]),
                                      .tail_slots   = Tensor(dtail.p, DType::I32, {B}).slice(0, b, 1),
                                      .batch        = 1,
                                      .width        = commit[b]};
            Tensor row_keys(draw.p + (static_cast<std::size_t>(l) * T + b * W) * Di, DType::BF16, {Di, commit[b]});
            ninfer::ops::qsa_pool_keys(row_keys, Tensor(dnorm.p, DType::BF16, {Di}), tref, layer_of(pooled_ref), one,
                                       geometry, nullptr);
        }
    }
    cuda_check(cudaDeviceSynchronize(), "commit tails");
    check(tails_verify.download() == tails_ref.download(), tag + ": tail commit equals pooling the committed prefix");
}

} // namespace

int main() {
    if (ninfer::test::cuda_unavailable()) {
        std::printf("SKIP: no usable CUDA device\n");
        return 77;
    }
    try {
        for (const auto [W, B] : {std::pair{2, 1}, std::pair{8, 1}, std::pair{4, 3}, std::pair{16, 8}}) {
            test_conv(10240, W, B);
            test_ple(1024, W, B, 4, 2);
            for (const int p0 : {0, 5, 62}) { test_qsa(W, B, p0); }
        }
        test_conv(8192, 64, 1);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
    if (g_failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("all speculative state checks passed\n");
    return 0;
}
