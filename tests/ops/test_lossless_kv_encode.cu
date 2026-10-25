// The design reference's KV storage (lossless_kv_encode, docs/maintainer/lossless-route.md §3b) against the
// production append: for the VQ formats every key's window row must equal, byte for byte, the window slot the
// production kv_cache_append writes for that position (the sinks and the call's last 1,024 keys), and every code
// plane the production pages of the same keys (an identity block table).

#include "ops/op_family.h"
#include "core/arena.h"
#include "core/device.h"
#include "core/paged_kv_storage.h"
#include "infernix/ops/kv_cache_append.h"
#include "infernix/ops/lossless_ops.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

using namespace infernix;

namespace {

constexpr int kDim = 256, kPage = 64;

std::uint16_t random_bf16(std::mt19937& rng) {
    std::normal_distribution<float> normal(0.0F, 1.0F);
    const float value = normal(rng) * std::ldexp(1.0F, static_cast<int>(rng() % 9) - 4);
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<std::uint16_t>((bits + 0x7FFFU + ((bits >> 16) & 1U)) >> 16);
}

std::vector<std::uint8_t> download(const void* p, std::size_t bytes) {
    std::vector<std::uint8_t> h(bytes);
    CUDA_CHECK(cudaMemcpy(h.data(), p, bytes, cudaMemcpyDeviceToHost));
    return h;
}

int check(KvCacheStorage storage, const char* name, int kv_heads, int columns) {
    std::mt19937 rng(20261022U + static_cast<unsigned>(storage));
    std::vector<std::uint16_t> hk(static_cast<std::size_t>(kDim) * kv_heads * columns), hv(hk.size());
    for (auto& x : hk) { x = random_bf16(rng); }
    for (auto& x : hv) { x = random_bf16(rng); }
    DeviceBuffer k(hk.size() * 2), v(hv.size() * 2);
    k.copy_from_host(hk.data(), hk.size() * 2);
    v.copy_from_host(hv.data(), hv.size() * 2);
    const Tensor tk(k.p, DType::BF16, {kDim, kv_heads, columns}), tv(v.p, DType::BF16, {kDim, kv_heads, columns});

    DeviceBuffer buffer(ops::lossless_kv_encode_bytes(storage, kv_heads, columns));
    const ops::LosslessKvCodes codes = ops::lossless_kv_encode(tk, tv, storage, buffer.p, nullptr);

    // The production append into its own pages (identity table) with a window, positions 0 .. T - 1.
    const PagedKVStorageLayout layout = paged_kv_storage_layout(storage, kDim);
    const int pages = (columns + kPage - 1) / kPage;
    const std::size_t rows = static_cast<std::size_t>(pages) * kPage * kv_heads;
    auto bytes = [&](DType dtype, int leading) { return rows * leading * dtype_size(dtype); };
    DeviceBuffer kc(bytes(layout.key.data_dtype, layout.key.data_leading_extent)),
        ks(bytes(layout.key.scale_dtype, layout.key.scale_leading_extent)),
        vc(bytes(layout.value.data_dtype, layout.value.data_leading_extent)),
        vs(bytes(layout.value.scale_dtype, layout.value.scale_leading_extent));
    std::vector<std::int32_t> table(pages), positions(columns);
    for (int i = 0; i < pages; ++i) { table[i] = i; }
    for (int i = 0; i < columns; ++i) { positions[i] = i; }
    DeviceBuffer dt(table.size() * 4), dp(positions.size() * 4);
    dt.copy_from_host(table.data(), table.size() * 4);
    dp.copy_from_host(positions.data(), positions.size() * 4);
    const bool window = kv_storage_has_exact_window(storage);
    const std::size_t slot_codes = static_cast<std::size_t>(kDim) * kKVWindowSlots * kv_heads;
    DeviceBuffer wk(slot_codes), wv(slot_codes), wks(slot_codes / 32), wvs(slot_codes / 32), tags(slot_codes / 32);
    for (auto* b : {&kc, &ks, &vc, &vs, &wk, &wv, &wks, &wvs, &tags}) { b->fill(0); }
    PagedKVLayerView view{};
    view.k_pages       = Tensor(kc.p, layout.key.data_dtype, {layout.key.data_leading_extent, kPage, kv_heads, pages});
    view.v_pages       = Tensor(vc.p, layout.value.data_dtype, {layout.value.data_leading_extent, kPage, kv_heads, pages});
    view.k_scale_pages = Tensor(ks.p, layout.key.scale_dtype, {layout.key.scale_leading_extent, kPage, kv_heads, pages});
    view.v_scale_pages =
        Tensor(vs.p, layout.value.scale_dtype, {layout.value.scale_leading_extent, kPage, kv_heads, pages});
    view.block_table  = Tensor(dt.p, DType::I32, {pages});
    view.head_dim     = kDim;
    view.num_kv_heads = kv_heads;
    view.storage      = storage;
    if (window) {
        view.window = PagedKVWindowView{
            .k_codes  = Tensor(wk.p, DType::I8, {kDim, kKVWindowSlots, kv_heads, 1}),
            .v_codes  = Tensor(wv.p, DType::I8, {kDim, kKVWindowSlots, kv_heads, 1}),
            .k_scales = Tensor(wks.p, DType::FP16, {4, kKVWindowSlots, kv_heads, 1}),
            .v_scales = Tensor(wvs.p, DType::FP16, {4, kKVWindowSlots, kv_heads, 1}),
            .tags     = Tensor(tags.p, DType::I32, {2, kKVWindowSlots, kv_heads, 1}),
        };
    }
    ops::kv_cache_append(tk, tv, Tensor(dp.p, DType::I32, {columns}), view, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::size_t wrong = 0;
    auto compare = [&](const std::vector<std::uint8_t>& got, const std::vector<std::uint8_t>& want, const char* plane) {
        std::size_t differ = 0;
        for (std::size_t i = 0; i < want.size(); ++i) { differ += got[i] != want[i]; }
        if (differ != 0) { std::printf("%s %s: %zu of %zu bytes differ\n", name, plane, differ, want.size()); }
        wrong += differ;
    };
    compare(download(codes.k.codes, kc.bytes), download(kc.p, kc.bytes), "k codes");
    compare(download(codes.k.scales, ks.bytes), download(ks.p, ks.bytes), "k scales");
    compare(download(codes.v.codes, vc.bytes), download(vc.p, vc.bytes), "v codes");
    compare(download(codes.v.scales, vs.bytes), download(vs.p, vs.bytes), "v scales");
    std::size_t checked = 0;
    if (window) {
        const auto pk = download(wk.p, slot_codes), pv = download(wv.p, slot_codes);
        const auto pks = download(wks.p, slot_codes / 32), pvs = download(wvs.p, slot_codes / 32);
        const auto mk = download(codes.window_k.codes, rows * kDim), mv = download(codes.window_v.codes, rows * kDim);
        const auto mks = download(codes.window_k.scales, rows * 8), mvs = download(codes.window_v.scales, rows * 8);
        for (int p = 0; p < columns; ++p) {
            if (p >= kKVWindowSinkTokens && p < columns - kKVWindowRingTokens) { continue; }
            const int slot = p < kKVWindowSinkTokens ? p : kKVWindowSinkTokens + (p & (kKVWindowRingTokens - 1));
            for (int h = 0; h < kv_heads; ++h) {
                const std::size_t mine = (static_cast<std::size_t>(p / kPage) * kv_heads + h) * kPage + p % kPage;
                const std::size_t prod = static_cast<std::size_t>(h) * kKVWindowSlots + slot;
                wrong += std::memcmp(&mk[mine * kDim], &pk[prod * kDim], kDim) != 0;
                wrong += std::memcmp(&mv[mine * kDim], &pv[prod * kDim], kDim) != 0;
                wrong += std::memcmp(&mks[mine * 8], &pks[prod * 8], 8) != 0;
                wrong += std::memcmp(&mvs[mine * 8], &pvs[prod * 8], 8) != 0;
                ++checked;
            }
        }
    }
    std::printf("%s: code planes %s, %zu window rows %s\n", name, wrong == 0 ? "identical" : "DIFFER", checked,
                wrong == 0 ? "identical" : "checked");
    return wrong == 0 ? 0 : 1;
}

} // namespace

INFERNIX_OP_TEST(lossless_kv_encode) {
    int failures = 0;
    // The 27B's 4 KV heads; 1,200 keys so the window wraps its ring once (positions 176-1,199 and the sinks).
    failures += check(KvCacheStorage::Int8Group64, "int8", 4, 1200);
    failures += check(KvCacheStorage::Fp8E4M3Row256, "fp8", 4, 1200);
    failures += check(KvCacheStorage::Nvfp4Group16, "nvfp4", 4, 1200);
    failures += check(KvCacheStorage::Fp8KeyNvfp4Value, "k8v4", 2, 1200);
    failures += check(KvCacheStorage::Vq2, "vq2", 4, 1200);
    failures += check(KvCacheStorage::Q4KeyVq2Value, "k4v2", 2, 1200);
    if (failures != 0) {
        std::cerr << "FAIL " << failures << " storages differ from the production append\n";
        return 1;
    }
    return 0;
}
