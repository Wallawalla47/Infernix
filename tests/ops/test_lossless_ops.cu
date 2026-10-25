// The lossless route's elementwise and row operations (include/infernix/ops/lossless_ops.h) against
// lossless_ops_oracle.py's exact results: run as python lossless_ops_oracle.py <this executable>, which
// passes the case file (per case: op, dims, inputs, expected bits). Every output must match.

#include "ops/op_family.h"
#include "core/arena.h"
#include "core/device.h"
#include "infernix/ops/lossless_ops.h"
#include "infernix/ops/lossless_quantize.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace infernix;

namespace {

std::vector<std::uint32_t> read_u32(std::ifstream& in, std::size_t n) {
    std::vector<std::uint32_t> v(n);
    in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * 4));
    return v;
}

DeviceBuffer upload16(const std::vector<std::uint32_t>& v) {
    std::vector<std::uint16_t> h(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) { h[i] = static_cast<std::uint16_t>(v[i]); }
    DeviceBuffer d(h.size() * 2);
    d.copy_from_host(h.data(), h.size() * 2);
    return d;
}

DeviceBuffer upload32(const std::vector<std::uint32_t>& v) {
    DeviceBuffer d(v.size() * 4);
    d.copy_from_host(v.data(), v.size() * 4);
    return d;
}

std::vector<std::uint32_t> download16(const DeviceBuffer& d, std::size_t n) {
    std::vector<std::uint16_t> h(n);
    d.copy_to_host(h.data(), n * 2);
    return std::vector<std::uint32_t>(h.begin(), h.end());
}

std::vector<std::uint32_t> download32(const DeviceBuffer& d, std::size_t n) {
    std::vector<std::uint32_t> h(n);
    d.copy_to_host(h.data(), n * 4);
    return h;
}

const char* kNames[] = {"rmsnorm",       "rmsnorm_gated", "add",          "silu_mul",
                         "sigmoid_mul",   "gdn_gates",     "causal_conv",  "attention",
                         "delta_rule",    "rope",          "router",       "moe_combine",
                         "moe_route",     "rmsnorm_grouped", "rmsnorm_gated_sigmoid", "hc_gates",
                         "hc_mix",        "hc_inject",     "ple_gate",     "ple_conv",
                         "qsa_pool",      "qsa_select",    "attention_select", "router_bf16",
                         "moe_combine_entries", "conv_slots", "delta_rule_slots", "attention_paged",
                         "ple_conv_slots", "qsa_cache", "qsa_select_positions", "attention_paged_select",
                         "quantize_nvfp4", "quantize_nvfp4_scaled", "quantize_fp8_token", "round_fp16",
                         "quantize_fp8_static", "codec_attention"};

// One codec plane of the oracle's rows (key j, KV head h at j Hkv + h; per row its code units and scale units,
// each one u32) packed into lossless_codec_attention's page order: per row the code bytes (VQ2: 16-bit words)
// and the scale bytes (FP16, or NVFP4's E4M3 bytes).
struct PackedPlane {
    DeviceBuffer codes, scales;
};

PackedPlane pack_plane(int codec, const std::vector<std::uint32_t>& codes, const std::vector<std::uint32_t>& scales,
                       int kv_heads, int cols) {
    const auto c = static_cast<ops::LosslessKvCodec>(codec);
    const int code_units  = c == ops::LosslessKvCodec::Int8G64 || c == ops::LosslessKvCodec::Fp8Row ? 256
                            : c == ops::LosslessKvCodec::Vq2                                     ? 32
                                                                                                 : 128;
    const int code_bytes  = c == ops::LosslessKvCodec::Vq2 ? 64 : code_units;
    const int scale_units = c == ops::LosslessKvCodec::Int8G64 ? 4 : (c == ops::LosslessKvCodec::Nvfp4G16 ? 16 : 1);
    const int scale_bytes = c == ops::LosslessKvCodec::Nvfp4G16 ? 16 : 2 * scale_units;
    const std::size_t rows = static_cast<std::size_t>((cols + 63) / 64) * 64 * kv_heads;
    std::vector<std::uint8_t> hc(rows * code_bytes, 0), hs(rows * scale_bytes, 0);
    for (int j = 0; j < cols; ++j) {
        for (int h = 0; h < kv_heads; ++h) {
            const std::size_t src = static_cast<std::size_t>(j) * kv_heads + h;
            const std::size_t dst = (static_cast<std::size_t>(j / 64) * kv_heads + h) * 64 + j % 64;
            for (int u = 0; u < code_units; ++u) {
                const std::uint32_t value = codes[src * code_units + u];
                if (c == ops::LosslessKvCodec::Vq2) {
                    hc[dst * code_bytes + 2 * u]     = static_cast<std::uint8_t>(value);
                    hc[dst * code_bytes + 2 * u + 1] = static_cast<std::uint8_t>(value >> 8);
                } else {
                    hc[dst * code_bytes + u] = static_cast<std::uint8_t>(value);
                }
            }
            for (int u = 0; u < scale_units; ++u) {
                const std::uint32_t value = scales[src * scale_units + u];
                if (c == ops::LosslessKvCodec::Nvfp4G16) {
                    hs[dst * scale_bytes + u] = static_cast<std::uint8_t>(value);
                } else {
                    hs[dst * scale_bytes + 2 * u]     = static_cast<std::uint8_t>(value);
                    hs[dst * scale_bytes + 2 * u + 1] = static_cast<std::uint8_t>(value >> 8);
                }
            }
        }
    }
    PackedPlane p{DeviceBuffer(hc.size()), DeviceBuffer(hs.size())};
    p.codes.copy_from_host(hc.data(), hc.size());
    p.scales.copy_from_host(hs.data(), hs.size());
    return p;
}

} // namespace

INFERNIX_OP_TEST(lossless_ops) {
    if (argc != 2) {
        std::cerr << "usage: test_lossless_ops <cases.bin> (run through lossless_ops_oracle.py)\n";
        return 2;
    }
    std::ifstream in(argv[1], std::ios::binary);
    int failures = 0;
    while (in.peek() != std::char_traits<char>::eof()) {
        const auto head = read_u32(in, 2);
        const std::uint32_t op = head[0];
        const auto dims = read_u32(in, head[1]);
        const auto ninputs = read_u32(in, 1)[0];
        const auto sizes = read_u32(in, ninputs);
        const auto nexpected = read_u32(in, 1)[0];
        std::vector<std::vector<std::uint32_t>> inputs;
        for (const auto s : sizes) { inputs.push_back(read_u32(in, s)); }
        const auto expected = read_u32(in, nexpected);
        if (!in) {
            std::cerr << "FAIL could not read the cases\n";
            return 1;
        }
        std::vector<std::uint32_t> got;
        switch (op) {
        case 0:
        case 1: {
            const int width = static_cast<int>(dims[0]), rows = static_cast<int>(dims[1]);
            const bool w_fp32 = dims[2] != 0;
            DeviceBuffer x = upload16(inputs[0]);
            DeviceBuffer z = op == 1 ? upload16(inputs[1]) : DeviceBuffer{};
            const auto& wv = inputs[op == 1 ? 2 : 1];
            DeviceBuffer w = w_fp32 ? upload32(wv) : upload16(wv);
            DeviceBuffer out(static_cast<std::size_t>(width) * rows * 2);
            Tensor tx(x.p, DType::BF16, {width, rows});
            Tensor tw(w.p, w_fp32 ? DType::FP32 : DType::BF16, {width});
            Tensor tout(out.p, DType::BF16, {width, rows});
            if (op == 0) {
                ops::lossless_rmsnorm(tx, tw, 1e-6F, tout, nullptr);
            } else {
                Tensor tz(z.p, DType::BF16, {width, rows});
                ops::lossless_rmsnorm_gated(tx, tz, tw, 1e-6F, tout, nullptr);
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            got = download16(out, static_cast<std::size_t>(width) * rows);
            break;
        }
        case 2:
        case 3:
        case 4: {
            const int n = static_cast<int>(dims[0]);
            DeviceBuffer a = upload16(inputs[0]), b = upload16(inputs[1]), out(static_cast<std::size_t>(n) * 2);
            Tensor ta(a.p, DType::BF16, {n}), tb(b.p, DType::BF16, {n}), tout(out.p, DType::BF16, {n});
            if (op == 2) { ops::lossless_add(ta, tb, tout, nullptr); }
            if (op == 3) { ops::lossless_silu_mul(ta, tb, tout, nullptr); }
            if (op == 4) { ops::lossless_sigmoid_mul(ta, tb, tout, nullptr); }
            CUDA_CHECK(cudaDeviceSynchronize());
            got = download16(out, static_cast<std::size_t>(n));
            break;
        }
        case 5: {
            const int heads = static_cast<int>(dims[0]), cols = static_cast<int>(dims[1]);
            DeviceBuffer a = upload16(inputs[0]), b = upload16(inputs[1]), alog = upload32(inputs[2]),
                         dt = upload32(inputs[3]);
            DeviceBuffer g(static_cast<std::size_t>(heads) * cols * 4), beta(static_cast<std::size_t>(heads) * cols * 2);
            Tensor ta(a.p, DType::BF16, {heads, cols}), tb(b.p, DType::BF16, {heads, cols});
            Tensor tal(alog.p, DType::FP32, {heads}), tdt(dt.p, DType::FP32, {heads});
            Tensor tg(g.p, DType::FP32, {heads, cols}), tbeta(beta.p, DType::BF16, {heads, cols});
            ops::lossless_gdn_gates(ta, tb, tal, tdt, tg, tbeta, nullptr);
            CUDA_CHECK(cudaDeviceSynchronize());
            got = download32(g, static_cast<std::size_t>(heads) * cols);
            const auto bb = download16(beta, static_cast<std::size_t>(heads) * cols);
            got.insert(got.end(), bb.begin(), bb.end());
            break;
        }
        case 6: {
            const int channels = static_cast<int>(dims[0]), cols = static_cast<int>(dims[1]), taps = static_cast<int>(dims[2]);
            DeviceBuffer x = upload16(inputs[0]), w = upload16(inputs[1]),
                         out(static_cast<std::size_t>(channels) * cols * 2);
            Tensor tx(x.p, DType::BF16, {channels, cols}), tw(w.p, DType::BF16, {channels, taps});
            Tensor tout(out.p, DType::BF16, {channels, cols});
            ops::lossless_causal_conv(tx, tw, tout, nullptr);
            CUDA_CHECK(cudaDeviceSynchronize());
            got = download16(out, static_cast<std::size_t>(channels) * cols);
            break;
        }
        case 7: {
            const int heads = static_cast<int>(dims[0]), kv_heads = static_cast<int>(dims[1]), cols = static_cast<int>(dims[2]);
            DeviceBuffer q = upload16(inputs[0]), k = upload16(inputs[1]), v = upload16(inputs[2]);
            DeviceBuffer out(static_cast<std::size_t>(256) * heads * cols * 2);
            DeviceBuffer ws(ops::lossless_attention_workspace_bytes(heads, kv_heads, cols));
            Tensor tq(q.p, DType::BF16, {256, heads, cols}), tk(k.p, DType::BF16, {256, kv_heads, cols});
            Tensor tv(v.p, DType::BF16, {256, kv_heads, cols}), tout(out.p, DType::BF16, {256, heads, cols});
            ops::lossless_attention(tq, tk, tv, tout, ws.p, nullptr);
            CUDA_CHECK(cudaDeviceSynchronize());
            got = download16(out, static_cast<std::size_t>(256) * heads * cols);
            break;
        }
        case 8: {
            const int hk = static_cast<int>(dims[0]), hv = static_cast<int>(dims[1]), cols = static_cast<int>(dims[2]);
            DeviceBuffer q = upload16(inputs[0]), k = upload16(inputs[1]), v = upload16(inputs[2]);
            DeviceBuffer g = upload32(inputs[3]), beta = upload16(inputs[4]);
            DeviceBuffer out(static_cast<std::size_t>(128) * hv * cols * 2);
            Tensor tq(q.p, DType::BF16, {128 * hk, cols}), tk(k.p, DType::BF16, {128 * hk, cols});
            Tensor tv(v.p, DType::BF16, {128 * hv, cols}), tout(out.p, DType::BF16, {128 * hv, cols});
            Tensor tg(g.p, DType::FP32, {hv, cols}), tbeta(beta.p, DType::BF16, {hv, cols});
            ops::lossless_gated_delta_rule(tq, tk, tv, tg, tbeta, nullptr, tout, nullptr);
            CUDA_CHECK(cudaDeviceSynchronize());
            got = download16(out, static_cast<std::size_t>(128) * hv * cols);
            break;
        }
        case 9: {
            const int heads = static_cast<int>(dims[0]), cols = static_cast<int>(dims[1]), axes = static_cast<int>(dims[2]);
            DeviceBuffer pos = upload32(inputs[0]), x = upload16(inputs[1]);
            DeviceBuffer inv(32 * 4), table(static_cast<std::size_t>(32) * cols * 2 * 2);
            Tensor tpos(pos.p, DType::I32, {cols, axes}), tinv(inv.p, DType::FP32, {32});
            Tensor ttable(table.p, DType::BF16, {32, cols, 2}), tx(x.p, DType::BF16, {256, heads, cols});
            ops::lossless_rope_frequencies(1.0e7F, 64, tinv, nullptr);
            ops::lossless_rope_table(tpos, tinv, ttable, nullptr);
            ops::lossless_rope_apply(ttable, tx, nullptr);
            CUDA_CHECK(cudaDeviceSynchronize());
            got = download16(x, static_cast<std::size_t>(256) * heads * cols);
            break;
        }
        case 10: {
            const int experts = static_cast<int>(dims[0]), top_k = static_cast<int>(dims[1]), cols = static_cast<int>(dims[2]);
            DeviceBuffer logits = upload16(inputs[0]);
            DeviceBuffer w(static_cast<std::size_t>(top_k) * cols * 4), ids(static_cast<std::size_t>(top_k) * cols * 4);
            Tensor tl(logits.p, DType::BF16, {experts + 1, cols});
            Tensor tw(w.p, DType::FP32, {top_k, cols}), tids(ids.p, DType::I32, {top_k, cols});
            ops::lossless_router(tl, experts, top_k, tw, tids, nullptr);
            CUDA_CHECK(cudaDeviceSynchronize());
            got = download32(w, static_cast<std::size_t>(top_k) * cols);
            const auto gi = download32(ids, static_cast<std::size_t>(top_k) * cols);
            got.insert(got.end(), gi.begin(), gi.end());
            break;
        }
        case 11: {
            const int width = static_cast<int>(dims[0]), cols = static_cast<int>(dims[1]), top_k = static_cast<int>(dims[2]),
                      gathered = static_cast<int>(dims[3]);
            DeviceBuffer eo = upload16(inputs[0]), w = upload32(inputs[1]), ids = upload32(inputs[2]),
                         sc = upload32(inputs[3]), so = upload16(inputs[4]), sl = upload16(inputs[5]),
                         res = upload16(inputs[6]);
            Tensor teo(eo.p, DType::BF16, {width, gathered}), tw(w.p, DType::FP32, {top_k, cols});
            Tensor tids(ids.p, DType::I32, {top_k, cols}), tsc(sc.p, DType::I32, {top_k, cols});
            Tensor tso(so.p, DType::BF16, {width, cols}), tsl(sl.p, DType::BF16, {3, cols});
            Tensor tres(res.p, DType::BF16, {width, cols});
            ops::lossless_moe_combine(teo, tw, tids, tsc, &tso, &tsl, 2, tres, nullptr);
            CUDA_CHECK(cudaDeviceSynchronize());
            got = download16(res, static_cast<std::size_t>(width) * cols);
            break;
        }
        case 12: {
            // Routing structure: every (slot, column) at a gathered column holding its activations in a tile
            // of its expert, distinct columns, padding zero, unused tiles marked.
            const int experts = static_cast<int>(dims[0]), top_k = static_cast<int>(dims[1]), cols = static_cast<int>(dims[2]);
            const int width = 64, c = ops::lossless_moe_columns(cols, top_k, experts);
            DeviceBuffer ids = upload32(inputs[0]), x = upload16(inputs[1]);
            DeviceBuffer counts(static_cast<std::size_t>(experts) * 4), tiles(static_cast<std::size_t>(c / 64) * 4),
                slots(static_cast<std::size_t>(top_k) * cols * 4), tokens(static_cast<std::size_t>(c) * 4),
                gathered(static_cast<std::size_t>(width) * c * 2);
            Tensor tids(ids.p, DType::I32, {top_k, cols}), tcounts(counts.p, DType::I32, {experts});
            Tensor ttiles(tiles.p, DType::I32, {c / 64}), tslots(slots.p, DType::I32, {top_k, cols});
            Tensor ttokens(tokens.p, DType::I32, {c}), tx(x.p, DType::BF16, {width, cols});
            Tensor tg(gathered.p, DType::BF16, {width, c});
            ops::lossless_moe_route(tids, experts, tcounts, ttiles, tslots, ttokens, nullptr);
            ops::lossless_moe_gather(tx, ttokens, ttiles, tg, nullptr);
            CUDA_CHECK(cudaDeviceSynchronize());
            const auto tile_e = download32(tiles, static_cast<std::size_t>(c / 64));
            const auto slot_c = download32(slots, static_cast<std::size_t>(top_k) * cols);
            const auto token  = download32(tokens, static_cast<std::size_t>(c));
            const auto g      = download16(gathered, static_cast<std::size_t>(width) * c);
            bool ok = true;
            std::vector<int> used(static_cast<std::size_t>(c), 0);
            for (int t = 0; t < cols; ++t) {
                for (int s = 0; s < top_k; ++s) {
                    const std::size_t at = static_cast<std::size_t>(t) * top_k + s;
                    const int col = static_cast<int>(slot_c[at]);
                    ok = ok && col >= 0 && col < c && used[static_cast<std::size_t>(col)]++ == 0;
                    ok = ok && static_cast<int>(tile_e[static_cast<std::size_t>(col / 64)]) == static_cast<int>(inputs[0][at]);
                    ok = ok && static_cast<int>(token[static_cast<std::size_t>(col)]) == t;
                    for (int h = 0; h < width && ok; ++h) {
                        ok = g[static_cast<std::size_t>(col) * width + h] == inputs[1][static_cast<std::size_t>(t) * width + h];
                    }
                }
            }
            for (int col = 0; col < c && ok; ++col) {
                const int e = static_cast<int>(tile_e[static_cast<std::size_t>(col / 64)]);
                if (e >= 0 && used[static_cast<std::size_t>(col)] == 0) {
                    for (int h = 0; h < width && ok; ++h) { ok = g[static_cast<std::size_t>(col) * width + h] == 0; }
                }
            }
            got = {ok ? 1U : 0U};
            break;
        }
        case 13: {
            const int width = static_cast<int>(dims[0]), groups = static_cast<int>(dims[1]), cols = static_cast<int>(dims[2]);
            const bool w_fp32 = dims[3] != 0;
            DeviceBuffer x = upload16(inputs[0]), w = w_fp32 ? upload32(inputs[1]) : upload16(inputs[1]);
            DeviceBuffer out(static_cast<std::size_t>(width) * groups * cols * 2);
            Tensor tx(x.p, DType::BF16, {width * groups, cols}), tout(out.p, DType::BF16, {width * groups, cols});
            Tensor tw(w.p, w_fp32 ? DType::FP32 : DType::BF16, {width * groups});
            ops::lossless_rmsnorm_grouped(tx, tw, groups, 1e-6F, tout, nullptr);
            CUDA_CHECK(cudaDeviceSynchronize());
            got = download16(out, static_cast<std::size_t>(width) * groups * cols);
            break;
        }
        case 14: {
            const int width = static_cast<int>(dims[0]), rows = static_cast<int>(dims[1]);
            DeviceBuffer x = upload16(inputs[0]), z = upload16(inputs[1]), w = upload16(inputs[2]);
            DeviceBuffer out(static_cast<std::size_t>(width) * rows * 2);
            Tensor tx(x.p, DType::BF16, {width, rows}), tz(z.p, DType::BF16, {width, rows});
            Tensor tw(w.p, DType::BF16, {width}), tout(out.p, DType::BF16, {width, rows});
            ops::lossless_rmsnorm_gated(tx, tz, tw, 1e-6F, tout, nullptr, /*sigmoid_gate=*/true);
            CUDA_CHECK(cudaDeviceSynchronize());
            got = download16(out, static_cast<std::size_t>(width) * rows);
            break;
        }
        case 15: {
            const int rank = static_cast<int>(dims[0]), streams = static_cast<int>(dims[1]), cols = static_cast<int>(dims[2]);
            DeviceBuffer dj = upload16(inputs[0]), mix(static_cast<std::size_t>(rank) * cols * 2),
                         inject(static_cast<std::size_t>(streams) * cols * 2);
            Tensor tdj(dj.p, DType::BF16, {rank + streams, cols}), tmix(mix.p, DType::BF16, {rank, cols});
            Tensor tinj(inject.p, DType::BF16, {streams, cols});
            ops::lossless_hc_gates(tdj, rank, streams, tmix, &tinj, nullptr);
            CUDA_CHECK(cudaDeviceSynchronize());
            got = download16(mix, static_cast<std::size_t>(rank) * cols);
            const auto gi = download16(inject, static_cast<std::size_t>(streams) * cols);
            got.insert(got.end(), gi.begin(), gi.end());
            break;
        }
        case 16: {
            const int width = static_cast<int>(dims[0]), streams = static_cast<int>(dims[1]), cols = static_cast<int>(dims[2]);
            DeviceBuffer up = upload16(inputs[0]), nm = upload16(inputs[1]), x(static_cast<std::size_t>(width) * cols * 2);
            Tensor tup(up.p, DType::BF16, {width * streams, cols}), tnm(nm.p, DType::BF16, {width * streams, cols});
            Tensor tx(x.p, DType::BF16, {width, cols});
            ops::lossless_hc_mix(tup, tnm, streams, tx, nullptr);
            CUDA_CHECK(cudaDeviceSynchronize());
            got = download16(x, static_cast<std::size_t>(width) * cols);
            break;
        }
        case 17: {
            const int width = static_cast<int>(dims[0]), streams = static_cast<int>(dims[1]), cols = static_cast<int>(dims[2]);
            DeviceBuffer y = upload16(inputs[0]), j = upload16(inputs[1]), r = upload16(inputs[2]);
            Tensor ty(y.p, DType::BF16, {width, cols}), tj(j.p, DType::BF16, {streams, cols});
            Tensor tr(r.p, DType::BF16, {width * streams, cols});
            ops::lossless_hc_inject(ty, tj, tr, nullptr);
            CUDA_CHECK(cudaDeviceSynchronize());
            got = download16(r, static_cast<std::size_t>(width) * streams * cols);
            break;
        }
        case 18: {
            const int width = static_cast<int>(dims[0]), streams = static_cast<int>(dims[1]), cols = static_cast<int>(dims[2]);
            DeviceBuffer kn = upload16(inputs[0]), qn = upload16(inputs[1]), v = upload16(inputs[2]);
            DeviceBuffer scores(static_cast<std::size_t>(streams) * cols * 2),
                gated(static_cast<std::size_t>(width) * streams * cols * 2);
            Tensor tk(kn.p, DType::BF16, {width * streams, cols}), tq(qn.p, DType::BF16, {width * streams, cols});
            Tensor tv(v.p, DType::BF16, {width, cols}), ts(scores.p, DType::BF16, {streams, cols});
            Tensor tg(gated.p, DType::BF16, {width * streams, cols});
            ops::lossless_ple_gate(tk, tq, tv, streams, ts, tg, nullptr);
            CUDA_CHECK(cudaDeviceSynchronize());
            got = download16(scores, static_cast<std::size_t>(streams) * cols);
            const auto gg = download16(gated, static_cast<std::size_t>(width) * streams * cols);
            got.insert(got.end(), gg.begin(), gg.end());
            break;
        }
        case 19: {
            const int channels = static_cast<int>(dims[0]), cols = static_cast<int>(dims[1]), taps = static_cast<int>(dims[2]),
                      dilation = static_cast<int>(dims[3]);
            DeviceBuffer g = upload16(inputs[0]), n = upload16(inputs[1]), w = upload16(inputs[2]), r = upload16(inputs[3]);
            Tensor tg(g.p, DType::BF16, {channels, cols}), tn(n.p, DType::BF16, {channels, cols});
            Tensor tw(w.p, DType::BF16, {channels, taps}), tr(r.p, DType::BF16, {channels, cols});
            ops::lossless_ple_conv_inject(tg, tn, tw, dilation, tr, nullptr);
            CUDA_CHECK(cudaDeviceSynchronize());
            got = download16(r, static_cast<std::size_t>(channels) * cols);
            break;
        }
        case 20: {
            const int dim = static_cast<int>(dims[0]), ratio = static_cast<int>(dims[1]), cols = static_cast<int>(dims[2]);
            const int blocks = cols / ratio;
            DeviceBuffer k = upload16(inputs[0]), pooled(static_cast<std::size_t>(dim) * blocks * 2);
            Tensor tk(k.p, DType::BF16, {dim, cols}), tp(pooled.p, DType::BF16, {dim, blocks});
            ops::lossless_qsa_pool(tk, ratio, tp, nullptr);
            CUDA_CHECK(cudaDeviceSynchronize());
            got = download16(pooled, static_cast<std::size_t>(dim) * blocks);
            break;
        }
        case 21: {
            const int heads = static_cast<int>(dims[0]), ratio = static_cast<int>(dims[1]), budget = static_cast<int>(dims[2]),
                      dim = static_cast<int>(dims[3]), cols = static_cast<int>(dims[4]);
            const int blocks = cols / ratio, words = (blocks + 31) / 32;
            DeviceBuffer sc = upload32(inputs[0]), sel(static_cast<std::size_t>(cols) * words * 4),
                ws(ops::lossless_qsa_select_workspace_bytes(cols, blocks));
            Tensor tsc(sc.p, DType::FP32, {blocks, heads * cols});
            ops::lossless_qsa_select(tsc, heads, ratio, budget, dim, static_cast<std::uint32_t*>(sel.p), words, ws.p,
                                     nullptr);
            CUDA_CHECK(cudaDeviceSynchronize());
            got = download32(sel, static_cast<std::size_t>(cols) * words);
            break;
        }
        case 22: {
            const int heads = static_cast<int>(dims[0]), kv_heads = static_cast<int>(dims[1]), cols = static_cast<int>(dims[2]),
                      ratio = static_cast<int>(dims[3]);
            DeviceBuffer q = upload16(inputs[0]), k = upload16(inputs[1]), v = upload16(inputs[2]), sel = upload32(inputs[3]);
            DeviceBuffer out(static_cast<std::size_t>(256) * heads * cols * 2);
            DeviceBuffer ws(ops::lossless_attention_workspace_bytes(heads, kv_heads, cols));
            Tensor tq(q.p, DType::BF16, {256, heads, cols}), tk(k.p, DType::BF16, {256, kv_heads, cols});
            Tensor tv(v.p, DType::BF16, {256, kv_heads, cols}), tout(out.p, DType::BF16, {256, heads, cols});
            ops::lossless_attention(tq, tk, tv, tout, ws.p, nullptr, static_cast<const std::uint32_t*>(sel.p), 1, ratio);
            CUDA_CHECK(cudaDeviceSynchronize());
            got = download16(out, static_cast<std::size_t>(256) * heads * cols);
            break;
        }
        case 23: {
            const int experts = static_cast<int>(dims[0]), top_k = static_cast<int>(dims[1]), cols = static_cast<int>(dims[2]);
            DeviceBuffer logits = upload16(inputs[0]);
            DeviceBuffer w(static_cast<std::size_t>(top_k) * cols * 4), ids(static_cast<std::size_t>(top_k) * cols * 4);
            Tensor tl(logits.p, DType::BF16, {experts + 1, cols});
            Tensor tw(w.p, DType::FP32, {top_k, cols}), tids(ids.p, DType::I32, {top_k, cols});
            ops::lossless_router(tl, experts, top_k, tw, tids, nullptr, /*bf16_weights=*/true);
            CUDA_CHECK(cudaDeviceSynchronize());
            got = download32(w, static_cast<std::size_t>(top_k) * cols);
            const auto gi = download32(ids, static_cast<std::size_t>(top_k) * cols);
            got.insert(got.end(), gi.begin(), gi.end());
            break;
        }
        case 24: {
            const int width = static_cast<int>(dims[0]), cols = static_cast<int>(dims[1]), top_k = static_cast<int>(dims[2]);
            DeviceBuffer eo = upload16(inputs[0]), w = upload32(inputs[1]), ids = upload32(inputs[2]),
                         so = upload16(inputs[3]), sl = upload16(inputs[4]), out(static_cast<std::size_t>(width) * cols * 2);
            Tensor teo(eo.p, DType::BF16, {width, cols * top_k}), tw(w.p, DType::FP32, {top_k, cols});
            Tensor tids(ids.p, DType::I32, {top_k, cols}), tso(so.p, DType::BF16, {width, cols});
            Tensor tsl(sl.p, DType::BF16, {1, cols}), tout(out.p, DType::BF16, {width, cols});
            ops::lossless_moe_combine(teo, tw, tids, Tensor{}, &tso, &tsl, 0, tout, nullptr, /*add=*/false);
            CUDA_CHECK(cudaDeviceSynchronize());
            got = download16(out, static_cast<std::size_t>(width) * cols);
            break;
        }
        case 25: {
            const int channels = static_cast<int>(dims[0]), width = static_cast<int>(dims[1]),
                      seqs = static_cast<int>(dims[2]), taps = static_cast<int>(dims[3]), slots = static_cast<int>(dims[4]);
            const int cols = width * seqs;
            DeviceBuffer x = upload16(inputs[0]), w = upload16(inputs[1]), st = upload16(inputs[2]);
            DeviceBuffer src = upload32(inputs[3]), dst = upload32(inputs[4]);
            DeviceBuffer out(static_cast<std::size_t>(channels) * cols * 2);
            Tensor tx(x.p, DType::BF16, {channels, cols}), tw(w.p, DType::BF16, {channels, taps});
            Tensor tst(st.p, DType::BF16, {channels, taps - 1, slots}), tout(out.p, DType::BF16, {channels, cols});
            Tensor tsrc(src.p, DType::I32, {seqs}), tdst(dst.p, DType::I32, {seqs});
            ops::lossless_causal_conv_slots(tx, tw, seqs, tst, tsrc, tdst, tout, nullptr);
            CUDA_CHECK(cudaDeviceSynchronize());
            got = download16(out, static_cast<std::size_t>(channels) * cols);
            const auto fs = download16(st, static_cast<std::size_t>(channels) * (taps - 1) * slots);
            got.insert(got.end(), fs.begin(), fs.end());
            break;
        }
        case 26: {
            const int hk = static_cast<int>(dims[0]), hv = static_cast<int>(dims[1]), width = static_cast<int>(dims[2]),
                      seqs = static_cast<int>(dims[3]), slots = static_cast<int>(dims[4]);
            const int cols = width * seqs;
            DeviceBuffer q = upload16(inputs[0]), k = upload16(inputs[1]), v = upload16(inputs[2]);
            DeviceBuffer g = upload32(inputs[3]), beta = upload16(inputs[4]), st = upload32(inputs[5]);
            DeviceBuffer src = upload32(inputs[6]), dst = upload32(inputs[7]);
            DeviceBuffer out(static_cast<std::size_t>(128) * hv * cols * 2);
            Tensor tq(q.p, DType::BF16, {128 * hk, cols}), tk(k.p, DType::BF16, {128 * hk, cols});
            Tensor tv(v.p, DType::BF16, {128 * hv, cols}), tout(out.p, DType::BF16, {128 * hv, cols});
            Tensor tg(g.p, DType::FP32, {hv, cols}), tbeta(beta.p, DType::BF16, {hv, cols});
            Tensor tst(st.p, DType::FP32, {128, 128, hv, slots});
            Tensor tsrc(src.p, DType::I32, {seqs}), tdst(dst.p, DType::I32, {seqs});
            ops::lossless_gated_delta_rule_slots(tq, tk, tv, tg, tbeta, seqs, tst, tsrc, tdst, tout, nullptr);
            CUDA_CHECK(cudaDeviceSynchronize());
            got = download16(out, static_cast<std::size_t>(128) * hv * cols);
            const auto fs = download32(st, static_cast<std::size_t>(128) * 128 * hv * slots);
            got.insert(got.end(), fs.begin(), fs.end());
            break;
        }
        case 27:
        case 31: {
            const int heads = static_cast<int>(dims[0]), kv_heads = static_cast<int>(dims[1]),
                      width = static_cast<int>(dims[2]), seqs = static_cast<int>(dims[3]),
                      stride = static_cast<int>(dims[4]), pages = static_cast<int>(dims[5]),
                      max_keys = static_cast<int>(dims[6]);
            const int cols = width * seqs;
            const auto& lens = inputs[4];
            const auto& tables = inputs[5];
            // The sequences' earlier keys and values into their pages ([256, 64, Hkv, pages]); the query
            // columns' own through lossless_kv_append.
            const std::size_t page_elems = static_cast<std::size_t>(256) * 64 * kv_heads;
            std::vector<std::uint32_t> kp(page_elems * pages, 0), vp(page_elems * pages, 0);
            std::vector<std::uint32_t> knew, vnew;
            std::size_t base = 0;
            for (int b = 0; b < seqs; ++b) {
                for (std::uint32_t j = lens[b] - width; j < lens[b]; ++j) {
                    const std::size_t src = base + static_cast<std::size_t>(j) * kv_heads * 256;
                    knew.insert(knew.end(), inputs[1].begin() + src, inputs[1].begin() + src + kv_heads * 256);
                    vnew.insert(vnew.end(), inputs[2].begin() + src, inputs[2].begin() + src + kv_heads * 256);
                }
                for (std::uint32_t j = 0; j + width < lens[b]; ++j) {
                    const std::uint32_t page = tables[static_cast<std::size_t>(b) * stride + j / 64];
                    for (int h = 0; h < kv_heads; ++h) {
                        for (int d = 0; d < 256; ++d) {
                            const std::size_t dst = ((static_cast<std::size_t>(page) * kv_heads + h) * 64 + j % 64) * 256 + d;
                            const std::size_t src = base + (static_cast<std::size_t>(j) * kv_heads + h) * 256 + d;
                            kp[dst] = inputs[1][src];
                            vp[dst] = inputs[2][src];
                        }
                    }
                }
                base += static_cast<std::size_t>(lens[b]) * kv_heads * 256;
            }
            std::vector<std::uint32_t> rows(seqs);
            for (int b = 0; b < seqs; ++b) { rows[b] = static_cast<std::uint32_t>(b); }
            DeviceBuffer q = upload16(inputs[0]), kpg = upload16(kp), vpg = upload16(vp);
            DeviceBuffer kn = upload16(knew), vn = upload16(vnew);
            DeviceBuffer pos = upload32(inputs[3]), tab = upload32(tables), trow = upload32(rows);
            DeviceBuffer out(static_cast<std::size_t>(256) * heads * cols * 2);
            DeviceBuffer ws(ops::lossless_paged_attention_workspace_bytes(heads, kv_heads, width, seqs, max_keys));
            Tensor tq(q.p, DType::BF16, {256, heads, cols}), tout(out.p, DType::BF16, {256, heads, cols});
            Tensor tpos(pos.p, DType::I32, {cols}), trows(trow.p, DType::I32, {seqs});
            PagedKVBatchLayerView cache{};
            cache.k_pages      = Tensor(kpg.p, DType::BF16, {256, 64, kv_heads, pages});
            cache.v_pages      = Tensor(vpg.p, DType::BF16, {256, 64, kv_heads, pages});
            cache.block_tables = Tensor(tab.p, DType::I32, {stride, seqs});
            cache.head_dim     = 256;
            cache.num_kv_heads = kv_heads;
            cache.storage      = KvCacheStorage::BFloat16;
            Tensor tkn(kn.p, DType::BF16, {256, kv_heads, cols}), tvn(vn.p, DType::BF16, {256, kv_heads, cols});
            ops::lossless_kv_append(tkn, tvn, tpos, seqs, trows, cache, nullptr);
            // Case 31: with a QSA selection (U32 [T], one word per query).
            const int ratio = op == 31 ? static_cast<int>(dims[7]) : 0;
            DeviceBuffer sel = op == 31 ? upload32(inputs[6]) : DeviceBuffer{};
            ops::lossless_paged_attention(tq, tpos, seqs, trows, cache, max_keys, max_keys, tout, ws.p, nullptr,
                                          op == 31 ? static_cast<const std::uint32_t*>(sel.p) : nullptr, op == 31 ? 1 : 0,
                                          ratio);
            CUDA_CHECK(cudaDeviceSynchronize());
            got = download16(out, static_cast<std::size_t>(256) * heads * cols);
            break;
        }
        case 28: {
            const int channels = static_cast<int>(dims[0]), width = static_cast<int>(dims[1]),
                      seqs = static_cast<int>(dims[2]), taps = static_cast<int>(dims[3]),
                      dilation = static_cast<int>(dims[4]), slots = static_cast<int>(dims[5]);
            const int cols = width * seqs, span = (taps - 1) * dilation;
            DeviceBuffer gt = upload16(inputs[0]), nz = upload16(inputs[1]), w = upload16(inputs[2]);
            DeviceBuffer res = upload16(inputs[3]), hist = upload16(inputs[4]);
            DeviceBuffer src = upload32(inputs[5]), dst = upload32(inputs[6]);
            Tensor tg(gt.p, DType::BF16, {channels, cols}), tn(nz.p, DType::BF16, {channels, cols});
            Tensor tw(w.p, DType::BF16, {channels, taps}), tr(res.p, DType::BF16, {channels, cols});
            Tensor th(hist.p, DType::BF16, {channels, span, slots});
            Tensor tsrc(src.p, DType::I32, {seqs}), tdst(dst.p, DType::I32, {seqs});
            ops::lossless_ple_conv_inject_slots(tg, tn, tw, dilation, seqs, th, tsrc, tdst, tr, nullptr);
            CUDA_CHECK(cudaDeviceSynchronize());
            got = download16(res, static_cast<std::size_t>(channels) * cols);
            const auto fh = download16(hist, static_cast<std::size_t>(channels) * span * slots);
            got.insert(got.end(), fh.begin(), fh.end());
            break;
        }
        case 29: {
            const int dim = static_cast<int>(dims[0]), ratio = static_cast<int>(dims[1]), width = static_cast<int>(dims[2]),
                      seqs = static_cast<int>(dims[3]), slots = static_cast<int>(dims[4]), stride = static_cast<int>(dims[5]),
                      pages = static_cast<int>(dims[6]);
            const int cols = width * seqs, rows = static_cast<int>(inputs[7].size()) / stride;
            DeviceBuffer keys = upload16(inputs[0]), pos = upload32(inputs[1]), tails = upload16(inputs[2]);
            DeviceBuffer tslots = upload32(inputs[3]), rope = upload32(inputs[4]), bstart = upload32(inputs[5]);
            DeviceBuffer plane = upload16(inputs[6]), tables = upload32(inputs[7]), trows = upload32(inputs[8]);
            DeviceBuffer pooled(static_cast<std::size_t>(dim) * cols * 2), brope(static_cast<std::size_t>(3) * cols * 4);
            Tensor tk(keys.p, DType::BF16, {dim, cols}), tp(pos.p, DType::I32, {cols});
            Tensor tt(tails.p, DType::BF16, {dim, ratio - 1, slots}), ts(tslots.p, DType::I32, {seqs});
            Tensor tro(rope.p, DType::I32, {cols, 3}), tbs(bstart.p, DType::I32, {seqs, 3});
            Tensor tpl(plane.p, DType::BF16, {dim / ratio, 64, 1, pages}), ttab(tables.p, DType::I32, {stride, rows});
            Tensor trw(trows.p, DType::I32, {seqs});
            Tensor tpo(pooled.p, DType::BF16, {dim, cols}), tbr(brope.p, DType::I32, {cols, 3});
            ops::lossless_qsa_pool_cache(tk, tp, seqs, tt, ts, tro, tbs, ratio, tpo, tbr, nullptr);
            ops::lossless_qsa_pooled_store(tpo, tp, seqs, ttab, trw, ratio, tpl, nullptr);
            ops::lossless_qsa_tails_store(tk, tp, seqs, tt, ts, ratio, nullptr);
            std::vector<std::uint32_t> gathered;
            for (int b = 0; b < seqs; ++b) {
                const int n = (static_cast<int>(inputs[1][static_cast<std::size_t>(b) * width + width - 1]) + 1) / ratio;
                DeviceBuffer g(static_cast<std::size_t>(dim) * n * 2);
                Tensor tg(g.p, DType::BF16, {dim, n});
                ops::lossless_qsa_pooled_gather(tpl, ttab, trw, tp, seqs, b, ratio, tg, nullptr);
                CUDA_CHECK(cudaDeviceSynchronize());
                const auto part = download16(g, static_cast<std::size_t>(dim) * n);
                gathered.insert(gathered.end(), part.begin(), part.end());
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            got = download16(pooled, static_cast<std::size_t>(dim) * cols);
            const auto br = download32(brope, static_cast<std::size_t>(3) * cols);
            const auto pl = download16(plane, inputs[6].size());
            const auto tl = download16(tails, inputs[2].size());
            got.insert(got.end(), br.begin(), br.end());
            got.insert(got.end(), pl.begin(), pl.end());
            got.insert(got.end(), tl.begin(), tl.end());
            got.insert(got.end(), gathered.begin(), gathered.end());
            break;
        }
        case 30: {
            const int heads = static_cast<int>(dims[0]), ratio = static_cast<int>(dims[1]), budget = static_cast<int>(dims[2]),
                      dim = static_cast<int>(dims[3]), cols = static_cast<int>(dims[4]), blocks = static_cast<int>(dims[5]);
            DeviceBuffer sc = upload32(inputs[0]), pos = upload32(inputs[1]), sel(static_cast<std::size_t>(cols) * 4);
            DeviceBuffer ws(ops::lossless_qsa_select_workspace_bytes(cols, blocks));
            Tensor tsc(sc.p, DType::FP32, {blocks, heads * cols}), tp(pos.p, DType::I32, {cols});
            ops::lossless_qsa_select(tsc, heads, ratio, budget, dim, static_cast<std::uint32_t*>(sel.p), 1, ws.p, nullptr,
                                     &tp);
            CUDA_CHECK(cudaDeviceSynchronize());
            got = download32(sel, static_cast<std::size_t>(cols));
            break;
        }
        case 32: { // lossless_quantize_nvfp4: dims (K, T, divisor bits)
            const int k = static_cast<int>(dims[0]), cols = static_cast<int>(dims[1]);
            float divisor = 0.0F;
            std::memcpy(&divisor, &dims[2], sizeof(divisor));
            DeviceBuffer x = upload16(inputs[0]), out(static_cast<std::size_t>(k) * cols * 2);
            Tensor tx(x.p, DType::BF16, {k, cols}), tout(out.p, DType::BF16, {k, cols});
            ops::lossless_quantize_nvfp4(tx, divisor, tout, nullptr);
            CUDA_CHECK(cudaDeviceSynchronize());
            got = download16(out, static_cast<std::size_t>(k) * cols);
            break;
        }
        case 33: { // lossless_quantize_nvfp4_scaled: dims (K, source columns, T)
            const int k = static_cast<int>(dims[0]), src = static_cast<int>(dims[1]), cols = static_cast<int>(dims[2]);
            DeviceBuffer x = upload16(inputs[0]), map = upload32(inputs[1]), scales = upload32(inputs[2]);
            DeviceBuffer out(static_cast<std::size_t>(k) * cols * 2);
            Tensor tx(x.p, DType::BF16, {k, src}), tout(out.p, DType::BF16, {k, cols});
            ops::lossless_quantize_nvfp4_scaled(tx, static_cast<const std::int32_t*>(map.p),
                                                static_cast<const float*>(scales.p), tout, nullptr);
            CUDA_CHECK(cudaDeviceSynchronize());
            got = download16(out, static_cast<std::size_t>(k) * cols);
            break;
        }
        case 34: { // lossless_quantize_fp8_token: dims (K, T); the values, then the scales
            const int k = static_cast<int>(dims[0]), cols = static_cast<int>(dims[1]);
            DeviceBuffer x = upload16(inputs[0]), out(static_cast<std::size_t>(k) * cols * 2),
                         scales(static_cast<std::size_t>(cols) * 4);
            Tensor tx(x.p, DType::BF16, {k, cols}), tout(out.p, DType::BF16, {k, cols});
            Tensor ts(scales.p, DType::FP32, {cols});
            ops::lossless_quantize_fp8_token(tx, tout, ts, nullptr);
            CUDA_CHECK(cudaDeviceSynchronize());
            got = download16(out, static_cast<std::size_t>(k) * cols);
            const auto s = download32(scales, static_cast<std::size_t>(cols));
            got.insert(got.end(), s.begin(), s.end());
            break;
        }
        case 35: { // lossless_round_fp16, in place
            const int n = static_cast<int>(dims[0]);
            DeviceBuffer x = upload16(inputs[0]);
            Tensor tx(x.p, DType::BF16, {n});
            ops::lossless_round_fp16(tx, nullptr);
            CUDA_CHECK(cudaDeviceSynchronize());
            got = download16(x, static_cast<std::size_t>(n));
            break;
        }
        case 36: { // lossless_quantize_fp8_static: dims (K, T, input scale bits); the values, then the scales
            const int k = static_cast<int>(dims[0]), cols = static_cast<int>(dims[1]);
            float scale = 0.0F;
            std::memcpy(&scale, &dims[2], sizeof(scale));
            DeviceBuffer x = upload16(inputs[0]), out(static_cast<std::size_t>(k) * cols * 2),
                         scales(static_cast<std::size_t>(cols) * 4);
            Tensor tx(x.p, DType::BF16, {k, cols}), tout(out.p, DType::BF16, {k, cols});
            Tensor ts(scales.p, DType::FP32, {cols});
            ops::lossless_quantize_fp8_static(tx, scale, tout, ts, nullptr);
            CUDA_CHECK(cudaDeviceSynchronize());
            got = download16(out, static_cast<std::size_t>(k) * cols);
            const auto s = download32(scales, static_cast<std::size_t>(cols));
            got.insert(got.end(), s.begin(), s.end());
            break;
        }
        case 37: { // lossless_codec_attention: dims (K codec, K rotated, V codec, V rotated, window, Hq, Hkv, T,
                   // sinks, recent, ratio); q, K codes, K scales, V codes, V scales[, the window's four][, select]
            const int kc = static_cast<int>(dims[0]), vc = static_cast<int>(dims[2]);
            const bool window = dims[4] != 0;
            const int heads = static_cast<int>(dims[5]), kv_heads = static_cast<int>(dims[6]), cols = static_cast<int>(dims[7]);
            const int ratio = static_cast<int>(dims[10]);
            DeviceBuffer q = upload16(inputs[0]), out(static_cast<std::size_t>(256) * heads * cols * 2);
            PackedPlane k = pack_plane(kc, inputs[1], inputs[2], kv_heads, cols);
            PackedPlane v = pack_plane(vc, inputs[3], inputs[4], kv_heads, cols);
            ops::LosslessKvCodes kv{};
            kv.k = {static_cast<ops::LosslessKvCodec>(kc), dims[1] != 0, static_cast<const std::uint8_t*>(k.codes.p),
                    static_cast<const std::uint8_t*>(k.scales.p)};
            kv.v = {static_cast<ops::LosslessKvCodec>(vc), dims[3] != 0, static_cast<const std::uint8_t*>(v.codes.p),
                    static_cast<const std::uint8_t*>(v.scales.p)};
            const int int8 = static_cast<int>(ops::LosslessKvCodec::Int8G64);
            PackedPlane wk{DeviceBuffer(1), DeviceBuffer(1)}, wv{DeviceBuffer(1), DeviceBuffer(1)};
            if (window) {
                wk = pack_plane(int8, inputs[5], inputs[6], kv_heads, cols);
                wv = pack_plane(int8, inputs[7], inputs[8], kv_heads, cols);
                kv.window_k = {ops::LosslessKvCodec::Int8G64, true, static_cast<const std::uint8_t*>(wk.codes.p),
                               static_cast<const std::uint8_t*>(wk.scales.p)};
                kv.window_v = {ops::LosslessKvCodec::Int8G64, true, static_cast<const std::uint8_t*>(wv.codes.p),
                               static_cast<const std::uint8_t*>(wv.scales.p)};
                kv.window_sinks  = static_cast<int>(dims[8]);
                kv.window_recent = static_cast<int>(dims[9]);
            }
            DeviceBuffer sel = ratio > 0 ? upload32(inputs.back()) : DeviceBuffer(4);
            DeviceBuffer ws(ops::lossless_codec_attention_workspace_bytes(heads, kv_heads, cols));
            Tensor tq(q.p, DType::BF16, {256, heads, cols}), tout(out.p, DType::BF16, {256, heads, cols});
            ops::lossless_codec_attention(tq, kv, kv_heads, tout, ws.p, nullptr,
                                          ratio > 0 ? static_cast<const std::uint32_t*>(sel.p) : nullptr, ratio > 0 ? 1 : 0,
                                          ratio);
            CUDA_CHECK(cudaDeviceSynchronize());
            got = download16(out, static_cast<std::size_t>(256) * heads * cols);
            break;
        }
        default: std::cerr << "FAIL unknown op " << op << "\n"; return 1;
        }
        std::size_t wrong = 0;
        for (std::size_t i = 0; i < expected.size(); ++i) {
            if (i >= got.size() || got[i] != expected[i]) {
                if (wrong < 5) {
                    std::printf("%s [%zu]: got 0x%x want 0x%x\n", kNames[op], i, i < got.size() ? got[i] : 0U, expected[i]);
                }
                ++wrong;
            }
        }
        std::printf("%s (dims", kNames[op]);
        for (const auto d : dims) { std::printf(" %u", d); }
        std::printf("): %zu of %zu outputs correctly rounded\n", expected.size() - wrong, expected.size());
        failures += wrong != 0;
    }
    if (failures != 0) {
        std::cerr << "FAIL " << failures << " cases differ from the exact results\n";
        return 1;
    }
    return 0;
}
