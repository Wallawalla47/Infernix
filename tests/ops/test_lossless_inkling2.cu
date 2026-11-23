// The lossless route's Inkling 2 operations (include/infernix/ops/lossless_inkling2.h) against
// lossless_inkling2_oracle.py's exact results: run as python lossless_inkling2_oracle.py <this executable>, which
// passes the case file (per case: op, dims, inputs, expected bits). Every output must match.

#include "ops/op_family.h"
#include "core/arena.h"
#include "core/device.h"
#include "infernix/ops/lossless_inkling2.h"
#include "infernix/ops/lossless_linear.h"
#include "ops/lossless/wide.cuh"

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

DeviceBuffer upload16(const std::vector<std::uint32_t>& v, std::size_t at_least = 0) {
    std::vector<std::uint16_t> h(std::max(v.size(), at_least), 0);
    for (std::size_t i = 0; i < v.size(); ++i) { h[i] = static_cast<std::uint16_t>(v[i]); }
    DeviceBuffer d(std::max<std::size_t>(h.size() * 2, 4));
    if (!h.empty()) { d.copy_from_host(h.data(), h.size() * 2); }
    return d;
}

DeviceBuffer upload32(const std::vector<std::uint32_t>& v, std::size_t at_least = 0) {
    std::vector<std::uint32_t> h(std::max(v.size(), at_least), 0);
    std::copy(v.begin(), v.end(), h.begin());
    DeviceBuffer d(std::max<std::size_t>(h.size() * 4, 4));
    if (!h.empty()) { d.copy_from_host(h.data(), h.size() * 4); }
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

// A [rows, columns] tensor over a buffer whose columns are `ld` elements apart.
Tensor strided(void* p, DType dtype, int rows, int columns, int ld) {
    Tensor t(p, dtype, {rows, columns});
    t.nb[1] = static_cast<std::int64_t>(ld) * static_cast<std::int64_t>(dtype_size(dtype));
    t.nb[2] = t.nb[1] * columns;
    t.nb[3] = t.nb[2];
    return t;
}

float bits_float(std::uint32_t b) {
    float f;
    std::memcpy(&f, &b, 4);
    return f;
}

const char* kNames[] = {"rmsnorm", "mix",   "message_gate", "clip",         "conv",   "dynamic_conv",
                        "sigmoid_gate", "add", "probs",     "router",       "four_over_six", "mxfp8",
                        "swiglu",  "combine", "softcap", "sqrt_magnitude", "ring_load", "ring_store", "keys_load",
                        "keys_store", "linear_planes"};

// wide.cuh's sqrt_magnitude over ProductSum-sized magnitudes (the clips' norms): case i's 20 words, then its unit.
__global__ void sqrt_cases(const std::uint32_t* words, int n, std::uint32_t* out) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) { return; }
    std::uint32_t mag[ops::lossless::kProductSumWords];
    for (int w = 0; w < ops::lossless::kProductSumWords; ++w) { mag[w] = words[static_cast<std::size_t>(i) * 21 + w]; }
    const int unit = static_cast<int>(words[static_cast<std::size_t>(i) * 21 + 20]);
    out[i]         = __float_as_uint(ops::lossless::sqrt_magnitude(mag, unit));
}

} // namespace

INFERNIX_OP_TEST(lossless_inkling2) {
    if (argc != 2) {
        std::cerr << "usage: test_lossless_inkling2 <cases.bin> (run through lossless_inkling2_oracle.py)\n";
        return 2;
    }
    std::ifstream in(argv[1], std::ios::binary);
    int failures = 0;
    while (in.peek() != std::char_traits<char>::eof()) {
        const auto head = read_u32(in, 2);
        const std::uint32_t op = head[0];
        const auto d = read_u32(in, head[1]);
        const auto ninputs = read_u32(in, 1)[0];
        const auto sizes = read_u32(in, ninputs);
        const auto nexpected = read_u32(in, 1)[0];
        std::vector<std::vector<std::uint32_t>> inputs;
        for (const auto s : sizes) { inputs.push_back(read_u32(in, s)); }
        const auto expected = read_u32(in, nexpected);
        if (!in || op >= sizeof(kNames) / sizeof(kNames[0])) {
            std::cerr << "FAIL could not read the cases\n";
            return 1;
        }
        const auto I = [&](int i) { return static_cast<int>(d[i]); };
        // The softmaxes (mix, probs) run twice: their binary64 route, and every element through the double-double
        // fallback (lossless_inkling2_force_exact_fallback), each against the same exact results.
        const int passes = op == 1 || op == 8 ? 2 : 1;
        for (int pass = 0; pass < passes; ++pass) {
        ops::lossless_inkling2_force_exact_fallback(pass == 1);
        std::vector<std::uint32_t> got;
        switch (op) {
        case 0: { // rmsnorm: width, rows, x_fp32, weighted, out_fp32, out_ld
            const int width = I(0), rows = I(1), ld = I(5);
            const bool x32 = d[2] != 0, weighted = d[3] != 0, o32 = d[4] != 0;
            DeviceBuffer x = x32 ? upload32(inputs[0]) : upload16(inputs[0]);
            DeviceBuffer w = upload16(inputs[1]);
            DeviceBuffer out(static_cast<std::size_t>(ld) * rows * (o32 ? 4 : 2));
            Tensor tx(x.p, x32 ? DType::FP32 : DType::BF16, {width, rows}), tw(w.p, DType::BF16, {width});
            Tensor to = strided(out.p, o32 ? DType::FP32 : DType::BF16, width, rows, ld);
            ops::lossless_inkling2_rmsnorm(tx, weighted ? &tw : nullptr, 1e-6F, to, nullptr);
            CUDA_CHECK(cudaDeviceSynchronize());
            const auto all = o32 ? download32(out, static_cast<std::size_t>(ld) * rows)
                                 : download16(out, static_cast<std::size_t>(ld) * rows);
            for (int r = 0; r < rows; ++r) {
                got.insert(got.end(), all.begin() + static_cast<std::ptrdiff_t>(r) * ld,
                           all.begin() + static_cast<std::ptrdiff_t>(r) * ld + width);
            }
            break;
        }
        case 1: { // mix: H, T, sources, queries, kinds...
            const int H = I(0), T = I(1), S = I(2), Q = I(3);
            std::vector<DeviceBuffer> src;
            std::vector<Tensor> ts;
            for (int j = 0; j < S; ++j) {
                const bool fp32 = d[4 + j] != 0;
                src.push_back(fp32 ? upload32(inputs[j]) : upload16(inputs[j]));
                ts.emplace_back(src.back().p, fp32 ? DType::FP32 : DType::BF16, std::initializer_list<std::int32_t>{H, T});
            }
            DeviceBuffer q = upload32(inputs[S]);
            std::vector<DeviceBuffer> outs;
            std::vector<Tensor> to;
            for (int k = 0; k < Q; ++k) {
                outs.emplace_back(static_cast<std::size_t>(H) * T * 4);
                to.emplace_back(outs.back().p, DType::FP32, std::initializer_list<std::int32_t>{H, T});
            }
            std::vector<const Tensor*> sp;
            for (auto& t : ts) { sp.push_back(&t); }
            std::vector<Tensor*> op_out;
            for (auto& t : to) { op_out.push_back(&t); }
            Tensor tq(q.p, DType::FP32, {H, Q});
            ops::lossless_inkling2_mix(sp, tq, op_out, nullptr);
            CUDA_CHECK(cudaDeviceSynchronize());
            for (int k = 0; k < Q; ++k) {
                const auto v = download32(outs[k], static_cast<std::size_t>(H) * T);
                got.insert(got.end(), v.begin(), v.end());
            }
            break;
        }
        case 2: { // message gate: n
            const int n = I(0);
            DeviceBuffer f = upload32(inputs[0]), g = upload32(inputs[1]), x = upload32(inputs[2]), m = upload32(inputs[3]);
            DeviceBuffer out(static_cast<std::size_t>(n) * 4);
            Tensor tf(f.p, DType::FP32, {n}), tg(g.p, DType::FP32, {n}), tx(x.p, DType::FP32, {n}), tm(m.p, DType::FP32, {n}),
                to(out.p, DType::FP32, {n});
            ops::lossless_inkling2_message_gate(tf, tg, tx, tm, to, nullptr);
            CUDA_CHECK(cudaDeviceSynchronize());
            got = download32(out, n);
            break;
        }
        case 3: { // clip: D, heads, T, scaled, logscale, threshold, alpha, floor
            const int D = I(0), heads = I(1), T = I(2);
            const bool scaled = d[3] != 0, logscale = d[4] != 0;
            DeviceBuffer x = upload16(inputs[0]), s = upload16(inputs[1]), p = upload32(inputs[2]);
            DeviceBuffer out(static_cast<std::size_t>(D) * heads * T * 2);
            Tensor tx(x.p, DType::BF16, {D * heads, T}), ts(s.p, DType::BF16, {D, heads}), tp(p.p, DType::I32, {T}),
                to(out.p, DType::BF16, {D * heads, T});
            ops::LosslessLogScale ls;
            if (logscale) {
                ls.positions = &tp;
                ls.alpha     = bits_float(d[6]);
                ls.floor     = static_cast<float>(d[7]);
            }
            ops::lossless_inkling2_clip(tx, scaled ? &ts : nullptr, heads, bits_float(d[5]), ls, to, nullptr);
            CUDA_CHECK(cudaDeviceSynchronize());
            got = download16(out, static_cast<std::size_t>(D) * heads * T);
            break;
        }
        case 4: { // conv: C, T, k
            const int C = I(0), T = I(1), k = I(2);
            DeviceBuffer x = upload16(inputs[0]), w = upload16(inputs[1]), out(static_cast<std::size_t>(C) * T * 2);
            Tensor tx(x.p, DType::BF16, {C, T}), tw(w.p, DType::BF16, {C, k}), to(out.p, DType::BF16, {C, T});
            ops::lossless_inkling2_conv(tx, tw, to, nullptr);
            CUDA_CHECK(cudaDeviceSynchronize());
            got = download16(out, static_cast<std::size_t>(C) * T);
            break;
        }
        case 5: { // dynamic conv: C, heads, T, k
            const int C = I(0), heads = I(1), T = I(2), k = I(3);
            DeviceBuffer x = upload16(inputs[0]), dyn = upload16(inputs[1]), w = upload16(inputs[2]);
            DeviceBuffer out(static_cast<std::size_t>(C) * T * 2);
            Tensor tx(x.p, DType::BF16, {C, T}), td(dyn.p, DType::BF16, {heads * k, T}), tw(w.p, DType::BF16, {C, k}),
                to(out.p, DType::BF16, {C, T});
            ops::lossless_inkling2_dynamic_conv(tx, td, tw, heads, to, nullptr);
            CUDA_CHECK(cudaDeviceSynchronize());
            got = download16(out, static_cast<std::size_t>(C) * T);
            break;
        }
        case 6: { // sigmoid gate: n
            const int n = I(0);
            DeviceBuffer o = upload16(inputs[0]), g = upload16(inputs[1]), out(static_cast<std::size_t>(n) * 2);
            Tensor to_(o.p, DType::BF16, {n, 1}), tg(g.p, DType::BF16, {n, 1}), tout(out.p, DType::BF16, {n, 1});
            ops::lossless_inkling2_sigmoid_gate(to_, tg, tout, nullptr);
            CUDA_CHECK(cudaDeviceSynchronize());
            got = download16(out, n);
            break;
        }
        case 7: { // add: n, a32, b32, o32, via
            const int n = I(0);
            const bool a32 = d[1] != 0, b32 = d[2] != 0, o32 = d[3] != 0, via = d[4] != 0;
            DeviceBuffer a = a32 ? upload32(inputs[0]) : upload16(inputs[0]);
            DeviceBuffer b = b32 ? upload32(inputs[1]) : upload16(inputs[1]);
            DeviceBuffer out(static_cast<std::size_t>(n) * (o32 ? 4 : 2));
            Tensor ta(a.p, a32 ? DType::FP32 : DType::BF16, {n}), tb(b.p, b32 ? DType::FP32 : DType::BF16, {n}),
                to(out.p, o32 ? DType::FP32 : DType::BF16, {n});
            ops::lossless_inkling2_add(ta, tb, to, nullptr, via);
            CUDA_CHECK(cudaDeviceSynchronize());
            got = o32 ? download32(out, n) : download16(out, n);
            break;
        }
        case 8: { // probs: keys, ld, T, begin, window, head, heads, rel, extent, scale, heads in the call
            const int keys = I(0), ld = I(1), T = I(2), heads = I(6), rel = I(7), extent = I(8), count = I(10);
            DeviceBuffer s = upload16(inputs[0]), r = upload16(inputs[1]), tab = upload16(inputs[2]), sk = upload16(inputs[3]);
            Tensor ts(s.p, DType::BF16, {ld, T * count}), tr(r.p, DType::BF16, {rel * heads, T}),
                tt(tab.p, DType::BF16, {rel, extent}), tk(sk.p, DType::BF16, {heads, T});
            ops::LosslessInkling2Probs a;
            a.scores      = &ts;
            a.ld          = ld;
            a.keys        = keys;
            a.query_begin = I(3);
            a.window      = I(4);
            a.head        = I(5);
            a.head_count  = count;
            a.heads       = heads;
            a.relative    = &tr;
            a.table       = &tt;
            a.sinks       = &tk;
            a.scale       = bits_float(d[9]);
            ops::lossless_inkling2_attention_probs(a, nullptr);
            CUDA_CHECK(cudaDeviceSynchronize());
            got = download16(s, static_cast<std::size_t>(ld) * T * count);
            break;
        }
        case 9: { // router: E, k, T, scaling
            const int E = I(0), k = I(1), T = I(2);
            DeviceBuffer l = upload32(inputs[0]), b = upload32(inputs[1]);
            DeviceBuffer w(static_cast<std::size_t>(k) * T * 4), id(static_cast<std::size_t>(k) * T * 4);
            Tensor tl(l.p, DType::FP32, {E, T}), tb(b.p, DType::FP32, {E}), tw(w.p, DType::FP32, {k, T}), ti(id.p, DType::I32, {k, T});
            ops::lossless_inkling2_router(tl, tb, k, bits_float(d[3]), tw, ti, nullptr);
            CUDA_CHECK(cudaDeviceSynchronize());
            got = download32(w, static_cast<std::size_t>(k) * T);
            const auto ids = download32(id, static_cast<std::size_t>(k) * T);
            got.insert(got.end(), ids.begin(), ids.end());
            break;
        }
        case 10:
        case 11: { // quantizers: K, T, columns
            const int K = I(0), T = I(1), C = I(2);
            DeviceBuffer x = upload16(inputs[0]), cols = upload32(inputs[1]);
            DeviceBuffer out(static_cast<std::size_t>(K) * C * 2), outer(static_cast<std::size_t>(C) * 4);
            Tensor tx(x.p, DType::BF16, {K, T}), to(out.p, DType::BF16, {K, C}), tou(outer.p, DType::FP32, {C});
            if (op == 10) {
                ops::lossless_inkling2_quantize_four_over_six(tx, static_cast<const std::int32_t*>(cols.p), C, to, tou, nullptr);
            } else {
                ops::lossless_inkling2_quantize_mxfp8(tx, static_cast<const std::int32_t*>(cols.p), C, to, nullptr);
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            got = download16(out, static_cast<std::size_t>(K) * C);
            if (op == 10) {
                const auto o = download32(outer, C);
                got.insert(got.end(), o.begin(), o.end());
            }
            break;
        }
        case 12: { // swiglu: I, C, gate cap, up cap
            const int In = I(0), C = I(1);
            DeviceBuffer gu = upload16(inputs[0]), g = upload32(inputs[1]), out(static_cast<std::size_t>(In) * C * 2);
            Tensor tgu(gu.p, DType::BF16, {2 * In, C}), tg(g.p, DType::FP32, {C}), to(out.p, DType::BF16, {In, C});
            ops::lossless_inkling2_swiglu(tgu, tg, bits_float(d[2]), bits_float(d[3]), to, nullptr);
            CUDA_CHECK(cudaDeviceSynchronize());
            got = download16(out, static_cast<std::size_t>(In) * C);
            break;
        }
        case 13: { // combine: H, T, k, entries
            const int H = I(0), T = I(1), k = I(2), entries = I(3);
            DeviceBuffer y = upload16(inputs[0]), sc = upload32(inputs[1]), out(static_cast<std::size_t>(H) * T * 2);
            Tensor ty(y.p, DType::BF16, {H, entries}), ts(sc.p, DType::I32, {T * k}), to(out.p, DType::BF16, {H, T});
            ops::lossless_inkling2_combine(ty, ts, k, to, nullptr);
            CUDA_CHECK(cudaDeviceSynchronize());
            got = download16(out, static_cast<std::size_t>(H) * T);
            break;
        }
        case 14: { // softcap: n, cap
            const int n = I(0);
            DeviceBuffer x = upload32(inputs[0]), out(static_cast<std::size_t>(n) * 4);
            Tensor tx(x.p, DType::FP32, {n}), to(out.p, DType::FP32, {n});
            ops::lossless_inkling2_softcap(tx, bits_float(d[1]), to, nullptr);
            CUDA_CHECK(cudaDeviceSynchronize());
            got = download32(out, n);
            break;
        }
        case 15: { // sqrt_magnitude: per case 20 magnitude words and the unit (two's complement)
            const int n = I(0);
            DeviceBuffer w = upload32(inputs[0]), out(static_cast<std::size_t>(n) * 4);
            sqrt_cases<<<(n + 127) / 128, 128>>>(static_cast<const std::uint32_t*>(w.p), n, static_cast<std::uint32_t*>(out.p));
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            got = download32(out, n);
            break;
        }
        case 16: { // ring_load: rows, C, first, count, columns, ld (the whole destination compared)
            const int rows = I(0), C = I(1), columns = I(4), ld = I(5);
            DeviceBuffer ring = upload16(inputs[0]), out = upload16(inputs[1]);
            Tensor to = strided(out.p, DType::BF16, C, columns, ld);
            ops::lossless_inkling2_ring_load(static_cast<const std::uint16_t*>(ring.p), rows, I(2), I(3), to, nullptr);
            CUDA_CHECK(cudaDeviceSynchronize());
            got = download16(out, static_cast<std::size_t>(ld) * columns);
            break;
        }
        case 17: { // ring_store: rows, C, first, T, ld
            const int rows = I(0), C = I(1), T = I(3), ld = I(4);
            DeviceBuffer x = upload16(inputs[0]), ring = upload16(inputs[1]);
            const Tensor tx = strided(x.p, DType::BF16, C, T, ld);
            ops::lossless_inkling2_ring_store(tx, I(2), static_cast<std::uint16_t*>(ring.p), rows, nullptr);
            CUDA_CHECK(cudaDeviceSynchronize());
            got = download16(ring, static_cast<std::size_t>(rows) * C);
            break;
        }
        case 18: { // keys_load: heads, ring pages (0: a pool), pages, first, count, stride
            const int heads = I(0), ring_pages = I(1), stride = I(5);
            DeviceBuffer k = upload16(inputs[0]), v = upload16(inputs[1]), table = upload32(inputs[2]);
            DeviceBuffer kh = upload16(inputs[3]), vt = upload16(inputs[4]);
            ops::LosslessInkling2Keys s;
            s.k          = static_cast<std::uint16_t*>(k.p);
            s.v          = static_cast<std::uint16_t*>(v.p);
            s.table      = ring_pages != 0 ? nullptr : static_cast<const std::int32_t*>(table.p);
            s.ring_pages = ring_pages;
            s.heads      = heads;
            s.dim        = 128;
            Tensor tk(kh.p, DType::BF16, {128, stride, heads}), tv(vt.p, DType::BF16, {stride, 128, heads});
            ops::lossless_inkling2_keys_load(s, I(3), I(4), stride, tk, tv, nullptr);
            CUDA_CHECK(cudaDeviceSynchronize());
            const std::size_t n = static_cast<std::size_t>(heads) * 128 * stride;
            got = download16(kh, n);
            const auto values = download16(vt, n);
            got.insert(got.end(), values.begin(), values.end());
            break;
        }
        case 19: { // keys_store: heads, ring pages (0: a pool), pages, first, T, ld
            const int heads = I(0), ring_pages = I(1), pages = I(2), T = I(4), ld = I(5);
            DeviceBuffer xk = upload16(inputs[0]), xv = upload16(inputs[1]), table = upload32(inputs[2]);
            DeviceBuffer k = upload16(inputs[3]), v = upload16(inputs[4]);
            ops::LosslessInkling2Keys s;
            s.k          = static_cast<std::uint16_t*>(k.p);
            s.v          = static_cast<std::uint16_t*>(v.p);
            s.table      = ring_pages != 0 ? nullptr : static_cast<const std::int32_t*>(table.p);
            s.ring_pages = ring_pages;
            s.heads      = heads;
            s.dim        = 128;
            const Tensor tk = strided(xk.p, DType::BF16, heads * 128, T, ld), tv = strided(xv.p, DType::BF16, heads * 128, T, ld);
            ops::lossless_inkling2_keys_store(tk, tv, I(3), s, nullptr);
            CUDA_CHECK(cudaDeviceSynchronize());
            const std::size_t n = static_cast<std::size_t>(ring_pages != 0 ? ring_pages : pages) * heads * 64 * 128;
            got = download16(k, n);
            const auto values = download16(v, n);
            got.insert(got.end(), values.begin(), values.end());
            break;
        }
        case 20: { // lossless_linear_planes: K, rows, stride, planes, plane columns, FP32 out, out rows (the whole out compared)
            const int K = I(0), rows = I(1), stride = I(2), planes = I(3), plane_columns = I(4), out_rows = I(6);
            const bool fp32 = d[5] != 0;
            DeviceBuffer x = upload16(inputs[0]), w = upload16(inputs[1]), tile_planes = upload32(inputs[2]);
            DeviceBuffer out = fp32 ? upload32(inputs[3]) : upload16(inputs[3]);
            const int columns = planes * plane_columns;
            Tensor tx(x.p, DType::BF16, {K, columns}), tp(tile_planes.p, DType::I32, {planes});
            Tensor to(out.p, fp32 ? DType::FP32 : DType::BF16, {out_rows, columns});
            Weight weight{};
            weight.qtype    = QType::BF16;
            weight.layout   = QuantLayout::Contiguous;
            weight.ndim     = 2;
            weight.n        = planes * stride;
            weight.k        = K;
            weight.shape[0] = weight.padded_shape[0] = weight.n;
            weight.shape[1] = weight.padded_shape[1] = K;
            weight.qdata    = w.p;
            weight.payload  = w.p;
            WorkspaceArena arena(ops::lossless_linear_planes_workspace_bytes(weight, planes, stride, plane_columns, rows) + 4096);
            ops::lossless_linear_planes(tx, weight, tp, stride, plane_columns, rows, to, arena, nullptr);
            CUDA_CHECK(cudaDeviceSynchronize());
            const std::size_t n = static_cast<std::size_t>(out_rows) * columns;
            got = fp32 ? download32(out, n) : download16(out, n);
            break;
        }
        default: std::cerr << "FAIL unknown op " << op << "\n"; return 1;
        }
        const bool undecided = ops::lossless_inkling2_take_undecided();
        std::size_t wrong = 0;
        for (std::size_t i = 0; i < expected.size(); ++i) {
            if (i >= got.size() || got[i] != expected[i]) {
                if (wrong < 5) {
                    std::printf("%s [%zu]: got 0x%x want 0x%x\n", kNames[op], i, i < got.size() ? got[i] : 0U, expected[i]);
                }
                ++wrong;
            }
        }
        std::printf("%s%s (dims", kNames[op], pass == 1 ? " (double-double fallback)" : "");
        for (const auto v : d) { std::printf(" %u", v); }
        std::printf("): %zu of %zu outputs correctly rounded%s\n", expected.size() - wrong, expected.size(),
                    undecided ? ", an element left undecided" : "");
        failures += wrong != 0 || undecided;
        }
        ops::lossless_inkling2_force_exact_fallback(false);
    }
    if (failures != 0) {
        std::cerr << "FAIL " << failures << " cases differ from the exact results\n";
        return 1;
    }
    return 0;
}
