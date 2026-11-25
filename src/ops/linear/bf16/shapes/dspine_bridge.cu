#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_skinny.cuh"

#include <algorithm>
#include <cstdint>

namespace infernix::ops::detail {
namespace {

// A DSpine drafter's bridge (docs/maintainer/dspine.md): every layer projects its code read [1024, 5120], the shared
// bridge projection [1536, 1024] and its write [5120, 512], 15 BF16 products a 27B round, at the block's columns (up
// to 16 for one row, up to 128 for eight). The runtime-shape fallback tiles a call past eight columns with its SIMT
// GEMM: ninfer-all measured that at 2.8 ms a round on an RTX 5090 at 16 columns, against 0.14 ms streamed, and
// Gate 2 (C:/AI/bench/dspine) found DSpine's Infernix round 2-3 ms dearer than DFlash2's. These shapes stream each
// weight row once (the skinny GEMV), 16 columns a launch. A column's sum does not depend on how many columns share
// the launch (bf16_skinny.cuh), so every width gives a column the bits a one-column call gives it.
constexpr std::int32_t kBridgeColumns    = 16;
constexpr std::int32_t kBridgeMaxColumns = 128;

Bf16Launch bridge_columns(std::int32_t columns) {
    switch (columns) {
    case 1: return launch_bf16_skinny_gemv<1, 8, 2, 0>;
    case 2: return launch_bf16_skinny_gemv<2, 8, 2, 0>;
    case 3: return launch_bf16_skinny_gemv<3, 8, 2, 0>;
    case 4: return launch_bf16_skinny_gemv<4, 8, 2, 0>;
    case 5: return launch_bf16_skinny_gemv<5, 8, 2, 0>;
    case 6: return launch_bf16_skinny_gemv<6, 8, 2, 0>;
    case 7: return launch_bf16_skinny_gemv<7, 8, 2, 0>;
    case 8: return launch_bf16_skinny_gemv<8, 8, 2, 0>;
    case 9: return launch_bf16_skinny_gemv<9, 8, 2, 0>;
    case 10: return launch_bf16_skinny_gemv<10, 8, 2, 0>;
    case 11: return launch_bf16_skinny_gemv<11, 8, 2, 0>;
    case 12: return launch_bf16_skinny_gemv<12, 8, 2, 0>;
    case 13: return launch_bf16_skinny_gemv<13, 8, 2, 0>;
    case 14: return launch_bf16_skinny_gemv<14, 8, 2, 0>;
    case 15: return launch_bf16_skinny_gemv<15, 8, 2, 0>;
    default: return launch_bf16_skinny_gemv<16, 8, 2, 0>;
    }
}

bool aligned16(const void* pointer) { return reinterpret_cast<std::uintptr_t>(pointer) % 16 == 0; }

// Up to 128 columns in launches of 16; wider calls (no bridge makes them) and rows the GEMV cannot read in 16-byte
// chunks take the runtime-shape fallback.
void bridge_gemv(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const std::int32_t columns = x.ne[1];
    if (columns > kBridgeMaxColumns || weight.k % 8 != 0 || !aligned16(x.data) || !aligned16(weight.qdata)) {
        select_bf16_general_launch(columns)(x, weight, out, stream);
        return;
    }
    for (std::int32_t first = 0; first < columns; first += kBridgeColumns) {
        const std::int32_t n = std::min(kBridgeColumns, columns - first);
        const Tensor part_x  = x.slice(1, first, n);
        Tensor part_out      = out.slice(1, first, n);
        bridge_columns(n)(part_x, weight, part_out, stream);
    }
}

} // namespace

Bf16Launch select_bf16_n1024_k5120(std::int32_t tokens) {
    (void)tokens;
    return &bridge_gemv;
}

Bf16Launch select_bf16_n1536_k1024(std::int32_t tokens) {
    (void)tokens;
    return &bridge_gemv;
}

Bf16Launch select_bf16_n5120_k512(std::int32_t tokens) {
    (void)tokens;
    return &bridge_gemv;
}

} // namespace infernix::ops::detail
