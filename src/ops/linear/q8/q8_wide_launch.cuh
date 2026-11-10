#pragma once

// The widest Q8 A16 GEMM route of an Op, with the Op's own output and epilogue: the staging ring
// (ops/linear/common/rowsplit_ring_linear.cuh) from ring_min_tokens columns, the 64-row MMA tile below. Both give
// the same bits (the same FP32 chain of m16n8k16 MMAs over the same decoded BF16 weights).

#include "ops/linear/common/rowsplit_ring_linear.cuh"
#include "ops/linear/q8/q8_mma_launch.cuh"
#include "ops/linear/q8/q8_ring_codec.cuh"

#include <cstdint>

namespace infernix::ops::detail {

// The fused Q8 projections' first width on the ring. RTX 5090, their benches (CUDA graph, cold L2, A B B A,
// 2026-11-10), ring against the 64-row MMA tile at 1,024 / 2,048 / 3,584 / 4,096 columns: GDN input [12288, 2048]
// -4.0 / +0.1 / -4.5 / -2.1 %; attention input [9216, 2048] +4.8 / -4.7 / -4.6 / -3.9 % and [6144, 2048] +8.0 /
// -4.6 / -5.1 / -0.2 %; residual add [2048, 4096] -10.2 / -7.6 / -6.5 / +5.7 % and [2048, 6144] -11.4 / -7.2 / -6.3 /
// +5.8 % (the two losses are the launch q8_ring_fills_waves turns away). The 35B's prefill: -1.4 % at 16,384 tokens.
inline constexpr std::int32_t kQ8InputRingMinTokens = 2048;
inline constexpr std::int32_t kQ8AddRingMinTokens   = 1024;

// Whether a ring launch of rows x tokens fills its waves: a ring CTA owns its SM, so a last wave of a few CTAs
// costs a whole wave's time, which matters while the launch is only a few waves (Q8 linear_add, 2,048 rows x
// 4,096 columns: 512 CTAs on 170 SMs, three waves and two CTAs, +5.8 % against the MMA tile; 3,584 columns,
// 448 CTAs, -6.5 %).
inline bool q8_ring_fills_waves(std::int32_t rows, std::int32_t tokens) {
    const int sms = rowsplit_ring_sms();
    const int ctas = div_up(rows, RowSplitRingTile128::BM) * div_up(tokens, RowSplitRingTile128::BN);
    const int tail = ctas % sms;
    return tail == 0 || tail * 4 >= sms || ctas / sms >= 8;
}

// Schedule: the Op's MMA tile for calls below ring_min_tokens.
template <class Schedule, class Output, class Epilogue>
void launch_q8_a16_wide(const Q8LinearOperands& p, Output output, Epilogue epilogue, std::int32_t ring_min_tokens,
                        cudaStream_t stream) {
    if (p.tokens >= ring_min_tokens && q8_ring_fills_waves(p.rows, p.tokens)) {
        validate_q8_operands(p);
        launch_rowsplit_ring_linear<Q8RingFormat, RowSplitRingTile128, kRowSplitRingStages, kRowSplitRingProducerWarps>(
            p.x, Q8RingFormat::Weights{p.codes, p.scales}, p.rows, p.k, p.padded_k, p.tokens, output, epilogue, stream);
        return;
    }
    launch_q8_a16_mma<Schedule>(p, output, epilogue, stream);
}

} // namespace infernix::ops::detail
