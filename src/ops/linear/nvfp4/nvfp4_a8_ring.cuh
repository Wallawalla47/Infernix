#pragma once

// The 8-bit NVFP4 128 x 128 prefill tile (LinearPolicy::A8Only, nvfp4_a8_mma.cuh's arithmetic) on a decode-warp ring:
// one producer warpgroup stages each K tile (the activations' codes and slice scales and the weights' block scales by
// cp.async; the weight codes loaded into registers and widened to int8 into the stage) and publishes it on the
// stage's full barrier; eight consumer warps (2 x 4, 64 x 32 outputs each) wait for it, run its MMAs and release it on
// its empty barrier. The MMA warps never stop for a widen pass or a CTA barrier, which nvfp4_a8_mma.cuh's
// barrier-synchronised 128 x 128 MMA tile (these widths' tile until 2026-11-19) ran twice a K tile. Every output is
// that tile's: the same int8 operands at the same K positions, one m16n8k16 s8 MMA a block and fma(P_b S_b, d_s, acc)
// in K order per thread, so the bits are its (compared when the ring replaced it: ops::linear bit for bit at 129 and
// 257 columns, and the --a8 quick-corpus perplexity of NVIDIA's NVFP4-QAD 27B, each of its 124 windows' NLL equal).
//
// The producers hand registers to the consumers (setmaxnreg: 168 at launch, three warps a sub-partition; producers
// 40, consumers 232), which relocatable device code drops (src/ops/CMakeLists.txt): the kernel is built only in the
// families' non-RDC translation units, behind nvfp4_a8_ring.h's launchers. The grid runs the token tiles fastest, so
// the CTAs running together share their weight rows in L2.
//
// Measured (RTX 5090, 2026-11-19, NVIDIA's NVFP4-QAD 27B shapes, op benches, ABBA, rounds within 0.5 %): against the
// barrier-synchronised 128 x 128 MMA tile the ring took 0.86-0.90 of its time on the gate/up [34816,5120] at 65-3,584
// columns and 0.87-0.92 on the 5,120-row projections (K 6144 and 17408) at 257-3,584, up to 224 TOPS; --a8 prefill on
// the exact artifact (INT8 KV, ABBA) pp128 / 256 / 384 / 512 / 1024 +7.5 / +5.1 / +9.1 / +7.9 / +7.2 %, pp4096
// 3,974 -> 4,287 and pp16384 3,875 -> 4,167 tok/s (+7.9 and +7.5 %). Measured and not taken: the grid's row tiles
// fastest (the gate/up level with the MMA tile from 1,792 columns: each token tile then rereads the 89 MB of weights),
// BK 64 with three or four stages (0.97-1.09), and the MMA tile with its token tiles fastest (gate/up 0.94-1.01,
// 5,120-row projections 1.00-1.05). The investigation before it (design 19.3.31 (8)): ncu found the MMA tile's tensor
// pipe 44 % and issue 57 % busy at two warps a sub-partition; 16-warp layouts, A fragments decoded in registers, the
// conversion by an integer magic add and two 64-deep CTAs an SM all ran slower.

#include "ops/common/mbarrier.cuh"
#include "ops/linear/nvfp4/nvfp4_a8_mma.cuh"

namespace infernix::ops::detail {

struct Nvfp4A8RingSchedule {
    static constexpr int kBlockRows         = 128;
    static constexpr int kBlockTokens       = 128;
    static constexpr int kBlockK            = 128;
    static constexpr int kStages            = 2; // two 40 KB stages: a third does not fit 99 KB
    static constexpr int kWarpsRows         = 2;
    static constexpr int kWarpsTokens       = 4;
    static constexpr int kConsumerWarps     = kWarpsRows * kWarpsTokens;
    static constexpr int kConsumerThreads   = kConsumerWarps * 32;
    static constexpr int kProducerThreads   = 128; // one warpgroup: setmaxnreg acts on whole warpgroups
    static constexpr int kThreads           = kConsumerThreads + kProducerThreads;
    static constexpr int kWarpRows          = kBlockRows / kWarpsRows;
    static constexpr int kWarpTokens        = kBlockTokens / kWarpsTokens;
    static constexpr int kMmaRows           = kWarpRows / 16;
    static constexpr int kMmaTokens         = kWarpTokens / 8;
    static constexpr int kRowBytes          = kBlockK + 16; // a widened weight row or an activation row, padded
    static constexpr int kWeightBytes       = kBlockRows * kRowBytes;
    static constexpr int kScaleBytes        = kBlockRows * (kBlockK / 16);
    static constexpr int kXBytes            = kBlockTokens * kRowBytes;
    static constexpr int kXScaleBytes       = kBlockTokens * (kBlockK / 32) * 4;
    static constexpr int kStageBytes        = kWeightBytes + kScaleBytes + kXBytes + kXScaleBytes;
    static constexpr int kTableBytes        = 256 * 4;
    static constexpr int kBarrierOffset     = kTableBytes + kStages * kStageBytes;
    static constexpr int kSharedBytes       = kBarrierOffset + kStages * 2 * 8;
    // 12 warps put three on each sub-partition (16,384 registers): 170 a thread, 168 in units of 8.
    static constexpr int kLaunchRegisters   = 168;
    static constexpr int kProducerRegisters = 40;
    // A sub-partition's producer warp releases (168 - 40) x 32 registers to its two consumer warps: 232 each. One
    // consumer instance (the gate/up's full tiles) keeps a uniform address in local memory, one load a K tile beside
    // its 128 MMAs (cuda_stack_allowlist.json); producers at 24 registers (consumers 240) spilled 16-72 bytes.
    static constexpr int kConsumerRegisters = kLaunchRegisters + (kLaunchRegisters - kProducerRegisters) / 2 / 8 * 8;
    // Weight-code chunks (32 codes, 16 bytes) a producer thread loads and widens a K tile.
    static constexpr int kChunks = kBlockRows * (kBlockK / 32) / kProducerThreads;
    static_assert(kChunks >= 1 && kBlockRows * (kBlockK / 32) % kProducerThreads == 0);
    static_assert(kStageBytes % 16 == 0 && kWeightBytes % 16 == 0 && kScaleBytes % 16 == 0 && kXBytes % 16 == 0);
    static_assert(kSharedBytes <= 99 * 1024 && kConsumerRegisters <= 248);
};

template <class Schedule, bool FullTokens, class Output, class Epilogue, class RowPolicy>
__global__ __launch_bounds__(Schedule::kThreads, 1) void nvfp4_a8_ring_kernel(
    Nvfp4A8Operands operands, Output output, Epilogue epilogue, RowPolicy row_policy, int token_offset, int count) {
#ifdef __CUDACC_RDC__
    static_assert(sizeof(Schedule) == 0, "the 8-bit ring needs a non-RDC translation unit (setmaxnreg)");
#endif
    constexpr int BM  = Schedule::kBlockRows;
    constexpr int BN  = Schedule::kBlockTokens;
    constexpr int BK  = Schedule::kBlockK;
    constexpr int S   = Schedule::kStages;
    constexpr int WM  = Schedule::kWarpRows;
    constexpr int WN  = Schedule::kWarpTokens;
    constexpr int MT  = Schedule::kMmaRows;
    constexpr int NT  = Schedule::kMmaTokens;
    constexpr int ROW = Schedule::kRowBytes;
    const int M       = operands.rows;
    const int K       = operands.k;
    const int tokens  = token_offset + count;
    const int k_tiles = K / BK;

    extern __shared__ __align__(16) unsigned char nvfp4_a8_ring_shared[];
    auto* table          = reinterpret_cast<float*>(nvfp4_a8_ring_shared);
    unsigned char* ring  = nvfp4_a8_ring_shared + Schedule::kTableBytes;
    auto* full           = reinterpret_cast<std::uint64_t*>(nvfp4_a8_ring_shared + Schedule::kBarrierOffset);
    auto* empty          = full + S;
    const auto weight_at = [&](int s) { return ring + s * Schedule::kStageBytes; };
    const auto scales_at = [&](int s) { return weight_at(s) + Schedule::kWeightBytes; };
    const auto x_at      = [&](int s) { return scales_at(s) + Schedule::kScaleBytes; };
    const auto xs_at     = [&](int s) { return reinterpret_cast<float*>(x_at(s) + Schedule::kXBytes); };

    const int tid         = static_cast<int>(threadIdx.x);
    const int row_begin   = static_cast<int>(blockIdx.y) * (BM / (RowPolicy::kPaired ? 2 : 1));
    const int token_begin = token_offset + static_cast<int>(blockIdx.x) * BN;

    nvfp4_a8_fill_scale_table(table, tid, Schedule::kThreads);
    if (tid == 0) {
#pragma unroll
        for (int s = 0; s < S; ++s) {
            cta_mbarrier_init(full + s, Schedule::kProducerThreads);
            cta_mbarrier_init(empty + s, Schedule::kConsumerWarps);
        }
        cta_mbarrier_fence_init();
    }
    __syncthreads();

    if (tid >= Schedule::kConsumerThreads) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(Schedule::kProducerRegisters) : "memory");
        const int p = tid - Schedule::kConsumerThreads;
        constexpr int kXParts = BK / 16;
        for (int kt = 0; kt < k_tiles; ++kt) {
            const int s  = kt % S;
            const int k0 = kt * BK;
            // The weight codes first: their loads are in flight while the copies below are issued.
            // The weight codes first: their loads are in flight across the empty-barrier wait and the copies below.
            uint4 raw[Schedule::kChunks];
#pragma unroll
            for (int j = 0; j < Schedule::kChunks; ++j) {
                const int item = p + j * Schedule::kProducerThreads;
                const int row = item / (BK / 32), chunk = item % (BK / 32);
                const int parent = row_policy.weight_row(row_begin, row, M);
                raw[j] = ld_nc_na<uint4>(operands.codes + static_cast<std::int64_t>(parent) * (K / 2) + k0 / 2 +
                                         chunk * 16);
            }
            cta_mbarrier_wait(empty + s, 1U ^ ((kt / S) & 1U));
            // A column's BK codes (BK / 16 16-byte copies) and BK / 32 slice scales (one copy of 16 or 8 bytes).
            for (int item = p; item < BN * (kXParts + 1); item += Schedule::kProducerThreads) {
                const int local = item / (kXParts + 1), part = item - local * (kXParts + 1);
                const int token = token_begin + local;
                const bool live = FullTokens || token < tokens;
                const auto* col = operands.x + static_cast<std::int64_t>(live ? token : 0) * operands.x_stride;
                if (part < kXParts) {
                    cp_async_zfill<16>(x_at(s) + local * ROW + part * 16, col + k0 + part * 16, live ? 16 : 0);
                } else {
                    cp_async_zfill<BK / 8>(xs_at(s) + local * (BK / 32), col + K + (k0 / 32) * 4,
                                           live ? BK / 8 : 0);
                }
            }
            for (int item = p; item < BM * (BK / 64); item += Schedule::kProducerThreads) {
                const int row = item / (BK / 64), tile = item % (BK / 64);
                const int parent = row_policy.weight_row(row_begin, row, M);
                cp_async<4>(scales_at(s) + row * (BK / 16) + tile * 4,
                            operands.scales + nvfp4_scale_byte_offset(parent, k0 / 16 + tile * 4, K));
            }
            // Each chunk's 32 codes as 32 int8 doubled values (the MMA tile's widen: quad i at byte 4 i).
#pragma unroll
            for (int j = 0; j < Schedule::kChunks; ++j) {
                const int item = p + j * Schedule::kProducerThreads;
                const int row = item / (BK / 32), chunk = item % (BK / 32);
                unsigned char* dst = weight_at(s) + row * ROW + chunk * 32;
                store_vec(dst, make_uint4(nvfp4_a8_quad(raw[j].x & 0xFFFFU), nvfp4_a8_quad(raw[j].x >> 16),
                                          nvfp4_a8_quad(raw[j].y & 0xFFFFU), nvfp4_a8_quad(raw[j].y >> 16)));
                store_vec(dst + 16, make_uint4(nvfp4_a8_quad(raw[j].z & 0xFFFFU), nvfp4_a8_quad(raw[j].z >> 16),
                                               nvfp4_a8_quad(raw[j].w & 0xFFFFU), nvfp4_a8_quad(raw[j].w >> 16)));
            }
            // Arrives once this thread's copies have landed (its weight stores precede the arrive).
            asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" : : "r"(smem_addr(full + s))
                         : "memory");
        }
        asm volatile("cp.async.wait_all;\n" ::: "memory");
        return;
    }

    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(Schedule::kConsumerRegisters) : "memory");
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wm   = warp / Schedule::kWarpsTokens;
    const int wn   = warp - wm * Schedule::kWarpsTokens;
    const int gid  = lane >> 2;
    const int lid  = lane & 3;
    float accumulators[MT][NT][4] = {};
#pragma unroll 1
    for (int kt = 0; kt < k_tiles; ++kt) {
        const int s = kt % S;
        cta_mbarrier_wait(full + s, (kt / S) & 1U);
        const unsigned char* weight = weight_at(s);
        const unsigned char* sc     = scales_at(s);
        const unsigned char* xr     = x_at(s);
        const float* xs             = xs_at(s);
#pragma unroll
        for (int k_step = 0; k_step < BK / 16; ++k_step) {
            unsigned a[MT][2];
            float sa[MT][2];
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
                const int row = wm * WM + mi * 16 + gid;
                a[mi][0]      = *reinterpret_cast<const unsigned*>(weight + row * ROW + k_step * 16 + lid * 4);
                a[mi][1]      = *reinterpret_cast<const unsigned*>(weight + (row + 8) * ROW + k_step * 16 + lid * 4);
                sa[mi][0]     = table[sc[row * (BK / 16) + k_step]];
                sa[mi][1]     = table[sc[(row + 8) * (BK / 16) + k_step]];
            }
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                const int local   = wn * WN + ni * 8;
                const unsigned b0 =
                    *reinterpret_cast<const unsigned*>(xr + (local + gid) * ROW + k_step * 16 + lid * 4);
                const float d0    = xs[(local + 2 * lid) * (BK / 32) + (k_step >> 1)];
                const float d1    = xs[(local + 2 * lid + 1) * (BK / 32) + (k_step >> 1)];
#pragma unroll
                for (int mi = 0; mi < MT; ++mi) {
                    nvfp4_a8_block(accumulators[mi][ni], a[mi][0], a[mi][1], b0, sa[mi][0], sa[mi][1], d0, d1);
                }
            }
        }
        // Release only after every lane of the warp has finished its reads of the stage.
        __syncwarp();
        if (lane == 0) cta_mbarrier_arrive(empty + s);
    }

    const auto destination = linear_output_tile<BM / (RowPolicy::kPaired ? 2 : 1)>(output, row_begin);
    const float alpha      = operands.alpha;
    if constexpr (kLoraEpilogue<Epilogue>) {
        const auto row_of = [&](int mi, int i) { return row_policy.weight_row(row_begin, wm * WM + mi * 16 + i, M); };
        const auto token_of = [&](int ni, int j) {
            const int token = token_begin + wn * WN + ni * 8 + j;
            return token < tokens ? token : -1;
        };
        if constexpr (requires { RowPolicy::kThreadPaired; }) {
            // Gate and up: the scaled values (one rounded multiply, the SwiGLU's own product), the term, the SwiGLU.
#pragma unroll
            for (int mi = 0; mi < MT; ++mi)
#pragma unroll
                for (int ni = 0; ni < NT; ++ni)
#pragma unroll
                    for (int e = 0; e < 4; ++e) accumulators[mi][ni][e] = __fmul_rn(accumulators[mi][ni][e], alpha);
            lora_add_rows_tile<MT, NT>(epilogue.lora, accumulators, row_of, token_of);
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
                const int out_row = row_policy.output_row(row_begin, wm * WM + mi * 16 + gid);
#pragma unroll
                for (int ni = 0; ni < NT; ++ni) {
                    const int token0 = token_begin + wn * WN + ni * 8 + 2 * lid;
                    const auto& v    = accumulators[mi][ni];
                    if (FullTokens || token0 < tokens) epilogue.apply_pair(destination, out_row, token0, v[0], v[2]);
                    if (FullTokens || token0 + 1 < tokens)
                        epilogue.apply_pair(destination, out_row, token0 + 1, v[1], v[3]);
                }
            }
        } else {
            static_assert(!RowPolicy::kPaired, "paired rows require the thread-paired MMA rows");
            // The epilogue on the kernel's own v * alpha (its residual add may fuse the product), then the term.
            lora_finish_store_rows_tile<MT, NT>(epilogue, destination, accumulators, row_of, token_of,
                                                [&](int, int, float v) { return v * alpha; });
        }
        return;
    }
    if constexpr (requires { RowPolicy::kThreadPaired; }) {
        // Each m16 tile pairs C-fragment rows g (gate) and g+8 (up) of one output row.
#pragma unroll
        for (int mi = 0; mi < MT; ++mi) {
            const int out_row = row_policy.output_row(row_begin, wm * WM + mi * 16 + gid);
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                const int token0 = token_begin + wn * WN + ni * 8 + 2 * lid;
                const auto& v    = accumulators[mi][ni];
                if (FullTokens || token0 < tokens) {
                    epilogue.apply_pair(destination, out_row, token0, v[0] * alpha, v[2] * alpha);
                }
                if (FullTokens || token0 + 1 < tokens) {
                    epilogue.apply_pair(destination, out_row, token0 + 1, v[1] * alpha, v[3] * alpha);
                }
            }
        }
    } else {
        static_assert(!RowPolicy::kPaired, "paired rows require the thread-paired MMA rows");
#pragma unroll
        for (int mi = 0; mi < MT; ++mi) {
            const int row0 = row_policy.weight_row(row_begin, wm * WM + mi * 16 + gid, M);
            const int row1 = row_policy.weight_row(row_begin, wm * WM + mi * 16 + gid + 8, M);
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                const int token0 = token_begin + wn * WN + ni * 8 + 2 * lid;
                const auto& v    = accumulators[mi][ni];
                if (FullTokens || token0 < tokens) {
                    destination.store(row0, token0, epilogue.apply(row0, token0, v[0] * alpha));
                    destination.store(row1, token0, epilogue.apply(row1, token0, v[2] * alpha));
                }
                if (FullTokens || token0 + 1 < tokens) {
                    destination.store(row0, token0 + 1, epilogue.apply(row0, token0 + 1, v[1] * alpha));
                    destination.store(row1, token0 + 1, epilogue.apply(row1, token0 + 1, v[3] * alpha));
                }
            }
        }
    }
}

// A family's launch_nvfp4_a8_ring (nvfp4_a8_ring.h) in its non-RDC translation unit.
template <class Output, class Epilogue, class Rows>
void launch_nvfp4_a8_ring_tiles(const Nvfp4A8Operands& p, Output output, Epilogue epilogue, Rows rows,
                                cudaStream_t stream) {
    using Schedule = Nvfp4A8RingSchedule;
    validate_nvfp4_a8_operands(p);
    if (p.rows % Schedule::kBlockRows || p.k % Schedule::kBlockK)
        throw std::invalid_argument("NVFP4 A8 ring requires complete row/K tiles");
    for_each_token_slice(p.tokens, Schedule::kBlockTokens, [&](int offset, int count) {
        // The token tiles fastest (grid x), the row tiles slowest (grid y, at most 272 here).
        const dim3 grid(div_up(count, Schedule::kBlockTokens), p.rows / Schedule::kBlockRows);
        const auto launch = [&]<bool Full>() {
            static constexpr auto kernel = nvfp4_a8_ring_kernel<Schedule, Full, Output, Epilogue, Rows>;
            const int bytes              = prepare_dynamic_shared<Schedule::kSharedBytes, kernel, true>();
            kernel<<<grid, Schedule::kThreads, bytes, stream>>>(p, output, epilogue, rows, offset, count);
            CUDA_CHECK(cudaGetLastError());
        };
        if (count % Schedule::kBlockTokens == 0)
            launch.template operator()<true>();
        else
            launch.template operator()<false>();
    });
}

} // namespace infernix::ops::detail
