// The reference route's experts (offloaded_moe::WideA16::kPrecise; docs/maintainer/model-quality.md
// §6.2), included by moe_layer.cu after fp8_expert.cuh, whose pass plumbing and h plane they share.
// Every A16 bank (NVFP4 rg16 with its multipliers, GGUF, block FP8) and every column count: each
// weight decoded exactly (ops/precise/precise_decode.cuh), x and h as stored, exact binary32 products
// summed in double-binary32 arithmetic in ascending k (ops/precise/precise_dot.cuh), and one rounding
// at each of the exact route's BF16 boundaries: y_gate and y_up, h = SiLU(y_gate) * y_up (binary64),
// and the output. A gate/up CTA owns 64 intermediates of one job (each thread its gate and up row), a
// down CTA 64 output rows; both walk the job's columns in groups of precise::kDotColumns, so an
// output's bits depend only on its expert and column.

constexpr int kPreciseThreads = 64;
static_assert(moe::kIntermediate % kPreciseThreads == 0 && moe::kHidden % kPreciseThreads == 0);

__device__ __forceinline__ std::uint16_t bf16_bits_rn(double v) { return __bfloat16_as_ushort(__double2bfloat16(v)); }

// nvfp4_expert_rg16_v1 records: gate and up rows interleaved (2 i, 2 i + 1), then the down rows; the
// FP32 multipliers of an A16 bank in ExpertScales::alpha_* apply to each sum.
struct PreciseNvfp4Bank {
    __device__ static precise::Rg16 gate_up(const std::uint8_t* record, const MoeExpertSource&) {
        return {record, moe::kGateUpBlocks};
    }
    __device__ static precise::Rg16 down(const std::uint8_t* record, const MoeExpertSource&) {
        return {record + moe::kGateUpBytes, moe::kDownBlocks};
    }
    __device__ static int gate_row(int i) { return 2 * i; }
    __device__ static int up_row(int i) { return 2 * i + 1; }
    __device__ static double multiplier(const MoeExpertSource& source, int expert, int matrix) {
        const moe::ExpertScales& s = source.scales[expert];
        return static_cast<double>(matrix == 0 ? s.alpha_gate : matrix == 1 ? s.alpha_up : s.alpha_down);
    }
};

// GGUF records: the gate rows, the up rows (format GU), then the down rows (format D).
template <QType GU, QType D>
struct PreciseGgmlBank {
    __device__ static precise::Ggml<GU> gate_up(const std::uint8_t* record, const MoeExpertSource& source) {
        return {record, source.ggml.gate_row_bytes, false};
    }
    __device__ static precise::Ggml<D> down(const std::uint8_t* record, const MoeExpertSource& source) {
        return {record + 2 * static_cast<std::size_t>(moe::kIntermediate) * source.ggml.gate_row_bytes,
                source.ggml.down_row_bytes, false};
    }
    __device__ static int gate_row(int i) { return i; }
    __device__ static int up_row(int i) { return moe::kIntermediate + i; }
    __device__ static double multiplier(const MoeExpertSource&, int, int) { return 1.0; }
};

// fp8_expert_b128_v1 records: gate then up rows (their tile scales consecutive, so one decoder over
// 2 I rows), the down rows, then the scales (fp8_scales).
struct PreciseFp8Bank {
    __device__ static precise::Fp8B128 gate_up(const std::uint8_t* record, const MoeExpertSource&) {
        return {record, fp8_scales(record), moe::kHidden};
    }
    __device__ static precise::Fp8B128 down(const std::uint8_t* record, const MoeExpertSource&) {
        return {record + 2 * static_cast<std::size_t>(moe::kIntermediate) * moe::kHidden,
                fp8_scales(record) + 2 * kFp8GateTiles, moe::kIntermediate};
    }
    __device__ static int gate_row(int i) { return i; }
    __device__ static int up_row(int i) { return moe::kIntermediate + i; }
    __device__ static double multiplier(const MoeExpertSource&, int, int) { return 1.0; }
};

template <class Bank>
__global__ void __launch_bounds__(kPreciseThreads)
    gate_up_kernel_precise(const bf16* __restrict__ x, MoeDispatch dispatch, MoeExpertSource source, int top_k,
                           const std::uint8_t* const* __restrict__ job_records, const std::int32_t* __restrict__ cpu_flags,
                           int job_base, std::uint16_t* __restrict__ h_plane, PassPhase phase) {
    __shared__ precise::DotShared shared;
    const int job = job_base + static_cast<int>(blockIdx.y);
    if (job >= *dispatch.job_count || (cpu_flags != nullptr && cpu_flags[job] != 0)) { return; }
    const int expert = dispatch.jobs[job];
    const std::uint8_t* record = pass_record(phase, source, expert, job_records, job);
    if (record == nullptr) { return; }
    const int first = dispatch.offsets[expert], count = dispatch.offsets[expert + 1] - first;
    const int i     = static_cast<int>(blockIdx.x) * kPreciseThreads + static_cast<int>(threadIdx.x);
    const auto rows = Bank::gate_up(record, source);
    const double alpha_gate = Bank::multiplier(source, expert, 0), alpha_up = Bank::multiplier(source, expert, 1);
    for (int c0 = 0; c0 < count; c0 += precise::kDotColumns) {
        const int n = min(precise::kDotColumns, count - c0);
        const auto column = [&](int c, int chunk, float (&v)[8]) {
            const int token    = dispatch.entries[first + c0 + c] / top_k;
            const uint4 packed = *reinterpret_cast<const uint4*>(x + static_cast<std::size_t>(token) * moe::kHidden + 8 * chunk);
            const auto* bits   = reinterpret_cast<const std::uint16_t*>(&packed);
#pragma unroll
            for (int e = 0; e < 8; ++e) { v[e] = precise::bf16_float(bits[e]); }
        };
        const auto weights = [&](int q, int chunk, precise::Weights8& w) {
            rows.decode(q == 0 ? Bank::gate_row(i) : Bank::up_row(i), chunk, w);
        };
        double acc[2][precise::kDotColumns];
        precise::dot<2, decltype(Bank::gate_up(record, source))>(moe::kHidden, n, column, weights, true, shared, acc);
#pragma unroll
        for (int c = 0; c < precise::kDotColumns; ++c) {
            if (c < n) {
                const double g = precise::bf16_value(bf16_bits_rn(acc[0][c] * alpha_gate));
                const double u = precise::bf16_value(bf16_bits_rn(acc[1][c] * alpha_up));
                h_plane[static_cast<std::size_t>(first + c0 + c) * moe::wide::kA16HRowElements + i] =
                    bf16_bits_rn(g / (1.0 + exp(-g)) * u);
            }
        }
    }
}

template <class Bank>
__global__ void __launch_bounds__(kPreciseThreads)
    down_kernel_precise(MoeDispatch dispatch, MoeExpertSource source, const std::uint8_t* const* __restrict__ job_records,
                        const std::int32_t* __restrict__ cpu_flags, int job_base, const std::uint16_t* __restrict__ h_plane,
                        bf16* __restrict__ outputs, PassPhase phase) {
    __shared__ precise::DotShared shared;
    const int job = job_base + static_cast<int>(blockIdx.y);
    if (job >= *dispatch.job_count || (cpu_flags != nullptr && cpu_flags[job] != 0)) { return; }
    const int expert = dispatch.jobs[job];
    const std::uint8_t* record = pass_record(phase, source, expert, job_records, job);
    if (record == nullptr) { return; }
    const int first = dispatch.offsets[expert], count = dispatch.offsets[expert + 1] - first;
    const int row   = static_cast<int>(blockIdx.x) * kPreciseThreads + static_cast<int>(threadIdx.x);
    const auto rows = Bank::down(record, source);
    const double alpha = Bank::multiplier(source, expert, 2);
    for (int c0 = 0; c0 < count; c0 += precise::kDotColumns) {
        const int n = min(precise::kDotColumns, count - c0);
        const auto column = [&](int c, int chunk, float (&v)[8]) {
            const uint4 packed = *reinterpret_cast<const uint4*>(
                h_plane + static_cast<std::size_t>(first + c0 + c) * moe::wide::kA16HRowElements + 8 * chunk);
            const auto* bits = reinterpret_cast<const std::uint16_t*>(&packed);
#pragma unroll
            for (int e = 0; e < 8; ++e) { v[e] = precise::bf16_float(bits[e]); }
        };
        const auto weights = [&](int, int chunk, precise::Weights8& w) { rows.decode(row, chunk, w); };
        double acc[1][precise::kDotColumns];
        precise::dot<1, decltype(Bank::down(record, source))>(moe::kIntermediate, n, column, weights, true, shared, acc);
#pragma unroll
        for (int c = 0; c < precise::kDotColumns; ++c) {
            if (c < n) {
                const int entry = dispatch.entries[first + c0 + c];
                outputs[static_cast<std::size_t>(entry) * moe::kHidden + row] = __double2bfloat16(acc[0][c] * alpha);
            }
        }
    }
}

template <class Bank>
void launch_pass_precise_bank(const bf16* x, const MoeDispatch& dispatch, const MoeExpertSource& source, int top_k,
                              const std::uint8_t* const* job_records, const std::int32_t* flags, int base, int jobs,
                              std::uint16_t* h_plane, bf16* outputs, cudaStream_t stream, PassPhase phase) {
    gate_up_kernel_precise<Bank><<<dim3(moe::kIntermediate / kPreciseThreads, jobs), kPreciseThreads, 0, stream>>>(
        x, dispatch, source, top_k, job_records, flags, base, h_plane, phase);
    check_launch("precise gate/up");
    down_kernel_precise<Bank><<<dim3(moe::kHidden / kPreciseThreads, jobs), kPreciseThreads, 0, stream>>>(
        dispatch, source, job_records, flags, base, h_plane, outputs, phase);
    check_launch("precise down");
}

// One pass of the reference route over jobs [base, base + jobs), every bank format.
void launch_pass_precise(const bf16* x, const MoeDispatch& dispatch, const MoeExpertSource& source, int top_k,
                         const std::uint8_t* const* job_records, const std::int32_t* flags, int base, int jobs,
                         const moe::wide::ExpertsWorkspace& layout, bf16* outputs, cudaStream_t stream, PassPhase phase) {
    require(source.activation == moe::ExpertActivation::kA16, "the reference route's experts are A16");
    auto* h_plane = reinterpret_cast<std::uint16_t*>(layout.h_rows);
    const QType gu = source.ggml.gate_up_format, d = source.ggml.down_format;
    if (gu == QType::NVFP4_MUL) {
        launch_pass_precise_bank<PreciseNvfp4Bank>(x, dispatch, source, top_k, job_records, flags, base, jobs, h_plane,
                                                   outputs, stream, phase);
        return;
    }
    if (gu == QType::FP8_E4M3FN_B128) {
        launch_pass_precise_bank<PreciseFp8Bank>(x, dispatch, source, top_k, job_records, flags, base, jobs, h_plane,
                                                 outputs, stream, phase);
        return;
    }
#define INFERNIX_PRECISE_PAIR(GU, D)                                                                                   \
    if (gu == GU && d == D) {                                                                                          \
        launch_pass_precise_bank<PreciseGgmlBank<GU, D>>(x, dispatch, source, top_k, job_records, flags, base, jobs,   \
                                                         h_plane, outputs, stream, phase);                             \
        return;                                                                                                        \
    }
    INFERNIX_PRECISE_PAIR(QType::GGML_IQ2_S, QType::GGML_IQ4_NL)
    INFERNIX_PRECISE_PAIR(QType::GGML_IQ2_S, QType::GGML_Q2_0)
    INFERNIX_PRECISE_PAIR(QType::GGML_IQ3_XXS, QType::GGML_IQ4_NL)
    INFERNIX_PRECISE_PAIR(QType::GGML_IQ3_XXS, QType::GGML_Q2_0)
    INFERNIX_PRECISE_PAIR(QType::GGML_IQ3_S, QType::GGML_IQ4_NL)
    INFERNIX_PRECISE_PAIR(QType::GGML_IQ3_S, QType::GGML_Q2_0)
    INFERNIX_PRECISE_PAIR(QType::GGML_IQ4_XS, QType::GGML_IQ4_NL)
    INFERNIX_PRECISE_PAIR(QType::GGML_IQ4_XS, QType::GGML_Q2_0)
#undef INFERNIX_PRECISE_PAIR
    require(false, "the reference route has no kernels for this bank's formats");
}
