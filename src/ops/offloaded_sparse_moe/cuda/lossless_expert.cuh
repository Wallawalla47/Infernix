// The --lossless route's experts (offloaded_moe::WideA16::kLossless; docs/maintainer/lossless-route.md),
// included by moe_layer.cu after fp8_expert.cuh. Every bank (NVFP4 records with their multipliers,
// block FP8, GGUF) on the GPU: the call's per-job column counts are read back once, then each pass of
// staged jobs is cut into 64-column tiles of one job each, and column chunks run the gate/up projection
// (x's columns through the tile's input map), h = bf16(bf16(silu(y_gate)) y_up) and the down projection
// (written to each column's entry) through ops::lossless_linear_tiles. An output depends only on its
// expert's record and its token's x. The pass reads the bank's geometry (source.shape) at run time:
// Flash-Next's for every bank kind, Qwen3.5-122B-A10B's for its NVFP4 records (wide::shape_supported).
//
// The design reference (source.lossless_design, lossless-route.md §3b): a W4A4 bank's experts quantize their
// inputs as the bank declares, each column with its expert's input scale g (ModelOpt's rule,
// lossless_quantize_nvfp4_scaled): x for the gate and up rows (once, or once per scale where an expert's gate
// and up scales differ) and h for the down rows; each tile's act scales carry the g its rows' sums are
// multiplied by, exactly.

// (Its includes are at the top of moe_layer.cu: this file lies inside its anonymous namespace.)

constexpr int kLosslessChunkTiles = moe::wide::kLosslessChunkColumns / 64;

std::int32_t lossless_max_tiles(std::int32_t max_jobs, std::int32_t entries) { return entries / 64 + max_jobs + 1; }

// The region's parts (moe::wide::lossless_experts_bytes).
struct LosslessRegion {
    std::int32_t* counts; // [max_jobs + 1]: each job's columns, then the job count
    std::int32_t* plan;   // [max_tiles][3]: (job, first column within the job, columns)
    LosslessTile* tiles_gate_up;
    LosslessTile* tiles_down;
    std::int32_t* in_columns;  // [max_tiles * 64]: the column's token (-1: padding)
    std::int32_t* out_columns; // [max_tiles * 64]: the column's entry (-1: padding)
    std::uint16_t* gate_up;    // BF16 [2I, chunk]
    std::uint16_t* h;          // BF16 [I, chunk]
    // The design reference's A4 inputs: each column's input scales (gate, up, down; [3][max_tiles * 64]),
    // the chunk's quantized gate/up input (and, gate and up scales differing, the up input and its plane).
    float* column_scales;
    std::uint16_t* x_gate;     // BF16 [H, chunk]
    std::uint16_t* x_up;       // BF16 [H, chunk]
    std::uint16_t* gate_up_up; // BF16 [2I, chunk]
    std::uint8_t* linear;
    std::size_t linear_bytes;
    std::size_t bytes; // the whole region
};

LosslessBankMatrix lossless_gate_up_matrix(QType q, const moe::ExpertShape& shape) {
    LosslessBankMatrix m;
    m.qtype = q;
    m.rows  = 2 * shape.intermediate;
    m.k     = shape.hidden;
    return m;
}

std::size_t lossless_linear_bytes(const moe::ExpertShape& shape) {
    LosslessBankMatrix down;
    down.qtype = QType::NVFP4_MUL;
    down.rows  = shape.hidden;
    down.k     = shape.intermediate;
    return std::max(lossless_linear_tiles_workspace_bytes(lossless_gate_up_matrix(QType::NVFP4_MUL, shape),
                                                          moe::wide::kLosslessChunkColumns),
                    lossless_linear_tiles_workspace_bytes(down, moe::wide::kLosslessChunkColumns));
}

// The region's parts from base (null: only its size, LosslessRegion::bytes).
LosslessRegion carve_lossless(std::uint8_t* base, const moe::ExpertShape& shape, std::int32_t max_jobs,
                              std::int32_t entries) {
    const auto tiles = static_cast<std::size_t>(lossless_max_tiles(max_jobs, entries));
    const auto chunk = static_cast<std::size_t>(moe::wide::kLosslessChunkColumns);
    const auto I = static_cast<std::size_t>(shape.intermediate), H = static_cast<std::size_t>(shape.hidden);
    std::size_t at = 0;
    const auto take = [&](std::size_t bytes) {
        std::uint8_t* p = base == nullptr ? nullptr : base + at;
        at += (bytes + 255) / 256 * 256;
        return p;
    };
    LosslessRegion r{};
    r.counts        = reinterpret_cast<std::int32_t*>(take((static_cast<std::size_t>(max_jobs) + 1) * 4));
    r.plan          = reinterpret_cast<std::int32_t*>(take(tiles * 3 * 4));
    r.tiles_gate_up = reinterpret_cast<LosslessTile*>(take(tiles * sizeof(LosslessTile)));
    r.tiles_down    = reinterpret_cast<LosslessTile*>(take(tiles * sizeof(LosslessTile)));
    r.in_columns    = reinterpret_cast<std::int32_t*>(take(tiles * 64 * 4));
    r.out_columns   = reinterpret_cast<std::int32_t*>(take(tiles * 64 * 4));
    r.gate_up       = reinterpret_cast<std::uint16_t*>(take(2 * I * chunk * 2));
    r.h             = reinterpret_cast<std::uint16_t*>(take(I * chunk * 2));
    r.column_scales = reinterpret_cast<float*>(take(3 * tiles * 64 * sizeof(float)));
    r.x_gate        = reinterpret_cast<std::uint16_t*>(take(H * chunk * 2));
    r.x_up          = reinterpret_cast<std::uint16_t*>(take(H * chunk * 2));
    r.gate_up_up    = reinterpret_cast<std::uint16_t*>(take(2 * I * chunk * 2));
    r.linear_bytes  = lossless_linear_bytes(shape);
    r.linear        = take(r.linear_bytes);
    r.bytes         = at;
    return r;
}

__global__ void lossless_counts_kernel(MoeDispatch dispatch, int max_jobs, std::int32_t* counts) {
    const int job = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int jobs = *dispatch.job_count;
    if (job < max_jobs) {
        const int expert = job < jobs ? dispatch.jobs[job] : -1;
        counts[job]      = expert < 0 ? 0 : dispatch.offsets[expert + 1] - dispatch.offsets[expert];
    }
    if (job == 0) { counts[max_jobs] = jobs; }
}

// One CTA (64 threads) per planned tile: its records (the multipliers of an NVFP4 bank: gate rows even,
// up rows odd) and its columns' token and entry; for the design reference's A4 inputs (design), the tiles'
// act scales and each column's input scales (gate, up, down planes `stride` apart).
__global__ void lossless_tiles_kernel(MoeDispatch dispatch, MoeExpertSource source, int top_k,
                                      const std::uint8_t* const* __restrict__ job_records, const std::int32_t* __restrict__ plan,
                                      LosslessTile* tiles_gate_up, LosslessTile* tiles_down, std::int32_t* in_columns,
                                      std::int32_t* out_columns, bool design, float* __restrict__ column_scales,
                                      std::size_t stride) {
    const int t = static_cast<int>(blockIdx.x), c = static_cast<int>(threadIdx.x);
    const int job = plan[3 * t], c0 = plan[3 * t + 1], n = plan[3 * t + 2];
    const int expert = dispatch.jobs[job];
    moe::ExpertScales s{};
    if (source.scales != nullptr) { s = source.scales[expert]; }
    if (c == 0) {
        const std::uint8_t* record = pass_record(PassPhase::All, source, expert, job_records, job);
        const std::uint32_t act_gate = design ? __float_as_uint(s.input_gate) : 0U;
        const std::uint32_t act_up   = design ? __float_as_uint(s.input_up) : 0U;
        const std::uint32_t act_down = design ? __float_as_uint(s.input_down) : 0U;
        tiles_gate_up[t] = LosslessTile{record, {__float_as_uint(s.alpha_gate), __float_as_uint(s.alpha_up)}, {act_gate, act_up}};
        tiles_down[t]    = LosslessTile{record, {__float_as_uint(s.alpha_down), __float_as_uint(s.alpha_down)}, {act_down, act_down}};
    }
    const int entry = c < n ? dispatch.entries[dispatch.offsets[expert] + c0 + c] : -1;
    in_columns[64 * t + c]  = entry < 0 ? -1 : moe::wide::entry_column(entry, top_k, dispatch.shared_first);
    out_columns[64 * t + c] = entry;
    if (design) {
        // Every column of the tile (padding too) takes its expert's scales.
        column_scales[64 * t + c]              = s.input_gate;
        column_scales[stride + 64 * t + c]     = s.input_up;
        column_scales[2 * stride + 64 * t + c] = s.input_down;
    }
}

// The up rows (odd: interleaved) of `up` into `gate_up` (BF16 [2I, columns] each).
__global__ void lossless_up_rows_kernel(const std::uint16_t* __restrict__ up, int I, int columns,
                                        std::uint16_t* __restrict__ gate_up) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<std::size_t>(I) * columns) { return; }
    const std::size_t c = index / I, i = index % I;
    gate_up[c * 2 * I + 2 * i + 1] = up[c * 2 * I + 2 * i + 1];
}

// h[i, c] = bf16(bf16(silu(g)) u) of the gate/up plane's column c (gate row 2 i and up row 2 i + 1 when
// interleaved, else rows i and I + i).
__global__ void lossless_swiglu_kernel(const std::uint16_t* __restrict__ gate_up, int I, int columns,
                                       bool interleaved, std::uint16_t* __restrict__ h) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<std::size_t>(I) * columns) { return; }
    const int c = static_cast<int>(index / I), i = static_cast<int>(index % I);
    const std::uint16_t* col = gate_up + static_cast<std::size_t>(c) * 2 * I;
    const float g = __uint_as_float(static_cast<std::uint32_t>(col[interleaved ? 2 * i : i]) << 16);
    const float u = __uint_as_float(static_cast<std::uint32_t>(col[interleaved ? 2 * i + 1 : I + i]) << 16);
    const float a = __uint_as_float(static_cast<std::uint32_t>(lossless::cr_silu_bf16(g)) << 16);
    h[index]      = __bfloat16_as_ushort(__float2bfloat16_rn(__fmul_rn(a, u))); // the product is exact in binary32
}

void lossless_experts(const Tensor& x, const MoeDispatch& dispatch, const MoeExpertSource& source, int top_k,
                      int max_jobs, const moe::wide::ExpertsWorkspace& layout, FetchCall* fetch, Tensor& outputs,
                      cudaStream_t stream) {
    const int I = source.shape.intermediate, H = source.shape.hidden;
    const int entries = static_cast<int>(outputs.ne[1]);
    const LosslessRegion r = carve_lossless(layout.lossless, source.shape, max_jobs, entries);
    // The banks' matrices within a record (block FP8 and GGUF banks at Flash-Next's geometry only).
    const QType gu = source.ggml.gate_up_format;
    LosslessBankMatrix gate_up = lossless_gate_up_matrix(gu, source.shape), down{};
    down.rows = H;
    down.k    = I;
    bool interleaved = false;
    if (gu == QType::NVFP4_MUL) {
        down.qtype        = QType::NVFP4_MUL;
        down.codes_offset = source.shape.gate_up_bytes();
        interleaved       = true;
    } else if (gu == QType::FP8_E4M3FN_B128) {
        const std::uint64_t scales = 3ULL * I * H;
        gate_up.scales_offset = scales;
        down.qtype            = QType::FP8_E4M3FN_B128;
        down.codes_offset     = 2ULL * I * H;
        down.scales_offset    = scales + 2ULL * kFp8GateTiles * sizeof(std::uint16_t);
    } else {
        gate_up.row_bytes = static_cast<std::int32_t>(source.ggml.gate_row_bytes);
        down.qtype        = source.ggml.down_format;
        down.codes_offset = 2ULL * I * source.ggml.gate_row_bytes;
        down.row_bytes    = static_cast<std::int32_t>(source.ggml.down_row_bytes);
    }
    // The design reference's A4 inputs (an NVFP4 bank's interleaved gate and up rows).
    const bool design         = source.lossless_design != offloaded_moe::LosslessExpertDesign::kNone;
    const bool separate_up    = source.lossless_design == offloaded_moe::LosslessExpertDesign::kA4SeparateUp;
    const std::size_t stride  = static_cast<std::size_t>(lossless_max_tiles(max_jobs, entries)) * 64;
    if (design && gu != QType::NVFP4_MUL) {
        throw std::invalid_argument("lossless experts: the design reference's A4 inputs need an NVFP4 bank");
    }
    // Each job's columns, on the host (the route is eager).
    lossless_counts_kernel<<<(max_jobs + 255) / 256, 256, 0, stream>>>(dispatch, max_jobs, r.counts);
    check_launch("lossless counts");
    std::vector<std::int32_t> counts(static_cast<std::size_t>(max_jobs) + 1);
    CUDA_CHECK(cudaMemcpyAsync(counts.data(), r.counts, counts.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const int job_count = counts[static_cast<std::size_t>(max_jobs)];
    const int pass_jobs = source.staging_slots > 0 ? std::min(source.staging_slots, kMaxPassJobs) : max_jobs;
    std::vector<std::int32_t> plan;
    for (int base = 0; base < job_count; base += pass_jobs) {
        const int jobs = std::min(pass_jobs, job_count - base);
        CUDA_CHECK(pdl::launch_consumer({dim3(source.staging_slots > 0 ? kStageCtas : 1), dim3(kThreads), 0, stream},
                                        stage_kernel, dispatch, source, base, jobs, static_cast<const std::int32_t*>(nullptr),
                                        layout.job_records, fetch));
        plan.clear();
        for (int j = base; j < base + jobs; ++j) {
            const int n = counts[static_cast<std::size_t>(j)];
            for (int c0 = 0; c0 < n; c0 += 64) {
                plan.push_back(j);
                plan.push_back(c0);
                plan.push_back(std::min(64, n - c0));
            }
        }
        const int tiles = static_cast<int>(plan.size() / 3);
        if (tiles == 0) { continue; }
        CUDA_CHECK(cudaMemcpyAsync(r.plan, plan.data(), plan.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice, stream));
        lossless_tiles_kernel<<<tiles, 64, 0, stream>>>(dispatch, source, top_k, layout.job_records, r.plan,
                                                        r.tiles_gate_up, r.tiles_down, r.in_columns, r.out_columns,
                                                        design, r.column_scales, stride);
        check_launch("lossless tiles");
        for (int t0 = 0; t0 < tiles; t0 += kLosslessChunkTiles) {
            const int columns = 64 * std::min(kLosslessChunkTiles, tiles - t0);
            Tensor gate_up_plane(r.gate_up, DType::BF16, {2 * I, columns});
            Tensor h_plane(r.h, DType::BF16, {I, columns});
            WorkspaceArena arena(DeviceSpan{r.linear, r.linear_bytes});
            const std::size_t elements = static_cast<std::size_t>(I) * columns;
            const unsigned blocks      = static_cast<unsigned>((elements + 255) / 256);
            if (!design) {
                lossless_linear_tiles(x, r.in_columns + 64 * t0, columns, gate_up, r.tiles_gate_up + t0, gate_up_plane,
                                      nullptr, arena, stream);
            } else {
                // Each column's x quantized with its expert's gate scale (and, scales differing, its up scale for
                // the up rows); the tiles' act scales multiply the sums by them.
                Tensor x_gate(r.x_gate, DType::BF16, {H, columns});
                lossless_quantize_nvfp4_scaled(x, r.in_columns + 64 * t0, r.column_scales + 64 * t0, x_gate, stream);
                lossless_linear_tiles(x_gate, nullptr, columns, gate_up, r.tiles_gate_up + t0, gate_up_plane, nullptr,
                                      arena, stream);
                if (separate_up) {
                    Tensor x_up(r.x_up, DType::BF16, {H, columns});
                    Tensor up_plane(r.gate_up_up, DType::BF16, {2 * I, columns});
                    lossless_quantize_nvfp4_scaled(x, r.in_columns + 64 * t0, r.column_scales + stride + 64 * t0, x_up,
                                                   stream);
                    lossless_linear_tiles(x_up, nullptr, columns, gate_up, r.tiles_gate_up + t0, up_plane, nullptr,
                                          arena, stream);
                    lossless_up_rows_kernel<<<blocks, 256, 0, stream>>>(r.gate_up_up, I, columns, r.gate_up);
                    check_launch("lossless up rows");
                }
            }
            lossless_swiglu_kernel<<<blocks, 256, 0, stream>>>(r.gate_up, I, columns, interleaved, r.h);
            check_launch("lossless swiglu");
            if (!design) {
                lossless_linear_tiles(h_plane, nullptr, columns, down, r.tiles_down + t0, outputs,
                                      r.out_columns + 64 * t0, arena, stream);
            } else {
                // h quantized with each column's down scale (into the gate input's buffer, free again).
                Tensor h_quantized(r.x_gate, DType::BF16, {I, columns});
                lossless_quantize_nvfp4_scaled(h_plane, nullptr, r.column_scales + 2 * stride + 64 * t0, h_quantized,
                                               stream);
                lossless_linear_tiles(h_quantized, nullptr, columns, down, r.tiles_down + t0, outputs,
                                      r.out_columns + 64 * t0, arena, stream);
            }
        }
    }
}
