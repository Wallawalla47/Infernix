# Qwen3.8-Flash-Next GSQ-RCO IQ3_S: implementation plan

**Status (2026-10-11): partly implemented** in worktree `iq3s-gsq-rco` (not committed). The
artifact with vision is `D:\Infernix-staging\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix-00001-of-00002.infernix`
(57.4 GB, verified). Done:

- **M1.** Formats, codecs and converter; byte verification of the whole artifact and volume.
- **M2.** The model runs end to end; KL and perplexity measured (below).
- **M3.** Exact dp4a narrow kernels and exact int8 tensor-core tiles (64×64) for experts and dense
  projections (§6 integers, `src/ops/ggml/`); the CPU expert engine
  (`offloaded_sparse_moe/cpu/ggml_expert.cpp`: scalar, AVX2, AVX-VNNI and AVX-512 VNNI, int16 limbs
  X = hi·2048 + lo) serves decode misses and the prefill CPU split with 16 workers. Every route is
  bit-equal to the exact oracle at every width; teacher-forced logits are bit-identical to the first
  kernels on the 2,557 positions (`--chunk 1024`).
- **M6.** `--ngram-residency`.
- **M7.** MTP: greedy output is identical with and without it, acceptance measured.
- **Prefill stream.** The prefill expert stream copies GGUF records into frame-pitch ring slots,
  record by record at each layer's host stride.
- **M4, SSD tier.** `ExpertStore` keeps a record stride per layer and no scale tail for ggml banks;
  RAM slots take the largest record; the device fetch channel serves ggml banks. Measured with
  `--expert-ram-mib 24576` (after the size classes): tg512 107.4, pp4096 821 tok/s (98.5 and 807
  before); greedy output equal to full RAM at 24 and 8 GiB.
- **M4, size-class frames.** Frames of the largest record wasted 23 % (largest 2.54 MiB in one
  layer, 2.22 MiB in nine, mean 1.95 MiB). Replay (§7.5 variants; recipe B's 6,061-token decode
  trace `trace_plain_c1.bin`, the engine's LFRU and budgets, this artifact's 7 record strides,
  25.3 GiB): (a) one max-size frame, 10,206 frames, 10.44 % decode misses; (b) static classes with
  equal frames per layer, 13,248 frames, 6.37 %; byte-exact global LFRU (the bound of (c)), 6.23 %.
  (b) is built: one frame pool per record stride (`ExpertResidency` pools, each with its own LFRU
  controller, VMM arena and halving period scaled by its layer share); leases carry their pool and
  lend bytes from the pool with the most room; the prefill stream's lease is split into ring,
  wide-call and walk leases so the walk never needs one pool's whole room. The cache holds 13,008
  frames (25.3 GiB). Measured: tg512 103.3 → 109.6, MTP 166.9 → 187.4 tok/s, pp4096 ~2,215
  (2,261 in the table below, measured before the size classes: 2 % lower, not attributed; a first
  build lost 5 % to a general rounding routine in the GGUF epilogues, restored), pp131072 2,836 (2,399 before the lease split), vision encode
  1.9-2.0 s (1.7-1.8 s before; not attributed). Greedy output is unchanged in RAM, 24 GiB and 8 GiB
  modes.
- **M8, vision.** From the BF16 `mmproj` (`--mmproj`), the 441 tensors bit-equal to recipe B's
  tower; a 1024 x 1024 image encodes in 1.9-2.0 s (recipe B 1.4 s; the gap is not attributed).

Open:

- **M5.** Dense tuning beyond the current kernels; the tiles at 128K (about 50 effective TOPS,
  tensor pipe 33 %, 214 registers, one CTA per SM).
- **M8.** Device-side PLE (gated on measured after-draft GPU idle).
- **Qualification gaps.** C > 2 and prefix-cache greedy invariance; a cold 16K prompt with and
  without RAM residency; Strata on this machine.

Measured before the size classes (their results are under M4 above), RTX 5090 on PCIe 5.0 x8,
i9-13900K, INT8 KV, `infernix_bench` against recipe B in the same session:

| | IQ3_S | Recipe B |
|---|---:|---:|
| tg512 plain | 103.1 tok/s | 120.7 tok/s |
| tg512 MTP | 166.9 tok/s (78.4 % accepted) | 176.4 tok/s (75.5 %) |
| pp4096, `--prefill-chunk 4096` | 2,261 tok/s | 1,679 tok/s |
| pp131072, `--prefill-chunk 4096` | 2,837 tok/s | 9,144 tok/s |
| Perplexity over the §16.5 texts (`--chunk 1024`) | 4.794 | 4.653 |

Decode attribution: about 5 % of routed experts miss the 10,183-frame cache; a PCIe-staged miss
costs about 72 µs (measured from the decode-time slope against the frame count). The CPU engine
takes half the misses: decode 100.5 tok/s without it, 98.4 with six workers (the NVFP4 default),
103.7 with 16 and 103.6 with 20; 16 workers with divisor 3 gave 101.1.

Prefill attribution at 128K: compute-bound in the integer tiles, about 1.45 s per 4,096-token chunk.

Kernel notes (measured end to end unless stated):

- Encoding x in every rows CTA instead of in one encode launch measured 4 % slower, and was removed.
- Launching the dense rows kernel as a PDL consumer of its encoder gained 0.4 %.
- Tiles: 64×64 gained 16 % over the first tile shape at pp131072 (2,361 → 2,834); 32×128 was L2-bound; 64×32 with
  two CTAs per SM was slower; folding the sub-scale into extra MMAs (6 per weight) and batching
  the MMAs gave no gain.
- The CPU engine's int16 limbs were 1.5× faster than the first int32 version (5,896 experts/s at six
  workers, decode 90.65 tok/s).

**GPU kernel review (2026-10-12; user request, before the Nsight attribution).** The arithmetic
must stay §6's exact integers (every route bit-equal), which rules out floating-point tensor-core
paths. Findings from the source and the CUDA 13.4 SASS of `moe_layer.obj`:

- *Emulated byte SIMD in the decoders* (`ggml_dp4a.cuh`). `apply_signs4` (`__vcmpne4`, `__vsub4`)
  compiles to about 10 instructions per code word (IMAD by 0x01010101, LOP3, IADD 0x7f7f7f7f, PRMT
  sign replication; replaced, below), eight words per IQ3_S slice: the IQ3_S gate/up loop spent ~150
  instructions decoding a slice against 24 IDP per column (LiquidGEMM, arXiv 2509.01229, reports the
  same lowering of `vadd4`). IQ3_S/IQ3_XXS/IQ2_S grid bytes are never zero, so a byte negation
  cannot borrow: signs as m = (s4 · 0x00204081) & 0x01010101 and (v ^ m · 0xFF) + m are five
  native instructions with the same codes. Q2_0 and Q6_K subtract a byte offset with `__vsub4`;
  ((v | 0x80808080) − offset) ^ 0x80808080 gives the same bytes in three (no byte borrows from its
  neighbour). Applies to the narrow (decode) kernels and the dense rows kernel.
- *An int128 flush per slice in the narrow kernels* (`add_slice`): a 64-bit multiply and a
  variable-shift int128 add per 32 elements and column, about as many instructions as the slice's 24
  IDP. The tiles already flush per block (8 slices of a 256-element block: |P| < 2^45,
  |mant · P| < 2^56 in int64). In the narrow kernels each lane takes every 16th slice (loads
  coalesced across a row), so its slices lie in different blocks; flushing per block needs each lane
  to take a run of slices instead, trading the int128 adds for scattered loads: to be measured.
- *Small items:* `to_f32_rn` and `to_f32_scaled_rn` found the leading bit with a shift loop (now
  `exact::top_bit`: `__clzll`, `std::countl_zero`; once per output); the narrow kernels read
  110-byte IQ3_S blocks as 16-bit loads (`ld_u32_a2`) with no prefetch beyond the first slice. The
  gate/up kernels' division by `top_k` (~20 instructions) is once per column and pass: negligible.
- *Prefill tiles* (the M5 gap above: 50 effective TOPS, 33 % tensor pipe, 214 registers, one CTA per
  SM). The registers are mostly the int128 accumulators (16 per thread) and block sums. Candidates:
  (1) int64 accumulators relative to each row's smallest d exponent, valid when a row's d-exponent
  spread keeps |R| < 2^63 (checkable per matrix at load; int128 otherwise): fewer registers, a
  second CTA per SM; (2) the vector decoders above in place of the scalar `chunk<F>` (per-element
  sign branches, global table loads); (3) `ldmatrix` for the limb fragments instead of scalar LDS;
  (4) a producer/consumer split (decode warps feeding MMA warps, as Marlin, arXiv 2408.11743, and
  FLUTE, arXiv 2407.10960, pipeline dequantisation against the MMAs), with deeper `cp.async` or
  bulk-copy stages; (5) for expert tiles, a persistent grouped schedule over (job, tile) pairs in
  the manner of Stream-K (arXiv 2301.03598), so the ~80-column jobs of a 4K chunk do not leave
  64 + 16 partial tiles.
- *Sliced-K narrow kernels* (upstream's `*_a16_sliced_k_mma`): warps split K, multi-stage
  `cp.async`, MMAs at any width with zero-padded columns, cross-warp reduction in shared memory. In
  the exact form: int8 `mma.sync` m16n8k32 per limb plane and an order-free int32/int64 reduction.
  Most useful at 2-8 columns (MTP verify rounds, several lanes); at one column 7/8 of each MMA is
  padding, so it pays only if decode (above) no longer dominates.
- *Measured 2026-10-12* (RTX 5090; `infernix_ggml_linear_test --bench` against the binaries built
  before the change in the same session; IQ3_S artifact, `infernix_bench` int8 KV, 3 runs; all
  routes bit-equal to the oracle in `infernix_ggml_linear_test` and `infernix_offloaded_moe_ggml_test`):
  - Kept: native sign and offset arithmetic in `ggml_dp4a.cuh` (`apply_signs4`, `sub_bytes4`),
    `exact::top_bit`, the tiles' `tile_chunk` word decoders (IQ3_S, IQ4_NL, IQ4_XS, Q4_K, Q5_K, Q6_K;
    `chunk<F>` packed for the rest) and `ldmatrix.x4` limb fragments. Tiles at 4096 columns against
    the original tile source rebuilt in the same session: Q6_K -1.1 % (attn_qkv 4215 -> 4167 us),
    -1.0 %, -1.4 %, -0.1 % (shexp gate+up); Q5_K -3.8 %; Q4_K -2.4 %; IQ4_XS -8.5 %; shexp down
    -4.2 %. IQ3_S pp4096 2215.3 -> 2214.9 tok/s (dense tiles are a small share of prefill). Decode
    tg512 109.9 -> 110.4, MTP 187.7 -> 188.8 tok/s (within run spread).
  - Rejected: `ggml_narrow.cuh`, the sliced-K int8 narrow route (16 rows per CTA, 3-8 warps splitting
    K by units, cp.async one unit ahead, codes through shared memory per slice, a flush per block).
    Q6_K attn_qkv at one column 22.2 -> 34.9 us, lm_head 307 -> 452 us; tg512 109.9 -> 84.7 and MTP
    187.7 -> 180.2 tok/s. Nsight Compute: 317 warp instructions per 16-row slice, 10 warps per SM
    (shared memory: 43 KiB per 5-warp CTA), stalls on long scoreboard, MIO throttle and the per-slice
    `__syncwarp`s. `rows_kernel` restored. A second attempt would decode straight into fragment
    registers (k may be permuted per lane when A and B agree; the half-sub formats then exchange one
    word per row pair by `__shfl_xor`) with no code stage in shared memory.
  - Rejected: the double-buffered code stage in the tiles (stage s + 1 decoding beside stage s's
    MMAs): 3-5 % slower for Q6_K, Q5_K and Q4_K than decode-then-MMA (a single-buffered variant of
    the restructured loop was slower still), better only for IQ4_XS.
  - Whole-inference attribution (Nsight Systems, IQ3_S tg256): `cpu_wait_kernel` 19 % and
    `stage_kernel` 12 % of GPU time (the expert misses served by the CPU and host staging), dense
    `rows_kernel`s 24 %, expert dp4a kernels ~16 %, the encoder 6 %. A second profile the same day
    (expert cache warm; exclusive time): `rows_kernel` 31 %, `cpu_wait_kernel` 8 %, expert dp4a
    13 %, the HC mixer kernels 12 %, encoders 5 %, staging 4 %; the first profile's cold cache
    explains its larger miss share.
- *Tiles, second round (2026-10-12, measured against the round above; RTX 5090; exact on
  `infernix_ggml_linear_test` and `infernix_offloaded_moe_ggml_test` throughout).* Nsight Compute on
  the dense Q6_K tile at 512 columns: 15.5 % of the int8 tensor peak (2,048 ops/clk/SM), 254
  registers with spills, IPC 1.77, about 20 non-MMA instructions per IMMA (IMAD 6.7, MOV 2.5, IADD
  2.2), the limb-copy loop's index math 10.5 % of all instructions; the IQ2_S gate/up tile 255
  registers and 63 MB of local spills, 11 % of tensor peak; expert tiles 44 % padded columns.
  - Kept: per-thread limb copies resolved once per tile; codes and sub-scales in A-fragment order
    (one 128-bit load per m16 row group, no MOVs); each n-tile's MMAs scaled at once (fewer live
    products); word decoders for IQ2_S, IQ3_XXS and Q2_0 (every format now); int64 accumulators
    relative to each row's smallest scale shift when the tile's rows allow (`row_bases`: a CTA's
    rows within 62 - term bound - ceil(log2 blocks) binades; the GGUF model: 100 % of 64-row tiles
    for every expert format, Q6_K and IQ4_XS, 96.5 % Q4_K, 79.6 % Q5_K), int128 otherwise; narrow
    warp layouts (compile-time) for tiles of at most 32 or 16 valid columns. Dense Q6_K tiles at
    4096 columns 4,167 -> 3,882 us (-6.8 %), Q5_K -5 %, Q4_K -4.7 %, IQ4_XS -4.2 %, shexp down
    -10 %; expert calls (512 experts, top-10, 4,096 tokens, all resident) IQ2_S 17.97 -> 13.66 ms,
    IQ3_XXS 14.72 -> 13.27, IQ3_S 14.46 -> 13.12, IQ3_S/Q2_0 14.03 -> 12.81. IQ3_S pp131072
    2,839 -> 3,074 tok/s (before the layouts).
  - Rejected: runtime warp layouts (dense tiles 3-9 % slower than compile-time); 32-row tiles of
    four warps, two CTAs per SM (dense 6-27 % and experts 75 % slower).
- *Against Strata (2026-10-12, this machine, Strata v0.1.43 built from source for sm_120 with
  CUDA 13.4, IQ3_S GGUF and its pack, MTP spec 4, INT8 KV, 131,072-token context; both through
  their OpenAI servers with one client, `serve_bench.py`: unique-prefix code prompts, greedy, 256
  tokens, median of 3).* Strata 4K / 32K / 108K: prefill 3,184 / 5,198 / 5,185 tok/s, decode
  197 / 188.5 / 167. Infernix (warm expert state) 4K / 32K / 128K: prefill 2,812 / 2,945 / 3,065,
  decode 142 / 138 / 122. Strata's default `STRATA_ROUTE_TAIL_SKIP=7` skips a missed expert that
  every token of a verify window routes at rank 7 or lower (its log: about 12,000 skipped; "answers
  differ"): an inexact route Infernix does not take.
- *Strata's prefill (code read 2026-10-12, v0.1.43).* 8,192-token chunks, each through all 48
  layers; every non-resident expert streamed once per chunk through a ~531 MB ring on a copy stream
  that runs ahead across layers. Dense GGUF projections: dequantized to FP16 per chunk into a 64 MiB
  scratch, then `cublasGemmEx` FP16 x FP16 -> FP32. HC/router/indexer: cuBLAS BF16. Experts:
  llama.cpp MMQ (int8 `mma` on q8_1 activations, one 8-bit limb), host counting sort after a sync,
  SwiGLU and re-quantization between gate/up and down; its fused IQ `mma` kernel (256 x 64 or
  128 x 128 work tiles, shared-memory codebooks, 4-stage `cp.async`) is opt-in (+5-11 % in its own
  5090 bench). GDN token-sequential FP32; QSA FP16 hi/lo `m16n8k16`. Inexact throughout (FP16
  activations, q8_1 experts; no route tail skip in prefill).
- *Prefill budget against Strata (estimates).* Dense ~7.3 GFLOP/token, experts ~4.7. At peak,
  Strata needs ~35 us/token dense (FP16, 209 TFLOPS) + ~6 us experts (int8, one limb); Infernix
  ~26 us dense + ~17 us experts (int8, three limbs, 838 TOPS): equal peak budgets. Strata reaches
  ~30 % of its peak, Infernix ~13 % (IQ3_S pp65536 before the second tile round: GPU 97 % busy;
  expert tiles 46 %, dense tiles 36.5 %, HC 5 %, QSA 3 %, GDN 1.5 %; dense Q6_K tile 26.5 % of the
  tensor pipe, 243 registers, one 8-warp CTA per SM, top stalls fixed-latency waits and math
  throttle). Beating Strata by 10-20 % needs ~2-2.5x tile efficiency.
- *Prefill plan (gated; each step exact on the oracle tests, kept only when faster at its scope):*
  - P0 ceilings: cuBLAS int8 GEMM (three limb GEMMs) at the dense shapes, the attainable IMMA rate
    without decode or epilogue; Strata `STRATA_PREFILL_TIMING=1` on this machine for its phase
    split; a fresh Infernix pp32768 profile after the second tile round.
  - P1 dense decode-once: per call (encode batch raised to the whole chunk), decode the weights
    once into workspace planes (int8 codes [rows][K], sub-scales per 16/32 elements, mant/shift
    per block) and run a lean int8 GEMM without decode in its loop: 4-stage `cp.async` of A codes
    and B limbs, `ldmatrix` for both, the block epilogue and int64/int128 flush of today's tiles,
    target <= 128 registers and two CTAs per SM. Decode is now repeated per 64-column tile (64x per
    4,096 columns). Goal: dense tiles at >= 50 % of tensor peak, dense time -50 %.
  - P2 experts on the same core: in layer-walk spans each expert sees ~1,280 columns but is decoded
    per 64-column tile; decode experts in L2-sized groups (int8 planes, ~5 MB per expert) right
    before their gate/up and down GEMMs. Goal: expert time -40 %.
  - P3 the rest (~17 %): HC prefill projections, `split_vector` folded into epilogues, QSA prompt,
    the narrow small-job experts.
  - Decode (separate track): `mma_rows_kernel`, tensor-core dense rows for 3-8 columns (MTP
    verification rounds, where dp4a is compute-bound: dense rows ~40 % of a 14.8 ms server round).
- *Third round (2026-10-12; RTX 5090, PCIe 5.0 x8; IQ3_S; exact on `infernix_ggml_linear_test`,
  `infernix_offloaded_moe_ggml_test` and `infernix_qwen4_exp_expert_cache_test` throughout).*
  - Kept: `mma_rows_kernel` (3-8 columns; 16-row CTAs of four warps taking whole blocks; each lane
    decodes its own 8-element chunk of rows g and g + 8, the column's limb words 2t, 2t + 1 its B
    fragment; half-sub formats one masked MMA per half). At 5 columns against `rows_kernel`:
    Q6_K attn_qkv 36.8 -> 24.6 us, attn_gate 32.7 -> 20.7, ssm_out 28.7 -> 22.5, shexp gate+up
    17.1 -> 12.4; Q5_K 46.9 -> 22.9; Q4_K 33.1 -> 19.8; IQ4_XS 30.9 -> 20.5; IQ4_NL shexp down 16.9
    -> 13.5. At 2 columns mixed (kept on `rows_kernel`), at 3-4 faster. Bench MTP tg512 at 135K
    context 186.2 -> 197.1 tok/s (acceptance and rounds identical); through the server no
    measurable change (its rounds verify ~2.4 columns: the policy and the 0.5 draft-head
    confidence cut give ~1.4 drafts per round).
  - Kept: decoded planes for dense prefill (`plane_kernel` decodes each segment once per call into
    64 MiB of workspace in the tiles' stage images; `compute_plane` copies them by cp.async with no
    decode in the loop; Q6_K and IQ4_NL tiles at two CTAs per SM). At 4096 columns 15-34 % faster
    (Q6_K attn_qkv 3,912 -> 2,839 us; Q5_K 3,935 -> 2,820; IQ4_NL shexp down 424 -> 282); the plane
    tile's tensor pipe 57 % busy (Q6_K's k16 MMAs), top stall math throttle. pp32768 (chunk 4096)
    3,204 -> 3,484 tok/s; pp4096 unchanged (2,217).
  - Kept: a loan's give-back reloads the experts the loan evicted, best first (Strata refills its
    lent slots the same way). With 8,192-token chunks the server's decode-only hit rate after 32K
    and 128K prompts went 86-88 % / 81-85 % -> 94-95 % / 90-97 %. Narrowed on 2026-10-17: on NVIDIA
    NVFP4 Dense8 (one frame pool) the reload took the frames decode's misses land in, and MTP 4 at
    4K + 512 fell from 150.2 (main) to 136.8-139.2 tok/s (plain rose 94.4 -> 98.1). The give-back
    now reloads only when landing can reserve no frame (a GGUF import's five pools: the scarcest has
    none to spare) or behind the SSD tier (only victims with a RAM copy; landing still runs). Dense8
    then measured MTP 154.8 against main's 150.2 and plain equal to main (94.7-94.9, the same hits,
    promotions and landings); IQ3_S MTP 188-189 and plain 129 as with the reload.
  - Kept: `ld_words_a2` (N + 1 aligned 4-byte loads and funnel shifts for 2-byte-aligned GGUF
    words: Q6_K, IQ4_NL, IQ3_S, IQ3_XXS, Q2_0), against the rows kernel's LSU throttle at one column
    (Nsight: long scoreboard 3.9, lg throttle 2.8 per issue). Op 1 column -10 % to +5 %, tg512
    ~+1 %, MTP +0.3 %.
  - Kept: the layer walk's MoE runs once per group of chunks (up to 16,384 columns; design F4).
    Greedy ids of a 32K-token prompt (chunk 8192) identical; pp32768 3,845 -> 4,035 tok/s,
    pp131072 4,007; the CLI's 32K prompt 2.97K -> 3.36K tok/s.
  - Server with everything above (chunk 8192): prefill 4K / 32K / 128K 2,887 / 3,644 / 3,930 tok/s,
    decode 137.9 / 137.3 / 138.6 (decode-only expert hits 91-96 %); Strata 3,226 / 5,181 / 5,322
    and 198 / 188.5 / 160.
  - Expert tiles at 16,384 tokens (Nsight Compute): IQ2_S gate/up 253 registers, tensor pipe 29.8 %;
    IQ4_NL down 142 registers, tensor pipe 11.9 %, 16 % of warps active. Rejected: two down-tile
    CTAs per SM (128 registers, maximum shared carveout): no change at 4,096 or 16,384 tokens.
  - Rejected (decode, all exact): the narrow expert kernels' slice loops unrolled with every load in
    flight, and the down kernel's slices decoded once before the PDL wait (expert call at 4 tokens
    0.122 -> 0.149-0.159 ms); the dense rows kernel decoding one slice ahead (tg512 112.1 -> 109.1).
    Together: tg512 112.1 -> 105.0, MTP 197.7 -> 189.1 tok/s.
  - Rejected: an int64-run slice accumulator in the dp4a rows (exact; 5-column ops +1-27 %, MTP
    186.2 -> 177.9 tok/s); the prefill tile at 2-8 columns (125-234 us against 37); decoded planes
    for the experts' tiles (grouped by 18 experts, 128 MiB): tiles 5-13 % faster at 16,384 tokens
    but the plane decode costs ~7 %, net -2 %, and no end-to-end change since the walk's MoE calls
    are one chunk.
  - Server against the round above (chunk 4096 -> 8192, planes, restore; Strata uses 8,192): prefill
    4K / 32K / 128K 2,829 / 2,947 / 3,065 -> 2,864 / 3,570 / 3,735 tok/s; decode 141.6 / 138.9 /
    ~128 -> 144.6 / 129.1 / 140.6 (server decode medians of three move ~10 % with MTP acceptance).
  - Draft-length cost on IQ3_S (fixed K, bench corpus): rounds of K = 1..5 drafts cost 1.34, 1.62,
    1.85, 2.08, 2.33 plain rounds (~1.09 + 0.25 K) against the policy's 1 + 0.38 K (measured on
    NVFP4: 1.84, 2.13, 2.54 at K = 2..4); at 0.76 acceptance per position the true costs favour
    K = 3 where 0.38 picks 2 (~+4 % expected). Open: learn the cost per width online.
- *Fourth round (2026-10-12; RTX 5090, PCIe 5.0 x8; IQ3_S; every variant exact on
  `infernix_ggml_linear_test` and `infernix_offloaded_moe_ggml_test`, greedy ids of the short and
  32K prompts unchanged; A/B in one binary through temporary switches unless stated).*
  - Kept: the shared expert on its own stream in decode and verification calls (forked at the MoE
    input, joined before the combine), beside the routed experts: tg512 112.7 -> 121.5, MTP 199.6 ->
    205.6 tok/s. Kept: `mma_rows_kernel` at four CTAs per SM (128 registers): MTP 197.9 -> 199.6.
  - Kept: the draft-length cost learned online (FN §11.3): start-to-start times of one-row MTP rounds
    fit as a + b K by exponentially weighted least squares (decay 0.98, at least 16 rounds and spread
    K), the 0.38 prior until then. MTP tg512 200.1 against 191.3 with the fixed cost (+4.6 %, same
    binary); with the dp4a experts 214.1 tok/s. On NVIDIA NVFP4 Dense8 (2026-10-17, MTP 4 at 4K + 512,
    without the give-back reload): 159.4 / 157.3 against 151.7 / 151.7 with the fixed cost. The round
    counts then vary from run to run (the fit follows measured times); outputs do not.
  - Measured: int8 `mma.sync` k16 costs a k32's issue and pipe time (0.74 against 0.73 ms for the same
    instruction count; peaks 516 and 1,030 TOPS), so the half-sub formats (Q6_K, IQ2_S) run at half
    the int8 peak per MMA.
  - Rejected: tensor-core narrow expert kernels (`mma_rows` arithmetic per job: 8 intermediates as
    MMA rows g and g + 8 for an in-lane SwiGLU, 16 down rows per CTA of five warps, h encoded once per
    entry): expert call 0.034 -> 0.049-0.057 ms at 1 token and 0.093-0.112 -> 0.150-0.171 ms at 5;
    tg512 121.8 -> 112.3, MTP 214.1 -> 200.1. The dp4a kernels stay.
  - Rejected: a warp-specialised gate/up tile (eight decode warps filling a 3-stage ring of code
    stages and limbs by named barriers, eight MMA warps, 512 threads at 128 registers): no faster
    for IQ3_XXS and IQ3_S (expert call at 4,096 tokens 17.6-18.5 against 17.6-18.4 ms, both with the
    folded down below), IQ2_S 23.3 against 18.4 ms with the folded stage's spills.
  - Rejected: scale-folded down tiles (IQ4_NL and Q2_0: W = code x mant(d) x 2^(shift - base) as
    three byte planes, nine MMAs per slice accumulated by power over the whole K, one int128 add per
    output; 32-row tiles at 90 KiB of shared memory): expert calls at 4,096 tokens 12.2-12.9 -> 17.7-
    18.4 ms. With the ring above: pp32768 4,077 -> 3,701 tok/s.
  - Rejected: sub-scales folded into two byte planes for the half-sub formats (W = code x sub = L +
    256 H, six accumulating k32 MMAs per slice, no multiply-add): the dense Q6_K tile at 4,096 columns
    2,839 -> 3,047 us at two CTAs per SM (128 registers, 1.3 KiB of spills). At one CTA per SM (two
  builds, two rounds each): Q6_K at 4,096 columns attn_qkv 2,849 -> 2,816 us, ssm_out 1,705 ->
  1,655, shexp gate+up 483 -> 430, but attn_gate 1,782 -> 1,807; expert calls at 4,096 tokens mixed
  (11.8-12.9 ms either way); pp32768 4,093.4 / 4,093.1 -> 4,091.8 / 4,089.2 tok/s.
- *Fifth round (2026-10-13; the same setup; same-binary A/B through temporary switches unless
  stated; every variant exact on both Op tests and the expert cache test, greedy ids of the short
  and 32K prompts unchanged).*
  - Kept: one call per fused GGUF projection into its part tensors (`ggml_linear` over several
    outputs, `ggml_split_segments`): GDN qkv/z (and a/b when both are GGUF), attention q/gate/k/v and
    the index projection in one call (one encode of x), the shared expert's gate/up straight into
    SwiGLU's inputs; no split copies. Against the build before it: pp32768 4,093 -> 4,142 / 4,137
    tok/s (+1.1 %), tg512 121.0 -> 122.9 / 122.8 (+1.5 %), MTP 214.7 -> 214.8 / 216.0 (+0.1-0.6 %).
  - Rejected: 64 x 128 dense plane tiles (2 x 4 warps of 32 x 32, one CTA per SM): Q6_K at 4,096
    columns 2,850 -> 3,046 us (attn_qkv), Q5_K 2,821 -> 4,912, Q4_K 1,791 -> 3,462; pp32768 4,139 ->
    3,258 tok/s.
  - Rejected: narrow gate/up lanes taking runs of five consecutive slices (int64 sums per block, one
    int128 add per block; 78-80 registers with the run loop not unrolled): expert call at 1 token
    0.036-0.048 -> 0.034-0.039 ms, at 5 tokens 0.099-0.112 -> 0.102-0.115; tg512 122.9 -> 120.1,
    MTP 214.8 -> 211.9.
  - Measured (decode profile, MTP tg256 after warm-up; GPU 91.9 % busy): misses staged on the fork
    stream are hidden (the main stream waits 1.2 % of the time at the join), while the spin for the
    CPU-served misses costs 6.7 % (`cpu_wait_kernel`: median 1 us, p90 68 us, p99 186 us); the CPU
    path is the critical one. The decode miss divisor (derived d = 2) against fixed d, two rounds:
    tg512 at d = 1 / 2 / 3 / 4: 126.5 / 122.7-122.8 / 122.0-122.1 / 121.7-121.9 tok/s; MTP 197.2-197.4
    / 215.3-216.0 / 207.8-208.3 / 203.5-204.5. d = 1 in one-column calls only (same binary, two
    rounds): plain 123.2 / 122.9 -> 126.2 / 126.6 tok/s (+2.9 %), but MTP 216.2 / 216.1 -> 213.1 /
    212.8 (-1.5 %: an MTP run's one-column rounds lose). Adopted for GGUF imports without
    speculation only (FN §19.3.12); with MTP or n-gram drafts nothing changes.
  - MTP 5 drafts against 4 (the CLI's limit is 5; 7 was rejected by it), two rounds: 215.7 / 215.8 ->
    219.5 / 218.8 tok/s (+1.6 %), acceptance 0.74 -> 0.69-0.70. The default stays 4 in the guides;
    a cap above 5 is a product change (sampling windows are qualified for K = 1..5).
  - Prefill profile (pp32768, chunk 8192): GPU 95.6 % busy; H2D copies active 22.7 % (70.1 GB at
    27.8 GB/s), 19.5 % beside kernels, 3.2 % alone; idle 1.2 %. Prefill is compute-bound; the link
    has room for ~4x the compute rate. Kernel time: dense plane tiles 36.0 %, expert gate/up tiles
    25.6 %, down tiles 18.0 %, BF16 GEMMs 5.0 %, attention 3.6 %, HC inject/collapse/norm 3.3 %,
    `split_vector` 0.5 % (1.5 % before the fused projections). The small kernels bound any further
    fusion at ~3 %; not pursued.
  - After removing the rejected variants: pp32768 4,143 / 4,144 tok/s; tests and greedy unchanged.
  - End-to-end batch invariance, IQ3_S and FP8 F-A (greedy, generated text compared): the 32K
    prompt at prefill chunks 2,048 / 4,096 / 8,192; the short prompt plain, repeated, with MTP (K = 4)
    and with MTP plus n-gram (7) verification; `infernix-serve --max-concurrency 4` with MTP, four
    prompts each alone, A+B together and all four together (one ~6K-token prompt's prefill beside
    the others' decode), A alone again. Every comparison identical.
  - Teacher-forced exact log-probabilities (main's `infernix_decode_quality_forced` and
    `fast_path_gate identical`: 96 segments of 64 forced tokens, prompts 512 / 2,048 / 8,192, INT8
    KV, against prefill chunk 4,096), IQ3_S and FP8 F-A: bit-identical across a repeat, prefill
    chunks 2,048 and 8,192, concurrency 2 / 4 / 8 (FP8 at 8 not measured: its requests expired
    waiting for admission), MTP alone and at concurrency 4, and a prefix hit resuming at the whole
    prompt. A hit resuming inside the prompt (3,000 or 5,000 of 8,192 tokens warmed) differs in every
    such segment from the first forced token: mean |dlogp| 0.066-0.082, max 1.6 (IQ3_S) to 6.2 (FP8);
    the 27B shows the same. This is the documented contract (FN §19.3.1 E3/E4): a resume equals its
    capturing lineage, and a cold run only at a tap on the `--prefill-chunk` grid; GDN's chunked
    prefill runs in 16-token sub-chunks from each call's start, and 3,000 / 5,000 are off that grid.
  - Server (chunk 8192, warm; Infernix only, Strata from the third round): prefill 4K / 32K / 128K
    2,886 / 3,718 / 4,030 tok/s, decode 151.9 / 143.1 / 124.2; with `--ngram-draft-tokens 7`
    2,839 / 3,707 / 4,027 and 148.4 / 141.9 / 134.9.
- *Sixth round (2026-10-16; RTX 5090, PCIe 5.0 x8; after merging main; `infernix_bench` pp4096 +
  tg512, MTP 4 + n-gram 7 (`--lm-head-draft`), INT8 KV, five repetitions per run, rounds rotated;
  same-binary A/B through temporary switches unless stated; every kept variant exact on
  `infernix_ggml_linear_test`, the expert tests and teacher-forced identity).*
  - Merged build against the build before the merge, each with its kept switches: 195.4-196.0 against
    188.2-191.8 tok/s (three rounds); the exact route is bit-identical on the segments both builds'
    forced tools cut alike (one MTP segment at offset 0 with 512-, 2,048- and 8,192-token prompts).
  - Kept, leave-one-out from all ten exact narrow-kernel switches (all: 202.0 / 202.9; without each,
    two rounds): segment grouping (one launch per format per call: 196.7 / 195.5 without), the mixed-
    format verification launch (201.3 / 199.7), the 16-byte encoder (200.8 / 201.2), scalar CPU grid
    loads (200.2 / 195.7, the second noisy); staged MMA rows on top of all: 206.6 / 206.9. Op level
    at one column, GDN q/k/v/z 33.4 -> 23.0 us, attention q/gate/k/v 27.0 -> 19.5.
  - Rejected: L2 lookahead of the next projections (204.7 / 204.8 without), early PDL triggers and L2
    prefetch of a CTA's rows, CPU encodes in parts, the GDN fold on a side stream, KV page tables read
    through the host mapping (each within 0.3 % of all); the engine thread pinned to an E-core (decode
    194.9 against 193.8 average, within run spread; prefill 0.3 % slower in every pair).
  - Staging slots sized to a verification call's misses (top-k x width: 80 at MTP 4 + n-gram 7) without
    the SSD tier: 204.0 against 190.3 tok/s (+7.2 %, three runs); the FP8 import (SSD tier) keeps 64
    (48.6-49.8 against 51.2: the larger ring took RAM residents).
  - The Q2 MTP drafter artifact against the original: 201.4-203.1 against 196.1-196.7 tok/s
    (acceptance 0.90 against 0.885); exact-route bits are the main model's either way.
  - Rejected, model-quality §6 decode gate (Q2 artifact, MTP, 256 segments, a16/BF16-KV reference):
    Strata's route tail skip at rank 7 (INT8: TV excess bound 4.26e-3 > 4e-3, top-1 0.75 pt, p99.9 KL
    ratio 1.41) and a +1 logit bias for resident experts (TV 2.93e-2, top-1 3.8 pt, KL 2.09e-2): both
    fail. The HC-Q8 artifact (hyper-connection and final mixer projections in `q8_g32_fp16`) fails it
    too (INT8: TV 4.71e-3, top-1 0.73 pt, tail 1.36; BF16: TV 8.93e-3, top-1 1.21 pt, tail 1.70) and
    decodes 14.5 % slower (172.8-173.6 tok/s, acceptance 0.68).
  - Plain decode's one-column divisor, d = 2 against d = 1 (Q2 artifact, kept switches, two rounds):
    512 tokens 139.2-141.4 against 151.2-151.3 tok/s (hits 95.7 %), 4K 115.1-115.2 against 99.5
    (+15.8 %, hits 87.3 %), 32K 127.2 against 119.5 (+6.5 %, one round); keeping a call's lone miss on
    the link (the GPU keeps ceil(M / 2)) gave 141.4-141.8 / 107.7 / 120.4: rejected. The better
    divisor follows the hit rate, not the prompt length; a fixed choice loses at one end.
  - The `ggml_tiles16_v1` dense layout (the same blocks in 16-row tiles, so a tile's block is one
    contiguous run; artifact checked: 239 matrices untile byte-equal, everything else identical):
    against the row-major route with the kept switches, weighted op time per token (36 GDN, 12
    attention, 48 shared-expert calls, one head) +6.8 / +8.3 / +8.5 / +5.5 / -5.8 % at 1 / 2 / 4 / 5 / 8
    columns (best route per width among rows_kernel with tiled addressing, a staged dp4a kernel of 23
    shapes and a staged MMA kernel of 10 shapes); end to end MTP 4 + n-gram 7 217.1-217.6 against
    214.6-216.2 tok/s (+0.9 %), plain decode 98.6-98.7 against 99.4-99.5 (-0.8 %); exact route
    bit-identical (48 segments). Not adopted: no gain at verification widths below 8 and a loss at one
    column. The gap is in the IQ4_XS-holding projections (single Q4_K / Q6_K matrices equal) and the
    two-format verification launch, which the tiled route lacks.
  - Second tiled iteration (the dp4a kernel removed; a tensor-core tiled kernel whose warps copy a
    tile's 16-row block runs by 16-byte cp.async, L2 evict-first, into two-slot rings, `unit_sums`
    MMAs, K split across warps with exact int128 partials in shared memory, and a two-format tiled
    launch; rows_kernel with tiled addressing for one-column Q6_K and IQ4_XS): the kept defaults are
    bit-identical to the slot before on 256 teacher-forced MTP segments; end to end MTP 4 + n-gram 7
    221.1-222.1 against 214.2-215.3 tok/s (+3.3 %, three rotated rounds), plain decode 98.7-98.8
    against 99.5 (-0.8 %). The shape sweep (16 shapes, five widths) found one cause for every outlier:
    a second CTA wave costs a call 25-30 % (GDN q/k/v/z, 1,024 tiles: 4 warps split 2 is 512 CTAs
    against 510 resident, 28.8 us at one column; unsplit, 21.9); with a shape that fits one wave the
    tiled GDN q/k/v/z, attention q/gate/k/v and shared-expert calls beat the row-major route at every
    width, one column included (21.9 / 17.1 / 6.4 against 22.7 / 19.3 / 12.1 us), while single Q6_K
    calls at one column stay 1-3 % behind (the GDN output 13.0 against 12.7 us).
  - Adopted: the tiled layout with CTA shapes from the occupancy API (the first of 8/4/2/4/4/2 warps
    split 8/4/2/2/1/1 whose CTAs fit one wave and that keeps the call within 1,024 warps or gives a
    warp at least four blocks; past one wave, 4 warps unsplit) and two-format tiled launches at every
    narrow width. Weighted op time per token against the row-major routes -11.9 / -22.9 / -25.6 /
    -26.7 / -31.8 % at 1 / 2 / 4 / 5 / 8 columns; end to end MTP 4 + n-gram 7 221.7-222.6 against
    214.0-214.5 tok/s (+3.7 %, three rotated rounds; every tiled round ahead of every row-major one),
    plain decode 99.6-99.7 against 99.5-99.6 (+0.1 %, within the runs' spread), prefill pp4096
    3,452-3,462 against 3,449-3,462; exact route bit-identical (64 teacher-forced MTP segments). The
    row-major route's own two-format launch at one and two columns, as a control: 213.0-214.8 MTP
    and 98.6 plain (its GDN and attention calls lose more than its shared expert gains), so the
    gain is the layout's. The IQ3_S recipes store their Q4_K, Q5_K, Q6_K and IQ4_XS dense matrices
    tiled.
  - Plain decode's divisor against the hit rate over 4,096 routed uses (d = 1 at 92 % or more): 148.0 /
    150.4 / 124.0 / 107.6 / 120.7 tok/s at 512 / 1K / 2K / 4K / 32K prompts against d = 1 151.5 /
    152.4 / 124.9 / 99.5 / 119.5 and d = 2 139-141 / 143.8 / 129.9 / 114.4 / 126.3 (hits 95.9 / 92.9
    / 89.2 / 87.3 / 88.3 %): below the better fixed divisor at every length (47 switches in a 4K run:
    eight-token windows swing around any threshold). Rejected; d = 1 wins up to 1K and d = 2 from 2K.
    Adopted instead: d by context, 1 below 1,536 tokens and 2 from there (151.4 tok/s at 512 + 256,
    115.2 at 4K + 512; at 1K + 1,024, 114.4 against d = 1 105.5 and d = 2 121.0: a long output from a
    short prompt would gain from d = 2 before the switch). Never below the previous d = 1 in these
    runs; the 2K and 32K points take d = 2 by the rule, so they rest on the fixed-d runs above.
- *Deferred, with reasons:* a persistent Stream-K expert schedule waits for a new attribution (the fourth round's warp-specialised tile showed the
  expert tiles' decode, not its alternation with the MMAs, bounds them).
- *Not applicable:* int4 `mma` (IQ3_S codes reach ±15, IQ4_NL ±127); FP8/FP6/FP4 and block-scaled
  `mma` (floating-point accumulation, not exact); `ldmatrix .b4x16_p64` unpacking (PTX 8.6,
  sm_120a: FP4 container layout, no codebook); T-MAC/LUT-GEMM activation tables (a grid index with
  its sign bits is 13 bits per 4 elements: 32 KiB per group and column); fewer activation limbs
  (X needs 21 bits).

This is a temporary planning document for a bit-exact Infernix
artifact of [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
`IQ3_S/` with an MTP drafter sourced from Qwen's checkpoint, and for the optional RAM-resident n-gram
table. When the work lands, its stable content moves into the authorities named in
[§14](#14-documentation-and-authority-changes) and this file is removed. The Flash-Next design is
cited as **FN §x** ([qwen3_8-flash-next-design.md](qwen3_8-flash-next-design.md)) and the GLM-5.3
design as **GLM §x** ([glm5_3-flash-design.md](glm5_3-flash-design.md)).

Code references are to the newest local tree, branch `Infernix` at `217135bba` (worktree
`E:\NInfer-V3\local\workdirs\NInfer-V3-issue1`, 22 commits ahead of `origin/main`). The companion
plan for the official FP8 checkpoint is [qwen3_8-flash-next-fp8-plan.md](qwen3_8-flash-next-fp8-plan.md);
§13 lists the work both plans share.

Labels: **measured** (this machine, method named), **upstream** (read from the source file or
code), **model** (arithmetic on measured or upstream inputs), **estimated** (judgement, with a
range). A milestone in brackets replaces each model or estimate with a measurement.

---

## Contents

1. [Deliverable, baselines and acceptance](#1-deliverable-baselines-and-acceptance)
2. [Source facts](#2-source-facts)
3. [What other implementations do](#3-what-other-implementations-do)
4. [Design decisions](#4-design-decisions)
5. [Artifact and conversion](#5-artifact-and-conversion)
6. [Arithmetic contracts](#6-arithmetic-contracts)
7. [Routed experts](#7-routed-experts)
8. [Dense path](#8-dense-path)
9. [PLE n-gram table: IQ4_NL rows and RAM residency](#9-ple-n-gram-table-iq4_nl-rows-and-ram-residency)
10. [MTP drafter from Qwen's checkpoint](#10-mtp-drafter-from-qwens-checkpoint)
11. [Performance model](#11-performance-model)
12. [Qualification](#12-qualification)
13. [Milestones](#13-milestones)
14. [Documentation and authority changes](#14-documentation-and-authority-changes)
15. [Risks, options and rejected alternatives](#15-risks-options-and-rejected-alternatives)
16. [Sources](#16-sources)

---

## 1. Deliverable, baselines and acceptance

### 1.1 Deliverable

- A recipe `qwen3_8_flash_next_gsq_rco_iq3s` producing `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix`:
  every tensor of the GGUF stored **bit-exactly** (codes and scales verbatim; only lossless
  re-layouts: plane splits, whole-row permutations, per-expert records, alignment padding), plus an
  IQ4_NL n-gram volume holding shard 2's rows verbatim.
- An MTP drafter and proposal head built from Qwen's MTP tensors (§10). The GGUF has none
  (llama.cpp's exporter sets `supports_mtp_export = False`). The drafter only proposes; verification
  keeps the target's output, so it does not affect exactness.
- Vision from the repository's BF16 `mmproj` (exact), as a later milestone.
- Runtime support through the public Engine for every route Flash-Next has today: plain and MTP
  decode, prefill, the hybrid prefix cache, C ≤ 8, VRAM/RAM/SSD expert tiers.
- `--ngram-residency auto|ram|volume`: the whole n-gram table in host RAM, or on NVMe as today (§9).

### 1.2 Baselines (RTX 5090 on PCIe 5.0 x8, i9-13900K, 96 GB DDR5-5800)

| Baseline | Decode | Prefill | Source |
|---|---|---|---|
| Infernix recipe B (NVFP4 experts, 8-bit dense), INT8 KV | plain 132.2, MTP 157.6 tok/s (pg1024+256 corpus) | pp131072 5,767 tok/s; agentic no-hit prefill 5,879 tok/s | measured, FN §19.3.21, README |
| Strata 0.1.38-0.1.39, this GGUF (IQ3_S), RTX 5090 | 161-193 tok/s (9950X3D/9700X, Linux); 127-142 (0.1.33, Windows, shared GPU) | 3,072-6,036 tok/s (4K-128K) | upstream, Strata `bench/results/2026-10-0{3,5}*` |
| llama.cpp #29887 GPU expert cache, Flash-Next Q4_0, RTX 5090, PCIe 4.0 | 67.8 tok/s | — | upstream, PR #29887 |

Strata's numbers are on different CPUs (Zen 4/5 with AVX-512), use GPU/CPU paths with different
rounding (q8_1 vs Q8_K activations), and by default skip low-rank misses (route tail skip, which
changes outputs). Before the design is frozen, M0 measures Strata on this machine with the same file,
with tail skip both on and off.

### 1.3 Acceptance

- **Must, exactness.** `verify_artifact` reconstructs every GGUF tensor's block bytes from the
  artifact and compares them byte for byte (transformed tensors: value for value through their
  stated exact maps, §5.3). Every n-gram row matches. Every Op passes its oracle (§12).
- **Must, invariance.** Greedy output is independent of expert placement (VRAM frame, staging, CPU
  on every ISA), decode width, C ≤ 8, and MTP on/off at C = 1, as for recipes A–C (FN §16.4).
- **Must, quality.** The §16.5 protocol (KL to the 16-bit reference on the frozen texts at several
  prefill chunk sizes) is reported against recipe B. ISTA's own claim (IQ3_S task average 93.26
  against 93.12 BF16) is upstream only. Adoption as a recommended artifact needs KL no worse than
  recipe B's, or a documented trade.
- **Targets (estimated, §11).** Plain decode ≥ recipe B (≥ 132 tok/s at C = 1), stretch 150; MTP
  ≥ 160; pp131072 ≥ 5,800 tok/s; a cold 16K prompt faster than recipe B when the table is in RAM.
  These are gates for the performance milestones, not completion conditions for correctness.

### 1.4 Non-goals

- Arbitrary GGUFs or other mixes. The code is per-format and per-shape, as `ops::linear` is today.
  The `IQ3_XXS`/`IQ2_XS`/`Q2_0` builds of the same repository use the same formats (some only IQ2_S
  or Q2_0), so they come almost free; that is checked, not promised.
- Loading the `.gguf` in place (§15.3).
- Reproducing llama.cpp's logits bit for bit: llama.cpp quantizes activations to Q8_1, Infernix keeps
  its exact A16 arithmetic (§6).

---

## 2. Source facts

Read on 2026-10-10 from the GGUF headers of the local `GSQ-RCO-Coder` build (same writer, same type
histogram), from ISTA's IQ3_S `tensor-allocation/*.rco-allocation.txt`, and from llama.cpp
`3cf03257` (vendored at `E:\Strata\Strata\third_party\llama.cpp`).

### 2.1 Files

| File | Size | Content |
|---|---:|---|
| `IQ3_S/…-00001-of-00002.gguf` | 54.8 GB | 1,223 tensors, 67 metadata keys, arch `qwen4exp`; SHA256 `4c1eb2ce…` (upstream) |
| `IQ3_S/…-00002-of-00002.gguf` | 28,800,138,240 B tensor | one tensor `per_layer_token_embd.weight` [160 × 320,001,536] IQ4_NL at file offset 192; SHA256 `316b46f3…`. **Already on disk**: `E:\Strata\Strata-data\models\coder-IQ1_M\…-00002-of-00002.gguf` has the same SHA256 (measured) |
| `mmproj-Qwen3.8-Flash-Next-BF16.gguf` | 0.91 GB | vision tower, BF16 |

Only shard 1 (54.8 GB) needs downloading; M0 pins the repository revision.

### 2.2 Types (shard 1, 1,223 tensors)

BF16 484, F32 292, Q6_K 129, IQ4_NL 86, Q4_K 47, IQ4_XS 45, IQ2_S 40, Q5_K 35, IQ3_XXS 34,
IQ3_S 20, Q2_0 9, F16 1, Q8_0 1. Per class (upstream allocation file):

| Class | Types (count of layers) |
|---|---|
| `token_embd` [2560 × 248,320] | IQ4_XS |
| `output` (lm_head) | Q6_K |
| GDN `attn_qkv` [10240 × 2560] | Q6_K 22, Q5_K 10, Q4_K 2, IQ4_XS 2 |
| GDN `attn_gate` (z) [6144 × 2560] | Q6_K 14, Q4_K 10, IQ4_XS 8, Q5_K 4 |
| GDN `ssm_out` [2560 × 6144] | Q6_K 29, Q5_K 4, Q4_K 3 |
| QSA `attn_q` [12288 × 2560] | Q6_K 5, Q5_K 3, IQ4_XS 3, Q4_K 1 |
| QSA `attn_k`, `attn_v` [512 × 2560] | Q6_K (k 11, v 10), IQ4_XS 1 (k), Q5_K 2 (v) |
| QSA `attn_output` [2560 × 6144] | Q6_K 11, Q4_K 1 |
| shared `ffn_gate_shexp`, `ffn_up_shexp` [640 × 2560] | Q4_K/IQ4_XS/Q6_K/Q5_K mixed per layer |
| shared `ffn_down_shexp` [2560 × 640] | IQ4_NL 47, Q8_0 1 (layer 47) |
| routers, HC mixers (`hc_*_down/up/inject`, `output_hc_*`), GDN α/β, indexer q/k, PLE key/value | BF16 |
| norms, `ssm_a`, `ssm_conv1d`, `ssm_dt.bias` | F32 |
| `ple_conv1d` | F16 |

### 2.3 Routed experts: seven per-layer combinations

Gate and up always share a type (upstream). Bytes per expert follow from the block sizes
(IQ2_S 2.5625, IQ3_XXS 3.0625, IQ3_S 3.4375, IQ4_XS 4.25, IQ4_NL 4.5, Q2_0 2.25 bits per weight):

| gate/up + down | Layers | Record bytes (model) | 4 KiB pages |
|---|---|---:|---:|
| IQ2_S + IQ4_NL | 2-6, 9, 11, 15, 19, 23, 24, 26, 32, 34, 39 (15) | 1,971,200 | 481.25 |
| IQ3_XXS + IQ4_NL | 0, 7, 14, 16, 18, 22, 31, 33, 37, 38, 41-43, 46 (14) | 2,176,000 | 531.25 |
| IQ3_S + IQ4_NL | 17, 25, 27-29, 35, 40, 44, 45 (9) | 2,329,600 | 568.75 |
| IQ2_S + Q2_0 | 1, 8, 10, 13, 36 (5) | 1,510,400 | 368.75 |
| IQ3_XXS + Q2_0 | 12, 20, 30 (3) | 1,715,200 | 418.75 |
| IQ3_S + Q2_0 | 21 (1) | 1,868,800 | 456.25 |
| IQ4_XS + IQ4_NL | 47 (1) | 2,662,400 | 650 |

All experts: 50.29 GB (46.8 GiB), mean 2.046 MB per expert, against 2,764,800 B for NVFP4 (−26 %).
The tensors are fused 3-D: `ffn_gate_exps`/`ffn_up_exps` [2560, 640, E], `ffn_down_exps` [640, 2560, E]
(ggml order), so each expert is a contiguous slice of each tensor.

### 2.4 Formats (upstream `ggml-common.h`, llama.cpp `3cf03257`)

| Type | Block | Bytes | Value |
|---|---:|---:|---|
| Q2_0 | 64 | 18 | `(c − 1) · d`, c ∈ {0..3}, fp16 d (llama.cpp #24448) |
| IQ4_NL | 32 | 18 | `d · kvalues_iq4nl[nibble]`, 16 values in [−127, 113]; low nibbles are elements 0-15, high 16-31 |
| IQ4_XS | 256 | 136 | `d · (ls − 32) · kvalues_iq4nl[nibble]`, 6-bit ls per 32 |
| IQ2_S | 256 | 82 | `d · (0.5 + s) · 0.25 · iq2s_grid[10-bit][j] · ±1`, 4-bit s per 16, grid values {8, 25, 43}, explicit sign bits |
| IQ3_XXS | 256 | 98 | `d · (0.5 + s) · 0.5 · iq3xxs_grid[8-bit][j] · ±1`, 4-bit s per 32, signs 7 bits + parity |
| IQ3_S | 256 | 110 | `d · (1 + 2s) · iq3s_grid[9-bit][j] · ±1`, 4-bit s per 32, explicit sign bits |
| Q4_K | 256 | 144 | `d · sc · q − dmin · m`, 6-bit sc and m per 32 |
| Q5_K | 256 | 176 | as Q4_K with a fifth bit |
| Q6_K | 256 | 210 | `d · sc · (q − 32)`, int8 sc per 16 |
| Q8_0 | 32 | 34 | `d · q`; equals Infernix `q8_g32_fp16` byte for byte when no code is −128 |

Every value is **an integer code times an integer sub-scale times one fp16 scale** (Q4_K/Q5_K: two
such terms). This is what makes the exact arithmetic of §6 possible.

### 2.5 Export transforms to undo (upstream `conversion/qwen.py`, `conversion/qwen4exp.py`)

| Transform | Tensors | Plan |
|---|---|---|
| `+1` on every `*norm.weight` except `linear_attn.norm.weight` (Qwen3-Next rule), and explicitly on PLE `norm_key/query/conv` and indexer `q/k_layernorm` | F32 gammas | §5.3 |
| `A_log → −exp(A_log)` | `ssm_a` F32 | bind A directly (§8.4) |
| GDN V-head reorder (ggml tiled order) | rows of `attn_qkv` V part, `attn_gate`, α, β, `ssm_dt`, `ssm_a`, conv V channels; **columns** of `ssm_out` | §5.3, §8.4 |
| indexer `index_qk_proj` split into `indexer.q_proj` / `indexer.k_proj` | BF16 | concatenate rows |
| `ple.conv1d` squeezed; PLE shards concatenated into one table | F16, IQ4_NL | §5.3, §9 |
| MTP dropped | — | §10 |

---

## 3. What other implementations do

Read 2026-10-10. [M] = the source reports a run, [C] = a claim, [I] = our inference.

| Source | Technique | Takeaway here |
|---|---|---|
| **Strata** (`E:\Strata\Strata`, 0.1.42) | Per-layer-sized expert slots that never move between layers (`expert_cache.hpp`); raw GGUF blocks per expert `[gate|up|down]`; grouped expert kernels with grid staging in shared memory only for IQ2_* (not IQ3_*); IQ4_NL via `__byte_perm` in registers; `kvalues` in `__constant__` was 2.12× slower [M]; decode-once-apply-to-all-columns with a 16-lane sub-warp that reproduces the 32-lane add tree; fused SwiGLU; PDL with weight loads before `griddepcontrol.wait`; opt-in int8 `mma.sync` prefill with smem codebooks and `cp.async` 4 stages (+4-11 % [M]); Q8_0/Q6_K plane repacks +10-12 % [M]; CPU AVX2 `vpgatherdd` grid decode chosen per core (IQ3_S row 0.34 → 0.19 ms on a 14900KF P-core [M]); host-dequantized PLE rows; `--ple-io ram` (map + VirtualLock after a threaded fault-in) | Per-layer slot sizing, plane repacks, decode-once, smem grids per format by measurement, byte-perm IQ4_NL, CPU gather choice per core: adopted. Strata's GPU/CPU paths round differently (q8_1 vs Q8_K), so output depends on placement (its issues #152, #410): rejected (§6). |
| **llama.cpp** mainline | MMVQ dp4a against Q8_1; signs via `__vcmpne4`/`__vsub4`; IQ4 via two `__byte_perm` 8-entry lookups; MMQ loads i-quants into int8 tiles and reuses the Q8_0 int8-MMA; Q2_0 symbols via `__byte_perm(…, 0x5140/0x7362)` (+36 % tg, #25603 [M]); GPU LRU for host experts (#29887): 67.8 tok/s at 89 % hits [M] | Sign and nibble tricks adopted verbatim in the weight-decode stage. Q8_1 activations rejected (§6). |
| **ik_llama.cpp** | Row-interleaved repacks (`_R4/_R8`) so one activation load feeds 4-8 rows; for slow-to-unpack i-quants, unpack a row block once to 8-bit and run one int8 GEMM over all columns (#515, IQ2_XXS pp 2.7× [M]); AVX-VNNI `dpbusd` kernels for IQ3_S/IQ3_XXS/IQ4_NL/Q4_K/Q5_K/Q6_K R4 (#1446-#1482) | Converter-side row interleave of gate/up (§5.4) and of CPU expert rows; unpack-once-then-int8 for CPU multi-column misses and prefill CPU share (§7.4). |
| **KTransformers** (SOSP'25) | AMX/AVX-512 experts; expert deferral (changes the mathematics) | No AMX/AVX-512 here; deferral rejected (FN §1.4). |
| QTIP / EXL3 / AQLM | Computed or tiny codebooks; codebooks ≤ 8 KB in shared memory are fast, 2^16 entries are not [C] | Confirms that IQ2_S's 8 KB grid in shared memory is fine. |

---

## 4. Design decisions

| # | Decision | Reason |
|---|---|---|
| D1 | **Exact import** of every GGUF block; Infernix-native re-layouts chosen by the converter (planes, interleaves, records), never by the runtime | AGENTS: one artifact, no runtime repacking; bit-exactness is checkable byte for byte |
| D2 | **One canonical arithmetic per class, exact up to a single final rounding** (§6): exact integer products of the stored codes with the existing A16 activation encoding, sub-scales applied in integers, and an exact 128-bit combine of the fp16 scales | Placement invariance (FN §16.2.1, GLM L13) without constraining reduction order; each route (SIMT, IMMA, CPU) may split work freely |
| D3 | **Expert records per layer**, each a whole number of 4 KiB pages; frames in **size classes** with one global LFRU (§7.5) | Seven record sizes; one max-size frame wastes 23 % of VRAM (§11.1) |
| D4 | **Dense layers**: per-(format, shape) routes in `ops::linear`, SIMT and column-invariant for T ≤ 8, tensor-core for prefill; multi-format **segmented** launches where the converter cannot group same-format parents (§8.2) | RCO varies the format per tensor and per layer |
| D5 | **The n-gram table in IQ4_NL**, verbatim rows, 45 per 4 KiB block; `--ngram-residency` RAM mode holding the whole table (26.8 GiB) in locked host memory (§9) | Shard 2 is IQ4_NL; the RAM mode removes NVMe from the PLE path |
| D6 | **MTP from Qwen's checkpoint**, re-quantized as recipes A–C do (`q4_g64_fp16` experts, `q8_g32_fp16` dense) (§10) | The GGUF has no MTP; the existing drafter path is measured and qualified |
| D7 | **Keep the BF16 tensors BF16** (HC mixers, routers, indexer, PLE key/value) | Bit-exactness; a re-quantized variant is a separate gated option (§15.2) |

---

## 5. Artifact and conversion

### 5.1 Recipe

| Class | GGUF | Stored as | Kernel family |
|---|---|---|---|
| Routed experts | IQ2_S/IQ3_XXS/IQ3_S/IQ4_XS + IQ4_NL/Q2_0 | `ggml_expert_v1` records (§5.4), per-layer format pair in the bank's metadata | `offloaded_sparse_moe` ggml family (§7) |
| Dense projections, shared experts, lm_head | Q4_K/Q5_K/Q6_K/IQ4_XS/IQ4_NL | new formats `ggml_q4_k`, `ggml_q5_k`, `ggml_q6_k`, `ggml_iq4_xs`, `ggml_iq4_nl` in plane layouts (§5.2) | `ops::linear`, `projection_fp32` (lm_head) |
| `ffn_down_shexp` layer 47 | Q8_0 | `q8_g32_fp16` (exact; converter asserts no −128 code) | existing Q8 |
| BF16 tensors | BF16 | BF16 | existing |
| `token_embd` | IQ4_XS | `ggml_iq4_xs`, host-resident | host embedding gather with exact decode (§8.5) |
| F32 norms | F32 (some `+1`) | per §5.3 | existing norms, or a direct-gamma FP32 variant |
| `ssm_a` | F32 (−exp(A_log)) | FP32, role `ssm_a` | GDN A-direct variant (§8.4) |
| `ssm_conv1d`, `ssm_dt.bias` | F32 | BF16 if every value round-trips exactly, else FP32 | existing |
| `ple_conv1d` | F16 | FP32 (exact widening, the precedent for BF16 → FP32) | PLE conv FP32 variant |
| n-gram table | IQ4_NL | volume `iq4_nl_rows_v1` (§9.1) | `ple_embed` IQ4_NL variant |
| MTP, proposal head | — (Qwen checkpoint) | as recipes A–C (§10) | existing |
| Vision | `mmproj` BF16 | BF16 | existing (`--vision-offload`) |

### 5.2 Formats and plane layouts

Each new format is defined in [tensor formats](tensor-formats.md) with an **exact decode oracle**
(Python, `tools/artifact/codecs/ggml_*.py`) and an **exact inverse** to GGUF block bytes. The stored
layout splits each block into aligned planes so every kernel load is a vector load (GGUF blocks are
2-byte aligned: 18, 34, 82, 98, 110, 136, 144, 176, 210 bytes):

- per row, planes `codes` (16-byte aligned), `signs`/`qh` (where present), `sub_scales`,
  `d`/`dmin` (fp16), each contiguous over the row's blocks;
- rows of a matrix in tiles of 16 (`rg16`), the unit the SIMT routes read;
- Strata measured +10-12 % for this split on Q8_0 and bitwise equality on Q6_K (§3).

The converter writes planes from raw blocks; `verify_artifact` re-assembles the blocks and compares
bytes. No runtime repacking.

### 5.3 Name map and exact transforms

`tools/convert/sources/gguf.py` gains `qwen4exp_name_map` (all classes of §2.2) and a raw-block
reader (`GGUFSource.read_blocks`) that returns stored bytes, never dequantized values. Transforms,
each exact and checked by the converter:

- **Norm `+1`.** For each shifted F32 gamma, compute g − 1 exactly in FP64. If it is representable
  in the format the Infernix binding uses (BF16 for the HC/QSA/indexer/PLE norms) and round-trips,
  store that; otherwise store FP32 g − 1 with an FP32-gamma norm, or the GGUF value with a
  direct-gamma norm variant. Which tensors carry the shift is decided per tensor in M1 and checked
  against Qwen's BF16 values of the same tensors (§10.1 fetches them).
- **V-head order.** Row-wise tensors are permuted back to Hugging Face order. This is exact: whole
  rows, and every K-block lies within one row. `ssm_out`'s permutation is on its **input** axis at
  128-element head granularity, but Q4_K/Q5_K/Q6_K superblocks span 256 elements (two heads), so it
  **cannot** be undone exactly. `ssm_out` stays in GGUF order, and the GDN output stage writes head h
  to slot π(h) (§8.4).
- **Indexer split**: the q and k rows are concatenated into the QSA indexer parent.
- **F16, F32**: exact widening, or exact narrowing when every value round-trips (checked).

### 5.4 Expert records (`ggml_expert_v1`)

One record per expert, `record_bytes(layer)` a multiple of 4,096:

```text
[ gate/up rows, interleaved in 16-row groups (g0..g15, u0..u15, g16..), split into planes per group ]
[ down rows, in 16-row groups, split into planes ]
[ padding to 4 KiB ]
```

- Gate/up interleaving gives one pass over x for both and a fused SwiGLU, as `nvfp4_expert_rg16_v1`
  does (FN §6.2) and Strata's S2 pack does. Interleaving whole rows is a lossless re-layout.
- Planes per 16-row group put `codes` first (the bulk), then `signs`/`qh`, `sub_scales`, `d`, all
  16-byte aligned. Bank metadata records the per-layer format pair and plane offsets.
- Padding overhead is 0-0.2 % per record (§2.3). The bank tail keeps no multiplier plane: GGUF scales
  live in the blocks.
- The converter writes 32-expert chunks as `import_expert_bank` does (`tools/convert/qwen4_exp.py`).
  `verify_artifact` decodes every record back to the GGUF slices.

---

## 6. Arithmetic contracts

### 6.1 Expert arithmetic: exact up to one rounding

Reuse the A16 activation encoding of FN §16.2.1 unchanged: a BF16 column v (K = 2,560 for x, 640 for
h) becomes integers X_j (|X_j| < 2^21) with one column exponent emax; v_j = X_j · 2^(emax−147)
exactly for elements within 2^13 of the column maximum. X is stored as three byte limbs (lo, mid
unsigned; hi signed).

For every format, an element's weight is w = Σ_t d_t · κ_t · m_t · c, where c is a small integer code
(|c| ≤ 127), m_t an integer sub-scale, d_t an fp16 scale and κ_t a power of two fixed by the format
(t ranges over one term, or two for Q4_K/Q5_K mins). The canonical value of one output row is

```text
P_s   = Σ_{j ∈ sub-block s} c_j · X_j                         exact: per byte limb in int32, limbs combined in int64
Q_b   = Σ_{s ∈ superblock b} m_s · P_s                        exact int64
R     = Σ_b mant(d_b) · Q_b · 2^(exp(d_b) − e_min)             exact int128 (fp16 exponents span ≤ 30 binades)
y     = bf16_rn( fl32_rn(R) · 2^(e_min + emax − 147 + log2 κ) )   one rounding to FP32, one to BF16
```

- Bounds (model): IQ4_NL |c| ≤ 127 over 32 elements gives 127 · 32 · 2^21 < 2^33, so P is formed per
  byte limb in int32 and combined in int64 (as FN §16.2.1 does). IQ2_S/IQ3_* |c| ≤ 62 per 16/32;
  Q6_K |q − 32| ≤ 32 per 16; Q2_0 |c − 1| ≤ 2 per 64. Q4_K/Q5_K add Σ_s m_s · ΣX_s for the min term.
- **Any order.** Every sum before R's rounding is exact, so SIMT, IMMA and CPU routes may split K,
  rows and columns however they like and still return the same bits. Column count, placement and
  route never change a bit, which is FN's invariant and the reason GLM L13 adopted exact arithmetic.
- The int128 combine runs once per superblock per output (10 per row at K = 2,560), so its cost is
  negligible. On the CPU it is two `_addcarry_u64`/`_mulx` steps (MSVC has no `__int128`).
- SwiGLU, h encoding and the down product follow FN §16.2.1 exactly:
  `x (BF16) → A16 → gate/up → BF16 → SwiGLU → BF16 h → A16 → down → BF16 y_e → combine`.

**Oracle.** Exact rationals (Python `Fraction`) over the decoded GGUF values and the encoded X, then
the same two roundings. Production output must be **bit-equal** to the oracle, not just close (§12.1).

### 6.2 Dense arithmetic

Dense layers run only on the GPU, so they need column invariance (T ≤ 8 on SIMT routes, FN §19.2) but
not CPU equality.

- **Decode and verify (T ≤ 8): exact-integer SIMT, the same contract as §6.1.** The bytes per token
  are ~4.2 GB (§11.2); the integer work is ≤ 3 dp4a per 4 weights per column, below DRAM time at
  T ≤ 8 (model: ~1.3 ms of integer issue against ~2.7 ms of DRAM at T = 8). One contract for experts
  and dense also means one oracle family.
- **Prefill (tensor cores):** codes are exact in BF16 (|c| ≤ 127 < 2^8); a BF16 `mma.m16n8k16` per
  16-element sub-block, with the integer sub-scale and fp16 scale applied in an FP32 epilogue per
  k16 step. Qualified normwise against FP64 (FN's dense rule), not bit-equal to the SIMT route; this
  is the same split recipe B already has between its SIMT and tensor-core routes.
  - Option, gated: exact IMMA prefill with the §6.1 sums (three byte-limb MMAs per k32). It is chosen
    only if M5 measures it within 10 % of the BF16 route.
- `lm_head` (Q6_K): a `projection_fp32` variant (FP32 logits), tensor-core at 1..16 columns as recipe
  A's BF16 projection is (701-710 µs against a 709 µs DRAM floor).

### 6.3 PLE rows and embedding

- n-gram rows: `out = bf16_rn(fl32(d) · kvalue)`. The product is exact in FP32, then one rounding at
  the same BF16 boundary `ple_embed` has today (`include/infernix/ops/ple.h`).
- Embedding (IQ4_XS): decoded exactly to FP32 on the host. Whether the first consumer takes FP32 rows
  or a BF16 rounding is a semantic-boundary decision measured in M2 by KL (one extra rounding of
  2,560 values per token).

---

## 7. Routed experts

### 7.1 Op contract changes (`include/infernix/ops/offloaded_sparse_moe.h`)

- `MoeExpertSource` gains `ExpertFormat format` (`nvfp4_rg16` or `ggml{gate_up, down}`) and a
  per-call `RecordGeometry` (plane offsets, record bytes), replacing the global
  `kRecordBytes`/`kGateUpBytes` constants (`cpu/w4a4_expert.h:16-36`, `wide_expert.cuh:44-58`,
  `moe_layer.cu:390, 1763-1767`).
- `stage_kernel` copies `record_bytes` from the geometry. `launch_pass` and the wide `prepare`/`run`
  dispatch on the format.
- The NVFP4 family is unchanged. The ggml family is a second implementation behind the same entry
  points, as `model-quality.md` §4.4 anticipated.
- Kernels are templated on `<GateUpFormat, DownFormat, Columns>`. Seven format pairs × narrow
  (Columns 1..8) × wide × CPU.

### 7.2 GPU narrow route (decode, ≤ 8 columns)

One CTA per (expert, 16-row group), `gate_up` then `down`, PDL-chained, as `gate_up_kernel<Columns>`
and `down_kernel<Columns>` are today:

- **Weight decode once, apply to every column** (Strata's `Split::load`/`apply`), so the column loop
  never re-decodes.
- **Grids**: IQ2_S's 8 KB grid staged in shared memory per CTA; IQ3_S (2 KB) and IQ3_XXS (1 KB)
  chosen between shared memory and L1-cached global by microbenchmark (Strata: staging helped grouped
  kernels and hurt single-matrix ones). IQ4_NL values via the two-`__byte_perm` lookup in registers,
  never `__constant__` (divergent indices serialize: 2.12× slower, Strata).
- **Signs**: IQ2_S/IQ3_S explicit bits and IQ3_XXS parity bits through `__vcmpne4`/`__vsub4`
  (branch-free). Q2_0 symbols via `__byte_perm(…, 0x5140/0x7362)` (llama.cpp #25603).
- **Products**: dp4a of the decoded int8 codes against X's byte limbs (3 dp4a per 4 weights per
  column), int32 per sub-block, int64 per superblock, int128 combine (§6.1).
- **x staging**: X limbs for up to 8 columns in shared memory once per CTA (≤ 61 KB at 8 columns of
  2,560 × 3 bytes).
- **Layer-adaptive**: the format pair is fixed per layer, so each layer's launch is one template
  instance; the graph captures the right instance per layer.
- Budget (M3 gate): each kernel within 10 % of its byte-time at 1.55 TB/s for n = 1, 2, 4, 8 (FN's
  cost-gate rule, §16.2).

### 7.3 GPU wide route (prefill, > 8 columns)

- Integer tensor cores: `mma.sync.m16n8k32.s8.u8/s8` (k16 for per-16 sub-scale formats: IQ2_S, Q6_K),
  one MMA per X limb (3 per k-step), int32 per sub-block, int64/int128 combine.
- **Decode to an int8 code tile in shared memory, double-buffered**; X tiles via `cp.async` 4 stages
  and `ldmatrix` (Strata's `moe_fused_iq.cu` pattern). Grids in shared memory.
- Same exact sums as §7.2, so wide and narrow return the same bits.
- Cost (model): 3 IMMA per 32 K-elements per 8 columns, against 4 for FN's W4A16 wide route (2 MMAs
  per 16 elements with the `[8·c2 | c2]` packing). So it is no more expensive than the W4A16 route
  recipe C already runs.

### 7.4 CPU expert engine (i9-13900K: AVX2 + AVX-VNNI, no AVX-512)

- One engine per format pair under `cpu/ggml_expert.{h,cpp}`, built `/fp:precise`, with ISA variants
  scalar (oracle), AVX2 and AVX-VNNI. AVX-512 VNNI is optional for other machines.
- **Products**: `vpdpbusd` (AVX-VNNI) with unsigned X lo/mid bytes against signed codes, and the
  biased-code trick for the signed hi limb (FN §16.2.1). That gives the same P_s as the GPU.
- **Grid decode**: IQ2_S/IQ3_* via scalar index loads packed with `_mm256_set_epi64x` (`vpgatherdd`
  measured 0.9 % slower end to end on the i9-13900K, MTP 4 + n-gram 7 at 4K, 2026-10-16, though
  Strata measured gather 0.34 → 0.19 ms per IQ3_S row on a Raptor Lake P-core); signs via
  `pshufb + and + cmpeq + vpsignb` (ggml `bit_selector`). IQ4_NL/IQ4_XS via `vpshufb` on `kvalues`;
  Q2_0 via shifts and `vpsubb`.
- **Multi-column**: unpack a 16-row group once into int8 codes in L1 (ik_llama.cpp #515), then
  `vpdpbusd` over all columns (Strata: 2.0-2.4× at 3 tokens).
- Team, phases, miss service and fetch channel unchanged (`cpu/expert_team.cpp`, `miss_service.cpp`).
  `CpuExpertJob` carries the record geometry. `cpu_rates.cpp` measures each format pair at startup
  (the decode-divisor input).

### 7.5 Residency with seven record sizes

Today the frame pool, staging slots, `ExpertStore`, host tier, SSD tier, VRAM sizing and the PCIe
divisor all use one `record_stride` (`program_impl.h:543-561, 750-792`, `load.cpp:540`,
`expert_store.cpp`, `host_expert_tier.cpp`, `qwen4_exp_instance.cpp:234`).

- **Per-layer geometry** everywhere a stride is read. The divisor `d = floor(1.65 + C·t)` uses that
  layer's `t = record_bytes/link`.
- **VRAM: size classes, one global LFRU.** The VMM arena is cut into one slab per record size
  (seven). Each slab's capacity is set at startup from the expert-state file (or the route-trace
  profile) in proportion to that class's byte-weighted demand. Eviction picks the lowest LFRU score
  within the needed class; at epoch boundaries a slab may cede whole 2 MiB granules to another class
  when the score gap exceeds a threshold. FN's rule that hard per-layer quotas lose (L21) concerns
  per-layer quotas; classes group 1-15 layers, and M0 replays three variants before freezing:
  - (a) one max-size frame (2,662,400 B), wasting 23 %;
  - (b) static classes;
  - (c) classes with rebalancing.
- **Host and SSD tiers**: records are contiguous per layer in the pinned arena and the artifact, each
  layer its own segment. The SSD tier reads in place (FN L12).
- **Staging**: 64 slots of the maximum record size (170 MB).
- `tools/expert_cache_replay/host_tier.py` replaces `RECORD_BYTES = 2_764_800` with per-layer sizes
  read from the artifact, and gains `--frame-model max|classes|rebalance`.

---

## 8. Dense path

### 8.1 Linear routes

New folders `src/ops/linear/{q4k,q5k,q6k,iq4xs,iq4nl}/`, each with `*_dispatch.cpp`, `*_shapes.h`
and `shapes/nN_kK.cu`, registered in `linear.cpp`'s `QType` switch (the FP8-row folder is the
template). The shapes that occur:

| Shape (n × k) | Classes | Formats |
|---|---|---|
| 10240 × 2560 | GDN `attn_qkv` | Q6_K, Q5_K, Q4_K, IQ4_XS |
| 6144 × 2560 | GDN `attn_gate` | Q6_K, Q4_K, IQ4_XS, Q5_K |
| 2560 × 6144 | `ssm_out`, `attn_output` | Q6_K, Q5_K, Q4_K |
| 12288 × 2560 | QSA `attn_q` (q + gate) | Q6_K, Q5_K, IQ4_XS, Q4_K |
| 512 × 2560 | `attn_k`, `attn_v` | Q6_K, IQ4_XS, Q5_K |
| 640 × 2560 | shared gate/up | Q4_K, IQ4_XS, Q6_K, Q5_K |
| 2560 × 640 | shared down | IQ4_NL (Q8_0 existing) |
| 248320 × 2560 | `lm_head` | Q6_K (`projection_fp32`) |

Each shape has a SIMT route family for T = 1..8 (column-invariant, exact §6.2) and a tensor-core
route for prefill, with tiles from a sweep bench (`bench/ops/ggml_flash_next_sweep.cu`, modelled on
`fp8_flash_next_sweep.cu`). Kernels use PDL with weight loads before `griddepcontrol.wait`
(already used for Flash-Next and Strata) and evict-first weight streams (FN L18).

### 8.2 Fusion across formats

Today the converter groups q/k/v/z (GDN), QSA q/gate/k/v/index and shared gate+up into parents and
splits rows after (`recipe.group`, `ops::split_rows`, `execution/forward.cpp`). In this file, members
of a group often differ in format (e.g. layer 0: `attn_qkv` Q6_K, `attn_gate` Q4_K).

- **Bring-up (M2)**: separate launches, one per member; up to ~84 more launches per token (model).
- **Optimized (M5)**: a **segmented linear**, one launch whose CTAs are assigned to segments by a
  prefix table, each segment with its own format and pointer, writing into one output with the same
  row offsets `split_rows` uses. Per-segment code paths are the per-format SIMT tiles of §8.1. A
  graph captures one segmented launch per group. Gate: adopted if the per-token time saved is
  ≥ 2 % at C = 1.
- Same-format groups stay single parents.

### 8.3 BF16 classes

HC mixers, routers, indexer, GDN α/β and PLE key/value use the existing BF16 routes. That includes
recipe A's fused hyper-connection mixer with the BF16 codec, measured ~18 % slower decode than
recipe B's 8-bit dense overall (FN §6.1). The HC mixers are ~1.26 GB per token in BF16 (model), the
one class where this artifact reads more than recipe B (§11.2).

### 8.4 GDN, QSA, norms

- **GDN A-direct**: the gate `g = A · softplus(a + dt_bias)` with A read from `ssm_a`. The kernel
  drops its `−exp(A_log)` (llama.cpp computes exactly this). A per-artifact role selects the variant.
- **GDN output order**: the gated-norm output store writes V head h to slot π(h) (π = ggml's tiled
  order, defined by `reorder_linear_attention_v_heads`). That is an index change in the store, at no
  cost.
- QSA, indexer and HC Ops are unchanged. Norm variants (FP32 gamma, direct gamma) are added only for
  the tensors §5.3 cannot narrow exactly.

### 8.5 Token embedding

Host-resident as today (FN §6.3). The engine worker gathers IQ4_XS rows (136 B per 256 values,
1,360 B per token) and decodes them exactly. This adds ~1 µs per token on the host (model), already
off the GPU's critical path.

---

## 9. PLE n-gram table: IQ4_NL rows and RAM residency

### 9.1 Volume

- New `NgramTableConfig.format = iq4_nl` with `row_bytes = 90` (5 blocks of fp16 d + 16 nibble
  bytes); 45 rows per 4 KiB block (4,050 B used, 1.1 % padding): 7,111,146 blocks, 29.13 GB. The
  converter copies shard 2's rows verbatim. The header carries the format and the volume id.
  `config.cpp:78-112` accepts both formats.
- Nibble order is llama.cpp's (elements 0-15 low nibbles, 16-31 high), decoded by an exact oracle
  and the kernel.
- `ple_embed` gains an IQ4_NL variant (§6.3); `ngram_scale` disappears for this format.
- The converter's `--ngram-reuse` works by volume id; `verify_artifact` compares every row with the
  GGUF.

### 9.2 Residency modes

`--ngram-residency auto|ram|volume` (CLI, serve, `include/infernix/types.h`):

- **volume** is today's path: unbuffered reads, a 2^20-row cache, gated verify reads (FN §12.3).
  It works with 90-byte rows unchanged apart from the geometry.
- **ram** holds the whole table in host memory:
  - The volume is read at load into a **locked** host arena with 16 overlapped 8 MiB unbuffered
    reads (the `ReadOnlyFile::read_direct` path used for experts): ~4-5 s from the Optane `G:` at
    ~7 GB/s (model).
  - `NgramVolume::read_rows` becomes a gather of 16 × 90 B per column. The 4 KiB padding is kept, so
    a row's address is `block · 4096 + (r mod 45) · 90`.
  - The arena is **pinned** (`cudaHostRegister`) because step 2 below needs it. Locked memory and
    pinned memory cost the same RAM.
- **auto** chooses ram when the ledger fits (§9.4) after experts are fully resident. Otherwise it
  uses volume, with a larger set-associative row cache sized from the remaining headroom
  (`--ngram-cache-mib`, default 160 MiB as today).

### 9.3 What RAM residency changes on the decode path

- **Step 1 (M6).** Rows are always available, so the host gathers them **before** every launch,
  including verify rounds. The anchor and copy-proposal columns are known before drafting; draft
  columns are gathered in ~1-2 µs after drafting. The gate machinery stays for draft columns but
  never waits on I/O.
  - What this removes: NVMe latency on decode misses (S4b), and, for prefill, the cold-row reads. S1
    measured these at ~1.46 s for a cold 16K prompt (FN §12.3, 2026-10-05). How much of that is still
    exposed after the later prefill work is measured in M0; it is not assumed.
- **Step 2 (option, M8 gate).** Device-side PLE:
  - The `ngram_row_ids` hash (exact int64, FN §16.1) runs on the GPU;
  - `ple_embed` reads the 16 rows of each column from the pinned table over UVA: 1,440 B per
    column, latency-bound at ~1-2 µs within layer 0.
  - This removes the host from the PLE path and lets the verify graph launch without waiting for the
    drafter's rows. It is FN's planned "device-driven PLE" (M6 there).
  - Zero-copy is wrong for bulk transfers (FN L3) but right for a few cache lines per column.
  - Gate: adopted when S0-style events show ≥ 0.1 ms of after-draft GPU idle per verify round that
    step 2 removes.

### 9.4 Host RAM ledger

95.8 GiB visible (measured, GLM §3.1). Model:

| Item | Volume mode | RAM mode |
|---|---:|---:|
| Expert records, pinned (24,576 × 2.05 MB with padding) | 46.8 GiB | 46.8 GiB |
| n-gram table | 0.16 GiB (row cache, pageable) | 26.8 GiB (pinned) |
| Embedding (IQ4_XS) | 0.31 GiB | 0.31 GiB |
| Vision tower (`--vision-offload`) | 0.84 GiB | 0.84 GiB |
| Prefix-cache Host tier (default) | 4.0 GiB | 4.0 GiB |
| Program allowance, lock overhead, load staging | ~0.6 GiB | ~0.6 GiB |
| Headroom (`--ram-headroom-mib` default) | 2.0 GiB | 2.0 GiB |
| **Total** | **~54.5 GiB** | **~81.2 GiB** |

Recipe B needs 69.3 GiB today (GLM §3.1). The RAM mode needs ~12 GiB more than recipe B. It fits a
quiet machine (95.8 GiB visible). It does **not** fit the 79 GiB available measured with other
sessions running (GLM §3.1), unless the prefix tier shrinks to 1-2 GiB or vision offload is off.

`HostMemoryDemand` (`models/qwen4_exp/memory_plan.h`) gains an `ngram_table` term placed after
experts. `auto` never displaces experts to the SSD tier to make room for the table. The pinned ceiling
on Windows is measured first (GLM R0: 67 % pinned works today; this plan pins ~78.5 GiB, 82 %,
counting the prefix Host tier).

---

## 10. MTP drafter from Qwen's checkpoint

### 10.1 Source

- No local checkpoint has the MTP head: the NVIDIA checkpoint the current recipes read is gone, and
  no BF16/FP8 Qwen checkpoint is local (measured).
- Fetch only what is needed from **`Qwen/Qwen3.8-Flash-Next-FP8`**, pinned to one revision with
  SHA256 per object:
  - the 31 dense `mtp.*` tensors (BF16);
  - the 512 MTP experts `mtp.layers.0.mlp.experts.{e}.{gate,up,down}_proj.weight` plus
    `weight_scale_inv` (block FP8, 2.52 GB; upstream index). These are the tensors and the form the
    existing converter already decodes exactly from NVIDIA's checkpoint (`fp8_block_values`,
    `dequantize_fp8_block128`, FN §6.1);
  - for §5.3's checks, the BF16 dense tensors whose GGUF counterparts carry transforms (norms,
    `A_log`, conv). Total ~3-6 GB.
- The fetch is a small tool, `tools/convert/fetch_tensors.py`: it reads safetensors headers, then
  HTTP range-reads the named tensors into a local partial checkpoint directory with `config.json`.
  Strata's `tools/mtp_fetch.py` does this from the BF16 repository (5.2 GB).
- Alternative source: Strata's local `E:\Strata\Strata-data\mtp\tensors\*.bin` (BF16, from the BF16
  repository at a pinned revision). Used only if the fetch is unavailable; its provenance is
  recorded.

### 10.2 Conversion and runtime

- The builder accepts a **drafter source** (`--drafter-source DIR`) separate from the base. Today
  `_Qwen4ExpBuilder.validate_source` refuses alternate sources (`tools/convert/qwen4_exp.py:320`);
  this narrows that refusal to the base model.
- The drafter recipe is `_flash_next_drafter` unchanged (`official_recipes.py:278`): experts
  `q4_g64_fp16` (`grouped_mse`, 1.34 GB resident), dense `q8_g32_fp16`, router and shared gate BF16.
- **Proposal head**: the 131,072 most frequent `lm_head` rows, `q4_g64_fp16`. They are taken from the
  GGUF's Q6_K `output` decoded exactly, so the draft head approximates this target's head, not
  Qwen's BF16 one.
- MTP shares the token embedding and `lm_head` (`mtp_use_dedicated_embeddings: false`). It uses the
  IQ4_XS embedding gather and the Q6_K head at runtime.
- Draft policy unchanged (`choose_draft_length`, FN §11.3).
- Expected acceptance (estimated): Strata measured 0.85-0.89 step-1..3 offline greedy acceptance
  with a 2.25-bit MTP drafter against this exact IQ3_S target. FN measured 75.6-81.8 % sampled on
  AIME with q4 experts against NVFP4. M7 measures it here.

---

## 11. Performance model

### 11.1 VRAM frames

Recipe B holds ~9,400 frames of 2.76 MB (~26.0 GB). This artifact's GPU dense bytes are ~4.2 GB
against recipe B's ~4.4 GB (§11.2), so ~26.2 GB is available for experts (model):

| Frame model | Frames | Share of expert bytes |
|---|---:|---:|
| Ideal byte packing | ~12,800 | 52 % |
| (b)/(c) size classes, 2-5 % fragmentation | ~12,250-12,600 | 50-51 % |
| (a) one max-size frame | ~9,840 | 40 % |

For comparison, Strata reports 10,733-12,331 slots on its 5090 runs (upstream).

### 11.2 Dense bytes per token (model, from §2.2 shapes and bits per weight)

| Class | This artifact | Recipe B |
|---|---:|---:|
| GDN projections (36 layers) | 1.58 GB | ~2.07 GB (8-bit) |
| QSA projections and indexer (12) | 0.48 GB | ~0.60 GB |
| Shared experts (48) | 0.14 GB | ~0.24 GB |
| Routers (BF16) | 0.13 GB | 0.13 GB |
| HC mixers | **1.27 GB (BF16)** | ~0.67 GB (Q8) |
| lm_head | 0.52 GB (Q6_K) | ~0.67 GB (Q8) |
| **Total** | **~4.2 GB** | **~4.4 GB** |

### 11.3 Decode (C = 1)

- LFRU replay on Flash-Next (FN §9.3) gives miss rates of 18.4 / 5.2 / 1.4 % of activations at
  16.3 / 32.6 / 48.8 % capacity. At ~50 % capacity against recipe B's ~38 %, the replay model
  predicts **about one third of recipe B's misses**, each 26 % smaller. Live cold-start rates are
  higher than replay (recipe B tg512 hit rate 88 % measured), but the ratio is the claim, and M0
  replays it on real traces with real sizes.
- Hit-expert bytes: 480 × 2.04 MB = 0.98 GB per token, against 1.33 GB (−0.2 ms at 1.55 TB/s).
- The dense path is unchanged within ±5 %, provided the k-quant/i-quant SIMT routes reach the Q8/FP8
  routes' DRAM efficiency. **This is the largest risk** (§15.1).
- **Estimate**: plain decode 135-160 tok/s against 132.2. MTP gains as for recipe B, since verify
  windows add no misses per accepted token (FN L8).
- The CPU miss rate per byte for i-quants is lower than W4A4's (grid decode); with ~1/3 of the misses
  it matters less. It is measured by `cpu_rates.cpp`.

### 11.4 Prefill

- **Link**: every expert not in VRAM crosses PCIe once per walk span. That is ~0.49 × 50 GB = 24.5 GB
  at 27.6 GB/s = 0.9 s per span, against ~1.5 s for recipe B (model).
- **Compute**: exact IMMA expert GEMMs at ≤ the W4A16 wide-route cost (§7.3). Dense prefill on BF16
  tensor cores with an FP32 epilogue per k16 (+~25 % FMA work against a pure BF16 GEMM, model).
- **Estimate**: pp131072 within ±10 % of recipe B's 5,767 tok/s. Short prompts (one walk span) gain
  the most, since streaming is ~40 % shorter.
- **RAM-resident n-gram table**: removes the cold-row reads of each prompt (up to ~1.46 s for 16K
  tokens at S1's rates; current exposure measured in M0).

---

## 12. Qualification

Per [Op development](op-development.md) §6; claims follow [model quality](model-quality.md).

### 12.1 Codecs and Ops

| Item | Oracle | Criterion | Coverage |
|---|---|---|---|
| Each `ggml_*` format | Exact decode in Python (`Fraction`/FP64) from the GGUF block bytes; cross-checked against `gguf-py` `dequantize` (no Q2_0 there: independent oracle) | Exact, every element of every tensor in the artifact; round-trip to GGUF bytes | All 1,223 tensors |
| `offloaded_sparse_moe` ggml family | §6.1 exact rationals, two roundings | **Bit-equal**, every output | 7 format pairs × columns 1..8, 9, 16, 476 × routes (narrow, staged, wide, CPU scalar/AVX2/AVX-VNNI) × random residency mixes; real geometry; non-finite columns |
| Linear (new formats) | FP64 over the exact decoded weights | SIMT: bit-equal to §6.2's exact contract; tensor-core: normwise with gross pointwise cap (FN rule) | Every (format, shape) of §8.1; T = 1..8 boundaries ± 1, prefill tiles |
| Segmented linear | Composition of per-segment oracles | Bit-equal to separate launches | Every group of §8.2 |
| `ple_embed` IQ4_NL | Exact decode | Bit-equal | Random rows + boundary rows |
| `NgramVolume` RAM backend | Volume backend | Byte-equal rows; counters | Closed-form row oracle (existing test) |
| GDN A-direct, output permutation | FP64 / exact index map | Normwise / exact | Real shapes |

### 12.2 Model and system

- `verify_artifact` (§1.3) over the whole artifact and volume.
- Greedy invariance: all-GPU vs all-CPU vs random placements, C = 1..8, MTP on/off
  (`test_forward_real`, `vram_pressure_real`, `expert_fault_real` with the new artifact).
- Quality: the §16.5 KL protocol against recipe B; llama.cpp b10221 (local binary) on the same GGUF
  as an external sanity check (top-1 agreement and KL, not bit-equality, because of Q8_1).
- Performance: tg512 ABBA, the agentic A/B (`bench/agentic_ab`), pp131072, a cold 16K prompt with
  and without RAM residency, and Strata on this machine with the same file, all against recipe B.
  Kernel changes are judged by per-round kernel time, not greedy throughput (FN L20).

---

## 13. Milestones

Estimates are focused engineering time, ±50 %.

| M | Content | Exit gate | Est. |
|---|---|---|---|
| **M0** facts and models | Download shard 1 (pinned, SHA256); full header dump; confirm §2. Strata on this machine (tail skip on/off), recipe B re-measured the same day. Replay with per-layer sizes (`host_tier.py`), three frame models (§7.5). Current exposure of n-gram reads in cold prefill. Windows pinned ceiling at ~78.5 GiB. | §11 table replaced by measurements; frame model chosen | 3 d |
| **M1** formats and converter | `ggml_*` codecs and planes; raw-block GGUF reader; `qwen4exp` name map and transforms; `ggml_expert_v1` records; IQ4_NL volume; `verify_artifact` extension | Bit-exact verify of the whole artifact and volume | 5-7 d |
| **M2** correctness bring-up | Correctness-first kernels: SIMT dense for every (format, shape), generic narrow expert kernel, scalar CPU engine; GDN A-direct and permutation; IQ4_NL `ple_embed`; IQ4_XS embedding; per-layer strides with one max-size frame | Model runs end to end; greedy invariance; KL measured (first quality read: stop here if KL is unacceptable) | 6-8 d |
| **M3** expert kernels | Tuned narrow (per format, smem grids), wide IMMA, AVX2/AVX-VNNI CPU engines; `cpu_rates` per format | Bit-equality matrix (§12.1); each kernel ≤ 110 % of byte-time | 8-10 d |
| **M4** residency | Size-class frames (per M0), per-layer stride in store/host/SSD tiers, divisor, staging, replay tool, expert state | Tier tests pass; replay ≈ live within FN's tolerance | 4-5 d |
| **M5** dense performance | Tuned SIMT and tensor-core tiles per shape; segmented linear; Q6_K `lm_head` projection; PDL chain audit | Decode ≥ recipe B, or attributed by nsys/ncu | 5-7 d |
| **M6** n-gram RAM residency | `--ngram-residency`, ledger term, RAM backend, pre-launch gather | RAM = volume rows; cold 16K prompt measured | 2-3 d |
| **M7** MTP | `fetch_tensors.py`, `--drafter-source`, drafter and proposal head; acceptance and draft-policy check | MTP greedy = plain; acceptance and tok/s reported | 2-3 d |
| **M8** options and vision | Device-side PLE (gated), exact IMMA dense prefill (gated), `mmproj` vision import | Each option by its gate | 3-5 d |
| **M9** documentation | §14; model card | Links and `git diff --check` | 1 d |

Total ≈ 40-55 days. The sequence front-loads the quality read (M2) before kernel investment.

---

## 14. Documentation and authority changes

- [tensor formats](tensor-formats.md): the `ggml_*` formats, plane layouts, `ggml_expert_v1`.
- [Flash-Next design](qwen3_8-flash-next-design.md): the ggml expert family (§16.2.2: the exact
  arithmetic of §6.1), size-class frames, n-gram residency modes, IQ4_NL volume.
- [weight conversion](../weight-conversion.md) and [model quality](model-quality.md) §4.4 (replace
  "neither path is built").
- User guide `docs/qwen3_8-flash-next.md`, `docs/cli.md`, `docs/serving.md`: `--ngram-residency`,
  `--ngram-cache-mib`, the new artifact.
- `model-cards/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix/`: provenance (ISTA revision, Qwen MTP
  revision), the license (Apache-2.0), ISTA's GSQ/RCO and imatrix credits.
- `docs/README.md` row for this plan, removed with it.

---

## 15. Risks, options and rejected alternatives

### 15.1 Risks

| Risk | Effect | Mitigation |
|---|---|---|
| k-quant/i-quant SIMT dense routes below Q8/FP8 DRAM efficiency | Decode below recipe B even with fewer misses | Plane layouts (§5.2), per-shape tiles, M5 attribution; segmented launches |
| CPU grid-decode cost (IQ2_S/IQ3_*) | Lower CPU miss rate; divisor shifts to PCIe | Per-core gather choice, unpack-once multi-column; the divisor already adapts to measured rates |
| Size-class fragmentation | Fewer effective frames | M0 replay picks the frame model; fall back to (a) |
| Windows pinned ceiling below ~78.5 GiB | RAM mode unavailable | `auto` falls back to volume plus a larger row cache |
| MTP acceptance lower against this target | Smaller MTP gain | Measured in M7; draft policy adapts |
| GSQ-RCO quality below recipe B | Artifact not recommended | M2 quality read before kernel work |

### 15.2 Options (each behind its own gate)

- `IQ3_S-dense8` variant: the HC mixers, PLE key/value and router re-quantized to recipe B's 8-bit
  classes (−0.6 GB per token). Not bit-exact, so it is a separate recipe name, qualified by §16.5.
- Exact IMMA dense prefill (§6.2).
- Device-side PLE (§9.3).
- A route-tail skip à la Strata is **not** an option: it changes outputs.

### 15.3 Rejected

- **Loading the GGUF in place**: `.infernix` is the only C++ artifact and runtime repacking is
  forbidden (AGENTS). The blocks' 2-byte alignment would also slow every kernel.
- **Q8_1/Q8_K activations** (llama.cpp, Strata): less accurate than A16, and Strata shows the two
  forms disagree across placements.
- **Dequantizing to FP32 and re-quantizing** (the current `gguf.py` route): a second quantization,
  not bit-exact.
- **One max-size frame without measurement**: 23 % fewer frames (§11.1).
- **Expert deferral, SER-style expert dropping**: change the mathematics.

---

## 16. Sources

- ISTA-DASLab GGUF repository and `tensor-allocation/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.rco-allocation.txt` (read 2026-10-10).
- llama.cpp `3cf03257` (vendored in `E:\Strata\Strata\third_party\llama.cpp`): `ggml-common.h`,
  `ggml-cuda/{vecdotq.cuh,mmq.cuh,dequantize.cuh}`, `conversion/{qwen.py,qwen4exp.py}`,
  `src/models/qwen4exp.cpp`; PRs #24448, #25603, #25707, #26348, #27742, #27794, #29887.
- Strata `E:\Strata\Strata` (HEAD `61b3fb5d`): `tools/iq_pack.py`, `src/kernels/cuda/iq_kernels.cu`,
  `moe_fused_iq.cu`, `native_mmvq.cu`, `src/kernels/cpu/iq_avx2.cpp`, `src/ngram/ple_reader.cpp`,
  `tools/mtp_fetch.py`, `bench/results/`.
- ik_llama.cpp discussions #164, PRs #515, #1446, #1467, #1474, #1482.
- Qwen/Qwen3.8-Flash-Next-FP8 `config.json` and `model.safetensors.index.json` (read 2026-10-10).
