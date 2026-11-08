# Output quality against the exact model

How closely each engine's next-token distributions match the exact computation of the same weights,
measured with llama.cpp's KL-divergence statistics on WikiText-2. The [README](../README.md#quality)
summarises the results; this page gives them in full with their intervals, defines each statistic,
records every engine's version and settings, and describes how to repeat the measurement.

Measured 2026-11-05/06 (FP4-Local 2026-11-08) on one RTX 5090 (32 GB, PCIe Gen5 x8), Core i9-13900K,
96 GB DDR5, Windows 11.
NInfer and vLLM ran in WSL2 (Ubuntu) on the same machine; every other engine ran natively on Windows.

## Results

Brackets are 95 % intervals: a bootstrap over chunks (10,000 resamples), or, for llama.cpp's rows,
llama-perplexity's own ± (it gives no interval for the 99.9 % KLD). "× rounding" is the row's mean
KLD divided by honest rounding's.

### Qwen3.8-27B NVFP4

All four engines run one model: NInfer's official artifact
[neroued/Qwen3.8-27B-nvfp4-NInfer](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) (Unsloth's
NVFP4/FP8 weights; SHA256 `74d2c57145e6ff11d1d2faa79594477f9bc903a611af1fb20218189fbbb77d82`, the file
TensorSharp pins). vLLM cannot load the artifact, so it loads
[unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) with its embedding replaced
by the artifact's ([below](#vllms-checkpoint)). BF16 KV for every engine. The reference's perplexity on
the scored positions is 7.0353.

| Path | Engine | Mean KLD | × rounding | KLD 99.9 % | Top-1 agreement | PPL vs reference |
|---|---|---|---:|---|---|---|
| — | Honest rounding (noise twin, seed 1) | 0.00160 [0.00141, 0.00186] | 1.0 | 0.046 [0.036, 0.063] | 98.15 % [98.07, 98.22] | +0.07 % [+0.03, +0.10] |
| Prompt | Infernix (default) | 0.01690 [0.01604, 0.01798] | 10.5 | 0.655 [0.588, 0.786] | 94.47 % [94.34, 94.60] | +0.69 % [+0.56, +0.82] |
| Prompt | NInfer `a754fcf3b` | 0.01660 [0.01588, 0.01753] | 10.4 | 0.660 [0.561, 0.794] | 94.48 % [94.34, 94.61] | +0.67 % [+0.50, +0.84] |
| Prompt | vLLM 0.33.0 | 0.01796 [0.01712, 0.01898] | 11.2 | 0.685 [0.598, 0.802] | 94.28 % [94.14, 94.41] | +0.70 % [+0.55, +0.84] |
| Prompt | TensorSharp `0c45b34c` | 0.01695 [0.01612, 0.01791] | 10.6 | 0.645 [0.552, 0.722] | 94.42 % [94.28, 94.55] | +0.76 % [+0.62, +0.90] |
| Decode | Infernix (default: DFlash2 + n-gram drafts) | 0.00442 [0.00393, 0.00507] | 2.76 | 0.197 [0.165, 0.257] | 97.38 % [97.29, 97.47] | −0.10 % [−0.20, +0.00] |
| Decode | NInfer (MTP, 3 drafts) | 0.00439 [0.00390, 0.00502] | 2.74 | 0.201 [0.163, 0.282] | 97.52 % [97.44, 97.61] | −0.02 % [−0.10, +0.06] |
| Decode | vLLM (MTP, 3 drafts) | 0.01783 [0.01695, 0.01890] | 11.1 | 0.647 [0.583, 0.773] | 94.23 % [94.09, 94.36] | +0.79 % [+0.64, +0.94] |
| Decode | TensorSharp (DFlash2, 7 drafts) | 0.01504 [0.01427, 0.01605] | 9.4 | 0.611 [0.540, 0.706] | 94.94 % [94.82, 95.06] | +0.60 % [+0.45, +0.75] |

### Qwen3.8-Flash-Next IQ3_S

All engines run one GGUF's weights:
[ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
IQ3_S. llama.cpp loads the GGUF itself, Infernix its import
[Wallawalla47/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix)
(with its n-gram volume), Strata and Basalt their own packs built from it. KV is 8-bit for every engine:
INT8 for Infernix, Strata and Basalt, Q8_0 for llama.cpp. The reference's perplexity on the scored
positions is 4.6465.

| Path | Engine | Mean KLD | × rounding | KLD 99.9 % | Top-1 agreement | PPL vs reference |
|---|---|---|---:|---|---|---|
| — | Honest rounding (noise twin, seed 1) | 0.01601 [0.01545, 0.01659] | 1.00 | 0.577 [0.509, 0.636] | 95.36 % [95.21, 95.50] | +0.04 % [−0.08, +0.15] |
| Prompt | Infernix (default) | 0.01465 [0.01409, 0.01522] | 0.91 | 0.496 [0.458, 0.545] | 95.59 % [95.45, 95.74] | +0.08 % [−0.03, +0.19] |
| Prompt | llama.cpp b12128 | 0.01505 [0.01484, 0.01527] | 0.94 | 0.499 | 95.43 % [95.32, 95.54] | −0.03 % [−0.13, +0.08] |
| Decode | Infernix (default: DFlash2 drafts) | 0.01443 [0.01392, 0.01495] | 0.90 | 0.472 [0.435, 0.522] | 95.59 % [95.45, 95.73] | −0.04 % [−0.15, +0.07] |
| Decode | llama.cpp (forced 4-token windows) | 0.01514 [0.01492, 0.01537] | 0.95 | 0.508 | 95.53 % [95.42, 95.64] | +0.05 % [−0.06, +0.16] |
| Decode | Basalt v0.2.1 (verify windows of 4) | 0.01922 [0.01859, 0.01986] | 1.20 | 0.579 [0.522, 0.632] | 95.29 % [95.14, 95.43] | +0.47 % [+0.33, +0.60] |
| Decode | Strata v0.2.11 (tail skip on, its default) | 0.02314 [0.02225, 0.02408] | 1.45 | 0.803 [0.725, 0.881] | 94.44 % [94.26, 94.60] | −0.00 % [−0.16, +0.14] |

### Qwen3.8-Flash-Next NVIDIA FP4-Local

Infernix only, on its conversion of NVIDIA's quantization-aware-distilled FP4 checkpoint
[Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix)
(NVIDIA's NVFP4 experts with 16-bit activations and MXFP8 dense layers, bit-exact; its n-gram volume),
INT8 KV. The first 200 chunks only (51,000 scored positions), so the intervals are wider than the
tables above. Besides the default route, the reproducible route (`--reproducible`: no fast paths) and
each default fast path alone were measured. The reference's perplexity on the scored positions is
4.1642.

| Path | Engine | Mean KLD | × rounding | KLD 99.9 % | Top-1 agreement | PPL vs reference |
|---|---|---|---:|---|---|---|
| — | Honest rounding (noise twin, seed 1) | 0.01475 [0.01388, 0.01566] | 1.00 | 0.526 [0.465, 0.610] | 95.60 % [95.35, 95.85] | +0.15 % [−0.03, +0.33] |
| Prompt | Infernix (`--reproducible`, BF16 KV) | 0.01310 [0.01235, 0.01387] | 0.89 | 0.401 [0.375, 0.481] | 96.00 % [95.76, 96.24] | −0.00 % [−0.20, +0.19] |
| Prompt | Infernix (`--reproducible`) | 0.01313 [0.01238, 0.01392] | 0.89 | 0.454 [0.381, 0.519] | 95.98 % [95.74, 96.21] | −0.02 % [−0.22, +0.17] |
| Prompt | Infernix (default) | 0.01364 [0.01283, 0.01446] | 0.92 | 0.482 [0.433, 0.539] | 96.01 % [95.75, 96.26] | +0.07 % [−0.12, +0.26] |
| Decode | Infernix (`--reproducible`, DFlash2 drafts) | 0.01318 [0.01241, 0.01396] | 0.89 | 0.431 [0.370, 0.508] | 95.98 % [95.75, 96.21] | −0.01 % [−0.18, +0.16] |
| Decode | Infernix (default: DFlash2 drafts) | 0.01323 [0.01248, 0.01403] | 0.90 | 0.457 [0.410, 0.514] | 96.04 % [95.78, 96.29] | +0.09 % [−0.07, +0.27] |

The default route's fast paths on this model, each alone against the same route without it (paired on
the same positions; "worst" divides the upper bound of each statistic's change by honest rounding's
value and takes the largest):

| Fast path | Path | Δ Mean KLD | Δ KLD 99.9 % | Δ Top-1 agreement | Worst / honest rounding |
|---|---|---|---|---|---:|
| `expert-a4-residual` | prompt | +0.00060 [+0.00025, +0.00096] | +0.012 [−0.062, +0.096] | −0.16 pt [−0.34, +0.02] | 0.18 |
| `screened-lm-head` | decode | +0.00000 [−0.00000, +0.00000] | +0.000 [−0.000, +0.000] | +0.00 pt [+0.00, +0.00] | 0.00 |

### Observations

- **27B prompt path.** All four engines sit at 10.4-11.2× honest rounding, with perplexity up
  0.67-0.76 %. Their intervals overlap, so this is the artifact's prompt path, not any one engine's:
  it computes with quantized activations (4-bit in the NVFP4 layers with the checkpoint's calibrated
  scales, 8-bit in the FP8 layers).
- **27B decode path.** Infernix and NInfer keep 16-bit activations in their decode and verify steps and
  fall to 2.7× honest rounding, with no measurable perplexity change. vLLM stays where its prompt path
  is (11.1×), consistent with 4-bit activations in its NVFP4 layers on every path. TensorSharp's
  verify quantizes the NVFP4 layers' inputs to 4 bits in any window of 4 or more rows, and its shipped
  configuration verifies 8 (7 drafts), hence 9.4×; its exact 16-bit verify module
  (`TS_Q35_NINFER_EXACT_VERIFY_PTX`) is an opt-in that its shipped configuration does not load.
- **IQ3_S floor.** Honest rounding is 10× higher here than on the 27B: a small rounding difference
  can change which experts the router picks, and the change then spreads. Infernix's and llama.cpp's
  defaults come in slightly *below* it (0.90-0.95×), most likely because rounding to nearest at the
  same precisions has about half the error variance of stochastic rounding.
- **Basalt** is 1.20× honest rounding, and its +0.47 % perplexity is the only IQ3_S change whose
  interval excludes zero. Basalt computes some parts of the model with fewer bits (the README's
  [IQ3_S comparison](../README.md#qwen38-flash-next-iq3_s-infernix-vs-strata-and-basalt) lists them).
- **Strata** is 1.45× honest rounding with no perplexity change. Its default route tail skip
  (`STRATA_ROUTE_TAIL_SKIP=7`: a missed expert that every token of a verify window routes at rank 7 or
  lower is not computed) skipped 3.9 million expert computations over the run.
- **FP4-Local** sits below honest rounding on both paths (0.89-0.92×), as IQ3_S does. Its two default
  fast paths are within rounding (at most 0.18× honest rounding), and its decode statistics are the
  same with DFlash2 drafting as with one-token decode (to every printed digit), as exact verification
  should leave them.
- **Perplexity alone would hide most of this.** Strata's distributions are 1.45× further from the
  exact model than honest rounding's, yet its perplexity matches the reference's to two decimals. See
  [PPL vs reference](#ppl-vs-reference).

## What the statistics mean

Every statistic compares, position by position, the engine's next-token distribution with the
reference's for the same preceding text.

- **The reference** is Infernix's lossless route on the same weights: every operation of the model
  correctly rounded at the model's own dtypes ([definition](maintainer/lossless-route.md)), BF16 KV,
  one pass per chunk. It is the model as specified, computed without error.
- **Honest rounding** (the noise twin) is the lossless route with every rounding made stochastic:
  each correctly rounded result goes up or down with probabilities set by the discarded fraction
  ([definition](maintainer/lossless-route.md#5-the-noise-twin)). It is an ideal implementation that is
  off by at most one unit in the last place at every rounding and by nothing else, so its distance
  from the reference is the model's own sensitivity to rounding. An engine near 1× computes as
  faithfully as rounding allows; 10× means ten times that distance.
- **Mean KLD** is the mean over positions of KL(reference ‖ engine): how much probability mass the
  engine moves, on average. Lower is better; 0 means identical distributions.
- **KLD 99.9 %** is the 99.9th percentile of the per-position KLD: how far the worst 0.1 % of
  positions (148 of 147,900) move.
- **Top-1 agreement** is the share of positions where the engine's most likely token is the
  reference's: how often greedy decoding would pick the same token.
- **PPL vs reference** is the engine's perplexity on the text as a change from the reference's:
  100 × (exp(mean(ln p_ref − ln p_engine)) − 1) %, with p the probability given to the text's actual
  next token, on the same positions.

### PPL vs reference

Perplexity measures how well a model predicts the text; here every engine runs the same weights, so
an engine's absolute perplexity is the reference's times the ratio above. The absolute values differ
in the third decimal, and their own intervals (±0.22 on the 27B) measure which text was sampled
rather than the engine, so this page quotes only the paired change. That change shows a systematic
shift: the 27B's quantized prompt path makes the text 0.7 % less likely on average. It cannot show
noise. Rounding errors raise some tokens' probabilities and lower others about equally, so they
cancel in the mean log-likelihood: Strata's IQ3_S row is 1.45× honest rounding by KLD and 0.00 % by
perplexity. KLD, its tail and top-1 agreement measure the distance from the intended distributions,
not whether the noise happened to favour this text, which is why llama.cpp's own quality comparisons
lead with them.

## How it was measured

### Text, chunks and positions

The WikiText-2 raw test set (`wiki.test.raw`), tokenised by each model's tokenizer and cut into 580
chunks of 512 tokens (FP4-Local: the first 200). As in `llama-perplexity --kl-divergence`, each chunk is scored on its second
half: the predictions of tokens 257-511 from positions 256-510, 255 per chunk, 147,900 in all. The
reference is written as llama.cpp's `_logits_` base file (`infernix-certify reference --format
llama-kld`), which holds the tokens and the reference's log-probabilities, so every engine reads the
same tokens and is scored against the same distributions.

### Logits

Every row is scored at BF16 precision: the reference writes BF16 logits; NInfer and vLLM produce BF16
logits; Infernix's, TensorSharp's, Strata's and Basalt's rows are rounded to BF16 by the scorer
(`infernix-certify llama-rows --round-logits bf16`), and llama.cpp's by its patch
(`LLAMA_KLD_ROUND_BF16`). Rounding logits to BF16 is itself a small KLD; applying it to every row keeps
the comparison fair.

### Prompt path

Each chunk is one prompt forward, and the scored rows are that forward's logits: what an engine
computes when it reads a prompt. Strata and Basalt compute no logits on their prompt paths, so they
have decode rows only.

### Decode path

Each chunk's first 257 tokens are one prompt, whose last row is the first scored row. The remaining
255 positions are generated in one request with the engine's own speculative decoding, the text's
tokens forced: a draft is accepted only when it equals the text's next token, every committed
position's row is the one the engine computed for the text's actual history, and rows computed
after a rejected draft are discarded and recomputed by the next window. This scores the kernels an
engine decodes with (decode steps and verify windows at its own window sizes) rather than its
prompt kernels. How each engine is forced:

| Engine | Drafting | Forcing |
|---|---|---|
| Infernix | its default drafter (27B: the artifact's DFlash2 plus n-gram copy drafts; IQ3_S and FP4-Local: the DFlash2 companion), automatic draft length | `infernix-certify decode`: each step's row is recorded, then the text token's logit is raised so the engine's own acceptance takes it |
| NInfer | MTP, 3 drafts | the same in its decision step (patch) |
| vLLM | MTP, 3 drafts | a `compute_logits` hook records each row and raises the text token's logit ([`vllm_rows.py`](../tools/quality/engines/vllm_rows.py)) |
| TensorSharp | the artifact's DFlash2, 7 drafts | a tool (patch) makes its speculative loop's own calls (draft, verify the window, commit) and accepts drafts that equal the text |
| llama.cpp | none in `llama-perplexity` | the patch feeds windows of 4 text tokens (a verify of 3 drafts that are all correct) |
| Basalt | verify windows of 4 | its own `ppl_gate` protocol, with the patch dumping each row |
| Strata | — | it computes logits only in its verify windows and sends any prompt longer than `--short-read` wholly through its batched path, so every token of the chunk is read through its verify windows (as its own quality tool reads them); the first half is not one batch and its MTP drafter is idle |

Infernix's 27B decode rows are recorded on eager rounds (`--no-cuda-graph`); its first 40 chunks with
CUDA graphs gave bit-identical rows.

### Scoring and intervals

Rows go to `infernix-certify llama-rows`, which computes llama-perplexity's per-position statistics
against the base file (`apps/certify/llama_kld.h`); llama.cpp's rows come from llama-perplexity
itself. [`tools/quality/llama_kld.py`](../tools/quality/llama_kld.py) turns the per-position values
into the published statistics, with 95 % intervals from 10,000 bootstrap resamples of whole chunks
(seed 20261025). [`tools/quality/engine_compare.py`](../tools/quality/engine_compare.py) runs the
steps in order, resumes after a failure, and writes `report.md` and `report.json`.

## Engines

Each engine ran its latest version on 2026-11-05 with its own default or recommended configuration;
the patches only take logits out (and, for TensorSharp, compile on Windows).

| Engine | Version | Platform | Settings |
|---|---|---|---|
| Infernix | `main` `68cb6992` with this comparison's tooling (FP4-Local: `main` `3279ceb4`) | Windows, CUDA 13.4 | default route; 27B BF16 KV; IQ3_S and FP4-Local INT8 KV and the n-gram volume; decode `--spec dflash2 --draft-tokens auto --lm-head-draft` (27B also `--ngram-draft-tokens auto`), one request |
| NInfer | `master` `a754fcf3b` + [patch](../bench/engine_quality/patches/ninfer-a754fcf3b.patch) | WSL2, CUDA 13.4 | prompt: its CausalScoring route; decode: its Generation route, `--spec mtp --draft-tokens 3`, one request, no CUDA graphs; BF16 KV, BF16 logits |
| vLLM | 0.33.0 (pip, torch 2.14, cu130) | WSL2 | its defaults (KV auto = BF16), `enforce_eager`, prefix caching off; decode `speculative_config={"method": "mtp", "num_speculative_tokens": 3}`, one request |
| TensorSharp | `main` `0c45b34c` (its NInfer route is not in release v2026.10.23) + [patch](../bench/engine_quality/patches/tensorsharp-0c45b34c.patch) | Windows, .NET 10.0.403, CUDA 13.4 | its shipped `config/qwen3.8-27b-ninfer.json`: direct-CUDA backend, n_ctx 40,960, ubatch 2,048, the artifact's DFlash2 (7 drafts, pMin 0, `TS_SPEC_ADAPTIVE=0`), BF16 KV |
| llama.cpp | b12128 + [patch](../bench/engine_quality/patches/llama.cpp-b12128.patch) | Windows, CUDA 13.4 | `llama-perplexity -c 512 -b 512 -ctk q8_0 -ctv q8_0`, its default placement (`--fit`: experts that do not fit run on the CPU) |
| Strata | v0.2.11 (`c3b3f503`) + [patch](../bench/engine_quality/patches/strata-v0.2.11.patch) | Windows, CUDA 13.0 | its setup's IQ3_S configuration for this PC ([config](../bench/engine_quality/configs/strata-qwen3.8-flash-next-iq3_s.json)): INT8 KV, `--expert-cache auto`, `--prefill auto`, `--spec 4 --spec-min-p 0.5` with its MTP head, route tail skip at its default (on) |
| Basalt | v0.2.1 (`1b3f6964`) + [patch](../bench/engine_quality/patches/basalt-v0.2.1.patch) | Windows, CUDA 13.4 | its GSQ-RCO IQ3_S pack (format 2), `--kv int8`, `ppl_gate` with verify windows of 4 |

What each patch does:

- **NInfer**: `ninfer-kld-rows` reads the base file's chunks and writes each row: on the prompt path
  through its scoring route (`NINFER_SCORE_LOGITS_DUMP`), on the decode path (`--decode`) through a hook
  in the decision step that records the row and raises the text token's logit
  (`NINFER_DECODE_ROWS_DUMP`).
- **llama.cpp**: `LLAMA_KLD_ROUND_BF16` rounds the logits to BF16 before scoring;
  `LLAMA_KLD_DECODE_T=N` scores the second half through forced windows of N tokens instead of the
  prompt batch.
- **Strata**: `STRATA_LOGPOS_ROWS=FILE` dumps every verify-window row (position, vocabulary, logits).
- **Basalt**: `BASALT_PPL_ROWS=FILE` dumps every row its `ppl_gate` scores.
- **TensorSharp**: `eng/KldRows` (prompt: one forward per chunk, then the engine's own final norm and
  head over the scored rows, 64 at a time, instead of the last row only; decode: the speculative calls
  above); and `__restrict__` written out in three explicit kernel instantiations, which the MSVC host
  compiler's front end otherwise fails to match (the compiled kernels are unchanged).

### vLLM's checkpoint

NInfer's artifact keeps Unsloth's NVFP4 and FP8 words unchanged and stores the embedding as FP8 rows
with one BF16 scale per row; Unsloth's checkpoint stores it in BF16.
[`make_vllm_27b_copy.py`](../bench/engine_quality/make_vllm_27b_copy.py) copies Unsloth's checkpoint and
overwrites only its embedding with BF16(code × scale) of the artifact's rows, the exact product rounded
once. An embedding row's output is BF16 in the model, so this is the same model the artifact holds.

## Repeating the measurement

1. **Weights.** Download the artifacts and checkpoints named above (the 27B `.ninfer`, Unsloth's
   27B checkpoint for vLLM, the IQ3_S GGUF, Infernix's IQ3_S and FP4-Local artifacts with their
   `.ngram` volumes, and the DFlash2 companion) and
   build Strata's and Basalt's packs of the GGUF with their own tools. Make vLLM's checkpoint:

   ```bash
   python bench/engine_quality/make_vllm_27b_copy.py UNSLOTH_DIR qwen3_8_27b_nvfp4.ninfer vllm-27b
   ```

2. **Engines.** Build `infernix-certify` from this repository. Check out each other engine at its
   version, apply its patch with `git apply` (which also works in llama.cpp's unpacked release tarball),
   and build:

   ```bash
   # llama.cpp b12128 (Windows; vcvars64 and CUDA 13.4 on PATH)
   cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DLLAMA_CURL=OFF
   cmake --build build --target llama-perplexity
   ```

   ```bash
   # NInfer a754fcf3b (Linux or WSL2)
   cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DNINFER_TARGET=rtx5090 -DNINFER_BUILD_ENGINE=ON -DNINFER_BUILD_APPS=ON -DBUILD_TESTING=OFF
   cmake --build build --target ninfer-kld-rows
   ```

   ```bash
   # Strata v0.2.11 (Windows; CUDA 13.0 with MSVC 14.44, whose headers CUDA 13.0's front end accepts)
   cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DSTRATA_PORTABLE=ON -DSTRATA_BUILD_TESTS=OFF
   cmake --build build --target strata
   ```

   ```bash
   # Basalt v0.2.1 (Windows)
   cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120 -DBUILD_TESTING=OFF
   cmake --build build --target basalt
   ```

   ```bat
   REM TensorSharp 0c45b34c (Windows cmd; .NET 10 SDK, CUDA 13.4, vcvars64). The direct-CUDA route needs
   REM no GGML library; building it would also download cuDNN.
   set TENSORSHARP_GGML_NATIVE_SKIP=true
   set TENSORSHARP_MLX_NATIVE_SKIP=true
   dotnet build eng\KldRows\KldRows.csproj -c Release -p:CudaArch=compute_120 -o tensorsharp-build
   ```

   ```bash
   # vLLM 0.33.0 (Linux or WSL2)
   uv venv --python 3.12 vllm-venv
   uv pip install --python vllm-venv/bin/python --torch-backend=cu130 "vllm==0.33.0" numpy
   ```

3. **Configurations.** [`bench/engine_quality/configs/`](../bench/engine_quality/configs/) holds the
   exact configurations of these runs. Their paths are this machine's: edit the artifact, text,
   binary and work paths (and the WSL paths in the NInfer and vLLM command lines).

4. **Run.** From the repository root, with Python 3.11 and numpy:

   ```bash
   python -m tools.quality.engine_compare plan bench/engine_quality/configs/qwen3.8-27b-nvfp4.json
   python -m tools.quality.engine_compare run bench/engine_quality/configs/qwen3.8-27b-nvfp4.json
   python -m tools.quality.engine_compare report bench/engine_quality/configs/qwen3.8-27b-nvfp4.json
   ```

   `plan` lists the steps; `run` executes the ones not yet done (`--only ID,...` selects some) and
   logs each to `WORK/logs`; `report` writes `WORK/report.md` and `report.json`. A test run on a
   few chunks: set `"chunks"` (and a separate `"work"`) in a copy of the configuration.
   On this machine the 27B took 5.0 GPU hours (vLLM's eager decode 2.1 of them), IQ3_S 2.7 and
   FP4-Local's 200 chunks about 1.8 (`qwen3.8-flash-next-fp4-local.json`: its `"arms": "split"` runs
   the reproducible and default routes and each fast path alone).

## Limitations

- **One text, one context.** WikiText-2 in 512-token chunks: short contexts in English prose.
  Long contexts, code, chat and thinking were not measured.
- **One honest-rounding seed per model.** The spread between seeds was not measured.
- **Strata's and Basalt's decode rows only**, and Strata's protocol differs as described above.
- **Speculative drafting differs by engine.** It changes which window sizes compute each row, not
  which tokens are scored; acceptance under forced text is low for some drafters (TensorSharp's
  DFlash2 accepted 9.4 % of drafts), which only means more windows.
- **These builds.** Later versions of any engine, Infernix included, were not measured.
