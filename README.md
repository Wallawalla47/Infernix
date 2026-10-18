# Infernix

> **AI disclaimer:** Everything added to this engine since it forked from NInfer, including most of
> this README, was written with AI (mostly Claude Opus 5.5, Qwen3.8-27B running on Infernix, plus a
> few other AI systems I’ve been testing). It is likely to be neither complete nor entirely
> accurate. This is hobby development.

Infernix is a C++/CUDA inference engine that serves Qwen models on a single NVIDIA
RTX 50-series GPU (tuned on the RTX 5090). It began as a fork of
[NInfer](https://github.com/Neroued/ninfer) by [Neroued](https://github.com/Neroued) and has grown
into its own engine; it reads NInfer's `.ninfer` artifacts as well as its own `.infernix` files.

## Why Infernix

### Qwen3.8-Flash-Next on one RTX 5090

Qwen3.8-Flash-Next has 24,576 routed experts (63 GiB at NVFP4), far more than a 32 GB GPU holds.
Infernix keeps the experts in pinned system RAM, caches the hot ones in VRAM, computes part of each
layer's cache misses on the CPU while the rest cross PCIe, and reads the 52 GB n-gram embedding
table from an NVMe drive.

- **Faster than [Strata](https://github.com/Niko1221/Strata)** on the same machine: 5.4-7.0× sooner
  to the first token and 1.4-1.6× the decode speed from 8K to 250K tokens of context. In a replayed
  agentic coding workload it reaches the first token in 5.2 s on average against 19.3 s and
  finishes in about a third of the time ([benchmarks](#qwen38-flash-next-nvfp4-infernix-vs-strata)).
- **At least Strata's quality**: perplexity 4.654 against 4.844 on the same token ids.
- **Long context**: 262K tokens natively and up to 1M with YaRN; a 250K-token prompt prefills in
  about 30 s.
- **Less RAM than the model needs?** An SSD expert tier keeps what fits in RAM and reads the rest
  from the artifact.
- **Uncensored too**: the
  [Qwen3.8-Flash-Next-Uncensored-NVFP4-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-Uncensored-NVFP4-Infernix)
  conversion of orcarouter's abliterated build keeps its weights bit-exact (perplexity 4.772 against
  Strata's 4.844 for the base model) and decodes about as fast as the NVIDIA Dense8 conversion
  ([models](#models)).

Setup and tuning: the [Qwen3.8-Flash-Next guide](docs/qwen3_8-flash-next.md).

### Qwen3.8-27B: more cache hits, faster prefill and decode

- **Hybrid prefix cache** built for Qwen's linear-attention layers: KV cached by content in 64-token
  blocks, snapshots of the recurrent state at the points that get reused, and three tiers (GPU,
  host RAM, and a file that survives restarts). In an agentic workload it serves 85 % of prompt
  tokens from cache against NInfer's 79 %, reaches the first token a third sooner on average and
  more than halves the median ([benchmarks](#qwen38-27b-infernix-vs-ninfer)).
- **Faster prefill**: 1.4-1.5× NInfer's on long prompts (FlashAttention-2 style INT8 prompt
  attention, FP4/FP8 Tensor Core paths, wave-aligned chunks).
- **Faster decode**: DFlash2 tree verification and n-gram copy drafting alongside DFlash2 and MTP.
- **Smaller KV**: the `vq2` and `k4v2` formats hold three to four times the context of INT8.

### Built for agents

- **APIs**: OpenAI Chat Completions and Responses, Anthropic Messages and streaming, with the
  request options that Claude Code, Qwen Code, Codex, Zed and GitHub Copilot send.
- **Tool calling that holds up**: the XML call forms agents emit, tolerant recovery of broken calls
  (`--tolerant-tool-calls`), calls stranded in unclosed thinking recovered, and calls an answer only
  quotes in Markdown code kept as text.
- **Constrained decoding**: JSON object and JSON Schema outputs, strict tool arguments and tool
  choice, and GBNF, regex and choice grammars, on every model and speculative mode.
- **Thinking control**: budgets with a custom wrap-up message, reasoning effort levels and a guard
  for thinking that keeps repeating itself (`--reasoning-loop`).
- **Concurrency**: one to eight requests at once, with preemption under memory pressure.
- **Resilience**: the prefix cache persists across restarts (and every N minutes with
  `--prefix-cache-save-mins`), and on Qwen3.8-Flash-Next a failed expert read fails only the
  requests running at the time, not the server.
- **`/v1/decide`**: decisions read from token probabilities (yes/no, choice, score, number, point,
  box).

### And also

- **Speculative decoding**: MTP, DFlash, DFlash2 and n-gram drafting, with exact sampling.
- **Every KV format**: `int8`, `bf16`, `fp8`, `nvfp4`, `k8v4`, `vq2` and `k4v2`.
- **Vision**: images and video, with the vision encoder optionally in system RAM.
- **Conversion tools**: Hugging Face, NVIDIA ModelOpt, Quasar and GGUF sources, with recipes for
  Q4-Q8, FP8, NVFP4 and mixed artifacts.
- **Windows and Linux**: native MSVC and GCC builds, WSL2 included.

## Benchmarks

One machine: an RTX 5090 (32 GB) in a **PCIe Gen5 x8** link, a Core i9-13900K, 96 GB of DDR5 and
Windows 11. At Gen5 x16 speeds are likely to be higher, most of all for Qwen3.8-Flash-Next, whose
expert-cache misses cross PCIe on every token. Each engine runs as its own OpenAI-compatible server
and is driven by the same client, and every time to first token is measured the same way, on the
client: from sending the request to the first streamed token. The latest Infernix
(`23d310d6`) is compared with its competitor and with Infernix's first release
(`windows-2026.10.09`). In the single-request tables the two Infernix builds run in interleaved
sessions (latest, first, first, latest), pooled per build; the other engine runs one session. The
agentic replay ([`bench/agentic_ab`](bench/agentic_ab/README.md)) interleaves three coding-agent
sessions and eleven subagents with tool calls, compactions, retries and aborts, up to eight
requests in flight, sampling at temperature 1.0, at half length (68 requests per seed); cells are
means over the seeds with the range in brackets.

### Qwen3.8-Flash-Next-NVFP4: Infernix vs Strata

Infernix runs the [Dense8 conversion](#models) of NVIDIA's checkpoint; Strata runs Unsloth's
`Qwen3.8-Flash-Next-UD-Q4_K_XL` GGUF, its native format. Both use the model's MTP drafter and INT8
KV at 262,144 tokens, so this compares each engine as it is run in practice, each with its own
quantization and drafter.

**Context length** (one request at a time, greedy, 1,024 output tokens; Strata 0.1.42 in its
default single-slot configuration):

| Context | Time to first token: Infernix | first release | Strata | Decode tok/s: Infernix | first release | Strata |
|---|---:|---:|---:|---:|---:|---:|
| ~8K | 2.00 s | **1.95 s** | 12.17 s | **139.8** | 138.7 | 97.6 |
| ~128K | **14.71 s** | 17.95 s | 103.02 s | **137.8** | 134.5 | 92.7 |
| ~250K | **30.09 s** | 34.14 s | 163.52 s | **170.7** | 157.7 | 104.7 |

Long prompts prefill 18 % (128K) and 12 % (250K) faster than in the first release: the prefix
cache now snapshots a long prompt inside its layer walk instead of ending a walk at each snapshot,
which made the walk stream the uncached experts again. At 8K the first latest-build session, which
ran straight after another GPU job, was the slowest (2.06-2.10 s); the second took 1.92 s, against
1.91-1.97 s for the first release.

**Agentic replay** (three seeds; Strata 0.1.42 with `"parallel": 2` and `--batch-mtp`, so that both
engines serve two requests at once):

| Metric | Strata | Infernix | Infernix first release | Infernix vs Strata | vs first release |
|---|---:|---:|---:|---:|---:|
| Average time to first token | 19.3 s (17.4-20.9) | **5.2 s** (5.0-5.5) | 6.3 s (6.0-6.6) | −73 % | −17 % |
| Median time to first token | 8.94 s (7.15-9.99) | **2.78 s** (2.61-2.93) | 2.85 s (2.68-3.11) | −68 % | −2 % |
| 90th-percentile time to first token | 53.1 s (47.7-57.9) | **11.9 s** (11.4-12.5) | 14.4 s (12.8-15.3) | −77 % | −17 % |
| Prompt tokens served from cache | 72.1 % (71.0-73.8) | **84.7 %** (84.7-84.8) | 84.7 % (84.6-84.8) | +12.6 points | 0 points |
| Prefill tok/s, requests with no cache hit | 1,059 | **6,081** | 5,097 | 5.7× | +19 % |
| Output tok/s, one request decoding | 78 (78-79) | **131** (127-135) | 127 (123-130) | +68 % | +3 % |
| Output tok/s, at the run's own batching | 44 (39-47) | **129** (122-137) | 120 (115-123) | 2.9× | +8 % |
| Tokens per round (draft acceptance) | 2.06 | 2.37 | **2.38** | +15 % | 0 % |
| Workload wall time | 20.1 min (18.9-20.8) | **7.2 min** (6.7-8.0) | 7.7 min (6.9-8.3) | −64 % | −6 % |

With two slots and batched MTP, Strata failed 5, 6 and 7 of the 68 requests of the three seeds
("BGEN: no such free slot"); the table covers the 61-63 requests per seed that every engine
completed. Both engines verify every draft, so drafting changes speed, not output.

### Qwen3.8-27B: Infernix vs NInfer

Both serve NInfer's official
[Qwen3.8-27B-nvfp4-NInfer](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) artifact with
DFlash2 drafting, INT8 KV and the same flags (NInfer `master` at `81c8ce09` with the Windows port;
Infernix also uses n-gram drafting in the agentic replay).

**Context length** (one request at a time, greedy, 1,024 output tokens; the latest Infernix ran as
`f7dae66e`, whose later commits leave the 27B unchanged):

| Context | Time to first token: Infernix | first release | NInfer | DFlash2 decode tok/s: Infernix | first release | NInfer |
|---|---:|---:|---:|---:|---:|---:|
| ~8K | **0.75 s** | 0.76 s | 0.76 s | **209.0** | **209.0** | 198.9 |
| ~128K | **21.61 s** | 21.64 s | 29.39 s | **212.7** | **212.7** | 186.8 |
| ~230K | 53.79 s | **53.78 s** | 82.23 s | 161.9 | 162.0 | **227.5** |

The two Infernix builds compute the same tokens at the same speed. Long prompts prefill 1.36×
(128K) and 1.53× (230K) faster than NInfer. At 230K NInfer's drafts were accepted more often on
this prompt (4.67 against 3.15 tokens per round), so it decoded faster; without speculation decode
is level (59.5 tok/s against 58.9 at 230K, 79.7 against 78.9 at 8K). NInfer kept the 8K prompt
cached through the two 230K prompts that followed it (a repeat took 0.29 s); Infernix had evicted
it (0.85 s).

**Agentic replay** (five seeds; `--max-context 150000` for every engine: the largest at which
NInfer starts with two requests and DFlash2 on 32 GB):

| Metric | NInfer | Infernix | Infernix first release | Infernix vs NInfer | vs first release |
|---|---:|---:|---:|---:|---:|
| Average time to first token | 4.1 s (3.2-5.0) | 2.8 s (2.0-3.2) | **2.5 s** (2.3-2.7) | −33 % | +12 % |
| Median time to first token | 1.10 s (0.93-1.58) | 0.43 s (0.38-0.50) | **0.41 s** (0.35-0.46) | −59 % | +5 % |
| 90th-percentile time to first token | 12.1 s (9.3-15.2) | 8.4 s (5.0-10.1) | **8.1 s** (6.3-9.7) | −30 % | +4 % |
| Prompt tokens served from cache | 79.2 % (78.7-79.4) | **85.4 %** (85.0-85.6) | **85.4 %** (85.0-85.6) | +6.2 points | 0 points |
| Prompt tokens prefilled | 562K | **398K** | 399K | −29 % | −0.2 % |
| Prefill tok/s, requests with no cache hit | 5,228 | **7,193** | 7,150 | +38 % | +1 % |
| Output tok/s, one request decoding | 193 (189-202) | **202** (195-213) | 200 (194-205) | +5 % | +1 % |
| Output tok/s, at the run's own batching | 231 (222-250) | **261** (254-269) | 254 (240-278) | +13 % | +3 % |
| Workload wall time | 7.1 min (6.2-8.0) | **6.2 min** (5.6-6.6) | 6.3 min (5.9-6.7) | −13 % | −2 % |

Against the first release the engine does the same work at the same speed: each request's prompt
takes the engine 0.74 s on average on both builds, and the requests hold a lane for 2,253 s in all
against 2,277 s (5.16 against 5.22 ms per output token). The higher time to the first token is time
spent waiting for one of the two lanes (2.03 s against 1.81 s per request, more on four of the five
seeds), which depends on how long the sampled replies ahead of a request ran: both builds wrote
390K output tokens in all, but in different turns (67-88K per seed). Wall time depends on it too.
Quality is level: on the repository's 1.04M-token perplexity corpus (16K
context, INT8 KV cache) Infernix scores 4.468 against 4.479 for NInfer (`master` at `81c8ce09` with
the Windows port).

## Quality

Infernix runs two routes:

- **The exact route** (`--disable-fast-paths`) gives bit-identical output from run to run and from
  release to release. Where an artifact's formats allow it (the NVFP4 Qwen3.8-27B and the NVFP4,
  FP8 and GGUF Flash-Next imports, for example), it is also request-invariant: other requests, concurrency,
  prefix-cache state and memory pressure do not change a request's output.
- **Fast paths** (on by default) are faster kernels and scheduling choices that change the
  arithmetic. Each one runs only for the kernels it was qualified on, only after passing the quality
  gate below, and only where it was measured faster.

### The quality gate

A fast path is measured against a high-precision reference: the same model with 16-bit activations
in every layer and a BF16 KV cache. Infernix charges a path only for the distance it adds beyond
the exact route's own, so the quantization a model already carries is not counted against it.

- **Prefill**: teacher-forced over a 1.04M-token corpus at four chunk sizes, measuring the KL
  divergence from the reference and top-1 token agreement.
- **Decode**: 1,024 forced decoding segments (prompts of 512 to 8,192 tokens, 65,536 scored
  positions), measuring total variation (TV) and top-1 agreement. The fast run reproduces the
  conditions the path acts in: concurrent requests, speculative verification, cache resumes and a
  short KV pool.
- **Budgets**, shared by all of a configuration's fast paths together: KL 1.5e-3 (about what
  llama.cpp's q8_0 costs over FP16), top-1 0.5 percentage points and decode TV 4e-3. There are also
  checks on the worst positions (the 99.9th-percentile KL may grow by at most 25 %), on every prompt
  length and context range, and caps on KL and perplexity in decode.
- **The decision** uses the 95 % upper confidence bound, not the average, so a noisy measurement
  makes a path harder to pass, never easier.
- **Every KV format** is gated. A qualification follows the kernels a path changes, so artifacts
  that run the same kernels share it.

The full policy is in [model quality §6](docs/maintainer/model-quality.md#6-numerical-acceptance-policy-and-fast-paths).
Other engines check speed-motivated changes more loosely:

| Engine | How a numerics change is accepted |
|---|---|
| vLLM, SGLang (CI) | each side's next token within the other's top 5, or top-5 log-probabilities within 0.05-0.08 of Hugging Face, plus GSM8K accuracy floors |
| llama.cpp | kernel error against the CPU backend; KL is reported for new quantizations without a threshold |
| Strata | KL against a reference or its own run-to-run noise, with no fixed bound; its two non-exact defaults moved KL by 4e-3 and 5.8e-3 |
| Fireworks | KL below 7e-3 for a whole quantized deployment |
| **Infernix** | **1.5e-3 KL, 0.5 points of top-1 and 4e-3 TV for all fast paths together, at 95 % confidence** |

### Defaults elsewhere that Infernix rejected

These routes are on by default in another engine (NInfer as of `master` 64d10fbf9 and `dev`
4126ecf4, checked on 2026-10-18; Strata 0.1.42). As implemented and measured in Infernix, each
failed the gate:

| Route | On by default in | What it does | Why it failed |
|---|---|---|---|
| Expert tail skip at rank 7 | Strata (`STRATA_ROUTE_TAIL_SKIP=7`) | In a decode round, skips an expert missing from VRAM that every token routes at rank 7 or lower (of its top 10), without renormalising the rest | GGUF IQ3_S Flash-Next, INT8 KV, MTP decode: TV bound 4.26e-3 against 4e-3, top-1 0.75 against 0.5 points and the 99.9th-percentile KL up 41 % against 25 %. The dropped experts' contributions are small but not zero, and which ones are dropped depends on what the cache happens to hold. Infernix offers it as an opt-in, `--expert-tail-skip R` (+12 % to +33 % decode on an RTX 5090) |
| 4-bit activations in wide decode rounds | NInfer, for artifacts that permit it, as its official NVFP4 Qwen3.8-27B does (and Infernix, inherited, until 2026-10-14) | Wide rounds quantize the NVFP4 MLP projections' input activations once to 4 bits (gate/up from 5 columns, down from 8) | Qwen3.8-27B NVFP4, INT8 KV: decode TV 1.57e-2 (about four times the budget) and top-1 +1.8 points; the 4-bit activations add about two thirds of the model's whole quantization noise. Perplexity does not show it (its excess was negative). Replaced by `batched-a4-residual-decode`, which sums two 4-bit terms and passes |
| 8-bit activations in wide decode rounds | NInfer, likewise (and Infernix, inherited, until 2026-10-14) | Rounds of 5 columns (17 when verifying drafts) or more quantize the FP8 projections' input activations once to 8 bits | Passes plain decode, where few rounds are that wide, but fails speculative verification of 32 columns (DFlash2 with n-gram drafts, four requests): TV bound 4.38e-3 against 4e-3 and top-1 0.55 against 0.5 points. Replaced by `batched-a8-residual-decode` (two 8-bit terms), which passes |

The replacements cost some speed. When they landed (2026-10-15), speculative decode on the 27B
(DFlash2 with n-gram drafts) measured 4 % slower than with the rejected routes at one request, 6 %
at two and 20 % at four. In the agentic replay the 27B now runs
3.7 % fewer decode rounds a second than the release benchmarked above, which still used them.

## Quick start (Windows)

Prerequisites: Visual Studio 2026 (MSVC; the VS 2022 compiler, MSVC 14.44, also builds it), the
CUDA 13 toolkit, and FFmpeg + curl from vcpkg (`x64-windows`, at `C:\vcpkg`); adjust the paths at
the top of `build_native.bat` for your machine (and, for Visual Studio 2022, its generator).

```bat
build_native.bat configure
build_native.bat build
```

The server is `build-windows\apps\Release\infernix-serve.exe`, with the FFmpeg, curl and zlib DLLs
next to it. Qwen3.8-Flash-Next (see the [guide](docs/qwen3_8-flash-next.md#run)):

```bat
infernix-serve.exe Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix-00001-of-00003.infernix --ngram-volume Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram --kv-dtype int8 --max-context 262144 --spec mtp --draft-tokens auto --lm-head-draft --max-concurrency 2
```

The uncensored Flash-Next takes the same options and the same n-gram volume:

```bat
infernix-serve.exe Qwen3.8-Flash-Next-Uncensored-NVFP4-Infernix-00001-of-00003.infernix --ngram-volume Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram --kv-dtype int8 --max-context 262144 --spec mtp --draft-tokens auto --lm-head-draft --max-concurrency 2
```

Qwen3.8-27B on a 32 GB RTX 5090:

```bat
infernix-serve.exe qwen3_8_27b_nvfp4-nvidia.ninfer --max-context 240000 --max-concurrency 2 --spec dflash2 --draft-tokens 7 --lm-head-draft --ngram-draft-tokens 15 --ngram-min-match 12 --kv-dtype int8 --preserve-thinking --host-context-mib 52000 --prefill-chunk 4096 --kv-capacity auto --tolerant-tool-calls
```

Add `--prefix-cache-file PATH` to keep the prefix cache across restarts. Stop the server with
Ctrl+C twice; it saves the cache and exits. `infernix-serve.exe --help` lists every option by
category; `--config FILE` reads them from a JSON file instead
([configuration file](docs/serving.md#configuration-file)).

Running on another PC needs an RTX 50-series GPU and an NVIDIA driver of 580 or later (CUDA 13); no
CUDA toolkit is needed. Copy the DLLs next to `infernix-serve.exe` and install the Visual C++
redistributable if it is missing. `packaging\windows\package_release.ps1` packs the server, its
DLLs and every licence they need into a release zip
([third-party notices](packaging/windows/THIRD_PARTY_NOTICES.md)).

## Quick start (Linux)

Builds and runs on 64-bit Linux, including WSL2 (tested on Ubuntu 24.04 with CUDA 13.4 and GCC
13.3). Prerequisites: a CUDA 13 toolkit, CMake 3.28 or newer, a C++20 compiler, Ninja, `pkg-config`,
and the FFmpeg and curl development packages:

```bash
sudo apt-get install -y build-essential cmake ninja-build pkg-config libavformat-dev libavcodec-dev libavutil-dev libswscale-dev libcurl4-openssl-dev
cmake --preset release
cmake --build build -j
```

The server is `build/apps/infernix-serve` and takes the same options. Under WSL2 use the Windows
NVIDIA driver (do not install a Linux one inside WSL) and copy models into the Linux filesystem
first; reads through `/mnt/` are slow.

## Models

- **[Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix)**
  (**recommended** for Qwen3.8-Flash-Next): [nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4)
  with NVIDIA's NVFP4 experts and n-gram table bit-exact and the dense projections in 8 bits. It
  decodes about 22 % faster than the bit-exact conversion at the same quality (perplexity 4.654
  against 4.666).
- **[Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix)**:
  every weight the model computes with exactly as NVIDIA stores it, for when the weights must be
  unchanged.
- **[Qwen3.8-Flash-Next-Uncensored-NVFP4-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-Uncensored-NVFP4-Infernix)**:
  [orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4),
  an abliterated build of Qwen3.8-Flash-Next whose refusal behaviour has been removed (see its model
  card). Its weight-only NVFP4 experts and FP8 dense projections are kept bit-exact and run with BF16
  activations; the other dense classes are stored in 8 bits as in Dense8. Perplexity 4.772 (Strata's
  base-model GGUF 4.844, NVIDIA Dense8 4.653: abliteration changes the weights). It needs Infernix
  `windows-2026.10.09.2` or later; earlier releases refuse it.
- **[Swift-1.5-Qwen3.8-Flash-Next-NVFP4-Dense8-Infernix](https://huggingface.co/Wallawalla47/Swift-1.5-Qwen3.8-Flash-Next-NVFP4-Dense8-Infernix)**:
  [ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4),
  UkisAI's fine-tune of Qwen3.8-Flash-Next that thinks with fewer tokens (see its model card),
  converted like NVIDIA Dense8: experts bit-exact, dense projections in 8 bits. Smoke-tested; its
  quality and speed are not yet measured.
- **[Qwen3.8-27B-NVIDIA-NVFP4-NInferV3](https://huggingface.co/Wallawalla47/Qwen3.8-27B-NVIDIA-NVFP4-NInferV3)**
  (**recommended** for the 27B): [nvidia/Qwen3.8-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4)
  with a DFlash2 draft model and a proposal head for `--lm-head-draft`.
- **[Qwen3.8-27B-Quasar-NVFP4-NInferV3](https://huggingface.co/Wallawalla47/Qwen3.8-27B-Quasar-NVFP4-NInferV3)**:
  the QAT-trained [QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4](https://huggingface.co/QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4),
  with the same draft model and proposal head.
- **[Qwen3.8-27B-Uncensored-NVFP4-NInferV3](https://huggingface.co/Wallawalla47/Qwen3.8-27B-Uncensored-NVFP4-NInferV3)**:
  [orcarouter/Qwen3.8-27B-Uncensored-NVFP4](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-NVFP4),
  an abliterated build of Qwen3.8-27B whose safety alignment has been substantially removed (see its
  model card), with the same draft model and proposal head.
- **[Swift-Qwen3.8-27B-NVFP4-NInferV3](https://huggingface.co/Wallawalla47/Swift-Qwen3.8-27B-NVFP4-NInferV3)**:
  [ukisai/Swift-Qwen3.8-27B-NVFP4](https://huggingface.co/ukisai/Swift-Qwen3.8-27B-NVFP4), UkisAI's
  token-efficient fine-tune of Qwen3.8-27B in NVIDIA's NVFP4/FP8 layout, converted like the NVIDIA
  artifact with the same draft model and proposal head. Smoke-tested on Infernix and NInfer; its
  quality and speed are not yet measured.

All four Flash-Next conversions use the same 52 GB n-gram volume,
[`Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram`](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix/blob/main/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram)
(the uncensored and Swift repos do not carry their own copy). `python -m tools.convert` converts
your own checkpoints ([weight conversion](docs/weight-conversion.md)).

## Documentation

[CLI](docs/cli.md), [serving](docs/serving.md), [Qwen3.8-Flash-Next](docs/qwen3_8-flash-next.md),
[perplexity](docs/perplexity.md), [performance](docs/performance.md) and the
[documentation map](docs/README.md).

## Contributing

Issues and pull requests of any kind are very welcome: bug reports, questions, measurements on
other GPUs and machines (PCIe x16 results especially), fixes, features and documentation.

## Support

Infernix is developed with Claude Code. If you would like to help pay for that subscription, you can
[sponsor me on GitHub](https://github.com/sponsors/Wallawalla47).

## Thanks

Infernix exists because of **[Neroued](https://github.com/Neroued)**, who created NInfer and built
the engine, artifact format and kernels this project grew from — thank you. Please support the
creator of the original NInfer [on Ko-fi](https://ko-fi.com/neroued). A particular thank you
also to the creator of **[Strata](https://github.com/Niko1221/Strata)** ([Niko1221](https://github.com/Niko1221)),
whose engine set the bar for Qwen3.8-Flash-Next on a single GPU, whose transcription of the model's
mathematics and published measurements guided this implementation, and whose community's ideas
(including wangmeng's reasoning-loop recovery and eddoursul's work on host-thread placement)
shaped several features here.

Thank you as well to everyone whose issues, pull requests, forks and research Infernix has drawn on:
[Michael Dementii](https://github.com/MichaelDementii),
[Minnnn](https://github.com/Minnnn),
[DuncanBetts](https://github.com/DuncanBetts),
[Valerio Dolci](https://github.com/ValerioDolci),
[Thireus](https://github.com/Thireus),
[remesis](https://github.com/remesis),
[Valeriy Selitskiy (iamwavecut)](https://github.com/iamwavecut),
[Hector Ramon Jimenez (hecrj)](https://github.com/hecrj),
[giveen](https://github.com/giveen),
[bingchengcc](https://github.com/bingchengcc),
[pkochubey](https://github.com/pkochubey),
[paq85](https://github.com/paq85),
[Macasacker](https://github.com/Macasacker),
[Sha1rholder](https://github.com/Sha1rholder),
[adubkov](https://github.com/adubkov),
[Gideon Zenz (gzenz)](https://github.com/gzenz),
[cometkim (Hyeseong Kim)](https://github.com/cometkim),
[Woesch-Nich](https://github.com/Woesch-Nich),
[Hundsbuah](https://github.com/Hundsbuah),
[lonelystarCX](https://github.com/lonelystarCX),
[gpillon](https://github.com/gpillon) (ignis),
[CaptainArni](https://github.com/CaptainArni),
[David Oelfke](https://github.com/Doelfke),
Fedor Suchkov, Yunado,
[IST-DASLab](https://github.com/IST-DASLab) (llmq), the authors of HyperQuant and Four Over Six,
the Qwen team for the models, and NVIDIA for the NVFP4 checkpoints.
