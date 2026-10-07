# Infernix

> **Infernix began as a fork of [NInfer](https://github.com/Neroued/ninfer)** by
> [Neroued](https://github.com/Neroued), a from-scratch C++/CUDA inference engine for single-GPU
> serving of Qwen models. It has since grown into its own engine. It runs
> **Qwen3.8-Flash-Next-NVFP4** — a mixture-of-experts model whose experts do not fit in VRAM — on a
> single RTX 5090, **faster than [Strata](https://github.com/Niko1221/Strata)** on the
> same machine, and it serves the dense **Qwen3.8-27B** models with a hybrid prefix cache, broader
> tool-calling support, faster prefill and decode kernels and many other improvements over NInfer.
> Infernix still reads NInfer's `.ninfer` artifacts as well as its own `.infernix` files.

> **AI disclaimer:** Infernix and this README were written with AI (mostly Claude Opus 5.5, with
> Qwen3.8-27B running on this engine and a few other AI systems). It is hobby development; it is
> likely to be neither complete nor entirely accurate.

## Highlights

### Qwen3.8-Flash-Next-NVFP4 on one RTX 5090

Qwen3.8-Flash-Next has 24,576 routed experts (63 GiB at NVFP4) — far more than a 32 GB GPU holds.
Infernix keeps the experts in pinned system RAM, caches the hot ones in VRAM, computes part of
each layer's cache misses on the CPU while the rest cross PCIe, and streams the 52 GB per-layer
n-gram embedding table from an NVMe drive.

- **Faster than Strata** on the same RTX 5090: about 102 tok/s plain decode and about 132 tok/s
  with the model's MTP drafter at 8K context, where Strata measured 72-80 tok/s.
- **Quality at least Strata's**: every optimisation is held to teacher-forced quality measured
  against Strata's output on the same text.
- **Long context**: 262K tokens natively, up to 1M with YaRN (`--rope-yarn-factor`); a 245K-token
  prompt prefills at ~8.2K tok/s and still decodes at ~74 tok/s plain, ~88 with MTP.
- **Every KV format** (`int8`, `bf16`, `fp8`, `nvfp4`, `k8v4`, `vq2`, `k4v2`), images and video,
  the hybrid prefix cache, concurrent requests with resource-pressure preemption, an SSD expert
  tier for machines with less RAM, and a VRAM pool that grows with the context.

| Benchmark | Infernix | Strata |
|---|---:|---:|
| *Table to follow* | | |

Setup, conversion and tuning: [Qwen3.8-Flash-Next guide](docs/qwen3_8-flash-next.md).

### Qwen3.8-27B: prefix caching, tool calling and speed

- **Hybrid prefix cache** (the default), built for Qwen's linear-attention layers: KV cached per
  64-token block by content, sparse snapshots of the recurrent state at useful points, three tiers
  (GPU, pinned host RAM, and a file that survives restarts). On a replayed agentic coding workload
  it serves 90.5 % of prompt tokens from cache against 85.8 % for NInfer's newest cache, and
  finishes 26 % sooner.
- **Tool calling for agent clients**: the XML call forms Claude Code and other agents emit,
  tolerant recovery of broken calls (`--tolerant-tool-calls`), Anthropic and OpenAI options used by
  Claude Code, Qwen Code, Codex, Zed and GitHub Copilot, and correct handling of reasoning output.
- **Faster prefill**: FlashAttention-2 style INT8 prompt attention, FP4/FP8 Tensor Core paths for
  NVFP4, FP8 and K8V4 KV, wave-aligned chunks — a third to two thirds less prompt-attention time
  on long prompts.
- **Faster decode**: overlapped (PDL) decode kernels, verification-width kernel tuning, DFlash2
  tree verification and n-gram copy drafting alongside MTP/DFlash2.
- **Smaller KV**: the `vq2` and `k4v2` formats hold three to four times the context of INT8.

## Performance: Qwen3.8-27B against NInfer

Measured on an RTX 5090 under Windows with the official Qwen3.8-27B NVFP4 artifact, October 2026.
The closed-loop suite in [`bench/agentic_ab/`](bench/agentic_ab/README.md) replays three
coding-agent sessions plus eleven subagents (130 requests, prompts of 25K-135K tokens, thinking
on) on three workload seeds. "NInfer" is upstream `master` at `abb7f14f` with its new context
cache, plus only the Windows port. Each cell is the mean over the three seeds (lowest and highest
in brackets).

| Metric | NInfer + Windows port | Infernix |
|---|---|---|
| Average time to first token (s) | 4.2 (3.7-4.8) | 2.5 (2.2-2.7), −39.2 % |
| Median time to first token (s) | 1.21 (1.09-1.31) | 0.63 (0.60-0.69), −47.5 % |
| 90th-percentile time to first token (s) | 11.4 (9.6-13.4) | 7.5 (6.6-8.1), −32.2 % |
| Prompt tokens served from cache | 85.8 % (85.1-87.0) | 90.5 % (89.9-91.2) |
| Prompt tokens prefilled | 817K (748-863K) | 547K (518-587K), −32.6 % |
| Prefill tok/s, requests with no cache hit | 6,234 (6,152-6,306) | 8,482 (8,347-8,558), +36.1 % |
| Output tok/s, one request decoding | 212 (208-216) | 219 (210-237), +3.4 % |
| Output tok/s, two requests decoding (combined) | 360 (346-370) | 348 (339-359), −3.4 % |
| Output tok/s, all decoding at the run's own batching | 254 (253-256) | 280 (273-283), +10.2 % |
| Workload wall time (min) | 15.4 (13.1-18.1) | 11.1 (9.8-12.0), −26.0 % |

Two requests decoding at once is the one place Infernix does not win: its n-gram drafting costs
host time per round. Perplexity on a fixed 409,687-token corpus (16K context): Infernix 2.30073
(INT8 KV), NInfer 2.30836.

Concurrent decode (DFlash2 K=7, no n-gram drafting, September 2026, against NInfer `d44ab584`):

| Concurrent requests | NInfer tok/s | Infernix tok/s |
|---|---:|---:|
| 1 | 198.3 | 196.6 |
| 2 | 365.8 | 380.0 |
| 4 | 649.6 | 677.4 |
| 8 | 1076.0 | 1092.2 |

## Quick start (Windows)

Prerequisites: Visual Studio 2026 (MSVC), the CUDA 13 toolkit, and FFmpeg + curl from vcpkg
(`x64-windows`, at `C:\vcpkg`); adjust the paths at the top of `build_native.bat` for your machine.

```bat
build_native.bat configure
build_native.bat build
```

The server is `build-windows\apps\Release\infernix-serve.exe`, with the FFmpeg, curl and zlib DLLs
copied next to it. The launch I use for Qwen3.8-27B on a single 32 GB RTX 5090 (stop any other
resident model first):

```bat
infernix-serve.exe qwen3_8_27b_nvfp4-nvidia.ninfer --host 127.0.0.1 --port 8080 --max-context 240000 --max-concurrency 2 --spec dflash2 --draft-tokens 7 --lm-head-draft --ngram-draft-tokens 15 --ngram-min-match 12 --kv-dtype int8 --preserve-thinking --host-context-mib 52000 --pending-timeout-ms 900000 --prefill-chunk 4096 --kv-capacity auto --vram-headroom-mib 0 --log-colours on --ngram-archive-mib 2048 --ngram-session-mib 256 --ngram-native-sessions --request-log-jsonl log.json --default-thinking-budget 16384 --thinking-budget-message "Considering the limited time available to the user, I must stop thinking now. Time to act:" --tolerant-tool-calls
```

For Qwen3.8-Flash-Next, see the [Flash-Next guide](docs/qwen3_8-flash-next.md#run):

```bat
infernix-serve.exe qwen3_8_flash_next_nvfp4_dense8.infernix --ngram-volume qwen3_8_flash_next.ngram --kv-dtype int8 --max-context 65536 --prefill-chunk 4096 --spec mtp --draft-tokens 4 --lm-head-draft
```

Add `--prefix-cache-file PATH` to keep the prefix cache across restarts, and
`--reasoning-loop conclude` to end a thinking phase that keeps repeating itself. Stop the server
with Ctrl+C twice (the first press asks for confirmation); it then saves the cache and exits.
`infernix-serve.exe --help` lists every option by category.

Running on another PC needs an RTX 50-series GPU (the build targets `sm_120a`) and an NVIDIA driver
of 580 or later (CUDA 13); no CUDA toolkit is needed. Copy the DLLs next to `infernix-serve.exe`
and install the Visual C++ redistributable if it is missing.

## Quick start (Linux)

Infernix builds and runs on 64-bit Linux too, including WSL2 (tested on Ubuntu 24.04 with CUDA 13.4
and GCC 13.3). Prerequisites: a CUDA 13 toolkit, CMake 3.28 or newer, a C++20 compiler, Ninja,
`pkg-config`, and the FFmpeg and curl development packages:

```bash
sudo apt-get install -y build-essential cmake ninja-build pkg-config libavformat-dev libavcodec-dev libavutil-dev libswscale-dev libcurl4-openssl-dev
cmake --preset release
cmake --build build -j
```

The server is `build/apps/infernix-serve` and takes the same options. Under WSL2 the GPU driver is
the Windows NVIDIA driver; do not install a Linux NVIDIA driver inside WSL, and copy artifacts into
the Linux filesystem first (reads through `/mnt/` are slow).

## Models

- **[Qwen3.8-27B-NVIDIA-NVFP4-NInferV3](https://huggingface.co/Wallawalla47/Qwen3.8-27B-NVIDIA-NVFP4-NInferV3)**
  (recommended for the 27B): [nvidia/Qwen3.8-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4)
  converted with the `qwen3_8_27b_nvfp4_nvidia` recipe, with the DFlash2 draft model and a proposal
  head for `--lm-head-draft`.
- **[Qwen3.8-27B-Quasar-NinferV3](https://huggingface.co/Wallawalla47/Qwen3.8-27B-Quasar-NinferV3)**:
  [QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4](https://huggingface.co/QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4),
  the QAT-trained NVFP4 checkpoint, with the same draft model and proposal head.
- **Qwen3.8-Flash-Next-NVFP4** (`.infernix`): coming to Hugging Face; until then, convert it from
  NVIDIA's checkpoint with the [Flash-Next guide](docs/qwen3_8-flash-next.md#convert).

Infernix loads `.infernix` and `.ninfer` artifacts (the same format); `python -m tools.convert`
writes either.

## What else is in the engine

- **Serving**: OpenAI Chat Completions and Responses, Anthropic Messages, `/v1/decide` for
  decisions read from probabilities (yes/no, choice, score, number, point, box), streaming,
  bounded FIFO admission with one to eight concurrent requests, a request log with rotation, a
  console statistics panel and a categorised `--help`.
- **Speculative decoding**: MTP, DFlash, DFlash2 (chain and tree verification) and n-gram copy
  drafting, with exact sampling.
- **Thinking control**: thinking budgets with a custom wrap-up message, reasoning effort levels, and
  an opt-in reasoning-loop guard (`--reasoning-loop stop|conclude`) for thinking that keeps
  repeating whole passages.
- **Vision**: images and video, with the vision encoder optionally offloaded to system RAM.
- **Conversion**: Hugging Face, NVIDIA ModelOpt, Quasar and GGUF sources, with recipes for Q4-Q8,
  FP8, NVFP4 and mixed artifacts.
- **Platforms**: native Windows (MSVC) and Linux builds, tuned on the RTX 5090 (`sm_120a`).

Documentation: [CLI](docs/cli.md), [serving](docs/serving.md), [perplexity](docs/perplexity.md),
[Flash-Next](docs/qwen3_8-flash-next.md), and the [documentation map](docs/README.md).

## Thanks

Infernix exists because of **[Neroued](https://github.com/Neroued)**, who created NInfer and built
the engine, artifact format and kernels this project grew from — thank you. A particular thank you
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
[gpillon](https://github.com/gpillon) (ignis),
[CaptainArni](https://github.com/CaptainArni),
David Oelfke, Fedor Suchkov, Yunado,
[IST-DASLab](https://github.com/IST-DASLab) (llmq), the authors of HyperQuant and Four Over Six,
the Qwen team for the models, and NVIDIA for the NVFP4 checkpoints.
