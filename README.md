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

- **Faster than [Strata](https://github.com/Niko1221/Strata)** on the same machine: 4.6-6.4× sooner
  to the first token and 1.6-1.8× the decode speed from 8K to 250K tokens of context. In a replayed
  agentic coding workload it reaches the first token in 5.9 s on average against 32.9 s and
  finishes in about a third of the time ([benchmarks](#qwen38-flash-next-nvfp4-infernix-vs-strata)).
- **At least Strata's quality**: perplexity 4.654 against 4.844 on the same token ids.
- **Long context**: 262K tokens natively and up to 1M with YaRN; a 250K-token prompt prefills in
  about 34 s.
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
  host RAM, and a file that survives restarts). In an agentic workload it serves 90 % of prompt
  tokens from cache and halves the average time to the first token against NInfer
  ([benchmarks](#qwen38-27b-infernix-vs-ninfer)).
- **Faster prefill**: 1.3-1.5× NInfer's on long prompts (FlashAttention-2 style INT8 prompt
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
and is driven by the same client. The agentic replay
([`bench/agentic_ab`](bench/agentic_ab/README.md)) interleaves three coding-agent sessions and
eleven subagents with tool calls, compactions, retries and aborts, up to eight requests in flight,
sampling at temperature 1.0; cells are means over seeds with the range in brackets.

### Qwen3.8-Flash-Next-NVFP4: Infernix vs Strata

Infernix runs the [Dense8 conversion](#models) of NVIDIA's checkpoint; Strata runs Unsloth's
`Qwen3.8-Flash-Next-UD-Q4_K_XL` GGUF, its native format. Both use the model's MTP drafter and INT8
KV at 262,144 tokens, so this compares each engine as it is run in practice, each with its own
quantization and drafter.

**Context length** (one request at a time, greedy, 1,024 output tokens; Infernix `3b35ccf9`,
Strata 0.1.40):

| Context | Time to first token, Infernix | Strata | Decode tok/s, Infernix | Strata |
|---|---:|---:|---:|---:|
| ~8K | **1.9 s** | 12.3 s | **138.8** | 86.4 |
| ~128K | **17.9 s** | 103.3 s | **136.2** | 77.2 |
| ~250K | **34.2 s** | 158.1 s | **153.3** | 95.8 |

**Agentic replay** (three seeds, 68 requests each; Infernix `63c672e4`, Strata 0.1.41):

| Metric | Strata | Infernix | Change |
|---|---:|---:|---:|
| Average time to first token | 32.9 s (30.9-36.8) | **5.9 s** (5.5-6.2) | −82 % |
| 90th-percentile time to first token | 67.8 s (62.0-75.1) | **13.3 s** (12.9-13.7) | −80 % |
| Prompt tokens served from cache | 53.8 % (49.3-60.4) | **84.6 %** (84.3-84.8) | +31 points |
| Prefill tok/s, requests with no cache hit | 1,423 | **5,879** | 4.1× |
| Output tok/s, one request decoding | 86 (83-88) | **125** (121-128) | +45 % |
| Output tok/s, at the run's own batching | 86 (83-88) | **143** (140-147) | +67 % |
| Tokens per round (draft acceptance) | **2.08** | 1.97 | −6 % |
| Workload wall time | 22.3 min (19.6-24.5) | **7.8 min** (6.5-9.0) | −65 % |

Strata's drafts are accepted slightly more often; Infernix decodes faster because each round takes
about a third less time. Both verify every draft, so drafting changes speed, not output.

### Qwen3.8-27B: Infernix vs NInfer

Both serve NInfer's official
[Qwen3.8-27B-nvfp4-NInfer](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) artifact with
DFlash2 drafting and the same flags (NInfer `master` at `68c54356` with the Windows port; Infernix
`a436fb5a`, which also uses n-gram drafting).

**Agentic replay** (three seeds, 130 requests each):

| Metric | NInfer | Infernix | Change |
|---|---:|---:|---:|
| Average time to first token | 5.7 s (4.9-7.0) | **2.7 s** (2.3-3.0) | −52 % |
| Prompt tokens served from cache | 83.7 % | **90.4 %** | +6.7 points |
| Prompt tokens prefilled | 982K | **569K** | −42 % |
| Prefill tok/s, requests with no cache hit | 6,127 | **8,254** | +35 % |
| Output tok/s, one request decoding | 199 (193-202) | **235** (220-246) | +18 % |
| Output tok/s, at the run's own batching | 249 (239-257) | **282** (276-290) | +13 % |
| Workload wall time | 16.9 min (15.1-18.9) | **12.5 min** (9.9-15.7) | −25 % |

Wall time also depends on how much each run wrote: on one seed Infernix wrote 31 % more output and
finished 4 % later; on the other two it wrote 24-30 % less and finished 37-41 % sooner.

Single requests with long prompts prefill 1.34-1.51× faster from 128K tokens, and decode without
speculation is level. Quality is level: on the repository's 1.04M-token perplexity corpus (16K
context, INT8 KV cache) Infernix scores 4.468 against 4.479 for NInfer (`master` at `81c8ce09` with
the Windows port).

## Quick start (Windows)

Prerequisites: Visual Studio 2026 (MSVC), the CUDA 13 toolkit, and FFmpeg + curl from vcpkg
(`x64-windows`, at `C:\vcpkg`); adjust the paths at the top of `build_native.bat` for your machine.

```bat
build_native.bat configure
build_native.bat build
```

The server is `build-windows\apps\Release\infernix-serve.exe`, with the FFmpeg, curl and zlib DLLs
next to it. Qwen3.8-Flash-Next (see the [guide](docs/qwen3_8-flash-next.md#run)):

```bat
infernix-serve.exe Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix-00001-of-00003.infernix --ngram-volume Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram --kv-dtype int8 --max-context 262144 --prefill-chunk 4096 --spec mtp --draft-tokens 4 --lm-head-draft --max-concurrency 2
```

The uncensored Flash-Next takes the same options and the same n-gram volume:

```bat
infernix-serve.exe Qwen3.8-Flash-Next-Uncensored-NVFP4-Infernix-00001-of-00003.infernix --ngram-volume Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram --kv-dtype int8 --max-context 262144 --prefill-chunk 4096 --spec mtp --draft-tokens 4 --lm-head-draft --max-concurrency 2
```

Qwen3.8-27B on a 32 GB RTX 5090:

```bat
infernix-serve.exe qwen3_8_27b_nvfp4-nvidia.ninfer --max-context 240000 --max-concurrency 2 --spec dflash2 --draft-tokens 7 --lm-head-draft --ngram-draft-tokens 15 --ngram-min-match 12 --kv-dtype int8 --preserve-thinking --host-context-mib 52000 --prefill-chunk 4096 --kv-capacity auto --tolerant-tool-calls
```

Add `--prefix-cache-file PATH` to keep the prefix cache across restarts. Stop the server with
Ctrl+C twice; it saves the cache and exits. `infernix-serve.exe --help` lists every option by
category.

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

All three Flash-Next conversions use the same 52 GB n-gram volume,
[`Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram`](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix/blob/main/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram)
(the uncensored repo does not carry its own copy). `python -m tools.convert` converts
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
