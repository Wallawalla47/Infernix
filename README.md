# Infernix

> **AI disclaimer:** everything added since the fork from NInfer, including most of this README, was
> written with AI (mostly Claude Opus 5.5 and Qwen3.8-27B running on Infernix). It is likely to be
> neither complete nor entirely accurate. This is hobby development.

Infernix is a C++/CUDA inference engine that serves Qwen models on one NVIDIA RTX 50-series GPU
(tuned on the RTX 5090). It began as a fork of [NInfer](https://github.com/Neroued/ninfer) by
[Neroued](https://github.com/Neroued) and reads NInfer's `.ninfer` artifacts as well as its own
`.infernix` files.

## Headline results

Release `windows-2026.10.18`, one RTX 5090 (PCIe Gen5 x8), 2026-10-19, against each engine's latest
version. Context columns: one request at a time, ~8K to ~250K tokens. Replay: an agentic coding
workload, two requests at once. Details: [Benchmarks](#benchmarks).

| Infernix vs | Time to first token | Decode tok/s | Replay: mean time to first token | Replay: wall time |
|---|---|---|---|---|
| [Strata](https://github.com/Niko1221/Strata) 0.1.45, Unsloth UD-Q4_K_XL (Infernix: NVIDIA Dense8) | 4.6-7.4× sooner | 1.5-1.7× | 4.9 s vs 17.0 s | 6.7 vs 15.9 min |
| [strata-nvfp4](https://github.com/sergqwer/strata-nvfp4), same NVIDIA NVFP4 weights | 10-19 % sooner | +6-9 % at 8K and 128K, −4 % at 250K | 4.9 s vs 10.7 s | 6.7 vs 11.5 min |
| Strata 0.1.45, same IQ3_S GGUF | Strata 20-30 % sooner | level at 8K, +9 % and +25 % at 128K and 250K | 4.9 s vs 8.6 s | 6.2 vs 9.1 min |
| [NInfer](https://github.com/Neroued/ninfer) `64d10fbf` (Windows port), 27B NVFP4 | 15 % sooner at 8K, 1.6-1.8× at 128K-230K | DFlash2 +4 % and +14 % at 8K and 128K; plain +3-4 % | 3.1 s vs 3.4 s (median 0.52 vs 1.14 s) | 6.0 vs 7.1 min |

## Why Infernix

### Qwen3.8-Flash-Next on one RTX 5090

Its 24,576 routed experts (63 GiB at NVFP4) live in pinned system RAM; the hot ones are cached in
VRAM, part of each layer's misses is computed on the CPU while the rest cross PCIe, and the 52 GB
n-gram table is read from NVMe.

- **Faster than [Strata](https://github.com/Niko1221/Strata)**: 4.6-7.4× sooner to the first token
  and 1.5-1.7× the decode speed from 8K to 250K tokens; in the agentic replay 4.9 s against 17.0 s
  to the first token and under half the wall time. Against
  [strata-nvfp4](https://github.com/sergqwer/strata-nvfp4) on the same weights, 10-19 % sooner
  ([benchmarks](#qwen38-flash-next-nvfp4-infernix-vs-strata-and-strata-nvfp4)). On the IQ3_S GGUF
  Strata prefills 20-30 % faster and Infernix decodes up to 25 % faster
  ([benchmarks](#qwen38-flash-next-iq3_s-infernix-vs-strata)).
- **At least Strata's quality**: perplexity 4.654 against 4.844 on the same token ids.
- **DFlash2 drafting** with Inco AI's drafter
  ([Qwen3.8-Flash-Next-DFlash2-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-DFlash2-Infernix),
  776 MB, any conversion): 8-11 % faster than the model's MTP drafter (26 % on sampled code), with
  VRAM for 330 more cached experts ([benchmarks](#qwen38-flash-next-dflash2-against-mtp)).
- **Long context**: 262K tokens natively, up to 1M with YaRN; a 250K prompt prefills in about 25 s.
- **Less RAM than the model?** An SSD expert tier reads what does not fit.
- **Uncensored too**:
  [Qwen3.8-Flash-Next-Uncensored-NVFP4-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-Uncensored-NVFP4-Infernix)
  keeps orcarouter's abliterated weights bit-exact (perplexity 4.772) and decodes about as fast as
  Dense8.

Setup and tuning: the [Qwen3.8-Flash-Next guide](docs/qwen3_8-flash-next.md).

### Qwen3.8-27B: more cache hits, faster prefill and decode

- **Hybrid prefix cache** for Qwen's linear-attention layers: content-addressed KV in 64-token
  blocks, recurrent-state snapshots where prompts get reused, and GPU, RAM and file tiers. In an
  agentic workload it serves 85 % of prompt tokens from cache (NInfer 79 %) and more than halves the
  median time to the first token ([benchmarks](#qwen38-27b-infernix-vs-ninfer)).
- **Faster prefill**: 1.6-1.8× NInfer's on long prompts.
- **Faster decode**: DFlash2 trees and n-gram copy drafting beside DFlash2 and MTP.
- **Smaller KV**: `vq2` and `k4v2` hold three to four times the context of INT8.

### Built for agents

- **APIs**: OpenAI Chat Completions and Responses, Anthropic Messages, streaming, and the options
  Claude Code, Qwen Code, Codex, Zed and GitHub Copilot send.
- **Robust tool calling**: the XML call forms agents emit, tolerant recovery of broken calls
  (`--tolerant-tool-calls`), calls stranded in unclosed thinking, and quoted calls kept as text.
- **Deferred tools**: `defer_loading` with regex/BM25 tool search, `tool_reference` and the
  Responses `tool_search` tool, so large MCP catalogs stay out of the prompt.
- **Constrained decoding**: JSON object and Schema output, strict tool arguments and choice, GBNF,
  regex and choice grammars, with every model and drafter.
- **Thinking control**: budgets with a wrap-up message, effort levels and a repetition guard
  (`--reasoning-loop`).
- **Concurrency**: one to eight requests, with preemption under memory pressure.
- **Resilience**: the prefix cache persists across restarts (`--prefix-cache-save-mins` saves it
  periodically); on Flash-Next a failed expert read fails only the requests running at the time.
- **`/v1/decide`**: decisions from token probabilities (yes/no, choice, score, number, point, box).

### And also

- **Speculative decoding**: MTP, DFlash, DFlash2 and n-gram drafting, with exact sampling.
- **KV formats**: `int8`, `bf16`, `fp8`, `nvfp4`, `k8v4`, `vq2` and `k4v2`.
- **Vision**: images and video; the vision encoder can stay in system RAM.
- **Conversion**: Hugging Face, NVIDIA ModelOpt, Quasar and GGUF sources; Q4-Q8, FP8, NVFP4 and
  mixed recipes.
- **Windows and Linux**: native MSVC and GCC builds, WSL2 included.

## Benchmarks

One machine: RTX 5090 (32 GB) on **PCIe Gen5 x8**, Core i9-13900K, 96 GB DDR5, Windows 11. Gen5 x16
should be faster, most of all for Flash-Next, whose expert misses cross PCIe every token. Infernix
is release [`windows-2026.10.18`](https://github.com/Wallawalla47/Infernix/releases/tag/windows-2026.10.18)
from its published zip; every engine runs as an OpenAI-compatible server driven by the same client,
which times the first token from sending the request to the first streamed token (2026-10-19).

- **Context length**: one request, greedy, 1,024 output tokens; three ~8K prompts and two each of
  ~128K and ~250K (~230K on the 27B). Two Infernix sessions bracket the other engines' (pooled);
  the others run one session in their default single-slot configuration.
- **Agentic replay** ([`bench/agentic_ab`](bench/agentic_ab/README.md)): three coding-agent sessions
  and eleven subagents with tool calls, compactions, retries and aborts, up to eight requests in
  flight, temperature 1.0, 68 requests per seed. Every engine serves two requests at once (Strata
  and strata-nvfp4 with `"parallel": 2` and `--batch-mtp`); every arm replays the same observations.
  Cells: means over seeds (range in brackets); comparison columns: the mean of per-seed differences.
  Every engine completed every request.
- Flash-Next engines use MTP and INT8 KV at 262,144 tokens; Infernix starts each session from the
  same saved expert-cache state, Strata from its bundled profiles. Every engine verifies every draft.

### Qwen3.8-Flash-Next NVFP4: Infernix vs Strata and strata-nvfp4

Infernix runs the [Dense8 conversion](#models) of NVIDIA's checkpoint (`--draft-tokens auto`);
[Strata](https://github.com/Niko1221/Strata) 0.1.45 Unsloth's `UD-Q4_K_XL` GGUF, its native format;
[strata-nvfp4](https://github.com/sergqwer/strata-nvfp4) 0.1.44.1-nvfp4.2 NVIDIA's checkpoint as its
`prepare-model` converts it (`--spec 6 --spec-min-p 0.7`, no control vector or vision, tail skip off
with NVFP4). Each engine runs with its own quantization and drafter.

Strata uses its setup's configuration for this machine (`--pcie-frac 0.00 --spec-min-p 0.70
--pool-workers 15`, `STRATA_HC_Q8=1`) except one change: the all-in-RAM mode would pin 74.8 GiB
(the limit is 76) and left no VRAM to start, so it runs `--resident-budget-gib 68`, where route tail
skip stays on.

**Context length**:

| Context | TTFT: Infernix | Strata | strata-nvfp4 | Decode tok/s: Infernix | Strata | strata-nvfp4 |
|---|---:|---:|---:|---:|---:|---:|
| ~8K | **2.29 s** | 10.51 s | 2.54 s | **157.2** | 103.9 | 143.6 |
| ~128K | **12.58 s** | 93.59 s | 15.22 s | **142.7** | 92.1 | 134.9 |
| ~250K | **25.17 s** | 139.64 s | 31.22 s | 155.2 | 93.0 | **161.0** |

An earlier Strata session that also kept `STRATA_ROUTE_TAIL_SKIP=0` by mistake reached the first
token about as soon (10.19 / 92.22 / 141.67 s) but decoded at 91.7 / 87.0 / 81.7 tok/s.

**Agentic replay** (three seeds):

| Metric | Strata | strata-nvfp4 | Infernix | vs Strata | vs strata-nvfp4 |
|---|---:|---:|---:|---:|---:|
| Mean TTFT | 17.0 s (16.5-17.8) | 10.7 s (9.3-11.6) | **4.9 s** (4.2-5.9) | −71 % | −54 % |
| Median TTFT | 9.77 s (9.64-9.92) | 6.74 s (5.41-7.53) | **3.55 s** (3.36-3.92) | −64 % | −46 % |
| 90th-percentile TTFT | 41.1 s (38.7-44.0) | 24.4 s (19.4-28.8) | **9.6 s** (8.6-11.5) | −77 % | −59 % |
| Mean TTFT, new long prompts | 31.4 s (29.4-35.4) | 10.1 s (9.1-10.8) | **6.5 s** (6.3-6.7) | −79 % | −35 % |
| Prompt tokens from cache | 72.5 % (68.8-75.0) | 68.9 % (67.5-71.0) | **84.9 %** (84.7-85.1) | +12.4 pt | +16.0 pt |
| Prompt tokens prefilled | 726K | 826K | **402K** | −44 % | −51 % |
| Prefill tok/s, no cache hit | 1,977 | 5,674 | **6,536** | 3.3× | +15 % |
| Output tok/s, one request | 94 (92-96) | 109 (105-112) | **135** (123-145) | +43 % | +24 % |
| Output tok/s, batched | 59 (57-60) | 85 (81-90) | **121** (106-128) | 2.1× | +42 % |
| Tokens per round | 2.01 | 2.17 | **2.24** | +12 % | +4 % |
| Wall time | 15.9 min (15.5-16.6) | 11.5 min (11.0-12.0) | **6.7 min** (6.4-7.0) | −58 % | −41 % |

Strata re-prefilled the whole prompt on 4-7 of the 41 continuing main-session turns per seed,
strata-nvfp4 on 6-7, Infernix on none. With two slots Strata applies no tail skip, so the mistaken
session's replay ran the same configuration (and gave the same picture: 18.5 s, 95 tok/s, 16.2 min).

### Qwen3.8-Flash-Next IQ3_S: Infernix vs Strata

Both run ISTA-DASLab's GSQ-RCO `IQ3_S` GGUF: Infernix its [bit-exact conversion](#models) with the
GGUF's IQ4_NL n-gram table, Strata 0.1.45 the GGUF with its setup's configuration (`--spec 4
--spec-min-p 0.5 --pool-workers 15`, every expert in RAM), where route tail skip (a missed expert
every token of a verify window ranks 7th or lower is skipped; faster, slightly different answers)
is on. The third column turns it off (`STRATA_ROUTE_TAIL_SKIP=0`).

**Context length**:

| Context | TTFT: Infernix | Strata | Strata, no tail skip | Decode tok/s: Infernix | Strata | Strata, no tail skip |
|---|---:|---:|---:|---:|---:|---:|
| ~8K | 2.14 s | **1.65 s** | 1.66 s | 195.9 | **198.7** | 183.7 |
| ~128K | 29.37 s | **23.92 s** | **23.92 s** | **184.9** | 169.1 | 160.3 |
| ~250K | 58.77 s | 49.04 s | **48.99 s** | **205.6** | 164.4 | 166.7 |

Strata reaches the first token 20-30 % sooner; Infernix decodes level at 8K and 9 % and 25 % faster
at 128K and 250K. Tail skip off cost Strata 8 % and 5 % of decode at 8K and 128K.

**Agentic replay** (three seeds):

| Metric | Strata | Strata, no tail skip | Infernix | vs Strata |
|---|---:|---:|---:|---:|
| Mean TTFT | 8.6 s (7.7-9.6) | 9.0 s (8.6-9.3) | **4.9 s** (4.0-5.7) | −43 % |
| Median TTFT | 5.29 s (5.20-5.36) | 5.05 s (3.95-5.90) | **2.70 s** (2.41-2.86) | −49 % |
| 90th-percentile TTFT | 19.5 s (16.3-21.6) | 20.9 s (19.1-21.8) | **13.1 s** (8.2-16.8) | −31 % |
| Mean TTFT, new long prompts | 14.3 s (12.8-15.2) | 13.2 s (11.9-13.8) | **12.7 s** (12.4-12.8) | −11 % |
| Prompt tokens from cache | 73.5 % (72.7-74.3) | 72.0 % (70.6-73.4) | **84.9 %** (84.7-85.1) | +11.4 pt |
| Prompt tokens prefilled | 700K | 742K | **401K** | −43 % |
| Prefill tok/s, no cache hit | 4,695 | **4,731** | 3,722 | −21 % |
| Output tok/s, one request | 144 (142-146) | 144 (143-147) | **164** (160-167) | +14 % |
| Output tok/s, batched | 107 (103-110) | 103 (100-110) | **158** (152-167) | +48 % |
| Tokens per round | 2.19 | 2.13 | **2.40** | +10 % |
| Wall time | 9.1 min (8.6-9.4) | 9.3 min (8.9-9.6) | **6.2 min** (5.8-6.6) | −32 % |

Infernix serves more of each prompt from cache (Strata re-prefilled whole prompts on 4-6 of 41
continuing turns, Infernix on none), though it prefills an uncached prompt 21 % slower. With two
slots Strata applies no tail skip, so its two columns ran the same computation.

### Qwen3.8-Flash-Next: DFlash2 against MTP

One request with each drafter, both `--draft-tokens auto --lm-head-draft`: the model's MTP drafter
(`--spec mtp`) and Inco AI's DFlash2
([Qwen3.8-Flash-Next-DFlash2-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-DFlash2-Infernix),
`--spec dflash2 --draft-model FILE`). Prompts: 24 slices of 1,280 tokens of Infernix's sources
(code) and Markdown docs at `e59fa1c7`, each continued for 384 tokens, greedy or sampled
(temperature 1.0, top-k 20, top-p 0.95); INT8 KV; `infernix_decode_quality_gen` (2026-10-20). Cells:
decode tok/s, the mean of the runs (Dense8 code: DFlash2 209-217, MTP 186-203; other repeats within
4 %).

| Conversion | Text | Plain | MTP | **DFlash2** | DFlash2 vs MTP |
|---|---|---:|---:|---:|---:|
| NVIDIA Dense8 | code, greedy | 120 | 198 | **213** | +8 % |
| NVIDIA Dense8 | code, sampled | | 162 | **204** | +26 % |
| NVIDIA Dense8 | docs, greedy | 121 | 164 | **179** | +9 % |
| NVIDIA Dense8 | docs, sampled | 122 | 145 | **161** | +11 % |
| GSQ-RCO IQ3_S | code, greedy | 121 | 233 | **258** | +11 % |
| GSQ-RCO IQ3_S | docs, greedy | | 192 | **210** | +9 % |
| GSQ-RCO IQ3_S | docs, sampled | | 167 | **183** | +10 % |

DFlash2 drafts a block of seven in one pass and reports each draft's probability; `auto` verifies
only the drafts worth their verification columns, which cost about a third of a plain round each
in expert misses here (all seven every round was slower than MTP). It also takes less VRAM:
9,070 cached experts against MTP's 8,740 on Dense8 (IQ3_S 12,432 against 12,288). Greedy output is
the same with either drafter; sampling keeps its distribution.

### Qwen3.8-27B: Infernix vs NInfer

Both serve NInfer's [Qwen3.8-27B-nvfp4-NInfer](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer)
with INT8 KV and the same flags; NInfer is `master` at
[`64d10fbf`](https://github.com/Neroued/ninfer/commit/64d10fbf9) with a Windows port (Win32 I/O,
an IOCP read queue, MSVC fixes).

**Context length** (DFlash2 at `--max-context 228000`, NInfer's largest with DFlash2 on 32 GB, so
no ~230K row for it; time to first token and plain decode at 240000):

| Context | TTFT: Infernix | NInfer | DFlash2 tok/s: Infernix | NInfer | Plain tok/s: Infernix | NInfer |
|---|---:|---:|---:|---:|---:|---:|
| ~8K | **0.65 s** | 0.76 s | **207.7** | 199.9 | **82.6** | 80.1 |
| ~128K | **18.75 s** | 29.51 s | **209.4** | 183.3 | **69.6** | 67.1 |
| ~230K | **47.30 s** | 83.24 s | 175.2 | — | **61.4** | 59.0 |

DFlash2 speed follows each engine's accepted drafts (the two 128K prompts: NInfer 206 and 161,
Infernix 218 and 201 tok/s). NInfer kept the 8K prompt cached through the 128K prompts (repeat
0.15 s); Infernix had evicted it (0.76 s).

**Agentic replay** (five seeds; `--max-context 200000`, NInfer's largest with two requests and
DFlash2; Infernix also drafts n-grams):

| Metric | NInfer | Infernix | vs NInfer |
|---|---:|---:|---:|
| Mean TTFT | 3.4 s (2.7-3.8) | **3.1 s** (2.8-3.3) | −7 % |
| Median TTFT | 1.14 s (0.95-1.35) | **0.52 s** (0.45-0.61) | −54 % |
| 90th-percentile TTFT | 10.0 s (7.4-11.8) | 10.0 s (8.8-11.4) | +3 % |
| Mean TTFT, continuing turns | **2.57 s** (1.77-3.26) | 2.93 s (2.62-3.39) | +18 % |
| Mean TTFT, new long prompts | 10.2 s (9.4-10.9) | **6.4 s** (6.1-6.9) | −37 % |
| Prompt tokens from cache | 79.0 % (78.7-79.4) | **85.3 %** (85.0-85.6) | +6.3 pt |
| Prompt tokens prefilled | 562K | **396K** | −30 % |
| Prefill tok/s, no cache hit | 4,986 | **7,176** | +44 % |
| Output tok/s, one request | **194** (187-199) | 192 (187-198) | −1 % |
| Output tok/s, batched | 229 (213-241) | **254** (238-274) | +11 % |
| Tokens per round | **3.23** | 3.17 | −2 % |
| Wall time | 7.1 min (6.2-8.0) | **6.0 min** (5.6-6.6) | −13 % |

Continuing turns waited longer for a lane behind other sessions' replies, whose sampled lengths
differ between the arms (2.63 s against 2.04 s of queueing per turn); without the queue Infernix
was sooner on 249 of 255 turns (0.29 s against 0.52 s). Quality is level: perplexity 4.468 against
NInfer's 4.479 on the 1.04M-token corpus (16K context, INT8 KV; NInfer `81c8ce09`).

## Quick start (Windows)

Prerequisites: Visual Studio 2026 (or 2022, MSVC 14.44) with its C++ CMake tools (they bring Ninja),
the CUDA 13 toolkit, and FFmpeg, libjpeg-turbo and curl from vcpkg (`x64-windows`, at `C:\vcpkg`;
`vcpkg install libjpeg-turbo:x64-windows` adds libjpeg-turbo to an existing tree); adjust the paths at
the top of `build_native.bat`. It builds with Ninja Multi-Config (about 30 % faster than the Visual
Studio generator); a `build-windows` tree from the Visual Studio generator is configured afresh on its
next build.

```bat
build_native.bat configure
build_native.bat build
```

The server is `build-windows\apps\Release\infernix-serve.exe`, with its DLLs beside it.
Qwen3.8-Flash-Next ([guide](docs/qwen3_8-flash-next.md#run)):

```bat
infernix-serve.exe Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix-00001-of-00003.infernix --ngram-volume Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram --kv-dtype int8 --max-context 262144 --spec dflash2 --draft-model Qwen3.8-Flash-Next-DFlash2-Infernix.infernix --draft-tokens auto --lm-head-draft --max-concurrency 2
```

The drafter is [Qwen3.8-Flash-Next-DFlash2-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-DFlash2-Infernix);
without it, `--spec mtp --draft-tokens auto --lm-head-draft` uses the model's own MTP drafter, about
10 % slower. The uncensored conversion takes the same options, n-gram volume and drafter:

```bat
infernix-serve.exe Qwen3.8-Flash-Next-Uncensored-NVFP4-Infernix-00001-of-00003.infernix --ngram-volume Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram --kv-dtype int8 --max-context 262144 --spec dflash2 --draft-model Qwen3.8-Flash-Next-DFlash2-Infernix.infernix --draft-tokens auto --lm-head-draft --max-concurrency 2
```

Qwen3.8-27B on a 32 GB RTX 5090:

```bat
infernix-serve.exe qwen3_8_27b_nvfp4-nvidia.ninfer --max-context 240000 --max-concurrency 2 --spec dflash2 --draft-tokens auto --lm-head-draft --ngram-draft-tokens auto --kv-dtype int8 --preserve-thinking --host-context-mib 52000 --prefill-chunk 4096 --kv-capacity auto --tolerant-tool-calls
```

Ornith-1.5-122B (experts in about 70 GiB of RAM; [guide](docs/qwen3_5-122b.md)):

```bat
infernix-serve.exe Ornith-1.5-122B-NVFP4-Infernix-00001-of-00003.infernix --max-context 262144 --max-concurrency 2 --kv-dtype int8 --spec mtp --draft-tokens auto --host-context-mib 4096
```

The uncensored conversion (`Ornith-1.5-122B-UNCENSORED-NVFP4-Infernix-00001-of-00003.infernix`, preliminary
support) takes the same options.

**Recommended settings per model** (each with `--kv-dtype int8`; the drafter flags are the fastest
measured overall, and `auto` picks the draft length):

| Model | Drafting | What `--draft-tokens auto` does there |
|---|---|---|
| Qwen3.8-Flash-Next, every conversion | `--spec dflash2 --draft-model Qwen3.8-Flash-Next-DFlash2-Infernix.infernix --draft-tokens auto --lm-head-draft`, plus `--ngram-volume` (the conversion's own or the FP8 volume, [models](#qwen38-flash-next)) | verifies the drafts of a 7-token block that their calibrated probabilities make worthwhile; without the drafter, `--spec mtp --draft-tokens auto --lm-head-draft` (0-7 drafts a round, about 10 % slower) |
| Qwen3.8-27B NVFP4 (NVIDIA, Quasar, Uncensored, Swift, NInfer's nvfp4) | `--spec dflash2 --draft-tokens auto --lm-head-draft --ngram-draft-tokens auto` | a 15-draft block verified at the width its probabilities favour (batched rounds: 7), n-gram copies of up to 63 tokens |
| Qwen3.8-27B groupwise (NInfer's `Qwen3.8-27B-NInfer`) | `--spec dflash2 --draft-tokens auto --lm-head-draft` | 7 drafts every round (its per-round policies were no faster); at one request 54 % faster than MTP's 3 on code, 40 % on docs |
| Qwen3.6-35B-A3B | `--spec mtp --draft-tokens auto --lm-head-draft` | the per-round MTP length, 1-5 (DFlash `auto` is faster only on code at one request) |
| Qwen3.6-27B (NVFP4, groupwise) | `--spec mtp --draft-tokens auto --lm-head-draft` | the per-round MTP length where its configuration registers `adaptive-mtp` (startup reports it), else 3 every round |
| Ornith-1.5-122B (and its UNCENSORED conversion) | `--spec mtp --draft-tokens auto` | 2 confidence-gated drafts: 14 % faster than plain decode at one request, about 6 % slower while two decode together; DFlash2's `auto` decodes plain |

Flash-Next keeps every routed expert in RAM by default (`--expert-ram-layout full`);
`--expert-ram-layout exclusive` keeps only the experts VRAM does not hold, about 15 GiB less RAM on the
NVFP4 conversions and 7 GiB on IQ3_S, at 1-2 % (NVFP4) and 12 % (IQ3_S) slower decode
([guide](docs/qwen3_8-flash-next.md)).

`--prefix-cache-file PATH` keeps the prefix cache across restarts (Ctrl+C twice saves it and exits).
`infernix-serve.exe --help` lists every option; `--config FILE` reads them from a TOML file
([configuration file](docs/serving.md#configuration-file), example
[`infernix.example.toml`](infernix.example.toml)).

Another PC needs an RTX 50-series GPU and NVIDIA driver 580 or later (no CUDA toolkit): copy the
DLLs beside `infernix-serve.exe` and install the Visual C++ redistributable if missing.
`packaging\windows\package_release.ps1` packs the server, DLLs and licences into a release zip
([third-party notices](packaging/windows/THIRD_PARTY_NOTICES.md)).

## Quick start (Linux)

64-bit Linux and WSL2 (tested on Ubuntu 24.04, CUDA 13.4, GCC 13.3). Prerequisites: CUDA 13, CMake
3.28+, a C++20 compiler, Ninja, `pkg-config`, and the FFmpeg, libjpeg-turbo and curl development
packages:

```bash
sudo apt-get install -y build-essential cmake ninja-build pkg-config libavformat-dev libavcodec-dev libavutil-dev libswscale-dev libjpeg-turbo8-dev libcurl4-openssl-dev
cmake --preset release
cmake --build build -j
```

The server is `build/apps/infernix-serve`, with the same options. Under WSL2 use the Windows NVIDIA
driver (install none inside WSL) and copy models into the Linux filesystem; `/mnt/` reads are slow.

## Models

Each repository's model card gives its files, requirements, recommended command, quality and speed.
`python -m tools.convert` converts your own checkpoints of these architectures
([weight conversion](docs/weight-conversion.md)).

### Qwen3.8-Flash-Next

| Repository | Converted from | What it is |
|---|---|---|
| **[Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix)** (**recommended**) | [nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) | NVIDIA's NVFP4 experts and FP8 n-gram table bit-exact, dense projections in 8 bits: about 22 % faster than the bit-exact conversion at the same quality (perplexity 4.654 against 4.666); the one [benchmarked](#qwen38-flash-next-nvfp4-infernix-vs-strata-and-strata-nvfp4). |
| [Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix) | [nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) | Every weight exactly as NVIDIA stores it. |
| [Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix) | [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) (`IQ3_S`) | Every GGUF tensor bit-exact in exact integer arithmetic, plus MTP and vision. Fastest decode (146 tok/s plain, 260 with MTP, against Dense8's 110 and 143), experts in 46.9 GiB of RAM; slower long prefill (4,324 against 10,933 tok/s at 131K). |
| [Qwen3.8-Flash-Next-FP8-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-FP8-Infernix) | [Qwen/Qwen3.8-Flash-Next-FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8) | Qwen's FP8 checkpoint bit-exact with its BF16 n-gram table: the most accurate and the slowest (about 44 tok/s plain; its 112.6 GiB of experts need the SSD tier on 96 GB). |
| [Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix) | [nvidia/Qwen3.8-Flash-Next-FP4-Local](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local) | NVIDIA's quantization-aware-distilled FP4 checkpoint bit-exact (NVFP4 experts, MXFP8 dense, its own NVFP4 n-gram volume), 16-bit expert activations as NVIDIA serves it (`--expert-a4-activations`: 4-bit, changes the output). Perplexity 3.4865 against FP8's 3.4854 (quick corpus). |
| [Qwen3.8-Flash-Next-Uncensored-NVFP4-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-Uncensored-NVFP4-Infernix) | [orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4) | Abliterated (refusals removed; see its card): weight-only NVFP4 experts and FP8 projections bit-exact, other dense classes 8-bit as Dense8. Perplexity 4.772. |
| [Swift-1.5-Qwen3.8-Flash-Next-NVFP4-Dense8-Infernix](https://huggingface.co/Wallawalla47/Swift-1.5-Qwen3.8-Flash-Next-NVFP4-Dense8-Infernix) | [ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4) | UkisAI's fine-tune that thinks in fewer tokens, converted like Dense8. Smoke-tested; quality and speed not measured. |

Every conversion can draft with **[Qwen3.8-Flash-Next-DFlash2-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-DFlash2-Infernix)**
(**recommended**, 776 MB), Inco AI's [DFlash2 drafter](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2)
in 8 bits (`--spec dflash2 --draft-model FILE --draft-tokens auto`), 8-11 % faster than MTP
([benchmarks](#qwen38-flash-next-dflash2-against-mtp)). On the 1M-token perplexity corpus (4,096
context, INT8 KV) FP8 scores 3.487, Dense8 3.508 and IQ3_S 3.522.

Every conversion also needs a PLE n-gram volume (`--ngram-volume`) on an NVMe drive. Any of the
three runs any conversion; **the FP8 volume is recommended**:

| Volume | Format | Bytes | Repository |
|---|---|---:|---|
| [`Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram`](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix/blob/main/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram) (**recommended**) | FP8, one scale (NVIDIA's) | 52,429,058,048 | [NVIDIA-NVFP4-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix), copy in [Dense8](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix) |
| [`Qwen3.8-Flash-Next-BF16.ngram`](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-FP8-Infernix/blob/main/Qwen3.8-Flash-Next-BF16.ngram) | BF16 (Qwen's) | 109,227,196,416 | [FP8-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-FP8-Infernix) |
| [`Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix.ngram`](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix/blob/main/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix.ngram) | IQ4_NL, 4.5 bits (the GGUF's) | 29,127,258,112 | [GSQ-RCO-IQ3_S-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix) |

The NVFP4 conversions were measured with the FP8 table, IQ3_S with its IQ4_NL table, FP8 with BF16;
the uncensored and Swift repositories carry none
([n-gram table formats](docs/qwen3_8-flash-next.md#n-gram-table-formats)). IQ3_S and FP8 need
Infernix `windows-2026.10.18` or later, the uncensored conversion `windows-2026.10.09.2`, the
DFlash2 drafter a build whose `--help` lists `--draft-model`; older builds accept only an
artifact's own n-gram table or BF16.

### Qwen3.8-27B

| Repository | Converted from | What it is |
|---|---|---|
| **[Qwen3.8-27B-NVIDIA-NVFP4-NInferV3](https://huggingface.co/Wallawalla47/Qwen3.8-27B-NVIDIA-NVFP4-NInferV3)** (**recommended**) | [nvidia/Qwen3.8-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4) | NVIDIA's NVFP4 checkpoint with a DFlash2 drafter and a proposal head for `--lm-head-draft`. |
| [Qwen3.8-27B-Quasar-NVFP4-NInferV3](https://huggingface.co/Wallawalla47/Qwen3.8-27B-Quasar-NVFP4-NInferV3) | [QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4](https://huggingface.co/QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4) | The QAT-trained NVFP4 model, same drafter and head. |
| [Qwen3.8-27B-Uncensored-NVFP4-NInferV3](https://huggingface.co/Wallawalla47/Qwen3.8-27B-Uncensored-NVFP4-NInferV3) | [orcarouter/Qwen3.8-27B-Uncensored-NVFP4](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-NVFP4) | Abliterated (safety alignment largely removed; see its card), same drafter and head. |
| [Swift-Qwen3.8-27B-NVFP4-NInferV3](https://huggingface.co/Wallawalla47/Swift-Qwen3.8-27B-NVFP4-NInferV3) | [ukisai/Swift-Qwen3.8-27B-NVFP4](https://huggingface.co/ukisai/Swift-Qwen3.8-27B-NVFP4) | UkisAI's token-efficient fine-tune, converted like the NVIDIA artifact. Smoke-tested; quality and speed not measured. |
| [neroued/Qwen3.8-27B-nvfp4-NInfer](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | NInfer's official NVFP4 artifact, the one [benchmarked](#qwen38-27b-infernix-vs-ninfer). |
| [neroued/Qwen3.8-27B-NInfer](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | NInfer's official `groupwise-int` artifact. |

### Qwen3.6

| Repository | Converted from | What it is |
|---|---|---|
| [neroued/Qwen3.6-27B-NInfer](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) | [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) | NInfer's official `groupwise-int` artifact. |
| [neroued/Qwen3.6-27B-nvfp4-NInfer](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer) | [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) | NInfer's official NVFP4 artifact. |
| [neroued/Qwen3.6-35B-A3B-NInfer](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) | [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) | NInfer's official `groupwise-int` MoE artifact, with a DFlash drafter. |

The `neroued` artifacts are NInfer's (loaded unchanged); their model cards are versioned in
[`model-cards/`](model-cards/) with the others.

### Qwen3.5-122B-A10B

A 122B mixture-of-experts model on one 32 GB GPU: the routed experts (60.75 GiB) live in host RAM
behind a VRAM expert cache, as for Flash-Next ([guide](docs/qwen3_5-122b.md)).

| Repository | Converted from | What it is |
|---|---|---|
| **[Ornith-1.5-122B-NVFP4-Infernix](https://huggingface.co/Wallawalla47/Ornith-1.5-122B-NVFP4-Infernix)** | [ornith-ai/Ornith-1.5-122B-NVFP4](https://huggingface.co/ornith-ai/Ornith-1.5-122B-NVFP4) | Every computed weight as the checkpoint stores it (NVFP4 experts, FP8 projections), plus the MTP drafter, the [DFlash2 drafter](https://huggingface.co/incoai/Ornith-1.5-122B-DFlash2) re-quantized, vision, a Q4 proposal head and the card's sampling defaults. About 64-97 tok/s plain decode depending on the text, 6,500 tok/s prefill at 32K; `--spec mtp --draft-tokens auto` (2 gated drafts) decodes 14 % faster at one request ([performance](docs/qwen3_5-122b.md#performance-rtx-5090-one-request-bf16-kv-default-route)). |
| **[Ornith-1.5-122B-UNCENSORED-NVFP4-Infernix](https://huggingface.co/Wallawalla47/Ornith-1.5-122B-UNCENSORED-NVFP4-Infernix)** | [dealignai/Ornith-1.5-122B-UNCENSORED-NVFP4](https://huggingface.co/dealignai/Ornith-1.5-122B-UNCENSORED-NVFP4) | The refusal-removed export, every computed weight as it stores it (NVFP4 experts; per-row FP8 projections, shared experts and head), with the same drafters, vision, proposal head and its card's sampling defaults (temperature 0.6, presence penalty 0). Same prefill and decode as the original at the same settings ([guide](docs/qwen3_5-122b.md#ornith-15-122b-uncensored-nvfp4)). **Support is preliminary: it runs and is verified, but optimisation is still in progress, so expect speeds to change.** |

## Documentation

[CLI](docs/cli.md), [serving](docs/serving.md), [Qwen3.8-Flash-Next](docs/qwen3_8-flash-next.md),
[Qwen3.5-122B-A10B](docs/qwen3_5-122b.md), [perplexity](docs/perplexity.md),
[performance](docs/performance.md) and the [documentation map](docs/README.md).

## Contributing

Issues and pull requests of any kind are welcome: bug reports, questions, measurements on other GPUs
and machines (PCIe x16 especially), fixes, features and documentation.

## Support

Infernix is developed with Claude Code. To help pay for that subscription, you can
[sponsor me on GitHub](https://github.com/sponsors/Wallawalla47).

## Thanks

Infernix exists because of **[Neroued](https://github.com/Neroued)**, who created NInfer and built
the engine, artifact format and kernels this project grew from — thank you; please support NInfer's
creator [on Ko-fi](https://ko-fi.com/neroued). Thank you also to
**[Niko1221](https://github.com/Niko1221)**, creator of [Strata](https://github.com/Niko1221/Strata),
whose engine set the bar for Qwen3.8-Flash-Next on one GPU, whose transcription of the model's
mathematics and published measurements guided this implementation, and whose community's ideas
(including wangmeng's reasoning-loop recovery and eddoursul's host-thread placement) shaped several
features here.

And to everyone whose issues, pull requests, forks and research Infernix has drawn on:
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
