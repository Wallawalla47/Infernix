# Infernix

> **AI disclaimer:** Everything added to this engine since it forked from NInfer, including most of
> this README, was written with AI (mostly Claude Opus 5.5, Qwen3.8-27B running on Infernix, plus a
> few other AI systems I’ve been testing). It is likely to be neither complete nor entirely
> accurate. This is hobby development.

Infernix is a C++/CUDA inference engine that serves Qwen models on a single NVIDIA
RTX 50-series GPU (tuned on the RTX 5090). It began as a fork of
[NInfer](https://github.com/Neroued/ninfer) by [Neroued](https://github.com/Neroued) and has grown
into its own engine; it reads NInfer's `.ninfer` artifacts as well as its own `.infernix` files.

## Headline results

Release `windows-2026.10.18` on one RTX 5090 (PCIe Gen5 x8), measured on 2026-10-19 against each
engine's latest version. The context-length columns cover one request at a time with ~8K to ~250K
tokens of context; the replay is an agentic coding workload with two requests served at once.
Method, configurations and full results: [Benchmarks](#benchmarks).

| Infernix vs | Time to first token, by context | Decode tok/s, by context | Replay: average time to first token | Replay: wall time |
|---|---|---|---|---|
| [Strata](https://github.com/Niko1221/Strata) 0.1.45 on Unsloth's UD-Q4_K_XL (Infernix on NVIDIA Dense8) | 4.6-7.4× sooner | 1.5-1.7× as fast | 4.9 s against 17.0 s | 6.7 min against 15.9 min |
| [strata-nvfp4](https://github.com/sergqwer/strata-nvfp4) on the same NVIDIA NVFP4 weights | 10-19 % sooner | 6-9 % faster at 8K and 128K, 4 % slower at 250K | 4.9 s against 10.7 s | 6.7 min against 11.5 min |
| Strata 0.1.45 on the same IQ3_S GGUF | Strata 20-30 % sooner | level at 8K, 9 % and 25 % faster at 128K and 250K | 4.9 s against 8.6 s | 6.2 min against 9.1 min |
| [NInfer](https://github.com/Neroued/ninfer) `64d10fbf` + Windows port on the official 27B NVFP4 | 15 % sooner at 8K, 1.6× and 1.8× sooner at 128K and 230K | DFlash2 4 % and 14 % faster at 8K and 128K; plain 3-4 % faster | 3.1 s against 3.4 s (median 0.52 s against 1.14 s) | 6.0 min against 7.1 min |

## Why Infernix

### Qwen3.8-Flash-Next on one RTX 5090

Qwen3.8-Flash-Next has 24,576 routed experts (63 GiB at NVFP4), far more than a 32 GB GPU holds.
Infernix keeps the experts in pinned system RAM, caches the hot ones in VRAM, computes part of each
layer's cache misses on the CPU while the rest cross PCIe, and reads the 52 GB n-gram embedding
table from an NVMe drive.

- **Faster than [Strata](https://github.com/Niko1221/Strata)** on the same machine: 4.6-7.4× sooner
  to the first token and 1.5-1.7× the decode speed from 8K to 250K tokens of context. In a replayed
  agentic coding workload it reaches the first token in 4.9 s on average against 17.0 s and
  finishes in under half the time. Against the
  [strata-nvfp4](https://github.com/sergqwer/strata-nvfp4) fork on the same NVIDIA weights it
  reaches the first token 10-19 % sooner, and in the replay 4.9 s against 10.7 s
  ([benchmarks](#qwen38-flash-next-nvfp4-infernix-vs-strata-and-strata-nvfp4)). On the IQ3_S GGUF
  Strata prefills 20-30 % faster and Infernix decodes up to 25 % faster
  ([benchmarks](#qwen38-flash-next-iq3_s-infernix-vs-strata)).
- **At least Strata's quality**: perplexity 4.654 against 4.844 on the same token ids.
- **DFlash2 drafting**: Inco AI's DFlash2 drafter, a separate 776 MB download
  ([Qwen3.8-Flash-Next-DFlash2-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-DFlash2-Infernix)) that runs beside any Flash-Next conversion,
  decodes 8-11 % faster than the model's own MTP drafter with `--draft-tokens auto` (26 % on
  sampled code), and leaves
  VRAM for 330 more cached experts on NVIDIA Dense8
  ([benchmarks](#qwen38-flash-next-dflash2-against-mtp)).
- **Long context**: 262K tokens natively and up to 1M with YaRN; a 250K-token prompt prefills in
  about 25 s.
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
  tokens from cache against NInfer's 79 % and more than halves the median time to the first token
  ([benchmarks](#qwen38-27b-infernix-vs-ninfer)).
- **Faster prefill**: 1.6-1.8× NInfer's on long prompts (FlashAttention-2 style INT8 prompt
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
expert-cache misses cross PCIe on every token. Infernix is release
[`windows-2026.10.18`](https://github.com/Wallawalla47/Infernix/releases/tag/windows-2026.10.18), run
from its published zip. Each engine runs as its own OpenAI-compatible server and is driven by the
same client, and every time to first token is measured the same way, on the client: from sending
the request to the first streamed token. All were measured on 2026-10-19.

- **Context length**: one request at a time, greedy, 1,024 output tokens; three ~8K prompts and two
  each of ~128K and ~250K (~230K on the 27B), means per size. Two Infernix sessions bracket the
  other engines' sessions (Infernix, the others, Infernix) and are pooled; each other engine runs
  one session, in its default single-slot configuration.
- **Agentic replay** ([`bench/agentic_ab`](bench/agentic_ab/README.md)): three coding-agent sessions
  and eleven subagents with tool calls, compactions, retries and aborts, up to eight requests in
  flight, sampled at temperature 1.0, at half length (68 requests per seed). Every engine serves two
  requests at once (Strata and strata-nvfp4 with `"parallel": 2` and `--batch-mtp`), and every arm
  of a seed replays the same observations. Cells are means over the seeds with the range in
  brackets, comparison columns the mean of the per-seed differences; every engine completed every
  request.
- Every Flash-Next engine uses the model's MTP drafter and INT8 KV at 262,144 tokens. Infernix
  starts each session from the same saved expert-cache state; Strata and strata-nvfp4 from their
  bundled expert profiles. Every engine verifies every draft, so drafting changes speed, not output.

### Qwen3.8-Flash-Next NVFP4: Infernix vs Strata and strata-nvfp4

Infernix runs the [Dense8 conversion](#models) of NVIDIA's NVFP4 checkpoint (`--draft-tokens auto`).
[Strata](https://github.com/Niko1221/Strata) 0.1.45 runs Unsloth's
`Qwen3.8-Flash-Next-UD-Q4_K_XL` GGUF, its native format.
[strata-nvfp4](https://github.com/sergqwer/strata-nvfp4) 0.1.44.1-nvfp4.2, a Strata fork for
NVIDIA's checkpoint, runs NVIDIA's NVFP4 checkpoint as its `prepare-model` converts it (`--spec 6
--spec-min-p 0.7`; no control vector or vision; route tail skip off, its default with NVFP4 experts). So this compares each engine as it is run in
practice, each with its own quantization and drafter.

Strata runs the configuration its 0.1.45 setup writes for this machine (calibrated `--pcie-frac
0.00`, `--spec-min-p 0.70`, `--pool-workers 15`, `STRATA_HC_Q8=1`), with one change: the all-in-RAM
mode that setup picks pins 71.7 GiB of experts and 3.1 GiB of KV, close to the 76 GiB a process
can pin on this machine, and the server did not start (no VRAM was left). It runs with
`--resident-budget-gib 68`, the RAM-budget mode, in which route tail skip stays on by default.

**Context length**:

| Context | Time to first token: Infernix | Strata | strata-nvfp4 | Decode tok/s: Infernix | Strata | strata-nvfp4 |
|---|---:|---:|---:|---:|---:|---:|
| ~8K | **2.29 s** | 10.51 s | 2.54 s | **157.2** | 103.9 | 143.6 |
| ~128K | **12.58 s** | 93.59 s | 15.22 s | **142.7** | 92.1 | 134.9 |
| ~250K | **25.17 s** | 139.64 s | 31.22 s | 155.2 | 93.0 | **161.0** |

Infernix reaches the first token 4.6-7.4× sooner than Strata and decodes 1.5-1.7× as fast.
Against strata-nvfp4, on the same NVIDIA weights, it reaches the first token 10-19 % sooner and
decodes 6-9 % faster at 8K and 128K, and 4 % slower at 250K. The Strata row comes from a session
run after the others. An earlier session kept the all-in-RAM mode's `STRATA_ROUTE_TAIL_SKIP=0`
beside the RAM budget by mistake; it reached the first token about as soon (10.19, 92.22 and 141.67 s)
but decoded at 91.7, 87.0 and 81.7 tok/s.

**Agentic replay** (three seeds):

| Metric | Strata | strata-nvfp4 | Infernix | Infernix vs Strata | vs strata-nvfp4 |
|---|---:|---:|---:|---:|---:|
| Average time to first token | 17.0 s (16.5-17.8) | 10.7 s (9.3-11.6) | **4.9 s** (4.2-5.9) | −71 % | −54 % |
| Median time to first token | 9.77 s (9.64-9.92) | 6.74 s (5.41-7.53) | **3.55 s** (3.36-3.92) | −64 % | −46 % |
| 90th-percentile time to first token | 41.1 s (38.7-44.0) | 24.4 s (19.4-28.8) | **9.6 s** (8.6-11.5) | −77 % | −59 % |
| Average TTFT, new long prompts | 31.4 s (29.4-35.4) | 10.1 s (9.1-10.8) | **6.5 s** (6.3-6.7) | −79 % | −35 % |
| Prompt tokens served from cache | 72.5 % (68.8-75.0) | 68.9 % (67.5-71.0) | **84.9 %** (84.7-85.1) | +12.4 points | +16.0 points |
| Prompt tokens prefilled | 726K | 826K | **402K** | −44 % | −51 % |
| Prefill tok/s, requests with no cache hit | 1,977 | 5,674 | **6,536** | 3.3× | +15 % |
| Output tok/s, one request decoding | 94 (92-96) | 109 (105-112) | **135** (123-145) | +43 % | +24 % |
| Output tok/s, at the run's own batching | 59 (57-60) | 85 (81-90) | **121** (106-128) | 2.1× | +42 % |
| Tokens per round (draft acceptance) | 2.01 | 2.17 | **2.24** | +12 % | +4 % |
| Workload wall time | 15.9 min (15.5-16.6) | 11.5 min (11.0-12.0) | **6.7 min** (6.4-7.0) | −58 % | −41 % |

Strata re-prefilled the whole prompt on 4-7 of the 41 continuing main-session turns per seed and
strata-nvfp4 on 6-7, Infernix on none. With two slots Strata does not apply route tail skip, so the
Strata arm's configuration is the same as in the mistaken first session, whose replay gave the same
picture (average time to first token 18.5 s, 95 output tok/s for one request, 16.2 min).

### Qwen3.8-Flash-Next IQ3_S: Infernix vs Strata

Both run ISTA-DASLab's GSQ-RCO `IQ3_S` GGUF: Infernix its [bit-exact conversion](#models) with
the GGUF's IQ4_NL n-gram table, Strata 0.1.45 the GGUF with the configuration its setup writes
(`--spec 4 --spec-min-p 0.5 --pool-workers 15`, every expert in RAM). In that configuration
Strata's route tail skip is on: when it is active, a missed expert that every token of a verify
window ranks 7th or lower is neither copied nor computed, which is faster and changes the answers
slightly. The third column turns it off (`STRATA_ROUTE_TAIL_SKIP=0`).

**Context length**:

| Context | Time to first token: Infernix | Strata | Strata, tail skip off | Decode tok/s: Infernix | Strata | Strata, tail skip off |
|---|---:|---:|---:|---:|---:|---:|
| ~8K | 2.14 s | **1.65 s** | 1.66 s | 195.9 | **198.7** | 183.7 |
| ~128K | 29.37 s | **23.92 s** | **23.92 s** | **184.9** | 169.1 | 160.3 |
| ~250K | 58.77 s | 49.04 s | **48.99 s** | **205.6** | 164.4 | 166.7 |

Strata prefills the IQ3_S model faster: Infernix takes 30 % longer to the first token at 8K, 23 %
at 128K and 20 % at 250K. Infernix decodes level at 8K (−1 %) and faster at 128K (+9 %) and 250K
(+25 %). Turning tail skip off cost Strata 8 % and 5 % of decode at 8K and 128K and gained 1 % at
250K; prefill did not change.

**Agentic replay** (three seeds):

| Metric | Strata | Strata, tail skip off | Infernix | Infernix vs Strata |
|---|---:|---:|---:|---:|
| Average time to first token | 8.6 s (7.7-9.6) | 9.0 s (8.6-9.3) | **4.9 s** (4.0-5.7) | −43 % |
| Median time to first token | 5.29 s (5.20-5.36) | 5.05 s (3.95-5.90) | **2.70 s** (2.41-2.86) | −49 % |
| 90th-percentile time to first token | 19.5 s (16.3-21.6) | 20.9 s (19.1-21.8) | **13.1 s** (8.2-16.8) | −31 % |
| Average TTFT, new long prompts | 14.3 s (12.8-15.2) | 13.2 s (11.9-13.8) | **12.7 s** (12.4-12.8) | −11 % |
| Prompt tokens served from cache | 73.5 % (72.7-74.3) | 72.0 % (70.6-73.4) | **84.9 %** (84.7-85.1) | +11.4 points |
| Prompt tokens prefilled | 700K | 742K | **401K** | −43 % |
| Prefill tok/s, requests with no cache hit | 4,695 | **4,731** | 3,722 | −21 % |
| Output tok/s, one request decoding | 144 (142-146) | 144 (143-147) | **164** (160-167) | +14 % |
| Output tok/s, at the run's own batching | 107 (103-110) | 103 (100-110) | **158** (152-167) | +48 % |
| Tokens per round (draft acceptance) | 2.19 | 2.13 | **2.40** | +10 % |
| Workload wall time | 9.1 min (8.6-9.4) | 9.3 min (8.9-9.6) | **6.2 min** (5.8-6.6) | −32 % |

Infernix reaches the first token sooner because it serves more of each prompt from cache (Strata
re-prefilled the whole prompt on 4-6 of the 41 continuing main-session turns, Infernix on none),
although it prefills a prompt with no cache hit 21 % slower. With two slots Strata does not apply
tail skip at all (its documentation: a window then mixes requests), and neither run printed its
tail-skip line, so the two Strata columns of this table ran the same computation and differ only by
run-to-run spread.

### Qwen3.8-Flash-Next: DFlash2 against MTP

Decode speed of one request with each drafter, both with `--draft-tokens auto` and the proposal
head (`--lm-head-draft`): the model's MTP drafter (`--spec mtp`) and Inco AI's DFlash2 drafter
([Qwen3.8-Flash-Next-DFlash2-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-DFlash2-Infernix), `--spec dflash2 --draft-model FILE`). The prompts
are 24 slices of 1,280 tokens, each continued for 384 tokens, from Infernix's own C++/CUDA/Python
sources (code) and its Markdown documentation (docs) at commit `e59fa1c7`; greedy, or sampled at
temperature 1.0 with top-k 20 and top-p 0.95; INT8 KV; `infernix_decode_quality_gen`, on the
machine above (2026-10-20). Cells are decode tok/s, the mean of every run where there were
several: Dense8 code had four DFlash2 runs (209-217) and eight MTP runs (186-203), docs one and
three, sampled docs three each; repeats otherwise agreed within 4 %.

| Conversion | Text | Plain | MTP | **DFlash2** | DFlash2 vs MTP |
|---|---|---:|---:|---:|---:|
| NVIDIA Dense8 | code, greedy | 120 | 198 | **213** | +8 % |
| NVIDIA Dense8 | code, sampled | | 162 | **204** | +26 % |
| NVIDIA Dense8 | docs, greedy | 121 | 164 | **179** | +9 % |
| NVIDIA Dense8 | docs, sampled | 122 | 145 | **161** | +11 % |
| GSQ-RCO IQ3_S | code, greedy | 121 | 233 | **258** | +11 % |
| GSQ-RCO IQ3_S | docs, greedy | | 192 | **210** | +9 % |
| GSQ-RCO IQ3_S | docs, sampled | | 167 | **183** | +10 % |

DFlash2 drafts a whole block of seven tokens in one pass, where MTP runs one step per token, and its
candidate selector reports how likely each draft is; with `--draft-tokens auto` a round verifies the
number of drafts that pays for its verification columns, read from those probabilities. On this
offloaded model each verified column costs about a third of a plain round in expert-cache misses,
so verifying only likely drafts matters more than drafting many: verifying all seven every round
(`--draft-tokens 7`) was slower than MTP. The drafter also takes less VRAM than MTP's: the
Dense8 model's resident weights shrink from 6.27 to 5.66 GiB, leaving 9,070 cached experts against
8,740 (IQ3_S, whose MTP experts are already 2-bit: 12,432 against 12,288). Greedy output is the same
with either drafter; sampled output keeps the sampler's distribution.

### Qwen3.8-27B: Infernix vs NInfer

Both serve NInfer's official
[Qwen3.8-27B-nvfp4-NInfer](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) artifact with
INT8 KV and the same flags. NInfer is upstream `master` at
[`64d10fbf`](https://github.com/Neroued/ninfer/commit/64d10fbf9) with the Windows port (Win32 file
I/O, an IOCP read queue and MSVC build fixes in place of the Linux-only pieces).

**Context length** (DFlash2 decode at `--max-context 228000`, the largest NInfer starts at with
DFlash2 on 32 GB, so no ~230K row for it; time to first token and plain decode at
`--max-context 240000`):

| Context | Time to first token: Infernix | NInfer | DFlash2 decode tok/s: Infernix | NInfer | Plain decode tok/s: Infernix | NInfer |
|---|---:|---:|---:|---:|---:|---:|
| ~8K | **0.65 s** | 0.76 s | **207.7** | 199.9 | **82.6** | 80.1 |
| ~128K | **18.75 s** | 29.51 s | **209.4** | 183.3 | **69.6** | 67.1 |
| ~230K | **47.30 s** | 83.24 s | 175.2 | — | **61.4** | 59.0 |

Long prompts prefill 1.57× (128K) and 1.76× (230K) faster than NInfer; plain decode is 3-4 %
faster. DFlash2 decode depends on how many drafts each engine's text accepts: NInfer decoded the
two 128K prompts at 206 and 161 tok/s, Infernix at 218 and 201. NInfer kept the 8K prompt cached
through the two 128K prompts that followed it (a repeat took 0.15 s); Infernix had evicted it
(0.76 s).

**Agentic replay** (five seeds; `--max-context 200000` for both engines, the largest at which NInfer
starts with two requests and DFlash2; Infernix also uses n-gram drafting):

| Metric | NInfer | Infernix | Infernix vs NInfer |
|---|---:|---:|---:|
| Average time to first token | 3.4 s (2.7-3.8) | **3.1 s** (2.8-3.3) | −7 % |
| Median time to first token | 1.14 s (0.95-1.35) | **0.52 s** (0.45-0.61) | −54 % |
| 90th-percentile time to first token | 10.0 s (7.4-11.8) | 10.0 s (8.8-11.4) | +3 % |
| Average TTFT, continuing-session turns | **2.57 s** (1.77-3.26) | 2.93 s (2.62-3.39) | +18 % |
| Average TTFT, new long prompts | 10.2 s (9.4-10.9) | **6.4 s** (6.1-6.9) | −37 % |
| Prompt tokens served from cache | 79.0 % (78.7-79.4) | **85.3 %** (85.0-85.6) | +6.3 points |
| Prompt tokens prefilled | 562K | **396K** | −30 % |
| Prefill tok/s, requests with no cache hit | 4,986 | **7,176** | +44 % |
| Output tok/s, one request decoding | **194** (187-199) | 192 (187-198) | −1 % |
| Output tok/s, at the run's own batching | 229 (213-241) | **254** (238-274) | +11 % |
| Tokens per round (draft acceptance) | **3.23** | 3.17 | −2 % |
| Workload wall time | 7.1 min (6.2-8.0) | **6.0 min** (5.6-6.6) | −13 % |

Infernix's continuing-session turns reached the first token later on average (2.93 s against
2.57 s; later on three of the five seeds, by 15-48 %, and sooner on two), and the 90th percentile
is level. That time is mostly waiting for one of the two lanes behind other sessions' replies,
whose lengths are sampled and differ between the arms (2.63 s against 2.04 s of queueing per
turn). Without the queue, as the servers report it, Infernix reached the first token sooner on 249
of the 255 continuing turns (0.29 s against 0.52 s on average). The six exceptions were
0.05-0.12 s slower: one had more to prefill, and five were the first prefill after the GPU had been
idle for 3 s or more, which costs both engines about 0.29 s. Infernix also holds a lane for less
time per output token (6.1-6.4 ms against 6.6-7.8 ms per seed). Single-request decode is level.
Quality is level: on the repository's
1.04M-token perplexity corpus (16K context, INT8 KV cache) Infernix scores 4.468 against 4.479 for
NInfer (measured with `master` at `81c8ce09` and the Windows port).

## Quality

Infernix runs three routes:

- **The lossless route** (`--lossless`; Qwen3.5/3.6/3.8 and Qwen3.8-Flash-Next, generation and scoring)
  computes the model's own arithmetic exactly:
  every operation of its Hugging Face forward pass correctly rounded to the dtype it has there
  (weights decoded exactly, sums exact). Its output depends on the request and the artifact alone,
  not on the build, the prefill chunk, prefill or decode, other requests or the prefix cache; it is slower
  than the other routes
  ([definition](docs/maintainer/lossless-route.md)).
- **The reproducible route** (`--reproducible`) gives bit-identical output from run to run and from
  release to release. It is also request-invariant (other requests, concurrency, prefix-cache state
  and memory pressure do not change a request's output), and speculation does not change it: a
  sampled request draws each token by Gumbel-max keyed by its seed, position and token, so drafts
  are verified against the token plain decoding would draw.
- **Fast paths** (on by default) are faster kernels and scheduling choices that change the
  arithmetic. Each one runs only for the kernels it was qualified on, only after passing the quality
  gate below, and only where it was measured faster.

### The quality gate

A fast path is measured against a high-precision reference: the lossless route (`--lossless`, every
operation of the model's Hugging Face forward pass correctly rounded) with a BF16 KV cache. Infernix
charges a path only for the distance it adds beyond the reproducible route's own, so the
quantization a model already carries is not counted against it.

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
infernix-serve.exe Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix-00001-of-00003.infernix --ngram-volume Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram --kv-dtype int8 --max-context 262144 --spec dflash2 --draft-model Qwen3.8-Flash-Next-DFlash2-Infernix.infernix --draft-tokens auto --lm-head-draft --max-concurrency 2
```

`Qwen3.8-Flash-Next-DFlash2-Infernix.infernix` is the DFlash2 drafter
([Qwen3.8-Flash-Next-DFlash2-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-DFlash2-Infernix)); without it, `--spec mtp --draft-tokens auto
--lm-head-draft` drafts with the model's own MTP drafter, about 10 % slower
([benchmarks](#qwen38-flash-next-dflash2-against-mtp)).

The uncensored Flash-Next takes the same options, the same n-gram volume and the same drafter:

```bat
infernix-serve.exe Qwen3.8-Flash-Next-Uncensored-NVFP4-Infernix-00001-of-00003.infernix --ngram-volume Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram --kv-dtype int8 --max-context 262144 --spec dflash2 --draft-model Qwen3.8-Flash-Next-DFlash2-Infernix.infernix --draft-tokens auto --lm-head-draft --max-concurrency 2
```

Qwen3.8-27B on a 32 GB RTX 5090:

```bat
infernix-serve.exe qwen3_8_27b_nvfp4-nvidia.ninfer --max-context 240000 --max-concurrency 2 --spec dflash2 --draft-tokens auto --lm-head-draft --ngram-draft-tokens auto --kv-dtype int8 --preserve-thinking --host-context-mib 52000 --prefill-chunk 4096 --kv-capacity auto --tolerant-tool-calls
```

Add `--prefix-cache-file PATH` to keep the prefix cache across restarts. Stop the server with
Ctrl+C twice; it saves the cache and exits. `infernix-serve.exe --help` lists every option by
category; `--config FILE` reads them, and the model path, from a TOML file that `infernix` can
share ([configuration file](docs/serving.md#configuration-file);
[`infernix.example.toml`](infernix.example.toml) is an example).

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

Every model Infernix supports, with the Hugging Face repository to download it from. Each
repository's model card gives its files, requirements, recommended command, quality and speed.
`python -m tools.convert` converts your own checkpoints of these architectures
([weight conversion](docs/weight-conversion.md)).

### Qwen3.8-Flash-Next

| Repository | Converted from | What it is |
|---|---|---|
| **[Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix)** (**recommended**) | [nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) | NVIDIA's NVFP4 experts and FP8 n-gram table bit-exact, the dense projections in 8 bits. It decodes about 22 % faster than the bit-exact conversion at the same quality (perplexity 4.654 against 4.666) and is the one benchmarked [above](#qwen38-flash-next-nvfp4-infernix-vs-strata-and-strata-nvfp4). |
| [Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix) | [nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) | Every weight the model computes with exactly as NVIDIA stores it, for when the weights must be unchanged. |
| [Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix) | [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) (`IQ3_S`) | Every GGUF tensor bit-exact, computed in exact integer arithmetic, plus an MTP drafter and vision. The fastest decode: 146 tok/s plain and 260 with MTP against Dense8's 110 and 143 in the same session, with the experts in 46.9 GiB of RAM. Long prompts prefill slower than NVFP4 (4,324 against 10,933 tok/s at 131K tokens) and than Strata on the same GGUF ([benchmarks](#qwen38-flash-next-iq3_s-infernix-vs-strata)). |
| [Qwen3.8-Flash-Next-FP8-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-FP8-Infernix) | [Qwen/Qwen3.8-Flash-Next-FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8) | Qwen's FP8 checkpoint bit-exact, with Qwen's BF16 n-gram table: the most accurate conversion and the slowest (about 44 tok/s plain), since its 112.6 GiB of experts exceed what a 96 GB machine can pin and the SSD tier reads the rest. |
| [Qwen3.8-Flash-Next-Uncensored-NVFP4-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-Uncensored-NVFP4-Infernix) | [orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4) | An abliterated build whose refusal behaviour has been removed (see its model card). Its weight-only NVFP4 experts and FP8 dense projections are kept bit-exact and run with BF16 activations; the other dense classes are stored in 8 bits as in Dense8. Perplexity 4.772 (Strata's base-model GGUF 4.844, NVIDIA Dense8 4.653: abliteration changes the weights). Needs Infernix `windows-2026.10.09.2` or later. |
| [Swift-1.5-Qwen3.8-Flash-Next-NVFP4-Dense8-Infernix](https://huggingface.co/Wallawalla47/Swift-1.5-Qwen3.8-Flash-Next-NVFP4-Dense8-Infernix) | [ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4) | UkisAI's fine-tune that thinks with fewer tokens (see its model card), converted like NVIDIA Dense8: experts bit-exact, dense projections in 8 bits. Smoke-tested; its quality and speed are not yet measured. |
| [Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix) | [nvidia/Qwen3.8-Flash-Next-FP4-Local](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local) | NVIDIA's quantization-aware-distilled FP4 checkpoint, bit-exact: NVFP4 routed experts, MXFP8 dense projections and an NVFP4 n-gram table, which ships with the model as its own volume. Experts run with 16-bit activations as NVIDIA serves them; `--expert-a4-activations` opts into 4-bit activations with NVIDIA's calibrated scales (changes the output). Perplexity 3.4865 against 3.4854 for the FP8 conversion on the quick corpus. **Support is preliminary: optimisation is still in progress.** |

Every Flash-Next conversion can draft with **[Qwen3.8-Flash-Next-DFlash2-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-DFlash2-Infernix)**
(**recommended**, 776 MB), Inco AI's
[DFlash2 drafter](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2) with its projections in
8 bits: pass it with `--spec dflash2 --draft-model FILE --draft-tokens auto`. It decodes 8-11 % faster
than the conversions' own MTP drafter (26 % on sampled code) ([benchmarks](#qwen38-flash-next-dflash2-against-mtp)) and
needs an Infernix build whose `--help` lists `--draft-model`.

On the 1M-token perplexity corpus (4,096-token context, INT8 KV) the FP8 conversion scores 3.487,
NVIDIA Dense8 3.508 and IQ3_S 3.522. The IQ3_S and FP8 conversions need Infernix
`windows-2026.10.18` or later.

Every Flash-Next conversion also needs a PLE n-gram volume (`--ngram-volume`), kept apart from the
model files and read from an NVMe drive. There are three, one per row format, and any of them runs
any Flash-Next conversion; **the FP8 volume is the recommended one**:

| Volume | Format | Bytes | Repository |
|---|---|---:|---|
| [`Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram`](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix/blob/main/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram) (**recommended**) | FP8, one scale for the table (NVIDIA's) | 52,429,058,048 | [Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix), identical copy in [Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix) |
| [`Qwen3.8-Flash-Next-BF16.ngram`](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-FP8-Infernix/blob/main/Qwen3.8-Flash-Next-BF16.ngram) | BF16, unquantized (Qwen's) | 109,227,196,416 | [Qwen3.8-Flash-Next-FP8-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-FP8-Infernix) |
| [`Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix.ngram`](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix/blob/main/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix.ngram) | IQ4_NL, 4.5 bits per value (the GSQ-RCO GGUF's) | 29,127,258,112 | [Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix) |

Each conversion was made and measured with one of them: the four NVFP4 conversions with the FP8
table, IQ3_S with its IQ4_NL table and FP8 with the BF16 table; the uncensored and Swift
repositories carry no volume. Older Infernix builds accept only an artifact's own table or BF16.
The [n-gram table formats](docs/qwen3_8-flash-next.md#n-gram-table-formats) section of the guide explains
the formats.

### Qwen3.8-27B

| Repository | Converted from | What it is |
|---|---|---|
| **[Qwen3.8-27B-NVIDIA-NVFP4-NInferV3](https://huggingface.co/Wallawalla47/Qwen3.8-27B-NVIDIA-NVFP4-NInferV3)** (**recommended**) | [nvidia/Qwen3.8-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4) | NVIDIA's NVFP4 checkpoint with a DFlash2 draft model and a proposal head for `--lm-head-draft`. |
| [Qwen3.8-27B-Quasar-NVFP4-NInferV3](https://huggingface.co/Wallawalla47/Qwen3.8-27B-Quasar-NVFP4-NInferV3) | [QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4](https://huggingface.co/QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4) | The QAT-trained NVFP4 model, with the same draft model and proposal head. |
| [Qwen3.8-27B-Uncensored-NVFP4-NInferV3](https://huggingface.co/Wallawalla47/Qwen3.8-27B-Uncensored-NVFP4-NInferV3) | [orcarouter/Qwen3.8-27B-Uncensored-NVFP4](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-NVFP4) | An abliterated build whose safety alignment has been substantially removed (see its model card), with the same draft model and proposal head. |
| [Swift-Qwen3.8-27B-NVFP4-NInferV3](https://huggingface.co/Wallawalla47/Swift-Qwen3.8-27B-NVFP4-NInferV3) | [ukisai/Swift-Qwen3.8-27B-NVFP4](https://huggingface.co/ukisai/Swift-Qwen3.8-27B-NVFP4) | UkisAI's token-efficient fine-tune in NVIDIA's NVFP4/FP8 layout, converted like the NVIDIA artifact with the same draft model and proposal head. Smoke-tested on Infernix and NInfer; its quality and speed are not yet measured. |
| [neroued/Qwen3.8-27B-nvfp4-NInfer](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | NInfer's official NVFP4 artifact, the one benchmarked [above](#qwen38-27b-infernix-vs-ninfer). |
| [neroued/Qwen3.8-27B-NInfer](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | NInfer's official `groupwise-int` artifact. |

### Qwen3.6

| Repository | Converted from | What it is |
|---|---|---|
| [neroued/Qwen3.6-27B-NInfer](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) | [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) | NInfer's official `groupwise-int` artifact. |
| [neroued/Qwen3.6-27B-nvfp4-NInfer](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer) | [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) | NInfer's official NVFP4 artifact. |
| [neroued/Qwen3.6-35B-A3B-NInfer](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) | [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) | NInfer's official `groupwise-int` artifact of the MoE model, with a DFlash drafter. |

The `neroued` artifacts are NInfer's (Infernix loads `.ninfer` files unchanged); their model cards
are versioned in [`model-cards/`](model-cards/) beside those of the other repositories.

### Qwen3.5-122B-A10B

A 122B-parameter mixture-of-experts model on one 32 GB GPU: the routed experts (60.75 GiB) live in host
RAM behind a VRAM expert cache, as for Flash-Next ([guide](docs/qwen3_5-122b.md)). **Support is preliminary: it runs and is verified, but optimisation is still in progress, so expect speeds to change.**

| Repository | Converted from | What it is |
|---|---|---|
| **[Ornith-1.5-122B-NVFP4-Infernix](https://huggingface.co/Wallawalla47/Ornith-1.5-122B-NVFP4-Infernix)** | [ornith-ai/Ornith-1.5-122B-NVFP4](https://huggingface.co/ornith-ai/Ornith-1.5-122B-NVFP4) | Every weight the model computes with exactly as the checkpoint stores it (NVFP4 experts, FP8 projections), plus the MTP drafter, the [DFlash2 drafter](https://huggingface.co/incoai/Ornith-1.5-122B-DFlash2) re-quantized, the vision tower, a Q4 proposal head for `--lm-head-draft` and the model card's sampling defaults. About 90-96 tok/s plain decode, 6,500 tok/s prefill at 32K, and MTP with 2 drafts about 5 % faster than plain decode on sampled requests ([performance](docs/qwen3_5-122b.md#performance-rtx-5090-one-request-bf16-kv-default-route)). |

## Documentation

[CLI](docs/cli.md), [serving](docs/serving.md), [Qwen3.8-Flash-Next](docs/qwen3_8-flash-next.md),
[Qwen3.5-122B-A10B](docs/qwen3_5-122b.md), [perplexity](docs/perplexity.md), [performance](docs/performance.md) and the
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
