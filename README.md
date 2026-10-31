# Infernix

> **AI disclaimer:** everything added since the fork from NInfer, including most of this README, was
> written with AI (mostly Claude Opus 5.5 and Qwen3.8-27B running on Infernix). It is likely to be
> neither complete nor entirely accurate. This is hobby development.

Infernix is a C++/CUDA inference engine that serves Qwen models on one NVIDIA RTX 50-series GPU
(tuned on the RTX 5090). It began as a fork of [NInfer](https://github.com/Neroued/ninfer) by
[Neroued](https://github.com/Neroued) and reads NInfer's `.ninfer` artifacts as well as its own
`.infernix` files.

## Headline results

`main` at `6b066a54`, one RTX 5090 (PCIe Gen5 x8), 2026-10-30/31, against each engine's latest
version, every engine native on Windows under the same 84 GiB RAM budget. Context columns: one
request at a time, ~8K to ~250K tokens. Replay: an agentic coding workload, two requests at once.
Activation precision differs: the other engines round to 8 bits what Infernix keeps at 16 (every
activation on IQ3_S; all but the NVFP4 routed experts, which Infernix runs at the checkpoint's own
4 bits and the others at 8 in decode); see each section's notes. Details: [Benchmarks](#benchmarks).

| Infernix vs | Time to first token | Decode tok/s | Replay: mean time to first token | Replay: wall time |
|---|---|---|---|---|
| [Strata](https://github.com/Niko1221/Strata) v0.2.8, same NVIDIA NVFP4 weights (Infernix: Dense8) | 1.4-1.6× sooner | +7-15 % | 3.3 s vs 12.3 s | 5.7 vs 12.9 min |
| [strata-nvfp4](https://github.com/sergqwer/strata-nvfp4) 0.2.0-beta-nvfp4.3, same weights | 1.3× sooner | level at 8K, +5 % and +14 % at 128K and 250K | 3.3 s vs 10.8 s | 5.7 vs 11.7 min |
| [Basalt](https://github.com/jesdga95/basalt) `c08284cb`, NVIDIA NVFP4 (its pack) | 5-8× sooner† | Basalt 3-13 % faster | 3.3 s vs 7.2 s | 5.7 vs 8.9 min |
| Strata* v0.2.8, same IQ3_S GGUF | 15 % sooner at 8K, level at 128K-250K | +3-12 % | 3.1 s vs 6.9 s | 5.5 vs 8.1 min |
| Basalt `c08284cb`, IQ3_S (its pack) | Basalt 4 % sooner at 8K, 1.3× at 128K-250K | Basalt 1.4-1.7× faster | 3.1 s vs 3.7 s | Basalt sooner: 5.5 vs 4.4 min |
| [NInfer](https://github.com/Neroued/ninfer) `1882c6adb` (Windows port), 27B NVFP4 | 9 % sooner at 8K, 1.6-1.8× at 128K-230K | own drafters +11-17 %; no drafting +2-3 % | 1.9 s vs 2.5 s (median 0.59 vs 0.93 s) | 12.7 vs 16.4 min |

\* Strata's default route tail skip (it skips some missed experts) was turned off for these runs
because it changes which experts run; see the [IQ3_S section](#qwen38-flash-next-iq3_s-infernix-vs-strata-and-basalt).

† Basalt's NVFP4 pack reads its 50.7 GiB n-gram table from the SSD on every prompt: within the RAM
budget the table does not fit beside its experts (in RAM it pushed 28 % of them out and the first
request failed). On IQ3_S, whose table fits in RAM, Basalt is the soonest. Not model loading: Basalt
loads its model during its first request, and every engine's first request is an uncounted warm-up.

## Why Infernix

### Qwen3.8-Flash-Next on one RTX 5090

Its 24,576 routed experts (63 GiB at NVFP4) live in pinned system RAM; the hot ones are cached in
VRAM, part of each layer's misses is computed on the CPU while the rest cross PCIe, and the 52 GB
n-gram table is read from NVMe.

- **Soonest to the first token on NVIDIA's NVFP4 weights**: 1.3-1.6× sooner than
  [Strata](https://github.com/Niko1221/Strata) v0.2.8 and
  [strata-nvfp4](https://github.com/sergqwer/strata-nvfp4) from 8K to 250K tokens, and 5-8× sooner
  than [Basalt](https://github.com/jesdga95/basalt), which reads its n-gram table from the SSD
  here; in the agentic replay 3.3 s against 7.2-12.3 s and 35-55 % less wall time. Decode is ahead
  of Strata, level with strata-nvfp4 at 8K and ahead at longer contexts, and behind Basalt by 3-13 %
  (level on one request in the replay)
  ([benchmarks](#qwen38-flash-next-nvfp4-infernix-vs-strata-strata-nvfp4-and-basalt)).
- **On the IQ3_S GGUF** Infernix decodes 3-12 % faster than Strata and needs a third less replay
  time, but Basalt decodes about 1.5× as fast as Infernix and finishes the replay 26 % sooner
  ([benchmarks](#qwen38-flash-next-iq3_s-infernix-vs-strata-and-basalt)).
- **16-bit activations by default** where Strata and Basalt round them to 8 bits (on NVIDIA's NVFP4
  weights the routed experts take the checkpoint's own 4-bit activations, which the others compute
  at 8 bits in decode); measured quality for every engine follows in a later update. Perplexity was 4.654
  against Strata 0.1.45's 4.844 on the same token ids (2026-10-19).
- **DFlash2 drafting** with Inco AI's drafter
  ([Qwen3.8-Flash-Next-DFlash2-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-DFlash2-Infernix),
  776 MB, any conversion): 8-11 % faster than the model's MTP drafter (26 % on sampled code), with
  VRAM for 330 more cached experts ([benchmarks](#qwen38-flash-next-dflash2-against-mtp)).
- **Long context**: 262K tokens natively, up to 1M with YaRN; a 250K prompt prefills in about 24 s.
- **Less RAM than the model?** An SSD expert tier reads what does not fit.
- **Uncensored too**:
  [Qwen3.8-Flash-Next-Uncensored-NVFP4-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-Uncensored-NVFP4-Infernix)
  keeps orcarouter's abliterated weights bit-exact (perplexity 4.772) and decodes about as fast as
  Dense8.

Setup and tuning: the [Qwen3.8-Flash-Next guide](docs/qwen3_8-flash-next.md).

### Qwen3.8-27B: faster prefill and decode

- **Hybrid prefix cache** for Qwen's linear-attention layers: content-addressed KV in 64-token
  blocks, recurrent-state snapshots where prompts get reused, and GPU, RAM and file tiers. In an
  agentic workload against NInfer `1882c6adb` (both serve 91 % of prompt tokens from cache) it
  reaches the first token 36 % sooner at the median and finishes 21 % sooner
  ([benchmarks](#qwen38-27b-infernix-vs-ninfer)).
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
is `main` at [`6b066a54`](https://github.com/Wallawalla47/Infernix/commit/6b066a54) built from
source; every engine runs natively on Windows as an OpenAI-compatible server driven by the same
client, which times the first token from sending the request to the first streamed token
(2026-10-30/31, GPU power limit 517.5 W, clocks checked before every session).

- **Context length**: one request, greedy, 1,024 output tokens; three ~8K prompts and two each of
  ~128K and ~250K (~230K on the 27B). Two Infernix sessions bracket the other engines' (pooled);
  the others run one session each.
- **Agentic replay** ([`bench/agentic_ab`](bench/agentic_ab/README.md)): three coding-agent sessions
  and eleven subagents with tool calls, compactions, retries and aborts, up to eight requests in
  flight, temperature 1.0, 68 requests per seed. Every engine serves two requests at once; every arm
  replays the same observations. Cells: means over seeds (range in brackets); comparison columns:
  the mean of per-seed differences. Every engine completed every request.
- **Settings**: each engine's own recommended configuration for this PC (its setup, probe or example
  config) and its own drafter, with the same rules for all: INT8 KV; 262,144 tokens on Flash-Next;
  an 8 GiB conversation or prefix cache; one RAM budget of 84 GiB of the 96 (a server that takes
  more is stopped; every engine stayed below it); every n-gram table on the same kind of drive
  (Samsung 990 PRO); no thinking caps (answer room off everywhere); and every switchable option
  that computes below the checkpoint's own precision or changes which experts run turned off
  (marked * where that differs from the engine's default). Every engine verifies every draft.

### Qwen3.8-Flash-Next NVFP4: Infernix vs Strata, strata-nvfp4 and Basalt

Infernix runs the [Dense8 conversion](#models) of NVIDIA's checkpoint with Inco AI's DFlash2 drafter
(`--spec auto`, draft length auto, `--lm-head-draft`, `--vram-past-budget`);
[Strata](https://github.com/Niko1221/Strata) v0.2.8 and
[strata-nvfp4](https://github.com/sergqwer/strata-nvfp4) 0.2.0-beta-nvfp4.3 NVIDIA's checkpoint as
their NVFP4 pack, with the model's MTP drafter (Strata: what its setup writes for this PC plus its
`--prefill auto:32768` tip for 96 GB PCs; strata-nvfp4: its bundled `--spec 6 --spec-min-p 0.7`);
[Basalt](https://github.com/jesdga95/basalt) `c08284cb` its published NVFP4 pack with MTP windows of
8. Basalt reads its 50.7 GiB n-gram table from the SSD here: keeping it in RAM pushed 28 % of the
experts out to the SSD and the first request failed, so its time to the first token is long (its
own A/B on IQ3_S: reading the table from the SSD made a 128K prompt 5× slower). Model loading is not
in these figures: Basalt loads its model during its first request, and every session starts with an
uncounted warm-up request.

**Context length** (Infernix: two sessions pooled):

| Context | TTFT: Infernix | Strata | strata-nvfp4 | Basalt | Decode tok/s: Infernix | Strata | strata-nvfp4 | Basalt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ~8K | **1.88 s** | 2.67 s | 2.48 s | 9.55 s | 178 | 159 | 177 | **202** |
| ~128K | **12.17 s** | 18.83 s | 15.34 s | 92.94 s | 168 | 146 | 161 | **173** |
| ~250K | **23.90 s** | 37.55 s | 31.78 s | 125.92 s | 195 | 183 | 172 | **216** |

Decode varies most at ~250K, where each request's thinking decides how many drafts are accepted
(per request: Infernix 151-246, Basalt 190-242 tok/s).

**Agentic replay** (three seeds):

| Metric | Strata | strata-nvfp4 | Basalt | Infernix | vs Strata | vs strata-nvfp4 | vs Basalt |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mean TTFT | 12.3 s (10.8-13.9) | 10.8 s (9.2-12.0) | 7.2 s (6.4-7.9) | **3.3 s** (3.3-3.4) | −73 % | −69 % | −54 % |
| Median TTFT | 9.92 s | 7.62 s | 3.81 s | **2.17 s** | −78 % | −72 % | −43 % |
| 90th-percentile TTFT | 25.9 s | 24.7 s | 17.5 s | **7.8 s** | −69 % | −68 % | −55 % |
| Mean TTFT, new long prompts | 11.8 s | 10.2 s | 20.8 s | **6.1 s** | −48 % | −39 % | −70 % |
| Prompt tokens from cache | 46.1 % | 53.2 % | 83.1 % | **84.2 %** | +38.1 pt | +31.0 pt | +1.1 pt |
| Prefill tok/s, no cache hit | 5,159 | 5,701 | 2,108 | **7,380-7,515** | +46 % | +29 % | 3.5× |
| Output tok/s, one request | 116 (114-120) | 110 (108-113) | **166** (160-170) | **166** (165-167) | +43 % | +51 % | level |
| Output tok/s, batched | 67 | 69 | 129 | **145** (141-149) | 2.2× | 2.1× | +12 % |
| Tokens per round | 2.26 | 2.21 | 2.37 | **2.61** | +16 % | +19 % | +11 % |
| Wall time | 12.9 min (12.0-13.8) | 11.7 min (11.0-12.9) | 8.9 min (8.6-9.1) | **5.7 min** (5.5-6.2) | −55 % | −51 % | −35 % |

Strata re-prefilled the whole prompt on 15-17 of the 41 continuing main-session turns per seed,
strata-nvfp4 on 9-14, Basalt on 0-1 and Infernix on 1.

**What each engine computes** (none of these options is switchable unless noted):

| | Infernix | Strata v0.2.8 | strata-nvfp4 | Basalt |
|---|---|---|---|---|
| Routed experts' activations (4-bit in the checkpoint) | 4-bit with NVIDIA's calibrated scales, as the checkpoint specifies | 8-bit | 8-bit; in the prompt, gate/up two 4-bit terms per value | 8-bit; prompt chunks of 1,024+ tokens 4-bit |
| Other activations (BF16 in the checkpoint) | 16-bit | **8-bit** | **8-bit** | **8-bit** |
| Dense layers (BF16 in the checkpoint) | 8-bit (Dense8; the bit-exact conversion keeps BF16) | 8-bit | 8-bit | **6-bit** (Basalt measures +10 % KL divergence from it) |
| n-gram table (FP8 in the checkpoint) | FP8 | FP8 | FP8 | re-blocked to 8-bit |

NVIDIA's checkpoint specifies 4-bit activations for the routed experts, which Infernix follows and
the others compute more finely in decode; it keeps every other activation in BF16, which Infernix
follows and the others round to 8 bits. None of the NVFP4 engines had a lossy option on by default
that could be turned off. Measured quality (KL divergence against an exact reference) follows in a
later update.

**Infernix with 8-bit expert activations** (the opt-in `--a8-activations`, closer to the others'
arithmetic in decode; not the comparison above): context length TTFT 1.88 / 15.53 / 30.58 s and
decode 199 / 185 / 268 tok/s at ~8K / ~128K / ~250K; replay mean TTFT 3.4 s, output 175 tok/s at one
request (+5 % on Basalt), wall time 5.7 min. Decode gains 5-9 %, but prompts read up to 28 % slower
than on the default 4-bit route, so the default stays the faster choice for agents.

### Qwen3.8-Flash-Next IQ3_S: Infernix vs Strata and Basalt

All three run ISTA-DASLab's GSQ-RCO `IQ3_S` GGUF: Infernix its [bit-exact conversion](#models) with
the GGUF's IQ4_NL n-gram table and the DFlash2 drafter (settings as above), Strata* v0.2.8 the GGUF
with its setup's configuration (`--spec 4 --spec-min-p 0.5 --pool-workers 15`, every expert in RAM,
`--prefill auto:32768`), Basalt `c08284cb` its published IQ3_S pack with the n-gram table in RAM.
Infernix reads its table from the SSD: in RAM it would cost a RAM copy of 6.2 GiB of experts, and
that measured 4 % sooner to the first token but 3-7 % slower decode.

**Context length** (Infernix: two sessions pooled):

| Context | TTFT: Infernix | Strata* | Basalt | Decode tok/s: Infernix | Strata* | Basalt |
|---|---:|---:|---:|---:|---:|---:|
| ~8K | 1.40 s | 1.65 s | **1.35 s** | 229 | 209 | **380** |
| ~128K | 18.10 s | 17.97 s | **13.47 s** | 201 | 196 | **298** |
| ~250K | 35.59 s | 36.56 s | **27.98 s** | 250 | 223 | **339** |

**Agentic replay** (three seeds):

| Metric | Strata* | Basalt | Infernix | vs Strata* | vs Basalt |
|---|---:|---:|---:|---:|---:|
| Mean TTFT | 6.9 s (6.1-7.5) | 3.7 s (3.2-4.2) | **3.1 s** (2.9-3.4) | −55 % | −14 % |
| Median TTFT | 4.49 s | 1.97 s | **1.48 s** | −67 % | −24 % |
| 90th-percentile TTFT | 15.4 s | 9.5 s | **9.2 s** | −39 % | +2 % |
| Mean TTFT, new long prompts | 9.3 s | **8.4 s** | 9.4 s | +2 % | +12 % |
| Prompt tokens from cache | 72.9 % | **83.2 %** | 82.6 % | +9.8 pt | −0.6 pt |
| Prefill tok/s, no cache hit | 6,298 | **7,714** | 5,860-5,968 | −7 % | −22 % |
| Output tok/s, one request | 159 (157-161) | **296** (292-298) | 201 (198-203) | +26 % | −32 % |
| Output tok/s, batched | 111 | **249** | 184 (171-202) | +66 % | −26 % |
| Tokens per round | 2.17 | 2.48 | **2.64** | +22 % | +7 % |
| Wall time | 8.1 min (7.7-8.5) | **4.4 min** (4.2-4.6) | 5.5 min (5.1-6.2) | −32 % | +26 % |

Basalt decodes IQ3_S about 1.5 times as fast as Infernix and finishes the replay 26 % sooner;
Infernix still reaches the first token soonest on the replay's short continuing turns.

**What each engine computes:**

| | Infernix | Strata* v0.2.8 | Basalt |
|---|---|---|---|
| Activations in matrix multiplies | 16-bit | **8-bit** | **8-bit** |
| Experts that run | every routed expert | every routed expert (route tail skip **off**, see below) | every routed expert |
| Output head | the GGUF's | the GGUF's | logits outside the top 64 from a **2-bit copy** of the head (the choices themselves exact; Basalt measures +1.1 % KL divergence) |

\* Strata's default for this weight set skips a missed expert that every token of a verify window
ranks 7th or lower (route tail skip; faster, but different answers). It was turned off for these
runs (`STRATA_ROUTE_TAIL_SKIP=0`) because it changes which experts the model uses and can degrade
quality, so the default would not be a like-for-like comparison. The engines' 8-bit activations
cannot be turned off. Measured quality follows in a later update.

**Infernix with 8-bit activations** (the opt-in `--a8-activations`, the others' arithmetic; not the
comparison above):

| | TTFT ~8K / ~128K / ~250K | Decode tok/s | Replay: mean TTFT | Replay: output tok/s | Replay: wall time |
|---|---:|---:|---:|---:|---:|
| Infernix `--a8-activations` | 1.30 / 13.99 / 27.68 s | 224 / 211 / 278 | 2.9 s | 208 | 5.2 min |
| Basalt (as above) | 1.35 / 13.47 / 27.98 s | 380 / 298 / 339 | 3.7 s | 296 | 4.4 min |

With 8-bit activations Infernix reads prompts about as fast as Basalt, but Basalt still decodes
about 1.4 times as fast.

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
with INT8 KV at 240,000 tokens and two lanes. NInfer is `master` at
[`1882c6adb`](https://github.com/Neroued/ninfer/commit/1882c6adb) (2026-10-29) with a Windows port
(Win32 I/O, an IOCP read queue, MSVC fixes; its greedy text matched the previous port's exactly on
the four prompts checked). Each runs its own recommended drafter: NInfer its RTX 5090 example
(MTP 3 with the optimized proposal head, automatic capacity, 256 MiB headroom, 1,024-token prefill
chunks), Infernix DFlash2 with draft length auto, the proposal head and n-gram drafts. Neither caps
thinking.

**Context length**:

| Context | TTFT: Infernix | NInfer | Drafting tok/s: Infernix | NInfer | No drafting tok/s: Infernix | NInfer |
|---|---:|---:|---:|---:|---:|---:|
| ~8K | **0.65 s** | 0.71 s | **228** | 195 | **81.1** | 79.2 |
| ~128K | **18.61 s** | 29.88 s | **210** | 182 | **68.4** | 66.5 |
| ~230K | **47.04 s** | 84.43 s | **187** | 169 | **60.3** | 58.7 |

The drafting columns follow each engine's accepted drafts (Infernix 3.5-4.9 tokens per round, NInfer
2.9-3.4); without drafting Infernix decodes 2-3 % faster.

**Agentic replay** (five seeds):

| Metric | NInfer | Infernix | vs NInfer |
|---|---:|---:|---:|
| Mean TTFT | 2.5 s (2.2-2.8) | **1.9 s** (1.6-2.2) | −21 % |
| Median TTFT | 0.93 s (0.85-1.13) | **0.59 s** (0.55-0.62) | −36 % |
| 90th-percentile TTFT | 6.6 s (5.1-8.2) | **5.3 s** (4.1-6.9) | −17 % |
| Mean TTFT, continuing turns | 2.12 s (1.88-2.52) | **1.67 s** (1.30-1.96) | −21 % |
| Mean TTFT, new long prompts | 10.9 s (10.6-11.2) | **7.5 s** (7.4-7.6) | −32 % |
| Prompt tokens from cache | 90.9 % (90.7-91.1) | 91.1 % (91.0-91.3) | +0.2 pt |
| Prefill tok/s, no cache hit | 4,749 | **7,088** | +49 % |
| Output tok/s, one request | 160 (154-164) | **220** (206-238) | +37 % |
| Output tok/s, batched | 193 (188-195) | **255** (245-262) | +32 % |
| Tokens per round | 2.68 | **3.68** | +38 % |
| Wall time | 16.4 min (14.0-19.0) | **12.7 min** (11.7-13.6) | −21 % |

Both engines decode the same number of rounds per second (59.8 against 59.9); the output lead is
Infernix's drafter. Without a thinking cap the replies run longer than in earlier versions of this
table, so wall times are not comparable with them. Quality is level: perplexity 4.468 against
NInfer's 4.479 on the 1.04M-token corpus (16K context, INT8 KV; NInfer `81c8ce09`, 2026-10-19).

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

`--spec auto --draft-tokens auto` makes the table's drafter choice by itself from the artifact's drafters
(on Flash-Next DFlash2 when `--draft-model` names its drafter, else MTP) and reports it at startup;
`--lm-head-draft`, `--draft-model` and `--ngram-draft-tokens auto` are still given where the table lists them.

Flash-Next keeps in RAM only the routed experts VRAM does not hold by default (`--expert-ram-layout
exclusive`); `--expert-ram-layout full` keeps a copy of every expert, 4-6 GiB more RAM on Dense8 and IQ3_S at
the same decode and prefill speed ([guide](docs/qwen3_8-flash-next.md)). The Qwen3.5-122B-A10B keeps every
expert in RAM by default; `--expert-ram-layout exclusive` holds 3.9 GiB less at about 1 % of decode after long
prompts ([guide](docs/qwen3_5-122b.md)).

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
| **[Ornith-1.5-122B-NVFP4-Infernix](https://huggingface.co/Wallawalla47/Ornith-1.5-122B-NVFP4-Infernix)** | [ornith-ai/Ornith-1.5-122B-NVFP4](https://huggingface.co/ornith-ai/Ornith-1.5-122B-NVFP4) | Every computed weight as the checkpoint stores it (NVFP4 experts, FP8 projections), plus the MTP drafter, the [DFlash2 drafter](https://huggingface.co/incoai/Ornith-1.5-122B-DFlash2) re-quantized, vision, a Q4 proposal head and the card's sampling defaults. About 64-97 tok/s plain decode depending on the text, 6,500 tok/s prefill at 32K; `--spec mtp --draft-tokens auto` (2 gated drafts) decodes 14 % faster at one request, and the decode split of 2026-10-30 adds 8-9 % to that on real code (3 % plain) ([performance](docs/qwen3_5-122b.md#performance-rtx-5090-one-request-bf16-kv-default-route)). |
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
