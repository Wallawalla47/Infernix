# Infernix

> **AI disclaimer:** Everything added to this engine since it forked from NInfer, including most of
> this README, was written with AI (mostly Claude Opus 5.5, Qwen3.8-27B running on Infernix, plus a
> few other AI systems I’ve been testing). It is likely to be neither complete nor entirely
> accurate. This is hobby development.

> **Infernix began as a fork of [NInfer](https://github.com/Neroued/ninfer)** by
> [Neroued](https://github.com/Neroued), a from-scratch C++/CUDA inference engine for single-GPU
> serving of Qwen models. It has since grown into its own engine. It runs
> **Qwen3.8-Flash-Next-NVFP4** — a mixture-of-experts model whose experts do not fit in VRAM — on a
> single RTX 5090, **faster than [Strata](https://github.com/Niko1221/Strata)** on the
> same machine, and it serves the dense **Qwen3.8-27B** models with a hybrid prefix cache, broader
> tool-calling support, faster prefill and decode kernels and many other improvements over NInfer.
> Infernix still reads NInfer's `.ninfer` artifacts as well as its own `.infernix` files.

## Highlights

### Qwen3.8-Flash-Next-NVFP4 on one RTX 5090

Qwen3.8-Flash-Next has 24,576 routed experts (63 GiB at NVFP4) — far more than a 32 GB GPU holds.
Infernix keeps the experts in pinned system RAM, caches the hot ones in VRAM, computes part of
each layer's cache misses on the CPU while the rest cross PCIe, and streams the 52 GB per-layer
n-gram embedding table from an NVMe drive.

- **Faster than Strata** on the same RTX 5090, both with the model's MTP drafter: 4.6-6.3× sooner
  to the first token and about 1.6× the decode speed from 8K to 250K tokens of context, and in a
  replayed agentic coding workload a quarter of Strata's average time to first token and less than
  half its wall time ([benchmarks](#qwen38-flash-next-nvfp4-infernix-vs-strata)).
- **Quality at least Strata's**: teacher-forced perplexity 4.600 against Strata's 4.864 on the same
  texts (lower is better); every optimisation was held to that bar.
- **Long context**: 262K tokens natively, up to 1M with YaRN (`--rope-yarn-factor`); a 250K-token
  prompt prefills at ~7.3K tok/s (34 s), and decoding with MTP stayed at 126-166 tok/s from 8K to
  250K tokens.
- **Every KV format** (`int8`, `bf16`, `fp8`, `nvfp4`, `k8v4`, `vq2`, `k4v2`), images and video,
  the hybrid prefix cache, concurrent requests with resource-pressure preemption, an SSD expert
  tier for machines with less RAM, and a VRAM pool that grows with the context.

Setup, conversion and tuning: [Qwen3.8-Flash-Next guide](docs/qwen3_8-flash-next.md).

### Qwen3.8-27B: prefix caching, tool calling and speed

- **Hybrid prefix cache** (the default), built for Qwen's linear-attention layers: KV cached per
  64-token block by content, sparse snapshots of the recurrent state at useful points, three tiers
  (GPU, pinned host RAM, and a file that survives restarts). On a replayed agentic coding workload
  it serves 90.4 % of prompt tokens from cache against 83.7 % for NInfer's newest cache, prefills
  42 % fewer tokens and halves the average time to first token
  ([benchmarks](#qwen38-27b-infernix-vs-ninfer)).
- **Tool calling for agent clients**: the XML call forms Claude Code and other agents emit,
  tolerant recovery of broken calls (`--tolerant-tool-calls`), Anthropic and OpenAI options used by
  Claude Code, Qwen Code, Codex, Zed and GitHub Copilot, and correct handling of reasoning output.
- **Faster prefill**: FlashAttention-2 style INT8 prompt attention, FP4/FP8 Tensor Core paths for
  NVFP4, FP8 and K8V4 KV, wave-aligned chunks — a third to two thirds less prompt-attention time
  on long prompts.
- **Faster decode**: overlapped (PDL) decode kernels, verification-width kernel tuning, DFlash2
  tree verification and n-gram copy drafting alongside MTP/DFlash2.
- **Smaller KV**: the `vq2` and `k4v2` formats hold three to four times the context of INT8.

## Benchmarks

Measured on 7 October 2026 on one machine: an RTX 5090 (32 GB, driver 617.14) in a **PCIe Gen5
x8** link, a Core i9-13900K, 96 GB of DDR5-5800, Samsung 990 PRO NVMe drives and Windows 11. The
card runs at x8 in this machine; at Gen5 x16 absolute speeds are likely to be higher, most of all
for Qwen3.8-Flash-Next, whose expert-cache misses cross PCIe on every token.

### How the figures were measured

Every engine runs as its own OpenAI-compatible server and is driven over `/v1/chat/completions`
by the same client, so all engines are measured the same way.

- **Context-length test.** One request at a time after a discarded warm-up request. Prompts are
  real repository text (documentation and C++/CUDA source) cut to ~8K, ~128K and ~250K tokens
  (~230K for Qwen3.8-27B, see below), three prompts at 8K and two at each longer size. Every prompt
  starts with a unique line, so no prompt can reuse another's cache. Requests are greedy with
  thinking on and write 1,024 tokens. Cells are means with the range in brackets.
  - **Follow-ups** then re-send earlier conversations as a chat client does (the prompt, the
    engine's own reply and a new question): F1 continues the newest long conversation, F2 the
    128K one with two longer prompts after it, F3 an 8K one from before those.
  - **TTFT** is the time from sending a request to its first streamed token on the client's
    clock; **prefill tok/s** is prompt tokens ÷ TTFT; **decode tok/s** is (output tokens − 1) ÷
    (last token − first token). **Tokens per round** (output tokens ÷ verify rounds) is how much
    of each round's speculative draft was accepted, and **rounds/s** is the engine's speed with
    that effect removed. **Cached** is what each server reports it reused.
  - Decode with speculation depends on how predictable the generated text is, and each engine
    (and each setting) writes slightly different text, so single-prompt decode figures move by up
    to ±20 %. Tokens per round and rounds/s show which of the two moved; the agentic replay
    measures decode over hundreds of outputs.
- **Agentic replay** ([`bench/agentic_ab`](bench/agentic_ab/README.md)). Three interleaved
  coding-agent sessions plus eleven subagents: 130 requests per run with 25K-135K-token prompts,
  tool calls, compactions, retries and aborted requests, sampled at temperature 1.0, closed loop
  (each engine's own answers are fed back) with up to eight requests in flight. Qwen3.8-27B runs
  it in full on three workload seeds; Qwen3.8-Flash-Next at half length (`--scale 0.5`, 68 scored
  requests) on two, which keeps Strata's runs under 20 minutes. Cells are the mean over seeds with
  the range in brackets. Infernix and NInfer figures come from each server's own request log,
  joined to the client's record of every request.
- **How Strata's figures are derived.** Strata writes no per-request log of this kind, so its
  rows come from what it returns with every response, without any change to Strata:
  - prompt and output tokens from the response's `usage`;
  - cached tokens, prefill time, decode time and speculative drafts from Strata's own per-request
    `timings` (`cache_n`, `prompt_ms`, `predicted_ms`, `draft_n`, `draft_n_accepted`); verify
    rounds are output tokens − accepted drafts;
  - TTFT from the client's clock. In a run that includes Strata every engine's TTFT is taken from
    the client's clock, so the columns compare like with like;
  - queueing before a request starts is TTFT − prefill time (Strata answers one request at a
    time), and its decode rate is decode tokens ÷ decode time per request (it never decodes two
    requests at once), where Infernix's comes from its own 5-second decode records.

### Qwen3.8-Flash-Next-NVFP4: Infernix vs Strata

| | Infernix | Strata |
|---|---|---|
| Version | branch `Infernix` at `a436fb5a` (agentic replay: `c9ee0445` and `9061adb8`) | 0.1.40 (release engine; server `82f46a8`, 0.1.40.1) |
| Model | [nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) converted with recipe B: NVIDIA's NVFP4 experts bit-exact, dense projections and `lm_head` in Q8 (group 32), MTP drafter in Q8 with Q4 experts, proposal head ([Hugging Face](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVFP4-Infernix)) | Unsloth `Qwen3.8-Flash-Next-UD-Q4_K_XL` GGUF (four shards) as Strata's native pack, with Strata's Q2_0 MTP drafter |
| KV cache and context | INT8, 262,144 tokens | INT8, 262,144 tokens (32,768 cells per layer resident in VRAM) |
| Speculation | MTP, `--draft-tokens 4 --lm-head-draft` | MTP, `--spec 4 --spec-min-p 0.70` |
| Other settings | n-gram volume on NVMe, `--prefill-chunk 4096` | `--expert-cache auto --prefill auto --resident-budget-gib 68 --pool-workers 15 --pcie-frac 0.00` |
| Agentic replay adds | `--max-concurrency 2 --pending-timeout-ms 900000`; the default 8,192 MiB host prefix cache | `--conversation-cache-mib 8192 --conversation-cache-slots 4` (its opt-in cache for several conversations); one request at a time |

Strata runs with its configuration on this machine (`strata-unsloth-ud-q4_k_xl.json`). Both
engines run with MTP: Strata refuses to start this model pack without speculation (`--spec` 2 or
more).

**Context length** (one request at a time):

| Context | Engine | TTFT (s) | Prefill tok/s | Decode tok/s | Tokens per round | Rounds/s |
|---|---|---:|---:|---:|---:|---:|
| ~8K | Infernix | **1.96** (1.94-1.99) | **4,129** | **139.7** (133.9-149.3) | 2.13 | **66.6** |
| ~8K | Strata | 12.28 (11.30-13.38) | 661 | 86.4 (83.4-90.3) | 1.94 | 44.9 |
| ~128K | Infernix | **17.95** (17.93-17.97) | **7,135** | **130.8** (126.0-135.5) | 2.38 | **56.0** |
| ~128K | Strata | 103.27 (95.54-110.99) | 1,247 | 77.2 (74.0-80.3) | 2.27 | 34.2 |
| ~250K | Infernix | **34.11** (33.63-34.58) | **7,335** | **152.4** (139.0-165.8) | 3.21 | **48.4** |
| ~250K | Strata | 158.07 (136.53-179.60) | 1,612 | 95.8 (86.0-105.7) | 2.89 | 33.2 |

**Follow-ups:**

| Follow-up | Infernix: cached / prompt tokens | Infernix TTFT (s) | Strata: cached / prompt tokens | Strata TTFT (s) |
|---|---:|---:|---:|---:|
| F1, newest ~250K conversation | 251,180 / 251,213 (99.99 %) | **0.24** | 250,136 / 250,163 (99.99 %) | 1.13 |
| F2, earlier 128K conversation | 0 / 129,147 | **17.91** | 0 / 128,111 | 99.12 |
| F3, earlier 8K conversation | 0 / 9,135 | **2.10** | 0 / 8,101 | 11.44 |

Neither engine kept the two older conversations: Strata keeps one conversation by default, and
Infernix's default 8 GiB host cache had been filled by the two 250K contexts after them. The
follow-up prompts differ in size because Infernix re-sends the previous answer's 1,024 reasoning
tokens in the prompt and Strata leaves them out.

**Agentic replay** (both engines with MTP; the replay at half length, `--scale 0.5`: 68 scored
requests per run with all three sessions and eleven subagents, prompts up to ~98K tokens; two
seeds):

| Metric | Strata | Infernix | Change |
|---|---:|---:|---:|
| Average time to first token (s) | 25.3 (24.5-26.2) | **6.3** (6.3-6.4) | −74.9 % |
| Median time to first token (s) | 16.50 (15.80-17.20) | **2.37** (2.35-2.39) | −85.6 % |
| 90th-percentile time to first token (s) | 59.4 (58.7-60.1) | **12.8** (12.7-13.0) | −78.4 % |
| Average TTFT, turns that continue a conversation (s) | 24.82 (24.54-25.09) | **5.93** (5.74-6.11) | −76.1 % |
| Average TTFT, new long prompts (s) | 37.2 (37.1-37.3) | **16.1** (14.6-17.6) | −56.7 % |
| Prompt tokens served from cache | 64.3 % (59.1-69.5) | **83.9 %** (83.4-84.4) | +19.6 points |
| Prompt tokens prefilled | 932K (796K-1,067K) | **425K** (408K-442K) | −53.6 % |
| Main-session turns that re-prefilled the whole prompt (of 41) | 8 (6-10) | **0** | −8 |
| Prefill tok/s, requests with no cache hit | 1,507 (1,488-1,527) | **5,360** (4,727-5,994) | 3.6× |
| Output tok/s, one request decoding | 86 (85-87) | **123** (122-124) | +43.4 % |
| Output tok/s, two requests decoding (combined) | (one at a time) | 151 (150-152) | |
| Output tok/s, at the run's own batching | 86 (85-87) | **136** (134-137) | +57.8 % |
| Decode rounds/s, one request (engine speed) | 40.5 (40.2-40.9) | **70.4** (69.5-71.3) | +73.7 % |
| Tokens per round, one request (draft acceptance) | **2.12** | 1.75 (1.71-1.79) | −17.4 % |
| Workload wall time (min) | 18.3 (17.6-19.0) | **8.4** (7.5-9.2) | −54.4 % |

Infernix wrote more in both runs (39.8K and 48.5K output tokens against Strata's 35.8K and
31.3K) and still finished in less than half the time. Strata's drafts are accepted more often
(2.12 against 1.75 tokens per round); Infernix decodes faster because each round takes 43 % less
time. With two requests decoding, Infernix drafts for neither, so the pair shares 151 tok/s (see
the Flash-Next guide's `--max-concurrency`).

The agentic replay found two crashes, both fixed on the `Infernix` branch: `c9ee0445` (a long
prompt's layer walk outlived the prefix-cache events it waited on) and `9061adb8` (with the SSD
expert tier, which this machine uses only when the experts do not fit in RAM). Seed 42's Infernix
run used `c9ee0445`, seed 43's `9061adb8`; with every expert in RAM the two behave the same. The
context-length tables above ran on `a436fb5a`, before both fixes, and were not repeated; neither
fix changes the arithmetic.

### Qwen3.8-27B: Infernix vs NInfer

Both engines serve NInfer's official
[Qwen3.8-27B-nvfp4-NInfer](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) artifact
(`qwen3_8_27b_nvfp4.ninfer`, with its DFlash2 drafter and proposal head). "NInfer" is upstream
`master` at `68c54356` (5 October 2026) plus only the Windows port; Infernix is branch `Infernix`
at `a436fb5a`.

- **Agentic replay:** the launch line I use for this model, the same flags on both engines:
  `--max-concurrency 2 --spec dflash2 --draft-tokens 7 --lm-head-draft --kv-dtype int8
  --preserve-thinking --host-context-mib 52000 --prefill-chunk 4096 --kv-capacity auto
  --default-thinking-budget 16384`. Infernix also takes the flags NInfer does not have:
  `--ngram-draft-tokens 31 --ngram-min-match 12`, the n-gram archive options,
  `--vram-headroom-mib 0`, `--tolerant-tool-calls` and `--thinking-budget-message`. Both run at a
  170,000-token context, the largest at which NInfer starts with two concurrent requests.
- **Context-length test:** `--kv-dtype int8 --max-context 240000 --kv-capacity 240000` on both,
  without speculation and with `--spec dflash2 --draft-tokens 7 --lm-head-draft`. NInfer cannot
  reserve 262,144 tokens of KV beside this artifact with DFlash2 loaded, so the long point is
  ~230K tokens. Each engine keeps its default host prefix cache (NInfer 9.46 GiB, Infernix
  8 GiB).

**Agentic replay** (three seeds):

| Metric | NInfer | Infernix | Change |
|---|---:|---:|---:|
| Average time to first token (s) | 5.7 (4.9-7.0) | **2.7** (2.3-3.0) | −51.7 % |
| Median time to first token (s) | 1.47 (1.14-1.95) | **0.58** (0.50-0.70) | −59.5 % |
| 90th-percentile time to first token (s) | 15.7 (14.4-17.8) | **7.5** (6.9-8.0) | −51.6 % |
| Average TTFT, turns that continue a conversation (s) | 5.64 (4.59-7.21) | **2.42** (1.98-2.97) | −54.8 % |
| Average TTFT, new long prompts (s) | 10.7 (10.4-11.0) | **6.8** (6.6-7.0) | −36.4 % |
| Prompt tokens served from cache | 83.7 % (82.4-84.6) | **90.4 %** (90.3-90.6) | +6.7 points |
| Prompt tokens prefilled | 982K (931K-1,081K) | **569K** (549K-581K) | −41.9 % |
| Prefill tok/s, requests with no cache hit | 6,127 (6,094-6,153) | **8,254** (8,234-8,284) | +34.7 % |
| Output tok/s, one request decoding | 199 (193-202) | **235** (220-246) | +17.9 % |
| Output tok/s, two requests decoding (combined) | 349 (344-357) | **367** (354-387) | +5.2 % |
| Output tok/s, at the run's own batching | 249 (239-257) | **282** (276-290) | +13.2 % |
| Decode rounds/s, one request (engine speed) | 58.4 (58.2-58.7) | **60.7** (60.5-60.9) | +3.8 % |
| Tokens per round, one request (draft acceptance) | 3.41 (3.32-3.46) | **3.87** (3.63-4.04) | +13.5 % |
| Workload wall time (min) | 16.9 (15.1-18.9) | **12.5** (9.9-15.7) | −24.7 % |

Wall time also depends on how much each run happened to write, which sampling changes from run to
run: on seed 42 Infernix wrote 31 % more output (219K against 167K tokens) and finished 4 % later;
on seeds 43 and 44 it wrote 24-30 % less and finished 37-41 % sooner. Every other row compares
rates or the same requests.

**Context length** (one request at a time):

| Context | Engine and speculation | TTFT (s) | Prefill tok/s | Decode tok/s | Tokens per round | Rounds/s |
|---|---|---:|---:|---:|---:|---:|
| ~8K | NInfer, none | 0.75 | 10,728 | 79.4 | | |
| ~8K | Infernix, none | 0.75 | 10,758 | 79.8 | | |
| ~8K | NInfer, DFlash2 | 0.77 | 10,460 | 200.0 (198.5-201.8) | 3.13 | 64.0 |
| ~8K | Infernix, DFlash2 | 0.77 | 10,556 | 209.1 (195.0-220.3) | 3.15 | 66.5 |
| ~8K | Infernix, DFlash2 + n-gram | 0.78 | 10,425 | 236.5 (221.7-254.4) | 3.55 | 66.6 |
| ~128K | NInfer, none | 29.36 | 4,363 | 66.6 | | |
| ~128K | Infernix, none | **21.88** | **5,854** | 67.2 | | |
| ~128K | NInfer, DFlash2 | 29.57 | 4,332 | 187.0 (167.5-206.5) | 3.43 | 54.5 |
| ~128K | Infernix, DFlash2 | **22.14** | **5,785** | 212.2 (200.3-224.1) | 3.72 | 57.2 |
| ~128K | Infernix, DFlash2 + n-gram | 22.21 | 5,768 | 235.2 (233.5-236.9) | 4.14 | 56.9 |
| ~230K | NInfer, none | 82.58 | 2,786 | 58.9 | | |
| ~230K | Infernix, none | **54.67** | **4,209** | 59.5 | | |
| ~230K | NInfer, DFlash2 | 82.70 | 2,782 | **227.3** (186.5-268.1) | 4.67 | 48.8 |
| ~230K | Infernix, DFlash2 | **55.16** | **4,172** | 161.6 (144.5-178.6) | 3.15 | 51.3 |
| ~230K | Infernix, DFlash2 + n-gram | 55.29 | 4,161 | 163.4 (153.3-173.6) | 3.20 | 51.2 |

Prefill is level at 8K and 1.34-1.51× faster from 128K. Decode without speculation is level
(within 1 %); with DFlash2 Infernix runs 4-5 % more rounds per second. At ~230K NInfer's DFlash2
decode is faster because the text it wrote there accepted more of each draft (4.67 against 3.15
tokens per round); the agentic replay, over hundreds of outputs, shows Infernix ahead.

**Follow-ups:**

| Follow-up | Engine and speculation | Cached / prompt tokens | TTFT (s) |
|---|---|---:|---:|
| F1, newest ~230K conversation | NInfer, none / DFlash2 | 231,104 / 231,137 | 0.19 / 0.22 |
| F1, newest ~230K conversation | Infernix, none / DFlash2 | 231,104 / 231,137 | **0.12 / 0.12** |
| F2, earlier 128K conversation | NInfer, none / DFlash2 | 0 / 129,147 | 30.84 / 31.14 |
| F2, earlier 128K conversation | Infernix, none / DFlash2 | 0 / 129,147 | **22.21 / 22.45** |
| F3, earlier 8K conversation | NInfer, none / DFlash2 | 0 and 8,072 / 9,135 | 1.00 / **0.30** |
| F3, earlier 8K conversation | Infernix, none / DFlash2 | 0 / 9,135 | 0.87 / 0.87 |

With each engine's default host cache, both kept only the newest long conversation; NInfer, whose
default is 1.46 GiB larger, kept the earlier 8K one in one of its two runs. The agentic replay
gives both engines the same 52,000 MiB host cache.

**Quality.** Perplexity on the repository's fixed corpus (all 16 streams, 1,044,876 scored tokens,
16K context, 8K stride, INT8 KV) with each engine's own `perplexity` tool: NInfer 4.8264, Infernix
4.8398 (+0.28 %). The [perplexity guide](docs/perplexity.md) measured run-to-run shifts of up to
~1 % on this NVFP4 artifact from the prefill chunking alone, so the two engines are level. The
offline scorer ran at 5,434 tok/s on NInfer and 5,303 tok/s on Infernix (−2.4 %, one run).

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
- **[Qwen3.8-Flash-Next-NVFP4-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVFP4-Infernix)**
  (`.infernix`): [nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4)
  converted with recipe B (NVIDIA's NVFP4 experts bit-exact, Q8 dense projections), with the MTP
  drafter and a proposal head for `--lm-head-draft`; the 52 GB n-gram volume is built from the same
  checkpoint (see the [Flash-Next guide](docs/qwen3_8-flash-next.md#convert)).

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
