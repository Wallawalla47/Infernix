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

- **Faster than Strata** on the same RTX 5090, both with the model's MTP drafter: 4.6-6.4× sooner
  to the first token and 1.6-1.8× the decode speed from 8K to 250K tokens of context, and in a
  replayed agentic coding workload under a quarter of Strata's average time to first token (5.4 s
  against 25.3 s) and less than half its wall time
  ([benchmarks](#qwen38-flash-next-nvfp4-infernix-vs-strata)).
- **Quality at least Strata's**: teacher-forced perplexity 4.654 against Strata's 4.844 on the same
  texts (lower is better); every optimisation was held to that bar.
- **Two conversions of NVIDIA's checkpoint** on Hugging Face:
  [Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix)
  (**recommended**: the dense projections in 8 bits, ~22 % faster decode, perplexity 4.654) and
  [Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix)
  (bit-exact in every weight the model computes with, perplexity 4.666); see [models](#models).
- **Long context**: 262K tokens natively, up to 1M with YaRN (`--rope-yarn-factor`); a 250K-token
  prompt prefills at ~7.3K tok/s (34 s), and decoding with MTP stayed at 136-164 tok/s from 8K to
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
  tolerant recovery of broken calls (`--tolerant-tool-calls`, which also returns complete calls a
  model strands in thinking it never closed; from Woesch-Nich's fix with Hundsbuah's review), a
  call the answer only quotes inside a Markdown code fence or inline code kept as text (as Strata
  0.1.40, #1058),
  Anthropic and OpenAI options used by
  Claude Code, Qwen Code, Codex, Zed and GitHub Copilot, and correct handling of reasoning output.
- **Faster prefill**: FlashAttention-2 style INT8 prompt attention, FP4/FP8 Tensor Core paths for
  NVFP4, FP8 and K8V4 KV, wave-aligned chunks — a third to two thirds less prompt-attention time
  on long prompts.
- **Faster decode**: overlapped (PDL) decode kernels, verification-width kernel tuning, DFlash2
  tree verification and n-gram copy drafting alongside MTP/DFlash2.
- **Smaller KV**: the `vq2` and `k4v2` formats hold three to four times the context of INT8.

## Benchmarks

Measured on 7-8 October 2026 on one machine: an RTX 5090 (32 GB, driver 617.14) in a **PCIe Gen5
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
  - Qwen3.8-Flash-Next servers start from a saved expert-cache state, as Infernix does in normal
    use after its first session (`<artifact>.expert-state`, here written by an earlier session on
    unrelated Wikipedia text). Started cold instead, Infernix measured within 0.5 % on the
    context-length test.
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
| Version | branch `Infernix` at `3b35ccf9` (artifact converted at `e3be72cd`) | 0.1.40 (release engine; server `82f46a8`, 0.1.40.1) |
| Model | [nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) as the Dense8 conversion (recipe B): NVIDIA's NVFP4 experts bit-exact, dense projections and `lm_head` in Q8 (group 32), MTP drafter in Q8 with Q4 experts, proposal head ([Hugging Face](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix)) | Unsloth `Qwen3.8-Flash-Next-UD-Q4_K_XL` GGUF (four shards) as Strata's native pack, with Strata's Q2_0 MTP drafter |
| KV cache and context | INT8, 262,144 tokens | INT8, 262,144 tokens (32,768 cells per layer resident in VRAM) |
| Speculation | MTP, `--draft-tokens 4 --lm-head-draft` | MTP, `--spec 4 --spec-min-p 0.70` |
| Other settings | n-gram volume on NVMe, `--prefill-chunk 4096` | `--expert-cache auto --prefill auto --resident-budget-gib 68 --pool-workers 15 --pcie-frac 0.00` |
| Agentic replay adds | `--max-concurrency 2 --pending-timeout-ms 900000`; the default 8,192 MiB host prefix cache | `--conversation-cache-mib 8192 --conversation-cache-slots 4` (its opt-in cache for several conversations); one request at a time |

Strata runs with its configuration on this machine (`strata-unsloth-ud-q4_k_xl.json`). Both
engines run with MTP: Strata refuses to start this model pack without speculation (`--spec` 2 or
more).

This compares each engine as it is run in practice, each with its own conversion of the model and
its own drafter, so it does not separate engine speed from weight format; the quality rows below
cover the formats. Each engine ran in its own server sessions, one engine after the other rather
than interleaved, so drift between sessions is not controlled beyond the discarded warm-up request
and the saved expert-cache state.

The two engines run different quantizations of the same model: Strata cannot run NVFP4, and
Infernix cannot run the GGUF. Neither file is uniformly higher precision. The GGUF is smaller
overall (111 GB, against about 127 GB for the Infernix artifact with its n-gram volume) but stores
the routed experts, which dominate the bytes read per token, at about 5.1 bits per weight against
NVFP4's 4.5, so Strata reads slightly more expert data per token. Infernix keeps the dense layers
(8-bit and BF16) and the n-gram table (8-bit) at higher precision. On the same token ids Infernix's
quality is slightly better (perplexity 4.664 at the measured version and 4.654 with the
2026-10-08 kernels, against 4.844).

**Context length** (one request at a time):

| Context | Engine | TTFT (s) | Prefill tok/s | Decode tok/s | Tokens per round | Rounds/s |
|---|---|---:|---:|---:|---:|---:|
| ~8K | Infernix | **1.93** (1.92-1.93) | **4,192** | **138.8** (138.0-139.9) | **2.03** | **68.7** |
| ~8K | Strata | 12.28 (11.30-13.38) | 661 | 86.4 (83.4-90.3) | 1.94 | 44.9 |
| ~128K | Infernix | **17.86** (17.84-17.87) | **7,174** | **136.2** (135.8-136.7) | **2.66** | **51.5** |
| ~128K | Strata | 103.27 (95.54-110.99) | 1,247 | 77.2 (74.0-80.3) | 2.27 | 34.2 |
| ~250K | Infernix | **34.24** (34.03-34.45) | **7,306** | **153.3** (142.6-164.1) | **3.13** | **49.7** |
| ~250K | Strata | 158.07 (136.53-179.60) | 1,612 | 95.8 (86.0-105.7) | 2.89 | 33.2 |

**Follow-ups:**

| Follow-up | Infernix: cached / prompt tokens | Infernix TTFT (s) | Strata: cached / prompt tokens | Strata TTFT (s) |
|---|---:|---:|---:|---:|
| F1, newest ~250K conversation | 251,180 / 251,213 (99.99 %) | **0.25** | 250,136 / 250,163 (99.99 %) | 1.13 |
| F2, earlier 128K conversation | 0 / 129,147 | **17.79** | 0 / 128,111 | 99.12 |
| F3, earlier 8K conversation | 0 / 9,135 | **2.09** | 0 / 8,101 | 11.44 |

Neither engine kept the two older conversations: Strata keeps one conversation by default, and
Infernix's default 8 GiB host cache had been filled by the two 250K contexts after them. The
follow-up prompts differ in size because Infernix re-sends the previous answer's 1,024 reasoning
tokens in the prompt and Strata leaves them out.

**Agentic replay** (both engines with MTP; the replay at half length, `--scale 0.5`: 68 scored
requests per run with all three sessions and eleven subagents, prompts up to ~98K tokens; two
seeds):

| Metric | Strata | Infernix | Change |
|---|---:|---:|---:|
| Average time to first token (s) | 25.3 (24.5-26.2) | **5.4** (4.6-6.2) | −78.5 % |
| Median time to first token (s) | 16.50 (15.80-17.20) | **2.55** (2.23-2.88) | −84.6 % |
| 90th-percentile time to first token (s) | 59.4 (58.7-60.1) | **13.0** (10.4-15.5) | −78.2 % |
| Average TTFT, turns that continue a conversation (s) | 24.82 (24.54-25.09) | **4.87** (3.94-5.80) | −80.3 % |
| Average TTFT, new long prompts (s) | 37.2 (37.1-37.3) | **14.5** (12.4-16.7) | −60.9 % |
| Prompt tokens served from cache | 64.3 % (59.1-69.5) | **83.7 %** (83.2-84.1) | +19.4 points |
| Prompt tokens prefilled | 932K (796K-1,067K) | **436K** (419K-453K) | −52.4 % |
| Main-session turns that re-prefilled the whole prompt (of 41) | 8 (6-10) | **0** | −8 |
| Prefill tok/s, requests with no cache hit | 1,494 (1,460-1,527) | **5,830** (5,646-6,015) | 3.9× |
| Output tok/s, one request decoding | 86 (85-87) | **129** (127-130) | +49.8 % |
| Output tok/s, two requests decoding (combined) | (one at a time) | 167 (164-170) | |
| Output tok/s, at the run's own batching | 86 (85-87) | **146** (143-149) | +69.8 % |
| Decode rounds/s, one request (engine speed) | 40.5 (40.2-40.9) | **64.9** (64.7-65.2) | +60.2 % |
| Tokens per round, one request (draft acceptance) | **2.12** | 1.98 (1.95-2.01) | −6.5 % |
| Workload wall time (min) | 18.3 (17.6-19.0) | **8.1** (7.9-8.3) | −55.8 % |

Strata's columns come from its runs of the same two seeds on 7 October (same Strata, same
workload); Infernix's were run on `3b35ccf9` with the NVIDIA-source artifact. Infernix wrote more in
both runs (45.6K and 45.2K output tokens against Strata's 35.8K and 31.3K) and still finished in
less than half the time.

Strata's drafts are still accepted a little more often (2.12 against 1.98 tokens per round);
Infernix decodes faster because each round takes nearly 40 % less time. The difference is which
drafts each engine offers, not output quality: both verify every draft, and on the same texts
Infernix's perplexity is 4.664 against Strata's 4.844. Strata offers a draft only while its draft
layer is at least 70 % sure of it; Infernix cuts a single request's drafts at the first below
50 %, and two requests decoding together now draft for both with one shared length (output
unchanged either way; see the Flash-Next guide's `--max-concurrency`).

Earlier versions of these tables were measured on an artifact converted by mistake from
`RadixArk/Qwen3.8-Flash-Next-NVFP4`, another Model Optimizer quantization of the same model with
its own expert calibration. NVIDIA's checkpoint measured within a few per cent of it on speed
(context-length decode 1.6-4.1 % lower, agentic replay level) and slightly higher perplexity (4.664
against 4.600, both below Strata's).

The agentic replay found two crashes, both fixed on the `Infernix` branch: `c9ee0445` (a long
prompt's layer walk outlived the prefix-cache events it waited on) and `9061adb8` (with the SSD
expert tier, which this machine uses only when the experts do not fit in RAM).

### Bit-exact against Dense8

The two Qwen3.8-Flash-Next conversions ([models](#models)) on the same build (`Infernix` with the
2026-10-08 BF16 kernels), the context-length test's prompts and settings, three interleaved
sessions per artifact. Cells are means over the prompts of each request's median.

| Context | Artifact | TTFT (s) | Prefill tok/s | Decode tok/s | Tokens per round | Rounds/s |
|---|---|---:|---:|---:|---:|---:|
| ~8K | Dense8 | **1.92** | **4,204** | **140.1** | 2.74 | 54.9 |
| ~8K | Bit-exact | 2.07 | 3,905 | 116.1 | 2.06 | 56.6 |
| ~128K | Dense8 | 17.96 | 7,130 | **136.3** | 2.47 | 55.5 |
| ~128K | Bit-exact | 17.91 | 7,152 | 107.5 | 2.29 | 47.6 |
| ~250K | Dense8 | 34.22 | 7,311 | **159.9** | 3.47 | 47.4 |
| ~250K | Bit-exact | 34.01 | 7,358 | 130.7 | 3.05 | 44.1 |

Per request, the bit-exact artifact decoded 18 % slower (geometric mean of ten requests; 11-25 %),
reached the first token 2.5 % later (6-8 % at 8K, level from 128K) and scored the same quality
(4.666 against 4.654). It reads ~8.6 GB of BF16 dense weights per token against ~5.1 GB, and its
dense weights take 3.5 GB more VRAM, which leaves about 1,270 fewer expert-cache frames. Tokens
per round differ because the two write different text; one 8K prompt sent Dense8 into a repetitive
answer (3.86 tokens per round), which lifts its 8K mean.

Before the BF16 kernel work (`1df65d19`), the bit-exact artifact ran its BF16 projections on routes
tuned for other models' shapes. The same prompts, settings and artifact on that build and on the
current one, three interleaved sessions each:

| Context | Build | TTFT (s) | Prefill tok/s | Decode tok/s | Tokens per round |
|---|---|---:|---:|---:|---:|
| ~8K | before | 4.70 | 1,718 | 116.3 | 2.01 |
| ~8K | current | **2.05** | **3,944** | 115.9 | 2.06 |
| ~128K | before | 73.35 | 1,746 | 111.1 | 2.80 |
| ~128K | current | **17.89** | **7,158** | 107.6 | 2.29 |
| ~250K | before | 145.05 | 1,724 | 136.5 | 3.74 |
| ~250K | current | **33.97** | **7,366** | 131.0 | 3.05 |

Prefill is 2.3× faster at 8K and 4.1-4.3× from 128K. Decode is level: −1.2 % per request
(geometric mean of ten; −9 % to +7 %), within the spread that the two builds' different texts give
(tokens per round moved by up to 0.7). The decode kernels save little against a ~20 ms round: with
MTP's 4 drafts a round verifies at most 5 columns, where the old BF16 head already ran at the DRAM
floor, and the tensor-core head pays from 6 columns (n-gram and tree rounds: 1,056 → 706 µs at 8
columns, 1,502 → 707 µs at 9-16). The bit-exact artifact's remaining decode gap to Dense8 is its
BF16 weight bytes, not kernel time. Dense8 on the same two builds measured level (decode +0.4 % per
request, −7 % to +7 %; TTFT −0.2 %), so the Strata comparison above stands for the current build.

### Qwen3.8-27B: Infernix vs NInfer

Both engines serve NInfer's official
[Qwen3.8-27B-nvfp4-NInfer](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) artifact
(`qwen3_8_27b_nvfp4.ninfer`, with its DFlash2 drafter and proposal head). "NInfer" is
[Neroued/ninfer](https://github.com/Neroued/ninfer) `master` at `68c54356` (5 October 2026) plus only the Windows port; Infernix is branch `Infernix`
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
4.8398 (+0.28 %). Both scored a Windows checkout of the corpus in which Git had converted its line
endings to CRLF; the corpus is now kept as LF on every checkout, which tokenizes slightly
differently, so a new checkout's absolute values will differ a little while the comparison stands.
The [perplexity guide](docs/perplexity.md) measured run-to-run shifts of up to ~1 % on this NVFP4
artifact from the prefill chunking alone, so the two engines are level. The
offline scorer ran at 5,434 tok/s on NInfer and 5,303 tok/s on Infernix (−2.4 %, one run).

## Quick start (Windows)

Prerequisites: Visual Studio 2026 (MSVC), the CUDA 13 toolkit, and FFmpeg + curl from vcpkg
(`x64-windows`, at `C:\vcpkg`); adjust the paths at the top of `build_native.bat` for your machine.

```bat
build_native.bat configure
build_native.bat build
```

Sources compile as UTF-8 (`/utf-8`), so the build works under any Windows system code page
(reported by Woesch-Nich).

The server is `build-windows\apps\Release\infernix-serve.exe`, with the FFmpeg, curl and zlib DLLs
copied next to it. The launch I use for Qwen3.8-27B on a single 32 GB RTX 5090 (stop any other
resident model first):

```bat
infernix-serve.exe qwen3_8_27b_nvfp4-nvidia.ninfer --host 127.0.0.1 --port 8080 --max-context 240000 --max-concurrency 2 --spec dflash2 --draft-tokens 7 --lm-head-draft --ngram-draft-tokens 15 --ngram-min-match 12 --kv-dtype int8 --preserve-thinking --host-context-mib 52000 --pending-timeout-ms 900000 --prefill-chunk 4096 --kv-capacity auto --vram-headroom-mib 0 --log-colours on --ngram-archive-mib 2048 --ngram-session-mib 256 --ngram-native-sessions --request-log-jsonl log.json --default-thinking-budget 16384 --thinking-budget-message "Considering the limited time available to the user, I must stop thinking now. Time to act:" --tolerant-tool-calls
```

For Qwen3.8-Flash-Next, see the [Flash-Next guide](docs/qwen3_8-flash-next.md#run):

```bat
infernix-serve.exe Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix-00001-of-00003.infernix --ngram-volume Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram --kv-dtype int8 --max-context 65536 --prefill-chunk 4096 --spec mtp --draft-tokens 4 --lm-head-draft
```

Add `--prefix-cache-file PATH` to keep the prefix cache across restarts (with
`--prefix-cache-save-mins N` it is also saved every N minutes, so a crash loses less), and
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
- **Qwen3.8-Flash-Next**, two `.infernix` conversions of
  [nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4), each
  with the MTP drafter, a proposal head for `--lm-head-draft` and the same 52 GB n-gram volume
  (see the [Flash-Next guide](docs/qwen3_8-flash-next.md#convert)):
  - **[Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix)
    (recommended)**: NVIDIA's NVFP4 experts and n-gram table bit-exact, the dense projections and
    `lm_head` in 8 bits. It decodes ~22 % faster than the bit-exact conversion at the same measured
    quality, and it is the artifact the benchmarks above use.
  - **[Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix)**
    (bit-exact): every weight of the main model and vision tower exactly as NVIDIA stores it. Choose
    it when the weights must be NVIDIA's unchanged; it is slower.

  | | [Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix) (bit-exact, recipe A) | [Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix) (recipe B, **recommended**) |
  |---|---|---|
  | Routed experts (24,576, NVFP4) | bit-exact | bit-exact |
  | PLE n-gram table (FP8) | bit-exact | bit-exact |
  | Dense projections and `lm_head` (629 tensors, 97 % of the dense bytes) | bit-exact (BF16) | **8-bit** `q8_g32_fp16` from NVIDIA's BF16 |
  | Router, gates, GDN `a`/`b`, QSA query/gate/key/value/indexer, embedding, norms, vision | bit-exact (BF16) | bit-exact (BF16) |
  | MTP drafter (drafting only, never changes output) | `q8_g32_fp16`, experts `q4_g64_fp16` | same |
  | Proposal head (drafting only, not in NVIDIA's checkpoint) | `q4_g64_fp16` | same |
  | Model files | 80.4 GB | 76.9 GB |
  | Perplexity, frozen texts (Strata 4.844) | 4.666 | 4.654 |
  | Decode speed ([measured](#bit-exact-against-dense8)) | ~18 % below Dense8 | 136-160 tok/s from 8K to 250K |

  "Bit-exact" was checked tensor by tensor against the checkpoint with
  `python -m tools.flash_next.verify_artifact`: every stored word equal, nothing re-derived. The
  two measured level on quality; Dense8 is the faster of the two because it reads fewer dense bytes
  per token and leaves 3.5 GB more VRAM to the expert cache.

Infernix loads `.infernix` and `.ninfer` artifacts (the same format); `python -m tools.convert`
writes either, and `python -m tools.artifact.rename` renames an existing one in place.

## What else is in the engine

- **Serving**: OpenAI Chat Completions and Responses, Anthropic Messages, `/v1/decide` for
  decisions read from probabilities (yes/no, choice, score, number, point, box), streaming,
  bounded FIFO admission with one to eight concurrent requests, a request log with rotation, a
  console statistics panel (with the VRAM expert-cache hit rate for Qwen3.8-Flash-Next, and n-gram
  columns only when n-gram drafting is on) and a categorised `--help`.
- **Constrained decoding** (from NInfer, October 2026): JSON object and JSON Schema response
  formats, strict tool arguments and tool choice, and GBNF, regex and choice through the CLI and
  the `structured_outputs` extension, on every model and speculative round type, Qwen3.8-Flash-Next
  included.
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
David Oelfke, Fedor Suchkov, Yunado,
[IST-DASLab](https://github.com/IST-DASLab) (llmq), the authors of HyperQuant and Four Over Six,
the Qwen team for the models, and NVIDIA for the NVFP4 checkpoints.
