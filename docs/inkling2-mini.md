# Inkling 2 Mini (`Inkling2LoopForCausalLM`)

Infernix runs Thinking Machines'
[Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) on one RTX 5090 (32 GB),
with the whole model resident in VRAM. The published artifact,
[Inkling-2-Mini-Infernix](https://huggingface.co/wallawalla47/Inkling-2-Mini-Infernix), is one
`.infernix` file of about 20.3 GB that holds the text model, the model's DSpark drafter, its vision
tower and a token-frequency ranking for `--lm-head-draft`. The conversion is bit-exact: the
checkpoint's weights are unchanged ([what is stored](#what-the-artifact-stores)).

Generation works through `infernix` and `infernix-serve` (OpenAI Chat Completions and Responses,
Anthropic Messages) with DSpark or [DFlash2](#dflash2-drafting) drafting, images, one to eight lanes, every KV format and the prefix
cache, and [constrained output and tool calls](#constrained-output-and-tool-calls).
`infernix-perplexity` scores it, on the default route or with `--lossless`, and `infernix-serve` answers
[`/v1/decide`](serving.md#decisions)'s readout questions (each prompt ends with the answer's opener, so the model answers
without thinking first). It takes [LoRA adapters](lora.md#inkling-2) and [ablation sidecars](ablate.md). Several features
of the Qwen models are [not available on Inkling yet](#not-supported-yet), among them video.

**Engine version:** this model needs a build from `main` at `a487ad24f` or later, or a Windows release after
`windows-2026.11.20`.

**Launcher:** the launcher runs this model and shows the options it takes, `--lora` and `--ablate` among them.

## Model

| | |
|---|---|
| Architecture | `Inkling2LoopForConditionalGeneration` in the checkpoint; its text model is `Inkling2LoopForCausalLM`, model family `inkling2_loop` |
| Layers | 24 physical layers; the middle 12 run three times, so a token makes 48 layer visits |
| Experts | 128 routed experts a layer, 8 per token. Layers 0-19 NVFP4, layers 20-23 MXFP8 |
| Attention | 20 sliding-window layers (1,024 tokens) and 4 full-attention layers |
| Hidden size, vocabulary | 2,304; 200,058 assigned tokens |
| Context | up to 1,048,576 tokens |
| Drafter | DSpark (3 layers, blocks of 8: up to 7 drafts a round), bundled; or incoai's DFlash2 drafter (5 layers, blocks of 8) as a [companion artifact](#dflash2-drafting) |
| Input | text and images (no video) |
| License | Apache 2.0, with Thinking Machines' [Model Acceptable Use Policy](https://thinkingmachines.ai/model-acceptable-use-policy/) linked from its card |

## Requirements

- **GPU.** Built and tuned on an RTX 5090 (`sm_120a`, 32 GB). Other cards have not been tested.
- **VRAM.** The weights take 17.8 GiB, 18.0 GiB with `--lm-head-draft` and 18.6 GiB with `--lm-head-draft --vision`.
  The execution workspace takes a further 1.8 GiB (3.1 GiB with `--vision`). What is left holds the KV cache, so the
  context you can run depends on the options: [`--plan`](#check-a-configuration-first) reports it for your machine.
- **RAM.** A peak working set of 1.9 GiB for a short `infernix` run at an 8,192-token context, including the load. The
  prefix cache's Host tier takes further RAM, down to `--ram-free-mib`, within what Windows lets the GPU page-lock
  (about 80 % of RAM on the RTX 5090 machine this was built on). Under `--ram-resize auto` (the server's default) it
  gives RAM back while other programs need it and grows again, past its startup size, as RAM frees; the startup line
  `Host tier auto: ...` names the bound that set its size.
- **Disk.** About 20.3 GB for the artifact.
- **Operating system.** Run and tested on Windows. The engine and launcher also build on Linux (GCC 13, CUDA 13.4), where
  this model has not been run.

## Download

```text
hf download wallawalla47/Inkling-2-Mini-Infernix Inkling-2-Mini-Infernix.infernix --local-dir <dir>
```

## Run

```bat
infernix-serve.exe Inkling-2-Mini-Infernix.infernix --max-context auto --kv-dtype int8 --spec dspark --draft-tokens auto --lm-head-draft
```

- `--max-context auto` takes the largest context one request can hold in the free VRAM: the whole KV
  pool. Without it `infernix-serve` uses 8,192 tokens and `infernix` 2,048.
- `--kv-dtype int8` stores the KV cache, the sliding windows included, in 8 bits: about half the memory
  of the default `bf16`, for slightly different output ([KV formats](#kv-formats)). `--kv-window-dtype`
  chooses the sliding windows' format on its own.
- `--spec dspark --draft-tokens auto` is what the default `--spec auto` chooses for this artifact, also with
  incoai's [DFlash2 drafter](#dflash2-drafting) converted beside it: DFlash2 took 2.4 % more wall time over the agentic
  replay. Name the DFlash2 companion with `--draft-model` for short prompts (measured up to about 600 tokens of
  context), where it is faster, and for 9 % more KV capacity.
- `--lm-head-draft` is optional: it changes the speed of drafting, never the output.
- Add `--vision` for images. Its tower and image buffers stay in VRAM and reduce the context.
- Add `--max-concurrency N` (1 to 8) for several lanes. The lanes share one KV pool, and each
  request takes pages as it grows, so any one of them can reach the full context. When the
  lanes together outgrow the pool, one request pauses and resumes later with the same output
  ([Context, lanes and the prefix cache](#context-lanes-and-the-prefix-cache)).

A single request from the command line (`--max-concurrency` and the cache options belong to
`infernix-serve`):

```bat
infernix.exe Inkling-2-Mini-Infernix.infernix --prompt "Explain what a looped transformer is." --max-context 32768 --max-new 2048 --kv-dtype int8 --spec dspark --draft-tokens auto --lm-head-draft
```

When a request sets no sampling options the engine uses temperature 1.0, top-p 1.0 and top-k 20: the
checkpoint's generation config sets the temperature (over the whole vocabulary), and top-k 20 is the shipped
default. `top_k: 0` (`--top-k 0`) with top-p 1 and min-p 0 samples the whole vocabulary, as the generation config
asks, by Gumbel-max (each verify column of DSpark draws the same way, so drafting changes nothing). At temperature 1
the top-20 cap leaves out 2.5 % of the probability on average on prose answers (median 0.34 %, 90th percentile 8.1 %,
99th 22 %; 2,182 sampled positions of eight answers, 2026-11-24). With top-p below 1 or min-p above 0, top-k 0 keeps
the 20-candidate domain, and `--lossless` always keeps it (its bits do not change). The whole-vocabulary draw is the
cheaper sampler call (an RTX 5090 op bench: 14 against 19 us at one row, 17-18 against 44 us at eight), but DSpark
accepts fewer drafts, since more draws leave the drafter's top token: on four seeded prose prompts (each run three
times, DSpark auto, one request) decode went from 222.0 to 213.7 tokens/s (-3.7 %) and acceptance from 0.364 to
0.340, with different texts in the two settings (2026-11-24).

### Check a configuration first

`--plan FILE` sizes a start on this machine without loading the model, writes the plan as JSON and
exits `0` (fits), `3` (does not fit) or `1` (no plan):

```bat
infernix-serve.exe Inkling-2-Mini-Infernix.infernix --max-context auto --max-concurrency 2 --kv-dtype int8 --spec dspark --draft-tokens auto --lm-head-draft --vision --plan plan.json
```

See [Sizing a start without loading](serving.md#sizing-a-start-without-loading).

## Not supported yet

Each of these fails with a clear error instead of running differently, except where the table says
otherwise. The engine refuses an option before the model loads. The options only `infernix-serve` or `infernix` reads
(the tool options, `--reasoning-loop`, `--expert-state FILE`, `--kv-capacity` without the cache) are refused
once the model has loaded; `--plan` reports them without loading.

| Feature | On Inkling 2 Mini |
|---|---|
| Video input | the request is refused; images work |
| `/v1/decide`'s generated types (`number`, `scalar`, `point`, `box`) | the request is refused (`question_type_unsupported`): they spell a number one digit a step, and Inkling's tokenizer writes digits in groups of up to three, so the answer would be noise. `noul`, `choice` and `score` work |
| A request ending in an assistant message (prefill) | the request is refused: only a new assistant turn is rendered |
| The lossless route's noise twin (`infernix-certify --noise-seed`) | refused: Inkling 2's own lossless kernels round to nearest only |
| `--reasoning-loop stop` or `conclude` | refused: the model has no reasoning-loop guard |
| `--preserve-thinking`, `--tolerant-tool-calls`, `--tool-parameter-names declared` | refused: Qwen template and tool-call options. The template renders the history's reasoning as given |
| N-gram drafting: `--ngram-volume`, `--ngram-draft-tokens`, `--ngram-min-match`, `--ngram-tree-nodes`, `--ngram-archive-mib`, `--ngram-session-mib`, `--ngram-residency` | refused |
| The MTP and DFlash backends | `--spec mtp` and `dflash` are refused: the model drafts with DSpark or a [DFlash2 companion](#dflash2-drafting). `--spec dflash2` without `--draft-model` is refused, and so is `--draft-model` with `--spec dspark` or `off`. So are `--spec-min-p`, `--draft-tree-nodes` and `--draft-tree-paths` (both drafters verify a chain) |
| `--lm-head-draft` when no drafter runs | refused: no DSpark in the artifact and no DFlash2 companion, `--spec off` or `--lossless` |
| Offloaded-expert options: `--expert-state`, `--expert-ram-mib`, `--expert-ram-layout`, `--expert-pin-mib`, `--expert-csf`, `--expert-stripe`, `--expert-tail-skip`, `--route-trace` | refused: the experts stay in VRAM |
| Qwen kernel, fast-path and media options: `--use-original-int8-prefill-kernel`, `--use-original-nvfp4-prefill-kernel`, `--prefill-split-workspace-mib`, `--fast-path`, `--no-fast-path`, `--media-cache-mib`, `--media-live-mib`, `--media-preprocess-threads`, `--context-cost-presets` | refused: Inkling has its own prompt route and image path, registers no fast path, and its prefix cache weighs a read against a re-prefill with a fixed fit of its own rates, which presets do not calibrate |
| Sizing options that a start would not use: `--vram-headroom-mib` and `--vram-past-budget` without `--max-context auto` or `--kv-capacity auto`; `--ram-free-mib` without the automatic Host tier and without `--ram-resize auto`; `--vision-max-merged` without `--vision` | refused |
| `--kv-capacity N` below `--max-context` | refused: the pool holds at least one request's whole context |
| Generation options on a `--causal-scoring` start (`--prefix-cache-file`, `--host-context-mib`, `--ssd-context-gib`, `--lm-head-draft`, the sizing options) | not refused: scoring runs without them, and the start lists them as unused in its log, as the other models do |
| `--logits-dtype bf16` | refused at startup: the logits are FP32 |
| `--rope-yarn-factor` other than 1 | refused at startup: the model has no RoPE and reaches its full context natively |
| `--expert-a4-activations` | refused at startup (it belongs to one Qwen3.8-Flash-Next artifact) |
| `--cache-tool-hold-secs` | refused unless `0`: not built for this model |
| `--device-snapshot-slots`, `--cache-host-state` other than `fp32`, `--host-context-mib 0` | refused: snapshots live in the Host tier, exactly. Use `--no-prefix-reuse` to turn the cache off |

## DSpark drafting

DSpark is the drafter Thinking Machines ships with the model. Each round it drafts up to seven tokens
and the full model verifies them, so greedy and sampled output are the same as plain decoding, token
for token.

- `--spec auto` (the default) drafts with DSpark when the artifact bundles it. `--spec off` decodes
  plainly.
- `--draft-tokens auto` uses the calibrated number of drafts each round, up to 7. With `--spec dspark`
  a fixed count from 1 to 7 also works.
- Requests decoding together draft together: their blocks are drafted at the same time, and one
  verify call checks the round's drafts, shared out so that the requests whose drafts are likelier to
  be accepted verify more of theirs. How many drafts a round verifies is the number their calibrated
  drafts make fastest on the verify times the rounds measure (a batch size whose drafted rounds turn
  out slower than its plain ones decodes plainly, trying drafts again now and then). Each request's
  tokens are still its plain decoding's. A request with presence or frequency penalties, a token constraint, forced tokens
  pending or less than a block of context left decodes plainly, and so does every request beside it
  in that round; so does a round under `--no-cuda-graph`.
- `--lossless` runs without a drafter.

### `--lm-head-draft`

With this option DSpark drafts from a shortlist of the 65,536 most frequent tokens instead of the whole
vocabulary, which makes each round's draft cheaper.
Infernix builds the shortlist at load from the ranking stored in the artifact
(`proposal/token_rank`). A token outside the list is never drafted; verification still produces it,
so **the output does not change**. An artifact without a ranking refuses the option, and it needs a
drafter: a start that runs none (`--spec off`, an artifact without DSpark, `--lossless`) refuses it.

On nine English, code, Chinese and other-language prompts (RTX 5090, DSpark `auto`, two runs each) it decoded 2.4 to 5.8 %
faster than DSpark on the full head on the default route, 2.9 to 4.9 % faster with `--q8-weights` and 2.5 to 4.0 % faster
with `--a8-activations`. Its tables take about 180 MB of VRAM.

## DFlash2 drafting

incoai's [Inkling-2-Mini-DFlash2](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2) is a DFlash2 drafter trained for
this model: five sliding-window layers (2,048 tokens) that read the model's hidden states (the inputs of layer visits 5,
23 and 41 and the final normed state), draft a block of 8 (the last token and seven masks) in one pass, and pick each
draft among the head's 16 best tokens with a conditional selector. Infernix runs it from a companion artifact beside the
main one ([conversion](weight-conversion.md#start-with-an-official-recipe): recipe `inkling2_mini_dflash2`, 635 MB), as
it runs Qwen3.8-Flash-Next's DFlash2 drafter:

```bat
infernix-serve.exe Inkling-2-Mini-Infernix.infernix --max-context auto --kv-dtype int8 --spec dflash2 --draft-model Inkling-2-Mini-DFlash2-Infernix.infernix --draft-tokens auto --lm-head-draft
```

- Verification is DSpark's: each verify column samples as plain decoding would, and a draft stands only while it equals
  that draw, so greedy and sampled output are plain decoding's, token for token.
- With `--spec dflash2` the bundled DSpark drafter is not loaded: none of its weights or KV pages take VRAM, which goes
  to the context instead.
- `--lm-head-draft` drafts from the same shortlist of 65,536 tokens, and needs the artifact's ranking as with DSpark.
- `--draft-tokens auto` verifies the calibrated number of the block's 7 drafts each round, a draft costing more on a
  longer context; a fixed count from 1 to 7
  also works, but `auto` beat every fixed count from 3 to 7 by 5-8 % one request at a time and tied the best at four
  requests ([draft length](maintainer/dflash.md#inkling-2-mini)). Up to eight requests decoding together draft their blocks in one call and share out one verify call, as
  with DSpark.
- The prefix cache stores the drafter's rings with each snapshot, so a resumed request drafts as a fresh one would. A
  `--prefix-cache-file` written with another drafter is not restored.
- `--spec auto` drafts with the companion that `--draft-model` names, otherwise with DSpark ([why](#agentic-replay));
  an artifact converted without DSpark drafts with the only companion of this model lying beside it.
- A request sampling the whole vocabulary (`top_k: 0`) has its drafts drawn with the noise its own verification draws
  with (the target's Gumbel noise at each verify column), so a draft is the target's own token far more often than
  the drafter's likeliest token alone would be; its output is unchanged. Greedy requests and the default top-20
  sampling draft the selector's likeliest path.

### Speed and memory against DSpark

Served on an RTX 5090 in the launcher's configuration (eight lanes, `--max-context 240000`, `--kv-dtype int8`,
`--lm-head-draft`, `--draft-tokens auto`, `--vision --vision-offload on`, the thinking budget), 512 tokens a request,
seeded, the arms run in the order DSpark, DFlash2, the NVFP4-MLP companion twice, DFlash2, DSpark (2026-11-26, DSpark
with its INT8 full-layer context, no warm-up arm: the two DSpark runs agree within 0.6 % one request at a time). The
one-request rows send eight short prompts (or four questions on a shared 122K-token prefix) one after another; the
others send them at once. Output tokens per second over the phase's wall time, prefill and first token included (the
mean, the two runs' range in brackets); every arm produced the same text in every phase. Several requests at once vary
from run to run (the rounds' interleaving), so those rows carry the spread their ranges show:

| Phase | DSpark | DFlash2 | DFlash2 against DSpark |
|---|---|---|---|
| 1 request at a time, 8 prompts, default sampling (top 20, T 1) | 278.0 (277.8-278.1) | 319.4 (317.4-321.3) | +14.9 % |
| 1 request at a time, 8 prompts, greedy | 309.9 (309.4-310.4) | 392.1 (391.8-392.4) | +26.5 % |
| 1 request at a time, 8 prompts, whole vocabulary (`top_k 0`) | 281.2 (280.4-282.1) | 355.5 (355.4-355.5) | +26.4 % |
| 1 request at a time, 4 questions on a 122K-token prefix | 180.0 (179.5-180.5) | 168.2 (167.9-168.4) | -6.6 % |
| 4 requests at once, default sampling | 499.2 (496.1-502.3) | 578.5 (576.8-580.2) | +15.9 % |
| 8 requests at once, default sampling | 683.5 (669.0-698.0) | 814.4 (804.7-824.2) | +19.2 % |
| 4 questions at once on the 122K-token prefix | 317.0 (315.1-319.0) | 294.2 (292.0-296.4) | -7.2 % |

- DFlash2 drafts a round in 0.78-0.79 ms against DSpark's 1.03-1.06 ms (device means over the arms' rounds), and
  accepts more of its drafts on short contexts (48-57 % against 44-49 % one request at a time).
- This table predates DFlash2's cheaper rounds (2026-11-27: a fused conv prepare, a feature-projection schedule, a
  faster top-16 and the commit context as a CUDA graph; [DFlash](maintainer/dflash.md#inkling-2-mini)), which cut
  its draft by 13 % and ran 1.6 % faster on the short prompts one at a time (0.5-3.6 % elsewhere, within the runs'
  spread), and its context-dependent draft lengths (0.7 % faster on the 122K-token prefix). DSpark was not re-run
  against them here; the agentic replay below was.
- On the 122K-token prefix DSpark is faster: it accepted 57 % of its drafts there against DFlash2's 49 %, likely
  because DSpark has a full-attention layer over the whole context while DFlash2's layers see the last 2,048 positions.
  (The same question resumed from the prefix cache and prefilled cold drafted and accepted exactly alike.)
- In an earlier run (DSpark before its INT8 context) eight short requests at once gave 666.8 tok/s without a drafter
  (one plain run), 674.5 with DSpark and 795.6 with DFlash2.
- The KV pool holds 548,992 tokens with DFlash2 against 502,528 with DSpark (+9.2 %; plain decoding 653,120): DSpark's
  full-layer KV shares the pool's pages, DFlash2's rings are 40 MiB a lane beside it. Its weights take 605 MiB. With
  the cheaper rounds it holds 547,456 (+8.9 %): the commit context's CUDA graphs take 24 MiB more of the graph
  allowance.
- `--lm-head-draft` took DFlash2's draft from 2.40 to 0.93 ms (an earlier run's phase mix, before DSpark's INT8
  context) and DFlash2 0.2-11 % faster in every phase but the 122K-token question (-2.0 %; one run without it).
- The companion's MLP in NVFP4 (recipe `inkling2_mini_dflash2_nvfp4_mlp`, 515 MB) drafts 3 % faster (0.761 ms) and
  gives the KV pool 7,040 more tokens (+1.3 %). One request at a time it ran 0.5-0.9 % faster than the Q8 companion
  on the short prompts (beyond the runs' small spread) and level on the long questions; at four and eight short
  requests at once -0.9 to +0.3 % (inside the spread); with four questions at once on the 122K-token prefix 4.1 %
  faster (306.3 against 294.2 tok/s, beyond the spread). Its acceptance was the Q8 companion's on the short prompts
  (2,364 against 2,364 accepted drafts, 2,500 against 2,499, 2,419 against 2,421) and 0.9 % lower on the long
  questions; at a fixed 7 drafts on the real test's four prompts it accepted 1.4-2.0 % fewer tokens a round, which is
  why the Q8 companion stays the recommended one.

### Agentic replay

The agentic replay (`bench/agentic_ab`: an agent's main-session and subagent turns, 130 requests a seed with prompts of
11K-134K tokens, median about 30K, 90 % of their tokens from the prefix cache), in the launcher configuration above,
three workload seeds, the DFlash2 arm first in seeds 42 and 44 and second in seed 43 (2026-11-27, with DFlash2's
cheaper rounds and context-dependent draft lengths, [DFlash](maintainer/dflash.md#inkling-2-mini)). Mean over the
seeds, their range in brackets:

| Metric | DSpark | DFlash2 | DFlash2 against DSpark, per seed |
|---|---|---|---|
| Workload wall time (min) | 6.0 (5.6-6.2) | 6.1 (5.7-6.4) | +2.4 % (+1.3 to +4.2) |
| Output tok/s, one request decoding | 281 (270-293) | 281 (271-293) | -0.0 % (-0.6 to +0.4) |
| Output tok/s, all decoding | 276 (271-280) | 266 (261-273) | -3.4 % (-5.5 to -2.2) |
| Decode rounds/s, one request decoding | 97.0 (93.1-102) | 99.4 (96.7-102) | +2.5 % (+0.5 to +3.8) |
| Tokens per round, one request decoding | 2.90 (2.75-3.15) | 2.83 (2.72-3.04) | -2.4 % (-3.6 to -1.1) |
| Average time to first token (s) | 0.7 (0.6-0.7) | 0.7 (0.6-0.7) | +1.3 % (+0.9 to +1.6) |

- Both arms produced the same number of tokens in every seed (as in the earlier run). DFlash2 was slower in each seed,
  by 1.3 and 1.6 % where it ran first and 4.2 % where it ran second (the seed it trailed most by in the earlier run
  too).
- The earlier run (2026-11-26, before the cheaper rounds, DFlash2 first in every seed) took 2.9 % more wall time with
  DFlash2 (+1.7 to +4.8 %), one request decoding 2.0 % slower. Since then DFlash2's walls fell 0.5-0.8 % in every seed
  and DSpark's stayed within 0.2 %. One request at a time DFlash2 now decodes level with DSpark (faster rounds, fewer
  tokens a round); with several requests decoding at once it is still 3.4 % slower.
- By prompt length (each request of 16 tokens or more against itself in the other arm): up to 32K -3.1 % decode (55 %
  of drafts accepted against DSpark's 61 %), 32K-64K -0.7 % (58 % against 60 %), 64K and beyond -3.5 % (61 % against
  66 %); the earlier run -2.3, -2.0 and -4.8 %. By how well DSpark drafts a request: where it accepts under 65 % of its
  drafts (290 requests) DFlash2 decodes level or faster (+3.6 and +0.6 %), where it accepts 65 % or more (100 requests,
  the most predictable text, often what the context already holds) 8.0-14.9 % slower. DSpark's full-attention layer
  sees the whole context; DFlash2's layers see the last 2,048 positions, as in the reference.
- So `--spec auto` keeps DSpark, and the DFlash2 companion is for short prompts, named with `--draft-model`. Its gain
  is measured up to about 600 tokens of context and its loss from 11K; nothing between was measured.

## Routes and precision

By default Infernix computes the model in the arithmetic it was trained with: the NVFP4 experts on
4-bit activations with the checkpoint's four-over-six rule, the MXFP8 experts on 8-bit activations,
and everything else in BF16 or FP32.

| Option | What it does | Output |
|---|---|---|
| (default) | the checkpoint's arithmetic | the reference for speed claims |
| `--reproducible` | the reproducible route. It refuses the two approximate options below and accepts `--a16-activations` and every KV format | deterministic and request-invariant |
| `--lossless` | every operation correctly rounded; a lane at a time. Text, `--kv-dtype bf16`, no drafter; refuses `--vision`, `--q8-weights` and `--a16-activations`. Much slower ([definition](maintainer/lossless-route.md)) | depends only on the request and the artifact |
| `--a16-activations` | the routed experts on 16-bit activations. Prompts run narrower kernels and read much more slowly | changes the output; the default is recommended |
| `--q8-weights` | **approximate**: the attention projections and output head become 8-bit weights at load, leaving about 770 MiB more VRAM for the KV cache. Their activations stay 16-bit and the experts are unchanged | changes the output |
| `--a8-activations` | **approximate**: includes `--q8-weights`, and runs the NVFP4 experts, attention projections and head on 8-bit activations. The MXFP8 experts keep their 8-bit arithmetic; the message gates and router stay 16-bit | changes the output |

Measured quality of the two approximate options (perplexity quick set, 291,842 tokens, BF16 KV; lower perplexity means
the model predicted the text better):

| Launch parameter | Perplexity | KL from the default | Same top token as the default |
|---|---:|---:|---:|
| default | 4.0314 | | |
| `--q8-weights` | 4.0342 | 0.0105 | 96.1 % |
| `--a8-activations` | 4.0261 | 0.0084 | 96.5 % |

`--a8-activations` scored better than the default on three of the four text sets and very slightly worse on the
fourth, so treat it as a small gain on this corpus rather than a guarantee.

`--reproducible` and `--lossless` refuse `--q8-weights` and `--a8-activations`, and
`--a8-activations` excludes `--a16-activations`. Published speeds for this model are default-route
speeds. Measured effects of the options are in the [CLI options](cli.md#common-options).

## KV formats

`--kv-dtype` selects how the eight full-attention visits store their keys and values, and with it the
40 sliding visits' windows: `bf16` and `bf16p` keep their windows in the same format, and every other
format stores its windows as `int8`. `int8` is recommended: its windows and pools take about half the
memory of `bf16`'s, so the same VRAM holds more context and more of the prefix cache. It decodes within
0.4 % of `bf16` and prefills up to 1.5 % slower ([performance](#performance)).

`--kv-window-dtype bf16|bf16p|int8` sets the windows' format apart from the pools' (`auto`, the default, is the
mapping above). The windows are a fixed share of each lane whatever its context: 170 MiB as `bf16`, 138 MiB as `bf16p`
(the same values, so `bf16`'s output) and 88 MiB as `int8`. On an RTX 5090 (2026-11-25), over `bf16p` pools, `bf16p`
windows decode 0.6 % slower than `bf16` windows with one request and 1.8 % slower with four, and prefill 0.3 % slower;
`int8` windows over `int8`, `fp8` and `nvfp4` pools decode 0.1 to 0.7 % faster than `bf16` windows with one request and
prefill up to 0.5 % slower. `--lossless` keeps its windows `bf16`.

| Format | Stored as | Output against `bf16` |
|---|---|---|
| `bf16` (default) | BF16 keys, FP16 values | the reference |
| `bf16p` | the same values packed losslessly to 12 bits | identical; a request with a row that has no packed form fails |
| `int8` (recommended) | 8-bit, two scales a row | changes |
| `fp8` | E4M3, one scale a row | changes |
| `nvfp4` | NVFP4 | changes |
| `k8v4` | `fp8` keys, `nvfp4` values | changes |
| `k4v2`, `vq2` | Hadamard-rotated 2-4 bit codes, no exact recent-key window | changes the most |

Measured quality per format: [design §7.6](maintainer/inkling2_mini-design.md#76-kv-profiles).

## Context, lanes and the prefix cache

- **One KV pool for every lane.** `--kv-capacity` sizes the pool (`auto`, the default with the
  prefix cache, takes the VRAM the model leaves free, read as the Qwen models read it: the
  `--vram-headroom-mib` comes off free memory, a 64 MiB margin off the Windows budget, and the
  CUDA graphs' allowance counts in the plan), and it always holds at least one `--max-context`.
  If a start still runs out of memory, an automatic pool starts once more with 1 GiB more headroom. A request takes pages as its prompt and output grow, so four lanes with short
  contexts all stay on the GPU. Pages no request holds keep cached prompt blocks.
- **Pauses keep the output.** When the lanes together outgrow the pool, the Engine pauses a
  request at a round's end, as it does for the Qwen models. The paused request's KV blocks and
  its state go to the prefix cache: Host RAM, and the SSD tier when there is one. It resumes
  from there once pages are free, or rebuilds what the cache no longer holds. The rebuild runs
  the prompt route over the prompt and the decode route over the reply, which is how those
  positions were first computed, so a paused request's output is bit for bit the same on every
  route. Without the prefix cache (`--no-prefix-reuse`, the CLI) a resume rebuilds the whole
  request. One request's context can never exceed `--max-context`, and the pool is allocated
  once at startup.
- **Prefill chunks** default to 4,096 tokens. `--prefill-chunk` takes multiples of 128 up to 8,064.
  Every chunk size gives the same bits.
- **The prefix cache** is on by default in `infernix-serve` (the `infernix` CLI runs without it). It
  caches only state computed from prompts, so a continuing turn would resume at the previous prompt's
  end and prefill the previous reply again; the server's idle [next-turn prefill](serving.md#next-turn-prefill)
  (`--cache-next-turn-ms`, 1 s by default, off on `--lossless`) prefills the reply as the next request will
  render it once nothing else runs, so the next turn resumes at the history's end. Requests arriving together that share a long prefix (one
  document, different questions) prefill it once: the others wait for the first's snapshot where they
  diverge (four questions on a 122K-token document, 7.5K tokens past its last snapshot: 1.4-2.0 s to
  the first token instead of 1.2-7.1 s), and a later prompt that diverges where an earlier one did
  resumes there (a second question 187 ms instead of 1.2 s; four after it 185-745 ms).
  `--kv-capacity` then defaults to `auto`. An explicit
  `--kv-capacity N` must hold `--max-context`, and caps `--max-context auto`. Queued and paused
  requests' resume points are evicted last (`--no-queue-holds` turns that off). `--prefix-cache-file` keeps
  it across restarts, and `--ssd-context-gib` with `--ssd-context-dir` adds an SSD tier. Details:
  [Inkling 2's prefix cache](serving.md#inkling-2s-prefix-cache).
- **`--embedding-host`** keeps the token embedding in pinned RAM and gives its VRAM to the KV and
  prefix caches, with the same output.

## Images

`--vision` enables image input, as local paths, URLs or base64 in the
[structured messages](cli.md#structured-messages) and the HTTP APIs. An image takes one token per
40-pixel cell. `--vision-max-merged N` (64 to 32,768, the default) caps the tokens of one image; a
larger image is downscaled to whole cells that fit. `--vision-offload on` keeps the tower in pinned
RAM and streams it through a small VRAM window for each image.

## Reasoning and tools

The chat template maps the effort names (`--reasoning-effort` on the CLI, the request's effort field
over HTTP) to a level; without one it uses 0.9.

| Effort | `none` | `minimal` | `low` | `medium` | `high` | `xhigh`, `max` |
|---|---:|---:|---:|---:|---:|---:|
| Level | 0 | 0.1 | 0.2 | 0.7 | 0.9 | 0.99 |

- `--no-thinking` selects level 0, the template's lowest. **It does not turn reasoning off**, as
  Thinking Machines' card also says.
- A thinking budget caps the thinking tokens of the whole turn: `--thinking-budget` (CLI), and in
  `infernix-serve` a request's own budget (Anthropic `budget_tokens`, the Chat Completions budget fields) or
  `--default-thinking-budget`. Inkling 2 may think in several blocks; they count together, and once the
  budget is used up (the control closes the block, or a block ends at the budget) the model cannot open
  another thinking block in that turn. The control is `--thinking-budget-message` (if set) and
  `<|end_message|>`.
- `infernix-serve` also keeps the answer room (as on every model): from an output limit of 1,024 tokens,
  thinking closes early enough to leave a quarter of the limit for the answer (`--no-thinking-answer-room`
  turns this off). Because the lowest effort still thinks, the budget, the answer room and
  `--end-in-thinking` apply to every request, including those sent with thinking off.
- The checkpoint's card warns that the model can get stuck in long reasoning loops. Infernix's
  `--reasoning-loop` guard is not available here, so set an output limit and a thinking budget for agent
  work.
- Tool calls work through the OpenAI and Anthropic APIs. `--preserve-thinking` and
  `--tolerant-tool-calls` are Qwen options and are refused.

## Constrained output and tool calls

Constrained decoding works as on the Qwen models ([output constraints](serving.md#openai-chat-completions),
[tool constraints](serving.md#tool-constraints), [design](maintainer/constrained-decoding.md#44-inkling-2-mini)): JSON object
and JSON Schema answers, GBNF, regex and choice (`--json-object`, `--json-schema-file`, `--grammar-file`, `--regex`,
`--choice` and the APIs' structured-output fields), and tool constraints: `tool_choice` required or a named tool, tools
declared with `strict: true`, `parallel_tool_calls: false`, and the protection of every call
(`tool_constraints:"basic"`, the default).

- The model still thinks first: any number of thinking blocks, then the answer in one text block, then the end of the
  turn. Only the answer follows the constraint, and thinking budgets work as without one. The grammar never lets the
  turn end inside a thinking block, whatever `--end-in-thinking` says.
- A tool call is the block the chat template writes, `<|message_model|>NAME<|content_invoke_tool_json|>` and
  `{"name": "NAME", "args": {...}}`: the name must be a callable tool's, and the arguments an object (a strict tool's
  must satisfy its schema, whose root must be an object). An `auto` choice lets the model think and answer in text before
  its calls; `required` and named choices allow thinking, then calls; `none` allows no call.
- A JSON answer and the call JSON take whitespace in runs of at most 32 characters. A JSON answer beside callable
  tools (`auto`) is either the answer or the calls.
- Calls arrive whole when they close, not streamed argument by argument as Qwen's constrained calls are. A call's
  arguments are its `args` object as the model wrote it (with or without a constraint).
- The model sometimes writes a call in an XML notation it was never shown, as a text block:
  `<invoke name="NAME"><parameter name="KEY">VALUE</parameter>...</invoke>`, bare or inside
  `<tool_call><function_calls>`, a parameter with or without `string="true"` (18 of 50 replies on Basalt #540's
  one-tool edit probe, temperature 0, here as on Basalt; 2026-11-24). Such a block is the calls it holds, on every route
  (Basic protection included, unlike Qwen, whose constrained content keeps these spellings as text), when all of these
  hold: it begins with `<tool_call>`, `<function_calls>` or `<invoke ` after whitespace; it holds nothing but closed
  invokes of the turn's callable tools (wrappers closed); a call block or the turn's end follows it. A strict tool's
  call stays text (its arguments were never held to its schema), as do more than one call under `parallel_tool_calls:
  false`, any XML in thinking, `tool_choice: none`, a block cut by the output limit or followed by the limit, and a
  block a stop string ends (matched as if never held). A value ends at the `</parameter>` that a parameter or
  `</invoke>` follows (one quoted inside the value is the value's); it is the model's text, entities undecoded;
  `string="true"` keeps it as a string; otherwise it is the JSON it holds when that is a type its property declares
  (any when it declares none) and a string otherwise. Such a block publishes nothing until it is decided; text then
  publishes exactly as unheld. `tool_call_parse.notation_calls` and `infernix_tool_call_notation_total` count these
  calls.
- A constrained request needs the model's own end of turn: custom stop strings and raw output are refused beside a
  constraint. `--tool-parameter-names declared` remains a Qwen option and is refused.
- DSpark keeps drafting under a constraint: each verified position samples under its own mask, so the output is the
  same as without the drafter. The masks are computed on the CPU while the GPU runs the forward pass: on an RTX 5090
  (DSpark, one request) a turn with a tool declared decoded within 0.5 % of the same turn free, with the same tokens
  (2026-11-24; several requests at once and `--spec off` not measured yet).

## Convert

To convert the checkpoint and verify the result:

```text
python -m tools.convert --model <Inkling-2-Mini checkpoint> --recipe inkling2_mini \
  --components text,dspark,vision --token-rank <rank.order.i32> --device cpu \
  --name Inkling-2-Mini-Infernix --out <dir>/Inkling-2-Mini-Infernix.infernix
python -m tools.convert.verify_inkling2 --model <Inkling-2-Mini checkpoint> \
  --artifact <dir>/Inkling-2-Mini-Infernix.infernix --token-rank <rank.order.i32>
```

The second command re-reads the checkpoint with its own reader, compares every stored word and fails
on any difference or on a checkpoint tensor the artifact did not consume.

### What the artifact stores

- **Experts:** the NVFP4 and MXFP8 codes and scales unchanged, regrouped into one record per expert.
- **Other weights:** the checkpoint's BF16 and FP32 values, some reordered without changing a value.
- **Padding dropped:** the embedding, output head and DSpark Markov embedding keep the 200,058
  assigned tokens' rows of the checkpoint's 201,024. No token id can reach the other 966.
- **Derived exactly as the reference does:** DSpark's `Fold` and `fold_bias` (from the head and
  `hidden_out`), the vision class row (the BF16 sum of its two tensors), and each NVFP4 expert's
  multiplier (the checkpoint's `amax` and denominator are stored beside it).
- **Added:** `proposal/token_rank`, the ranking below.
- **At load, not in the file:** the output head is packed losslessly, and DSpark's layers run in
  8 bits. The drafter's precision changes draft speed, never the output.

### Token ranking

The ranking orders the vocabulary by how often Inkling itself predicts each token: its top prediction
at every assistant position of a chat corpus, with the corpus's text as context. The counts are
weighted **English 50 %, Chinese 20 %, code 15 %, other languages 12 %, maths 3 %**; ties break by the
corpus's own token counts and then by id, and the tokenizer's special tokens come first. Only the
order is stored, so no dataset text is in the artifact.

The corpus draws on [WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M) (ODC-BY),
[OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct) and
[OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2) (CC-BY-4.0),
[Aya](https://huggingface.co/datasets/CohereLabs/aya_dataset) and
[OASST2](https://huggingface.co/datasets/OpenAssistant/oasst2) (Apache-2.0), and
[GSM8K](https://huggingface.co/datasets/openai/gsm8k) (MIT). The tools and steps to measure a ranking
are in [`tools/freq_corpus`](../tools/freq_corpus/README.md).

The published ranking counts 51.7 million assistant positions. On 1.35 million held-out positions:

| Shortlist | Weighted | English | Code | Chinese | Maths | Other languages |
|---:|---:|---:|---:|---:|---:|---:|
| 32,768 | 96.63 % | 97.07 % | 99.48 % | 98.75 % | 99.45 % | 87.03 % |
| 49,152 | 98.25 % | 98.55 % | 99.81 % | 99.44 % | 99.79 % | 92.64 % |
| **65,536 (used)** | **99.03 %** | 99.21 % | 99.93 % | 99.71 % | 99.92 % | 95.77 % |
| 98,304 | 99.71 % | 99.75 % | 99.98 % | 99.91 % | 99.97 % | 98.78 % |

## Performance

RTX 5090, PCIe Gen5 x8, default route, `--kv-dtype bf16`, one request unless noted.

| Measure | Result |
|---|---:|
| Decode, plain | 212 tok/s; 195 after a 10,000-token prompt |
| Decode, plain, on the chat / structured / prose prompts | 209 / 201 / 210 tok/s |
| Decode, DSpark `auto`, same prompts | 299 / 640 / 217 tok/s |
| Decode, DSpark `auto` with `--lm-head-draft`, same prompts | 316 / 667 / 228 tok/s |
| Prefill | 16,261 tok/s at 2,048 tokens, 16,184 at 10,000, 15,733 at 16,384 |
| Decode, several requests (total) | 312 / 452 / 648 tok/s at 2 / 4 / 8 requests |
| Largest `--max-context auto`, BF16 KV (`--plan`, about 30 GiB of VRAM for the model; 2026-11-25, with the CUDA graphs' allowance) | 300,160 tokens; 295,360 with `--lm-head-draft`; 241,216 with `--lm-head-draft --vision` |
| The same with `--kv-dtype int8` | 522,560 tokens; 514,304 with `--lm-head-draft`; 420,608 with `--lm-head-draft --vision` |

With `--kv-dtype int8` (2026-11-25, the same bench against `bf16` in one build, two runs each): plain decode 205.6
tok/s after a 2,048-token prompt and 194.5 after 10,000 (`bf16` 204.8 / 195.0); prefill 16,243 tok/s at 2,048 tokens
and 16,003 at 10,000 (`bf16` 16,344 / 16,243).

Measured with `infernix_bench` on the published artifact (2026-11-24). Plain decode, prefill and several requests are
means of three runs after a warmup. The DSpark rows are two runs each of a chat and a prose prompt (1,200 tokens,
temperature 0.6) and a greedy structured prompt, with 52 %, 94 % and 37 % of the shortlist's drafts accepted; other text
accepts more or fewer. Perplexity is under [Routes and precision](#routes-and-precision).

## License

Inkling 2 Mini is licensed under the
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) by Thinking Machines Lab, Inc.,
whose card also links its
[Model Acceptable Use Policy](https://thinkingmachines.ai/model-acceptable-use-policy/). The datasets
behind the token ranking are credited in the artifact's `NOTICE`.
