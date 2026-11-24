# Inkling 2 Mini (`Inkling2LoopForCausalLM`)

Infernix runs Thinking Machines'
[Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) on one RTX 5090 (32 GB),
with the whole model resident in VRAM. The published artifact,
[Inkling-2-Mini-Infernix](https://huggingface.co/wallawalla47/Inkling-2-Mini-Infernix), is one
`.infernix` file of about 20.3 GB that holds the text model, the model's DSpark drafter, its vision
tower and a token-frequency ranking for `--lm-head-draft`. The conversion is bit-exact: the
checkpoint's weights are unchanged ([what is stored](#what-the-artifact-stores)).

Generation works through `infernix` and `infernix-serve` (OpenAI Chat Completions and Responses,
Anthropic Messages) with DSpark drafting, images, one to eight lanes, every KV format and the prefix
cache. `infernix-perplexity` scores it, on the default route or with `--lossless`. It takes
[LoRA adapters](lora.md#inkling-2) and [ablation sidecars](ablate.md). Several features
of the Qwen models are [not available on Inkling yet](#not-supported-yet), among them constrained
output and video.

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
| Drafter | DSpark (3 layers, blocks of 8: up to 7 drafts a round) |
| Input | text and images (no video) |
| License | Apache 2.0, with Thinking Machines' [Model Acceptable Use Policy](https://thinkingmachines.ai/model-acceptable-use-policy/) linked from its card |

## Requirements

- **GPU.** Built and tuned on an RTX 5090 (`sm_120a`, 32 GB). Other cards have not been tested.
- **VRAM.** The weights take 17.8 GiB, 18.0 GiB with `--lm-head-draft` and 18.6 GiB with `--lm-head-draft --vision`.
  The execution workspace takes a further 1.8 GiB (3.1 GiB with `--vision`). What is left holds the KV cache, so the
  context you can run depends on the options: [`--plan`](#check-a-configuration-first) reports it for your machine.
- **RAM.** A peak working set of 1.9 GiB for a short `infernix` run at an 8,192-token context, including the load. The
  prefix cache's Host tier takes further RAM, down to `--ram-free-mib`.
- **Disk.** About 20.3 GB for the artifact.
- **Operating system.** Run and tested on Windows. The engine and launcher also build on Linux (GCC 13, CUDA 13.4), where
  this model has not been run.

## Download

```text
hf download wallawalla47/Inkling-2-Mini-Infernix Inkling-2-Mini-Infernix.infernix --local-dir <dir>
```

## Run

```bat
infernix-serve.exe Inkling-2-Mini-Infernix.infernix --max-context auto --spec dspark --draft-tokens auto --lm-head-draft
```

- `--max-context auto` takes the largest context the free VRAM holds for every lane. Without it
  `infernix-serve` uses 8,192 tokens and `infernix` 2,048.
- `--spec dspark --draft-tokens auto` is what the default `--spec auto` chooses for this artifact.
- `--lm-head-draft` is optional: it changes the speed of drafting, never the output.
- Add `--vision` for images. Its tower and image buffers stay in VRAM and reduce the context.
- Add `--max-concurrency N` (1 to 8) for several lanes. Each lane reserves a whole context, so
  `--max-context auto` gives each a share of the memory.

A single request from the command line (`--max-concurrency` and the cache options belong to
`infernix-serve`):

```bat
infernix.exe Inkling-2-Mini-Infernix.infernix --prompt "Explain what a looped transformer is." --max-context 32768 --max-new 2048 --spec dspark --draft-tokens auto --lm-head-draft
```

When a request sets no sampling options the engine uses temperature 1.0, top-p 1.0 and top-k 20: the
checkpoint's generation config sets the temperature, and 20 is the sampler's candidate limit.

### Check a configuration first

`--plan FILE` sizes a start on this machine without loading the model, writes the plan as JSON and
exits `0` (fits), `3` (does not fit) or `1` (no plan):

```bat
infernix-serve.exe Inkling-2-Mini-Infernix.infernix --max-context auto --max-concurrency 2 --spec dspark --draft-tokens auto --lm-head-draft --vision --plan plan.json
```

See [Sizing a start without loading](serving.md#sizing-a-start-without-loading).

## Not supported yet

Each of these fails with a clear error instead of running differently, except where the table says
otherwise. The engine refuses an option before the model loads. The options only `infernix-serve` or `infernix` reads
(the tool options, `--reasoning-loop`, `--expert-state FILE`, `--ram-resize`, `--kv-capacity` without the cache) are refused
once the model has loaded; `--plan` reports them without loading.

| Feature | On Inkling 2 Mini |
|---|---|
| Video input | the request is refused; images work |
| Constrained output: `--grammar-file`, `--json-object`, `--json-schema-file`, `--choice`, `--regex` and the APIs' structured-output fields | the request is refused |
| `tool_choice` set to required or to a named tool | the request is refused. `auto` and `none` work; tool calls are parsed from the model's free output, so no grammar protects their arguments |
| A tool declared with `strict: true` | the request is refused, since nothing would check its arguments against the schema. Tools with `strict: false` or no `strict` field work |
| A request ending in an assistant message (prefill) | the request is refused: only a new assistant turn is rendered |
| `--reasoning-loop stop` or `conclude` | refused: the model has no reasoning-loop guard |
| `--preserve-thinking`, `--tolerant-tool-calls`, `--tool-parameter-names declared` | refused: Qwen template and tool-call options. The template renders the history's reasoning as given |
| N-gram drafting: `--ngram-volume`, `--ngram-draft-tokens`, `--ngram-min-match`, `--ngram-tree-nodes`, `--ngram-archive-mib`, `--ngram-session-mib`, `--ngram-residency` | refused |
| The MTP, DFlash and DFlash2 backends | `--spec mtp`, `dflash` and `dflash2` are refused: the model's drafter is DSpark. So are `--draft-model`, `--spec-min-p`, `--draft-tree-nodes` and `--draft-tree-paths` (DSpark verifies a chain) |
| `--lm-head-draft` when no drafter runs | refused: an artifact without DSpark, `--spec off` or `--lossless` |
| Offloaded-expert options: `--expert-state`, `--expert-ram-mib`, `--expert-ram-layout`, `--expert-pin-mib`, `--expert-csf`, `--expert-stripe`, `--expert-tail-skip`, `--route-trace`, and `--ram-resize` | refused: the experts stay in VRAM and the Host tier keeps its startup size |
| Qwen kernel, fast-path and media options: `--use-original-int8-prefill-kernel`, `--use-original-nvfp4-prefill-kernel`, `--prefill-split-workspace-mib`, `--fast-path`, `--no-fast-path`, `--media-cache-mib`, `--media-live-mib`, `--media-preprocess-threads`, `--context-cost-presets`, `--no-queue-holds` | refused: Inkling has its own prompt route and image path, registers no fast path, and its prefix cache has no cost model or queue holds |
| Sizing options that a start would not use: `--vram-headroom-mib` and `--vram-past-budget` without `--max-context auto` or the cache's `--kv-capacity auto`; `--ram-free-mib` without the automatic Host tier; `--vision-max-merged` without `--vision` | refused |
| `--kv-capacity N` below every lane's context, with `--max-context auto`, or without the prefix cache (`--no-prefix-reuse`, the CLI) | refused: each lane reserves its whole context |
| Generation options on a `--causal-scoring` start (`--prefix-cache-file`, `--host-context-mib`, `--ssd-context-gib`, `--lm-head-draft`, the sizing options) | not refused: scoring runs without them, and the start lists them as unused in its log, as the other models do |
| `--logits-dtype bf16` | refused at startup: the logits are FP32 |
| `--rope-yarn-factor` other than 1 | refused at startup: the model has no RoPE and reaches its full context natively |
| `--expert-a4-activations` | refused at startup (it belongs to one Qwen3.8-Flash-Next artifact) |
| `--cache-next-turn-ms`, `--cache-tool-hold-secs` | refused unless `0`: not built for this model |
| `--device-snapshot-slots`, `--cache-host-state` other than `fp32`, `--host-context-mib 0` | refused: snapshots live in the Host tier, exactly. Use `--no-prefix-reuse` to turn the cache off |

## DSpark drafting

DSpark is the drafter Thinking Machines ships with the model. Each round it drafts up to seven tokens
and the full model verifies them, so greedy and sampled output are the same as plain decoding, token
for token.

- `--spec auto` (the default) drafts with DSpark when the artifact bundles it. `--spec off` decodes
  plainly.
- `--draft-tokens auto` uses the calibrated number of drafts each round, up to 7. With `--spec dspark`
  a fixed count from 1 to 7 also works.
- DSpark drafts only while **one request is decoding**. A round with several requests decodes
  plainly.
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

`--kv-dtype` selects how the eight full-attention visits store their keys and values. The 40 sliding
visits' windows stay BF16 keys and FP16 values in every format.

| Format | Stored as | Output against `bf16` |
|---|---|---|
| `bf16` (default) | BF16 keys, FP16 values | the reference |
| `bf16p` | the same values packed losslessly to 12 bits | identical; a request with a row that has no packed form fails |
| `int8` | 8-bit, two scales a row | changes |
| `fp8` | E4M3, one scale a row | changes |
| `nvfp4` | NVFP4 | changes |
| `k8v4` | `fp8` keys, `nvfp4` values | changes |
| `k4v2`, `vq2` | Hadamard-rotated 2-4 bit codes, no exact recent-key window | changes the most |

Measured quality per format: [design §7.6](maintainer/inkling2_mini-design.md#76-kv-profiles).

## Context, lanes and the prefix cache

- **No request is paused.** Every lane reserves its whole context when it starts, so Inkling does
  not preempt requests. The price is that the context is divided between the lanes up front.
- **Prefill chunks** default to 4,096 tokens. `--prefill-chunk` takes multiples of 128 up to 8,064.
  Every chunk size gives the same bits.
- **The prefix cache** is on by default in `infernix-serve` (the `infernix` CLI runs without it). It
  caches only state computed from prompts, so a continuing turn resumes at the previous prompt's end
  and prefills the previous reply again. `--kv-capacity` then defaults to `auto`, so with a fixed
  `--max-context` the cache also gets the VRAM beyond the lanes' contexts. An explicit `--kv-capacity N`
  must hold every lane's whole context and needs a fixed `--max-context`. `--prefix-cache-file` keeps
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
| Largest `--max-context auto`, one lane, BF16 KV (`--plan`, about 30 GiB of VRAM for the model) | 302,528 tokens; 297,728 with `--lm-head-draft`; 243,584 with `--lm-head-draft --vision` |

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
