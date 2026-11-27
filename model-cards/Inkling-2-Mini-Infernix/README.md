---
library_name: infernix
pipeline_tag: image-text-to-text
inference: false
license: apache-2.0
license_link: https://www.apache.org/licenses/LICENSE-2.0
base_model:
  - thinkingmachines/Inkling-2-Mini
base_model_relation: quantized
tags:
  - infernix
  - inkling
  - inkling-2-mini
  - inkling2_loop
  - nvfp4
  - mxfp8
  - moe
  - dspark
  - dflash2
  - speculative-decoding
  - blackwell
  - rtx-5090
  - multimodal
  - conversational
  - cuda
---

# Inkling 2 Mini for Infernix

> **AI disclaimer:** this artifact, its conversion recipe, the engine that runs it and most of this
> card were made with AI (mostly Claude Opus 5.5). It is hobby work and is likely to be neither
> complete nor entirely accurate.

> [!NOTE]
> **Engine version:** this artifact needs [Infernix](https://github.com/Wallawalla47/Infernix) built from
> GitHub `main` at commit `a487ad24f` or later, or a Windows release
> after `windows-2026.11.20`. Older builds do not support this model. The recommended `--kv-dtype int8`
> stores the sliding windows in 8 bits too from `main` at `c0e69d556`. The optional DFlash2 drafter in
> [`drafter`](drafter) needs `main` at `6dbeb011d` or later.

Thinking Machines' [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini), packed for
[Infernix](https://github.com/Wallawalla47/Infernix), a C++/CUDA engine that runs it on **one RTX 5090**
(32 GB) with the whole model in VRAM. One file holds the text model, the model's DSpark drafter, its vision
tower and a token-frequency ranking for `--lm-head-draft`. The [`drafter`](drafter) folder adds incoai's DFlash2
drafter as an optional companion, faster than DSpark on short prompts ([DFlash2 drafter](#dflash2-drafter)).

**The conversion is bit-exact.** Every weight the model can use is stored unchanged ([What is stored](#what-is-stored)
lists the derived tables and the padding rows left out), and by default Infernix computes them in the arithmetic the
model was trained with: NVFP4 experts on 4-bit activations by the checkpoint's
four-over-six rule, MXFP8 experts on 8-bit activations, everything else in BF16 or FP32.

## Files

| File | Bytes | What it is |
|---|---:|---|
| `Inkling-2-Mini-Infernix.infernix` | 20,287,047,912 | the artifact |
| `Inkling-2-Mini-Infernix.infernix.conversion.json` | 425,200 | the converter's per-tensor report |
| `config.json` | | not read by Infernix; lets the Hub count downloads |
| `LICENSE`, `NOTICE` | | the Apache License 2.0; the changes made and the credits for the ranking's datasets |
| `drafter/Inkling-2-Mini-DFlash2-NVFP4-MLP-Infernix.infernix` | 515,118,080 | optional: incoai's DFlash2 drafter for Inkling 2 Mini, an Infernix companion artifact ([its README](drafter/README.md)) |
| `drafter/Inkling-2-Mini-DFlash2-NVFP4-MLP-Infernix.infernix.conversion.json`, `drafter/README.md`, `drafter/LICENSE`, `drafter/NOTICE` | | its conversion report, description, Apache-2.0 licence and notice |

```text
hf download wallawalla47/Inkling-2-Mini-Infernix Inkling-2-Mini-Infernix.infernix --local-dir <dir>
```

## What is stored

| Part | Stored as |
|---|---|
| Routed experts, layers 0-19 | the checkpoint's NVFP4 codes and scales unchanged, one record per expert, with its `amax` and denominator |
| Routed experts, layers 20-23 | the checkpoint's MXFP8 codes and scales unchanged, one record per expert |
| Attention, gates, router, norms, convolutions | the checkpoint's BF16 and FP32 values (some reordered, no value changed) |
| Token embedding, output head, DSpark Markov embedding | the rows of the 200,058 assigned tokens. The checkpoint's other 966 rows are padding no token id can reach |
| DSpark drafter | the checkpoint's BF16 tensors. `Fold` and `fold_bias` are derived from the head and `hidden_out`, as the reference does |
| Vision tower | the checkpoint's BF16 tensors; the class row is the BF16 sum of its two tensors, as the reference computes it |
| Tokenizer, chat template, generation and processor configs | unchanged |
| `proposal/token_rank` | **added**: the ranking below |

`python -m tools.convert.verify_inkling2` re-read the checkpoint with its own reader and compared every stored
word with it, and it passed on this file: 1,168 tensors, 24 expert banks, the 3 derived tensors and the ranking.

## Token ranking and `--lm-head-draft`

With `--lm-head-draft`, DSpark drafts from a shortlist of the 65,536 most frequent tokens, built at load from the
stored ranking. The full model verifies every token, so **the output is unchanged**. The option is optional.

I measured the ranking with Inkling itself: its top prediction at each assistant position of a chat corpus, with
the corpus's text as context, weighted **English 50 %, Chinese 20 %, code 15 %, other languages 12 %, maths 3 %**.
The tokenizer's special tokens come first. The artifact stores only the order of the token ids, and no dataset
text. The tool is [`tools/freq_corpus`](https://github.com/Wallawalla47/Infernix/tree/main/tools/freq_corpus).

The ranking counts 51.7 million assistant positions. On 1.35 million held-out positions, the 65,536-token shortlist
holds 99.0 % under that weighting: English 99.2 %, code 99.9 %, Chinese 99.7 %, maths 99.9 % and other languages 95.8 %.

The corpus comes from these datasets, with thanks to their authors:

| Dataset | License |
|---|---|
| [WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M) (Allen Institute for AI) | [ODC-BY 1.0](https://opendatacommons.org/licenses/by/1-0/) |
| [OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct) (NVIDIA) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| [OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2) (NVIDIA) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| [Aya Dataset](https://huggingface.co/datasets/CohereLabs/aya_dataset) (Cohere Labs) | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| [OASST2](https://huggingface.co/datasets/OpenAssistant/oasst2) (OpenAssistant) | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| [GSM8K](https://huggingface.co/datasets/openai/gsm8k) (OpenAI) | [MIT](https://opensource.org/license/mit) |

## Requirements

- **GPU:** built and tuned on an NVIDIA RTX 5090 (Blackwell, `sm_120a`, 32 GB). Other cards have not been tested.
- **VRAM:** the weights take 17.8 GiB, 18.0 GiB with `--lm-head-draft` and 18.6 GiB with `--lm-head-draft --vision`.
  The execution workspace takes a further 1.8 GiB (3.1 GiB with `--vision`). What is left holds the KV cache: on my
  card, one lane with the recommended `--kv-dtype int8` reaches 526,656 tokens of context, 518,400 with
  `--lm-head-draft` and 424,704 with `--lm-head-draft --vision` (with the default BF16 KV: 302,528, 297,728 and
  243,584). `infernix-serve --plan plan.json` reports what fits on your machine.
- **RAM:** a peak working set of 1.9 GiB for a short `infernix` run at an 8,192-token context, including the load. The
  server's prefix cache uses further RAM for its Host tier, down to `--ram-free-mib`.
- **Disk:** about 20.3 GB.
- **Engine:** [Infernix](https://github.com/Wallawalla47/Infernix) at the version above, run through its launcher,
  `infernix-serve` or `infernix`. I run it on Windows; the engine also builds on Linux, where I have not run this model.

## Run

```bat
infernix-serve.exe Inkling-2-Mini-Infernix.infernix --max-context auto --kv-dtype int8 --spec dspark --draft-tokens auto --lm-head-draft
```

It serves the OpenAI Chat Completions and Responses APIs and the Anthropic Messages API. `--max-context auto` takes
the largest context the free VRAM holds (without it the server uses 8,192 tokens). `--kv-dtype int8`
stores the KV cache, the 40 sliding-window visits included, in 8 bits with two scales a 128-value row: about half the
memory of the default `bf16`, so the same VRAM holds about 1.7 times the context and more of the prefix cache. It
decodes within 0.4 % of `bf16` and prefills up to 1.5 % slower, and changes the output slightly
([KV formats](#kv-formats)). Add `--vision` for
images and `--max-concurrency N` (1 to 8) for several lanes. From `main` at `8dfb51c69` the lanes share one KV pool:
each request takes pages as it grows, so any one of them can reach the whole pool, and when the lanes together outgrow
it one request pauses and resumes later with the same output (earlier builds reserve a whole context for each lane).
From `main` at `84c82ac45`, requests decoding together draft with DSpark together, one verify call checking the
round's drafts (earlier builds draft while one request decodes, and rounds of several decode plainly). The
[Inkling 2 Mini guide](https://github.com/Wallawalla47/Infernix/blob/main/docs/inkling2-mini.md) covers the KV formats,
the prefix cache, images, reasoning effort and conversion.

From `main` at `8ee3149fa`, `--lora PATH` applies a Hugging Face PEFT adapter at runtime (its attention projections,
convolution taps and loop message gates, rank up to 128) and `--ablate FILE` applies a directional-ablation sidecar, on
every route but `--ablate` with `--lossless`. The launcher offers both. See
[LoRA adapters](https://github.com/Wallawalla47/Infernix/blob/main/docs/lora.md#inkling-2) and
[ablation sidecars](https://github.com/Wallawalla47/Infernix/blob/main/docs/ablate.md), which give their measured speed
costs. No adapter or sidecar for this model has been published yet, so the adapter module names Infernix accepts follow
the Transformers implementation and have not been checked against a real adapter's files.

From `main` at `f9c2e42cb`, constrained output works on this model: JSON object and JSON Schema answers, GBNF, regex and
choice (`--json-schema-file`, `--grammar-file`, `--regex`, `--choice` and the APIs' structured-output fields), and tool
constraints: `tool_choice` required or a named tool, `strict: true` tools, `parallel_tool_calls: false`, and every call
checked against its declared tool by default. The model still thinks first; the answer, or the call, follows the
constraint. DSpark keeps drafting under a constraint, and on one request a text turn with a tool declared decoded
within 0.5 % of the same turn free. `/v1/decide` answers its readout questions (`noul`, `choice`, `score`). See
[constrained output and tool calls](https://github.com/Wallawalla47/Infernix/blob/main/docs/inkling2-mini.md#constrained-output-and-tool-calls).

## DFlash2 drafter

The [`drafter`](drafter) folder holds incoai's [DFlash2 drafter for Inkling 2 Mini](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2)
(Apache-2.0), converted as a 515 MB companion artifact with its MLP in NVFP4. It replaces the artifact's DSpark drafter,
which is then not loaded, and needs Infernix `main` at `6dbeb011d` or later:

```bat
hf download wallawalla47/Inkling-2-Mini-Infernix drafter/Inkling-2-Mini-DFlash2-NVFP4-MLP-Infernix.infernix --local-dir .
infernix-serve.exe Inkling-2-Mini-Infernix.infernix --max-context auto --kv-dtype int8 --spec dflash2 --draft-model drafter\Inkling-2-Mini-DFlash2-NVFP4-MLP-Infernix.infernix --draft-tokens auto --lm-head-draft
```

Use it for **short prompts**: there DFlash2 decoded 15-27 % faster than DSpark one request at a time and 16-19 % faster
at four and eight requests (measured with an all-Q8 conversion, which this one matches within 2 %). On a 122K-token
prefix it was 7 % slower, and over an agentic replay (prompts of 11K-134K
tokens) it took 2.4 % more wall time, so the default `--spec auto` keeps DSpark unless `--draft-model` names the
drafter. Without DSpark's KV pages the pool holds about 10 % more tokens. Output is unchanged either way: every draft
is verified. Measurements, the formats and how it works are in [its README](drafter/README.md).

## KV formats

`--kv-dtype` selects how the 8 full-attention visits store their keys and values, and with it the 40 sliding
visits' windows: `bf16` and `bf16p` keep their windows in their own format, and every other format stores its
windows as `int8`.

| Format | Stored as | Output against `bf16` |
|---|---|---|
| `bf16` (default) | BF16 keys, FP16 values | the reference |
| `bf16p` | the same values packed losslessly to 12 bits | identical; a request with a row that has no packed form fails |
| `int8` (recommended) | 8 bits, two scales a row | changes slightly |
| `fp8`, `nvfp4`, `k8v4` | E4M3 / NVFP4 / both, with INT8 windows | changes slightly |
| `k4v2`, `vq2` | Hadamard-rotated 2-4 bit codes, with INT8 windows | changes the most |

On Infernix's perplexity quick set, `int8` pools read a KL of 0.0094 from `bf16` (with BF16 windows), and its INT8
windows add 0.0013 of KL against BF16 windows over the same pools (perplexity 4.0317 → 4.0322). Measured quality of
every format: the [design](https://github.com/Wallawalla47/Infernix/blob/main/docs/maintainer/inkling2_mini-design.md#76-kv-profiles).

## Limits

Not supported on this model yet:

- video input (images work);
- `/v1/decide`'s generated types (`number`, `scalar`, `point`, `box`): they spell a number one digit at a time, and this
  model's tokenizer writes digits in groups of up to three, so they are refused;
- a request ending in an assistant message (prefill);
- the `--reasoning-loop` guard, n-gram drafting, and `--logits-dtype bf16`.

`--no-thinking` selects the template's lowest reasoning effort; it does not turn reasoning off. Thinking Machines'
card warns that the model can get stuck in long reasoning loops, so set an output limit for agent work.

## Activations, weights and precision

By default Infernix runs the model as its checkpoint was designed: the routed experts of layers 0-19 are NVFP4 on
4-bit activations (the checkpoint's four-over-six rule), those of layers 20-23 are MXFP8 on 8-bit activations, and
everything else is BF16 or FP32. Two optional launch parameters change that arithmetic:

- **`--a8-activations`** runs the NVFP4 experts on 8-bit activations instead of 4-bit, and the attention projections
  and output head on 8-bit weights and activations. The MXFP8 experts keep their 8-bit arithmetic, and the message
  gates and router stay 16-bit. It includes `--q8-weights`.
- **`--q8-weights`** converts the attention projections and output head from BF16 to 8-bit weights at load, which
  leaves about 770 MiB more VRAM for the KV cache. Their activations stay 16-bit and the experts are unchanged.

| Launch parameter | Perplexity | KL from the default | Same top token as the default |
|---|---:|---:|---:|
| default | 4.0314 | | |
| `--q8-weights` | 4.0342 | 0.0105 | 96.1 % |
| `--a8-activations` | 4.0261 | 0.0084 | 96.5 % |

Measured on Infernix's perplexity quick set (291,842 tokens, BF16 KV); lower perplexity means the model predicted the
text better. `--a8-activations` scored better than the default on three of the four text sets and very slightly worse
on the fourth, so treat it as a small gain on this corpus rather than a guarantee. Both options change the output.

| Other route | What it does | Output |
|---|---|---|
| `--reproducible` | the reproducible route; refuses the two options above | deterministic and request-invariant |
| `--lossless` | every operation correctly rounded; text, `--kv-dtype bf16`, no drafter; much slower | depends only on the request and the artifact |
| `--a16-activations` | the experts on 16-bit activations; prompts read much more slowly | changes |

## Performance

RTX 5090 on PCIe Gen5 x8, default route, BF16 KV, one request unless noted.

| Measure | Result |
|---|---:|
| Decode, plain | 212 tok/s; 195 after a 10,000-token prompt |
| Decode, plain, on the chat / structured / prose prompts | 209 / 201 / 210 tok/s |
| Decode, DSpark `auto`, same prompts | 299 / 640 / 217 tok/s |
| Decode, DSpark `auto` with `--lm-head-draft`, same prompts | 316 / 667 / 228 tok/s |
| Prefill | 16,261 tok/s at 2,048 tokens, 16,184 at 10,000, 15,733 at 16,384 |
| Decode, several requests (total) | 312 / 452 / 648 tok/s at 2 / 4 / 8 requests |

Measured with `infernix_bench` on this artifact (2026-11-24). Plain decode, prefill and several requests are means of
three runs after a warmup. The DSpark rows are two runs each of a chat and a prose prompt (1,200 tokens, temperature
0.6) and a greedy structured prompt, with 52 %, 94 % and 37 % of the shortlist's drafts accepted; other text accepts
more or fewer.

With `--kv-dtype int8` (2026-11-25, `main` at `c0e69d556`, against `bf16` in the same build, two runs each): plain
decode 205.6 tok/s after a 2,048-token prompt and 194.5 after 10,000 (`bf16` 204.8 / 195.0); prefill 16,243 tok/s at
2,048 tokens and 16,003 at 10,000 (`bf16` 16,344 / 16,243). Compare each pair with each other rather than with the
table, which comes from another day's runs.

## Provenance

Converted with `tools.convert` (recipe `inkling2_mini`, `--components text,dspark,vision`, `--token-rank`) from
Thinking Machines' revision `6e7c8248f75265bdd4fbab5273304243b5d8c2cb`. The `artifact_id` is
`9031a532d9ff45bcab717dce99abcea8`. The conversion report and the artifact's own provenance record the
paths of the machine that converted it.

## License

Inkling 2 Mini is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) by Thinking
Machines Lab, Inc.; `NOTICE` lists the changes made here and credits the ranking's datasets. Thinking Machines' card
also links its [Model Acceptable Use Policy](https://thinkingmachines.ai/model-acceptable-use-policy/). The drafter
in `drafter` is incoai's, also under the Apache License 2.0 (`drafter/LICENSE`); `drafter/NOTICE` lists the changes
made in its conversion. The Infernix engine has its own license in its repository.
