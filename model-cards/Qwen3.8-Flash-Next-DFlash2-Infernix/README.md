---
library_name: infernix
pipeline_tag: text-generation
inference: false
license: apache-2.0
base_model:
  - incoai/Qwen3.8-Flash-Next-DFlash2
base_model_relation: quantized
tags:
  - infernix
  - qwen3.8
  - qwen4_exp
  - dflash2
  - speculative-decoding
  - draft-model
  - blackwell
  - rtx-5090
  - cuda
---

# Qwen3.8-Flash-Next DFlash2 drafter for Infernix

> **AI disclaimer:** this artifact, its conversion recipe, the engine that runs it and most of this
> card were made with AI (mostly Claude Opus 5.5). It is hobby work and is likely to be neither
> complete nor entirely accurate.

Inco AI's [DFlash 2 drafter for Qwen3.8-Flash-Next](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2),
packed for [Infernix](https://github.com/Wallawalla47/Infernix), a C++/CUDA engine that runs
Qwen3.8-Flash-Next on **one RTX 5090**. It is not a model by itself: Infernix loads it beside a
Qwen3.8-Flash-Next artifact (`--spec dflash2 --draft-model`), and each decoding round it drafts a
block of up to seven tokens in one pass, which the full model then verifies. Greedy output is the
same as without it, and sampling keeps its distribution: only the speed changes.

One copy serves every Infernix Flash-Next artifact
([FP8](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-FP8-Infernix),
[NVIDIA NVFP4 Dense8](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix),
[NVIDIA NVFP4](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix),
[GSQ-RCO IQ3_S](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix),
[Uncensored NVFP4](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-Uncensored-NVFP4-Infernix)):
the drafter reads the main artifact's token embedding and its output or proposal head, so this
file holds only the drafter's own layers.

## Files

| File | Bytes | What it is |
|---|---:|---|
| `Qwen3.8-Flash-Next-DFlash2-Infernix.infernix` | 776,075,776 | the drafter (an Infernix companion artifact) |
| `Qwen3.8-Flash-Next-DFlash2-Infernix.infernix.conversion.json` | 32,510 | conversion report: sources, methods and formats per object |
| `config.json` | | metadata only, so the Hub counts downloads; Infernix does not read it |
| `LICENSE`, `NOTICE` | | the Apache License 2.0; attribution and changes |

## Get Infernix

Infernix is on GitHub: [Wallawalla47/Infernix](https://github.com/Wallawalla47/Infernix). Windows
release zips are on its [Releases](https://github.com/Wallawalla47/Infernix/releases) page; on
Linux (or to build the latest source on Windows) follow the build section of its README. DFlash2
for Qwen3.8-Flash-Next needs a build whose `infernix-serve --help` lists `--draft-model`; older
builds refuse this file.

You also need a Qwen3.8-Flash-Next artifact for Infernix and an n-gram volume: the model cards
linked above list them, and the
[Flash-Next guide](https://github.com/Wallawalla47/Infernix/blob/main/docs/qwen3_8-flash-next.md)
covers everything else (KV formats, the prefix cache, the SSD expert tier, images and video).

## Use

Download this file next to your main artifact:

```text
hf download Wallawalla47/Qwen3.8-Flash-Next-DFlash2-Infernix Qwen3.8-Flash-Next-DFlash2-Infernix.infernix --local-dir <dir>
```

Then select DFlash2 instead of the built-in MTP drafter:

```text
infernix-serve Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix-00001-of-00003.infernix \
  --ngram-volume <nvme>/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram \
  --max-context 262144 --kv-dtype int8 \
  --spec dflash2 --draft-model <dir>/Qwen3.8-Flash-Next-DFlash2-Infernix.infernix \
  --draft-tokens auto --lm-head-draft
```

- `--spec dflash2 --draft-model FILE` loads the drafter beside the main artifact (any Flash-Next
  artifact; Infernix checks that the drafter was made for this model).
- `--draft-tokens auto` (recommended): every round drafts the block of 7 the drafter was trained
  with (8 columns: the last accepted token and 7 drafts) and verifies only as many of them as the
  drafter's own probabilities, calibrated per request, make worthwhile. `--draft-tokens N` (1-7)
  verifies N every round, which was slower than MTP (see [Performance](#performance));
  `--spec-min-p P` then verifies the drafts only up to the first whose probability is below P.
- `--lm-head-draft` scores the drafts with the main artifact's small proposal head (the 131,072
  most frequent tokens, 4-bit) instead of its full output head; the figures below use it. Every
  Infernix Flash-Next artifact listed above has one. It decodes faster than the full head on every
  conversion measured (2026-10-25, one request): Dense8 1.5 %, FP4-Local 3.0 %, IQ3_S 1.5 %.
- `--ngram-draft-tokens 7` (copy proposals) works beside it, as with MTP.
- The same options work with the one-shot CLI (`infernix`) and `infernix_bench`.

DFlash2 replaces MTP: the main artifact's MTP drafter is not loaded (no `--spec mtp`), so the
expert cache gets that VRAM back, less the drafter's 776 MB and a 40 MiB context ring per request
lane. With `--max-concurrency 2` or more, rounds of two or more requests decode without drafts
(as MTP's do without the `adaptive-speculation` fast path); a request decoding alone speculates.

## Quantization

Converted from Inco AI's checkpoint (revision `a22a1744e85c17364dcc54a6a420af7135d4c7f6`) with
Infernix's recipe `qwen3_8_flash_next_dflash2`, stored as NInfer's official Qwen3.8-27B artifact
stores its DFlash2 drafter:

| Tensors | Format |
|---|---|
| Target-feature projection, attention query/key/value/output, MLP gate/up/down (46 projections; the context key/value projections share the query block's key/value words, as in the source) | `q8_g32_fp16`: 8-bit codes, one FP16 absmax scale per 32 weights |
| Dynamic-convolution kernel projections and base kernels, candidate selector projection and codebooks, every norm (45 tensors) | BF16, unchanged |

A check against the checkpoint found every BF16 tensor byte-exact and every 8-bit weight within
half a quantization step of its source value. The drafter only proposes tokens, so its precision
can change how many drafts are accepted, never what is generated.

## Performance

RTX 5090 on PCIe 5.0 x8, i9-13900K with 96 GB of RAM, INT8 KV, one request; 24 prompts of 1,280
tokens each continued for 384 tokens, from Infernix's own source code (code) and its Markdown
documentation (docs), greedy or sampled at temperature 1.0 (top-k 20, top-p 0.95); both drafters
with `--draft-tokens auto` and `--lm-head-draft` (2026-10-20). Decode tok/s, the mean of every run
where there were several (Dense8 code: four DFlash2 runs at 209-217, eight MTP runs at 186-203):

| Main artifact | Text | Plain | MTP (`--spec mtp`) | **DFlash2 (this drafter)** | DFlash2 vs MTP |
|---|---|---:|---:|---:|---:|
| NVIDIA NVFP4 Dense8 | code, greedy | 120 | 198 | **213** | +8 % |
| NVIDIA NVFP4 Dense8 | code, sampled | | 162 | **204** | +26 % |
| NVIDIA NVFP4 Dense8 | docs, greedy | 121 | 164 | **179** | +9 % |
| NVIDIA NVFP4 Dense8 | docs, sampled | 122 | 145 | **161** | +11 % |
| GSQ-RCO IQ3_S | code, greedy | 121 | 233 | **258** | +11 % |
| GSQ-RCO IQ3_S | docs, greedy | | 192 | **210** | +9 % |
| GSQ-RCO IQ3_S | docs, sampled | | 167 | **183** | +10 % |

With `--draft-tokens auto` each round drafts seven tokens and verifies only as many as pay for
their verification: on this offloaded model a verified column costs about a third of a plain round
in expert-cache misses, so verifying all seven every round (`--draft-tokens 7`) was slower than
MTP. The drafter also leaves more VRAM to the expert cache than the MTP drafter: 9,070 cached
experts against 8,740 on Dense8 (12,432 against 12,288 on IQ3_S, whose MTP experts are 2-bit).

## How it works

DFlash 2 is a block-diffusion drafter: from the target's hidden features (the mean of the four
hyper-connection streams of layers 5, 15, 25, 35 and 45), five sliding-window layers with dynamic
convolutions turn a block of one known token and seven mask tokens into candidates for the next
seven positions in one pass, and a candidate selector traces one path through the top 16 at each
position. See the [Inco AI blog post](https://inco.ai/blog/dflash2/) and Infernix's
[DFlash reference](https://github.com/Wallawalla47/Infernix/blob/main/docs/maintainer/dflash.md).

## License

Apache License 2.0, as the source checkpoint ([LICENSE](LICENSE)); see `NOTICE` for the
attribution and the changes made in this conversion. The main model it drafts for has its own
license (the Qwen Community License 1.0 for Qwen3.8-Flash-Next).
