---
library_name: infernix
pipeline_tag: image-text-to-text
inference: false
license: other
license_name: swift-open-license-1.0
license_link: LICENSE
base_model:
  - ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4
base_model_relation: quantized
tags:
  - infernix
  - qwen3.8
  - qwen4_exp
  - swift
  - efficient-thinking
  - nvfp4
  - moe
  - mtp
  - blackwell
  - rtx-5090
  - multimodal
  - conversational
  - cuda
---

# Swift 1.5 Qwen3.8-Flash-Next NVFP4 Dense8 for Infernix

> **AI disclaimer:** this artifact, its conversion recipe, the engine that runs it and most of this
> card were made with AI (mostly Claude Opus 5.5). It is hobby work and is likely to be neither
> complete nor entirely accurate.

> [!IMPORTANT]
> **You also need a PLE n-gram volume, which is not in this repository.** Swift 1.5 did not change
> the n-gram table, so the volumes of the other conversions serve it. The recommended one is the
> 52 GB FP8 volume
> **[`Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram`](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix/blob/main/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram)**
> from [Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix)
> (the Dense8 repository has an identical copy). The 109 GB BF16 volume
> ([Qwen3.8-Flash-Next-FP8-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-FP8-Infernix))
> and the 29 GB IQ4_NL volume
> ([Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix))
> also run it. Put the volume on an NVMe drive and pass it with `--ngram-volume`:
>
> ```text
> hf download Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram --local-dir <nvme dir>
> ```
>
> If you already have a volume for another Flash-Next conversion of mine, use it. See
> [PLE n-gram tables](#ple-n-gram-tables); [`NGRAM-VOLUME.md`](NGRAM-VOLUME.md) in this repository
> repeats these links.

UkisAI's [Swift-1.5-Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4)
(revision `3ff05202`) — a reasoning-efficient fine-tune of Qwen3.8-Flash-Next with 48 layers, 512
routed experts per layer (top-10), an MTP drafter and vision — packed for
[Infernix](https://github.com/Wallawalla47/Infernix), a C++/CUDA engine that runs it on **one RTX
5090** (32 GB). The 24,576 routed experts (63 GiB) live in pinned host RAM, a VRAM cache holds the hot
ones, the CPU computes part of each layer's misses, and the 52 GB PLE n-gram table is read from an
NVMe drive.

**About Swift 1.5** (UkisAI's claims, not measured here): UkisAI fine-tuned Qwen3.8-Flash-Next to
think less on the same problems. On their evaluation of the BF16 models at `xhigh` reasoning effort,
Swift 1.5 used 55.8 % fewer mean (63.4 % fewer median) thinking tokens on GPQA-Diamond while scoring
within 0.2 points of the base model; across the other benchmarks on its card the scores range from
3.1 points lower (IFBench) to 2.0 points higher (LiveCodeBench v6, Terminal-Bench 2.1). At medium and
low effort it saved fewer tokens and lost about 2.5 points on GPQA-Diamond. See
[UkisAI's model card](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4#evaluation) for
the tables and methods.

UkisAI's checkpoint uses the same layout as
[NVIDIA's NVFP4 checkpoint](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) (ModelOpt W4A4
NVFP4 experts, BF16 everything else), so it is converted with the same recipe as my
[Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix):
the experts are kept bit-exact and the BF16 dense projections and `lm_head` are stored in 8 bits
("Dense8"). Every object of this artifact has the same format and shape as in that conversion.

> **Other Flash-Next conversions:**
> [Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix),
> the bit-exact [Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix),
> [Qwen3.8-Flash-Next-Uncensored-NVFP4-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-Uncensored-NVFP4-Infernix),
> [Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix)
> and the bit-exact [Qwen3.8-Flash-Next-FP8-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-FP8-Infernix).

## Files

| File | Bytes | What it is |
|---|---:|---|
| `Swift-1.5-Qwen3.8-Flash-Next-NVFP4-Dense8-Infernix-00001-of-00003.infernix` | 32,000,000,000 | model, part 1 of 3 (also holds the table of contents) |
| `Swift-1.5-Qwen3.8-Flash-Next-NVFP4-Dense8-Infernix-00002-of-00003.infernix` | 32,000,000,000 | model, part 2 of 3 |
| `Swift-1.5-Qwen3.8-Flash-Next-NVFP4-Dense8-Infernix-00003-of-00003.infernix` | 12,865,512,192 | model, part 3 of 3 |
| `Swift-1.5-Qwen3.8-Flash-Next-NVFP4-Dense8-Infernix.conversion.json` | 894,525 | conversion report: sources, methods and formats per object |
| `NGRAM-VOLUME.md` | | where to download an n-gram volume (none is in this repository) |
| `LICENSE`, `LICENSE-QWEN`, `NOTICE` | | the Swift Open License v1.0; the Qwen Community License 1.0 of the base model; UkisAI's attribution notice and my change notice |

Keep the three model parts in one directory and give Infernix part 1, as with a split GGUF.

## DFlash2 drafting

This model can draft with **[Qwen3.8-Flash-Next-DFlash2-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-DFlash2-Infernix)**
(776 MB), Inco AI's [DFlash2 drafter](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2) packed
for Infernix. It decodes 8-11 % faster than the model's own MTP drafter with `--draft-tokens auto`
(26 % on sampled code). Download it next to the model and pass it with `--spec dflash2 --draft-model`:

```text
hf download Wallawalla47/Qwen3.8-Flash-Next-DFlash2-Infernix Qwen3.8-Flash-Next-DFlash2-Infernix.infernix --local-dir <dir>
infernix-serve <this model's part 1>.infernix --spec dflash2 \
  --draft-model <dir>/Qwen3.8-Flash-Next-DFlash2-Infernix.infernix --draft-tokens auto
```

It needs an Infernix build whose `--help` lists `--draft-model`; see the drafter's card for details.

## PLE n-gram tables

Qwen3.8-Flash-Next's PLE n-gram table (320,001,536 rows) is not in the model files: it is a
separate volume, read in random 4 KiB blocks from an NVMe drive and passed with `--ngram-volume`.
Three volumes of the table exist, one per row format, none of them in this repository, and any of
them runs any Qwen3.8-Flash-Next model. **The FP8 volume is the recommended one.**

| Table | File | Bytes | Download from |
|---|---|---:|---|
| **FP8**, one scale for the table (NVIDIA's), **recommended** | `Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram` | 52,429,058,048 | [Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix/blob/main/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram), or the identical copy in [Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix/blob/main/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram) |
| BF16, unquantized (Qwen's) | `Qwen3.8-Flash-Next-BF16.ngram` | 109,227,196,416 | [Qwen3.8-Flash-Next-FP8-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-FP8-Infernix/blob/main/Qwen3.8-Flash-Next-BF16.ngram) |
| IQ4_NL, 4.5 bits per value (ISTA-DASLab's GSQ-RCO GGUF) | `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix.ngram` | 29,127,258,112 | [Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix/blob/main/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix.ngram) |

```text
hf download Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram --local-dir <nvme dir>
hf download Wallawalla47/Qwen3.8-Flash-Next-FP8-Infernix Qwen3.8-Flash-Next-BF16.ngram --local-dir <nvme dir>
hf download Wallawalla47/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix.ngram --local-dir <nvme dir>
```

One copy of a volume serves every Flash-Next model.

Swift 1.5 left Qwen's n-gram table unchanged (UkisAI's n-gram shards have the same SHA-256 as Qwen's
base checkpoint). Quantized to FP8 by the rule NVIDIA uses, it equals NVIDIA's table byte for byte:
all 320,001,536 rows were checked against the volume. So this model was converted with the FP8
table, and the tests below ran with it. UkisAI's table is Qwen's byte for byte, so with the BF16
volume the PLE uses this model's own BF16 values unrounded; the IQ4_NL table rounds them to 4.5 bits
per value. Neither has been measured on this model (on the GSQ-RCO IQ3_S conversion, BF16 lowered
perplexity from 3.5223 to 3.5198). On a 96 GB machine no table fits in RAM beside this model's
experts, so each is read from the drive. Older Infernix builds run this artifact with the FP8 and
BF16 volumes only.

## What is bit-exact and what is quantized

Converted with Infernix's recipe B, `qwen3_8_flash_next_nvfp4_dense8` (components text, vision and
MTP, plus a proposal head), with `--ngram-reuse` binding the artifact to the NVIDIA volume.
`python -m tools.flash_next.verify_artifact` compared the artifact with UkisAI's checkpoint tensor by
tensor: 24,576 experts (with every activation scale), 987 tensors and 320,001,536 n-gram rows with no
mismatch, and 629 text-model and 1,556 drafter parameters re-quantized.

**Bit-exact to UkisAI's checkpoint:**

- **Routed experts** (24,576): the NVFP4 words (E2M1 codes, E4M3 scale per 16, FP32 multiplier and
  FP32 activation input scale per matrix), repacked into Infernix's expert layout without changing a
  bit.
- **PLE n-gram table:** BF16 in the checkpoint; every row quantized to FP8 per tensor equals the
  volume's row byte for byte (above).
- **Router, shared-expert gate, GDN `a`/`b`, QSA query/gate/key/value/indexer projections, token
  embedding, norms, convolution weights and vision:** BF16 as stored; the GDN `A_log` and `dt_bias`
  widened from BF16 to FP32, value for value.

**Quantized from UkisAI's BF16** (`q8_g32_fp16`: 8-bit integers, one FP16 scale per 32 weights):

- **Dense projections and `lm_head`** (629 tensors): GDN q/k/v/z and output projections, QSA output
  projections, shared experts (gate, up, down), hyper-connection mixers (down, inject and up of every
  layer, and the final mixer), PLE key/value projections and `lm_head`, as in the NVIDIA Dense8
  conversion.

**Drafting only** (never changes output, since the full model verifies every draft):

- **MTP drafter** (1,556 parameters, BF16 in UkisAI's checkpoint): projections `q8_g32_fp16` (router
  and shared-expert gate BF16), routed experts `q4_g64_fp16` with MSE-chosen scales.
- **Proposal head** for `--lm-head-draft`: `q4_g64_fp16` copies of the `lm_head` rows of the 131,072
  most frequent tokens.

**Tokenizer.** UkisAI re-serialized `tokenizer.json` (vocabulary, merges and special tokens are
unchanged from Qwen's). The converter rewrites its pre-tokenizer split pattern to the standard Qwen
pattern, which UkisAI's own `tokenizer_config.json` also specifies, and builds the
`added_tokens_decoder` that Infernix's tokenizer requires.

Average bits per weight of the text decoder (176.9 billion weights; scales counted; vision and the
MTP drafter excluded), the same as the NVIDIA Dense8 conversion since every format matches:

| | Routed experts | Dense and other | PLE n-gram table | All | Without the n-gram table |
|---|---:|---:|---:|---:|---:|
| This artifact (+ n-gram volume) | 4.50 | 10.34 | 8.19 | 5.73 | 4.73 |

The n-gram volume's 8.19 includes its 4 KiB block padding; the table itself is 8.00.

## Requirements

- **GPU:** NVIDIA RTX 5090 (Blackwell, `sm_120a`), all of its 32 GB.
- **RAM:** about 69.3 GiB available for the experts, 73.3 GiB for `infernix-serve` with its prefix
  cache. With less, Infernix's SSD expert tier keeps what fits in RAM and reads the rest from the
  artifact (slower).
- **Disk:** ~77 GB for the artifact, plus the 52 GB n-gram volume on an NVMe drive.
- **Engine:** [Infernix](https://github.com/Wallawalla47/Infernix) (Windows or Linux).

## Run

```text
infernix-serve Swift-1.5-Qwen3.8-Flash-Next-NVFP4-Dense8-Infernix-00001-of-00003.infernix \
  --ngram-volume <nvme>/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram \
  --max-context 262144 --kv-dtype int8 \
  --spec mtp --draft-tokens auto --lm-head-draft --max-concurrency 2
```

It serves the OpenAI Chat Completions and Responses APIs and the Anthropic Messages API. Add
`--vision` for images and video. The
[Flash-Next guide](https://github.com/Wallawalla47/Infernix/blob/main/docs/qwen3_8-flash-next.md)
covers the KV formats, YaRN beyond 262K, images and video, the prefix cache, the SSD expert tier
and tuning.

## Tested

On 2026-10-11 on an RTX 5090 (Windows) with the Infernix `windows-2026.10.09` release, the command
above (32K context, with `--vision --vision-offload on`, on another port) started with this artifact
and passed Infernix's serving contract check (`tools.smoke.serve_contract`: streamed Chat
Completions, Responses with stored state and continuation, Anthropic Messages, and an image request
that expanded through the vision frontend). Greedy answers were correct: "Paris", "391" for 17 × 23,
a working iterative Fibonacci function, and "3" r's in "strawberry" with thinking on (84 output
tokens in all). MTP accepted 87.7-91.5 % of drafts on the longer replies.

UkisAI's own manifest lists native Blackwell validation of their NVFP4 checkpoint as not run (they
validated it on H100 with BF16 activations); this is, as far as I know, its first run with native FP4
activations.

These are smoke tests, not benchmarks: I have not measured this artifact's quality (perplexity) or
speed, and the speeds above were with a cold expert cache. Expect speed close to the NVIDIA Dense8
conversion, whose storage formats it shares, though speculative decoding depends on the text.

## Provenance

Converted with `tools.convert` (recipe `qwen3_8_flash_next_nvfp4_dense8`, `--ngram-reuse`) on CPU in
17 minutes and renamed into numbered parts with `tools.artifact.rename --numbered`; `artifact_id`
`826514ee14d94830b0726b0ef93d04a0`. The conversion report and the artifact's own provenance record the
paths of the machine that converted it.

## License

The Swift Contribution (UkisAI's fine-tuned weights) is licensed under the
**[Swift Open License v1.0](LICENSE)**: personal, research, educational, evaluation and commercial use
are free for individuals and organizations with gross annual revenue (including affiliates) below
US$1,000,000; above that threshold, commercial use needs a separate Swift Enterprise License from
[UkisAI](https://ukisai.com/contact). The underlying
[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) is subject to the
[Qwen Community License 1.0](LICENSE-QWEN), which among other conditions requires a separate license
from Qwen for some commercial uses. [`NOTICE`](NOTICE) carries UkisAI's attribution notice and a
description of the changes made in this conversion.

Copyright 2026 UkisAI. Swift Contribution licensed under the Swift Open License v1.0. Derivative of
Qwen3.8-Flash-Next, Copyright (c) 2026 Qwen, Qwen Community License 1.0.
