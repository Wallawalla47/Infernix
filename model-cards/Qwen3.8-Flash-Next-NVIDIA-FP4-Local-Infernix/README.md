---
library_name: infernix
pipeline_tag: image-text-to-text
inference: false
license: other
license_name: nvidia-open-model-agreement
license_link: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/
base_model:
  - nvidia/Qwen3.8-Flash-Next-FP4-Local
base_model_relation: quantized
tags:
  - infernix
  - qwen3.8
  - qwen4_exp
  - nvfp4
  - mxfp8
  - moe
  - mtp
  - blackwell
  - rtx-5090
  - multimodal
  - conversational
  - cuda
---

# Qwen3.8-Flash-Next NVIDIA FP4-Local for Infernix

> **AI disclaimer:** this artifact, its conversion recipe, the engine that runs it and most of this
> card were made with AI (mostly Claude Opus 5.5). It is hobby work and is likely to be neither
> complete nor entirely accurate.

NVIDIA's [Qwen3.8-Flash-Next-FP4-Local](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local),
a quantization-aware distillation of Qwen3.8-Flash-Next (48 layers, 512 routed experts per layer,
top 10, an MTP drafter and vision), packed for [Infernix](https://github.com/Wallawalla47/Infernix), a
C++/CUDA engine that runs it on **one RTX 5090** (32 GB). The 24,576 routed experts live in pinned
host RAM, a VRAM cache holds the hot ones, the CPU computes part of each layer's misses, and the
29 GB PLE n-gram table is read from an NVMe drive.

The conversion is bit-exact: NVIDIA's NVFP4 experts and block scales, MXFP8 projections and NVFP4
n-gram table are kept as the checkpoint stores them.

## Files

| File | Bytes | What it is |
|---|---:|---|
| `Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix-00001-of-00003.infernix` | 32,000,000,000 | model, part 1 of 3 (also holds the table of contents) |
| `Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix-00002-of-00003.infernix` | 32,000,000,000 | model, part 2 of 3 |
| `Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix-00003-of-00003.infernix` | 13,375,571,200 | model, part 3 of 3 |
| `Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix.ngram` | 29,127,258,112 | PLE n-gram volume (read in random 4 KiB blocks), NVIDIA's NVFP4 rows |
| `Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix-00001-of-00003.infernix.conversion.json` | 883,511 | conversion report: sources, methods and formats per object |
| `LICENSE`, `NOTICE` | | the Qwen Community License 1.0; attribution and the NVIDIA agreement |

Keep the three model parts in one directory and give Infernix part 1, as with a split GGUF. Put the
n-gram volume on an NVMe drive and pass it with `--ngram-volume`. Any of the other
[Qwen3.8-Flash-Next n-gram tables](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix#ple-n-gram-tables)
also runs this model.

This version (2026-10-24) needs an Infernix build from GitHub `main` at commit `257789da` or later, which
runs the vision MLPs in NVFP4; with an older build, update Infernix first.

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

## What is bit-exact and what is converted

Converted with Infernix's recipe `qwen3_8_flash_next_fp4_local` (components text, vision and MTP,
plus a proposal head and `--screen-heads`). `python -m tools.flash_next.verify_artifact` compared the
artifact with the checkpoint: 24,576 experts, 1,616 tensors and 320,001,536 n-gram rows with no mismatch
and no tensor rounded.

- **Routed experts:** NVIDIA's NVFP4 codes, block scales (the checkpoint stores them in a lossless
  compressed encoding, which the converter restores), FP32 multipliers and activation scales. By
  default they run with BF16 activations, as NVIDIA serves the checkpoint;
  `--expert-a4-activations` runs them with 4-bit activations at NVIDIA's calibrated scales, which
  changes the output. Only this artifact offers the choice.
- **GDN, attention (indexer included) and shared-expert projections:** NVIDIA's MXFP8 words, run
  with BF16 activations; the GDN a/b projections and the vision attention are their exact BF16
  values; the trained FP32 norms stay FP32.
- **Vision MLPs and merger:** NVIDIA's NVFP4 codes, block scales and FP32 multipliers, run with BF16
  activations and exact sums (the experts' default arithmetic).
- **N-gram table:** every NVFP4 row, byte for byte.
- **Converted:** only the MTP drafter (1,556 tensors, re-quantized as for the other imports).
- **Added:** the MTP drafter's proposal head (`--lm-head-draft`) and 2-bit copies of the LM head and
  the proposal head (273 MB, the artifact's last objects) for Infernix's experimental screened heads,
  which score every row from the copy and compute only the best ones exactly; nothing else reads them.

## Run

```text
infernix-serve Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix-00001-of-00003.infernix \
  --ngram-volume Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix.ngram --kv-dtype int8 \
  --max-context 262144 --max-concurrency 2 \
  --spec dflash2 --draft-model Qwen3.8-Flash-Next-DFlash2-Infernix.infernix --draft-tokens auto --lm-head-draft
```

The DFlash2 drafter is
[Qwen3.8-Flash-Next-DFlash2-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-DFlash2-Infernix);
`--lm-head-draft` decodes 3.0 % faster than drafting with this model's BF16 text head. `infernix`
takes the same arguments. See the
[Qwen3.8-Flash-Next guide](https://github.com/Wallawalla47/Infernix/blob/main/docs/qwen3_8-flash-next.md#nvidia-fp4-local-checkpoint)
for the options, KV formats and requirements.

## Quality and speed (RTX 5090, 2026-10-20)

On the 1M-token perplexity corpus (quick, 4K context, BF16 KV, exact route) the model scores 3.4865
with its default 16-bit expert activations and 3.5024 with `--expert-a4-activations`, against 3.4854
for the official FP8 conversion and 3.5114 for NVIDIA's NVFP4 Dense8.

| Measure (INT8 KV, one request) | 16-bit expert activations | `--expert-a4-activations` |
|---|---:|---:|
| Prefill, 8K prompt | 3,352 tok/s | |
| Prefill, 32K prompt | 11,299 tok/s | 11,985 tok/s |
| Decode, plain | 110 tok/s | 135 tok/s |
| Decode, MTP | 150-162 tok/s | 216 tok/s |

Decode with the DFlash2 drafter (`--draft-tokens auto --lm-head-draft`, 2026-10-25, one request, INT8
KV, 384-token continuations of 1,280-token slices of code and docs): 220 tok/s on code (greedy), 147
tok/s on docs (sampled at temperature 1). The 2026-10-24 version changes only the vision MLPs and adds
the head screens: its text path gives the previous version's logits bit for bit, and it scores the
same 3.4865.

## License

Use of this model is governed by the
[NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/);
the Qwen3.8-Flash-Next base model is governed by the Qwen Community License 1.0 (`LICENSE`). The
Infernix engine has its own license in its repository.
