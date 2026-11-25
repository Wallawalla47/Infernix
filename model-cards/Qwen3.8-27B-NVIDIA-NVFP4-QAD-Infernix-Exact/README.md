---
library_name: infernix
pipeline_tag: image-text-to-text
inference: false
license: other
license_name: nvidia-open-model-license
license_link: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/
base_model:
  - nvidia/Qwen3.8-27B-NVFP4-QAD
base_model_relation: quantized
tags:
  - infernix
  - qwen3.8
  - qwen3.8-27b
  - nvfp4
  - fp8
  - qad
  - mtp
  - dflash2
  - speculative-decoding
  - blackwell
  - rtx-5090
  - multimodal
  - conversational
  - cuda
---

# Qwen3.8-27B NVIDIA NVFP4-QAD for Infernix (Exact)

> **AI disclaimer:** this artifact, its conversion recipe, the engine that runs it and most of this
> card were made with AI (mostly Claude Opus 5.5). It is hobby work and is likely to be neither
> complete nor entirely accurate.

> [!NOTE]
> **Engine version:** this artifact needs [Infernix](https://github.com/Wallawalla47/Infernix) built from
> GitHub `main` at commit `3ad591aa` or later, or a Windows release after `windows-2026.11.18`. Older builds
> refuse to load it: they do not know its `nvfp4_mul` weights (NVIDIA's NVFP4 codes with NVIDIA's FP32 scale as a
> multiplier).

NVIDIA's [Qwen3.8-27B-NVFP4-QAD](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4-QAD) (revision `8158ac56`), a
quantization-aware distillation of Qwen3.8-27B, packed for [Infernix](https://github.com/Wallawalla47/Infernix), a
C++/CUDA engine that runs it on **one RTX 5090** (32 GB), with an MTP drafter, a DFlash2 drafter, an indexed
proposal head for `--lm-head-draft`, and vision.

**The conversion is bit-exact.** Every NVFP4 and FP8 projection keeps NVIDIA's codes, block scales and FP32
per-tensor weight and input scales exactly as the checkpoint stores them (no scale is rounded into another format),
and Infernix runs them in the arithmetic NVIDIA serves them with: 4-bit activations at the NVFP4 layers and static
8-bit activations at the FP8 layers in prompts, at NVIDIA's calibrated input scales, and 16-bit activations in
decode and verification (rounds of several requests give the MLP a two-term 4-bit residual, a fast path). The BF16 tensors (embedding, norms, vision, MTP) are the checkpoint's.

> **NInfer form:** [Qwen3.8-27B-NVIDIA-NVFP4-QAD-NInferV3](https://huggingface.co/Wallawalla47/Qwen3.8-27B-NVIDIA-NVFP4-QAD-NInferV3)
> is the same checkpoint in the form NInfer can run (which Infernix runs too). It rounds the per-tensor scales into
> NInfer's stored forms and stores the head in Q8 and the embedding in FP8. In Infernix this exact artifact scored
> a lower perplexity than both it and my earlier NVIDIA conversion, and decoded faster (below).

## How it differs from NVIDIA's NVFP4 release

NVIDIA's QAD checkpoint stores the linear-attention (Gated Delta-Net) projections of 38 of its 48 GDN layers in
NVFP4 (W4A4, groups of 16), where its NVFP4 release (and my
[Qwen3.8-27B-NVIDIA-NVFP4-NInferV3](https://huggingface.co/Wallawalla47/Qwen3.8-27B-NVIDIA-NVFP4-NInferV3), made
from an earlier revision of it) keeps every GDN projection in FP8. The MLP projections are NVFP4 and the 16
full-attention layers' q/k/v/o FP8 in both.

## Files

| File | Contents |
|---|---|
| `Qwen3.8-27B-NVIDIA-NVFP4-QAD-Infernix-Exact.infernix` | the artifact (22,205,138,944 bytes) |
| `Qwen3.8-27B-NVIDIA-NVFP4-QAD-Infernix-Exact.infernix.conversion.json` | the converter's per-tensor report |
| `config.json` | not read by Infernix; lets the Hub count downloads |
| `LICENSE`, `NOTICE` | licences (below) |

The artifact holds the `text` backbone (64 layers, hidden 5120, vocabulary 248,320, native context 262,144), the
`vision` tower, the `mtp` drafter, the `dflash2` drafter and an indexed 131,072-row `proposal` head.

## What is bit-exact and what is converted

| Part | Stored as |
|---|---|
| 205 NVFP4 projections (128 MLP, 76 GDN in 38 layers, the output head) | NVIDIA's packed FP4 codes and E4M3 block scales unchanged; `weight_scale_2` and `input_scale` kept as FP32 (`nvfp4_mul`: the scale a multiplier, not a rounded divisor) |
| 52 FP8 projections (the 16 attention layers' q/k/v/o, the GDN projections of layers 0, 1, 2, 4, 5, 57, 58, 60, 61, 62) | NVIDIA's E4M3 codes unchanged with the FP32 per-tensor weight scale and static input scale (`fp8_e4m3fn_row_f32`) |
| Embedding, norms, convolutions, GDN `a`/`b` projections, `dt_bias`/`a_log`, vision tower | the checkpoint's BF16/FP32 values |
| MTP drafter | the checkpoint's decoder layer; its input projection `mtp.fc` and three norms, which the QAD checkpoint omits, from nvidia/Qwen3.8-27B-NVFP4 (revision `482ca0f3`, whose MTP decoder layer is byte-identical to this one's) |
| DFlash2 drafter | copied byte for byte from my earlier NVIDIA conversion (its gate/up in grouped Q8) |
| Proposal head | 131,072 rows gathered from the output head, ranked by a frequency corpus, grouped Q4 with an INT32 token table |

An independent word check compared every imported projection's codes and scales with the checkpoint: no
differences.

### Activation precision

| Option | Prompts (prefill) | Decode and verification |
|---|---|---|
| default | NVFP4 layers 4-bit with NVIDIA's input scales, FP8 layers static 8-bit (W4A4 and W8A8, as NVIDIA serves it) | 16-bit |
| `--a16-activations` | 16-bit everywhere | 16-bit |
| `--a8-activations` | every NVFP4 and FP8 projection 8-bit (int8 codes per 32 values at the NVFP4 layers, one E4M3 term at the FP8 ones) | the same, at every width |

The two options are for precision comparisons: they read prompts about 4x and 5x slower than the default.

## Requirements

- An NVIDIA RTX 50-series GPU (built and tuned on an RTX 5090, 32 GB). The weights take 17.0 GiB of VRAM, 19.4 GiB
  with the DFlash2 drafter and the proposal head.
- Infernix at the version above.

## Run

```bat
infernix-serve.exe Qwen3.8-27B-NVIDIA-NVFP4-QAD-Infernix-Exact.infernix --max-context auto --max-concurrency 2 --spec dflash2 --draft-tokens auto --lm-head-draft --ngram-draft-tokens auto --kv-dtype int8 --preserve-thinking --tolerant-tool-calls
```

It runs DFlash2 drafting with `--draft-tokens auto` (each round verifies the width its probabilities favour), the
proposal head, n-gram copies and INT8 KV. Add `--vision` for images and video.

## Quality

Perplexity on Infernix's perplexity corpus (the quick run: 261,167 positions, INT8 KV, the default route), RTX 5090,
2026-11-19:

| Artifact | Perplexity |
|---|---:|
| **This artifact** | **4.3373** |
| This artifact, `--a16-activations` | 4.2869 |
| This artifact, `--a8-activations` | 4.2871 |
| My earlier NVIDIA conversion ([Qwen3.8-27B-NVIDIA-NVFP4-NInferV3](https://huggingface.co/Wallawalla47/Qwen3.8-27B-NVIDIA-NVFP4-NInferV3)) | 4.3472 |
| The NInfer form of this checkpoint ([Qwen3.8-27B-NVIDIA-NVFP4-QAD-NInferV3](https://huggingface.co/Wallawalla47/Qwen3.8-27B-NVIDIA-NVFP4-QAD-NInferV3)) | 4.3423 |

Against my earlier NVIDIA conversion the paired difference is -0.0023 nats per token (95 % interval -0.0044 to
-0.0003, a block bootstrap over the corpus windows); against the NInfer form it is -0.0012 (-0.0029 to +0.0002, not
significant). At 16-bit activations the QAD weights are the better ones by more (4.2869 against 4.3048), but they
lose more to NVIDIA's 4-bit activations. `--a8-activations` (int8 activations with a scale per 32 values) gives up
nothing measurable against 16-bit ones (+0.00004 nats per token, -0.0003 to +0.0004) and gains 0.0117 (0.0098 to
0.0135) on the default route. Lower perplexity on this corpus is not proof of better answers.

## Performance

`infernix_bench` on an RTX 5090 (PCIe Gen5 x8), INT8 KV, one request, a 2,048-token prompt and 256 decoded tokens,
2026-11-19, against my earlier NVIDIA conversion in the same runs:

| Measure | This artifact | Against the earlier conversion |
|---|---:|---:|
| Prefill, 4,096-token prompt | 17,110 tok/s | |
| Prefill, 16,384-token prompt | 15,486 tok/s | |
| Prefill, 2,048-token prompt | 18,800 tok/s | 1.06-1.12x |
| Decode, plain | 90.4 tok/s | 1.12x |
| Decode, MTP auto | | 1.08x |
| Decode, DFlash2 auto + proposal head | 570 tok/s | 1.04x (1.05x at two requests) |

The benchmark text is very predictable (DFlash2 accepts about 9 of a 15-draft block); treat the speculative speeds
as the upper end. Its 38 NVFP4 GDN layers read fewer bytes than the earlier conversion's FP8 ones, which is most of
the plain decode gain.

## Provenance

Converted with `tools.convert` (recipe `qwen3_8_27b_nvfp4_nvidia_exact`, `--components text,vision,mtp`,
`--proposal`, `--graft dflash2=<my earlier NVIDIA conversion>`, `--source mtp=<the four MTP tensors above>`,
`--resource chat_template.jinja=tools/chat_templates/qwen3_8.jinja`) from NVIDIA's revision
`8158ac567c235346e4b3629f77bd73d5852a4ded`. The `artifact_id` is `f3f94d1e57234ef2a27edc47dca7c59c`. The conversion
report and the artifact's own provenance record the paths of the machine that converted it.

## License

Use of this model is governed by the
[NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/),
as for the source checkpoint (see `NOTICE`); the Qwen3.8-27B base model is licensed under Apache 2.0 (`LICENSE`,
copied from the source checkpoint). The Infernix engine has its own license in its repository.
