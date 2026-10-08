---
library_name: infernix
pipeline_tag: image-text-to-text
inference: false
license: other
license_name: nvidia-open-model-license
license_link: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/
base_model:
  - nvidia/Qwen3.8-Flash-Next-NVFP4
base_model_relation: quantized
tags:
  - infernix
  - qwen3.8
  - qwen4_exp
  - nvfp4
  - moe
  - mtp
  - blackwell
  - rtx-5090
  - multimodal
  - conversational
  - cuda
---

# Qwen3.8-Flash-Next NVIDIA NVFP4 for Infernix (bit-exact)

> **AI disclaimer:** this artifact, its conversion recipe, the engine that runs it and most of this
> card were made with AI (mostly Claude Opus 5.5). It is hobby work and is likely to be neither
> complete nor entirely accurate.

NVIDIA's [Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) — 48
layers, 512 routed experts per layer (top-10), an MTP drafter and vision — packed for
[Infernix](https://github.com/Wallawalla47/ninfer-custom/tree/Infernix), a C++/CUDA engine that runs
it on **one RTX 5090** (32 GB). The 24,576 routed experts (63 GiB) live in pinned host RAM, a VRAM
cache holds the hot ones, the CPU computes part of each layer's misses, and the 52 GB PLE n-gram
table is read from an NVMe drive.

**The main model and the vision tower are bit-exact to NVIDIA's checkpoint**: every weight the
full model computes with is NVIDIA's, word for word. Only the MTP drafter, which proposes tokens
for the full model to verify, is quantized further.

> **Faster alternative:**
> [Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix)
> keeps the same NVFP4 experts and n-gram table bit-exact but stores the dense projections and
> `lm_head` in 8 bits. On the same machine and build it decodes about **22 % faster** (136-160
> against 108-131 tok/s from 8K to 250K tokens of context), reaches the first token up to 7 % sooner
> at 8K, and measured the same quality (perplexity 4.654 against this artifact's 4.666). Choose this
> artifact when the weights must be NVIDIA's unchanged; otherwise Dense8 is the faster choice. See
> [Performance](#performance).

Until 2026-10-08 this repository held the 8-bit dense conversion under this name; it is now the
Dense8 repository above.

## Files

| File | Bytes | What it is |
|---|---:|---|
| `Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix-00001-of-00003.infernix` | 32,000,000,000 | model, part 1 of 3 (also holds the table of contents) |
| `Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix-00002-of-00003.infernix` | 32,000,000,000 | model, part 2 of 3 |
| `Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix-00003-of-00003.infernix` | 16,370,610,944 | model, part 3 of 3 |
| `Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram` | 52,429,058,048 | PLE n-gram volume (read in random 4 KiB blocks); the same file serves both conversions |
| `Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.conversion.json` | 901,847 | conversion report: sources, methods and formats per object |
| `NVIDIA-Open-Model-License.txt`, `NOTICE`, `LICENSE` | | the NVIDIA Open Model License; NVIDIA's attribution notice; the Qwen Community License 1.0 |

Keep the three model parts in one directory and give Infernix part 1, as with a split GGUF. Put the
n-gram volume on an NVMe drive and pass it with `--ngram-volume`.

## What is bit-exact and what is quantized

Converted from [nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4)
with Infernix's recipe A, `qwen3_8_flash_next_nvfp4` (components text, vision and MTP, plus a
proposal head). `python -m tools.flash_next.verify_artifact` compared the artifact with the
checkpoint tensor by tensor: 24,576 experts, 1,616 tensors and 320,001,536 n-gram rows with no
mismatch, and only the drafter's 1,556 parameters re-quantized.

**Bit-exact to NVIDIA's checkpoint** (the whole main model and vision):

- **Routed experts** (24,576): NVIDIA's NVFP4 words (E2M1 codes, E4M3 scale per 16, FP32
  multiplier and FP32 activation input scale per matrix), repacked into Infernix's expert layout
  without changing a bit.
- **PLE n-gram table:** every FP8 row byte-identical in the n-gram volume, with its scale.
- **Every BF16 tensor, stored BF16:** the dense projections (GDN, QSA, shared experts,
  hyper-connection mixers, PLE key/value projections), `lm_head`, router, gates, token embedding,
  norms, convolution weights and the vision tower. The GDN `A_log` and `dt_bias` are widened from
  BF16 to FP32, value for value.
- The PLE hash tables are computed from the model's config, which conversion checks against the
  checkpoint's buffers.

**Drafting only** (never changes output, since the full model verifies every draft):

- **MTP drafter** (1,556 parameters): projections `q8_g32_fp16` (8-bit, one FP16 scale per 32
  weights; router and shared-expert gate BF16); its routed experts, block-scaled FP8 in NVIDIA's
  checkpoint, `q4_g64_fp16` with MSE-chosen scales from their exact values.
- **Proposal head** for `--lm-head-draft` (not in NVIDIA's checkpoint): `q4_g64_fp16` copies of
  the `lm_head` rows of the 131,072 most frequent tokens.

Bit-exact weights do not make the output bit-identical to another engine's: Infernix computes with
BF16 activations and its own kernels' FP32 summation order, as every engine does in its own way.

Average bits per weight of the text decoder (176.9 billion weights; scales counted; vision and the
MTP drafter excluded, as Unsloth's GGUF holds neither):

| | Routed experts | Dense and other | PLE n-gram table | All | Without the n-gram table |
|---|---:|---:|---:|---:|---:|
| This artifact (+ n-gram volume) | 4.50 | 16.00 | 8.19 | 5.89 | 4.95 |
| The Dense8 conversion (+ n-gram volume) | 4.50 | 10.34 | 8.19 | 5.73 | 4.73 |
| `nvidia/Qwen3.8-Flash-Next-NVFP4` | 4.50 | 16.00 | 8.00 | 5.83 | 4.95 |
| Unsloth `Qwen3.8-Flash-Next-UD-Q4_K_XL` | 5.10 | 8.90 | 4.50 | 5.03 | 5.25 |

The n-gram volume's 8.19 includes its 4 KiB block padding; the table itself is 8.00.

## Requirements

- **GPU:** NVIDIA RTX 5090 (Blackwell, `sm_120a`), all of its 32 GB. The BF16 dense weights take
  3.5 GB more VRAM than the Dense8 conversion's, which leaves about 1,270 fewer expert-cache frames
  (6,057 against 7,319 with 3.6 GB of INT8 KV allocated).
- **RAM:** about 69.3 GiB available for the experts, 73.3 GiB for `infernix-serve` with its
  prefix cache. With less, Infernix's SSD expert tier keeps what fits in RAM and reads the rest
  from the artifact (slower).
- **Disk:** ~80 GB for the artifact, plus the 52 GB n-gram volume on an NVMe drive. A fast
  consumer drive is enough; a SATA SSD or a hard disk is far slower.
- **Engine:** Infernix (Windows or Linux), branch `Infernix`.

## Run

```text
infernix-serve Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix-00001-of-00003.infernix \
  --ngram-volume <nvme>/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram \
  --max-context 262144 --kv-dtype int8 --prefill-chunk 4096 \
  --spec mtp --draft-tokens 4 --lm-head-draft --max-concurrency 2
```

It serves the OpenAI Chat Completions and Responses APIs and the Anthropic Messages API. The
[Flash-Next guide](https://github.com/Wallawalla47/ninfer-custom/blob/Infernix/docs/qwen3_8-flash-next.md)
covers the KV formats, YaRN beyond 262K, images and video, the prefix cache, the SSD expert tier
and tuning.

## Quality

Teacher-forced on the same token ids (2,557 positions of code, a document and a chat transcript;
INT8 KV in both engines), against Strata 0.1.40 running Unsloth's UD-Q4_K_XL GGUF, with the Dense8
conversion beside it (Infernix at the 2026-10-08 kernels):

| Text | Perplexity, this artifact | ΔNLL vs Strata | Perplexity, Dense8 | ΔNLL vs Strata | Perplexity, Strata |
|---|---:|---:|---:|---:|---:|
| Code | 1.917 | +0.009 ± 0.012 | 1.913 | +0.007 ± 0.013 | 1.899 |
| Document | 9.707 | −0.001 ± 0.018 | 9.666 | −0.006 ± 0.016 | 9.721 |
| Chat | 3.499 | **−0.096 ± 0.021** | 3.493 | **−0.098 ± 0.022** | 3.852 |
| All | **4.666** | **−0.037 ± 0.011** | 4.654 | **−0.040 ± 0.011** | 4.844 |

ΔNLL is the mean per-token difference in nats (negative: Infernix assigns the actual text higher
probability) with its standard error. The two conversions are level within that error.

## Performance

RTX 5090 in a PCIe Gen5 **x8** link (x16 is likely faster, most of all for this model, whose
expert-cache misses cross PCIe on every token), Core i9-13900K, 96 GB DDR5, Windows 11; Infernix
`Infernix` branch with the 2026-10-08 BF16 kernels, INT8 KV, MTP with 4 drafts and the proposal
head; one request at a time, greedy, 1,024 output tokens, means over two or three prompts of each
request's median of three sessions. Beside it, the Dense8 conversion on the same build:

| Context | TTFT, this artifact | TTFT, Dense8 | Decode tok/s, this artifact | Decode tok/s, Dense8 |
|---|---:|---:|---:|---:|
| ~8K | 2.07 s | 1.92 s | 116.1 | 140.1 |
| ~128K | 17.91 s | 17.96 s | 107.5 | 136.3 |
| ~250K | 34.01 s | 34.22 s | 130.7 | 159.9 |

Per request this artifact decoded 18 % slower than Dense8 (11-25 %) and reached the first token
2.5 % later (6-8 % at 8K, level from 128K): it reads ~8.6 GB of BF16 dense weights per token against
~5.1 GB and holds fewer experts in VRAM. Decode with speculation depends on how predictable the
text is, so single prompts vary by up to ±20 %. Against Strata 0.1.40 on the same machine (Unsloth
UD-Q4_K_XL with its MTP drafter; 12.3 s, 103 s and 158 s to the first token and 86, 77 and 96
tok/s at the same contexts, measured on 7 October), this artifact reaches the first token 4.6-5.9×
sooner and decodes 1.3-1.4× faster. Methods are in the
[Infernix README](https://github.com/Wallawalla47/ninfer-custom/blob/Infernix/README.md#benchmarks).

## Provenance

Converted with `tools.convert` (recipe `qwen3_8_flash_next_nvfp4`) and renamed into numbered parts
with `tools.artifact.rename --numbered`; `artifact_id` `86e6d1fd9f9048df810e766809552032`. The
conversion report and the artifact's own provenance record the paths of the machine that converted
it. The n-gram volume was written from the same FP8 table (byte-identical to NVIDIA's) and is the
one the Dense8 conversion uses.

## License

Use of this model is governed by the
[NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)
(copy in `NVIDIA-Open-Model-License.txt`; "Licensed by NVIDIA Corporation under the NVIDIA Open
Model License", see `NOTICE`). The underlying
[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) is also subject to the
[Qwen Community License 1.0](LICENSE), which among other conditions requires a separate license from
Qwen for some commercial uses.
