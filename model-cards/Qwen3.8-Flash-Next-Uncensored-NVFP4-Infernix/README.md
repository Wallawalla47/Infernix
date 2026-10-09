---
library_name: infernix
pipeline_tag: image-text-to-text
inference: false
license: apache-2.0
base_model:
  - orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4
base_model_relation: quantized
tags:
  - infernix
  - qwen3.8
  - qwen4_exp
  - nvfp4
  - fp8
  - moe
  - mtp
  - abliterated
  - uncensored
  - blackwell
  - rtx-5090
  - multimodal
  - conversational
  - cuda
---

# Qwen3.8-Flash-Next Uncensored NVFP4 for Infernix

> **AI disclaimer:** this artifact, its conversion recipe, the engine that runs it and most of this
> card were made with AI (mostly Claude Opus 5.5). It is hobby work and is likely to be neither
> complete nor entirely accurate.

> **Uncensored model.** orcarouter removed the base model's refusal behaviour ("abliteration"). It
> will answer requests that Qwen's and NVIDIA's releases decline, and its output is not filtered.
> You are responsible for how you use it and for what it produces.

orcarouter's [Qwen3.8-Flash-Next-Uncensored-NVFP4](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4)
(revision `cddc6ec5`) — 48 layers, 512 routed experts per layer (top-10), an MTP drafter and vision
— packed for [Infernix](https://github.com/Wallawalla47/Infernix), a C++/CUDA engine that runs it on
**one RTX 5090** (32 GB). The 24,576 routed experts (63 GiB) live in pinned host RAM, a VRAM cache
holds the hot ones, the CPU computes part of each layer's misses, and the 52 GB PLE n-gram table is
read from an NVMe drive.

orcarouter exported the model **weight-only** (NVFP4 experts and FP8 dense projections without
activation scales, for runtimes that keep activations in BF16). Infernix runs it exactly that way:
every quantized weight is imported bit-exactly and multiplied with BF16 activations in an exact
integer arithmetic, identical on the GPU and the CPU. The BF16 tensors that NVIDIA's "Dense8"
conversion stores in 8 bits are stored in 8 bits here as well.

> **Other Flash-Next conversions:**
> [Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix)
> and the bit-exact
> [Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix),
> both from NVIDIA's (not abliterated) checkpoint.

## Files

| File | Bytes | What it is |
|---|---:|---|
| `Qwen3.8-Flash-Next-Uncensored-NVFP4-Infernix-00001-of-00003.infernix` | 32,000,000,000 | model, part 1 of 3 (also holds the table of contents) |
| `Qwen3.8-Flash-Next-Uncensored-NVFP4-Infernix-00002-of-00003.infernix` | 32,000,000,000 | model, part 2 of 3 |
| `Qwen3.8-Flash-Next-Uncensored-NVFP4-Infernix-00003-of-00003.infernix` | 12,302,074,624 | model, part 3 of 3 |
| `Qwen3.8-Flash-Next-Uncensored-NVFP4-Infernix.conversion.json` | 967,282 | conversion report: sources, methods and formats per object |

Keep the three model parts in one directory and give Infernix part 1, as with a split GGUF.

**PLE n-gram volume.** The abliteration left the n-gram table untouched: quantized to FP8 by the
rule NVIDIA uses, it equals NVIDIA's table byte for byte (all 320,001,536 rows checked). So this
model uses the same 52 GB volume as the NVIDIA conversions,
`Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram` from
[Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix).
Put it on an NVMe drive and pass it with `--ngram-volume`; one copy serves all three models.

## What is bit-exact and what is quantized

Converted with Infernix's recipe C, `qwen3_8_flash_next_nvfp4_orcarouter` (components text, vision and
MTP, plus a proposal head). `python -m tools.flash_next.verify_artifact` compared the artifact with
the checkpoint: 24,576 experts, 1,323 tensors and 320,001,536 n-gram rows with no mismatch; 293
text-model and 1,556 drafter parameters re-quantized.

**Bit-exact to orcarouter's checkpoint:**

- **Routed experts** (24,576): the NVFP4 words (E2M1 codes, E4M3 scale per 16, the per-matrix FP32
  global scale) repacked into Infernix's expert layout. The checkpoint stores the global scale as a
  divisor; Infernix stores its FP32 reciprocal, the import's only rounding.
- **FP8 dense projections** (per-row E4M3 with BF16 row scales): GDN query/key/value/z and output,
  attention query/gate, key, value and output, and the shared experts.
- **Router, shared-expert gate, GDN `a`/`b`, QSA indexer, token embedding, norms, convolution weights
  and vision:** BF16 as stored; the GDN `A_log` and `dt_bias` widened from BF16 to FP32, value for
  value.

**Quantized from orcarouter's BF16** (`q8_g32_fp16`: 8-bit integers, one FP16 scale per 32 weights):

- **Hyper-connection mixers, PLE key/value projections and `lm_head`** (the classes the Dense8
  conversion also stores in 8 bits).
- **PLE n-gram table:** BF16 in the checkpoint, FP8 per tensor here (above).

**Drafting only** (never changes output, since the full model verifies every draft):

- **MTP drafter** (1,556 parameters, BF16 in the checkpoint): projections `q8_g32_fp16`, routed
  experts `q4_g64_fp16` with MSE-chosen scales.
- **Proposal head** for `--lm-head-draft`: `q4_g64_fp16` copies of the `lm_head` rows of the 131,072
  most frequent tokens.

**Arithmetic.** The checkpoint's experts and dense projections carry no activation scales, so
Infernix does not quantize activations for them (W4A16 experts, W8A16 dense). Each expert input is
aligned to its largest exponent as 21-bit integers, exact for every element within 13 binades of
the largest, and multiplied by the codes in exact integer sums on every route: the GPU's narrow and
wide (tensor-core) kernels and the CPU's AVX kernels give the same bits, so outputs do not depend on
which device or kernel served an expert.

Average bits per weight of the text decoder (176.9 billion weights; scales counted; vision and the
MTP drafter excluded):

| | Routed experts | Dense and other | PLE n-gram table | All | Without the n-gram table |
|---|---:|---:|---:|---:|---:|
| This artifact (+ n-gram volume) | 4.50 | 9.42 | 8.19 | 5.71 | 4.69 |

The n-gram volume's 8.19 includes its 4 KiB block padding; the table itself is 8.00.

## Requirements

- **GPU:** NVIDIA RTX 5090 (Blackwell, `sm_120a`), all of its 32 GB.
- **RAM:** about 69 GiB available for the experts, more for `infernix-serve` with its prefix cache.
  With less, Infernix's SSD expert tier keeps what fits in RAM and reads the rest from the artifact
  (slower).
- **Disk:** ~76 GB for the artifact, plus the 52 GB n-gram volume on an NVMe drive.
- **Engine:** [Infernix](https://github.com/Wallawalla47/Infernix) at a version with recipe C
  (weight-only W4A16 experts); earlier releases refuse the artifact.

## Run

```text
infernix-serve Qwen3.8-Flash-Next-Uncensored-NVFP4-Infernix-00001-of-00003.infernix \
  --ngram-volume <nvme>/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram \
  --max-context 262144 --kv-dtype int8 --prefill-chunk 4096 \
  --spec mtp --draft-tokens 4 --lm-head-draft --max-concurrency 2
```

It serves the OpenAI Chat Completions and Responses APIs and the Anthropic Messages API. The
[Flash-Next guide](https://github.com/Wallawalla47/Infernix/blob/main/docs/qwen3_8-flash-next.md)
covers the KV formats, YaRN beyond 262K, images and video, the prefix cache, the SSD expert tier
and tuning.

## Quality

Teacher-forced on the same token ids (2,557 positions of code, a document and a chat transcript;
INT8 KV), against Strata 0.1.40 running Unsloth's UD-Q4_K_XL GGUF of the (not abliterated) base
model, with NVIDIA's Dense8 conversion on the same Infernix build beside it:

| Text | Perplexity, this artifact | ΔNLL vs Strata | Perplexity, NVIDIA Dense8 | Perplexity, Strata |
|---|---:|---:|---:|---:|
| Code | 1.918 | +0.010 ± 0.014 | 1.905 | 1.899 |
| Document | 9.823 | +0.010 ± 0.013 | 9.720 | 9.721 |
| Chat | 3.656 | **−0.052 ± 0.021** | 3.479 | 3.852 |
| All | **4.772** | −0.015 ± 0.010 | 4.653 | 4.844 |

ΔNLL is the mean per-token difference in nats (negative: Infernix assigns the actual text higher
probability) with its standard error. On these texts the uncensored model is level with Strata's
base-model quant (better on chat) and 2.5 % above NVIDIA's conversion in perplexity, a difference of
the models (abliteration changes weights), not of the engine.

## Performance

RTX 5090 in a PCIe Gen5 **x8** link, Core i9-13900K, 96 GB DDR5, Windows 11; INT8 KV; `infernix_bench`
(three repetitions after a warm-up, one request).

| Workload | This artifact | NVIDIA Dense8, same build |
|---|---:|---:|
| Prefill 4,096 tokens (chunk 4096) | 1,660 tok/s | 1,659 tok/s |
| Prefill 16,384 tokens (chunk 4096) | 6,176 tok/s | 6,480 tok/s |
| Decode 512 tokens, plain | 125.8 tok/s | 116.3 tok/s |
| Decode 512 tokens, MTP 4 drafts | 165.7 tok/s | 167.5 tok/s |

The Dense8 column and the decode rows come from a session earlier the same day (the decode path did
not change in between); the prefill rows of this artifact are from the final build. The decode rows
generate each model's own text, so they compare the models' outputs as much as the engine. Long warm prefill is a few percent slower than Dense8 because the exact BF16-activation
expert arithmetic costs more tensor-core work than NVIDIA's FP4-activation experts. Not yet measured
against Strata on this model.

## Provenance

Converted with `tools.convert` (recipe `qwen3_8_flash_next_nvfp4_orcarouter`, `--ngram-reuse`);
`artifact_id` `df806b6e2c7547d8bfda7859e824d39c`. The conversion report and the artifact's own
provenance record the paths of the machine that converted it.

## License

orcarouter publishes the source checkpoint under the
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). It derives from
[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), released under the Qwen Community
License 1.0, whose conditions (among them a separate license from Qwen for some commercial uses) may
apply to derivatives as well; check both before use.
