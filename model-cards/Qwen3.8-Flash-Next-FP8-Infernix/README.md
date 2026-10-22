---
library_name: infernix
pipeline_tag: image-text-to-text
inference: false
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model:
  - Qwen/Qwen3.8-Flash-Next-FP8
base_model_relation: quantized
tags:
  - infernix
  - qwen3.8
  - qwen4_exp
  - fp8
  - moe
  - mtp
  - blackwell
  - rtx-5090
  - multimodal
  - conversational
  - cuda
---

# Qwen3.8-Flash-Next FP8 for Infernix (bit-exact)

> **AI disclaimer:** this artifact, its conversion recipe, the engine that runs it and most of this
> card were made with AI (mostly Claude Opus 5.5). It is hobby work and is likely to be neither
> complete nor entirely accurate.

Qwen's official [Qwen3.8-Flash-Next-FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8) — 48
layers, 512 routed experts per layer (top-10), an MTP drafter and vision — packed for
[Infernix](https://github.com/Wallawalla47/Infernix), a C++/CUDA engine that runs it on **one RTX
5090** (32 GB). The 24,576 routed FP8 experts (112.6 GiB) are more than a 96 GB machine can pin, so
Infernix's SSD expert tier keeps as many as fit in pinned host RAM and reads the rest from the
artifact; a VRAM cache holds the hot ones, and the CPU computes part of each layer's misses.

**The main model and the vision tower are bit-exact to Qwen's checkpoint**: every expert keeps its
E4M3FN codes and BF16 128×128 tile scales, every BF16 or FP32 tensor keeps its values, and with
this repository's n-gram volume the PLE n-gram table is Qwen's BF16 table, of which the
checkpoint's FP8 table is the quantization (see [PLE n-gram tables](#ple-n-gram-tables)). Infernix
computes the experts with BF16 activations in exact integer arithmetic, rounded once, so the result
has the same bits wherever an expert runs (VRAM, PCIe, CPU, SSD). Only the MTP drafter, which
proposes tokens for the full model to verify, is quantized further.

This is the most accurate Flash-Next artifact for Infernix, and the slowest: see
[Performance](#performance). The 4-bit conversions decode several times faster.

## Files

| File | Bytes | What it is |
|---|---:|---|
| `Qwen3.8-Flash-Next-FP8-Infernix-00001-of-00005.infernix` | 32,000,000,000 | model, part 1 of 5 (also holds the table of contents) |
| `Qwen3.8-Flash-Next-FP8-Infernix-00002-of-00005.infernix` | 32,000,000,000 | model, part 2 of 5 |
| `Qwen3.8-Flash-Next-FP8-Infernix-00003-of-00005.infernix` | 32,000,000,000 | model, part 3 of 5 |
| `Qwen3.8-Flash-Next-FP8-Infernix-00004-of-00005.infernix` | 32,000,000,000 | model, part 4 of 5 |
| `Qwen3.8-Flash-Next-FP8-Infernix-00005-of-00005.infernix` | 4,689,672,704 | model, part 5 of 5 |
| `Qwen3.8-Flash-Next-BF16.ngram` | 109,227,196,416 | PLE n-gram volume: Qwen's BF16 table (read in random 4 KiB blocks) |
| `Qwen3.8-Flash-Next-FP8-Infernix-00001-of-00005.infernix.conversion.json` | 799,119 | conversion report: sources, methods and formats per object |
| `config.json` | | metadata only, so the Hub counts downloads; Infernix does not read it |
| `LICENSE`, `NOTICE` | | the Qwen Community License 1.0; attribution and changes |

Keep the five model parts in one directory and give Infernix part 1, as with a split GGUF. Put the
n-gram volume on an NVMe drive and pass it with `--ngram-volume`.

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
separate volume, read in random 4 KiB blocks from an NVMe drive (or held in RAM when it fits) and
passed with `--ngram-volume`. Three volumes of the table exist, one per row format, and any of them
runs any Qwen3.8-Flash-Next model. **The FP8 volume is the recommended one.**

| Table | File | Bytes | Download from |
|---|---|---:|---|
| **FP8**, one scale for the table (NVIDIA's), **recommended** | `Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram` | 52,429,058,048 | [Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix/blob/main/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram), or the identical copy in [Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix/blob/main/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram) |
| BF16, unquantized (Qwen's) | `Qwen3.8-Flash-Next-BF16.ngram` | 109,227,196,416 | this repository |
| IQ4_NL, 4.5 bits per value (ISTA-DASLab's GSQ-RCO GGUF) | `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix.ngram` | 29,127,258,112 | [Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix/blob/main/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix.ngram) |

```text
hf download Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram --local-dir <nvme dir>
hf download Wallawalla47/Qwen3.8-Flash-Next-FP8-Infernix Qwen3.8-Flash-Next-BF16.ngram --local-dir <nvme dir>
hf download Wallawalla47/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix.ngram --local-dir <nvme dir>
```

One copy of a volume serves every Flash-Next model.

This artifact was converted with the BF16 table, the one in this repository: with it the PLE uses
Qwen's BF16 values unrounded, the whole main model is bit-exact to Qwen's checkpoint, and the
[Quality](#quality) and [Performance](#performance) figures below were measured with it. The FP8
and IQ4_NL tables round those values (FP8 with one scale for the whole table, IQ4_NL to 4.5 bits per
value); their effect on this model has not been measured. On a 96 GB machine no table fits in RAM
beside this model's experts, so each is read from the drive. Older Infernix builds run this
artifact with the BF16 volume only.

## What is bit-exact and what is quantized

Converted from Qwen's checkpoint (revision `236dfdf285828023ca3bcd3f37366c58a3469b13`) with
Infernix's recipe `qwen3_8_flash_next_fp8` (components text, vision and MTP, plus a proposal head).
Infernix's `verify_artifact` compared the artifact with the checkpoint and found no difference: all
24,576 experts, 1,615 tensors and all 320,001,536 n-gram rows (the drafter's 1,556 re-quantized
tensors are listed, not compared).

**Bit-exact to the checkpoint:**

- **Routed experts** (24,576): each expert's E4M3FN codes and BF16 128×128 tile scales, unchanged
  (`fp8_e4m3fn_b128`, a 4.69 MiB record per expert).
- **Every other tensor of the main model** (attention, GDN, hyper-connections, router, shared
  experts, norms, embeddings and `lm_head`): its BF16 or FP32 values.
- **PLE n-gram table:** Qwen's BF16 table. The converter checks that its rows quantize to the
  checkpoint's FP8 codes.
- **Vision tower:** the checkpoint's BF16 values.

**Drafting only** (never changes output, since the full model verifies every draft):

- **MTP drafter:** its projections in `q8_g32_fp16` (router and shared-expert gate BF16) and its
  routed experts in `q2_g64_fp16` with MSE-chosen scales (2.25 bits per weight). Drafting with
  4-bit experts instead gave identical outputs and the same speed within run-to-run spread, so
  the smaller drafter leaves 0.63 GB more VRAM to the expert cache.
- **Proposal head** for `--lm-head-draft`: `q4_g64_fp16` copies of the `lm_head` rows of the 131,072
  most frequent tokens.

## Requirements

- **GPU:** NVIDIA RTX 5090 (Blackwell, `sm_120a`), all of its 32 GB.
- **RAM:** as much as possible. Infernix pins what the operating system allows (on a 96 GB Windows
  machine, 73.9 GiB of the 112.6 GiB of experts) and reads the rest from the artifact; the n-gram
  table is read from its volume.
- **Disk:** ~133 GB for the artifact on a fast NVMe drive (the SSD tier reads from it), plus an
  n-gram volume on an NVMe drive: 52 GB for the recommended FP8 table, 109 GB for this repository's
  BF16 one.
- **Engine:** [Infernix](https://github.com/Wallawalla47/Infernix) (Windows or Linux).

## Run

```text
infernix-serve Qwen3.8-Flash-Next-FP8-Infernix-00001-of-00005.infernix \
  --ngram-volume <nvme>/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram \
  --max-context 262144 --kv-dtype int8 \
  --spec mtp --draft-tokens auto --lm-head-draft --max-concurrency 2
```

The command uses the recommended FP8 n-gram table; for the bit-exact BF16 table, with which the
figures below were measured, pass `--ngram-volume <nvme>/Qwen3.8-Flash-Next-BF16.ngram` instead.

N-gram copy drafts (`--ngram-draft-tokens 7`) speed up highly repetitive text (the benchmark row
below) but made no difference in a replayed agentic coding workload, so the command leaves them
off. Builds before 2026-10-18 could stop serving during a long prompt at the default prefill chunk
("no run of frames to lend"); on those, add `--prefill-chunk 4096`.

It serves the OpenAI Chat Completions and Responses APIs and the Anthropic Messages API. The
[Flash-Next guide](https://github.com/Wallawalla47/Infernix/blob/main/docs/qwen3_8-flash-next.md)
covers the KV formats, YaRN beyond 262K, images and video, the prefix cache, the SSD expert tier
and tuning. This artifact stores the checkpoint's own chat template; `--chat-template qwen3_8.jinja`
(the maintained Qwen3.8 template, in the release zip's `chat_templates`) keeps a System message
that arrives after the first turn in place.

## Quality

On the 1M-token perplexity corpus (261,167 tokens, 4,096-token context with a 2,048 stride, INT8
KV; the reference is NVIDIA's NVFP4 Dense8 conversion's top-32 distribution):

| Artifact | Perplexity | Mean KL from the reference | Top-1 agreement |
|---|---:|---:|---:|
| This artifact (BF16 n-gram table) | 3.4870 | 0.0496 | 92.1 % |
| NVIDIA NVFP4 Dense8 (the reference) | 3.5077 | — | — |
| [Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix) | 3.5223 | 0.0905 | 89.3 % |

## Performance

RTX 5090 on PCIe 5.0 x8, i9-13900K with 96 GB of DDR5-5800 (74.1 GiB of experts page-locked, 4-8
GiB more kept in RAM for the CPU, the rest read from a Samsung 990 PRO), INT8 KV, one request,
`infernix_bench` (2026-10-16/17):

| Workload | tok/s |
|---|---:|
| Decode, plain (tg512) | 43.8-44.4 |
| Decode after a 4K-token prompt, MTP 4 with the proposal head | 43.5-49.3 |
| The same with n-gram drafts (`--ngram-draft-tokens 7`) | 53.8-54.2 |
| Prompt, 4,096 tokens, `--prefill-chunk 4096` | 690 |

Decode is bound by the experts that miss the VRAM cache (40-50 % of routed experts; the CPU
computes about a third of them) and prefill by reading ~38 GiB of experts from the drive per
4K-token prompt. More RAM helps directly: every expert held in RAM is one less SSD read. For speed,
the 4-bit conversions are several times faster
([Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix):
146 tok/s plain, 260 with MTP on the same machine).

With the experts taking all the RAM they may, little stays free for other programs while this model
runs; `--ram-headroom-mib N` (default 2048) leaves more, at some decode speed.

## Provenance

Converted with `tools.convert` (recipe `qwen3_8_flash_next_fp8`, `--ngram-reuse` with the BF16
volume) and renamed into numbered parts with `tools.artifact.rename --numbered`; `artifact_id`
`fd83b9068589467cb776f0354139f63e`, the recipe as in Infernix `4ee28eb1`. The conversion report and the artifact's own
provenance record the paths of the machine that converted it. The BF16 n-gram volume was written
with `tools.flash_next.ngram_bf16` from the BF16 table of
[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) and checked against it.

## License

The model is subject to the [Qwen Community License 1.0](LICENSE), which among other conditions
requires a separate license from Qwen for some commercial uses; see `NOTICE` for the attribution and
the changes made in this conversion.
