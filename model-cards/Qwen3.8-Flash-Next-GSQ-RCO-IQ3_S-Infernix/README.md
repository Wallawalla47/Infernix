---
library_name: infernix
pipeline_tag: image-text-to-text
inference: false
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model:
  - ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF
base_model_relation: quantized
tags:
  - infernix
  - qwen3.8
  - qwen4_exp
  - gguf
  - gsq
  - rco
  - iq3_s
  - moe
  - mtp
  - blackwell
  - rtx-5090
  - multimodal
  - conversational
  - cuda
---

# Qwen3.8-Flash-Next GSQ-RCO IQ3_S for Infernix

> **AI disclaimer:** this artifact, its conversion recipe, the engine that runs it and most of this
> card were made with AI (mostly Claude Opus 5.5). It is hobby work and is likely to be neither
> complete nor entirely accurate.

ISTA-DASLab's [Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
`IQ3_S` — Qwen3.8-Flash-Next's 48 layers and 512 routed experts per layer (top-10), quantized by
GSQ with RCO's per-tensor bit allocation — packed for [Infernix](https://github.com/Wallawalla47/Infernix),
a C++/CUDA engine that runs it on **one RTX 5090** (32 GB), with an MTP drafter and vision. The
24,576 routed experts (46.9 GiB) live in pinned host RAM, a VRAM cache holds the hot ones, the CPU
computes part of each layer's misses, and the PLE n-gram table is read from an NVMe drive (or held
in RAM when it fits, as this repository's 29 GB IQ4_NL table does on a 96 GB machine).

Every GGUF tensor is kept bit for bit, and Infernix computes them with BF16 activations in exact
integer arithmetic: each output is the exact sum of the GGUF weights times the activations, rounded
once, so the result has the same bits wherever an expert runs (VRAM, PCIe, CPU) and at any batch
width. (llama.cpp rounds the activations to 8 bits instead, so its logits differ slightly.)

## Files

| File | Bytes | What it is |
|---|---:|---|
| `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix-00001-of-00002.infernix` | 32,000,000,000 | model, part 1 of 2 (also holds the table of contents) |
| `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix-00002-of-00002.infernix` | 24,755,219,968 | model, part 2 of 2 |
| `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix.ngram` | 29,127,258,112 | PLE n-gram volume: the GGUF's `IQ4_NL` table (read in random 4 KiB blocks) |
| `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix-00001-of-00002.infernix.conversion.json` | 923,312 | conversion report: sources, methods and formats per object |
| `config.json` | | metadata only, so the Hub counts downloads; Infernix does not read it |
| `LICENSE`, `LICENSE-APACHE-2.0`, `NOTICE` | | the Qwen Community License 1.0; the Apache License 2.0 of ISTA-DASLab's GGUF; attribution and changes |

Keep the two model parts in one directory and give Infernix part 1, as with a split GGUF. Put the
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
| BF16, unquantized (Qwen's) | `Qwen3.8-Flash-Next-BF16.ngram` | 109,227,196,416 | [Qwen3.8-Flash-Next-FP8-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-FP8-Infernix/blob/main/Qwen3.8-Flash-Next-BF16.ngram) |
| IQ4_NL, 4.5 bits per value (ISTA-DASLab's GSQ-RCO GGUF) | `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix.ngram` | 29,127,258,112 | this repository |

```text
hf download Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram --local-dir <nvme dir>
hf download Wallawalla47/Qwen3.8-Flash-Next-FP8-Infernix Qwen3.8-Flash-Next-BF16.ngram --local-dir <nvme dir>
hf download Wallawalla47/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix.ngram --local-dir <nvme dir>
```

One copy of a volume serves every Flash-Next model.

This artifact was converted with the IQ4_NL table, the one in this repository (the GGUF's own rows,
byte for byte), and the [Quality](#quality) and [Performance](#performance) figures below were
measured with it held in RAM. With the BF16 table perplexity measured slightly better (3.5198
against 3.5223); the FP8 table has not been measured on this model. On a 96 GB machine only the
IQ4_NL table (27.1 GiB) fits in RAM beside the experts; the FP8 (48.8 GiB) and BF16 (101.7 GiB)
tables are read from the drive. Older Infernix builds run this artifact with the IQ4_NL and BF16
volumes only.

## What is bit-exact and what is quantized

Converted from ISTA-DASLab's GGUF (revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09`, the two
`IQ3_S` shards and the BF16 `mmproj`) with Infernix's recipe `qwen3_8_flash_next_gsq_rco_iq3s`
(components text, vision and MTP, plus a proposal head). Infernix's `verify_gguf_artifact` compared
the artifact with the GGUF and found no difference: all 24,576 experts, the 385 quantized dense
tensors (the tiled ones after untiling), 1,218 BF16 and FP32 tensors and all 320,001,536 n-gram
rows.

**Bit-exact to the GGUF:**

- **Routed experts** (24,576): the imatrix mix of `IQ2_S`, `IQ3_XXS`, `IQ3_S` and `IQ4_XS`
  gate/up and `IQ4_NL` or `Q2_0` down matrices, every block byte for byte, one record per expert.
- **Dense projections and `lm_head`**: `Q4_K`, `Q5_K`, `Q6_K`, `IQ4_XS` and `IQ4_NL` blocks byte for
  byte. Those of `Q4_K`, `Q5_K`, `Q6_K` and `IQ4_XS` are stored in tiles of 16 rows (the same blocks
  reordered so a tile's block is one contiguous run for the decode kernels); untiling returns the
  GGUF rows exactly. The one `Q8_0` tensor is stored as the identical `q8_g32_fp16` words.
- **PLE n-gram table:** the GGUF's `IQ4_NL` rows, byte for byte, in the n-gram volume.
- **BF16 and FP32 tensors** (router, norms, convolutions, GDN `a`/`b` and the rest): their values;
  llama.cpp's export transforms (norms stored as 1 + g, the GDN value-head order, `ssm_a` =
  −exp(A_log), the concatenated indexer projections) are undone exactly and checked.
- **Vision tower:** the GGUF `mmproj`'s BF16 values, which equal Qwen's own tower bit for bit.

**Drafting only** (never changes output, since the full model verifies every draft):

- **MTP drafter** (not in the GGUF): from
  [Qwen/Qwen3.8-Flash-Next-FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8) (revision
  `236dfdf285828023ca3bcd3f37366c58a3469b13`), its projections in `q8_g32_fp16` (router and
  shared-expert gate BF16) and its routed experts in `q2_g64_fp16` with MSE-chosen scales (2.25 bits
  per weight, as Strata keeps them).
- **Proposal head** for `--lm-head-draft`: `q4_g64_fp16` copies of the `lm_head` rows of the 131,072
  most frequent tokens.

## Requirements

- **GPU:** NVIDIA RTX 5090 (Blackwell, `sm_120a`), all of its 32 GB.
- **RAM:** 46.9 GiB for the experts, and to hold the n-gram table in RAM (the default when it
  fits; otherwise its rows are read from the volume as needed) 27.1 GiB more for the IQ4_NL table or
  48.8 GiB for the FP8 one. With less RAM, Infernix's SSD expert tier keeps what fits and reads the
  rest from the artifact (slower).
- **Disk:** ~57 GB for the artifact, plus an n-gram volume on an NVMe drive: 52 GB for the
  recommended FP8 table, 29 GB for this repository's IQ4_NL one.
- **Engine:** [Infernix](https://github.com/Wallawalla47/Infernix) (Windows or Linux).

## Run

```text
infernix-serve Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix-00001-of-00002.infernix \
  --ngram-volume <nvme>/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram \
  --max-context 262144 --kv-dtype int8 \
  --spec mtp --draft-tokens auto --lm-head-draft --max-concurrency 2
```

The command uses the recommended FP8 n-gram table; for this repository's IQ4_NL table, with which
the figures below were measured, pass
`--ngram-volume <nvme>/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix.ngram` instead.

N-gram copy drafts (`--ngram-draft-tokens 7`) speed up highly repetitive text (the benchmark row
below) but were level within 1 % in a replayed agentic coding workload, so the command leaves them
off.

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
| This artifact (its `IQ4_NL` n-gram table) | 3.5223 | 0.0905 | 89.3 % |
| This artifact with the BF16 n-gram table | 3.5198 | 0.0893 | — |
| [Qwen3.8-Flash-Next-FP8-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-FP8-Infernix) (BF16 table) | 3.4870 | 0.0496 | 92.1 % |
| NVIDIA NVFP4 Dense8 (the reference) | 3.5077 | — | — |

Teacher-forced on three frozen texts (2,557 positions of code, a document and a chat transcript;
INT8 KV), its perplexity is 4.794 against 4.653 for the NVFP4 Dense8 conversion (+0.030 ± 0.011
nats per token) and 4.864 for Strata running Unsloth's UD-Q4_K_XL (−0.015 ± 0.012, level within the
error).

## Performance

RTX 5090 on PCIe 5.0 x8, i9-13900K with 96 GB of DDR5-5800 (every expert and the n-gram table in
RAM), INT8 KV, one request, `infernix_bench`, against NVIDIA's NVFP4 Dense8 conversion measured in
the same session (2026-10-16):

| Workload | This artifact | NVFP4 Dense8 |
|---|---:|---:|
| Decode, plain (tg512) | 145.6 tok/s | 110.3 tok/s |
| Decode, MTP 4 with the proposal head (tg512) | 259.9 tok/s | 143.1 tok/s |
| Decode after a 4K-token prompt, MTP 4 + n-gram drafts 7 | 207.5-218.4 tok/s | — |
| Prompt, 4,096 tokens, `--prefill-chunk 4096` | 2,228 tok/s | 1,662 tok/s |
| Prompt, 131,072 tokens of code and docs | 4,324 tok/s | 10,933 tok/s |

Decode is fast because the IQ3_S experts are small: the VRAM cache holds 13,008 of them, and
98.6 % of routed experts hit it in tg512. Long prompts are slower than NVFP4's: the exact integer
tensor-core tiles are the limit there, against NVFP4's FP4 tensor cores.

## Provenance

Converted with `tools.convert` (recipe `qwen3_8_flash_next_gsq_rco_iq3s`) and renamed into numbered
parts with `tools.artifact.rename --numbered`; `artifact_id` `8fcdca815634481d9e1af0bf392713db`,
the recipe as in Infernix `4ee28eb1`. The conversion report and the artifact's own provenance record the
paths of the machine that converted it. The n-gram volume was written from the GGUF's `IQ4_NL`
table.

## License

ISTA-DASLab publish their GGUF under the [Apache License 2.0](LICENSE-APACHE-2.0); see `NOTICE`
for the attribution and the changes made in this conversion. The underlying
[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) is subject to the
[Qwen Community License 1.0](LICENSE), which among other conditions requires a separate license from
Qwen for some commercial uses.
