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

# Qwen3.8-Flash-Next NVIDIA NVFP4 Dense8 for Infernix

> **AI disclaimer:** this artifact, its conversion recipe, the engine that runs it and most of this
> card were made with AI (mostly Claude Opus 5.5). It is hobby work and is likely to be neither
> complete nor entirely accurate.

NVIDIA's [Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) — 48
layers, 512 routed experts per layer (top-10), an MTP drafter and vision — packed for
[Infernix](https://github.com/Wallawalla47/ninfer-custom/tree/Infernix), a C++/CUDA engine that runs
it on **one RTX 5090** (32 GB). The 24,576 routed experts (63 GiB) live in pinned host RAM, a VRAM
cache holds the hot ones, the CPU computes part of each layer's misses, and the 52 GB PLE n-gram
table is read from an NVMe drive.

This is the faster of the two Infernix conversions: NVIDIA's NVFP4 experts and FP8 n-gram table
are kept bit-exact, and the BF16 dense projections and `lm_head` are stored in 8 bits ("Dense8").

> **Bit-exact alternative:**
> [Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix)
> keeps every weight of the main model and vision tower exactly as NVIDIA stores it (BF16 dense
> projections and `lm_head` included). It is **slower**: on the same machine and build it decoded
> about 18 % slower per request (108-131 against 136-160 tok/s from 8K to 250K tokens of context)
> and reached the first token 6-8 % later at 8K (level from 128K), because it reads ~8.6 GB of dense
> weights per token instead of ~5.1 GB and caches fewer experts in VRAM. Quality measured the same
> (perplexity 4.666 against this artifact's 4.654), so choose it only when the weights must be
> NVIDIA's unchanged.

## Files

| File | Bytes | What it is |
|---|---:|---|
| `Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix-00001-of-00003.infernix` | 32,000,000,000 | model, part 1 of 3 (also holds the table of contents) |
| `Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix-00002-of-00003.infernix` | 32,000,000,000 | model, part 2 of 3 |
| `Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix-00003-of-00003.infernix` | 12,867,265,280 | model, part 3 of 3 |
| `Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram` | 52,429,058,048 | PLE n-gram volume (read in random 4 KiB blocks); the same file serves both conversions |
| `Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix.conversion.json` | 906,018 | conversion report: sources, methods and formats per object |
| `NVIDIA-Open-Model-License.txt`, `NOTICE`, `LICENSE` | | the NVIDIA Open Model License; NVIDIA's attribution notice; the Qwen Community License 1.0 |

Keep the three model parts in one directory and give Infernix part 1, as with a split GGUF. Put the
n-gram volume on an NVMe drive and pass it with `--ngram-volume`.

## What is bit-exact and what is quantized

Converted from [nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4)
with Infernix's recipe B, `qwen3_8_flash_next_nvfp4_dense8` (components text, vision and MTP, plus a
proposal head). `python -m tools.flash_next.verify_artifact` compared the artifact with the
checkpoint tensor by tensor: 24,576 experts, 987 tensors and 320,001,536 n-gram rows with no
mismatch, and 629 text-model and 1,556 drafter parameters re-quantized.

**Bit-exact to NVIDIA's checkpoint:**

- **Routed experts** (24,576): NVIDIA's NVFP4 words (E2M1 codes, E4M3 scale per 16, FP32
  multiplier and FP32 activation input scale per matrix), repacked into Infernix's expert layout
  without changing a bit.
- **PLE n-gram table:** every FP8 row byte-identical in the n-gram volume, with its scale.
- **Router, shared-expert gate, GDN `a`/`b`, QSA query/gate/key/value/indexer projections, token
  embedding, norms, convolution weights and vision:** BF16 as NVIDIA stores them; the GDN `A_log`
  and `dt_bias` widened from BF16 to FP32, value for value.

**Quantized from NVIDIA's BF16** (`q8_g32_fp16`: 8-bit integers, one FP16 scale per 32 weights):

- **Dense projections and `lm_head`** (629 tensors, 97 % of the dense bytes): GDN q/k/v/z and
  output projections, QSA output projections, shared experts (gate, up, down), hyper-connection
  mixers (down, inject and up of every layer, and the final mixer), PLE key/value projections and
  `lm_head`. Against keeping them BF16 (the bit-exact conversion) this saves
  3.5 GB of VRAM for the expert cache and reads ~5.1 GB instead of ~8.6 GB of dense weights per
  token; the two conversions measured level on quality (see [Quality](#quality)).

**Drafting only** (never changes output, since the full model verifies every draft):

- **MTP drafter** (1,556 parameters): projections `q8_g32_fp16` (router and shared-expert gate
  BF16); its routed experts, block-scaled FP8 in NVIDIA's checkpoint, `q4_g64_fp16` with
  MSE-chosen scales from their exact values.
- **Proposal head** for `--lm-head-draft` (not in NVIDIA's checkpoint): `q4_g64_fp16` copies of
  the `lm_head` rows of the 131,072 most frequent tokens.

Average bits per weight of the text decoder (176.9 billion weights; scales counted; vision and the
MTP drafter excluded, as Unsloth's GGUF holds neither):

| | Routed experts | Dense and other | PLE n-gram table | All | Without the n-gram table |
|---|---:|---:|---:|---:|---:|
| This artifact (+ n-gram volume) | 4.50 | 10.34 | 8.19 | 5.73 | 4.73 |
| The bit-exact conversion (+ n-gram volume) | 4.50 | 16.00 | 8.19 | 5.89 | 4.95 |
| `nvidia/Qwen3.8-Flash-Next-NVFP4` | 4.50 | 16.00 | 8.00 | 5.83 | 4.95 |
| Unsloth `Qwen3.8-Flash-Next-UD-Q4_K_XL` | 5.10 | 8.90 | 4.50 | 5.03 | 5.25 |

The n-gram volume's 8.19 includes its 4 KiB block padding; the table itself is 8.00.

## Requirements

- **GPU:** NVIDIA RTX 5090 (Blackwell, `sm_120a`), all of its 32 GB.
- **RAM:** about 69.3 GiB available for the experts, 73.3 GiB for `infernix-serve` with its
  prefix cache. With less, Infernix's SSD expert tier keeps what fits in RAM and reads the rest
  from the artifact (slower).
- **Disk:** ~77 GB for the artifact, plus the 52 GB n-gram volume on an NVMe drive. A fast
  consumer drive is enough; a SATA SSD or a hard disk is far slower.
- **Engine:** Infernix (Windows or Linux), branch `Infernix`.

## Run

```text
infernix-serve Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix-00001-of-00003.infernix \
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
INT8 KV in both engines), against Strata 0.1.40 running Unsloth's UD-Q4_K_XL GGUF, with the
bit-exact conversion beside it (Infernix at the 2026-10-08 kernels):

| Text | Perplexity, this artifact | ΔNLL vs Strata | Perplexity, bit-exact | ΔNLL vs Strata | Perplexity, Strata |
|---|---:|---:|---:|---:|---:|
| Code | 1.913 | +0.007 ± 0.013 | 1.917 | +0.009 ± 0.012 | 1.899 |
| Document | 9.666 | −0.006 ± 0.016 | 9.707 | −0.001 ± 0.018 | 9.721 |
| Chat | 3.493 | **−0.098 ± 0.022** | 3.499 | **−0.096 ± 0.021** | 3.852 |
| All | **4.654** | **−0.040 ± 0.011** | 4.666 | **−0.037 ± 0.011** | 4.844 |

ΔNLL is the mean per-token difference in nats (negative: Infernix assigns the actual text higher
probability) with its standard error. The two conversions are level within that error; storing the
dense projections in 8 bits cost no measurable quality on these texts.

## Performance

RTX 5090 in a PCIe Gen5 **x8** link (x16 is likely faster, most of all for this model, whose
expert-cache misses cross PCIe on every token), Core i9-13900K, 96 GB DDR5, Windows 11; Infernix
at `3b35ccf9`, INT8 KV, MTP with 4 drafts and the proposal head. Strata 0.1.40 on the same machine
runs the Unsloth UD-Q4_K_XL GGUF with its own MTP drafter.

| Context | TTFT, Infernix | TTFT, Strata | Decode tok/s, Infernix | Decode tok/s, Strata |
|---|---:|---:|---:|---:|
| ~8K | 1.93 s | 12.28 s | 138.8 | 86.4 |
| ~128K | 17.86 s | 103.27 s | 136.2 | 77.2 |
| ~250K | 34.24 s | 158.07 s | 153.3 | 95.8 |

One request at a time, greedy, 1,024 output tokens, means over two or three prompts. Decode with
speculation depends on how predictable the text is, so single prompts vary by up to ±20 %.
Re-measured with the 2026-10-08 kernels on the same prompts, Dense8 was level with these figures
(decode +0.4 % per request, −7 % to +7 %; TTFT −0.2 %); the bit-exact conversion decoded 18 % slower
than it on the same build.

In a replayed agentic coding workload (three sessions plus subagents, up to eight requests in
flight, two seeds), Infernix averaged 5.4 s to the first token against Strata's 25.3 s, served
83.7 % of prompt tokens from its prefix cache against 64.3 %, decoded one request at 129 tok/s
against 86, and finished in 8.1 minutes against 18.3. Methods and the full tables are in the
[Infernix README](https://github.com/Wallawalla47/ninfer-custom/blob/Infernix/README.md#benchmarks).

## Provenance

Converted with `tools.convert` (recipe `qwen3_8_flash_next_nvfp4_dense8`) and renamed into numbered
parts with `tools.artifact.rename --numbered`; `artifact_id` `c70f7dee5052435eb0425e06e1bdcc6c`. The
conversion report and the artifact's own provenance record the paths of the machine that converted
it. The artifact was first published under the name `Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix`
and renamed with `-Dense8` on 2026-10-08, when the bit-exact conversion took that name; only the
file names changed (no payload byte moved, and the `artifact_id` is the same). The n-gram volume was written from the same FP8 table (byte-identical to
NVIDIA's).

## License

Use of this model is governed by the
[NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)
(copy in `NVIDIA-Open-Model-License.txt`; "Licensed by NVIDIA Corporation under the NVIDIA Open
Model License", see `NOTICE`). The underlying
[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) is also subject to the
[Qwen Community License 1.0](LICENSE), which among other conditions requires a separate license from
Qwen for some commercial uses.
