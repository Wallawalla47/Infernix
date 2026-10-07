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

# Qwen3.8-Flash-Next NVIDIA NVFP4 for Infernix

> **AI disclaimer:** this artifact, its conversion recipe, the engine that runs it and most of this
> card were made with AI (mostly Claude Opus 5.5). It is hobby work and is likely to be neither
> complete nor entirely accurate.

NVIDIA's [Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) — 48
layers, 512 routed experts per layer (top-10), an MTP drafter and vision — packed for
[Infernix](https://github.com/Wallawalla47/ninfer-custom/tree/Infernix), a C++/CUDA engine that runs
it on **one RTX 5090** (32 GB). The 24,576 routed experts (63 GiB) live in pinned host RAM, a VRAM
cache holds the hot ones, the CPU computes part of each layer's misses, and the 52 GB PLE n-gram
table is read from an NVMe drive.

## Files

| File | Bytes | What it is |
|---|---:|---|
| `Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix-00001-of-00003.infernix` | 32,000,000,000 | model, part 1 of 3 (also holds the table of contents) |
| `Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix-00002-of-00003.infernix` | 32,000,000,000 | model, part 2 of 3 |
| `Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix-00003-of-00003.infernix` | 12,867,265,280 | model, part 3 of 3 |
| `Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram` | 52,429,058,048 | PLE n-gram volume (read in random 4 KiB blocks) |
| `Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.conversion.json` | 905,990 | conversion report: sources, methods and formats per object |
| `NVIDIA-Open-Model-License.txt`, `NOTICE`, `LICENSE` | | the NVIDIA Open Model License; NVIDIA's attribution notice; the Qwen Community License 1.0 |

Keep the three model parts in one directory and give Infernix part 1, as with a split GGUF. Put the
n-gram volume on an NVMe drive and pass it with `--ngram-volume`.

## Contents

Converted from [nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4)
with Infernix's recipe B (components text, vision and MTP, plus a proposal head):

- **Routed experts:** NVIDIA's NVFP4 tensors (W4A4, E4M3 scale per 16, FP32 multiplier and input
  scale per matrix), imported bit-exactly.
- **PLE n-gram table:** NVIDIA's FP8 table, imported bit-exactly into the n-gram volume.
- **Dense projections and `lm_head`:** `q8_g32_fp16` (8-bit, groups of 32, FP16 scales) from
  NVIDIA's BF16. Against keeping them BF16 this measured +0.008 ± 0.010 nats per token (not
  significant) and decodes about 22 % faster.
- **Router, gates, norms, embedding, vision:** BF16 as NVIDIA stores them.
- **MTP drafter:** dense parts `q8_g32_fp16`; its routed experts, block-scaled FP8 in NVIDIA's
  checkpoint, `q4_g64_fp16` from their exact values. The drafter only proposes tokens; every draft is
  verified by the full model, so its precision changes speed, not output.
- **Proposal head** for `--lm-head-draft`, a smaller draft output head.

Average bits per weight of the text decoder (176.9 billion weights; scales counted; vision and the
MTP drafter excluded, as Unsloth's GGUF holds neither):

| | Routed experts | Dense and other | PLE n-gram table | All | Without the n-gram table |
|---|---:|---:|---:|---:|---:|
| This artifact (+ n-gram volume) | 4.50 | 10.34 | 8.19 | 5.73 | 4.73 |
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
INT8 KV in both engines), against Strata 0.1.40 running Unsloth's UD-Q4_K_XL GGUF:

| Text | Perplexity, this artifact | Perplexity, Strata | ΔNLL (Infernix − Strata) |
|---|---:|---:|---:|
| Code | 1.905 | 1.899 | +0.003 ± 0.013 |
| Document | 9.781 | 9.721 | +0.006 ± 0.014 |
| Chat | 3.477 | 3.852 | **−0.102 ± 0.023** |
| All | **4.664** | 4.844 | **−0.038 ± 0.011** |

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

In a replayed agentic coding workload (three sessions plus subagents, up to eight requests in
flight, two seeds), Infernix averaged 5.4 s to the first token against Strata's 25.3 s, served
83.7 % of prompt tokens from its prefix cache against 64.3 %, decoded one request at 129 tok/s
against 86, and finished in 8.1 minutes against 18.3. Methods and the full tables are in the
[Infernix README](https://github.com/Wallawalla47/ninfer-custom/blob/Infernix/README.md#benchmarks).

## Provenance

Converted with `tools.convert` (recipe `qwen3_8_flash_next_nvfp4_dense8`) and renamed into numbered
parts with `tools.artifact.rename --numbered`; `artifact_id` `c70f7dee5052435eb0425e06e1bdcc6c`. The
conversion report and the artifact's own provenance record the paths of the machine that converted
it. The n-gram volume was written from the same FP8 table (byte-identical to NVIDIA's).

## License

Use of this model is governed by the
[NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)
(copy in `NVIDIA-Open-Model-License.txt`; "Licensed by NVIDIA Corporation under the NVIDIA Open
Model License", see `NOTICE`). The underlying
[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) is also subject to the
[Qwen Community License 1.0](LICENSE), which among other conditions requires a separate license from
Qwen for some commercial uses.
