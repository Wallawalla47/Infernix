---
library_name: infernix
pipeline_tag: image-text-to-text
inference: false
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model:
  - Qwen/Qwen3.8-Flash-Next
  - RadixArk/Qwen3.8-Flash-Next-NVFP4
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

# Qwen3.8-Flash-Next NVFP4 for Infernix

> **AI disclaimer:** this artifact, its conversion recipe, the engine that runs it and most of this
> card were made with AI (mostly Claude Opus 5.5). It is hobby work and is likely to be neither
> complete nor entirely accurate.

[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) — 48 layers, 512 routed
experts per layer (top-10), an MTP drafter and vision — packed for
[Infernix](https://github.com/Wallawalla47/ninfer-custom/tree/Infernix), a C++/CUDA engine that runs
it on **one RTX 5090** (32 GB). The 24,576 routed experts (63 GiB) live in pinned host RAM, a VRAM
cache holds the hot ones, the CPU computes part of each layer's misses, and the 52 GB PLE n-gram
table is read from an NVMe drive.

## Files

| File | Bytes | What it is |
|---|---:|---|
| `qwen3_8_flash_next_nvfp4_dense8m-00001-of-00003.infernix` | 32,000,000,000 | model, part 1 of 3 (also holds the table of contents) |
| `qwen3_8_flash_next_nvfp4_dense8m-00002-of-00003.infernix` | 32,000,000,000 | model, part 2 of 3 |
| `qwen3_8_flash_next_nvfp4_dense8m-00003-of-00003.infernix` | 12,867,265,280 | model, part 3 of 3 |
| `qwen3_8_flash_next_nvfp4_dense8m.ngram` | 52,429,058,048 | PLE n-gram volume (read in random 4 KiB blocks) |
| `qwen3_8_flash_next_nvfp4_dense8m.conversion.json` | 914,141 | conversion report: sources, methods and formats per object |

Keep the three model parts in one directory and give Infernix part 1, as with a split GGUF. Put the
n-gram volume on an NVMe drive and pass it with `--ngram-volume`.

## Contents

Converted from [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)
(routed experts quantized to NVFP4 W4A4 with NVIDIA Model Optimizer; everything else BF16) with
Infernix's recipe B (`qwen3_8_flash_next_nvfp4_dense8`, components text, vision and MTP, plus a
proposal head):

- **Routed experts:** the checkpoint's NVFP4 tensors, imported bit-exactly.
- **Dense projections and `lm_head`:** `q8_g32_fp16` (8-bit, groups of 32, FP16 scales). Against
  recipe A, which keeps them BF16, this measured +0.008 ± 0.010 nats per token (not significant)
  and decodes about 22 % faster.
- **MTP drafter:** `q8_g32_fp16`, its routed experts `q4_g64_fp16`. The drafter only proposes
  tokens; every draft is verified by the full model, so its precision changes speed, not output.
- **Proposal head** for `--lm-head-draft`, a smaller draft output head.
- **Vision** encoder and preprocessor resources, BF16.

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
infernix-serve qwen3_8_flash_next_nvfp4_dense8m-00001-of-00003.infernix \
  --ngram-volume <nvme>/qwen3_8_flash_next_nvfp4_dense8m.ngram \
  --max-context 262144 --kv-dtype int8 --prefill-chunk 4096 \
  --spec mtp --draft-tokens 4 --lm-head-draft --max-concurrency 2
```

It serves the OpenAI Chat Completions and Responses APIs and the Anthropic Messages API. The
[Flash-Next guide](https://github.com/Wallawalla47/ninfer-custom/blob/Infernix/docs/qwen3_8-flash-next.md)
covers the KV formats, YaRN beyond 262K, images and video, the prefix cache, the SSD expert tier
and tuning.

## Quality

Teacher-forced perplexity on the same texts: **4.600**, against 4.864 for Unsloth's
`Qwen3.8-Flash-Next-UD-Q4_K_XL` GGUF (lower is better).

## Performance

RTX 5090 in a PCIe Gen5 **x8** link (x16 is likely faster, most of all for this model, whose
expert-cache misses cross PCIe on every token), Core i9-13900K, 96 GB DDR5, Windows 11; Infernix
at `cc76e684`, INT8 KV, MTP with 4 drafts and the proposal head. Strata 0.1.40 on the same machine
runs the Unsloth UD-Q4_K_XL GGUF with its own MTP drafter.

| Context | TTFT, Infernix | TTFT, Strata | Decode tok/s, Infernix | Decode tok/s, Strata |
|---|---:|---:|---:|---:|
| ~8K | 1.92 s | 12.28 s | 142.9 | 86.4 |
| ~128K | 17.84 s | 103.27 s | 138.4 | 77.2 |
| ~250K | 33.87 s | 158.07 s | 159.9 | 95.8 |

One request at a time, greedy, 1,024 output tokens, means over two or three prompts. Decode with
speculation depends on how predictable the text is, so single prompts vary by up to ±20 %.

In a replayed agentic coding workload (three sessions plus subagents, up to eight requests in
flight, two seeds), Infernix averaged 5.5 s to the first token against Strata's 25.3 s, served
84.5 % of prompt tokens from its prefix cache against 64.3 %, and finished in 8.1 minutes against
18.3. Methods and the full tables are in the
[Infernix README](https://github.com/Wallawalla47/ninfer-custom/blob/Infernix/README.md#benchmarks).

## Provenance

The conversion report and the artifact's own provenance record the paths of the machine that
converted it. `artifact_id` `c7d4b790549443fb9db6ef9c13a3b92d`. The artifact was written as
`.ninfer` and renamed to these three files in place with
`python -m tools.artifact.rename --numbered` (the same bytes; only the recorded part names
changed). The n-gram volume was written by the same conversion.

## License

[Qwen Community License 1.0](LICENSE), inherited from
[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) through RadixArk's NVFP4
checkpoint, whose card defers to the source model's terms. Read the license before using the model;
among other conditions, it requires a separate license from Qwen for some commercial uses.
