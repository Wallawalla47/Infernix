---
library_name: infernix
pipeline_tag: image-text-to-text
inference: false
license: mit
license_link: https://huggingface.co/ornith-ai/Ornith-1.5-122B/blob/main/LICENSE
base_model:
  - ornith-ai/Ornith-1.5-122B-NVFP4
base_model_relation: quantized
tags:
  - infernix
  - ornith
  - qwen3.5
  - nvfp4
  - moe
  - mtp
  - dflash2
  - blackwell
  - rtx-5090
  - multimodal
  - conversational
  - cuda
---

# Ornith-1.5-122B NVFP4 for Infernix

> **Preliminary support:** this model runs and is verified, but I am still working on optimisations,
> so its speeds will change.

> **AI disclaimer:** this artifact, its conversion recipe, the engine that runs it and most of this
> card were made with AI (mostly Claude Opus 5.5). It is hobby work and is likely to be neither
> complete nor entirely accurate.

[ornith-ai/Ornith-1.5-122B-NVFP4](https://huggingface.co/ornith-ai/Ornith-1.5-122B-NVFP4) (a
`Qwen3_5MoeForCausalLM` at the Qwen3.5-122B-A10B geometry: 48 layers, hidden size 3072, 256 routed
experts per layer, top 8) packed for [Infernix](https://github.com/Wallawalla47/Infernix), a C++/CUDA
engine that runs it on **one RTX 5090** (32 GB). The 12,288 routed experts (60.75 GiB) live in pinned
host RAM, a VRAM cache holds the hot ones (about 4,450-4,850 of them), and the CPU computes part of
each layer's misses while the GPU reads the rest over PCIe.

The artifact also carries the model's MTP drafter, the
[incoai/Ornith-1.5-122B-DFlash2](https://huggingface.co/incoai/Ornith-1.5-122B-DFlash2) drafter, the
vision tower, a Q4 proposal head for `--lm-head-draft` and the model card's recommended sampling
settings (temperature 1.0, top-p 0.95, top-k 20, min-p 0, presence penalty 1.5, in both thinking
modes).

## Files

| File | Bytes | What it is |
|---|---:|---|
| `Ornith-1.5-122B-NVFP4-Infernix-00001-of-00003.infernix` | 32,000,000,000 | model, part 1 of 3 (also holds the table of contents) |
| `Ornith-1.5-122B-NVFP4-Infernix-00002-of-00003.infernix` | 32,000,000,000 | model, part 2 of 3 |
| `Ornith-1.5-122B-NVFP4-Infernix-00003-of-00003.infernix` | 11,331,296,516 | model, part 3 of 3 |
| `Ornith-1.5-122B-NVFP4-Infernix-00001-of-00003.infernix.conversion.json` | 620,329 | conversion report: sources, methods and formats per object |
| `LICENSE-APACHE-2.0`, `NOTICE` | | the DFlash2 drafter's licence; attribution and the changes made |

Keep the three model parts in one directory and give Infernix part 1, as with a split GGUF. No
n-gram volume is needed.

## Requirements

- An RTX 5090 (or another `sm_120a` GPU with 32 GB).
- About 70 GiB of free host RAM for the all-in-RAM mode. With less, the RAM ledger keeps as many
  experts as fit and reads the rest from the artifact (the SSD expert tier); `--expert-ram-mib` sets
  the split. Windows caps page-locked memory at 80 % of RAM.
- About 75 GB of disk. An Infernix build from `main` at or after commit `5069e375`.

## What is bit-exact and what is quantized

Converted with Infernix's recipe `qwen3_5_moe_nvfp4_modelopt` (components text, MTP, vision and
DFlash2, plus a proposal head). `python -m tools.convert.verify_qwen3_5` compared the artifact with
the checkpoint word for word: 12,336 expert records, 1,178 tensors and 240 FP8 input scales, no
mismatch.

- **Bit-exact to the checkpoint:** every routed and shared expert (NVFP4 words, block scales,
  multipliers), the FP8 (E4M3, row-scaled) attention, GDN and output projections with their input
  scales, the router, norms, embeddings and the vision tower.
- **Re-quantized (drafters only):** the 777 MTP tensors and the 55 DFlash2 tensors, to the engine's
  Q4 and Q8 layouts, plus the 131,072-row Q4 proposal head built from the output head.

## Run

```text
infernix Ornith-1.5-122B-NVFP4-Infernix-00001-of-00003.infernix \
  --prompt "Explain why the sky is blue." --max-context 8192 --max-new 256
infernix-serve Ornith-1.5-122B-NVFP4-Infernix-00001-of-00003.infernix --max-context 32768
```

- **Context memory:** only the first 32,768 tokens of KV are held at startup; longer contexts take
  their KV from the expert cache as they grow and give it back when they end.
- **Speculation:** `--spec mtp --draft-tokens 2` verifies only the drafts the drafter is confident
  enough of to pay for their experts; `--spec dflash2` uses the DFlash2 drafter. `--draft-tokens auto`
  decodes without a drafter on this model.
- `--reproducible` runs the bit-exact route, `--lossless` the correctly rounded one (about 5 tok/s).

## Speed (RTX 5090, i9-13900K, 96 GB DDR5, PCIe 5.0 x8, bf16 KV, one request)

| Measure | Result |
|---|---:|
| Plain decode, greedy (tg256 after a 512-token prompt) | 90-97 tok/s |
| Plain decode, the sampled request with the card's settings | 74-78 tok/s |
| MTP with 2 drafts, the same sampled request | 78-81 tok/s |
| Prefill, 4K / 32K prompt | 1,750 / 6,500 tok/s |

Decode depends on the text, through the share of its experts already in VRAM (74-80 % hit rate),
because it is bound by the bytes of the missing experts. Quality has not been measured beyond the
bit-exact comparison above. Details and the options that did not help are in the
[design document](https://github.com/Wallawalla47/Infernix/blob/main/docs/maintainer/qwen3_5-122b-offload-design.md#01-second-pass-2026-10-22-measured-the-same-day-from-2025)
and the [guide](https://github.com/Wallawalla47/Infernix/blob/main/docs/qwen3_5-122b.md).

## License

The model is [Ornith-1.5-122B](https://huggingface.co/ornith-ai/Ornith-1.5-122B), declared MIT by
its authors (see the link in the header). The DFlash2 drafter is Inco AI's Apache-2.0 model
(`LICENSE-APACHE-2.0`). The Infernix engine has its own license in its repository.
