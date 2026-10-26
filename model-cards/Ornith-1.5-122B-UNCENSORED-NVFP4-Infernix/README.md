---
library_name: infernix
pipeline_tag: image-text-to-text
inference: false
license: mit
license_link: https://huggingface.co/ornith-ai/Ornith-1.5-122B/blob/main/LICENSE
base_model:
  - dealignai/Ornith-1.5-122B-UNCENSORED-NVFP4
base_model_relation: quantized
tags:
  - infernix
  - ornith
  - qwen3.5
  - nvfp4
  - fp8
  - moe
  - mtp
  - dflash2
  - uncensored
  - abliteration
  - blackwell
  - rtx-5090
  - multimodal
  - conversational
  - cuda
---

# Ornith-1.5-122B UNCENSORED NVFP4 for Infernix

> **Preliminary support:** this model runs and is verified, but I am still working on optimisations,
> so its speeds will change.

> **AI disclaimer:** this artifact, its conversion recipe, the engine that runs it and most of this
> card were made with AI (mostly Claude Opus 5.5). It is hobby work and is likely to be neither
> complete nor entirely accurate.

> **Content warning:** the source model is a refusal-removed build. As its authors say, it answers
> requests the base model refuses, including harmful ones; do not deploy it to end users without your
> own safety controls.

[dealignai/Ornith-1.5-122B-UNCENSORED-NVFP4](https://huggingface.co/dealignai/Ornith-1.5-122B-UNCENSORED-NVFP4)
(revision `e6a1c178`; a refusal-removed NVFP4 export of
[ornith-ai/Ornith-1.5-122B](https://huggingface.co/ornith-ai/Ornith-1.5-122B), a `Qwen3_5MoeForCausalLM`
at the Qwen3.5-122B-A10B geometry: 48 layers, hidden size 3072, 256 routed experts per layer, top 8)
packed for [Infernix](https://github.com/Wallawalla47/Infernix), a C++/CUDA engine that runs it on
**one RTX 5090** (32 GB). The 12,288 routed experts (60.75 GiB) live in pinned host RAM, a VRAM cache
holds the hot ones, and the CPU computes part of each layer's misses while the GPU reads the rest over
PCIe.

The artifact also carries the model's MTP drafter, the
[incoai/Ornith-1.5-122B-DFlash2](https://huggingface.co/incoai/Ornith-1.5-122B-DFlash2) drafter
(the copy bundled in dealignai's repository, byte-identical to Inco AI's), the vision tower, a Q4
proposal head for `--lm-head-draft` and the source card's recommended sampling settings (temperature
0.6, top-p 0.95, top-k 20, min-p 0, presence penalty 0, in both thinking modes).

## Files

| File | Bytes | What it is |
|---|---:|---|
| `Ornith-1.5-122B-UNCENSORED-NVFP4-Infernix-00001-of-00003.infernix` | 32,000,000,000 | model, part 1 of 3 (also holds the table of contents) |
| `Ornith-1.5-122B-UNCENSORED-NVFP4-Infernix-00002-of-00003.infernix` | 32,000,000,000 | model, part 2 of 3 |
| `Ornith-1.5-122B-UNCENSORED-NVFP4-Infernix-00003-of-00003.infernix` | 11,864,907,776 | model, part 3 of 3 |
| `Ornith-1.5-122B-UNCENSORED-NVFP4-Infernix-00001-of-00003.infernix.conversion.json` | 636,919 | conversion report: sources, methods and formats per object |
| `LICENSE-APACHE-2.0`, `NOTICE` | | the DFlash2 drafter's licence; attribution and the changes made |

Keep the three model parts in one directory and give Infernix part 1, as with a split GGUF. No
n-gram volume is needed.

## Requirements

- An RTX 5090 (or another `sm_120a` GPU with 32 GB).
- About 70 GiB of free host RAM for the all-in-RAM mode. With less, the RAM ledger keeps as many
  experts as fit and reads the rest from the artifact (the SSD expert tier); `--expert-ram-mib` sets
  the split. Windows caps page-locked memory at 80 % of RAM.
- About 76 GB of disk. An Infernix build from `main` at or after commit `c17e24a8`.

## What is bit-exact and what is quantized

Converted with Infernix's recipe `qwen3_5_moe_nvfp4_modelopt` (components text, MTP, vision and
DFlash2, plus a proposal head). `python -m tools.convert.verify_qwen3_5` compared the artifact with
the checkpoint word for word: 12,288 expert records and 1,322 tensors, no mismatch.

- **Bit-exact to the checkpoint:** every routed expert (NVFP4 words, block scales, multipliers), the
  FP8 (E4M3, one FP32 scale per row) attention, GDN, output, shared-expert and head projections, the
  router, norms, embeddings and the vision tower.
- **Re-quantized (drafters only):** the 777 MTP tensors (from the checkpoint's per-row and block-FP8
  values) and the 55 DFlash2 tensors, to the engine's Q4 and Q8 layouts, plus the 131,072-row Q4
  proposal head built from the output head.
- **Arithmetic:** the checkpoint quantizes activations per token at run time
  (`FP8_PER_CHANNEL_PER_TOKEN`); Infernix runs the attention and GDN projections with per-token 8-bit
  activations in prefill and 16-bit activations in decode, and the shared experts and head with
  16-bit activations, as for the original Ornith artifact.

## Run

```text
infernix Ornith-1.5-122B-UNCENSORED-NVFP4-Infernix-00001-of-00003.infernix \
  --prompt "Explain why the sky is blue." --max-context 8192 --max-new 256
infernix-serve Ornith-1.5-122B-UNCENSORED-NVFP4-Infernix-00001-of-00003.infernix --max-context 262144 --kv-dtype int8 --spec mtp --draft-tokens auto
```

- **Context memory:** only the first 32,768 tokens of KV are held at startup; longer contexts take
  their KV from the expert cache as they grow and give it back when they end.
- **Speculation (recommended: `--spec mtp --draft-tokens auto`):** two MTP drafts each round, verified
  only when the drafter is confident enough of them to pay for their experts, as for the original
  Ornith (+18 % decode greedy, +7.5 % at the card's sampling). `--spec dflash2` uses the DFlash2 drafter; with `--draft-tokens auto` it decodes
  without it, as on the original.
- `--reproducible` runs the bit-exact route, `--lossless` the correctly rounded one.

## Speed (RTX 5090, i9-13900K, 96 GB DDR5, PCIe 5.0 x8, one request)

INT8 KV, fresh engine each run (empty expert cache), three texts; ranges are across the texts.

| Workload | Speed |
|---|---|
| Decode, 1,280-token prompts, greedy: plain / `--spec mtp --draft-tokens auto` | 64-80 / 73-97 tok/s (MTP +18 % median) |
| The same at the card's sampling (temperature 0.6) | 64-80 / 68-97 tok/s (MTP +7.5 % median) |
| Prefill, 4K / 32K prompt | 1,700 / 4,530 tok/s |

Measured side by side with the original
[Ornith-1.5-122B-NVFP4-Infernix](https://huggingface.co/Wallawalla47/Ornith-1.5-122B-NVFP4-Infernix)
at these settings, prefill is the same (32K +1 %, 4K -0.4 %) and decode differs by less than the
run-to-run spread (+2.4 % median over two pairs, worst -5 %). The original's card quotes its prefill
with BF16 KV.

Decode depends on the text, through the share of its experts already in VRAM, because it is bound by
the bytes of the missing experts. Quality has not been measured beyond the bit-exact comparison above.
The [guide](https://github.com/Wallawalla47/Infernix/blob/main/docs/qwen3_5-122b.md#ornith-15-122b-uncensored-nvfp4)
has the conversion and verification commands.

## License

The model is dealignai's refusal-removed build of
[Ornith-1.5-122B](https://huggingface.co/ornith-ai/Ornith-1.5-122B), declared MIT by its authors (see
the link in the header). The DFlash2 drafter is Inco AI's Apache-2.0 model (`LICENSE-APACHE-2.0`). The
Infernix engine has its own license in its repository.
