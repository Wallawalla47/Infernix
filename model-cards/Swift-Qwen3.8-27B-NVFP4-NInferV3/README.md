---
library_name: ninfer
pipeline_tag: image-text-to-text
license: other
license_name: swift-open-license-1.0
license_link: LICENSE
url: https://github.com/Wallawalla47/Infernix
base_model: ukisai/Swift-Qwen3.8-27B-NVFP4
base_model_relation: quantized
tags:
  - Qwen3.8
  - swift
  - efficient-thinking
  - token-efficient
  - ModelOpt
  - nvfp4
  - fp4
  - w4a4
  - 4bit
  - fp8
  - quantization
  - quantized
  - blackwell
  - infernix
  - ninfer
  - qwen3.8-27b
  - 4-bit precision
  - multimodal
  - vision
  - reasoning
  - thinking
  - long-context
  - agentic
  - mtp
  - speculative-decoding
---

# Swift-Qwen3.8-27B-NVFP4-NInferV3

**A single-file NInfer V3 (`.ninfer`) artifact of
[ukisai/Swift-Qwen3.8-27B-NVFP4](https://huggingface.co/ukisai/Swift-Qwen3.8-27B-NVFP4),
built for an NVIDIA RTX 5090 (sm_120a), for [Infernix](https://github.com/Wallawalla47/Infernix)
(recommended) and [NInfer](https://github.com/Neroued/ninfer). No conversion step at load time:
point `infernix-serve` (or `ninfer-serve`) at the `.ninfer` file and run.**

> **AI disclaimer:** this artifact, its conversion and most of this card were made with AI (mostly
> Claude Opus 5.5). It is hobby work and is likely to be neither complete nor entirely accurate.

This repository is the NInfer form of UkisAI's checkpoint above: a mixed NVFP4/FP8 quantization of
[Swift-Qwen3.8-27B](https://huggingface.co/ukisai/Swift-Qwen3.8-27b), UkisAI's reasoning-efficient
fine-tune of Qwen3.8-27B. UkisAI quantized it with NVIDIA Model Optimizer following the precision
mapping of [nvidia/Qwen3.8-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4), so it has the
same layout, and this artifact is built exactly like my
[Qwen3.8-27B-NVIDIA-NVFP4-NInferV3](https://huggingface.co/Wallawalla47/Qwen3.8-27B-NVIDIA-NVFP4-NInferV3):
the NVFP4/FP8 weights are carried over **bit-exactly** for every transformer linear except the
language-model head, and two speculative-decoding components (DFlash2 draft model, indexed proposal
head) are added so the artifact runs with the engines' fastest decode paths.

**About Swift:** UkisAI trained Swift to reach the same answers with shorter thinking. Their claims
and measurements are on [UkisAI's model card](https://huggingface.co/ukisai/Swift-Qwen3.8-27B-NVFP4)
and the [BF16 model card](https://huggingface.co/ukisai/Swift-Qwen3.8-27b); I have not measured them.

**Recommended engine: [Infernix](https://github.com/Wallawalla47/Infernix)**, my engine grown from
NInfer, which replaces my earlier [ninfer-custom](https://github.com/Wallawalla47/ninfer-custom)
fork. In an agentic coding replay on the 27B it served 90 % of prompt tokens from its prefix cache
and reached the first token in about half NInfer's average time, and it prefills long prompts
1.3-1.5× faster, at the same quality
([benchmarks](https://github.com/Wallawalla47/Infernix#qwen38-27b-infernix-vs-ninfer), measured on
NInfer's official 27B artifact). It also runs Qwen3.8-Flash-Next, including
[Swift 1.5 Qwen3.8-Flash-Next](https://huggingface.co/Wallawalla47/Swift-1.5-Qwen3.8-Flash-Next-NVFP4-Dense8-Infernix).
The artifact also works with [NInfer](https://github.com/Neroued/ninfer) itself and with
ninfer-custom (see [Tested](#tested)).

## What is in the file

| Component | Contents |
|---|---|
| `text` | Qwen3.5-family hybrid backbone: 64 layers (Gated Delta-Net linear attention + full attention every 4th layer), hidden 5120, vocab 248,320, native context 262,144 |
| `vision` | qwen3_5_vision tower (depth 27) for image/video input |
| `mtp` | Qwen3.5 MTP companion (UkisAI's fine-tuned head) for MTP speculative decoding |
| `dflash2` | 5-layer sliding-attention draft model (window 2048, selector rank 256 / top-16) — the `--spec dflash2` path |
| `proposal` | **Indexed 131,072-row proposal head** gathered from the output head — enables cheap copy/proposal drafting via `--lm-head-draft` |

Format census of the 1,072 prepared weight objects, identical to the NVIDIA artifact: **128 NVFP4**
(MLP projections, imported bit-exact), **130 FP8** (e4m3fn, row-scale; attention + GDN projections
imported bit-exact, plus the re-quantised embedding and output head), 579 BF16, 96 FP32, 55 Q4, 54
Q5, 28 Q8 and 1 Q6 (grouped, FP16 scales), and 1 INT32 proposal index table.

| File | Bytes |
|---|---:|
| `Swift-Qwen3.8-27B-NVFP4-NInfer.ninfer` | 22,783,240,704 |
| `Swift-Qwen3.8-27B-NVFP4-NInfer.ninfer.conversion.json` (per-tensor conversion report) | 506,626 |
| `LICENSE`, `LICENSE-APACHE-2.0`, `NOTICE` (Swift Open License v1.0; Apache 2.0 of Qwen3.8-27B and DFlash2; UkisAI's notice and my change notice) | |

## How it was created

The artifact was built with Infernix's converter using the `qwen3_8_27b_nvfp4_nvidia` recipe
([`tools/convert/official_recipes.py`](https://github.com/Wallawalla47/Infernix/blob/main/tools/convert/official_recipes.py))
on CPU (~4-5 min), with components `text,vision,mtp` and the proposal head; the DFlash2 component was
then added as described below. Sources and treatment:

1. **ukisai/Swift-Qwen3.8-27B-NVFP4** at revision `17d5f9df` (NVIDIA Model Optimizer mixed
   NVFP4/FP8; calibrated by UkisAI with Local-Hessian NVFP4 on 2,048 Nemotron post-training samples,
   per its card) — the dominant source.
   - **MLP projections (gate/up/down, all 128 NVFP4 objects)** are **imported in their encoded NVFP4
     form, bit-exact** (`import_encoded`): the packed FP4 weights, block scales and activation scales
     are carried over unchanged, with no re-quantisation round-trip. At runtime these drive the
     engines' W4A4 NVFP4 kernels.
   - **Self-attention and linear-attention (GDN) projections (128 FP8 objects)** are likewise
     **imported in their encoded per-row FP8 form, bit-exact**.
   - **Output head** — stored as NVFP4 in the source; the engines register the vocabulary
     projection only for Q8/Q6/FP8, so it is dequantised and re-quantised to row-scale FP8
     (`fp8_row_maxabs`), as in the NVIDIA artifact.
   - **Token embedding** is quantised from its source BF16 to row-scale FP8.
   - **GDN a/b projections**, norms, convolutions, `dt_bias`/`a_log` and the other non-projection
     tensors keep their source BF16/FP32 formats.
   - **Vision tower and MTP companion** are taken from the source's BF16 tensors and stored with the
     converter's grouped formats (the vision tower in mixed BF16 + Q4/Q5/Q6/Q8; the MTP companion in
     BF16 + Q8).
2. **DFlash2 draft** — [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)
   (Apache 2.0), **copied byte for byte** from my
   [Qwen3.8-27B-NVIDIA-NVFP4-NInferV3](https://huggingface.co/Wallawalla47/Qwen3.8-27B-NVIDIA-NVFP4-NInferV3)
   artifact: all 91 of its bindings point to byte-identical objects, and its component configuration
   and uses are identical. It was trained for Qwen3.8-27B, not retrained for Swift's weights.
3. **Indexed proposal head** — 131,072 rows gathered from the output head, ranked by a frequency
   corpus, stored as grouped Q4 plus an INT32 token-id table; this is what `--lm-head-draft` serves
   copy-heavy proposals from.
4. **Resources** — the six tokenizer / chat-template / generation-config / preprocessor files are
   byte-identical to UkisAI's (which are Qwen's, plus UkisAI's `min_p`/`repetition_penalty` defaults
   in `generation_config.json`).

Every object has the same format, layout and shape as in the NVIDIA artifact, and the component
configurations, bindings and uses are identical; only the weights' values (UkisAI's fine-tune), the
model name and the provenance differ. No checkpoint-level byte comparison of the imported weights
was run for this artifact (the Flash-Next tooling has one, the 27B does not); the smoke tests below
are the evidence that the weights load and compute correctly.

## How to run

Requires an NVIDIA RTX 50-series GPU (built and tuned on an RTX 5090; the weights take ~21 GB of
VRAM). Prebuilt Windows executables of Infernix are on its
[releases page](https://github.com/Wallawalla47/Infernix/releases/latest) and need an NVIDIA driver
of 580 or later; on Linux, build from source
([quick start](https://github.com/Wallawalla47/Infernix#quick-start-linux)).

```bat
infernix-serve.exe Swift-Qwen3.8-27B-NVFP4-NInfer.ninfer --host 127.0.0.1 --port 8080 --max-context 240000 --max-concurrency 2 --spec dflash2 --draft-tokens 7 --lm-head-draft --ngram-draft-tokens 15 --ngram-min-match 12 --kv-dtype int8 --preserve-thinking --host-context-mib 52000 --pending-timeout-ms 900000 --prefill-chunk 4096 --kv-capacity auto --vram-headroom-mib 0 --log-colours on --ngram-archive-mib 2048 --ngram-session-mib 256 --ngram-native-sessions --request-log-jsonl log.json --default-thinking-budget 16384 --thinking-budget-message "Considering the limited time available to the user, I must stop thinking now. Time to act:" --tolerant-tool-calls
```

It runs DFlash2 speculative decoding with 7 draft tokens, the indexed proposal head
(`--lm-head-draft`) and n-gram copy drafting, int8 KV, and the hybrid prefix cache with a
52,000 MiB tier in system RAM (`--host-context-mib`). `--kv-capacity auto` sizes the KV pool to the
free VRAM. Lower `--host-context-mib` on a machine with less system RAM, and on a single 32 GB GPU
stop any other resident model first. Add `--prefix-cache-file PATH` to keep the prefix cache across
restarts. `infernix-serve.exe --help` lists every option.

- **Images and video:** add `--vision`. `--vision-offload on` keeps the vision tower in system RAM
  instead of VRAM, which frees about 0.8 GiB of VRAM for the KV cache.
- **Longer context:** beyond the native 262,144 tokens with `--rope-yarn-factor F` (F from 1 to 4,
  up to 1M tokens).

**With NInfer:** the artifact also runs on [NInfer](https://github.com/Neroued/ninfer) and on
[ninfer-custom](https://github.com/Wallawalla47/ninfer-custom) with `ninfer-serve` and the same
command; drop any option that engine's `ninfer-serve --help` does not list.

## Tested

On 2026-10-11 on an RTX 5090 (Windows), at 32K context with DFlash2 (7 drafts), the proposal head and
`--vision`, the artifact started and answered correctly on three engines:

| Engine | Build | KV | Checks |
|---|---|---|---|
| Infernix | `windows-2026.10.09` release | K4V2 | Infernix's serving contract check (`tools.smoke.serve_contract`: streamed Chat Completions, Responses with stored state and continuation, Anthropic Messages, an image request through the vision frontend) |
| NInfer (upstream) | [Neroued/ninfer](https://github.com/Neroued/ninfer) master at `81c8ce09` plus the Windows build commit of ninfer-custom's `upstream-Windows-Port` branch (`c36fcbd4`) | INT8 | chat, streamed chat, Anthropic Messages, an image request |
| ninfer-custom | `windows-2026.10.08-5` | K4V2 | chat, Anthropic Messages, an image request |

On every engine the greedy answers were correct: "Paris", "391" for 17 × 23, a working iterative
Fibonacci function, and "3" r's in "strawberry" with thinking on (93-134 output tokens in all).
DFlash2 accepted 85.7 % of drafts on the code reply and 46-57 % on the thinking reply, so the
draft model trained for Qwen3.8-27B still drafts usefully for Swift's weights; the code reply decoded
at 270-480 tokens per second.

These are smoke tests, not benchmarks: I have not measured this artifact's speed or quality against
the other artifacts.

## License

The Swift Contribution (UkisAI's fine-tuned weights) is licensed under the
**[Swift Open License v1.0](LICENSE)**: personal, research, educational, evaluation and commercial use
are free for individuals and organizations with gross annual revenue (including affiliates) below
US$1,000,000; above that threshold, commercial use needs a separate Swift Enterprise License from
[UkisAI](https://ukisai.com/contact). The underlying
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) and the
[DFlash2 draft model](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) are licensed under the
[Apache License 2.0](LICENSE-APACHE-2.0). [`NOTICE`](NOTICE) carries UkisAI's attribution notice and
a description of the changes made in this conversion.

Copyright 2026 UkisAI. Swift Contribution licensed under the Swift Open License v1.0. Derivative of
Qwen3.8-27B, Copyright 2026 Alibaba Cloud, Apache License 2.0.
