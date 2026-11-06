---
library_name: infernix
pipeline_tag: image-text-to-text
inference: false
license: other
license_name: nvidia-open-model-agreement
license_link: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/
base_model:
  - edp1096/Huihui-Qwen3.8-Flash-Next-abliterated-FP4-Local
base_model_relation: quantized
tags:
  - infernix
  - qwen3.8
  - qwen4_exp
  - nvfp4
  - mxfp8
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

# Huihui Qwen3.8-Flash-Next abliterated FP4-Local for Infernix

> **AI disclaimer:** this artifact, its conversion recipe, the engine that runs it and most of this
> card were made with AI (mostly Claude Opus 5.5). It is hobby work and is likely to be neither
> complete nor entirely accurate.

> **Uncensored model.** Huihui's abliteration removed the base model's refusal behaviour. It will
> answer requests that Qwen's and NVIDIA's releases decline, and its output is not filtered. You
> are responsible for how you use it and for what it produces.

> [!IMPORTANT]
> **You also need a PLE n-gram volume, which is not in this repository.** The recommended one is
> NVIDIA FP4-Local's own 29 GB NVFP4 volume,
> **[`Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix.ngram`](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix/blob/main/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix.ngram)**
> from [Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix):
> this checkpoint keeps that table byte for byte. The other Flash-Next volumes (FP8, BF16, IQ4_NL)
> also run it. Put the volume on an NVMe drive and pass it with `--ngram-volume`:
>
> ```text
> hf download Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix.ngram --local-dir <nvme dir>
> ```
>
> If you already run NVIDIA FP4-Local with Infernix, use the volume you have. See
> [PLE n-gram tables](#ple-n-gram-tables); [`NGRAM-VOLUME.md`](NGRAM-VOLUME.md) in this repository
> repeats these links.

edp1096's [Huihui-Qwen3.8-Flash-Next-abliterated-FP4-Local](https://huggingface.co/edp1096/Huihui-Qwen3.8-Flash-Next-abliterated-FP4-Local)
(revision `d1ec8b01`) — NVIDIA's [Qwen3.8-Flash-Next-FP4-Local](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local)
with the changes of Huihui's [abliterated v2](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-v2-GGUF)
transferred into it — packed for [Infernix](https://github.com/Wallawalla47/Infernix), a C++/CUDA
engine that runs it on **one RTX 5090** (32 GB). 48 layers, 512 routed experts per layer (top 10),
an MTP drafter and vision. The 24,576 routed experts live in pinned host RAM, a VRAM cache holds
the hot ones, the CPU computes part of each layer's misses, and the 29 GB PLE n-gram table is read
from an NVMe drive.

The conversion is bit-exact: every main-model weight is kept as edp1096's checkpoint stores it.

## How it differs from NVIDIA FP4-Local

edp1096 took NVIDIA's checkpoint and changed **96 projections** (the attention or GDN output
projection and the shared expert's down projection of each of the 48 layers): each was decoded,
moved by the difference between Huihui's abliterated Q8_0 GGUF and Unsloth's Q8_0 GGUF of the
original model, rounded to BF16 and encoded again as MXFP8, the format NVIDIA stores them in.
Everything else — the routed experts and their scales, the other projections, the norms, the
n-gram table, the MTP drafter, vision, tokenizer and config — is NVIDIA's checkpoint byte for byte
(the source's own manifest lists 19 of its 21 shards as identical to NVIDIA's, and so do Hugging
Face's file hashes).

So it runs on Infernix exactly as NVIDIA FP4-Local does: same recipe, formats, kernels and options.
This artifact differs from
[Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix)
in those 96 weight objects (859,668,480 bytes) and its artifact id; every other object, the vision
tower, the quantized MTP drafter, the proposal head and the screened heads included, is the same byte
for byte. (Its `tokenizer_config.json` also lacks the unused copy of the chat template that NVIDIA's
file carries: Infernix reads the template from `chat_template.jinja`, the same in both artifacts.)
Points to know:

- **Activation scales.** The routed experts keep NVIDIA's activation scales, calibrated on the
  original model; edp1096 did not recalibrate. They matter only with `--expert-a4-activations`
  (4-bit expert activations); the default 16-bit activations do not use them.
- **Drafters.** The MTP drafter is NVIDIA's (the abliteration did not change it), and the DFlash2
  drafter was trained on the original model. Drafts only change speed, never the output, since the
  full model verifies every draft.
- **Source qualification.** edp1096 records the transfer as *completed with findings*
  (`runtime_validated: false`): on DGX Spark the checkpoint passed retrieval at 1,048,056 tokens,
  code execution, structured JSON, tool calls and an image test, and failed one check (a correct
  answer that ignored an integer-only instruction). No new QAD training or calibration was done.

## Files

| File | Bytes | What it is |
|---|---:|---|
| `Huihui-Qwen3.8-Flash-Next-abliterated-FP4-Local-Infernix-00001-of-00003.infernix` | 32,000,000,000 | model, part 1 of 3 (also holds the table of contents) |
| `Huihui-Qwen3.8-Flash-Next-abliterated-FP4-Local-Infernix-00002-of-00003.infernix` | 32,000,000,000 | model, part 2 of 3 |
| `Huihui-Qwen3.8-Flash-Next-abliterated-FP4-Local-Infernix-00003-of-00003.infernix` | 13,375,558,912 | model, part 3 of 3 |
| `Huihui-Qwen3.8-Flash-Next-abliterated-FP4-Local-Infernix-00001-of-00003.infernix.conversion.json` | 945,710 | conversion report: sources, methods and formats per object |
| `NGRAM-VOLUME.md` | | where to download an n-gram volume (none is in this repository) |
| `LICENSE`, `NOTICE` | | the Qwen Community License 1.0; attribution and the NVIDIA agreement |

Keep the three model parts in one directory and give Infernix part 1, as with a split GGUF.

## DFlash2 drafting

This model can draft with **[Qwen3.8-Flash-Next-DFlash2-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-DFlash2-Infernix)**
(776 MB), Inco AI's [DFlash2 drafter](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2) packed
for Infernix. On NVIDIA FP4-Local it decodes 8-11 % faster than the model's own MTP drafter with
`--draft-tokens auto` (26 % on sampled code). Download it next to the model and pass it with
`--spec dflash2 --draft-model`:

```text
hf download Wallawalla47/Qwen3.8-Flash-Next-DFlash2-Infernix Qwen3.8-Flash-Next-DFlash2-Infernix.infernix --local-dir <dir>
infernix-serve <this model's part 1>.infernix --spec dflash2 \
  --draft-model <dir>/Qwen3.8-Flash-Next-DFlash2-Infernix.infernix --draft-tokens auto
```

It needs an Infernix build whose `--help` lists `--draft-model`; builds from 2026-10-31 also find it
in the model's folder by themselves when no `--spec` is given. See the drafter's card for details.

## PLE n-gram tables

Qwen3.8-Flash-Next's PLE n-gram table (320,001,536 rows) is not in the model files: it is a
separate volume, read in random 4 KiB blocks from an NVMe drive and passed with `--ngram-volume`.
Four volumes of the table exist, one per row format, none of them in this repository, and any of
them runs any Qwen3.8-Flash-Next model. **For this model the NVFP4 volume of NVIDIA FP4-Local is the
recommended one:** this checkpoint stores NVIDIA FP4-Local's table unchanged, and the artifact was
converted with that volume and verified against it row for row.

| Table | File | Bytes | Download from |
|---|---|---:|---|
| **NVFP4**, NVIDIA FP4-Local's rows, **recommended** | `Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix.ngram` | 29,127,258,112 | [Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix/blob/main/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix.ngram) |
| FP8, one scale for the table (NVIDIA's NVFP4 checkpoint) | `Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram` | 52,429,058,048 | [Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix/blob/main/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram), or the identical copy in [Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix/blob/main/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram) |
| BF16, unquantized (Qwen's) | `Qwen3.8-Flash-Next-BF16.ngram` | 109,227,196,416 | [Qwen3.8-Flash-Next-FP8-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-FP8-Infernix/blob/main/Qwen3.8-Flash-Next-BF16.ngram) |
| IQ4_NL, 4.5 bits per value (ISTA-DASLab's GSQ-RCO GGUF) | `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix.ngram` | 29,127,258,112 | [Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix/blob/main/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix.ngram) |

```text
hf download Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix.ngram --local-dir <nvme dir>
hf download Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram --local-dir <nvme dir>
hf download Wallawalla47/Qwen3.8-Flash-Next-FP8-Infernix Qwen3.8-Flash-Next-BF16.ngram --local-dir <nvme dir>
hf download Wallawalla47/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix.ngram --local-dir <nvme dir>
```

One copy of a volume serves every Flash-Next model. The FP8, BF16 and IQ4_NL tables hold the same
table at other precisions (NVIDIA's NVFP4 checkpoint, Qwen's original and ISTA-DASLab's GGUF); none
of them has been measured on this model.

## What is bit-exact and what is converted

Converted with Infernix's recipe `qwen3_8_flash_next_fp4_local` (components text, vision and MTP,
plus a proposal head and `--screen-heads`), as NVIDIA FP4-Local's artifact.
`python -m tools.flash_next.verify_artifact` compared the artifact with edp1096's checkpoint:
24,576 experts, 1,616 tensors and all 320,001,536 rows of NVIDIA FP4-Local's n-gram volume with no
mismatch and no tensor rounded; only the MTP drafter's 1,556 tensors are re-quantized.

- **Routed experts:** NVIDIA's NVFP4 codes, block scales (the checkpoint stores them in a lossless
  compressed encoding, which the converter restores), FP32 multipliers and activation scales. By
  default they run with BF16 activations, as NVIDIA serves the checkpoint;
  `--expert-a4-activations` runs them with 4-bit activations at NVIDIA's calibrated scales, which
  changes the output (see above).
- **GDN, attention (indexer included) and shared-expert projections:** the checkpoint's MXFP8 words,
  edp1096's re-encoded ones included, run with BF16 activations; the GDN a/b projections and the
  vision attention are their exact BF16 values; the trained FP32 norms stay FP32.
- **Vision MLPs and merger:** NVIDIA's NVFP4 codes, block scales and FP32 multipliers, run with BF16
  activations and exact sums (the experts' default arithmetic).
- **N-gram table:** bound to NVIDIA FP4-Local's NVFP4 volume, whose every row equals this
  checkpoint's table byte for byte.
- **Converted:** only the MTP drafter (1,556 tensors, re-quantized as for the other imports).
- **Added:** the MTP drafter's proposal head (`--lm-head-draft`) and 2-bit copies of the LM head and
  the proposal head (273 MB, the artifact's last objects) for Infernix's screened heads.
- **Chat template:** Infernix's maintained Qwen3.8 template
  ([`qwen3_8.jinja`](https://github.com/Wallawalla47/Infernix/blob/main/tools/chat_templates/qwen3_8.jinja)),
  as in the NVIDIA FP4-Local artifact; the checkpoint's own template is NVIDIA's, unchanged.

## Requirements

- **GPU:** NVIDIA RTX 5090 (Blackwell, `sm_120a`), all of its 32 GB.
- **RAM:** about 70 GiB available for the experts, more for `infernix-serve` with its prefix cache.
  With less, Infernix's SSD expert tier keeps what fits in RAM and reads the rest from the artifact
  (slower).
- **Disk:** ~77 GB for the artifact, plus the 29 GB n-gram volume on an NVMe drive.
- **Engine:** [Infernix](https://github.com/Wallawalla47/Infernix) from GitHub `main` at commit
  `51bdd498` or later (the same requirement as NVIDIA FP4-Local's artifact).

## Run

```text
infernix-serve Huihui-Qwen3.8-Flash-Next-abliterated-FP4-Local-Infernix-00001-of-00003.infernix \
  --ngram-volume <nvme dir>/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix.ngram --kv-dtype int8 \
  --max-context 262144 --max-concurrency 2 \
  --spec dflash2 --draft-model <dir>/Qwen3.8-Flash-Next-DFlash2-Infernix.infernix --draft-tokens auto --lm-head-draft \
  --vram-past-budget
```

`--vram-past-budget` (Windows) also gives the expert cache the ~0.6 GiB of VRAM the per-process
budget withholds; leave it out when another program needs that memory. Infernix builds from
2026-10-31 speculate even without `--spec` (`--spec auto`): with DFlash2 when its drafter is the only
one in the model's folder, else with MTP. `infernix` takes the same arguments. It serves the OpenAI
Chat Completions and Responses APIs and the Anthropic Messages API. See the
[Qwen3.8-Flash-Next guide](https://github.com/Wallawalla47/Infernix/blob/main/docs/qwen3_8-flash-next.md#nvidia-fp4-local-checkpoint)
for the options, KV formats and requirements.

## Quality

On the 1M-token perplexity corpus (quick: one stream per domain, 4K context, BF16 KV, reproducible
route), against NVIDIA FP4-Local on the same Infernix build (2026-11-01, `main` `4ea9cd0a`):

| Domain | This artifact | NVIDIA FP4-Local |
|---|---:|---:|
| English reference | 3.1271 | 3.1088 |
| English long-form | 6.5605 | 6.5579 |
| Chinese reference | 4.3463 | 4.3364 |
| Code | 1.6600 | 1.6602 |
| **All** | **3.4939** | **3.4865** |

Perplexity is 0.21 % higher overall (the most on English reference text, +0.59 %; code is level).
Each next-token distribution sits at a mean KL divergence of 0.018 from NVIDIA FP4-Local's, and the
two pick the same top token at 95.2 % of positions. These figures measure closeness to NVIDIA's model on ordinary text, not the effect on
refusals: with the same greedy settings NVIDIA FP4-Local declined two test requests (how to hotwire
a car, an insulting roast) and this artifact answered both.

## Performance

This artifact runs the same kernels and fast paths as NVIDIA FP4-Local's (the server reports the
same fast-path configuration), so
[that card's figures](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix#quality-and-speed-rtx-5090-2026-10-20)
are a guide. A short interleaved check of decode speed (2026-11-01; `infernix-serve` of the
`windows-2026.11.01` release; RTX 5090 in a PCIe Gen5 x8 link, Core i9-13900K, 96 GB DDR5, Windows
11; INT8 KV, every expert in RAM, greedy answers of up to 256 tokens to four prompts; two servers per
artifact in A-B-A-B order, each with a warm-up pass and three measured passes; mean decode tok/s):

| Prompt | Same answer? | DFlash2, NVIDIA FP4-Local | DFlash2, this artifact | Plain, NVIDIA FP4-Local | Plain, this artifact |
|---|---|---:|---:|---:|---:|
| Explain mixture-of-experts | no | 172.2 | 167.9 (−2.5 %) | 126.8 | 120.1 (−5.3 %) |
| Fibonacci function | yes | 96.3 | 95.4 (−0.9 %) | 76.0 | 75.6 (−0.5 %) |
| Hotwiring a car | no (NVIDIA's declines) | 171.3 | 157.8 (−7.9 %) | 123.6 | 117.4 (−5.0 %) |
| Insulting roast | no (NVIDIA's declines) | 166.8 | 156.9 (−5.9 %) | 124.2 | 121.4 (−2.2 %) |

DFlash2 is `--spec dflash2 --draft-tokens auto --lm-head-draft` with
[the DFlash2 drafter](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-DFlash2-Infernix); plain is
`--spec off`. Where both models write the same answer, the speed is the same within the run-to-run
spread (each artifact's six measurements overlap). Where the answers differ, this artifact decoded
2-8 % slower on these prompts, with and without drafting, so the drafters are not the only cause;
which experts the different text routes to is the likely one, but four prompts cannot separate the
model from the text. Prefill was not measured separately.

## Provenance

Converted with `tools.convert` (recipe `qwen3_8_flash_next_fp4_local`, `--proposal --screen-heads`,
`--resource chat_template.jinja=tools/chat_templates/qwen3_8.jinja`, `--ngram-reuse` on NVIDIA
FP4-Local's volume) from edp1096's revision `d1ec8b01dd859336d7905db2ad2ceb725146008c`;
`artifact_id` `dc20f5a9951e46408ca240ff5d431fff`. edp1096's `transfer-manifest.json` and `source-audit.json` in the source
repository record the transfer's sources, changed tensors and errors. The conversion report and the
artifact's own provenance record the paths of the machine that converted it.

## License

edp1096 publishes the source checkpoint under the
[NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/),
NVIDIA FP4-Local's license. The Qwen3.8-Flash-Next base model, and Huihui's abliterated v2 derived
from it, are governed by the Qwen Community License 1.0 (copy in `LICENSE`), whose conditions
(among them a separate license from Qwen for some commercial uses) may apply to derivatives as
well; check both before use. The Infernix engine has its own license in its repository.
