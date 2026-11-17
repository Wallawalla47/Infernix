---
library_name: infernix
pipeline_tag: image-text-to-text
inference: false
license: other
license_name: nvidia-open-model-license
license_link: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/
base_model:
  - nvidia/Qwen3.8-Flash-Next-NVFP4-QAD
base_model_relation: quantized
tags:
  - infernix
  - qwen3.8
  - qwen4_exp
  - nvfp4
  - mxfp8
  - qad
  - moe
  - mtp
  - blackwell
  - rtx-5090
  - multimodal
  - conversational
  - cuda
---

# Qwen3.8-Flash-Next NVIDIA NVFP4-QAD Dense8 for Infernix

> **AI disclaimer:** this artifact, its conversion recipe, the engine that runs it and most of this
> card were made with AI (mostly Claude Opus 5.5). It is hobby work and is likely to be neither
> complete nor entirely accurate.

> [!IMPORTANT]
> **You also need a PLE n-gram volume, which is not in this repository.** The recommended one is
> the 52 GB FP8 volume
> **[`Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram`](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix/blob/main/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram)**
> from [Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix)
> (an identical copy is in the NVFP4 Dense8 repository). NVIDIA kept its NVFP4 release's FP8 n-gram
> table frozen through QAD, and that volume holds it byte for byte. Put the volume on an NVMe drive and
> pass it with `--ngram-volume`:
>
> ```text
> hf download Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram --local-dir <nvme dir>
> ```
>
> If you already run NVIDIA's NVFP4 conversions with Infernix, use the volume you have. See
> [PLE n-gram tables](#ple-n-gram-tables); [`NGRAM-VOLUME.md`](NGRAM-VOLUME.md) in this repository
> repeats these links.

> [!NOTE]
> **Engine version:** this artifact needs [Infernix](https://github.com/Wallawalla47/Infernix) built from
> GitHub `main` at commit `eac2b3d8` or later, which runs its NVFP4 shared experts, or a Windows release after
> `windows-2026.11.07`. Older builds refuse to load it: they stop at startup with "missing logical parameter
> text/layers/0/moe/shared/gate" (checked with a build of `main` at `49893d9c`, 120 commits after
> `windows-2026.11.07`, on the bit-exact QAD artifact).

NVIDIA's [Qwen3.8-Flash-Next-NVFP4-QAD](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4-QAD)
(revision `a874339c`), a quantization-aware distillation of Qwen3.8-Flash-Next with a more aggressive
recipe than NVIDIA's NVFP4 release, packed for [Infernix](https://github.com/Wallawalla47/Infernix), a
C++/CUDA engine that runs it on **one RTX 5090** (32 GB). The model has 48 layers, 512 routed experts
per layer (top 10), an NVFP4 shared expert per layer, an MTP drafter and vision. The 24,576 routed
experts (63.3 GiB) are split between a VRAM cache, which holds the hot ones, and pinned host RAM, which
holds the rest. The CPU computes part of each layer's misses. The 52 GB PLE n-gram table is read from
an NVMe drive.

This is the faster of the two Infernix conversions ("Dense8"). NVIDIA's NVFP4 experts, MXFP8
projections and FP8 n-gram table are kept bit-exact, and the classes the checkpoint keeps in BF16
(hyper-connection mixers, PLE projections, final mixer and `lm_head`) are stored in 8 bits.

> **Bit-exact alternative:**
> [Qwen3.8-Flash-Next-NVIDIA-NVFP4-QAD-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-QAD-Infernix)
> keeps every weight of the main model and vision tower exactly as NVIDIA stores it. It measured level
> on quality (perplexity 3.4422 against this artifact's 3.4385) and decoded 1.8 % and 7.4 % slower in two
> runs (249.4 and 250.8 against 254.0 and 269.4 tok/s). Choose it only when every weight must be
> NVIDIA's unchanged.

## How it differs from NVIDIA's NVFP4 release

According to NVIDIA's card, the checkpoint comes from 100 optimizer updates of quantization-aware
distillation from a frozen BF16 teacher. It also quantizes more of the model than
[Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4):

| Component | NVIDIA NVFP4 | NVIDIA NVFP4-QAD |
|---|---|---|
| Routed experts | NVFP4 | NVFP4, trained by QAD |
| Shared experts | BF16 | **NVFP4**, trained by QAD |
| GDN, attention and indexer projections | BF16 | **MXFP8** (indexer frozen) |
| PLE n-gram table, MTP drafter, vision, routers, `lm_head` | | kept from the NVFP4 release, frozen |

NVIDIA validated the checkpoint on GB300 with vLLM, with MTP disabled. It reports no statistically
established task-accuracy gain over the same recipe without QAD, and did not evaluate MTP on these
weights. In Infernix, drafting only changes speed: the full model verifies every draft.

Infernix runs the NVFP4 shared expert as one more expert record per layer, kept in VRAM, as the last
job of the routed experts' call. It uses the routed experts' 4-bit arithmetic and its own activation
scales.

## Files

| File | Bytes | What it is |
|---|---:|---|
| `Qwen3.8-Flash-Next-NVIDIA-NVFP4-QAD-Dense8-Infernix-00001-of-00003.infernix` | 32,000,000,000 | model, part 1 of 3 (also holds the table of contents) |
| `Qwen3.8-Flash-Next-NVIDIA-NVFP4-QAD-Dense8-Infernix-00002-of-00003.infernix` | 32,000,000,000 | model, part 2 of 3 |
| `Qwen3.8-Flash-Next-NVIDIA-NVFP4-QAD-Dense8-Infernix-00003-of-00003.infernix` | 12,536,650,752 | model, part 3 of 3 |
| `Qwen3.8-Flash-Next-NVIDIA-NVFP4-QAD-Dense8-Infernix-00001-of-00003.infernix.conversion.json` | 886,663 | conversion report: sources, methods and formats per object |
| `NGRAM-VOLUME.md` | | where to download an n-gram volume (none is in this repository) |
| `NVIDIA-Open-Model-License.txt`, `NOTICE`, `LICENSE` | | the NVIDIA Open Model License; attribution notice; the Qwen Community License 1.0 |

Keep the three model parts in one directory and give Infernix part 1, as with a split GGUF.

## DFlash2 drafting

This model can draft with **[Qwen3.8-Flash-Next-DFlash2-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-DFlash2-Infernix)**
(776 MB), Inco AI's [DFlash2 drafter](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2) packed
for Infernix. It was trained on the original model. On the bit-exact QAD artifact's benchmark text it
decoded 264 tok/s, against 250 with the model's own MTP drafter. The two figures come from separate
sessions, see [Performance](#performance). Download it next to the model and pass it with
`--spec dflash2 --draft-model`:

```text
hf download Wallawalla47/Qwen3.8-Flash-Next-DFlash2-Infernix Qwen3.8-Flash-Next-DFlash2-Infernix.infernix --local-dir <dir>
```

## PLE n-gram tables

Qwen3.8-Flash-Next's PLE n-gram table (320,001,536 rows) is not in the model files. It is a separate
volume, read in random 4 KiB blocks from an NVMe drive and passed with `--ngram-volume`. Four volumes
of the table exist, one per row format, none of them in this repository, and any of them runs any
Qwen3.8-Flash-Next model.

**The FP8 volume is the recommended one.** This checkpoint stores that table unchanged. The artifact
was converted with that volume, and every figure on this card was measured with it. Its SHA-256 equals
the Hub file's.

| Table | File | Bytes | Download from |
|---|---|---:|---|
| **FP8**, one scale for the table (NVIDIA's), **recommended** | `Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram` | 52,429,058,048 | [Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix/blob/main/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram), or the identical copy in [Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix/blob/main/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram) |
| BF16, unquantized (Qwen's) | `Qwen3.8-Flash-Next-BF16.ngram` | 109,227,196,416 | [Qwen3.8-Flash-Next-FP8-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-FP8-Infernix/blob/main/Qwen3.8-Flash-Next-BF16.ngram) |
| NVFP4, 4.5 bits per value (NVIDIA FP4-Local's) | `Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix.ngram` | 29,127,258,112 | [Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix/blob/main/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix.ngram) |
| IQ4_NL, 4.5 bits per value (ISTA-DASLab's GSQ-RCO GGUF) | `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix.ngram` | 29,127,258,112 | [Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix/blob/main/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix.ngram) |

```text
hf download Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram --local-dir <nvme dir>
hf download Wallawalla47/Qwen3.8-Flash-Next-FP8-Infernix Qwen3.8-Flash-Next-BF16.ngram --local-dir <nvme dir>
hf download Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix.ngram --local-dir <nvme dir>
hf download Wallawalla47/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix.ngram --local-dir <nvme dir>
```

One copy of a volume serves every Flash-Next model. The other three hold the same table at other
precisions: Qwen's original, NVIDIA FP4-Local's and ISTA-DASLab's GGUF. None of them has been measured
on this model.

## What is bit-exact and what is quantized

Converted with Infernix's recipe `qwen3_8_flash_next_nvfp4_qad_dense8` (components text, vision and
MTP, plus a proposal head and `--screen-heads`). `python -m tools.flash_next.verify_artifact` compared
the artifact with the checkpoint:

- **No mismatch:** 24,624 expert records (the 24,576 routed experts and the 48 shared ones), 1,179
  tensors and a 200,000-row sample of the n-gram volume. The bit-exact QAD artifact, bound to the same
  volume, matched all 320,001,536 rows.
- **Re-quantized:** 293 text-model parameters and the MTP drafter's 1,556.

**Bit-exact to NVIDIA's checkpoint:**

- **Routed and shared experts:** NVIDIA's NVFP4 codes (E2M1), E4M3 block scales per 16 weights, FP32
  multipliers and FP32 activation input scales, repacked into Infernix's expert layout without changing a
  bit. They run **W4A4**, with activations quantized to NVFP4 at NVIDIA's scales, as NVIDIA serves the
  checkpoint. Decode and verification use exact integer sums. Wide prefill calls use the FP4 tensor
  cores with FP32 accumulation.
- **GDN, attention and indexer projections:** NVIDIA's MXFP8 words (E4M3 codes, one power-of-two scale
  per 32 weights), run with BF16 activations (W8A16). The GDN a/b projections become their exact BF16
  values.
- **N-gram table:** bound to NVIDIA's FP8 volume, whose every row equals this checkpoint's table byte
  for byte.
- **Routers, the shared-expert gates, the token embedding, norms, convolutions and vision:** NVIDIA's
  stored values.

**Quantized from NVIDIA's BF16** (`q8_g32_fp16`: 8-bit integers, one FP16 scale per 32 weights):

- **293 tensors:** the hyper-connection mixers (six per layer), the PLE key and value projections, the
  final mixer and `lm_head`. Of the classes Infernix's NVIDIA NVFP4 Dense8 conversion stores in 8 bits,
  these are the ones this checkpoint still keeps in BF16. Its other dense projections are already
  MXFP8 or NVFP4.

**Drafting only** (never changes output, since the full model verifies every draft):

- **MTP drafter** (1,556 tensors): its projections are `q8_g32_fp16`; its routed experts, block-scaled
  FP8 in the checkpoint, are `q4_g64_fp16` from their exact values.
- **Proposal head** for `--lm-head-draft`, plus 2-bit copies of the LM head and the proposal head for
  Infernix's screened heads. These are the artifact's last objects, and nothing else reads them.

**Chat template:** Infernix's maintained Qwen3.8 template
([`qwen3_8.jinja`](https://github.com/Wallawalla47/Infernix/blob/main/tools/chat_templates/qwen3_8.jinja)).

### Expert activation precision

Three expert arithmetics are available. The default is the one NVIDIA trained and serves. The figures
below were measured on the bit-exact QAD artifact; this artifact runs the same expert kernels.

| Option | Expert arithmetic | Perplexity (quick corpus, bit-exact artifact) | MTP decode on the benchmark (bit-exact artifact) |
|---|---|---:|---:|
| default | W4A4: 4-bit activations at NVIDIA's scales | 3.4422 | 250.8 / 249.4 tok/s |
| `--a16-activations` | W4A16: BF16 activations, exact sums | 3.4138 | 248.4 / 249.0 tok/s |
| `--a8-activations` | 8-bit activations (opt-in, changes the output) | 3.4139 (default route, with the `expert-a4-residual` fast path; the default scores 3.4422 there) | 253.2 / 250.6 tok/s |

Under `--a16-activations` the `expert-a4-residual` fast path takes long prefills, as on NVIDIA
FP4-Local; `--reproducible` turns it off. The shared experts follow the routed experts' choice in every
mode.

## Requirements

- **GPU:** NVIDIA RTX 5090 (Blackwell, `sm_120a`), all of its 32 GB.
- **RAM:**
  - **Experts:** at the recommended settings below (262K context, two requests), Infernix planned
    64.4 GiB of RAM for the experts and 2.8 GiB for everything else.
  - **Prefix cache:** gets what remains above 4 GiB free; that was 9.8 GiB on a 96 GB machine.
  - **With less RAM:** Infernix's SSD expert tier keeps what fits in RAM and reads the rest from the
    artifact. It is slower, with the same output.
- **Disk:** 76.5 GB for the artifact, plus the 52 GB n-gram volume on an NVMe drive.
- **Engine:** [Infernix](https://github.com/Wallawalla47/Infernix) from GitHub `main` at commit
  `eac2b3d8` or later (Windows or Linux), or a Windows release after `windows-2026.11.07`.

## Run

```text
infernix-serve Qwen3.8-Flash-Next-NVIDIA-NVFP4-QAD-Dense8-Infernix-00001-of-00003.infernix \
  --ngram-volume <nvme dir>/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram --kv-dtype int8 \
  --max-context 262144 --max-concurrency 2 \
  --spec dflash2 --draft-model <dir>/Qwen3.8-Flash-Next-DFlash2-Infernix.infernix --draft-tokens auto --lm-head-draft \
  --vram-past-budget
```

This is the recommended configuration, Infernix's settings for every Flash-Next conversion: INT8 KV
cache, DFlash2 drafting with the proposal head, and `--draft-tokens auto` to pick the draft length.
Without the drafter, `--spec mtp --draft-tokens auto --lm-head-draft` drafts with the model's MTP
layer. On Windows, `--vram-past-budget` also gives the expert cache the ~0.6 GiB of VRAM the
per-process budget withholds; leave it out when another program needs that memory. Without `--spec`,
Infernix speculates as `--spec auto`: with DFlash2 when its drafter is the only one in the model's
folder, else with MTP. `infernix` takes the same arguments.

This exact command was checked on this artifact on 2026-11-17 (RTX 5090, 96 GB, with an upload
loading the CPU). It ran one request alone, two at once, the first again, and an 8,100-token prompt,
all greedy with thinking on:

- **Errors:** none.
- **Output:** coherent, and the repeated request gave the same output.
- **Decode:** each request alone at 200-224 tok/s, the pair at 123-131 tok/s each.
- **8,100-token prompt:** first token in 1.9 s.

This was a functional check, not a benchmark. Its VRAM held 9,407 experts in its expert cache,
against 9,102 for the bit-exact artifact, because its 8-bit dense weights take 0.8 GiB less.

It serves the OpenAI Chat Completions and Responses APIs and the Anthropic Messages API. See the
[Qwen3.8-Flash-Next guide](https://github.com/Wallawalla47/Infernix/blob/main/docs/qwen3_8-flash-next.md#nvidia-nvfp4-qad-checkpoint)
for the options, KV formats and the SSD expert tier.

## Quality

Perplexity on Infernix's 1M-token perplexity corpus. The quick run uses one stream per domain, 4K
context, BF16 KV and the reproducible route, on Infernix builds of 2026-11-17 (this artifact and the bit-exact one
gave the same figures on both). NVIDIA FP4-Local is shown with `--expert-a4-activations`, the same 4-bit expert
arithmetic:

| Domain | This artifact | QAD bit-exact | QAD bit-exact, `--a16-activations` | NVIDIA FP4-Local, `--expert-a4-activations` | NVIDIA NVFP4 Dense8 |
|---|---:|---:|---:|---:|---:|
| English reference | 3.0242 | 3.0257 | 2.9925 | 3.1448 | 3.1179 |
| English long-form | 6.5092 | 6.5225 | 6.4712 | 6.5934 | 6.6153 |
| Chinese reference | 4.2976 | 4.3080 | 4.2621 | 4.3568 | 4.3840 |
| Code | 1.6413 | 1.6401 | 1.6346 | 1.6624 | 1.6663 |
| **All** | **3.4385** | **3.4422** | **3.4138** | **3.5066** | **3.5095** |

Storing the BF16 classes in 8 bits cost no measurable quality on these texts: this artifact is within
0.1 % of the bit-exact one overall and in every domain. Its perplexity is 2.0 % lower than NVIDIA NVFP4 Dense8's,
the conversion it replaces as the recommended one, and lower in every domain. Lower perplexity on this corpus is
not proof of better answers: NVIDIA's own task evaluations found no statistically established gain from QAD.

## Performance

**Setup:**

- **Hardware:** RTX 5090 in a PCIe Gen5 **x8** link, Core i9-13900K, 96 GB DDR5, Windows 11.
- **Tool and settings:** `infernix_bench` built from the code of commit `eac2b3d8`; INT8 KV, 16K context,
  one request; 256 tokens decoded after a 4,096-token prompt.
- **Expert state:** `--expert-state off`, so no saved expert-cache state, and part of the experts were
  read from the artifact.
- **Runs:** two runs each, interleaved with the bit-exact QAD artifact and NVIDIA FP4-Local's,
  2026-11-17.

The benchmark text is very predictable: the MTP drafter has 6.9 of 7 drafts a round accepted. Treat
the speculative decode speeds as the upper end; real text decodes slower and varies with the text.

| Measure | This artifact | QAD bit-exact | NVIDIA FP4-Local, default (W4A16) | NVIDIA FP4-Local, `--expert-a4-activations` (W4A4) |
|---|---:|---:|---:|---:|
| Decode, MTP auto + proposal head | 269.4 / 254.0 tok/s | 250.8 / 249.4 | 249.0 / 255.9 | 253.6 / 253.0 |
| Prefill, 8,192-token prompt | 3,309 / 3,315 | 3,323 / 3,320 | 3,319 / 3,334 | 3,330 / 3,332 |

This artifact decoded fastest in both runs, by 1.8 % and 7.4 % over the bit-exact one, with prefill
level. DFlash2, plain decode and two concurrent requests were measured on the bit-exact artifact only:

- **DFlash2 auto:** 264.0 / 263.8 tok/s, against FP4-Local's 270.3 / 269.7.
- **Plain decode:** 117.5 / 117.5 tok/s.
- **Two requests at once (MTP):** 116 tok/s.

The [bit-exact card](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-QAD-Infernix#performance)
has those rows.

**Against NVIDIA NVFP4 Dense8**, the conversion this one replaces as Infernix's recommended Flash-Next model. Both ran
on one build (commit `b5242057`, 2026-11-17 evening) with the same settings and the same n-gram volume file, three
runs each in alternating order. Each run measured an 8,192-token prefill before the decode test, which costs decode
about 7 % against decode-only runs (the bit-exact artifact's DFlash2 decode was 259.5 tok/s this way, 280.2 without
the prefill test):

| Measure | This artifact | NVIDIA NVFP4 Dense8 |
|---|---:|---:|
| Decode, DFlash2 auto + proposal head (the recommended command) | 271.0 / 272.8 / 272.6 tok/s | 280.1 / 281.1 / 281.3 |
| Decode, MTP auto + proposal head | 244.3 / 258.8 / 258.8 | 242.8 / 242.9 / 250.8 |
| Prefill, 8,192-token prompt | 3,345 / 3,373 / 3,371 | 3,368 / 3,372 / 3,383 |
| Perplexity (quick corpus, BF16 KV, reproducible) | 3.4385 | 3.5095 |

- **DFlash2:** this artifact decoded 3.1 % slower, in every run. Its drafts were accepted slightly less often (7.53
  against 7.63 tokens a round), which accounts for part of the gap; the rest is time per round, not attributed.
- **MTP:** 3.5 % faster on average: +0.6 %, +6.5 % and +3.2 % by round, with higher MTP acceptance in two rounds
  (6.56 against 6.24-6.32 tokens a round).
- **Prefill:** level (−0.4 %).
- **Quality:** perplexity 2.0 % lower.

## Provenance

Converted with `tools.convert` (recipe `qwen3_8_flash_next_nvfp4_qad_dense8`, `--proposal --screen-heads`,
`--resource chat_template.jinja=tools/chat_templates/qwen3_8.jinja`, and `--ngram-reuse` on NVIDIA's FP8
volume) from NVIDIA's revision `a874339cdb171ace5f36c494b6d0d3e54fdb92be`, then renamed into numbered parts
with `tools.artifact.rename --numbered`. The `artifact_id` is `044ad3ee0d384c0bb24db0b490d852cb`. The
conversion report and the artifact's own provenance record the paths of the machine that converted it.

## License

Use of this model is governed by the
[NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)
(copy in `NVIDIA-Open-Model-License.txt`; "Licensed by NVIDIA Corporation under the NVIDIA Open Model
License", see `NOTICE`). The underlying [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
is also subject to the [Qwen Community License 1.0](LICENSE), which among other conditions requires a
separate license from Qwen for some commercial uses. The Infernix engine has its own license in its
repository.
