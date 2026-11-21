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

# Qwen3.8-Flash-Next NVIDIA NVFP4-QAD for Infernix

> **AI disclaimer:** this artifact, its conversion recipe, the engine that runs it and most of this
> card were made with AI (mostly Claude Opus 5.5). It is hobby work and is likely to be neither
> complete nor entirely accurate.

> [!IMPORTANT]
> **You also need a PLE n-gram volume, which is not in this repository.** The recommended one is
> the 52 GB FP8 volume
> **[`Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram`](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix/blob/main/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram)**
> from [Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix)
> (an identical copy is in the Dense8 repository). NVIDIA kept its NVFP4 release's FP8 n-gram table frozen through
> QAD, and that volume holds it byte for byte. Put the volume on an NVMe drive and pass it with
> `--ngram-volume`:
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
> text/layers/0/moe/shared/gate" (checked with a build of `main` at `49893d9c`, 120 commits after `windows-2026.11.07`).

NVIDIA's [Qwen3.8-Flash-Next-NVFP4-QAD](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4-QAD)
(revision `a874339c`), a quantization-aware distillation of Qwen3.8-Flash-Next with a more aggressive
recipe than NVIDIA's NVFP4 release, packed for [Infernix](https://github.com/Wallawalla47/Infernix), a
C++/CUDA engine that runs it on **one RTX 5090** (32 GB). The model has 48 layers, 512 routed experts
per layer (top 10), an NVFP4 shared expert per layer, an MTP drafter and vision. The 24,576 routed
experts (63.3 GiB) are split between a VRAM cache, which holds the hot ones, and pinned host RAM, which
holds the rest. The CPU computes part of each layer's misses. The 52 GB PLE n-gram table is read from
an NVMe drive.

The conversion is bit-exact. NVIDIA's NVFP4 experts, MXFP8 projections, BF16 weights and FP8 n-gram
table are kept as the checkpoint stores them, and the experts run in the 4-bit arithmetic NVIDIA
serves them with.

> **Dense8 alternative:**
> [Qwen3.8-Flash-Next-NVIDIA-NVFP4-QAD-Dense8-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-QAD-Dense8-Infernix)
> stores the classes this checkpoint keeps in BF16 (hyper-connection mixers, PLE projections, final
> mixer and `lm_head`) in 8 bits and leaves everything else the same. It measured level on quality
> (perplexity 3.4385 against this artifact's 3.4422) and decoded 1.8 % and 7.4 % faster in two runs
> (254.0 and 269.4 against 249.4 and 250.8 tok/s). Choose this artifact when every weight must be
> NVIDIA's unchanged.

## How it differs from NVIDIA's NVFP4 release

According to NVIDIA's card, the checkpoint comes from 100 optimizer updates of quantization-aware
distillation from a frozen BF16 teacher. It also quantizes more of the model than
[Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4):

| Component | NVIDIA NVFP4 | NVIDIA NVFP4-QAD (this artifact) |
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
| `Qwen3.8-Flash-Next-NVIDIA-NVFP4-QAD-Infernix-00001-of-00003.infernix` | 32,000,000,000 | model, part 1 of 3 (also holds the table of contents) |
| `Qwen3.8-Flash-Next-NVIDIA-NVFP4-QAD-Infernix-00002-of-00003.infernix` | 32,000,000,000 | model, part 2 of 3 |
| `Qwen3.8-Flash-Next-NVIDIA-NVFP4-QAD-Infernix-00003-of-00003.infernix` | 13,695,446,016 | model, part 3 of 3 |
| `Qwen3.8-Flash-Next-NVIDIA-NVFP4-QAD-Infernix-00001-of-00003.infernix.conversion.json` | 884,462 | conversion report: sources, methods and formats per object |
| `NGRAM-VOLUME.md` | | where to download an n-gram volume (none is in this repository) |
| `NVIDIA-Open-Model-License.txt`, `NOTICE`, `LICENSE` | | the NVIDIA Open Model License; attribution notice; the Qwen Community License 1.0 |

Keep the three model parts in one directory and give Infernix part 1, as with a split GGUF.

## DFlash2 drafting

This model can draft with **[Qwen3.8-Flash-Next-DFlash2-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-DFlash2-Infernix)**
(776 MB), Inco AI's [DFlash2 drafter](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2) packed
for Infernix. It was trained on the original model. On this artifact's benchmark text it decoded
264 tok/s, against 250 with the model's own MTP drafter. The two figures come from separate sessions,
see [Performance](#performance). Download it next to the model and pass it with
`--spec dflash2 --draft-model`:

```text
hf download Wallawalla47/Qwen3.8-Flash-Next-DFlash2-Infernix Qwen3.8-Flash-Next-DFlash2-Infernix.infernix --local-dir <dir>
```

## PLE n-gram tables

Qwen3.8-Flash-Next's PLE n-gram table (320,001,536 rows) is not in the model files. It is a separate
volume, read in random 4 KiB blocks from an NVMe drive and passed with `--ngram-volume`. Five volumes
of the table exist, one per row format, none of them in this repository, and any of them runs any
Qwen3.8-Flash-Next model.

**The FP8 volume is the recommended one.** This checkpoint stores that table unchanged. The artifact
was converted with that volume and verified against it row for row, and every figure on this card was
measured with it. Its SHA-256 equals the Hub file's.

| Table | File | Bytes | Download from |
|---|---|---:|---|
| **FP8**, one scale for the table (NVIDIA's), **recommended** | `Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram` | 52,429,058,048 | [Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix/blob/main/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram), or the identical copy in [Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix/blob/main/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram) |
| BF16, unquantized (Qwen's) | `Qwen3.8-Flash-Next-BF16.ngram` | 109,227,196,416 | [Qwen3.8-Flash-Next-FP8-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-FP8-Infernix/blob/main/Qwen3.8-Flash-Next-BF16.ngram) |
| BF16 P12, the BF16 values bit for bit in 12.8 bits each | `Qwen3.8-Flash-Next-BF16-P12.ngram` | 81,920,397,312 | [Qwen3.8-Flash-Next-FP8-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-FP8-Infernix/blob/main/Qwen3.8-Flash-Next-BF16-P12.ngram) |
| NVFP4, 4.5 bits per value (NVIDIA FP4-Local's) | `Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix.ngram` | 29,127,258,112 | [Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix/blob/main/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix.ngram) |
| IQ4_NL, 4.5 bits per value (ISTA-DASLab's GSQ-RCO GGUF) | `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix.ngram` | 29,127,258,112 | [Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix/blob/main/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix.ngram) |

```text
hf download Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram --local-dir <nvme dir>
hf download Wallawalla47/Qwen3.8-Flash-Next-FP8-Infernix Qwen3.8-Flash-Next-BF16.ngram --local-dir <nvme dir>
hf download Wallawalla47/Qwen3.8-Flash-Next-FP8-Infernix Qwen3.8-Flash-Next-BF16-P12.ngram --local-dir <nvme dir>
hf download Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix.ngram --local-dir <nvme dir>
hf download Wallawalla47/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix.ngram --local-dir <nvme dir>
```

One copy of a volume serves every Flash-Next model. The other four hold the same table at other
precisions: Qwen's original (as BF16, or packed losslessly as BF16 P12, which gives the BF16
volume's output in three quarters of the space and needs Infernix `main` at `fed37c0e` or later),
NVIDIA FP4-Local's and ISTA-DASLab's GGUF. None of them has been measured on this model.

## What is bit-exact and what is converted

Converted with Infernix's recipe `qwen3_8_flash_next_nvfp4_qad` (components text, vision and MTP, plus
a proposal head and `--screen-heads`). `python -m tools.flash_next.verify_artifact` compared the
artifact with the checkpoint and found no mismatch in 24,624 expert records (the 24,576 routed experts
and the 48 shared ones), 1,472 tensors or any of the 320,001,536 n-gram volume rows. Only the MTP
drafter's 1,556 tensors are re-quantized.

- **Routed and shared experts:** NVIDIA's NVFP4 codes (E2M1), E4M3 block scales per 16 weights, FP32
  multipliers and FP32 activation input scales, repacked into Infernix's expert layout without changing a
  bit. They run **W4A4**, with activations quantized to NVFP4 at NVIDIA's scales, as NVIDIA serves the
  checkpoint. Decode and verification use exact integer sums. Wide prefill calls use the FP4 tensor
  cores with FP32 accumulation.
- **GDN, attention and indexer projections:** NVIDIA's MXFP8 words (E4M3 codes, one power-of-two scale
  per 32 weights), run with BF16 activations (W8A16). The GDN a/b projections become their exact BF16
  values.
- **Everything else in the main model and vision tower:** NVIDIA's stored values (BF16, with the
  trained norms in FP32). This includes the hyper-connection mixers, PLE projections, routers, the token
  embedding and `lm_head`.
- **N-gram table:** bound to NVIDIA's FP8 volume, whose every row equals this checkpoint's table byte
  for byte.
- **Converted (drafting only):** the MTP drafter (1,556 tensors). Its projections are `q8_g32_fp16`;
  its routed experts, block-scaled FP8 in the checkpoint, are `q4_g64_fp16` from their exact values.
- **Added:** the MTP drafter's proposal head (`--lm-head-draft`), plus 2-bit copies of the LM head and
  the proposal head for Infernix's screened heads. These are the artifact's last objects, and nothing
  else reads them.
- **Chat template:** Infernix's maintained Qwen3.8 template
  ([`qwen3_8.jinja`](https://github.com/Wallawalla47/Infernix/blob/main/tools/chat_templates/qwen3_8.jinja)).

### Expert activation precision

Three expert arithmetics are available. The default is the one NVIDIA trained and serves.

| Option | Expert arithmetic | Perplexity (quick corpus) | MTP decode on the benchmark |
|---|---|---:|---:|
| default | W4A4: 4-bit activations at NVIDIA's scales | 3.4422 | 250.8 / 249.4 tok/s |
| `--a16-activations` | W4A16: BF16 activations, exact sums | 3.4138 | 248.4 / 249.0 tok/s |
| `--a8-activations` | 8-bit activations (opt-in, changes the output) | 3.4139 (default route, see below) | 253.2 / 250.6 tok/s |

`--a16-activations` measured 0.8 % lower perplexity at decode within 1 % of the default. `--a8-activations` runs on
the default route only. It scored 3.4139 there, with its 8-bit routes and the `expert-a4-residual` fast path that
its 16-bit-bank configuration registers; the default arithmetic scores the same 3.4422 on the default route as on
the reproducible route. Under `--a16-activations` the
`expert-a4-residual` fast path takes long prefills, as on NVIDIA FP4-Local; `--reproducible` turns it
off. The shared experts follow the routed experts' choice in every mode.

## Requirements

- **GPU:** NVIDIA RTX 5090 (Blackwell, `sm_120a`), all of its 32 GB.
- **RAM:**
  - **Experts:** at the recommended settings below (262K context, two requests), Infernix planned
    64.4 GiB of RAM for the experts and 2.8 GiB for everything else.
  - **Prefix cache:** gets what remains above 4 GiB free; that was 9.8 GiB on a 96 GB machine.
  - **With less RAM:** Infernix's SSD expert tier keeps what fits in RAM and reads the rest from the
    artifact. It is slower, with the same output.
- **Disk:** 77.7 GB for the artifact, plus the 52 GB n-gram volume on an NVMe drive.
- **Engine:** [Infernix](https://github.com/Wallawalla47/Infernix) from GitHub `main` at commit
  `eac2b3d8` or later (Windows or Linux), or a Windows release after `windows-2026.11.07`.

## Run

```text
infernix-serve Qwen3.8-Flash-Next-NVIDIA-NVFP4-QAD-Infernix-00001-of-00003.infernix \
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
- **Decode:** each request alone at 176-200 tok/s, the pair at about 118 tok/s each.
- **8,100-token prompt:** first token in 2.0 s.

This was a functional check, not a benchmark.

It serves the OpenAI Chat Completions and Responses APIs and the Anthropic Messages API. See the
[Qwen3.8-Flash-Next guide](https://github.com/Wallawalla47/Infernix/blob/main/docs/qwen3_8-flash-next.md#nvidia-nvfp4-qad-checkpoint)
for the options, KV formats and the SSD expert tier.

## Quality

Perplexity on Infernix's 1M-token perplexity corpus. The quick run uses one stream per domain, 4K
context, BF16 KV and the reproducible route, on Infernix builds of 2026-11-17 (this artifact and QAD Dense8 gave
the same figures on both). NVIDIA FP4-Local is shown with `--expert-a4-activations`, the same 4-bit expert
arithmetic:

| Domain | This artifact | This artifact, `--a16-activations` | QAD Dense8 | NVIDIA FP4-Local, `--expert-a4-activations` | NVIDIA NVFP4 Dense8 |
|---|---:|---:|---:|---:|---:|
| English reference | 3.0257 | 2.9925 | 3.0242 | 3.1448 | 3.1179 |
| English long-form | 6.5225 | 6.4712 | 6.5092 | 6.5934 | 6.6153 |
| Chinese reference | 4.3080 | 4.2621 | 4.2976 | 4.3568 | 4.3840 |
| Code | 1.6401 | 1.6346 | 1.6413 | 1.6624 | 1.6663 |
| **All** | **3.4422** | **3.4138** | **3.4385** | **3.5066** | **3.5095** |

This artifact's perplexity is 1.8 % lower than FP4-Local's at the same expert arithmetic and 1.9 % lower than
NVIDIA's NVFP4 Dense8 conversion's, and lower than both in every domain. Lower perplexity on this corpus is not proof of better answers: NVIDIA's own task
evaluations found no statistically established gain from QAD.

## Performance

**Setup:**

- **Hardware:** RTX 5090 in a PCIe Gen5 **x8** link, Core i9-13900K, 96 GB DDR5, Windows 11.
- **Tool and settings:** `infernix_bench` built from the code of commit `eac2b3d8`; INT8 KV, 16K context
  (20K for the DFlash2 and plain rows), one request; 256 tokens decoded after a 4,096-token prompt.
- **Expert state:** `--expert-state off`, so no saved expert-cache state, and part of the experts were
  read from the artifact.
- **Runs:** two runs each, interleaved with NVIDIA FP4-Local's artifact, 2026-11-17.

The benchmark text is very predictable: the MTP drafter has 6.9 of 7 drafts a round accepted, and
DFlash2 7.5. Treat the speculative decode speeds as the upper end; real text decodes slower and varies
with the text.

| Measure | This artifact (W4A4) | NVIDIA FP4-Local, default (W4A16) | NVIDIA FP4-Local, `--expert-a4-activations` (W4A4) |
|---|---:|---:|---:|
| Decode, MTP auto + proposal head | 250.8 / 249.4 tok/s | 249.0 / 255.9 | 253.6 / 253.0 |
| Decode, DFlash2 auto + proposal head | 264.0 / 263.8 | | 270.3 / 269.7 |
| Decode, plain | 117.5 / 117.5 | | 126.2 / 118.6 |
| Prefill, 8,192-token prompt | 3,323 / 3,320 | 3,319 / 3,334 | 3,330 / 3,332 |
| Prefill, 16,384-token prompt | 6,628 / 6,624 | | 6,638 / 6,631 |

**Re-measured on a later build** (commit `b5242057`, 2026-11-17 evening; the same settings with both n-gram volumes
on one drive, three runs each in alternating order; each run only the decode test, as above):

| Measure | This artifact (W4A4) | NVIDIA FP4-Local, `--expert-a4-activations` (W4A4) |
|---|---:|---:|
| Decode, DFlash2 auto + proposal head | 280.7 / 280.0 / 279.8 tok/s | 288.0 / 288.7 / 289.2 |
| Decode, plain | 127.8 / 118.7 / 118.9 | 120.5 / 120.0 / 120.3 |

**Results:**

- **MTP decode:** level with FP4-Local.
- **Prefill:** 0.15-0.5 % behind FP4-Local in every pair.
- **DFlash2:** 2.3 % slower than FP4-Local on the first build and 2.9 % on the later one, in every run, with the
  same draft acceptance. This artifact's n-gram volume sat on another drive than FP4-Local's in the first
  measurement. Moving it to FP4-Local's drive changed nothing: 259.5 tok/s mean on either drive, since 99 % of the
  rows came from the host cache. Those check runs measured an 8,192-token prefill before decoding, which costs
  decode about 7 %, so 259.5 compares only with runs of that kind. So the drive is not the cause. In the same rounds this model's routing sends about
  3 % more expert misses to the CPU.
- **Plain decode:** 1.2 % slower than FP4-Local in two of three runs on the later build (118.7 and 118.9 against
  120.0-120.5). Its other run, at 127.8, took a different expert-cache course (475 landings against 671); the cause
  is not established.
- **Two requests at once (MTP):** this artifact decoded at 116 tok/s. FP4-Local's lower 85 tok/s comes
  from a lower expert-cache hit rate in that run, which is an FP4-Local matter, not something this
  artifact gains.
- **Against NVIDIA NVFP4 Dense8:** see the
  [QAD Dense8 card](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-QAD-Dense8-Infernix#performance)
  for the head-to-head of the two Dense8 conversions.

## Provenance

Converted with `tools.convert` (recipe `qwen3_8_flash_next_nvfp4_qad`, `--proposal --screen-heads`,
`--resource chat_template.jinja=tools/chat_templates/qwen3_8.jinja`, and `--ngram-reuse` on NVIDIA's FP8
volume) from NVIDIA's revision `a874339cdb171ace5f36c494b6d0d3e54fdb92be`, then renamed into numbered parts
with `tools.artifact.rename --numbered`. The `artifact_id` is `ccb368e6ffa24cf39c83f55af0aec88a`. The
conversion report and the artifact's own provenance record the paths of the machine that converted it.

## License

Use of this model is governed by the
[NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)
(copy in `NVIDIA-Open-Model-License.txt`; "Licensed by NVIDIA Corporation under the NVIDIA Open Model
License", see `NOTICE`). The underlying [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
is also subject to the [Qwen Community License 1.0](LICENSE), which among other conditions requires a
separate license from Qwen for some commercial uses. The Infernix engine has its own license in its
repository.
