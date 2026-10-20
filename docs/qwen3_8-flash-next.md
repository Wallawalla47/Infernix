# Qwen3.8-Flash-Next (`Qwen4ExpForCausalLM`)

Infernix runs NVIDIA's NVFP4 checkpoint of Qwen3.8-Flash-Next
([nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4)) on one
RTX 5090 (32 GB), and ISTA-DASLab's GSQ-RCO IQ3_S GGUF bit-exactly
([GGUF import](#gguf-import-ista-daslab-gsq-rco-iq3_s)). The 24,576
routed experts (63 GiB) live in pinned host RAM. A VRAM expert cache holds the hot ones, and the
CPU computes part of each layer's misses. The 52 GB PLE n-gram table lives on an NVMe volume. The
design, measurements and open work are in the
[design document](maintainer/qwen3_8-flash-next-design.md) (§16.5, §19.2).

Generation works through `infernix`, `infernix-serve` and `infernix_bench`, with Inco AI's DFlash2
drafter (`--spec dflash2 --draft-model FILE`, [recommended](#run)) or the model's MTP drafter
(`--spec mtp`), n-gram copy proposals, images and video (`--vision`), and in
`infernix-serve` the hybrid prefix cache. Causal scoring (`infernix-perplexity`, the
decode-quality judge) runs teacher-forced through the generation path; `--a16-activations` scores
with 16-bit activations in every linear and every expert bank (the exact W4A16 expert
arithmetic): the [quality reference](maintainer/model-quality.md#6-numerical-acceptance-policy-and-fast-paths).
Pass the n-gram volume with `--ngram-volume` when it is not `<artifact>.ngram`, and an expert stripe
with `--expert-stripe` in the SSD tier.

## Requirements

- **GPU.** An RTX 5090 (sm_120a). A **PCIe x16** link matters: expert misses cross PCIe, and an
  x8 link halves their bandwidth. Check with
  `nvidia-smi --query-gpu=pcie.link.width.current --format=csv`.
- **RAM.** About 96 GB for full speed. A run pins ~64.5 GiB. At startup a RAM ledger plans every
  allocation and keeps `--ram-headroom-mib` (default 2048) free for the system; with the defaults
  about 69.3 GiB must be available, and 73.3 GiB for `infernix-serve` with its prefix cache (see
  [Prefix cache](#prefix-cache)). With less, the [SSD expert tier](#ssd-expert-tier) keeps the
  experts that fit in RAM and reads the others from the artifact; the ledger line names every term.
  Windows lets the GPU page-lock RAM only up to its *shared GPU memory* (Task Manager → Performance
  → GPU; often half of RAM). Experts past that limit stay in RAM unpinned and the CPU serves them
  (see [Page-locking limit](#page-locking-limit)), so the limit costs some speed, not RAM.
  An artifact whose experts exceed RAM (the FP8 conversion's 112.6 GiB) takes everything the
  ledger allows, so only about the headroom stays available while it runs: other programs page,
  and Windows may stall Infernix's own threads for a moment (rounds wait up to 10 s for its CPU
  expert service and SSD reads before they fail). A larger `--ram-headroom-mib` leaves more for
  other programs and costs the experts it keeps out of RAM: with the FP8 artifact on a 96 GB
  machine (2026-10-17, one run each), plain tg512 measured 43.8 tok/s at 2048, 38.3 at 5120 and
  35.6 at 8192; MTP 4 at 4K + 512 measured 36.3-43.5, 40.4-40.9 and 39.6-39.7 (two runs each; how
  much RAM is free at startup moves it as much as the headroom does).
- **VRAM.** All of it: the expert cache takes what the dense weights, the KV cache, the
  workspaces and a reserve for CUDA graphs leave, less a headroom for the display and other
  programs (see [VRAM](#vram)).
- **Disk.** ~80 GB for the artifact, plus a 52 GB n-gram volume on an NVMe drive.
  - The volume is read in random 4 KiB blocks: about 1.1M reads in a 128K-token prompt, a few
    per decode round.
  - A fast consumer drive is enough. On an RTX 5090 machine, a Samsung 990 PRO and an Intel Optane
    P5800X gave the same prefill (5,766 against 5,768 tok/s at 128K) and decode (111.3 tok/s).
  - Per-read latency was ~200 µs on the 990 PRO against ~60 µs on the Optane, a few milliseconds
    over a whole generation.
  - A SATA SSD or a hard disk would be far slower.

## Convert

Two recipes convert NVIDIA's checkpoint. Both import the NVFP4 routed experts (codes, block scales,
per-matrix multipliers and input scales) and the FP8 n-gram table bit-exactly, and both store the
MTP drafter in `q8_g32_fp16` with `q4_g64_fp16` routed experts: the drafter only proposes tokens
that the full model verifies, so its precision changes speed, never output.

| | Recipe A, `qwen3_8_flash_next_nvfp4` | Recipe B, `qwen3_8_flash_next_nvfp4_dense8` |
|---|---|---|
| Published as | `Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix` | `Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix` |
| Main model and vision | **bit-exact**: every tensor as NVIDIA stores it (BF16 stays BF16) | dense projections and `lm_head` (629 tensors, 97 % of the dense bytes) in `q8_g32_fp16`; everything else bit-exact |
| Model files | 80.4 GB | 76.9 GB |
| Dense weights read per token | ~8.6 GB (BF16) | ~5.1 GB |

The two measured level on quality (teacher-forced perplexity 4.666 and 4.654 on the frozen texts
below, both 0.037-0.040 nats per token better than Strata's). Recipe B leaves more VRAM for the
expert cache and reads fewer bytes per token, so it decodes faster; see
[Performance](#performance). Use recipe A where outputs must come from NVIDIA's weights unchanged.
`--proposal` adds the smaller draft head that `--lm-head-draft` uses, to either.

```text
python -m tools.convert --model <nvidia/Qwen3.8-Flash-Next-NVFP4 dir> --recipe qwen3_8_flash_next_nvfp4_dense8 \
  --components text,vision,mtp --proposal --device cpu --name Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix \
  --resource chat_template.jinja=tools/chat_templates/qwen3_8.jinja \
  --out <dir>/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix.infernix --ngram-out <nvme>/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram
python -m tools.artifact.rename <dir>/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix.infernix Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix.infernix --numbered
```

The second command names the three files as numbered parts
(`Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix-00001-of-00003.infernix` and so on); Infernix
opens part 1. `--resource` stores the maintained Qwen3.8 template, as for Qwen3.8-27B (see
[Chat template](#chat-template)); keep it in every recipe below.

- **Recipe A.** Use `--recipe qwen3_8_flash_next_nvfp4` and the name without `-Dense8`.
- **Recipe C, weight-only exports.** `--recipe qwen3_8_flash_next_nvfp4_orcarouter` converts
  [orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4)
  (published as `Qwen3.8-Flash-Next-Uncensored-NVFP4-Infernix`): its NVFP4 experts and FP8 dense
  projections, stored without activation scales, are imported bit-exactly and run with BF16
  activations in an exact integer arithmetic (W4A16 experts, W8A16 dense); its BF16 classes follow
  recipe B. Its n-gram table (BF16 in the checkpoint) quantizes to NVIDIA's FP8 table byte for
  byte, so pass `--ngram-reuse` with the NVIDIA volume. The arithmetic and its trade-offs are in
  the [design notes](maintainer/qwen3_8-flash-next-design.md) (§6.1.1, §16.2.1).
- **Verification.** `python -m tools.flash_next.verify_artifact --model <checkpoint> --artifact
  <part 1> --ngram <volume>` compares every stored tensor, expert and n-gram row with the
  checkpoint and lists the re-quantized parameters per component.
- **Second artifact of the same checkpoint.** It can share an existing n-gram volume: pass
  `--ngram-reuse <volume>` instead of `--ngram-out`. The converter checks the volume's geometry and
  512 of its rows against the checkpoint, so a volume of another checkpoint is refused.
- **Duration.** A conversion takes ~11 minutes from a warm disk cache; the MTP experts' MSE
  scale search accounts for several of them.

### GGUF import: ISTA-DASLab GSQ-RCO IQ3_S

Recipe `qwen3_8_flash_next_gsq_rco_iq3s` imports
[ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
`IQ3_S` bit-exactly into one artifact, `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix` (not published):

- Every quantized GGUF tensor is stored as its own GGUF blocks, byte for byte. That covers the
  imatrix mix of `IQ2_S`, `IQ3_XXS`, `IQ3_S`, `IQ4_XS` and `IQ4_NL` gate/up experts, `IQ4_NL` and
  `Q2_0` down experts, and `Q4_K`, `Q5_K`, `Q6_K`, `IQ4_XS` and `IQ4_NL` dense projections. The one
  `Q8_0` tensor is stored as the identical `q8_g32_fp16` words, and BF16/F32 tensors keep their
  values.
- The `Q4_K`, `Q5_K`, `Q6_K` and `IQ4_XS` dense projections keep the same blocks in tiles of 16 rows
  (`ggml_tiles16_v1`), which the decode and verification kernels read in 16-byte runs. The outputs
  are bit-identical; MTP with n-gram drafts decodes about 3.7 % faster than from GGUF row order.
- The converter undoes llama.cpp's layout transforms exactly, and checks every one: norms stored
  as 1 + g, the GDN value-head order, `ssm_a` = −exp(A_log) and the concatenated indexer
  projections.
- The imatrix only shaped the quantization, so it needs nothing at run time.
- The PLE n-gram table is the GGUF's `IQ4_NL` table, a 29.1 GB volume (27.1 GiB).
- The GGUF has no MTP head, so `--model` names a Qwen checkpoint directory that supplies it: the
  `mtp.*` tensors, `config.json` and the tokenizer files. `tools/convert/fetch_tensors.py` range-reads
  just those from
  [Qwen/Qwen3.8-Flash-Next-FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8) at a pinned
  revision (~2.7 GB instead of the whole checkpoint). The drafter is re-quantized as in recipes
  A-C, except that its routed experts are `q2_g64_fp16` instead of `q4_g64_fp16`. It only proposes
  tokens, so the outputs are unchanged, and MTP with n-gram drafts decodes about 3 % faster.
- The vision tower comes from the repository's `mmproj` (BF16; unsloth's `mmproj-BF16.gguf` is the
  same file content): `--components text,vision,mtp --mmproj <mmproj-BF16.gguf>`. Every value must
  be exact in BF16, and the 441 tensors equal the tower recipes A-C import from Qwen's checkpoint
  bit for bit (`verify_gguf_artifact --mmproj ... --vision-reference <recipe B artifact>`).

Every weight is computed with BF16 activations in an exact integer arithmetic. Each output is the
exact sum, rounded once, so the GPU's result has the same bits wherever an expert is computed and
at any batch width. The arithmetic is described in
[the design](maintainer/qwen3_8-flash-next-design.md#1622-exact-arithmetic-over-gguf-blocks-and-block-fp8-2026-10-10)
§16.2.2. llama.cpp quantizes activations to
Q8_1 instead, so its logits differ slightly from these.

```text
python -m tools.convert.fetch_tensors --repo Qwen/Qwen3.8-Flash-Next-FP8 --revision 236dfdf285828023ca3bcd3f37366c58a3469b13 \
  --out <dir>/Qwen3.8-Flash-Next-FP8-partial --pattern "mtp.*"
python -m tools.convert --model <dir>/Qwen3.8-Flash-Next-FP8-partial --gguf <IQ3_S-00001-of-00002.gguf> \
  --gguf <IQ3_S-00002-of-00002.gguf> --mmproj <mmproj-BF16.gguf> --recipe qwen3_8_flash_next_gsq_rco_iq3s \
  --components text,vision,mtp --proposal --device cpu \
  --name Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix --out <dir>/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix.infernix \
  --ngram-out <nvme>/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix.ngram
python -m tools.artifact.rename <dir>/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix.infernix Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix.infernix --numbered
python -m tools.flash_next.verify_gguf_artifact --model <dir>/Qwen3.8-Flash-Next-FP8-partial \
  --gguf <shard 1> --gguf <shard 2> --mmproj <mmproj-BF16.gguf> \
  --artifact <dir>/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix-00001-of-00002.infernix --ngram <volume> --ngram-rows all
```

The rename names the two files as numbered parts, as for recipe B; open part 1. The conversion took
11-14 minutes. The verification compares every expert record, GGUF row, tensor
and n-gram row with the GGUF (a tiled matrix after untiling) and found no difference. The artifact
is 56.8 GB, of which the routed experts are 46.9 GiB.

- **Vision** (`--vision`) runs the same tower and kernels as recipe B. A 1024 x 1024 image encoded
  in 1.9-2.0 s against recipe B's 1.4 s (1.7-1.8 s before the expert cache's size classes; the
  difference is not attributed); the IQ3_S model read the image's label and described it as
  recipe B did.
- **Experts beyond RAM.** The SSD tier serves this artifact as it serves recipe B: with
  `--expert-ram-mib 24576` (24 of 46.9 GiB in RAM) tg512 measured 107.4 tok/s and pp4096 821 tok/s,
  against 109.6 and ~2,215 with every expert in RAM. Greedy output is the same with every expert in
  RAM, with 24 GiB and with 8 GiB (400 tokens).
- **Expert cache.** The records come in five sizes (mean 1.95 MiB, largest 2.54 MiB). The cache
  keeps one frame pool per size, each with its own frames and LFRU counters, so it holds 13,008
  experts where frames of the largest size held about 10,200.

**Speed and quality** (RTX 5090 on PCIe 5.0 x8, i9-13900K, INT8 KV, one request, against recipe B
in the same session on 2026-10-16; the n-gram table in RAM, `auto`; one run of `-r 3`, or `-r 1`
at 128K):

| `infernix_bench` | GSQ-RCO IQ3_S | Recipe B |
|---|---:|---:|
| tg512, plain | 145.6 tok/s | 110.3 tok/s |
| tg512, `--spec mtp --draft-tokens 4 --lm-head-draft` (then a per-round length up to 4, now `auto`) | 259.9 tok/s (75 % of drafts accepted) | 143.1 tok/s |
| pp4096, `--prefill-chunk 4096` | 2,228 tok/s | 1,662 tok/s |
| pp131072, `--prefill-chunk 4096`, code and docs | 4,324 tok/s | 10,933 tok/s |

Earlier the IQ3_S column measured 109.6, 187.4, 2,215 and 2,836 tok/s. Between the two, the
narrow-width IQ3_S kernels, the tiled dense layout, the Q2 drafter and plain decode's divisor by
context landed, each measured on its own in the [design](maintainer/qwen3_8-flash-next-design.md) §19.3.25;
this table does not split the gain between them. Recipe B's column measured
120.5 and 176.3 tok/s for decode before; its lower decode here (84.5 % of routed experts hit the
cache in tg512) is not attributed.

- **Decode** is faster than recipe B's: the IQ3_S experts are smaller (the cache holds 13,008 of
  them, and 98.6 % of routed experts hit it in tg512), and the CPU expert engine computes part of
  the misses with the same bits as the GPU.
- **Prefill** of 4K tokens is faster: the smaller experts cross PCIe sooner, and dense and expert
  projections run as exact integer tensor-core tiles. At 128K the tiles' integer arithmetic is the
  limit against recipe B's FP4 tensor cores, so recipe B is 2.5× faster.
- **Quality.** Teacher-forced perplexity over the three frozen texts below (2,557 positions, INT8
  KV) is 4.794, against 4.653 for recipe B (+0.030 ± 0.011 nats per token, mostly on chat:
  +0.058 ± 0.019). Against Strata's UD-Q4_K_XL (4.864) it is −0.015 ± 0.012 nats, level within
  the error. The mean KL divergence from recipe B's distribution is 0.10.

### Official FP8 checkpoint

Recipe `qwen3_8_flash_next_fp8` imports
[Qwen/Qwen3.8-Flash-Next-FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8) bit-exactly into
one artifact, `Qwen3.8-Flash-Next-FP8-Infernix` (not published):

- Each routed expert's E4M3FN codes and BF16 128×128 tile scales are stored unchanged
  (`fp8_e4m3fn_b128`, a 4.69 MiB record). They are computed with BF16 activations in exact integer
  arithmetic, rounded once, so every placement (VRAM, PCIe, CPU, SSD) gives the same bits.
- Every other tensor of the checkpoint keeps its values (its dense layers are BF16).
- The n-gram table is the [BF16 n-gram table](#bf16-n-gram-table): the checkpoint's FP8 table is
  that table quantized, so `--ngram-reuse` with the BF16 volume checks that sampled rows quantize to
  the checkpoint's FP8 codes and records the volume as the artifact's own
  (`verify_artifact --ngram ... --ngram-rows all` checks every row).
- The MTP drafter is re-quantized as for the IQ3_S import: projections in `q8_g32_fp16`, routed
  experts in `q2_g64_fp16`. With `q4_g64_fp16` drafter experts instead, the outputs were identical
  (24 teacher-forced MTP segments) and decode measured the same within run-to-run spread (MTP 4 at
  4K + 512, four alternating pairs: the 2-bit drafter ahead in two by 1.0-2.0 tok/s and behind in
  two by 0.6-1.2), so the smaller drafter leaves its 0.63 GB of VRAM to the expert cache.

```text
python -m tools.convert --model <dir>/Qwen3.8-Flash-Next-FP8 --recipe qwen3_8_flash_next_fp8 \
  --components text,vision,mtp --proposal --device cpu --name Qwen3.8-Flash-Next-FP8-Infernix \
  --out <dir>/Qwen3.8-Flash-Next-FP8-Infernix.infernix --ngram-reuse <fast drive>/Qwen3.8-Flash-Next-BF16.ngram
python -m tools.artifact.rename <dir>/Qwen3.8-Flash-Next-FP8-Infernix.infernix Qwen3.8-Flash-Next-FP8-Infernix.infernix --numbered
python -m tools.flash_next.verify_artifact --model <dir>/Qwen3.8-Flash-Next-FP8 \
  --artifact <dir>/Qwen3.8-Flash-Next-FP8-Infernix-00001-of-0000N.infernix \
  --ngram <fast drive>/Qwen3.8-Flash-Next-BF16.ngram --ngram-rows all
```

The artifact is 132.7 GB in five numbered parts, of which the routed experts are 112.6 GiB: more
than a 96 GB machine can hold, so it always runs in the [SSD expert tier](#ssd-expert-tier) (on
this machine 74.1 GiB of experts page-locked plus up to 8 GiB of unpinned records, by the RAM free
at startup). The verification found every expert, all 1,615 imported tensors and all 320,001,536
n-gram rows equal to the checkpoint (2026-10-17).

Measured with default options (INT8 KV, one request, the BF16 n-gram table; 2026-10-16/17):

| `infernix_bench` | FP8 |
|---|---:|
| tg512, plain | 43.8-44.4 tok/s |
| pp4096 + tg512, `--spec mtp --draft-tokens 4 --lm-head-draft` (then a per-round length up to 4, now `auto`) (decode) | 43.5-49.3 tok/s |
| the same with `--ngram-draft-tokens 7 --ngram-min-match 4` | 53.8-54.2 tok/s |
| pp4096, `--prefill-chunk 4096` | 690 tok/s |

Decode is bound by the CPU expert engine serving about a third of routed experts (50-60 % hit the
VRAM cache), and prefill by reading ~38 GiB of experts per 4K-token prompt from one drive; the
MTP range follows the unpinned records the RAM free at startup allows (4.2 to 7.4 GiB in those
runs, rising run by run). On 2026-10-12, before the FP8 route work and the unpinned records, the
4-bit-drafter conversion measured tg512 26.4, tg512 with MTP 39.0 and pp4096 449 tok/s.

On the 1M-token perplexity corpus (quick, 4K context, INT8 KV) it scores 3.487 against recipe B's
3.508, with a mean KL divergence of 0.050 from recipe B's distribution (the exact route, which the
drafter does not change).

## Run

```text
infernix <artifact>.infernix --ngram-volume <volume>.ngram --kv-dtype int8 --max-context 16384 \
  --prompt "..."
infernix-serve <artifact>.infernix --ngram-volume <volume>.ngram --kv-dtype int8 --max-context 16384
```

- `--kv-dtype` accepts every Infernix KV format: `int8` (recommended), `bf16`, `fp8`, `nvfp4`,
  `k8v4`, `vq2` and `k4v2`.
  - `fp8`, `nvfp4` and `k8v4` measured within noise of `int8` in teacher-forced quality. `nvfp4`
    holds a context in 55 % of `int8`'s KV memory, which leaves more VRAM to the expert cache.
  - `vq2` and `k4v2` keep each sequence's recent and first tokens exact and quantize older ones
    harder (design §19.3.15).
- **Context.** The model's native context is 262,144 tokens. Long contexts work the same way, but
  their KV cache takes VRAM from the expert cache.
  - Measured on an RTX 5090 at `--max-context 262144`, INT8 KV, with a 245,760-token prompt of code
    and docs:
    - prefill ran at 8.2K tok/s (30 s);
    - decode ran at 73.7 tok/s plain and 87.9 tok/s with MTP, against ~102 and ~132 at 8K;
    - the KV took 3.2 GB, which cut the expert cache from 9,541 to 8,469 frames.
  - At that length `nvfp4` KV decoded 74.2 tok/s and `vq2` 76.9, because their smaller KV leaves
    more frames.
- **Beyond 262,144 tokens: `--rope-yarn-factor F`** (1-4). It enables YaRN on every rotation of
  the model and raises the context ceiling to 262,144 × F (1,048,576 at 4). It does not raise
  `--max-context` itself: set both. Without a factor, a `--max-context` above 262,144 is refused.
  YaRN also changes shorter contexts slightly (design §19.3.18), so use it only when you need the
  length. Retrieval holds up less far than the ceiling. A catalog test asked for the first, middle
  and last of thousands of code records (greedy, INT8 KV). At 470K tokens, all three were right at
  factor 2 and at 4. At 970K tokens and factor 4, the middle one came from a neighbouring record,
  while the engine's block selection and rotations matched their oracles at that length
  (GitHub #22).
- The prefill chunk defaults to 16,384 tokens. The exact route's output measured identical at
  chunks of 1,024, 4,096 and 12,288, and wide chunks prefill long prompts faster at no decode cost
  (Dense8, INT8 KV, RTX 5090, 2026-10-16: 32K prompts 5,570 / 9,640 / 11,210 tok/s at chunks
  1,024 / 4,096 / 16,384, 64K prompts 5,700 / 9,890 / 11,270; tg512 unchanged; 26 fewer expert
  frames of ~8,560). A smaller `--prefill-chunk` only shrinks the prefill workspace. A long
  prompt's prefill workspace is lent from the expert cache; a chunk whose workspace the cache could
  never lend (more than three quarters of its frames, e.g. 65,536 tokens under the SSD tier, ~27
  GiB) is halved at startup until it can, with a warning, since the output does not depend on the
  chunk.
- **`--spec dflash2 --draft-model Qwen3.8-Flash-Next-DFlash2-Infernix.infernix --draft-tokens auto
  --lm-head-draft` (recommended)** drafts with Inco AI's DFlash2 drafter, a separate 776 MB
  companion artifact ([Qwen3.8-Flash-Next-DFlash2-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-DFlash2-Infernix)) that runs beside any Flash-Next
  conversion.
  - Each round drafts a block of seven tokens in one pass. With `auto` it verifies the number of
    them that pays for its verification columns: each draft's acceptance is read from the
    drafter's own probability for it, calibrated per request.
  - One request, 384-token continuations of code and Markdown docs on an RTX 5090 (INT8 KV,
    2026-10-20): Dense8 213 / 204 / 179 / 161 tok/s (code / code sampled at temperature 1 / docs /
    docs sampled) against MTP's 198 / 162 / 164 / 145; IQ3_S 258 / 210 / 183 (code / docs / docs
    sampled) against 233 / 192 / 167.
  - `--draft-tokens N` (1-7) verifies N every round, which on this offloaded model was slower than
    MTP: a verified column costs about a third of a plain round in expert-cache misses.
    `--spec-min-p P` then verifies only the drafts up to the first whose probability is below P
    (0.6 was the best fixed cut, still below `auto` on docs).
  - The drafter takes less VRAM than MTP's: 9,070 cached experts against 8,740 on Dense8 (IQ3_S:
    12,432 against 12,288), plus its 40 MiB context ring per lane.
  - Greedy output is the same as without it when one request runs at a time; sampling keeps the
    sampler's distribution. With two or more requests decoding together, rounds decode without
    drafts (as MTP's do without `adaptive-speculation`).
- `--spec mtp --draft-tokens auto --lm-head-draft` drafts with the model's MTP layer.
  - `auto`: each round drafts the number of tokens, 0 to 7, the request's recent acceptance makes
    worthwhile (down to none on text it predicts poorly), and verifies them up to the first the
    drafter gives below 0.5 probability (`--spec-min-p`).
  - `--draft-tokens N` (1-7): every round drafts and verifies N tokens (fewer only near the output
    limit), whatever the acceptance; `--spec-min-p` then defaults to 0.
  - The drafter's 512 experts (1.34 GB), its workspace and the proposal head (178 MB) stay in
    VRAM, so the expert cache gets about 710 fewer frames.
  - Greedy output is the same as without it when one request runs at a time. For several lanes,
    see `--max-concurrency` below.
- `--ngram-draft-tokens 7` verifies copy proposals. It works alone or beside MTP; a longer copy
  proposal replaces a round's MTP drafts. On code-editing prompts it accepts most drafts; on prose
  it finds none and costs nothing. In the agentic replay (`bench/agentic_ab`, greedy, seed 42, two
  lanes, 2026-10-18) copy drafts came to under 3 % of the output tokens, and 7/12, 7/4 and 15/12 (drafts /
  minimum match) were level with none on IQ3_S (within 1 %) and no faster on Dense8 or FP8, so the
  recommended commands leave it off. See [ngram copy proposals](ngram.md).
- `--max-concurrency N` (1-8) runs up to N requests at once. Its main gain is a shorter wait for
  requests that would otherwise queue (time to first token): requests that decode together share
  each round's dense weight reads, but their routed experts barely overlap.
  - Speculation (`--spec mtp`, `--ngram-draft-tokens`) depends on how many requests decode
    together:
    - one: with `auto`, the request picks its own draft length, and verifies its drafts only up
      to the first the drafter gives less than 0.5 probability (C = 1 sampled MTP on an RTX 5090:
      138.8 -> 147.2 tok/s, 75.6 -> 81.8 % of drafts accepted; output unchanged, since every draft
      is still verified); with N, it drafts and verifies N;
    - two: both draft the same number of tokens (`auto`: 0-2 chosen from their acceptance; N:
      min(N, 2)), so neither is
      padded to the other's draft. Such a round has at most 6 columns, where concurrent requests
      give the same greedy output as each alone (design §19.3.5). Measured on an RTX 5090 with
      sampled AIME answers: 164.9 tok/s for the pair against 143.8 without drafts (+14.7 %);
    - three or more: one token per request per round without drafts (one draft each measured
      slower at four requests).
  - Each lane takes VRAM from the expert cache: its KV extent (14.6 KB per token of
    `--max-context` with INT8 KV, unless `--kv-capacity` fixes one pool for all lanes) and its
    recurrent state (113 MB), records and workspace. At startup the engine prints the cost per
    lane, as `expert cache: <F> frames of 2.64 MiB at <N> lanes; each lane holds <n> frames (...)`.
    With INT8 KV a lane holds about 130 frames at a 16K context and about 390 at 64K, of roughly
    9,000.
  - A request running alone at `--max-concurrency 2` therefore has a slightly smaller expert
    cache than at 1 (estimated ~0.5 % slower decode at 16K and ~1.5 % at 64K).
- The engine starts six CPU worker threads for missed experts (up to 16 per layer call). They
  spin while decoding. More
  workers measured slower: the CPU and the PCIe stage share the host's memory bandwidth.
  The GSQ-RCO IQ3_S artifact's CPU route decodes its GGUF blocks before multiplying, so it is
  compute-bound and starts 16 workers instead (decode 98.4 tok/s with six, 103.7 with 16).
- On Windows the engine opts its process out of power throttling. Windows otherwise slows a
  process it considers in the background (a server whose console is not in front, a job without a
  window) and moves its threads to efficiency cores; a hidden run decoded 37.6 tok/s at 8K throttled
  and 98.7 exempt (design §19.3.19).
- `--vision` enables images and video. The vision tower stays in pinned RAM and borrows expert
  frames only while it encodes (`--vision-offload auto`).
- The expert cache fills the VRAM that remains; see [VRAM](#vram).
- **Warm start.** The engine saves the expert cache's state (which experts it holds and how often each
  was used) to `<artifact>.expert-state` when it stops and every 10 minutes between requests. The
  next start loads the highest-ranked experts into half of the expert cache before the first
  request (under a second), so the first few hundred tokens decode faster when the work resembles
  the last session's (measured +24 % plain and +30 % with MTP on a repeated coding prompt), and
  about as fast as a cold start when it does not; the other half stays free to fill quickly. With
  no saved state yet (a first start, or a file of another artifact), the engine seeds the same way
  from a built-in calibrated expert order (NInfer's, ranked over 128 tasks in nine domains), which
  every Qwen3.8-Flash-Next artifact uses (measured against an empty cache on the first request, MTP
  decode: +9 % on a coding prompt, +15 % on a math problem, +56 % with the SSD tier). `--expert-state
  FILE` moves the file; `--expert-state off` starts with an empty cache and no seed, as cold
  benchmarks need (`infernix_bench` always starts that way). Output is the same either way: where an
  expert is computed never changes a bit.
- The `infernix-serve` statistics panel's `experts` column shows the expert cache's hit rate over
  the session and the last ten requests.
- After each request the engine logs two Debug lines (`--log-level debug`): the expert cache's
  hit rate, and the request's n-gram row traffic (`n-gram rows: N requested, H% host-cache hits,
  R NVMe reads, T ms of reads`, and with speculation `; G gated rounds: R NVMe reads (T ms) behind the gate, the GPU
  waited in W (mean X us)`). The row cache outlives requests, so repeated text hits it, while new
  text reads most of its rows from the volume. Rows missing from that cache are read while the
  GPU starts the round (draft tokens' rows while the verification round starts), so most of that
  read time is hidden.

## N-gram table in RAM

`--ngram-residency auto|ram|volume` chooses where the PLE n-gram table lives:

- `volume` reads rows from the NVMe volume on demand, through a host cache of 2^20 rows.
- `ram` reads the whole volume into pinned RAM at startup and never touches the drive again. The
  GGUF import's `IQ4_NL` table takes 27.1 GiB; NVIDIA's FP8 table takes 48.8 GiB. The read uses
  unbuffered 1 GiB requests. The RAM ledger counts the table, and startup stops if it does not fit.
- `auto`, the default, takes RAM only when the ledger still holds every routed expert in RAM
  beside the table, and the volume otherwise. On a 96 GB machine that means RAM for the GGUF
  import (46.9 + 27.1 GiB) and the volume for recipes A and B (64.5 + 48.8 GiB do not fit).

The startup line names the choice: `n-gram table: in RAM (N MiB pinned)`, or `n-gram table: read
from its volume`, which `auto` follows with `(the RAM ledger keeps the experts resident first)`.
The rows are the same bytes either way, so outputs do not change.

## N-gram table formats

The PLE n-gram table has three volume formats: NVIDIA's FP8 rows with one scale for the whole table
(recipes A-C bind it), the GGUF import's IQ4_NL rows (4.5 bits per value), and the BF16 rows of
Qwen's checkpoint (the FP8 checkpoint's exact artifact binds it). Every artifact binds one of them as
its own, and **any Qwen3.8-Flash-Next artifact runs with a volume of any of the three formats**:
pass it with `--ngram-volume`. The engine reads the row format from the volume's header and logs
`n-gram table: <format> rows from <volume> in place of the artifact's <format> table`; the PLE
decodes the FP8 or IQ4_NL rows (or takes the BF16 rows as its embedding) as it does for the
artifact's own table. The three volumes hold the same table row for row: on 3,002 sampled rows,
the FP8 rows' relative L2 error against the BF16 rows was 2.7 % (median; 3.1 % at the 99th
percentile) and the IQ4_NL rows' 7.6 % (8.7 %), against 142 % for rows shifted by one. With the
NVIDIA NVFP4 Dense8 artifact (the 1M-token corpus's quick subset, 4,096-token windows at stride
2,048, INT8 KV; 2026-10-18) the perplexity was 3.5102 with its own FP8 table, 3.5101 with the BF16
table and 3.5140 with the IQ4_NL table.

FP8 rows need the table's scale. An FP8 artifact stores it (`ngram_scale`), so its own volume always
works. For any other artifact the engine reads it from the volume's header, which FP8 volumes record
since 2026-10-18, or, for NVIDIA's published FP8 volume (written before), recognises that file by
the volume id in its header and uses its known scale, so the downloaded file works unchanged. An
FP8 volume with neither serves only its own artifact.

The FP8 checkpoint's artifact, converted with the BF16 table, is bit-exact to Qwen's checkpoint
only with its own BF16 volume; another table rounds the PLE's values as that table's artifacts do.

### BF16 n-gram table

A BF16 volume holds the table as Qwen's BF16 checkpoint stores it:

```text
python -m tools.convert.fetch_tensors --repo Qwen/Qwen3.8-Flash-Next --revision <sha> --out <dir> \
  --pattern "model.language_model.layers.*.ple.ple_embedding.*"
python -m tools.flash_next.ngram_bf16 --model <dir> --out <fast drive>/Qwen3.8-Flash-Next-BF16.ngram
python -m tools.flash_next.ngram_bf16 --model <dir> --out <fast drive>/Qwen3.8-Flash-Next-BF16.ngram --verify
infernix-serve <artifact part 1> --ngram-volume <fast drive>/Qwen3.8-Flash-Next-BF16.ngram ...
```

Any checkpoint that carries Qwen's BF16 table unchanged serves as `--model`: UkisAI's Swift 1.5
NVFP4 export keeps it in its `embedding-*.safetensors` files, and all 128 shards are byte-identical to
Qwen/Qwen3.8-Flash-Next's (31 files by SHA256, the other 4 tensors compared byte for byte). Writing
the volume from local shards took 205 s; `--verify` of 1,000,000 sampled rows 215 s.

The table is ~102 GB in the checkpoint and the volume ~109 GB (rows of 320 bytes, 12 per 4 KiB
block, against 52 GB for FP8). A token reads as many 4 KiB blocks as with the FP8 volume, so a drive
with fast random reads serves it as fast; it does not fit in RAM (`--ngram-residency auto` keeps it
on the volume). A persistent prefix cache saved with one table is not restored with the other.
The quality change against the FP8 table has not been measured yet.

## Chat template

The conversion above stores [qwen3_8.jinja](../tools/chat_templates/qwen3_8.jinja), the template
Qwen3.8-27B uses; Flash-Next's own template is Qwen3.8-27B's. Among its Infernix changes, it keeps
a System or developer message after the first turn in place, as serving documents (Claude Code's
normal mode sends them). Artifacts converted without it, including the published
Flash-Next artifacts, store the checkpoint's template, which rejects such a message with
"System message must be at the beginning". Start them with
`--chat-template tools/chat_templates/qwen3_8.jinja` (the file is also in the Windows release
zip's `chat_templates` folder).

## Prefix cache

`infernix-serve` reuses prompt prefixes with the hybrid prefix cache (its default;
[spec](maintainer/hybrid-prefix-cache-spec.md#17-qwen4exp-binding-qwen38-flash-next)). A
conversation's next turn resumes from a snapshot of the model state and prefills only the new
tokens; KV blocks are shared across requests. `infernix` (one request) runs without it.

- **Host RAM.** Snapshots (116 MB each) and KV blocks live in one pinned Host pool,
  `--host-context-mib` (default 8192, must be positive; it comes out of the RAM the automatic
  `--expert-ram-mib` and `--expert-pin-mib` leave the experts). It is pinned at startup and counted in the
  RAM ledger (`prefix cache` in the ledger line that `--log-level debug` prints). `--no-prefix-reuse` turns the cache off and frees
  that RAM for the experts.
- **Snapshots.** Each turn leaves one at the generation opener (the assistant turn's start) and one
  at its end; shared prefixes (the system and tools block, client breakpoints) get one where they
  end. Copying a snapshot to the Host overlaps the next call. The opener costs one small extra
  call per turn (estimated 25-40 ms); a system-block or breakpoint snapshot inside a prefill chunk
  costs one extra chunk call (estimated ~1.3 s), once per distinct prefix.
- **Queue holds.** What each waiting request would resume from is evicted after every other cache
  entry, the request furthest back in the queue first, so parallel agents whose contexts together
  exceed the cache lose whole conversations at the back of the queue instead of every waiting
  request losing part of its own ([spec §9.6](maintainer/hybrid-prefix-cache-spec.md#96-queue-holds)).
  `--no-queue-holds` turns this off.
- **Exactness.** A resumed request computes what the request that left the snapshot computed; it
  may differ from an uncached run where two tokens are near ties. `--no-prefix-reuse` gives
  uncached runs.
- `--prefix-cache-file PATH` saves the Host tier when the server stops and restores it at the next
  start, as for Qwen3.5 ([serving](serving.md)); a file from another artifact, KV format, drafter
  or `infernix-serve` build is ignored and replaced. `--prefix-cache-save-mins N` also saves it
  every N minutes while serving (only when it changed; requests wait while the file is written, a
  few seconds for several GB; the file is replaced only once the new one is complete). Each save
  rewrites the whole file, so a short interval adds a lot of SSD wear: prefer 15-60 minutes.
- Not available for this model: `--device-snapshot-slots`.

## SSD expert tier

When the routed experts do not all fit in RAM, or `--expert-ram-mib N` caps them below the ~64.5 GiB
they need, or Windows cannot page-lock all of them (see [Page-locking limit](#page-locking-limit)),
the experts stay in the artifact: startup pins N MiB (or what the ledger leaves, up to the
page-locking limit) of expert slots and fills all but ~430 of them (kept free for reads and
demotions) before the first request; RAM past the limit holds unpinned records. RAM and VRAM hold different experts: the VRAM warm start takes the top of the saved
ranking of `--expert-state`, and RAM the next ones, then the experts the last session used most,
then the rest spread evenly across the layers. With no saved state, both take the calibrated
order instead (with `--expert-state off`, RAM takes the even spread). The tier's
usage ranking starts from the saved counts, so the best-ranked RAM experts are not the first to
be replaced. A VRAM eviction of an expert worth keeping copies it back into RAM.
A round that routes an expert in neither VRAM nor RAM reads it from the artifact while the layer's
other experts compute: decode and verification hand it to the CPU expert engine, prefill copies it
from the read as soon as it lands. Read experts that rank above the coldest RAM expert replace it.
Records that follow each other in the artifact are read as one request (the startup fill up to
32 MiB, a prefill's fetched experts up to 12 MiB), which the drive serves faster than one request
per expert. Outputs are the same bits as with every expert in RAM; only speed changes, with the share of routed
experts read from disk. The tier needs at least 2.6 GiB (1,024 slots). A read that fails fails the
requests running in that round with an error and the server keeps serving; queued requests wait as
usual, and the prefix cache is emptied (the failed round may have published state it never
computed). If the tier's read thread or the CPU expert service itself stops, every later round
would fail too, so the server stops with that cause. A round waits for either while it shows signs
of life, and fails only when one has been silent for 10 s ("the CPU expert service stopped
answering" or "the SSD expert tier's read agent stopped answering"); until 2026-10-17 that was 1 s,
which RAM pressure alone could exceed (FP8 runs at the default headroom failed 8 times in 34).

**A second drive (`--expert-stripe FILE`).** One drive bounds the tier's reads, most of all in
prefill, which reads every expert missing from RAM (about 38 GiB for a 4K-token prompt with the FP8
artifact). An expert stripe holds the tail of every expert record on another drive; the tier then
reads each record's head from the artifact and its tail from the stripe at the same time, at the
same queue depth on each drive. The artifact is unchanged and keeps every record whole, so the
stripe costs a copy of the tails only (by default half the experts: 56 GiB for the FP8 artifact).
The stripe names its artifact, which refuses any other. Write it once, on a drive other than the
artifact's (and the n-gram volume's), and check it against the artifact:

```text
python -m tools.flash_next.expert_stripe <artifact part 1> D:\stripes\<model>.stripe
python -m tools.flash_next.expert_stripe <artifact part 1> D:\stripes\<model>.stripe --verify
infernix-serve <artifact part 1> --expert-stripe D:\stripes\<model>.stripe ...
```

`--tail-fraction F` (default 0.5) moves more of each record to a faster second drive. Records then
are read in two parts each, never as runs of neighbouring records. With every expert in RAM the
stripe is not used. Outputs are the same bits with or without it.

The tier and the n-gram volume read with unbuffered, 4 KiB-aligned I/O. On ZFS (NInfer #404) a
dataset's `recordsize` (128 KiB by default) is the unit the filesystem reads, so each small read
can cost a whole record; keeping the artifact and the `.ngram` volume on a dataset with
`recordsize=16K` or less avoids that (not measured here; Windows NTFS is unaffected).

### Page-locking limit

The GPU reads RAM experts only from page-locked (pinned) memory, and Windows page-locks at most
the adapter's shared system memory for it: on an RTX 5090 with 95.8 GiB of RAM page-locking stopped
at 76.0 GiB of a reported 76.6 GiB (80 % of RAM there; many machines report half of RAM). The RAM
ledger reads that size and its current use at startup and keeps 1 GiB below it. Every other pin
comes first (the embedding, the prefix cache's Host tier, the Program's buffers); the experts pin
what is left, and their RAM beyond it holds **unpinned records**:

- The RAM the experts get does not change: a 96 GB machine with a half-of-RAM limit still holds
  every expert in RAM (the SSD tier with nothing on disk), a 64 GB machine the same experts as
  without the limit.
- Unpinned records are ordinary memory kept resident (a raised minimum working set and
  `VirtualLock`), filled at startup with the experts ranked after the slots' (none of them in VRAM)
  and never replaced; the slots keep adapting.
- Decode and verification hand an unpinned expert to the CPU expert engine, which reads it in
  place (as fast as from pinned memory). A prefill streams it two layers ahead like a pinned one,
  through pinned bounce slots that the CPU fills first (four parts, each a layer's unpinned share of its 512 experts, 64-256 records: 2.2 GiB with 40 % of the experts unpinned); past that
  a layer's call copies it into a pinned ring slot instead of reading it from disk. A VRAM
  promotion stages it through 32 more bounce slots. The bounce slots come out of the pinned
  share. Each bounced copy crosses RAM three times, so these paths reach about a third of the RAM
  bandwidth (~25-30 GB/s with two DDR5 channels), below a PCIe 5.0 x16 link: pinned experts stay
  faster for prefill, which is why the coldest experts are the unpinned ones. Outputs are the same
  bits.
- If Windows refuses a slot chunk anyway (another program pinned memory meanwhile), the tier gives
  back enough for the prefix cache and the Program's buffers, and the slots it lost become unpinned
  records; a warning says so.
- `--expert-pin-mib N` page-locks at most N MiB of the experts' RAM (the rest unpinned), for
  example to leave shared GPU memory to other programs.

The info line at startup names the split (`… page-locked for the GPU and … kept in RAM for the
CPU`), and the tier's line counts the unpinned records.

## VRAM

At startup the engine reads the GPU's free memory, the Windows video-memory budget of the process
and whether the GPU drives a display (DXGI outputs, then NVML). The expert cache takes

`min(free - fixed - headroom, budget - fixed - 64 MiB) - reserve`

where `fixed` is the dense weights plus the KV cache, workspaces and staging, `reserve` is
max(256 MiB, 128 MiB + 4 MiB per CUDA graph the engine may capture, counting at most 32), and
`headroom` is `--vram-headroom-mib`: by default 512 MiB when the GPU drives a display (or cannot
tell) and 256 MiB when it does not. The console shows a one-line summary (`VRAM: model …, KV cache
…, expert cache …`); `--log-level debug` adds the full ledger, for example:

```text
VRAM ledger: 32607 MiB card, no display, 547 in use before loading; dense weights 4198 MiB, KV 1024,
workspace 1236, expert staging 169, state and io 310, expert tables 7; reserve 256 (20 graphs);
headroom 256 (auto); expert frames 9300 (23.95 GiB); 380 free after startup
```

- **Before reading the weights**, the engine checks that the fixed allocations fit; if they do not,
  it stops with the largest terms and what to lower (`--max-context`, `--max-concurrency`,
  `--prefill-chunk`, `--kv-dtype int8`). Below 2,048 expert frames it warns that decode will be slow.
- **No silent spill.** On Windows the driver can place an allocation that does not fit in shared
  system memory without an error, which slows everything down. The engine checks that every startup
  allocation lowered free VRAM by its size: dense weights or fixed buffers that spilled stop
  startup; expert frames that spilled are given back once and the cache starts smaller. Setting the
  NVIDIA Control Panel's *CUDA - Sysmem Fallback Policy* to *Prefer No Sysmem Fallback* for the
  Infernix executables makes such an allocation fail instead (optional).
- **At runtime** the cache follows free VRAM. When another program (a browser, a game, the
  desktop) takes memory and free VRAM falls below half the headroom, the cache gives back the
  least valuable experts at the next round boundary, or at once when idle, and logs it; it grows
  back 30 s after the memory is released. Decode is slower while the cache is smaller, and a
  warning is logged if it falls below a quarter of its startup size.
- **The KV cache is elastic.**
  - Startup backs only its first 32K tokens; the ledger shows `KV 482 (grows to 3360 from the
    expert cache)`.
  - A request that needs more KV takes the memory from the expert cache when it is admitted, and
    gives it back when it ends.
  - Cached prefix blocks that also have a host copy return their memory after 60 s with no request
    running; a later match restores them from host RAM.
  - A long `--max-context` therefore costs expert frames only while long contexts are in use. On
    an RTX 5090 at `--max-context 262144`: 9,299 frames instead of 8,197, and plain decode 109.6
    instead of 103.5 tok/s.
- **`--vram-past-budget`** (Windows) also sizes the expert cache into VRAM that the Windows
  per-process budget withholds.
  - On an RTX 5090 the budget is 31,419 MiB against 32,187 MiB of physical memory less the driver's
    reserve. A further 640 MiB allocates and runs at full VRAM speed: about 240 more expert frames.
  - The engine measures this allowance at startup: physical memory less the reserve, less the
    budget, less a 128 MiB margin. It reports it in the ledger (`… in use before loading (640 MiB
    past the OS budget)`).
  - If Windows lowers the budget later, the cache shrinks by the same amount.
  - The cost: memory past the budget may be moved to system memory if another program needs VRAM,
    which slows decode until the cache shrinks. It is off by default.
- <a id="expert-tail-skip"></a>**`--expert-tail-skip R`** (approximate, off by default) trades
  quality for decode speed. Each token routes 10 experts, ranked by router weight (rank 0 the
  largest). In a decode or verification round, an expert missing from VRAM that every token of the
  round routes at rank `R` or lower (`1..9`) is neither copied nor computed: it adds zero, and the
  other experts keep the weights the router gave them (no renormalisation).
  - It saves the PCIe copies and CPU work of those misses, so it helps most when many experts miss
    VRAM (the SSD tier, small cards); with every expert in VRAM it changes nothing.
  - The output changes and is slightly worse, and it depends on which experts the cache holds at
    the time, so it is neither the exact route nor request-invariant (`--disable-fast-paths` refuses
    it, and the load summary reports the Engine as not request-invariant). Prefill, forced and scored
    tokens are unchanged, and a skipped expert still counts as routed for the cache, so it can be
    promoted.
  - Experts that a round routes from more than 8 tokens are always computed.
  - Measured (2026-10-18, RTX 5090 on PCIe 5.0 x8, INT8 KV, `infernix_bench` pp2048 + tg512, MTP
    `auto`, two runs per arm alternating; each arm decodes its own text): rank 7 took Dense8 from
    155 to 207 tok/s (+33 %), FP8 from 47 to 61 (+28 %) and IQ3_S from 210 to 234 (+12 %), every run
    ahead; rank 9 on FP8 changed nothing beyond spread (+0.4 %). Rank 7 was gated before the
    option was built, on an IQ3_S prototype (INT8 KV, MTP decode, 256 segments, 2026-10-16): TV
    excess bound 4.26e-3 against 4e-3, top-1 0.75 against 0.5 points, p99.9 KL ratio 1.41 against
    1.25. The option as built has not been gated again.
  - Strata turns the same rule on by default at rank 7 (`STRATA_ROUTE_TAIL_SKIP=7`, single-request
    windows; it reports KL 0.003-0.006 and +10-20 % decode on 12-16 GB cards). Rank 7 failed
    Infernix's fast-path decode gate (model-quality §6; [design §19.3](maintainer/qwen3_8-flash-next-design.md)),
    which is why it is an option rather than a fast path. 9 skips least, 1 most.
- On a system without CUDA virtual memory management the cache cannot resize: the headroom with a
  display is then 1 GiB, and a warning names the headroom to set when free VRAM runs short.
- `--kv-capacity auto` means `--max-context` x `--max-concurrency`, the same as omitting it.

## Performance

RTX 5090 on PCIe Gen5 x8, i9-13900K, DDR5-5800, recipe B, INT8 KV, one request. The expert cache
warms up over the first few hundred tokens of a session.

The speed figures in this guide were measured before 2026-10-07 on an artifact converted from
`RadixArk/Qwen3.8-Flash-Next-NVFP4` instead of NVIDIA's checkpoint. Re-measured on NVIDIA's, the
README's context-length decode came out 1.6-4.1 % lower and the agentic replay level; the
[README benchmarks](../README.md#qwen38-flash-next-nvfp4-infernix-vs-strata-and-strata-nvfp4) are the current figures.
Recipe A (bit-exact) decodes about 18 % slower than recipe B and reaches the first token 6-8 %
later at 8K, level from 128K (the bit-exact
[model card's performance table](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix#performance)).

| Workload | tok/s |
|---|---:|
| `infernix_bench` tg512, `--spec mtp --draft-tokens 4 --lm-head-draft` (then a per-round length up to 4, now `auto`) (warm cache) | ~134 |
| `infernix_bench` tg512, plain decode (warm cache) | ~87 |
| Single CLI request, cold cache, plain: code rewrite / prose story | ~66 / ~65 |
| Single CLI request, cold cache, MTP: code rewrite / prose story | ~84 / ~67 |
| Prompt, 4,096 tokens, `--prefill-chunk 4096` | ~670-690 |
| Prompt, 4,096 tokens, `--prefill-chunk 1024` (the default then) | ~410 |

tg512 generates its own greedy text, so it moves with any change in rounding. Its current text
has a lower expert hit rate (88 % against 93 %) than before the last kernel change, which is
why it reads ~139 / ~90 in earlier notes. On a prompt whose text stayed identical, that change
was faster (design §19.2).

MTP accepts most drafts on code and other predictable text, and fewer on free prose. The draft
length follows the measured acceptance, so prose mostly drafts one token and stays at least as
fast as plain decode (design §19.2).

**Quality.** Teacher-forced perplexity over three frozen texts (2,557 positions, INT8 KV): recipe
B 4.654 and recipe A 4.666 (2026-10-08 kernels; recipe B measured 4.664 before them). For comparison, Strata 0.1.40 with UD-Q4_K_XL and INT8 KV gives 4.844 on the same token ids
(−0.038 ± 0.011 nats for Infernix; level on code and documents, better on chat).
