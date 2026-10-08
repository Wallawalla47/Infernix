# Qwen3.8-Flash-Next (`Qwen4ExpForCausalLM`)

Infernix runs NVIDIA's NVFP4 checkpoint of Qwen3.8-Flash-Next
([nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4)) on one
RTX 5090 (32 GB). The 24,576
routed experts (63 GiB) live in pinned host RAM. A VRAM expert cache holds the hot ones, and the
CPU computes part of each layer's misses. The 52 GB PLE n-gram table lives on an NVMe volume. The
design, measurements and open work are in the
[design document](maintainer/qwen3_8-flash-next-design.md) (§16.5, §19.2).

Generation works through `infernix`, `infernix-serve` and `infernix_bench`, with the model's MTP
drafter (`--spec mtp`), n-gram copy proposals, images and video (`--vision`), and in
`infernix-serve` the hybrid prefix cache. Not yet supported: CausalScoring (perplexity).

## Requirements

- **GPU.** An RTX 5090 (sm_120a). A **PCIe x16** link matters: expert misses cross PCIe, and an
  x8 link halves their bandwidth. Check with
  `nvidia-smi --query-gpu=pcie.link.width.current --format=csv`.
- **RAM.** About 96 GB for full speed. A run pins ~64.5 GiB. At startup a RAM ledger plans every
  allocation and keeps `--ram-headroom-mib` (default 2048) free for the system; with the defaults
  about 69.3 GiB must be available, and 73.3 GiB for `infernix-serve` with its prefix cache (see
  [Prefix cache](#prefix-cache)). With less, the [SSD expert tier](#ssd-expert-tier) keeps the
  experts that fit in RAM and reads the others from the artifact; the ledger line names every term.
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
  --out <dir>/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix.infernix --ngram-out <nvme>/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram
python -m tools.artifact.rename <dir>/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix.infernix Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix.infernix --numbered
```

The second command names the three files as numbered parts
(`Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix-00001-of-00003.infernix` and so on); Infernix
opens part 1.

- **Recipe A.** Use `--recipe qwen3_8_flash_next_nvfp4` and the name without `-Dense8`.
- **Verification.** `python -m tools.flash_next.verify_artifact --model <checkpoint> --artifact
  <part 1> --ngram <volume>` compares every stored tensor, expert and n-gram row with the
  checkpoint and lists the re-quantized parameters per component.
- **Second artifact of the same checkpoint.** It can share an existing n-gram volume: pass
  `--ngram-reuse <volume>` instead of `--ngram-out`. The converter checks the volume's geometry and
  512 of its rows against the checkpoint, so a volume of another checkpoint is refused.
- **Duration.** A conversion takes ~11 minutes from a warm disk cache; the MTP experts' MSE
  scale search accounts for several of them.

## Run

```text
infernix <artifact>.infernix --ngram-volume <volume>.ngram --kv-dtype int8 --max-context 16384 \
  --prompt "..."
infernix-serve <artifact>.infernix --ngram-volume <volume>.ngram --kv-dtype int8 --max-context 16384 \
  --prefill-chunk 4096
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
  length.
- `--prefill-chunk 2048` or `4096` speeds up long prompts by 1.4-1.6×. Each costs ~2-5 % of
  decode speed, because the larger prefill workspace takes expert frames.
- `--spec mtp --draft-tokens 4 --lm-head-draft` drafts with the model's MTP layer.
  - `--draft-tokens` sets the maximum. Each round drafts the number of tokens the recent
    acceptance makes worthwhile, down to none on text it predicts poorly.
  - The drafter's 512 experts (1.34 GB), its workspace and the proposal head (178 MB) stay in
    VRAM, so the expert cache gets about 710 fewer frames.
  - Greedy output is the same as without it when one request runs at a time. For several lanes,
    see `--max-concurrency` below.
- `--ngram-draft-tokens 7` verifies copy proposals. It works alone or beside MTP; a longer copy
  proposal replaces a round's MTP drafts. On code-editing prompts it accepts most drafts; on prose
  it finds none and costs nothing. See [ngram copy proposals](ngram.md).
- `--max-concurrency N` (1-8) runs up to N requests at once. Its main gain is a shorter wait for
  requests that would otherwise queue (time to first token): requests that decode together share
  each round's dense weight reads, but their routed experts barely overlap.
  - Speculation (`--spec mtp`, `--ngram-draft-tokens`) depends on how many requests decode
    together:
    - one: the request picks its own draft length, and verifies its drafts only up to the first
      the drafter gives less than 0.5 probability (C = 1 sampled MTP on an RTX 5090: 138.8 ->
      147.2 tok/s, 75.6 -> 81.8 % of drafts accepted; output unchanged, since every draft is still
      verified);
    - two: both draft the same number of tokens, 0-2 chosen from their acceptance, so neither is
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
  about as fast as a cold start when it does not; the other half stays free to fill quickly. `--expert-state FILE` moves the file; `--expert-state off` starts with an empty cache,
  as cold benchmarks need. Output is the same either way: where an expert is computed never
  changes a bit.
- The `infernix-serve` statistics panel's `experts` column shows the expert cache's hit rate over
  the session and the last ten requests.
- After each request the engine logs two Info lines: the expert cache's hit rate, and the
  request's n-gram row traffic (`n-gram rows: N requested, H% host-cache hits, R NVMe reads, T ms
  of reads`, and with speculation `; G gated rounds: R NVMe reads (T ms) behind the gate, the GPU
  waited in W (mean X us)`). The row cache outlives requests, so repeated text hits it, while new
  text reads most of its rows from the volume. Rows missing from that cache are read while the
  GPU starts the round (draft tokens' rows while the verification round starts), so most of that
  read time is hidden.

## Prefix cache

`infernix-serve` reuses prompt prefixes with the hybrid prefix cache (its default;
[spec](maintainer/hybrid-prefix-cache-spec.md#17-qwen4exp-binding-qwen38-flash-next)). A
conversation's next turn resumes from a snapshot of the model state and prefills only the new
tokens; KV blocks are shared across requests. `infernix` (one request) runs without it.

- **Host RAM.** Snapshots (116 MB each) and KV blocks live in one pinned Host pool,
  `--host-context-mib` (default 4096, must be positive). It is pinned at startup and counted in the
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
- Not available for this model: `--use-original-prefix-caching`, `--device-snapshot-slots`.

## SSD expert tier

When the routed experts do not all fit in RAM, or `--expert-ram-mib N` caps them below the ~64.5 GiB
they need, the experts stay in the artifact: startup pins N MiB (or what the ledger leaves) of
expert slots and fills all but ~430 of them (kept free for reads and demotions) before the first
request. RAM and VRAM hold different experts: the VRAM warm start takes the top of the saved
ranking of `--expert-state`, and RAM the next ones, then the experts the last session used most,
then the rest spread evenly across the layers (the same spread with no saved state). The tier's
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
would fail too, so the server stops with that cause.

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
- On a system without CUDA virtual memory management the cache cannot resize: the headroom with a
  display is then 1 GiB, and a warning names the headroom to set when free VRAM runs short.
- `--kv-capacity auto` means `--max-context` x `--max-concurrency`, the same as omitting it.

## Performance

RTX 5090 on PCIe Gen5 x8, i9-13900K, DDR5-5800, recipe B, INT8 KV, one request. The expert cache
warms up over the first few hundred tokens of a session.

The speed figures in this guide were measured before 2026-10-07 on an artifact converted from
`RadixArk/Qwen3.8-Flash-Next-NVFP4` instead of NVIDIA's checkpoint. Re-measured on NVIDIA's, the
README's context-length decode came out 1.6-4.1 % lower and the agentic replay level; the
[README benchmarks](../README.md#qwen38-flash-next-nvfp4-infernix-vs-strata) are the current figures.
Recipe A (bit-exact) decodes about 18 % slower than recipe B and reaches the first token 6-8 %
later at 8K, level from 128K ([README](../README.md#bit-exact-against-dense8)).

| Workload | tok/s |
|---|---:|
| `infernix_bench` tg512, `--spec mtp --draft-tokens 4 --lm-head-draft` (warm cache) | ~134 |
| `infernix_bench` tg512, plain decode (warm cache) | ~87 |
| Single CLI request, cold cache, plain: code rewrite / prose story | ~66 / ~65 |
| Single CLI request, cold cache, MTP: code rewrite / prose story | ~84 / ~67 |
| Prompt, 4,096 tokens, `--prefill-chunk 4096` | ~670-690 |
| Prompt, 4,096 tokens, default chunk 1024 | ~410 |

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
