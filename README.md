# NInfer — custom fork

> **AI disclaimer:** Everything added to this fork, including most of this README, was written with
> AI (mostly Claude Opus 5.5, Qwen3.8-27B running on NInfer, plus a few other AI systems I’ve been
> testing). It is likely to be neither complete nor entirely accurate. This is hobby development.

This is a personal fork of [Neroued/ninfer](https://github.com/Neroued/ninfer). It follows upstream
closely and adds changes on top: the history is upstream `master` (`d44ab584`), the Windows port,
then one commit per fork change. The sections below explain what is different, grouped by topic,
with credit given as best my AI agents can where a change came from someone else. The upstream
README follows, copied unchanged, under the "Upstream README" heading. A huge thank you to Neroued
for creating NInfer!

**The short version.** Compared with upstream, this fork:

1. builds and runs natively on Windows (and still on Linux)
2. uses a new prefix caching system, designed and implemented by Claude Opus 5.5, as the default.
   You set how much system RAM it may use with `--host-cache-mib N`; `--prefix-cache-file PATH`
   keeps the cache across restarts. On the agentic benchmark below it serves far more of each
   prompt from cache than upstream's system. Stop the server with Ctrl+C (twice) rather than by
   closing the window, because Windows does not always leave enough time after a window closes to
   save a large cache
3. keeps upstream's original prefix caching, with a raft of fixes, behind
   `--use-original-prefix-caching` (I worked on it before switching to a new design; I found the
   original too complex and fragile)
4. prefills INT8 and NVFP4 KV with faster prompt-attention kernels by default (a third to two
   thirds less prompt-attention time on long prompts); `--use-original-int8-prefill-kernel` and
   `--use-original-nvfp4-prefill-kernel` select upstream's kernels, and `--int8-prefill-8bit-pv`
   opts in to an 8-bit P×V version of the INT8 one (about a tenth less prompt-attention time at
   long context, at a small numerical cost)
5. adds ngram copy drafting (based on an implementation by [remesis](https://github.com/remesis)),
   which greatly speeds up copy-heavy workloads, with more than one concurrent request
6. overlaps each decode kernel's launch and weight loading with the kernel before it, and tunes
   the decode-width kernels, for faster speculative decode rounds with the same output
7. can offload the vision encoder to system RAM (`--vision-offload on`), based on the work of
   Valeriy Selitskiy ([iamwavecut](https://github.com/iamwavecut))
8. supports YaRN context extension up to 1M tokens (`--rope-yarn-factor F`, F from 1 to 4)
9. sizes the CUDA Graph memory allowance from measurement instead of an estimate that could reserve
   far more VRAM than ever used
10. lets `--vram-headroom-mib N` shrink the 1 GiB VRAM headroom left after `--kv-capacity auto`
11. adds a custom thinking budget message (`--default-thinking-budget N` with
    `--thinking-budget-message "..."`)
12. accepts more tool-call formats and API options used by agent clients such as Claude Code, Qwen
    Code, Codex, Zed and GitHub Copilot, and fixes several tool-call and reasoning-output issues
    (mostly based on the work of others credited below); `--tolerant-tool-calls` recovers some
    broken tool calls
13. improves the console: optional colours (`--log-colours on`), a statistics panel at the bottom
    (`--log-stats-panel off` removes it) and a `--help` screen organised by category, and can
    rotate the request log by size (`--request-log-max-mib N`)
14. contains various other fixes and improvements, including upstream pull requests merged before
    upstream did

I recommend using this with the NVIDIA NVFP4 artifact I’ve uploaded here, which runs a bit faster
than the original artifact based on the Unsloth quant and takes up less VRAM:
<https://huggingface.co/wallawalla47/Qwen3.8-27B-NVIDIA-NVFP4-NInferV3>

## Quick start (Windows)

Prerequisites: Visual Studio 2026 (MSVC), the CUDA 13 toolkit, and FFmpeg + curl from vcpkg
(`x64-windows`, at `C:\vcpkg`); adjust the paths at the top of `build_native.bat` for your machine.

```bat
build_native.bat configure
build_native.bat build
```

The server is `build-windows\apps\Release\ninfer-serve.exe`, with the FFmpeg, curl and zlib DLLs
copied next to it. The launch I use on a single 32 GB RTX 5090 (stop any other resident model
first):

```bat
ninfer-serve.exe qwen3_8_27b_nvfp4-nvidia.ninfer --host 127.0.0.1 --port 8080 --max-context 240000 --max-concurrency 2 --spec dflash2 --draft-tokens 7 --lm-head-draft --ngram-draft-tokens 15 --ngram-min-match 12 --kv-dtype int8 --preserve-thinking --host-cache-mib 52000 --pending-timeout-ms 900000 --prefill-chunk 4096 --kv-capacity auto --vram-headroom-mib 0 --log-colours on --ngram-archive-mib 2048 --ngram-session-mib 256 --ngram-native-sessions --request-log-jsonl log.json --default-thinking-budget 16384 --thinking-budget-message "Considering the limited time available to the user, I must stop thinking now. Time to act:" --tolerant-tool-calls
```

Add `--prefix-cache-file PATH` to keep the prefix cache across restarts. Stop the server by
pressing Ctrl+C twice (the first press shows a prompt at the bottom of the console). The server then
cancels running and queued requests, saves the cache and exits; pressing Ctrl+C once more exits
without saving and deletes the unfinished file. `ninfer-serve.exe --help` lists every option by
category.

Running on another PC needs an RTX 50-series GPU (the build targets `sm_120a`) and an NVIDIA driver
of 580 or later (CUDA 13); no CUDA toolkit is needed. Copy the DLLs next to `ninfer-serve.exe` and
install the Visual C++ redistributable if it is missing.

## Quick start (Linux)

The fork builds and runs on 64-bit Linux too, including WSL2 (tested on Ubuntu 24.04 under WSL2
with CUDA 13.4 and GCC 13.3). Prerequisites: a CUDA 13 toolkit, CMake 3.28 or newer, a C++20
compiler, Ninja, `pkg-config`, and the FFmpeg and curl development packages. On Ubuntu 24.04:

```bash
sudo apt-get install -y build-essential cmake ninja-build pkg-config libavformat-dev libavcodec-dev libavutil-dev libswscale-dev libcurl4-openssl-dev
cmake --preset release
cmake --build build -j
```

The server is `build/apps/ninfer-serve` and takes the same options as on Windows, for example the
launch line above with `./build/apps/ninfer-serve` in place of `ninfer-serve.exe`.

Under WSL2 the GPU driver is the Windows NVIDIA driver (580 or later); do not install a Linux NVIDIA
driver inside WSL. NVIDIA's `wsl-ubuntu` CUDA repository stops at CUDA 13.3, so for 13.4 add its
`ubuntu2404` repository and install only `cuda-toolkit-13-4` (not `cuda` or `cuda-drivers`, which
pull in a driver). Artifacts on a Windows drive load slowly through `/mnt/`, so copy the `.ninfer`
file into the Linux filesystem first.

## Performance: this fork vs upstream

Both benchmarks below compare this fork (the commits before this README; the measured build
`fe869b56` has the same source) with **upstream + Windows port**: upstream at the commit this fork
is rebased on (`d44ab584`) plus only the Windows port (commit `d0abd0cb` on the branch
`upstream-Windows-Port`). Everything ran on an RTX 5090 under Windows with the official
Qwen3.8-27B NVFP4 artifact (`qwen3_8_27b_nvfp4-official.ninfer`), in September 2026.

### Agentic coding workload

The closed-loop suite in [`bench/agentic_ab/`](bench/agentic_ab/README.md) replays three
coding-agent sessions plus eleven subagents: 130 requests with fan-outs, a concurrent subagent
pair, compaction, retries, an abort and a solo wrap-up, with prompts of 25K-135K tokens and
thinking on. Each arm's own answers are fed back as an agent client does, and the three main
sessions take their turns in lock-step rounds, so every build meets the same order of session
turns whatever its speed. Both builds completed every request on each of three workload seeds
(42, 43, 44), which replay different observations.

Settings:

- **Fork arm:** the launch flags of `LaunchQwen3.8-27B-official-dflash2-ngram.bat`:

  ```text
  --max-context 170000 --max-concurrency 2 --spec dflash2 --draft-tokens 7 --lm-head-draft
  --ngram-draft-tokens 15 --ngram-min-match 12 --kv-dtype int8 --preserve-thinking
  --host-cache-mib 52000 --pending-timeout-ms 900000 --prefill-chunk 4096 --kv-capacity auto
  --vram-headroom-mib 0 --ngram-archive-mib 2048 --ngram-session-mib 256 --ngram-native-sessions
  --default-thinking-budget 16384 --thinking-budget-message "Considering the limited time
  available to the user, I must stop thinking now. Time to act:" --tolerant-tool-calls
  ```

- **Upstream arm:** the same flags minus those upstream does not have, with the host RAM split
  the fork's original cache resolves from the same 52,000 MiB passed as explicit flags:

  ```text
  --max-context 170000 --max-concurrency 2 --spec dflash2 --draft-tokens 7 --lm-head-draft
  --kv-dtype int8 --preserve-thinking --pending-timeout-ms 900000 --prefill-chunk 4096
  --kv-capacity auto --default-thinking-budget 16384 --host-state-slots 123 --host-kv-mib 29020
  --max-private-continuations 4 --max-long-anchors-per-continuation 27 --max-shared-prefixes 7
  ```

- **Context:** 170,000 tokens, the largest context the upstream build starts with under these
  flags; at it the fork's device KV holds 227,648 tokens and upstream's 175,424. **Sampling:**
  temperature 1.0, top_p 0.95, top_k 20, `max_tokens: 64000` on agent turns.

Each cell is the mean over the three seeds, with the lowest and highest seed in brackets; changes
are computed per seed against that seed's upstream run.

| Metric | Upstream + Windows port | Fork |
|---|---|---|
| Average time to first token (s) | 14.0 (11.6-15.6) | 2.7 (2.4-3.1), −80.5 % |
| Median time to first token (s) | 8.72 (8.20-9.68) | 0.63 (0.47-0.76), −92.7 % |
| 90th-percentile time to first token (s) | 36.6 (24.0-43.8) | 6.7 (5.5-8.0), −80.5 % |
| Average TTFT, continuing-session turns (s) | 13.97 (12.03-15.51) | 2.57 (2.31-3.08), −81.6 % |
| Average TTFT, new long prompts (s) | 10.5 (10.4-10.7) | 8.3 (6.4-11.6), −21.6 % |
| Prompt tokens served from cache | 69.2 % (66.7-73.5) | 91.1 % (90.9-91.2) |
| Prompt tokens prefilled | 1.83M (1.61-1.98) | 0.51M (0.49-0.54), −71.7 % |
| Main-session turns that re-prefilled the whole prompt (of 75) | 11.7 (9.0-15.0) | 0 |
| Subagent turns that re-prefilled the whole prompt (of 37) | 23.7 (17.0-31.0) | 0 |
| Prefill tok/s, requests with no cache hit in any arm | 6,085 (6,018-6,151) | 8,389 (8,380-8,407), +37.9 % |
| Prefill tok/s, the same requests from 32K tokens | 5,982 (5,913-6,051) | 8,291 (8,281-8,308), +38.6 % |
| Output tok/s, one request decoding | 197 (193-205) | 214 (204-219), +8.8 % |
| Decode rounds/s, one request decoding (engine speed) | 58.4 (56.1-59.7) | 60.2 (59.2-61.8), +3.2 % |
| Tokens per round, one request decoding (acceptance) | 3.38 (3.24-3.45) | 3.56 (3.30-3.71) |
| Output tok/s, two requests decoding (combined) | 360 (349-368) | 355 (343-378), −1.3 % |
| Decode rounds/s, two requests decoding (engine speed) | 57.3 (55.0-58.6) | 54.7 (53.7-55.5), −4.3 % |
| Decode rounds that ran two requests | 9.8 % (5.8-12.3) | 35.4 % (30.6-38.4) |
| Output tok/s, all decoding at the run's own batching | 214 (204-227) | 270 (264-282), +26.3 % |
| Workload wall time (min) | 20.7 (18.3-22.8) | 12.0 (10.0-14.6), −41.7 % |

- Time to first token includes queueing: up to seven requests are in flight on two lanes. The
  average queue wait was 11.7 s upstream and 2.2 s on the fork, and without it the average time
  to first token was 2.31 s and 0.58 s.
- Output tok/s counts decode tokens per second of the engine's own decode time. It splits into
  decode rounds/s (engine speed) and tokens per round (speculative acceptance, which moves with
  what the model happened to write). The fork's ngram drafting supplied 6-13 % of its output;
  upstream has none.
- With two requests decoding, the fork's rounds also carry ngram drafting for both, which costs
  host time per round, and upstream decoded two requests together for only 15-70 s per seed.
  The concurrent decode benchmark below compares two-request decode without ngram drafting.

### Concurrent decode

The decode-saturation suite of `tools/bench/run_serve_concurrency.py` starts a fresh server at
each `--max-concurrency` C and decodes C requests at once, each up to 8,192 tokens, with DFlash2
K=7 and `--lm-head-draft` (no ngram drafting), stochastic sampling, `--max-context 32768
--kv-capacity auto`. The builds alternated point by point in two passes, C=1 to 8 and back.

| C | Upstream tok/s | Fork tok/s | Fork time per decode round vs upstream (pass 1, pass 2) |
|---|---|---|---|
| 1 | 198.3 | 196.6 | −2.3 %, −2.4 % |
| 2 | 365.8 | 380.0 | −2.2 %, −2.3 % |
| 3 | 516.1 | 533.9 | −3.1 %, −3.1 % |
| 4 | 649.6 | 677.4 | −3.2 %, −3.2 % |
| 5 | 748.6 | 789.0 | −3.5 %, −3.5 % |
| 6 | 868.7 | 879.7 | −1.8 %, −1.7 % |
| 7 | 965.7 | 1002.6 | −1.4 %, −1.3 % |
| 8 | 1076.0 | 1092.2 | −1.3 %, −1.3 % |

Tok/s is the mean of both passes. The fork's decode rounds are 1.3-3.5 % faster at every
concurrency, the same in both passes; tok/s also moves with speculative acceptance on the sampled
text, which is why C=1 is lower despite faster rounds (2.99 against 3.08 tokens per round).

### Running the benchmarks

Commits: [`e2db558`][c-agentic-ab], [`2838371`][c-serve-concurrency],
[`9189a7e`][c-ab-rig].

Build this fork, then the upstream control from the branch `upstream-Windows-Port`
([details](bench/agentic_ab/README.md#running-it)); stop any other server on the port first:

```bat
build_native.bat configure
build_native.bat build
git worktree add C:\ab\control\src upstream-Windows-Port
set AB_CONTROL_SRC=C:\ab\control\src
set AB_CONTROL_BUILD=C:\ab\control\build
bench\agentic_ab\build_control.bat configure
bench\agentic_ab\build_control.bat build
```

Agentic workload (about 1.7 hours for the two arms on three seeds). The runner reads
the model path and launch flags from `AB_LAUNCH_BAT`, calibrates the largest context the control
starts with, and writes `report.md` under `profiles\bench\agentic_ab\`. `treatment` is the fork
and `control` upstream; an optional `alt` arm runs another build (`AB_ALT_EXE`) or the fork with
other flags (`AB_ALT_EXTRA_FLAGS`, default `--use-original-prefix-caching`):

```bat
set AB_CONTROL_EXE=C:\ab\control\build\apps\Release\ninfer-serve.exe
set AB_LAUNCH_BAT=<a launch .bat with the fork flags above>
py -3.11 bench\agentic_ab\runner.py --arms treatment,control --seeds 42,43,44
```

Concurrent decode, one call per build (repeat `--concurrency` to sweep, or alternate single-point
calls between the builds as above):

```bat
py -3.11 tools\bench\run_serve_concurrency.py --serve build-windows\apps\Release\ninfer-serve.exe ^
  --artifact q38=qwen3_8_27b_nvfp4-official.ninfer --mode dflash2_7 --suite decode-saturation ^
  --max-context 32768 --kv-capacity auto --concurrency 1 --concurrency 8 --output profiles\bench\cc-fork
```

Upstream uses the same command with the control checkout's own copy of the script and its
`ninfer-serve.exe`, because its server writes an older request-log schema. On Windows that copy
needs the fork's `wait_for_final_throughput` (commit [`2838371`][c-serve-concurrency]),
which reads the final statistics interval that Windows otherwise loses when the server is stopped.

## What this fork changes

Each topic lists everything that affects it, whether written here or taken from elsewhere.
"Upstream PR" means an open pull request on `Neroued/ninfer` that this fork merged before upstream
did.

**Picking individual changes.** This fork's
[history](https://github.com/Wallawalla47/ninfer-custom/commits/master) is upstream `master`
(`d44ab584`), then the Windows port, then one commit per fork change in dependency order, each
a whole feature with its fixes folded in. Each item below links to its commit, and each commit
message lists the earlier fork commits it builds on, so a change can be cherry-picked into another
fork together with those prerequisites.

### Hybrid prefix cache (the default)

Commit: [`a4665be`][c-hybrid].

Designed around how Qwen3.5-family models work: most of their layers are linear-attention (GDN)
layers, whose recurrent state cannot be rebuilt from the KV cache, so resuming a prompt needs the
KV of every earlier token plus a saved state at the exact token where the new prompt continues.
The cache keeps the two apart and stores each as cheaply as it can. The
[design document](docs/maintainer/hybrid-prefix-cache-spec.md) explains the design and records
every decision taken while building and running it, with the measurements behind it and the
alternatives that were tried and reverted.

- **KV is cached per 64-token block, keyed by content** (its tokens, any image in it, and the
  block before it), in a radix tree. A shared system prompt is stored once and costs its GPU pages
  once at any concurrency.
- **Saved model state is sparse.** Snapshots are taken only at useful points: the end of the
  system prompt and tools, client cache breakpoints, the start of the assistant reply, the end of
  each answer, and a few points spread back through long history. Most cost no extra prefill work
  because they fall on prefill chunk boundaries.
- **Three tiers.** Free VRAM after the model becomes GPU block cache (`--kv-capacity` defaults to
  `auto`); `--host-cache-mib` (default 8192, `0` = GPU only) is one pinned host RAM pool that
  blocks and snapshots share, split by how much prefill time each entry saves; and
  `--prefix-cache-file PATH` saves the host tier on shutdown and reloads it at startup (a smaller
  host tier keeps the most valuable entries). A file from a different model, KV format or
  `ninfer-serve` build is ignored and replaced.
- **Restores overlap the request's own work.** Host RAM copies run on a separate stream in layer
  order and each layer waits only for its own data, so a long restored context costs little more
  than its new tokens.
- **Parallel requests with a new shared prefix prefill it once.** Later requests wait for the
  first one's snapshot where the prompts diverge. Four requests with a new 13.9K-token system
  prompt: mean time to first token 1.48 s instead of 3.52 s.
- **Host eviction keeps what the next turns reuse**, and **a request waiting for a lane prefetches
  its host-only blocks**: at the 52 GB production host tier, 9.2 % fewer prompt tokens prefilled
  and a 7.8 % shorter agentic workload.
- Everything except `--host-cache-mib` is derived from `--max-concurrency` and `--prefill-chunk`;
  `--device-snapshot-slots`, `--cache-taps-per-request`, `--cache-tap-ladder` and
  `--cache-tap-min-gap` are optional overrides.
- **Ctrl+C stops cleanly and saves the cache**: the first press asks for confirmation, the second
  answers running and queued requests with 503 and saves; one more press exits without saving.
  Commit: [`b14f8d6`][c-ctrl-c-stop].

### Original prefix cache: `--use-original-prefix-caching`

Upstream's checkpoint catalog (a saved state plus the KV at that exact point), kept with this
fork's fixes. It is configured with `--host-cache-mib` or upstream's separate capacity flags,
which require `--use-original-prefix-caching`. Details:
[resource scheduling and context cache](docs/maintainer/resource-scheduling-and-context-cache.md).

- **GPU KV grows with the answer instead of being reserved up front**: a request reserves its
  prompt plus a 4,096-token window and extends it as the answer grows, freeing just enough idle
  cache when the pool is full. Commit: [`3614562`][c-kv-lease].
- **Eviction takes the oldest entries first, only as many as needed, and moves them to host RAM
  before deleting them.** Commit: [`b3c6442`][c-eviction].
- **A checkpoint loses its value only when its own conversation has moved past it.** Builds on
  upstream PR #300 by [pkochubey](https://github.com/pkochubey) (upstream issue
  [#178](https://github.com/Neroued/ninfer/issues/178)). Commit: [`dbb1964`][c-lineage-value].
- **One host RAM setting, `--host-cache-mib`**, sizes the saved-state pool, the long anchors and
  host KV. Commit: [`a5c4039`][c-host-budget].
- **Long anchors are placed automatically** at message boundaries, spaced further apart further
  back (`--long-anchor-spacing`). Commits: [`b83ac4b`][c-anchors],
  [`99e0552`][c-anchor-spacing].
- **Aborted requests keep their prefilled prefix**, so a retry carries on from there.
  Commit: [`1dac1a2`][c-salvage].
- **More time to find a cache plan** (5 ms to 250 ms with the request's cost; upstream issue
  [#229](https://github.com/Neroued/ninfer/issues/229), approach suggested by Gene0Liu), and **the
  shared-prefix list no longer fills up for good** (upstream issue
  [#251](https://github.com/Neroued/ninfer/issues/251), approach suggested by albertov).
  Commits: [`16fced4`][c-planning-budget], [`65b168e`][c-shared-catalog].
- **The default shared-prefix catalog is sized for a request's full candidate set** (upstream PR
  #274 by [giveen](https://github.com/giveen)); **cache planning cannot race with itself** (by
  [Gideon Zenz (gzenz)](https://github.com/gzenz)); **no resource-underflow HTTP 500s** when a
  release leaves shared pages exclusive.
  Commits: [`34f7d18`][c-pr274], [`831326f`][c-seal-window],
  [`3f43c97`][c-entitlement].
- **Real-model prefix-reuse scenarios and a smoke runner** cover these fixes.
  Commit: [`ea722fa`][c-prefix-tests].

### Prefill speed

- **Fast INT8 prompt attention** (default for `--kv-dtype int8`): a FlashAttention-2 style kernel
  in which each warp keeps its query rows, scores and output in registers and accumulates P×V in
  FP16. The prefill chunk is rounded down to whole GPU waves (`4096` runs as `3584`). Per attention
  layer it takes 0.61-0.66× the time of upstream's INT8 prompt kernel on 3584-token chunks from an
  empty context to 128K. On the full `ninfer-ppl-1m-v1` corpus it is closer to BF16 KV than the
  original INT8 kernel (4K windows: 4.8986 against 4.9027, BF16 KV 4.8948).
  `--use-original-int8-prefill-kernel` keeps upstream's kernel.
  Commit: [`4c9a949`][c-fast-int8].
- **8-bit P×V in the fast INT8 prompt kernel** (opt-in, `--int8-prefill-8bit-pv`): the kernel
  multiplies P×V on INT8 Tensor Cores (4× the FP16 rate with FP32 accumulation on RTX 5090). Each
  row's probabilities, scaled by the V group scale, are quantized to 8-bit codes per 64-key tile and
  group against the stored INT8 V codes, which stay exact. Per attention layer, 3584-token chunks
  take 3.5 % less time from an empty context and 8-11 % less from 16K to 128K; end to end, prefill
  is 1.5 % faster at a 16K-token prompt and 3.9 % at 64K. It is not a free speed-up: probabilities
  below half a code step of their tile's largest round to zero. Perplexity on the full
  `ninfer-ppl-1m-v1` corpus moved from 4.90771 to 4.90551 with 4K windows and from 4.90412 to
  4.85164 (-1.1 %) with 64K windows; the 64K change is larger than any other kernel change here
  and is a systematic deviation from exact attention, even though it lowers perplexity on this
  corpus. Off by default.
- **Fast NVFP4 prompt attention** (default for `--kv-dtype nvfp4` over more than 2048 cached keys):
  QK runs on block-scaled FP4 Tensor Cores (8× the FP16 rate on RTX 5090) straight from the stored
  K codes, with Q as two NVFP4 terms (0.9 % RMS error); a single chunk whose rows alone would leave
  SMs idle splits its keys across CTAs. 0.34-0.67× the time of upstream's NVFP4 prompt kernel per
  attention layer; end to end, prefill is 3.5 % faster at a 16K-token prompt, 6.4 % at 32K and
  14.4 % at 64K. Perplexity moves by +1.1e-3 nats per token (standard error 2.0e-3).
  `--use-original-nvfp4-prefill-kernel` keeps upstream's kernel.
  Commit: [`8dcd89a`][c-nvfp4-kv].
- **Several requests can prefill at the same time**, overlapping one request's prefill with
  others' prefill and decode. By David Oelfke in the [gzenz/ninfer](https://github.com/gzenz/ninfer)
  fork. Commit: [`25e52f9`][c-concurrent-prefill].
- **`--prefill-round-robin`** (opt-in) serves concurrently prefilling requests in turn and, while
  another request is active, prefills in steps of at most 1024 tokens (896 with the fast INT8
  kernel), so a short prompt or a decoding request beside a long prompt waits one short step
  instead of the whole long prefill. A prompt alone still prefills in whole chunks, and KV capacity
  is unchanged. Based on the round-robin prefill and narrow prefill width by
  [giveen](https://github.com/giveen) in [giveen/ninfer-ext](https://github.com/giveen/ninfer-ext).
  Commit: [`dae362c`][c-prefill-rr].
- **Fused text q/k RMSNorm + RoPE at every width** (14-22 % faster than three separate calls at
  the 3584-token chunk, one sincos per lane), and only for checkpoints with its built-in RoPE theta
  and epsilon. Commit: [`2c8be5e`][c-rope-fused].
- **Kernel tuning from upstream PRs:** the text `rmsnorm_rope` route (#273, Michael Dementii), a
  Q6 34,816×5120 shape for the fused MLP gate_up (#284, [bingchengcc](https://github.com/bingchengcc)),
  and, adapted from [llmq](https://github.com/IST-DASLab/llmq) (IST-DASLab, Erik Schultheis) by
  [DuncanBetts](https://github.com/DuncanBetts), a fused NVFP4 RMSNorm + quantise for the attention
  input projection (#305) and a single-pass target log-probability kernel (#307).
  Commits: [`3681fed`][c-pr273], [`a4c112f`][c-pr284], [`2e68ce8`][c-pr305],
  [`53f51c5`][c-pr307].

### Decode speed

- **Overlapped decode kernels.** Kernels in the decode CUDA Graph launch as programmatic
  dependents of the kernel before them (PDL): weight-streaming kernels load their first weight
  tiles while the previous kernel runs, and release the next kernel only after their own main
  loop. Output is unchanged token for token; greedy decode rounds were 2.2-2.5 % faster when this
  was introduced. Ideas tried and dropped are in [`RESEARCH_NOTES.md`](RESEARCH_NOTES.md).
  Commit: [`0c59ca6`][c-pdl].
- **Decode kernels sized for verification widths**: a third pipeline stage for the NVFP4 down
  projection up to 64 tokens (45.5-47.7 µs instead of 57.7-60.0 µs, and the A4 route from 8
  tokens), and two 16-row tiles sharing each staged activation in the FP8 head and the Q8 DFlash2
  drafter.
  Commits: [`5db3795`][c-nvfp4-linear-add], [`5db53c7`][c-row-tiles].
- **Faster split-KV attention for verification rows** (DFlash2/MTP drafts and n-gram copies):
  short rows split their keys into more, balanced KV splits so they fill the GPU; the split merge
  launches as a programmatic dependent of the attention kernel; and INT8 KV rows of 2-16 columns
  use a kernel that decodes V in registers and keeps the next K/V tile in flight. 8-column INT8
  rows spend 3 % less time in attention at 131K keys and up to 47 % less at 2K, 16-column rows
  21-39 % less at every length, and FP8, K8V4 and NVFP4 rows 9-35 % less at 512-2K keys. Output
  is unchanged except for rows shorter than about 11K-22K keys, whose split merge now rounds in a
  different order. On the NVIDIA artifact a single ~100K-token request decodes 2.4 % faster with
  identical output.
  Commits: [`88df116`][c-split-balance], [`fad95fa`][c-merge-pdl], [`8777710`][c-int8-pipelined].
- **NVFP4 DFlash2 drafter MLP** (`qwen3_8_27b_nvfp4_nvidia` recipe): the drafter's MLP gate/up
  projections are stored as NVFP4 (new `nvfp4_mse` quantizer) instead of Q8 and run with 16-bit
  activations through the fused SwiGLU kernel, which now covers every width (sliced kernels to 32
  tokens, a Tensor Core route above). On the decode-saturation suite (DFlash2 K=7, n-gram 15/12,
  INT8 KV) rounds are 1.6 % shorter with one request (+1.2 % tokens/s) and 0.2 % shorter with four
  (tokens/s within noise). Acceptance moved by -0.35 % on 24 sampled agent prompts and -0.4 %
  (tokens per round) on the suite. Existing artifacts must be reconverted to use it.
- **FP8 LM head for 42-64 tokens per round**: rounds of 42-64 verified tokens (three or four
  requests verifying 16-column n-gram or tree blocks, four verifying 12-column trees) ran the FP8 LM
  head on its 64-token MMA schedule; they now stay on the sliced-K kernel, with two K warps and a
  double-buffered stage above 48 tokens. The LM head is 17-21 % faster at 42-48 tokens, 12-15 % at
  49-56 and 3-8 % at 57-64, and unchanged at other widths (it saves 0.12 ms per 12-column tree round
  with four requests). Logits at those widths round in a different order.
- **Reciprocal NVFP4 activation quantizer on the Linear MMA route** (upstream #327 by
  [DuncanBetts](https://github.com/DuncanBetts)): 2-5 % faster at 8-64 tokens; the other A4 routes
  keep the divisions, because opting them in changed the generated text.
  Commit: [`58808ee`][c-pr327].
- **Ngram copy drafting** proposes the next tokens by copying matching text from earlier in the
  context, alongside MTP/DFlash/DFlash2. The single-request version is the original work of
  [remesis](https://github.com/remesis) in the [remesis/ninfer](https://github.com/remesis/ninfer)
  fork (upstream issue [#234](https://github.com/Neroued/ninfer/issues/234)); this fork extends it
  to `--max-concurrency` above 1 and builds each prompt's index while the prompt is prepared, off
  the engine worker. See [ngram copy proposals](docs/ngram.md).
  Commits: [`d2209f6`][c-ngram], [`c5e390b`][c-ngram-concurrency],
  [`c54dacb`][c-ngram-prep].
- **DFlash2 tree verification** (opt-in, `--draft-tree-nodes auto` or a per-batch-size list):
  instead of one proposal path, a round verifies a small tree of proposals that the GPU builds every
  round from the DFlash2 drafter's candidate lattice, spending the extra columns where the drafter
  is least sure. Sampling stays exact (recursive rejection sampling over each node's alternatives),
  and the accepted path is moved onto the main-chain columns so the KV cache, GDN state and drafter
  see an ordinary round. It works with every `--kv-dtype` on artifacts whose GDN input projections
  are single FP8 or NVFP4 matrices. A column buys the same acceptance at any batch size but costs
  more of the round as the batch and the context grow, so `auto` measures both while it runs: the
  round time of each width per batch size and context length, and, from every tree round's accepted
  path, the tokens each narrower tree and the chain would have emitted on the same text. Each round
  then verifies the chain or a tree of K+5 or K+9 columns, whichever gives the most tokens per
  second. Each round computes every tree column's GDN recurrence once, in one depth-first walk per
  row, so 12- and 16-column tree rounds cost 0.4-0.6 % less than replaying every root-to-leaf path
  with one request, 1.7-2.0 % with two and 2.9-4.0 % with four. On the decode-saturation suite
  (DFlash2 K=7, n-gram 15/12, INT8 KV) `auto` decodes an estimated 6.9 % faster with one request and
  4.4 % with two (measured before the walk), and about 2.9 % with three and 2.7 % with four (from
  2.0 % and 0.3 % before the walk and the LM head change above, in the same runs); at 128K tokens of
  context it loses 0.6-2.3 % in a fresh process (a fixed 16-column tree loses 4.2-5.4 %), and on
  text the drafter already predicts it keeps chain verification. Below about 64K tokens the fixed
  table `16,12,12,0` gains up to 2 points more (7-8.5 % with one request and 4-5 % with two on INT8,
  K8V4 and NVFP4 KV) and keeps seeded one-request runs reproducible, which `auto` does not, since
  its choice depends on measured time. See [tree
  verification](docs/maintainer/tree-verification.md).
- **NVFP4 KV groups pick the best of five scales** (NVFP4 K and V, K8V4 V): each 16-value group
  maps its largest magnitude to 6, 4, 4.5, 5 or 5.5 and keeps the scale with the least squared
  error (Four Over Six, arXiv:2512.02010, generalized). RMS error of the 27B model's rotated K rows
  falls from 9.5 % to 8.5 %; perplexity moves within one standard error.
  Commit: [`a79c2cd`][c-nvfp4-targets].

### Tool calls and reasoning output

- **More tool-call formats**: the XML forms emitted by Claude Code and other agent tools
  (`<function name="…">`, `<invoke>`, `<function_calls>`, short `<param>` tags), also while
  streaming. Upstream PR #300 by [pkochubey](https://github.com/pkochubey) (upstream issue
  [#276](https://github.com/Neroued/ninfer/issues/276)). Commit: [`00ef353`][c-xml-tools].
- **Repeated tool-call parameters keep the last value** (upstream PR #299 by
  [adubkov](https://github.com/adubkov)), and **a quoted `</parameter>` stays in the value**
  (upstream PR #318). Commits: [`c09e929`][c-pr299], [`69b1760`][c-pr318].
- **Quoting `</think>` no longer ends the reasoning early**: it only ends the reasoning when a line
  break or the end of the turn follows. Adapted from upstream PR #309 by Fedor Suchkov.
  Commit: [`080af40`][c-think-quote].
- **`--tolerant-tool-calls`** keeps a good call followed by junk, a final call cut off by the
  output limit (if a parameter is complete), repairs a missing `>` after the function name, and
  returns calls to undeclared tools. By David Oelfke in the gzenz/ninfer fork, ported onto this
  fork's parser. The request log's `tool_call_parse.tolerant_recovered` shows when it rescued a
  call (from giveen's giveen/ninfer-ext). Commits: [`9e28ab8`][c-tolerant-tools],
  [`d2ab752`][c-tolerant-recovered].
- **A reasoning effort the chat template rejects renders as its nearest accepted one** (the
  official Qwen3.8 template accepts only low, medium and xhigh), and `--chat-template` gains the
  froggeric v22.5 template. Commits: [`9b7c58b`][c-effort-nearest],
  [`c65c819`][c-chat-template].

### API and client compatibility

- **llama.cpp-style model details on `/v1/models`** (upstream PR #162 by
  [Hector Ramon Jimenez (hecrj)](https://github.com/hecrj)) and **`ignore_eos` on chat
  completions** (upstream PR #197 by [Thireus](https://github.com/Thireus)).
  Commits: [`ad36334`][c-pr162], [`f235041`][c-pr197].
- **GitHub Copilot and other agent-host requests** (`custom` tools, advisory `tool_choice` /
  `strict` / `parallel_tool_calls`, tool names up to 256 bytes, `--usage-chunk-choice`): the
  serving commits of upstream PR #316 by [paq85](https://github.com/paq85) (Damian Sromek).
  Commit: [`36fc03b`][c-pr316].
- **Responses API options used by Codex and Zed Agent** (`reasoning.summary`,
  `include: ["reasoning.encrypted_content"]`): upstream PR #295 by
  [Macasacker](https://github.com/Macasacker), based on an earlier PR by
  [Sha1rholder](https://github.com/Sha1rholder). Commit: [`d03fd36`][c-pr295].
- **`response_format` `json_object` / `json_schema` is accepted** (not enforced), a **tool call
  cut off by the output or context limit is reported as cut off**, and a request ending with an
  assistant message **continues that reply** (thinking off only): upstream PR #300 by
  [pkochubey](https://github.com/pkochubey). Commits: [`d9f6c95`][c-response-format],
  [`ad0600e`][c-cut-tool-call], [`c2e2339`][c-continuation].
- **Anthropic clients such as Qwen Code**: forced, named and strict tool choices are accepted as
  advisory (upstream issue #223), a thinking budget at or above `max_tokens` is accepted, and API
  routes answer under a doubled `/v1` prefix (a base URL ending in `/v1`).
  Commits: [`e56b408`][c-pr223], [`f863ddc`][c-thinking-budget-max],
  [`e49a360`][c-doubled-v1].
- **Claude Code's `thinking.display: "omitted"`** is accepted instead of rejected: Thinking blocks
  come back with empty text and the reasoning in their signature, which is restored when the
  client sends the block back, so retained thinking and prefix-cache reuse are unchanged. Ported
  from giveen's change in giveen/ninfer-ext. Commit: [`ca1f50e`][c-thinking-omitted].

### Stability

- **Out-of-memory no longer stops the engine**: only the affected requests fail and the engine
  carries on with the queue (by David Oelfke in the gzenz/ninfer fork), and **recovery really
  leaves the engine empty**. Commits: [`a8569f4`][c-oom-recovery],
  [`f8f23a2`][c-idle-recovery].
- **An aborted prefill can no longer write into another conversation's cached prefix**: a lane's
  block-table copy still queued from an aborted request is waited for before the next request
  overwrites it, and activated block tables are published on the compute stream (upstream PR #320).
  Commits: [`92d0cf7`][c-kv-fence], [`4291a7e`][c-pr320].
- **Smaller fixes:** a workspace scope opened before an arena reset no longer rolls the next
  phase's allocations back; a request an idle engine can never admit gets 503 instead of 500;
  token-count requests are bounded like generation requests; Windows servers detect clients that
  vanish without closing the connection; a vision overlay suffix is encoded at the right
  position (fork PR #1 by Yunado); an Anthropic stream whose client leaves while it is still
  queued is logged as a disconnect (499) instead of an internal error (500), by Gideon Zenz in
  the gzenz/ninfer fork; and a full main KV pool no longer ends an MTP or DFlash answer early when
  only its draft KV lease needed room, from giveen's giveen/ninfer-ext.
  Commits: [`f67a284`][c-arena-scope], [`edc9785`][c-idle-503],
  [`7936838`][c-count-bound], [`f6af07f`][c-win-keepalive],
  [`22e6ef1`][c-pr1], [`bcac0a8`][c-queued-cancel], [`6ee864c`][c-lease-thin].

### Models, conversion and vision

- **GGUF files as conversion sources**: upstream PR #282 by [giveen](https://github.com/giveen).
  Commit: [`a4f6aab`][c-gguf].
- **NVIDIA ModelOpt NVFP4 and FP8 checkpoints** as conversion sources, with the
  `qwen3_8_27b_nvfp4_nvidia` recipe storing the output head as FP8; **Quasar NVFP4 conversion**
  with DFlash2 heads and an indexed proposal head; **third-party tokenizer settings** rebuilt during
  conversion. Commits: [`33afed8`][c-modelopt], [`de4623a`][c-quasar],
  [`5b73bba`][c-tokenizer].
- **`nvfp4_absmax` and `nvfp4_mse` conversion methods** quantize BF16 sources to NVFP4 for
  16-bit-activation parents; the NVIDIA recipe uses them for the DFlash2 drafter MLP. `nvfp4_mse`
  picks each 16-value group's scale from all 126 E4M3 values by squared error (based on
  giveen's scale sweep in giveen/ninfer-ext): on the drafter's gate projection the relative
  error is 0.0812, against 0.0847 for the earlier best of five targets and 0.0952 for absmax.
  Commit: [`2309ead`][c-nvfp4-mse].
- **A `qwen3_8_27b_q6` recipe** and a `grouped_mse` scale-search method for groupwise
  quantisation; **Q8 MTP** and a **general BF16 GEMM fallback** for shapes without a dedicated
  kernel. Commits: [`a4c112f`][c-pr284], [`afb274c`][c-grouped-mse],
  [`e172c02`][c-q8-mtp], [`961ce1b`][c-bf16-gemm].
- **`--rope-yarn-factor F`** for YaRN context extension (F from 1 to 4, up to 1M tokens of
  context). Commit: [`59de7e3`][c-yarn].
- **`--vision-offload on`** keeps the vision tower in pinned system RAM instead of VRAM and streams
  it to the GPU while an image is encoded, and **`--vision-max-merged N`** bounds the merged vision
  tokens per image or video. Based on the original work by
  [Valeriy Selitskiy (iamwavecut)](https://github.com/iamwavecut), rewritten for this engine.
  Commit: [`5f7350f`][c-vision-offload].

### Windows

- **Native build and run** with MSVC and CUDA (see [Quick start](#quick-start-windows)): static
  CUDA runtime, FFmpeg/curl from vcpkg, non-RDC NVFP4 kernels, TMA descriptors staged into device
  memory by a kernel (launches captured into a decode graph read a copy written once, so decode
  rounds replay no staging kernels), a built-in PNG decoder for the vision path, drive letters in
  converter recipe paths, and a build id in every product binary. The same port without any
  other fork change is the branch `upstream-Windows-Port`. Commits: [`57d77e4`][c-win-extras],
  [`5065bc5`][c-build-id].
- **Linux still builds and runs** (see [Quick start (Linux)](#quick-start-linux)), checked under
  WSL2 Ubuntu 24.04 with CUDA 13.4; the FP64 attention oracle of the tests uses every host core.
  Commit: [`18795ee`][c-oracle-threads].

### Options and console

- **`--log-colours on`** colours the console statistics, and a **session statistics panel**
  beneath the log shows session and last-ten averages of TTFT, cache hit rate, prefill and decode
  speed, the mean decode batch and drafter acceptance (`--log-stats-panel off` removes it). Engine
  messages and FFmpeg's log are ordinary log records that scroll above the panel.
  Commits: [`9ff604a`][c-log-colours], [`94f0ea6`][c-stats-panel],
  [`6fda154`][c-diagnostics], [`3ee3c1b`][c-ffmpeg-log].
- **Grouped `--help`** by category on `ninfer-serve` and the `ninfer` CLI, with separate sections
  for the two prefix caching systems. Commit: [`3c9b129`][c-help].
- **`--vram-headroom-mib N`** sets how much GPU memory `--kv-capacity auto` leaves spare (upstream
  always leaves 1 GiB). Commit: [`1684538`][c-vram-headroom].
- **Measured CUDA Graph allowance**: the KV sizing reserves 64 MiB plus 4 MiB per decode-graph
  executable, measured on an RTX 5090 across every speculative mode and concurrency; startup
  warns if the graphs ever use more. Commit: [`61e082f`][c-graph-allowance].
- **`--thinking-budget-message S`** sets the message inserted when a request reaches its
  `--default-thinking-budget N`. Commit: [`f3aaad7`][c-thinking-message].
- **`--request-log-max-mib N`** rotates the `--request-log-jsonl` file once it reaches `N` MiB,
  keeping `--request-log-keep K` older files (default 4); each new file starts with a copy of the
  server's start record. Based on the rotation by
  [Gideon Zenz (gzenz)](https://github.com/gzenz) in the gzenz/ninfer fork.
  Commit: [`80d73bb`][c-log-rotation].

### Kept in sync with upstream

This fork is rebased onto upstream `master` whenever upstream moves; the latest is `d44ab584`
(September 2026). Where upstream rewrote code this fork had changed, the rebase starts from
upstream's version and carries the fork's change onto it only where an A/B test on the RTX 5090
shows the fork's version is faster. Once upstream adopts a change, it leaves this list.

Dropped in the rebase onto `d44ab584`, with the measurement that decided each:

- **K8V4 prompt and decode kernels**: upstream's reorganized K8V4 attention was faster (decode by
  up to 43 %, prefill from 32K keys and at 1024-token chunks).
- **NVFP4 decode kernels with FP4 Tensor Core QK**: not carried over. They were 0.64-0.92× the time
  of upstream's new NVFP4 decode at long contexts but up to 1.21× at 8K, and porting them means a
  new split-KV family in upstream's routes. The NVFP4 prompt kernel was carried over (above).
- **TMA-staged FP8 prefill GEMM (upstream PR #167 by Michael Dementii)**: faster than upstream's
  new TMA split-K schedules in operator timings (8-16 % on the input projections at 2048-4096
  tokens), but prefill on the NVIDIA NVFP4/FP8 artifact was 1.5-3.5 % slower end to end with it at
  every call site from 2048 tokens, and 0.5-0.8 % slower on the output projections alone.
- **Narrow FP8 A8 decode-width linear_add tiles**: upstream's tuned TMA schedules are as fast at
  17-128 tokens for K=6144 (within 5 %) and faster for K=17408 (up to 24 %).
- **Split-KV attention for short prefill passes and verification widths, and its split-count fix
  at very long contexts**: upstream's grouped split-KV routes now take single-row passes of up to
  256 (BF16, INT8), 192 (NVFP4) or 80 (FP8, K8V4) new tokens, and its 300,001-key test passes.
- **Folding the attention output gate into the reduce (#268 and batched verification)**: worth
  about 1.5-2 µs per full-attention layer (about 0.2 % of a decode step), and upstream's per-format
  kernels would each need the fold.
- **Per-profile CUDA Graph topology classes**: upstream made decode attention update-compatible
  across resource tiers.
- Adopted by upstream: FP8 MMA on the MX datapath (#328 by
  [DuncanBetts](https://github.com/DuncanBetts)).

## Model artifacts

- **[Qwen3.8-27B-NVIDIA-NVFP4-NInferV3](https://huggingface.co/Wallawalla47/Qwen3.8-27B-NVIDIA-NVFP4-NInferV3)**
  (recommended): [nvidia/Qwen3.8-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4), the
  Model Optimizer mixed NVFP4/FP8 checkpoint, converted with the `qwen3_8_27b_nvfp4_nvidia` recipe
  in `tools/convert/official_recipes.py`. The weights are imported bit-exact except the output
  head (NVFP4 → row-scale FP8), with the DFlash2 draft model and an indexed 131,072-row proposal
  head for `--lm-head-draft`.
- **[Qwen3.8-27B-Quasar-NinferV3](https://huggingface.co/Wallawalla47/Qwen3.8-27B-Quasar-NinferV3)**:
  [QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4](https://huggingface.co/QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4),
  the QAT-trained NVFP4 checkpoint, converted with `tools/convert/quasar_nvfp4.py`, with the same
  DFlash2 draft model and proposal head.

Both are single-file `.ninfer` artifacts for an RTX 5090 (`sm_120a`); each Hugging Face page has
the creation outline and conversion report. The [Quick start](#quick-start-windows) launch works
for either.

## Thanks

A big thank you to all the contributors to upstream NInfer and to the forks this one draws on —
[Neroued](https://github.com/Neroued),
[Michael Dementii](https://github.com/MichaelDementii),
[Minnnn](https://github.com/Minnnn),
[DuncanBetts](https://github.com/DuncanBetts),
[Thireus](https://github.com/Thireus),
[remesis](https://github.com/remesis),
[Valeriy Selitskiy (iamwavecut)](https://github.com/iamwavecut),
[Hector Ramon Jimenez (hecrj)](https://github.com/hecrj),
[giveen](https://github.com/giveen),
[bingchengcc](https://github.com/bingchengcc),
[pkochubey](https://github.com/pkochubey),
[paq85](https://github.com/paq85),
[Macasacker](https://github.com/Macasacker),
[Sha1rholder](https://github.com/Sha1rholder),
[adubkov](https://github.com/adubkov),
[Gideon Zenz (gzenz)](https://github.com/gzenz), David Oelfke, Fedor Suchkov,
Yunado, and everyone else whose pull
requests, reviews and commits made this fork possible — and a particular thank you to
**[Neroued](https://github.com/Neroued)** for creating NInfer, maintaining upstream so
well, and for the work this branch builds on.

[c-agentic-ab]: https://github.com/Wallawalla47/ninfer-custom/commit/e2db558508f907457425626872a1e1b3c26c6bde
[c-serve-concurrency]: https://github.com/Wallawalla47/ninfer-custom/commit/2838371d7f243222bf95352e6d7680955de5cce0
[c-ab-rig]: https://github.com/Wallawalla47/ninfer-custom/commit/9189a7e114393be93e35391d68462117ab3b993d
[c-hybrid]: https://github.com/Wallawalla47/ninfer-custom/commit/a4665be499db14b44e6e6efd99b3bf913c00884d
[c-ctrl-c-stop]: https://github.com/Wallawalla47/ninfer-custom/commit/b14f8d6bd05a04f7f69e5127d1520b22c2af06fc
[c-kv-lease]: https://github.com/Wallawalla47/ninfer-custom/commit/3614562450a737ffacc623254ffd81c823ce55fc
[c-eviction]: https://github.com/Wallawalla47/ninfer-custom/commit/b3c64426a0a3d8186fe5a675932cc0c925591eea
[c-lineage-value]: https://github.com/Wallawalla47/ninfer-custom/commit/dbb1964968b91794f50516b8b8b50990f0553944
[c-host-budget]: https://github.com/Wallawalla47/ninfer-custom/commit/a5c4039418b5e88165e985a74bc707ed100c8940
[c-anchors]: https://github.com/Wallawalla47/ninfer-custom/commit/b83ac4bb361e10cd1f3857b56f4d8a28e9f34cd9
[c-anchor-spacing]: https://github.com/Wallawalla47/ninfer-custom/commit/99e05522716fcdf9e19e43bdde4f1734f49ebead
[c-salvage]: https://github.com/Wallawalla47/ninfer-custom/commit/1dac1a236b948aa01ba49728f33a3f0689a08a87
[c-planning-budget]: https://github.com/Wallawalla47/ninfer-custom/commit/16fced46fc28f453c1aec429dbfe7791ae380a4e
[c-shared-catalog]: https://github.com/Wallawalla47/ninfer-custom/commit/65b168ec12f6afb78a319223e3d3d57ccaa576dc
[c-pr274]: https://github.com/Wallawalla47/ninfer-custom/commit/34f7d188f5f2dfb75d1b02f04b84ffe2f646a6fb
[c-seal-window]: https://github.com/Wallawalla47/ninfer-custom/commit/831326fbb68d37ce6898d00c3a3d7ea1ec23e4c8
[c-entitlement]: https://github.com/Wallawalla47/ninfer-custom/commit/3f43c97852baaa75bfd13ffef2121b98b65d5278
[c-prefix-tests]: https://github.com/Wallawalla47/ninfer-custom/commit/ea722fa330c541fb8f3d61d9ef85e634d857fe8d
[c-fast-int8]: https://github.com/Wallawalla47/ninfer-custom/commit/4c9a949eee73b3cfbdcaf79ae87121af46dea177
[c-nvfp4-kv]: https://github.com/Wallawalla47/ninfer-custom/commit/8dcd89a1029a568e48e2387ec37208e0432b9fb8
[c-concurrent-prefill]: https://github.com/Wallawalla47/ninfer-custom/commit/25e52f915a9184ed1c76ec2a77848e6fa87c8b71
[c-rope-fused]: https://github.com/Wallawalla47/ninfer-custom/commit/2c8be5e7ad5467c7bd6936138d28e1680a4fd79a
[c-pr273]: https://github.com/Wallawalla47/ninfer-custom/commit/3681fed90539527f1c1d8bdc1efe472c9730aa30
[c-pr284]: https://github.com/Wallawalla47/ninfer-custom/commit/a4c112fab9a264cc9cfd0552e0a606822dd7ba54
[c-pr305]: https://github.com/Wallawalla47/ninfer-custom/commit/2e68ce8c79391a612fcd0c537393ea811625ec73
[c-pr307]: https://github.com/Wallawalla47/ninfer-custom/commit/53f51c5b758d89a18caaccf9cd19e2960ecca315
[c-pdl]: https://github.com/Wallawalla47/ninfer-custom/commit/0c59ca61b00a641f9164174a868b26402ef0841f
[c-nvfp4-linear-add]: https://github.com/Wallawalla47/ninfer-custom/commit/5db37954cce4689cb7fcfe90ba2b4c93c4243fea
[c-row-tiles]: https://github.com/Wallawalla47/ninfer-custom/commit/5db53c7991dfe420a7b4072a24ef6ad0a0f4c112
[c-split-balance]: https://github.com/Wallawalla47/ninfer-custom/commit/88df116fe8e7b07b77180815b590fd935db3ab44
[c-merge-pdl]: https://github.com/Wallawalla47/ninfer-custom/commit/fad95faa79707ce7e513c805a7ee361a8c28793e
[c-int8-pipelined]: https://github.com/Wallawalla47/ninfer-custom/commit/8777710b8e0e5206d32d519b3d7325c0650140b8
[c-pr327]: https://github.com/Wallawalla47/ninfer-custom/commit/58808ee2d2d0ce35aa8f989d64b9a9e4a251c47d
[c-ngram]: https://github.com/Wallawalla47/ninfer-custom/commit/d2209f60ad3a520ffb1886fe6329183bf66c389b
[c-ngram-concurrency]: https://github.com/Wallawalla47/ninfer-custom/commit/c5e390b1bcc4b25e0ea9d649b7db0e11d1956160
[c-ngram-prep]: https://github.com/Wallawalla47/ninfer-custom/commit/c54dacb01c77a517d1d8127f5bb167a0b74089e9
[c-nvfp4-targets]: https://github.com/Wallawalla47/ninfer-custom/commit/a79c2cd9859aa03c503942be24c5c76b9485e025
[c-xml-tools]: https://github.com/Wallawalla47/ninfer-custom/commit/00ef353a5521bad001bbbcafe8cf336c8ae22cca
[c-pr299]: https://github.com/Wallawalla47/ninfer-custom/commit/c09e929022e7390b7e2b5d098fc4e2c84508872f
[c-pr318]: https://github.com/Wallawalla47/ninfer-custom/commit/69b17600072b954d25a45cdfbf26e4b64e6c30a3
[c-think-quote]: https://github.com/Wallawalla47/ninfer-custom/commit/080af402250ee1bcc4b7686cc3c8b1a476555ed8
[c-tolerant-tools]: https://github.com/Wallawalla47/ninfer-custom/commit/9e28ab819c975bbd05e63729c032f3718f07ae52
[c-effort-nearest]: https://github.com/Wallawalla47/ninfer-custom/commit/9b7c58b48bff12023eb22528c6b82fe49cadfab4
[c-chat-template]: https://github.com/Wallawalla47/ninfer-custom/commit/c65c819d4cd283be6f37ee3eb9fea507634e161b
[c-pr162]: https://github.com/Wallawalla47/ninfer-custom/commit/ad363341c12c06b05127eec587c071c080ec4173
[c-pr197]: https://github.com/Wallawalla47/ninfer-custom/commit/f23504133730b8b59453041f1e4a1f59c2324067
[c-pr316]: https://github.com/Wallawalla47/ninfer-custom/commit/36fc03ba9e0802428688f09b9e621af31a15abaf
[c-pr295]: https://github.com/Wallawalla47/ninfer-custom/commit/d03fd36e8796602a580871c6746a006cc4ca0792
[c-response-format]: https://github.com/Wallawalla47/ninfer-custom/commit/d9f6c959cb3f62eb2df11b4e842b5bd3f1337d51
[c-cut-tool-call]: https://github.com/Wallawalla47/ninfer-custom/commit/ad0600e806502428649432c318728ed5fd83fd05
[c-continuation]: https://github.com/Wallawalla47/ninfer-custom/commit/c2e2339fc5743fbfd61c58c045aff273c9d8ff19
[c-pr223]: https://github.com/Wallawalla47/ninfer-custom/commit/e56b40880fe753180f0364063bc25cd9bda2437a
[c-thinking-budget-max]: https://github.com/Wallawalla47/ninfer-custom/commit/f863ddcb46a3498a8a874ad8ec85733fb14beca0
[c-doubled-v1]: https://github.com/Wallawalla47/ninfer-custom/commit/e49a36074e3051e9b79f9f37a32add67a80031bf
[c-oom-recovery]: https://github.com/Wallawalla47/ninfer-custom/commit/a8569f483dce236208e8d77eb184aa4c6db40ce1
[c-idle-recovery]: https://github.com/Wallawalla47/ninfer-custom/commit/f8f23a2583002e88e5abaf5e25aef082353186a2
[c-kv-fence]: https://github.com/Wallawalla47/ninfer-custom/commit/92d0cf7e8f1b1acdaf3d120cb1c34c4262877370
[c-pr320]: https://github.com/Wallawalla47/ninfer-custom/commit/4291a7eb0ce65f453b8565b1dae0045eda1b0bb9
[c-arena-scope]: https://github.com/Wallawalla47/ninfer-custom/commit/f67a284bb1f35fe9a2b6b4596569e2c8aec8ac68
[c-idle-503]: https://github.com/Wallawalla47/ninfer-custom/commit/edc97851ff97e13bbefd972e7b25740afca6d640
[c-count-bound]: https://github.com/Wallawalla47/ninfer-custom/commit/793683850754459283c203c4a4f28b41e5b0db31
[c-win-keepalive]: https://github.com/Wallawalla47/ninfer-custom/commit/f6af07ff62010c9c06d0e5dc1f0644d75cd6778c
[c-pr1]: https://github.com/Wallawalla47/ninfer-custom/commit/22e6ef1c0988a4892775687d39035c2b76157cfd
[c-gguf]: https://github.com/Wallawalla47/ninfer-custom/commit/a4f6aaba39bf631889fa498c46ac716c71eb4fc7
[c-modelopt]: https://github.com/Wallawalla47/ninfer-custom/commit/33afed8d3e6272f0da63cd4bc207c3281d6016ed
[c-quasar]: https://github.com/Wallawalla47/ninfer-custom/commit/de4623a31827df73c74a46e7eeaf670a334ec434
[c-tokenizer]: https://github.com/Wallawalla47/ninfer-custom/commit/5b73bba1a3252542b4bfbdbe7808800b2697dcc4
[c-grouped-mse]: https://github.com/Wallawalla47/ninfer-custom/commit/afb274c66a7f09636142e0e021e563149bb32ffe
[c-q8-mtp]: https://github.com/Wallawalla47/ninfer-custom/commit/e172c02790ac1316b1cb75e6080f4bf54622fa58
[c-bf16-gemm]: https://github.com/Wallawalla47/ninfer-custom/commit/961ce1b4b82bae7b1451cf5ce700841cd6b581af
[c-yarn]: https://github.com/Wallawalla47/ninfer-custom/commit/59de7e3927ade15912a6fda55420f08a67074a42
[c-vision-offload]: https://github.com/Wallawalla47/ninfer-custom/commit/5f7350f547ff4e7b3b2aad7082120b5507ef18f1
[c-win-extras]: https://github.com/Wallawalla47/ninfer-custom/commit/57d77e4ba144cebeb374da9b7dae24390820bf36
[c-build-id]: https://github.com/Wallawalla47/ninfer-custom/commit/5065bc550df1523d57c37c4e5f988b6b2701cfcb
[c-oracle-threads]: https://github.com/Wallawalla47/ninfer-custom/commit/18795ee43dc32b0c9f5638d113def82f6c31d710
[c-log-colours]: https://github.com/Wallawalla47/ninfer-custom/commit/9ff604abe753d00a925cc9e1a1cbe5e4f25489cd
[c-stats-panel]: https://github.com/Wallawalla47/ninfer-custom/commit/94f0ea643e00d07689d25af2647940b8e4f10682
[c-diagnostics]: https://github.com/Wallawalla47/ninfer-custom/commit/6fda154a5f4dd205328f68d32294b1722666b78c
[c-ffmpeg-log]: https://github.com/Wallawalla47/ninfer-custom/commit/3ee3c1bfc85b6ffb7b774b203253aabe4625d7c6
[c-help]: https://github.com/Wallawalla47/ninfer-custom/commit/3c9b1292e9571d508570ec2ee87658bd8440a4a9
[c-vram-headroom]: https://github.com/Wallawalla47/ninfer-custom/commit/1684538e4cba676dc8e4832b253bdd0f25246f4f
[c-graph-allowance]: https://github.com/Wallawalla47/ninfer-custom/commit/61e082f37a5aba0a23477e1698075daa82e47eb7
[c-thinking-message]: https://github.com/Wallawalla47/ninfer-custom/commit/f3aaad7c3a8e0d6a66746aa6558e5cb05ceeba57
[c-log-rotation]: https://github.com/Wallawalla47/ninfer-custom/commit/80d73bb9ee5d83fa88a4ced068a1ec792bf2bc22
[c-queued-cancel]: https://github.com/Wallawalla47/ninfer-custom/commit/bcac0a84e115341c78ece01e905c9831c95b712e
[c-nvfp4-mse]: https://github.com/Wallawalla47/ninfer-custom/commit/2309ead35d7d2e2b532c88b061f5a29ae811ad33
[c-prefill-rr]: https://github.com/Wallawalla47/ninfer-custom/commit/dae362c9ae56babbbc430fae4bdcfd29d6516d47
[c-thinking-omitted]: https://github.com/Wallawalla47/ninfer-custom/commit/ca1f50edd8754aa468a5a099e6a8dd5141cca902
[c-lease-thin]: https://github.com/Wallawalla47/ninfer-custom/commit/6ee864cd60ba17f41f29f1dfafbc4c807f68ff6c
[c-tolerant-recovered]: https://github.com/Wallawalla47/ninfer-custom/commit/d2ab7524324919dd2319c4b7b6fcf724910a4105

---

## Upstream README (direct copy)

Everything below is a copy of the upstream
[NInfer README](https://github.com/Neroued/ninfer/blob/master/README.md) as of the upstream
commit this fork is rebased on (`d44ab584`), unchanged except for one added link to
the fork's [ngram copy proposals](docs/ngram.md) guide.

# NInfer

> Selected checkpoints. Maximum single-GPU inference performance.

NInfer is a from-scratch C++/CUDA inference engine for Qwen3.5 Dense and MoE architectures on a
single NVIDIA GeForce RTX 5090. It runs text, image, and video prompts through a local CLI or
OpenAI-/Anthropic-compatible HTTP APIs. The runtime is deliberately specialized: one GPU, one
resident model, and one to eight execution lanes fixed at startup.

Five official artifacts are available. The quick-start commands use Qwen3.8-27B NVFP4.

| Model | Weights | Artifact | Download and model card |
|---|---|---|---|
| Qwen3.6-27B | `groupwise-int` | `qwen3_6_27b.ninfer` | [Qwen3.6-27B](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) |
| Qwen3.6-27B | `nvfp4` | `qwen3_6_27b_nvfp4.ninfer` | [Qwen3.6-27B NVFP4](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer) |
| Qwen3.8-27B | `groupwise-int` | `qwen3_8_27b.ninfer` | [Qwen3.8-27B](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) |
| Qwen3.8-27B | `nvfp4` | `qwen3_8_27b_nvfp4.ninfer` | [Qwen3.8-27B NVFP4](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) |
| Qwen3.6-35B-A3B | `groupwise-int` | `qwen3_6_35b_a3b.ninfer` | [Qwen3.6-35B-A3B](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) |

Each v3 `.ninfer` artifact carries model configuration, encoded weights, logical bindings and
frontend resources. Runtime execution uses those facts with the implemented model and Op
capabilities. You can also [convert your own weights](docs/weight-conversion.md), reuse an official
recipe or choose another supported mixture of formats.

The current engine requires v3 artifacts. Existing official v2 downloads can be
[upgraded locally](docs/weight-conversion.md#upgrade-an-existing-v2-artifact) without downloading
the weights again.

## Quick start

NInfer requires 64-bit Linux, an NVIDIA GeForce RTX 5090, a CUDA toolkit supporting `sm_120a`,
CMake 3.28 or newer, a C++20 host compiler, Ninja, `pkg-config`, FFmpeg development libraries
(`libavformat`, `libavcodec`, `libavutil`, and `libswscale`), and `libcurl >= 7.85`.
CUDA 13.1 is the validated development toolkit; CMake does not impose a CUDA version floor.
The build rejects CUDA architectures other than `sm_120a`.

Build the product binaries:

```bash
git clone https://github.com/Neroued/ninfer.git
cd ninfer

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Tests and benchmarks are excluded from the default build. `cmake --preset release` configures
the same product build; `cmake --preset dev` also enables tests and benchmarks and finds a
Python 3 interpreter. Both presets use `build/` and explicitly reset the build options.
Machine-specific compiler and Python paths belong in the ignored `CMakeUserPresets.json`.
See [build organization and configuration](docs/maintainer/build-system.md) for details.

There is no install target or packaged binary distribution; run NInfer from its source build tree.
Python tools run independently of CMake; the standalone HBM probe has its own
[build command](tools/README.md#standalone-hbm-probe).

Download the artifact used by this example with the Hugging Face CLI:

```bash
hf download neroued/Qwen3.8-27B-nvfp4-NInfer \
  qwen3_8_27b_nvfp4.ninfer \
  --local-dir models
```

Start a long-running text/agent server with two execution lanes and prefix caching:

```bash
./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --max-context 240000 \
  --kv-capacity 240000 \
  --max-concurrency 2 \
  --kv-dtype fp8 \
  --device-state-slots 2 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft \
  --preserve-thinking
```

Each request has a 240,000-token logical ceiling. A shared 240,000-token Device KV pool serves
resident requests and retained prefixes. Requests acquire KV pages as execution advances; under
pressure, the scheduler can pause a request and resume it later. The profile provides two extra
Device StateImages and the default shared pinned Host budget: 8 GiB plus eight model StateImages,
used for retained state, KV and pause snapshots.

Send an OpenAI-style request:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Reply with one short sentence."}],
    "max_tokens": 64
  }'
```

Run a one-shot CLI request with a 32,768-token allocation:

```bash
./build/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --prompt "Explain prefill and decode, then give a concise conclusion." \
  --max-context 32768 \
  --max-new 8192 \
  --kv-dtype fp8 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

Answer content is written to stdout. Human-readable startup/runtime diagnostics and the CLI-owned
reasoning, timing, throughput, memory, and speculative-decoding report are written to stderr;
reasoning and the result report remain unprefixed product output. On a terminal, weight
materialization uses one transient progress line followed by a compact Engine-ready summary.
Redirected stderr receives persistent readable progress without terminal control sequences. Use
`--log-level debug` for complete startup detail. Option and local input errors remain direct command
diagnostics. Use `--messages FILE` and `--vision` for structured image/video input; see the
[CLI guide](docs/cli.md) and [committed examples](examples/cli/).

## Resource-aware long-context reuse

A reusable checkpoint combines KV with the complete continuation state at an exact token frontier.
The engine retains completed conversation endpoints and stable input boundaries for multi-turn and
agent reuse. Inactive checkpoints share Device and pinned Host capacity; pressure reclaims retained
resources before pausing resident requests. Paused requests resume from a snapshot or rebuild their
state by replaying already committed tokens.

See [Resource scheduling and context cache](docs/maintainer/resource-scheduling-and-context-cache.md)
for the algorithm and [Serve TTFT benchmark](tools/bench/ttft/) for public-HTTP coverage of hot
reuse, Host resume, eviction, shared prefixes, scheduling boundaries, and multimodal load.

## Performance

Published measurements use an RTX 5090. The [performance index](docs/performance.md) links to
per-model run records and the [measurement rules](docs/performance/methodology.md). The tables
below are excerpts from those detailed results. Qwen3.8 uses FP8 E4M3 row-256 KV;
Qwen3.6 uses INT8 group-64 KV.

### Concurrent MTP3 decode

Saturated decode used CUDA Graphs, MTP3, and one 8,192-token generation per active
request. Throughput uses aggregate committed decode tokens from complete intervals whose actual
decode batch equaled the configured concurrency. Acceptance covers the complete request wave;
these rates are steady decode (tok/s).

| Model profile | C=1 tok/s / accept | C=2 tok/s / accept | C=4 tok/s / accept | C=8 tok/s / accept |
|---|---:|---:|---:|---:|
| [Qwen3.6-27B](docs/performance/qwen3.6-27b.md#decode-saturation) `groupwise-int` | 185.8 / 68.2% | 247.0 / 69.0% | 309.5 / 68.4% | 535.0 / 68.3% |
| [Qwen3.6-27B](docs/performance/qwen3.6-27b.md#decode-saturation) `nvfp4` | 202.4 / 69.3% | 399.7 / 71.4% | 699.7 / 69.3% | 1,146.9 / 68.6% |
| [Qwen3.6-35B-A3B](docs/performance/qwen3.6-35b-a3b.md#decode-saturation) `groupwise-int` | 642.5 / 68.6% | 907.2 / 66.3% | 1,213.5 / 69.6% | 1,380.7 / 68.0% |
| [Qwen3.8-27B](docs/performance/qwen3.8-27b.md#decode-saturation) `groupwise-int` | 136.5 / 44.4% | 253.3 / 45.2% | 398.1 / 46.1% | 582.4 / 46.4% |
| [Qwen3.8-27B](docs/performance/qwen3.8-27b.md#decode-saturation) `nvfp4` | 147.7 / 46.2% | 291.0 / 48.7% | 522.2 / 45.8% | 922.4 / 46.1% |

### Single-request serving

The serial serving corpus used CUDA Graphs, a 1,024-token prefill chunk, and five
fixed seeds after warm-up. The table keeps one short-prefill, one extreme-prefill, and one
structured-output MTP3 point for each published profile; the full context and scenario matrices are
linked from each model below.

| Model profile | 7,680-token prefill | 260,096-token prefill | Structured MTP3 decode |
|---|---:|---:|---:|
| [Qwen3.6-35B-A3B](docs/performance/qwen3.6-35b-a3b.md#single-request-speculative-decode) `groupwise-int` | 17,705.4 tok/s | 5,247.0 tok/s | 779.6 tok/s |
| [Qwen3.6-27B](docs/performance/qwen3.6-27b.md#single-request-speculative-decode) `groupwise-int` | 3,218.1 tok/s | 1,614.8 tok/s | 193.0 tok/s |
| [Qwen3.6-27B](docs/performance/qwen3.6-27b.md#single-request-speculative-decode) `nvfp4` | 11,191.5 tok/s | 2,510.6 tok/s | 252.2 tok/s |
| [Qwen3.8-27B](docs/performance/qwen3.8-27b.md#single-request-speculative-decode) `groupwise-int` | 3,331.9 tok/s | 2,139.4 tok/s | 214.7 tok/s |
| [Qwen3.8-27B](docs/performance/qwen3.8-27b.md#single-request-speculative-decode) `nvfp4` | 12,819.1 tok/s | 4,016.4 tok/s | 231.7 tok/s |

## Evaluation

Capability scores were measured through NInfer's OpenAI-compatible serving route with thinking
enabled, MTP3, and EvalScope 1.9.0 (0-shot, rule scoring, one sample per problem):

| Model profile | AIME 2025 | AIME 2026 | GPQA-Diamond | ERQA | RealWorldQA |
|---|---:|---:|---:|---:|---:|
| [Qwen3.6-27B groupwise-int](model-cards/Qwen3.6-27B-NInfer/README.md) | 86.67% | 93.33% | 86.87% | — | — |
| [Qwen3.6-27B NVFP4](model-cards/Qwen3.6-27B-nvfp4-NInfer/README.md) | 93.33% | 93.33% | 84.34% | — | — |
| [Qwen3.6-35B-A3B groupwise-int](model-cards/Qwen3.6-35B-A3B-NInfer/README.md) | 90.00% | 90.00% | 85.35% | — | — |
| [Qwen3.8-27B groupwise-int](model-cards/Qwen3.8-27B-NInfer/README.md) | 96.67% | 96.67% | 87.37% | 66.25% | 82.22% |
| [Qwen3.8-27B NVFP4](model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md) | 96.67% | 96.67% | 90.40% | 66.25% | 83.53% |

The Qwen3.6 rows used temperature 0.6 and presence penalty 1.0; the Qwen3.8 rows used temperature
1.0 and presence penalty 0.0. Multimodal evaluation used `--vision` and an 81,920-token context
limit. Text evaluation used 262,144 tokens except Qwen3.8-27B NVFP4, which used 252,928 tokens to
fit the RTX 5090 after weights. Each score is one sample per problem; model cards contain the
correct/total counts and evaluation notes.

## Startup notes

GPU residency is fixed at process startup. `--spec` selects speculative decoding residency, and
`--vision` independently selects Vision residency. Qwen3.6-35B-A3B DFlash can be combined with
Vision; it accelerates generated-text decode after multimodal prefill, not Vision encode itself.

## Docker

Build the runtime image on a host with the NVIDIA Container Toolkit:

```bash
docker build --tag ninfer:local .
```

Mount the downloaded model and run the same example server profile:

```bash
docker run --rm \
  --gpus '"device=0"' \
  --publish 8080:8080 \
  --volume "$PWD/models:/models:ro" \
  ninfer:local \
  ninfer-serve /models/qwen3_8_27b_nvfp4.ninfer \
  --host 0.0.0.0 \
  --max-context 240000 \
  --kv-capacity 240000 \
  --max-concurrency 2 \
  --kv-dtype fp8 \
  --device-state-slots 2 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft \
  --preserve-thinking
```

## Capabilities and limits

The official artifacts provide the following capabilities, with optional components enabled at startup:

- text generation with thinking and non-thinking prompt modes;
- image, multi-image, video, and mixed multimodal messages;
- chunked prefill, exact-batch CUDA Graph decode, and startup-bounded batched decode;
- MTP speculative decoding with draft windows from one to five;
- BF16, INT8, FP8, NVFP4, and K8V4 KV storage;
- offline causal-perplexity scoring;
- private and shared exact-prefix reuse with Device/Host State and KV retention;
- model-aware sampling defaults and explicit sampler overrides;
- OpenAI Responses Core, OpenAI Chat Completions, and Anthropic Messages, including streaming,
  tools, local response state, token counting, and usage accounting.

The 35B-A3B target additionally supports DFlash with draft windows from one to fifteen for Text and
image/video Vision prompts. Qwen3.8-27B artifacts with the DFlash2 companion weights support
`--spec dflash2 --draft-tokens 7` for the same Text/Vision Engine path, with draft counts 1..15
and either full or optimized proposal heads.

The product boundary remains intentionally small:

- one RTX 5090 and one resident model per Engine;
- one to eight resident execution lanes with bounded FIFO ingress;
- resource-pressure preemption with snapshot or token-replay recovery;
- no priority/QoS, weight offload, multi-GPU, or distributed serving;
- one shared startup-fixed KV pool across active requests and retained prefixes;
- model architectures and format/shape combinations use explicitly implemented native paths;
- parsed tool calls are returned to the client; NInfer does not execute tools;
- the in-tree C++ headers are not distributed as an installed SDK.

`--max-context` is each sequence's logical limit. `--kv-capacity` sizes the shared Main Text KV pool
used by active requests and retained prefixes; `auto` resolves the largest legal capacity at
startup from the memory remaining after weights while keeping 1 GiB of sizing headroom. Explicit
capacities remain fixed for the process lifetime.

## Documentation

- [Documentation index](docs/README.md)
- [CLI](docs/cli.md)
- [HTTP serving](docs/serving.md)
- [Ngram copy proposals](docs/ngram.md)
- [Performance](docs/performance.md)
- [Perplexity evaluation](docs/perplexity.md)
- [Weight conversion and custom recipes](docs/weight-conversion.md)
- [Resource scheduling and context cache](docs/maintainer/resource-scheduling-and-context-cache.md)
- [Serve TTFT benchmark](tools/bench/ttft/)
- [CLI examples](examples/cli/)
- [Contributing](CONTRIBUTING.md)

Run the relevant `--help` for the exact current option contract.

## Support

NInfer is a personal project that I develop out of interest. If you find it useful and would like
to support its continued development, you can [support the project on Ko-fi](https://ko-fi.com/neroued).

Support is entirely voluntary. It is not a purchase or investment and does not come with financial
returns, promised services or features, or a role in project decisions. The project's direction,
priorities, technical choices, and release schedule remain independently determined by the
maintainer.

## License

NInfer is licensed under the [Apache License 2.0](LICENSE).

The published artifacts are derived from
[Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B),
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), and
[Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B). The Qwen3.6-27B NVFP4 artifact
also uses the fixed packed weights from
[rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm](https://huggingface.co/rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm).
The Qwen3.8-27B NVFP4 artifact also uses the fixed mixed FP8/NVFP4 weights from
[unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4). These source
repositories are distributed under Apache-2.0. Vendored dependencies retain their own license files
under `third_party/`.
