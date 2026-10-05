# NInfer — custom fork

> **AI disclaimer:** Everything added to this fork, including most of this README, was written with
> AI (mostly Claude Opus 5.5, Qwen3.8-27B running on NInfer, plus a few other AI systems I’ve been
> testing). It is likely to be neither complete nor entirely accurate. This is hobby development.

This is a personal fork of [Neroued/ninfer](https://github.com/Neroued/ninfer). It follows upstream
closely and adds changes on top: the history is upstream `master` (`68c54356`), the Windows port,
then one commit per fork change. The sections below explain what is different, grouped by topic,
with credit given as best my AI agents can where a change came from someone else. The upstream
README follows, copied unchanged, under the "Upstream README" heading. A huge thank you to Neroued
for creating NInfer!

**The short version.** Compared with upstream, this fork:

1. builds and runs natively on Windows (and still on Linux)
2. uses a new prefix caching system, designed and implemented by Claude Opus 5.5, as the default.
   You set how much system RAM it may use with `--host-context-mib N`; `--prefix-cache-file PATH`
   keeps the cache across restarts. It still beats upstream's new context cache (October 2026):
   on the agentic benchmark below it serves 90.5 % of prompt tokens from cache against 85.8 %,
   prefills a third fewer prompt tokens, cuts the average time to first token by 39 % and finishes
   the workload 26 % sooner. Stop the server with Ctrl+C (twice) rather than by
   closing the window, because Windows does not always leave enough time after a window closes to
   save a large cache
3. keeps upstream's own prefix caching (its continuation/checkpoint cache with request preemption
   and replay) behind `--use-original-prefix-caching`
4. prefills INT8, NVFP4, FP8 and K8V4 KV with faster prompt-attention kernels by default (a third
   to two thirds less prompt-attention time on long INT8 and NVFP4 prompts than upstream's kernels,
   a sixth to a half less for FP8 and K8V4); `--use-original-int8-prefill-kernel` and
   `--use-original-nvfp4-prefill-kernel` select upstream's INT8 and NVFP4 kernels; NVFP4 and K8V4
   P×V runs on 8-bit Tensor Cores by default (5-7 % less end-to-end long-prompt prefill, KL
   divergence from BF16 KV within 1.1× the FP16 form's), while INT8 KV keeps FP16 P×V unless
   `--prefill-8bit-pv` asks for its 8-bit form, and `--no-prefill-8bit-pv` forces FP16 everywhere
5. adds two compact KV formats, `--kv-dtype vq2` and `k4v2` (a quarter and three eighths of INT8's
   KV memory, based on HyperQuant and [cometkim](https://github.com/cometkim)'s implementation),
   which keep the most recent keys exact, always run P×V in FP16, and decode faster than INT8 at
   long context
6. adds ngram copy drafting (based on an implementation by [remesis](https://github.com/remesis)),
   which greatly speeds up copy-heavy workloads, with more than one concurrent request
7. overlaps each decode kernel's launch and weight loading with the kernel before it, and tunes
   the decode-width kernels, for faster speculative decode rounds with the same output
8. can offload the vision encoder to system RAM (`--vision-offload on`), based on the work of
   Valeriy Selitskiy ([iamwavecut](https://github.com/iamwavecut))
9. supports YaRN context extension up to 1M tokens (`--rope-yarn-factor F`, F from 1 to 4)
10. sizes the CUDA Graph memory allowance from measurement instead of an estimate that could reserve
    far more VRAM than ever used
11. lets `--vram-headroom-mib N` shrink the 1 GiB VRAM headroom left after `--kv-capacity auto`
12. adds a custom thinking budget message (`--default-thinking-budget N` with
    `--thinking-budget-message "..."`)
13. accepts more tool-call formats and API options used by agent clients such as Claude Code, Qwen
    Code, Codex, Zed and GitHub Copilot, and fixes several tool-call and reasoning-output issues
    (mostly based on the work of others credited below); `--tolerant-tool-calls` recovers some
    broken tool calls
14. improves the console: optional colours (`--log-colours on`), a statistics panel at the bottom
    (`--log-stats-panel off` removes it) and a `--help` screen organised by category, and can
    rotate the request log by size (`--request-log-max-mib N`)
15. contains various other fixes and improvements, including upstream pull requests merged before
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
ninfer-serve.exe qwen3_8_27b_nvfp4-nvidia.ninfer --host 127.0.0.1 --port 8080 --max-context 240000 --max-concurrency 2 --spec dflash2 --draft-tokens 7 --lm-head-draft --ngram-draft-tokens 15 --ngram-min-match 12 --kv-dtype int8 --preserve-thinking --host-context-mib 52000 --pending-timeout-ms 900000 --prefill-chunk 4096 --kv-capacity auto --vram-headroom-mib 0 --log-colours on --ngram-archive-mib 2048 --ngram-session-mib 256 --ngram-native-sessions --request-log-jsonl log.json --default-thinking-budget 16384 --thinking-budget-message "Considering the limited time available to the user, I must stop thinking now. Time to act:" --tolerant-tool-calls
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

The agentic benchmark and perplexity below compare this fork with **upstream + Windows port**:
upstream `master` at `abb7f14f`, which replaced upstream's prefix cache with its new context cache
(incremental KV reservation, preemption and replay), plus only the Windows port (build `07f7a944`
of the branch `upstream-Windows-Port`). The measured fork build is `580c91ea`: this history before
the rebase onto `68c54356`, whose one upstream commit changes only how the chat template keeps
cache boundaries when it trims text. A third arm runs the fork as it was before the rebase onto
`abb7f14f` (`cfe8d4d4`, its own cache on upstream's previous engine), to check that the rebase lost
nothing. Everything ran on an RTX 5090 under Windows with the official Qwen3.8-27B NVFP4 artifact
(`qwen3_8_27b_nvfp4-official.ninfer`), in October 2026.

**The fork's hybrid prefix cache still beats upstream's new cache**: it serves 90.5 % of prompt
tokens from cache against 85.8 %, prefills 33 % fewer prompt tokens, answers 39 % sooner on
average (48 % at the median) and finishes the workload 26 % sooner. Upstream's new cache closed
much of the gap to its previous one (69 % served from cache in September's run), but it still
re-prefilled whole prompts on main-session turns where the fork reused them.

### Agentic coding workload

The closed-loop suite in [`bench/agentic_ab/`](bench/agentic_ab/README.md) replays three
coding-agent sessions plus eleven subagents: 130 requests with fan-outs, a concurrent subagent
pair, compaction, retries, an abort and a solo wrap-up, with prompts of 25K-135K tokens and
thinking on. Each arm's own answers are fed back as an agent client does, and the three main
sessions take their turns in lock-step rounds, so every build meets the same order of session
turns whatever its speed. Every build completed every request on each of three workload seeds
(42, 43, 44), which replay different observations.

Settings:

- **Fork arm:** the [Quick start](#quick-start-windows) launch flags:

  ```text
  --max-context 170000 --max-concurrency 2 --spec dflash2 --draft-tokens 7 --lm-head-draft
  --ngram-draft-tokens 15 --ngram-min-match 12 --kv-dtype int8 --preserve-thinking
  --host-context-mib 52000 --pending-timeout-ms 900000 --prefill-chunk 4096 --kv-capacity auto
  --vram-headroom-mib 0 --ngram-archive-mib 2048 --ngram-session-mib 256 --ngram-native-sessions
  --default-thinking-budget 16384 --thinking-budget-message "Considering the limited time
  available to the user, I must stop thinking now. Time to act:" --tolerant-tool-calls
  ```

- **Upstream arm:** the same flags minus those upstream does not have; upstream's
  `--host-context-mib 52000` gives its new cache the same host RAM:

  ```text
  --max-context 170000 --max-concurrency 2 --spec dflash2 --draft-tokens 7 --lm-head-draft
  --kv-dtype int8 --preserve-thinking --host-context-mib 52000 --pending-timeout-ms 900000
  --prefill-chunk 4096 --kv-capacity auto --default-thinking-budget 16384
  ```

- **Context:** 170,000 tokens, the largest context the upstream build starts with under these
  flags; at it the fork's device KV holds 223,424 tokens and upstream's 175,424. **Sampling:**
  temperature 1.0, top_p 0.95, top_k 20, `max_tokens: 64000` on agent turns.

Each cell is the mean over the three seeds, with the lowest and highest seed in brackets; changes
are computed per seed against that seed's upstream run and averaged.

| Metric | Upstream + Windows port | Fork | Fork before the rebase |
|---|---|---|---|
| Average time to first token (s) | 4.2 (3.7-4.8) | 2.5 (2.2-2.7), −39.2 % | 3.2 (3.0-3.2), −23.5 % |
| Median time to first token (s) | 1.21 (1.09-1.31) | 0.63 (0.60-0.69), −47.5 % | 0.75 (0.54-0.87), −38.4 % |
| 90th-percentile time to first token (s) | 11.4 (9.6-13.4) | 7.5 (6.6-8.1), −32.2 % | 7.5 (6.9-8.5), −33.1 % |
| Average TTFT, continuing-session turns (s) | 4.00 (3.37-4.75) | 2.26 (1.86-2.54), −41.4 % | 3.18 (3.05-3.29), −18.8 % |
| Average TTFT, new long prompts (s) | 10.0 (9.9-10.0) | 6.5 (6.4-6.6), −34.5 % | 6.7 (6.6-6.8), −32.8 % |
| Prompt tokens served from cache | 85.8 % (85.1-87.0) | 90.5 % (89.9-91.2) | 90.9 % (90.6-91.3) |
| Prompt tokens prefilled | 817K (748-863K) | 547K (518-587K), −32.6 % | 526K (522-531K), −35.3 % |
| Main-session turns that re-prefilled the whole prompt (of 75) | 1.3 (1-2) | 0 | 0 |
| Subagent turns that re-prefilled the whole prompt (of 37) | 0 | 0 | 0 |
| Prefill tok/s, requests with no cache hit in any arm | 6,234 (6,152-6,306) | 8,482 (8,347-8,558), +36.1 % | 8,465 (8,360-8,589), +35.8 % |
| Prefill tok/s, the same requests from 32K tokens | 6,052 (6,020-6,089) | 8,316 (8,251-8,357), +37.4 % | 8,300 (8,246-8,388), +37.2 % |
| Output tok/s, one request decoding | 212 (208-216) | 219 (210-237), +3.4 % | 220 (198-258), +3.7 % |
| Decode rounds/s, one request decoding (engine speed) | 58.9 (58.3-59.8) | 61.7 (61.4-62.2), +4.7 % | 61.4 (60.9-61.7), +4.2 % |
| Tokens per round, one request decoding (acceptance) | 3.60 (3.53-3.69) | 3.55 (3.39-3.85) | 3.58 (3.22-4.19) |
| Output tok/s, two requests decoding (combined) | 360 (346-370) | 348 (339-359), −3.4 % | 376 (359-395), +4.5 % |
| Decode rounds/s, two requests decoding (engine speed) | 55.2 (54.9-55.5) | 54.9 (52.7-56.4), −0.7 % | 56.3 (54.9-57.7), +1.9 % |
| Decode rounds that ran two requests | 29.2 % (29.1-29.4) | 40.9 % (36.9-43.0) | 38.3 % (31.1-42.0) |
| Output tok/s, all decoding at the run's own batching | 254 (253-256) | 280 (273-283), +10.2 % | 281 (271-301), +10.7 % |
| Workload wall time (min) | 15.4 (13.1-18.1) | 11.1 (9.8-12.0), −26.0 % | 11.5 (10.1-12.3), −24.5 % |

- Time to first token includes queueing: up to seven requests are in flight on two lanes.
- Output tok/s counts decode tokens per second of the engine's own decode time. It splits into
  decode rounds/s (engine speed) and tokens per round (speculative acceptance, which moves with
  what the model happened to write). The fork's ngram drafting supplied 8-9 % of its output;
  upstream has none.
- **Two requests decoding is the one place the fork does not win.** Its rounds carry ngram
  drafting for both requests, which costs host time per round, and it is 0.7 % slower than
  upstream per two-request round on average. Against the fork before the rebase, it was +2.7 %,
  −1.4 % and −8.7 % per seed; the −8.7 % seed had only 49 s of two-request decoding, so whether
  the rebase costs anything here is unresolved. Each arm ran only 49-160 s of two-request decoding
  per seed. The concurrent decode benchmark below compares two-request decode without ngram
  drafting.
- Perplexity on a fixed 409,687-token corpus (16K context, 8K stride): the fork 2.30073 with INT8
  KV and 2.31073 with NVFP4 KV, bit-identical to the fork before the rebase; upstream 2.30836
  (+0.33 %) and 2.32046 (+0.42 %).

### Concurrent decode

The decode-saturation suite of `tools/bench/run_serve_concurrency.py` starts a fresh server at
each `--max-concurrency` C and decodes C requests at once, each up to 8,192 tokens, with DFlash2
K=7 and `--lm-head-draft` (no ngram drafting), stochastic sampling, `--max-context 32768
--kv-capacity auto`. The builds alternated point by point in two passes, C=1 to 8 and back. This
benchmark was measured in September 2026 against upstream `d44ab584` + Windows port and the fork
rebased on it (build `fe869b56`), and has not been repeated since the rebase onto `abb7f14f`.

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

Commits: [`b1d99ca`][c-agentic-ab], [`021e3cd`][c-serve-concurrency],
[`a94c095`][c-ab-rig].

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
needs the fork's `wait_for_final_throughput` (commit [`021e3cd`][c-serve-concurrency]),
which reads the final statistics interval that Windows otherwise loses when the server is stopped.

## What this fork changes

Each topic lists everything that affects it, whether written here or taken from elsewhere.
"Upstream PR" means an open pull request on `Neroued/ninfer` that this fork merged before upstream
did.

**Picking individual changes.** This fork's
[history](https://github.com/Wallawalla47/ninfer-custom/commits/master) is upstream `master`
(`68c54356`), then the Windows port, then one commit per fork change in dependency order, each
a whole feature with its fixes folded in. Each item below links to its commit, and each commit
message lists the earlier fork commits it builds on, so a change can be cherry-picked into another
fork together with those prerequisites.

### Hybrid prefix cache (the default)

Commit: [`2184a98`][c-hybrid].

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
  `auto`); `--host-context-mib` (default 8192, `0` = GPU only) is one pinned host RAM pool that
  blocks and snapshots share, split by how much prefill time each entry saves; and
  `--prefix-cache-file PATH` saves the host tier on shutdown and reloads it at startup (a smaller
  host tier keeps the most valuable entries). A file from a different model, KV format or
  `ninfer-serve` build is ignored and replaced.
- **Restores overlap the request's own work.** Host RAM copies run on a separate stream in layer
  order and each layer waits only for its own data, so a long restored context costs little more
  than its new tokens.
- **Parallel requests with a new shared prefix prefill it once.** Later requests wait for the
  first one's snapshot where the prompts diverge. Four requests with a new 13.9K-token system
  prompt: mean time to first token 1.48 s instead of 3.52 s. A shared prefix may hold images:
  requests that carry the same image (by content) behind the same text wait for the first one's
  snapshot past it, so the image is encoded once.
- **Host eviction keeps what the next turns reuse**, and **a request waiting for a lane prefetches
  its host-only blocks**: at the 52 GB production host tier, 9.2 % fewer prompt tokens prefilled
  and a 7.8 % shorter agentic workload.
- Everything except `--host-context-mib` is derived from `--max-concurrency` and `--prefill-chunk`;
  `--device-snapshot-slots`, `--cache-taps-per-request`, `--cache-tap-ladder` and
  `--cache-tap-min-gap` are optional overrides.
- **It runs on upstream's execution engine.** A cached path is one more binding source for
  upstream's lane binding, so requests grow their KV unit by unit, cached blocks are evicted first
  when growth runs short, and only then is a younger request paused and later replayed.
- **Ctrl+C stops cleanly and saves the cache**: the first press asks for confirmation, the second
  answers running and queued requests with 503 and saves; one more press exits without saving.
  Commit: [`2184a98`][c-ctrl-c-stop].

### Original prefix cache: `--use-original-prefix-caching`

Upstream's own context cache: private continuations and shared checkpoints with incremental KV
reservation, resource-pressure preemption and snapshot or replay recovery. `--device-state-slots`
requires `--use-original-prefix-caching`, and `--host-context-mib` sizes its shared Host budget.
Details: [resource scheduling and context cache](docs/maintainer/resource-scheduling-and-context-cache.md).

The fixes this fork carried for upstream's previous checkpoint-catalog cache (answer-sized KV
leases, recency eviction, lineage value, a single host budget, automatic long anchors, prefix
salvage on abort, a longer planning budget and shared-catalog reclaim) were retired with that cache
when upstream replaced it in `abb7f14f`.

### Prefill speed

- **Fast INT8 prompt attention** (default for `--kv-dtype int8`): a FlashAttention-2 style kernel
  in which each warp keeps its query rows, scores and output in registers and accumulates P×V in
  FP16. The prefill chunk is rounded down to whole GPU waves (`4096` runs as `3584`). Per attention
  layer it takes 0.61-0.66× the time of upstream's INT8 prompt kernel on 3584-token chunks from an
  empty context to 128K. On the full `ninfer-ppl-1m-v1` corpus it is closer to BF16 KV than the
  original INT8 kernel (4K windows: 4.8986 against 4.9027, BF16 KV 4.8948).
  `--use-original-int8-prefill-kernel` keeps upstream's kernel.
  Commit: [`60e67d0`][c-fast-int8].
- **Key splits for short INT8 prompt chunks**: a chunk of up to about 1300 tokens over a long
  cached prefix leaves SMs idle with one CTA per row block, so the fast INT8 kernel divides its keys
  among eight-warp CTAs and merges their FP32 partial rows, as the NVFP4 kernel does. It splits
  only when the time saved outweighs writing and merging the partial rows (a cost model fitted to
  every split count measured at 0-64K keys), so chunks over less than about 1K keys and widths whose
  row blocks already fill the GPU stay unsplit. Per attention layer against the kernel before
  splits: 257-384-token chunks take 4-7 % less time over 1K cached keys, 10-14 % over 2K, 15-19 %
  over 4K and 28-37 % over 32K-128K; 512 tokens 6-34 % less from 2K keys, 576-640 tokens 7-20 % less
  from 8K and 1024-1280 tokens 8-23 % less from 4K. Slower: 448 tokens over 8K keys (+2.9 %), 576
  over 4K (+1.9 %) and 1024-1280 over 2K (+1.9 to +2.7 %); other widths, including full 3584-token
  chunks, are within 1 %. End to end, a 267-token follow-up over a 128K-token cached prefix
  prefilled 21 % faster, and 550-620-token follow-ups 12 % faster (5 % over 32K; measured with an
  earlier plan that splits these widths the same way); uncached prompts, full documents and longer
  follow-ups were unchanged within about 2 %. Perplexity on the full `ninfer-ppl-1m-v1` corpus moved
  from 4.9077081 to 4.9076804 with 4K windows and from 4.9041197 to 4.9041624 with 64K windows, each
  within 0.7 standard errors of the per-window differences. The partials took up to 64 MiB of
  workspace: 576 fewer KV tokens (0.23 %) at startup with `--max-context 140000`; their bound is now
  `--prefill-split-workspace-mib` (below).
- **8-bit P×V in prompt attention** (default for NVFP4, K8V4 and VQ2 KV; `--prefill-8bit-pv`
  turns it on for INT8 and K4V2 KV, which default to FP16 P×V, and `--no-prefill-8bit-pv`
  forces FP16 everywhere).
  NVFP4 and K8V4 KV: the MX-FP8 tiled prompt kernel rounds each probability (as 256 p against its
  tile's own row maximum, which the softmax reference follows) and each decoded V value to E4M3 and
  multiplies them on block-scaled E4M3 Tensor Cores with FP32 accumulation, four times the rate of
  upstream's FP16 P×V with FP32 accumulation; a per-tile power of two keeps V in E4M3's range and
  returns through the MMA's block scale. Its error against exact attention over the stored values is
  about 15 times the FP16 form's (relative L2 0.026-0.037 against 0.0017), but the divergence it
  causes does not follow that: on a Q6 artifact with BF16 activations, where the measurement is not
  buried under A4 activation noise, its KL divergence from a BF16 KV reference over the full corpus
  at 64K context is 0.0203 against the FP16 form's 0.0205 (NVFP4) and 0.0141 against 0.0134 (K8V4),
  within 1.1× of FP16 in every context bucket and within 0.06 pp of top-1 agreement. Per attention
  layer, against the FP16 form, 3584-token chunks take 19 % (K8V4) and 18 % (NVFP4) less time over
  128K keys and up to 20 % less elsewhere; end to end, one request's prefill (DFlash2, 4096-token
  chunks) is 4.7-5.1 % (NVFP4) and 5.0-5.4 % (K8V4) faster at 64K tokens and 6.8-7.1 % and 7.5-7.7 %
  at 128K (two runs differing by 0.4-0.7 pp).
  Perplexity on the full `ninfer-ppl-1m-v1` corpus: 4.89624 → 4.91573 (K8V4) and 4.91651 → 4.91384
  (NVFP4) with 4K windows, 4.84164 → 4.97411 and 4.89965 → 4.87335 with 64K windows; the median
  stream moved by under 0.002 nats per token, but at 64K single streams moved by up to 0.36 (K8V4) —
  the corpus moves that much for any small numeric change, which is why the KL comparison above is
  the direct measure.
  VQ2 and K4V2 KV: the vector-quantized prompt kernel runs the same integer form over the tile's
  INT8 V rows, scaling each group's probabilities by that key's row scale and rounding them to
  u8 codes against their row maximum. VQ2 takes it by default: against a BF16 KV reference on a
  BF16-activation model its KL divergence moves 0.0367 → 0.0378 (1.03x, every bucket within
  1.05x, top-1 within 0.04 pp) while prefill of one request is 5.0 % faster at 64K tokens and
  6.8 % at 128K. K4V2 stays opt-in: its divergence moves 0.0212 → 0.0232, 1.09x overall but
  1.11x in the 32-64K bucket, against the 1.10x bound the other formats were held to, for
  5.3 % and 7.2 % the same way. Per attention layer the two are 3.3-18 % faster at 3584-key
  chunks over long context (geomean 0.97 over 256-3584 keys at 0-64K) and up to 5 % slower for
  512-1024-key calls over an empty context.
  INT8 KV (`--prefill-8bit-pv`, off by default): the fast kernel multiplies P×V on INT8 Tensor Cores
  (4× the FP16 rate with FP32 accumulation on RTX 5090); each row's probabilities, scaled by the V
  group scale, are quantized to 8-bit codes per 64-key tile and multiplied against the stored INT8 V
  codes, which stay exact. Per attention layer, 3584-token chunks take 3.5 % less time from an empty
  context and 8-11 % less from 16K to 128K; end to end, prefill is 1.9-2.6 % faster at a 16K-token
  prompt, 2.6-4.2 % at 64K and 4.0-5.1 % at 128K (two runs). Unlike the E4M3 form it is not free: probabilities below
  half a code step of their tile's largest round to zero, and on the same Q6 artifact the KL
  divergence from BF16 KV about doubles, 0.0051 → 0.0111 at 64K, with all sixteen streams worse and
  the growth concentrated in longer context (the 32-64K bucket ×2.28, top-1 98.9 % → 98.6 %). The
  fast kernel's FP16 P×V matches the original INT8 prompt kernel there (0.0051 against 0.0050), so
  the whole cost is the 8-bit P, and FP16 P×V stays INT8's default.
- **Fast NVFP4 prompt attention** (default for `--kv-dtype nvfp4` over more than 768 cached
  keys, re-tuned from 2048 with the decode-once kernel: forced at every visible count the fast
  kernel is within 0.03 % of the per-shape best above 768 visible keys in both head geometries
  (24 query heads x 4 KV heads and 16 x 2), where 2048 costs 2.3-2.7 % overall and up to 64 % in
  its worst cell):
  QK runs on block-scaled FP4 Tensor Cores (8× the FP16 rate on RTX 5090) straight from the stored K
  codes, with Q as two NVFP4 terms (0.9 % RMS error). It now runs on the MX-FP8 tiled kernel (below)
  with NVFP4 keys: each 64-key V tile is decoded once per CTA into shared memory instead of in every
  warp's registers. Per attention layer it takes 6-26 % less time than the previous fast NVFP4
  kernel wherever it runs (3584-token chunks 8 % less over an empty context, 15 % over 8K-32K and
  17 % over 128K keys), and 0.32-0.63× the time of upstream's NVFP4 prompt kernel; end to end,
  prefill of one request (DFlash2) is 2.1 % faster at 16K tokens, 3.6 % at 32K, 6.0 % at 64K and
  9.2 % at 128K than with the previous fast kernel, both with the previous kernel's 64 MiB split
  budget; the bound is now `--prefill-split-workspace-mib` (below). Perplexity:
  4.90646 → 4.91651 with 4K windows and 4.90946 → 4.89965 with 64K windows, moves of the size the
  corpus shows for any small numeric change (below). `--use-original-nvfp4-prefill-kernel` keeps
  upstream's kernel.
  Commit: [`205d7f2`][c-nvfp4-kv].
- **Faster FP8 and K8V4 prompt attention** (default): upstream's MX-FP8 tiled prompt kernel now
  accumulates P×V on FP16 Tensor Cores per 64-key tile (FP32 accumulation runs at half their rate on
  RTX 5090) under a per-tile power-of-two V shift, issues the heaviest row blocks first, and chooses
  each launch's key splits by a cost model that counts writing and merging the split partials, which
  upstream's wave count ignored. Per attention layer against master, every width and context
  measured (256-4096 tokens over 0-128K cached keys) is faster: K8V4 5.5-58 % less time (3584-token
  chunks 31 % less over an empty context, 26 % over 8K, 22 % over 32K, 15 % over 128K;
  1408-2048-token chunks about half the time over an empty context), FP8 5.7-59 % less. End to end,
  prefill of one request (DFlash2) is 2-2.4 % faster at 16K tokens, 4 % at 32K, 6 % at 64K and 8.5 %
  (K8V4) and 9.4 % (FP8) at 128K. Perplexity: K8V4 4.91185 → 4.89624 and FP8 4.89797 → 4.89736 with
  4K windows, 4.88656 → 4.84164 and 4.84961 → 4.89144 with 64K windows. These come from one or two
  streams that react strongly to any small numeric change: a control build that differed only by
  FP32 accumulation scored K8V4 4.92107 (4K), and one stream moved by 0.08-0.11 nats per token
  between the three. The median stream moved by under 0.001 nats per token, and the kernel's error
  against exact attention is within 2 % of master's (relative L2 0.0017-0.0018). These measurements
  kept upstream's unbounded split workspace (about 550 MiB at 3328-token launches); it is now bounded
  by `--prefill-split-workspace-mib` (next).
- **One split-workspace bound for prompt attention** (`--prefill-split-workspace-mib`, default
  256 MiB). The fast INT8 and NVFP4 prompt kernels and the FP8/K8V4 prompt kernel split a launch's
  keys across SMs when its row blocks alone would leave SMs idle, which needs FP32 partial rows in
  workspace. INT8 and NVFP4 were capped at 64 MiB and FP8 and K8V4 were unbounded (about 550 MiB with
  4096-token chunks; upstream's default 1024-token chunks need 169 MiB). One bound now applies to all
  four, and a launch that would need more runs fewer splits. Per attention layer against unbounded
  (3 interleaved passes, widths 256-4096 over 0-128K cached keys), 1024-2048-token chunks take
  17-25 % longer at 64 MiB over 32K-128K keys (27 % at worst), 2-5 % at 128 MiB and under 1 % from
  192 MiB, the same for all four formats; at the 256 MiB default 3-4K-token chunks over 128K keys
  still take up to 5 % longer, and 3584-token chunks and short contexts are unaffected. These widths
  are follow-up turns over a cached prefix and `--prefill-round-robin` steps beside another request;
  a fresh 64K or 128K prompt prefills in the same time within 0.3 % end to end. The startup
  workspace with 4096-token chunks and `--max-context 220000` is 515 MiB at 256 MiB against 380 MiB
  at 64 MiB: about 4K fewer INT8 KV tokens (7.7K NVFP4); FP8 and K8V4 gain about 300 MiB of KV
  cache against unbounded.
- **Wave-aligned prefill chunks for NVFP4, FP8 and K8V4 KV**: their prompt kernel (the MX-FP8
  tiled kernel, which fast NVFP4 prompt attention now also uses) runs one 128-row CTA per SM, like
  the fast INT8 kernel, so
  `--prefill-chunk` is now rounded down to whole attention waves for them too (`4096` runs as
  `3584`), which keeps each full chunk's attention free of a mostly idle last wave. One request,
  DFlash2: FP8 prefill 1.3 % faster at 32K tokens, 1.2 % at 64K and 0.8 % at 128K (same-session A/B,
  two passes), K8V4 0.7-1.6 %, NVFP4 1.4-2.2 % at 128K and within noise at 32-64K.
- **Fused text q/k RMSNorm + RoPE at every width** (14-22 % faster than three separate calls at
  the 3584-token chunk, one sincos per lane), and only for checkpoints with its built-in RoPE theta
  and epsilon. Commit: [`441ca14`][c-rope-fused].
- **Kernel tuning from upstream PRs:** the text `rmsnorm_rope` route (#273, Michael Dementii), a
  Q6 34,816×5120 shape for the fused MLP gate_up (#284, [bingchengcc](https://github.com/bingchengcc)),
  and, adapted from [llmq](https://github.com/IST-DASLab/llmq) (IST-DASLab, Erik Schultheis) by
  [DuncanBetts](https://github.com/DuncanBetts), a fused NVFP4 RMSNorm + quantise for the attention
  input projection (#305) and a single-pass target log-probability kernel (#307).
  Commits: [`1a78f38`][c-pr273], [`99d61b4`][c-pr284], [`be08a9b`][c-pr305],
  [`d9f9679`][c-pr307].

### Decode speed

- **Overlapped decode kernels.** Kernels in the decode CUDA Graph launch as programmatic
  dependents of the kernel before them (PDL): weight-streaming kernels load their first weight
  tiles while the previous kernel runs, and release the next kernel only after their own main
  loop. Output is unchanged token for token; greedy decode rounds were 2.2-2.5 % faster when this
  was introduced. Ideas tried and dropped are in [`RESEARCH_NOTES.md`](RESEARCH_NOTES.md).
  Commit: [`6094ea4`][c-pdl].
- **Decode kernels sized for verification widths**: a third pipeline stage for the NVFP4 down
  projection up to 64 tokens (45.5-47.7 µs instead of 57.7-60.0 µs, and the A4 route from 8
  tokens), and two 16-row tiles sharing each staged activation in the FP8 head and the Q8 DFlash2
  drafter.
  Commits: [`3c0d1e2`][c-nvfp4-linear-add], [`fc97916`][c-row-tiles].
- **NVFP4 MLP tiles for verification widths**: the fused gate/up projection runs 64-row tiles at
  two CTAs per SM up to 64 tokens (with a third pipeline stage up to 32), and the [5120,17408]
  down projection streams 512 K per stage up to 8 tokens. Gate/up takes 65.3 instead of 66.1 µs
  at 8 tokens, 64.9 instead of 66.1 at 12 and 66.1 instead of 68.0-68.2 at 33-49 (unchanged at
  16-32 and 56-64); the down projection 36.6 instead of 37.6 µs at 8 tokens. Every output keeps its
  K order, so output is unchanged bit for bit. On the decode-saturation suite (DFlash2 K=7, n-gram
  15/12, INT8 KV, four passes) one request's rounds are 0.56 % shorter with chain verification and
  0.23 % with 12-column trees, four requests' 0.18 % (chain); four requests verifying 12-column
  trees (48 columns, on the 33-64-token tiles) changed by -0.15 % and +0.23 % in two passes, which is
  within noise.
- **Faster split-KV attention for verification rows** (DFlash2/MTP drafts and n-gram copies):
  short rows split their keys into more, balanced KV splits so they fill the GPU; the split merge
  launches as a programmatic dependent of the attention kernel; and INT8 KV rows of 2-16 columns
  use a kernel that decodes V in registers and keeps the next K/V tile in flight. 8-column INT8
  rows spend 3 % less time in attention at 131K keys and up to 47 % less at 2K, 16-column rows
  21-39 % less at every length, and FP8, K8V4 and NVFP4 rows 9-35 % less at 512-2K keys. Output
  is unchanged except for rows shorter than about 11K-22K keys, whose split merge now rounds in a
  different order. On the NVIDIA artifact a single ~100K-token request decodes 2.4 % faster with
  identical output.
  Commits: [`82830da`][c-split-balance], [`6049964`][c-merge-pdl], [`8bb2ce1`][c-int8-pipelined].
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
  Commit: [`233b16d`][c-pr327].
- **Ngram copy drafting** proposes the next tokens by copying matching text from earlier in the
  context, alongside MTP/DFlash/DFlash2. The single-request version is the original work of
  [remesis](https://github.com/remesis) in the [remesis/ninfer](https://github.com/remesis/ninfer)
  fork (upstream issue [#234](https://github.com/Neroued/ninfer/issues/234)); this fork extends it
  to `--max-concurrency` above 1 and builds each prompt's index while the prompt is prepared, off
  the engine worker. See [ngram copy proposals](docs/ngram.md).
  Commits: [`f232333`][c-ngram], [`178f005`][c-ngram-concurrency],
  [`4f891f5`][c-ngram-prep].
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
  Commit: [`b00ad84`][c-nvfp4-targets].

### Smaller KV cache: `--kv-dtype vq2` and `k4v2`

- **Two vector-quantized KV formats.** `vq2` stores K and V in 2 bits per value, and `k4v2` stores K
  in 4 bits and V in 2. Per token and KV head they take 132 and 196 bytes, against 528 for INT8 and
  288 for NVFP4. For the 27B models' attention layers, 240K tokens of context need 1.9 GiB of KV as
  `vq2` and 2.8 GiB as `k4v2`, against 7.6 GiB as INT8 and 4.1 GiB as NVFP4, so the same VRAM holds
  three or four times the context or requests. How it works:
  - each 256-value row is rotated with the fork's fixed Hadamard transform;
  - V, and K for `vq2`, then store every 8 values as one 16-bit code (one of 512 trained magnitude
    patterns plus 7 sign bits, the 8th sign set by parity);
  - K for `k4v2` stores 4-bit Lloyd-Max levels instead;
  - each row keeps an FP16 scale chosen so its reconstruction is unbiased.
- **Recent keys stay exact.** The attention sinks (the first 64 tokens) and the 768 tokens before
  each query are read from INT8 copies held in the request's state. Which keys a query reads
  exactly depends only on positions, so a context reads the same values however it was split into
  prefill chunks, decode steps or cache reuse points. These copies take about 40 MB per state slot
  for the 27B models; each running request, each GPU checkpoint slot and each host checkpoint image
  holds one.
- **Quality** (perplexity at 64K context, 32K stride, `ninfer-ppl-1m-v1` quick corpus, NVIDIA NVFP4
  27B artifact):

  | KV | Perplexity | vs BF16 |
  |---|---:|---:|
  | BF16 | 4.1661 | |
  | INT8 | 4.1728 | +0.16 % |
  | NVFP4 | 4.1764 | +0.25 % |
  | `k4v2` | 4.1709 | +0.12 % |
  | `vq2` | 4.1859 | +0.48 % |
- **Speed** (one request, official NVFP4 27B artifact, RTX 5090, `ninfer_bench` against INT8 measured
  in the same run):

  | Run | Format | Prefill | Decode |
  |---|---|---:|---:|
  | 32K context | `vq2` / `k4v2` | -4.5 % | +1.6 % / +1.4 % |
  | 128K context | `vq2` / `k4v2` | -7.3 % / -8.7 % | +7.4 % / +6.7 % |
  | DFlash2 K=7, 32K context | `vq2` / `k4v2` | -4.2 % / -4.4 % | -3.4 % / -3.6 % |

  In the DFlash2 rows, draft acceptance is 0.82 for both formats against INT8's 0.85.
  Per attention layer, against INT8 measured in the same run (`ninfer_causal_softmax_attention_bench`,
  40 calls, cold, first row / four rows):

  | Call | 2K keys | 8K keys | 32K keys | 128K keys |
  |---|---:|---:|---:|---:|
  | one-token decode | 1.54x / 2.23x | 1.19x / 0.96x | 0.80x / 0.62x | 0.62x / 0.59x |
  | 4-column verification | 1.61x / 2.21x | 1.23x / 1.07x | 0.89x / 0.75x | 0.76x / 0.73x |
  | 8-column verification | 1.54x / 2.48x | 1.23x / 1.42x | 1.09x / 1.10x | 1.04x / 1.03x |
  | 16-column verification | 1.71x / 3.29x | 1.28x / 1.64x | 1.30x / 1.31x | 1.22x / 1.33x |

  Below 32K keys these formats pay for expanding their codes; from 32K, calls of up to four columns
  overtake INT8 at 0.6-0.9x, because the attention reads a quarter of the KV bytes. Wider
  verification stays above INT8 at 1.0-1.4x, where its G64 kernel fills the SM better. `k4v2` is
  within about 0.05x of `vq2` except on 16-column verification, where it is 0.1-0.4x slower
  (1.4-1.9x at 8K keys and below). Prompt attention is 1.2-1.3x at 32-128K keys
  and 1.4x at 8K against INT8's fast prompt kernel (896-3584 columns, 20 calls), and on short
  prompts the encoding of each new row costs more than the attention.
- They work with the hybrid and original prefix caches (a restored checkpoint reads exactly what
  the original request read), cache files, MTP, DFlash2 chain and tree verification, n-gram
  drafting, concurrent requests, CUDA Graphs and vision.
- Based on the HyperQuant KV cache (arXiv 2606.23406) and the implementation of it by
  [cometkim (Hyeseong Kim)](https://github.com/cometkim). Measured on this model's K and V rows, a
  fixed-rate 8-value code beat HyperQuant's E8 lattice with Rice coding at fewer bytes. Keeping
  recent keys exact mattered more than either, so the fork implements the fixed-rate code with an
  exact window.

### Tool calls and reasoning output

- **More tool-call formats**: the XML forms emitted by Claude Code and other agent tools
  (`<function name="…">`, `<invoke>`, `<function_calls>`, short `<param>` tags), also while
  streaming. Upstream PR #300 by [pkochubey](https://github.com/pkochubey) (upstream issue
  [#276](https://github.com/Neroued/ninfer/issues/276)). Commit: [`b4e3c6c`][c-xml-tools].
- **Repeated tool-call parameters keep the last value** (upstream PR #299 by
  [adubkov](https://github.com/adubkov)), and **a quoted `</parameter>` stays in the value**
  (upstream PR #318). Commits: [`65847ef`][c-pr299], [`943157e`][c-pr318].
- **Quoting `</think>` no longer ends the reasoning early**: it only ends the reasoning when a line
  break or the end of the turn follows. Adapted from upstream PR #309 by Fedor Suchkov.
  Commit: [`b13e0ee`][c-think-quote].
- **`--tolerant-tool-calls`** keeps a good call followed by junk, a final call cut off by the
  output limit (if a parameter is complete), repairs a missing `>` after the function name, and
  returns calls to undeclared tools. By David Oelfke in the gzenz/ninfer fork, ported onto this
  fork's parser. The request log's `tool_call_parse.tolerant_recovered` shows when it rescued a
  call (from giveen's giveen/ninfer-ext). Commits: [`09420f6`][c-tolerant-tools],
  [`a404af2`][c-tolerant-recovered].
- **A reasoning effort the chat template rejects renders as its nearest accepted one** (the
  official Qwen3.8 template accepts only low, medium and xhigh), and `--chat-template` gains the
  froggeric v22.5 template. Commits: [`003b60a`][c-effort-nearest],
  [`87ea617`][c-chat-template].

### API and client compatibility

- **llama.cpp-style model details on `/v1/models`** (upstream PR #162 by
  [Hector Ramon Jimenez (hecrj)](https://github.com/hecrj)) and **`ignore_eos` on chat
  completions** (upstream PR #197 by [Thireus](https://github.com/Thireus)).
  Commits: [`4496549`][c-pr162], [`0e865f9`][c-pr197].
- **GitHub Copilot and other agent-host requests** (`custom` tools, advisory `tool_choice` /
  `strict` / `parallel_tool_calls`, tool names up to 256 bytes, `--usage-chunk-choice`): the
  serving commits of upstream PR #316 by [paq85](https://github.com/paq85) (Damian Sromek).
  Commit: [`44a43a2`][c-pr316].
- **Responses API options used by Codex and Zed Agent** (`reasoning.summary`,
  `include: ["reasoning.encrypted_content"]`): upstream PR #295 by
  [Macasacker](https://github.com/Macasacker), based on an earlier PR by
  [Sha1rholder](https://github.com/Sha1rholder). Commit: [`cdc5f04`][c-pr295].
- **`response_format` `json_object` / `json_schema` is accepted** (not enforced), a **tool call
  cut off by the output or context limit is reported as cut off**, and a request ending with an
  assistant message **continues that reply** (thinking off only): upstream PR #300 by
  [pkochubey](https://github.com/pkochubey). Commits: [`cb67100`][c-response-format],
  [`b232e5e`][c-cut-tool-call], [`bdc34cd`][c-continuation].
- **Anthropic clients such as Qwen Code**: forced, named and strict tool choices are accepted as
  advisory (upstream issue #223), a thinking budget at or above `max_tokens` is accepted, and API
  routes answer under a doubled `/v1` prefix (a base URL ending in `/v1`).
  Commits: [`e53645c`][c-pr223], [`d16c7a7`][c-thinking-budget-max],
  [`4e03738`][c-doubled-v1].
- **Claude Code's `thinking.display: "omitted"`** is accepted instead of rejected: Thinking blocks
  come back with empty text and the reasoning in their signature, which is restored when the
  client sends the block back, so retained thinking and prefix-cache reuse are unchanged. Ported
  from giveen's change in giveen/ninfer-ext. Commit: [`9a655ca`][c-thinking-omitted].

### Stability

- **An aborted prefill can no longer write into another conversation's cached prefix**: a lane's
  block-table copy still queued from an aborted request is waited for before the next request
  overwrites it. Commit: [`fad0dd9`][c-kv-fence].
- **Smaller fixes:** a workspace scope opened before an arena reset no longer rolls the next
  phase's allocations back; a request an idle engine can never admit gets 503 instead of 500;
  token-count requests are bounded like generation requests; Windows servers detect clients that
  vanish without closing the connection; a vision overlay suffix is encoded at the right
  position (fork PR #1 by Yunado).
  Commits: [`1729d4a`][c-arena-scope], [`95d29dd`][c-idle-503],
  [`565024f`][c-count-bound], [`bacb651`][c-win-keepalive],
  [`0583304`][c-pr1].

### Models, conversion and vision

- **GGUF files as conversion sources**: upstream PR #282 by [giveen](https://github.com/giveen).
  Commit: [`9f7ca43`][c-gguf].
- **NVIDIA ModelOpt NVFP4 and FP8 checkpoints** as conversion sources, with the
  `qwen3_8_27b_nvfp4_nvidia` recipe storing the output head as FP8; **Quasar NVFP4 conversion**
  with DFlash2 heads and an indexed proposal head; **third-party tokenizer settings** rebuilt during
  conversion. Commits: [`84c1133`][c-modelopt], [`1f4f04c`][c-quasar],
  [`1a99c1d`][c-tokenizer].
- **`nvfp4_absmax` and `nvfp4_mse` conversion methods** quantize BF16 sources to NVFP4 for
  16-bit-activation parents; the NVIDIA recipe uses them for the DFlash2 drafter MLP. `nvfp4_mse`
  picks each 16-value group's scale from all 126 E4M3 values by squared error (based on
  giveen's scale sweep in giveen/ninfer-ext): on the drafter's gate projection the relative
  error is 0.0812, against 0.0847 for the earlier best of five targets and 0.0952 for absmax.
  Commit: [`8cc75d8`][c-nvfp4-mse].
- **A `qwen3_8_27b_q6` recipe** and a `grouped_mse` scale-search method for groupwise
  quantisation; **Q8 MTP** and a **general BF16 GEMM fallback** for shapes without a dedicated
  kernel. Commits: [`99d61b4`][c-pr284], [`4ee02bb`][c-grouped-mse],
  [`f085b82`][c-q8-mtp], [`c2af41c`][c-bf16-gemm].
- **`--rope-yarn-factor F`** for YaRN context extension (F from 1 to 4, up to 1M tokens of
  context). Commit: [`ea7e50c`][c-yarn].
- **`--vision-offload on`** keeps the vision tower in pinned system RAM instead of VRAM and streams
  it to the GPU while an image is encoded, and **`--vision-max-merged N`** bounds the merged vision
  tokens per image or video. Based on the original work by
  [Valeriy Selitskiy (iamwavecut)](https://github.com/iamwavecut), rewritten for this engine.
  Commit: [`529ca5b`][c-vision-offload].

### Windows

- **Native build and run** with MSVC and CUDA (see [Quick start](#quick-start-windows)): static
  CUDA runtime, FFmpeg/curl from vcpkg, non-RDC NVFP4 kernels, TMA descriptors staged into device
  memory by a kernel (launches captured into a decode graph read a copy written once, so decode
  rounds replay no staging kernels), a built-in PNG decoder for the vision path, drive letters in
  converter recipe paths, and a build id in every product binary. The same port without any
  other fork change is the branch `upstream-Windows-Port`. Commits: [`743da8c`][c-win-extras],
  [`9021804`][c-build-id].
- **Linux still builds and runs** (see [Quick start (Linux)](#quick-start-linux)), checked under
  WSL2 Ubuntu 24.04 with CUDA 13.4.

### Options and console

- **`--log-colours on`** colours the console statistics, and a **session statistics panel**
  beneath the log shows session and last-ten averages of TTFT, cache hit rate, prefill and decode
  speed, the mean decode batch and drafter acceptance (`--log-stats-panel off` removes it). Engine
  messages and FFmpeg's log are ordinary log records that scroll above the panel.
  Commits: [`75b1954`][c-log-colours], [`190c80a`][c-stats-panel],
  [`3974165`][c-diagnostics], [`f617271`][c-ffmpeg-log].
- **Grouped `--help`** by category on `ninfer-serve` and the `ninfer` CLI, with separate sections
  for the two prefix caching systems. Commit: [`2b070bc`][c-help].
- **`--vram-headroom-mib N`** sets how much GPU memory `--kv-capacity auto` leaves spare (upstream
  always leaves 1 GiB). Commit: [`d4d2e12`][c-vram-headroom].
- **Measured CUDA Graph allowance**: the KV sizing reserves 64 MiB plus 4 MiB per decode-graph
  executable, measured on an RTX 5090 across every speculative mode and concurrency; startup
  warns if the graphs ever use more. Commit: [`966fa5a`][c-graph-allowance].
- **`--thinking-budget-message S`** sets the message inserted when a request reaches its
  `--default-thinking-budget N`. Commit: [`9304521`][c-thinking-message].
- **`--request-log-max-mib N`** rotates the `--request-log-jsonl` file once it reaches `N` MiB,
  keeping `--request-log-keep K` older files (default 4); each new file starts with a copy of the
  server's start record. Based on the rotation by
  [Gideon Zenz (gzenz)](https://github.com/gzenz) in the gzenz/ninfer fork.
  Commit: [`1b84b06`][c-log-rotation].

### Kept in sync with upstream

This fork is rebased onto upstream `master` whenever upstream moves; the latest is `68c54356`
(October 2026). Where upstream rewrote code this fork had changed, the rebase starts from
upstream's version and carries the fork's change onto it only where an A/B test on the RTX 5090
shows the fork's version is faster. Once upstream adopts a change, it leaves this list.

Dropped in the rebase onto `abb7f14f`, because upstream now does the same thing:

- **Fork fixes to upstream's previous prefix cache** (see above) and **worker recovery from out of
  memory**: upstream replaced that cache and its admission with incremental reservation and
  preemption, which reports a shortage instead of failing an allocation.
- **Round-robin prefill (`--prefill-round-robin`)** and **concurrent staged-prefill lanes** (by
  David Oelfke in the gzenz/ninfer fork): upstream now prefills several lanes and rotates prefill
  between them by default. The narrow prefill steps beside other requests were not carried over.
- **A full main KV pool ending an MTP or DFlash answer early when only its draft KV lease needed
  room** (from giveen/ninfer-ext): upstream's incremental reservation grows each request's KV unit
  by unit.
- **Publishing activated block tables on the compute stream** (upstream PR #320): upstream now
  requires an explicit publication stream for every block-table copy.
- **Logging an Anthropic stream cancelled while queued as a disconnect (499)** (by Gideon Zenz in
  the gzenz/ninfer fork) and **the FP64 attention test oracle on every host core**: upstream does
  the same.

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
[Gideon Zenz (gzenz)](https://github.com/gzenz),
[cometkim (Hyeseong Kim)](https://github.com/cometkim), David Oelfke, Fedor Suchkov,
Yunado, and everyone else whose pull
requests, reviews and commits made this fork possible — and a particular thank you to
**[Neroued](https://github.com/Neroued)** for creating NInfer, maintaining upstream so
well, and for the work this branch builds on.

[c-agentic-ab]: https://github.com/Wallawalla47/ninfer-custom/commit/b1d99ca467759f53eeb1355dee42b69513c94872
[c-serve-concurrency]: https://github.com/Wallawalla47/ninfer-custom/commit/021e3cdd5e9f63f7b71af80174a2bdab517c21e6
[c-ab-rig]: https://github.com/Wallawalla47/ninfer-custom/commit/a94c0950d085663eb2f5f2af916bd58791e86396
[c-hybrid]: https://github.com/Wallawalla47/ninfer-custom/commit/2184a98121417ca6968ff3062af36370bae26c81
[c-ctrl-c-stop]: https://github.com/Wallawalla47/ninfer-custom/commit/2184a98121417ca6968ff3062af36370bae26c81
[c-fast-int8]: https://github.com/Wallawalla47/ninfer-custom/commit/60e67d06a6292f848c1a21578b3e4a53d704c123
[c-nvfp4-kv]: https://github.com/Wallawalla47/ninfer-custom/commit/205d7f250b9f4db179bd207cb964bb6f407f6279
[c-rope-fused]: https://github.com/Wallawalla47/ninfer-custom/commit/441ca14096eb1880bea7dc87df6b06cd7cd25b1d
[c-pr273]: https://github.com/Wallawalla47/ninfer-custom/commit/1a78f3832ae0f5736b091c431033078c49fcd41e
[c-pr284]: https://github.com/Wallawalla47/ninfer-custom/commit/99d61b40aeda068e6f244c40eca4d95dd6d6ac5c
[c-pr305]: https://github.com/Wallawalla47/ninfer-custom/commit/be08a9bd468130d49b75c40ba2495d60944a1754
[c-pr307]: https://github.com/Wallawalla47/ninfer-custom/commit/d9f9679e00b9ff4dbd959ec1582740a86a8d9b14
[c-pdl]: https://github.com/Wallawalla47/ninfer-custom/commit/6094ea4460e140ef76b7a24f4ab47b405635beb7
[c-nvfp4-linear-add]: https://github.com/Wallawalla47/ninfer-custom/commit/3c0d1e213f8c69a0361068a004400dbce76ed23f
[c-row-tiles]: https://github.com/Wallawalla47/ninfer-custom/commit/fc97916f0e37ab21978cfa3860b8b826a4be7ed0
[c-split-balance]: https://github.com/Wallawalla47/ninfer-custom/commit/82830da614f52b31e5050b89a66797c226b111c3
[c-merge-pdl]: https://github.com/Wallawalla47/ninfer-custom/commit/6049964ffeceee5bcbcbaef7c2494d64d09764bc
[c-int8-pipelined]: https://github.com/Wallawalla47/ninfer-custom/commit/8bb2ce10e80386ab14d72b6ec1577f4818741e5a
[c-pr327]: https://github.com/Wallawalla47/ninfer-custom/commit/233b16dbeeefc1cf26ef9c258df8b3f7a16d62ec
[c-ngram]: https://github.com/Wallawalla47/ninfer-custom/commit/f2323337abbacefbccc011f76dc047b283864637
[c-ngram-concurrency]: https://github.com/Wallawalla47/ninfer-custom/commit/178f005d8b025a1923615ec381c210e0e7d8ba54
[c-ngram-prep]: https://github.com/Wallawalla47/ninfer-custom/commit/4f891f5c9a57d254abfc060d48f306ef3043299d
[c-nvfp4-targets]: https://github.com/Wallawalla47/ninfer-custom/commit/b00ad843db1e54c57dd320c4a2a0bac7b02ce2c9
[c-xml-tools]: https://github.com/Wallawalla47/ninfer-custom/commit/b4e3c6c093bc6abc6eacfb5fd51f7feaf081bf98
[c-pr299]: https://github.com/Wallawalla47/ninfer-custom/commit/65847ef6ceaf1ba96c43caef1b33322459041489
[c-pr318]: https://github.com/Wallawalla47/ninfer-custom/commit/943157e22eeeca9d1ac2a0081ee812b0a06a7089
[c-think-quote]: https://github.com/Wallawalla47/ninfer-custom/commit/b13e0ee32b2d067e2ff898590c8c6fb0eb6a3fc4
[c-tolerant-tools]: https://github.com/Wallawalla47/ninfer-custom/commit/09420f61d81a26b983138a9b9623ea17fbdd65fd
[c-effort-nearest]: https://github.com/Wallawalla47/ninfer-custom/commit/003b60afccbd31c20ab4df039153b1ecb1edea77
[c-chat-template]: https://github.com/Wallawalla47/ninfer-custom/commit/87ea617ac086dc45fbe3919fbcc0e9f76dfbf0a2
[c-pr162]: https://github.com/Wallawalla47/ninfer-custom/commit/4496549debef249915d35ddb0068c8581aed15ee
[c-pr197]: https://github.com/Wallawalla47/ninfer-custom/commit/0e865f9d098e416dd8068a97ae2042b5e6daeca9
[c-pr316]: https://github.com/Wallawalla47/ninfer-custom/commit/44a43a2169483b5a6ecfb308f2e8138b0797cdb8
[c-pr295]: https://github.com/Wallawalla47/ninfer-custom/commit/cdc5f04961fa3155c3b94dc2a47345551684530c
[c-response-format]: https://github.com/Wallawalla47/ninfer-custom/commit/cb6710085377ff5df1f36d287dc272299dffe67c
[c-cut-tool-call]: https://github.com/Wallawalla47/ninfer-custom/commit/b232e5e788c905e7cb7a73dc1d642fd8aa771510
[c-continuation]: https://github.com/Wallawalla47/ninfer-custom/commit/bdc34cd5185e4cb598bc5a4b338f76919ad2d94e
[c-pr223]: https://github.com/Wallawalla47/ninfer-custom/commit/e53645c39c1702728e81d98f956ea2897d69488c
[c-thinking-budget-max]: https://github.com/Wallawalla47/ninfer-custom/commit/d16c7a748dfa0233a086a667f07298e2a1555561
[c-doubled-v1]: https://github.com/Wallawalla47/ninfer-custom/commit/4e03738c007e81d897db04babd7844b8e171e61a
[c-kv-fence]: https://github.com/Wallawalla47/ninfer-custom/commit/fad0dd9e78d1cec82cf92cab0923dfb7fa522976
[c-arena-scope]: https://github.com/Wallawalla47/ninfer-custom/commit/1729d4aebbbc2f103aeb1a474109108a2c308aea
[c-idle-503]: https://github.com/Wallawalla47/ninfer-custom/commit/95d29dde134aa13f50ca835afdaff4ae6e42637f
[c-count-bound]: https://github.com/Wallawalla47/ninfer-custom/commit/565024f910f57a5fdcc6614d97a9732bf616e165
[c-win-keepalive]: https://github.com/Wallawalla47/ninfer-custom/commit/bacb6516942ecc1f758e85ad00937a6667f31fec
[c-pr1]: https://github.com/Wallawalla47/ninfer-custom/commit/0583304f023082c0cff70977886c10fc6713be1e
[c-gguf]: https://github.com/Wallawalla47/ninfer-custom/commit/9f7ca43cbea0064bc7c1c9368ec99c6922a3d1b0
[c-modelopt]: https://github.com/Wallawalla47/ninfer-custom/commit/84c1133e15c570346a170593ad0cb21565be2f6e
[c-quasar]: https://github.com/Wallawalla47/ninfer-custom/commit/1f4f04c0a0ddcae37c65283094b63eb8c7f94038
[c-tokenizer]: https://github.com/Wallawalla47/ninfer-custom/commit/1a99c1d110145c5a7580a41981ade95e18ffa74a
[c-grouped-mse]: https://github.com/Wallawalla47/ninfer-custom/commit/4ee02bb169303482c641d580b11657b34914ce0b
[c-q8-mtp]: https://github.com/Wallawalla47/ninfer-custom/commit/f085b821e273c4e2ca227a6bfa6b5ba73dc41951
[c-bf16-gemm]: https://github.com/Wallawalla47/ninfer-custom/commit/c2af41c4857356f4ab385c7ea1c1ad3c4d2da2fe
[c-yarn]: https://github.com/Wallawalla47/ninfer-custom/commit/ea7e50cc73fc35a3ee7aa5c2ec9aa43b6084ea11
[c-vision-offload]: https://github.com/Wallawalla47/ninfer-custom/commit/529ca5b9c31eef5d3f59ab3f50c626e100061770
[c-win-extras]: https://github.com/Wallawalla47/ninfer-custom/commit/743da8c28d99a31ad7d631ef5f21da9ffa0bdf3d
[c-build-id]: https://github.com/Wallawalla47/ninfer-custom/commit/90218044f8cef270e44b65ae8a2f03f841982377
[c-log-colours]: https://github.com/Wallawalla47/ninfer-custom/commit/75b1954924b90a747658611305b30d1f961bef3c
[c-stats-panel]: https://github.com/Wallawalla47/ninfer-custom/commit/190c80ab4ace45b3c76b2a781ce899a618523ca8
[c-diagnostics]: https://github.com/Wallawalla47/ninfer-custom/commit/3974165107cc26bf750ae144606aae1427aaffe5
[c-ffmpeg-log]: https://github.com/Wallawalla47/ninfer-custom/commit/f617271e2c6ac62dea68035e4fc80876f84f5c31
[c-help]: https://github.com/Wallawalla47/ninfer-custom/commit/2b070bc141cc97e5466eb1b500f0e5c1aac9e40b
[c-vram-headroom]: https://github.com/Wallawalla47/ninfer-custom/commit/d4d2e125326d72ad59a4917a467bea7bf5cea99a
[c-graph-allowance]: https://github.com/Wallawalla47/ninfer-custom/commit/966fa5abc72f7bcdbbf96d2d4ead4a29dc34d8cb
[c-thinking-message]: https://github.com/Wallawalla47/ninfer-custom/commit/930452173d9ce4d7665dae224017d79f55ff9649
[c-log-rotation]: https://github.com/Wallawalla47/ninfer-custom/commit/1b84b060f858974de7dbf8c9232841dc4ffeb24c
[c-nvfp4-mse]: https://github.com/Wallawalla47/ninfer-custom/commit/8cc75d8c7918acb2b40f460bcbc970a1e5a0e01e
[c-thinking-omitted]: https://github.com/Wallawalla47/ninfer-custom/commit/9a655cac1ed2efba3e5c2539a1009aa005ee3042
[c-tolerant-recovered]: https://github.com/Wallawalla47/ninfer-custom/commit/a404af27efd7b828b3943ac8f534bdb18a029ab2

---

## Upstream README (direct copy)

Everything below is a copy of the upstream
[NInfer README](https://github.com/Neroued/ninfer/blob/master/README.md) as of the upstream
commit this fork is rebased on (`68c54356`), unchanged except for one added link to
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
