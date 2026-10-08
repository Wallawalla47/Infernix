# Agentic A/B benchmark: NInfer + Windows port vs Infernix

A black-box A/B benchmark of two `infernix-serve` builds serving the same model on the same GPU,
driven by a closed-loop replay of real agentic coding traffic over the OpenAI chat-completions
API. Infernix runs its default hybrid prefix cache; an optional third arm runs Infernix build with
the original prefix cache (`--use-original-prefix-caching`). It produces a
README-style comparison table covering:

- **prefix-cache hits** (tokens served from cache, continuing turns that had to re-prefill);
- **time to first token**, split into continuing-session turns (cache retention) and new long
  prompts (prefill speed);
- **raw prefill tok/s** on requests that had no cache hit in any arm;
- **output tok/s** per second of engine decode time, with one request decoding, two requests
  decoding, and at the run's own batching, each with a 95 % interval, split into engine speed
  (decode rounds/s) and speculative acceptance (tokens per round).

Metrics come from each serve's own request log (`--request-log-jsonl`); the client only supplies
the request classification.

## Files

| File | Purpose |
|---|---|
| `workload.py` | Deterministic scenario: personas, observations and session timeline (`python workload.py` prints a summary). |
| `runner.py` | Calibrates a common `--max-context`, runs the treatment, the optional alt arm, then the control, drives the workload, calls the analyzer. |
| `analyze.py` | `python analyze.py <run_dir>`: joins client and serve logs, writes `report.md` and `summary.json`; `--aggregate` combines several seeds' runs. |
| `build_control.bat` | Windows build of the control's `ninfer-serve` from a control checkout. |

Each run writes to `profiles/bench/agentic_ab/<timestamp>/` (gitignored): `runner.log`,
`config.json`, `plan.json`, `report.md`, `summary.json`, and per arm `client.jsonl`,
`request_log.jsonl`, `serve.log`, `arm.json`. A run over several seeds has one such directory per
seed (`seed-<n>/`) and the combined `report.md` above them.

## What the workload models

The shape comes from the deploy folder's production request logs (~3,300 requests from Qwen Code
and Claude Code style clients): prompts of 30K-210K tokens (median ~100K) that grow by a few
hundred to a few thousand tokens per turn, short tool-calling answers (median ~380 tokens,
p90 ~4,000) with thinking on, `max_tokens: 64000` on every agent turn, periodic compaction,
subagent fan-outs, whole-history side calls, retries, aborted requests, and several sessions
sharing one serve with up to eight requests in flight.

One run replays three interleaved sessions plus eleven subagents (130 requests at scale 1.0):

| Actor | Persona | What it does |
|---|---|---|
| A | Claude-Code-like (20 tools, ~13K-token system+tools) | Resumes a ~85K-token C++ session (cold), runs a tool loop, fans out 4 explore subagents, loops on to ~125K, compacts, restarts from the summary, fans out 3 more subagents, then waits for B and C to finish and wraps up alone: a last loop, a review subagent and a long pull-request write-up. |
| B | Qwen-Code-like (17 tools, ~11K) | Resumes a ~60K Python session, loops (a user retry, a loop-detection side call), launches two research subagents at once, sits idle while A compacts, comes back, loops, sends a second side call, has its older tool results cleared by the client (all but the last 20), loops. |
| C | Claude-Code-like | Resumes a ~25K tests/docs session, loops, receives a pasted ~16K-token CI log that the client aborts after 1.5 s and re-sends, launches a review subagent, loops. |
| explore-1..7, survey-1..2, review-1..2 | Subagent personas (6 tools, ~12-15K shared prefix) | 4-5 read-only research or review turns each, final report. |

The three resumes run one after another before the sessions start interleaving, so they are
clean, isolated cold prefills of ~25K, ~55K and ~80K tokens. Four agent turns ask the model to
rewrite or edit a file it has just read (copy-heavy output). B's concurrent subagent pair and A's
solo wrap-up give every arm sustained stretches of two-request and one-request decoding, whatever
its prefill speed does to the interleaving elsewhere.

**Closed loop, identical observations.** Every observation the client sends (tool results, user
messages, subagent reports, the post-compaction summary) is generated before the run from real
files, diffs and history of this repository at a pinned commit (`AB_CORPUS_COMMIT`, default
`e48a0d28`), so every arm receives byte-identical observations. The assistant turns are each
arm's own streamed output (reasoning, content and tool calls), fed back as an agent client does;
that is what makes the engine's private-endpoint reuse behave as in production. Observations
answer the model's tool calls by id whatever it asked for, so every arm sees the same token
growth. Each request carries a `seed` that is identical in every arm; it also tags the request
in the serve log.

**Lock-step main sessions.** Once resumed, A, B and C take their turns in rounds: a session
sends its next request only when every other session in the rounds is also ready to send, and
within a round the scripted tool and user delays, identical in every arm, set the order. A
session leaves the rounds while it waits on its subagents or on another session, and when it
finishes, so the rounds never stall. Without them, a faster build runs one session ahead of the
others and changes which session's prefix is evicted at the device-KV limit, so the cache rows
would partly measure speed.

**Sampling.** Every request sends `temperature 1.0`, `top_p 0.95`, `top_k 20`, as 97 % of the
production requests did.

What differs between arms is only what should: the model's sampled text (the kernels differ
numerically, so outputs diverge after a few tokens), and therefore interleaving and queueing.
Rates, cache fractions and per-class averages are comparable; absolute completion totals are not.
Sampled answer lengths are heavy-tailed, as in production: the longest 8 % of turns write about
45 % of all output, and which turns run long changes with every run, so one run is one draw.
Several workload seeds (`--seeds`) show how much a result depends on the particular session.

## Fairness

- **Control = NInfer at the commit Infernix has merged, plus only the Windows port.** A
  control built from an older NInfer would credit Infernix with NInfer's own newer work.
- **Same launch configuration.** The treatment runs the deploy folder's launch bat
  (`AB_LAUNCH_BAT`, default the official-artifact launcher) plus `AB_TREATMENT_EXTRA_FLAGS`
  (default none). The control runs the same flags minus the ones its `--help` does not
  advertise.
- **Same host RAM.** Both builds size their pinned Host tier with `--host-context-mib`, so the
  control gets the treatment's value unchanged.
- **Same context.** NInfer keeps a fixed 1 GiB of VRAM spare under `--kv-capacity auto`, so it
  may not start at the bat's context. The runner tries the bat's value first, then 200000,
  180000, 170000, 160000, and runs every arm at the first one the control starts with.
  Infernix's larger device KV at that context (`--vram-headroom-mib 0` and its measured CUDA Graph
  allowance) is part of what is being compared and is shown in the report header.

## Running it

1. Stop any server on the port and let the GPU go idle (the runner checks both and refuses
   otherwise; it never starts or stops a production server).
2. Build the control:

   ```bat
   git worktree add -b ab/ninfer-windows-port <dir>\src origin/master
   git -C <dir>\src cherry-pick <windows-port-commit>
   set AB_CONTROL_SRC=<dir>\src
   set AB_CONTROL_BUILD=<dir>\build
   build_control.bat configure
   build_control.bat build
   ```

3. Run both arms (about 35 minutes on an RTX 5090, including model loads):

   ```bat
   set AB_CONTROL_EXE=<dir>\build\apps\Release\ninfer-serve.exe
   python runner.py
   ```

   `--arms treatment,alt,control` adds the original-prefix-cache arm (about 50 minutes; the arms
   always run in that order). `--seeds 42,43,44` runs every arm once per workload seed, seed by seed, into
   `<out>/seed-<n>`, and writes a combined report to `<out>/report.md`; each seed replays
   different observations. `--ctx N` skips calibration, `--scale F` stretches or shrinks the
   session loops (0.3 is a quick smoke run), `--dry-run` prints the plans and flags.
   `python analyze.py <run_dir>` re-renders a report from whichever arms the run directory holds;
   every arm is compared with the control. `python analyze.py --aggregate <out> <run_dir>...`
   re-renders the combined report.

| Variable | Default |
|---|---|
| `AB_LAUNCH_BAT` | `<AB_DEPLOY>\LaunchQwen3.8-27B-official-dflash2-ngram.bat`; one of the two is required |
| `AB_DEPLOY` | none (the folder holding the launch .bat) |
| `AB_MODEL` | the model path in the launch bat |
| `AB_TREATMENT_EXE` | `build-windows\apps\Release\infernix-serve.exe` in this checkout |
| `AB_CONTROL_EXE` | `bench\agentic_ab\control\build\apps\Release\ninfer-serve.exe` |
| `AB_TREATMENT_EXTRA_FLAGS` | none |
| `AB_ALT_EXTRA_FLAGS` | `--use-original-prefix-caching` (added to the treatment's flags) |
| `AB_ALT_EXE` | the treatment executable; another build puts a second Infernix build in the alt arm |
| `AB_CONTROL_LABEL` / `AB_TREATMENT_LABEL` / `AB_ALT_LABEL` | none (the report calls the arms "NInfer + Windows port", "Infernix" and "Infernix, original prefix cache") |
| `AB_CONTROL_KIND` | `serve`; `strata` runs the control arm as a Strata server (below) |
| `AB_STRATA_DIR` / `AB_STRATA_CONFIG` | none, required for a Strata control / its `strata-unsloth-ud-q4_k_xl.json` |
| `AB_STRATA_EXTRA_ARGS` | none (engine arguments added to the Strata config's) |
| `AB_REQUEST_TIMEOUT_S` | `1200` (per request; raise it for an engine that queues requests one at a time) |
| `AB_HOST` / `AB_PORT` | `127.0.0.1` / `8080` |
| `AB_OUT` | `profiles\bench\agentic_ab` in this checkout |
| `AB_CORPUS_REPO` / `AB_CORPUS_COMMIT` | this checkout / `e48a0d28` |

The launch bat's serve line may start `infernix-serve` or `ninfer-serve` (deploy folders from
before the rename). A run is marked invalid (non-zero exit, warning in the report) if any
workload request fails.

### Strata as the control arm

`AB_CONTROL_KIND=strata` compares Infernix with [Strata](https://github.com/Niko1221/Strata) on
Qwen3.8-Flash-Next. The runner starts Strata's own server (`serve\server.py --engine strata`) with
the Strata config plus `AB_STRATA_EXTRA_ARGS`, waits until it answers a request (it returns 503
while its engine starts), and needs `--ctx` (no calibration). The treatment's launch line comes
from `AB_LAUNCH_BAT` as usual.

Strata writes no request log of this kind, so after its arm the runner writes `request_log.jsonl`
from the client's rows: prompt and output tokens from each response's `usage`; cached tokens,
prefill time, decode time and drafts from Strata's per-request `timings` (`cache_n`,
`prompt_ms`, `predicted_ms`, `draft_n`, `draft_n_accepted`), with verify rounds = output tokens −
accepted drafts; queue wait = TTFT − prefill time (Strata runs one request at a time); and one
decode record per request (it never decodes two at once). TTFT comes from the client's clock, and
in a run with a Strata arm the analyzer takes every arm's TTFT from the client's clock
(`client_ttft` in `config.json`).
The report also flags when the client's context guard had to clear old tool results to keep a
prompt 24K tokens under `--max-context`.

## Reading the report

- **TTFT** includes queueing: two lanes serve up to seven requests in flight. The report also
  gives the average queue wait and TTFT without it.
- **Brackets on the TTFT and cache rows** are 95 % block-bootstrap intervals over stretches of
  10 consecutive requests: how much the number moves with which stretches of the run it
  contains. They cannot show how differently another run would interleave; the combined report
  over several seeds shows that, as the mean with the min-max over seeds, and each arm's change
  against the control computed per seed.
- **Continuing-session turns** are tool-loop turns, retries, the post-idle turn, the
  post-history-edit turn and the abort retry; cache retention decides their TTFT. **New long
  prompts** are the resumes, compaction and side calls; prefill speed decides theirs.
- **Re-prefilled turns** are split into main-session and subagent turns: losing a 100K main
  session costs far more tokens than losing a 15K subagent, so the two tell different stories.
- **Prefill tok/s with no cache hit** is token-weighted (total prefilled tokens over total
  prefill time) over requests that had no hit in any arm and prefilled at least 4,096 tokens,
  with a separate 32K+ row; the per-request table shows them by size.
- **Output tok/s** is the decode tokens the server committed per second of the engine's own
  decode time: device wait plus host work of decode rounds, from the serve's ~5 s throughput
  records. Prefill chunks and idle time do not dilute it, and ngram copies count as output.
  The *one request* and *two requests* rows use only the records in which every decode round ran
  that many requests, so a build's batching mix cannot move them. The *all* row takes every decode
  record at the batching the run produced, so faster prefill that keeps both lanes decoding shows
  up there. Brackets are 95 % block-bootstrap intervals over ~30 s stretches of decoding: how much
  the rate moves with which turns happened to decode. A row with less than ~150 s of decoding
  shows its seconds instead of an interval.
- **Output tok/s = decode rounds/s × tokens per round.** Rounds/s is the engine's own speed at
  that batch size (kernels and host work) and varies only a few percent across a run. Tokens per
  round is speculative acceptance: it moves with what the model happened to write (a file copy
  accepts several times more than fresh reasoning) and carries most of the interval on output
  tok/s. Compare rounds/s for engine speed and tokens per round for drafting; each arm samples
  its own text, so acceptance also differs between repeated runs of one build. The notes give
  each arm's decode seconds per row and per-request completion over decode wall time, which is
  what one stream saw, including other lanes' batching and prefill.
- **Output volume** is sampled, not controlled: the notes give each arm's completion and thinking
  totals and how many turns ran into the thinking budget. A few long turns shift cache
  pressure, batching and wall time, so compare seeds before attributing them to a build.
- **Where cache hits came from** and **cache pressure** (eviction, degradation, spill and
  host/device transfer counters summed from the serve's throughput records) explain the cache
  rows.
