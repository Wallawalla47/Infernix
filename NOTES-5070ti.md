# Infernix-5070ti — fork notes

Fork of [Wallawalla47/Infernix](https://github.com/Wallawalla47/Infernix) specialized for
**RTX 5070 Ti (sm_120 / Blackwell)** deployments on high-core EPYC hosts.
Maintained by `vault-watcher-sketch` (AI-dedicated account).

## The CPU placement fix (2026-10-28, epyc7532: EPYC 7532 32C/64T + 2x RTX 5070 Ti)

Decode throughput was CPU-pipeline-bound (engine telemetry: `host 97.9% ... limited`, GPU SM
17-40% bursty, PCIe rx only 1.3-3.5 of 28.5 GB/s). The expert offload worker team (default
**6** threads, `cpu_expert_workers`) is the critical path and is **not** scaled by
`OMP_NUM_THREADS` — but it *is* configurable through the undocumented switch
`INFERNIX_TMP_CPU_PLACEMENT=<mode>[,<workers>]` (mode in `default|physical|physical-set`,
workers 1-40; `physical` pins workers to distinct physical cores). Documented in
[docs/cpu-placement.md](docs/cpu-placement.md) (upstream PR
[Wallawalla47/Infernix#44](https://github.com/Wallawalla47/Infernix/pull/44)).

Measured on Qwen3.8-Flash-Next FP4-Local (256-token decode probe, median of 5 interleaved
rounds, identical flags, scheduling only, **no output change**):

| Placement   | tok/s | vs default |
|-------------|------:|-----------:|
| 6 (default) |  70.4 |        - |
| default,16  |  73.8 |      +8 % |
| physical,16 |  76.0 |     +12 % |
| physical,32 |  83.3 |     +19 % |

## Recommended launch (this hardware)

    -e INFERNIX_TMP_CPU_PLACEMENT=physical,32   # all 32 physical cores

Speed tier (max decode, speculative flags alter output regime):

    infernix-serve <model>.infernix --ngram-volume <model>.ngram --kv-dtype int8 \
      --max-context 65536 --max-concurrency 2 --spec dflash2 \
      --draft-model <dflash2-drafter>.infernix --draft-tokens auto --lm-head-draft \
      --spec-min-p 0.7 --expert-tail-skip 7

Quality tier (vanilla output: no tail-skip, min-p 0.5) = 61.4 tok/s, still +25-39 % over the
pre-fix 6-worker default.

## Build

Repo `Dockerfile` is the production recipe (nvidia/cuda:13.1.2-devel-ubuntu24.04, ninja,
targets `infernix` + `infernix-serve`):

    docker build -t infernix-5070ti:vanilla-<date> -f Dockerfile .

Live container: `infernix-p8080` on epyc7532 (GPU 0). Full tuning process:
`LLM-Weights/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix_Wallawalla47/docs/
tuning-process-epyc7532-5070ti-p8080-20261028.md` (sec 10); benchmark rows:
aios-benchmark.db runs 652-654.
