# tools/ablate

Fits a refusal-ablation sidecar for `infernix-serve --ablate` (every model it runs on: the Qwen3.5 family,
Qwen3.8-Flash-Next and Inkling 2) on your own setup, and evaluates it against the stock model. One entry point with subcommands; all state lives in one run directory, and every step is
resumable. The guide, with requirements, prompt sets, timings and how to read the results, is
[docs/ablate-fitting.md](../../docs/ablate-fitting.md).

```bash
pip install -r tools/ablate/requirements.txt          # eval also: requirements-eval.txt
python tools/ablate/ablate.py init RUN --production production.txt
python tools/ablate/ablate.py doctor RUN
python tools/ablate/ablate.py all RUN --harmful harmful.txt --harmless harmless.txt --study s1
python tools/ablate/ablate.py datasets fetch advbench-alpaca --out prompts   # public prompt sets, licences: datasets list
python tools/ablate/ablate.py auto RUN --harmful H --harmless G --out sidecars --name my-cut   # all of it, to a sidecar
python tools/ablate/ablate.py --help                   # every command: list, pin, grade, chat, eval, export
```

The launcher's Ablation page runs `init --production-argv` and `auto` for a model in its library (a release ships
this folder as `ablate` beside the launcher; `tools/launcher/ablate_stub.py` stands in for it in launcher_check).

`ablatelib/` holds the parts: `production.py` reads the production setup, `serve_flags.py` classifies every
`infernix-serve` option (carried, set by the tool, left out, refused), `capture.py` records each prompt's hidden
states through a scoring engine (`POST /v1/score` with `hidden_states`), `direction.py` computes the direction, `fit.py` runs the Optuna study, `measure.py` grades refusals and KL,
`candidates.py` pins, grades and exports, `evaluate.py` runs the capability tasks, `doctor.py` checks a run
before it starts, `datasets.py` downloads and converts the public prompt sets, and `auto.py` runs everything to an
exported sidecar with a `progress.json`. `tests/test_ablate_tools.py` runs every command on CPU against the fake engine in
`tests/ablate_fakes.py`.
