# tools/ablate

Fits a refusal-ablation sidecar for `infernix-serve --ablate` (Qwen3.8-Flash-Next) on your own setup, and evaluates
it against the stock model. One entry point with subcommands; all state lives in one run directory, and every step is
resumable. The guide, with requirements, prompt sets, timings and how to read the results, is
[docs/ablate-fitting.md](../../docs/ablate-fitting.md).

```bash
pip install -r tools/ablate/requirements.txt          # eval also: requirements-eval.txt
python tools/ablate/ablate.py init RUN --production production.txt \
    --capture-binary build/tests/infernix_qwen4_exp_forward_real_test
python tools/ablate/ablate.py doctor RUN
python tools/ablate/ablate.py all RUN --harmful harmful.txt --harmless harmless.txt --study s1
python tools/ablate/ablate.py --help                   # every command: list, pin, grade, chat, eval, export
```

`ablatelib/` holds the parts: `production.py` reads the production setup, `serve_flags.py` classifies every
`infernix-serve` option (carried, set by the tool, left out, refused), `capture.py` drives the forward-pass test,
`direction.py` computes the direction, `fit.py` runs the Optuna study, `measure.py` grades refusals and KL,
`candidates.py` pins, grades and exports, `evaluate.py` runs the capability tasks, and `doctor.py` checks a run
before it starts. `tests/test_ablate_tools.py` runs every command on CPU against the fake engine in
`tests/ablate_fakes.py`.
