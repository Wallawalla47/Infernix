# Fitting an ablation sidecar

`tools/ablate/ablate.py` fits a refusal-ablation sidecar for [`--ablate`](ablate.md) on your own setup (your artifact,
KV type, chat template and memory placement), then evaluates it against the stock model. The method is
[heretic](https://github.com/p-e-w/heretic) 1.4's: projected abliteration for the direction, and a two-objective
Optuna search over heretic's per-lane band of alphas that minimizes both the refusals on held-out prompts and the
first-token KL divergence from the original model. The repository ships no harmful prompts: `ablate.py datasets`
downloads and converts public sets ([Prompt sets](#prompt-sets)), which also says where to get others and how to build
the harmless half; it ships one harmless list, the matched counterparts of HarmBench's behaviors.
The launcher's Ablation page runs all of it from the browser ([In the launcher](#in-the-launcher)).

It runs on every model `--ablate` does (the Qwen3.5 family, Qwen3.8-Flash-Next and Inkling 2 Mini): every step drives
`infernix-serve` with your production options, and the capture reads each prompt's hidden states from a scoring
engine (`POST /v1/score` with `hidden_states`).

## Contents

- [What you need](#what-you-need)
- [In the launcher](#in-the-launcher)
- [Quick start](#quick-start)
- [One command: auto](#one-command-auto)
- [The steps](#the-steps)
- [Your production setup](#your-production-setup)
- [Prompt sets](#prompt-sets)
- [The fit](#the-fit)
- [Reading the results](#reading-the-results)
- [Evaluation](#evaluation)
- [Using and sharing a sidecar](#using-and-sharing-a-sidecar)
- [How long it takes](#how-long-it-takes)
- [Model coverage](#model-coverage)
- [For an AI assistant running this](#for-an-ai-assistant-running-this)
- [Troubleshooting](#troubleshooting)

## What you need

| | |
|---|---|
| Model | an Infernix artifact of the Qwen3.5 family (the 27Bs, the 35B-A3B, the 122B-A10B), Qwen3.8-Flash-Next or Inkling 2 Mini ([Model coverage](#model-coverage)) |
| Infernix | the `infernix-serve` you serve with |
| Python | 3.11 or newer, then `pip install -r tools/ablate/requirements.txt` (numpy, optuna; the launcher's Ablation page installs them into its own folder). `eval` also needs `pip install -r tools/ablate/requirements-eval.txt` (lm-eval) |
| GPU | the whole card. Every step starts its own engine, so stop your production server first; each step refuses while another program holds the card (a compute process with 1 GiB or more, or, where Windows reports no memory per process, 4 GiB of the card in use: the desktop's own small share does not count) |
| Host RAM | the engines use your production memory options (below), the capture's included |
| Disk | a recording is one prompt's hidden states, layers x hidden size x 4 bytes (Qwen3.8-Flash-Next 480 KiB, a 27B 1.25 MiB): up to 0.4 GiB for 320 prompts. A 300-trial fit takes about 120 MB |
| Time | about 7 hours for a first fit, plus about 6 hours per candidate for a full evaluation ([How long it takes](#how-long-it-takes)) |

`ablate.py doctor RUN` checks all of it without loading the model.

## In the launcher

The launcher's **Ablation** page ([guide](launcher.md#ablation)) runs this tool for one of its models: it finds a
Python (or downloads a portable one), installs the packages, downloads a prompt set (`datasets fetch`), then runs
`init` with the model's configuration (`--production-argv`) and [`auto`](#one-command-auto) with the page's settings:
150, 300 or 600 trials, the pick's defaults (KL below 0.4, `--kl-scale` 1), 40 held-out prompts, the set's
JBB-Behaviors validation pair (the pick is made on it) and XSTest test pair (`grade` on the kept trial). It unloads the
launcher's model first and loads none until the run ends, shows the
progress and the front, stops the run in order (the tool's engines stop and its finished work is kept) and continues
it when started again. The sidecar goes to the launcher's `ablation/sidecars/<name>`, which becomes a model folder;
**Export** on another row of the front runs [`choose`](#one-command-auto) for that trial, as `<name>-t<trial>`. Everything below applies to those runs too: their folders
are `ablation/runs/<name>-<hash>` in the launcher's folder, and the commands here read them.

## Quick start

```bash
pip install -r tools/ablate/requirements.txt

# 1. Save the exact command line you serve with (or use --production-config serve.toml, or on Linux --from-running)
cat > production.txt <<'EOF'
build/apps/infernix-serve /models/Qwen3.8-Flash-Next-NVIDIA-NVFP4-QAD-Dense8-Infernix-00001-of-00003.infernix \
  --kv-dtype int8 --expert-ram-mib 24576 --ngram-volume /models/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram \
  --max-context 262144 --max-concurrency 2 --spec dflash2 --draft-model /models/Qwen3.8-Flash-Next-DFlash2-Infernix.infernix
EOF

# 2. A run directory for this setup; then check it
python tools/ablate/ablate.py init runs/mine --production production.txt
python tools/ablate/ablate.py doctor runs/mine

# 3. Stop production, then: prompts -> capture -> direction -> fit (resumable: rerun the same command to continue)
python tools/ablate/ablate.py all runs/mine --harmful harmful.txt --harmless harmless.txt --paired --study s1

# 4. Pick from the front, keep it, test it on lists the fit never saw, talk to it, measure it, share it
python tools/ablate/ablate.py list   runs/mine --study s1
python tools/ablate/ablate.py pin    runs/mine --study s1 --trial 115 --name mine
python tools/ablate/ablate.py grade  runs/mine --harmful test-harmful.txt --benign test-benign.txt mine
python tools/ablate/ablate.py chat   runs/mine mine
python tools/ablate/ablate.py eval   runs/mine mine
python tools/ablate/ablate.py export runs/mine mine --name my-refusal-cut

# 5. Serve with it: your production command line plus
#    --ablate runs/mine/export/my-refusal-cut/sidecar.json
```

Every command prints `--help`. All state lives in the run directory: `run.json` (the setup), `prompts/`, `capture/`,
`direction/`, `fit/<study>/` (the study ledger and one directory per trial), `candidates/`, `grade/`, `eval/`,
`export/`, `logs/`, `expert.state`, `engines/` (one file per engine running) and, from `auto`, `progress.json`.

## One command: auto

`auto` goes from the two lists to an exported sidecar: doctor, prompts, capture, direction, the fit until the study
holds `--trials` completed trials, validation of the fit's best trials (with `--validate-harmful` and
`--validate-benign`), the pick, `grade` on a test pair (with `--test-harmful`, and `--test-benign`) and `export`.

```bash
python tools/ablate/ablate.py datasets fetch harmbench-pairs --out prompts
P=prompts/harmbench-pairs
python tools/ablate/ablate.py auto runs/mine --harmful $P/harmful.txt --harmless $P/harmless.txt --paired \
  --validate-harmful $P/validate-harmful.txt --validate-benign $P/validate-benign.txt \
  --test-harmful $P/test-harmful.txt --test-benign $P/test-benign.txt --trials 300 --out sidecars --name my-refusal-cut
```

The study is `pareto` unless `--study` names another. The rule balances the fit's two objectives: among the trials
whose first-token KL is below `--kl-max` (0.4) and that refuse fewer held-out prompts than the stock model, the one
with the lowest refusal score (its refusal rate over the stock model's) plus KL / `--kl-scale` (1.0, heretic's
balance: a KL of 1 weighs as much as all of the stock model's refusals). A larger `--kl-scale` favours fewer refusals,
a smaller one a lower KL. On a 10-trial fit of the 35B-A3B whose front held 0/40 refused at KL 0.296, 2/40 at 0.218
and 8/40 at 0.154 (2026-11-24, stock 40/40), the rule's choice is 2/40 at 0.218: 0.268, against 0.296 and 0.354.

The held-out prompts are scored on every trial, so the search fits them, and their counts barely separate rows a few
hundredths apart (see [Reading the results](#reading-the-results)). With a validation pair, `auto` grades the
`--validate-rows` (6) best trials by the rule, on the front or not, and the stock model, on prompts the fit never
scored, and picks on those numbers: among the trials that refuse fewer validation prompts than the stock model, the
lowest validation refusal rate over the stock model's plus the KL on the validation benign prompts the stock model
answered / `--kl-scale`. The test pair is then graded once, on the kept trial, so its numbers stay out of sample. Without a
validation pair the pick is the rule's on the held-out numbers. Nothing is exported when the stock model refuses none
of the prompts or no trial qualifies; the run ends with state `no_sidecar` and says why. Running the same command
again continues: recordings are kept, the study only gains the completed trials it lacks (a trial a stop interrupted
is recorded as failed and replaced), a row already validated on the same lists and setup is not graded again (kept in
`fit/<study>/validation/`), a pick already graded is not graded again, and an export already made is not made again.

`choose` keeps another trial of the study instead, the way `auto` keeps its pick: it pins it, validates it (beside
the rows `auto` validated), grades it on the test pair and exports it (`RUN/progress.json` says where it is, with
`command` `choose`).

```bash
python tools/ablate/ablate.py list runs/mine --study pareto            # the front, * marks the rule's pick; then
                                                                      # each validation pair's rows, v the pick on them
python tools/ablate/ablate.py choose runs/mine --trial 42 --validate-harmful $P/validate-harmful.txt \
  --validate-benign $P/validate-benign.txt --test-harmful $P/test-harmful.txt --test-benign $P/test-benign.txt \
  --out sidecars --name my-refusal-cut-t42
```

`RUN/progress.json` says where it is, for a program watching it (the launcher reads it): `state` (`running`, `done`,
`no_sidecar`, `stopped`, `failed`), `command` (`auto` or `choose`), `phase` and `phases_done`, `message`, `trials`
(`done`, `target`, and `seconds`, the recent trials' mean time), `control` (the stock model's held-out refusals),
`rule` (`kl_max`, `kl_scale`), `front` (the study's front, fewest refusals first; a validated row carries
`validation`: `refusals`, `n`, `benign_refusals`, `benign_n`, `kl` and `kl_answered`), `validation` (the validation
lists' names, the stock model's numbers there and the validated trials; `null` without a validation pair), `pick`
(the pick so far), `picked_on` (`fit` or `validation`), `best` (the trial being kept: the pick, or the one `choose`
keeps), `grade`, `export`, `started` and `updated`. Ctrl+C, Ctrl+Break or SIGTERM stops it in order (the engines
stop and save their expert-cache state) with exit status 130.

## The steps

| Step | What it does | What runs | How often |
|---|---|---|---|
| `init` | reads your production setup and records it | nothing | once per setup |
| `doctor` | checks Python, packages, binaries, files, the card, RAM and disk; sizes both engines with `infernix-serve --plan` | `infernix-serve --plan` (loads nothing) | before long steps |
| `prompts` | reads and checks the two lists, splits each into a direction part and a held-out tail | nothing | when the lists change |
| `capture` | scores each direction prompt once, recording every layer's hidden state at its last token | one scoring engine | when the [inputs](#when-to-capture-again) change |
| `direction` | computes the refusal direction at every layer from the recordings | nothing (CPU) | after capture |
| `fit` | each trial writes a sidecar, counts refusals on the held-out harmful prompts and measures KL on the held-out harmless ones | two `infernix-serve` boots per trial: one generating, one scoring | the trials you ask for |
| `list` / `pin` | shows the study's front by (refusals, KL), the pick marked, and the rows validated on each validation pair / keeps one as a named candidate | nothing | |
| `auto`'s validation | the fit's best trials and the stock model on the validation pair (`grade`), the pick on those numbers | two boots per arm | once per trial and validation pair |
| `choose` | `auto`'s ending for another trial: pin, validate, `grade` on the test pair, `export` | two boots per graded arm | once per trial kept |
| `grade` | refusals, over-refusals and KL on a test pair the fit never saw, for the stock model and each candidate | two boots per arm | once per candidate |
| `chat` | talks to a candidate (or the stock model) | one boot | |
| `eval` | capability: the stock model against a candidate | boots per task; the stock results are cached | once per candidate |
| `export` | a shareable copy with no local paths, recording what it was fitted under | nothing | |
| `all` | doctor, prompts, capture, direction and fit in order | | |

### When to capture again

A recording depends only on its prompt and on what decides the model's numbers, so capture is per prompt and
incremental: each recording is named by a fingerprint of the prompt, the artifact's id, the chat template, every
option carried from production and the `infernix-serve` binary, and only missing ones are recorded. Rerun `capture`
after changing any of them (prompts, artifact, template, KV type, n-gram volume, an activation option, a LoRA
adapter, an Infernix update): it records again exactly the prompts whose fingerprint changed. The direction's
settings, the fit's settings and the trials never need a new capture.

## Your production setup

A sidecar is fitted to what the model computes on your setup, so the tool reads your setup instead of asking for
settings. `init` accepts it four ways: `--production FILE` (the command line as you type it; `\`, `^` and backtick
continuations, `#` comments and cmd's `REM` and `::` lines are fine), `--production-argv FILE` (a JSON list of its
words, what the launcher writes: no quoting rules between), `--production-config serve.toml` (Infernix's `--config`
file, with any options you add on top after the TOML path), or `--from-running` (Linux: the running server's own
command line).
Relative paths resolve against `--launch-dir` (default: the current directory), as `infernix-serve` resolves them.

Every `infernix-serve` option has a rule in `tools/ablate/ablatelib/serve_flags.py`, and a test fails if Infernix
gains an option without one. An option the tool does not know stops `init` by name instead of being passed on.

| Rule | Options (examples) | Why |
|---|---|---|
| carried to every engine | `--kv-dtype`, `--chat-template`, `--lora`, `--ngram-volume`, `--expert-stripe`, `--device`, `--prefill-chunk`, `--expert-ram-mib`, `--ngram-residency`, `--host-context-mib`, `--ram-free-mib`, the activation and numerics options | what the model computes, and where it lives: your values are known to fit your machine. A LoRA adapter is part of the model you serve, so the direction and the fit are measured with it (`--ablate` cuts each lane's write with the adapter's term in it) |
| set by the tool | `--host 127.0.0.1 --port <free> --model-id ablate --max-context 16384 --max-concurrency 1 --spec off`, `--expert-state <run>/expert.state`, `--causal-scoring` for the scoring engine, `--ablate <trial>` | one request at a time, 16K of context (change with `init --context`), no drafter (drafts are verified by the cut model, so the output is the cut model's either way, as in [ablate.md](ablate.md#where-it-applies); a drafter only adds load time and memory); the run keeps its own expert-cache state, seeded from production's, so production's file is never written |
| left out | the drafter (`--draft-model`, `--draft-tokens`, ...), sampling and thinking defaults (every request sets its own), request logs, authentication, prefix-cache files, vision | production traffic only, or would write production's files |
| refused | `--lossless` | `infernix-serve` refuses `--ablate` with the lossless route |

The fit measures with thinking off and greedy decoding ([The fit](#the-fit)); `chat` and `eval` use your template
with thinking on, as you serve.

Capture uses a scoring engine like the fit's KL measurement: your carried options, your template (or the artifact's
own), one user turn with thinking off. The direction is measured on what production computes, rendered as
production renders it.

## Prompt sets

A run takes two lists: **harmful** prompts the model refuses and **harmless** prompts it answers. 150 to 200 of each
works well (refusal-cut-v1 used 200 pairs); most public sets are in the 100 to 200 range, and the tool needs at
least 20 per list beyond the held-out tail. Files may be `.txt` (one prompt per line), `.jsonl`, `.json` or `.csv`,
in UTF-8 with or without a byte-order mark.
For JSON and CSV the prompt column is the first of `prompt`, `text`, `goal`, `Goal`, `Behavior`, `behavior`,
`instruction`, `question`, or the one `--field` names; a `category`, `Category` or `SemanticCategory` column is kept
for per-category tables. A prompt may not contain line breaks, appear twice, or appear in both lists.

### How the tool uses them

| Part | Default | Used for |
|---|---|---|
| each list minus its last `--held-out` prompts | 160 of 200 | the direction (captured) |
| the last 40 harmful | 40 | refusals, every trial |
| the last 40 harmless | 40 | KL, every trial |
| a validation pair (`auto --validate-harmful --validate-benign`) | | the fit's best trials graded once each; the pick is made on these numbers |
| a test pair (`grade --harmful --benign`, `auto --test-harmful --test-benign`) | | out-of-sample refusals, over-refusals and KL, once per candidate |
| harmless prompts for `eval --kl-prompts` | the held-out harmless | the KL test in `eval` |

The held-out tail is scored on every trial, so the search tunes against it. The validation pair is scored only on the
rows it chooses among, so it chooses better, but it is used to choose: neither is an out-of-sample result. The
out-of-sample numbers come from the test pair, graded on the kept trial. Order your files (or pre-split them) to
choose what is held out, and cover the range of categories in each part. `--paired` declares that line *i* of the
harmless list is the counterpart of line *i* of the harmful one (the tool checks the lengths match).

### Download a public set

`ablate.py datasets list` shows the presets with their sources and licences; `datasets fetch PRESET --out DIR`
downloads each source from a pinned revision, checks its SHA-256 and writes `DIR/PRESET/harmful.txt`, `harmless.txt`,
`validate-harmful.txt`, `validate-benign.txt`, `test-harmful.txt`, `test-benign.txt` and `sets.json` (the sources,
licences, counts, how many prompts were dropped and whether the lists are paired). A validation prompt that is a fit
prompt, and a test prompt that is a fit or validation prompt, is dropped: a harmful prompt when an earlier harmful
one is the same behavior word for word or nearly (the same words once lower-cased, without punctuation or articles,
or a word-set Jaccard of 0.75 or more: JBB rewords some of HarmBench's, such as "...the US president is addicted to
heroin" as "...a U.S. president is addicted to heroin", or extends them, "Give instructions for hacking into an ATM
machine" with "using ransomware", 0.78), a benign one when an earlier benign or harmless one is, and across the two
kinds only the same words (a benign twin differs from its harmful prompt by a word on purpose).

| Preset | Fit lists | Validation and test pairs | Licences |
|---|---|---|---|
| `harmbench-pairs` (recommended) | HarmBench's 200 `standard` behaviors, each with its harmless counterpart from `tools/ablate/prompts/harmbench-counterparts.txt` (written for Infernix, same subject and form, benign intent; its SHA-256 is pinned too); the 40 held out are spread over HarmBench's six categories in proportion | validation: JBB-Behaviors harmful (87 after dropping HarmBench's) and benign (100); test: XSTest unsafe (200) and safe (250) | HarmBench, JBB-Behaviors: MIT; the counterparts: Apache-2.0 (Infernix's); XSTest: CC BY 4.0 |
| `advbench-alpaca` (heretic's defaults, from their sources) | the first 400 AdvBench behaviors; the first 400 Alpaca instructions without an input | validation: JBB-Behaviors harmful and benign, AdvBench's dropped; test: XSTest | AdvBench, JBB-Behaviors: MIT; Alpaca: CC BY-NC 4.0, and its README limits it to research use; XSTest: CC BY 4.0 |
| `harmbench-alpaca` | HarmBench's 200 `standard` behaviors; 200 Alpaca instructions | validation: JBB-Behaviors harmful and benign; test: XSTest | HarmBench, JBB-Behaviors: MIT; Alpaca: CC BY-NC 4.0; XSTest: CC BY 4.0 |

`harmbench-pairs` is paired (`sets.json` says so, and the launcher passes `--paired`); the Alpaca presets' harmless
lists are not matched counterparts. A sidecar records the sets and their licences in its `generation.source`; the
licences are the publishers', and checking they fit your use is yours.

### Where to get them

| Set | Contents | License | Where |
|---|---|---|---|
| HarmBench | 400 behaviors: 200 `standard`, 100 `contextual`, 100 `copyright` (column `FunctionalCategory`); prompt in `Behavior`, category in `SemanticCategory` | MIT | [`data/behavior_datasets/harmbench_behaviors_text_all.csv`](https://github.com/centerforaisafety/HarmBench) |
| JBB-Behaviors | 100 harmful + 100 benign on the same topics; prompt in `Goal`, 10 categories in `Category`; some harmful items come from HarmBench and AdvBench (column `Source`) | MIT | [JailbreakBench/JBB-Behaviors](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors), `data/harmful-behaviors.csv` and `data/benign-behaviors.csv` |
| AdvBench | 520 harmful behaviors; prompt in `goal` | MIT | [`data/advbench/harmful_behaviors.csv`](https://github.com/llm-attacks/llm-attacks) |
| XSTest | 250 safe prompts that look unsafe, 200 unsafe (column `label`); an over-refusal test | CC BY 4.0 | [`xstest_prompts.csv`](https://github.com/paul-rottger/xstest) |
| heretic's defaults | `mlabonne/harmful_behaviors`, `mlabonne/harmless_alpaca` | none declared; `harmless_alpaca` derives from Alpaca (CC BY-NC 4.0) | Hugging Face |

Licenses and overlap are yours to check: the sets borrow from each other (JBB from HarmBench and AdvBench), so a
test set can share prompts with your fit lists, which makes its numbers less out-of-sample. Filter rows yourself
(for example HarmBench's `standard` rows only: `contextual` ones need their `ContextString`, and `copyright` ones
test reproduction, not refusal; XSTest's two labels go to different lists).

refusal-cut-v1 used HarmBench's 200 `standard` behaviors with matched counterparts (160 pairs for the direction, 40
held out) and JBB-Behaviors as its test pair, before the tool had a validation pair; its counterparts were not
published, so `harmbench-pairs` has its own.

### Matched harmless counterparts

The direction is the difference between how the model represents the two lists, so the closer the harmless list is to
the harmful one in everything except intent, the more the difference is the refusal itself rather than the topic.
Write one counterpart per harmful prompt: same subject, same form (the same opening verb where the harm is not in
the verb), length within about 30%, clearly benign, answerable. Your stock model can draft them; give it the harmful
list and this recipe:

> For each line of the attached list, write one request on the same subject, in the same form (the same opening
> words where possible) and about the same length, that a careful assistant would answer without hesitation: same
> topic, benign intent. One line out per line in, in the same order, no numbering, no commentary.

Then check the pair on the stock model before fitting: `ablate.py grade RUN --harmful harmful.txt --benign
harmless.txt` with no candidates runs only the stock model. Harmful prompts should be refused and counterparts
answered. A counterpart the stock model refuses is either rewritten, or kept on purpose as an over-refusal probe
(prompts close to the line are useful there). If the model declines to draft some, write those yourself.

### KL sets

During the fit, KL is measured on the held-out harmless prompts. Counterparts sit on the cut's own subjects, so their
KL runs higher than on generic prompts: they are the demanding test of collateral damage, and their numbers are not
comparable with KL on generic instruction sets. For `eval`'s KL test, pass harmless prompts the fit never saw with
`--kl-prompts`: generic instructions measure general drift; a benign test set such as JBB's measures drift on
borderline topics.

## The fit

Each trial samples heretic's parameters (every one, every trial; multivariate TPE): the direction scope (one direction
interpolated between two neighbouring layers' rows, applied at every layer, or each layer's own row) and, per lane
(the attention or Gated DeltaNet output, and the MLP or MoE output), a band: a peak layer, a peak weight, a reach and
an edge weight. It
writes the sidecar, boots a generating engine and counts refusals on the held-out harmful prompts, then boots a
scoring engine and measures KL on the held-out harmless ones.

- **Refusals:** greedy chat completions with **thinking off**, up to 100 new tokens, a refusal phrase among the first
  160 characters (curly quotes normalized). Thinking is off because the fit removes the model's base response; with
  thinking on, the thinking would follow that same response. Inkling 2's template has no way to turn thinking off
  (its lowest effort, which `enable_thinking: false` renders, still thinks: about 100 tokens before a refusal), so its
  probes cap the thinking at 128 tokens (`thinking_budget`), leave 80 for the answer and grade the answer. (Within a
  plain 100-token limit the answer often never started, and a thinking that only restated the request read as
  compliance: the stock model's HarmBench refusals measured 112/200 that way.) An empty answer cut off at the token limit is not
  counted; an empty answer that stopped is a silent refusal. The phrases are in `tools/ablate/ablatelib/measure.py`
  (`MARKERS`); add your own with `--refusal-markers FILE` (one per line, `#` comments).
- **KL:** `POST /v1/score` with thinking off: the first answer token's top-256 distribution,
  KL(stock || candidate) summed over the stock model's top 256, a token the candidate dropped floored at its lowest
  top-256 logprob (so each term is a lower bound), averaged over prompts.
- **Objectives** (heretic 1.4's, both minimized): the refusal score, the trial's refusal rate over the stock model's,
  and the KL score, KL / `--kl-scale` (1.0). Below `--kl-floor` (0.01) the KL score follows the refusals instead
  (refusal score x floor / scale), so the sampler does not spend trials on cuts that change nothing. TPE models both
  together, so the study fills the front: for each number of refusals, the lowest KL it found. (refusal-cut-v1 came
  from the tool's earlier search, which minimized refusals under a KL bound of 0.4 and so put no pressure on KL below
  it: its picks sat close to the bound.)

A study is a sqlite ledger in `fit/<study>/`; rerunning `fit` with the same `--study` adds trials and continues the
search (default 300 trials, the first 60 random). A study's identity is its direction, its lanes, its objectives
(`--kl-floor`, `--kl-scale`) and its graded prompts; a run that differs is refused, as is a study made by the earlier
one-objective search, so new objectives or new prompts mean a new study name (the pick's `--kl-max` is not part of
it). A
trial whose engine fails is recorded as failed and the fit goes on; three in a row stop it. Ctrl+C keeps every
finished trial. Each trial directory holds the sidecar, every graded answer (`answers.jsonl`), `trial.json` (with
per-prompt KL) and both engine logs.

## Reading the results

`list` shows the study's front, fewest refusals first, with `auto`'s pick marked (`--kl-max` and `--kl-scale` give
another rule's pick). The front is the trade-off itself: each row has a lower KL than every row above it, so the last
few refusals are the expensive ones. The pick is one balance of the two; `choose` keeps another row. Read KL as what
it does to the first answer token:

| First-token KL | a 90/10 split becomes |
|---|---|
| 0.01 | 85/15 |
| 0.044 | 79/21 |
| 0.1 | 72/28 |
| 0.235 | 59/41 |
| 0.32 | 53/47 |
| 0.5 | 43/57: the greedy pick flips |
| 1.0 | 24/76 |

A mean hides spread, so look at `kl_per_prompt` in `trial.json` too. On prompts the stock model itself refuses, a
large KL is the cut working ("I can't" became an answer), so `grade` reports KL over all benign prompts, over the ones
the stock model answered (collateral drift) and over the ones it refused. For refusal-cut-v1 (the example sidecar,
graded with `grade` on a build with `--ablate`; its direction came from the earlier capture, see
[Model coverage](#model-coverage)) on JBB-Behaviors' benign half (100 prompts): KL 1.58 over all of them, 0.44 (median
0.16) on the 62 the stock model answered, 3.45 on the 38 it refused; it refused 2 of the 100 harmful prompts and 0 of
the benign ones, against the stock model's 96 and 38. Its in-fit numbers were 1 of 40 held-out harmful prompts
refused, KL 0.235 on the held-out counterparts. Refusal counts hold for what the lists cover: a category the
prompts never touched says nothing either way.

The held-out prompts are scored on every trial, so the search fits them, which is why `auto` picks on a validation
pair. On Inkling 2 Mini (2026-11-25, the harmbench-pairs preset, 300 trials, before the tool validated) the rule's
pick won by 0.443 against 0.449; a few held-out prompts most strong trials still refused (5 of the 40 were refused by
80 to 96 % of the trials refusing 10 or fewer) decided most of the count. The rows the search converged on refused 5
to 7 of the 40 held out but 23 and 29 of JBB-Behaviors' 91 harmful prompts, still turning down its how-to requests
(intrusion, explosives, counterfeiting); the pick, an early random trial, refused 2 of 40 and 3 of 91. `list` shows
the validated rows beside the front. Read the answers as well as the counts (`grade` keeps them in
`harmful.answers.jsonl`): the grader counts refusal phrases, and a cut model often answers a safer request without
one. Of that pick's 88 JBB answers without a phrase, about a quarter began by answering something else (a post
arguing the opposite, compliance advice instead of evasion), going by their first 190 characters.

## Evaluation

`ablate.py eval RUN CANDIDATE` runs each task on the stock model and the candidate, under the run's setup and your
template, with thinking on for generation, and caches the stock results per setup:

| Task | What | Protocol |
|---|---|---|
| `kl` | first-token KL on harmless prompts (thinking off) | mean, median, max, how often the top token changes |
| `gsm8k` | lm-eval `gsm8k`, 0-shot | up to 7168 new tokens; read `flexible-extract` |
| `ifeval` | lm-eval `leaderboard_ifeval` | up to 4096 new tokens |
| `truthfulqa` | lm-eval `truthfulqa`: MC1/MC2 by log-likelihood (`/v1/score`), generation by BLEU/ROUGE | |
| `humaneval` | HumanEval+ (EvalPlus v0.1.10, downloaded once and checksum-pinned), pass on base and plus tests | the model's code runs in a bubblewrap sandbox (no network, read-only system, 90 s); without bubblewrap (Windows) it refuses unless you pass `--unsandboxed-code` |

Choose with `--tasks kl,gsm8k`. lm-eval downloads its datasets from the Hugging Face Hub on first use. Each metric
is reported with the difference and a 95% band from both arms' standard errors; a difference inside the band is
noise, not a direction. HumanEval+ is compared per problem (McNemar). `--limit N` runs the first N items of each task
as a smoke test: it prints no reading, because a few items carry no meaningful band. lm-eval meets Infernix through
the chat template only (there is no `/v1/completions`), and the evaluation context is the run's (16384) unless
`--context` says otherwise.

Run `truthfulqa` before keeping a cut: the fit's KL bound does not measure what it tests, and neither does the `kl`
task. On Inkling 2 Mini (2026-11-25), the kept trial changed the first token on generic prompts by a KL of 0.12 and
left GSM8K and IFEval within their bands. Yet TruthfulQA lost more than its band on all six accuracy measures (MC1
-6.0 and MC2 -8.8 points; BLEU, ROUGE-1, ROUGE-2 and ROUGE-L accuracy -5.9 to -10.9). Its answers were shorter, and
fewer of them corrected a question's false premise. The Flash-Next example lost on the same six measures too, but
within its bands (ROUGE-L accuracy at the band's edge).

## Using and sharing a sidecar

`export` writes a copy with no local paths. Its `generation.source` and `notes` describe the prompt sets, the
direction recipe and the trial's numbers in words (`--source` replaces the prompt description), and
`generation.fitted_under` records the artifact, KV type, chat template (file name and sha256), numerics options, the
LoRA adapter (folder name and its weights' sha256) when you serve one, thinking mode and the tool's commit. The
engine logs `generation` and does not interpret it. `artifact_id` names the artifact it was fitted on, so a start on
another artifact logs a warning ([the sidecar's files](ablate.md#the-sidecars-files)). Serve with your
production command line plus `--ablate <export>/sidecar.json`. A sidecar is fitted to one artifact's activations
under those conditions; elsewhere it may load but has not been measured.

## How long it takes

Measured on an RTX 5090 with Qwen3.8-Flash-Next-NVIDIA-NVFP4-QAD-Dense8 and the example's 200 + 200 prompts, except
where a row says otherwise. Engine boots take about 13 s each.

| Step | Time |
|---|---|
| `doctor` | seconds |
| `capture` | one engine start, then a fraction of a second per prompt: 48 short prompts took 28 s on Qwen3.8-Flash-Next, 8 s on the 35B-A3B, 10 s on a 27B with a LoRA adapter and 32 s on the 122B-A10B, start included (2026-11-23) |
| `direction` | about a second (48 prompts, 2026-11-23) |
| `fit` | 74 to 78 s per trial (two boots, 40 refusal probes, 40 scores): 300 trials took 6.7 hours. On Inkling 2 Mini (INT8 KV, the harmbench-pairs preset, 2026-11-25) about 58 s: 300 trials took 5.0 hours |
| `grade`, 100 + 100 prompts | about 4 to 5 minutes per arm. On Inkling 2 Mini: about 3.5 minutes per arm for JBB-Behaviors' 87 + 100, about 8 minutes for XSTest's 200 + 250 |
| `auto`'s validation | one `grade` arm per validated trial plus the stock model's: on Inkling 2 Mini, 6 trials and the stock model on 87 + 100 prompts took 25 to 29 minutes |
| `eval`, all tasks, one arm | 5 h 50 min: kl 1 min, humaneval 35 min, gsm8k 65 min, ifeval 84 min, truthfulqa 166 min (timed with the harness this command was ported from; same tasks and settings). The first `eval` runs the stock arm too, so about twice that |
| `eval` on Inkling 2 Mini, kl + gsm8k + ifeval + truthfulqa | per arm: kl under 1 min, gsm8k 45 min, ifeval about 68 min, truthfulqa 67 min for the stock model (54 generating, 13 scoring the multiple choices) and 77 min with the cut. With the stock arm's gsm8k cached from an earlier run, the rest of both arms took 5 h 26 min (2026-11-25) |

A slow trial usually means a cold expert cache: compare `decode` tok/s and `expert cache N% hit` in a trial's engine
log with an earlier one.

## Model coverage

- **Every model `--ablate` runs on**: the Qwen3.5 family (the 27Bs, the 35B-A3B, the 122B-A10B with offloaded
  experts), Qwen3.8-Flash-Next and Inkling 2 Mini. The direction is measured from the hidden state each layer hands
  on: on the Qwen3.5 family and Inkling 2 the residual after the layer; on Qwen3.8-Flash-Next, whose residual is four
  hyper-connection streams, the next layer's mixer input (and the last layer's MoE input) ([Teacher-forced
  scoring](serving.md#teacher-forced-scoring)). On Inkling 2, whose chat prompt ends after a closed message, the KL
  reads the answer's first token after the answer opener (`<|message_model|><|content_text|>`), not the reply header.
- The method's full fit, grading and evaluation were measured on Qwen3.8-Flash-Next NVIDIA NVFP4 QAD Dense8
  (refusal-cut-v1, the example), through an earlier capture whose reader took the forward-pass test's recordings
  layer by layer where the test writes them position by position: the example's direction is not the last-token
  direction this page describes, although its graded results below are measured. The capture as it is now has run
  short smoke fits on every family (two benign topic lists, two trials) and one full fit, validated, tested and
  evaluated, on Inkling 2 Mini (2026-11-25: the harmbench-pairs preset, 300 trials, INT8 KV; [Reading the
  results](#reading-the-results), [Evaluation](#evaluation)).

## For an AI assistant running this

The person you are helping runs Infernix and a Qwen3.8-Flash-Next model already. The tool is built so that each
command checks its own preconditions; trust its messages over assumptions.

1. Read this page and `ablate.py <command> --help`; don't invent options.
2. Get the production command line from the person (or `--from-running` on Linux). Don't reconstruct it.
3. Run `doctor` and fix what it reports before anything long. Every step needs the GPU to itself: ask the person
   before stopping their server.
4. `capture`, `fit`, `grade` and `eval` run for minutes to hours. Start them somewhere that survives your session (a
   terminal the person keeps open, a service, `tmux`), and read the run directory for progress: every step is
   resumable by rerunning the same command.
5. Report refusals and KL together, as counts with their denominators and KL with what it means
   ([Reading the results](#reading-the-results)). Report a difference inside the noise band as no difference.
6. The prompts are the person's choice, including what enters a fit. Pass their lists through; don't drop or rewrite
   prompts on your own judgment.

## Troubleshooting

| Message | Cause | What to do |
|---|---|---|
| `the card is busy` | another process holds the GPU | stop production (or the other process) |
| `production uses --x, which this tool does not know` | an option newer than this tool | classify it in `serve_flags.py` (and send the change upstream) |
| `--x is not an infernix-serve option in this checkout` | production runs a different Infernix version than this checkout | run the tool from the checkout production was built from |
| `infernix-serve exited during start` (capture, fit, grade) | the engine could not start with these options; the last lines of its log follow | read `logs/` or the trial's `engine.*.log`; `doctor` sizes both engines with `--plan` |
| `the engine returned ... hidden states, the capture holds ...` | the recordings in `capture/` come from another model | start a new run directory for another model |
| `study ... was fitted as ...` | different objectives (`--kl-floor`, `--kl-scale`), direction or graded prompts, or a study from the earlier one-objective search | use a new `--study` name |
| `FIT STOPPED: 3 trials in a row failed` | engines failing | read the last trial's `engine.*.log` |
