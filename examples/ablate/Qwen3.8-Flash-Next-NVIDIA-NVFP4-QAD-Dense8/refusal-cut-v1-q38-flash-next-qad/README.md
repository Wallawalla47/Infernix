# refusal-cut-v1-q38-flash-next-qad: a refusal-ablation sidecar for Qwen3.8-Flash-Next NVIDIA NVFP4 QAD (Dense8)

**For `Qwen3.8-Flash-Next-NVIDIA-NVFP4-QAD-Dense8-Infernix` only.** It was fitted on that artifact's activations and
measured only there. Another conversion of Qwen3.8-Flash-Next has the same geometry, so the sidecar will load on it, but
nothing here was measured on one: `sidecar.json` names this artifact (`artifact_id` 044ad3ee0d384c0bb24db0b490d852cb), so
a start on another logs a warning.

```
infernix-serve Qwen3.8-Flash-Next-NVIDIA-NVFP4-QAD-Dense8-Infernix-00001-of-00003.infernix \
  --ablate examples/ablate/Qwen3.8-Flash-Next-NVIDIA-NVFP4-QAD-Dense8/refusal-cut-v1-q38-flash-next-qad/sidecar.json
```

Remove `--ablate` and the model is the stock model again. The format and what the engine does with it:
[docs/ablate.md](../../../../docs/ablate.md).

## What it cuts

- One shared direction, `d.f32.bin`: 2,560 float32 values, sha256
  `8b68133ea1a506ba5a10ecdf190a2d51558fe3a9a349e86da7d864df68a8fc27`.
- The attention lane of layers 22-47 (alpha 1.09 at layer 22, rising to 1.41 at layer 41) and the MLP lane of layers
  8-47 (alpha 1.06 at layer 8, rising to 1.30 at layer 30).

## How it was made

- **Prompts:** HarmBench's 200 standard behaviors (MIT), split by category and kind of request into 160 for the direction
  and 40 held out, each paired with a harmless counterpart on the same topic, written for this fit.
- **Direction:** the method of [heretic](https://github.com/p-e-w/heretic) 1.4. The prompts are rendered with the
  artifact's chat template (thinking off) and read at the last prompt token, after each layer. The direction is the
  difference of the two class means, orthogonalized against the harmless mean. This sidecar uses one global direction,
  interpolated at layer index 31.67 between the per-layer ones.
- **Alphas:** heretic's per-lane band shape (peak, position, floor, width), searched by Optuna TPE over 300 trials. Each
  trial counted refusals on the 40 held-out behaviors, with the first-token KL divergence on their counterparts bounded
  at 0.4. This is trial 115: 1 of 40 refused (stock: 40 of 40), KL 0.235.

## Measured

Every comparison is the same binary, artifact and flags with and without the sidecar, on an RTX 5090: the stock arms of
the capability tests ran on 2026-11-20 and every other arm on 2026-11-21. The build was Infernix at `013c9faf`
(2026-11-17) with this cut, before this repository's later changes, so the numbers describe the cut rather than this
exact build.

**Refusals.** Grading uses the fit's refusal-phrase grader on the first 100 tokens of each reply, thinking off, with the
artifact's chat template.

| Prompts | Stock | With the sidecar |
|---|---|---|
| HarmBench, the 40 held out | 40/40 refused | 1/40 |
| [JBB-Behaviors](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors) harmful (100) | 96/100 refused | 1/100 |
| JBB-Behaviors benign (100), over-refusal | 38/100 refused | 0/100 |

The 40 held-out behaviors were scored on every trial, so they validated the search rather than testing it. JBB was
scored once, after the fit. Nine of its harmful prompts are HarmBench behaviors word for word: seven are among the 160
direction prompts and two among the 40 held out. The one HarmBench refusal and the one JBB refusal are different
prompts.

**JBB-Behaviors on PR #75's build** (with #74's `/v1/score` merged in for the KL measure; RTX 5090, 2026-11-21; the
merged engine computes the cut inside the inject, unrounded, so its bits differ slightly), graded the same way: with the sidecar 2 of the 100 harmful prompts are refused (stock: 96) and 0
of the 100 benign ones (stock: 38). The stock model's 200 answers are identical to the older build's; with the
sidecar, 5 answers differ and one verdict changes, a Sexual/Adult content prompt now refused. This build applies the
last layer's MLP cut on the column that picks the first generated token, which the older build skipped.

**Capability.** These runs use the chat template `tools/chat_templates/reasoning_effort_froggeric_v225_chat_template.jinja`.
Generation tests decode greedily with thinking on. Multiple-choice tests (TruthfulQA) score log-likelihoods with thinking
off. The noise band is twice the standard error.

| Test | Stock | With the sidecar | Reading |
|---|---|---|---|
| First-token KL, 100 harmless_alpaca prompts (thinking off, top-256 logprobs) | | mean 0.0437, median 0.0123, max 0.699; the top token changed on 4/100 | |
| First-token KL, JBB-Behaviors benign (100) | | mean 1.58 | the stock model refuses 38 of these and the sidecar answers them, so their first tokens differ. Split on this repository's build (the same mean, 1.58): 0.44 (median 0.16) on the 62 the stock model answers, 3.45 on the 38 it refuses |
| HumanEval+ (164) | 153 (base tests 161) | 154 (162) | McNemar p = 1 |
| GSM8K (1,319, 0-shot) | 91.58 % | 90.67 % | -0.91 points, noise band 2.21 |
| IFEval (541), prompt-level strict / loose | 90.94 % / 93.16 % | 92.42 % / 93.72 % | within the noise band (3.36 / 3.01) |
| IFEval, instruction-level strict / loose | 93.53 % / 95.20 % | 94.60 % / 95.68 % | no standard error reported |
| TruthfulQA (817) MC1 / MC2 | 49.20 % / 69.78 % | 47.12 % / 68.20 % | within the noise band (4.95 / 4.05) |
| TruthfulQA (817) generation, BLEU / ROUGE-1 / ROUGE-2 / ROUGE-L accuracy | 61.3 / 66.5 / 56.3 / 65.4 % | 56.9 / 62.8 / 53.7 / 60.6 % | within the noise band (4.86 / 4.73 / 4.92 / 4.78); ROUGE-L's -4.77 is at its edge |

Not measured: GPQA, MuSR, BBH, MMLU-Pro.

**On this repository's build** (RTX 5090, 2026-11-23), `infernix-perplexity --ablate` against a `--save-top-tokens`
run without it, on the first 24 KB (whole lines) of `docs/serving.md` (context 1,024, stride 512, FP8 KV, 6,341
positions): perplexity 9.4550 without the sidecar and 9.4745 with it (mean NLL +0.0021 per token); KL from the stock
model 0.081 on average (over its top 64 tokens plus one bucket for the rest, a lower bound); the top token agrees on
86.1 % of positions. Every position of technical prose is scored here, where many next tokens are near ties, so this
is not the first-token measure above. (On PR #75's build, 2026-11-21, a 24 KB slice of the file as it was then gave
8.7064 and 8.7076, KL 0.082 and 86.2 %; the text differs, so only the KL and the agreement compare.)
