# Ablation sidecars

`--ablate FILE` removes one direction from Qwen3.8-Flash-Next's residual writes at runtime (directional ablation, the
technique behind "abliterated" models). The artifact is unchanged: the engine reads a small sidecar at startup, and
without the flag nothing loads and nothing is hooked. A sidecar fitted to the refusal direction removes the model's
refusals while the rest of its behaviour stays close to the original; [examples/ablate](../examples/ablate) ships one,
with its measurements.

```
infernix-serve Qwen3.8-Flash-Next-NVIDIA-NVFP4-QAD-Dense8-Infernix-00001-of-00003.infernix \
  --ablate examples/ablate/Qwen3.8-Flash-Next-NVIDIA-NVFP4-QAD-Dense8/refusal-cut-v1-q38-flash-next-qad/sidecar.json
```

The startup log names the cut and where its direction came from:

```
INFO  engine | ablation sidecar .../refusal-cut-v1-q38-flash-next-qad/sidecar.json: v2, one shared direction, attn lane 26 layers (22-47), mlp lane 40 layers (8-47); source=HarmBench standard behaviors (MIT): 160 of the 200, ...
```

## What it computes

At each cut lane of layer L, for every token column `y` of that lane's output:

```
y -= alpha[L][lane] * dot(d, y) * d
```

with `d` a unit direction (one shared vector, or one per layer). Two lanes are cut independently:

| Lane | Where |
|---|---|
| `attn` | the layer's attention or Gated DeltaNet output, before it is added to the residual streams |
| `mlp` | the MoE output (routed and shared experts combined), before it is added to the residual streams |

`alpha` 1 removes the component along `d` entirely; values above 1 reverse part of it. The dot product is summed in
FP32 in a fixed order (one CUDA block per column), so a cut run is as deterministic as the run without it.

## The sidecar's files

A sidecar is a `sidecar.json` with its direction file beside it (raw little-endian float32, `direction_dim` values per
row). Every field is checked at load; anything malformed fails the start, naming the problem, rather than serving uncut.

| Field | Meaning |
|---|---|
| `format` | `ablate-direction-v2` (or `ablate-direction-v1`: one table for both lanes, no per-layer stack) |
| `direction_dtype` | `float32` |
| `direction_dim` | the model's hidden size (2560 on Qwen3.8-Flash-Next) |
| `direction_layout` | `shared` (one row for every layer) or `per_layer` (`num_hidden_layers` rows) |
| `direction_file` | the direction file's name, beside `sidecar.json` |
| `default_alpha` | must be 0: a layer missing from a table is not cut |
| `alpha_attn_per_layer`, `alpha_mlp_per_layer` | layer index (a string) to alpha, each in [0, 4] |
| `alpha_per_layer` | one table for both lanes (v1, or instead of the two in v2) |
| `generation` | v2 requires `generation.source`, saying what produced the direction; `notes` is optional. The engine logs it and does not interpret it |

Rows a cut layer reads must be finite with a norm above 0.5 (they are normalized at load); rows of uncut layers may be
zero. Refused, naming the cause: a missing or unreadable file, an unknown format, a wrong dtype or dimension, a non-zero
`default_alpha`, a layer key outside the model, a non-finite or out-of-range alpha, a direction file of the wrong size,
a degenerate row on a cut layer, and a v2 sidecar without `generation.source`.

## Where it applies

- Qwen3.8-Flash-Next (`qwen4_exp`) artifacts. Every other architecture refuses `--ablate` at startup.
- Every product command takes it: `infernix-serve`, `infernix` (the CLI), `infernix-perplexity` (which scores the cut
  model) and `infernix-bench`, and the launcher's configure page has an **Ablation sidecar** field for Flash-Next
  models. What a run applied is recorded beside the startup log line: `server_start.artifact.ablate_sidecar` in the
  request log, `execution.ablate_sidecar` in `infernix-perplexity`'s `report.json`, and `config.ablate` and
  `load.ablate_sidecar` in an `infernix-bench` report.
- Not with `--lossless` (refused).
- The drafters (MTP, DFlash2, n-gram) are not cut: they propose and the cut model verifies, so the output is the cut
  model's. A cut that moves the model's choices lowers the acceptance rate.
- The saved prefix cache (`--prefix-cache-file`) belongs to the artifact and the sidecar: a cache saved without the
  sidecar, or with another (or a changed one), is not reused.
- A sidecar is fitted to one artifact's activations. One fitted on another conversion of the same model may load (the
  geometry matches) but has not been measured there.

## Memory and speed

The direction takes `rows x 2560 x 4` bytes of VRAM: 10 KiB shared, 480 KiB as a per-layer stack. Each cut lane adds
one kernel per layer. A layer whose MLP lane is cut also forms the MoE output before adding it, as its own kernel
instead of the fused add, which takes one BF16 hidden-width buffer per call column (budgeted in the forward's
workspace when `--ablate` is given).

Measured cost, against the same build without `--ablate` (RTX 5090, a production argv: DFlash2, `--max-concurrency 2`,
`--max-context 262144`, INT8 KV; two fresh prompts of 11,431 and 11,413 tokens, 600 tokens out, one run per arm, the example sidecar):

| | Prefill, first / second prompt | Decode, first / second prompt | DFlash2 acceptance |
|---|---|---|---|
| without | 4.65k / 6.32k tok/s | 99.4 / 166.7 tok/s | 58.6 % / 62.7 % |
| with the example | 5.15k / 6.28k tok/s | 136.4 / 172.2 tok/s | 59.3 % / 66.0 % |

The first prompt after a start runs cold and varies most; on the second the cost is -0.6 % of prefill, inside a
run-to-run spread measured at about 5 %. Decode follows DFlash2's acceptance, which changes with the output.

## Making a sidecar

A direction is the difference between the mean hidden states of prompts the model refuses and of matched prompts it
answers, at the last prompt token, orthogonalized against the harmless mean; the alphas are then searched per lane to
minimize refusals on held-out prompts while bounding the first-token KL divergence from the original model. This
follows [heretic](https://github.com/p-e-w/heretic) 1.4's projected abliteration.
[Fitting an ablation sidecar](ablate-fitting.md) covers the tool that does it on your own setup
(`tools/ablate/ablate.py`): prompt sets, the fit, evaluation and export.
