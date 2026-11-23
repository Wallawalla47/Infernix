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
INFO  engine | ablation sidecar .../refusal-cut-v1-q38-flash-next-qad/sidecar.json: one shared direction, attn lane 26 layers (22-47), mlp lane 40 layers (8-47); source=HarmBench standard behaviors (MIT): 160 of the 200, ...
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

`alpha` 1 removes the component along `d` entirely; values above 1 reverse part of it. The cut runs inside the
lane's inject into the residual streams (`ops::hyper_connection_cut_inject`, and `ops::moe_combine_cut_inject` where
the MoE combine injects its output directly): the dot product is summed in FP32 in a fixed order, one CUDA block per
column, and the cut value goes into the inject unrounded, so each residual element keeps the inject's one BF16
rounding. A column's result depends only on that column, so a cut run is as deterministic and request-invariant as
the run without it. A layer or lane the sidecar leaves uncut runs the plain inject, and without `--ablate` nothing
loads and every kernel is the one that runs without the flag.

Cutting a direction out of a projection's output is a rank-1 low-rank term in principle, but `--lora`'s kernels do not
serve it: its term is fused into Qwen3.5's adapted projections, which Qwen3.8-Flash-Next does not run, and the MoE
output is a sum over experts rather than one projection. `--ablate` follows `--lora`'s layout instead: the sidecar
is read at load as Model data (`models/qwen4_exp/load/ablate`), its direction is the Parameters' device copy
(`execution/ablate_parameters`, reported as `adapter_bytes`), and the saved prefix cache's fingerprint stamps it.

## The sidecar's files

A sidecar is a `sidecar.json` with its direction file beside it (raw little-endian float32, `direction_dim` values per
row). Every field is checked at load; anything malformed fails the start, naming the problem, rather than serving uncut.

| Field | Meaning |
|---|---|
| `format` | `ablate-direction-v2` |
| `direction_dtype` | `float32` |
| `direction_dim` | the model's hidden size, an integer (2560 on Qwen3.8-Flash-Next) |
| `direction_layout` | optional: `shared` (one row for every layer, the default) or `per_layer` (`num_hidden_layers` rows) |
| `direction_file` | the direction file's name, beside `sidecar.json` |
| `default_alpha` | optional; must be 0: a layer missing from a table is not cut |
| `alpha_attn_per_layer`, `alpha_mlp_per_layer` | layer index (a string) to alpha, each in [0, 4] |
| `alpha_per_layer` | one table for both lanes, instead of the two |
| `generation` | required: `generation.source` says what produced the direction; `seed` and `notes` are optional strings. The engine logs them and does not interpret them |
| `artifact_id` | optional: the 32-hex-digit id of the artifact the direction was fitted on. A start on another artifact logs a warning (the sidecar loads, as the geometry matches, but was not measured there) |

Any other key is refused, so a misspelt table cannot load as a lane left uncut. Rows a cut layer reads must be finite
with a norm above 0.5 (they are normalized at load); rows of uncut layers may be zero. Refused, naming the cause: a
missing or unreadable file, an unknown key or format, a wrong dtype or dimension, a non-zero `default_alpha`, a layer
key outside the model, a non-finite or out-of-range alpha, no `direction_file`, a direction file of the wrong size, a
degenerate row on a cut layer, and no `generation.source`.

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

The direction takes `rows x 2560 x 4` bytes of VRAM: 10 KiB shared, 480 KiB as a per-layer stack (the memory
summary's `adapter_bytes`). A cut lane launches no extra kernel and takes no workspace: its inject is the cut inject in
place of the plain one.

## Making a sidecar

A direction is the difference between the mean hidden states of prompts the model refuses and of matched prompts it
answers, at the last prompt token, orthogonalized against the harmless mean; the alphas are then searched per lane to
minimize refusals on held-out prompts while bounding the first-token KL divergence from the original model. This
follows [heretic](https://github.com/p-e-w/heretic) 1.4's projected abliteration. The tooling that captures the
activations and fits the alphas is not part of this repository yet.
