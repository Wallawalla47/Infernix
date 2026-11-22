# Ablation sidecars

`--ablate FILE` removes one direction from a model's residual writes at runtime (directional ablation, the technique
behind "abliterated" models), on every architecture Infernix runs: Qwen3.5 and Qwen3.6/3.8 dense (the 27Bs), Qwen3.5-MoE
(the 35B-A3B, the 122B-A10B) and Qwen3.8-Flash-Next. The artifact is unchanged: the engine reads a small sidecar at
startup, and without the flag nothing loads and nothing is hooked. A sidecar fitted to the refusal direction removes
the model's refusals while the rest of its behaviour stays close to the original; [examples/ablate](../examples/ablate)
ships one for Qwen3.8-Flash-Next NVIDIA NVFP4 QAD Dense8, with its measurements. No sidecar has been fitted for a
Qwen3.5-family model yet.

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
| `attn` | the layer's attention or Gated DeltaNet output, before it is added to the residual |
| `mlp` | the MLP output (a dense MLP's down projection, or the MoE's routed and shared experts combined), before it is added to the residual |

`alpha` 1 removes the component along `d` entirely; values above 1 reverse part of it. The dot product is summed in
FP32 in a fixed order, one CUDA block per column, so a column's result depends only on that column: a cut run is as
deterministic and request-invariant as the run without it. A layer or lane the sidecar leaves uncut runs its plain
kernels, and without `--ablate` nothing loads and every kernel is the one that runs without the flag.

- **Qwen3.8-Flash-Next** writes its lanes into four hyper-connection streams. The cut runs inside the lane's inject
  (`ops::hyper_connection_cut_inject`, and `ops::moe_combine_cut_inject` where the MoE combine injects its output
  directly) and goes into it unrounded, so each residual element keeps the inject's one BF16 rounding.
- **Qwen3.5 and Qwen3.5-MoE** write one residual. A cut lane writes it with the kernels it runs without a sidecar (the
  fused attention, Gated DeltaNet and down projections, the 27B's fused NVFP4 MLP, the 35B's sparse MoE, the MoE
  combines, a LoRA term), after copying the residual columns the write covers; `ops::residual_delta_cut` then removes
  the direction's component of what the write added and rounds the residual to BF16 once more. A cut too small to
  move a residual element keeps its bits, and the result differs from the ideal `x + y - alpha (d . y) d` by the
  write's own rounding (its component along `d` scaled by `1 - alpha`) and that last rounding.

Cutting a direction out of a projection's output is a rank-1 low-rank term in principle, but `--lora`'s kernels do not
serve it: on Qwen3.8-Flash-Next its term is in none of the projections the model runs, the MoE lanes of every MoE
model are a sum over experts rather than one projection, and on Qwen3.5 a cut term could not share a projection with a
loaded `--lora` adapter (one term a row segment) and would round its coefficient to BF16 twice. `--ablate` follows
`--lora`'s layout instead: the sidecar is read at load as Model data (`models/ablate/binding`), its direction is the
Parameters' device copy (`models/ablate/parameters`, reported as `adapter_bytes`), and the saved prefix cache's
fingerprint stamps it. On Qwen3.5 a cut lane runs its LoRA term unchanged, since the cut acts after the write.

## The sidecar's files

A sidecar is a `sidecar.json` with its direction file beside it (raw little-endian float32, `direction_dim` values per
row). Every field is checked at load; anything malformed fails the start, naming the problem, rather than serving uncut.

| Field | Meaning |
|---|---|
| `format` | `ablate-direction-v2` |
| `direction_dtype` | `float32` |
| `direction_dim` | the model's hidden size, an integer (5120 on the 27Bs, 3072 on the 122B-A10B, 2560 on Qwen3.8-Flash-Next, 2048 on the 35B-A3B) |
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

- Every architecture: Qwen3.5 dense, Qwen3.5-MoE (resident and offloaded experts) and Qwen3.8-Flash-Next.
- Every product command takes it: `infernix-serve`, `infernix` (the CLI), `infernix-perplexity` (which scores the cut
  model) and `infernix-bench`, and the launcher's configure page has an **Ablation sidecar** field. What a run applied
  is recorded beside the startup log line: `server_start.artifact.ablate_sidecar` in the request log,
  `execution.ablate_sidecar` in `infernix-perplexity`'s `report.json`, and `config.ablate` and `load.ablate_sidecar`
  in an `infernix-bench` report.
- Not with `--lossless` (refused).
- The drafters (MTP, DFlash, DFlash2, n-gram) are not cut: they propose and the cut model verifies, so the output is
  the cut model's. A cut that moves the model's choices lowers the acceptance rate.
- With `--lora` (the Qwen3.5 dense models) both apply: the adapter's term is part of each lane's write the cut acts on.
- The saved prefix cache (`--prefix-cache-file`) belongs to the artifact and the sidecar: a cache saved without the
  sidecar, or with another (or a changed one), is not reused.
- A sidecar is fitted to one artifact's activations. One fitted on another conversion of the same model may load (the
  geometry matches) but has not been measured there.

## Memory and speed

The direction takes `rows x hidden_size x 4` bytes of VRAM (Qwen3.8-Flash-Next: 10 KiB shared, 480 KiB as a per-layer
stack), the memory summary's `adapter_bytes`: `infernix-serve --plan` counts it in `vram.adapter_bytes` (and with
offloaded experts, before the expert cache's frames), the server's memory telemetry in the device part `adapter`,
and the launcher shows it as **Adapters** in **Will it fit?** (checked again when the sidecar changes) and in the
dashboard's Memory panel. On Qwen3.5 a cut lane's copy of the residual is workspace, in `vram.workspace_bytes`.

On Qwen3.8-Flash-Next a cut lane launches no extra kernel and takes no workspace: its inject is the cut inject in place
of the plain one. Against the plain inject a cut inject costs about 0.4 us per call at one to 32 columns, and the fused
MoE combine with the cut about 1.5 us; at prefill widths both are faster than the plain kernels
([Direction cut Op benchmark](../bench/README.md#direction-cut-op-benchmark)). On Qwen3.5 a cut lane adds a copy of
the residual columns it writes and the cut kernel (and one BF16 [H, T] of workspace for the copy, reserved only with a
sidecar); its own kernels are the ones it runs without one.

Measured on Qwen3.8-Flash-Next (2026-11-23, RTX 5090, NVIDIA NVFP4 QAD Dense8, `infernix_bench` on the default route, INT8
KV, one lane, plain decode without a drafter, the n-gram table read from its volume; four fresh processes an arm in
A B C C B A blocks, each the mean of three repetitions):

| | pp1280 prefill | pp4096 prefill | tg256 decode |
|---|---|---|---|
| without `--ablate` | 940.7 tok/s | 2,410.5 tok/s | 149.97 tok/s |
| the example sidecar | 939.5 (-0.1 %) | 2,402.5 (-0.3 %) | 144.97 (-3.3 %) |
| the example's layers and lanes at alpha 1e-30 | 937.1 (-0.4 %) | 2,403.3 (-0.3 %) | 148.58 (-0.9 %) |

The prefill differences are inside the runs' spread (1.9 to 3.6 %). The third row runs the example's kernels with the
plain run's output, so -0.9 % of decode is the cut's own cost (its 40 fused MoE cuts a token). The rest of the
example's -3.3 % follows from the text it changes: its 256 tokens routed more experts outside the VRAM cache (95.4 %
hits against 96.5 %, 5,143 experts served by the CPU against 4,040). With a drafter the cut also changes the
acceptance rate.

Measured on Qwen3.5 (2026-11-23, RTX 5090, `infernix_bench` on the default route, INT8 KV, one lane, plain decode
without a drafter; four fresh processes an arm in A B C C B A blocks, each the mean of three repetitions), with a cut
of both lanes of every layer at alpha 1e-30: the most a sidecar can cut, at a strength too small to move the output
(on the reproducible route it keeps the plain run's bits). No Qwen3.5 sidecar has been fitted, so a real one's cost
(fewer lanes, and a changed text) is not measured.

| | pp1280 prefill | pp4096 prefill | tg256 decode |
|---|---|---|---|
| Qwen3.8-27B NVFP4 (NInfer's official artifact), without `--ablate` | 14,139.9 tok/s | 13,466.6 tok/s | 82.74 tok/s |
| the same, every lane cut (128 cuts a token) | 13,755.1 (-2.7 %) | 13,066.2 (-3.0 %) | 79.95 (-3.4 %) |
| Qwen3.6-35B-A3B, without `--ablate` | 23,159.5 tok/s | 24,891.3 tok/s | 414.01 tok/s |
| the same, every lane cut (80 cuts a token) | 22,797.5 (-1.6 %) | 24,571.2 (-1.3 %) | 381.51 (-7.9 %) |

The rows' run-to-run spreads are 0.04 to 0.84 %. A cut costs about 3 us of decode per lane (a copy and a kernel per cut), which
is a larger share of the 35B's 2.4 ms token than of the 27B's 12 ms; a sidecar cutting half the lanes costs about half.

## Making a sidecar

A direction is the difference between the mean hidden states of prompts the model refuses and of matched prompts it
answers, at the last prompt token, orthogonalized against the harmless mean; the alphas are then searched per lane to
minimize refusals on held-out prompts while bounding the first-token KL divergence from the original model. This
follows [heretic](https://github.com/p-e-w/heretic) 1.4's projected abliteration.
[Fitting an ablation sidecar](ablate-fitting.md) covers the tool that does it on your own setup
(`tools/ablate/ablate.py`): prompt sets, the fit, evaluation and export.
