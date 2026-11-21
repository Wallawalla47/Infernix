# LoRA adapters

`--lora PATH` applies a LoRA adapter to a Qwen3.5-family dense model (the Qwen3.8-27B artifacts) at runtime. The
artifact is unchanged: the engine reads the adapter's own Hugging Face files at load and adds the adapter's term to
every projection it targets. Every product command takes the option: `infernix-serve`, `infernix` (CLI),
`infernix-perplexity` and `infernix-bench`, and the launcher's model page has a LoRA adapter field.

## The adapter's files

A LoRA adapter on Hugging Face (one trained with PEFT, the format `transformers` and most fine-tuning tools save)
is a folder holding two files:

- `adapter_model.safetensors`: the factors, a `lora_A` and a `lora_B` matrix for each adapted projection;
- `adapter_config.json`: the rank, `lora_alpha`, the per-module `rank_pattern` / `alpha_pattern`, `use_rslora` and
  the options that would change the computation.

`--lora` takes that folder, or its `adapter_model.safetensors` with `adapter_config.json` beside it. The factors may
be stored as F32, BF16 or F16. A GGUF adapter (llama.cpp's export of the same factors) is not read: download the
safetensors repository instead.

```
infernix-serve Qwen3.8-27B-NVIDIA-NVFP4-QAD-Infernix.infernix --lora C:\adapters\my-adapter --model-id my-adapter
```

`--model-id` names the adapted model in `/v1/models` and the responses; without it the server reports the
artifact's name. The startup log names the adapter, what it adapts and its VRAM:

```
LoRA adapter C:\adapters\my-adapter: 64 attention and 192 MLP projections in 64 layers, rank 32, scale 2, BF16 factors, base Qwen/Qwen3.8-27B; 304 MiB of VRAM
```

## What an adapter may adapt

For a projection `y = W x` the adapter makes `y = W x + s B (A x)`, with `A` its `lora_A` ([r, in]), `B` its
`lora_B` ([out, r]) and `s` its scale: `lora_alpha / r`, or `lora_alpha / sqrt(r)` with `use_rslora`, each module's
`r` and `lora_alpha` from `rank_pattern` and `alpha_pattern` where a key matches its path (as PEFT matches them).

| Module | Where |
|---|---|
| `self_attn.q_proj`, `k_proj`, `v_proj`, `o_proj` | the full-attention layers (every fourth). `q_proj`'s rows are each head's query and then its output gate's, as in the checkpoint |
| `mlp.gate_proj`, `up_proj`, `down_proj` | every layer's dense MLP |

Ranks up to 64. Module paths are the checkpoint's (`model.language_model.layers.N...`, or a text-only export's
`model.layers.N...`) and every module's shapes must be the model's, so an adapter trained for another base model is
refused by name.

Refused, naming the module or option:

- the Gated DeltaNet projections (`linear_attn.*`), vision, MTP, the embeddings and the head;
- the mixture-of-experts models (Qwen3.6-35B-A3B, Qwen3.5-122B-A10B, Qwen3.8-Flash-Next);
- everything that is not plain LoRA on linear weights: DoRA (`use_dora`), LoRA biases (`lora_bias`, `bias` other than
  `none`), `modules_to_save`, trainable tokens, `layer_replication`, `fan_in_fan_out`, aLoRA, QALoRA,
  `target_parameters`, and every variant's own `*_config` (an initialization method's config — LoftQ, EVA, CorDA,
  LoRA-GA — is accepted: it acts before training).

## How it is computed

The factors are rounded to BF16 at load (round to nearest even, PEFT's cast). Each projection input's adapters are
stacked: one call computes `U = A x` for q/k/v together, for gate/up together, and alone for o_proj and down_proj.
`U` is summed in FP32 in a fixed order (chunks of 256 values on the tensor cores, their sums in ascending order) and
stored as BF16. The term `c = B U` is a fixed chain of BF16 tensor-core steps from zero; its value `fl32(c s)` is
added in FP32 before the projection's BF16 rounding: to gate and up before the SwiGLU, and, where the projection adds
into the residual stream, before or after the residual's add as the kernel's own arithmetic forms that sum (a kernel
that fuses its scale's product into the residual's add takes the term on the sum, so that its own arithmetic is
unchanged). Every kernel of every route adds the same term, the kernels of one column-invariant route in the same
place, so:

- the term is the same whatever the route, the call's width and the column's position: the reproducible route
  (`--reproducible`) stays deterministic and request-invariant with an adapter;
- `--lora` changes no fast path: an artifact's qualified fast paths run as without it, and the term is exact
  arithmetic beside them;
- without `--lora` every kernel is the one it was (the adapter forms are separate instances).

PEFT (as `PeftModel.from_pretrained` loads an adapter) computes the adapter path in FP32 and adds it to the
projection's BF16-rounded output. Here the factors and `U` are rounded to BF16, and the term is added before the
projection's own rounding instead of after it; the [lossless route](maintainer/lossless-route.md#2-qwen35-operations)
defines PEFT's own arithmetic.

On the lossless route (`--lossless`, the quality reference) the adapter is computed as PEFT computes it when it
loads an adapter: the factors in FP32, `u = A x` and `B u` each an exact sum rounded once to FP32, the scaled term
added to the projection's BF16 output in FP32 and rounded back to BF16
([definition](maintainer/lossless-route.md#2-qwen35-operations)). Its factors (A as three exact BF16 parts, B in
FP32) take about two and a half times the default route's VRAM.

The fused routes keep their fusion: the NVFP4 27Bs' fused dense MLP and Quasar's fused norm/attention projection
store the BF16 input the adapter reads (`h`, and the SwiGLU product for `down_proj`) beside the quantized planes
they already write.

## Speculative decoding and the prefix cache

The drafters (MTP, DFlash2) are not adapted: they propose the base model's likely tokens and the adapted model
verifies them, so the output is the adapted model's whatever the drafter. An adapter that moves the model's
choices a long way lowers the acceptance rate, and with it the speed-up.

The saved prefix cache (`--prefix-cache-file`) belongs to the artifact and the adapter: a cache saved without the
adapter, or with another (or a changed) one, is not reused.

## Memory and speed

The factors take `2 r (in + out)` bytes per adapted projection: a rank-32 adapter on all seven projections of the
27B takes 304 MiB of VRAM, which the KV cache gives up (a start allocates the factors before it sizes the KV, and
the CUDA Graph allowance grows with the adapted sites). The startup log names the adapter's VRAM; `--plan` reports
it as `vram.adapter_bytes` (in `planned_bytes`), the server's memory telemetry as the device part `adapter`, and the
launcher shows it in **Will it fit?** (checked again when the adapter changes) and in the dashboard's Memory panel.
Loading reads one adapted site's factors at a time, so it holds a few MB of RAM, not a copy of the factors.

Each adapted input costs one more kernel (the `A x` of its stacked adapters). Each projection adds the term in its
epilogue. The decode kernels copy their rows of `B` and columns of `U` into shared memory beside their main loop
(their weights stream through L2, so `B` would be gone by the epilogue), and the NVFP4 prompt tile copies them once
for the whole tile; the other tiles read them from L2, one fragment at a time.

The cost of a rank-32 adapter on all seven projections, against the same artifact without one (RTX 5090,
`infernix_bench` with INT8 KV, one run of three repetitions each, 2026-11-21; the term's cost depends on the rank and
the adapted modules, not on the factors' values, so a zero adapter measured it):

| 27B artifact | Prefill, 2,048 tokens | Decode | Decode, `--spec auto` |
|---|---|---|---|
| NVIDIA NVFP4 QAD | −9.9 % | −5.2 % | −5.0 % |
| NVIDIA NVFP4 | −8.3 % | −4.9 % | −5.4 % |
| Official NVFP4 | −9.5 % | −4.7 % | −5.0 % |
| Quasar NVFP4 | −10.9 % | −5.6 % | −6.0 % |
| Uncensored | −8.4 % | −4.5 % | −5.3 % |
| Groupwise (Q4/Q5) | −2.1 % | −5.7 % | −6.0 % |

On the QAD and groupwise artifacts, decode at C=2 costs −7.4 % and −7.1 %, prefill of 4,096 tokens −9.8 % and
−2.2 % (16,384 tokens: −8.7 % and −2.0 %), and on the reproducible route prefill of 2,048 tokens −12.1 % and −2.5 %
and decode −4.9 % and −9.1 %; decode with `--spec mtp` costs −3.0 to −5.3 % and −5.5 % (two runs each).

The zero adapter left the drafters' acceptance exactly as without it; a trained adapter changes acceptance only
through the model's choices ([above](#speculative-decoding-and-the-prefix-cache)).
