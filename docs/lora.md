# LoRA adapters

`--lora PATH` applies a LoRA adapter to a Qwen3.5-family dense model (the Qwen3.8-27B artifacts) at runtime. The
artifact is unchanged: the engine reads the adapter's own Hugging Face files at load and adds the adapter's term to
every projection it targets. Every product command takes the option: `infernix-serve`, `infernix` (CLI),
`infernix-perplexity` and `infernix-bench`. The launcher finds adapters on Hugging Face and downloads them
(**Discover → LoRA adapters**), lists the ones in its model folders, and its model page has a LoRA adapter picker
([launcher guide](launcher.md#download-models)).

## The adapter's files

A LoRA adapter on Hugging Face (one trained with PEFT, the format `transformers` and most fine-tuning tools save)
is a folder holding two files:

- `adapter_model.safetensors`: the factors, a `lora_A` and a `lora_B` matrix for each adapted projection;
- `adapter_config.json`: the rank, `lora_alpha`, the per-module `rank_pattern` / `alpha_pattern`, `use_rslora` and
  the options that would change the computation.

`--lora` takes that folder, or its `adapter_model.safetensors` with `adapter_config.json` beside it. The factors may
be stored as F32, BF16 or F16. A GGUF adapter (llama.cpp's export of the same factors) is not read: download the
safetensors repository instead. Only these two files are needed; an adapter repository's tokenizer and chat template
copies are not read (the artifact's own are used).

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
| `linear_attn.in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj` | the Gated DeltaNet (linear-attention) layers: the query, key and value channels the convolution reads, the output gate, the decay and beta controls, and the output projection |
| `mlp.gate_proj`, `up_proj`, `down_proj` | every layer's dense MLP |

Ranks up to 128 (the largest of the adapter's modules' ranks, `r` and `rank_pattern`). Module paths are the
checkpoint's (`model.language_model.layers.N...`, or a text-only export's `model.layers.N...`) and every module's
shapes must be the model's, so an adapter trained for another base model is refused by name.

Refused, with a message that names the adapter and says why ("LoRA adapter C:\adapters\x is refused: ..."):

- a rank above 128: "its rank is 256 (module '...'), and Infernix currently supports LoRA adapters up to rank 128";
- vision, MTP, the embeddings and the head, and other linear-attention leaves (Qwen3-Next's fused `in_proj_qkvz` and
  `in_proj_ba` among them);
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

In a Gated DeltaNet layer the four adapters that read the normalized input `h` (`in_proj_qkv`, `in_proj_z`,
`in_proj_a`, `in_proj_b`) are stacked into one `U = A h`. The projection's term reaches each query, key and value
channel before the convolution reads it, and `z` before its gate; decode and verification then run the projection
materialized and the convolution over it, where without an adapter some routes fuse the two. The control's term is
added to `a` and `b` before their gates (`g = -exp(A_log) softplus(a + dt_bias)`, `beta = sigmoid(b)`): an adapted
control runs the input norm on its own, then the shrink, then one kernel for both controls and their terms (the same
schedule at every width, so the reproducible route stays request-invariant). Those two are other kernels than the
unadapted model runs, so a zero adapter on these modules does not give its bits (everywhere else it does). `out_proj`
takes its term as `o_proj` does. On the lossless route the terms are added to the BF16 `qkv`, `z`, `a` and `b` and
to `out_proj`'s output, as PEFT adds them.

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

Each adapted input costs one more kernel, the `A x` of its stacked adapters. A prompt's call takes the schedule
modelled fastest for its shape: for most prompts a tile schedule, a cluster of CTAs over K's groups in rounds whose
tile and cluster size follow how many clusters the GPU holds at once; for decode, verification and short prompts of
the narrow stacks, the split and stream schedules. Every schedule gives the same bits. Each projection adds the term
in its epilogue. The decode kernels copy their rows of `B` and columns of `U` into shared memory beside their main
loop (their weights stream through L2, so `B` would be gone by the epilogue); the NVFP4 prompt tile copies them once
for the whole tile, a wide adapter's FP8 tiles copy them too, and so do the groupwise ring tiles (the plain row-split
ring only for a wide adapter); the other tiles read
them from L2, one fragment at a time.

Every adapted kernel has two instances, chosen by the adapter's rank (padded to a multiple of 16): up to 64,
the narrow one, which holds the term's operands for 64; above, to 128, the wide one, which holds them for 128. They
compute the same chain (a rank-32 adapter padded with zeros to 128 gives the same bits through either), and the narrow
instances do not depend on the wide class, so supporting wide adapters costs a narrow adapter nothing.

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
and decode −4.9 % and −9.1 %; decode with `--spec mtp` costs −3.0 to −5.3 % and −5.5 % (two runs each). The
2026-11-22 shrink and epilogue changes below made rank-32 prefill 0.7 % and 0.5 % faster on QAD and groupwise and
left decode as it was.

A wider adapter costs more. The same trained rank-32 adapter, padded with zeros to ranks 96 and 128 (so all three
give the same bits), against the same artifact without one (the same benchmark, prefill of 2,048 tokens and decode of
128, 2026-11-22; each adapter in two pairs of runs against the build before the changes below, one run without an
adapter; `--spec auto`: 2026-11-21, a column the changes below moved by under 0.05 % at rank 128):

| 27B artifact | Rank | Prefill, 2,048 tokens | Decode | Decode, `--spec auto` |
|---|---|---|---|---|
| NVIDIA NVFP4 QAD | 32 | −9.5 % | −5.0 % | +0.6 % |
| | 96 | −15.8 % | −7.3 % | −3.3 % |
| | 128 | −18.1 % | −8.6 % | −4.9 % |
| Groupwise (Q4/Q5) | 32 | −1.7 % | −5.7 % | −4.6 % |
| | 96 | −4.2 % | −8.7 % | −10.2 % |
| | 128 | −5.1 % | −9.9 % | −12.8 % |

Unlike the zero adapter above, this one changes the model's choices and with them the drafters' acceptance (161
accepted tokens against 148 without it on QAD, 144 against 141 on groupwise, the same at every rank), so the
`--spec auto` column holds that change too; the rank's own cost is the difference between its rows. Prefill's cost
grows less than in proportion to the rank (on QAD 13 ms more than the 128 ms of 2,048 tokens without an adapter at
rank 32, 28 ms at rank 128), decode's much less. At rank 128, prefill of 2,048 tokens on the reproducible route costs
−20.6 % and −5.0 % and decode −10.6 % and −16.0 % (decode at C=2, 2026-11-21: −11.8 % and −15.8 %). On the
NVIDIA NVFP4, Official NVFP4, Uncensored and Quasar 27Bs prefill of 2,048 tokens costs −15.3, −15.3, −15.6 and
−19.4 % at rank 128 (−9.1, −8.9, −9.3 and −10.9 % at rank 32, 2026-11-21). Harness Hub's super-adapter (rank 96,
trained) costs what the padded rank-96 adapter does (decode −7.3 % and with `--spec auto` −2.3 % on QAD, 162 accepted
tokens against 148; −8.9 and −10.4 % on groupwise, 2026-11-21). A rank-128 adapter on all seven projections takes
1,216 MiB of VRAM, a rank-96 one 912 MiB.

Against the build before them, the 2026-11-22 changes made prefill of 2,048 tokens 10.2 % faster on QAD at rank 128
(7.8 % at 96, 0.7 % at 32), 8.2-9.7 % on the other NVFP4 27Bs at rank 128, 3.3 % on groupwise (2.4 % at 96, 0.5 % at
32) and on the reproducible route 10.7 % and 5.1 %. Shorter prompts gain too: at rank 128, 256 and 1,024 tokens
7.6 and 9.6 % on QAD, 5.6 and 5.9 % on groupwise; at rank 32 within 0.3 %. Decode, with and without `--spec auto`,
moved by at most 0.13 % in any pair of runs. The `A x` kernel took most of it: a cluster that the GPU cannot hold all at once runs in waves (the RTX 5090 holds 22 clusters of
6 CTAs at one CTA an SM), so the tile and the cluster now follow the GPU's capacity, and its operands arrive by TMA
under per-stage barriers instead of a barrier for the whole CTA. At rank 128 and 2,048 tokens it takes 91 us for
`down_proj` (222 before), 74 for q/k/v (134) and 55 for gate/up (89), and 80 for a Gated DeltaNet layer's input stack
(the split schedule, the previous rule's choice for that height, takes 280). One narrow case is slower: o_proj at
rank 32 and 1,536 tokens, 16.5 us against 14.5.

An adapter of the Gated DeltaNet modules costs more than one of the same rank on the others, mostly in decode: those
layers then run their projection materialized and the convolution over it, and an adapted control its own kernels.
Against the same artifact without an adapter (the same benchmark, 2026-11-22, one run each):

| Adapter | VRAM | QAD prefill | QAD decode | Groupwise prefill | Groupwise decode |
|---|---|---|---|---|---|
| code-tools (rank 64, every module of every layer) | 891 MiB | −18.1 % | −10.5 % | −4.3 % | −11.5 % |
| Atlassian v2 (rank 128, layers 32-63: `in_proj_qkv`, `out_proj`, attention and MLP) | 801 MiB | −12.6 % | −5.9 % | −3.5 % | −7.2 % |
| a rank-128 adapter of every module of every layer | 1,782 MiB | −26.1 % | −14.4 % | −7.1 % | −15.5 % |

The zero adapter left the drafters' acceptance exactly as without it; a trained adapter changes acceptance only
through the model's choices ([above](#speculative-decoding-and-the-prefix-cache)).
