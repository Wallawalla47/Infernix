# Perplexity evaluation

`ninfer-perplexity` measures the causal perplexity produced by a v3 `.ninfer` artifact.
It uses the artifact's tokenizer, Text model, selected Main KV representation, final normalization,
and main output head. It is an offline evaluator, not a serving endpoint or a logits-export API.
Only Text weights and resources are loaded; Vision and speculative components are not required.

## Run the fixed corpus

The repository includes `ninfer-ppl-1m-v1`, a fixed set of 16 independent UTF-8 streams covering
English reference text, English long-form text, Chinese reference text, and NInfer C++/CUDA code.
`full` selects all streams; `--quick` selects one stream from each domain.

```bash
./build/apps/ninfer-perplexity models/qwen3_8_27b_nvfp4.ninfer \
  --corpus eval/corpora/perplexity-1m/manifest.json \
  --quick \
  --kv-dtype fp8
```

The default evaluation uses a 4,096-token context and a 2,048-token stride. Use `--context` and
`--stride` to change that protocol, or score one UTF-8 file with `--text FILE`. Each window is
prefilled in passes of `--prefill-chunk` tokens (a multiple of 128, default 1024), recorded in
`report.json` as `prefill_chunk_tokens`; INT8, NVFP4, FP8 and K8V4 KV round it down to whole
prompt-attention waves (896 tokens on Qwen3.8-27B). The pass size changes only the order of
rounding, but on artifacts with 4-bit activations that order matters more than it seems: on the
official Qwen3.8-27B NVFP4 artifact, chunks of 256-7168 tokens moved perplexity by -0.6 % to
+0.8 % against one pass per window, with no trend in the chunk size, and every chunked run sat at
a KL divergence of 0.025-0.033 from the one-pass run (about 4 % of positions change their top
token), as large as switching the KV cache between BF16 and INT8. The Q6 artifact, with 16-bit
activations, moved 15 times less (KL 0.0021). So treat a perplexity difference under about 1 %
between two NVFP4 builds or settings as noise unless it holds at several chunk sizes.
`--a16-activations` runs every linear with 16-bit activations (about 3x slower on NVFP4
artifacts): saved with `--save-top-tokens`, it is the reference that `--kl-reference` measures
other runs against. [Model-level numerical quality](maintainer/model-quality.md) has the method and
the recorded results. The available Main
KV representations are `bf16`, `int8`, `fp8`, `nvfp4`, `k8v4`, `vq2`, and `k4v2`; unlike `ninfer` and
`ninfer-serve`, which default to `bf16`, `ninfer-perplexity` defaults to `fp8`. `int8` scores with the fast
prompt-attention kernel, and `nvfp4` with its fast kernel over more than 768 visible keys, as
`ninfer-serve` prefills them by default; `--use-original-int8-prefill-kernel` and
`--use-original-nvfp4-prefill-kernel` score them with the original kernels and require the matching
`--kv-dtype`. Prompt attention's P×V form follows `ninfer-serve`'s per-format default: the fast
NVFP4 kernel, `k8v4` and `vq2` use their 8-bit forms; INT8 and `k4v2` keep FP16. `--prefill-8bit-pv` and
`--no-prefill-8bit-pv` force either form. `report.json` records them as
`original_int8_prefill_kernel`, `prefill_8bit_pv` and `original_nvfp4_prefill_kernel`.

```bash
./build/apps/ninfer-perplexity models/qwen3_8_27b.ninfer \
  --text notes.txt \
  --context 16384 --stride 8192 \
  --kv-dtype int8
```

Run `./build/apps/ninfer-perplexity --help` for the complete command surface. The evaluator loads
the model once, reads and tokenizes every selected stream before scoring, and writes readable
startup, corpus, scoring, and per-stream summaries to stderr. Interactive weight loading and
scoring use one transient progress line; redirected scoring emits persistent progress every ten
seconds. `--log-level debug` exposes internal startup and stream-begin detail. The final
domain/overall table remains product output on stdout; the independent full-precision machine
report is `report.json` under `profiles/perplexity/` unless `--output` supplies an empty directory.

For KV-format comparisons, the recommended long-context profile is the full corpus with
`--context 65536 --stride 32768` and without `--quick`.

## KL divergence from a reference run

Perplexity only sees each target token's probability, so a numerical change can move it in either
direction, and a few streams can dominate the result. A KL divergence comparison measures how far
each next-token distribution moved. First record a reference, normally with `--kv-dtype bf16`:

```bash
./build/apps/ninfer-perplexity models/qwen3_8_27b.ninfer \
  --corpus eval/corpora/perplexity-1m/manifest.json --context 65536 --stride 32768 \
  --kv-dtype bf16 --save-top-tokens ref-bf16-64k.bin
```

`--save-top-tokens FILE` writes each scored position's 32 most probable tokens and their
log-probabilities (about 270 MB for the full corpus). Then score the setting under test with the
same corpus, mode, context and stride:

```bash
./build/apps/ninfer-perplexity models/qwen3_8_27b.ninfer \
  --corpus eval/corpora/perplexity-1m/manifest.json --context 65536 --stride 32768 \
  --kv-dtype nvfp4 --kl-reference ref-bf16-64k.bin
```

The run scores the reference's 32 tokens at every position and reports KL(reference ‖ this run)
over those tokens plus one bucket for all other tokens. Merging the other tokens can only lower the
divergence, so the result is a lower bound of the full KL divergence; `mean_reference_top_mass`
shows how much probability the 32 tokens cover. It also reports top-1 agreement (the same most
probable token) and the mean target-token NLL change. `report.json` gains a `kl_divergence`
object: overall mean, standard deviation and quantiles, and the same statistics by context length,
the number of tokens before the predicted one, in buckets `[0,1K)`, `[1K,2K)`, `[2K,4K)` and so on.
Each stream carries its own `kl_divergence`, so stream-level paired comparisons are possible. The
evaluator refuses a reference recorded for a different corpus, mode, context or stride, or whose
windows hold different tokens.

## Runtime YaRN override

`--rope-yarn-factor F` accepts finite values in `[1,4]` (default `1`, native RoPE unchanged).
This startup-fixed Engine override does not alter the artifact or converter. It only extends the
allowed ceiling: request longer scoring windows explicitly with `--context`; the default remains
4,096 tokens and the stride remains 2,048. Memory limits still apply. Long-context extrapolation
is not a guarantee of quality. Keep the factor fixed when comparing other numerical settings;
`report.json` records it as `rope_yarn_factor` in the execution configuration.

## Metric

For a stream `x[0..N)`, every token after `x[0]` is scored exactly once. A window `[b,e)` with target
suffix `[s,e)` contributes:

```text
log p(x[i] | x[b], ..., x[i-1])  for i in [s,e)
```

Each window starts from empty State and Main KV, so history before `b` is deliberately excluded.
The reported metric is therefore fixed-window, truncated-context causal perplexity:

```text
mean_nll = -sum(logprob) / scored_tokens
perplexity = exp(mean_nll)
```

The first window scores `[1,min(context,N))`. Each later window advances by `stride` targets while
retaining up to `context-stride` preceding tokens as local context. Streams never share history.

## Comparing runs

For a numerical comparison, keep the corpus, context, stride, and execution settings fixed except
the variable being measured. Compare KV formats with the same artifact and weight formats with the
same KV format.

The corpus name is a workload scale, not an exact token count. Exact input and scored-token counts
are runtime results from the current artifact tokenizer and are recorded in each report. Reports
contain unrounded NLL/PPL values for every window, stream, domain, and the token-weighted overall
aggregate.

The schema-v3 report identifies the artifact's architecture, public name, actual weight formats
and prefill signature alongside the workload and numerical results.
