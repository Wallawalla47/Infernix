# Ngram copy proposals

Ngram proposals accelerate repeated text, such as copying source code from a tool result into
an edit. They supplement the selected neural drafter; they are not an external knowledge store,
a model quantization mode, or a change to thinking settings. The feature is disabled by default.

## Enable

Add these options to an otherwise working single-request CLI or server configuration:

```sh
--max-concurrency 1 --ngram-draft-tokens 15 --ngram-min-match 12
```

For example, an MTP artifact can use:

```sh
build/apps/ninfer-serve /path/to/model.ninfer \
  --max-concurrency 1 --spec mtp --draft-tokens 3 --lm-head-draft \
  --ngram-draft-tokens 15 --ngram-min-match 12
```

Keep the artifact, context capacity, KV format and other residency options appropriate for
your GPU. Wider target verification needs additional graph, replay and workspace memory.
This example is not a memory-sizing recommendation.

| Selected backend | Neural draft tokens | Ngram draft tokens |
|---|---:|---:|
| MTP | 1..5 | 1..63 |
| DFlash | 1..15 | 1..63 |
| DFlash2 | 1..15 | 1..63 |

The artifact must contain the selected drafter. Qwen3.6-35B-A3B uses original DFlash;
that is distinct from the DFlash2 companion in supported Qwen3.8-27B artifacts.
Ngram does not convert one drafter into another. Multiple active requests with ngram
enabled are rejected. `--ngram-draft-tokens 0` disables the feature; the minimum match
defaults to 12 and its supported enabled range is 4..64.

Ngram width is a separate upper bound, not the neural drafter's step count.
Larger widths can reduce target rounds on long copy spans but increase per-round
attention, projection, replay and workspace costs. Measure both short and long
contexts before choosing a width; the longest supported width need not be fastest.
Each admitted provider keeps its configured physical verification width, including
partial copy spans. Unused columns are masked; output budgets do not select a
different ngram arithmetic shape. Source-ending admission can instead return the
round to the ordinary neural provider, as described below. DFlash and DFlash2 retain
an append buffer sized for the widest provider: even a narrow neural round must catch
up target features from a preceding wide copy round. That padded append work is an
additional cost when ngram is enabled.

The mixed-FP8 27B target retains 16-bit activations for FP8 residual projections
during 17..64-column single-request verification. Its ordinary narrow path uses
that precision already; crossing the core's 22/25-column A8 thresholds otherwise
adds another quantization change. Weight and KV formats, prefill, neural-only
decoding and multi-request batches are unchanged. This does not promise identical
floating-point results between different widths.

## Operation

Each request owns a bounded CPU index of source positions. Lookups use 16-, 8- and 4-token
suffixes, compare hits exactly and extend matches backward through up to 64 committed
tokens to select a source. A proposal
copies a contiguous span from that source. Hash collisions and overwritten positions cannot
license target output. Short or absent matches fall back to the selected neural drafter.

Source endings can need different surrounding text, such as tool-call framing.
Admission looks ahead by two extra CPU tokens, leaving the last source token beyond
the target's bonus for a neural round. Useful partial spans remain eligible; a short
tail offering no more drafts than the neural width falls back to that provider.
This policy does not inspect whitespace, token identities or client names, and adds
no GPU graph family. It is not a guarantee of correct file contents: every output
still needs the same caller-side validation as ordinary generation.

The index includes the prepared prompt and committed output, never rejected or pending
verification columns. Known special tokens end source spans. The default index has about
36 MiB of token and bucket storage per active request, plus bounded proposal-only sources
in the prepared prompt. This index is released with the request. Optional session retention
uses a separate bounded CPU archive, described below; unrelated clients never share it.
The prompt is indexed before derived tool sources. The larger index retains useful old
spans under diverse long-context pressure; bounded storage still does not guarantee a hit
for every previous span.

Tool results containing at least three consecutively numbered lines can provide an additional
de-numbered source. Recognized separators include colon-space, tab, an arrow and pipe-space;
an optional `L` prefix is supported. Indentation, CRLF and terminal newlines are preserved.
The original tool message, target tokens, positions and prefix-cache identity are unchanged.
Plain tool results and earlier generated edits can match directly without normalization.
This is protocol-neutral and does not require Kilo-, Codex- or Claude-specific engine paths.
Derived-source tokenization uses the engine's context budget and stops when that
budget is exhausted. This only limits optional proposal sources: it never truncates
the target prompt to make an oversized request valid. Preparation cancellation
is checked around source tokenization.

Every proposed token is verified by the target. MTP and DFlash retain their deterministic-draft
acceptance paths; DFlash2 represents the proposal as a one-hot distribution in its existing
sparse verifier. Existing target commit, stop, cancellation and cache-restore ownership remain
authoritative. Ordinary neural rounds keep their original verification width.

This implementation is position-indexed, not a literal port of llama.cpp's hashed next-token
chain. It does not promise bit-identical output across different verification widths: floating
point execution and quantized projection choices can depend on width, just as in other batched
or speculative decoding configurations. This caveat does not permit using unverified output
or publishing speculative state before commit.

## Retain Sources Through Compaction

For a local single-owner server, opt in to a RAM-only archive:

```sh
--ngram-archive-mib 512 --ngram-session-mib 128
```

Requests identify a conversation with `X-NInfer-Draft-Session`. Without an identity,
drafting remains request-local. Add `--ngram-native-sessions` to recognize Kilo's
`x-session-affinity`, Codex's `client_metadata.thread_id`, and Claude's JSON-encoded
`metadata.user_id.session_id`. This separate local opt-in permits draft retention
even when the client sends `store:false`; it does not enable Responses-object storage.
Conflicting or invalid identities fall back to request-local drafting. Session names
are routing identifiers inside the existing local trust boundary, not authentication.

The Engine fixes an immutable view at admission and publishes validated input and
committed output after successful generation. Cancellation and failure discard the
private overlay. Retention is independent of GPU/host prefix-cache survival. Retained
tokens never enter the target prompt or KV: they can accelerate a continuation the
target already considers likely, not recover knowledge absent from its context.

`X-NInfer-Draft-Reset: 1` clears the identified archive before a request and revokes
its old views/cursors. Restarting the Engine clears all archives. A new identity starts
empty. Explicit-header clients can fork the parent's latest completed generation using
`X-NInfer-Draft-Parent` and `X-NInfer-Draft-Generation`; a missing/stale parent or an
already existing destination does not merge histories. Native client forks receive
their own identity; inheritance is not guessed from similar prompts. Deleting a stored
Responses object is separate from clearing a draft session.
Generation numbers are opaque, Engine-lifetime identifiers, not per-session turn
counts. Clearing or evicting a session never recycles its number; clients must
discard generation references when the server restarts.

A successful publication reports its completed generation in the non-streaming
`X-NInfer-Draft-Generation` response header. Streaming responses put the standard SSE
comment `: ninfer-draft-generation: N` before their terminal event. Ordinary SSE
clients ignore this comment; explicit-fork clients can read it without access to
server logs. No generation is advertised for unbound or unpublished requests.

Immutable spans use exact-checked 16/8/4-token position indexes. Unchanged sources
deduplicate, and a validated copy cursor avoids repeat lookup while a span continues.
The cursor rechecks committed correction/bonus/control tokens and uses the same
source-ending handoff as request-local drafting. The archive is bounded by session
and Engine capacities, including indexes, metadata, staging and still-pinned sources;
pressure can evict sources or bypass retention without failing inference. It allocates
no VRAM and creates no archive files. The existing per-request index is additional.

Retention-enabled requests mix the supplied sampling seed with fresh request entropy.
This prevents retained generated proposals from depending on the same random draws
used to accept them during a retry. A seed alone does not reproduce an archived
conversation; disable retention for deterministic same-seed replay. Greedy target
selection is unaffected by this random-domain separation. The generation result
and structured request log expose the effective seed as `ngram_archive.sampling_seed`;
the ordinary request seed remains the caller's input. Reconstructing the same
proposal path additionally requires the same archive snapshot and model geometry.

## Evaluate

Compare ngram off and on with the same binary, artifact, precision, sampling, prompt, output
budget and context-cache state. Verify output correctness before reporting speed. Measure raw
file copies, numbered tool-file copies, edits with insertions/deletions, and non-copying reasoning
separately. Include short and near-capacity contexts and both cold and cached prompts.

Report computed prefill tokens separately from reused tokens. Decode throughput excludes the
first prefill-sampled output token; whole-request latency includes prefill and cache restoration.
Building the CPU index and derived sources adds host work, especially visible on short prefills.
Faster copies do not imply faster prefill or the same improvement across an entire coding turn.
Reasoning, novel code and short matches can leave little reusable text.

Request statistics include `ngram_rounds`, `ngram_drafted_tokens` and
`ngram_accepted_tokens` in addition to aggregate speculative statistics. A useful performance
test must demonstrate accepted ngram proposals, not just successfully enabling an option.
The product `ninfer_bench` accepts the same ngram flags and preserves these
counters in table, CSV and JSON reports. Its ordinary one-token decode seed is
not a copy workload; see [benchmark guidance](../bench/README.md).

The [scope proposal](https://github.com/Neroued/ninfer/issues/234) links RTX 5090
copy, edit and compaction measurements with their artifacts, toolchain, commands
and correctness checks. Those pre-v3 results belong to the source revisions in
the [evidence release](https://github.com/remesis/ninfer/releases/tag/ngram-mod-evidence-20260912),
not to later rebases. Remeasure the relevant workload before attributing the same
gain to another revision or configuration.

## Real-Artifact Checks

`ninfer_ngram_lifecycle_real` uses `NINFER_NGRAM_TEST_WEIGHTS`. For example,
`63 5 1 1 nvfp4 12 0 mtp` selects NG63, neural K5, CUDA Graphs, optimized proposal
head, NVFP4 KV, minimum match 12, no additional soak and MTP. It checks retained
device versus host-restored state, cancellation, queued isolation, output budgets,
partial pages and context tails. Append `--strict-fresh` to additionally require
bit-identical fresh-prefill and cached output. That separate diagnostic may return
2 on quantized paths, including upstream without ngram; all retained-state checks
remain mandatory. It skips without the artifact environment variable.

`ninfer_ngram_archive_real` uses `NINFER_NGRAM_TEST_WEIGHTS` and takes the backend
(`mtp`, `dflash` or `dflash2`) and optional graph mode (`0` or `1`). It exercises
NG63/K5 source-absent regeneration, isolated and explicitly forked sessions, reset,
cancellation/retry and publication generations with prefix reuse disabled. This
greedy, thinking-disabled contract fixture is not an agent-performance benchmark.
It skips without the artifact environment variable.

`ninfer_ngram_thinking_real` uses `NINFER_NGRAM_TEST_WEIGHTS` and accepts a backend,
neural draft size and ngram draft size, for example `mtp 5 15`. It checks output and
thinking budgets, forced control-token transitions, exact source-prefix answers,
positive ngram acceptance, and token-identical consecutive cached repeats.
It skips when the artifact environment variable is absent.

The optional fourth argument `--strict-fresh` additionally requires bit-identical
fresh-versus-cached generation and returns 2 when that diagnostic differs. Quantized
prefill/cache paths can already differ on upstream without ngram, including in
reasoning text; this is distinct from cached-repeat determinism and budget correctness.
Keep that diagnostic separate when comparing upstream and patched engines.

`ninfer_ngram_stop_chat_real` uses the same artifact environment variable and accepts
a backend and ngram draft size, for example `dflash2 15`. Neural draft size is five.
It uses canonical assistant-prefilled copy requests, stops inside speculative rounds,
and compares repeated retained continuations with equivalent output-limited ones.
It requires exact source-prefix text, complete prefix reuse and valid output-budget
and finish-reason accounting. An optional `--strict-fresh` argument retains
the separate fresh-prefill identity diagnostic; natural stop tokens are allowed to
end an answer before its maximum output budget. `--no-cuda-graph` selects eager
execution. This test also skips without an artifact.
