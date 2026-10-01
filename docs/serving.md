# HTTP serving

`build/apps/ninfer-serve` loads one v3 `.ninfer` artifact and exposes OpenAI- and
Anthropic-compatible HTTP endpoints over one resident NInfer Engine.

## Start the server

See [CUDA synchronization](cli.md#cuda-synchronization) for the shared `NINFER_CUDA_SYNC` setting.

```bash
./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --host 127.0.0.1 \
  --port 8080 \
  --max-context 240000 \
  --kv-capacity 240000 \
  --max-concurrency 2 \
  --kv-dtype fp8 \
  --device-state-slots 2 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft \
  --preserve-thinking
```

The command uses Qwen3.8-27B NVFP4. Each request has a 240,000-token logical ceiling. A shared
240,000-token Main Text KV pool serves admitted requests; either request may use the full capacity
when running alone. Requests acquire KV pages as execution advances; if concurrent growth exhausts
the pool, the scheduler can pause a request and restore it later.

With `C=2` and two extra Device slots, the process owns four Device StateImages. The default shared
pinned Host budget is 8 GiB plus eight model StateImages. It holds retained state, KV and pause
snapshots, including in-flight destinations; `--host-context-mib` sets an explicit total instead.

Other artifacts use the same command shape with their own path. For 35B-A3B DFlash, replace the MTP
selection with `--spec dflash --draft-tokens 7 --lm-head-draft`. Qwen3.8-27B
artifacts with DFlash2 companion weights also support `--spec dflash2 --draft-tokens 7`, with
`--lm-head-draft` optional. DFlash2 accepts draft counts 1..15 and supports the same sampling,
concurrency, prefix reuse, and image/video request surfaces. It may remain combined with
`--vision`.

When `--model-id` is omitted, the server advertises and accepts the artifact's `metadata.name`,
falling back to its architecture name when no name is stored. An explicit `--model-id` is a public
HTTP alias override and does not select or alter model execution.

Vision is disabled by default: its weights and Vision-specific unified-workspace extent are not
allocated, and media requests and token-count requests fail with HTTP 400 `vision_disabled`. Add
`--vision` when the server must accept image or video input. Speculative residency is likewise
frozen by `--spec mtp|dflash|dflash2` and `--draft-tokens`; omitting `--spec` loads no speculative backend.
`--lm-head-draft` additionally loads the optimized proposal head. DFlash on 35B-A3B and DFlash2 on Qwen3.8-27B can be combined
with `--vision`; each accelerates generated-text decode after multimodal prefill, while Vision encode
and prefill remain outside speculative acceleration. A later request cannot enable a capability
omitted at startup. The artifact need only contain the Text backbone and the optional components
selected for this process.

## Endpoints

| Method and path | Behavior |
|---|---|
| `GET /health` | Engine readiness |
| `GET /metrics` | Prometheus counters, gauges and latency histograms |
| `GET /v1/models` | configured OpenAI model alias, effective `max_model_len`, and a llama.cpp-compatible `meta` object |
| `GET /v1/models/{id}` | lookup of the configured alias, `max_model_len`, and `meta` object |
| `POST /v1/chat/completions` | OpenAI-style chat generation |
| `POST /v1/responses` | OpenAI Responses Core generation, state, typed Items, and SSE |
| `POST /v1/responses/input_tokens` | Responses prompt-token count without generation |
| `GET /v1/responses/{id}` | retrieve a locally stored terminal Response |
| `DELETE /v1/responses/{id}` | delete a locally stored Response |
| `GET /v1/responses/{id}/input_items` | list that Response's normalized input Items |
| `POST /v1/messages` | Anthropic-style message generation |
| `POST /v1/messages/count_tokens` | checkpoint-native expanded input-token count |

Anthropic SDKs append `/v1/messages` to their base URL, so their base URL is
`http://127.0.0.1:8080`, while OpenAI SDKs take `http://127.0.0.1:8080/v1`. A client given the
OpenAI-style base URL requests `/v1/v1/...`; every path above except `/health` also answers under
that doubled prefix, with the same behavior, errors, and request IDs.

`GET /health` returns HTTP 200 with `{"status":"ok"}` while the Engine can accept work. After an
Engine-wide failure it returns HTTP 503 with `{"status":"unavailable"}`. Temporary queue
saturation does not make the Engine unavailable. The endpoint remains unauthenticated.

How failures end:

- An out-of-memory condition or a failed internal check during one unit of work fails the requests
  that unit involved, logs `worker out of memory: ... - recovering` or `worker recovering from a
  failed request`, and the Engine carries on with its queue.
- A failure after eight such recoveries in a row, with no successful unit between them, fails every
  queued and running request and stops the Engine: `/health` then answers 503 and every request
  fails until the server is restarted. Any other exception on the worker (`worker crash: ...`)
  ends the Engine the same way.
- A CUDA error other than out of memory (an illegal address, a lost device) cannot be recovered
  in-process: the server prints `CUDA_CHECK(...) failed` with the error and exits, without final
  request-log records for the requests in flight.

Every OpenAI-compatible response carries a unique `x-request-id` header, including streaming and
error responses. Anthropic endpoints use their separate `request-id` contract.

All three generation SSE endpoints emit the standard `: keep-alive` comment after five seconds
without a protocol event. The comment is transport-only: SSE clients ignore it, and it does not
change generated text, event ordering, usage, stored Responses, or request logs. Accepted
connections also use TCP keepalive and a 15-second retransmission limit (`TCP_USER_TIMEOUT` on
Linux, `TCP_MAXRTMS` on Windows); together with the heartbeat,
a dead or unacknowledging peer is normally cancelled within about 20 seconds, including while the
request is waiting or prefilling. A peer whose TCP stack remains connected and acknowledges data
cannot be distinguished from a reading application; proxies must close their upstream NInfer
connection when the downstream client disappears.

## Models

`GET /v1/models` and `GET /v1/models/{id}` return the configured public OpenAI model alias
(defaults to the loaded model name, overridable with `--model-id`) together with the effective
`max_model_len` and a `meta` object in the shape exposed by `llama.cpp`. The `meta` facts describe
the model behind the alias and the running instance:

| Field | Meaning |
|---|---|
| `n_vocab` | tokenizer token domain |
| `n_ctx` | this instance's configured per-request context ceiling (equal to `max_model_len`) |
| `n_ctx_train` | model native/training context (informational; not the instance ceiling) |
| `n_embd` | model embedding width |
| `n_params` | total logical weight elements over the loaded artifact's distinct weight tensors |
| `size` | encoded weight payload bytes of the loaded artifact |
| `ftype` | dominant loaded weights format (the NInfer quantization name, by payload bytes) |

`n_ctx` is a property of the running `ninfer-serve` instance, so it reflects the `--max-context`
that this server was started with, not the model's native context. `n_ctx_train` reports the model's
own trained context separately.

`GET /v1/models/{id}` returns the same object for the single configured alias and a `404` for any
other id.

## OpenAI Chat Completions

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [
      {"role": "system", "content": "Answer concisely."},
      {"role": "user", "content": "What is speculative decoding?"}
    ],
    "max_tokens": 128
  }'
```

The endpoint supports:

- `system`, `developer`, `user`, `assistant`, and `tool` history, plus legacy `function` history;
- string content and ordered text/refusal parts; adjacent parts are preserved without inserted
  separators, and empty wire content remains an empty turn;
- User `image_url` parts, tool-result `image_url` parts used by compatible clients, and the User
  `video_url` extension using HTTP(S) or data URIs; image detail is omitted or `auto`;
- nonnegative `max_completion_tokens` and the legacy `max_tokens` spelling; zero performs prompt
  processing without generation;
- `temperature`, `top_p`, presence/frequency penalties, and signed integer `seed`;
- the compatible `top_k` (`0..20`) and `min_p` (`0..1`) sampler extensions;
- up to four non-empty stop strings, applied to both reasoning and answer output;
- `n:1`, text-only `modalities`, and `response_format` (`{"type":"text"}`, `{"type":"json_object"}`, or `{"type":"json_schema"}`; the type is accepted so clients that always send one are not refused, but NInfer does not constrain generation to it);
- non-streaming responses and server-sent event streams;
- `stream_options.include_usage`, optionally shaped by `--usage-chunk-choice` for strict client
  parsers;
- llama.cpp-compatible terminal `timings`, plus opt-in `timings_per_token` and
  streaming `return_progress` observations;
- function tools and free-form `custom` tools, the latter served to the model as a
  single-string-input function under the caller's own tool name so callers that dispatch by name
  keep working;
- `tool_choice` `auto`/`none`, and `required`, named-function, `custom`, or function-only
  `allowed_tools` selections, which are accepted and treated as advisory narrowing because the
  Engine cannot force a call;
- `strict:true` and `parallel_tool_calls:false` as advisory flags: the Engine does not enforce JSON
  Schema through constrained decoding and cannot limit the model to one call;
- assistant tool-call history, tool-result messages, and legacy function-call history;
- the top-level `reasoning_effort` field, where the `default` and `auto` aliases resolve to the
  server-configured level;
- `enable_thinking` and `preserve_thinking`, either at top level or in
  `chat_template_kwargs`;
- Assistant `reasoning_content` and `reasoning` history aliases.

Options whose observable behavior the Engine cannot provide are rejected when they request that
behavior. This includes nonzero `logit_bias`, requested log probabilities, audio/file input or
audio output, explicit low/high image detail, web search, moderation, low/high verbosity, stored
Chat Completions, and non-empty legacy `functions`.
Each capability rejection identifies the affected field and the guarantee NInfer cannot provide.
Known constrained-decoding aliases (`grammar`, `structured_outputs`, `guided_json`, `guided_regex`,
`guided_choice`, and `guided_grammar`) receive the same explicit rejection instead of being treated
as unknown hints.

Semantically neutral fields do not make an otherwise executable request fail. All-zero
`logit_bias`, `logprobs:false`, `top_logprobs:0`, `verbosity:"medium"`, empty legacy tool controls,
text-only `audio` configuration, and `prediction` are accepted without changing Engine execution.
Metadata, user/safety identifiers, service-tier and prompt-cache hints are likewise advisory.
Unknown top-level fields are ignored.

A string `name` on a `tool` message is accepted as an ignored, output-neutral compatibility
extension for clients that mirror the function name onto tool results. It does not participate in
tool identity, prompt rendering, or output. Non-string values are malformed; non-empty names on
other message roles remain unsupported because they carry participant identity that the loaded chat
template cannot represent.

For commonly generated OpenAI-compatible payloads, `repetition_penalty` is accepted only at its
neutral value `1`, and `mm_processor_kwargs` when empty or containing only null values. String-form
image/video URLs are also accepted.

Malformed protocol values return field-specific HTTP 400 errors. Invalid media sources, bytes, or
decoded content use `invalid_media`; remote fetch and timeout failures retain their dedicated
server-error codes. Failures in the normalized prompt contract use `invalid_prompt`; typed capacity
and availability failures retain their dedicated codes. Internal invariant failures are not
relabeled as client input errors.

The request `model` must equal the public model ID: the artifact `metadata.name` by default
(falling back to its architecture name when absent), or the explicit `--model-id` override.
Reasoning is returned separately as `reasoning_content`; answer text remains in `content`.

Across Chat Completions, Responses, and Anthropic Messages, a direct top-level tool-parameter
`type`, or an `anyOf`/`oneOf` composed entirely of explicit primitive types, guides conversion of
Qwen's untyped parameter text. It does not decide whether structurally complete markup is a tool
call. String-admitting values remain strings, including the empty string. An empty block for a
declared non-string parameter is omitted. Admitted JSON values retain their JSON type;
case-insensitive boolean text is normalized to `true` or `false`. A nonempty schema mismatch remains
a structured call: valid JSON retains its represented type and other text becomes a JSON string so
the tool consumer can report the validation error and continue the agent loop. Schemas without a
supported explicit type retain untyped inference. NInfer does not apply defaults, enforce required
properties, perform recursive JSON Schema validation, or use constrained decoding.

String parameters preserve function/tool-call markers and balanced nested
`<parameter=...>...</parameter>` text as value bytes. The Qwen wire format has no delimiter escape,
so a standalone `</parameter>` ends a value only when whitespace and then another parameter, the
function's closer, or the end of the output follow it; any other one is value text, such as a shell
command that echoes the markup. An unmatched nested parameter opener, or a quoted closer that is
followed by the next token, cannot be represented unambiguously and makes that tool-call region
ordinary content. Later content is still examined:
the first tool-call region (any accepted marker form) that parses becomes the structured turn, and any
quoted markup before it stays ordinary content. Generated reasoning closes only at a `</think>`
followed by a line break or the end of the turn, so a marker the model quotes while reasoning (followed
by a space, punctuation or an escaped `
`) stays in the reasoning channel.

By default the parser keeps that all-or-nothing behaviour. With `--tolerant-tool-calls` the server
recovers a call instead when the model adds a suffix after a complete call, a second call is
malformed, a single final call is cut by the output budget before its closing tags, or the closing
bracket after the function name is missing: the recovered call is reported structurally with a
`truncated_tail` diagnostic (logged at Info severity) rather than demoted to text, and an
undeclared tool name stays structured for the consumer to judge.

Messages enter the selected template in their input order. The maintained Qwen templates keep
system/developer messages at their original positions. A final assistant message is an assistant
prefill: generation continues that turn in place instead of opening a new assistant turn. Because
a reasoning opener cannot be continued, a prefill that carries media or whose thinking is not
explicitly disabled (top-level or `chat_template_kwargs` `enable_thinking`, or
`reasoning_effort: "none"`) is refused with `invalid_prompt`.

Prompt-bearing JSON objects retain their received member order through request parsing and prompt
rendering, including tool schemas and historical tool inputs. Canonical model-origin tool arguments
retain that member order in aggregate and streaming responses, so an unmodified replay reconstructs
the same ordered tool call. NInfer does not canonicalize semantically equivalent JSON: if a client
reorders members, inserts defaults, or otherwise rewrites a tool object, the changed rendered input
does not match the model-held endpoint and can reuse only an earlier exact checkpoint.

`--chat-template FILE` selects a local Jinja template; by default, the server uses the template
stored in the artifact. See the [CLI guide](cli.md#text-input) for an example.

Control-token spellings quoted in message content, tool data or ordinary template kwargs are
encoded as text. Media placeholders come from the template and bind to actual image/video inputs.

`chat_template_kwargs` passes a JSON object to the template in Chat Completions, Responses and
Anthropic Messages. Values duplicated in typed request fields must agree. Null standard options
mean unspecified; other null values remain `none`. Messages, tools, generation mode and tokenizer
special tokens cannot be overridden through kwargs.

`--default-thinking-budget N` sets a positive default thinking-token cap for requests that start
in thinking mode. Non-thinking requests receive no cap. It may coexist with `--no-thinking`
because requests can explicitly enable thinking. Anthropic
`thinking:{"type":"enabled","budget_tokens":N}` overrides this default for that request.

Add `--default-thinking-budget 512` to the startup command to cap model-origin thinking at 512
tokens for every thinking-enabled request.

At the cap boundary, Engine first honors a natural `</think>`, stop condition, cancellation, or
total output/context limit. If thinking remains open, it commits Qwen's canonical early-close
guidance and close marker to the same model sequence without sampling, streams the guidance as a
reasoning delta, and continues normal content or tool-call generation. `--thinking-budget-message S`
replaces that guidance with `S`, appending the close marker when `S` lacks it. Inserted tokens count in
completion usage and the request's `max_tokens`/`max_output_tokens` budget. If the effective output
capacity extends past the cap but cannot fit the complete tokenizer-derived control suffix plus one
post-close model token, preparation is rejected with HTTP 400 code
`thinking_budget_capacity_insufficient` rather than partially inserting control. The server does
not promise that the model will emit nonempty content or a tool call after the marker.

For Chat Completions, `reasoning_effort: "none"` requests disabled thinking. The selected template
interprets the other standard values (`minimal`, `low`, `medium`, `high`, `xhigh`, `max`).
Conflicting explicit `enable_thinking` and effort values return `conflicting_template_option`.

Templates accept different subsets of those values: the official Qwen3.8 template raises for
anything but `low`, `medium` and `xhigh`. NInfer renders each value once when the template loads,
and a request for a value the template rejects renders with the nearest accepted value, a tie
rounding up: on Qwen3.8, `high` and `max` render as `xhigh` and `minimal` as `low`. This applies to
every endpoint and to `--reasoning-effort`. The values are tried on a one-message chat; a template
that rejects every value, or cannot render that chat at all (one that requires a system message, for
example), keeps the request's value and its error. Request logs record the effort the client asked
for.

`preserve_thinking` controls reasoning retention according to the selected template. Request
options override server defaults set with `--no-thinking` and `--preserve-thinking`. Unspecified
thinking, effort and preservation options use the template's defaults.

Streaming begins with an assistant-role chunk, sends separate reasoning and content deltas, then a
finish-reason chunk and `[DONE]`. When `stream_options.include_usage` is true, a final empty
`choices` chunk contains completed usage. Aggregate and streamed usage include cached prompt tokens
and reasoning-token details; choices carry `logprobs: null` when log probabilities were not
requested, and aggregate assistant messages carry `refusal: null` because refusal output is not
supported.

A failure after the stream has started is sent as OpenAI does: one unnamed `data:` event whose
payload is the usual error body (`{"error": {...}}`, which OpenAI SDKs raise as an API error), then
the connection closes without a finish-reason chunk, usage chunk or `[DONE]`.

### llama.cpp-compatible request observations

Every successful Chat Completions response includes a top-level `timings` object. This is a
llama.cpp-compatible response extension, not an OpenAI field. In a stream it is attached to the
last JSON chunk before `[DONE]`: the empty `choices` usage chunk when
`stream_options.include_usage` is true, otherwise the finish-reason chunk.

```json
{
  "timings": {
    "cache_n": 4096,
    "prompt_n": 4096,
    "prompt_ms": 83.0,
    "prompt_per_token_ms": 0.020263671875,
    "prompt_per_second": 49349.39759036145,
    "predicted_n": 129,
    "predicted_ms": 1140.0,
    "predicted_per_token_ms": 8.90625,
    "predicted_per_second": 112.28070175438596
  }
}
```

`cache_n` is the exact Engine-proven reused prompt prefix and `prompt_n` is the remaining prompt
suffix, so `cache_n + prompt_n` equals `usage.prompt_tokens`. Prompt time starts when admission
commits that exact reuse choice and ends when the first output token is committed. Generation time
starts at that first token and ends at the last committed output token. Accordingly, generation
speed uses `max(predicted_n - 1, 0)` token intervals; the first token belongs to prompt latency and
is not counted again as a decode interval. Zero-token, one-token, zero-duration, and exact-cache-hit
cases report finite zero rates rather than `NaN` or infinity. Speculative requests additionally
include terminal `draft_n` and `draft_n_accepted` when draft work occurred.

Set top-level `timings_per_token: true` on a streaming request to attach the latest cumulative
timing snapshot to each visible reasoning or content chunk. This does not enable terminal timings,
which are always present. A model commit that is temporarily hidden by UTF-8, stop-string,
reasoning, or tool-call buffering still advances the cumulative token count; the next visible chunk
observes that committed frontier. The option increases response serialization and transport volume
and is off by default.

Set top-level `return_progress: true` together with `stream: true` to receive prompt-processing
chunks:

```json
{
  "prompt_progress": {
    "total": 8192,
    "cache": 4096,
    "processed": 6144,
    "time_ms": 41
  }
}
```

The initial event has `processed == cache`. Later cumulative events are published only after the
corresponding prefill unit commits, may be coalesced when the consumer is slower than prefill, and
never move backwards. The final event has `processed == total` and precedes the first output delta.
For an exact full-prefix hit, the initial event already has `cache == processed == total` and no
synthetic prompt work is reported. `time_ms` is elapsed wall time since committed admission;
clients may calculate actual suffix progress as `(processed-cache)/(total-cache)` when the
denominator is nonzero.

### Multimodal request

Start the server with `--vision` before sending media:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
        {"type": "text", "text": "Describe this image."}
      ]
    }],
    "max_tokens": 128
  }'
```

OpenAI image and video sources may be HTTP(S) URLs or base64 data URLs.

Text and media requests use one complete-prompt context contract. After chat-template rendering and
media-token expansion, the result must fit Engine `--max-context`. The current Vision runtime also
has a 32,768 merged-token envelope (131,072 raw patches); the effective Vision limit is therefore
`min(--max-context, 32768)`. There is no fixed image/video item-count limit: item count is admitted
through aggregate source-byte, decoded-pixel, raw-patch, Vision-token, and live-memory budgets.

Media cache misses run as independent decode → resize → BF16-pack tasks on a bounded host worker
pool. Prepared payloads are keyed by SHA-256 of the acquired bytes plus modality, so repeated media
in later requests reuses the exact immutable BF16 patch input; concurrent identical misses use one
single-flight build. `--media-cache-mib` bounds LRU-retained payloads, while
`--media-live-mib` bounds every cache-, request-, or runtime-referenced payload. Cache eviction does
not invalidate a request reference, and live bytes are returned only when the final reference is
released. A request-level preparation gate derived from the live limit prevents concurrent partial
builds from deadlocking the memory account.

An expanded prompt beyond `--max-context` returns HTTP 400 `context_length_exceeded`, including
the prepared token count and configured context ceiling. A media preprocessing resource rejection
returns HTTP 400 `media_budget_exceeded`. HTTP 413 `request_too_large` is reserved for a raw request
body that exceeds `--max-request-mib` before JSON parsing; it is not used for model-context or media
resource errors.

## OpenAI prompt caching

Chat Completions and Responses translate OpenAI cache hints into optional shared-prefix write
candidates:

- omitted `prompt_cache_options` creates a default implicit candidate at the end of the latest
  cacheable content part;
- `mode:"implicit"` requests the same automatic candidate explicitly;
- `mode:"explicit"` disables that implicit write for the request;
- `prompt_cache_breakpoint:{"mode":"explicit"}` on supported content creates an explicit
  candidate.

The automatic candidate precedes the message closing tokens, allowing reuse when the same message
body grows and its previous tokens remain an exact prefix. An explicit marker keeps its requested
location. Responses applies this policy after expanding stored history.

One request carries at most four writes. Explicit markers take precedence: four explicit candidates
leave no extra slot for an automatic candidate. If more explicit markers appear in the history, the
latest four remain write candidates. Exact reads of already-published prefixes do not require the
request to repeat a marker.

These fields are optimization hints. A legal boundary that cannot be represented as an exact
rendered-token frontier is ignored without changing prompt content. `prompt_cache_key` is not an
Engine session key or prefix identity. Valid TTL/retention values are accepted, but NInfer does not
promise their wall-clock residency; physical retention follows the resource scheduler.

## OpenAI Responses Core

NInfer implements the typed-Item and semantic-event core of the OpenAI
[Responses API](https://developers.openai.com/api/reference/resources/responses/overview). All
supported model instances use this same adapter and Engine route. It is intentionally not
advertised as full parity with OpenAI-hosted tools, durable cloud storage, background jobs,
Conversations, or compaction.

### Create a Response

```bash
curl http://127.0.0.1:8080/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "instructions": "Answer concisely.",
    "input": "What is speculative decoding?",
    "max_output_tokens": 128,
    "store": true
  }'
```

The same endpoint works with OpenAI SDKs by replacing their base URL:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local-secret")
response = client.responses.create(
    model="qwen3.8-27b",
    instructions="Answer concisely.",
    input="What is speculative decoding?",
    max_output_tokens=128,
)
print(response.output_text)  # SDK helper derived from response.output
```

`output_text` is an SDK convenience property. It is not emitted as a top-level wire field; the
wire response contains typed `output` Items.

### Create request fields

| Field | NInfer Responses Core contract |
|---|---|
| `model` | required non-empty string; must equal the artifact-derived public model ID or explicit `--model-id` override |
| `input` | string or typed Item array; it may be omitted or empty only when `previous_response_id` already supplies a user query |
| `instructions` | optional string, inserted before the reconstructed conversation for this request only |
| `previous_response_id` | optional ID of a retained local Response |
| `max_output_tokens` | non-negative integer; omission executes with `--default-max-tokens` but remains `null` in the Response object |
| `stream` | boolean; `true` selects Responses SSE rather than a JSON body |
| `store` | boolean, default `true`; controls local retrieval and continuation state |
| `temperature` | finite number in `[0,2]` |
| `top_p` | finite number in `[0,1]` |
| `metadata` | at most 16 string pairs; keys at most 64 characters and values at most 512 |
| `client_metadata` | Codex client extension; an object or `null`, accepted as opaque tracing metadata with no generation effect |
| `reasoning.effort` | `none` requests disabled thinking; other standard effort values pass to the selected template |
| `reasoning.summary` | omitted, `null`, or any string; every string requests the same fixed protocol placeholder without changing model execution, and the original value is echoed in the response |
| `chat_template_kwargs` | template parameters as a JSON object; standard options merge with typed fields |
| `preserve_thinking` | alias for `chat_template_kwargs.preserve_thinking`; conflicting values are rejected |
| `text.format` | omitted or `{"type":"text"}` only |
| `tools` | direct function definitions or namespace groups containing function definitions; see below |
| `tool_choice` | `auto`, `none`, or function-only `allowed_tools` with mode `auto`; a namespaced selection carries both `namespace` and `name` |
| `parallel_tool_calls` | `true` by default; `false` is accepted only when no effective tool is callable |
| `max_tool_calls` | non-negative integer accepted as a hosted-tool no-op; NInfer does not execute hosted tools |
| `truncation` | omitted or `disabled`; overlong input fails instead of silently dropping Items |
| `top_logprobs` | omitted or `0` |
| `service_tier` | omitted, `auto`, or `default`; the response reports `default` |
| `background` | omitted or `false` |
| `include` | omitted, empty, or `["reasoning.encrypted_content"]`; the supported value requests the local raw-reasoning mirror described below |
| `stream_options.include_obfuscation` | optional boolean; accepted as a transport hint, but this local server emits no padding |
| cache and client hints | `prompt_cache_key`, `prompt_cache_options`, `prompt_cache_retention`, and explicit breakpoints follow [OpenAI prompt caching](#openai-prompt-caching); `safety_identifier` and `user` are accepted as client hints |

Unknown top-level fields fail with `unknown_parameter`. Recognized but unsupported features fail
with a field-specific 400 error instead of being silently ignored. When a rejection is caused by a
specific value, the error names that value (ASCII-escaped and truncated), its size where relevant,
and its location in the request, both in the error `param` (for example `tools[0].function.name`
or `messages[2].tool_calls[0].function.name`) and in the message text.

### Input Item contract

String `input` is normalized to one user `message` with an `input_text` part. Array input accepts:

| Item | Supported form |
|---|---|
| `message` | roles `user`, `assistant`, `system`, and `developer`; string content or typed content array |
| `input_text` | message content part containing string `text` |
| `output_text` | assistant-message replay part containing string `text` |
| `refusal` | assistant-message replay part; its text enters assistant history |
| `input_image` | user- or assistant-message part with HTTP(S) or data-URI `image_url`; detail omitted or `auto`; requires server `--vision` |
| `input_video` | NInfer extension with HTTP(S) or data-URI `video_url`; requires server `--vision` |
| `reasoning` | raw replay Item with `reasoning_text` content; summary/encrypted metadata may accompany raw text but cannot replace it |
| `function_call` | completed assistant call with optional `id` and namespace, plus required `call_id`, `name`, and JSON-object string `arguments` |
| `function_call_output` | completed result with required `call_id` and optional matching name/namespace assertion; `output` may be a string or a non-empty array of `input_text`/`input_image` parts |

Contiguous assistant-owned Items form one assistant history turn in the representable order
`reasoning` -> assistant message content -> `function_call`. Multiple message Items append their
content parts, multiple calls retain declaration order, and a reasoning-only turn is retained. A
user, system, developer, or `function_call_output` Item ends the group; an order that would require
rearranging assistant content fails with `invalid_assistant_history`. Results are validated by
`call_id` and reordered to call declaration order before prompt rendering; unknown, duplicate, or
unrepresentable partial result sets fail with `invalid_tool_history`. Canonical input Items retain
client order. Input Item IDs are preserved when supplied and generated otherwise; duplicate IDs
fail.

System and developer message Items retain their positions in the input array. Top-level
`instructions` is represented as a leading developer turn for the current request; target-specific
role lowering occurs only in the Qwen family frontend.

An `input_text`, `input_image`, or tool-result part may carry
`prompt_cache_breakpoint:{"mode":"explicit"}`. Write selection follows
[OpenAI prompt caching](#openai-prompt-caching); boundaries affect reuse opportunities, not prompt
identity or output semantics. String message status/phase metadata is accepted but has no Qwen
prompt representation.

`input_file`, `input_audio`, image `file_id`, non-`auto` image detail, reasoning metadata without raw
reasoning text, partial tool Items, and other Item/content types are not supported. HTTP media URLs
stored in a response chain are fetched again when that chain is continued; use data URIs when the
historical media bytes must be immutable.

### Function tools

Responses function definitions may be declared directly rather than inside Chat Completions'
nested `function` object:

```json
{
  "type": "function",
  "name": "get_weather",
  "description": "Get current weather",
  "parameters": {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"]
  },
  "strict": false
}
```

They may also be grouped in a Responses namespace:

```json
{
  "type": "namespace",
  "name": "mcp__weather",
  "description": "Weather service",
  "tools": [{"type": "function", "name": "get_current"}]
}
```

NInfer gives each namespace/function pair a distinct internal Engine identity and restores the
separate `namespace` and `name` fields in aggregate output, SSE events, and replayed Items. The same
function name may therefore appear in different namespaces. Namespace members remain ordinary
client-executed functions; this does not add a remote MCP executor.

NInfer renders these definitions in the Qwen prompt and parses model output into separate
`function_call` output Items. Each output has a protocol Item `id` (`fc_...`) and a distinct
`call_id` (`call_...`). The client executes the function and sends a `function_call_output` Item in
a later request. Only functions in the current effective tool set can become structured calls;
undeclared model output remains ordinary text. `allowed_tools` with mode `auto` filters that set
without changing declaration order, while `tool_choice:"none"` disables structured tool output even
when the history contains earlier calls.

NInfer does not execute functions or enforce JSON Schema through constrained decoding, so
`strict:true`, required or named tool choice, hosted tools, remote MCP tools, and custom free-form
tools are rejected. Deferred loading, output schemas, and caller restrictions that exclude direct
invocation are also rejected because their semantics cannot be honored.

### Response object and usage

A terminal wire response has `object: "response"`, one of `completed`, `incomplete`, or
`cancelled` in `status`, and a typed `output` array. NInfer may emit:

- a `reasoning` item containing raw `reasoning_text`; it returns a placeholder summary if
  `reasoning.summary` is requested;
- an assistant `message` containing an `output_text` part;
- one or more `function_call` Items.

When `include:["reasoning.encrypted_content"]` is requested, reasoning Items also
carries en `encrypted_content` equal to its raw `reasoning_text`. This field is **not**
**encrypted** and provides no confidentiality.

Ordinary model/string stops produce `completed`. Output-token or context-capacity exhaustion
produces `incomplete` with `incomplete_details.reason: "max_output_tokens"`. Errors accepted after
an SSE response has started produce `response.failed`; validation and preparation errors remain
normal HTTP error responses. `completed_at` is populated only for completed Responses. A
reasoning-only incomplete result contains no invented empty assistant message.

Usage is checkpoint-native:

```json
{
  "input_tokens": 42,
  "input_tokens_details": {"cached_tokens": 17},
  "output_tokens": 12,
  "output_tokens_details": {"reasoning_tokens": 5},
  "total_tokens": 54
}
```

`input_tokens` includes the chat template and expanded media tokens. `cached_tokens` is the exact
checkpoint-proven prompt prefix reused by Engine. `output_tokens` is the count of accepted generated token
IDs, including a withheld stop token when applicable. `reasoning_tokens` is counted in the Qwen
output decoder while accepted tokens are still in the reasoning channel; it is not estimated by
re-tokenizing decoded text.

### Responses streaming

Set `stream:true` for semantic Server-Sent Events. Every frame uses both the SSE event name and a
matching JSON `type`, and every JSON event has a monotonically increasing `sequence_number`:

```text
event: response.output_text.delta
data: {"type":"response.output_text.delta","sequence_number":7,...}

```

The normal lifecycle is:

1. `response.created`, then `response.in_progress`;
2. `response.output_item.added` and `response.content_part.added`;
3. zero or more `response.reasoning_text.delta` or `response.output_text.delta` events;
4. matching `*.done`, `response.content_part.done`, and `response.output_item.done` events;
5. exactly one `response.completed`, `response.incomplete`, or `response.failed` terminal event.

For a reasoning Item requested with any string-valued `reasoning.summary`, its
`response.output_item.added` and `.done` payloads carry the same placeholder summary. Immediately
after the Item is added, the stream emits `response.reasoning_summary_part.added`,
`response.reasoning_summary_text.delta`, `response.reasoning_summary_text.done`, and
`response.reasoning_summary_part.done` with `summary_index:0`, then continues with the raw
`reasoning_text` content lifecycle. Omitted or `null` summary requests emit none of these summary
events and retain an empty Item `summary` array.

For `include:["reasoning.encrypted_content"]`, the in-progress
`response.output_item.added` Item omits `encrypted_content` because the complete reasoning text is
not available yet. `response.output_item.done` and the terminal Response output contain the same
complete raw mirror. A response with no reasoning Item emits no encrypted placeholder.

Function arguments use `response.function_call_arguments.delta` and `.done`. IDs, output indices,
and content indices remain stable, and concatenated deltas equal the terminal Item. Responses SSE
does not emit the Chat Completions `[DONE]` sentinel. With tools enabled, ordinary answer text still
streams immediately; only an ambiguous `<tool_call>` suffix or the structured tool region is held.
Malformed tool markup is flushed back as ordinary text without losing bytes.

### Local response state and resources

`store` defaults to `true`. Stored Responses live only in this server process and are bounded by an
LRU store. They are lost on restart and are not OpenAI's durable cloud retention service.

`previous_response_id` reconstructs the complete stored input/output Item history before the new
input. The current `instructions` value is placed first but is not saved into the continuation
context, matching the Responses rule that previous top-level instructions do not carry forward.
Function definitions are request configuration rather than conversation Items and must be sent
again on tool-result turns. The reconstructed prompt follows the ordinary Engine path, so compatible
checkpoint reuse applies naturally.

A stored Response also retains its resolved `preserve_thinking` value. A child which omits the
field inherits the parent value. An explicit different value creates a new semantic branch; prompt
rendering and identity still determine reuse. Changing the boolean alone never invalidates an exact
checkpoint already proved compatible by the model runtime.

For Engine-local reuse, a stored root Response receives one bounded session key derived from its
response ID, and every `previous_response_id` child inherits that key. `store:false` roots remain
anonymous; a `store:false` child may read its inherited session checkpoint but does not replace the
stored chain's latest endpoint. Response-store eviction or deletion removes the HTTP object, not an
independently retained Engine checkpoint; the latter remains bounded by the Engine's own retention
and pressure policy. No session key or cache marker is added to the HTTP schema.

Resource behavior:

| Endpoint | Contract |
|---|---|
| `GET /v1/responses/{id}` | returns the stored terminal object, or 404 `response_not_found`; stream recovery and non-empty `include` are rejected rather than ignored |
| `DELETE /v1/responses/{id}` | removes public retrieval and returns `response.deleted`; descendant contexts already retained by other Responses remain usable |
| `GET /v1/responses/{id}/input_items` | returns normalized Items supplied to that request; supports `after`, `limit` `1..100` (default `20`), and `order` `asc|desc` (default `desc`); image URLs are redacted unless `include=message.input_image.image_url` |
| `POST /v1/responses/{id}/cancel` | explicitly fails because background execution is unsupported |
| `POST /v1/responses/compact` | explicitly fails with `compaction_not_supported` |

`store:false` Responses cannot be retrieved or used as `previous_response_id`. LRU eviction and
explicit deletion also make an ID unavailable. A single Response larger than the configured store
capacity fails with `response_store_capacity_exceeded` rather than silently pretending it was
stored. Generation has finished by then, but its output is not returned: an aggregate request gets
only the error and a streamed one ends with `response.failed`. Repeat it with `store:false` or a
larger `--response-store-max-mib`.

### Responses input token count

`POST /v1/responses/input_tokens` uses the same prompt path as Create and does not run generation.
It accepts `model`, `input`, `instructions`, `previous_response_id`, reasoning, function tools and
tool choice, supported text/truncation values, and the `preserve_thinking` extension. Parent lookup,
call-ID normalization, template rendering, and media expansion are therefore identical to the
corresponding Create request:

```bash
curl http://127.0.0.1:8080/v1/responses/input_tokens \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","input":"Count this prompt."}'
```

```json
{"object":"response.input_tokens","input_tokens":11}
```

Unsupported Create fields include Conversations, prompt templates, context management, hosted
moderation, Structured Outputs/JSON mode, `include` values other than
`reasoning.encrypted_content`, background execution, compaction, files/audio, and
OpenAI-hosted/MCP/custom tools. Except for the two explicitly documented placeholders,
these are compatibility boundaries rather than silently accepted approximations.

## Anthropic Messages

```bash
curl http://127.0.0.1:8080/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "max_tokens": 128,
    "messages": [
      {"role": "user", "content": "Explain prefix reuse in one sentence."}
    ]
  }'
```

The endpoint accepts top-level System text, ordered User/Assistant/System history, text and image
blocks, Thinking history, tool-use history, tool results, user-defined tools, aggregate responses,
and Anthropic SSE. Consecutive User or Assistant messages are joined without adding separators.
Mid-conversation System messages retain their input position. A final Assistant message
is an Assistant prefill: generation continues its existing text instead of opening another turn.
Assistant prefill cannot contain media and cannot start with Thinking enabled; a turn cut by the
output limit may carry reasoning content. It cannot carry `tool_use` blocks: every `tool_use`
needs its `tool_result` in the next User message, so a trailing one is rejected with
`invalid_tool_history`, as the Anthropic API rejects it.

Claude Code may place its attribution metadata in the first block of a top-level System array. If
that block is a text block beginning exactly with `x-anthropic-billing-header:`, NInfer consumes the
whole block before token counting, prompt preparation, and cache identity construction. The rule is
positional: a string-form System value, a later array block, or an inline System message with the
same text remains ordinary prompt content. A `cache_control` marker attached to the consumed block
is consumed with it rather than moved to adjacent content.

`max_tokens` is optional for local clients and otherwise uses `--default-max-tokens`; a positive
value is the complete output budget. `max_tokens:0` is rejected because NInfer does not expose a
completed zero-output cache-prewarm lifecycle. `temperature`, `top_p`, `top_k`, and
`stop_sequences` enter Engine execution. A matched custom stop is returned as
`stop_reason:"stop_sequence"` together with the actual `stop_sequence`; context exhaustion returns
`model_context_window_exceeded`.

Thinking supports `disabled`, `adaptive`, and `enabled`. Enabled Thinking requires
`budget_tokens >= 1024`, and that budget is passed to Engine. Unlike the Anthropic API, a budget at
or above `max_tokens` is accepted: the output limit ends thinking before the budget can, so the
budget takes no effect. Clients such as Qwen Code send a fixed budget while shrinking `max_tokens`
to the context left. Visible
Thinking is returned with an opaque compatibility signature; SSE emits its `signature_delta`
before closing the block. Request lowering reconstructs the local prompt from the visible
`thinking` text and treats `signature` as non-semantic transport metadata, so retained history
remains usable across serve restarts.
`display:"omitted"` is rejected because NInfer cannot provide Anthropic's
encrypted hidden-reasoning restore semantics. `preserve_thinking` remains a NInfer extension for
closed-turn reasoning history. `output_config.effort` passes its protocol-validated value to the
selected template, substituting the nearest value the template accepts as described above.

User-defined tools support `name`, `description`, object `input_schema`, and `input_examples`.
`tool_choice:auto` and `none` are executable. As on the OpenAI endpoints, `tool_choice` `any` and
named `tool`, `strict:true`, and `disable_parallel_tool_use:true` are accepted as advisory: the
Engine cannot force a call, constrain arguments to the schema, or limit the model to one call, so
the tools stay offered under automatic selection. A named choice must name a declared tool, and
`any` requires tools. Qwen Code sends `any` for its JSON side queries (permission classifier,
session title, next-speaker check). Deferred tools, tools that exclude direct model calls,
Anthropic-provided/server tools, toolsets, MCP, and containers are rejected because their executor
is absent. `tool_result` preserves text/image order and marks
`is_error:true` explicitly in the model prompt. For a visible Assistant tool-use turn, the next
User turn must provide exactly one leading result for every declared ID; valid results are matched
by ID and normalized to call order. A history that begins with results remains valid as a truncated
or imported conversation.

Block-level ephemeral `cache_control` on tools and supported System/User/Assistant/tool-history
blocks creates explicit shared-prefix candidates. At most four distinct block-level breakpoints are
accepted. Request-level `cache_control` targets the last cacheable block: it merges with an explicit
breakpoint at the same target and TTL, conflicts at the same target with a different TTL, and needs
an available fifth slot when four different explicit targets already exist. TTL must be `5m` or
`1h`; it is a protocol hint, not a wall-clock residency guarantee.

NInfer maps representable boundaries to exact prompt frontiers and ignores a legal but
unrepresentable advisory boundary without changing the prompt. Reuse still requires exact rendered
identity and can read an existing owner without another `cache_control`. Aggregate usage reports
verified reused tokens in `cache_read_input_tokens` and leaves cache creation unknown. Streaming
emits `message_start` after Engine admission commits the prefix selection and before
transfer/prefill output, so its uncached/cache-read split is already exact; terminal cumulative
usage matches the aggregate response.

Documents, Search Results, Files, Structured Outputs, server-tool results, container uploads, and
other execution-dependent blocks are rejected with the missing capability identified. Metadata,
service tier, inference geography, protocol-version/beta headers, cache TTL, and unknown advisory
fields do not block an otherwise executable request. The request `model` is any non-empty local
proxy label and is echoed in the response; it does not select the resident artifact.

Every Messages response carries a `request-id` header; error bodies also carry `request_id` and use
Anthropic error categories. Local admission overload maps to HTTP 529 and queue/media timeouts to
HTTP 504. Streaming owns the full Anthropic block lifecycle for Thinking, text, and tool use.

`POST /v1/messages/count_tokens` uses the artifact's tokenizer, chat template, and media expansion
without generation. It shares the same prompt normalization, tools, Thinking mode, Assistant
prefill, media processing, and cache-marker interpretation as Messages; output-only sampling and
streaming fields do not affect the count:

```bash
curl http://127.0.0.1:8080/v1/messages/count_tokens \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Count this prompt."}]
  }'
```

## Authentication and CORS

Pass `--api-key VALUE` to require the same value as an OpenAI bearer token or Anthropic
`x-api-key` header. `GET /health` and CORS preflight requests remain unauthenticated.

```bash
curl http://127.0.0.1:8080/v1/models \
  -H 'Authorization: Bearer local-secret'
```

`--cors` adds permissive browser CORS headers. It is disabled by default.

## Server options

The table lists executable defaults. The startup example selects a long-context FP8/MTP3 profile.

| Option | Meaning | Default |
|---|---|---:|
| `--host H` | listen address | `127.0.0.1` |
| `--port N` | listen port | `8080` |
| `--api-key KEY` | required bearer or `x-api-key` value | unset |
| `--model-id ID` | override the public OpenAI model alias | artifact `metadata.name`, or architecture name |
| `--chat-template FILE` | replace the artifact's chat template at startup; it must be one the target accepts | artifact template |
| `--rope-yarn-factor F` | startup-fixed runtime YaRN factor, finite `[1,4]`; extends allowed ceiling only | `1` |
| `--max-context N` | logical context ceiling of each sequence | `8192` |
| `--kv-capacity N\|auto` | explicit shared Main Text KV capacity, or maximize it from remaining GPU memory; omitted means `--max-context` | `8192` |
| `--vram-headroom-mib N` | VRAM in MiB that `--kv-capacity auto` leaves free after sizing the KV pool; requires `auto` | `1024` |
| `--max-concurrency N` | resident execution lanes; valid range `1..8` | `1` |
| `--max-pending-requests N` | additional requests allowed to wait for admission | `16` |
| `--pending-timeout-ms N` | maximum preparation-plus-admission wait | `30000` |
| `--prefill-chunk N` | text-prefill chunk | `1024` |
| `--use-original-int8-prefill-kernel` | prefill INT8-KV prompt attention with the original kernel at the requested `--prefill-chunk`. Without it INT8 KV uses the fast kernel (FP16 per-tile PV accumulation) and rounds `--prefill-chunk` down to whole prompt-attention waves (896 tokens for the 24-head model on RTX 5090: `4096` runs as `3584`); requires `--kv-dtype int8` (startup rejects it with any other KV format) | off |
| `--int8-prefill-8bit-pv` | run the fast INT8 prompt kernel's P×V on INT8 Tensor Cores: each row's probabilities, scaled by the V group scale, become 8-bit codes per 64-key tile against the exact stored V codes. Prompt attention takes 8-11 % less time from 16K context up; probabilities below half a code step of their tile's largest round to zero, which moved 64K-window perplexity by 1.1 % on the bundled corpus. Requires `--kv-dtype int8` and the fast kernel (startup rejects it with `--use-original-int8-prefill-kernel`) | off |
| `--use-original-nvfp4-prefill-kernel` | prefill NVFP4-KV prompt attention with the tiled kernel. Without it a chunk that sees more than 2048 keys uses the fast kernel, which runs QK on block-scaled FP4 Tensor Cores directly over the stored K codes (Q as two NVFP4 terms), decodes V in registers with FP16 per-tile PV accumulation, and splits the chunk's keys across CTAs when its row blocks alone would leave SMs idle (at most 64 MiB of workspace); shorter chunks keep the tiled kernel. Requires `--kv-dtype nvfp4` (startup rejects it with any other KV format) | off |
| `--log-stats-interval-ms N` | aggregate throughput report interval; `0` disables it | `5000` |
| `--log-stats-panel on\|off` | pin the session statistics panel beneath the console log on an interactive terminal | `on` |
| `--log-colours on\|off` | colour the console statistics lines; never applies to file logs | `off` |
| `--log-level trace\|debug\|info\|warning\|error\|critical\|off` | pretty stderr verbosity | `info` |
| `--device N` | CUDA device index | `0` |
| `--context-cost-presets FILE` | optional runtime context-cost preset registry | generic + compiled defaults |
| `--max-request-mib N` | body-size limit before JSON parsing | `384` |
| `--media-cache-mib N` | LRU-retained prepared BF16 media payloads; `0` disables retention | `1024` |
| `--media-live-mib N` | all live prepared BF16 media payloads | `2048` |
| `--media-preprocess-threads N` | bounded media preprocessing workers; `0` selects at most 16 from host concurrency | `0` |
| `--request-log-jsonl FILE` | append full-precision server/request records | disabled |
| `--response-store-max-records N` | maximum locally retained Responses objects | `1024` |
| `--response-store-max-mib N` | total local Response envelope/Item/context budget | `256` |
| `--kv-dtype bf16\|int8\|fp8\|nvfp4\|k8v4` | KV-cache storage | `bf16` |
| `--spec mtp\|dflash\|dflash2` | speculative backend | off |
| `--draft-tokens N` | MTP `1..5`; DFlash/DFlash2 `1..15` | unset |
| `--lm-head-draft` | optimized proposal head | off |
| `--draft-tree-nodes auto\|LIST` | DFlash2 tree verification, each row's tree built on the device from the drafter's candidate lattice; works with every `--kv-dtype`. `auto` (recommended) lets every all-neural round of up to four rows choose the chain or a tree of `draft tokens + 5` or `+ 9` columns from measured round time and acceptance at its batch size and context length; on an artifact whose GDN input projections cannot verify trees it keeps chain verification and logs a warning. A LIST fixes the column count by batch size: entry *c* (anchor included, `draft tokens + 2..32`) applies to rounds of *c* rows, the last entry repeats for larger batches and `0` keeps chain verification; see [tree verification](maintainer/tree-verification.md) | off |
| `--draft-tree-paths N` | most root-to-leaf paths per tree row (each one a parallel GDN replay), `2..8` | `8` |
| `--ngram-draft-tokens N` | verified n-gram copy proposals per round beside the `--spec` drafter, `1..63`; `0` disables; see [ngram copy proposals](ngram.md) | `0` |
| `--ngram-min-match N` | minimum matched tokens for an n-gram proposal, `4..64` | `12` |
| `--ngram-archive-mib N` | RAM archive that keeps n-gram sources across the requests of a conversation; requires `--ngram-draft-tokens`; `0` keeps drafting request-local | `0` |
| `--ngram-session-mib N` | one conversation's share of the n-gram archive; no effect without `--ngram-archive-mib` | `128` |
| `--ngram-native-sessions` | also recognize Kilo, Codex and Claude session metadata as conversation identities; requires `--ngram-archive-mib` | off |
| `--default-max-tokens N` | output limit when omitted by a request | `8192` |
| `--default-thinking-budget N` | positive thinking cap inherited by thinking-enabled requests | unset |
| `--thinking-budget-message S` | text committed when a thinking budget ends thinking, replacing Qwen's early-close guidance; the close marker is appended when `S` lacks it | Qwen guidance |
| `--vision` | enable media input and load Vision GPU allocations | off |
| `--vision-offload on\|off` | keep the vision tower in pinned system RAM and borrow Device memory only while encoding; requires `--vision` | `off` |
| `--vision-max-merged N` | merged vision tokens per image or video, `64..32768` | `32768` |
| `--no-cuda-graph` | disable CUDA Graph decode | graphs on |
| `--no-prefix-reuse` | disable compatible-prefix caching | prefix reuse on |
| `--device-state-slots N` | extra Device StateImages beyond `max-concurrency` | `max-concurrency` |
| `--host-context-mib N` | shared pinned Host budget for StateImages, KV and pause snapshots, including in-flight destinations | `8192 MiB + 8 native StateImages` |
| `--no-thinking` | disable thinking by default | thinking on |
| `--preserve-thinking` | preserve closed-turn assistant reasoning by default | off |
| `--tolerant-tool-calls` | recover complete tool calls cut by a malformed wrapper, a trailing suffix or the output budget instead of demoting them to text | off |
| `--cors` | permissive browser CORS headers | off |
| `--usage-chunk-choice` | give the streamed usage chunk a zero-delta choice, for strict client parsers that reject the OpenAI-conformant empty `choices` array | off |
| `--temperature F` | process-level temperature override | unset |
| `--top-p F` | process-level top-p override | unset |
| `--top-k N` | process-level top-k override (`0..20`; zero selects the top-20 cap) | unset |
| `--min-p F` | process-level min-p override | unset |
| `--presence-penalty F` | process-level presence-penalty override | unset |
| `--frequency-penalty F` | process-level frequency-penalty override | unset |
| `--seed N` | fixed seed when a request omits one | fresh random seed per request |
| `--greedy` | force exact argmax for all requests | off |

Context-cost coefficients resolve once at startup from generic defaults, matching compiled values,
and optional transfer or prefill entries from `--context-cost-presets FILE`. Prefill entries match
the hardware and a signature derived from the actual Text/Vision configuration, bindings and Uses.
A new representation without a matching measurement uses generic prefill coefficients. A malformed
file aborts startup; the operational context-cost record and JSONL `server_start` identify the
selected source.

Engine selects sampling defaults from the loaded architecture and the request's resolved thinking mode.
Qwen3.6-27B and Qwen3.8-27B use `1.0/0.95/20/0/0` for
temperature/top-p/top-k/min-p/presence penalty in thinking mode and `0.7/0.80/20/0/1.5` in
non-thinking mode. Qwen3.6-35B-A3B differs only in its thinking presence penalty, which is `1.5`.
Frequency penalty is `0` for all registered presets. Process flags override registered values,
request fields override process flags, and `--greedy` finally forces temperature `0`.

For `C=--max-concurrency` and `H=--device-state-slots`, total Device StateImage capacity is `C+H`.
Host state and Main/Backend KV share one startup-fixed byte budget; this is context storage, not a
limit on total process RAM. `--host-context-mib 0` disables Host context backing.
`--no-prefix-reuse` disables cross-request history reads and writes; pause/replay recovery remains
available, and the capacity flags may still be specified.

Run `./build/apps/ninfer-serve --help` for the exact option contract.

Serve writes human-readable operational records to stderr using
`YYYY-MM-DD HH:MM:SS.mmm  LEVEL  message`. Normal output covers material startup milestones,
readiness, request lifecycle, fixed-interval throughput, and shutdown; `--log-level debug` exposes
internal startup and resource-planning detail. Engine runtime diagnostics are ordinary records
prefixed `engine |`: a Device KV lease extended by releasing retained cache is `debug`; a lease
that cannot grow, recovery from out of memory or a failed request, and a failed prefix-cache save
are warnings or errors; the prefix-cache save at shutdown is `info`. FFmpeg's media-decoding
messages are records prefixed `media |`, so they never write inside the statistics panel: FFmpeg
errors are warnings, and its warnings and notices, such as swscaler's `deprecated pixel format`
notice for each JPEG, are `debug`. A terminal may use one
transient line during startup, but Serve throughput is always a persistent record. Redirected
stderr contains no terminal control sequences.

On an interactive terminal that accepts VT cursor control, and at `info` verbosity or more, Serve
pins a session statistics panel beneath the scrolling records (`--log-stats-panel off` removes it).
Records scroll above it and remain in the scrollback; the panel is redrawn after each record and
left on screen as ordinary output at exit. It has one row over every completed request and, once
more than ten have completed, one over the last ten:

| Column | Aggregate |
|---|---|
| TTFT | mean time to first token |
| cached | prefix-cache hit tokens / prompt tokens |
| prefill | computed (non-cached) prompt tokens / prefill seconds, in tok/s |
| decode | aggregate decode throughput in tok/s: output tokens after the first / decode seconds, where each batched decode round's time is split across the requests in it, so concurrent requests add up rather than each showing its per-stream rate |
| batch | mean decode batch size, each decode round weighted by its duration; decode / batch is the per-stream rate. The throughput record's `batch` counts rounds equally over its interval instead |
| `<DRAFTER>`, acc/rnd | model-drafter (MTP or DFlash) accepted / drafted tokens, and accepted tokens per model-drafted round; n-gram rounds are excluded |
| ngram, ng rnds | n-gram accepted / drafted tokens and verification rounds |
| archive | n-gram archive accepted / drafted tokens, shown once the archive has drafted |

The table is 84 columns wide, 93 with the archive column, so it fits a console window snapped to
half of a 1920-pixel screen; a narrower window cuts the rows at its edge.

Ratios and rates divide summed tokens by summed seconds, so each request weighs by its size. The
title counts completed, failed, cancelled, and rejected requests and, while throughput reporting is
enabled, the current running and waiting requests. The panel only reads the same outcomes as the
`req#N done` records; it changes no request behavior. Pretty values use readable units and rounded rates; use the independent request JSONL for
complete fields and full precision. Operational records never contain prompts, request bodies,
credentials, or arbitrary client error messages, and carry generated text in one case only: if a
tool marker is returned to text because its structure or tool identity cannot be represented,
Serve emits one warning with the failure classification and the returned markup's first 240
bytes from `<tool_call>` on (tabs and line breaks shown as spaces, `...` when cut), so a malformed
call can be diagnosed from the console. The request JSONL records only the classification.

## Live metrics

`GET /metrics` serves Prometheus text format 0.0.4 on the same port. It follows the server's
API-key authentication and works independently of `--request-log-jsonl` and
`--log-stats-interval-ms`. Engine failure leaves the endpoint readable with
`ninfer_engine_ready 0`. Counters start after startup warmup and reset when the server restarts.

```bash
curl http://127.0.0.1:8080/metrics
```

| Metrics | Meaning |
|---|---|
| `ninfer_model_info`, `ninfer_max_concurrency`, `ninfer_max_context_tokens` | Model/backend identity and startup limits |
| `ninfer_requests_running`, `waiting`, `paused`, `prefilling`, `decode_ready`, `replaying`, `materializing` | Current Engine gauges; prefill/decode/replay are subsets of resident requests |
| `ninfer_prompt_tokens_total`, `ninfer_prompt_tokens_cached_total` | Full input and reused tokens counted once on initial binding |
| `ninfer_prefill_tokens_total`, `ninfer_replayed_tokens_total` | Actual initial prefill and separate recovery recomputation |
| `ninfer_generation_tokens_total`, `ninfer_decode_tokens_total` | All committed outputs, or decode/control outputs excluding the first token; include thinking and injected control tokens |
| `ninfer_spec_decode_{rounds,draft_tokens,accepted_tokens,fallback_steps}_total` | Live native speculative work, including MTP and DFlash/DFlash2 |
| `ninfer_{preemptions,snapshot_restores,replay_restores}_total` | Pressure pauses and recovery routes |
| `ninfer_device_kv_{used,capacity}_pages`, `ninfer_device_state_{used,capacity}_slots` | Physical Main KV and StateImage occupancy; retained history also occupies these pools |
| `ninfer_host_context_{used,reserved,capacity,peak}_bytes` | Unified Host backing; reserved bytes are already included in used bytes |
| `ninfer_context_transfer_bytes_total{resource,direction}` | Actual State/Main KV/backend KV payload transfers |
| `ninfer_host_work_seconds_total{phase}`, `ninfer_device_wait_seconds_total` | Instrumented worker wall time; device wait is not CUDA kernel time |
| `ninfer_requests_total{outcome}`, `ninfer_response_failures_total` | Generation attempts entering preparation and subsequent response failures; protocol/model validation failures and token-count requests are excluded |
| `ninfer_time_to_first_token_seconds` | Histogram updated once at the first committed token, including preparation, queueing and binding |
| `ninfer_request_duration_seconds`, `ninfer_request_queue_seconds` | Histograms for settled generation outcomes, including cancellation; exceptional failures have separate counts |

Histograms expose `_bucket`, `_sum` and `_count`. Metrics have bounded labels; they do not retain
request IDs or request text. Rates are calculated by the consumer, for example:

```promql
rate(ninfer_generation_tokens_total[1m])

rate(ninfer_spec_decode_accepted_tokens_total[1m])
/ rate(ninfer_spec_decode_draft_tokens_total[1m])
```

The handler copies published snapshots and formats them outside the Engine worker. Scraping does
not reset counters or initiate device work. Detailed per-request records remain available through
the JSONL log. Monitoring tools with engine-specific metric names need an NInfer adapter.

## Structured request log

`--request-log-jsonl FILE` enables the machine-readable measurement log. The server opens `FILE`
in append mode and flushes every event, so successive model or MTP blocks may share one campaign
file. The parent directory must already exist. Failure to open the file aborts startup; the log path
is also rejected if it resolves to the model artifact.

Every line is one `ninfer_serve_request_log` schema-v25 JSON object. All events carry
`timestamp_unix_ms` and a process-unique `server_instance_id`; request IDs are monotonic only within
that server instance. Successful request-start records include request-scoped acquisition,
media-preprocessing wall/work, tokenizer, cache hit/miss/single-flight, and payload-size fields;
they do not infer request behavior from process-global counter deltas.

| Event | Contents |
|---|---|
| `server_start` | artifact path, architecture, public name, actual formats and prefill signature; resolved Engine and context-cache capacities, prompt-attention kernel and n-gram drafting options, thinking/non-thinking sampler defaults plus process overrides, thinking-history and thinking-budget defaults, Device arenas, the optional non-additive Vision layout inside the unified workspace, unified Host context capacity and occupancy, KV sizing ledger, CUDA Graph allowance, CUDA/GPU environment, and redacted argv |
| `request_start` | protocol, resolved sampler and seed, requested reasoning effort, actual initial thinking mode and optional budget, Responses semantic-change flag, output budget, stream/message/tool shape |
| `request_rejected` | parsed request shape, requested reasoning effort, media-item count, `phase: "prepare"`, and the exact HTTP status/type/code/parameter/message for a synchronous preparation rejection |
| `request_done` | finish reason, prompt/completion/cache/computed-prefill tokens, prefix reuse path, tool-call parse diagnostics, preemption/recovery counters, thinking-budget application counters, unrounded request-stage seconds, per-request Engine Host exposure, and complete speculative-decoding counters including n-gram and draft-archive counters |
| `request_scheduling` | request identity, pause/restore/recovery transitions, Snapshot revocation, Engine observation time and cumulative global/request work counters |
| `request_error` | the resolved request configuration and the generation, cancellation, or pre-outcome transport terminal message |
| `throughput` | interval token/decode/context-cache pressure counter deltas, authoritative worker Host-work deltas, current scheduler/resource gauges including Device KV growth leases, decode-round batch statistics, and hybrid prefix-cache gauges and counters when that cache is in use |

`requested_reasoning_effort` and `preserve_thinking` record the explicit options, or `null` when
unspecified. `enable_thinking` records whether the response starts in thinking mode.

`request_done.result.tool_call_parse` records whether a complete marker was seen, the structured
call count, empty non-string arguments omitted during normalization, schema-mismatched arguments
preserved for consumer validation, `duplicate_parameters_repaired`, and a stable fallback reason.
A parameter named more than once in one call keeps its last value, as in JSON object syntax, and
counts once in `duplicate_parameters_repaired` for each repeat instead of demoting the call to text.
Fallback reasons are `none`, `malformed_structure`, `invalid_tool_name`, `undeclared_tool`,
`trailing_content`, and `truncated_tail`. `truncated_tail` occurs only with `--tolerant-tool-calls`:
with a nonzero `structured_call_count` the recovered calls were returned structurally (a discarded
suffix or a call cut at the region end), and with none the region was returned as text. These
counters contain no tool arguments or generated text.

`request_done.timings_seconds` contains `prepare`, `ttft`, `vision`, `prefill`, `decode`, and `total`
as full-precision JSON numbers. Its `speculative` object contains `backend`, `draft_window`, `rounds`,
`drafted_tokens`, `accepted_tokens`, `fallback_steps`, and `accepted_per_position`. Rates can be
derived downstream from raw token counts and seconds instead of rounded stderr strings.
`generation.scheduling` records preemptions, snapshot/replay restores, replayed tokens, paused time
and request-owned transfer bytes. Replay rebuilds committed state without adding new output usage.

`request_scheduling` records `pause_started`, `paused`, `restore_started`, `restored`,
`replay_complete`, `recovery_complete`, `snapshot_revoked`, and a `terminal` boundary for preempted
requests. `preemption_index` identifies each pause cycle; `route` is `snapshot`, `replay`, or `null`
while pause preparation has not yet selected the saved representation. `steady_ns` is captured on
the Engine worker, and `elapsed_ns` starts at Engine submission; JSONL writes happen on the request
consumer thread. Compare `steady_ns` rather than delivery order across requests. `restored` ends
binding; `recovery_complete` marks the first fresh committed unit or normal terminal progress,
not merely rebuilding the old frontier. Cancellation can end the cycle without that event.

Each event's `progress` carries global and request-owned prefill, decode/control and replay token
counters. Between two boundaries, subtract the request delta from the global delta to measure
other requests' completed work. In particular, `restored` to `replay_complete` establishes whether
other work advanced during Replay without relying on periodic scheduler gauges. Events are enabled
only with request logging and add no per-token records.
`ngram_rounds`, `ngram_drafted_tokens`, and `ngram_accepted_tokens` are the part of `rounds`,
`drafted_tokens`, and `accepted_tokens` whose proposal came from n-gram copy drafting, and the
`ngram_archive_rounds`, `ngram_archive_drafted_tokens`, and `ngram_archive_accepted_tokens` counters
are the part of those whose copy source was the retained draft archive (see [n-gram
drafting](ngram.md)). With DFlash2 tree verification (`--draft-tree-nodes`), `tree_rounds` counts
rounds that verified a tree, `tree_side_rounds` those whose accepted path left the main chain, and
`tree_side_accepted_tokens` the drafts those paths accepted after leaving it, which a chain round
would not have accepted; `drafted_tokens` counts every verified tree column. The `speculative.ngram_archive` object reports that archive at the end of the
request: `enabled` when the server has one (`--ngram-archive-mib`), `bound` when the request was
bound to a draft session, `published` when its input and output were published into it,
`generation` as the session's latest completed generation, `sources` and `session_bytes` as the
bound session's retained source count and bytes, `total_bytes` as the whole archive's bytes, and
`sampling_seed` as the effective seed after request-domain separation for a request that named a
session, otherwise `null`.

`request_done.materialization.cached_prefix_tokens` and `restored_host_bytes` describe a hybrid
prefix-cache admission and are `0` with `--use-original-prefix-caching`.
`cached_prefix_tokens` is the longest prompt prefix held as cached KV blocks, whether or not it was
reusable: reuse also needs a state snapshot inside it, so a gap to `prefix_cache_hit_tokens` is
prefix lost to snapshot placement. `restored_host_bytes` is what the admission copied back from the
Host tier; those copies overlap the request's first prefill pass, so their time is part of its
prefill.

For `server_start.memory`, `workspace.capacity_bytes` is the only physical workspace allocation.
When Vision is enabled, `vision_workspace` reports the aggregate prompt and maximum-item token
bounds plus encode peak and handoff layout/usage within that same allocation; these bytes must not
be added to `workspace.capacity_bytes`. The field is `null` when Vision is disabled.
`host_state_image_bytes` is the Host size of one StateImage, the cost of one retained state
checkpoint whatever prefix depth it resumes; `host_kv_page_group_bytes` is the Host KV size of one
page group; and `host_cache_budget_bytes` is the `--host-cache-mib` budget those two are traded
under, `0` when that budget is not in use. `cuda_graph_allowance_bytes` is the CUDA Graph memory
the KV sizing reserved, and `cuda_graph_measured_bytes` the Device memory graph preparation
actually took at startup (`0` without CUDA Graphs); the startup log warns when the second exceeds
the first.

`server_start.engine.original_int8_prefill_kernel`, `int8_prefill_8bit_pv` and
`original_nvfp4_prefill_kernel` record `--use-original-int8-prefill-kernel`,
`--int8-prefill-8bit-pv` and `--use-original-nvfp4-prefill-kernel`. `draft_tree_nodes` (eight
entries, one per batch size, all zero for `auto`), `draft_tree_auto` and `draft_tree_paths`
record `--draft-tree-nodes` and `--draft-tree-paths`; `draft_tree_auto` is the resolved value,
false when the artifact cannot verify trees. `ngram_draft_window` and
`ngram_min_match` record `--ngram-draft-tokens` (`0` disables n-gram drafting) and
`--ngram-min-match`; `ngram_archive_bytes` and `ngram_session_bytes` are the draft-archive budgets
from `--ngram-archive-mib` (`0` keeps drafting request-local) and `--ngram-session-mib`; and
`ngram_native_sessions` records `--ngram-native-sessions`.

`request_done.engine_timing` separates FIFO `queue_wait_seconds`, blocking
`device_wait_exposed_seconds`, and five mutually exclusive Host-active exposure phases under
`host_exposed_seconds`: `engine_boundary`, `program_submit`, `program_post`,
`engine_commit_output`, and `engine_maintenance`. `total` is exactly their sum and excludes Device
wait. The nested `decode` object reports the request's decode-class Host exposure, Device wait, and
round count; `units` reports its prefill/control unit counts. In a compact batch every participating
request is delayed by the full round, so these values explain request latency but **must not be
summed across concurrent requests**.

`request_done.first_output_timing` freezes observations immediately before Engine publishes its
first nonempty output delta. It is `null` when no such output exists. This boundary differs from the
first accepted model token and the client's first HTTP output. Engine elapsed time begins at submit:
initial queue ends when the successful binding attempt starts, initial binding ends when the
binding is installed, and paused time includes pause preparation, waiting and restoration. The remaining
interval is resident time. Its `engine` observations describe resident Host/Device-wait exposure;
`prefill` and `replay` describe this request's submitted work. Terminal `engine_timing` still covers
the whole request.

The work `gpu_seconds` measures Text prefill stream intervals, including their MTP/DFlash work;
Vision encode, standalone bridges and exact-hit sampling fall outside that interval.
`context_transfers` reports completed request-owned State/Main KV/backend KV copies by direction,
using transfer-event time and bytes. GPU intervals overlap Host submission and waits, so they are
separate evidence, not additional wall-time stages. Background reclamation remains Engine-wide.

`result.generated_token_ids` records committed output token IDs for exact prefix analysis.
The JSONL file never records an API-key value; `argv` replaces it with `<redacted>`.
Operational stderr summaries are rounded and are not the
aggregation source. OpenAI Responses, OpenAI Chat, and Anthropic generation requests receive a
request ID when they enter synchronous preparation. Successful preparation produces
`request_start`; a preparation failure produces `request_rejected` without a matching start. Each
started generation transaction then has exactly one machine terminal: `request_done` when Engine
returns its outcome, or `request_error` when generation fails before an outcome exists. Later
response rendering, Responses storage, or terminal transport failures are operational response
events only and do not add a second JSONL terminal. Schema/model validation rejections before
preparation and token-count-only calls are not measurement requests and do not receive request IDs.

By default the server persistently reports aggregate activity every five seconds. `prefill` counts
prompt suffix tokens actually computed during the interval, excluding prefix-cache hits; `decode`
counts tokens finally committed by decode rounds, excluding the first token produced by prefill.
For MTP, DFlash and DFlash2 this is the accepted committed output, not draft or rejected tokens.
Pretty `batch` and JSONL `average_size` are decode row-rounds divided by decode rounds during the
same interval. The
`running`, `prefilling`, `decode_ready`, `waiting`, `paused`, `replaying`, `materializing`,
`capture_pending`, and `terminal_pending` fields are the Engine scheduler snapshot at the end of the
interval. The JSONL `context_cache` object reports selections, captures, StateImage operations,
transfers, tail-page COW and pressure spills as interval deltas; `occupancy` and `last_selection` are
end-of-interval gauges. The separate `scheduling` object reports preemptions, restores and replayed
tokens. Occupancy includes Host reservations while transfers are in flight.

The JSONL `throughput.host_work` object is the aggregation authority: the Engine worker counts each
wall-time segment once, independent of batch size. `elapsed_seconds` contains the same five
mutually exclusive Host phases and their `total`; `device_wait_seconds` is separate.
`work_class_seconds` splits Host and Device-wait time into decode, prefill, and control classes.
`detail_subset_seconds` and `detail_invocations` expose stats-publication work; these detail values
are already contained in a top-level Host phase and must not be added to `total`.
Per-round, per-row-round, and per-invocation normalized values are
`null` when their denominator is zero. Pretty throughput contains nonzero token rates and counts,
the current running/prefill/decode-ready composition, nonzero waiting/materialization/terminal
states, average decode batch, and Host-active time plus its fraction of the interval. Use JSONL for
complete measurement analysis.
Intervals with context materialization or retention activity are retained even when they contain no
token execution; only fully idle intervals are omitted. Downstream measurement should prefer the
raw counters and seconds over rounded stderr rates.

## Execution behavior

The server owns one resident Engine with `1..8` execution lanes fixed at startup. At each decode
boundary, eligible decode-ready requests form one compact batch, processed by one model traversal
and, when graphs are enabled, one exact-batch CUDA Graph replay. A
request joins that batch only after its single-request prefill finishes; when it completes or is
cancelled, the next boundary rebuilds the batch without an empty row.

`--max-pending-requests` bounds the requests waiting behind the active set. The total generation
request lifetime capacity is `max_concurrency + max_pending_requests`, including requests still in
CPU/media preparation and completed model results whose response has not yet been released. A full
capacity returns HTTP 429 with code `server_overloaded`. Token-count requests (Anthropic
`count_tokens`, Responses `input_tokens`) run the same preparation without generating; a separate
capacity of the same size bounds how many run at once, and beyond it they are rejected as
overloaded, as generation requests are. The absolute
`--pending-timeout-ms` deadline starts before preparation, covers media acquisition and Engine FIFO
waiting, and returns HTTP 503 with code `request_queue_timeout` if admission does not occur in time.
Once admitted, a request may pause for resource pressure without restarting this initial-admission
deadline. Fresh requests enter in FIFO order with a bounded bypass allowance when an earlier request
cannot fit. There is no admission ETA or unbounded overflow queue.

Input memory is bounded by the outstanding-request count and the per-request
`--max-request-mib` limit. Media preparation uses a shared permit pool sized from `--media-live-mib`
and the maximum supported prepared-payload size per request. Waiting media requests retain the same
cancellation and timeout deadline. Model output is bounded by the same finite request count and
each request's effective output-token limit; output callbacks and network serialization run
outside the GPU executor and do not delay formation of the next batch.

`--rope-yarn-factor F` is a startup-fixed runtime override shared by every request in this server;
it is not a per-request API parameter and does not modify artifact configuration. It accepts finite
values in `[1,4]` (default `1`, native RoPE unchanged). It extends only the allowed ceiling: the
8,192-token default stays unchanged unless `--max-context` is explicitly increased. KV memory and
selected speculative-backend limits still apply. Long-context extrapolation does not guarantee
answer quality; validate the workload before deployment.

`--max-context` is each sequence's logical ceiling. `--kv-capacity` fixes the shared Main Text KV
pool used by active requests and retained prefixes. `auto` accounts for the complete enabled runtime
and leaves 1 GiB of sizing headroom (`--vram-headroom-mib`); omitting the option makes it follow `--max-context`. The
CUDA Graph driver-state allowance reserved against that budget is 64 MiB plus 4 MiB for every
decode-graph executable the engine instantiates: one per topology class of each captured family,
for every batch size up to `--max-concurrency` (DFlash and DFlash2 capture a second family when
n-gram drafting is enabled). Measured on an RTX 5090 an executable takes 2.2-2.9 MiB, and up to
4.1 MiB when MTP verifies a 15-wide n-gram window at batch 4-8, so DFlash2 with
`--max-concurrency 2` reserves 112 MiB and uses about 30 MiB. The startup log and `server_start`
report both the allowance and the memory the graphs actually used, and the log warns when the
second exceeds the first.
Capacity resolves once at startup.

Before each prefill, decode or replay unit, the runtime reserves the additional pages and temporary
storage required by that unit. It first reclaims inactive cache resources when capacity is short.
If resident requests still cannot advance together, it pauses a younger request while preserving
progress for the oldest resident request. A paused request does not block fresh requests that fit
the remaining capacity. Restoration follows original request order and reserves enough space to
rebuild the saved frontier and complete one new execution unit.

A paused request keeps its committed output and protocol state. With sufficient Host backing, it
can restore a snapshot; otherwise it rebuilds model state from retained input and committed tokens.
Replay does not resample or republish those tokens, but it consumes compute and can increase gaps in
the output stream. The policy and ownership rules are defined in
[Resource scheduling and context cache](maintainer/resource-scheduling-and-context-cache.md).

Compatible prefixes are reused for both text and multimodal histories unless the server starts with
`--no-prefix-reuse`. A multimodal hit additionally requires matching token types, three-axis MRoPE
positions, encoded-media digest, grid, and consumer spans. Media wholly inside a matched prefix
skips Vision execution, while new suffix media is encoded normally. The pretty completion record
shows `cache N (P%, path)` using readable path labels; JSONL retains the exact
`prefix_cache_hit_tokens` and `prefix_reuse_path` fields (`root` or `checkpoint`). Reuse requires
matching KV, recurrent state, hidden state, selected-backend state and exact prefix identity.

Completed conversation endpoints serve direct continuations. A separate input checkpoint preserves
the stable boundary before a response that a later prompt may normalize or replace. With
`preserve_thinking=true`, that boundary precedes the response's generation prologue; with `false`,
it precedes the assistant turn whose closed reasoning may be omitted. The next request can therefore
recompute the changed suffix while retaining the preceding conversation. Capture follows these
semantic boundaries and shared-prefix hints.

Changing `preserve_thinking` or reasoning effort changes the rendered prompt where applicable;
already retained exact prefixes remain usable. An appended mid-conversation system message is an
ordinary suffix. Modifying, removing or moving a historical message changes the prefix and may
require an earlier checkpoint or a root prefill.

Speculative backends preserve protocol output shapes, stop behavior, and usage accounting. If a stop
truncates a multi-token MTP, DFlash or DFlash2 round, the Engine commits the exact accepted target prefix so
a following compatible turn can reuse it. Output-limit and context-capacity finishes map to
`length`/ `max_tokens`; ordinary model or string stops map to `stop`/ `end_turn`.

Function tools are rendered into the model prompt and generated calls are parsed into protocol
responses. NInfer does not execute tools and does not enforce client JSON Schema through constrained
decoding.

Prompt-token usage includes chat-template and expanded media tokens. Generated-token usage comes
from accepted output token IDs, including a stop token whose decoded text may be withheld.
