# HTTP serving

`build/apps/infernix-serve` loads one v3 `.infernix` (or `.ninfer`) artifact and exposes OpenAI- and
Anthropic-compatible HTTP endpoints over one resident Infernix Engine.

## Start the server

See [CUDA synchronization](cli.md#cuda-synchronization) for the shared `INFERNIX_CUDA_SYNC` setting.

```bash
./build/apps/infernix-serve models/qwen3_8_27b_nvfp4.ninfer \
  --host 127.0.0.1 \
  --port 8080 \
  --max-context 240000 \
  --max-concurrency 2 \
  --kv-dtype fp8 \
  --host-context-mib 8192 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft \
  --preserve-thinking
```

The command uses Qwen3.8-27B NVFP4. Each request has a 240,000-token logical ceiling. The default
hybrid prefix cache ([spec](maintainer/hybrid-prefix-cache-spec.md)) sizes the shared Main Text KV
pool to the Device memory the model and runtime leave free (`--kv-capacity` defaults to `auto`).
Requests acquire KV pages as execution advances, and pages no active request holds are the Device
block cache: content-addressed 64-token KV blocks shared across requests. If concurrent growth
exhausts the pool, cached blocks are evicted first, then the scheduler can pause a request and
restore it later. `--host-context-mib 8192` is the one pinned Host pool that blocks and sparse state
snapshots share; every other cache capacity is derived from `--max-concurrency` and
`--prefill-chunk`.

Other artifacts use the same command shape with their own path. For 35B-A3B DFlash, replace the MTP
selection with `--spec dflash --draft-tokens 7 --lm-head-draft`. Qwen3.8-27B
artifacts with DFlash2 companion weights also support `--spec dflash2 --draft-tokens 7`, with
`--lm-head-draft` optional. DFlash2 accepts draft counts 1..15 and supports the same sampling,
concurrency, prefix reuse, and image/video request surfaces. It may remain combined with
`--vision`. Qwen3.8-Flash-Next takes its DFlash2 drafter from a companion artifact:
`--spec dflash2 --draft-model Qwen3.8-Flash-Next-DFlash2-Infernix.infernix --draft-tokens 7`, with
1..7 drafts or `auto` and `--lm-head-draft` optional (see the
[Flash-Next guide](qwen3_8-flash-next.md)).

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

### Stop the server

Press Ctrl+C twice to stop the server. The first press only shows a prompt on the console's bottom
line, beneath the statistics panel: `Press Ctrl+C again within 5 s to save the prefix cache and
close` (`… to close` without `--prefix-cache-file`). Without an interactive console, the prompt is
logged as a warning. A second press within those 5 seconds stops the server; otherwise the prompt
disappears and serving continues. Ctrl+Break, closing the console window and `SIGTERM` stop at once.

Stopping refuses new connections, and every running and queued request ends with an HTTP 503
`service_unavailable` error, or an error event on a stream that has already started. The Engine
then saves the prefix cache when `--prefix-cache-file` is set, while the bottom line shows
`Closing: saving the prefix cache | Press Ctrl+C again to exit without saving`. The process exits
after logging `server stopped` and the save result. One more Ctrl+C during the stop exits at once
without saving: an unfinished save stops, its temporary file is deleted, and the previous file is
kept. Ctrl+C with console text selected only copies the text. Windows ends the
process about 5 seconds after its console window is closed, so stop with Ctrl+C when a large Host
tier must be saved. Before the server is ready, one Ctrl+C ends startup at once.

## Endpoints

| Method and path | Behavior |
|---|---|
| `GET /health` | Engine readiness |
| `GET /metrics` | Prometheus counters, gauges and latency histograms |
| `GET /metrics?format=json` | JSON snapshot: engine identity, live gauges, totals, and the last 128 terminal requests (`Accept: application/json` also selects it) |
| `GET /v1/models` | configured OpenAI model alias, effective `max_model_len`, and llama.cpp-compatible `meta` and `chat_template_caps` objects |
| `GET /v1/models/{id}` | lookup of the configured alias, `max_model_len`, `meta` and `chat_template_caps` |
| `POST /v1/chat/completions` | OpenAI-style chat generation |
| `POST /v1/responses` | OpenAI Responses Core generation, state, typed Items, and SSE |
| `POST /v1/responses/input_tokens` | Responses prompt-token count without generation |
| `GET /v1/responses/{id}` | retrieve a locally stored terminal Response |
| `DELETE /v1/responses/{id}` | delete a locally stored Response |
| `GET /v1/responses/{id}/input_items` | list that Response's normalized input Items |
| `POST /v1/messages` | Anthropic-style message generation |
| `POST /v1/messages/count_tokens` | Engine-native expanded input-token count |
| `POST /v1/decide` (alias `/v1/systemone`) | typed decisions read from the next-token distribution, with nothing generated ([Decisions](#decisions)) |

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
- Logits with no finite value to sample from (NaN or infinity, a numerical failure of the model's
  execution) fail only the request whose row produced them, with an HTTP 500 `internal_error`
  (an error event once its stream has started), and log `the model produced non-finite logits:
  failed 1 request; the Engine keeps serving`; the other requests of the round continue. The
  prefix-cache entries that request published are withdrawn, so a retry of the same prompt starts
  from what was cached before it; a `--prefix-cache-file` saved while the request ran is deleted
  and written again at the next save. A grammar
  that refuses a committed token, or reaches a position with no legal token, likewise fails only
  its own request.
- A failure after eight such recoveries in a row, with no successful unit between them, fails every
  queued and running request and stops the Engine. Any other exception on the worker (`worker
  crash: ...`) ends the Engine the same way. Nothing can be served until a restart, so within about
  250 ms the server stops accepting connections, logs `inference engine failed; server stopped,
  exit status 2`, and `infernix-serve` exits with status **2**. A launcher or service manager can
  restart it on that status; until it exits, `/health` answers 503.
- A CUDA error other than out of memory (an illegal address, a lost device) cannot be recovered
  in-process: the server prints `CUDA_CHECK(...) failed` with the error and aborts (a non-zero
  status other than 2), without final request-log records for the requests in flight.

`infernix-serve` exits with status 0 after an orderly stop (Ctrl+C or a termination signal), 130
when a further Ctrl+C exits before that stop finished, 2 after an Engine-wide failure while
serving, and 1 for an invalid command line, any startup, bind, warm-up or listen failure, or
another server exception.

Every OpenAI-compatible response carries a unique `x-request-id` header, including streaming and
error responses. Anthropic endpoints use their separate `request-id` contract.

A generation SSE endpoint commits its response only once the Engine has admitted the request: while
the request is still queued the client receives no response bytes, and the queue's own answers (a
queue timeout, an orderly stop) are plain HTTP statuses ([Execution behavior](#execution-behavior)).
A started stream emits the standard `: keep-alive` comment after five seconds without a protocol
event. The comment is transport-only: SSE clients ignore it, and it does not change generated text,
event ordering, usage, stored Responses, or request logs. Accepted connections also use TCP
keepalive and a 15-second retransmission limit (`TCP_USER_TIMEOUT` on Linux, `TCP_MAXRTMS` on
Windows); together with the heartbeat, a dead or unacknowledging peer is normally cancelled within
about 20 seconds, including while the request is waiting (TCP alone) or prefilling. A peer whose TCP stack remains connected and acknowledges data
cannot be distinguished from a reading application; proxies must close their upstream Infernix
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
| `ftype` | dominant loaded weights format (the Infernix quantization name, by payload bytes) |

`n_ctx` is a property of the running `infernix-serve` instance, so it reflects the `--max-context`
that this server was started with, not the model's native context. `n_ctx_train` reports the model's
own trained context separately.

Each model object also carries `chat_template_caps`, the chat-template capability object `llama.cpp`
reports, with one field:

| Field | Meaning |
|---|---|
| `supports_tools` | the loaded chat template (the artifact's, or `--chat-template FILE`) renders tool declarations into the prompt |

The server measures it once at startup, after warmup, through the same prompt path requests take: it
counts a one-message conversation that declares a probe tool, under a short name and under a longer
one, and the template supports tools when the longer name lengthens the prompt. A template that
ignores tools, or adds a fixed text whenever tools are present without the declaration itself,
counts both alike and reports `false`, as does a template that refuses a declared tool. A request
may still declare tools then, but the model never sees them. Generated tool calls are always read by
the model family's own tool-call parser, so the probe does not test a template that teaches the
model another call format. The Qwen templates, including those in `tools/chat_templates`, write each
declared tool as JSON into the System turn, so they report `true`.

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
  `video_url` extension using HTTP(S) or data URIs; image detail is omitted, `auto` or `high` (both the
  model's native full-quality preprocessing);
- nonnegative `max_completion_tokens` and the legacy `max_tokens` spelling; zero performs prompt
  processing without generation. As an extension (OpenAI rejects it), `-1` means no explicit limit:
  the field is treated as omitted, so the other spelling or `--default-max-tokens` applies;
- `temperature`, `top_p`, presence/frequency penalties, and signed integer `seed`;
- the compatible `top_k` (`0..20`) and `min_p` (`0..1`) sampler extensions;
- up to four non-empty stop strings, applied to both reasoning and answer output;
- `n:1`, text-only `modalities`, and `response_format: {"type":"text"}`;
- non-streaming responses and server-sent event streams;
- `stream_options.include_usage`, optionally shaped by `--usage-chunk-choice` for strict client
  parsers;
- llama.cpp-compatible terminal `timings`, plus opt-in `timings_per_token` and
  streaming `return_progress` observations;
- function tools with optional `strict:true` argument schemas, and free-form `custom` tools, the
  latter served to the model as a single-string-input function under the caller's own tool name so
  callers that dispatch by name keep working;
- `tool_choice` `auto`/`none`/`required`, named function or `custom` selection, `allowed_tools`, and
  `parallel_tool_calls`; assistant tool-call history, tool-result messages, and legacy
  function-call history;
- the top-level `reasoning_effort` field, where the `default` and `auto` aliases resolve to the
  server-configured level;
- `enable_thinking` and `preserve_thinking`, either at top level or in
  `chat_template_kwargs`;
- Assistant `reasoning_content` and `reasoning` history aliases.

Options whose observable behavior the Engine cannot provide are rejected when they request that
behavior. This includes nonzero `logit_bias`, requested log probabilities, audio/file input or
audio output, image detail other than `auto` or `high`, web search, moderation, low/high verbosity, stored
Chat Completions, and non-empty legacy `functions`.
Each capability rejection identifies the affected field and the guarantee Infernix cannot provide.

JSON mode and JSON Schema use the standard protocol fields:

| Endpoint | Field |
|---|---|
| Chat Completions | `response_format: {"type":"json_object"}` or `{"type":"json_schema","json_schema":{"name":"answer","schema":{...},"strict":true}}` |
| Responses | `text.format: {"type":"json_object"}` or `{"type":"json_schema","name":"answer","schema":{...},"strict":true}` |
| Anthropic Messages | `output_config.format: {"type":"json_schema","schema":{...}}` |

`json_object` requires an object root. Schema mode follows the supplied root type and enforces the
[supported assertions](maintainer/constrained-decoding.md#42-the-json-and-schema-execution-contract), including when
`strict` is omitted or false. Unsupported assertions return HTTP 400 before generation. OpenAI
errors distinguish `invalid_json_schema`, `unsupported_json_schema` and `unsatisfiable_json_schema`;
`param` identifies the request field followed by the schema JSON Pointer. Anthropic uses its
`invalid_request_error` envelope with the schema location in the message. Responses echoes the
selected `text.format` in aggregate responses and SSE response objects.

JSON output uses declared property order and admits JSON's insignificant whitespace, in runs of at
most 32 characters, so a model may write the pretty-printed layout it prefers; this includes the
inside of an object or array `const`/`enum` value (its keys, strings and numbers stay verbatim, its
members in the schema's order). State the desired
content in the prompt; the schema is not inserted into it. Only one output constraint may be supplied.

Schemas support positional arrays (`prefixItems` plus tail `items`) and inclusive/exclusive
`number` ranges. Bounded numbers use exact int64 integers or finite binary64-compatible decimal
and scientific notation with up to 17 significant digits. Bounds must retain their value when the
schema is parsed; numbers requiring greater precision receive `unsupported_json_schema`.
These capabilities also apply to strict tool parameters.

GBNF, choice and regex are available through the Infernix extension `structured_outputs`
on Chat Completions, Responses and Anthropic Messages. Supply exactly one member:

```json
{"structured_outputs": {"grammar": "root ::= \"yes\" | \"no\""}}
```

```json
{"structured_outputs": {"choice": ["positive", "neutral", "negative"]}}
```

```json
{"structured_outputs": {"regex": "(BUG|TASK)-[0-9]{4}"}}
```

Choice returns one literal string, preserving case and whitespace. The list must be nonempty;
duplicate entries have no extra weight, and an empty-string entry permits empty content.
Regex matches the complete content. It supports character classes, groups, alternatives and
repetition; `.` excludes line terminators, `\d`/`\w` use ASCII ranges, and `\s` includes Unicode
whitespace. Empty regex permits only empty content. Anchors are supported at the ends of top-level
alternatives. Lookaround, backreferences, word boundaries, Unicode properties, flags and unknown
escapes return HTTP 400. See the [language contract](maintainer/constrained-decoding.md#41-gbnf--regex--choice).
Invalid choices and regexes use `invalid_choice` and `invalid_regex`, with the request field in `param`.

These constraints apply to answer content; thinking is separate. GBNF supports recursive rules,
Unicode and repetition. All modes support streaming and all speculative backends, including
Qwen3.8-Flash-Next MTP and n-gram copy rounds. For assistant
continuation, the grammar covers the existing assistant content plus the generated suffix. Completion uses the model's EOS tokens;
output limits and cancellation can produce an incomplete answer. JSON modes can be combined with
active tools; GBNF, choice and regex require no active tools or `tool_choice:"none"`.
Output constraints reject custom stops. OpenAI errors use
`invalid_grammar` for invalid grammars and `constraint_dead_end` for a reachable prefix without a
legal next token. Anthropic reports these through its `invalid_request_error` envelope.
The `grammar` and `guided_*` aliases are not accepted.

Constrained responses include a Infernix `constraint` observation. `branch` is `undecided`, `content`,
or `tools`; `complete` means the committed language can end, and `terminated` means it accepted EOS.
A complete JSON value can therefore have `complete:true`, `terminated:false` and a length finish
reason. The observation also includes `cache` (`hit`, `built`, `waited`), `mask_positions`,
`mask_upload_bytes`, and `timings_seconds` for preparation, CPU mask work and matcher work.
These times are parts of existing request time and can overlap GPU execution.
Streaming sends the observation once: the Chat finish/usage chunk, the Responses terminal response
object, or Anthropic `message_delta`. Unconstrained responses omit it.

Semantically neutral fields do not make an otherwise executable request fail. All-zero
`logit_bias`, `logprobs:false`, `top_logprobs:0`, `verbosity:"medium"`, empty legacy tool controls,
text-only `audio` configuration, and `prediction` are accepted without changing Engine execution.
Metadata, user/safety identifiers, service-tier and prompt-cache hints are likewise advisory.
Unknown top-level fields are ignored.

A string `name` on a message of any role is accepted and ignored, as llama.cpp does: agent
frameworks stamp names onto system, assistant and tool messages, and the Qwen chat templates have no
place for a participant name. It does not participate in tool identity, prompt rendering, or output,
so two participants that share a role are not told apart. Non-string values are malformed. The
legacy `function` role keeps `name` as the function name.

For commonly generated OpenAI-compatible payloads, `repetition_penalty` is accepted only at its
neutral value `1`, and `mm_processor_kwargs` when empty or containing only null values. String-form
image/video URLs are also accepted.

Malformed protocol values return field-specific HTTP 400 errors. Invalid media sources, bytes, or
decoded content use `invalid_media`; remote fetch and timeout failures retain their dedicated
server-error codes. Failures in the normalized prompt contract use `invalid_prompt`; typed capacity
and availability failures retain their dedicated codes. Internal invariant failures are not
relabeled as client input errors.

The request `model` must equal the public model ID: the artifact `metadata.name` by default
(falling back to its architecture name when absent), or the explicit `--model-id` override. With
`--any-model-name` any non-empty name is served (a client's hard-coded or default name, such as
`gpt-4o-mini`), on Chat Completions, Responses, Responses input tokens and decide; the response and
the request log report the public model ID, as OpenAI reports the snapshot an alias resolved to.
`/v1/models` and `/v1/models/{id}` still list and retrieve only the public model ID.
Reasoning is returned separately as `reasoning_content`; answer text remains in `content`.

For non-strict tools, the JSON types a top-level tool-parameter schema admits guide conversion of
Qwen's untyped parameter text. They are read from its `type`, the values of its `enum` or `const`, a
local `$ref` (such as Pydantic's `{"$ref": "#/$defs/Status"}`), `allOf` (each branch narrows them), and
`anyOf`/`oneOf` (the union of their branches, when every branch names types), so a string enum's value
`"1"` stays the string `"1"`. They do not decide whether structurally complete markup is a tool
call. String-admitting values remain strings, including the empty string, except that the text `null`
is the null value where `null` is admitted too. An empty block for a
declared non-string parameter is omitted. Admitted JSON values retain their JSON type;
case-insensitive boolean text is normalized to `true` or `false`. A nonempty schema mismatch remains
a structured call: valid JSON retains its represented type and other text becomes a JSON string so
the tool consumer can report the validation error and continue the agent loop. Schemas that name no
type this way (no type keyword, an invalid type name, a remote, missing or circular reference)
retain untyped inference. Infernix does not apply defaults, enforce required
properties, or perform recursive JSON Schema validation on this route.

On the unconstrained route, string parameters preserve function/tool-call markers and balanced nested
`<parameter=...>...</parameter>` text as value bytes. The Qwen wire format has no delimiter escape,
so a standalone `</parameter>` ends a value only when whitespace and then another parameter, the
function's closer, or the end of the output follow it; any other one is value text, such as a shell
command that echoes the markup. An unmatched nested parameter opener, or a quoted closer that is
followed by the next token, cannot be represented unambiguously and makes that tool-call region
ordinary content. Later content is still examined:
the first tool-call region (any accepted marker form) that parses becomes the structured turn, and any
quoted markup before it stays ordinary content. A marker inside a Markdown code fence (```` ``` ```` or
`~~~`) or inline code of the answer is a quoted example and stays content even when it is a complete
call; only the text before the marker decides, so streamed and complete responses agree. A marker
after a closed fence, or after a sentence on the same line, still opens a call.

By default the unconstrained parser keeps that all-or-nothing behaviour. With
`--tolerant-tool-calls` the server recovers a call instead when the model adds a suffix after a
complete call, a second call is malformed, a single final call is cut by the output budget before
its closing tags, or the closing bracket after the function name is missing: the recovered call is
reported structurally with a `truncated_tail` diagnostic (logged at Info severity) rather than
demoted to text, and an undeclared tool name stays structured for the consumer to judge.

Tolerant mode also reads a drifted function opener inside `<tool_call>` as `<function=NAME>`: a
doubled or dropped `<` or a leaked `<|im_start|>` before it, `<=NAME>`, `<function:NAME>`
(`<invoke:NAME>`) and a bare `<NAME>` (closed by `</function>` or `</NAME>`). The forms without
the `function`/`invoke` keyword followed by `=` or a space are calls only when `NAME` is a declared
tool; for any other name, or with no tools declared, the region stays text. `<function =NAME>` is
ordinary markup in both modes. Inside a call, a parameter opener that lost its keyword, `<=NAME>`
(where the next `<parameter=NAME>` belongs, or first in the call), reads as `<parameter=NAME>` when
the call's tool declares `NAME`; otherwise it stays text of the value it follows, and the strict
parser keeps it as value text too. A repaired opener reports `tolerant_recovered`.

Tolerant mode also recovers calls the model wrote inside its thinking when it ends the turn with
its stop token without closing the thinking, so no call reached the answer. If, from some marker to
the end of that thinking, there is nothing but complete calls to declared tools (parsed strictly),
those calls become the structured turn (logged at Info severity). The thinking before them stays
reasoning and is never published as content. Nothing is recovered when the thinking was closed,
when the turn published any content, when it ended at the output limit, context capacity or a
cancellation, or when prose or a cut call follows the last call. Constrained tool output (below) is
well formed by construction and does not use either recovery.

Markers that are tokens of the model's vocabulary are recognized by the generated token, never by
their spelling. Generated reasoning ends at the model's `</think>` token wherever it appears, with or
without a following line break, and `<tool_call>` opens a call only as the model's `<tool_call>`
token, on both the constrained and the unconstrained route and in the recovery from unclosed
thinking. The same spellings written with ordinary text tokens (for example while the model
discusses a chat template) stay text in their channel. The model writes the marker tokens when it
quotes them too: a quoted `</think>` token ends the reasoning, and a quoted `<tool_call>` token stays
text only by the Markdown code rule above or when its region does not parse. Leading whitespace of
the answer after `</think>` is dropped; with an output constraint the grammar requires exactly `\n\n`
after the token, and only that separator is dropped. The wrapper-less forms agent harnesses drift
into (`<function=...>`, `<invoke ...>`, `<function_calls>`) have no token of their own and are read
by their spelling on the unconstrained route, except directly inside a `<tool_call>` spelled with
ordinary tokens, which stays text. An assistant prefill is the client's text, so a `<tool_call>` it
spells is the opener of the call it continues.

### Tool constraints

The three protocols share one constrained tool implementation:

| Choice | Generated calls |
|---|---|
| `auto` | Text or calls; zero to many |
| `none` | No tool calls; declarations remain in the prompt |
| OpenAI `required` / Anthropic `any` | One or more calls |
| OpenAI named function | Exactly one call to that function |
| Anthropic named `tool` | One or more calls to that tool |
| OpenAI `parallel_tool_calls:false` / Anthropic `disable_parallel_tool_use:true` | At most one call; exactly one when a call is required |

OpenAI `allowed_tools` supports `auto` and `required`. Selection changes generation permissions,
while all declarations retain their original order in the prompt. Requests with tools enable
**basic structural constraints by default**, including ordinary `auto` calls without `strict`.
The model can answer normally or start a tool call; a call must use the model's tool framing and a
declared function name.

Non-strict parameter names and order remain open. Their schema supplies the existing value
normalization hints; it is not compiled as a strict constraint. Open objects, root unions, and
unsupported schema assertions therefore remain usable. A call names each parameter at most once
(the constraint refuses a repeated name), so the published JSON object, streamed or whole, contains
each key once. `strict:true` additionally enforces the parameter contract below.

Top-level `tool_constraints:"auto"` opts into request-driven constraints: ordinary non-strict
`tool_choice:"auto"` then uses free generation. Strict tools, selection/count restrictions, and
`tool_choice:"none"` still enforce their requirements. `tool_constraints:"basic"` is the default.

With JSON object/schema output, `auto` permits either a JSON answer or a complete tool-call sequence.
Required/named choices permit calls for that turn; after supplying the tool result, use `auto` for
the final JSON answer. `none` permits only JSON. The JSON schema is validated on every turn.
This combination enforces tool framing even with `tool_constraints:"auto"`; `strict` continues to
control argument-value validation. Tool markers inside JSON strings remain ordinary string data.

A function's `strict:true` also constrains its argument values against its schema. Its parameter
root must reduce to a `type:"object"` schema with `additionalProperties:false`, including supported
`allOf` and local-reference combinations.
Properties are emitted in declaration order; optional properties may be omitted. Root
const/enum/unions are not supported. Values use the supported JSON Schema subset described above.
Top-level pure string parameters use raw text and preserve whitespace. A nullable string (a type list
of `string` and `null`, an `enum` or `const` of strings and `null`, or an `anyOf`/`oneOf` of such
strings and null: OpenAI strict mode's form of an optional field) is raw text too, and the text `null`
is the null value: the grammar admits the string's own values (its enum members, or text within its
bounds and pattern) and `null`, even where those bounds or that pattern refuse it. An enum or const
that lists the string `"null"` beside null is rejected, since its raw text is the null value's. Other
values use JSON; any other top-level string/non-string union (such as string/integer) is rejected
because the Qwen parameter format cannot distinguish those branches. Such unions inside JSON objects
or arrays are supported.
Raw values cannot contain the delimiter `\n</parameter>`; unsatisfiable required values are rejected.
Integer arguments use signed 64-bit values; number arguments use finite binary64-compatible
representations. Unsupported schemas fail with HTTP 400 before generation.

For `auto` without JSON output, text can precede the first call. Required/named choices start directly with calls
(after thinking, if enabled). Once a constrained call starts, the suffix consists of complete calls
and model EOS. Active tool constraints require model EOS and reject custom stop strings.
For ordinary non-strict auto calls that need custom stops, select `tool_constraints:"auto"`.
Token limits and cancellation can still stop generation: only completed calls are published.
A later call truncated by the token limit keeps `length`/`max_tokens`/Responses `incomplete` as the
terminal status.

Streaming publishes a constrained call as it is generated: its name when the function opener is
complete, then its arguments JSON in fragments, which concatenate to the arguments of the complete
call. A string value streams as it is generated (a strict string parameter, or a non-strict one whose
declared type admits strings); a strict JSON value streams verbatim; a value whose JSON depends on all
of it (a nullable string, a non-strict declared non-string type, or no declared type) arrives when its
parameter closes. A call cut by the token limit or the context keeps the fragments it streamed (its arguments
are then incomplete JSON) and closes with the terminal status; it stays out of the aggregate response
and of stored history. Each protocol renders the steps in its own events (below). Free tool output
(`tool_constraints:"auto"` without a constraint that applies) still publishes its calls whole in the
terminal event sequence, since a malformed region there reverts to text.
Assistant continuation may finish a partial call; a prefix containing a completed call is rejected.
The call such a prefix began streams from its start, prefix included.

Messages enter the selected template in their input order. The maintained Qwen templates keep
system/developer messages at their original positions. Where the selected template cannot render an
instruction role at its position, that turn is rendered under a role it does accept — a later
system or developer turn under `user`, a leading developer turn under `system` — and only the
role changes: the text, the number of turns and their positions stay as they arrived, and a
leading system turn is never touched, so the conversation's prefix-cache anchor is unchanged. A
template that refuses a system turn at every position is stating its own rule about instructions
rather than placing one, and the request fails as that template asks. The lowered turn count is
`instruction_role_lowerings` in the preparation block of `--request-log-jsonl`, and both
`infernix-serve` and `infernix` state the lowering once at startup. Pass
`--chat-template tools/chat_templates/qwen3_8.jinja` to render every turn under its real role. A
lowered turn counts as a user query for a template that selects its thinking history from the last
user turn, so `preserve_thinking: false` can drop earlier `<think>` blocks that the maintained
template keeps. A final assistant message is an assistant
prefill: generation continues that turn in place instead of opening a new assistant turn. Because
a reasoning opener cannot be continued, a prefill that carries media or whose thinking is not
explicitly disabled (top-level or `chat_template_kwargs` `enable_thinking`, or
`reasoning_effort: "none"`) is refused with `invalid_prompt`.

Prompt-bearing JSON objects retain their received member order through request parsing and prompt
rendering, including tool schemas and historical tool inputs. Canonical model-origin tool arguments
retain that member order in aggregate and streaming responses, so an unmodified replay reconstructs
the same ordered tool call. Infernix does not canonicalize semantically equivalent JSON: if a client
reorders members, inserts defaults, or otherwise rewrites a tool object, the changed rendered input
does not match the model-held endpoint and can reuse only an earlier exact cached prefix.
Generated token segmentation can also differ from re-encoding the same text, limiting prefix reuse.

`--chat-template FILE` selects a local Jinja template; by default, the server uses the template
stored in the artifact. See the [CLI guide](cli.md#text-input) for an example.

Only markup the chat template writes itself becomes control tokens. Every request string the
template receives is encoded as ordinary text: message content of every role, assistant reasoning
history, tool-call names and arguments, tool definitions and ordinary template kwargs. A spelling
such as `<|im_start|>`, `<|im_end|>`, `<think>` or `<|image_pad|>` there becomes ordinary text
tokens, so it cannot open, close or imitate a turn, reasoning block or media item. This differs
from Hugging Face `apply_chat_template` plus the tokenizer and from vLLM, which turn such spellings
into control tokens wherever they occur, so a prompt that contains them gets different token ids
here; a prompt without them encodes exactly as one tokenizer pass over the rendered text. Media
placeholders come from the template and bind to actual image/video inputs.

`chat_template_kwargs` passes a JSON object to the template in Chat Completions, Responses and
Anthropic Messages. Values duplicated in typed request fields must agree. Null standard options
mean unspecified; other null values remain `none`. Messages, tools, generation mode and tokenizer
special tokens cannot be overridden through kwargs.

`--default-thinking-budget N` sets a positive default thinking-token cap for requests that start
in thinking mode. Non-thinking requests receive no cap. It may coexist with `--no-thinking`
because requests can explicitly enable thinking. Anthropic
`thinking:{"type":"enabled","budget_tokens":N}` overrides this default for that request, and so
does a top-level `thinking_budget_tokens`, `thinking_token_budget`, `thinking_budget` or
`reasoning_budget_tokens` in Chat Completions and Responses (the names other servers' clients send;
two of them must agree). There, a positive N caps that request's thinking, `0` disables thinking
(rejected with `enable_thinking: true`), and `-1` keeps the server default.

Add `--default-thinking-budget 512` to the startup command to cap model-origin thinking at 512
tokens for every thinking-enabled request.

`max_tokens` counts the thinking, so a request that thinks for its whole output limit ends with
no answer (a harness that sends `max_completion_tokens: 8192` on every request hits this on hard
tasks). A thinking request with no budget of its own and no `--default-thinking-budget` therefore
gets one when its output limit is at least 1,024: thinking closes early enough to leave the
answer a quarter of that limit (at least 256 tokens, at most half), through the same early close
as any budget. 8,192 leaves 2,048 for the answer, 1,024 leaves 256. The output limit is
`max_tokens`, or the room the prompt leaves in `--max-context` when that is less: a client that
sends `max_tokens` 32,000 with a 250,000-token prompt at 262,144 context can emit 12,145 tokens,
so thinking closes after 9,109 rather than running to the end of the context. Below 1,024
nothing changes, and `--no-thinking-answer-room` lets thinking use all of the output limit again.
The request log's `thinking_budget` shows the budget a request ran with.

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

`--reasoning-loop off|stop|conclude` (default `off`) guards thinking-enabled requests without a token
constraint against a model that keeps rewriting the same passages in its thinking, which token
penalties do not catch because every pass varies a little.
- **Check.** Every 512 model-origin thinking tokens, at the end of a decode round, the thinking is
  split into words (letters, digits and non-ASCII bytes; ASCII case folded) and punctuation marks.
  The guard fires when at least 25 % of the last 2,000 words lie inside a 12-word passage that
  occurs at least three times in the last 30,000 words. A passage repeated twice (a recap) does not
  count. The thresholds are those of Strata's measured `reasoning_loop_recovery`.
- **`stop`** ends the reply there with `finish_reason` `length` (Anthropic `max_tokens`).
- **`conclude`** commits the same control span a reached thinking budget does (Qwen's early-close
  guidance, or `--thinking-budget-message`, plus `</think>`), so the model answers from the thinking
  it has; no prompt is read again. When the remaining output budget cannot hold the span and one
  more token, it stops instead.
- The console line and the request log (`reasoning_loop_detected`, `reasoning_loop_thinking_tokens`,
  `reasoning_loop_coverage`) record each firing. It can mistake a long, deliberately repeated
  checklist for a loop; keep it off where the thinking legitimately repeats whole passages.

For Chat Completions, `reasoning_effort: "none"` requests disabled thinking. The selected template
interprets the other standard values (`minimal`, `low`, `medium`, `high`, `xhigh`, `max`).
Conflicting explicit `enable_thinking` and effort values return `conflicting_template_option`.

Templates accept different subsets of those values: the official Qwen3.8 template raises for
anything but `low`, `medium` and `xhigh`. Infernix renders each value once when the template loads,
and a request for a value the template rejects renders with the nearest accepted value, a tie
rounding up: on Qwen3.8, `high` and `max` render as `xhigh` and `minimal` as `low`. This applies to
every endpoint and to `--reasoning-effort`. The values are tried on a one-message chat; a template
that rejects every value, or cannot render that chat at all (one that requires a system message, for
example), keeps the request's value and its error. Request logs record the effort the client asked
for.

`preserve_thinking` controls reasoning retention according to the selected template. Request
options override server defaults set with `--no-thinking` and `--preserve-thinking`. Unspecified
thinking, effort and preservation options use the template's defaults.

Many agent clients send an assistant tool-call message back without its `reasoning_content`. Within
a tool loop the template renders that reasoning, so the history then differs from the tokens the
model generated, and each later turn re-prefills from that message. With `--restore-tool-reasoning`,
the server remembers the reasoning of its last 4,096 tool-calling Chat Completions responses (at most
64 MiB) by tool-call id. An assistant message whose tool calls all belong to one remembered response,
and which carries no reasoning of its own, renders with that response's reasoning. Messages that
carry reasoning are left unchanged. `infernix_chat_reasoning_missing_total` counts the tool-call
messages received without reasoning, with or without the option, and
`infernix_chat_reasoning_restored_total` counts those restored.

Streaming begins with an assistant-role chunk, sends separate reasoning and content deltas, then a
finish-reason chunk and `[DONE]`. A constrained tool call streams as `delta.tool_calls` chunks: the
first carries the call's `index`, `id`, `type` and `function.name` with the first argument
fragment, later ones its `index` and the next `function.arguments` fragment. Free tool output sends
each call whole in one chunk before the finish reason. When `stream_options.include_usage` is true, a final empty
`choices` chunk contains completed usage. Aggregate and streamed usage include cached prompt tokens
and reasoning-token details; choices carry `logprobs: null` when log probabilities were not
requested, and aggregate assistant messages carry `refusal: null` because refusal output is not
supported.

A failure after the stream has started is sent as OpenAI does: one unnamed `data:` event whose
payload is the usual error body (`{"error": {...}}`, which OpenAI SDKs raise as an API error), then
the connection closes without a finish-reason chunk, usage chunk or `[DONE]`. A stream starts at
Engine admission, so it carries only mid-generation failures: a queue timeout or an orderly stop
before admission answers as a plain HTTP status on every route.

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
include terminal `draft_n` and `draft_n_accepted` when draft work occurred. Qwen3.8-Flash-Next
responses additionally include `expert_routed_n`, the request's routed (layer, expert) uses, and
`expert_hit_n`, how many of them its VRAM expert cache served. The cumulative per-token snapshots
below carry neither the draft nor the expert fields.

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

An expanded prompt beyond `--max-context` returns HTTP 400 `context_length_exceeded` with OpenAI's
wording, which agent clients match to compact the conversation and retry: "This model's maximum
context length is N tokens. However, your messages resulted in M tokens. Please reduce the length
of the messages." The Anthropic route returns `invalid_request_error` with the Anthropic API's
"prompt is too long: M tokens > N maximum", which Claude Code matches the same way. A streamed
Responses request (`stream:true` on `/v1/responses`) answers inside its event stream instead:
HTTP 200 with `response.created`, `response.in_progress`, then `response.failed` whose `error`
carries the code `context_length_exceeded` and the same message, the only form in which Codex
starts compacting; non-streamed Responses and the other routes keep the HTTP 400. M is exact up
to four times the limit; a longer prompt is not tokenized further and reads "more than M tokens".
The prompt's images and videos together beyond the 32,768-token Vision envelope (every turn of an
agent session resends its screenshots, so a long session reaches it) answer in the same
`context_length_exceeded` form, so the client compacts instead of failing every later turn: N is
the envelope, M the vision tokens counted up to the item that crossed it ("more than M"), and the
message ends with a parenthesized note that the counts are image and video tokens against the
per-prompt vision budget. A single image or video beyond the Vision execution capacity, and the
other media preprocessing resource rejections (source bytes, decoded pixels, live memory), return
HTTP 400 `media_budget_exceeded`. Qwen3.8-Flash-Next encodes media in memory lent from its expert
cache: a request whose encode window the cache cannot lend while other requests run waits in the
queue (subject to the queue timeout), and one it cannot lend with the request's own context
reserved and nothing else running also returns HTTP 400 `media_budget_exceeded` (fewer or smaller
images, or a shorter conversation, fit); the server keeps serving either way. HTTP 413 `request_too_large` is reserved for a raw request
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
rendered-token frontier is ignored without changing prompt content. `prompt_cache_key` does not
affect prefix identity. Valid TTL/retention values are accepted, but Infernix does not promise their
wall-clock residency; physical retention follows the prefix cache's eviction.

## OpenAI Responses Core

Infernix implements the typed-Item and semantic-event core of the OpenAI
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

| Field | Infernix Responses Core contract |
|---|---|
| `model` | required non-empty string; must equal the artifact-derived public model ID or explicit `--model-id` override (any non-empty name with `--any-model-name`) |
| `input` | string or typed Item array; it may be omitted or empty only when `previous_response_id` already supplies a user query |
| `instructions` | optional string, inserted before the reconstructed conversation for this request only |
| `previous_response_id` | optional ID of a retained local Response |
| `max_output_tokens` | non-negative integer; omission executes with `--default-max-tokens` but remains `null` in the Response object. `-1` (an extension) is treated as omitted |
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
| `text.format` | `text` (default), `json_object`, or `json_schema`; see output constraints above |
| `tools` | direct function and free-form `custom` definitions, or namespace groups containing them; see below |
| `tool_choice` | `auto`, `none`, `required`, a named function or `custom` tool, or `allowed_tools` of function and custom entries with mode `auto`/`required`; namespaced selection carries both `namespace` and `name` |
| `parallel_tool_calls` | `true` by default; `false` enforces at most one call |
| `max_tool_calls` | non-negative integer accepted as a hosted-tool no-op; Infernix does not execute hosted tools |
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
| `input_image` | user- or assistant-message part with HTTP(S) or data-URI `image_url`; detail omitted, `auto` or `high`; requires server `--vision` |
| `input_video` | Infernix extension with HTTP(S) or data-URI `video_url`; requires server `--vision` |
| `reasoning` | raw replay Item with `reasoning_text` content; summary/encrypted metadata may accompany raw text but cannot replace it |
| `function_call` | completed assistant call with optional `id` and namespace, plus required `call_id`, `name`, and JSON-object string `arguments` |
| `function_call_output` | completed result with required `call_id` and optional matching name/namespace assertion; `output` may be a string or a non-empty array of `input_text`/`input_image` parts |
| `custom_tool_call` | completed assistant call of a custom tool with optional `id` and namespace, plus required `call_id`, `name`, and string `input` |
| `custom_tool_call_output` | as `function_call_output`, for a custom tool's call |

Contiguous assistant-owned Items form one assistant history turn in the representable order
`reasoning` -> assistant message content -> `function_call` (or `custom_tool_call`). Multiple message Items append their
content parts, multiple calls retain declaration order, and a reasoning-only turn is retained. A
user, system, developer, or `function_call_output` Item ends the group; an order that would require
rearranging assistant content fails with `invalid_assistant_history`. Results are validated by
`call_id` and reordered to call declaration order before prompt rendering; unknown, duplicate, or
unrepresentable partial result sets fail with `invalid_tool_history`. Canonical input Items retain
client order. Input Item IDs are preserved when supplied and generated otherwise; duplicate IDs
fail.

System and developer message Items retain their positions in the input array. Top-level
`instructions` is represented as a leading developer turn for the current request; target-specific
role lowering occurs only in the Qwen family frontend, and it lowers a leading `developer` turn to
`system` when the selected template has no `developer` role at all, so `instructions` renders under
its own text instead of failing.

An `input_text`, `input_image`, or tool-result part may carry
`prompt_cache_breakpoint:{"mode":"explicit"}`. Write selection follows
[OpenAI prompt caching](#openai-prompt-caching); boundaries affect reuse opportunities, not prompt
identity or output semantics. String message status/phase metadata is accepted but has no Qwen
prompt representation.

`input_file`, `input_audio`, image `file_id`, image detail other than `auto` or `high`, reasoning metadata without raw
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
    "required": ["city"],
    "additionalProperties": false
  },
  "strict": true
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

Infernix gives each namespace/function pair a distinct internal Engine identity and restores the
separate `namespace` and `name` fields in aggregate output, SSE events, and replayed Items. The same
function name may therefore appear in different namespaces. Namespace members remain ordinary
client-executed functions; this does not add a remote MCP executor.

Infernix renders these definitions in the Qwen prompt and parses model output into separate
`function_call` output Items. Each output has a protocol Item `id` (`fc_...`) and a distinct
`call_id` (`call_...`). The client executes the function and sends a `function_call_output` Item in
a later request. Selection and strict argument enforcement follow the common tool contract above.

Free-form `custom` tools (`{"type":"custom","name",...,"format"}`, as Codex declares
`apply_patch`) are served as on the Chat Completions endpoint: the model sees a function with one
string parameter, `input`, under the tool's own name, and the declared format (`text`, or a `grammar`
with `syntax` `lark` or `regex` and its `definition`) is described in that parameter. The grammar is
not enforced. A generated call returns as a `custom_tool_call` Item (`ctc_...`) whose `input` is the
model's text exactly; streaming emits `response.output_item.added`, then
`response.custom_tool_call_input.delta` and `.done`, then `response.output_item.done`. The client
sends its result back as a `custom_tool_call_output` Item.

Functions and custom tools, top-level or in a namespace, accept `defer_loading:true` together with a
`tool_search` tool; `tool_search_call`, `tool_search_output` and `additional_tools` Items are accepted
as input. See [Deferred tools and tool search](#deferred-tools-and-tool-search). Other hosted tools,
remote MCP tools, output schemas, and caller restrictions that exclude direct invocation remain
unsupported.

### Response object and usage

A terminal wire response has `object: "response"`, one of `completed`, `incomplete`, or
`cancelled` in `status`, and a typed `output` array. Infernix may emit:

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
normal HTTP error responses, except a streamed request's prompt over the context limit, which
fails inside its stream after `response.created` and `response.in_progress`, as described with
the context limit above. `completed_at` is populated only for completed Responses. A
reasoning-only incomplete result contains no invented empty assistant message.

Usage reports the Engine's exact counts:

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
prompt prefix the Engine's prefix cache supplied. `output_tokens` is the count of accepted generated token
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
and content indices remain stable, and concatenated deltas equal the terminal Item. On the
constrained route a call's `response.output_item.added` arrives when its function opener is complete
(the reasoning and message Items are done by then); a `function_call` streams its argument deltas as
they are generated and ends with `.done` and `response.output_item.done` when the call completes.
`custom_tool_call` and `tool_search_call` Items complete when their generation ends: their input is
taken from the whole call. A call the output limit cut ends with an `incomplete` Item carrying the
arguments it streamed; the terminal Response lists it, and stored history does not. Free tool output
publishes every call Item at the end of its generation. Responses SSE
does not emit the Chat Completions `[DONE]` sentinel. With tools enabled, ordinary answer text still
streams immediately; only trailing whitespace, a possible wrapper-less opener (`<function...`,
`<invoke...`) on the unconstrained route, or the structured tool region is held. `<tool_call>` is one
token, so it never waits for following bytes.
On the unconstrained route, malformed tool markup is flushed back as ordinary text without losing bytes.

### Local response state and resources

`store` defaults to `true`. Stored Responses live only in this server process and are bounded by an
LRU store. They are lost on restart and are not OpenAI's durable cloud retention service.

`previous_response_id` reconstructs the complete stored input/output Item history before the new
input. The current `instructions` value is placed first but is not saved into the continuation
context, matching the Responses rule that previous top-level instructions do not carry forward.
Function definitions are request configuration rather than conversation Items and must be sent
again on tool-result turns. The reconstructed prompt follows the ordinary Engine path, so prefix
reuse applies naturally.

A stored Response also retains its resolved `preserve_thinking` value. A child which omits the
field inherits the parent value. An explicit different value creates a new semantic branch; prompt
rendering and identity still determine reuse. Changing the boolean alone never invalidates a
cached prefix the rendered prompt still matches.

Engine-local reuse is content-addressed: a `previous_response_id` child reuses the prefix its
reconstructed prompt shares with the cached history, whether or not either Response is stored.
Response-store eviction or deletion removes the HTTP object, not the Engine's cached blocks and
snapshots; those remain bounded by the cache's own eviction. No session key or cache marker is added
to the HTTP schema.

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
moderation, `include` values other than `reasoning.encrypted_content`, background execution,
compaction, files/audio, and OpenAI-hosted/MCP tools. Except for the two explicitly
documented placeholders, these are compatibility boundaries rather than silently accepted
approximations.

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
Mid-conversation System messages retain their text and input position; where the active chat
template rejects them there, Infernix instead renders them under the User role (role only, never
text or position). A final Assistant message
is an Assistant prefill: generation continues its existing text instead of opening another turn.
Assistant prefill cannot contain media and cannot start with Thinking enabled; a turn cut by the
output limit may carry reasoning content. It cannot carry `tool_use` blocks: every `tool_use`
needs its `tool_result` in the next User message, so a trailing one is rejected with
`invalid_tool_history`, as the Anthropic API rejects it.

Claude Code may place its attribution metadata in the first block of a top-level System array. If
that block is a text block beginning exactly with `x-anthropic-billing-header:`, Infernix consumes the
whole block before token counting, prompt preparation, and cache identity construction. The rule is
positional: a string-form System value, a later array block, or an inline System message with the
same text remains ordinary prompt content. A `cache_control` marker attached to the consumed block
is consumed with it rather than moved to adjacent content.

`--claude-code-prompt local` rewrites Claude Code's own System prompt for a local model; the default,
`as-sent`, uses the client's System text unchanged. The rewrite applies only when the consumed first
block is that attribution block and the next block opens with Claude Code's identity line (`You are
Claude Code` or `You are a Claude agent`); every other System value is used as sent. The rules,
taken from strata-nvfp4's `claude_code_prompt: "local"`, each match a marker of Claude Code's prompt:

- the identity line is replaced by a short local-agent preamble: call only the declared tools, by
  their exact names and with arguments that match their schemas; never write a tool call as text;
  run dependent calls in turn and read each result before acting on it;
- paragraphs opening with `You are an agent working with the user toward their goals`,
  `IMPORTANT: Assist with authorized security testing` (Anthropic's security policy) or `When you
  use a pronoun for someone` are dropped;
- bullets opening with `The most recent Claude models`, `Claude Code is available as` or `Fast
  mode for Claude Code` (Claude Code's Environment section) are dropped, and the `# Environment`
  heading with them when nothing else remains under it;
- a `# Memory` section that names its directory (``memory at `PATH` ``) becomes one paragraph that
  keeps the path; it ends at the next `# ` heading outside a code fence;
- runs of blank lines collapse to one. Every other section passes unchanged.

The rewrite depends on the text alone, so every turn of a conversation renders the same System
prefix and the prefix cache keeps hitting; Count Tokens counts the rewritten prompt. Each System
block keeps its `cache_control` breakpoint, and a block the rewrite leaves empty hands its
breakpoint to the end of the previous block.

`max_tokens` is optional for local clients and otherwise uses `--default-max-tokens`; a positive
value is the complete output budget. `max_tokens:0` is rejected because Infernix does not expose a
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

`thinking.display:"omitted"`, which current Claude Code sends on every request, returns each
Thinking block with an empty `thinking` string and no `thinking_delta` events. Its `signature`
(`signature_delta` when streaming) is `ninfer-reasoning.v1:` followed by the Base64 reasoning text,
so the reasoning stays out of the visible transcript but comes back with the block the client
echoes. When a replayed Thinking block has empty `thinking` and such a signature, request lowering
restores the reasoning from it, so the prompt (and its prefix-cache identity) is the one
`summarized` display would give; non-empty `thinking` text takes precedence, any other signature
stays non-semantic metadata, and a malformed Infernix signature fails with
`invalid_thinking_signature`. The signature is not encrypted or authenticated: this is a local
trusted server, and the value is the same reasoning text `summarized` display shows. Count Tokens
ignores `display`. `preserve_thinking` remains a Infernix extension for closed-turn reasoning
history. `output_config.effort` passes its protocol-validated value to the
selected template, substituting the nearest value the template accepts as described above.
`output_config.format` accepts JSON Schema output as described above.

User-defined tools support `name`, `description`, object `input_schema`, `input_examples`, and
`strict`. `tool_choice` accepts `auto`, `none`, `any`, or named `tool`; `disable_parallel_tool_use`
enforces a single-call limit. See the common tool contract above for schema and framing details.
`defer_loading:true`, the tool search tools (`tool_search_tool_regex_20251119`,
`tool_search_tool_bm25_20251119`) and `tool_reference` blocks inside a `tool_result` follow
[Deferred tools and tool search](#deferred-tools-and-tool-search). Tools that exclude direct model
calls, other Anthropic-provided/server tools, toolsets, MCP, and containers remain unsupported.
`tool_result` preserves text/image order and marks
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

Infernix maps representable boundaries to exact prompt frontiers and ignores a legal but
unrepresentable advisory boundary without changing the prompt. Reuse still requires exact rendered
identity and can read an existing owner without another `cache_control`. Aggregate usage reports
verified reused tokens in `cache_read_input_tokens` and, with the prefix cache, in
`cache_creation_input_tokens` the prompt tokens past them whose state this request's snapshots made
reusable for a later request (its deepest snapshot within the prompt); as in the Anthropic API,
`input_tokens` is then the rest, so the three sum to the prompt. Without the prefix cache, or for a
prompt that publishes nothing, creation is `null`. Streaming emits `message_start` after Engine
admission commits the prefix selection and before transfer/prefill output, so its cache-read split
is already exact and creation is still `null` there; terminal cumulative usage matches the aggregate
response.

Every successful Messages response also carries the llama.cpp-compatible top-level `timings`
object described for [Chat Completions](#llamacpp-compatible-request-observations), with the same
fields and semantics (`cache_n` equals `cache_read_input_tokens`). It is not an Anthropic field. A
stream attaches it only to the terminal `message_delta` event; `timings_per_token` and
`return_progress` are Chat Completions options and have no Messages equivalent.

Documents, Search Results, Files, Structured Outputs, server-tool blocks other than tool search,
container uploads, and other execution-dependent blocks are rejected with the missing capability
identified. Metadata,
service tier, inference geography, protocol-version/beta headers, cache TTL, and unknown advisory
fields do not block an otherwise executable request. The request `model` is any non-empty local
proxy label and is echoed in the response; it does not select the resident artifact.

Every Messages response carries a `request-id` header; error bodies also carry `request_id` and use
Anthropic error categories. Local admission overload maps to HTTP 529 and queue/media timeouts to
HTTP 504. Streaming owns the full Anthropic block lifecycle for Thinking, text, and tool use. A
constrained call opens its `tool_use` block (`server_tool_use` for a tool search tool) when its name
is complete, streams `input_json_delta` fragments as the arguments are generated, and closes when the
call completes; a call the output limit cut closes before `message_delta` with the fragments it
streamed. Infernix streams every constrained call this way, so a tool's `eager_input_streaming` is
accepted and changes nothing. Free tool output sends each call as one `input_json_delta`.

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

## Deferred tools and tool search

Agent clients with large tool catalogs (many MCP servers) can mark tools `defer_loading:true` on
Anthropic Messages and OpenAI Responses. The client still sends every definition on every request,
but the prompt's tool list leaves deferred tools out: the model finds them with a tool search, and
a found tool's definition appears in the conversation at the search's result. Everything before that
result (System prompt, listed tools, earlier turns) keeps its tokens as tools load, so the prefix
cache keeps serving the conversation, and the first prompt carries none of the deferred schemas.

- **Rendering.** Listed tools render in the template's tool section exactly as before; a request
  that defers nothing renders token-identically to one without these features. A result that loads
  tools ends with `You now also have access to the following functions:`, a blank line, and the
  definitions inside `<tools>`...`</tools>`, one JSON function definition per line in the form the
  listed tools use (with `input_examples` when declared). The text depends only on the loaded
  definitions, so a continuation Infernix runs and a later request that replays the same history
  render the same tokens. Each tool loads once per result.
- **Calls.** Every declared tool, deferred or not, belongs to the tool-call contract, which the
  Engine builds from the listed tools followed by the deferred ones (`PromptOptions::deferred_tool_jsons`).
  A call to a deferred tool parses, and is constrained under basic constraints or `strict:true`,
  like a call to a listed one, whether or not it has been loaded; the contract does not change as
  tools load. A named `tool_choice` may select a deferred tool.
- **Searching.** Infernix searches the deferred tools' names, descriptions, and their input schema's
  top-level property names and property descriptions. The regex search follows Python `re.search`
  with `IGNORECASE`, field by field (a field is one string, so `^` and `$` anchor to a name or a
  description), in linear time: patterns that need backtracking (backreferences, lookaround,
  possessive quantifiers, atomic and conditional groups) are refused as invalid input, as is a
  pattern over 200 characters or one that expands past 10,000 automaton states. Name matches come
  first, then other matches, each in declaration order. The BM25 search (k1 1.2, b 0.75) ranks
  tools by word tokens split at non-alphanumerics and lower-to-upper case changes, lowercased, with a
  final plural `s` removed from words over three characters; name words count twice; queries are
  at most 500 characters. Results are capped by `limit` (1 to 10,000, default 5).
- **Server-side searches continue the response.** When the model calls a search tool Infernix
  executes (Anthropic's search tools, a hosted Responses `tool_search`), the response does not end:
  Infernix appends the call and its result to the conversation and runs another generation, as a
  client tool loop would. A generation that also calls a client tool, or that the output limit cut,
  ends the response after its searches run. One response runs at most 10 generations; the output
  budget (`max_tokens`, `max_output_tokens`) covers all of them; a continuation keeps the request's
  ingress reservation. Usage sums the generations' input, cached and output tokens; `timings`
  sum them too, except that time to first token is the first generation's. A tool choice that
  forced a call was satisfied by the search: continuations use `auto`, and a selection that named
  only the search tool is lifted.

### Anthropic Messages

Declare a search tool beside the tools:

```json
{"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"}
{"type": "tool_search_tool_bm25_20251119", "name": "tool_search_tool_bm25"}
```

The `name` may be omitted but no other name is accepted. The model sees an ordinary function with
`pattern` (regex) or `query` (BM25) and an optional integer `limit`. Requests follow Anthropic's
rules: at least one tool stays listed (`all_tools_deferred`), a search tool is never deferred
(`invalid_deferred_tool`), a deferred tool carries no `cache_control` (`invalid_cache_control`),
and at most 10,000 tools are deferred (`too_many_deferred_tools`).

A search appears in the response as Anthropic returns it: the generation's calls in call order
(`server_tool_use` with a `srvtoolu_` id for a search, `tool_use` for a client tool), then each
search's result, then the next generation's blocks:

```json
{"type": "server_tool_use", "id": "srvtoolu_…", "name": "tool_search_tool_regex",
 "input": {"pattern": "weather"}}
{"type": "tool_search_tool_result", "tool_use_id": "srvtoolu_…",
 "content": {"type": "tool_search_tool_search_result",
             "tool_references": [{"type": "tool_reference", "tool_name": "get_weather"}]}}
```

A failed search returns `content: {"type": "tool_search_tool_result_error", "error_code":
"invalid_tool_input" | "execution_time_exceeded", "error_message": "…"}` with HTTP 200, and the model
reads `Tool search failed (code): message` as an error result; a search that matches nothing returns
an empty `tool_references` array, which the model reads as `No deferred tools matched the search.`
Streaming emits a search call as a `server_tool_use` block (`content_block_start` with an empty
`input`, one `input_json_delta`, `content_block_stop`) and its result as one `tool_search_tool_result`
block carried whole by its `content_block_start`. A response that reaches the generation limit ends
with `stop_reason:"pause_turn"`; one whose budget or context runs out after a search ends with
`max_tokens` or `model_context_window_exceeded`.

The client sends the assistant content back unchanged. Request lowering splits it at search results:
the blocks after a result belong to the generation the server ran after it, so the history renders
as the continuation did. Results follow call order even when a generation mixed a search with a
client call. A `tool_result` for a `srvtoolu_` id, a `server_tool_use` without its result, a result
before its call, a client `tool_use` answered nowhere before the next generation, and a search
history for a search tool the request does not declare are rejected; a `tool_reference` to an
undeclared tool fails with `tool_reference_not_found`. An Assistant message that ends with a search
result (a paused response) resumes with a new assistant turn.

A client that runs its own search (Claude Code's tool search, for example) returns `tool_reference`
blocks inside an ordinary `tool_result`:

```json
{"type": "tool_result", "tool_use_id": "toolu_…",
 "content": [{"type": "tool_reference", "tool_name": "get_weather"}]}
```

The referenced definitions load after the result's text and images. A `tool_reference` outside a
`tool_result` is rejected. Count Tokens lowers the same history.

### OpenAI Responses

Declare one `tool_search` tool. Without `execution`, or with `"execution":"server"`, the search is
hosted: the tool's description lists what can be loaded, each namespace that has deferred members
(with the namespace description) and each top-level deferred function or custom tool (with its
description), and the model passes the names in `paths`. Infernix loads the named namespaces'
deferred members and the named functions; unknown names load nothing. The response shows

```json
{"id": "tsc_…", "type": "tool_search_call", "execution": "server", "call_id": null,
 "status": "completed", "arguments": {"paths": ["crm"]}}
{"id": "tso_…", "type": "tool_search_output", "execution": "server", "call_id": null,
 "status": "completed", "tools": [{"type": "namespace", "name": "crm", "tools": […]}]}
```

after the generation's other call Items, and the next generation's Items follow. A search Item has
no events of its own beyond `response.output_item.added` and `.done`. A Response that reaches the
generation limit, or the request's `max_tool_calls` hosted searches, ends `completed` after its last
searches; one whose budget or context runs out ends `incomplete`.

With `"execution":"client"`, the model sees the client's `description` and `parameters` (a default
`query` string otherwise), and its call ends the Response as a `tool_search_call` Item carrying a
`call_id`. The client answers with a `tool_search_output` Item of that `call_id` whose `tools` are the
definitions to load; they need not appear in `tools`, and remain callable in later turns. An
`additional_tools` Item (`role:"developer"`, non-empty `tools`) loads its tools at its own position,
which renders as a System message there: a template that refuses System messages after the first
message refuses the request.

`defer_loading:true` needs a `tool_search` tool (`tool_search_required`), and a request declares at
most one. Hosted `tool_search_call`/`tool_search_output` Items without `call_id` pair in order. A
Response stored for `previous_response_id` keeps its searches' results as history. The namespace of
a namespaced tool that only a client `tool_search_output` declared is restored on output only while
the request still carries that Item; afterwards its calls show the flattened `namespace__name`.

## Decisions

`POST /v1/decide` answers typed questions about one piece of evidence by reading the model's
next-token distribution where the answer would start: each option is named by a single-token label
(`A`-`Z`, `a`-`z`, `0`-`9`, then the uppercase bigrams the tokenizer encodes as one token), and the
answer is those labels' probabilities. Nothing is generated for these answers (`noul`, `choice`,
`score`), so such a question costs one prefill; `number`, `scalar`, `point` and `box` generate a
constrained digit chain (below). The wire shape is TypeSafe Jev's `POST /v1/systemone` (also served at that path), as served
by [ignis](https://github.com/gpillon/ignis); the prompt is the one ignis measured its accuracy with
(0.934 balanced accuracy over 144 authored decisions on the 27B, the declared labels holding a median
99.8 % of the distribution), byte for byte.

```bash
curl http://127.0.0.1:8080/v1/decide -H 'Content-Type: application/json' -d '{
  "state": "Help! My payouts have been failing for 3 days.",
  "questions": {
    "queue": {"type": "choice", "instructions": "Which queue should this ticket land in?",
              "criteria": {"payouts": "Money leaving the account",
                           "billing": "Money coming in",
                           "fraud": "Suspected fraud or account takeover"}},
    "urgent": {"type": "noul", "instructions": "Is the customer blocked right now?"},
    "mood": {"type": "score", "instructions": "How upset is the customer?",
             "criteria": ["calm", "annoyed", "angry", "furious"]}}}'
```

```json
{"model": "qwen3.6-27b",
 "answers": {
   "queue": {"type": "choice", "choice": "payouts", "confidence": 0.98,
             "probabilities": {"payouts": 0.98, "billing": 0.01, "fraud": 0.01},
             "answer_mass": 0.999},
   "urgent": {"type": "noul", "noul": 0.93, "answer_mass": 0.998},
   "mood": {"type": "score", "score": 1.62, "confidence": 0.71,
            "legend": {"0": "calm", "1": "annoyed", "2": "angry", "3": "furious"},
            "probabilities": {"0": 0.01, "1": 0.36, "2": 0.6, "3": 0.03},
            "answer_mass": 0.997}},
 "usage": {"input_tokens": 393, "output_tokens": 0, "cached_input_tokens": 220,
           "computed_prefill_tokens": 173}}
```

(Values illustrative.)

- **`state`** is a string, JSON object or array, or OpenAI content parts (an array whose every
  element is an object with a string `type`), so the evidence may include images. Object key order
  is kept: the model reads the evidence in the order it was written.
- **`questions`** maps ids of your choosing to questions, answered under the same ids in the same
  order. A question has `type`, `instructions` (alias `question`: a string, object or array) and
  `criteria` (alias `options`):
  - `noul`: optional `{"true": description, "false": description}` (defaults `Yes`, `No`); answers
    `noul`, the true option's probability.
  - `choice`: an object of option name to description, `null` meaning the name describes itself;
    answers `choice`, the per-option `probabilities` and `confidence`, the top probability.
  - `score`: an ordered array of at least two level descriptions; answers `score`, the expected level
    index (it can land between levels), a `legend`, the `probabilities` and `confidence`,
    `1 - standard deviation / half the level range` (an even split between adjacent levels is
    confident; one between the extremes is not).
- **Generated primitives** write their answer under a token constraint: every output step may only
  produce the tokens the schedule permits (a digit, or the one token of a forced literal such as
  `{"x":`), so the run cannot leave the answer's shape. They cost a prefill plus one output token per
  step, verified speculatively like any other generation. Each takes `instructions` and an optional
  `digits`, and answers with the per-digit trace (`digits`: each digit and its probability among the
  ten) and an `uncertainty` in units of the value, `sum((1 - p) * 10^place)` — the model's own
  statement of its resolution, not a bound:
  - `number`: a whole number in a field of `digits` digits (1-6, default 3), right-aligned and padded
    with zeros, which carry no uncertainty; answers `number`.
  - `scalar`: a number of at most `digits` digits (1-15, default 8) that may be negative or have a
    decimal part, closed by the model with `}`; answers `value`, the `text` the model wrote, and the
    errors `malformed_scalar`, `too_many_digits` or `run_cut_short` instead of a guess. Unlike ignis,
    the colon after `{"value` is a choice between `":` and `":-`, the single token the tokenizer
    writes for a colon followed by a minus sign: ignis forces `":` and offers a separate `-`, a
    sequence the model almost never writes, which left a negative answer about 1 % of the mass.
  - `point` and `box`: a position (`x`, `y`) or bounding box (`x0`, `y0`, `x1`, `y1`) on a
    0-`10^digits - 1` scale over the state's one image (`digits` 1-6, default 3), answered in that
    image's own `pixels` (each axis rescaled by its own side) beside the `normalized` reading, with
    `method: "chain"`. A state without exactly one image is refused (`spatial_needs_one_image`).
  Their prompts are ignis's texts (`point` at three digits is the one its pointing measurements
  used); `answer_mass` is the lowest share of the distribution any chosen step's tokens held.
- **Option order is part of the prompt**: each option is shown under the label at its position, so
  reordering options asks a different question.
- **`answer_mass`** is the declared labels' share of the whole next-token distribution. Near 1 the
  model answered inside the declared set; a low mass means the probabilities are renormalized noise,
  however plausible the winner looks. `confidence` is not calibrated beyond what ignis measured on
  its authored set (every answer above 0.9 correct).
- **Validation is all-or-nothing** before any prefill: a malformed question, more than 256 options
  (the width ignis measured without decay) or more than the tokenizer can label, an empty
  instruction, an unknown question field, `digits` on a readout or `criteria` on a generated type,
  digits out of range, or a type not listed here (`question_type_unsupported`; ignis's
  attention-read `locate` and head-based `point`/`box` are not served) refuses the whole request with
  HTTP 400. Thinking is refused (`thinking_unsupported`): the answer
  is read where the prompt ends, where a thinking prompt opens a reasoning block.
- **Evidence placement and reuse.** JSON evidence follows the instruction in the system message, and
  content parts lead the user turn and end at an explicit prefix-cache boundary. The first question
  runs alone; the rest run in waves of `--max-concurrency` and resume from the cached evidence, so
  N questions over one state (an image included) prefill and encode it once.
  `usage.cached_input_tokens` and `usage.computed_prefill_tokens` show the split.
  `usage.output_tokens` counts the generated primitives' output steps; readout questions add none.
- **Failures after validation are per question**: a question that fails during execution answers
  `{"type": "error", "code", "message"}` beside its siblings' answers. A request whose every
  question failed returns the first question's error status instead.
- Each question is one engine request in the request log and metrics (`protocol: "decide"`): a
  readout with one staged output token that is never returned, a generated primitive with its run.

## Authentication and CORS

Pass `--api-key VALUE` to require the same value as an OpenAI bearer token or Anthropic
`x-api-key` header. `GET /health` and CORS preflight requests remain unauthenticated. A server
started without `--api-key` on a `--host` other than loopback (`127.0.0.0/8`, `::1`, `localhost`),
such as `0.0.0.0`, logs a warning that anyone who can reach the port can use the model.

```bash
curl http://127.0.0.1:8080/v1/models \
  -H 'Authorization: Bearer local-secret'
```

`--cors` adds permissive browser CORS headers. It is disabled by default.

## Server options

The table lists executable defaults. The startup example selects a long-context FP8/MTP3 profile.

| Option | Meaning | Default |
|---|---|---:|
| `--config FILE` | read options, and the artifact path, from a TOML file ([Configuration file](#configuration-file)); command-line options override it | unset |
| `--host H` | listen address | `127.0.0.1` |
| `--port N` | listen port | `8080` |
| `--api-key KEY` | required bearer or `x-api-key` value | unset |
| `--model-id ID` | override the public OpenAI model alias | artifact `metadata.name`, or architecture name |
| `--any-model-name` | serve OpenAI-route requests whatever non-empty `model` they name, reporting the public model ID (see Chat Completions) | off: other names are 404 `model_not_found` |
| `--chat-template FILE` | replace the artifact's chat template at startup; it must be one the target accepts | artifact template |
| `--claude-code-prompt as-sent\|local` | Anthropic Messages from Claude Code: use its System prompt as sent, or replace its Claude-specific parts with a short local-agent preamble ([Anthropic Messages](#anthropic-messages)) | `as-sent` |
| `--rope-yarn-factor F` | startup-fixed runtime YaRN factor, finite `[1,4]`; extends allowed ceiling only | `1` |
| `--max-context N` | logical context ceiling of each sequence | `8192` |
| `--kv-capacity N\|auto` | explicit shared Main Text KV capacity, or maximize it from remaining GPU memory (with offloaded experts, as on Qwen3.8-Flash-Next and Qwen3.5-122B-A10B, `auto` is `--max-context` x `--max-concurrency` and the expert cache takes the rest); omitted means `auto` with the prefix cache and `--max-context` with `--no-prefix-reuse` | `auto` |
| `--vram-headroom-mib N\|auto` | VRAM in MiB that startup sizing leaves free. Qwen3.5: after `--kv-capacity auto` sizes the KV pool (requires `auto`; `auto` = 1024). Offloaded experts (Qwen3.8-Flash-Next, Qwen3.5-122B-A10B): after the expert cache takes the rest, for the display and other programs (`auto` = 1024 with a display on the GPU, 256 without; [details](qwen3_8-flash-next.md#vram)) | `auto` |
| `--vram-past-budget` | Offloaded experts (Qwen3.8-Flash-Next, Qwen3.5-122B-A10B) on Windows: also size the expert cache into the VRAM the Windows per-process budget withholds (~0.6 GiB, ~240 more expert frames on an RTX 5090), measured at startup; that memory may be moved to system memory if another program needs VRAM ([details](qwen3_8-flash-next.md#vram)) | off |
| `--expert-tail-skip R` | Artifacts with offloaded experts (Qwen3.8-Flash-Next, Qwen3.5-122B-A10B), approximate: trades quality for decode speed. In decode and verification rounds, a routed expert missing from VRAM that every token of the round routes at top-k rank `R` or lower (`1..k-1`, 0-based, with k = 10 on Flash-Next and 8 on the 122B; k-1 skips least) is neither copied nor computed: it adds zero and the other experts keep their router weights. The output changes, is slightly worse, and depends on which experts the cache holds; prefill is unchanged; the reproducible route (`--reproducible`) refuses it. Strata does the same by default at rank 7 (`STRATA_ROUTE_TAIL_SKIP=7`) ([details](qwen3_8-flash-next.md#expert-tail-skip)) | off |
| `--expert-state FILE\|off` | Offloaded experts (Qwen3.8-Flash-Next, Qwen3.5-122B-A10B): the expert cache's saved state. At startup the engine fills the VRAM expert cache with the experts the file ranks highest (about 1 s for a full cache) and starts their use counts from it; it writes the file when it stops and every 10 minutes between requests (through `FILE.tmp` and a rename). A file from another artifact or of another size is ignored and replaced. Without a usable file Qwen3.8-Flash-Next seeds the cache from a built-in calibrated expert order instead, and with one the SSD tier's RAM takes the experts the file never ranked in that order; the 122B seeds it the same way from its own calibrated order once one is generated into the build (none is yet, so today it starts with an empty cache that each prompt's routes seed). `off` starts with an empty cache, no seed, and saves nothing; cold benchmarks need it | `<artifact>.expert-state` |
| `--expert-stripe FILE` | Offloaded experts (Qwen3.8-Flash-Next, Qwen3.5-122B-A10B) with the SSD expert tier: an expert stripe (`tools/flash_next/expert_stripe.py`) holding the tail of every expert record on a second drive; each record not in RAM is then read in two parts, its head from the artifact and its tail from the stripe, at the same time. The stripe must belong to the artifact. Unused when every expert fits in RAM ([guide](qwen3_8-flash-next.md#ssd-expert-tier)) | none |
| `--max-concurrency N` | resident execution lanes; valid range `1..8` | `1` |
| `--max-pending-requests N` | additional requests allowed to wait for admission | `16` |
| `--pending-timeout-ms N` | maximum preparation-plus-admission wait | `30000` |
| `--prefill-chunk N` | text-prefill chunk; with INT8, NVFP4, FP8, K8V4, VQ2 or K4V2 KV (unless an original INT8 or NVFP4 prompt kernel is selected) it is rounded down to whole prompt-attention waves, 896 tokens for the 24-head model on RTX 5090 (`4096` runs as `3584`) | `16384` with offloaded experts: Qwen3.8-Flash-Next (its reproducible route measured identical across chunks, and wide chunks prefill long prompts faster) and Qwen3.5-122B-A10B (narrow chunks are bound by the PCIe link); `1024` otherwise |
| `--use-original-int8-prefill-kernel` | prefill INT8-KV prompt attention with the original kernel at the requested `--prefill-chunk`. Without it INT8 KV uses the fast kernel (FP16 per-tile PV accumulation; a chunk whose row blocks alone would leave SMs idle splits its keys across CTAs within `--prefill-split-workspace-mib`) and rounds `--prefill-chunk` down to whole prompt-attention waves (896 tokens for the 24-head model on RTX 5090: `4096` runs as `3584`); requires `--kv-dtype int8` (startup rejects it with any other KV format) | off |
| `--use-original-nvfp4-prefill-kernel` | prefill NVFP4-KV prompt attention with the tiled kernel. Without it a chunk that sees more than 768 keys uses the fast kernel (the MX-FP8 tiled kernel with NVFP4 keys), which runs QK on block-scaled FP4 Tensor Cores directly over the stored K codes (Q as two NVFP4 terms), decodes each V tile once per CTA with FP16 per-tile PV accumulation, and splits the chunk's keys across CTAs when that is cheaper than leaving SMs idle (within `--prefill-split-workspace-mib`); shorter chunks keep the tiled kernel. Requires `--kv-dtype nvfp4` (startup rejects it with any other KV format) | off |
| `--prefill-split-workspace-mib N` | workspace in MiB, `0..16384`, that the FP32 partials of one prompt-attention launch may take when it splits its keys across SMs (the INT8 and NVFP4 fast prompt kernels, FP8 and K8V4 KV, and the VQ2 and K4V2 prompt kernels); a launch that would need more runs fewer splits. Less frees KV cache but slows prompt chunks of about 1-2K tokens over long context, such as follow-up turns. Per attention layer against unlimited, 1024-2048-token chunks take 17-25 % longer at `64` over 32K-128K cached keys, 2-5 % at `128`, and under 1 % from `192`; 3-4K-token chunks over 128K keys take up to 5 % longer below `384`. `0` turns splitting off (short chunks over long context up to 3× slower). With 4096-token chunks the startup workspace is about 515 MiB at the default, 380 MiB at `64` and 810 MiB unlimited (FP8 and K8V4 used no limit before). BF16 KV and the original INT8 and NVFP4 kernels ignore it | `256` |
| `--reproducible` | run the reproducible route. By default every fast path qualified for the loaded artifact's configuration (the kernels it runs and its KV format; artifacts with the same kernels share one) and measured faster there runs: faster kernels (such as 8-bit P×V in prompt attention) whose output is not bit-exact with the reproducible route but whose measured quality cost stays within Infernix's [numerical acceptance policy](maintainer/model-quality.md#6-numerical-acceptance-policy-and-fast-paths). Startup logs the paths selected (`fast paths \| ...`) and `server_start` records them. The reproducible route is deterministic and request-invariant: on a given engine build and GPU, the same request (input, sampling parameters and seed) with the same artifact and options always gets the same response, and concurrent requests, the prefix cache's contents and KV pressure do not change it (another engine version may give other bits; the lossless route's never change) (an unsupported artifact keeps this at a large decode cost, and startup warns `UNSUPPORTED ARTIFACT`), and neither does speculation: sampled tokens are drawn by Gumbel-max keyed by the seed, position and token (see [`--reproducible` in the CLI reference](cli.md) for what it gives up); the [fast-path registry](maintainer/model-quality.md#67-fast-path-registry) lists each path, the configurations it is qualified for, and its measured cost and speedup | fast paths on |
| `--lossless` | run the lossless route: every operation of the model's Hugging Face forward pass computed exactly and rounded once to the dtype it has there (weights decoded exactly, sums exact, logits correctly rounded to `--logits-dtype`, FP32 by default), so a request's output depends on the request and the artifact alone: not on the build, the prefill chunk, prefill or decode, other requests, the prefix cache or preemption. Text requests with `--kv-dtype bf16` (whose pages then hold V as BF16) and no speculation; no fast path runs, so it is slower than the reproducible route. Sampled tokens are drawn by Gumbel-max keyed by the seed, position and token ([definition](maintainer/lossless-route.md)) | off |
| `--fast-path NAME` | also run this fast path (a name from the [registry](maintainer/model-quality.md#67-fast-path-registry)), qualified for the configuration or not; repeatable. For qualification and timing runs (with `--reproducible`, it times one path against the reproducible route): a path that cannot apply to the loaded model fails startup | none |
| `--expert-a4-activations` | [nvidia/Qwen3.8-Flash-Next-FP4-Local](qwen3_8-flash-next.md#nvidia-fp4-local-checkpoint) only: run the routed experts with 4-bit NVFP4 activations (W4A4) at the activation scales NVIDIA calibrated for this checkpoint, instead of the 16-bit activations NVIDIA serves it with. Changes the output (a saved prefix-cache file from the other setting is not loaded). Every other artifact refuses it: their experts' arithmetic is fixed by the conversion | off: 16-bit expert activations |
| `--a8-activations` | Approximate: 8-bit activations instead of a projection's own arithmetic where that measured faster (a route stays only where it does). Decode and verification calls: int8 codes with an FP32 scale per 32 values (as llama.cpp's Q8_1) in a GGUF import's dense projections (calls of up to 31 columns) and in routed experts (GGUF blocks, NVFP4 W4A16 banks; calls of up to 128 columns) with their CPU-served misses. Prefill calls: on the same codec, a GGUF import's dense projections (32 or more columns) and those routed experts (more than 128 columns, in place of `expert-a4-residual`'s A4 terms where that runs); a groupwise Q4 MLP gate/up with int8 activations per token (the groupwise 27B artifacts with Q4 gate/up; 256 or more columns); MXFP8 dense projections with two E4M3 terms per 32 values (NVIDIA FP4-Local). Every other call keeps its route: the Q8, FP8, NVFP4 and BF16 dense projections (the NVFP4 27B's FP8 and NVFP4 sites already take the 8- and 4-bit activations their artifact permits), resident experts, and W4A4 and block-FP8 experts ([guide](qwen3_8-flash-next.md#a8-activations)). Changes the output (a saved prefix-cache file from the other setting is not loaded); `--reproducible`, `--lossless` and artifacts with none of these routes refuse it (the NVFP4 Qwen3.8-27B, the Qwen3.6-35B-A3B, Flash-Next Dense8 and the official FP8 import among them) | off |
| `--embedding-host` | Qwen3.5-family artifacts: keep the token embedding in pinned host memory and read its rows zero-copy, as artifacts with offloaded experts always do, so its VRAM (1.18 GiB on the 27B) goes to the KV and prefix caches. Same output; a prompt chunk reads its rows over PCIe | off: in VRAM |
| `--log-stats-interval-ms N` | aggregate throughput report interval; `0` disables it | `5000` |
| `--log-stats-panel on\|off` | pin the session statistics panel beneath the console log on an interactive terminal | `on` |
| `--log-colours on\|off` | colour the console statistics lines; never applies to file logs | `off` |
| `--log-level trace\|debug\|info\|warning\|error\|critical\|off` | pretty stderr verbosity | `info` |
| `--device N` | CUDA device index | `0` |
| `--ram-free-mib N` | physical Host RAM in MiB that everything Infernix keeps in RAM (weights, offloaded experts, the n-gram table, the prefix cache's Host tier) leaves free for other programs; startup refuses a lock that would take it, and Qwen3.8-Flash-Next plans every allocation against it in its RAM ledger | `8192` |
| `--expert-ram-mib N\|auto` | Offloaded experts (Qwen3.8-Flash-Next, Qwen3.5-122B-A10B): Host RAM in MiB for the routed experts; less than all of them need (~64.5 GiB on Flash-Next, 60.75 GiB on the 122B) keeps that many in RAM and reads the rest from the artifact ([SSD expert tier](qwen3_8-flash-next.md#ssd-expert-tier)); more than is free is an error | `auto` (what the ledger leaves) |
| `--expert-ram-layout exclusive\|full` | Qwen3.8-Flash-Next, when VRAM cannot hold every routed expert: `exclusive` keeps in RAM only the experts VRAM does not hold (plus copies of the ones a prefill borrows), `full` keeps a RAM copy of every expert whenever RAM allows (about 15 GiB more on the NVFP4 artifacts). Same output bits either way. Ignored with `--expert-ram-mib N` | `full` |
| `--ngram-residency auto\|ram\|volume` | Qwen3.8-Flash-Next: where the PLE n-gram table lives. `ram` reads the whole table into Host RAM at startup, locked for the CPU but not page-locked for the GPU (27.1 GiB for a GGUF import's IQ4_NL table, 48.8 GiB for the FP8 table) and never reads the volume again; `volume` reads rows on demand through a 2^20-row cache; `auto` takes RAM only when the RAM ledger still keeps every expert in RAM beside it ([details](qwen3_8-flash-next.md#n-gram-table-in-ram)) | `auto` |
| `--expert-pin-mib N\|auto` | Offloaded experts (Qwen3.8-Flash-Next, Qwen3.5-122B-A10B): at most N MiB of the experts' RAM page-locked for the GPU; the rest stays in RAM unpinned and the CPU serves it ([Page-locking limit](qwen3_8-flash-next.md#page-locking-limit)) | `auto` (what Windows lets the GPU page-lock) |
| `--context-cost-presets FILE` | optional runtime context-cost preset registry | generic + compiled defaults |
| `--max-request-mib N` | body-size limit before JSON parsing | `384` |
| `--media-cache-mib N` | LRU-retained prepared BF16 media payloads; `0` disables retention | `1024` |
| `--media-live-mib N` | all live prepared BF16 media payloads | `2048` |
| `--media-preprocess-threads N` | bounded media preprocessing workers; `0` selects at most 16 from host concurrency | `0` |
| `--request-log-jsonl FILE` | append full-precision server/request records | disabled |
| `--request-log-max-mib N` | rotate the request log once it reaches `N` MiB ([Structured request log](#structured-request-log)); requires `--request-log-jsonl` | one file without a size limit |
| `--request-log-keep N` | rotated request-log files kept (`0..1000`); requires `--request-log-max-mib` | `4` |
| `--response-store-max-records N` | maximum locally retained Responses objects | `1024` |
| `--response-store-max-mib N` | total local Response envelope/Item/context budget | `256` |
| `--route-trace PATH` | Offloaded experts (Qwen3.8-Flash-Next, Qwen3.5-122B-A10B) only: write the expert-routing trace `tools/expert_cache_replay/host_tier.py` replays — a header, one record per prefill, forced, decode and verify round, and one per CUDA-graph instantiation; no decode-path work when unset | disabled |
| `--kv-dtype bf16\|int8\|fp8\|nvfp4\|k8v4\|vq2\|k4v2` | KV-cache storage; `vq2` and `k4v2` are the 2-bit and 4-bit-K vector-quantized formats ([CLI](cli.md#context-and-memory)); `bf16` stores K as BF16 and V as FP16 (exact for every BF16 value with 2^-17 <= \|v\| < 2^16), and with `--lossless` V as BF16 | `bf16` |
| `--logits-dtype fp32\|bf16` | the vocabulary head's logits, which the sampler, greedy selection, draft acceptance, constraints and scoring read, on every route: `fp32`, the head's sums without a final rounding, or `bf16`, each logit rounded once to BF16 as Hugging Face Transformers returns them (near-equal logits then become exact ties, which selection breaks by token id); for comparisons with engines and references that round them. With `--lossless` each exact sum is correctly rounded to the chosen dtype | `fp32` |
| `--spec mtp\|dflash\|dflash2` | speculative backend | off |
| `--draft-tokens N\|auto` | MTP `1..7` (Qwen3.5/3.6 `1..5`); DFlash/DFlash2 `1..15` (Qwen3.8-Flash-Next DFlash2 `1..7`). `N` drafts and verifies N tokens every round (fewer only near the output limit). `auto` with DFlash or DFlash2: each round drafts the whole block and verifies the number its drafts' calibrated probabilities make fastest; on Qwen3.8-Flash-Next DFlash2 a block of 7 ([guide](qwen3_8-flash-next.md#run)); on Qwen3.5/3.6 the `adaptive-dflash` fast path where that is qualified (the NVFP4 Qwen3.8-27B, a block of 15 whose rounds of several requests verify 7; the Qwen3.6-35B-A3B, a block of 7; [DFlash](maintainer/dflash.md#automatic-verify-width-qwen38-27b--qwen36-35b-a3b)), elsewhere, and with `--reproducible`, the artifact's fastest fixed length every round (7 on the 27B, 3 on the 35B-A3B). On the 122B-A10B, whose verify columns each bring their own expert misses, `auto` with either drafter decodes without one (no draft length was faster than plain decode; the drafter is not loaded). `auto` (MTP): each round drafts the length the request's learned acceptance makes fastest, `0..7` on Qwen3.8-Flash-Next (whose one-request rounds also apply the cumulative acceptance cut, [design §11.3](maintainer/qwen3_8-flash-next-design.md#113-draft-length)); on Qwen3.5/3.6 it is the `adaptive-mtp` fast path (`1..5`) where that is qualified (the NVFP4 Qwen3.8-27B and Qwen3.6-35B-A3B); elsewhere, and with `--reproducible`, every round drafts the artifact's fastest fixed length (5 on the NVFP4 27B, 3 otherwise; none on the 122B-A10B, as above), which startup reports | unset |
| `--lm-head-draft` | optimized proposal head | off |
| `--draft-model FILE` | Qwen3.8-Flash-Next with `--spec dflash2`: the DFlash2 drafter's companion artifact (`tools.convert --components dflash2`); Qwen3.5/3.6 artifacts bundle their drafters and refuse it | unset |
| `--spec-min-p P` | `--spec mtp`, and `--spec dflash2` on Qwen3.8-Flash-Next (other combinations refuse it at startup). On Qwen3.8-Flash-Next MTP it applies to a fixed `--draft-tokens N` only (startup refuses it with `auto`): with `auto` its one-request argmax rounds verify the drafts before the first whose product of binned acceptance rates falls below an adaptive threshold (0.35 falling toward 0.25 as the request's drafts beat the table; on Dense8 with `--draft-tokens auto` 1.05x on conversational text and flat on structured output against the 0.5 cut, [design §11.3](maintainer/qwen3_8-flash-next-design.md#113-draft-length)), and its sampled-draft rows verify every draft. On Qwen3.5/3.6 it defaults to 0 (no measured default yet); with `--draft-tokens auto` a cut round verifies at the cut's width while the drafter keeps proposing the `adaptive-mtp` length, and with `--fast-path sampled-mtp-drafts` every draft is verified. A one-request round verifies its drafts up to the first whose drafter probability (Qwen3.5/3.6 MTP's draft head, DFlash2's candidate selector) is below `P`, `0..1`; `0` verifies every draft. It changes how many drafts a round verifies, not the target's distribution: with `--reproducible` greedy output still equals plain decoding, fast paths whose arithmetic depends on the round's width (on Qwen3.8-Flash-Next the default `q8-sliced-k`, `gdn-conv-fused` and `qsa-prepare-fused`, [registry](maintainer/model-quality.md#67-fast-path-registry)) can change near-tie greedy tokens with the verify width, and sampling keeps its distribution (only the realization of a seeded run may differ) | `0` |
| `--draft-tree-nodes auto\|LIST` | DFlash2, DFlash or MTP tree verification, each row's tree built on the device from the drafter's candidates (DFlash2's candidate lattice, DFlash's 16 most probable tokens per position, alternatives beside each MTP draft, which need `--fast-path sampled-mtp-drafts`); works with every `--kv-dtype`. `auto` (recommended) lets every all-neural round of up to four rows choose the chain or a tree of `draft tokens + 5` or `+ 9` columns from measured round time and acceptance at its batch size and context length; on an artifact whose GDN input projections cannot verify trees, and beside `--draft-tokens auto`'s `adaptive-dflash` widths (which verify chains faster), it keeps chain verification and logs a warning. A LIST fixes the column count by batch size: entry *c* (anchor included, `draft tokens + 2..32`) applies to rounds of *c* rows, the last entry repeats for larger batches and `0` keeps chain verification; see [tree verification](maintainer/tree-verification.md) | off |
| `--draft-tree-paths N` | most root-to-leaf paths per tree row, `2..8` | `8` |
| `--ngram-draft-tokens N\|auto` | verified n-gram copy proposals per round beside the `--spec` drafter (Qwen3.8-Flash-Next: also alone, `1..15`), `1..63`; `0` disables. `auto` (Qwen3.8-27B / Qwen3.6-35B-A3B) takes the configuration's measured window and minimum match: beside the NVFP4 27B's `--spec dflash2 --draft-tokens auto`, copies of up to 63 drafts (31 in rounds of several requests) at a minimum match of 12; elsewhere copies stay off (none was measured faster), which startup reports; see [ngram copy proposals](ngram.md#automatic-window) | `0` |
| `--ngram-min-match N` | minimum matched tokens for an n-gram proposal, `4..64` | `12` |
| `--ngram-archive-mib N` | RAM archive that keeps n-gram sources across the requests of a conversation; requires `--ngram-draft-tokens`; `0` keeps drafting request-local | `0` |
| `--ngram-session-mib N` | one conversation's share of the n-gram archive; no effect without `--ngram-archive-mib` | `128` |
| `--ngram-native-sessions` | also recognize Kilo, Codex and Claude session metadata as conversation identities; requires `--ngram-archive-mib` | off |
| `--default-max-tokens N` | output limit when omitted by a request; startup logs an `output limit` note when `--max-context` is at least four times it | `8192` |
| `--default-thinking-budget N` | positive thinking cap inherited by thinking-enabled requests | unset |
| `--no-thinking-answer-room` | without a budget, thinking may use all of the output limit (by default, from an output limit of 1,024, it closes early enough to leave a quarter for the answer; the limit is `max_tokens` or the context the prompt leaves, whichever is less) | answer room on |
| `--thinking-budget-message S` | text committed when a thinking budget ends thinking, replacing Qwen's early-close guidance; the close marker is appended when `S` lacks it, and `S` may contain `</think>` only as that final close | Qwen guidance |
| `--reasoning-loop MODE` | `off`, `stop` or `conclude`: reasoning-loop guard for thinking requests without a token constraint (details under [OpenAI Chat Completions](#openai-chat-completions)) | `off` |
| `--vision` | enable media input and load Vision GPU allocations | off |
| `--vision-offload auto\|on\|off` | keep the vision tower in pinned system RAM and borrow Device memory only while encoding; `on` requires `--vision`. `auto` is on for artifacts with offloaded experts, Qwen3.8-Flash-Next and Qwen3.5-122B-A10B (the expert cache lends the encode memory), and off for the other Qwen3.5 artifacts | `auto` |
| `--vision-max-merged N` | merged vision tokens per image or video, `64..32768` | `32768` |
| `--no-cuda-graph` | disable CUDA Graph decode | graphs on |
| `--no-prefix-reuse` | disable the prefix cache: every request prefills from the root, and a paused request recovers by replay; rejected with `--host-context-mib` and the cache options below | prefix reuse on |
| `--host-context-mib N\|auto` | pinned Host budget in decimal MiB that resolve to whole bytes: the slab pool the prefix cache's KV blocks and state snapshots share ([spec](maintainer/hybrid-prefix-cache-spec.md)), split at run time by eviction value. `0` keeps the cache on the Device only, and a nonzero budget below one snapshot is rejected. Qwen3.8-Flash-Next: must be positive (its state snapshots live in this pool); `auto` sizes it after the experts and the n-gram table, 0.4 of the RAM they leave beyond `--ram-free-mib`, between 1 and 8 GiB, and the startup line `RAM plan:` shows the result (other models take 8192). | `8192 MiB` |
| `--device-snapshot-slots N` | Device state snapshot slots (`1..64`); not available for Qwen3.8-Flash-Next, whose snapshots live in the Host tier | `max-concurrency + 1`; `+ 2` without a Host tier; `1` for an artifact with offloaded experts (Qwen3.5-122B-A10B) with a Host tier, whose further slots would take expert frames |
| `--cache-taps-per-request N` | new prefill state snapshots per request (`0..64`) | `8`; `2` without a Host tier |
| `--cache-tap-ladder N` | history-snapshot ladder base G; flexible taps at `prompt − G·2^k` | `max(4096, 2 * prefill-chunk)`; Qwen3.8-Flash-Next counts the chunk as at most `4096` |
| `--cache-tap-min-gap N` | minimum tokens between ladder snapshots | `max(1024, prefill-chunk)`; Qwen3.8-Flash-Next counts the chunk as at most `4096` |
| `--no-queue-holds` | turn off queue holds. With them, the snapshot each waiting request would resume from and its blocks are evicted after every other cache entry, the request furthest back in the queue first, and new snapshots or Host block writes that would need a held snapshot's slabs are skipped instead; without them eviction is plain LRU (Device) and GDSF (Host), so under a working set larger than the cache a queued request can lose its prefix while it waits ([spec §9.6](maintainer/hybrid-prefix-cache-spec.md#96-queue-holds)) | holds on |
| `--cache-tool-hold-secs N` | Qwen3.5: after a reply that ends in a tool call, hold the snapshots its next turn resumes from (and their blocks) like a queued request's, after every queued one, for `N` seconds (`0..3600`), since an agent client usually sends the next turn once its tool has run; `0` turns the holds off. Qwen3.8-Flash-Next has none and refuses a nonzero value ([spec §9.7](maintainer/hybrid-prefix-cache-spec.md#97-tool-call-holds)) | `30` (Qwen3.5) |
| `--cache-next-turn-ms N` | after a reply that ended on its own, once the server has had nothing to run for `N` ms (`0..3600000`), prefill the conversation's history as its next request will render it ([next-turn prefill](#next-turn-prefill), [spec §7.9](maintainer/hybrid-prefix-cache-spec.md#79-next-turn-prefill)); it yields to any request at its next unit and never changes output; `0` turns it off. Qwen3.8-Flash-Next: off by default, refused with `--reproducible` and `--lossless` | `150` (Flash-Next `0`) |
| `--cache-host-state fp32\|fp24\|bf16` | Qwen3.5: Host-tier form of snapshot recurrent (Gated DeltaNet) state; the attention KV cache keeps `--kv-dtype` and the Device copies stay exact. `fp24` and `bf16` round it to 15 or 7 mantissa bits, shrinking each 27B snapshot by 36 or 72 MiB (about 19 % and 39 % of a DFlash2 image), so the same Host tier keeps more snapshots and restores move fewer bytes; a snapshot restored from the Host then resumes from the rounded state, so output is no longer bit-identical to `fp32`. A cache file written with another form is not loaded; `fp24` and `bf16` are refused with `--reproducible` (whether a turn resumes from the Host depends on eviction) and for Qwen3.8-Flash-Next ([spec §5.6](maintainer/hybrid-prefix-cache-spec.md#56-host-state-precision)). Under the `fp16-gdn-state` fast-path candidate the pool itself stores the state as FP16, and the Host keeps those bytes exactly whatever this option says | `fp32`: exact |
| `--prefix-cache-file PATH` | at startup, restore the Host tier from `PATH` if the file exists; when the server stops ([Stop the server](#stop-the-server)), save it there once running and queued requests are cancelled (every Host-backed snapshot and the block path it resumes through). A save cut short is abandoned and the previous file kept: one more Ctrl+C during the stop also deletes the unfinished `PATH.tmp`, while Windows ending a closed console's process about 5 s after the close leaves it until the next save, so stop large caches with Ctrl+C. `PATH` may be relative (resolved against the launch directory) or absolute, e.g. `--prefix-cache-file "e:\NInfer-Deploy-V3\file.cache"`. Its directory must exist, and the flag needs a Host tier (not `--host-context-mib 0`). A file written for another artifact, KV format, speculative backend, RoPE scaling or `infernix-serve` binary is ignored and replaced at shutdown. The startup log shows the read's progress and reports what was restored; with a `--host-context-mib` smaller than the file needs, the most valuable snapshots and only the blocks they resume through are restored, and the log warns with the size the file needs; when no snapshot fits, it warns that nothing was restored and that the save at shutdown replaces the file. Saving writes up to `--host-context-mib` of data. | off: nothing is saved or restored |
| `--prefix-cache-save-mins N` | with `--prefix-cache-file`: also save the Host tier there every `N` minutes while serving, counted from the end of the previous save, and only when it changed since the last save or load. The save runs at the first point between execution units once due, also while requests run: they wait while it is written (the log reports the size, the save time and how long requests waited). It writes `PATH.tmp` and replaces `PATH` only once complete, so a crash or power loss leaves the previous file; stopping the server during a periodic save abandons it before the shutdown save. Each save rewrites the whole file (up to `--host-context-mib`, several GB), so a short interval writes a lot to the drive and wears an SSD faster: at 7.5 GB every 5 minutes that is up to ~2 TB a day, every 30 minutes ~360 GB. | off: saved only when the server stops |
| `--no-thinking` | disable thinking by default | thinking on |
| `--preserve-thinking` | preserve closed-turn assistant reasoning by default | off |
| `--tolerant-tool-calls` | recover complete tool calls cut by a malformed wrapper, a trailing suffix or the output budget instead of demoting them to text, and complete calls stranded in thinking the model never closed | off |
| `--restore-tool-reasoning` | give a Chat Completions tool-call message sent back without reasoning the reasoning this server generated for it (see Chat Completions) | off |
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
non-thinking mode; Qwen3.8-Flash-Next uses the same values. Qwen3.6-35B-A3B differs only in its
thinking presence penalty, which is `1.5`. Frequency penalty is `0` for all registered presets.
An artifact that carries its checkpoint's recommended presets (the text resource `sampling.json`,
[weight conversion](weight-conversion.md#resources-files-and-inspection)) uses them instead:
Ornith-1.5-122B-NVFP4 uses its model card's general set, `1.0/0.95/20/0/1.5`, in both modes, and
Ornith-1.5-122B-UNCENSORED-NVFP4 its card's `0.6/0.95/20/0/0`.
Process flags override registered values, request fields override process flags, and `--greedy`
finally forces temperature `0`. A request with an output constraint (JSON mode, JSON Schema, GBNF,
regex or choice) does not take the registered presence and frequency penalties: the constrained
format repeats its quotes, separators and keys by construction, and a penalty on them made sampled
answers pick legal neighbouring tokens that change the content (a JSON string opened by `",`, so
the value starts with a comma). A penalty from a process flag or the request still applies.

For `C=--max-concurrency`, total Device StateImage capacity is `C+H`, where `H` is
`--device-snapshot-slots` (0 with `--no-prefix-reuse`). The Host budget is startup-fixed context
storage, not a limit on total process RAM. `--no-prefix-reuse` disables cross-request history reads
and writes; pause/replay recovery remains available.

Run `./build/apps/infernix-serve --help` for the exact option contract.

Serve writes human-readable operational records to stderr using
`YYYY-MM-DD HH:MM:SS.mmm  LEVEL  message`. Normal output covers material startup milestones,
readiness, request lifecycle, fixed-interval throughput, and shutdown; `--log-level debug` exposes
internal startup and resource-planning detail. Engine runtime diagnostics are ordinary records
prefixed `engine |`: a Device KV lease extended by releasing retained cache is `debug`; a lease
that cannot grow, recovery from out of memory or a failed request, and a failed prefix-cache save
are warnings or errors; the prefix-cache save at shutdown is `info`. FFmpeg's media-decoding
messages are records prefixed `media |`, so they never write inside the statistics panel: FFmpeg
errors are warnings, and its warnings and notices, such as swscaler's `deprecated pixel format`
notice for each full-range (JPEG-coded) video, are `debug`. A terminal may use one
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
| experts | Offloaded experts (Qwen3.8-Flash-Next, Qwen3.5-122B-A10B): routed experts the VRAM expert cache served / routed-expert lookups (the rest ran from RAM on the CPU or GPU, or from the SSD tier), shown once a request reported them |
| prefill | computed (non-cached) prompt tokens / prefill seconds, in tok/s |
| decode | aggregate decode throughput in tok/s: output tokens after the first / decode seconds, where each batched decode round's time is split across the requests in it, so concurrent requests add up rather than each showing its per-stream rate |
| batch | mean decode batch size, each decode round weighted by its duration; decode / batch is the per-stream rate. The throughput record's `batch` counts rounds equally over its interval instead |
| `<DRAFTER>`, acc/rnd | model-drafter (MTP or DFlash) accepted / drafted tokens, and accepted tokens per model-drafted round; n-gram rounds are excluded |
| ngram, ng rnds | n-gram accepted / drafted tokens and verification rounds, shown only when `--ngram-draft-tokens` enables n-gram drafting |
| archive | n-gram archive accepted / drafted tokens, shown once the archive has drafted |

The table is 66 columns wide without n-gram drafting, and 84 with it (93 with the archive column),
so it fits a console window snapped to half of a 1920-pixel screen; a narrower window cuts the rows
at its edge. The experts column adds 8 (74 without n-gram drafting).

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

### Configuration file

`--config FILE` reads options from a [TOML 1.0](https://toml.io/en/v1.0.0) file whose keys are the
long option names without the leading dashes. `infernix` (the [CLI](cli.md#configuration-file))
reads the same format, so one file can configure both: top-level keys apply to whichever program
reads the file, keys in a `[serve]` table only to `infernix-serve` and keys in a `[cli]` table only
to `infernix`. A key in the program's own table overrides the same key at the top level, and each
program skips the other's table. [`infernix.example.toml`](../infernix.example.toml) is a commented
example.

```toml
model = "models/qwen3_8_27b_nvfp4.ninfer"
max-context = 240000
kv-dtype = "int8"
spec = "dflash2"
draft-tokens = 7
lm-head-draft = true

[serve]
max-concurrency = 2
prefix-cache-file = "qwen3_8_27b.cache"
api-key = "local-secret"

[cli]
max-context = 32768
```

```bash
./build/apps/infernix-serve --config infernix.toml --port 8081
```

- `model` names the artifact. The command line's first argument, when it is not an option, takes
  precedence over it; startup refuses to run with neither.
- A flag takes `true` or `false` (`false` leaves it unset). Every other option takes a string, an
  integer or a float, which reaches the option exactly as its command-line text would (`"auto"`,
  `4096`, `0.7`).
- A repeatable option (`fast-path`; for the CLI also `stop`, `reasoning-stop`, `stop-token-id` and
  `choice`) takes an array, each element given once in order; any other option refuses an array.
- Each value is validated as on the command line. Relative paths resolve against the launch
  directory, not the file's.
- Options on the command line override the file wherever `--config` appears, and the file overrides
  the defaults. Values of a repeatable option accumulate: the command line's are added to the
  file's. A flag the file sets cannot be turned off from the command line, since flags have no
  negative form.
- Startup refuses a file it cannot parse as TOML, an unknown key (including a dashed name, an
  underscored one such as `max_context`, `help` and `config`), a table other than `[serve]` and
  `[cli]`, a table inside either, or a value of another type, naming the key and its line.
  `--config` may be given once.

The startup log has a line listing the options the file gave
(`config FILE: --max-context 240000 ...`). The `server_start` record's `config` holds the file's path and the same arguments
(`null` without `--config`), with the `api-key` value redacted as in `argv`.

## Live metrics

`GET /metrics` serves Prometheus text format 0.0.4 on the same port. It follows the server's
API-key authentication and works independently of `--request-log-jsonl` and
`--log-stats-interval-ms`. Engine failure leaves the endpoint readable with
`infernix_engine_ready 0`. Counters start after startup warmup and reset when the server restarts.

```bash
curl http://127.0.0.1:8080/metrics
```

| Metrics | Meaning |
|---|---|
| `infernix_model_info`, `infernix_max_concurrency`, `infernix_max_context_tokens` | Model/backend identity and startup limits |
| `infernix_requests_running`, `waiting`, `paused`, `prefilling`, `decode_ready`, `replaying`, `materializing` | Current Engine gauges; prefill/decode/replay are subsets of resident requests |
| `infernix_prompt_tokens_total`, `infernix_prompt_tokens_cached_total` | Full input and reused tokens counted once on initial binding |
| `infernix_prefill_tokens_total`, `infernix_replayed_tokens_total` | Actual initial prefill and separate recovery recomputation |
| `infernix_generation_tokens_total`, `infernix_decode_tokens_total` | All committed outputs, or decode/control outputs excluding the first token; include thinking and injected control tokens |
| `infernix_spec_decode_{rounds,draft_tokens,accepted_tokens,fallback_steps}_total` | Live native speculative work, including MTP and DFlash/DFlash2 |
| `infernix_expert_cache_frames`, `infernix_expert_cache_{routed,hits,promotions,landed,cpu_served}_total` | Qwen3.8-Flash-Next's VRAM expert cache: frames backed now (one expert each), routed (layer, expert) uses, those served from a frame, experts copied into frames by the policy or adopted from staged misses, and misses computed by the CPU expert team; absent for other models |
| `infernix_{preemptions,snapshot_restores,replay_restores}_total` | Pressure pauses and recovery routes; a snapshot restore resumed from a cache snapshot at the paused frontier |
| `infernix_{root,prefix}_selections_total` | Initial bindings from the root and from a prefix-cache snapshot |
| `infernix_device_kv_{used,capacity}_pages`, `infernix_device_state_{used,capacity}_slots` | Physical Main KV and StateImage occupancy; cached blocks and snapshots also occupy these pools |
| `infernix_prefix_cache_{blocks,device_blocks,evictable_blocks,snapshots}` | Prefix-cache tree blocks (Device or Host), those resident in Device KV, the Device ones no request holds, and state snapshots; zero without the cache |
| `infernix_host_context_{used,capacity}_bytes` | Host context-cache occupancy in occupied slabs and its capacity (`--host-context-mib`); size the tier from these |
| `infernix_prefix_cache_{snapshot_hits,evicted_blocks}_total`, `infernix_host_context_{snapshot_evictions,write_bytes,restore_bytes}_total` | Snapshot hits, tree-block evictions, Host-tier snapshot evictions and Host-tier copy volume |
| `infernix_host_work_seconds_total{phase}`, `infernix_device_wait_seconds_total` | Instrumented worker wall time; device wait is not CUDA kernel time |
| `infernix_executions_total{kind}`, `infernix_execution_seconds_total{kind}`, `infernix_execution_max_seconds{kind}` | The Engine worker's execution units (`prefill` steps, `replay` steps, `control` batches, `decode` rounds), each from its call to its return: counts and wall time since startup, and the longest since the warm-up, which shows a stall the totals hide |
| `infernix_decode_batch_rounds_total{batch}`, `infernix_decode_batch_seconds_total{batch}`, `infernix_decode_batch_max_seconds{batch}` | Decode rounds by batch size (1 to `--max-concurrency`): the cost of a batch size, and its longest round since the warm-up |
| `infernix_constraint_requests_total{outcome}`, `infernix_constraint_cache_total{result}` | Settled constrained requests by completion state and compilation-cache access |
| `infernix_constraint_{prepare,mask,matcher}_seconds_total`, `infernix_constraint_mask_{positions,upload_bytes}_total` | Constraint work aggregated at request settlement, including truncated/cancelled results |
| `infernix_constraint_draft_wait_seconds_total` | Live draft-ready wait counted once per batch; a subset of device wait |
| `infernix_requests_total{outcome}`, `infernix_response_failures_total` | Generation attempts entering preparation and subsequent response failures; protocol/model validation failures and token-count requests are excluded |
| `infernix_chat_reasoning_missing_total`, `infernix_chat_reasoning_restored_total` | Chat Completions assistant tool-call messages received without reasoning, and those `--restore-tool-reasoning` restored |
| `infernix_time_to_first_token_seconds` | Histogram updated once at the first committed token, including preparation, queueing and binding |
| `infernix_request_duration_seconds`, `infernix_request_queue_seconds` | Histograms for settled generation outcomes, including cancellation; exceptional failures have separate counts |

Histograms expose `_bucket`, `_sum` and `_count`. Metrics have bounded labels; they do not retain
request IDs or request text. Rates are calculated by the consumer, for example:

```promql
rate(infernix_generation_tokens_total[1m])

rate(infernix_spec_decode_accepted_tokens_total[1m])
/ rate(infernix_spec_decode_draft_tokens_total[1m])
```

The handler copies published snapshots and formats them outside the Engine worker. Scraping does
not reset counters or initiate device work. Detailed per-request records remain available through
the JSONL log. Monitoring tools with engine-specific metric names need an Infernix adapter.

### JSON representation

`GET /metrics?format=json` (or `Accept: application/json`) returns one JSON document instead of the
text exposition. Prometheus text remains the default; `?format=prometheus` selects it explicitly.
An unknown `format` is a `400`, never a silent fallback.

```bash
curl 'http://127.0.0.1:8080/metrics?format=json'
```

The document carries `schema`, `time`, `engine`, `live`, `totals`, and `requests`. It exists because
the text exposition cannot carry per-request records: a request id or a per-row timing in a label
would make the series cardinality unbounded.

| Field | Meaning |
|---|---|
| `schema` | `infernix.telemetry/1`; additive-only within v1 |
| `engine` | model id, architecture, weight and KV formats, speculative backend, context and concurrency limits |
| `live` | engine phase (`idle`, `reading`, `generating`), request counts by phase, and the interval prefill/decode rates. A rate is `null` until the throughput reporter has produced one |
| `totals` | monotonic counters since startup, plus terminal request counts |
| `requests` | the last 128 settled generation requests (completed, cancelled, failed), oldest first; requests rejected before generation appear only in `totals.rejected` |

Each request row carries its id, completion time in Unix seconds, protocol, finish reason, token
counts, cache reuse, TTFT/prompt/decode milliseconds, total seconds, speculative draft and accept
counts, and the Qwen3.8-Flash-Next expert-cache lookups. Time sums the engine does not accumulate
(`prompt_ms` and `decode_ms` at the totals level) are `null`. There is no `hardware` object: the
engine measures no GPU telemetry, and an absent block is honest where a zero would not be.

The route follows the server's API-key authentication. The document contains no prompt, completion,
tool argument, header value, key, or client address. `requests` is a bounded window, not a durable
log; use `--request-log-jsonl` when every request must be retained.

## Structured request log

`--request-log-jsonl FILE` enables the machine-readable measurement log. The server opens `FILE`
in append mode and flushes every event, so successive model or MTP blocks may share one campaign
file. The parent directory must already exist. Failure to open the file aborts startup; the log path
is also rejected if it resolves to the model artifact.

`--request-log-max-mib N` rotates the file by size; without it the file grows without limit. When
`FILE` reaches `N` MiB (at the record that crosses the limit, so a file can exceed it by one record)
it is renamed `FILE.1`, older files move up to `FILE.2` .. `FILE.K` for `--request-log-keep K`
(default `4`), the oldest beyond `K` is deleted, and a new `FILE` is started; with `K = 0` the full
file is deleted instead. A file already at the limit when the server starts is rotated before it
writes. Every new file begins with a verbatim copy of this server's `server_start` record (same
`server_instance_id` and timestamp), so each retained file can be read on its own; a reader that
combines files should count one start per `server_instance_id`. If the rename fails, for example
because another process holds `FILE` open without delete sharing on Windows, the server logs a
warning, keeps appending to `FILE`, and tries again after another `N` MiB. Rotated names are also
checked against the model artifact path.

Every line is one `infernix_serve_request_log` schema-v31 JSON object. All events carry
`timestamp_unix_ms` and a process-unique `server_instance_id`; request IDs are monotonic only within
that server instance. Successful request-start records include request-scoped acquisition,
media-preprocessing wall/work, tokenizer, cache hit/miss/single-flight, payload-size fields, and
`instruction_role_lowerings`, the turns the selected template could not render at their positions;
they do not infer request behavior from process-global counter deltas.

| Event | Contents |
|---|---|
| `server_start` | artifact path, architecture, public name, actual formats and prefill signature; resolved Engine and context-cache capacities, prompt-attention kernel and n-gram drafting options, thinking/non-thinking sampler defaults plus process overrides, thinking-history and thinking-budget defaults, Device arenas, the optional non-additive Vision layout inside the unified workspace, unified Host context capacity and occupancy, KV sizing ledger, CUDA Graph allowance, CUDA/GPU environment, redacted argv, and the `--config` file's path and redacted options (`null` without one) |
| `request_start` | protocol, `label`, resolved sampler and seed, requested reasoning effort, actual initial thinking mode and optional budget, Responses semantic-change flag, output budget, stream/message/tool shape |
| `request_rejected` | parsed request shape, `label`, requested reasoning effort, media-item count, `phase: "prepare"`, and the exact HTTP status/type/code/parameter/message for a synchronous preparation rejection |
| `request_done` | finish reason, prompt/completion/cache/computed-prefill tokens, prefix reuse path, `prefix_miss`, tool-call parse diagnostics, preemption/recovery counters, thinking-budget application counters, unrounded request-stage seconds, per-request Engine Host exposure, and complete speculative-decoding counters including n-gram and draft-archive counters |
| `request_scheduling` | request identity, pause/restore/recovery transitions, Engine observation time and cumulative global/request work counters |
| `request_error` | the resolved request configuration and the generation, cancellation, or pre-outcome transport terminal message |
| `throughput` | interval token/decode/context-cache pressure counter deltas, authoritative worker Host-work deltas, current scheduler/resource gauges including Device KV growth leases, decode-round batch statistics, hybrid prefix-cache gauges and counters when that cache is in use, and `gpu` (NVML, `null` without it): the device's SM and memory clocks, temperature, power and clock-event reasons at the interval's end, its energy, average power, joules per decode token (intervals without prefill) and the fractions of it the power and thermal policies held the clocks down; a figure the driver does not report is `null` |

`requested_reasoning_effort` and `preserve_thinking` record the explicit options, or `null` when
unspecified. `enable_thinking` records whether the response starts in thinking mode.

`request_done.prefix_miss` explains a large re-prefill, or is `null`. The server keeps the token
contexts (prompt and output) of its last 16 requests while the request log is on. A request that
computes at least 4,096 prompt tokens is compared with them, and the context sharing the longest
prefix with its prompt (`common_prefix_tokens`) decides:

- `kind: "diverged"`: the prompt departs from that context inside it, and they share at least half
  the prompt. The client changed history there (it dropped earlier reasoning, say). `earlier` and
  `current` hold the token ids and text 64 tokens either side of the divergence (`first` is the
  window's first index; characters a cut splits become U+FFFD).
- `kind: "evicted"`: that context is a prefix of the prompt (or the prompt of it), yet at least
  4,096 of its tokens were recomputed (`lost_reuse_tokens`): the cache had lost them.

Both also give the earlier request's `earlier_engine_request_id` and `earlier_context_tokens`, and
this request's `reused_tokens` and `computed_prefill_tokens`.

`label` is the request's `X-Infernix-Request-Label` header, or `null` without one: a client
(`tools/bench/run_serve_corpus.py` sends `run_serve_corpus`) marks its traffic so that benchmark
requests can be told from real use. Only a single header of 1-128 printable ASCII characters
counts; anything else is ignored, and the request itself is unaffected.

`request_done.result.tool_call_parse` records whether a complete marker was seen, the structured
call count, empty non-string arguments omitted during normalization, schema-mismatched arguments
preserved for consumer validation, `duplicate_parameters_repaired`, and a stable fallback reason.
On the unconstrained route a parameter named more than once in one call keeps its last value, as in
JSON object syntax, and counts once in `duplicate_parameters_repaired` for each repeat instead of
demoting the call to text (a constrained call cannot repeat a name).
Fallback reasons are `none`, `malformed_structure`, `invalid_tool_name`, `undeclared_tool`,
`trailing_content`, and `truncated_tail`. `truncated_tail` occurs only with `--tolerant-tool-calls`:
with a nonzero `structured_call_count` the recovered calls were returned structurally (a discarded
suffix or a call cut at the region end), and with none the region was returned as text.
`tolerant_recovered` is true when `--tolerant-tool-calls` turned output the strict parser rejects
into structured calls: a recovered `truncated_tail`, or a kept call whose opener (a dropped `<` or
keyword, a missing `>` after the name) was repaired or whose name is not a declared tool.
`recovered_from_reasoning` is true when the calls came from a turn that ended inside its thinking.
These counters contain no tool arguments or generated text.

`request_done.constraint` carries the same constraint observation as the HTTP terminal result,
or `null` for unconstrained requests. Preparation failures and execution errors use the existing
rejection/error records rather than successful constraint outcomes.

`request_done.timings_seconds` contains `prepare`, `ttft`, `vision`, `prefill`, `decode`, and `total`
as full-precision JSON numbers. Its `speculative` object contains `backend`, `draft_window`, `rounds`,
`drafted_tokens`, `accepted_tokens`, `fallback_steps`, `accepted_per_position`, and
`rounds_by_draft_length`: with MTP on Qwen3.5-family models, entry k-1 counts the all-neural rounds
that ran draft length k (all at `--draft-tokens` unless the `adaptive-mtp` fast path chooses the
length per round); empty for other backends. `round_steps` lists every verified round in order as
`[drafted, accepted, ngram]` (`ngram` 1 for n-gram copy drafts) and `accepted_length_histogram`
counts those rounds by accepted drafts (Qwen3.5-family MTP and DFlash, Qwen3.8-Flash-Next; empty
for other models). With the request log, `output_4grams` counts the output's token 4-grams and
`output_4grams_in_prompt` those that also occur in the prompt (`null` for an output or prompt
shorter than four tokens). Rates can be derived downstream from raw token counts and seconds
instead of rounded stderr strings.
`generation.scheduling` records preemptions, snapshot/replay restores, replayed tokens and paused
time. Replay rebuilds committed state without adding new output usage.

`generation.admission` records the initial `preferred_reused_tokens`, `source_wait_seconds`, and
`fallback_reason`. Source waiting is a subset of initial queue time; selecting a source does not
itself count as a cache hit. Fallback reasons are `none`, `source_invalid`, `capacity_limit`, and
`isolated_capacity`.

`request_scheduling` records `pause_started`, `paused`, `restore_started`, `restored`,
`replay_complete`, `recovery_complete`, and a `terminal` boundary for preempted requests.
`preemption_index` identifies each pause cycle; `route` is `replay`, or `null` during pause
preparation. `steady_ns` is captured on
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
drafting](ngram.md)). With tree verification (`--draft-tree-nodes`), `tree_rounds` counts
rounds that verified a tree, `tree_side_rounds` those whose accepted path left the main chain, and
`tree_side_accepted_tokens` the drafts those paths accepted after leaving it, which a chain round
would not have accepted; `drafted_tokens` counts every verified tree column. The `speculative.ngram_archive` object reports that archive at the end of the
request: `enabled` when the server has one (`--ngram-archive-mib`), `bound` when the request was
bound to a draft session, `published` when its input and output were published into it,
`generation` as the session's latest completed generation, `sources` and `session_bytes` as the
bound session's retained source count and bytes, `total_bytes` as the whole archive's bytes, and
`sampling_seed` as the effective seed after request-domain separation for a request that named a
session, otherwise `null`.

For `server_start.memory`, `workspace.capacity_bytes` is the only physical workspace allocation.
When Vision is enabled, `vision_workspace` reports the aggregate prompt and maximum-item token
bounds plus encode peak and handoff layout/usage within that same allocation; these bytes must not
be added to `workspace.capacity_bytes`. The field is `null` when Vision is disabled.
`cuda_graph_allowance_bytes` is the CUDA Graph memory
the KV sizing reserved, and `cuda_graph_measured_bytes` the Device memory graph preparation
actually took at startup (`0` without CUDA Graphs); the startup log warns when the second exceeds
the first. When the graphs find no Device memory at all, a Qwen3.5-family engine with
`--kv-capacity auto` warns and starts once more with 1 GiB more headroom (a smaller KV capacity);
an explicit `--kv-capacity` fails startup instead.

`server_start.engine.original_int8_prefill_kernel`, `prefill_split_workspace_mib`,
`original_nvfp4_prefill_kernel` and `fast_paths` record `--use-original-int8-prefill-kernel`,
`--prefill-split-workspace-mib`, `--use-original-nvfp4-prefill-kernel` and whether
`--reproducible` was absent; `server_start.artifact.fast_paths` lists the fast paths that run
and `fast_path_configuration` the configuration they were selected for. `spec_min_p` records
`--spec-min-p` (`null`: the default 0). `speculative_draft_window` and
`speculative_draft_auto` record `--draft-tokens` (`auto` records the resolved ceiling and true). `draft_tree_nodes` (eight
entries, one per batch size, all zero for `auto`), `draft_tree_auto` and `draft_tree_paths`
record `--draft-tree-nodes` and `--draft-tree-paths`; `draft_tree_auto` is the resolved value,
false when the artifact cannot verify trees or `--draft-tokens auto`'s `adaptive-dflash` widths
run. `ngram_draft_window` and `ngram_min_match` record `--ngram-draft-tokens` (`0` disables n-gram
drafting; `auto` records the resolved window) and `--ngram-min-match`; `ngram_batch_draft_window`
is the most drafts a copy of a round of several requests holds (`0`: the window; `auto` may set
it); `ngram_archive_bytes` and `ngram_session_bytes` are the draft-archive budgets
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
`constraint_draft_wait_exposed_seconds` is the request's exposure to the batch's draft-ready wait,
already included in `device_wait_exposed_seconds`. The `throughput.host_work.constraint_draft_wait_seconds`
interval and Prometheus counter count each batch once.

`request_done.first_output_timing` freezes observations immediately before Engine publishes its
first nonempty output delta. It is `null` when no such output exists. This boundary differs from the
first accepted model token and the client's first HTTP output. Engine elapsed time begins at submit:
initial queue ends when the successful binding attempt starts, initial binding ends when the
binding is installed, and paused time includes pause preparation, waiting and restoration. The remaining
interval is resident time. Its `engine` observations describe resident Host/Device-wait exposure;
`prefill` and `replay` describe this request's submitted work. Terminal `engine_timing` still covers
the whole request.

The work `gpu_seconds` measures Text prefill stream intervals, including their MTP/DFlash work;
Vision encode, standalone bridges and exact-hit sampling fall outside that interval. GPU intervals
overlap Host submission and waits, so they are separate evidence, not additional wall-time stages.

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
`running`, `prefilling`, `decode_ready`, `waiting`, `paused`, `replaying`, `materializing`, and
`terminal_pending` fields are the Engine scheduler snapshot at the end of the interval. The JSONL
`context_cache` object reports `selections` (`root`, `prefix`, reused prompt tokens), StateImage
`state_operations` (forks, forks at initial binding, Host restores) and `partial_tail_cow_pages` as
interval deltas; `occupancy` (Device StateImages, Main and backend KV pages) and `last_selection` are
end-of-interval gauges. The separate `scheduling` object reports preemptions, restores and replayed
tokens.

`context_cache.hybrid` is present once
the cache has inserted a block or holds a snapshot. `device_blocks` (Device-resident 64-token KV blocks),
`evictable_blocks` (those Device eviction may drop now), `tree_blocks` (blocks on the Device or
Host), `snapshots`, `host_capacity_bytes`, and `host_used_bytes` are end-of-interval gauges; the rest
are interval deltas. `snapshot_hits` counts admissions that resumed from a snapshot and
`reused_tokens` the prompt tokens they reused. `blocks_inserted` counts new tree blocks,
`blocks_reattached` blocks whose existing tree entry took a request's Device pages (a Host-only
entry, or one whose unused Device copy the request's page replaced), and `blocks_duplicate`
committed blocks whose Device copy another request was using, so the request's pages stayed private
until it finished.
`taps_created` and `taps_skipped` count planned prefill snapshots published and dropped, and
`endpoints_created` end-of-answer snapshots. Publication refusals are counted too:
`taps_skipped_no_slot` (of `taps_skipped`, no Device snapshot slot was available),
`endpoints_skipped` (an end-of-answer snapshot that passed its checks but found no slot or could
not be published), `host_image_write_refusals` and `host_block_write_refusals` (snapshots and
blocks refused Host slabs, which then stay Device-only), `device_slot_evictions` (Device snapshot
slots taken from another snapshot) and `device_slot_refusals` (slot requests refused);
`free_device_snapshot_slots` is an end-of-interval gauge. `host_image_writes`, `host_block_writes`,
`host_image_restores`, `host_block_restores`, `host_write_bytes`, and `host_restore_bytes` count
Host-tier write-through and restores, and `host_write_copies` and `host_restore_copies` the copy
calls that moved those bytes (many small copies queue behind other work on the copy engine; the
Qwen3.8-Flash-Next prefix cache reports 0); `prefetched_blocks` counts the restored blocks copied for a
waiting FIFO head before its admission. `evicted_blocks` counts Device block evictions,
`host_snapshot_evictions` snapshots evicted from the Host tier, `host_dead_reclaims` Host slabs
reclaimed from KV that no snapshot can reach, and `unbacked_node_losses` Device evictions of blocks
with no Host copy, which remove them and the blocks after them from the cache. With queue holds
(the default; `--no-queue-holds` turns them off), `held_snapshots` is the end-of-interval number of
snapshots held for waiting requests, `held_device_evictions` counts Device copies of held blocks
evicted anyway (Host-backed ones first), `held_snapshot_losses` held snapshots that were lost, and
`held_host_refusals` Host snapshot or block writes skipped because only held snapshots were left to
evict. `tool_call_holds` counts holds placed after a reply ending in a tool call
(`--cache-tool-hold-secs`). `next_turn_prefill` counts the [next-turn prefill](#next-turn-prefill)'s
passes over the interval: `started`, `completed` (run to the history's end), `yielded` (stopped at a
unit boundary because a request arrived), `skipped` (the cache already held the history),
`dropped` (a request or a newer reply came before the pass started, or the history could not be
rendered), and `computed_tokens`, the prompt tokens the passes computed (also counted in the
interval's prefill tokens).

The JSONL `throughput.host_work` object is the aggregation authority: the Engine worker counts each
wall-time segment once, independent of batch size. `elapsed_seconds` contains the same five
mutually exclusive Host phases and their `total`; `device_wait_seconds` is separate.
`work_class_seconds` splits Host and Device-wait time into decode, prefill, and control classes.
`detail_subset_seconds` and `detail_invocations` expose stats-publication work; these detail values
are already contained in a top-level Host phase and must not be added to `total`.
Per-round, per-row-round, and per-invocation normalized values are
`null` when their denominator is zero. Pretty throughput contains nonzero token rates and counts,
the current running/prefill/decode-ready composition, nonzero waiting/materialization/terminal
states, average decode batch, Host-active time plus its fraction of the interval, and with NVML the
device (`gpu SM/memory MHz, power, temperature`, the interval's average power, joules per decode
token, any clock-event reason other than idle, and the power-capped and thermal fractions when
nonzero). Use JSONL for complete measurement analysis.
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
capacity returns HTTP 429 with code `server_overloaded` and a `Retry-After` header (seconds: the
`--pending-timeout-ms` window, at least 1). Token-count requests (Anthropic
`count_tokens`, Responses `input_tokens`) run the same preparation without generating; a separate
capacity of the same size bounds how many run at once, and beyond it they are rejected as
overloaded, as generation requests are. The absolute
`--pending-timeout-ms` deadline starts before preparation, covers media acquisition and Engine FIFO
waiting, and returns HTTP 503 with code `request_queue_timeout` and the same `Retry-After` if admission
does not occur in time, on streaming routes too (their response starts at admission).
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
and leaves 1 GiB of sizing headroom (`--vram-headroom-mib`); omitting the option selects `auto`
with the prefix cache and follows `--max-context` with `--no-prefix-reuse`. The
CUDA Graph driver-state allowance reserved against that budget is 64 MiB plus 4 MiB for every
decode-graph executable the engine instantiates: one per topology class of each captured family,
for every batch size up to `--max-concurrency` (DFlash and DFlash2 capture a second family when
n-gram drafting is enabled). A speculative family is a Forward/Finish pair: MTP's Finish keeps
the Forward topology classes, a DFlash Finish adds one executable per batch size. Measured on an
RTX 5090 before that split, an executable took 2.2-2.9 MiB, and up to 4.1 MiB when MTP verified a
15-wide n-gram window at batch 4-8; DFlash2 with `--max-concurrency 2` now reserves 120 MiB. The
startup log and `server_start` report both the allowance and the memory the graphs actually
used, and the log warns when the second exceeds the first.
Capacity resolves once at startup.

Before each prefill, decode or replay unit, the runtime reserves the additional pages and temporary
storage required by that unit. It first reclaims inactive cache resources when capacity is short.
If resident requests still cannot advance together, it pauses a younger request while preserving
progress for the oldest resident request. A paused request does not block fresh requests that fit
the remaining capacity. Restoration follows original request order and reserves enough space to
rebuild the saved frontier and complete one new execution unit.

A paused request keeps its committed output and protocol state. It rebuilds model state from the
deepest cached prefix plus retained input and committed tokens.
Replay does not resample or republish those tokens, but it consumes compute and can increase gaps in
the output stream. The policy and ownership rules are defined in
[Resource scheduling and context cache](maintainer/resource-scheduling-and-context-cache.md).

Compatible prefixes are reused for both text and multimodal histories unless the server starts with
`--no-prefix-reuse`. A multimodal hit additionally requires matching token types, three-axis MRoPE
positions, encoded-media digest, grid, and consumer spans. Media wholly inside a matched prefix
skips Vision execution, while new suffix media is encoded normally. The pretty completion record
shows `cache N (P%, path)` using readable path labels; JSONL retains the exact
`prefix_cache_hit_tokens` and `prefix_reuse_path` fields (`root`, `hybrid_endpoint` or
`hybrid_snapshot`). Reuse requires matching KV, recurrent state, hidden state, selected-backend
state and exact prefix identity.

The prefix cache retains content-addressed KV blocks and sparse state snapshots; its admission, tap
placement, Host tier and eviction are defined in the
[hybrid prefix cache spec](maintainer/hybrid-prefix-cache-spec.md). Completed conversation endpoints
serve direct continuations, and snapshots at the generation opener and at structural boundaries
let a later prompt that normalizes or replaces a response recompute only the changed suffix.

Changing `preserve_thinking` or reasoning effort changes the rendered prompt where applicable;
already cached exact prefixes remain usable. An appended mid-conversation system message is an
ordinary suffix. Modifying, removing or moving a historical message changes the prefix and may
require an earlier snapshot or a root prefill.

### Next-turn prefill

A conversation's next request does not always render the previous reply as the tokens the model
generated, so the endpoint above cannot serve it: a template that drops earlier reasoning once a new
user turn arrives (Qwen3.5's and Qwen3.6's by default, any request with `preserve_thinking: false`)
re-renders the whole tail since the previous user turn without it, and on `--reproducible` (and any
configuration without the `free-cache-resume` fast path) no decode-computed state is reused at all,
so every next turn recomputes the previous reply. After each reply that ended on its own (a stop
token or stop string; not one cut by its limit or cancelled), the server renders the conversation
with that reply appended as its next request will (the request's template options; a tool result
follows a reply that calls tools, a user turn any other), and once the server has had nothing to run
for `--cache-next-turn-ms` (default 150 ms), prefills that history into the prefix cache as an
internal request. The next request then resumes at the history's end (on `--reproducible`, at the
grid position at or before it, so its output still equals an uncached run's).

- It never delays a request by more than one unit: a request that arrives before the pass starts
  drops it, and one that arrives while it runs makes it stop at its next prefill unit, keeping the
  blocks and snapshots it had published and the state it reached (Qwen3.8-Flash-Next: except inside
  a layer-walk span, which a stopped pass abandons).
- It never changes output: the next request reuses it only where its tokens are identical. A client
  that sends the reply back differently (reordered tool arguments, dropped reasoning in a template
  that keeps it) simply misses it.
- A history the cache already holds (the usual agent loop with a template that keeps reasoning: the
  endpoint already covers the reply) is skipped without running. Only the latest finished reply is
  prefilled; text conversations only (a request with media is not followed).
- It costs GPU time when no next request comes. `--cache-next-turn-ms 0` turns it off. The
  `throughput` record's `context_cache.hybrid.next_turn_prefill` object counts the passes.
- Qwen3.8-Flash-Next: off by default. Its default route already resumes nearly every agentic turn at
  the reply's end (94 % of continuing turns in a Dense8 replay), and on `--reproducible` a turn resumed
  from the pass's grid snapshot did not equal an uncached run, so `--reproducible` and `--lossless`
  refuse a non-zero `--cache-next-turn-ms` there.

Speculative backends preserve protocol output shapes, stop behavior, and usage accounting. If a stop
truncates a multi-token MTP, DFlash or DFlash2 round, the Engine commits the exact accepted target prefix so
a following compatible turn can reuse it. Output-limit and context-capacity finishes map to
`length`/ `max_tokens`; ordinary model or string stops map to `stop`/ `end_turn`.

Function tools are rendered into the model prompt and generated calls are parsed into protocol
responses. Infernix executes no client tools: the only tool it runs is the search of
[deferred tools](#deferred-tools-and-tool-search). It does not enforce tool-argument schemas through
constrained decoding.

Prompt-token usage includes chat-template and expanded media tokens. Generated-token usage comes
from accepted output token IDs, including a stop token whose decoded text may be withheld.
