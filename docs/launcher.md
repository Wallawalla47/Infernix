# Infernix Launcher

`infernix-launcher` is the front end for running Infernix: a web interface to download, configure,
run and chat with models, and one OpenAI- and Anthropic-compatible endpoint that coding tools point
at. When a request names a model that is not loaded, the launcher lets the running requests finish,
stops the current `infernix-serve`, starts the requested model with its saved configuration and
forwards the request to it, as LM Studio does.

The launcher runs no inference itself. It supervises one `infernix-serve` process at a time on a
private loopback port and relays requests and streamed answers to it.

## Start

Windows: run `infernix-launcher.exe` from the folder that holds `infernix-serve.exe` (the release zip
and `build-windows\apps\Release` both do). Linux: `build/apps/infernix-launcher`.

```bat
infernix-launcher.exe
```

The console prints the addresses and the browser opens the web UI:

| Address | Use |
|---|---|
| `http://127.0.0.1:1234` | the web UI |
| `http://127.0.0.1:1234/v1` | OpenAI SDKs and tools (Chat Completions and Responses) |
| `http://127.0.0.1:1234` | Anthropic SDKs and Claude Code (they append `/v1/messages`) |

| Option | Meaning |
|---|---|
| `--host H`, `--port N` | listen address for this run (default: the saved settings, `127.0.0.1:1234`) |
| `--data-dir DIR` | the folder of the launcher's memory file and logs (default below) |
| `--serve FILE` | the `infernix-serve` to run (default: the one beside the launcher) |
| `--no-browser` | do not open the web UI |
| `--ui-dir DIR` | serve the web UI from a folder (web UI development) |

Ctrl+C (or closing the console) stops the loaded model in order, so it saves its prefix cache and
expert-cache state, pauses running downloads and exits. Only the launcher's own events and server
errors appear in its console; the web UI shows the server's full log. On Windows the server shares
the launcher's console, which carries the orderly stop; started without one (by a program or a
shortcut that gives it none), the launcher makes a hidden console of its own.

## What it remembers

Everything persists between runs in one file beside the launcher, `infernix-launcher.json`, so the
launcher is portable and its memory travels with its folder:

| In `infernix-launcher.json` | |
|---|---|
| `settings` | launcher settings, model folders, model routes, API keys (their SHA-256, never the keys) and the Hugging Face token |
| `models` | each configuration's model id, listing, launch parameters, environment and idle unload, keyed by the artifact's id (so moving or renaming the files keeps them; a named configuration is `<artifact id>@<8 hex>`) |
| `last_model` | the last configuration loaded |
| `downloads` | download jobs, resumed after a restart, with the revision and file hashes they fetched |

Beside it are the logs ([Event log](#event-log); **Settings → Logging** can move them to another
folder), `infernix-launcher-usage.json` ([Usage](#usage)), `chats/` (one file per chat, with its
images), `launcher-prefix-cache/`, each model's saved prefix cache when **Keep each model's prefix
cache** is on (several GB per model), and the scratch folders `plans/` and `updates/`.

The memory belongs to that copy of the launcher, wherever it is started from: unzipping a new
release into another folder starts afresh, so copy `infernix-launcher.json` across (or start the new
launcher with `--data-dir` pointing at the old folder). A launcher in a folder it cannot write to
(such as `C:\Program Files`) keeps its memory in `%LOCALAPPDATA%\Infernix\launcher` instead (Linux
`$XDG_CONFIG_HOME/infernix/launcher` or `~/.config/infernix/launcher`) and says so at startup. The
file holds the Hugging Face token in plain text, like the folder's other files; API keys are shown
once when created and only their hash is kept.

A default configuration whose parameters were never changed follows the launcher's defaults, which
pick up a drafter or n-gram volume added later; once changed, the saved parameters are used until
**Restore defaults**.

## Dashboard

One page, top to bottom:

- **Status**: the loaded model, its state (idle, starting with the latest startup line, ready,
  stopping, failed with the reason), uptime, and a pending switch with the requests it waits for.
- **Inference gauges**: speedometer dials for decode tok/s (live while generating), prefill tok/s,
  time to first token, prompt-cache hit rate, and, when the model reports them, the VRAM expert-cache
  hit rate and drafter acceptance, each with its last-ten and session values.
- **System gauges**: CPU use and temperature, RAM, GPU use, temperature and power, VRAM, PCIe traffic
  (the expert and n-gram traffic of Qwen3.8-Flash-Next shows here) and the server process's memory,
  each with a 10-minute history. Everything comes from the operating system and the NVIDIA driver; no other
  software is needed. CPU temperature on Windows is the CPU package sensor from HWiNFO or Core Temp
  when one of them runs (HWiNFO needs *Shared Memory Support* on), otherwise the ACPI thermal zone
  Windows reports, a motherboard sensor rather than the package; a PC that exposes neither shows no
  temperature. Linux reads the kernel's hwmon sensors (coretemp, k10temp).
- **Inference statistics**: the same figures as `infernix-serve`'s console panel, over the last ten
  requests and the whole session of the loaded model: requests, time to first token, prompt-cache hit
  rate, prefill and decode tok/s, mean decode batch, drafter acceptance and accepted tokens per
  round, n-gram acceptance, and the VRAM expert-cache hit rate (Qwen3.8-Flash-Next and the
  Qwen3.5-122B-A10B). They come from `session_stats` in the server's `/metrics?format=json`
  ([serving](serving.md#json-representation)); with an older `infernix-serve` the UI derives them
  from its last 128 requests.
- **Requests**: per-request TTFT and decode speed, and the coding tools' requests as the launcher
  routed them (by pin, route, name or as the loaded model), with the client and API key and, while
  a long prompt is processed, its progress (tokens processed of the prompt, from the cache and
  computed).
- **Console**, at the bottom: the server's log, live, in a resizable box with level filters, search,
  follow and download (the last server's log stays after it stops). **Logs** shows it full height.

## Model folders and detection

Add folders under **Settings → Model folders** (or **Models → Add folder**). Each is scanned two
folder levels deep, reading file headers only:

- **Models**: `.infernix` and `.ninfer` v3 artifacts with a text component, recognised by family
  (Qwen3.8-Flash-Next, Qwen3.8-27B, Qwen3.6-27B, Qwen3.6-35B-A3B, Qwen3.5-122B-A10B). The first part of
  a numbered set (`-00001-of-00003`) stands for the set; missing or incomplete parts are reported and
  such a model cannot load. Copies of one artifact (same id) appear once, with every location.
- **DFlash2 drafters**: companion artifacts without a text component, such as
  `Qwen3.8-Flash-Next-DFlash2-Infernix.infernix`, matched to the models whose architecture and shape
  they were trained for.
- **N-gram volumes**: `.ngram` files of Qwen3.8-Flash-Next, matched to the models that can read them:
  a model's own table first, then FP8, NVFP4, IQ4_NL and BF16 (an FP8 table without a recorded scale
  serves only its own artifact).

Folders are rescanned when they change in the launcher, after a download and every minute.

**Models** shows each folder's size, the free space on its drive, and each model's, drafter's and
volume's size on disk. **Delete files** removes a model's artifact (every part), a drafter or an
n-gram volume from the disk after you type its file name. A model's deletion takes the files at
the location shown (every part, its `.conversion.json` and `.expert-state`) and its
configurations; files a running server uses cannot be deleted.

## Configure a model

Each model's page edits its launch parameters, the `infernix-serve` options after the artifact
path. The form and the **Launch parameters** text are two views of the same list: a control changes
its option in the text, an edit to the text moves the controls, and options without a control are
kept. The **DFlash2 drafter** and **N-gram volume** pickers (Qwen3.8-Flash-Next) list the compatible
files found and write `--draft-model` and `--ngram-volume`. The full command line is previewed with
warnings (an option this `infernix-serve` does not have, a missing file, a Flash-Next model without
a volume). Every option of the installed `infernix-serve --help` is listed in the reference drawer.

The **model id** is the name tools see in `/v1/models` and send back; it defaults to the artifact's
file name without its part numbering, and any other name can be set (two configurations cannot
share one). A configuration can be hidden from the list.

**Configurations.** An artifact has its default configuration and any number of named ones (**New
configuration**, copied from the one on screen), each with its own model id (by default the
artifact's id with the name appended, `Qwen3.8-27B-long-context`), launch parameters, environment
and idle unload, so one artifact can be served as several models (a long-context one and a
two-request one, say). Loading another configuration of the loaded artifact restarts the server
with its parameters. A named configuration can be renamed or deleted; the default one can only be
restored to the defaults.

**Environment.** Variables set for the configuration's `infernix-serve` only (`CUDA_VISIBLE_DEVICES`,
`INFERNIX_*` switches), on top of the launcher's own environment.

**Import and export.** **Import script** reads a `.bat`, `.cmd` or `.sh` that starts `infernix-serve`
(pasted or dropped): its `set`/`export` variables become the environment and the options after the
artifact become the launch parameters (the ones the launcher sets itself are dropped, and options
this `infernix-serve` lacks are named), and its `--model-id` the model id. When the artifact is in a
model folder (by path, else by file name), the import replaces one of that model's configurations
or becomes a new named one; otherwise the launcher says which artifact it needs. **Export** writes the
configuration as a `.bat` (Windows) or `.sh` that runs `infernix-serve` directly, with its model id.

**Will it fit?** asks the installed `infernix-serve` to size the configuration without loading it
(`infernix-serve --plan`): the VRAM it would take (weights, KV cache and the context it holds,
workspace, CUDA graphs, expert cache, headroom), the RAM (experts, pinned prefix cache, n-gram
table, what stays free), and the largest `--max-context` that fits. A configuration that does not
fit gets the same suggestions as a failed load ([Watchdog](#watchdog)). The plan reads the GPU as it
is now, so a loaded model's memory counts against it; unload first to plan for an empty GPU.
Turning on **Vision** checks at once: the vision tower takes VRAM a 27B's default context was sized
without, so the default context no longer fits beside it (on a 32 GB GPU the official NVFP4 27B
fits 177,856 tokens with `--vision` against 222,272 without).

**Speculative decoding.** **Off** writes `--spec off` when the server lists `off` among its `--spec`
values (such a server speculates when `--spec` is absent, and the field then shows **Auto**);
with an older server, which decodes plainly without the flag and refuses `off`, Off removes it.

**Idle unload.** Minutes without requests after which this configuration's server is stopped (the
next request loads it again); **Settings → Idle unload** is the default for configurations that do
not set their own, and 0 keeps it loaded. The dashboard shows when the loaded model will unload.

Defaults per family (README "Recommended settings per model"):

| Family | Default launch parameters |
|---|---|
| all | `--max-concurrency 2 --kv-dtype int8 --preserve-thinking --tolerant-tool-calls --pending-timeout-ms 900000`, and `--spec auto --draft-tokens auto` when a drafter is available |
| Qwen3.8-Flash-Next | `--max-context 262144 --lm-head-draft`, the best DFlash2 drafter found (`--draft-model`) and n-gram volume (`--ngram-volume`, unless `<artifact>.ngram` exists) |
| Qwen3.8-27B | `--lm-head-draft --ngram-draft-tokens auto --prefill-chunk 4096`, `--host-context-mib` 55 % of RAM up to 52000, and the `--max-context` that fits beside the artifact in the GPU's memory (below) |
| Qwen3.6-27B, Qwen3.6-35B-A3B | `--lm-head-draft` where the artifact has a proposal head; the 27B's context as for Qwen3.8-27B |
| Qwen3.5-122B-A10B | `--max-context 262144 --host-context-mib 4096` (the experts take the RAM) |

A 27B keeps all of its KV cache on the GPU, so its context is what the GPU holds beside the
weights: the launcher reckons 2.03 GB plus 33,792 bytes per token of context (measured with INT8 KV,
two requests and both draft kinds), the Engine's 1 GiB of headroom and 1 GB for the CUDA context and
the display, rounded down to 4,096 tokens and at most 240,000. On a 32 GB RTX 5090 that is 188,416
for the 23.7 GB official NVFP4 artifact (it starts at up to about 196,000 there; 240,000 needs
1.5 GB more) and 240,000 for artifacts of 21.6 GB or less. Raise it with `--vram-headroom-mib 0`
when nothing else uses the GPU.

The launcher sets `--host`, `--port`, `--model-id` (the model id), `--any-model-name`,
`--log-stats-panel off` and `--log-colours off` itself; they are removed from saved parameters.

## Download models

**Discover** lists the published Infernix conversions and NInfer's official artifacts. **Download**
lists the repository's files with their sizes (the model's parts are selected) and suggests what the
model also needs, unless a compatible one is already in a model folder: Qwen3.8-Flash-Next's DFlash2
drafter and an n-gram volume (the recommended table preselected). Choose the destination folder with
the folder browser; by default each repository gets its own subfolder. Any other repository can be
entered by its `owner/name`.

Downloads run one repository at a time, each file in 64 MiB pieces over several connections
(**Settings → Downloads**), into `<file>.part`. Pausing, a dropped connection or a launcher restart
resumes where it stopped. A finished file is checked against its SHA-256 from Hugging Face and
renamed into place; the destination becomes a model folder and is scanned. A Hugging Face token
(settings, `HF_TOKEN`, or the `huggingface-cli login` token file) is sent for gated repositories.

## Connect coding tools

Point a tool at the launcher instead of at `infernix-serve`. **Connect** shows the exact addresses and
the model list as tools see it, with snippets for Claude Code, Codex, Zed, Qwen Code, Hermes and the
OpenAI and Anthropic SDKs.

- `GET /v1/models` (also `/models`) lists the configurations not hidden, by model id, in the fields
  other servers use so each tool finds the ones it reads:
  - the OpenAI list fields; the Anthropic ones (`type`, `display_name`, `created_at`, `has_more`,
    `first_id`, `last_id`), paged by `limit` (default 20, at most 1,000), `after_id` and `before_id`
    when the request carries `anthropic-version`;
  - `name` and `description` (family, quantization, context, vision, whether it is loaded);
  - its context (`--max-context`) as `context_length`, `context_window`, `max_model_len`,
    `max_context_length`, `max_input_tokens` and `meta.n_ctx`, and the default output limit as
    `max_tokens`;
  - whether it is loaded, as `state` (`loaded`, `loading`, `not-loaded`; LM Studio's) and
    `status.value` (`loaded`, `loading`, `unloaded`; llama.cpp's and llama-swap's);
  - `architecture` with its `input_modalities` (`image` when started with `--vision`) and
    `output_modalities`, `capabilities`, and `supported_parameters` (the request fields
    `infernix-serve` honours, in OpenRouter's names); the model class is `meta.architecture`.

  `GET /v1/models/{id}` returns one entry, or `model_not_found` (Anthropic: `not_found_error`).
  The launcher does not answer `/api/tags`, `/api/v1/models` or `/version`, nor `/props` unless
  [llama.cpp endpoints](#llamacpp-endpoints) are on: tools such as Hermes tell Ollama, LM Studio,
  llama.cpp and vLLM apart by them.
- **Claude Code's model picker**: with `CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1`, Claude Code
  asks `/v1/models` at startup and lists only ids that contain `claude` or `anthropic`. To Claude Code
  (its `claude-cli/` user agent) the launcher lists each model as `<model id>:claude-code`, which
  requests may also name; a suffix rather than a `claude-` prefix keeps
  `CLAUDE_CODE_MAX_CONTEXT_TOKENS` (Claude Code otherwise assumes 200K) in effect. **Connect**'s
  snippet sets both, and Codex's sets `model_context_window` (Codex otherwise assumes 272K).
- Generation routes are forwarded unchanged, streaming included: `POST /v1/chat/completions`,
  `/v1/responses`, `/v1/responses/input_tokens`, `/v1/responses/compact`, `/v1/messages`,
  `/v1/messages/count_tokens` and `/v1/decide`; `GET`/`DELETE /v1/responses/{id}`,
  `GET /v1/responses/{id}/input_items` and `POST /v1/responses/{id}/cancel` go to the loaded model.
  `infernix-serve` refuses compaction (`compaction_not_supported`) and cancellation (it has no
  background Responses), and clients get those answers. The doubled `/v1/v1/...` prefix is accepted
  as by `infernix-serve`.
- **Which model serves a request** (**Settings → Model switching**), in this order:
  1. **Pin the loaded model** (off): every request goes to the loaded (or loading) model, whatever
     it names, so a tool that names another model cannot unload it. With nothing loaded, the rules
     below apply.
  2. **Model routes**: patterns over the requested name, case-insensitive with `*` wildcards, each
     to a configuration or to the loaded model (else the default model); the first match wins. For
     example `claude-*` → `Qwen3.8-Flash-Next`, so Claude Code's own model names need no setting
     in the tool. A route to a configuration that no longer exists is skipped.
  3. The name itself: a configuration's model id or id (case-insensitive), else an artifact's name,
     file name or default model id (its default configuration).
  4. An unknown name goes to the loaded model, else the default model (like `infernix-serve
     --any-model-name`), or is refused with `model_not_found` when the setting is **Reject**.

  The dashboard and the event log show which rule chose the model. A request that names no model
  (the Chat page's **Loaded model** sends none) follows the same rules and reaches the server with
  the chosen configuration's model id, since `infernix-serve` requires one.
- **Switching**: requests queue in arrival order. While requests for another model wait, the loaded
  model takes no new requests; once its running requests finish (or after the drain timeout, which
  cancels them) it stops and the next model starts. They fail with HTTP 503 (Anthropic: 529
  `overloaded_error`) if the model does not load within the load timeout (`Retry-After: 10`: the load
  goes on). **Load** in the UI queues the same way, or switches at once when asked to.
- **After a failed load** the model is held for 2 minutes: its requests are refused at once with the
  load's error and `Retry-After` in whole seconds (over 60, so Claude Code reports the error instead
  of retrying), rather than each retry starting another load that fails the same way (coding tools
  retry 5xx answers: Claude Code up to 10 times). **Load** in the UI, a watchdog restart, a
  changed configuration or changed Settings end the hold at once.
- **While a model loads** (a large model takes a minute or more), a streaming request that has to
  wait is answered at once with HTTP 200 and an event stream that carries a keep-alive every 5
  seconds (an SSE comment `: loading <model id>`; for Anthropic clients a `ping` event), so clients
  and proxies with idle timeouts do not give up; once the model is ready the server's stream follows
  in the same response. A load that fails, or an error answer from the server, then arrives as one
  error event in the client's protocol (OpenAI `{"error": ...}`, Anthropic `event: error`) instead of
  an HTTP error status. A non-streaming request receives no bytes until its model is ready.
- **Pooled connections** stay open 6 minutes after a response (advertised as `Keep-Alive:
  timeout=360`, closed 10 s after that), as `infernix-serve`'s `--keep-alive-timeout-sec` default
  ([serving](serving.md)): longer than Bun's 5-minute `fetch` idle timeout, so opencode, whose
  client ignores the header, and clients that follow it both close a connection before the
  launcher does, rather than writing a request onto one it is closing. Each forwarded request takes
  its own connection to the server.
- Every forwarded request carries `X-Infernix-Request-Label: launcher-<n>` (unless the client sent
  its own label), which ties the server's per-request figures to the launcher's record, and the
  answer carries `X-Infernix-Request-Id: <n>`.
- Stored Responses (`previous_response_id`, `GET /v1/responses/{id}`) live in the running server and
  end with it: after a switch, earlier response ids are unknown.

### API keys and remote access

**Settings → API keys** creates named keys (`inx-` and 32 letters and digits, shown once; the
launcher keeps their SHA-256), each **API** (the `/v1` routes) or **Admin** (also the web UI and its
management API from other computers), with when it was last used. With no key, `/v1` is open; with
any key, `/v1` needs one, as `Authorization: Bearer` or `x-api-key`. The event log and the usage page
name the key each request came with. Listening on another address than loopback without a key logs
a warning.

The web UI and its management API (`/launcher/api`) answer this computer without a key. From
another computer they need **Allow remote management** on and an Admin key: the UI asks for it once
and keeps a sign-in cookie (HttpOnly, SameSite=Strict, 30 days; signing out or restarting the
launcher ends it). The UI's own chat uses the endpoint without a key.

Other web pages cannot use the management API, whatever the CORS setting: it refuses requests whose
`Origin` is neither the launcher's own page nor a loopback one and, unless they carry an Admin key,
requests whose `Host` is a domain name rather than an IP address or `localhost` (against DNS
rebinding). Open the UI as `http://127.0.0.1:1234` or `http://localhost:1234` (from another computer: by its
IP address).

The launcher does not answer LM Studio's or Ollama's own APIs (`/api/v1/models`, `/api/tags`), nor
llama.cpp's unless asked ([llama.cpp endpoints](#llamacpp-endpoints)), so tools that probe them
treat it as a generic OpenAI-compatible server.

### llama.cpp endpoints

**Settings → Launcher server → llama.cpp endpoints** (off) answers the router API of
`llama-server`, for tools with a llama.cpp provider: Zed's lists every launcher model with its
context, vision and tool support and follows loads as they happen (**Connect** has its settings).
With it on, tools that tell servers apart by these paths (Hermes probes `/props`) treat the
launcher as llama.cpp; with it off the paths are not answered. It applies at once, and the routes
need an API key like `/v1` when any exists.

| Route | Answer |
|---|---|
| `GET /props`, `/v1/props` | `?model=<id>`: that configuration as `llama-server` describes its loaded model, without loading it: `default_generation_settings.n_ctx` (its `--max-context`), `default_generation_settings.speculative` (its `--spec`; without one, whether the server speculates by default with a drafter the artifact bundles or `--draft-model` names), `total_slots` (`--max-concurrency`), `modalities.vision` (`--vision`), `chat_template_caps` (tool calls; `supports_preserve_reasoning` with `--preserve-thinking`), `model_path`, `model_alias`, `is_sleeping`. Without `model`, the loaded configuration, else the default model, else `llama-server`'s router answer (`n_ctx` 0). An unknown id is HTTP 400 `model '<id>' not found` |
| `POST /models/load` `{"model": id}` | queues the load like a request (the loaded model's running requests finish first) and answers `{"success": true}` at once; 404 `model is not found`, 400 `model is already running` |
| `POST /models/unload` `{"model": id}` | stops that model when it is the loaded one; 400 `model is not found` or `model is not running` |
| `GET /models/sse` | an event stream of `data: {"model", "event", "data"}` lines: `model_status` with `status` `loading`, `loaded` or `unloaded` (a failed load: `unloaded` with a non-zero `exit_code`), and `models_reload` for the model `*` when the list or a model id changes; a `: ping` comment every 15 s while idle |

`GET /v1/models` (and `/models`) already carries `status.value`, `architecture.input_modalities`
and `meta.n_ctx`, which llama.cpp clients read. Generation goes through the `/v1` routes;
`llama-server`'s own `/completion`, `/infill`, `/tokenize` and `/slots` have no counterpart in
`infernix-serve`.

## Chat

**Chat** talks to a model through the launcher's own `/v1/chat/completions`, so it loads and
switches models like any client. The chats are listed in a sidebar (a sheet on narrow screens):
new, rename, delete; each is kept in `chats/` with its model, options and messages and reopens
where it was. A chat's model is a configuration, or **Loaded model** (whatever is loaded).

- Replies stream as Markdown (tables, task lists, maths, highlighted code with a copy button); a
  model's thinking (`reasoning_content`) shows as a collapsible block. **Stop** ends the request;
  a reply can be regenerated and a message edited and sent again.
- Images (PNG, JPEG, WebP, GIF, up to 20 MB each) are attached, pasted or dropped onto the chat and
  sent as `image_url` parts. A configuration takes images only when its parameters include
  `--vision` (and the artifact has a vision tower); otherwise the chat says so before sending.
- **Options**: system prompt, temperature, maximum tokens and thinking
  (`chat_template_kwargs.enable_thinking`).
- Under the prompt box, the last reply's figures, live while it streams: time to first token (as
  the page saw it, a model load included), prefill tok/s, decode tok/s and the prefix-cache hit
  rate (cached tokens of the prompt), from the server's `timings`, then its usage, then its prompt
  progress. While a long prompt is processed, the same row shows how far it has got. Each reply
  keeps its figures in a tooltip.

A reply keeps streaming when you switch to another chat or page and is saved when it ends; a page
closed mid-reply keeps the chat without that reply.

## Usage

**Usage** totals the requests the launcher served, per day, model and client (the API key's name,
`launcher chat` for the Chat page, else the client's address): requests, failures, prompt tokens and those from the prefix cache,
computed prefill tokens, output tokens, and prefill and decode time, with a chart and a table over
the last 7, 30 or 90 days or this month. The figures are the server's own for each request, matched to the
launcher's record by its request label, and are kept in `infernix-launcher-usage.json` (written at
most every 30 seconds and when the launcher stops). Requests a server ended before they reached its
records (a crash) are not counted.

## Updates

**Settings → Updates**.

**Models**: for every model downloaded through the launcher, the launcher compares the revision and
file hashes it downloaded with the repository on Hugging Face (two minutes after starting, then
every 6 hours, or **Check now**) and lists the changed files with their size. **Update** downloads
just those files over the old ones (a model loaded from that folder must be unloaded first); a
model the launcher did not download is not checked.

**Infernix** (Windows): the launcher compares its own build with the latest release on GitHub
(`Wallawalla47/Infernix`) and shows the release notes. **Download** fetches the release zip and
checks its SHA-256; **Install and restart** stops the loaded model in order, exits, copies the
release over the launcher's folder (adding and replacing files; `infernix-launcher.json`, logs,
chats and anything else of yours stay) and starts the new launcher with the same options. A
launcher in a folder it cannot write to cannot update itself. On Linux, update from the source.

## Watchdog

While a model serves, the launcher watches its `infernix-serve` (**Settings → Watchdog**):

| It notices | When | Then |
|---|---|---|
| a crash | the server exits without being asked (the exit status is named: access violation, out of memory, killed by the system...; on Linux a SIGKILL names both memory-pressure killers, the kernel's (`journalctl -k`) and systemd-oomd (`journalctl -u systemd-oomd`), which leaves no kernel log line) | restarts the model |
| a missing DLL | Windows cannot start `infernix-serve`: a DLL it needs is missing, from another version, damaged or failed to initialize (status 0xC0000135, 0xC0000139, 0xC000007B or 0xC0000142) | reports it with the release to take the DLLs from (`https://github.com/Wallawalla47/Infernix/releases/latest`; every DLL in its zip belongs beside `infernix-serve.exe`, and the Visual C++ Redistributable supplies the rest) |
| a lock-up | requests are in the server or waiting on it, and it answers nothing for the hang timeout (60 s; at least three checks a second apart) | asks it to stop, ends it after 15 s, restarts the model |
| a stall | requests are in the server, yet no prompt or output token is processed for the stall timeout (300 s; the progress counters move with every prompt chunk and token, so long prompts do not count) | the same |
| a load that does not fit | the server's own sizing error (the KV cache beside the weights, a GPU allocation, pinning RAM), or the computer's free RAM below the RAM floor (2 GiB) for three seconds while it loads (the computer would otherwise page heavily) | ends or reports the load, names the shortfall, and suggests changes |
| a port in use | `infernix-serve` binds its port exclusively and exits before loading with `cannot bind HOST:PORT` when another program holds it (or Windows reserves it) | with the automatic **Server port**, starts it again on another free port (up to 3 tries, noted in the event log only: another program can take a port in the moment between the launcher picking it and the server binding it); with a fixed port, or after 3 tries, reports it with advice: the automatic port, the program holding the port, Windows' reserved ranges |

A server sitting idle is never ended, however long it waits for requests. One that stops answering
while idle (just after the computer wakes from sleep, say) is noted once in the event log and left
running; if a request then waits on it and it still does not answer, it is treated as a lock-up at
once. Unloading an idle model after a while is a separate choice, **Idle unload** (off).

A restart loads the same model with the same parameters, at most 3 times in 15 minutes; after that,
and when automatic restarts are off, the next request loads it again. Requests running when the
server failed end with an error (their streams stop); requests waiting for the model are served by
the restarted one. Each incident shows as a banner on the dashboard and a notification, and goes into
the [event log](#event-log). With the watchdog off, a crash is still reported, nothing is restarted
and nothing ends a lock-up, a stall or a load.

For a load that did not fit, the banner lists changes worked out from the server's figures and the
model's parameters, most effective first: for VRAM, the largest context that fits by the server's
own plan (`infernix-serve --plan`, run once the failed server's memory is free; the suggestions show
a second or so after the error), giving up the 1 GiB of VRAM headroom, one request at a time, INT8 KV
when it was BF16; for RAM, more RAM left free with automatic expert sizing (`--ram-free-mib`,
`--expert-ram-mib auto`) and a smaller Host prefix cache (`--host-context-mib`); and how much other
programs hold. **Apply** saves a change into the model's parameters; **Apply and load** also loads it.

A GPU that fills up is normal for models with offloaded experts (their expert cache takes the VRAM
the rest leaves), so the launcher does not judge VRAM by how full the GPU is; the server sizes its
own allocations and reports a shortfall itself.

## Event log

`infernix-launcher.log`, beside the launcher (or in the folder **Settings → Logging** names), one
line per event in the server's record format:

```text
2026-10-31 07:14:56.433  ERROR  watchdog  Qwen3.8-27B: infernix-serve stopped responding: no answer for 61 s with requests waiting on it (Read); restarting it (1 of 3 in 15 minutes)
2026-10-31 07:14:58.120  INFO   api       POST /v1/chat/completions 200 in 2.1 s (first event 128 ms) · Qwen3.8-27B · stream · 127.0.0.1
```

It records the launcher starting and stopping, model loads, stops, failures and the watchdog's
incidents, downloads, settings and configuration changes, and every API request by its metadata:
client address and program (User-Agent), the API key's name, method and route, the model asked for,
the configuration that served it and the rule that chose it, streaming or not, status, duration,
time to the first event, and bytes in and out
(`GET /v1/models`, which tools poll, at debug level). It holds no prompts or answers. **Logs →
Launcher events** shows this session's records with filters and the full details of each.

**Record request and response contents** (off) also writes every proxied request's body and the
response (a stream's events as they were sent) to `infernix-launcher-requests.jsonl`, one JSON
object per request, each body cut at 16 MiB. It holds prompts and answers in full and grows quickly.

Both files rotate by size (16 MiB and 100 MiB) into `FILE.1` .. `FILE.5`. The server's own console,
one file per load, is kept in `serve-logs/` in the same folder (the newest 20). Moving the log folder
applies at once to later records; a folder that cannot be written leaves the records in memory and
says so.

## Settings

| Setting | Default | Meaning |
|---|---|---|
| Host, port | `127.0.0.1`, `1234` | listen address (applies after a restart) |
| API keys | none | named keys; with any, `/v1` needs one ([API keys](#api-keys-and-remote-access)) |
| Allow remote management | off | the web UI from other computers, with an Admin key |
| CORS | on | browser access headers on every answer (applies after a restart) |
| llama.cpp endpoints | off | `llama-server`'s router API for llama.cpp providers ([llama.cpp endpoints](#llamacpp-endpoints)) |
| `infernix-serve` | beside the launcher | the server executable; its `--help` drives the option reference |
| Server port | automatic | the loopback port of the managed server |
| Drain timeout | 120 s | how long running requests may finish before a switch cancels them |
| Load timeout | 1200 s | how long a model may take to become ready |
| Stop timeout | 180 s | how long an orderly stop may take (saving a large prefix cache) before the process is ended |
| Idle unload | never | unload after this many minutes without requests (a configuration can set its own) |
| Pin the loaded model | off | every request goes to the loaded model ([routing](#connect-coding-tools)) |
| Model routes | none | requested-name patterns to configurations |
| Unknown model names | use the loaded model | or reject them |
| Default model, load on start | none, off | the model for unknown names when none is loaded; load it (else the last one) at startup |
| Keep each model's prefix cache | off | adds `--prefix-cache-file` in the data folder for every model |
| Download connections, verify | 8, on | parallel connections per file; SHA-256 check |
| Log folder | the launcher's folder | the event log, the request-content log and `serve-logs/` (applies at once) |
| Record request and response contents | off | `infernix-launcher-requests.jsonl` ([Event log](#event-log)) |
| Watchdog, restart automatically | on, on | [Watchdog](#watchdog) |
| Hang timeout, stall timeout | 60 s, 300 s | no answer; no progress with requests running (0: no stall check) |
| RAM floor while loading | 2048 MiB | end a load when free RAM stays below it (0: off) |

## Web UI development

The UI is React, TypeScript, Vite, Tailwind CSS, Radix (shadcn/ui-style components), Recharts and
TanStack Query, in [`apps/launcher/web`](../apps/launcher/web). Its build is one gzip-compressed HTML
file, [`apps/launcher/assets/index.html.gz`](../apps/launcher/assets), committed so that building
Infernix needs no Node.js; CMake embeds it into `infernix-launcher` with the download catalog
([`apps/launcher/catalog.json`](../apps/launcher/catalog.json)). After changing the UI:

```bash
cd apps/launcher/web
npm ci
npm run build
```

`npm run dev` serves the UI with hot reload and forwards `/launcher`, `/v1` and `/health` to a
launcher on port 1234.

Without a GPU, `infernix_launcher_fake_serve` (built with the tests) stands in for `infernix-serve`:
start the launcher with `--serve <build>/tests/Release/infernix_launcher_fake_serve` (any artifact in
the model folders; only its header is read) and requests get plausible figures on the dashboard,
the chat (thinking, prompt progress, timings) and the usage page, and **Will it fit?** gets a plan
that fits up to 225,280 tokens of context. It crashes, locks up or stalls on request (`POST
/fake/crash`, `/fake/hang`, `/fake/stall` on its own port) and fails or overfills its load with
`--fake-fail-load TEXT` and `--fake-hog-mib N` in the model's parameters.
[`tools/launcher/launcher_check.py`](../tools/launcher/launcher_check.py) runs a launcher against it
and checks the features (configurations, model ids, environment, script export and import, the fit
plan, keep-alive streams, routes and the pin, prompt progress, usage, API keys and sign-in, chats,
idle unload) and the watchdog, the event log and the request log, and, on Windows, that a launcher
started on a console without a window still stops its server in order.
