# Infernix Launcher

`infernix-launcher` is the front end for running Infernix: a web interface to download, configure,
run and chat with models, and one OpenAI- and Anthropic-compatible endpoint that coding tools point
at. When a request names a model that is not loaded, the launcher lets the running requests finish,
stops the current `infernix-serve`, starts the requested model with its saved configuration and
forwards the request to it, as LM Studio does.

The launcher runs no inference itself. It supervises one `infernix-serve` process at a time on a
private loopback port and relays requests and streamed answers to it. The server never outlives
the launcher: if the launcher ends without stopping it (a crash, a forced kill), the system ends
the server too, on Windows and Linux, so it does not keep holding the GPU. The one other program it
runs is the ablation-sidecar fitter of the [Ablation](#ablation) page, which starts engines of its own
and holds the GPU while it runs.

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
| `--ablate-tool DIR` | the sidecar fitter the [Ablation](#ablation) page runs: the folder holding `ablate.py` (default: the `ablate` folder beside the launcher) |
Ctrl+C stops the loaded model in order, so it saves its prefix cache and expert-cache state, pauses
running downloads and exits. Closing the console window, signing out or shutting Windows down does
the same, but Windows ends the launcher and the model about 5 seconds after it: a save not finished
by then leaves the previous prefix-cache file in place (the launcher saves its own settings and usage
first). Only the launcher's own events and server
errors appear in its console; the web UI shows the server's full log. A console that holds its
output (a QuickEdit selection) or a slow log folder never holds up the server: the launcher keeps
reading its output and writes the console and `serve-logs/` from threads of their own. On Windows
the server shares the launcher's console, which carries the orderly stop; started without one (by a
program or a shortcut that gives it none), the launcher makes a hidden console of its own.

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
cache** is on (several GB per model), `ablation/` ([Ablation](#ablation): the fitter's packages, prompt
sets, run folders with their recordings, up to a few GB a run, and the sidecars it made), and the scratch
folders `plans/` and `updates/`. A portable Python the Ablation page downloaded is `ablate-python` beside the
launcher itself.

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
  When the Engine holds requests and has completed nothing for 30 seconds, a warning says for how
  long (`live.no_progress_s`; it heals itself after two minutes by failing the running requests,
  [serving](serving.md)), and **Stalls healed** counts the times it did.
- **Inference gauges**: speedometer dials for decode tok/s (live while generating), prefill tok/s,
  time to first token, prompt-cache hit rate, and, when the model reports them, the VRAM expert-cache
  hit rate and drafter acceptance. A dial reads the last ten requests; under it are the session's
  value and the best (or, for time to first token, the slowest) of the requests the server keeps.
- **System gauges**: CPU, RAM, GPU and VRAM use, each with a 10-minute history and, under the dial,
  the CPU and GPU temperatures, the GPU's power and the server process's memory. **All gauges** adds
  dials for the CPU and GPU temperature, GPU power and PCIe traffic to and from the GPU (the expert
  and n-gram traffic of Qwen3.8-Flash-Next shows here); the browser remembers the choice. Everything
  comes from the operating system and the NVIDIA driver; no other software is needed. CPU temperature
  on Windows is the CPU package sensor from HWiNFO or Core Temp when one of them runs (HWiNFO needs
  *Shared Memory Support* on), otherwise the ACPI thermal zone Windows reports, a motherboard sensor
  rather than the package; a PC that exposes neither shows no temperature. Linux reads the kernel's
  hwmon sensors (coretemp, k10temp). With more than one NVIDIA GPU the gauges show the inference GPU,
  taken to be the one with the most VRAM (the first of equals in the driver's order): CUDA numbers the
  fastest device first, so on a PC with a smaller second card `infernix-serve`'s default `--device 0`
  is that one. The same GPU sizes the default launch parameters and the advice for a load that does
  not fit.

  A dial is one arc in the accent colour, a needle and the reading. The dials with a limit turn
  amber and say **High** at the first limit, red and **Critical** at the second: CPU temperature at
  85 and 95 °C, GPU temperature at 80 and 90 °C, and RAM when less than 4 GiB and 2 GiB are
  available (how full it is says little: models with offloaded experts fill it on purpose). A
  temperature past its limit is also named beside **All gauges** while its dial is not shown.
- **Memory**: what the loaded `infernix-serve` holds, as one bar for VRAM and one for RAM: its
  parts (model weights, the drafter, a LoRA adapter, experts with their count, model state, the KV cache, the prefix
  cache, workspace, CUDA graphs, staging, the n-gram table, the n-gram archive; the KV cache and the
  prefix cache side by side, since they trade pages with every request, so only their shared boundary
  moves), then
  **Infernix other** (its memory no
  part names: the CUDA context, the driver, heaps), **Other processes** and **Free**, beside which
  VRAM shows what the server keeps free for the display (its headroom). The KV cache and the prefix
  cache are drawn faint, with the share holding tokens or entries solid. **Reserved** (Inkling 2) is
  VRAM the server's plan keeps for what it allocates later, such as CUDA graphs not yet captured: it is
  still free on the card, so it is drawn faint, counts towards the server's figure and comes out of
  **Free**, which then sits near the headroom. While the server uses an SSD
  tier (an SSD prefix cache, or experts read from the artifact on the SSD), each part on the SSD is
  listed with its size. A part keeps its colour in every bar; hovering one gives its exact size and
  what it is. The parts come from `memory` in the server's `/metrics?format=json`
  ([serving](serving.md#json-representation)), as do the free VRAM and RAM the other programs' share
  follows from (the server's own readings, which its headroom and reserve are kept against); an older
  `infernix-serve` reports no parts, and the panel says so. Under WSL2 RAM is the Linux VM's.
- **Inference statistics**: the same figures as `infernix-serve`'s console panel, over the last ten
  requests and the whole session of the loaded model: requests, time to first token, prompt-cache hit
  rate, prefill and decode tok/s, mean decode batch, drafter acceptance and accepted tokens per
  round, n-gram acceptance, and the VRAM expert-cache hit rate (Qwen3.8-Flash-Next and the
  Qwen3.5-122B-A10B). They come from `session_stats` in the server's `/metrics?format=json`
  ([serving](serving.md#json-representation)); with an older `infernix-serve` the UI derives them
  from its last 128 requests. The drafter's cells read **no drafts** when none of the requests'
  rounds drafted. Once they are not zero, the session's preemptions, replayed tokens and
  healed stalls are listed too, and **Now** names paused and replaying requests. The columns keep
  their widths while the figures and **Now** change.
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
- **LoRA adapters** ([`--lora`](lora.md)): a folder holding a Hugging Face PEFT adapter's
  `adapter_config.json` and `adapter_model.safetensors`, named by the folder, with the base model,
  rank and modules its config names. It is matched to the models of that base (an adapter of
  `Qwen/Qwen3.8-27B`, or of a repackaging such as `unsloth/Qwen3.8-27B-bnb-4bit`, to the Qwen3.8-27B
  artifacts; one of `thinkingmachines/Inkling-2-Mini` to the Inkling 2 Mini artifacts); whether `infernix-serve`
  can apply it is the server's to say (below).
- **Ablation sidecars** ([`--ablate`](ablate.md)): a folder's `sidecar.json` in the `ablate-direction-v2`
  format with the direction file it names, named by the folder, with the layers it cuts. It is matched
  to the models whose hidden size and layer count it fits, the artifact it was fitted on (its
  `artifact_id`) first. A base direction that cuts no layer, another program's `sidecar.json` and the
  trial sidecars inside a `tools/ablate` run folder are not listed; the run's exports are.

Folders are rescanned when they change in the launcher, after a download and every minute. A folder
whose listing fails part way (a network share that goes away, say) is marked as scanned only in
part, with the reason; the files found before it stay listed.

**Models** shows each folder's size, the free space on its drive (read by the last scan, so a
sleeping network drive never holds up the web UI or the API), and each model's, drafter's,
volume's, adapter's and sidecar's size on disk. **Delete files** removes a model's artifact (every
part), a drafter, an n-gram volume, a LoRA adapter or an ablation sidecar from the disk after you type
its file (an adapter's: folder; a sidecar's: `sidecar.json`) name. A model's deletion takes the files at
the location shown (every part, its `.conversion.json` and `.expert-state`) and its configurations; an
adapter's takes its two files and a sidecar's its `sidecar.json` and direction file, then the folder
when nothing else is left in it; files a running server uses cannot be deleted.

## Configure a model

Each model's page edits its launch parameters, the `infernix-serve` options after the artifact
path. The form and the **Launch parameters** text are two views of the same list: a control changes
its option in the text, an edit to the text moves the controls, and options without a control are
kept. The form shows only what the model takes: one table in the launcher (`src/launcher/option_support.cpp`) says
which options each model type (Qwen3.5 dense, Qwen3.5-MoE with resident or offloaded experts, Qwen3.8-Flash-Next,
Inkling 2) accepts, from what the engine accepts and reads for it, and which `--spec` values and whether
`--max-context auto` apply. The same table builds the default arguments and filters every launch, preview and
**Will it fit?** plan: a saved value for an option the model does not take is left out of the command line, and the
preview names it. A model type the launcher does not know is not restricted. Inkling 2's defaults are
`--max-context auto` and `--spec auto --draft-tokens auto` (DSpark when the artifact bundles it), with `--lm-head-draft`
when the artifact holds its `proposal/token_rank` parameter and DSpark drafter. The form offers **Sliding-window KV format**
(`--kv-window-dtype`) beside **KV cache format** for Inkling 2 only. The **DFlash2 drafter** and **N-gram volume**
pickers (Qwen3.8-Flash-Next) list the compatible
files found and write `--draft-model` and `--ngram-volume`; the **LoRA adapter** picker (the dense
27Bs and Inkling 2) lists the adapters found, those of the model's base first (another base's is marked, and
picking it warns), or takes a folder or file elsewhere, and writes `--lora`. The full command line is previewed with
warnings (an option this `infernix-serve` does not have, a missing file, a Flash-Next model without
a volume). Every option of the installed `infernix-serve --help` is listed in the reference drawer,
**All options** (read once per build of `infernix-serve`; a `--help` that fails is run again when
next needed).

The form has a control for every option `infernix-serve` accepts, apart from the ones the launcher
sets itself (below), in sections that open and close: **Serving**, **Context and memory**,
**Speculative decoding**, **N-gram table** (Qwen3.8-Flash-Next), **Prefix cache** (the Host tier,
the SSD tier, next-turn prefill), **Offloaded experts** (models that have them), **Thinking and
tools**, **Requests and clients**, **Sampling overrides**, **Vision**, **Numerical routes and
speed**, **Ablation sidecar**, **Logging and diagnostics** and **Environment variables**. A section's badge counts the
options the configuration sets in it, so a closed section still shows that it holds settings; its
less common options are under **More options**, which stays open while one of them is set. The
browser remembers which sections are open. A control appears only for a model it applies to and
only when the installed `infernix-serve` lists its option.
[`tests/test_launcher_option_coverage.py`](../tests/test_launcher_option_coverage.py) fails when an
option of the server has no control.

**⋮** beside **All options** and **Duplicate** holds the rest: a new configuration, import and
export of a launch script, copying the command line, **Restore defaults** and deleting a named
configuration.

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
or becomes a new named one; otherwise the launcher says which artifact it needs. **Download as .bat**
and **Download as .sh** (in the **⋮** menu) write the configuration as a `.bat` (Windows; UTF-8, with
every argument quoted for cmd) or `.sh` that runs `infernix-serve` directly, with its model id.

**Will it fit?** asks the installed `infernix-serve` to size the configuration without loading it
(`infernix-serve --plan`): the VRAM it would take (weights, DFlash2's draft screen, the adapters (a LoRA adapter's
factors, an ablation sidecar's direction), KV cache and the context it holds, workspace (on Qwen3.5 with a cut lane's copy of
the residual), CUDA graphs, expert cache, headroom), the RAM (experts, pinned prefix cache, n-gram
table, what stays free), and the largest `--max-context` that fits. A configuration that does not
fit gets the same suggestions as a failed load ([Watchdog](#watchdog)). The plan reads the GPU as it
is now, so a loaded model's memory counts against it; unload first to plan for an empty GPU.
Turning on **Vision** checks at once: the vision tower takes VRAM a 27B's default context was sized
without, so the default context no longer fits beside it (on a 32 GB GPU the official NVFP4 27B
fits 177,856 tokens with `--vision` against 222,272 without). Setting or changing the **LoRA adapter** checks at
once too: its factors take VRAM beside the weights (304 MiB for a rank-32 adapter on every projection of a 27B). So
does setting or changing the **Ablation sidecar**: the plan reads it against the model, so a malformed sidecar or one
fitted to another geometry is refused here rather than at launch, and its direction (a few KiB) and, on Qwen3.5, the
cut lanes' copies of the residual are counted.

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
| Qwen3.8-27B | `--lm-head-draft --ngram-draft-tokens auto --max-context auto` (the context the GPU's memory holds beside the artifact, below) |
| Qwen3.6-27B, Qwen3.6-35B-A3B | `--lm-head-draft` where the artifact has a proposal head; the 27B's context as for Qwen3.8-27B |
| Qwen3.5-122B-A10B | `--max-context 262144` |
| Inkling 2 Mini | `--max-context auto` and, with the artifact's token ranking and DSpark drafter, `--lm-head-draft` |

No family sets `--host-context-mib`: `infernix-serve`'s default (`auto`) gives the Host prefix cache all the RAM
the model and its experts leave beyond `--ram-free-mib` ([serving guide](serving.md)).

A 27B keeps all of its KV cache on the GPU, so its context is what the GPU holds beside the
weights: the launcher sets `--max-context auto`, and the server sizes the context at startup to the
largest its KV cache holds beside the weights and the automatic headroom (256 MiB when NVML reports no
display on the GPU, else 1 GiB). Raise it with `--vram-headroom-mib 0` when nothing else uses the
GPU. Before the model loads, the launcher reports the model's own context for such a configuration
(at most 262,144 tokens); once it is loaded, the one the server resolved. **Context length**'s
**Auto** sets it for any model whose experts stay in VRAM; the plan reports what it resolves to.

The launcher sets `--host`, `--port`, `--model-id` (the model id), `--any-model-name`,
`--log-stats-panel off` and `--log-colours off` itself; they are removed from saved parameters.

## Download models

**Discover** lists the published Infernix conversions and NInfer's official artifacts. **Download**
lists the repository's files with their sizes (the model's parts are selected) and suggests what the
model also needs, unless a compatible one is already in a model folder: Qwen3.8-Flash-Next's DFlash2
drafter and an n-gram volume (the recommended table preselected). Choose the destination folder with
the folder browser; by default each repository gets its own subfolder. Any other repository can be
entered by its `owner/name`.

**Discover → LoRA adapters** asks Hugging Face for the PEFT LoRA adapters of a base model
(Qwen3.8-27B, Qwen3.6-27B or Inkling 2 Mini) whose repository holds `adapter_config.json` and
`adapter_model.safetensors` at its root, the files [`--lora`](lora.md) reads (GGUF exports and merged
models are left out): [Harness Hub](https://huggingface.co/harness-hub)'s adapters for agent
harnesses first (the rank-32 specialists for Pi, OpenCode and Hermes, and the rank-96 super-adapter
for all three), then the 60 most downloaded, with a search by name. Gated
ones need a token. **Download** reads the adapter's config first and shows its base model, rank and
modules (an adapter above rank 128, which `infernix-serve` refuses, is flagged) and selects its two
files; an adapter always gets a folder of its own, named `owner--name` after the repository, since
every adapter's files have the same names (two owners' adapters of one name stay apart). Once it is scanned, pick it on a 27B's Configure page; **Will
it fit?** then asks the server whether it applies it and what it takes.

Downloads run one repository at a time, each file in 64 MiB pieces over several connections
(**Settings → Downloads**), into `<file>.part`. Pausing, a dropped connection or a launcher restart
resumes where it stopped (a piece counts as done once it is on the disk). A finished file is checked
against Hugging Face's hash (an LFS file's SHA-256, else its git blob id) and renamed into place;
the destination becomes a model folder and is scanned. A Hugging Face token (settings, `HF_TOKEN`, or the `huggingface-cli login` token file) is sent for gated repositories.

## Ablation

**Ablation** makes a refusal-ablation sidecar ([`--ablate`](ablate.md)) for one of the library's models
on this computer with the fitter in `tools/ablate` ([fitting guide](ablate-fitting.md)). It records the
model's hidden states on harmful and harmless prompts and computes the direction its activations use to
refuse. It then searches for the layers and strengths whose cut both removes refusals and keeps the first
token on harmless prompts close to the stock model's (first-token KL; 40 held-out prompts of each list
grade every trial), grades the best few cuts on a validation pair the search never scored, picks the best
balance of the two there, grades the pick on a test pair neither saw and exports it. The page has three
steps:

1. **The tool.** A release keeps the fitter in the `ablate` folder beside the launcher (`--ablate-tool`
   names another; a build tree finds `tools/ablate` of the checkout it is in). It needs Python 3.11 or
   newer with pip: the launcher tries the Pythons the Windows `py` launcher lists, then `python3` and
   `python` on PATH, runs each and takes the first new enough with pip (a Microsoft Store stub, a removed
   Python, or Debian's and Ubuntu's Python without its `python3-pip` package is passed over); **Use this
   Python** names one. When none is found, **Set up** first downloads a portable one: CPython 3.13 from
   [python-build-standalone](https://github.com/astral-sh/python-build-standalone) (the builds `uv`
   installs; PSF licence; x86-64 only; a 46 MiB download that unpacks to about 150 MiB on Windows, 73 MiB
   and about 560 MiB on Linux), one release pinned and
   checked against its SHA-256, unpacked with the system's `tar` into `ablate-python` beside
   `infernix-launcher` (whose folder must be writable for it); once there, it is the Python used. Set up
   then installs the fitter's packages (numpy and optuna, about 40 MB from PyPI) with that Python's pip
   into `ablation/packages` in the launcher's folder; nothing is installed into the Python itself.
   Another Python version, or a launcher update that changes the requirements, asks for Set up again.
2. **Prompt sets.** **Download** fetches a public set from its publishers' pinned files, checks their
   SHA-256 and converts them to the fitter's lists in `ablation/prompts/<set>`, the publishers'
   licences shown beside each source. A sidecar records its sets and their licences in its
   `generation.source`.

   | Set | Fit prompts | Validation and test pairs | Licences |
   |---|---|---|---|
   | HarmBench + matched counterparts (recommended) | 200 HarmBench standard behaviors, each paired with a harmless counterpart on the same subject (shipped with the fitter) | validation: JBB-Behaviors, harmful and benign; test: XSTest, unsafe and safe | HarmBench, JBB-Behaviors: MIT; the counterparts: Apache-2.0; XSTest: CC BY 4.0 |
   | AdvBench + Alpaca (heretic's defaults) | 400 AdvBench behaviors, 400 Alpaca instructions | the same | AdvBench, JBB-Behaviors: MIT; Alpaca: CC BY-NC 4.0 (non-commercial use; its README limits it to research); XSTest: CC BY 4.0 |
   | HarmBench + Alpaca | 200 HarmBench standard behaviors, 200 Alpaca instructions | the same | HarmBench, JBB-Behaviors: MIT; Alpaca: CC BY-NC 4.0; XSTest: CC BY 4.0 |

   A validation prompt that is a fit prompt, and a test prompt that is a fit or validation prompt, is
   dropped, word for word or nearly (13 of JBB's harmful 100 are HarmBench behaviors, some reworded; 11
   are AdvBench ones). A set downloaded before the fitter had a validation pair shows as not downloaded
   and is downloaded again. Matched pairs make the direction the refusal rather than the topic
   ([why](ablate-fitting.md#matched-harmless-counterparts)), and the fit is told the lists are paired.
3. **Make a sidecar.** Choose the model (a configuration: its launch parameters and environment are
   used as they are, without its own sidecar, since a sidecar is fitted to the model as you run it), the
   prompt set, the search (**Quick** 150 trials, **Standard** 300, the fitter's default, **Thorough**
   600) and the sidecar's name, then **Start the run**. The fitter reads the configuration first and
   refuses one it cannot fit, with its reason, before anything changes.

A run takes the whole GPU. The loaded model is unloaded first, and no model loads until the run ends:
requests for a model are refused with the reason, and **Load** and the memory plan say so. Each trial
starts the model twice (a generating engine for the refusals, a scoring one for the KL), so a trial takes
roughly half a minute to a minute and a half on a large model, and a standard run several hours. The
page shows the phases, the trials, the stock model's refusals, the pick so far, the trade-off table, an
estimate of the time left and the fitter's output.

**Stop** asks the fitter to stop its engines (each saves its expert-cache state) and keeps every
finished trial; **End now**, while it stops, ends it and its engines at once and loses the trial under
way. Starting the same configuration, prompt set and name again continues the same run folder
(`ablation/runs/<name>-<hash>`, with its recordings and trials), as **Run again** does after a stop or a
failure; a changed command line, environment or `infernix-serve` starts a fresh one. The launcher stops
a run before it exits; a run it could not finish shows as stopped when it starts again.

The search minimizes both refusals and KL ([The fit](ablate-fitting.md#the-fit)), and the **trade-off
table** shows what it found: for each number of held-out refusals, the lowest KL a trial reached (no
other trial is better on both; fewer refusals usually cost KL). The balance is the lowest share of the
stock model's refusals left plus the KL, so a KL of 1 weighs as much as all of them, among the trials
below KL 0.4 that refuse fewer prompts than the stock model. Every trial is scored on the same held-out
prompts, so the search fits them: the six rows best by that balance are graded on the validation pair
(about as long as grading one arm of the test pair each), the table shows their validation refusals and
KL, and the pick is the best balance on those numbers. No sidecar is written when the stock model refuses
none of the prompts or no trial qualifies; the page says which. A sidecar goes to `ablation/sidecars/<name>`, which becomes a model folder, so it is listed on
**Models** and offered first in its model's **Ablation sidecar** field; **Use it with** sets it there.
Its test-pair results (refusals on both halves, and the KL where the stock model answered) are shown
with it.

**Export** on another row of the table keeps that trial instead, as a sidecar of its own named
`<name>-t<trial>`: the fitter's `choose` validates it (when it is not one of the six), grades it on the
same test pair and exports it, with the GPU
taken as for a run (a few minutes). Its results replace the pick's on the page, the table stays, and the
pick's sidecar is kept.

## Connect coding tools

Point a tool at the launcher instead of at `infernix-serve`. **Connect** shows the exact addresses and
the model list as tools see it, with snippets for Claude Code, Codex, Zed, Qwen Code, Hermes and the
OpenAI and Anthropic SDKs. **Endpoints**, at the top of every page, copies either base URL.

- `GET /v1/models` (also `/models`) lists the configurations not hidden, by model id, in the fields
  other servers use so each tool finds the ones it reads:
  - the OpenAI list fields; the Anthropic ones (`type`, `display_name`, `created_at`, `has_more`,
    `first_id`, `last_id`), paged by `limit` (default 20, at most 1,000), `after_id` and `before_id`
    when the request carries `anthropic-version`; the list's entries again as a top-level `models`,
    which Codex CLI's model-catalog refresh reads;
  - `name` and `description` (family, quantization, context, vision, whether it is loaded);
  - its context (`--max-context`; with `auto`, the loaded server's, else the model's own) as
    `context_length`, `context_window`, `max_model_len`,
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
  `/v1/messages/count_tokens`, `/v1/decide` and `/v1/score`; `GET`/`DELETE /v1/responses/{id}`,
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
  an HTTP error status, as does a whole-body success answer the stream cannot carry
  (`upstream_not_a_stream`). A non-streaming request receives no bytes until its model is ready, and
  so does a request to a route that never streams (`count_tokens`, `input_tokens`, `compact`, `decide`,
  `systemone`, `score`), whatever its `stream` says: it gets the server's own JSON answer and status.
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
any key, `/v1` needs one, as `Authorization: Bearer` or `x-api-key`, except the health check
(`/health` and `/v1/health`, the state and the loaded model's alias), which stays open as
infernix-serve's `/health` does. The event log and the usage page name the key each request came
with. Listening on another address than loopback without a key logs a warning.

The web UI and its management API (`/launcher/api`) answer this computer without a key. From
another computer they need **Manage from other computers** on, and then follow the keys as `/v1`
does: with no key they are open to every computer that reaches the launcher (listening beyond
loopback with remote management on and no key is that choice); once any key exists they need an
Admin key, which the UI asks for once and keeps as a sign-in cookie (HttpOnly, SameSite=Strict, 30
days; signing out or restarting the launcher ends it). The UI's own chat uses the endpoint without a
key.

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
and when automatic restarts are off, the next request loads it again. A model unloaded around the
failure (from the launcher, idle, or for another model) is not restarted. Requests running when the
server failed end with an error (their streams stop); requests waiting for the model are served by
the restarted one. Each incident shows as a banner on the dashboard and a notification, and goes into
the [event log](#event-log). With the watchdog off, a crash is still reported, nothing is restarted
and nothing ends a lock-up, a stall or a load.

For a load that did not fit, the banner lists changes worked out from the server's figures and the
model's parameters, most effective first: for VRAM, the largest context that fits by the server's
own plan (`infernix-serve --plan`, run once the failed server's memory is free and before the next
load starts; the suggestions show a second or so after the error, and a plan not done in 20 s leaves
the estimate), giving up the 1 GiB of VRAM headroom, one request at a time, INT8 KV
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

Both files rotate by size (16 MiB and 100 MiB) into `FILE.1` .. `FILE.5`; a file another program holds
open (a tail tool) keeps growing until a later rotation can move it. The server's own console,
one file per load, is kept in `serve-logs/` in the same folder (the newest 20). Moving the log folder
applies at once to later records; a folder that cannot be written leaves the records in memory and
says so.

## Settings

| Setting | Default | Meaning |
|---|---|---|
| Host, port | `127.0.0.1`, `1234` | listen address (applies after a restart) |
| API keys | none | named keys; with any, `/v1` needs one ([API keys](#api-keys-and-remote-access)) |
| Manage from other computers | off | the web UI from other computers; an Admin key once any key exists |
| CORS | on | browser access headers on every answer: any origin, and `infernix-serve`'s request headers (`Authorization` named, which a wildcard does not cover), `api-key` and the rest (applies after a restart) |
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
| Download connections, verify | 8, on | parallel connections per file; hash check |
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
`--fake-fail-load TEXT` and `--fake-hog-mib N` in the model's parameters; `--fake-experts` there
makes it an offloaded-expert model (an expert cache, experts in RAM and an SSD tier on the
dashboard's **Memory**) and `--fake-no-memory` a server too old to report its memory. Its own
`--help` lists a
dozen options, and the Configure form shows a control only for an option the server lists: set
`INFERNIX_FAKE_SERVE_HELP` to a text file holding a real `infernix-serve --help` and it prints that
instead, so the form shows every control.
[`tools/launcher/launcher_check.py`](../tools/launcher/launcher_check.py) runs a launcher against it
and checks the features (configurations, model ids, environment, the memory breakdown, script export
and import, the fit plan, keep-alive streams, routes and the pin, prompt progress, usage, API keys
and sign-in, chats, idle unload) and the watchdog, the event log and the request log, and, on Windows, that a launcher
started on a console without a window still stops its server in order. Its `ablation` group runs the
[Ablation](#ablation) page's backend with `tools/launcher/ablate_stub.py` in place of the fitter (Set up installs a
local wheel, so no network): the GPU taken from a loaded model and given back, a refused configuration, another
trial of the front kept with `choose`, stop and resume, End now with a tool that ignores the stop, and the last run
after a restart. Its groups configure the first Qwen3.8-27B in `--models`, for which every check was written, else the
first supported model; a check of an option that model does not take (`--lora` on a Qwen MoE model, the n-gram archive
off Qwen3.5) or that needs two models is then skipped by name in its summary.
