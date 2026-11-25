// The launcher's management API (`/launcher/api/*`) and its event stream. The C++ backend
// (src/launcher) produces exactly these shapes; keep both in step.

/** Unix time in milliseconds. */
export type UnixMs = number;

export type ModelFamily =
  | "qwen3.8-flash-next"
  | "qwen3.5-122b-a10b"
  | "qwen3.6-35b-a3b"
  | "qwen3.8-27b"
  | "qwen3.6-27b"
  | "qwen3.5"
  | "inkling2-mini"
  | "unknown";

// ---------------------------------------------------------------------------------------------
// Settings

export type UnknownModelPolicy = "loaded" | "reject";

export interface Settings {
  /** Listen address of the launcher (OpenAI/Anthropic endpoint and UI). Restart to apply. */
  host: string;
  port: number;
  /** Named API keys. With none, /v1 needs no key. With any, /v1 needs one of them (as
   * `Authorization: Bearer`, `x-api-key` or `api-key`), except from the launcher's own page;
   * managing from another computer needs an `admin` key (a sign-in, see Session). Create and
   * revoke them with POST /launcher/api/keys and DELETE /launcher/api/keys/{id}. */
  api_keys: ApiKeyInfo[];
  cors: boolean;
  allow_remote_admin: boolean;
  /** Path of infernix-serve; "" = the one beside the launcher. */
  serve_executable: string;
  /** Folders scanned for artifacts, in order. */
  model_dirs: string[];
  scan_depth: number;
  /** Folder the download dialog proposes. */
  download_dir: string;
  /** Loopback port of the managed infernix-serve; 0 = pick a free one. */
  child_port: number;
  drain_timeout_s: number;
  load_timeout_s: number;
  stop_timeout_s: number;
  /** Unload the model after this many idle minutes; 0 = never. */
  idle_unload_min: number;
  /** Requests whose `model` matches a pattern go to its target, before the model ids are looked up
   * (only `pin_loaded` comes first); case-insensitive,
   * `*` matches any run of characters, first match wins. Target: a configuration id, or "loaded"
   * (whatever is loaded; the default model when nothing is). E.g. `claude-*haiku*` -> "loaded" keeps
   * Claude Code's background requests on the loaded model. */
  model_routes: ModelRoute[];
  /** Every request goes to the loaded model, whatever it names: no automatic switching (the UI's
   * Load still switches). With nothing loaded, requests resolve as usual. */
  pin_loaded: boolean;
  /** Answer llama-server's router API (/props, /models/load, /models/unload, /models/sse) for tools
   * with a llama.cpp provider (Zed). Off by default: tools that tell servers apart by these paths
   * (Hermes) would take the launcher for llama.cpp. */
  llamacpp_api: boolean;
  unknown_model: UnknownModelPolicy;
  /** Model key loaded when a request names an unknown model and none is loaded. */
  default_model: string | null;
  /** Load the default model (else the last one) when the launcher starts. */
  autoload: boolean;
  download_connections: number;
  verify_downloads: boolean;
  hf_token_set: boolean;
  open_browser: boolean;
  /** Keep each model's prefix cache across loads (adds --prefix-cache-file in the data folder). */
  persist_prefix_cache: boolean;
  /** Folder of the launcher's event log, request-content log and server logs; "" = the launcher's
   * own folder (`LauncherState.logs` has the resolved paths). Applies at once. */
  log_dir: string;
  /** Also record every proxied request's body and response (infernix-launcher-requests.jsonl). */
  log_bodies: boolean;
  /** Watch the running server for crashes, lock-ups (no answer) and stalls (no progress). */
  watchdog: boolean;
  /** Restart the model after the watchdog ended a crashed, locked-up or stalled server (at most
   * 3 times in 15 minutes). */
  restart_on_failure: boolean;
  /** Seconds without an answer, with requests in it or waiting on it, before a ready server counts
   * as locked up (10-3600). An idle server is never ended; one that stops answering while idle is
   * noted in the event log. */
  hang_timeout_s: number;
  /** Seconds with requests running but no prompt or output token processed before the server
   * counts as stalled; 0 = off. */
  stall_timeout_s: number;
  /** While the watchdog is on: stop a model load when the computer's free RAM stays below this many
   * MiB for three seconds (the system would otherwise page heavily); 0 = off (0-262144). */
  load_ram_floor_mib: number;
}

/** PUT /launcher/api/settings body: any subset; `hf_token` "" clears it. */
export type SettingsPatch = Partial<Omit<Settings, "api_keys" | "hf_token_set">> & {
  hf_token?: string;
};

export interface ModelRoute {
  pattern: string;
  /** A configuration id, or "loaded". */
  target: string;
}

export type ApiKeyScope = "api" | "admin";

export interface ApiKeyInfo {
  id: string;
  name: string;
  /** The key's first characters ("inx-3fK9..."), to recognise it. */
  prefix: string;
  /** api: the OpenAI/Anthropic routes; admin: those and the launcher's management. */
  scope: ApiKeyScope;
  created_at: UnixMs;
  last_used_at: UnixMs | null;
}

/** POST /launcher/api/keys `{ name, scope }` answers this; the secret is shown this once (the
 * launcher keeps only its SHA-256). */
export interface CreatedApiKey {
  key: ApiKeyInfo;
  secret: string;
}

/** GET /launcher/api/session (answers everyone): whether this browser may manage the launcher.
 * POST /launcher/api/session `{ key }` signs in with an admin key (an HttpOnly cookie), DELETE
 * signs out. Management calls answer 401 `{error: {code: "sign_in"}}` when a sign-in is needed. */
export interface Session {
  /** False from this computer, or when no key exists. */
  required: boolean;
  signed_in: boolean;
  key_name: string | null;
}

// ---------------------------------------------------------------------------------------------
// Library

/** One named configuration of an artifact. Every artifact has its default configuration (id = the
 * artifact key) and any number of named ones (id = `<artifact key>@<8 hex>`); each is a separate
 * model in /v1/models under its own model id, and loading one loads the artifact with its
 * parameters. Every route below that takes `{id}` takes a configuration id. */
export interface ModelConfig {
  /** Configuration id: the artifact key, or `<artifact key>@<8 hex>`. */
  id: string;
  /** The artifact's key. */
  key: string;
  /** "Default" for the default configuration; the user's name otherwise. */
  name: string;
  is_default: boolean;
  /** Model id coding tools see and send (editable; unique across configurations). */
  alias: string;
  default_alias: string;
  /** Listed in /v1/models. */
  listed: boolean;
  /** Launch parameters, without the artifact path and the launcher-managed flags. */
  args: string[];
  default_args: string[];
  /** Environment variables infernix-serve starts with (besides the launcher's own). */
  env: EnvVar[];
  /** Unload after this many minutes without requests; null = Settings.idle_unload_min; 0 = never. */
  idle_unload_min: number | null;
  /** args, env or alias differ from the defaults (named configurations: always true). */
  customized: boolean;
  /** Effective --max-context (8192 when absent); with --max-context auto, the loaded server's resolved
   * context, or the model's own before it loads. */
  context_length: number;
  /** --max-context auto: the server sizes the context at startup. */
  context_auto: boolean;
  /** The artifact has a vision tower and these parameters enable it (--vision): images can be sent. */
  vision: boolean;
  updated_at: UnixMs | null;
}

export interface EnvVar {
  name: string;
  value: string;
}

export interface ModelArtifact {
  /** artifact_id, 32 hex digits: stable across moves and renames. */
  key: string;
  name: string;
  file_name: string;
  /** Entry file (part 1 of a numbered set). */
  path: string;
  dir: string;
  /** Every entry path holding this artifact id (copies). */
  locations: string[];
  format: "infernix" | "ninfer";
  family: ModelFamily;
  family_label: string;
  model_type: string;
  recipe: string | null;
  /** The weight formats holding most parameters, ready to show: "NVFP4 + FP8", "IQ3_S". */
  quantization: string | null;
  bytes: number;
  file_count: number;
  missing_files: string[];
  components: string[];
  has_vision: boolean;
  has_mtp: boolean;
  has_dflash: boolean;
  has_dflash2: boolean;
  /** Inkling 2's bundled DSpark drafter. */
  has_dspark: boolean;
  has_proposal: boolean;
  /** What this model takes (src/launcher/option_support.cpp, the one owner): the form shows nothing else. */
  option_support: {
    /** infernix-serve options this model does not take. */
    unsupported: string[];
    /** The --spec values it takes. */
    spec_values: string[];
    /** Whether it takes --max-context auto. */
    context_auto: boolean;
  };
  offloaded_experts: boolean;
  /** Qwen3.8-Flash-Next: needs an n-gram volume and takes --draft-model. */
  needs_ngram_volume: boolean;
  accepts_draft_model: boolean;
  native_context: number;
  hidden_size: number;
  num_layers: number;
  vocab_size: number;
  num_experts: number | null;
  expert_state: string | null;
  /** `<entry>.ngram` when present (serve's default volume). */
  own_ngram_volume: string | null;
  modified: UnixMs;
  /** The default configuration (configs[0]). */
  config: ModelConfig;
  /** Every configuration, the default first. */
  configs: ModelConfig[];
}

export interface DrafterArtifact {
  key: string;
  name: string;
  file_name: string;
  path: string;
  dir: string;
  bytes: number;
  /** Components it carries, e.g. ["dflash2"]. */
  kind: string;
  target_architecture: string | null;
  /** Keys of library models it can draft for. */
  compatible_models: string[];
  modified: UnixMs;
}

export type NgramFormat = "fp8" | "iq4_nl" | "bf16" | "nvfp4" | "unknown";

export interface NgramVolume {
  /** volume_id, 32 hex digits. */
  key: string;
  name: string;
  file_name: string;
  path: string;
  dir: string;
  bytes: number;
  format: NgramFormat;
  rows: number;
  /** Keys of library models that can read it. */
  compatible_models: string[];
  /** Keys of models whose own table it is. */
  own_for: string[];
  modified: UnixMs;
}

/** What a PEFT LoRA adapter's adapter_config.json says (null where it does not say it). */
export interface AdapterConfig {
  peft_type: string | null;
  /** base_model_name_or_path: the model it was trained on. */
  base_model: string | null;
  /** r */
  rank: number | null;
  /** The largest of r and the rank_pattern's ranks. */
  max_rank: number | null;
  /** lora_alpha */
  alpha: number | null;
  /** target_modules, when the config lists them (empty for a pattern). */
  target_modules: string[];
  rslora: boolean;
}

/** A LoRA adapter (--lora): a folder holding adapter_config.json and adapter_model.safetensors. */
export interface LoraAdapter {
  /** 32 hex digits of its folder's path's SHA-256. */
  key: string;
  /** Its folder's name (the repository's, when the launcher downloaded it). */
  name: string;
  /** The folder: what --lora takes. */
  path: string;
  dir: string;
  /** adapter_model.safetensors */
  bytes: number;
  config: AdapterConfig;
  /** Keys of library models of the base model it names (dense Qwen3.8-27B or Qwen3.6-27B). */
  compatible_models: string[];
  modified: UnixMs;
}

/** An ablation sidecar (--ablate): a sidecar.json of format ablate-direction-v2 with its direction file beside it.
 * Sidecars that cut no layer, and the trial sidecars of a tools/ablate run's fit/ folder, are not listed. */
export interface AblationSidecar {
  /** 32 hex digits of its sidecar.json's path's SHA-256. */
  key: string;
  /** Its folder's name. */
  name: string;
  /** The sidecar.json: what --ablate takes. */
  path: string;
  dir: string;
  /** The direction file. */
  bytes: number;
  /** The hidden size it was fitted for. */
  direction_dim: number;
  /** 1 (one shared direction) or one per layer. */
  rows: number;
  /** Layers whose attention and MLP lanes it cuts. */
  attn_layers: number;
  mlp_layers: number;
  /** The artifact it was fitted on (32 hex digits), when it says. */
  artifact_id: string | null;
  /** That artifact's file name (generation.fitted_under.artifact), when it says. */
  fitted_artifact: string | null;
  /** generation.source: what produced its direction. */
  source: string | null;
  /** Keys of library models whose geometry it fits. */
  compatible_models: string[];
  /** Of those, the model it was fitted on. */
  fitted_models: string[];
  modified: UnixMs;
}

export interface ModelDir {
  path: string;
  exists: boolean;
  models: number;
  drafters: number;
  ngram_volumes: number;
  adapters: number;
  sidecars: number;
  /** Bytes of the models, drafters, volumes, adapters and sidecars found in it. */
  library_bytes: number;
  /** The drive holding it. */
  free_bytes: number | null;
  total_bytes: number | null;
  error: string | null;
}

export interface LibraryIssue {
  path: string;
  message: string;
}

export interface Library {
  scanned_at: UnixMs | null;
  scanning: boolean;
  dirs: ModelDir[];
  models: ModelArtifact[];
  drafters: DrafterArtifact[];
  ngram_volumes: NgramVolume[];
  adapters: LoraAdapter[];
  sidecars: AblationSidecar[];
  issues: LibraryIssue[];
}

/** PUT /launcher/api/models/{id}/config body (any subset). POST /launcher/api/models/{artifact
 * key}/configs with `{ name, from?: configuration id, alias?, args?, env? }` creates a named
 * configuration (a copy of `from`, else of the defaults) and returns its ModelConfig; DELETE
 * /launcher/api/models/{id}/config removes a named one; POST .../config/reset restores the
 * defaults (a named configuration keeps its name and model id). */
export interface ModelConfigPatch {
  name?: string;
  alias?: string;
  listed?: boolean;
  args?: string[];
  env?: EnvVar[];
  idle_unload_min?: number | null;
}

/** POST /launcher/api/scripts/import `{ text, file_name? }`: the infernix-serve (or ninfer-serve)
 * commands of a launch script (.bat/.cmd: `set VAR=...`, `^` continuations; .sh: `export VAR=...`,
 * `VAR=... cmd`, `\` continuations), each matched to a library artifact by its path or file name.
 * Apply one with PUT .../config or POST .../configs. */
export interface ScriptImport {
  commands: ImportedCommand[];
}

export interface ImportedCommand {
  /** The artifact path as written. */
  artifact: string;
  /** The library artifact it names; null when none matches. */
  model: { key: string; name: string; file_name: string } | null;
  /** --model-id, when the script set one. */
  alias: string | null;
  /** Launch parameters without the managed flags (--host, --port, --model-id, ...). */
  args: string[];
  env: EnvVar[];
  /** Flags this infernix-serve does not know, managed flags dropped, NInfer-only options... */
  warnings: string[];
}

/** GET /launcher/api/models/{id}/script?shell=cmd|sh: the configuration as a launch script. */
export interface LaunchScript {
  file_name: string;
  text: string;
}

/** POST /launcher/api/models/{id}/plan `{ args?, env? }` (default: the saved configuration):
 * `infernix-serve --plan`, the Engine's own sizing for these parameters on this machine without
 * loading the weights. Takes a few seconds; the loaded model keeps running (its memory counts as
 * used unless it is the one planned, then its own share is counted as available). */
export interface FitPlan {
  /** False when this infernix-serve has no --plan (an older build): nothing else is set. */
  supported: boolean;
  /** null when the plan could not be made (error). */
  fits: boolean | null;
  /** One line: "Fits: 29.1 of 32.0 GiB VRAM, 18.2 of 96 GiB RAM" or the shortfall. */
  summary: string;
  vram: PlanBudget | null;
  ram: PlanBudget | null;
  /** The largest --max-context that fits with these parameters, when the plan can tell. */
  max_context: number | null;
  /** KV capacity in tokens across all requests, as sized. */
  kv_tokens: number | null;
  error: string | null;
  /** When it does not fit: changes to try (as for a failed load). */
  suggestions: LoadSuggestion[];
  /** The server's own report, for the curious. */
  report: unknown;
}

export interface PlanBudget {
  total: number;
  /** Free for the model now. */
  available: number;
  /** What the plan uses. */
  planned: number;
  parts: { label: string; bytes: number }[];
}

export interface LaunchPreview {
  executable: string;
  /** Full argv including the artifact path and the managed flags. */
  argv: string[];
  /** Shell-quoted command line for the host OS. */
  command_line: string;
  /** Unknown flags, managed flags removed, missing files, missing n-gram volume... */
  warnings: string[];
  /** Flags the launcher sets itself. */
  managed: string[];
}

// ---------------------------------------------------------------------------------------------
// Server instance

export type InstanceState = "idle" | "starting" | "ready" | "stopping" | "failed";

export interface InstanceInfo {
  state: InstanceState;
  /** Increments on every start; log lines carry it. */
  instance: number;
  model_key: string | null;
  alias: string | null;
  name: string | null;
  pid: number | null;
  port: number | null;
  started_at: UnixMs | null;
  ready_at: UnixMs | null;
  /** Latest startup milestone ("loading weights | 6.12 GiB", "engine ready ..."). */
  phase: string;
  argv: string[];
  exit_code: number | null;
  error: string | null;
  /** Proxied requests running on the child / waiting for a model. */
  in_flight: number;
  waiting: number;
  /** Model a pending switch will load once the current one drains. */
  pending_model: string | null;
  /** model_key is a configuration id; these name the artifact and the configuration. */
  artifact_key: string | null;
  config_name: string | null;
  /** The last request's end (or the load), and when an idle model is unloaded (null: never). */
  last_request_at: UnixMs | null;
  idle_unload_at: UnixMs | null;
  /** Requests go to this model whatever they name (Settings.pin_loaded). */
  pinned: boolean;
  memory: { working_set: number; private_bytes: number } | null;
  /** The watchdog's latest incident in this launcher session; null when none. */
  incident: Incident | null;
  /** An ablation run (the Ablation page) has the GPU: no model loads until it ends. */
  ablation_holds_gpu: boolean;
}

/** crash/hang/stall: a ready server failed; vram/ram: a load did not fit (the server's own sizing
 * error, or the launcher stopped it at the RAM floor). */
/** `port`: infernix-serve could not bind its port (another program holds it); advice only. */
export type IncidentKind = "crash" | "hang" | "stall" | "vram" | "ram" | "port";

export interface Incident {
  t: UnixMs;
  kind: IncidentKind;
  /** "load" for vram/ram, "running" for crash/hang/stall. */
  phase: "load" | "running";
  model_key: string;
  alias: string;
  /** What happened, for people: "infernix-serve crashed: access violation (0xC0000005) ...". */
  message: string;
  /** What the launcher did: "restarting it (1 of 3 in 15 minutes)", "the load was stopped". */
  action: string;
  restarted: boolean;
  /** vram/ram: changes that may make the model fit, most effective first; empty otherwise. */
  suggestions: LoadSuggestion[];
}

export interface LoadSuggestion {
  title: string;
  detail: string;
  /** The model's full launch parameters with the change made (PUT .../config {args}); null for
   * advice without a flag (close other programs). */
  args: string[] | null;
}

// ---------------------------------------------------------------------------------------------
// The launcher's event log (infernix-launcher.log): its own events and the API's, by metadata.

export type EventLevel = "debug" | "info" | "warn" | "error";
export type EventCategory = "launcher" | "model" | "watchdog" | "api" | "download" | "settings";

export interface EventRecord {
  seq: number;
  t: UnixMs;
  level: EventLevel;
  category: EventCategory;
  message: string;
  /** Structured details; for `api` an ApiEventFields. */
  fields: Record<string, unknown> | null;
}

/** `fields` of an `api` event: one proxied (or refused) request. */
export interface ApiEventFields {
  id: number;
  client: string;
  user_agent: string | null;
  method: string;
  path: string;
  /** The `model` the request named, and the model key that served it. */
  model: string | null;
  model_key: string | null;
  stream: boolean;
  status: number;
  duration_ms: number;
  /** Time to the first response byte (streams: the first event). */
  first_byte_ms: number | null;
  bytes_in: number;
  bytes_out: number;
  error: string | null;
  /** The name of the API key it sent (null: none). */
  key_name: string | null;
  /** Why it went to that model: "name", "route", "pinned", "loaded". */
  routed_by: string | null;
}

export interface EventPage {
  lines: EventRecord[];
  first_seq: number;
  last_seq: number;
}

export interface LogPaths {
  /** The folder in use (Settings.log_dir, else the launcher's folder). */
  dir: string;
  events_file: string;
  /** Written only while Settings.log_bodies is on. */
  requests_file: string;
  /** Per-run server console logs (serve-<time>-<run>-<model>.log, newest 20 kept). */
  serve_dir: string;
}

export type LogLevel = "trace" | "debug" | "info" | "warn" | "error" | "critical" | "raw";

export interface LogLine {
  seq: number;
  instance: number;
  /** Parsed from the record's timestamp, else the time it was read. */
  t: UnixMs;
  level: LogLevel;
  source: "server" | "launcher";
  text: string;
}

export interface LogPage {
  lines: LogLine[];
  first_seq: number;
  last_seq: number;
}

// ---------------------------------------------------------------------------------------------
// Inference metrics: the child's GET /metrics?format=json passed through unchanged.

export interface StatsRow {
  requests: number;
  ttft_s: number | null;
  prompt_tokens: number;
  cached_tokens: number;
  cache_hit_rate: number | null;
  prefill_tokens: number;
  prefill_s: number;
  prefill_tok_s: number | null;
  decode_tokens: number;
  decode_s: number;
  decode_tok_s: number | null;
  batch: number | null;
  drafted: number;
  accepted: number;
  acceptance: number | null;
  accepted_per_round: number | null;
  ngram_drafted: number;
  ngram_accepted: number;
  ngram_acceptance: number | null;
  ngram_rounds: number;
  archive_drafted: number;
  archive_accepted: number;
  archive_acceptance: number | null;
  expert_lookups: number;
  expert_hits: number;
  expert_hit_rate: number | null;
}

export interface SessionStats {
  recent_window: number;
  /** "mtp" | "dflash" | "dflash2" once a request used a model drafter. */
  drafter: string | null;
  ngram_enabled: boolean;
  completed: number;
  failed: number;
  cancelled: number;
  rejected: number;
  session: StatsRow;
  recent: StatsRow;
}

export interface MetricsRequestRow {
  id: number;
  time: number; // unix seconds
  protocol: string;
  finish: string;
  duration_s: number;
  prompt_tokens: number;
  reused: number;
  output_tokens: number;
  ttft_ms: number;
  prompt_ms: number;
  decode_ms: number;
  prefill_tok_s: number | null;
  decode_tok_s: number | null;
  hit_rate: number | null;
  drafts_offered: number;
  drafts_accepted: number;
  ngram_drafts: number;
  ngram_accepted: number;
  expert_lookups: number;
  expert_hits: number;
}

export interface MetricsSnapshot {
  schema: string;
  time: number;
  engine: {
    name: string;
    model: string;
    architecture: string;
    weights: string;
    kv: string;
    speculative_backend: string;
    max_context: number;
    max_concurrency: number;
  };
  live: {
    state: "idle" | "reading" | "generating";
    running: number;
    waiting: number;
    prefilling: number;
    decode_ready: number;
    paused: number;
    replaying: number;
    materializing: number;
    /** Seconds the Engine has held requests without completing anything; 0 while it progresses
     * or is idle. It warns at 30 s and heals the stall at 120 s (the running requests fail).
     * Absent from servers built before 2026-11. */
    no_progress_s?: number;
    /** Stalls healed that way since the server started. */
    no_progress_recoveries?: number;
    tok_s: number | null;
    tok_s_mean: number | null;
    prefill_tok_s_mean: number | null;
    start_time: number;
    instance_id: string;
  };
  totals: {
    requests: number;
    completed: number;
    cancelled: number;
    failed: number;
    rejected: number;
    prompt_tokens: number;
    reused: number;
    computed_prefill_tokens: number;
    output_tokens: number;
    decode_tokens: number;
    drafts_offered: number;
    drafts_accepted: number;
    preemptions: number;
    replayed_tokens: number;
  };
  requests: MetricsRequestRow[];
  /** Absent from servers built before the launcher; the UI then derives rows from `requests`. */
  session_stats?: SessionStats;
  /** What the server holds in VRAM, RAM and on the SSD, by part. Absent from servers built before
   * 2026-11-13 (docs/serving.md, JSON representation). */
  memory?: MemoryBreakdown;
}

/** A closed vocabulary; a newer server's unknown key is shown under its raw name. */
export type MemoryPartKey =
  | "weights"
  | "drafter"
  | "adapter"
  | "experts"
  | "kv_cache"
  | "mtp_kv"
  | "state"
  | "prefix_cache"
  | "workspace"
  | "cuda_graphs"
  | "staging"
  | "ngram"
  | "ngram_archive";

/** One category of what the server holds in a pool. Parts never overlap. */
export interface MemoryPart {
  key: MemoryPartKey | (string & {});
  /** Allocated (reserved): what the operating system counts as used. */
  bytes: number;
  /** The occupied share inside the part (KV pages holding tokens, prefix-cache slabs holding
   * entries); at most `bytes`. */
  used_bytes?: number;
  /** Experts held. */
  count?: number;
}

export interface MemoryBreakdown {
  /** The GPU the server runs on. */
  device: {
    total_bytes: number;
    /** Free on the device now, all processes counted. */
    free_bytes: number;
    /** What the server keeps free for the display and other programs. */
    headroom_bytes: number;
    /** The server's own VRAM use as the OS counts it; null where the platform reports none. */
    process_bytes: number | null;
    parts: MemoryPart[];
  };
  host: {
    total_bytes: number;
    available_bytes: number;
    /** The server's resident memory (working set / RSS); null if unknown. */
    process_bytes: number | null;
    parts: MemoryPart[];
  };
  /** Present only while the server uses an SSD tier (prefix cache, or experts read from the SSD). */
  ssd: { parts: MemoryPart[] } | null;
}

// ---------------------------------------------------------------------------------------------
// System telemetry (1 Hz)

export interface GpuSample {
  index: number;
  name: string;
  /** 0..100 */
  usage: number | null;
  mem_usage: number | null;
  vram_total: number;
  vram_used: number;
  temp_c: number | null;
  power_w: number | null;
  power_limit_w: number | null;
  sm_clock_mhz: number | null;
  mem_clock_mhz: number | null;
  fan_pct: number | null;
  /** PCIe throughput, bytes per second. */
  pcie_rx_bps: number | null;
  pcie_tx_bps: number | null;
  pcie_gen: number | null;
  pcie_width: number | null;
}

/** Where the CPU temperature comes from. Every source is optional: a PC without any of them still
 * reports every other statistic, with temp_c null. */
export type CpuTempSource = "hwinfo" | "coretemp" | "thermal-zone" | "hwmon";

export interface SystemSample {
  t: UnixMs;
  /** temp_source: "hwinfo" (HWiNFO shared memory, CPU package) | "coretemp" (Core Temp shared
   * memory, CPU package) | "thermal-zone" (Windows ACPI thermal zone: a motherboard sensor, not the
   * CPU package) | "hwmon" (Linux) | null (no sensor; temp_c is null too). */
  cpu: { usage: number; temp_c: number | null; temp_source: CpuTempSource | null };
  ram: { total: number; used: number; available: number };
  /** Empty without an NVIDIA driver (NVML); any field NVML cannot read is null. */
  gpus: GpuSample[];
  /** The managed infernix-serve process. */
  server: { working_set: number; private_bytes: number; cpu: number | null } | null;
}

export interface SystemInfo {
  hostname: string;
  os: string;
  cpu_name: string;
  cores: number;
  threads: number;
  ram_total: number;
  gpus: { index: number; name: string; vram_total: number }[];
  driver_version: string | null;
  /** "hwinfo" | "coretemp" | "thermal-zone" | "hwmon" | null: where the CPU temperature comes
   * from (see SystemSample.cpu.temp_source). */
  cpu_temp_source: CpuTempSource | null;
}

export interface TelemetryHistory {
  interval_ms: number;
  samples: SystemSample[];
}

// ---------------------------------------------------------------------------------------------
// Proxy activity

export interface ProxyRecord {
  id: number;
  t: UnixMs;
  method: string;
  path: string;
  model: string | null;
  /** Configuration id the request resolved to. */
  model_key: string | null;
  /** Why it went there: "name", "route" (a model route), "pinned", "loaded" (unknown name). */
  routed_by: string | null;
  stream: boolean;
  /** 0 while running or waiting. */
  status: number;
  state: "waiting" | "running" | "done" | "failed";
  duration_ms: number | null;
  error: string | null;
  /** The client's address, and its API key's name when it sent one. */
  client: string;
  key_name: string | null;
  /** While it runs: the server's progress (prompt tokens processed of the total, tokens out). */
  progress: RequestProgress | null;
}

/** A running request's progress (from the server's metrics, matched by its label). */
export interface RequestProgress {
  phase: "queued" | "prefill" | "decode";
  prompt_tokens: number;
  cached_tokens: number;
  /** Prompt tokens done (cached included); prompt_tokens when the prompt is done. */
  processed_tokens: number;
  output_tokens: number;
  elapsed_ms: number;
}

export interface Activity {
  in_flight: number;
  waiting: number;
  recent: ProxyRecord[];
}

// ---------------------------------------------------------------------------------------------
// Catalog and downloads

export interface CatalogEntry {
  repo: string;
  title: string;
  family: ModelFamily;
  kind: "model" | "drafter" | "ngram";
  description: string;
  recommended: boolean;
  /** Repository it was converted from. */
  source_repo: string | null;
  tags: string[];
  approx_bytes: number;
  /** What the model also needs, as catalog repos ("Wallawalla47/...") or repo:file pairs. */
  drafter_repo: string | null;
  ngram_options: { repo: string; file: string; format: NgramFormat; bytes: number; recommended: boolean }[];
  notes: string | null;
}

export interface Catalog {
  entries: CatalogEntry[];
}

export type RepoFileRole =
  | "model"
  | "model-part"
  | "drafter"
  | "ngram"
  | "expert-state"
  | "report"
  | "adapter"
  | "other";

export interface RepoFile {
  path: string;
  bytes: number;
  sha256: string | null;
  role: RepoFileRole;
  default_selected: boolean;
  /** Local path of an existing file with the same name and size, if any (an adapter's: in a folder of the
   * repository's name). */
  present_locally: string | null;
}

export interface RepoListing {
  repo: string;
  revision: string;
  /** "adapter" for a LoRA adapter repository (adapter_config.json and adapter_model.safetensors at its root). */
  kind: "model" | "drafter" | "ngram" | "adapter";
  files: RepoFile[];
  /** An adapter repository's adapter_config.json, read before the download. */
  adapter: AdapterConfig | null;
  /** Why the adapter's config could not be read. */
  adapter_error: string | null;
}

/** GET /launcher/api/hf/adapters?base=&search=: LoRA adapters on the Hub that --lora reads. */
export interface HubAdapter {
  repo: string;
  /** The Hub's count of the last 30 days. */
  downloads: number;
  likes: number;
  /** ISO 8601 */
  last_modified: string;
  /** Its files need a token from an account that accepted its terms. */
  gated: boolean;
  /** Harness Hub's (huggingface.co/harness-hub), trained for an agent harness. */
  harness_hub: boolean;
}

export interface HubAdapterSearch {
  base: string;
  search: string;
  /** Harness Hub's first, then the most downloaded. */
  adapters: HubAdapter[];
}

export type DownloadFileState = "queued" | "downloading" | "verifying" | "done" | "failed" | "paused" | "skipped";

export interface DownloadFile {
  path: string;
  bytes: number;
  done: number;
  state: DownloadFileState;
  error: string | null;
}

export type DownloadState = "queued" | "running" | "paused" | "verifying" | "done" | "failed" | "cancelled";

export interface DownloadJob {
  id: string;
  repo: string;
  /** Folder the files land in (dest_dir, plus the repo name when subfolder was asked). */
  target_dir: string;
  files: DownloadFile[];
  total: number;
  done: number;
  speed_bps: number;
  eta_s: number | null;
  state: DownloadState;
  error: string | null;
  created_at: UnixMs;
  finished_at: UnixMs | null;
}

export interface DownloadRequest {
  repo: string;
  files: string[];
  dest_dir: string;
  /** Put the files in dest_dir/<repo name>. */
  subfolder: boolean;
}

// ---------------------------------------------------------------------------------------------
// File-system browsing (folder picker)

export interface FsEntry {
  name: string;
  path: string;
  kind: "dir" | "drive";
}

export interface FsListing {
  /** "" lists the roots (drives on Windows, "/" on Linux). */
  path: string;
  parent: string | null;
  entries: FsEntry[];
  free_bytes: number | null;
  total_bytes: number | null;
  error: string | null;
}

// ---------------------------------------------------------------------------------------------
// infernix-serve --help, parsed

export interface ServeOption {
  flag: string;
  /** Argument placeholder ("N", "FILE", "auto|N") or null for a switch. */
  arg: string | null;
  section: string;
  description: string;
}

export interface ServeOptions {
  executable: string;
  found: boolean;
  error: string | null;
  build: string | null;
  options: ServeOption[];
}

// ---------------------------------------------------------------------------------------------
// Whole-launcher state

export interface LauncherState {
  version: string;
  /** This launcher process; a restarted one numbers its console and event-log lines afresh. */
  run: string;
  /** The launcher's own folder (AppData only when that folder is read-only). */
  data_dir: string;
  /** Full path of infernix-launcher.json: settings, model configs, last model, download queue. */
  memory_file: string;
  logs: LogPaths;
  endpoints: {
    /** OpenAI SDK base URL, e.g. http://127.0.0.1:1234/v1 */
    openai: string;
    /** Anthropic SDK base URL, e.g. http://127.0.0.1:1234 */
    anthropic: string;
  };
  /** The address the launcher listens on (--host/--port override Settings until a restart). */
  listen: { host: string; port: number };
  restart_required: boolean;
  settings: Settings;
  instance: InstanceInfo;
  system: SystemInfo;
}

// ---------------------------------------------------------------------------------------------
// Server-sent events on GET /launcher/api/events (EventSource; `event:` name -> data type)

export interface LauncherEvents {
  hello: LauncherState;
  system: SystemSample;
  instance: InstanceInfo;
  metrics: MetricsSnapshot | null;
  log: { lines: LogLine[] };
  library: Library;
  downloads: DownloadJob[];
  config: ModelConfig;
  activity: Activity;
  toast: { level: "info" | "success" | "warning" | "error"; message: string };
  settings: Settings;
  /** New event-log records, batched (debug ones included). */
  eventlog: { lines: EventRecord[] };
  /** The resolved log paths, after the log folder changed. */
  logs: LogPaths;
  infernix_release: InfernixRelease;
  model_updates: ModelUpdates;
  ablation: AblationState;
}

// ---------------------------------------------------------------------------------------------
// Ablation (the Ablation page: the launcher runs tools/ablate, which fits a refusal-ablation sidecar). GET
// /launcher/api/ablation -> AblationState (the first call looks for a Python: a second or two). POST .../ablation/python
// `{ path }` ("" finds one again), .../ablation/python/detect, .../ablation/setup (pip installs the tool's packages),
// .../ablation/prompts/{preset} (download and convert), .../ablation/runs `{ config, preset, trials, name }` and
// .../ablation/stop `{ now? }` answer the same; the SSE event `ablation` carries it on every change.

export interface AblationSource {
  name: string;
  title: string;
  licence: string;
  /** What the licence asks of its users, in a few words. */
  licence_note: string;
  home: string;
}

/** A public prompt set the tool downloads (from pinned files, checked by SHA-256) and converts. */
export interface AblationPreset {
  name: string;
  title: string;
  description: string;
  /** Line i of the harmless list is the counterpart of line i of the harmful one (the fit is told so). */
  paired: boolean;
  harmful: { source: string; count: number };
  harmless: { source: string; count: number };
  /** The pair the fit's best trials are graded on to pick among them. */
  validate: { harmful: string; benign: string };
  /** The pair the kept trial is graded on once. */
  test: { harmful: string; benign: string };
  sources: AblationSource[];
}

/** A downloaded prompt set (<data>/ablation/prompts/<preset>). */
export interface AblationFetched {
  title: string;
  fetched: string;
  counts: Record<string, number>;
  /** The sets and their licences in words: the sidecar's generation.source. */
  source_text: string;
  dir: string;
}

export interface AblationTask {
  kind: "setup" | "fetch";
  preset: string;
  state: "running" | "done" | "failed";
  message: string;
  log: string[];
  started: UnixMs;
  ended: UnixMs | null;
}

export interface AblationRefusals {
  refusals: number;
  n: number;
  errors?: number;
  truncated?: number;
}

export type AblationPhase =
  | "doctor"
  | "prompts"
  | "capture"
  | "direction"
  | "fit"
  | "validate"
  | "pick"
  | "grade"
  | "export";

/** A trial (or the stock model) on the validation prompts: refusals of the `n` graded harmful ones, over-refusals of
 * the benign ones, first-token KL on the benign ones (all, and where the stock model answered). */
export interface AblationValidated {
  refusals: number;
  n: number;
  benign_refusals: number;
  benign_n: number;
  kl: number | null;
  kl_answered: number | null;
}

/** A trial of the fit: refusals of the `n` held-out harmful prompts, first-token KL on the held-out harmless ones,
 * and its validation numbers once graded on the validation pair. */
export interface AblationTrial {
  trial: number;
  refusals: number;
  n: number;
  kl: number;
  what: string;
  validation?: AblationValidated;
  /** A validated trial that is not on the front (validation takes the fit's best trials, on the front or not). */
  off_front?: boolean;
}

/** The tool's own account of the run (its run folder's progress.json). */
export interface AblationProgress {
  state: "running" | "done" | "no_sidecar" | "stopped" | "failed";
  /** `auto` (a fit and its pick) or `choose` (another trial of the same fit kept). */
  command: "auto" | "choose";
  phase: AblationPhase | null;
  phases_done: AblationPhase[];
  message: string;
  /** Completed trials in the study; `seconds`: the recent trials' mean time. */
  trials: { done: number; target: number; seconds?: number };
  /** The stock model on the held-out harmful prompts. */
  control: { refusals: number; n: number } | null;
  /** The pick's rule: KL below `kl_max`, the lowest refusal score (refusals over the stock model's) + KL / `kl_scale`. */
  rule: { kl_max: number; kl_scale: number };
  /** The study's Pareto front, fewest refusals first: no trial has both fewer refusals and a lower KL. Once
   * validated, each validated trial off the front too (`off_front`). */
  front: AblationTrial[];
  /** Once the fit's best trials are graded on the validation pair: the lists' names, the stock model's numbers, the
   * validated trials. Null without a validation pair. */
  validation: { harmful: string; benign: string; stock: AblationValidated; trials: number[] } | null;
  /** The pick so far by the rule. */
  pick: AblationTrial | null;
  /** What the pick was made on: the held-out numbers (until validation, or without a validation pair) or the
   * validation ones. */
  picked_on: "fit" | "validation";
  /** The trial being kept: the pick (auto), or the chosen trial (choose). */
  best: AblationTrial | null;
  /** The test prompts' results, per arm ("stock" and the kept trial). */
  grade: Record<
    string,
    { harmful: AblationRefusals; benign: AblationRefusals | null; kl: number | null; kl_stock_answered: number | null }
  > | null;
  export: string | null;
  started: string;
  updated: string;
}

export type AblationJobState = "waiting_gpu" | "running" | "stopping" | "done" | "no_sidecar" | "stopped" | "failed";

export interface AblationJob {
  run: { config_id: string; model_name: string; preset: string; trials: number; name: string };
  run_dir: string;
  /** A job that keeps another trial of the run (the tool's `choose`) as the sidecar `name`; null for a fit. */
  choice: { trial: number; name: string } | null;
  state: AblationJobState;
  /** The run is taking, or holds, the GPU. */
  active: boolean;
  message: string;
  started: UnixMs;
  ended: UnixMs | null;
  progress: AblationProgress | null;
  /** The tool's last lines of output. */
  log: string[];
  /** The exported sidecar.json (done). */
  exported: string | null;
}

export interface AblationState {
  tool: { found: true; dir: string } | { found: false; looked_in: string[] };
  python: {
    /** The Python chosen on the page ("": the launcher finds one). */
    chosen: string;
    found: { executable: string; version: string } | null;
    tried: { path: string; result: string }[];
    minimum: string;
    /** The portable Python (python-build-standalone) Set up downloads into `dir`, beside the launcher, when no Python
     * 3.11 or newer with pip is found; null where none is pinned for this platform. */
    portable: { version: string; bytes: number; url: string; dir: string; installed: boolean } | null;
  };
  /** The tool's packages (numpy, optuna), installed for the Python found. */
  packages: { ready: boolean; dir: string };
  catalogue: AblationPreset[] | null;
  catalogue_error: string | null;
  prompts: Record<string, AblationFetched>;
  prompts_dir: string;
  sidecars_dir: string;
  task: AblationTask | null;
  job: AblationJob | null;
  limits: { min_trials: number; max_trials: number };
}

// ---------------------------------------------------------------------------------------------
// Chat (the launcher's playground). Chats live in the launcher's folder (`chats/`), so every
// browser sees them and they survive restarts. GET /launcher/api/chats -> ChatSummary[] (newest
// first); GET/PUT/DELETE /launcher/api/chats/{id} (PUT stores the whole Chat; a new id creates
// it; ids are 1-64 of [A-Za-z0-9_-]). Messages go to the public POST /v1/chat/completions like any
// client's (the launcher's own page needs no API key), streamed with `stream_options:
// {include_usage: true}`, `return_progress: true` (prompt_progress chunks: {total, cache,
// processed, time_ms}) and `timings_per_token: true`; the last chunk's `timings` (llama-server
// shape: cache_n, prompt_n, prompt_ms, prompt_per_second, predicted_n, predicted_ms,
// predicted_per_second) give the statistics. Proxied answers carry the header
// `X-Infernix-Request-Id` (the activity id).

export interface ChatSummary {
  id: string;
  title: string;
  /** The configuration it talks to (null: whatever is loaded). */
  model: string | null;
  messages: number;
  created_at: UnixMs;
  updated_at: UnixMs;
}

export interface Chat extends Omit<ChatSummary, "messages"> {
  system_prompt: string;
  options: ChatOptions;
  messages: ChatMessage[];
}

export interface ChatOptions {
  /** null: the server's default. */
  temperature: number | null;
  max_tokens: number | null;
  /** Ask for thinking (chat_template_kwargs.enable_thinking). */
  thinking: boolean;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  /** Images sent with a user message, as data: URLs (PNG/JPEG/WebP/GIF, at most 20 MB each). */
  images: string[];
  /** The assistant's thinking (reasoning_content). */
  reasoning: string | null;
  stats: ChatStats | null;
  /** The configuration that answered (assistant messages). */
  model: string | null;
  error: string | null;
  t: UnixMs;
}

export interface ChatStats {
  /** Time to the first token as this page saw it (queueing and a model load included). */
  ttft_ms: number;
  prompt_tokens: number;
  /** Prompt tokens the prefix cache supplied. */
  cached_tokens: number;
  /** Prompt (prefill) rate over the uncached tokens. */
  prefill_tok_s: number | null;
  decode_tok_s: number | null;
  output_tokens: number;
  total_ms: number;
  /** Accepted / offered draft tokens, when speculative decoding ran. */
  drafted: number | null;
  accepted: number | null;
}

// ---------------------------------------------------------------------------------------------
// Usage: GET /launcher/api/usage?from=YYYY-MM-DD&to=YYYY-MM-DD (local dates, inclusive; default the
// last 30 days). One row per day, configuration and client, from the server's own per-request
// accounting; kept in infernix-launcher-usage.json beside the memory file.

export interface UsageRow {
  day: string;
  /** Configuration id and its model id at the time. */
  config_id: string;
  model: string;
  /** The API key's name, else the client's address ("launcher chat" for the playground). */
  client: string;
  requests: number;
  failed: number;
  prompt_tokens: number;
  cached_tokens: number;
  /** Prompt tokens computed (not from the cache). */
  prefill_tokens: number;
  output_tokens: number;
  prefill_s: number;
  decode_s: number;
}

export interface Usage {
  from: string;
  to: string;
  rows: UsageRow[];
}

// ---------------------------------------------------------------------------------------------
// Disk and updates

/** DELETE /launcher/api/models/{artifact key}/files `{ confirm: file_name }` deletes the artifact's
 * files (every part and its .conversion.json and .expert-state sidecars) and its configurations;
 * refused while it is loaded. DELETE /launcher/api/drafters/{key} and /launcher/api/ngram/{key}
 * (same body) delete a drafter or an n-gram volume. All answer the new Library. Downloads refuse to
 * start when the destination's drive has too little free space for what remains to fetch. */
export interface DeleteRequest {
  confirm: string;
}

/** GET /launcher/api/updates: Hugging Face repositories downloaded through the launcher whose
 * files changed since. POST /launcher/api/updates/check re-checks now (it also runs every 6
 * hours); POST /launcher/api/updates/{job id}/apply queues a download of the changed files into
 * the same folder (refused while that model is loaded). */
export interface ModelUpdates {
  checked_at: UnixMs | null;
  checking: boolean;
  updates: ModelUpdate[];
  error: string | null;
}

export interface ModelUpdate {
  /** The download job it came from. */
  job_id: string;
  repo: string;
  target_dir: string;
  revision: string;
  latest_revision: string;
  files: { path: string; bytes: number; change: "changed" | "new" }[];
  bytes: number;
}

/** GET /launcher/api/infernix/release: this build against the latest GitHub release of
 * Wallawalla47/Infernix. POST /launcher/api/infernix/update downloads it (SHA-256 checked against
 * GitHub's digest); POST /launcher/api/infernix/install stops the model in order, replaces the
 * program files beside the launcher (the memory file, logs and chats stay) and restarts the
 * launcher. SSE `infernix_release` carries changes. Windows only (releases are Windows builds). */
export interface InfernixRelease {
  /** This build's id, e.g. "windows-2026.10.18-174-g52bcb024". */
  current: string;
  /** Its release version ("2026.10.18"), when it is or follows one. */
  current_version: string | null;
  latest: {
    tag: string;
    version: string;
    published_at: string;
    /** Release notes (Markdown). */
    notes: string;
    url: string;
    asset: { name: string; bytes: number; sha256: string | null } | null;
  } | null;
  /** latest is a newer release than this build. */
  newer: boolean;
  /** Why it cannot be installed from here (not Windows, folder not writable, a source build...). */
  blocked: string | null;
  state: "idle" | "checking" | "downloading" | "ready" | "installing" | "failed";
  progress: { done: number; total: number } | null;
  checked_at: UnixMs | null;
  error: string | null;
}

/** Error body of every failed management call (HTTP 4xx/5xx). */
export interface ApiError {
  error: { message: string; code: string };
}
