import type {
  Activity,
  ApiError,
  ApiKeyScope,
  Catalog,
  Chat,
  ChatSummary,
  CreatedApiKey,
  DownloadJob,
  DownloadRequest,
  EnvVar,
  EventPage,
  FitPlan,
  FsListing,
  InfernixRelease,
  InstanceInfo,
  LaunchPreview,
  LaunchScript,
  LauncherState,
  Library,
  LogPage,
  MetricsSnapshot,
  ModelConfig,
  ModelConfigPatch,
  ModelUpdates,
  RepoListing,
  ScriptImport,
  ServeOptions,
  Session,
  Settings,
  SettingsPatch,
  TelemetryHistory,
  Usage,
} from "./api-types";

export const API_BASE = "/launcher/api";

/** A failed call: the HTTP status and the launcher's message. */
export class ApiFailure extends Error {
  readonly status: number;
  readonly code: string;

  constructor(message: string, status: number, code: string) {
    super(message);
    this.name = "ApiFailure";
    this.status = status;
    this.code = code;
  }
}

function isApiError(value: unknown): value is ApiError {
  if (typeof value !== "object" || value === null || !("error" in value)) return false;
  const error = (value as { error: unknown }).error;
  return typeof error === "object" && error !== null && "message" in error;
}

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}/${path}`, {
      method,
      headers: body === undefined ? undefined : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch {
    throw new ApiFailure("The launcher is not reachable", 0, "offline");
  }
  const text = await response.text();
  let data: unknown = null;
  if (text.length > 0) {
    try {
      data = JSON.parse(text);
    } catch {
      data = null;
    }
  }
  if (!response.ok) {
    if (isApiError(data)) {
      throw new ApiFailure(data.error.message, response.status, data.error.code ?? "error");
    }
    throw new ApiFailure(`${method} ${path} failed (HTTP ${response.status})`, response.status, "http");
  }
  return data as T;
}

const enc = encodeURIComponent;

/** `?after=..&limit=..` of a paged log read ("" without either). */
function pageQuery(options: { after?: number; limit?: number }): string {
  const query: string[] = [];
  if (options.after !== undefined) query.push(`after=${options.after}`);
  if (options.limit !== undefined) query.push(`limit=${options.limit}`);
  return query.length > 0 ? `?${query.join("&")}` : "";
}

export const api = {
  state: () => request<LauncherState>("GET", "state"),

  settings: () => request<Settings>("GET", "settings"),
  saveSettings: (patch: SettingsPatch) => request<Settings>("PUT", "settings", patch),

  library: () => request<Library>("GET", "library"),
  rescan: () => request<unknown>("POST", "library/rescan"),
  addDir: (path: string) => request<unknown>("POST", "dirs", { path }),
  removeDir: (path: string) => request<unknown>("DELETE", `dirs?path=${enc(path)}`),

  // `id` is a configuration id: the artifact key, or `<artifact key>@<8 hex>`.
  modelConfig: (id: string) => request<ModelConfig>("GET", `models/${enc(id)}/config`),
  saveModelConfig: (id: string, patch: ModelConfigPatch) =>
    request<ModelConfig>("PUT", `models/${enc(id)}/config`, patch),
  resetModelConfig: (id: string) => request<ModelConfig>("POST", `models/${enc(id)}/config/reset`),
  createModelConfig: (
    artifactKey: string,
    body: { name: string; from?: string; alias?: string; args?: string[]; env?: EnvVar[] },
  ) => request<ModelConfig>("POST", `models/${enc(artifactKey)}/configs`, body),
  deleteModelConfig: (id: string) => request<unknown>("DELETE", `models/${enc(id)}/config`),
  preview: (id: string, args: string[]) => request<LaunchPreview>("POST", `models/${enc(id)}/preview`, { args }),
  /** The Engine's sizing for these parameters without loading the weights (a few seconds). */
  plan: (id: string, body: { args?: string[]; env?: EnvVar[] } = {}) =>
    request<FitPlan>("POST", `models/${enc(id)}/plan`, body),
  script: (id: string, shell: "cmd" | "sh") => request<LaunchScript>("GET", `models/${enc(id)}/script?shell=${shell}`),
  importScript: (text: string, fileName?: string) =>
    request<ScriptImport>("POST", "scripts/import", { text, file_name: fileName }),
  deleteModelFiles: (artifactKey: string, confirm: string) =>
    request<Library>("DELETE", `models/${enc(artifactKey)}/files`, { confirm }),
  deleteDrafter: (key: string, confirm: string) => request<Library>("DELETE", `drafters/${enc(key)}`, { confirm }),
  deleteNgramVolume: (key: string, confirm: string) => request<Library>("DELETE", `ngram/${enc(key)}`, { confirm }),

  load: (id: string, force = false) => request<unknown>("POST", `models/${enc(id)}/load`, { force }),
  unload: (force = false) => request<unknown>("POST", "unload", { force }),
  restart: () => request<unknown>("POST", "restart"),

  instance: () => request<InstanceInfo>("GET", "instance"),
  logs: (options: { after?: number; limit?: number } = {}) => request<LogPage>("GET", `logs${pageQuery(options)}`),
  /** The launcher's event log; without `after`, the newest `limit` records. */
  eventLog: (options: { after?: number; limit?: number } = {}) =>
    request<EventPage>("GET", `eventlog${pageQuery(options)}`),
  /** Opens the log folder in the file manager of the computer running the launcher. */
  openLogFolder: () => request<{ ok: boolean }>("POST", "logs/open", {}),
  metrics: () => request<MetricsSnapshot | null>("GET", "metrics"),
  telemetry: () => request<TelemetryHistory>("GET", "telemetry"),
  activity: () => request<Activity>("GET", "activity"),

  catalog: () => request<Catalog>("GET", "catalog"),
  repoFiles: (repo: string) => request<RepoListing>("GET", `hf/files?repo=${enc(repo)}`),

  downloads: () => request<DownloadJob[]>("GET", "downloads"),
  startDownload: (body: DownloadRequest) => request<DownloadJob>("POST", "downloads", body),
  pauseDownload: (id: string) => request<unknown>("POST", `downloads/${enc(id)}/pause`),
  resumeDownload: (id: string) => request<unknown>("POST", `downloads/${enc(id)}/resume`),
  cancelDownload: (id: string) => request<unknown>("POST", `downloads/${enc(id)}/cancel`),
  removeDownload: (id: string) => request<unknown>("DELETE", `downloads/${enc(id)}`),

  listDir: (path: string) => request<FsListing>("GET", `fs?path=${enc(path)}`),
  mkdir: (path: string) => request<unknown>("POST", "fs/mkdir", { path }),

  serveOptions: () => request<ServeOptions>("GET", "serve/options"),

  session: () => request<Session>("GET", "session"),
  signIn: (key: string) => request<Session>("POST", "session", { key }),
  signOut: () => request<Session>("DELETE", "session"),
  createApiKey: (name: string, scope: ApiKeyScope) => request<CreatedApiKey>("POST", "keys", { name, scope }),
  deleteApiKey: (id: string) => request<unknown>("DELETE", `keys/${enc(id)}`),

  chats: () => request<ChatSummary[]>("GET", "chats"),
  chat: (id: string) => request<Chat>("GET", `chats/${enc(id)}`),
  saveChat: (chat: Chat) => request<ChatSummary>("PUT", `chats/${enc(chat.id)}`, chat),
  deleteChat: (id: string) => request<unknown>("DELETE", `chats/${enc(id)}`),

  usage: (range: { from?: string; to?: string } = {}) => {
    const query = [range.from ? `from=${range.from}` : "", range.to ? `to=${range.to}` : ""].filter(Boolean).join("&");
    return request<Usage>("GET", `usage${query ? `?${query}` : ""}`);
  },

  modelUpdates: () => request<ModelUpdates>("GET", "updates"),
  checkModelUpdates: () => request<ModelUpdates>("POST", "updates/check"),
  applyModelUpdate: (jobId: string) => request<DownloadJob>("POST", `updates/${enc(jobId)}/apply`),
  infernixRelease: () => request<InfernixRelease>("GET", "infernix/release"),
  downloadInfernixUpdate: () => request<InfernixRelease>("POST", "infernix/update"),
  installInfernixUpdate: () => request<InfernixRelease>("POST", "infernix/install"),
};

/** The public model list exactly as a coding tool requests it. */
export interface PublicModel {
  id: string;
  object?: string;
  display_name?: string;
  context_length?: number;
  max_model_len?: number;
  max_tokens?: number;
  state?: string;
  owned_by?: string;
}

export async function publicModels(apiKey: string): Promise<PublicModel[]> {
  const headers: Record<string, string> = {};
  if (apiKey.length > 0) headers.Authorization = `Bearer ${apiKey}`;
  let response: Response;
  try {
    response = await fetch("/v1/models", { headers });
  } catch {
    throw new ApiFailure("The launcher is not reachable", 0, "offline");
  }
  const data: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    const message =
      typeof data === "object" && data !== null && "error" in data
        ? String((data as { error: { message?: unknown } }).error?.message ?? `HTTP ${response.status}`)
        : `HTTP ${response.status}`;
    throw new ApiFailure(message, response.status, "http");
  }
  const list = (data as { data?: unknown }).data;
  return Array.isArray(list) ? (list as PublicModel[]) : [];
}

export function errorMessage(error: unknown): string {
  if (error instanceof Error) return error.message;
  return String(error);
}
