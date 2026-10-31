// The chat playground: the open chats, the replies streaming into them, and their storage in the
// launcher's folder (`/launcher/api/chats`, see the Chat section of api-types.ts). A reply goes to
// the public POST /v1/chat/completions like any client's request and streams as server-sent
// events; it keeps streaming while the page shows another chat or route. A chat is saved when the
// user's turn is added, when the reply ends, and after every other change.

import { useQuery, type QueryClient } from "@tanstack/react-query";
import { useSyncExternalStore } from "react";
import { toast } from "sonner";

import { ApiFailure, api, errorMessage } from "./api";
import type { Chat, ChatMessage, ChatOptions, ChatStats, ChatSummary, InstanceInfo, Library } from "./api-types";
import { getLive } from "./events";
import { allConfigs, findConfig } from "./library";
import { readStorage, writeStorage } from "./utils";

export const CHATS_KEY = ["chats"] as const;
/** Chat ids the launcher stores. */
export const CHAT_ID_PATTERN = /^[A-Za-z0-9_-]{1,64}$/;

export const IMAGE_TYPES: ReadonlySet<string> = new Set(["image/png", "image/jpeg", "image/webp", "image/gif"]);
export const IMAGE_ACCEPT = "image/png,image/jpeg,image/webp,image/gif";
export const MAX_IMAGE_BYTES = 20 * 1024 * 1024;

const TITLE_CHARS = 60;
/** Streamed text reaches the page at most this often (each update re-renders the reply). */
const FLUSH_MS = 50;
/** Typing in the chat options is saved this long after the last change. */
const SAVE_DELAY_MS = 800;
/** The model and options a new chat starts with: those of the last chat used. */
const DEFAULTS_KEY = "infernix.chat.defaults";

const DEFAULT_OPTIONS: ChatOptions = { temperature: null, max_tokens: null, thinking: true };

function num(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function positive(value: unknown): number | null {
  const n = num(value);
  return n !== null && n > 0 ? n : null;
}

// ---------------------------------------------------------------------------------------------
// Live statistics of a streaming reply

export type ReplyPhase = "waiting" | "prefill" | "decode";

/** A `prompt_progress` chunk: prompt tokens processed (cached included) of the total. */
export interface PromptProgress {
  total: number;
  cache: number;
  processed: number;
  time_ms: number;
}

/** What the page knows of a reply while it streams. */
export interface LiveStats {
  /** waiting: nothing back yet (queued, or a model loading); prefill: reading the prompt;
   * decode: tokens arriving. */
  phase: ReplyPhase;
  /** Date.now() when the request was sent. */
  sent_at: number;
  progress: PromptProgress | null;
  ttft_ms: number | null;
  prompt_tokens: number | null;
  cached_tokens: number | null;
  prefill_tok_s: number | null;
  decode_tok_s: number | null;
  output_tokens: number;
  drafted: number | null;
  accepted: number | null;
}

/** llama-server's `timings` (sent with every chunk under `timings_per_token`). */
interface Timings {
  cache_n?: number;
  prompt_n?: number;
  prompt_ms?: number;
  prompt_per_second?: number;
  predicted_n?: number;
  predicted_ms?: number;
  predicted_per_second?: number;
  draft_n?: number;
  draft_n_accepted?: number;
}

interface Usage {
  prompt_tokens?: number;
  completion_tokens?: number;
  prompt_tokens_details?: { cached_tokens?: number } | null;
}

interface StreamChunk {
  model?: unknown;
  choices?: { delta?: { content?: unknown; reasoning_content?: unknown; reasoning?: unknown } | null }[] | null;
  prompt_progress?: Partial<PromptProgress> | null;
  timings?: Timings | null;
  usage?: Usage | null;
  error?: unknown;
}

function progressOf(raw: Partial<PromptProgress>): PromptProgress | null {
  const total = num(raw.total);
  const processed = num(raw.processed);
  if (total === null || processed === null) return null;
  return { total, processed, cache: num(raw.cache) ?? 0, time_ms: num(raw.time_ms) ?? 0 };
}

/** The text of an `error` member: `{message}` or a string. */
function errorText(error: unknown): string {
  if (typeof error === "string" && error.length > 0) return error;
  if (typeof error === "object" && error !== null && "message" in error && typeof error.message === "string") {
    return error.message;
  }
  return "The server reported an error";
}

/** Collects one reply's chunks: its text, thinking and statistics. Times are milliseconds after
 * the request was sent. */
class ReplyTracker {
  content = "";
  reasoning = "";
  /** The model id the server answered as. */
  model: string | null = null;
  progress: PromptProgress | null = null;
  timings: Timings | null = null;
  usage: Usage | null = null;
  firstAt: number | null = null;
  lastAt: number | null = null;
  /** Chunks that carried text (one token each, without speculative decoding). */
  deltas = 0;
  readonly sentAt: number;

  constructor(sentAt: number) {
    this.sentAt = sentAt;
  }

  /** Takes one chunk; returns the error it carries, if any. */
  add(chunk: StreamChunk, at: number): string | null {
    if (chunk.error !== undefined && chunk.error !== null) return errorText(chunk.error);
    if (typeof chunk.model === "string" && chunk.model.length > 0) this.model = chunk.model;
    if (chunk.prompt_progress) this.progress = progressOf(chunk.prompt_progress) ?? this.progress;
    if (chunk.timings) this.timings = chunk.timings;
    if (chunk.usage) this.usage = chunk.usage;
    const delta = chunk.choices?.[0]?.delta;
    if (delta) {
      const thinking = delta.reasoning_content ?? delta.reasoning;
      let text = false;
      if (typeof thinking === "string" && thinking.length > 0) {
        this.reasoning += thinking;
        text = true;
      }
      if (typeof delta.content === "string" && delta.content.length > 0) {
        this.content += delta.content;
        text = true;
      }
      if (text) {
        if (this.firstAt === null) this.firstAt = at;
        this.lastAt = at;
        this.deltas += 1;
      }
    }
    return null;
  }

  get empty(): boolean {
    return this.content.length === 0 && this.reasoning.length === 0;
  }

  private promptTokens(): number | null {
    const promptN = num(this.timings?.prompt_n);
    if (promptN !== null) return (num(this.timings?.cache_n) ?? 0) + promptN;
    return num(this.usage?.prompt_tokens) ?? num(this.progress?.total);
  }

  private cachedTokens(): number | null {
    return (
      num(this.timings?.cache_n) ??
      num(this.usage?.prompt_tokens_details?.cached_tokens) ??
      num(this.progress?.cache)
    );
  }

  /** Over the uncached prompt tokens. */
  private prefillRate(): number | null {
    const rate = positive(this.timings?.prompt_per_second);
    if (rate !== null) return rate;
    const n = positive(this.timings?.prompt_n);
    const ms = positive(this.timings?.prompt_ms);
    if (n !== null && ms !== null) return n / (ms / 1000);
    const progress = this.progress;
    if (progress && progress.time_ms > 0 && progress.processed > progress.cache) {
      return (progress.processed - progress.cache) / (progress.time_ms / 1000);
    }
    return null;
  }

  private outputTokens(): number {
    return num(this.timings?.predicted_n) ?? num(this.usage?.completion_tokens) ?? this.deltas;
  }

  private decodeRate(): number | null {
    const rate = positive(this.timings?.predicted_per_second);
    if (rate !== null) return rate;
    const n = positive(this.timings?.predicted_n);
    const ms = positive(this.timings?.predicted_ms);
    if (n !== null && ms !== null) return n / (ms / 1000);
    // Without timings: the tokens after the first over the time since it.
    const tokens = this.outputTokens();
    if (this.firstAt !== null && this.lastAt !== null && tokens > 1 && this.lastAt > this.firstAt) {
      return (tokens - 1) / ((this.lastAt - this.firstAt) / 1000);
    }
    return null;
  }

  private drafts(): { drafted: number | null; accepted: number | null } {
    const drafted = num(this.timings?.draft_n);
    if (drafted === null || drafted <= 0) return { drafted: null, accepted: null };
    return { drafted, accepted: num(this.timings?.draft_n_accepted) ?? 0 };
  }

  live(): LiveStats {
    return {
      phase: this.firstAt !== null ? "decode" : this.progress ? "prefill" : "waiting",
      sent_at: this.sentAt,
      progress: this.progress,
      ttft_ms: this.firstAt,
      prompt_tokens: this.promptTokens(),
      cached_tokens: this.cachedTokens(),
      prefill_tok_s: this.prefillRate(),
      decode_tok_s: this.firstAt !== null ? this.decodeRate() : null,
      output_tokens: this.outputTokens(),
      ...this.drafts(),
    };
  }

  /** The reply's statistics once it ended; null when nothing came back. */
  stats(totalMs: number): ChatStats | null {
    if (this.firstAt === null && this.timings === null && this.usage === null) return null;
    return {
      ttft_ms: this.firstAt ?? totalMs,
      prompt_tokens: this.promptTokens() ?? 0,
      cached_tokens: this.cachedTokens() ?? 0,
      prefill_tok_s: this.prefillRate(),
      decode_tok_s: this.decodeRate(),
      output_tokens: this.outputTokens(),
      total_ms: totalMs,
      ...this.drafts(),
    };
  }
}

/** The share of the prompt the prefix cache supplied (0..1), when known. */
export function cacheHitRatio(stats: { prompt_tokens: number | null; cached_tokens: number | null }): number | null {
  if (stats.prompt_tokens === null || stats.cached_tokens === null || stats.prompt_tokens <= 0) return null;
  return Math.min(1, stats.cached_tokens / stats.prompt_tokens);
}

/** The statistics of the chat's latest finished reply. */
export function latestStats(messages: readonly ChatMessage[]): ChatStats | null {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message.role === "assistant" && message.stats) return message.stats;
  }
  return null;
}

// ---------------------------------------------------------------------------------------------
// Server-sent events

/** The `data` of each server-sent event of a response body. Comment lines (the launcher's
 * keep-alives while it loads a model) and the other fields carry nothing this page uses. */
async function* sseData(body: ReadableStream<Uint8Array>): AsyncGenerator<string> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let data: string[] = [];
  try {
    for (;;) {
      const { value, done } = await reader.read();
      buffer += done ? decoder.decode() : decoder.decode(value, { stream: true });
      let start = 0;
      for (let end = buffer.indexOf("\n"); end >= 0; end = buffer.indexOf("\n", start)) {
        const line = buffer.slice(start, end > start && buffer.charCodeAt(end - 1) === 13 ? end - 1 : end);
        start = end + 1;
        if (line.length === 0) {
          if (data.length > 0) {
            yield data.join("\n");
            data = [];
          }
        } else if (line.startsWith("data:")) {
          data.push(line.charCodeAt(5) === 32 ? line.slice(6) : line.slice(5));
        }
      }
      buffer = buffer.slice(start);
      if (done) break;
    }
    if (buffer.startsWith("data:")) data.push(buffer.charCodeAt(5) === 32 ? buffer.slice(6) : buffer.slice(5));
    if (data.length > 0) yield data.join("\n");
  } finally {
    reader.cancel().catch(() => undefined);
  }
}

/** The message of a failed (non-2xx) answer: `{error: {message}}`, else its text or status. */
async function responseError(response: Response): Promise<string> {
  const text = await response.text().catch(() => "");
  try {
    const data: unknown = JSON.parse(text);
    if (typeof data === "object" && data !== null && "error" in data) return errorText(data.error);
  } catch {
    // not JSON
  }
  const plain = text.trim();
  return plain.length > 0 && !plain.startsWith("<") ? plain.slice(0, 400) : `HTTP ${response.status} ${response.statusText}`.trim();
}

// ---------------------------------------------------------------------------------------------
// The store

/** A reply streaming into a chat. */
export interface ChatReply {
  message_id: string;
  live: LiveStats;
}

export interface ChatEntry {
  chat: Chat;
  /** The launcher stores it (false: a new chat nothing has been sent in yet). */
  saved: boolean;
  reply: ChatReply | null;
}

interface Store {
  entries: Readonly<Record<string, ChatEntry>>;
  /** Why a chat could not be opened, by id. */
  failures: Readonly<Record<string, string>>;
  /** Ids of the chats a reply is streaming into. */
  streaming: readonly string[];
}

let store: Store = { entries: {}, failures: {}, streaming: [] };
const listeners = new Set<() => void>();

function commit(patch: Partial<Store>): void {
  store = { ...store, ...patch };
  for (const listener of listeners) listener();
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

function streamingIds(entries: Readonly<Record<string, ChatEntry>>): readonly string[] {
  const ids = Object.values(entries)
    .filter((entry) => entry.reply !== null)
    .map((entry) => entry.chat.id);
  const same = ids.length === store.streaming.length && ids.every((id, index) => id === store.streaming[index]);
  return same ? store.streaming : ids;
}

function putEntry(entry: ChatEntry): void {
  const entries = { ...store.entries, [entry.chat.id]: entry };
  commit({ entries, streaming: streamingIds(entries) });
}

function dropEntry(id: string): void {
  if (!(id in store.entries)) return;
  const entries = { ...store.entries };
  delete entries[id];
  commit({ entries, streaming: streamingIds(entries) });
}

function patchEntry(id: string, change: (entry: ChatEntry) => ChatEntry): void {
  const entry = store.entries[id];
  if (!entry) return;
  const next = change(entry);
  if (next !== entry) putEntry(next);
}

function setFailure(id: string, message: string): void {
  commit({ failures: { ...store.failures, [id]: message } });
}

function clearFailure(id: string): void {
  if (!(id in store.failures)) return;
  const failures = { ...store.failures };
  delete failures[id];
  commit({ failures });
}

function withMessage(chat: Chat, messageId: string, patch: Partial<ChatMessage>): Chat {
  return { ...chat, messages: chat.messages.map((message) => (message.id === messageId ? { ...message, ...patch } : message)) };
}

export function useChatEntry(id: string | null): ChatEntry | null {
  return useSyncExternalStore(subscribe, () => (id === null ? null : (store.entries[id] ?? null)));
}

export function useChatFailure(id: string | null): string | null {
  return useSyncExternalStore(subscribe, () => (id === null ? null : (store.failures[id] ?? null)));
}

/** Ids of the chats a reply is streaming into. */
export function useStreamingChats(): readonly string[] {
  return useSyncExternalStore(subscribe, () => store.streaming);
}

/** Every stored chat, newest first. */
export function useChatSummaries() {
  return useQuery<ChatSummary[]>({ queryKey: CHATS_KEY, queryFn: api.chats, staleTime: 15_000 });
}

// ---------------------------------------------------------------------------------------------
// Chats

/** A random UUID; crypto.randomUUID exists only in secure contexts (not plain http on a LAN). */
export function newId(): string {
  if (typeof crypto.randomUUID === "function") return crypto.randomUUID();
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  bytes[6] = (bytes[6] & 0x0f) | 0x40;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;
  const hex = Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

export function chatTitle(chat: { title: string }): string {
  return chat.title.length > 0 ? chat.title : "New chat";
}

/** A chat's title from its first message: the first ~60 characters, cut at a word. */
export function autoTitle(text: string, images: number): string {
  const flat = text.replace(/\s+/g, " ").trim();
  if (flat.length === 0) return images > 0 ? "Image" : "New chat";
  if (flat.length <= TITLE_CHARS) return flat;
  const cut = flat.slice(0, TITLE_CHARS);
  const space = cut.lastIndexOf(" ");
  return `${(space > TITLE_CHARS / 2 ? cut.slice(0, space) : cut).trimEnd()}…`;
}

/** How the page names a configuration: its model id, or "Loaded model". */
export function modelName(library: Library | null | undefined, configId: string | null): string {
  if (configId === null) return "Loaded model";
  const found = findConfig(library, configId);
  if (found) return found.config.alias;
  return library ? "Unavailable model" : configId;
}

/** The model id a request names for a configuration (null: whatever is loaded). */
function requestModel(library: Library | undefined, configId: string | null): string | null {
  if (configId === null) return null;
  return findConfig(library, configId)?.config.alias ?? configId;
}

/** The configuration that answered: the chosen one, else the one the server answered as. */
function answeringConfig(library: Library | undefined, chosen: string | null, answeredAs: string | null): string | null {
  if (chosen !== null) return chosen;
  if (answeredAs !== null) {
    const match = allConfigs(library).find(({ config }) => config.alias === answeredAs);
    if (match) return match.config.id;
  }
  const instance = getLive().instance;
  return instance && instance.state === "ready" ? instance.model_key : null;
}

export interface ImageSupport {
  allowed: boolean;
  /** Why images cannot be sent (null while the library loads). */
  note: string | null;
}

/** Whether the chat's model takes images: its configuration was started with --vision. */
export function imageSupport(
  library: Library | null | undefined,
  configId: string | null,
  instance: InstanceInfo | null,
): ImageSupport {
  if (!library) return { allowed: false, note: null };
  let found: ReturnType<typeof findConfig>;
  if (configId !== null) {
    found = findConfig(library, configId);
    if (!found) return { allowed: false, note: "This chat's model is no longer in the library" };
  } else {
    const loaded = instance && (instance.state === "ready" || instance.state === "starting") ? instance.model_key : null;
    found = findConfig(library, loaded);
    if (!found) return { allowed: false, note: "No model is loaded: images need one started with --vision" };
  }
  return found.config.vision
    ? { allowed: true, note: null }
    : { allowed: false, note: `${found.config.alias} was not started with --vision: images are refused` };
}

function normalizeOptions(raw: unknown): ChatOptions {
  const value = (typeof raw === "object" && raw !== null ? raw : {}) as Partial<Record<keyof ChatOptions, unknown>>;
  return {
    temperature: num(value.temperature),
    max_tokens: num(value.max_tokens),
    thinking: typeof value.thinking === "boolean" ? value.thinking : DEFAULT_OPTIONS.thinking,
  };
}

function normalizeMessage(raw: unknown): ChatMessage | null {
  if (typeof raw !== "object" || raw === null) return null;
  const value = raw as Partial<Record<keyof ChatMessage, unknown>>;
  const role = value.role === "user" ? "user" : value.role === "assistant" ? "assistant" : null;
  if (role === null) return null;
  return {
    id: typeof value.id === "string" && value.id.length > 0 ? value.id : newId(),
    role,
    content: typeof value.content === "string" ? value.content : "",
    images: Array.isArray(value.images) ? value.images.filter((image): image is string => typeof image === "string") : [],
    reasoning: typeof value.reasoning === "string" && value.reasoning.length > 0 ? value.reasoning : null,
    stats: typeof value.stats === "object" && value.stats !== null ? (value.stats as ChatStats) : null,
    model: typeof value.model === "string" && value.model.length > 0 ? value.model : null,
    error: typeof value.error === "string" && value.error.length > 0 ? value.error : null,
    t: num(value.t) ?? 0,
  };
}

/** A stored chat as this page uses it (files written by hand or an older launcher may lack
 * fields). */
function normalizeChat(raw: Chat, id: string): Chat {
  const now = Date.now();
  return {
    id,
    title: typeof raw.title === "string" ? raw.title : "",
    model: typeof raw.model === "string" && raw.model.length > 0 ? raw.model : null,
    created_at: num(raw.created_at) ?? now,
    updated_at: num(raw.updated_at) ?? now,
    system_prompt: typeof raw.system_prompt === "string" ? raw.system_prompt : "",
    options: normalizeOptions(raw.options),
    messages: Array.isArray(raw.messages)
      ? raw.messages.map(normalizeMessage).filter((message): message is ChatMessage => message !== null)
      : [],
  };
}

type ChatDefaults = Pick<Chat, "model" | "options">;

function readDefaults(): ChatDefaults {
  const text = readStorage(DEFAULTS_KEY);
  if (text) {
    try {
      const value = JSON.parse(text) as { model?: unknown; options?: unknown };
      return { model: typeof value.model === "string" ? value.model : null, options: normalizeOptions(value.options) };
    } catch {
      // ignore a damaged entry
    }
  }
  return { model: null, options: { ...DEFAULT_OPTIONS } };
}

function rememberDefaults(chat: ChatDefaults): void {
  writeStorage(DEFAULTS_KEY, JSON.stringify({ model: chat.model, options: chat.options }));
}

function blankChat(id: string, defaults: ChatDefaults): Chat {
  const now = Date.now();
  return {
    id,
    title: "",
    model: defaults.model,
    created_at: now,
    updated_at: now,
    system_prompt: "",
    options: { ...defaults.options },
    messages: [],
  };
}

export function summaryOf(chat: Chat): ChatSummary {
  return {
    id: chat.id,
    title: chat.title,
    model: chat.model,
    messages: chat.messages.length,
    created_at: chat.created_at,
    updated_at: chat.updated_at,
  };
}

function isSummary(value: unknown): value is ChatSummary {
  return typeof value === "object" && value !== null && typeof (value as { id?: unknown }).id === "string";
}

/** A new chat, kept in the page until something is sent in it. Starts with the model and options
 * of `from` (else those of the last chat used). Returns its id. */
export function createDraft(from?: ChatDefaults | null): string {
  const chat = blankChat(newId(), from ?? readDefaults());
  putEntry({ chat, saved: false, reply: null });
  return chat.id;
}

/** Drops a new chat nothing was sent in (the page moved to another). */
export function discardDraft(id: string): void {
  const entry = store.entries[id];
  if (entry && !entry.saved && entry.chat.messages.length === 0 && entry.reply === null) dropEntry(id);
}

/** The chat the page showed last (this visit). */
let lastOpen: string | null = null;

export function lastOpenChat(): string | null {
  return lastOpen;
}

const opening = new Map<string, Promise<void>>();

function hasLocalChanges(id: string, entry: ChatEntry): boolean {
  return !entry.saved || entry.reply !== null || saveTimers.has(id) || saving.has(id);
}

/** Reads a stored chat into the page (again when it is open already and nothing local is
 * pending, since another browser may have changed it). An id the launcher does not know becomes a
 * new chat (a page reloaded on a chat nothing was sent in yet). */
export function openChat(id: string): Promise<void> {
  lastOpen = id;
  const pending = opening.get(id);
  if (pending) return pending;
  const known = store.entries[id];
  if (known && hasLocalChanges(id, known)) return Promise.resolve();
  const promise = api
    .chat(id)
    .then(
      (raw) => {
        const current = store.entries[id];
        const chat = normalizeChat(raw, id);
        if (current && (hasLocalChanges(id, current) || current.chat.updated_at >= chat.updated_at)) return;
        putEntry({ chat, saved: true, reply: null });
        clearFailure(id);
      },
      (error: unknown) => {
        if (store.entries[id]) return;
        if (error instanceof ApiFailure && error.status === 404) {
          putEntry({ chat: blankChat(id, readDefaults()), saved: false, reply: null });
          clearFailure(id);
        } else {
          setFailure(id, errorMessage(error));
        }
      },
    )
    .finally(() => opening.delete(id));
  opening.set(id, promise);
  return promise;
}

export function retryOpen(id: string): Promise<void> {
  clearFailure(id);
  return openChat(id);
}

// ---------------------------------------------------------------------------------------------
// Saving

const saveTimers = new Map<string, number>();
const saving = new Map<string, { again: boolean; done: Promise<void> }>();

function cancelSaveTimer(id: string): void {
  const timer = saveTimers.get(id);
  if (timer !== undefined) {
    window.clearTimeout(timer);
    saveTimers.delete(id);
  }
}

/** The chat as stored: a streaming reply is saved once it ends. */
function storedChat(entry: ChatEntry): Chat {
  if (entry.reply === null) return entry.chat;
  const streamingId = entry.reply.message_id;
  return { ...entry.chat, messages: entry.chat.messages.filter((message) => message.id !== streamingId) };
}

function upsertSummary(qc: QueryClient, summary: ChatSummary): void {
  const list = qc.getQueryData<ChatSummary[]>(CHATS_KEY);
  if (!list) {
    void qc.invalidateQueries({ queryKey: CHATS_KEY });
    return;
  }
  const next = [summary, ...list.filter((item) => item.id !== summary.id)].sort((a, b) => b.updated_at - a.updated_at);
  qc.setQueryData<ChatSummary[]>(CHATS_KEY, next);
}

/** Saves the chat now. Saves never overlap: a change during one is saved right after it. */
function persist(qc: QueryClient, id: string): Promise<void> {
  cancelSaveTimer(id);
  const running = saving.get(id);
  if (running) {
    running.again = true;
    return running.done;
  }
  const slot = { again: false, done: Promise.resolve() };
  slot.done = (async () => {
    try {
      do {
        slot.again = false;
        const entry = store.entries[id];
        if (!entry || (!entry.saved && entry.chat.messages.length === 0)) break;
        const chat = { ...storedChat(entry), updated_at: Date.now() };
        const answer = await api.saveChat(chat);
        // Deleted meanwhile: removeChat deletes it once this save is done.
        if (!store.entries[id]) break;
        patchEntry(id, (current) => ({ ...current, saved: true, chat: { ...current.chat, updated_at: chat.updated_at } }));
        upsertSummary(qc, isSummary(answer) ? answer : summaryOf(chat));
      } while (slot.again);
    } catch (error) {
      toast.error(`The chat was not saved: ${errorMessage(error)}`);
    }
  })().finally(() => saving.delete(id));
  saving.set(id, slot);
  return slot.done;
}

function persistLater(qc: QueryClient, id: string): void {
  cancelSaveTimer(id);
  saveTimers.set(
    id,
    window.setTimeout(() => {
      saveTimers.delete(id);
      void persist(qc, id);
    }, SAVE_DELAY_MS),
  );
}

function updateChat(qc: QueryClient, id: string, change: (chat: Chat) => Chat, when: "now" | "later"): void {
  const entry = store.entries[id];
  if (!entry) return;
  const chat = change(entry.chat);
  if (chat === entry.chat) return;
  putEntry({ ...entry, chat });
  if (chat.model !== entry.chat.model || chat.options !== entry.chat.options) rememberDefaults(chat);
  if (entry.saved) upsertSummary(qc, summaryOf({ ...storedChat({ ...entry, chat }), updated_at: Date.now() }));
  if (when === "now") void persist(qc, id);
  else persistLater(qc, id);
}

export function setChatModel(qc: QueryClient, id: string, model: string | null): void {
  updateChat(qc, id, (chat) => (chat.model === model ? chat : { ...chat, model }), "now");
}

export function setChatSettings(
  qc: QueryClient,
  id: string,
  patch: { system_prompt?: string; options?: ChatOptions },
): void {
  updateChat(qc, id, (chat) => ({ ...chat, ...patch }), "later");
}

export async function renameChat(qc: QueryClient, id: string, title: string): Promise<void> {
  const text = title.replace(/\s+/g, " ").trim();
  if (text.length === 0) return;
  if (!store.entries[id]) await openChat(id);
  if (!store.entries[id]) {
    toast.error(store.failures[id] ?? "The chat could not be opened");
    return;
  }
  updateChat(qc, id, (chat) => (chat.title === text ? chat : { ...chat, title: text }), "now");
}

export async function removeChat(qc: QueryClient, id: string): Promise<void> {
  stopReply(id);
  cancelSaveTimer(id);
  const entry = store.entries[id];
  dropEntry(id);
  clearFailure(id);
  qc.setQueryData<ChatSummary[]>(CHATS_KEY, (list) => list?.filter((item) => item.id !== id));
  // A save still running would store it again after the delete.
  await saving.get(id)?.done;
  if (entry && !entry.saved && entry.chat.messages.length === 0) return;
  try {
    await api.deleteChat(id);
  } catch (error) {
    if (error instanceof ApiFailure && error.status === 404) return;
    toast.error(`The chat was not deleted: ${errorMessage(error)}`);
    void qc.invalidateQueries({ queryKey: CHATS_KEY });
  }
}

export function deleteMessage(qc: QueryClient, id: string, messageId: string): void {
  if (store.entries[id]?.reply) return;
  updateChat(qc, id, (chat) => ({ ...chat, messages: chat.messages.filter((message) => message.id !== messageId) }), "now");
}

// ---------------------------------------------------------------------------------------------
// Sending

const controllers = new Map<string, AbortController>();

/** Stops the reply streaming into a chat; what arrived is kept. */
export function stopReply(id: string): void {
  controllers.get(id)?.abort();
}

export function sendMessage(qc: QueryClient, id: string, text: string, images: string[]): void {
  const entry = store.entries[id];
  if (!entry || entry.reply) return;
  const user: ChatMessage = {
    id: newId(),
    role: "user",
    content: text,
    images,
    reasoning: null,
    stats: null,
    model: null,
    error: null,
    t: Date.now(),
  };
  const title = entry.chat.title.length > 0 ? entry.chat.title : autoTitle(text, images.length);
  startReply(qc, { ...entry.chat, title, messages: [...entry.chat.messages, user] });
}

/** Answers the last user message again (replacing the reply after it). */
export function regenerate(qc: QueryClient, id: string): void {
  const entry = store.entries[id];
  if (!entry || entry.reply) return;
  const messages = [...entry.chat.messages];
  while (messages.length > 0 && messages[messages.length - 1].role === "assistant") messages.pop();
  if (messages.length === 0) return;
  startReply(qc, { ...entry.chat, messages });
}

/** Replaces a user message's text, drops everything after it and asks again. */
export function editAndResend(qc: QueryClient, id: string, messageId: string, text: string): void {
  const entry = store.entries[id];
  if (!entry || entry.reply) return;
  const { chat } = entry;
  const index = chat.messages.findIndex((message) => message.id === messageId && message.role === "user");
  if (index < 0) return;
  const old = chat.messages[index];
  const firstUser = chat.messages.find((message) => message.role === "user");
  const title =
    firstUser?.id === messageId && chat.title === autoTitle(old.content, old.images.length)
      ? autoTitle(text, old.images.length)
      : chat.title;
  startReply(qc, { ...chat, title, messages: [...chat.messages.slice(0, index), { ...old, content: text, t: Date.now() }] });
}

type RequestContent = string | ({ type: "text"; text: string } | { type: "image_url"; image_url: { url: string } })[];

function requestMessages(systemPrompt: string, history: readonly ChatMessage[]): { role: string; content: RequestContent }[] {
  const messages: { role: string; content: RequestContent }[] = [];
  if (systemPrompt.trim().length > 0) messages.push({ role: "system", content: systemPrompt });
  for (const message of history) {
    if (message.role === "user") {
      if (message.images.length === 0) {
        messages.push({ role: "user", content: message.content });
      } else {
        const parts: Exclude<RequestContent, string> = [];
        if (message.content.length > 0) parts.push({ type: "text", text: message.content });
        for (const url of message.images) parts.push({ type: "image_url", image_url: { url } });
        messages.push({ role: "user", content: parts });
      }
    } else if (message.content.length > 0) {
      // Failed or empty replies are left out; thinking is not sent back.
      messages.push({ role: "assistant", content: message.content });
    }
  }
  return messages;
}

function requestBody(chat: Chat, history: readonly ChatMessage[], model: string | null): Record<string, unknown> {
  const body: Record<string, unknown> = {
    messages: requestMessages(chat.system_prompt, history),
    stream: true,
    stream_options: { include_usage: true },
    return_progress: true,
    timings_per_token: true,
    chat_template_kwargs: { enable_thinking: chat.options.thinking },
  };
  if (model !== null) body.model = model;
  if (chat.options.temperature !== null) body.temperature = chat.options.temperature;
  if (chat.options.max_tokens !== null) body.max_tokens = chat.options.max_tokens;
  return body;
}

/** Adds an empty assistant message to `chat` (the entry's new content), saves the user's turn and
 * streams the reply into it. */
function startReply(qc: QueryClient, chat: Chat): void {
  const entry = store.entries[chat.id];
  if (!entry || entry.reply) return;
  const library = qc.getQueryData<Library>(["library"]);
  const history = chat.messages;
  const assistant: ChatMessage = {
    id: newId(),
    role: "assistant",
    content: "",
    images: [],
    reasoning: null,
    stats: null,
    model: chat.model,
    error: null,
    t: Date.now(),
  };
  const tracker = new ReplyTracker(Date.now());
  putEntry({
    ...entry,
    chat: { ...chat, messages: [...history, assistant] },
    reply: { message_id: assistant.id, live: tracker.live() },
  });
  rememberDefaults(chat);
  void persist(qc, chat.id);
  void streamReply(qc, chat.id, assistant.id, tracker, requestBody(chat, history, requestModel(library, chat.model)));
}

async function streamReply(
  qc: QueryClient,
  chatId: string,
  messageId: string,
  tracker: ReplyTracker,
  body: Record<string, unknown>,
): Promise<void> {
  const controller = new AbortController();
  controllers.set(chatId, controller);
  const started = performance.now();
  let flushTimer: number | undefined;
  const flush = (): void => {
    flushTimer = undefined;
    patchEntry(chatId, (entry) =>
      entry.reply?.message_id !== messageId
        ? entry
        : {
            ...entry,
            chat: withMessage(entry.chat, messageId, {
              content: tracker.content,
              reasoning: tracker.reasoning.length > 0 ? tracker.reasoning : null,
            }),
            reply: { message_id: messageId, live: tracker.live() },
          },
    );
  };

  let error: string | null = null;
  let received = false;
  try {
    const response = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    if (!response.ok) {
      error = await responseError(response);
    } else if (!response.body) {
      error = "The server answered without a stream";
    } else {
      for await (const data of sseData(response.body)) {
        if (data === "[DONE]") break;
        let chunk: StreamChunk;
        try {
          chunk = JSON.parse(data) as StreamChunk;
        } catch {
          continue;
        }
        if (typeof chunk !== "object" || chunk === null) continue;
        received = true;
        error = tracker.add(chunk, performance.now() - started);
        if (error !== null) break;
        if (flushTimer === undefined) flushTimer = window.setTimeout(flush, FLUSH_MS);
      }
    }
  } catch (caught) {
    if (!controller.signal.aborted) {
      error = caught instanceof TypeError
        ? received
          ? "The connection to the launcher was lost"
          : "The launcher is not reachable"
        : errorMessage(caught);
    }
  } finally {
    controllers.delete(chatId);
    if (flushTimer !== undefined) window.clearTimeout(flushTimer);
  }
  if (error === null && controller.signal.aborted && tracker.empty) error = "Stopped before the reply began.";
  if (error === null && !controller.signal.aborted && tracker.empty) error = "The model sent an empty reply.";

  const totalMs = performance.now() - started;
  const library = qc.getQueryData<Library>(["library"]);
  patchEntry(chatId, (entry) => {
    if (entry.reply?.message_id !== messageId) return entry;
    const chosen = entry.chat.messages.find((message) => message.id === messageId)?.model ?? null;
    return {
      ...entry,
      reply: null,
      chat: withMessage(entry.chat, messageId, {
        content: tracker.content,
        reasoning: tracker.reasoning.length > 0 ? tracker.reasoning : null,
        stats: tracker.stats(totalMs),
        model: answeringConfig(library, chosen, tracker.model),
        error,
      }),
    };
  });
  await persist(qc, chatId);
}

// ---------------------------------------------------------------------------------------------
// Images

function readDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error ?? new Error("The file could not be read"));
    reader.readAsDataURL(file);
  });
}

/** Reads image files as data: URLs; PNG, JPEG, WebP and GIF up to 20 MB each. */
export async function readImages(files: readonly File[]): Promise<{ images: string[]; rejected: string[] }> {
  const images: string[] = [];
  const rejected: string[] = [];
  for (const file of files) {
    const name = file.name || "The image";
    if (!IMAGE_TYPES.has(file.type)) {
      rejected.push(`${name} is not a PNG, JPEG, WebP or GIF image`);
    } else if (file.size > MAX_IMAGE_BYTES) {
      rejected.push(`${name} is larger than 20 MB`);
    } else {
      try {
        images.push(await readDataUrl(file));
      } catch (error) {
        rejected.push(`${name}: ${errorMessage(error)}`);
      }
    }
  }
  return { images, rejected };
}
