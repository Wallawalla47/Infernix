// One EventSource to the launcher feeds a small store of live data (telemetry, the server
// instance, inference metrics, the console log, the launcher's event log, proxy activity) and
// writes the slower data (library, settings, downloads, model configs, log paths, update states)
// into the TanStack Query cache. A browser that must sign in (any management call or the stream
// answers 401 sign_in) stops the stream until signedIn().

import type { QueryClient } from "@tanstack/react-query";
import { useSyncExternalStore } from "react";
import { toast } from "sonner";

import { INFERNIX_RELEASE_KEY, MODEL_UPDATES_KEY, isSignInFailure } from "./admin";
import { api } from "./api";
import type {
  Activity,
  EventRecord,
  InfernixRelease,
  InstanceInfo,
  LauncherEvents,
  LauncherState,
  Library,
  LogLine,
  LogPaths,
  MetricsSnapshot,
  ModelUpdates,
  Settings,
  SystemSample,
} from "./api-types";

export const HISTORY_MS = 10 * 60 * 1000;
export const LOG_LIMIT = 10_000;
/** Event-log records kept in the page (the launcher's own ring holds as many). */
export const EVENT_LIMIT = 5000;

export interface LiveState {
  /** The event stream is open. */
  connected: boolean;
  /** It has been open at least once (distinguishes "starting up" from "lost"). */
  everConnected: boolean;
  /** When the stream last failed, for the offline banner's grace period. */
  disconnectedAt: number | null;
  launcher: LauncherState | null;
  system: SystemSample | null;
  history: SystemSample[];
  instance: InstanceInfo | null;
  metrics: MetricsSnapshot | null;
  logs: LogLine[];
  /** The launcher's event log (its own events and one record per API request), oldest first. */
  eventLog: EventRecord[];
  activity: Activity | null;
  /** The launcher wants an admin key from this browser: the page shows the sign-in card. */
  signInRequired: boolean;
}

let state: LiveState = {
  connected: false,
  everConnected: false,
  disconnectedAt: null,
  launcher: null,
  system: null,
  history: [],
  instance: null,
  metrics: null,
  logs: [],
  eventLog: [],
  activity: null,
  signInRequired: false,
};

const listeners = new Set<() => void>();

function update(patch: Partial<LiveState>): void {
  state = { ...state, ...patch };
  for (const listener of listeners) listener();
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

export function getLive(): LiveState {
  return state;
}

/** Reads one field (or another stable reference) of the live store. */
export function useLive<T>(selector: (live: LiveState) => T): T {
  return useSyncExternalStore(
    subscribe,
    () => selector(state),
    () => selector(state),
  );
}

function appendSamples(samples: readonly SystemSample[]): void {
  if (samples.length === 0) return;
  const newest = samples[samples.length - 1];
  const cutoff = newest.t - HISTORY_MS;
  const merged = state.history.filter((sample) => sample.t >= cutoff && sample.t < samples[0].t);
  for (const sample of samples) {
    if (sample.t >= cutoff) merged.push(sample);
  }
  update({ system: newest, history: merged });
}

function appendLogs(lines: readonly LogLine[]): void {
  if (lines.length === 0) return;
  const lastSeq = state.logs.length > 0 ? state.logs[state.logs.length - 1].seq : -1;
  const fresh = lines.filter((line) => line.seq > lastSeq);
  if (fresh.length === 0) return;
  const merged = state.logs.concat(fresh);
  update({ logs: merged.length > LOG_LIMIT ? merged.slice(merged.length - LOG_LIMIT) : merged });
}

/** Drops the visible console history (the launcher keeps its own copy). */
export function clearLogView(): void {
  update({ logs: [] });
}

/** `base` plus `records` in sequence order, without duplicates, keeping the newest EVENT_LIMIT.
 * The page read at connect time and the stream overlap, and either can arrive first. */
function mergeEvents(base: readonly EventRecord[], records: readonly EventRecord[]): EventRecord[] {
  let lastSeq = base.length > 0 ? base[base.length - 1].seq : 0;
  let ordered = true;
  for (const record of records) {
    if (record.seq <= lastSeq) {
      ordered = false;
      break;
    }
    lastSeq = record.seq;
  }
  let merged: EventRecord[];
  if (ordered) {
    merged = base.concat(records);
  } else {
    const bySeq = new Map<number, EventRecord>();
    for (const record of base) bySeq.set(record.seq, record);
    for (const record of records) bySeq.set(record.seq, record);
    merged = Array.from(bySeq.values()).sort((a, b) => a.seq - b.seq);
  }
  return merged.length > EVENT_LIMIT ? merged.slice(merged.length - EVENT_LIMIT) : merged;
}

// Each hello re-reads the newest event-log page and replaces the history with it: a restarted
// launcher numbers its records afresh, so sequence numbers alone cannot join the two runs. The
// records the stream delivers between the hello and that page are kept aside and merged in.
let helloCount = 0;
let eventsSynced = false;
let streamedEvents: EventRecord[] = [];

function appendEvents(records: readonly EventRecord[]): void {
  if (records.length === 0) return;
  if (!eventsSynced) streamedEvents = streamedEvents.concat(records);
  update({ eventLog: mergeEvents(state.eventLog, records) });
}

function patchState(qc: QueryClient, patch: Partial<LauncherState>): void {
  qc.setQueryData<LauncherState>(["state"], (old) => (old ? { ...old, ...patch } : old));
  if (state.launcher) update({ launcher: { ...state.launcher, ...patch } });
}

async function seed(qc: QueryClient, hello: number): Promise<void> {
  const lastSeq = state.logs.length > 0 ? state.logs[state.logs.length - 1].seq : undefined;
  const [telemetry, logs, metrics, activity, events] = await Promise.allSettled([
    api.telemetry(),
    lastSeq === undefined ? api.logs({ limit: LOG_LIMIT }) : api.logs({ after: lastSeq, limit: LOG_LIMIT }),
    api.metrics(),
    api.activity(),
    api.eventLog({ limit: EVENT_LIMIT }),
  ]);
  if (telemetry.status === "fulfilled") appendSamples(telemetry.value.samples);
  if (logs.status === "fulfilled") appendLogs(logs.value.lines);
  if (metrics.status === "fulfilled") update({ metrics: metrics.value });
  if (activity.status === "fulfilled") update({ activity: activity.value });
  if (hello === helloCount) {
    // A failed read keeps the history and what the stream has added to it.
    if (events.status === "fulfilled") update({ eventLog: mergeEvents(events.value.lines, streamedEvents) });
    eventsSynced = true;
    streamedEvents = [];
  }
  void qc.invalidateQueries({ queryKey: ["library"] });
  void qc.invalidateQueries({ queryKey: ["downloads"] });
  // A launcher that restarted (an installed update) has new update states.
  void qc.invalidateQueries({ queryKey: INFERNIX_RELEASE_KEY });
  void qc.invalidateQueries({ queryKey: MODEL_UPDATES_KEY });
}

function listen<K extends keyof LauncherEvents>(
  source: EventSource,
  name: K,
  handler: (data: LauncherEvents[K]) => void,
): void {
  source.addEventListener(name, (event: MessageEvent) => {
    let data: LauncherEvents[K];
    try {
      data = JSON.parse(String(event.data)) as LauncherEvents[K];
    } catch {
      return;
    }
    handler(data);
  });
}

function showToast(level: LauncherEvents["toast"]["level"], message: string): void {
  switch (level) {
    case "success":
      toast.success(message);
      break;
    case "warning":
      toast.warning(message);
      break;
    case "error":
      toast.error(message);
      break;
    default:
      toast.info(message);
  }
}

// The running stream's controls, for requireSignIn() and signedIn().
let suspendStream: (() => void) | null = null;
let resumeStream: (() => void) | null = null;

/** The launcher answered 401 sign_in: show the sign-in card and stop the stream (it would only
 * fail again) until signedIn(). */
export function requireSignIn(): void {
  if (state.signInRequired) return;
  update({ signInRequired: true });
  suspendStream?.();
}

/** This browser signed in: reopen the stream and re-read everything. */
export function signedIn(qc: QueryClient): void {
  if (!state.signInRequired) return;
  update({ signInRequired: false });
  void qc.invalidateQueries();
  resumeStream?.();
}

/** Opens the event stream; reconnects with backoff. Returns the function that stops it. */
export function startEvents(qc: QueryClient): () => void {
  let stopped = false;
  let source: EventSource | null = null;
  let timer: number | undefined;
  let attempt = 0;

  const schedule = (): void => {
    if (stopped || state.signInRequired || timer !== undefined) return;
    const delay = Math.min(10_000, 500 * 2 ** attempt);
    attempt += 1;
    timer = window.setTimeout(() => {
      timer = undefined;
      connect();
    }, delay);
  };

  const connect = (): void => {
    if (stopped || state.signInRequired) return;
    const es = new EventSource(`/launcher/api/events`);
    source = es;

    listen(es, "hello", (launcher: LauncherState) => {
      attempt = 0;
      update({
        connected: true,
        everConnected: true,
        disconnectedAt: null,
        launcher,
        instance: launcher.instance,
      });
      qc.setQueryData(["state"], launcher);
      qc.setQueryData(["settings"], launcher.settings);
      helloCount += 1;
      eventsSynced = false;
      streamedEvents = [];
      void seed(qc, helloCount);
    });
    listen(es, "system", (sample: SystemSample) => appendSamples([sample]));
    listen(es, "instance", (instance: InstanceInfo) => {
      update({ instance });
      patchState(qc, { instance });
    });
    listen(es, "metrics", (metrics: MetricsSnapshot | null) => update({ metrics }));
    listen(es, "log", (payload) => appendLogs(payload.lines));
    listen(es, "eventlog", (payload) => appendEvents(payload.lines));
    listen(es, "logs", (logs: LogPaths) => patchState(qc, { logs }));
    listen(es, "library", (library: Library) => {
      qc.setQueryData(["library"], library);
    });
    listen(es, "downloads", (jobs) => {
      qc.setQueryData(["downloads"], jobs);
    });
    listen(es, "config", (config) => {
      // `config.key` is the artifact; `config.id` names this configuration (the artifact key for
      // its default one), so a named configuration never overwrites the default's entries.
      qc.setQueryData(["config", config.id], config);
      qc.setQueryData<Library>(["library"], (old) =>
        old
          ? {
              ...old,
              models: old.models.map((model) => {
                if (model.key !== config.key) return model;
                const known = model.configs.some((item) => item.id === config.id);
                return {
                  ...model,
                  config: config.is_default ? config : model.config,
                  configs: known
                    ? model.configs.map((item) => (item.id === config.id ? config : item))
                    : model.configs.concat(config),
                };
              }),
            }
          : old,
      );
    });
    listen(es, "activity", (activity: Activity) => update({ activity }));
    listen(es, "toast", (payload) => showToast(payload.level, payload.message));
    listen(es, "settings", (settings: Settings) => {
      qc.setQueryData(["settings"], settings);
      patchState(qc, { settings });
      void qc.invalidateQueries({ queryKey: ["state"] });
    });
    listen(es, "infernix_release", (release: InfernixRelease) => {
      qc.setQueryData(INFERNIX_RELEASE_KEY, release);
    });
    listen(es, "model_updates", (updates: ModelUpdates) => {
      qc.setQueryData(MODEL_UPDATES_KEY, updates);
    });

    es.onerror = () => {
      es.close();
      if (source === es) source = null;
      if (state.connected || state.disconnectedAt === null) {
        update({ connected: false, disconnectedAt: Date.now() });
      }
      if (stopped || state.signInRequired) return;
      // EventSource hides the HTTP status: ask the session whether a sign-in is what failed.
      api.session().then(
        (session) => {
          if (session.required && !session.signed_in) requireSignIn();
          else schedule();
        },
        () => schedule(),
      );
    };
  };

  suspendStream = () => {
    if (timer !== undefined) window.clearTimeout(timer);
    timer = undefined;
    source?.close();
    source = null;
    if (state.connected) update({ connected: false, disconnectedAt: Date.now() });
  };
  resumeStream = () => {
    if (stopped || source !== null) return;
    attempt = 0;
    connect();
  };

  // Any management call answering 401 sign_in (a query or an action) asks for a sign-in.
  const unsubscribeQueries = qc.getQueryCache().subscribe((event) => {
    if (event.type === "updated" && event.action.type === "error" && isSignInFailure(event.action.error)) requireSignIn();
  });
  const unsubscribeMutations = qc.getMutationCache().subscribe((event) => {
    if (event.type === "updated" && event.action.type === "error" && isSignInFailure(event.action.error)) requireSignIn();
  });

  connect();
  return () => {
    stopped = true;
    if (timer !== undefined) window.clearTimeout(timer);
    source?.close();
    unsubscribeQueries();
    unsubscribeMutations();
    suspendStream = null;
    resumeStream = null;
  };
}
