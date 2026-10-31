// The watchdog's latest incident (InstanceInfo.incident): a banner on the dashboard and a line in
// the sidebar's status card. Dismissing it remembers the incident's time in this browser, so the
// next incident shows again.

import { useState, useSyncExternalStore } from "react";
import { Link } from "react-router";
import { useMutation } from "@tanstack/react-query";
import {
  Check,
  ChevronDown,
  Gpu,
  Hourglass,
  MemoryStick,
  OctagonX,
  Play,
  RotateCw,
  ScrollText,
  TimerOff,
  TriangleAlert,
  Unplug,
  X,
  type LucideIcon,
} from "lucide-react";
import { toast } from "sonner";

import { useLoadModel } from "@/components/useModelActions";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Hint } from "@/components/ui/tooltip";
import { api } from "@/lib/api";
import type { Incident, IncidentKind } from "@/lib/api-types";
import { useLive } from "@/lib/events";
import { formatRelative } from "@/lib/format";
import { useNow } from "@/lib/hooks";
import { cn, readStorage, writeStorage } from "@/lib/utils";

interface KindInfo {
  title: string;
  /** For the sidebar's one line. */
  short: string;
  icon: LucideIcon;
  tone: "destructive" | "warning";
}

const KINDS: Record<IncidentKind, KindInfo> = {
  crash: { title: "infernix-serve crashed", short: "Server crashed", icon: OctagonX, tone: "destructive" },
  hang: { title: "infernix-serve stopped responding", short: "Server stopped responding", icon: TimerOff, tone: "warning" },
  stall: { title: "Requests stopped progressing", short: "Requests stalled", icon: Hourglass, tone: "warning" },
  vram: { title: "The model did not fit in GPU memory", short: "Did not fit in GPU memory", icon: Gpu, tone: "destructive" },
  ram: { title: "The model ran out of system RAM", short: "Ran out of system RAM", icon: MemoryStick, tone: "destructive" },
  port: { title: "infernix-serve could not use its port", short: "Port in use", icon: Unplug, tone: "destructive" },
};

const UNKNOWN_KIND: KindInfo = { title: "infernix-serve failed", short: "Server failed", icon: TriangleAlert, tone: "destructive" };

function kindInfo(incident: Incident): KindInfo {
  return KINDS[incident.kind] ?? UNKNOWN_KIND;
}

const TONE_TEXT: Record<KindInfo["tone"], string> = {
  destructive: "text-critical",
  warning: "text-warning",
};

const DISMISSED_KEY = "infernix.incident.dismissed";
let dismissed = readStorage(DISMISSED_KEY);
const dismissListeners = new Set<() => void>();

function subscribeDismissed(listener: () => void): () => void {
  dismissListeners.add(listener);
  return () => {
    dismissListeners.delete(listener);
  };
}

function dismissIncident(t: number): void {
  dismissed = String(t);
  writeStorage(DISMISSED_KEY, dismissed);
  for (const listener of dismissListeners) listener();
}

/** The watchdog's latest incident, unless it was dismissed in this browser. */
export function useActiveIncident(): Incident | null {
  const incident = useLive((live) => live.instance?.incident ?? null);
  const hidden = useSyncExternalStore(
    subscribeDismissed,
    () => dismissed,
    () => dismissed,
  );
  return incident !== null && String(incident.t) !== hidden ? incident : null;
}

function sentence(text: string): string {
  return text.length > 0 ? text.charAt(0).toUpperCase() + text.slice(1) : text;
}

/** What happened, what the launcher did and, for a load that did not fit, the changes that may
 * make it fit (each one applied to the model's launch parameters with a click). Render it with
 * `key={incident.t}` so a new incident starts with fresh state. */
export function IncidentBanner({ incident, className }: { incident: Incident; className?: string }) {
  const now = useNow(5000);
  const kind = kindInfo(incident);
  const Icon = kind.icon;
  const suggestions = incident.suggestions ?? [];
  const [showSuggestions, setShowSuggestions] = useState(true);
  const [applied, setApplied] = useState<ReadonlySet<number>>(() => new Set());
  const { load, dialog, pending: loading } = useLoadModel();

  const apply = useMutation({
    mutationFn: ({ args }: { index: number; args: string[]; andLoad: boolean }) =>
      api.saveModelConfig(incident.model_key, { args }),
    onSuccess: (_config, { index, andLoad }) => {
      setApplied((old) => new Set(old).add(index));
      toast.success(`${incident.alias}: ${suggestions[index]?.title ?? "change"} applied`);
      if (andLoad) load(incident.model_key, incident.alias);
    },
  });
  const busy = apply.isPending || loading;

  return (
    <Alert variant={kind.tone} className={cn("pr-11", className)}>
      <Icon />
      <AlertTitle className="flex flex-wrap items-baseline gap-x-2">
        <span>{kind.title}</span>
        <span className="text-xs font-normal text-muted-foreground">
          {incident.alias ? `${incident.alias} · ` : null}
          <Hint label={new Date(incident.t).toLocaleString()}>
            <time dateTime={new Date(incident.t).toISOString()} className="tabular-nums">
              {formatRelative(incident.t, now)}
            </time>
          </Hint>
        </span>
      </AlertTitle>
      <AlertDescription>
        <p className="break-words">{incident.message}</p>
        <div className="flex w-full flex-wrap items-center gap-x-4 gap-y-1 text-xs">
          {incident.action ? (
            <span className="inline-flex min-w-0 items-center gap-1">
              {incident.restarted ? <RotateCw className="size-3.5 shrink-0" /> : null}
              {sentence(incident.action)}
            </span>
          ) : null}
          <Link to="/logs?tab=events" className="inline-flex items-center gap-1 underline-offset-4 hover:text-foreground hover:underline">
            <ScrollText className="size-3.5" />
            Launcher events
          </Link>
        </div>
        {suggestions.length > 0 ? (
          <div className="mt-1 w-full">
            <button
              type="button"
              aria-expanded={showSuggestions}
              onClick={() => setShowSuggestions((open) => !open)}
              className="inline-flex items-center gap-1 text-xs font-medium text-foreground hover:underline"
            >
              <ChevronDown className={cn("size-3.5 transition-transform", !showSuggestions && "-rotate-90")} />
              {suggestions.length === 1 ? "1 suggestion" : `${suggestions.length} suggestions, most effective first`}
            </button>
            {showSuggestions ? (
              <ul className="mt-1.5 flex flex-col gap-1.5">
                {suggestions.map((suggestion, index) => {
                  const args = suggestion.args;
                  return (
                    <li
                      key={`${index}-${suggestion.title}`}
                      className="flex flex-col gap-2 rounded-md border bg-background/60 px-3 py-2 sm:flex-row sm:items-center"
                    >
                      <div className="min-w-0 flex-1">
                        <div className="flex flex-wrap items-center gap-1.5 text-sm font-medium text-foreground">
                          {suggestion.title}
                          {applied.has(index) ? (
                            <Badge variant="accent">
                              <Check />
                              applied
                            </Badge>
                          ) : null}
                        </div>
                        <div className="text-xs">{suggestion.detail}</div>
                      </div>
                      {args ? (
                        <div className="flex shrink-0 gap-1.5">
                          <Button
                            size="xs"
                            variant="outline"
                            disabled={busy}
                            onClick={() => apply.mutate({ index, args, andLoad: false })}
                          >
                            Apply
                          </Button>
                          <Button size="xs" disabled={busy} onClick={() => apply.mutate({ index, args, andLoad: true })}>
                            <Play />
                            Apply and load
                          </Button>
                        </div>
                      ) : null}
                    </li>
                  );
                })}
              </ul>
            ) : null}
          </div>
        ) : null}
      </AlertDescription>
      <Button
        variant="ghost"
        size="icon-xs"
        className="absolute top-2 right-2"
        aria-label="Dismiss"
        onClick={() => dismissIncident(incident.t)}
      >
        <X />
      </Button>
      {dialog}
    </Alert>
  );
}

/** The incident in one line, for the sidebar's status card. */
export function IncidentLine() {
  const incident = useActiveIncident();
  const now = useNow(5000);
  if (!incident) return null;
  const kind = kindInfo(incident);
  const Icon = kind.icon;
  return (
    <div className={cn("mt-2 flex items-center gap-1.5 text-xs", TONE_TEXT[kind.tone])} title={incident.message}>
      <Icon className="size-3.5 shrink-0" />
      <span className="truncate">{kind.short}</span>
      <span className="ml-auto shrink-0 text-muted-foreground tabular-nums">{formatRelative(incident.t, now)}</span>
    </div>
  );
}

/** The incident as one icon, for the compact sidebar; it links to the dashboard's banner. */
export function IncidentIcon() {
  const incident = useActiveIncident();
  const now = useNow(5000);
  if (!incident) return null;
  const kind = kindInfo(incident);
  const Icon = kind.icon;
  return (
    <Hint label={`${kind.short} · ${formatRelative(incident.t, now)}`} side="right">
      <Link to="/" aria-label={kind.short} className={cn("flex size-8 items-center justify-center rounded-md hover:bg-sidebar-accent/60", TONE_TEXT[kind.tone])}>
        <Icon className="size-4" />
      </Link>
    </Hint>
  );
}
