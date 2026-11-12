import { useState } from "react";
import { Link, useNavigate } from "react-router";
import { useMutation } from "@tanstack/react-query";
import {
  ArrowRightLeft,
  Boxes,
  Pin,
  Play,
  Power,
  RotateCw,
  ScrollText,
  SlidersHorizontal,
  TriangleAlert,
} from "lucide-react";

import { ConfirmDialog } from "@/components/common";
import { CopyButton } from "@/components/CopyButton";
import { IncidentBanner, useActiveIncident } from "@/components/IncidentBanner";
import { StateBadge } from "@/components/StateBadge";
import { useLoadModel } from "@/components/useModelActions";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Hint } from "@/components/ui/tooltip";
import { api } from "@/lib/api";
import { useLive } from "@/lib/events";
import { formatContext, formatUptime, quantizationLabel } from "@/lib/format";
import { useEndpoints, useLibrary, useNow, useSettings } from "@/lib/hooks";
import { findConfig, findModel } from "@/lib/library";

function Chip({ label, value }: { label: string; value: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 rounded-md border bg-background/50 px-2 py-0.5 text-xs">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-medium tabular-nums">{value}</span>
    </span>
  );
}

/** The Engine's own warning threshold for holding requests without progress (docs/serving.md). */
const NO_PROGRESS_WARNING_S = 30;

/** Time until an idle unload: "in 14 min", "in under a minute", "in 2 h 05 min". */
function untilUnload(ms: number): string {
  if (ms < 60_000) return "in under a minute";
  const minutes = Math.ceil(ms / 60_000);
  if (minutes < 60) return `in ${minutes} min`;
  return `in ${Math.floor(minutes / 60)} h ${String(minutes % 60).padStart(2, "0")} min`;
}

/** The loaded model at a glance: state, startup progress, actions and the endpoint. */
export function StatusHero() {
  const instance = useLive((live) => live.instance);
  const metrics = useLive((live) => live.metrics);
  const incident = useActiveIncident();
  const { data: library } = useLibrary();
  const { data: settings } = useSettings();
  const endpoints = useEndpoints();
  const now = useNow(1000);
  const navigate = useNavigate();
  const { load, dialog, pending: loading } = useLoadModel();
  const [confirmUnload, setConfirmUnload] = useState(false);

  const unload = useMutation({ mutationFn: (force: boolean) => api.unload(force) });
  const restart = useMutation({ mutationFn: () => api.restart() });

  const state = instance?.state ?? "idle";
  // model_key, pending_model and default_model are configuration ids (an artifact key names the
  // artifact's default configuration).
  const loaded = findConfig(library, instance?.model_key);
  const model = loaded?.model ?? findModel(library, instance?.artifact_key ?? instance?.model_key);
  const config = loaded?.config ?? null;
  const pendingKey = instance?.pending_model ?? null;
  const pending = findConfig(library, pendingKey);
  const pendingName = pending?.config.alias ?? pending?.model.name ?? null;
  const defaultKey = settings?.default_model ?? null;
  const defaultConfig = findConfig(library, defaultKey);
  const title = instance?.alias ?? config?.alias ?? instance?.name ?? null;
  const engine = metrics?.engine ?? null;
  const noProgress = metrics?.live.no_progress_s ?? 0;
  const recoveries = metrics?.live.no_progress_recoveries ?? 0;
  const active = state === "ready" || state === "starting";
  // A named configuration's name; the default one is just the model.
  const configName =
    config !== null
      ? config.is_default
        ? null
        : config.name
      : instance?.config_name && instance.config_name !== "Default"
        ? instance.config_name
        : null;
  const idleFor =
    state === "ready" && instance && instance.idle_unload_at !== null && instance.in_flight === 0 && instance.waiting === 0
      ? Math.max(0, instance.idle_unload_at - now)
      : null;
  // An incident about this run of this model already explains the failure (and how to fix a load
  // that did not fit); the plain failure notice returns once it is dismissed.
  const incidentExplainsFailure =
    incident !== null &&
    instance !== null &&
    incident.model_key === instance.model_key &&
    (instance.started_at === null || incident.t >= instance.started_at);

  const requestUnload = () => {
    if ((instance?.in_flight ?? 0) > 0) setConfirmUnload(true);
    else unload.mutate(false);
  };

  return (
    <Card className="hero-glow gap-3 py-4">
      {/* The buttons move below the title rather than cut it short. */}
      <div className="flex flex-wrap items-center justify-between gap-3 px-4">
        <div className="flex min-w-0 flex-auto items-center gap-3">
          <StateBadge state={state} className="shrink-0" />
          <div className="min-w-0">
            <div className="truncate text-lg leading-tight font-semibold tracking-tight md:text-xl" title={title ?? undefined}>
              {title ?? "No model loaded"}
            </div>
            <div className="mt-0.5 flex min-w-0 flex-wrap items-center gap-1.5 text-xs text-muted-foreground">
              {active && instance?.pinned ? (
                <Hint label="Every request goes to this model, whatever it names (Settings, Model routing)">
                  <Badge variant="accent" className="cursor-default">
                    <Pin />
                    Pinned
                  </Badge>
                </Hint>
              ) : null}
              {configName ? (
                <Badge variant="secondary" title="Configuration">
                  {configName}
                </Badge>
              ) : null}
              {model ? (
                <>
                  <Badge variant="outline">{model.family_label}</Badge>
                  {model.quantization ? <Badge variant="muted">{quantizationLabel(model.quantization)}</Badge> : null}
                  {model.name !== title ? <span className="truncate">{model.name}</span> : null}
                </>
              ) : (
                <span>Load a model here, or send a request: the launcher loads whichever model a tool asks for.</span>
              )}
            </div>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {active ? (
            <>
              {model ? (
                <Button variant="outline" size="sm" onClick={() => navigate(`/models/${model.key}`)}>
                  <SlidersHorizontal />
                  Configure
                </Button>
              ) : null}
              <Button variant="outline" size="sm" disabled={restart.isPending} onClick={() => restart.mutate()}>
                <RotateCw />
                Restart
              </Button>
              <Button variant="outline" size="sm" disabled={unload.isPending} onClick={requestUnload}>
                <Power />
                Unload
              </Button>
            </>
          ) : null}
          {state === "failed" && instance?.model_key ? (
            <Button size="sm" disabled={loading} onClick={() => load(instance.model_key ?? "", title ?? "the model")}>
              <RotateCw />
              Retry
            </Button>
          ) : null}
          {state === "idle" && defaultConfig ? (
            <Button size="sm" disabled={loading} onClick={() => load(defaultConfig.config.id, defaultConfig.config.alias)}>
              <Play />
              Load {defaultConfig.config.alias}
            </Button>
          ) : null}
          {state === "idle" || state === "failed" ? (
            <Button variant={state === "idle" && !defaultConfig ? "default" : "outline"} size="sm" asChild>
              <Link to="/models">
                <Boxes />
                Models
              </Link>
            </Button>
          ) : null}
        </div>
      </div>

      {state === "starting" || state === "stopping" ? (
        <div className="flex flex-col gap-1.5 px-4">
          <Progress value={null} tone={state === "starting" ? "primary" : "muted"} />
          <div className="flex items-center justify-between gap-3 text-xs text-muted-foreground">
            <span className="min-w-0 truncate font-mono">
              {instance?.phase || (state === "starting" ? "Starting infernix-serve…" : "Stopping…")}
            </span>
            {instance?.started_at && state === "starting" ? (
              <span className="shrink-0 tabular-nums">{formatUptime(now - instance.started_at)}</span>
            ) : null}
          </div>
        </div>
      ) : null}

      {incident ? (
        <div className="px-4">
          <IncidentBanner key={incident.t} incident={incident} />
        </div>
      ) : null}

      {state === "failed" && instance && !incidentExplainsFailure ? (
        <div className="px-4">
          <Alert variant="destructive">
            <TriangleAlert />
            <AlertTitle>
              infernix-serve stopped{instance.exit_code !== null ? ` (exit status ${instance.exit_code})` : ""}
            </AlertTitle>
            <AlertDescription>
              <p className="font-mono text-xs break-words">{instance.error ?? instance.phase ?? "No message."}</p>
              <span className="inline-flex items-center gap-1 text-xs">
                <ScrollText className="size-3.5" />
                The console below keeps its last lines.
              </span>
            </AlertDescription>
          </Alert>
        </div>
      ) : null}

      {state === "ready" && noProgress >= NO_PROGRESS_WARNING_S ? (
        <div className="px-4">
          <Alert variant="warning">
            <TriangleAlert />
            <AlertTitle>No progress for {Math.round(noProgress)} s</AlertTitle>
            <AlertDescription>
              <p>
                The Engine holds requests and has completed nothing in that time. After two minutes it heals itself: the
                running requests end with an error and waiting ones carry on. The console below has its report.
              </p>
            </AlertDescription>
          </Alert>
        </div>
      ) : null}

      {pendingKey && instance ? (
        <div className="px-4">
          <Alert variant="info">
            <ArrowRightLeft />
            <AlertTitle>Switching to {pendingName ?? "another model"}</AlertTitle>
            <AlertDescription>
              <p>
                Waiting for {instance.in_flight} running request{instance.in_flight === 1 ? "" : "s"} on {title ?? "the current model"}{" "}
                to finish{settings ? ` (at most ${settings.drain_timeout_s} s)` : ""}; {instance.waiting} request
                {instance.waiting === 1 ? " waits" : "s wait"} for the new model.
              </p>
              {pending ? (
                <Button size="xs" variant="outline" className="mt-1" onClick={() => load(pending.config.id, pending.config.alias)}>
                  Switch now
                </Button>
              ) : null}
            </AlertDescription>
          </Alert>
        </div>
      ) : null}

      <div className="flex flex-col gap-2 border-t px-4 pt-3 text-xs lg:flex-row lg:items-center lg:justify-between">
        <div className="flex flex-wrap items-center gap-1.5">
          {state === "ready" && instance?.ready_at ? <Chip label="Up" value={formatUptime(now - instance.ready_at)} /> : null}
          {idleFor !== null ? <Chip label="Unloads" value={`${untilUnload(idleFor)} if idle`} /> : null}
          {engine ? <Chip label="Context" value={formatContext(engine.max_context)} /> : null}
          {engine ? <Chip label="Lanes" value={String(engine.max_concurrency)} /> : null}
          {engine ? <Chip label="KV" value={engine.kv} /> : null}
          {engine && engine.speculative_backend && engine.speculative_backend !== "none" ? (
            <Chip label="Drafter" value={engine.speculative_backend} />
          ) : null}
          {state === "ready" && recoveries > 0 ? (
            <Hint label="Times the Engine made no progress for two minutes while holding requests and healed itself by failing them (this run of the server)">
              <span>
                <Chip label="Stalls healed" value={String(recoveries)} />
              </span>
            </Hint>
          ) : null}
          {!engine && model ? <Chip label="Context" value={formatContext((config ?? model.config).context_length)} /> : null}
        </div>
        <div className="flex min-w-0 flex-wrap items-center gap-x-4 gap-y-1">
          <span className="flex min-w-0 items-center gap-1">
            <span className="text-muted-foreground">OpenAI</span>
            <code className="truncate font-mono">{endpoints.openai}</code>
            <CopyButton text={endpoints.openai} size="icon-xs" toastMessage="OpenAI base URL copied" />
          </span>
          <span className="flex min-w-0 items-center gap-1">
            <span className="text-muted-foreground">Anthropic</span>
            <code className="truncate font-mono">{endpoints.anthropic}</code>
            <CopyButton text={endpoints.anthropic} size="icon-xs" toastMessage="Anthropic base URL copied" />
          </span>
        </div>
      </div>

      {dialog}
      <ConfirmDialog
        open={confirmUnload}
        onOpenChange={setConfirmUnload}
        title="Unload the model?"
        description={`${instance?.in_flight ?? 0} request(s) are running; unloading cancels them. The prefix cache is saved first when it is set to persist.`}
        confirmLabel="Unload"
        destructive
        onConfirm={() => unload.mutate(true)}
      />
    </Card>
  );
}
