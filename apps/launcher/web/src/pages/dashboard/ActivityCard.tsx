import { ArrowRightLeft, CircleCheck, CircleX, Hourglass, LoaderCircle } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Hint } from "@/components/ui/tooltip";
import type { Library, ProxyRecord, RequestProgress } from "@/lib/api-types";
import { useLive } from "@/lib/events";
import { formatClock, formatCount, formatMs } from "@/lib/format";
import { useEndpoints, useLibrary } from "@/lib/hooks";
import { findConfig } from "@/lib/library";
import { cn } from "@/lib/utils";

const SHOWN = 30;

function StateIcon({ record }: { record: ProxyRecord }) {
  switch (record.state) {
    case "waiting":
      return <Hourglass className="size-3.5 text-warning" aria-label="waiting for a model" />;
    case "running":
      return <LoaderCircle className="size-3.5 animate-spin text-series-1" aria-label="running" />;
    case "failed":
      return <CircleX className="size-3.5 text-critical" aria-label="failed" />;
    default:
      return <CircleCheck className="size-3.5 text-success" aria-label="done" />;
  }
}

function shortPath(path: string): string {
  return path.replace(/^\/v1\/v1\//, "/v1/").replace(/^\/v1\//, "");
}

/** Why a request went to its model, when it is not simply the model it named. */
const ROUTED: Record<string, { label: string; hint: string }> = {
  route: { label: "routed", hint: "A model route sent it here (Settings, Model routing)" },
  pinned: { label: "pinned", hint: "The loaded model is pinned: it answers every request" },
  loaded: { label: "loaded model", hint: "The name is unknown, so the loaded (or default) model answered" },
};

/** The model id a request was served by, when it differs from the one it named. */
function servedBy(record: ProxyRecord, library: Library | undefined): string | null {
  if (!record.model_key) return null;
  const served = findConfig(library, record.model_key)?.config.alias ?? null;
  return served !== null && served !== record.model ? served : null;
}

/** A running request's server-side progress: the prompt (prefill) as a bar, then output tokens. */
function ProgressLine({ progress }: { progress: RequestProgress }) {
  const total = progress.prompt_tokens;
  const promptShare = total > 0 ? Math.min(1, progress.processed_tokens / total) : 1;
  const decoding = progress.phase === "decode";
  const queued = progress.phase === "queued";
  return (
    <div className="mt-1.5 flex flex-col gap-1">
      <Progress
        value={queued ? 0 : promptShare * 100}
        tone={decoding ? "muted" : "primary"}
        aria-label={`Prompt ${Math.round(promptShare * 100)}% processed`}
      />
      <div className="flex flex-wrap items-center gap-x-2 text-[11px] text-muted-foreground tabular-nums">
        {queued ? (
          <span>queued in the server</span>
        ) : decoding ? (
          <span className="text-foreground">decoding · {formatCount(progress.output_tokens)} tokens out</span>
        ) : (
          <span className="text-foreground">prefill {Math.floor(promptShare * 100)}%</span>
        )}
        <span>
          prompt {formatCount(progress.processed_tokens)} / {formatCount(total)}
          {progress.cached_tokens > 0 ? ` (${formatCount(progress.cached_tokens)} cached)` : ""}
        </span>
        <span className="ml-auto">{formatMs(progress.elapsed_ms)}</span>
      </div>
    </div>
  );
}

/** What coding tools are asking the launcher for, including requests waiting for a model; running
 * requests show the server's progress on their prompt and output. */
export function ActivityCard({ className }: { className?: string }) {
  const activity = useLive((live) => live.activity);
  const { data: library } = useLibrary();
  const endpoints = useEndpoints();
  const records = activity ? activity.recent.slice(-SHOWN).reverse() : [];

  return (
    <Card className={cn("gap-3", className)}>
      <CardHeader>
        <CardTitle>
          <ArrowRightLeft className="size-4 text-muted-foreground" />
          API activity
        </CardTitle>
        <CardDescription>Requests coding tools send to the launcher</CardDescription>
        <CardAction>
          <Badge variant="outline" className="tabular-nums">
            {activity?.in_flight ?? 0} running
          </Badge>
          <Badge variant={(activity?.waiting ?? 0) > 0 ? "accent" : "outline"} className="tabular-nums">
            {activity?.waiting ?? 0} waiting
          </Badge>
        </CardAction>
      </CardHeader>
      <CardContent className="px-0">
        {records.length === 0 ? (
          <div className="mx-4 flex h-32 flex-col items-center justify-center gap-1 rounded-lg border border-dashed px-4 text-center text-sm text-muted-foreground">
            <span>No API requests yet.</span>
            <span className="text-xs">
              Point a coding tool at <code className="font-mono">{endpoints.openai}</code>
            </span>
          </div>
        ) : (
          <ul className="max-h-[340px] divide-y overflow-y-auto border-y text-sm">
            {records.map((record) => {
              const routed = record.routed_by && record.routed_by !== "name" ? ROUTED[record.routed_by] : undefined;
              const served = servedBy(record, library);
              const who = record.key_name ?? record.client;
              return (
                <li
                  key={record.id}
                  className={cn(
                    "flex items-start gap-2.5 px-4 py-2",
                    record.state === "waiting" && "bg-warning/5",
                    record.state === "failed" && "bg-critical/5",
                  )}
                >
                  <span className="mt-0.5 shrink-0">
                    <StateIcon record={record} />
                  </span>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <span className="truncate font-mono text-xs">
                        {record.method} {shortPath(record.path)}
                      </span>
                      {record.stream ? <Badge variant="muted">stream</Badge> : null}
                      {routed ? (
                        <Hint label={routed.hint}>
                          <Badge variant="outline" className="cursor-default">
                            {routed.label}
                          </Badge>
                        </Hint>
                      ) : null}
                    </div>
                    <div className="truncate text-xs text-muted-foreground">
                      {record.model ?? "no model named"}
                      {served ? <span className="text-foreground"> → {served}</span> : null}
                      {who ? (
                        <span title={record.key_name ? `API key ${record.key_name} from ${record.client}` : record.client}>
                          {" "}
                          · {who}
                        </span>
                      ) : null}
                      {record.error ? <span className="text-critical"> · {record.error}</span> : null}
                    </div>
                    {record.state === "running" && record.progress ? <ProgressLine progress={record.progress} /> : null}
                  </div>
                  <div className="shrink-0 text-right text-xs tabular-nums">
                    <div className={cn(record.status >= 400 ? "text-critical" : "text-foreground")}>
                      {record.status > 0 ? record.status : record.state === "waiting" ? "waiting" : "…"}
                    </div>
                    <div className="text-muted-foreground">
                      {record.duration_ms !== null ? formatMs(record.duration_ms) : formatClock(record.t)}
                    </div>
                  </div>
                </li>
              );
            })}
          </ul>
        )}
      </CardContent>
    </Card>
  );
}
