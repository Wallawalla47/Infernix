import type { ReactNode } from "react";
import { LoaderCircle } from "lucide-react";

import { Hint } from "@/components/ui/tooltip";
import type { ChatStats } from "@/lib/api-types";
import { cacheHitRatio, type LiveStats } from "@/lib/chat";
import { useLive } from "@/lib/events";
import { formatInt, formatMs, formatPercent, formatRatio, formatTokS } from "@/lib/format";
import { useNow } from "@/lib/hooks";
import { cn } from "@/lib/utils";

function Stat({ label, value, hint }: { label: string; value: ReactNode; hint?: string }) {
  return (
    <span className="inline-flex items-baseline gap-1 whitespace-nowrap" title={hint}>
      <span className="text-muted-foreground">{label}</span>
      <span className="font-medium text-foreground tabular-nums">{value}</span>
    </span>
  );
}

function Row({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div className={cn("flex min-h-5 flex-wrap items-center gap-x-4 gap-y-1 text-xs", className)}>
      {children}
    </div>
  );
}

function draftsLabel(drafted: number | null, accepted: number | null): string | null {
  if (drafted === null || drafted <= 0) return null;
  return `${formatInt(accepted ?? 0)} / ${formatInt(drafted)} (${formatRatio((accepted ?? 0) / drafted, 0)})`;
}

function cacheHint(stats: { prompt_tokens: number | null; cached_tokens: number | null }): string {
  return `${formatInt(stats.cached_tokens)} of ${formatInt(stats.prompt_tokens)} prompt tokens came from the prefix cache`;
}

/** A finished reply's statistics in one row. */
function FinishedRow({ stats }: { stats: ChatStats }) {
  const drafts = draftsLabel(stats.drafted, stats.accepted);
  return (
    <Row>
      <Stat label="TTFT" value={formatMs(stats.ttft_ms)} hint="Time to the first token, as this page saw it (queueing and a model load included)" />
      <Stat label="Prefill" value={formatTokS(stats.prefill_tok_s)} hint="Prompt tokens per second, over the tokens not in the prefix cache" />
      <Stat label="Decode" value={formatTokS(stats.decode_tok_s)} hint="Output tokens per second" />
      <Stat label="Cache hit" value={formatRatio(cacheHitRatio(stats), 0)} hint={cacheHint(stats)} />
      <Stat label="Prompt" value={formatInt(stats.prompt_tokens)} />
      <Stat label="Output" value={formatInt(stats.output_tokens)} />
      {drafts ? <Stat label="Drafts accepted" value={drafts} hint="Accepted of the draft tokens offered" /> : null}
      <Stat label="Total" value={formatMs(stats.total_ms)} />
    </Row>
  );
}

/** A streaming reply's progress: waiting (queued or a model loading), reading the prompt, then
 * the live decode rate. */
function LiveRow({ live }: { live: LiveStats }) {
  const now = useNow(250);
  const instance = useLive((state) => state.instance);
  const elapsed = Math.max(0, now - live.sent_at);

  if (live.phase === "waiting") {
    const loading =
      instance && instance.state === "starting"
        ? `Loading ${instance.alias ?? "the model"}${instance.phase ? `: ${instance.phase}` : ""}`
        : null;
    return (
      <Row className="text-muted-foreground">
        <span className="inline-flex min-w-0 items-center gap-1.5">
          <LoaderCircle className="size-3.5 shrink-0 animate-spin text-primary" />
          <span className="truncate">{loading ?? "Waiting for the model"}</span>
        </span>
        <Stat label="Elapsed" value={formatMs(elapsed)} />
      </Row>
    );
  }

  if (live.phase === "prefill") {
    const progress = live.progress;
    const percent = progress && progress.total > 0 ? (progress.processed / progress.total) * 100 : null;
    return (
      <Row>
        <span className="inline-flex items-center gap-1.5 text-muted-foreground">
          <LoaderCircle className="size-3.5 shrink-0 animate-spin text-primary" />
          Reading the prompt
        </span>
        <Stat label="Prompt" value={formatPercent(percent)} hint={progress ? `${formatInt(progress.processed)} of ${formatInt(progress.total)} tokens` : undefined} />
        <Stat label="Prefill" value={formatTokS(live.prefill_tok_s)} />
        <Stat label="Cache hit" value={formatRatio(cacheHitRatio(live), 0)} hint={cacheHint(live)} />
        <Stat label="Tokens" value={formatInt(live.prompt_tokens)} />
        <Stat label="Elapsed" value={formatMs(elapsed)} />
      </Row>
    );
  }

  const drafts = draftsLabel(live.drafted, live.accepted);
  return (
    <Row>
      <span className="relative inline-flex size-2 shrink-0" aria-hidden="true">
        <span className="absolute inset-0 animate-ping rounded-full bg-primary opacity-60" />
        <span className="relative inline-flex size-2 rounded-full bg-primary" />
      </span>
      <Stat label="TTFT" value={formatMs(live.ttft_ms)} />
      <Stat label="Decode" value={formatTokS(live.decode_tok_s)} />
      <Stat label="Output" value={formatInt(live.output_tokens)} />
      <Stat label="Prefill" value={formatTokS(live.prefill_tok_s)} />
      <Stat label="Cache hit" value={formatRatio(cacheHitRatio(live), 0)} hint={cacheHint(live)} />
      <Stat label="Prompt" value={formatInt(live.prompt_tokens)} />
      {drafts ? <Stat label="Drafts accepted" value={drafts} /> : null}
    </Row>
  );
}

/** The statistics row under the prompt box: the streaming reply's live values, else the latest
 * reply's. */
export function ReplyStatsBar({ stats, live }: { stats: ChatStats | null; live: LiveStats | null }) {
  if (live) return <LiveRow live={live} />;
  if (stats) return <FinishedRow stats={stats} />;
  return <Row className="text-muted-foreground">The latest reply's statistics appear here.</Row>;
}

function DetailLine({ label, value }: { label: string; value: string }) {
  return (
    <>
      <span className="text-muted-foreground">{label}</span>
      <span className="text-right tabular-nums">{value}</span>
    </>
  );
}

/** A reply's own statistics: a short footer line, the rest in its tooltip. */
export function MessageStats({ stats, model }: { stats: ChatStats; model: string }) {
  const drafts = draftsLabel(stats.drafted, stats.accepted);
  const detail = (
    <div className="grid grid-cols-[auto_auto] gap-x-4 gap-y-0.5">
      <DetailLine label="Model" value={model} />
      <DetailLine label="Time to first token" value={formatMs(stats.ttft_ms)} />
      <DetailLine label="Prefill" value={formatTokS(stats.prefill_tok_s)} />
      <DetailLine label="Decode" value={formatTokS(stats.decode_tok_s)} />
      <DetailLine label="Prompt tokens" value={formatInt(stats.prompt_tokens)} />
      <DetailLine label="From the prefix cache" value={`${formatInt(stats.cached_tokens)} (${formatRatio(cacheHitRatio(stats), 0)})`} />
      <DetailLine label="Output tokens" value={formatInt(stats.output_tokens)} />
      {drafts ? <DetailLine label="Drafts accepted" value={drafts} /> : null}
      <DetailLine label="Total" value={formatMs(stats.total_ms)} />
    </div>
  );
  return (
    <Hint label={detail}>
      <span className="min-w-0 cursor-default truncate text-[11px] text-muted-foreground tabular-nums">
        {model} · {formatTokS(stats.decode_tok_s)} · TTFT {formatMs(stats.ttft_ms)} · {formatInt(stats.output_tokens)} tokens
      </span>
    </Hint>
  );
}
