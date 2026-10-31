import { CircleCheck, CircleX, LoaderCircle, Power } from "lucide-react";

import type { InstanceState } from "@/lib/api-types";
import { STATE_LABELS } from "@/lib/format";
import { cn } from "@/lib/utils";

const DOT: Record<InstanceState, string> = {
  idle: "bg-muted-foreground/50",
  starting: "bg-warning",
  ready: "bg-success",
  stopping: "bg-serious",
  failed: "bg-critical",
};

const ICON_TONE: Record<InstanceState, string> = {
  idle: "text-muted-foreground",
  starting: "text-warning",
  ready: "text-success",
  stopping: "text-serious",
  failed: "text-critical",
};

/** A state is never colour alone: the dot sits beside its label, or carries it as a title. */
export function StateDot({ state, className, pulse = true }: { state: InstanceState; className?: string; pulse?: boolean }) {
  const busy = pulse && (state === "starting" || state === "stopping");
  return (
    <span className={cn("relative inline-flex size-2 shrink-0", className)} title={STATE_LABELS[state]}>
      {busy ? <span className={cn("absolute inset-0 animate-ping rounded-full opacity-60", DOT[state])} /> : null}
      <span className={cn("relative inline-flex size-2 rounded-full", DOT[state])} />
    </span>
  );
}

export function StateBadge({ state, className }: { state: InstanceState; className?: string }) {
  const Icon = state === "ready" ? CircleCheck : state === "failed" ? CircleX : state === "idle" ? Power : LoaderCircle;
  const spinning = state === "starting" || state === "stopping";
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border bg-background/60 px-2 py-0.5 text-xs font-medium text-foreground",
        className,
      )}
    >
      <Icon className={cn("size-3.5", ICON_TONE[state], spinning && "animate-spin")} aria-hidden="true" />
      {STATE_LABELS[state]}
    </span>
  );
}
