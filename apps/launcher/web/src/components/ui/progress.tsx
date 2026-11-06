import * as React from "react";
import { Progress as ProgressPrimitive } from "radix-ui";

import { cn } from "@/lib/utils";

type Tone = "primary" | "success" | "warning" | "critical" | "muted";

const INDICATOR: Record<Tone, string> = {
  primary: "bg-primary",
  success: "bg-success",
  warning: "bg-warning",
  critical: "bg-critical",
  muted: "bg-muted-foreground/60",
};

const TRACK: Record<Tone, string> = {
  primary: "bg-primary/15",
  success: "bg-success/15",
  warning: "bg-warning/20",
  critical: "bg-critical/20",
  muted: "bg-muted",
};

/** A bar of `value` percent (0..100); `value` null draws the indeterminate animation. */
export function Progress({
  className,
  value,
  tone = "primary",
  ...props
}: Omit<React.ComponentProps<typeof ProgressPrimitive.Root>, "value"> & { value: number | null; tone?: Tone }) {
  const clamped = value === null ? null : Math.max(0, Math.min(100, value));
  return (
    <ProgressPrimitive.Root
      data-slot="progress"
      value={clamped}
      className={cn("relative h-1.5 w-full overflow-hidden rounded-full", TRACK[tone], className)}
      {...props}
    >
      <ProgressPrimitive.Indicator
        data-slot="progress-indicator"
        className={cn(
          "h-full rounded-full transition-[width] duration-500 ease-out",
          INDICATOR[tone],
          clamped === null && "progress-indeterminate w-2/5",
        )}
        style={clamped === null ? undefined : { width: `${clamped}%` }}
      />
    </ProgressPrimitive.Root>
  );
}

export type ProgressTone = Tone;
