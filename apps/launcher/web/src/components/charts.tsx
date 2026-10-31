// Chart pieces shared by the dashboard. Marks follow the data-viz specs: 2px lines, recessive
// hairline grid, >=8px markers with a 2px surface ring, a crosshair on lines and a per-bar hover on
// bars, values leading in tooltips with series keyed by a short line.

import { Area, AreaChart, ReferenceDot, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { cn } from "@/lib/utils";

export interface TooltipEntry {
  value?: unknown;
  name?: unknown;
  dataKey?: unknown;
  payload?: unknown;
}

export interface RechartsTooltipProps {
  active?: boolean;
  payload?: readonly TooltipEntry[];
  label?: unknown;
}

export interface SeriesSpec {
  key: string;
  label: string;
  /** CSS colour of the mark (a series token). */
  color: string;
  format: (value: number) => string;
}

export const AXIS_TICK = { fill: "var(--chart-muted)", fontSize: 11 };

function numeric(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

/** Tooltip for one or more series: value first (strong), series name after, keyed by a line. */
export function SeriesTooltip({
  active,
  payload,
  label,
  series,
  labelFormat,
}: RechartsTooltipProps & { series: readonly SeriesSpec[]; labelFormat?: (label: unknown) => string }) {
  if (!active || !payload || payload.length === 0) return null;
  return (
    <div className="min-w-36 rounded-lg border bg-popover px-3 py-2 text-xs shadow-lg">
      {labelFormat ? <div className="mb-1.5 text-muted-foreground">{labelFormat(label)}</div> : null}
      <div className="flex flex-col gap-1">
        {series.map((spec) => {
          const entry = payload.find((item) => item.dataKey === spec.key);
          const value = numeric(entry?.value);
          return (
            <div key={spec.key} className="flex items-center gap-2">
              <span className="h-0.5 w-3 shrink-0 rounded-full" style={{ background: spec.color }} />
              <span className="font-semibold text-foreground tabular-nums">{value === null ? "—" : spec.format(value)}</span>
              <span className="text-muted-foreground">{spec.label}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export interface SparkPoint {
  t: number;
  v: number | null;
}

/** A small trend line: the de-emphasis hue for history, the accent for the current reading. */
export function Sparkline({
  data,
  format,
  label,
  domain,
  height = 40,
  className,
}: {
  data: readonly SparkPoint[];
  format: (value: number) => string;
  label: string;
  /** Fixed y-range (e.g. [0, 100] for percentages); otherwise from zero to the data maximum. */
  domain?: [number, number];
  height?: number;
  className?: string;
}) {
  let last: SparkPoint | null = null;
  let max = 0;
  for (let i = data.length - 1; i >= 0; i -= 1) {
    const value = data[i].v;
    if (value === null) continue;
    if (last === null) last = data[i];
    if (value > max) max = value;
  }
  const yDomain: [number, number] = domain ?? [0, max > 0 ? max * 1.15 : 1];
  const series: SeriesSpec[] = [{ key: "v", label, color: "var(--chart-muted)", format }];
  return (
    <div className={cn("w-full", className)} style={{ height }}>
      {data.length < 2 ? (
        <div className="flex h-full items-center px-2 text-[11px] text-muted-foreground/70">Collecting…</div>
      ) : (
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data as SparkPoint[]} margin={{ top: 6, right: 6, bottom: 2, left: 6 }}>
            <XAxis dataKey="t" type="number" domain={["dataMin", "dataMax"]} hide />
            <YAxis hide domain={yDomain} />
            <Tooltip
              isAnimationActive={false}
              cursor={{ stroke: "var(--chart-axis)", strokeWidth: 1 }}
              content={
                <SeriesTooltip
                  series={series}
                  labelFormat={(value) => (typeof value === "number" ? new Date(value).toLocaleTimeString() : "")}
                />
              }
            />
            <Area
              type="monotone"
              dataKey="v"
              stroke="var(--chart-muted)"
              strokeWidth={2}
              fill="var(--chart-muted)"
              fillOpacity={0.1}
              connectNulls
              dot={false}
              activeDot={{ r: 4, fill: "var(--primary)", stroke: "var(--card)", strokeWidth: 2 }}
              isAnimationActive={false}
            />
            {last !== null && last.v !== null ? (
              <ReferenceDot
                x={last.t}
                y={last.v}
                r={4}
                fill="var(--primary)"
                stroke="var(--card)"
                strokeWidth={2}
                ifOverflow="visible"
              />
            ) : null}
          </AreaChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}
