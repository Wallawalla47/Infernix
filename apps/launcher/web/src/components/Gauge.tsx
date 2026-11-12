// A speedometer dial in pure SVG, drawn with as few marks as it needs: one 240-degree track, the
// reading as one arc of one width over it, a needle, five quiet ticks, the scale's two ends, and
// the reading with its unit in the gap between the arc's ends. The arc and needle are the accent
// colour; a dial with limits (temperature, memory) turns amber at its warning limit and red at its
// critical one, and its card says so in a word.

import { useRef, type ReactNode } from "react";
import type { LucideIcon } from "lucide-react";
import { CircleHelp } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { Hint } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

const CX = 100;
const CY = 100;
const R = 80; // centre line of the track
const TRACK = 8; // width of the track and of the reading's arc
const START = 150; // degrees, clockwise from +x (screen coordinates): lower left
const SWEEP = 240; // to 390 = 30 degrees: lower right
const NEEDLE = 62;

/** What a reading means on a dial with limits. */
export type GaugeTone = "normal" | "warning" | "critical";

const TONE_COLOR: Record<GaugeTone, string> = {
  normal: "var(--primary)",
  warning: "var(--status-warning)",
  critical: "var(--status-critical)",
};

/** Where a reading stands against the dial's limits (higher is worse); normal without limits. */
export function gaugeTone(value: number | null, warnAt?: number, criticalAt?: number): GaugeTone {
  if (value === null || !Number.isFinite(value)) return "normal";
  if (criticalAt !== undefined && value >= criticalAt) return "critical";
  if (warnAt !== undefined && value >= warnAt) return "warning";
  return "normal";
}

function polar(radius: number, degrees: number): [number, number] {
  const rad = (degrees * Math.PI) / 180;
  return [CX + radius * Math.cos(rad), CY + radius * Math.sin(rad)];
}

function arc(radius: number, from: number, to: number): string {
  const [x1, y1] = polar(radius, from);
  const [x2, y2] = polar(radius, to);
  const large = to - from > 180 ? 1 : 0;
  return `M ${x1.toFixed(2)} ${y1.toFixed(2)} A ${radius} ${radius} 0 ${large} 1 ${x2.toFixed(2)} ${y2.toFixed(2)}`;
}

/** The smallest of 1, 2, 2.5, 5 x 10^k at or above `value`. */
export function niceCeil(value: number): number {
  if (!(value > 0)) return 1;
  const exponent = Math.floor(Math.log10(value));
  for (const mantissa of [1, 2, 2.5, 5, 10]) {
    const candidate = mantissa * 10 ** exponent;
    if (candidate >= value - 1e-9) return candidate;
  }
  return 10 ** (exponent + 1);
}

/** A scale maximum that starts at `base` and only grows (to the next round value above any
 * reading), so needles stay comparable over a session. */
export function useGrowingMax(reading: number | null | undefined, base: number): number {
  const max = useRef(base);
  if (typeof reading === "number" && Number.isFinite(reading) && reading > max.current) {
    max.current = niceCeil(reading * 1.05);
  }
  return max.current;
}

export function compactTick(value: number): string {
  const abs = Math.abs(value);
  if (abs >= 1000) {
    const k = value / 1000;
    return `${Number.isInteger(k) ? k : k.toFixed(1)}k`;
  }
  if (Number.isInteger(value)) return String(value);
  return abs >= 10 ? value.toFixed(0) : value.toFixed(1);
}

export interface GaugeProps {
  /** Accessible name. */
  label: string;
  value: number | null;
  min?: number;
  max: number;
  unit?: string;
  /** The big reading (default: compact). */
  format?: (value: number) => string;
  /** The two end labels of the scale (default: compact). */
  tickFormat?: (value: number) => string;
  /** Readings at or above these are a warning and critical (higher is worse); a dial without
   * limits is always the accent colour. */
  warnAt?: number;
  criticalAt?: number;
  className?: string;
}

const TICKS = [0, 0.25, 0.5, 0.75, 1];

export function Gauge({
  label,
  value,
  min = 0,
  max,
  unit,
  format = compactTick,
  tickFormat = compactTick,
  warnAt,
  criticalAt,
  className,
}: GaugeProps) {
  const span = max > min ? max - min : 1;
  const hasValue = value !== null && Number.isFinite(value);
  const fraction = hasValue ? Math.max(0, Math.min(1, (value - min) / span)) : 0;
  const angle = START + fraction * SWEEP;
  const tone = gaugeTone(value, warnAt, criticalAt);
  const color = TONE_COLOR[tone];

  const text = hasValue ? format(value) : "—";
  const valueText = hasValue
    ? `${text}${unit ? ` ${unit}` : ""}${tone === "normal" ? "" : tone === "warning" ? " (high)" : " (critical)"}`
    : "no reading";
  const [minX, endY] = polar(R, START);
  const [maxX] = polar(R, START + SWEEP);

  return (
    <div
      role="meter"
      aria-label={label}
      aria-valuemin={min}
      aria-valuemax={max}
      aria-valuenow={hasValue ? value : undefined}
      aria-valuetext={valueText}
      className={cn("relative mx-auto w-full max-w-[14rem]", className)}
    >
      <svg viewBox="0 0 200 162" className="block h-auto w-full" aria-hidden="true">
        {/* track */}
        <path
          d={arc(R, START, START + SWEEP)}
          fill="none"
          strokeWidth={TRACK}
          strokeLinecap="round"
          style={{ stroke: "color-mix(in oklab, var(--muted-foreground) 18%, transparent)" }}
        />
        {/* the reading */}
        <path
          d={arc(R, START, START + SWEEP)}
          fill="none"
          strokeWidth={TRACK}
          strokeLinecap="round"
          pathLength={100}
          strokeDasharray="100 200"
          className="gauge-anim"
          style={{ stroke: color, strokeDashoffset: 100 - fraction * 100, opacity: hasValue && fraction > 0.004 ? 1 : 0 }}
        />
        {/* quarter ticks */}
        {TICKS.map((tick) => {
          const deg = START + tick * SWEEP;
          const [x1, y1] = polar(R - 15, deg);
          const [x2, y2] = polar(R - 10, deg);
          return (
            <line
              key={tick}
              x1={x1}
              y1={y1}
              x2={x2}
              y2={y2}
              strokeWidth={1.25}
              strokeLinecap="round"
              style={{ stroke: "var(--chart-muted)", opacity: 0.7 }}
            />
          );
        })}
        {/* needle */}
        <g
          className="gauge-anim"
          style={{
            transform: `rotate(${angle}deg)`,
            transformOrigin: `${CX}px ${CY}px`,
            transformBox: "view-box",
            opacity: hasValue ? 1 : 0.3,
          }}
        >
          <line x1={CX} y1={CY} x2={CX + NEEDLE} y2={CY} strokeWidth={2.5} strokeLinecap="round" style={{ stroke: "var(--foreground)" }} />
        </g>
        <circle cx={CX} cy={CY} r={5} style={{ fill: "var(--foreground)" }} />
        {/* reading */}
        <text
          x={CX}
          y={138}
          textAnchor="middle"
          fontSize={26}
          fontWeight={650}
          style={{ fill: hasValue ? "var(--foreground)" : "var(--muted-foreground)", fontVariantNumeric: "tabular-nums" }}
        >
          {text}
        </text>
        {unit ? (
          <text x={CX} y={155} textAnchor="middle" fontSize={11} style={{ fill: "var(--muted-foreground)" }}>
            {unit}
          </text>
        ) : null}
        {/* the scale's ends */}
        {[
          { x: minX, text: tickFormat(min) },
          { x: maxX, text: tickFormat(max) },
        ].map((end) => (
          <text
            key={end.x}
            x={end.x}
            y={endY + 15}
            textAnchor="middle"
            fontSize={10.5}
            style={{ fill: "var(--muted-foreground)", fontVariantNumeric: "tabular-nums" }}
          >
            {end.text}
          </text>
        ))}
      </svg>
    </div>
  );
}

const TONE_WORD: Record<Exclude<GaugeTone, "normal">, string> = { warning: "High", critical: "Critical" };

/** A dial in a card: title (with an optional explanation), the dial, sub-labels and an optional
 * history sparkline beneath. A reading past a limit is named top right ("High", "Critical"). */
export function GaugeCard({
  title,
  icon: Icon,
  hint,
  aside,
  sub,
  footer,
  className,
  ...gauge
}: Omit<GaugeProps, "label"> & {
  title: string;
  icon?: LucideIcon;
  hint?: ReactNode;
  /** Top-right slot (a badge); a reading past a limit takes it. */
  aside?: ReactNode;
  /** Lines under the dial ("Session 142 tok/s"). */
  sub?: ReactNode;
  /** Under everything, e.g. a sparkline. */
  footer?: ReactNode;
}) {
  const tone = gaugeTone(gauge.value, gauge.warnAt, gauge.criticalAt);
  return (
    <Card className={cn("min-w-0 gap-1 overflow-hidden px-0 py-3", className)}>
      <div className="flex min-h-5 items-center justify-between gap-2 px-3">
        <span className="flex min-w-0 items-center gap-1.5 text-xs font-medium text-muted-foreground">
          {Icon ? <Icon className="size-3.5 shrink-0" /> : null}
          <span className="truncate">{title}</span>
          {hint ? (
            <Hint label={<span className="block max-w-xs leading-relaxed">{hint}</span>}>
              <CircleHelp className="size-3 shrink-0 cursor-help opacity-70" aria-label="About this gauge" />
            </Hint>
          ) : null}
        </span>
        {tone !== "normal" ? <Badge variant={tone === "critical" ? "critical" : "warning"}>{TONE_WORD[tone]}</Badge> : aside}
      </div>
      <div className="px-2">
        <Gauge label={title} {...gauge} />
      </div>
      {sub ? (
        <div className="flex min-h-4 flex-wrap items-center justify-center gap-x-1.5 gap-y-0.5 px-3 text-center text-[11px] text-muted-foreground tabular-nums [&>*+*]:before:mr-1.5 [&>*+*]:before:opacity-60 [&>*+*]:before:content-['·']">
          {sub}
        </div>
      ) : null}
      {footer ? <div className="mt-1 px-1">{footer}</div> : null}
    </Card>
  );
}
