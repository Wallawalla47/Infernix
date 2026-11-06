// A speedometer gauge in pure SVG: a 240-degree track with zone bands under it, major and minor
// ticks (labels at the majors), a value arc and a needle that animate to the reading, a peak marker,
// and the reading with its unit in the gap between the arc's ends.

import { useRef, type CSSProperties, type ReactNode } from "react";
import type { LucideIcon } from "lucide-react";
import { CircleHelp } from "lucide-react";

import { Card } from "@/components/ui/card";
import { Hint } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

const CX = 100;
const CY = 100;
const R = 80; // centre line of the track
const TRACK = 10; // track width
const BAND_INNER = R + 7.5; // inner edge of the zone bands; wider bands grow outwards
const BAND_WIDTH = 3; // a zone band's default width
const START = 150; // degrees, clockwise from +x (screen coordinates): lower left
const SWEEP = 240; // to 390 = 30 degrees: lower right
const NEEDLE = 70;

export interface GaugeZone {
  /** Upper bound of the zone in value units; zones run in order from `min`. */
  to: number;
  /** Any CSS colour (tokens welcome). */
  color: string;
  /** Band width (default 3); a wider band grows outwards from the same inner edge. */
  width?: number;
  /** The value arc's colour while the reading is in this zone (`valueColor="zone"`; default `color`). */
  arc?: string;
}

export const ACCENT_SOFT = "color-mix(in oklab, var(--primary) 28%, transparent)";
export const ACCENT_MID = "color-mix(in oklab, var(--primary) 60%, transparent)";
export const ACCENT = "var(--primary)";
/** The accent at its most contrast with the surface: lighter on dark, darker on light. */
export const ACCENT_STRONG = "var(--primary-strong)";

/** The accent ramp, light to strong: utilisation and rates. */
export function accentZones(max: number): GaugeZone[] {
  return [
    { to: max * 0.6, color: ACCENT_SOFT },
    { to: max * 0.85, color: ACCENT_MID },
    { to: max, color: ACCENT },
  ];
}

/** Severity bands where a high reading is worse (temperature, memory, time to first token): normal,
 * warning and critical in the accent alone, each band wider and of stronger contrast than the last,
 * so the scale reads without hue; the value arc takes the same steps. */
export function severityZones(max: number, warning: number, critical: number): GaugeZone[] {
  return [
    { to: Math.min(warning, max), color: ACCENT_SOFT, width: 3, arc: ACCENT_MID },
    { to: Math.min(critical, max), color: ACCENT, width: 6, arc: ACCENT },
    { to: max, color: ACCENT_STRONG, width: 10, arc: ACCENT_STRONG },
  ].filter((zone, index, all) => index === 0 || zone.to > all[index - 1].to);
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

function isNiceStep(step: number): boolean {
  if (!(step > 0)) return false;
  const exponent = Math.floor(Math.log10(step));
  const mantissa = step / 10 ** exponent;
  return [1, 2, 2.5, 5].some((nice) => Math.abs(mantissa - nice) < 1e-6);
}

/** Major tick values: 4-6 intervals of a round step when one exists, else 4 equal intervals. */
export function majorTicks(min: number, max: number): number[] {
  const span = max - min;
  let count = 4;
  for (const candidate of [4, 5, 6, 3]) {
    if (isNiceStep(span / candidate)) {
      count = candidate;
      break;
    }
  }
  return Array.from({ length: count + 1 }, (_, index) => min + (span * index) / count);
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
  tickFormat?: (value: number) => string;
  zones?: GaugeZone[];
  /** Colour the value arc by the zone the reading is in (severity scales), or the accent. */
  valueColor?: "zone" | "accent";
  /** Highest reading over the visible history (a small marker on the track). */
  peak?: number | null;
  peakLabel?: string;
  className?: string;
}

export function Gauge({
  label,
  value,
  min = 0,
  max,
  unit,
  format = compactTick,
  tickFormat = compactTick,
  zones,
  valueColor = "accent",
  peak = null,
  peakLabel = "peak",
  className,
}: GaugeProps) {
  const span = max > min ? max - min : 1;
  const clamp = (v: number) => Math.max(0, Math.min(1, (v - min) / span));
  const hasValue = value !== null && Number.isFinite(value);
  const fraction = hasValue ? clamp(value) : 0;
  const angle = START + fraction * SWEEP;

  const bands = zones ?? accentZones(max);
  let arcColor = ACCENT;
  if (hasValue && valueColor === "zone" && bands.length > 0) {
    const zone = bands.find((band) => value <= band.to) ?? bands[bands.length - 1];
    arcColor = zone.arc ?? zone.color;
  }

  const majors = majorTicks(min, max);
  const minors: number[] = [];
  for (let i = 0; i + 1 < majors.length; i += 1) {
    const step = (majors[i + 1] - majors[i]) / 4;
    for (let j = 1; j < 4; j += 1) minors.push(majors[i] + step * j);
  }

  let from = min;
  const zonePaths = bands.map((zone) => {
    const a = START + clamp(from) * SWEEP;
    const b = START + clamp(zone.to) * SWEEP;
    const width = zone.width ?? BAND_WIDTH;
    from = zone.to;
    return { d: b - a > 0.5 ? arc(BAND_INNER + width / 2, a, b) : null, color: zone.color, width };
  });
  const bandOuter = BAND_INNER + Math.max(...zonePaths.map((zone) => zone.width), BAND_WIDTH);

  const peakAngle = peak !== null && Number.isFinite(peak) ? START + clamp(peak) * SWEEP : null;
  const text = hasValue ? format(value) : "—";
  const valueText = hasValue ? `${text}${unit ? ` ${unit}` : ""}` : "no reading";

  return (
    <div
      role="meter"
      aria-label={label}
      aria-valuemin={min}
      aria-valuemax={max}
      aria-valuenow={hasValue ? value : undefined}
      aria-valuetext={valueText}
      className={cn("relative mx-auto w-full max-w-[15rem]", className)}
    >
      <svg viewBox="0 0 200 160" className="block h-auto w-full overflow-visible" aria-hidden="true">
        {/* track */}
        <path
          d={arc(R, START, START + SWEEP)}
          fill="none"
          strokeWidth={TRACK}
          strokeLinecap="round"
          style={{ stroke: "color-mix(in oklab, var(--muted-foreground) 16%, transparent)" }}
        />
        {/* zone bands */}
        {zonePaths.map((zone, index) =>
          zone.d ? (
            <path key={index} d={zone.d} fill="none" strokeWidth={zone.width} strokeLinecap="butt" style={{ stroke: zone.color }} />
          ) : null,
        )}
        {/* minor and major ticks */}
        {minors.map((tick) => {
          const deg = START + clamp(tick) * SWEEP;
          const [x1, y1] = polar(R - 13, deg);
          const [x2, y2] = polar(R - 8.5, deg);
          return <line key={`m${tick}`} x1={x1} y1={y1} x2={x2} y2={y2} strokeWidth={1} style={{ stroke: "var(--chart-muted)", opacity: 0.55 }} />;
        })}
        {majors.map((tick) => {
          const deg = START + clamp(tick) * SWEEP;
          const [x1, y1] = polar(R - 16, deg);
          const [x2, y2] = polar(R - 8.5, deg);
          const [lx, ly] = polar(R - 27, deg);
          return (
            <g key={`M${tick}`}>
              <line x1={x1} y1={y1} x2={x2} y2={y2} strokeWidth={1.75} strokeLinecap="round" style={{ stroke: "var(--chart-muted)" }} />
              <text
                x={lx}
                y={ly}
                textAnchor="middle"
                dominantBaseline="central"
                fontSize={10.5}
                style={{ fill: "var(--muted-foreground)", fontVariantNumeric: "tabular-nums" }}
              >
                {tickFormat(tick)}
              </text>
            </g>
          );
        })}
        {/* value arc */}
        <path
          d={arc(R, START, START + SWEEP)}
          fill="none"
          strokeWidth={TRACK}
          strokeLinecap="round"
          pathLength={100}
          strokeDasharray="100 200"
          className="gauge-anim gauge-value-arc"
          style={
            {
              stroke: arcColor,
              strokeDashoffset: 100 - fraction * 100,
              opacity: hasValue && fraction > 0.004 ? 1 : 0,
              "--gauge-glow": arcColor,
            } as CSSProperties
          }
        />
        {/* peak marker */}
        {peakAngle !== null ? (
          <line
            x1={polar(R - 7, peakAngle)[0]}
            y1={polar(R - 7, peakAngle)[1]}
            x2={polar(bandOuter + 2.5, peakAngle)[0]}
            y2={polar(bandOuter + 2.5, peakAngle)[1]}
            strokeWidth={2.25}
            strokeLinecap="round"
            style={{ stroke: "var(--foreground)", opacity: 0.6 }}
          >
            <title>{`${peakLabel} ${peak !== null ? format(peak) : ""}${unit ? ` ${unit}` : ""}`}</title>
          </line>
        ) : null}
        {/* needle */}
        <g
          className="gauge-anim"
          style={{
            transform: `rotate(${angle}deg)`,
            transformOrigin: `${CX}px ${CY}px`,
            transformBox: "view-box",
            opacity: hasValue ? 1 : 0.35,
          }}
        >
          <path
            d={`M ${CX - 12} ${CY} L ${CX} ${CY - 3.6} L ${CX + NEEDLE} ${CY - 0.6} L ${CX + NEEDLE} ${CY + 0.6} L ${CX} ${CY + 3.6} Z`}
            style={{ fill: "var(--foreground)" }}
          />
        </g>
        {/* hub */}
        <circle cx={CX} cy={CY} r={7.5} style={{ fill: "var(--card)", stroke: "var(--foreground)" }} strokeWidth={2.5} />
        <circle cx={CX} cy={CY} r={2.5} style={{ fill: hasValue ? arcColor : "var(--muted-foreground)" }} />
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
      </svg>
    </div>
  );
}

/** A gauge in a card: title (with an optional explanation), the dial, sub-labels and an optional
 * history sparkline beneath. */
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
  /** Top-right slot (a badge, a temperature source). */
  aside?: ReactNode;
  /** Lines under the dial ("Session 142 tok/s"). */
  sub?: ReactNode;
  /** Under everything, e.g. a sparkline. */
  footer?: ReactNode;
}) {
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
        {aside}
      </div>
      <div className="px-2">
        <Gauge label={title} {...gauge} />
      </div>
      {sub ? (
        <div className="flex min-h-4 flex-wrap items-center justify-center gap-x-2.5 gap-y-0.5 px-3 text-center text-[11px] text-muted-foreground tabular-nums">
          {sub}
        </div>
      ) : null}
      {footer ? <div className="mt-1 px-1">{footer}</div> : null}
    </Card>
  );
}
