import { useId } from "react";

import { cn } from "@/lib/utils";

const FLAME =
  "M16 3.5c1.4 4.6 6.6 7.4 7.9 12.4 1.6 6-2.4 12.6-7.9 12.6-5.6 0-9.6-4.4-8.9-9.8.4-3.2 2.3-5.4 3.6-7.4.4 2.4 1.5 3.9 2.8 4.5-.7-4.3.4-8.8 2.5-12.3z";
const CORE = "M16.3 16.2c.8 2.8 3.7 4.2 3.7 7.3 0 2.5-1.8 4-4 4s-3.9-1.6-3.7-3.9c.2-2.3 2-3 2.7-5 .6.9 1 1.1 1.3-2.4z";
// The flame at 90 %, centred on x 16, its base at y 23: just above the tokens.
const FLAME_AT = "translate(2.005 -2.65) scale(.9)";

/** A token: a chip with a two-word text line. */
function Token({ x, fill }: { x: number; fill: string }) {
  return (
    <>
      <rect x={x} y="24.1" width="9.6" height="6.3" rx="1.9" fill={fill} />
      <rect x={x + 1.9} y="26.6" width="3.36" height="1.3" rx=".65" fill="#ffdde9" />
      <rect x={x + 6.16} y="26.6" width="1.54" height="1.3" rx=".65" fill="#ffdde9" opacity=".65" />
    </>
  );
}

/** A spark: a tiny token rising from the fire, tilted by `angle` degrees about its centre. */
function Spark({ cx, cy, width, angle, fill, opacity }: { cx: number; cy: number; width: number; angle: number; fill: string; opacity: number }) {
  const height = width * 0.64;
  return (
    <rect
      x={cx - width / 2}
      y={cy - height / 2}
      width={width}
      height={height}
      rx={width * 0.2}
      fill={fill}
      opacity={opacity}
      transform={`rotate(${angle} ${cx} ${cy})`}
    />
  );
}

/** The Infernix mark: a flame above a row of AI tokens, its sparks rising as tiny tokens, in the
 * burgundy of the colour scheme (hue 356). The background is transparent, so every colour is one that
 * reads on light and dark surfaces alike. index.html's favicon is the same drawing. */
export function LogoMark({ className }: { className?: string }) {
  // useId's characters (":" or "«»") are not safe inside url(#...) references.
  const id = `flame${useId().replace(/[^a-zA-Z0-9_-]/g, "")}`;
  const token = `url(#${id}-token)`;
  return (
    <svg viewBox="0 0 32 32" className={cn("size-8", className)} aria-hidden="true">
      <defs>
        <linearGradient id={`${id}-flame`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#fa86b4" />
          <stop offset="0.5" stopColor="#ce337e" />
          <stop offset="1" stopColor="#931055" />
        </linearGradient>
        <linearGradient id={`${id}-core`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#ffeff6" />
          <stop offset="1" stopColor="#ffc7dc" />
        </linearGradient>
        <linearGradient id={`${id}-token`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#c65e89" />
          <stop offset="1" stopColor="#a43066" />
        </linearGradient>
      </defs>
      <Token x={0.6} fill={token} />
      <Token x={11.2} fill={token} />
      <Token x={21.8} fill={token} />
      <g transform={FLAME_AT}>
        <path d={FLAME} fill={`url(#${id}-flame)`} />
        <path d={CORE} fill={`url(#${id}-core)`} />
      </g>
      <Spark cx={5.4} cy={13.6} width={3} angle={-22} fill="#e2699b" opacity={0.95} />
      <Spark cx={26.8} cy={9.4} width={2.5} angle={18} fill="#fa86b4" opacity={0.9} />
      <Spark cx={3.8} cy={6.6} width={1.8} angle={-12} fill="#fa86b4" opacity={0.7} />
      <Spark cx={27.2} cy={16.8} width={2} angle={26} fill="#e2699b" opacity={0.85} />
    </svg>
  );
}

export function Logo({ className, compact = false }: { className?: string; compact?: boolean }) {
  return (
    <div className={cn("flex items-center gap-2.5", className)}>
      <LogoMark className="size-8 shrink-0" />
      {compact ? null : (
        <div className="flex min-w-0 flex-col leading-none">
          <span className="text-[15px] font-semibold tracking-tight">Infernix</span>
          <span className="mt-1 text-[11px] font-medium text-muted-foreground">Launcher</span>
        </div>
      )}
    </div>
  );
}
