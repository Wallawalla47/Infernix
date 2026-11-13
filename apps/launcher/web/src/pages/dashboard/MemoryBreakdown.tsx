// What the loaded server holds in VRAM and RAM (and on the SSD when it uses an SSD tier), by part,
// from `memory` in its /metrics?format=json: one stacked bar per pool, the server's parts first,
// then its unnamed memory, other programs and what is free.

import { useMemo } from "react";
import { Gpu, HardDrive, Info, Layers, MemoryStick } from "lucide-react";

import { Card, CardContent } from "@/components/ui/card";
import { Hint } from "@/components/ui/tooltip";
import type { GpuSample, MemoryBreakdown as Memory, MemoryPart, MemoryPartKey } from "@/lib/api-types";
import { useLive } from "@/lib/events";
import { formatInt } from "@/lib/format";
import { OTHER_COLOR, useCategoricalPalette } from "@/lib/palette";
import { cn } from "@/lib/utils";

const GiB = 1024 ** 3;
const MiB = 1024 ** 2;

type Pool = "device" | "host" | "ssd";

interface KeySpec {
  label: string;
  /** Palette slot (0-based). Each key keeps its colour in every pool. */
  slot: number;
  /** Staging and the n-gram table borrow a slot's hue, striped. */
  striped?: boolean;
  /** One sentence per pool the key appears in. */
  what: Partial<Record<Pool, string>>;
}

// The contract's keys in its order (the bars' order). Ten keys share eight hues: the slots were
// chosen so that neighbours in realistic bars (with or without a drafter, experts, device prefix
// slots, CUDA graphs) stay apart for normal and colour-blind vision in both themes; staging and the
// n-gram table, the two smallest and rarest, reuse the hue of workspace and of the drafter, striped.
const KEYS: Record<MemoryPartKey, KeySpec> = {
  weights: {
    label: "Model weights",
    slot: 0,
    what: {
      device: "The model's dense (non-expert) weights.",
      host: "Weights kept in RAM: pinned mirrors, the token embedding, an offloaded vision tower.",
    },
  },
  drafter: { label: "Drafter", slot: 3, what: { device: "The speculative drafter's weights and buffers (MTP, DFlash or DFlash2)." } },
  experts: {
    label: "Experts",
    slot: 5,
    what: {
      device: "The VRAM expert cache: routed experts held on the GPU.",
      host: "Expert records held in RAM, copied to the GPU when a step needs one the cache lacks.",
      ssd: "Experts read from the artifact on the SSD instead of being held in RAM.",
    },
  },
  kv_cache: {
    label: "KV cache",
    slot: 4,
    what: { device: "The attention KV cache's pages; the solid share holds tokens now, the faint rest is free for more." },
  },
  state: { label: "Model state", slot: 6, what: { device: "Recurrent (GDN) state, per-request rows and input/output buffers." } },
  prefix_cache: {
    label: "Prefix cache",
    slot: 7,
    what: {
      device: "Device slots of the prefix cache, which keep recent prompts' state for reuse; the solid share holds entries.",
      host: "The prefix cache's RAM tier; the solid share holds entries, the faint rest is free for more.",
      ssd: "The prefix cache's SSD tier; the solid share holds entries, the faint rest is free for more.",
    },
  },
  workspace: { label: "Workspace", slot: 2, what: { device: "Scratch memory for activations while a step runs." } },
  cuda_graphs: { label: "CUDA graphs", slot: 1, what: { device: "Captured CUDA Graphs, which launch a whole step at once." } },
  staging: {
    label: "Staging",
    slot: 2,
    striped: true,
    what: { device: "Copy buffers for expert misses and uploads.", host: "Pinned rings that copies to and from the GPU pass through." },
  },
  ngram: {
    label: "N-gram table",
    slot: 3,
    striped: true,
    what: { host: "The n-gram drafter's table in RAM (the whole table or a cache of its rows)." },
  },
};

const KEY_ORDER = Object.keys(KEYS);

type SegmentKind = "part" | "infernix" | "others" | "free";

interface Segment {
  id: string;
  kind: SegmentKind;
  label: string;
  bytes: number;
  used?: number;
  count?: number;
  color: string;
  striped: boolean;
  what: string;
}

/** Memory sizes: GiB from 1 GiB, MiB below. */
function formatMemory(bytes: number): string {
  if (bytes >= GiB) return `${(bytes / GiB).toFixed(bytes >= 100 * GiB ? 0 : 1)} GiB`;
  return `${Math.round(bytes / MiB)} MiB`;
}

/** A fill: the colour, or (faint) its reserved-but-empty shade; striped for the borrowed hues. */
function fill(color: string, striped: boolean, faint = false): string {
  const solid = faint ? `color-mix(in oklab, ${color} 38%, transparent)` : color;
  if (!striped) return solid;
  const soft = `color-mix(in oklab, ${color} ${faint ? 16 : 42}%, transparent)`;
  return `repeating-linear-gradient(135deg, ${solid} 0 2px, ${soft} 2px 4px)`;
}

function bytesOf(value: number | null | undefined): number {
  return typeof value === "number" && Number.isFinite(value) && value > 0 ? value : 0;
}

/** The pool's parts in the contract's order, unknown keys last under their raw names. */
function partSegments(parts: readonly MemoryPart[], pool: Pool, palette: readonly string[]): Segment[] {
  const rank = (key: string) => {
    const index = KEY_ORDER.indexOf(key);
    return index < 0 ? KEY_ORDER.length : index;
  };
  return parts
    .slice()
    .sort((a, b) => rank(a.key) - rank(b.key))
    .map((part) => {
      const spec = KEY_ORDER.includes(part.key) ? KEYS[part.key as MemoryPartKey] : undefined;
      const bytes = bytesOf(part.bytes);
      return {
        id: part.key,
        kind: "part" as const,
        label: spec?.label ?? part.key,
        bytes,
        used: part.used_bytes === undefined ? undefined : Math.min(bytesOf(part.used_bytes), bytes),
        count: part.count,
        color: spec ? palette[spec.slot] : OTHER_COLOR,
        striped: spec ? spec.striped === true : true,
        what: spec?.what[pool] ?? "A part this launcher does not know yet; a newer launcher names it.",
      };
    });
}

const OTHERS_COLOR = "color-mix(in oklab, var(--muted-foreground) 35%, transparent)";
const FREE_COLOR = "var(--muted)";

interface PoolView {
  segments: Segment[];
  total: number;
  /** In use on the device or in RAM, all programs. */
  used: number;
  /** The server's own share (its process, else its parts). */
  infernix: number;
}

/** Parts, then the server's unnamed memory, other programs and the free rest, summing to `total`. */
function poolView(
  parts: Segment[],
  total: number,
  used: number,
  process: number | null,
  text: { infernix: string; others: string; free: string },
): PoolView {
  const sum = parts.reduce((acc, part) => acc + part.bytes, 0);
  const infernixOther = process === null ? 0 : Math.max(0, process - sum);
  // The parts never exceed the process (the contract); should a reading lag, the parts win.
  const infernix = sum + infernixOther;
  const others = Math.max(0, used - infernix);
  const free = Math.max(0, total - infernix - others);
  const derived = (id: string, kind: SegmentKind, label: string, bytes: number, color: string, what: string): Segment => ({
    id,
    kind,
    label,
    bytes,
    color,
    striped: false,
    what,
  });
  return {
    segments: [
      ...parts,
      derived("infernix-other", "infernix", "Infernix other", infernixOther, OTHER_COLOR, text.infernix),
      derived("other-processes", "others", "Other processes", others, OTHERS_COLOR, text.others),
      derived("free", "free", "Free", free, FREE_COLOR, text.free),
    ],
    total,
    used: Math.max(used, infernix),
    infernix,
  };
}

/** Device memory in use: the driver's reading (NVML) for the same card when the launcher has one,
 * else what the server sees free. */
function deviceUsed(device: Memory["device"], gpu: GpuSample | null): number {
  const total = bytesOf(device.total_bytes);
  const sameCard = gpu !== null && bytesOf(gpu.vram_total) > 0 && Math.abs(gpu.vram_total - total) <= total * 0.05;
  if (sameCard && typeof gpu.vram_used === "number" && Number.isFinite(gpu.vram_used)) return Math.min(Math.max(0, gpu.vram_used), total);
  return Math.max(0, total - bytesOf(device.free_bytes));
}

function share(bytes: number, total: number): string {
  if (total <= 0) return "";
  const pct = (bytes / total) * 100;
  if (pct <= 0) return "0%";
  return pct >= 10 ? `${pct.toFixed(0)}%` : pct >= 0.1 ? `${pct.toFixed(1)}%` : "<0.1%";
}

function SegmentTip({ segment, total }: { segment: Segment; total: number | null }) {
  return (
    <div className="flex max-w-72 flex-col gap-0.5">
      <span className="font-medium">{segment.label}</span>
      <span className="tabular-nums">
        {formatMemory(segment.bytes)}
        {total !== null ? ` · ${share(segment.bytes, total)}` : ""} · {formatInt(segment.bytes)} bytes
      </span>
      {segment.used !== undefined ? (
        <span className="tabular-nums">
          {formatMemory(segment.used)} in use ({share(segment.used, segment.bytes) || "0%"} of the part)
        </span>
      ) : null}
      {segment.count !== undefined ? <span className="tabular-nums">{formatInt(segment.count)} experts</span> : null}
      <span className="text-muted-foreground">{segment.what}</span>
    </div>
  );
}

function Swatch({ segment }: { segment: Segment }) {
  return (
    <span
      className={cn("size-2.5 shrink-0 rounded-[3px]", segment.kind === "free" && "ring-1 ring-border ring-inset")}
      style={{ background: fill(segment.color, segment.striped) }}
      aria-hidden="true"
    />
  );
}

function StackedBar({ label, view }: { label: string; view: PoolView }) {
  const shown = view.segments.filter((segment) => segment.bytes > 0);
  const grow = (bytes: number) => (view.total > 0 ? (bytes / view.total) * 1000 : bytes);
  const summary = shown.map((segment) => `${segment.label} ${formatMemory(segment.bytes)}`).join(", ");
  return (
    <div className="flex h-3 w-full gap-0.5 overflow-hidden rounded-full" role="img" aria-label={`${label}: ${summary}`}>
      {shown.map((segment) => {
        const partial = segment.used !== undefined && segment.bytes > 0;
        return (
          <Hint key={segment.id} label={<SegmentTip segment={segment} total={view.total} />}>
            <div
              className={cn("relative h-full min-w-0.5", segment.kind === "free" && "min-w-0")}
              style={{
                flexGrow: grow(segment.bytes),
                flexBasis: 0,
                background: fill(segment.color, segment.striped, partial),
              }}
            >
              {partial && segment.used! > 0 ? (
                <div
                  className="absolute inset-y-0 left-0"
                  style={{ width: `${Math.min(100, (segment.used! / segment.bytes) * 100)}%`, background: fill(segment.color, segment.striped) }}
                />
              ) : null}
            </div>
          </Hint>
        );
      })}
    </div>
  );
}

function LegendItem({ segment, total, note }: { segment: Segment; total: number | null; note?: string }) {
  const extra = [
    segment.count !== undefined ? `${formatInt(segment.count)} experts` : null,
    segment.used !== undefined ? `${formatMemory(segment.used)} in use` : null,
    note ?? null,
  ].filter((part): part is string => part !== null);
  return (
    <li className="min-w-0">
      <Hint label={<SegmentTip segment={segment} total={total} />}>
        <span className="inline-flex cursor-default items-center gap-1.5">
          <Swatch segment={segment} />
          <span className="text-foreground">{segment.label}</span>
          <span className="text-muted-foreground tabular-nums">
            {formatMemory(segment.bytes)}
            {extra.length > 0 ? ` · ${extra.join(" · ")}` : ""}
          </span>
        </span>
      </Hint>
    </li>
  );
}

function PoolBar({
  label,
  icon: Icon,
  view,
  freeNote,
}: {
  label: string;
  icon: typeof Gpu;
  view: PoolView;
  freeNote?: string;
}) {
  const listed = view.segments.filter((segment) => segment.bytes > 0 || segment.kind === "free");
  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-0.5">
        <span className="flex items-center gap-1.5 text-sm font-medium">
          <Icon className="size-4 text-muted-foreground" />
          {label}
        </span>
        <span className="text-xs text-muted-foreground tabular-nums">
          Infernix <span className="font-semibold text-foreground">{formatMemory(view.infernix)}</span> · {formatMemory(view.used)} of{" "}
          {formatMemory(view.total)} in use
        </span>
      </div>
      <StackedBar label={label} view={view} />
      <ul className="flex flex-wrap gap-x-4 gap-y-1.5 text-xs" aria-label={`${label} parts`}>
        {listed.map((segment) => (
          <LegendItem key={segment.id} segment={segment} total={view.total} note={segment.kind === "free" ? freeNote : undefined} />
        ))}
      </ul>
    </div>
  );
}

function SsdPart({ segment }: { segment: Segment }) {
  const partial = segment.used !== undefined && segment.bytes > 0;
  return (
    <li className="flex min-w-0 flex-col gap-1.5 rounded-md border bg-muted/30 px-2.5 py-2 text-xs">
      <div className="flex items-baseline justify-between gap-3">
        <Hint label={<SegmentTip segment={segment} total={null} />}>
          <span className="inline-flex min-w-0 cursor-default items-center gap-1.5">
            <Swatch segment={segment} />
            <span className="truncate text-foreground">{segment.label}</span>
          </span>
        </Hint>
        <span className="shrink-0 text-muted-foreground tabular-nums">
          {partial ? (
            <>
              <span className="font-medium text-foreground">{formatMemory(segment.used!)}</span> in use of {formatMemory(segment.bytes)}
            </>
          ) : (
            <span className="font-medium text-foreground">{formatMemory(segment.bytes)}</span>
          )}
          {segment.count !== undefined ? ` · ${formatInt(segment.count)} experts` : ""}
        </span>
      </div>
      {partial ? (
        <div
          className="h-1.5 w-full overflow-hidden rounded-full"
          style={{ background: fill(segment.color, segment.striped, true) }}
          role="img"
          aria-label={`${segment.label}: ${formatMemory(segment.used!)} in use of ${formatMemory(segment.bytes)}`}
        >
          <div
            className="h-full rounded-full"
            style={{ width: `${Math.min(100, (segment.used! / segment.bytes) * 100)}%`, background: fill(segment.color, segment.striped) }}
          />
        </div>
      ) : null}
    </li>
  );
}

function Header({ subtitle }: { subtitle: string }) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      <h2 id="memory-breakdown" className="flex items-center gap-2 text-sm font-semibold">
        <Layers className="size-4 text-muted-foreground" />
        Memory
      </h2>
      <span className="min-w-0 truncate text-xs text-muted-foreground">{subtitle}</span>
    </div>
  );
}

/** VRAM and RAM as stacked bars of what infernix-serve holds (in the contract's key order), its
 * unnamed rest, other programs and the free space; the SSD tier's parts when it has one. Hidden
 * without a running server; a server too old to report it gets a one-line note. */
export function MemoryBreakdown() {
  const metrics = useLive((live) => live.metrics);
  const system = useLive((live) => live.system);
  const palette = useCategoricalPalette();
  const memory = metrics?.memory;
  const gpu = system && system.gpus.length > 0 ? system.gpus[0] : null;

  const views = useMemo(() => {
    if (!memory) return null;
    const device = memory.device;
    const host = memory.host;
    const headroom = bytesOf(device.headroom_bytes);
    const vram = poolView(
      partSegments(device.parts, "device", palette),
      bytesOf(device.total_bytes),
      deviceUsed(device, gpu),
      device.process_bytes ?? null,
      {
        infernix: "The server's VRAM that no part names: the CUDA context, driver allocations and small buffers.",
        others: "VRAM in use by other programs: the desktop, browsers and other GPU work.",
        free: `Free on the card now. Infernix keeps ${formatMemory(headroom)} free for the display and other programs.`,
      },
    );
    const total = bytesOf(host.total_bytes);
    const ram = poolView(
      partSegments(host.parts, "host", palette),
      total,
      Math.max(0, total - bytesOf(host.available_bytes)),
      host.process_bytes ?? null,
      {
        infernix:
          "The server's other memory: mostly the CUDA driver's copy of its GPU code (about 4 GiB), then heaps and small buffers.",
        others: "RAM in use by Windows and other programs.",
        free: "Available to programs now (free plus standby memory the system can reclaim).",
      },
    );
    const ssd = memory.ssd ? partSegments(memory.ssd.parts, "ssd", palette).filter((segment) => segment.bytes > 0) : null;
    return { vram, ram, ssd, headroom };
  }, [memory, gpu, palette]);

  if (!metrics) return null;

  if (!views) {
    return (
      <section className="flex flex-col gap-3" aria-labelledby="memory-breakdown">
        <Header subtitle="What the server holds in VRAM and RAM" />
        <p className="flex items-start gap-2 text-xs text-muted-foreground">
          <Info className="mt-0.5 size-3.5 shrink-0" />
          This infernix-serve build reports no memory breakdown; a newer build shows what its VRAM and RAM hold.
        </p>
      </section>
    );
  }

  const { vram, ram, ssd, headroom } = views;
  return (
    <section className="flex flex-col gap-3" aria-labelledby="memory-breakdown">
      <Header
        subtitle={`infernix-serve holds ${formatMemory(vram.infernix)} of VRAM and ${formatMemory(ram.infernix)} of RAM${ssd ? " · SSD tier in use" : ""}`}
      />
      <Card className="gap-5 py-4">
        <CardContent className="flex flex-col gap-5">
          <PoolBar label="VRAM" icon={Gpu} view={vram} freeNote={headroom > 0 ? `kept free ${formatMemory(headroom)}` : undefined} />
          <PoolBar label="RAM" icon={MemoryStick} view={ram} />
          {ssd ? (
            <div className="flex flex-col gap-2">
              <span className="flex items-center gap-1.5 text-sm font-medium">
                <HardDrive className="size-4 text-muted-foreground" />
                SSD
              </span>
              {ssd.length > 0 ? (
                <ul className="grid gap-2 sm:grid-cols-2 xl:grid-cols-3" aria-label="SSD parts">
                  {ssd.map((segment) => (
                    <SsdPart key={segment.id} segment={segment} />
                  ))}
                </ul>
              ) : (
                <p className="text-xs text-muted-foreground">Nothing on the SSD tier yet.</p>
              )}
            </div>
          ) : null}
        </CardContent>
      </Card>
    </section>
  );
}
