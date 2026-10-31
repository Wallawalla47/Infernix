import type { InstanceState, ModelFamily, NgramFormat } from "./api-types";

const DASH = "—";

type Maybe = number | null | undefined;

function finite(value: Maybe): value is number {
  return value !== null && value !== undefined && Number.isFinite(value);
}

/** Sizes on disk and downloads in decimal units (GB, as Hugging Face lists them); memory in
 * binary units (GiB) with `binary`. */
export function formatBytes(bytes: Maybe, options: { binary?: boolean; digits?: number } = {}): string {
  if (!finite(bytes)) return DASH;
  const binary = options.binary ?? false;
  const base = binary ? 1024 : 1000;
  const units = binary ? ["B", "KiB", "MiB", "GiB", "TiB"] : ["B", "KB", "MB", "GB", "TB"];
  let value = Math.abs(bytes);
  let unit = 0;
  while (value >= base && unit < units.length - 1) {
    value /= base;
    unit += 1;
  }
  const digits = options.digits ?? (unit === 0 || value >= 100 ? 0 : value >= 10 ? 1 : 2);
  return `${bytes < 0 ? "-" : ""}${value.toFixed(digits)} ${units[unit]}`;
}

export function formatGiB(bytes: Maybe, digits = 1): string {
  if (!finite(bytes)) return DASH;
  return `${(bytes / 1024 ** 3).toFixed(digits)} GiB`;
}

/** Bytes per second as a transfer rate (decimal units). */
export function formatByteRate(bytesPerSecond: Maybe): string {
  if (!finite(bytesPerSecond)) return DASH;
  return `${formatBytes(bytesPerSecond, { digits: bytesPerSecond >= 1e9 ? 2 : undefined })}/s`;
}

/** A token rate's number alone: 7.46k, 157, 42.3, 3.21. */
export function formatRateNumber(value: Maybe): string {
  if (!finite(value)) return DASH;
  if (value >= 10_000) return `${(value / 1000).toFixed(1)}k`;
  if (value >= 1000) return `${(value / 1000).toFixed(2)}k`;
  if (value >= 100) return value.toFixed(0);
  if (value >= 10) return value.toFixed(1);
  return value.toFixed(2);
}

export function formatTokS(value: Maybe): string {
  return finite(value) ? `${formatRateNumber(value)} tok/s` : DASH;
}

export function formatSeconds(seconds: Maybe): string {
  if (!finite(seconds)) return DASH;
  if (seconds < 0) return DASH;
  if (seconds < 1) return `${Math.round(seconds * 1000)} ms`;
  if (seconds < 10) return `${seconds.toFixed(2)} s`;
  if (seconds < 60) return `${seconds.toFixed(1)} s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ${String(Math.round(seconds % 60)).padStart(2, "0")}s`;
  const hours = Math.floor(minutes / 60);
  return `${hours}h ${String(minutes % 60).padStart(2, "0")}m`;
}

export function formatMs(ms: Maybe): string {
  return finite(ms) ? formatSeconds(ms / 1000) : DASH;
}

/** A ratio (0..1) as a percentage. */
export function formatRatio(ratio: Maybe, digits = 1): string {
  return finite(ratio) ? `${(ratio * 100).toFixed(digits)}%` : DASH;
}

/** A value already in percent (0..100). */
export function formatPercent(value: Maybe, digits = 0): string {
  return finite(value) ? `${value.toFixed(digits)}%` : DASH;
}

export function formatInt(value: Maybe): string {
  return finite(value) ? Math.round(value).toLocaleString("en-US") : DASH;
}

/** Compact counts: 1,284 / 12.9K / 4.2M. */
export function formatCount(value: Maybe): string {
  if (!finite(value)) return DASH;
  const abs = Math.abs(value);
  if (abs < 10_000) return Math.round(value).toLocaleString("en-US");
  if (abs < 1_000_000) return `${(value / 1000).toFixed(abs < 100_000 ? 1 : 0)}K`;
  if (abs < 1_000_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  return `${(value / 1_000_000_000).toFixed(1)}B`;
}

/** Context lengths as people quote them: 8K, 240K, 262K, 1.05M. */
export function formatContext(tokens: Maybe): string {
  if (!finite(tokens)) return DASH;
  if (tokens >= 1_000_000) return `${(tokens / 1_000_000).toFixed(2).replace(/\.?0+$/, "")}M`;
  if (tokens >= 1000) return `${Math.round(tokens / 1000)}K`;
  return String(tokens);
}

export function formatUptime(ms: Maybe): string {
  if (!finite(ms) || ms < 0) return DASH;
  const seconds = Math.floor(ms / 1000);
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ${String(seconds % 60).padStart(2, "0")}s`;
  const hours = Math.floor(minutes / 60);
  if (hours < 48) return `${hours}h ${String(minutes % 60).padStart(2, "0")}m`;
  return `${Math.floor(hours / 24)}d ${hours % 24}h`;
}

export function formatRelative(t: Maybe, now: number = Date.now()): string {
  if (!finite(t)) return DASH;
  const delta = Math.max(0, now - t) / 1000;
  if (delta < 5) return "just now";
  if (delta < 60) return `${Math.floor(delta)} s ago`;
  if (delta < 3600) return `${Math.floor(delta / 60)} min ago`;
  if (delta < 86_400) return `${Math.floor(delta / 3600)} h ago`;
  return new Date(t).toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
}

function pad(value: number, width = 2): string {
  return String(value).padStart(width, "0");
}

export function formatClock(t: number): string {
  const date = new Date(t);
  return `${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
}

export function formatClockMs(t: number): string {
  return `${formatClock(t)}.${pad(new Date(t).getMilliseconds(), 3)}`;
}

export function formatDateTime(t: Maybe): string {
  if (!finite(t)) return DASH;
  return new Date(t).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function formatEta(seconds: Maybe): string {
  if (!finite(seconds)) return DASH;
  if (seconds < 1) return "<1 s";
  return formatSeconds(seconds).replace(/\.\d+ s$/, " s");
}

export const FAMILY_LABELS: Record<ModelFamily, string> = {
  "qwen3.8-flash-next": "Qwen3.8-Flash-Next",
  "qwen3.5-122b-a10b": "Qwen3.5-122B-A10B",
  "qwen3.6-35b-a3b": "Qwen3.6-35B-A3B",
  "qwen3.8-27b": "Qwen3.8-27B",
  "qwen3.6-27b": "Qwen3.6-27B",
  "qwen3.5": "Qwen3.5 family",
  unknown: "Other",
};

export const FAMILY_ORDER: ModelFamily[] = [
  "qwen3.8-flash-next",
  "qwen3.8-27b",
  "qwen3.5-122b-a10b",
  "qwen3.6-35b-a3b",
  "qwen3.6-27b",
  "qwen3.5",
  "unknown",
];

export const NGRAM_FORMAT_LABELS: Record<NgramFormat, string> = {
  fp8: "FP8",
  iq4_nl: "IQ4_NL",
  bf16: "BF16",
  nvfp4: "NVFP4",
  unknown: "unknown",
};

export const STATE_LABELS: Record<InstanceState, string> = {
  idle: "Idle",
  starting: "Loading",
  ready: "Ready",
  stopping: "Stopping",
  failed: "Failed",
};

/** The launcher already names the weight formats ("NVFP4 + FP8", "IQ3_S"). */
export function quantizationLabel(value: string | null): string {
  return value ? value : DASH;
}
