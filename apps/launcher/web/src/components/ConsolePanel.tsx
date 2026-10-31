import { useEffect, useMemo, useRef, useState } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
import { ArrowDownToLine, Copy, Download, Eraser, Search, X } from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Hint } from "@/components/ui/tooltip";
import type { LogLevel, LogLine } from "@/lib/api-types";
import { clearLogView, useLive } from "@/lib/events";
import { formatClockMs } from "@/lib/format";
import { cn, copyText, downloadText } from "@/lib/utils";

type Filter = "debug" | "info" | "warn" | "error" | "launcher";

const FILTERS: { key: Filter; label: string }[] = [
  { key: "error", label: "Errors" },
  { key: "warn", label: "Warnings" },
  { key: "info", label: "Info" },
  { key: "debug", label: "Debug" },
  { key: "launcher", label: "Launcher" },
];

function bucket(level: LogLevel): Exclude<Filter, "launcher"> {
  switch (level) {
    case "error":
    case "critical":
      return "error";
    case "warn":
      return "warn";
    case "debug":
    case "trace":
      return "debug";
    default:
      return "info";
  }
}

const LEVEL_TEXT: Record<LogLevel, string> = {
  trace: "TRACE",
  debug: "DEBUG",
  info: "INFO",
  warn: "WARN",
  error: "ERROR",
  critical: "CRIT",
  raw: "",
};

const LEVEL_TONE: Record<LogLevel, string> = {
  trace: "text-muted-foreground/70",
  debug: "text-muted-foreground",
  info: "text-series-1",
  warn: "text-warning",
  error: "text-critical",
  critical: "text-critical",
  raw: "text-muted-foreground",
};

function lineText(line: LogLine): string {
  const level = LEVEL_TEXT[line.level];
  const prefix = line.source === "launcher" ? "launcher | " : "";
  return `${formatClockMs(line.t)}  ${level.padEnd(5)}  ${prefix}${line.text}`;
}

/** `text` with every case-insensitive occurrence of `query` marked. */
export function Highlight({ text, query }: { text: string; query: string }) {
  if (query.length === 0) return <>{text}</>;
  const lower = text.toLowerCase();
  const needle = query.toLowerCase();
  const parts: { text: string; hit: boolean }[] = [];
  let at = 0;
  for (;;) {
    const found = lower.indexOf(needle, at);
    if (found < 0) break;
    if (found > at) parts.push({ text: text.slice(at, found), hit: false });
    parts.push({ text: text.slice(found, found + needle.length), hit: true });
    at = found + needle.length;
  }
  if (at < text.length) parts.push({ text: text.slice(at), hit: false });
  return (
    <>
      {parts.map((part, index) =>
        part.hit ? (
          <mark key={index} className="rounded-sm bg-primary/30 text-foreground">
            {part.text}
          </mark>
        ) : (
          <span key={index}>{part.text}</span>
        ),
      )}
    </>
  );
}

/** The managed server's console, streamed from the launcher (ANSI already stripped). */
export function ConsolePanel({
  className,
  heightClass = "h-[26rem]",
  height,
}: {
  className?: string;
  heightClass?: string;
  /** Fixed height of the log body in pixels (overrides heightClass). */
  height?: number;
}) {
  const logs = useLive((live) => live.logs);
  const [enabled, setEnabled] = useState<Record<Filter, boolean>>({
    error: true,
    warn: true,
    info: true,
    debug: true,
    launcher: true,
  });
  const [query, setQuery] = useState("");
  const [follow, setFollow] = useState(true);
  const scrollRef = useRef<HTMLDivElement>(null);

  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return logs.filter((line) => {
      if (line.source === "launcher") {
        if (!enabled.launcher) return false;
      } else if (!enabled[bucket(line.level)]) {
        return false;
      }
      return needle.length === 0 || line.text.toLowerCase().includes(needle);
    });
  }, [logs, enabled, query]);

  const virtualizer = useVirtualizer({
    count: filtered.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => 20,
    overscan: 24,
  });

  useEffect(() => {
    if (follow && filtered.length > 0) {
      virtualizer.scrollToIndex(filtered.length - 1, { align: "end" });
    }
  }, [follow, filtered.length, virtualizer]);

  const onScroll = () => {
    const el = scrollRef.current;
    if (!el) return;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 32;
    if (atBottom !== follow) setFollow(atBottom);
  };

  const exportText = () => filtered.map(lineText).join("\n");
  const highlight = query.trim();

  return (
    <div className={cn("flex min-w-0 flex-col overflow-hidden rounded-xl border bg-card", className)}>
      <div className="flex flex-wrap items-center gap-2 border-b px-3 py-2">
        <div className="flex flex-wrap items-center gap-1">
          {FILTERS.map((filter) => (
            <button
              key={filter.key}
              type="button"
              aria-pressed={enabled[filter.key]}
              onClick={() => setEnabled((old) => ({ ...old, [filter.key]: !old[filter.key] }))}
              className={cn(
                "rounded-full border px-2 py-0.5 text-[11px] font-medium transition-colors",
                enabled[filter.key]
                  ? "border-primary/40 bg-primary/10 text-foreground"
                  : "text-muted-foreground hover:text-foreground",
              )}
            >
              {filter.label}
            </button>
          ))}
        </div>
        <div className="relative ml-auto w-full min-w-40 sm:w-56">
          <Search className="pointer-events-none absolute top-1/2 left-2.5 size-3.5 -translate-y-1/2 text-muted-foreground" />
          <Input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search the log"
            className="h-8 pl-8 text-xs"
            aria-label="Search the log"
          />
          {query.length > 0 ? (
            <button
              type="button"
              className="absolute top-1/2 right-2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
              onClick={() => setQuery("")}
              aria-label="Clear the search"
            >
              <X className="size-3.5" />
            </button>
          ) : null}
        </div>
        <div className="flex items-center gap-0.5">
          <Hint label={follow ? "Following new lines" : "Jump to the newest line"}>
            <Button
              variant={follow ? "secondary" : "ghost"}
              size="icon-sm"
              aria-label="Follow the newest line"
              onClick={() => setFollow(true)}
            >
              <ArrowDownToLine />
            </Button>
          </Hint>
          <Hint label="Copy the visible lines">
            <Button
              variant="ghost"
              size="icon-sm"
              aria-label="Copy the visible lines"
              onClick={() => {
                void copyText(exportText()).then((ok) =>
                  ok ? toast.success(`Copied ${filtered.length} lines`) : toast.error("Copying failed"),
                );
              }}
            >
              <Copy />
            </Button>
          </Hint>
          <Hint label="Download the visible lines">
            <Button
              variant="ghost"
              size="icon-sm"
              aria-label="Download the visible lines"
              onClick={() => downloadText(`infernix-serve-${new Date().toISOString().replace(/[:.]/g, "-")}.log`, exportText())}
            >
              <Download />
            </Button>
          </Hint>
          <Hint label="Clear this view (the launcher keeps the log)">
            <Button variant="ghost" size="icon-sm" aria-label="Clear the view" onClick={clearLogView}>
              <Eraser />
            </Button>
          </Hint>
        </div>
      </div>

      <div
        ref={scrollRef}
        onScroll={onScroll}
        className={cn("relative overflow-auto bg-background/40 font-mono text-[12px] leading-5", height === undefined && heightClass)}
        style={height === undefined ? undefined : { height }}
        role="log"
        aria-live="off"
      >
        {filtered.length === 0 ? (
          <div className="flex h-full items-center justify-center px-6 text-center font-sans text-sm text-muted-foreground">
            {logs.length === 0
              ? "The server's console appears here once a model is loading or loaded."
              : "No lines match the filters."}
          </div>
        ) : (
          <div style={{ height: virtualizer.getTotalSize(), position: "relative", width: "100%" }}>
            {virtualizer.getVirtualItems().map((item) => {
              const line = filtered[item.index];
              const launcher = line.source === "launcher";
              return (
                <div
                  key={line.seq}
                  data-index={item.index}
                  ref={virtualizer.measureElement}
                  className={cn(
                    "absolute top-0 left-0 flex w-full gap-3 px-3 hover:bg-muted/50",
                    launcher && "border-l-2 border-primary/70 bg-primary/5 pl-2.5",
                  )}
                  style={{ transform: `translateY(${item.start}px)` }}
                >
                  <span className="shrink-0 text-muted-foreground/70 tabular-nums select-none">
                    {formatClockMs(line.t)}
                  </span>
                  <span className={cn("w-10 shrink-0 font-semibold select-none", LEVEL_TONE[line.level])}>
                    {launcher ? "▸" : LEVEL_TEXT[line.level]}
                  </span>
                  <span
                    className={cn(
                      "min-w-0 flex-1 break-words whitespace-pre-wrap",
                      launcher ? "text-foreground" : line.level === "debug" || line.level === "trace" ? "text-muted-foreground" : "",
                    )}
                  >
                    <Highlight text={line.text} query={highlight} />
                  </span>
                </div>
              );
            })}
          </div>
        )}
      </div>
      <div className="flex items-center justify-between border-t px-3 py-1.5 text-[11px] text-muted-foreground tabular-nums">
        <span>
          {filtered.length.toLocaleString()} of {logs.length.toLocaleString()} lines
        </span>
        <span>{follow ? "Following" : "Paused (scroll to the bottom to follow)"}</span>
      </div>
    </div>
  );
}
