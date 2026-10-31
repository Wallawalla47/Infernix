import { Fragment, useEffect, useMemo, useRef, useState } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
import { ArrowDownToLine, ChevronRight, Search, X } from "lucide-react";

import { Highlight } from "@/components/ConsolePanel";
import { CopyButton } from "@/components/CopyButton";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Hint } from "@/components/ui/tooltip";
import type { ApiEventFields, EventCategory, EventLevel, EventRecord } from "@/lib/api-types";
import { useLive } from "@/lib/events";
import { formatBytes, formatClockMs, formatMs } from "@/lib/format";
import { cn } from "@/lib/utils";

const LEVELS: { key: EventLevel; label: string }[] = [
  { key: "error", label: "Errors" },
  { key: "warn", label: "Warnings" },
  { key: "info", label: "Info" },
  { key: "debug", label: "Debug" },
];

const CATEGORIES: { key: EventCategory; label: string }[] = [
  { key: "launcher", label: "Launcher" },
  { key: "model", label: "Model" },
  { key: "watchdog", label: "Watchdog" },
  { key: "api", label: "API" },
  { key: "download", label: "Downloads" },
  { key: "settings", label: "Settings" },
];

const LEVEL_TEXT: Record<EventLevel, string> = {
  debug: "DEBUG",
  info: "INFO",
  warn: "WARN",
  error: "ERROR",
};

const LEVEL_BADGE: Record<EventLevel, string> = {
  debug: "border-transparent bg-muted text-muted-foreground",
  info: "border-series-1/30 bg-series-1/10 text-series-1",
  warn: "border-warning/40 bg-warning/15 text-warning",
  error: "border-critical/40 bg-critical/15 text-critical",
};

/** Warnings and errors stand out with a coloured margin, like the console's launcher lines. */
const ROW_TONE: Record<EventLevel, string> = {
  debug: "",
  info: "",
  warn: "border-l-2 border-warning/70 bg-warning/5 pl-2.5",
  error: "border-l-2 border-critical/70 bg-critical/8 pl-2.5",
};

/** The `fields` of an `api` record, when they have the contract's shape. */
function apiFields(record: EventRecord): ApiEventFields | null {
  const fields = record.fields;
  if (record.category !== "api" || !fields) return null;
  return typeof fields.method === "string" && typeof fields.path === "string" && typeof fields.status === "number"
    ? (fields as unknown as ApiEventFields)
    : null;
}

/** HTTP status by class: success, redirect, client error, server error; 0 = no status. */
function statusTone(status: number): string {
  if (status <= 0) return "text-muted-foreground";
  if (status < 300) return "text-success";
  if (status < 400) return "text-series-1";
  if (status < 500) return "text-warning";
  return "text-critical";
}

function fieldText(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function absoluteTime(t: number): string {
  return `${new Date(t).toLocaleString()} (${formatClockMs(t)})`;
}

const haystacks = new WeakMap<EventRecord, string>();

/** What the search looks through: the category, the message and every field value. */
function haystack(record: EventRecord): string {
  let text = haystacks.get(record);
  if (text === undefined) {
    const parts = [record.category, record.message];
    if (record.fields) {
      for (const value of Object.values(record.fields)) {
        if (value !== null && value !== undefined) parts.push(fieldText(value));
      }
    }
    text = parts.join("\n").toLowerCase();
    haystacks.set(record, text);
  }
  return text;
}

function apiRows(fields: ApiEventFields): [string, string][] {
  return [
    ["Request", `#${fields.id}`],
    ["Route", `${fields.method} ${fields.path}`],
    ["Status", fields.status > 0 ? String(fields.status) : "—"],
    ["Streamed", fields.stream ? "yes" : "no"],
    ["Model named", fields.model ?? "none"],
    ["Model key", fields.model_key ?? "—"],
    ["Duration", formatMs(fields.duration_ms)],
    ["First byte", formatMs(fields.first_byte_ms)],
    ["Bytes in", formatBytes(fields.bytes_in)],
    ["Bytes out", formatBytes(fields.bytes_out)],
    ["Client", fields.client],
    ["User agent", fields.user_agent ?? "—"],
    ["Error", fields.error ?? "—"],
  ];
}

function Chip({ label, on, onToggle }: { label: string; on: boolean; onToggle: () => void }) {
  return (
    <button
      type="button"
      aria-pressed={on}
      onClick={onToggle}
      className={cn(
        "rounded-full border px-2 py-0.5 text-[11px] font-medium transition-colors",
        on ? "border-primary/40 bg-primary/10 text-foreground" : "text-muted-foreground hover:text-foreground",
      )}
    >
      {label}
    </button>
  );
}

/** One request on one line: route, status, duration, model and client. */
function ApiSummary({ fields, query }: { fields: ApiEventFields; query: string }) {
  return (
    <span className="flex min-w-0 flex-1 items-center gap-2.5 overflow-hidden whitespace-nowrap">
      <span className="shrink-0">
        <Highlight text={`${fields.method} ${fields.path}`} query={query} />
      </span>
      <span className={cn("shrink-0 font-semibold tabular-nums", statusTone(fields.status))}>
        {fields.status > 0 ? fields.status : "—"}
      </span>
      <span className="shrink-0 text-muted-foreground tabular-nums">{formatMs(fields.duration_ms)}</span>
      {fields.stream ? <span className="shrink-0 text-muted-foreground">stream</span> : null}
      {fields.model ? (
        <span className="min-w-0 truncate" title="Model the request named">
          <Highlight text={fields.model} query={query} />
        </span>
      ) : null}
      <span className="min-w-0 truncate text-muted-foreground" title="Client">
        <Highlight text={fields.client} query={query} />
      </span>
      {fields.error ? <span className="min-w-0 truncate text-critical">{fields.error}</span> : null}
    </span>
  );
}

function Details({ record, api }: { record: EventRecord; api: ApiEventFields | null }) {
  const rows: [string, string][] = api
    ? apiRows(api)
    : Object.entries(record.fields ?? {}).map(([key, value]): [string, string] => [key, fieldText(value)]);
  return (
    <div className="mt-0.5 mb-1.5 ml-6 rounded-md border bg-background/60 px-3 py-2 font-sans text-xs">
      <div className="flex items-start gap-2">
        <div className="min-w-0 flex-1">
          <div className="text-muted-foreground tabular-nums">{absoluteTime(record.t)}</div>
          {api ? <p className="mt-0.5 break-words whitespace-pre-wrap">{record.message}</p> : null}
        </div>
        <CopyButton text={JSON.stringify(record, null, 2)} size="icon-xs" toastMessage="Record copied as JSON" />
      </div>
      {rows.length > 0 ? (
        <dl className="mt-1.5 grid grid-cols-[max-content_minmax(0,1fr)] gap-x-4 gap-y-0.5 sm:grid-cols-[max-content_minmax(0,1fr)_max-content_minmax(0,1fr)]">
          {rows.map(([label, value]) => (
            <Fragment key={label}>
              <dt className="text-muted-foreground">{label}</dt>
              <dd className="font-mono break-all">{value}</dd>
            </Fragment>
          ))}
        </dl>
      ) : null}
    </div>
  );
}

function EventRow({
  record,
  query,
  open,
  onToggle,
}: {
  record: EventRecord;
  query: string;
  open: boolean;
  onToggle: () => void;
}) {
  const api = apiFields(record);
  const expandable = api !== null || (!!record.fields && Object.keys(record.fields).length > 0);
  const onRowClick = () => {
    if (!expandable) return;
    // Selecting text in a row does not toggle it.
    if ((window.getSelection()?.toString() ?? "").length > 0) return;
    onToggle();
  };
  return (
    <>
      <div className={cn("flex items-start gap-3", expandable && "cursor-pointer")} onClick={onRowClick}>
        <span className="flex h-5 w-3.5 shrink-0 items-center">
          {expandable ? (
            <button
              type="button"
              aria-expanded={open}
              aria-label={open ? "Hide the details" : "Show the details"}
              className="text-muted-foreground hover:text-foreground"
              onClick={(event) => {
                event.stopPropagation();
                onToggle();
              }}
            >
              <ChevronRight className={cn("size-3.5 transition-transform", open && "rotate-90")} />
            </button>
          ) : null}
        </span>
        <span className="shrink-0 text-muted-foreground/70 tabular-nums select-none" title={absoluteTime(record.t)}>
          {formatClockMs(record.t)}
        </span>
        <Badge
          variant="outline"
          className={cn("mt-px w-12 shrink-0 px-1 py-0 text-[10px] leading-4 select-none", LEVEL_BADGE[record.level])}
        >
          {LEVEL_TEXT[record.level] ?? record.level}
        </Badge>
        <span className="hidden w-16 shrink-0 text-muted-foreground select-none sm:inline">{record.category}</span>
        {api ? (
          <ApiSummary fields={api} query={query} />
        ) : (
          <span className={cn("min-w-0 flex-1 break-words whitespace-pre-wrap", record.level === "debug" && "text-muted-foreground")}>
            <Highlight text={record.message} query={query} />
          </span>
        )}
      </div>
      {open && expandable ? <Details record={record} api={api} /> : null}
    </>
  );
}

/** The launcher's own event log: model loads and stops, the watchdog, downloads, settings changes
 * and one record per API request. */
export function EventLogPanel({ className, heightClass = "h-[26rem]" }: { className?: string; heightClass?: string }) {
  const records = useLive((live) => live.eventLog);
  const paths = useLive((live) => live.launcher?.logs ?? null);
  const [levels, setLevels] = useState<Record<EventLevel, boolean>>({
    error: true,
    warn: true,
    info: true,
    debug: false,
  });
  const [categories, setCategories] = useState<Record<EventCategory, boolean>>({
    launcher: true,
    model: true,
    watchdog: true,
    api: true,
    download: true,
    settings: true,
  });
  const [query, setQuery] = useState("");
  const [follow, setFollow] = useState(true);
  const [expanded, setExpanded] = useState<ReadonlySet<number>>(() => new Set());
  const scrollRef = useRef<HTMLDivElement>(null);

  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase();
    // A level or category this page does not know stays visible.
    return records.filter(
      (record) =>
        levels[record.level] !== false &&
        categories[record.category] !== false &&
        (needle.length === 0 || haystack(record).includes(needle)),
    );
  }, [records, levels, categories, query]);

  const virtualizer = useVirtualizer({
    count: filtered.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => 22,
    overscan: 24,
    // Measured heights (expanded rows) follow the record, not its position in the filtered list.
    getItemKey: (index) => filtered[index].seq,
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

  const toggle = (seq: number) =>
    setExpanded((old) => {
      const next = new Set(old);
      if (next.has(seq)) next.delete(seq);
      else next.add(seq);
      return next;
    });

  const highlight = query.trim();

  return (
    <div className={cn("flex min-w-0 flex-col overflow-hidden rounded-xl border bg-card", className)}>
      <div className="flex flex-wrap items-center gap-2 border-b px-3 py-2">
        <div className="flex flex-wrap items-center gap-1" role="group" aria-label="Levels">
          {LEVELS.map((level) => (
            <Chip
              key={level.key}
              label={level.label}
              on={levels[level.key]}
              onToggle={() => setLevels((old) => ({ ...old, [level.key]: !old[level.key] }))}
            />
          ))}
        </div>
        <span className="hidden h-4 w-px bg-border sm:block" aria-hidden="true" />
        <div className="flex flex-wrap items-center gap-1" role="group" aria-label="Categories">
          {CATEGORIES.map((category) => (
            <Chip
              key={category.key}
              label={category.label}
              on={categories[category.key]}
              onToggle={() => setCategories((old) => ({ ...old, [category.key]: !old[category.key] }))}
            />
          ))}
        </div>
        <div className="relative ml-auto w-full min-w-40 sm:w-56">
          <Search className="pointer-events-none absolute top-1/2 left-2.5 size-3.5 -translate-y-1/2 text-muted-foreground" />
          <Input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search the events"
            className="h-8 pl-8 text-xs"
            aria-label="Search the events"
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
        <Hint label={follow ? "Following new records" : "Jump to the newest record"}>
          <Button
            variant={follow ? "secondary" : "ghost"}
            size="icon-sm"
            aria-label="Follow the newest record"
            onClick={() => setFollow(true)}
          >
            <ArrowDownToLine />
          </Button>
        </Hint>
      </div>

      <div
        ref={scrollRef}
        onScroll={onScroll}
        className={cn("relative overflow-auto bg-background/40 font-mono text-[12px] leading-5", heightClass)}
        role="log"
        aria-live="off"
      >
        {filtered.length === 0 ? (
          <div className="flex h-full items-center justify-center px-6 text-center font-sans text-sm text-muted-foreground">
            {records.length === 0
              ? "The launcher's events appear here: model loads and stops, the watchdog, downloads, settings changes and API requests."
              : "No records match the filters."}
          </div>
        ) : (
          <div style={{ height: virtualizer.getTotalSize(), position: "relative", width: "100%" }}>
            {virtualizer.getVirtualItems().map((item) => {
              const record = filtered[item.index];
              return (
                <div
                  key={item.key}
                  data-index={item.index}
                  ref={virtualizer.measureElement}
                  className={cn("absolute top-0 left-0 w-full px-3 py-px hover:bg-muted/50", ROW_TONE[record.level])}
                  style={{ transform: `translateY(${item.start}px)` }}
                >
                  <EventRow
                    record={record}
                    query={highlight}
                    open={expanded.has(record.seq)}
                    onToggle={() => toggle(record.seq)}
                  />
                </div>
              );
            })}
          </div>
        )}
      </div>
      <div className="flex items-center gap-3 border-t px-3 py-1 text-[11px] text-muted-foreground tabular-nums">
        <span className="shrink-0">
          {filtered.length.toLocaleString()} of {records.length.toLocaleString()} records
        </span>
        {paths ? (
          <span className="flex min-w-0 flex-1 items-center justify-center gap-1">
            <code className="truncate font-mono" title={paths.events_file}>
              {paths.events_file}
            </code>
            <CopyButton text={paths.events_file} size="icon-xs" className="size-5" toastMessage="Path copied" />
          </span>
        ) : (
          <span className="flex-1" />
        )}
        <span className="hidden shrink-0 sm:inline">{follow ? "Following" : "Paused (scroll to the bottom to follow)"}</span>
      </div>
    </div>
  );
}
