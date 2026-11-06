import { useRef, useState, type PointerEvent } from "react";
import { Link } from "react-router";
import { GripHorizontal, Maximize2, Terminal } from "lucide-react";

import { ConsolePanel } from "@/components/ConsolePanel";
import { StateBadge } from "@/components/StateBadge";
import { Button } from "@/components/ui/button";
import { useLive } from "@/lib/events";
import { readStorage, writeStorage } from "@/lib/utils";

const HEIGHT_KEY = "infernix.dashboard.consoleHeight";
const DEFAULT_HEIGHT = 360;
const MIN_HEIGHT = 180;
const MAX_HEIGHT = 1200;

function storedHeight(): number {
  const value = Number(readStorage(HEIGHT_KEY));
  return Number.isFinite(value) && value >= MIN_HEIGHT && value <= MAX_HEIGHT ? value : DEFAULT_HEIGHT;
}

/** The server console at the bottom of the dashboard, resizable by its handle (height remembered
 * in this browser). With no server running it keeps the last instance's lines. */
export function DashboardConsole() {
  const instance = useLive((live) => live.instance);
  const [height, setHeight] = useState(storedHeight);
  const drag = useRef<{ startY: number; startHeight: number } | null>(null);

  const onPointerDown = (event: PointerEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    drag.current = { startY: event.clientY, startHeight: height };
  };
  const onPointerMove = (event: PointerEvent<HTMLDivElement>) => {
    if (!drag.current) return;
    const next = drag.current.startHeight + (event.clientY - drag.current.startY);
    setHeight(Math.max(MIN_HEIGHT, Math.min(MAX_HEIGHT, Math.round(next))));
  };
  const onPointerUp = (event: PointerEvent<HTMLDivElement>) => {
    if (!drag.current) return;
    drag.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
    writeStorage(HEIGHT_KEY, String(height));
  };
  const nudge = (delta: number) => {
    const next = Math.max(MIN_HEIGHT, Math.min(MAX_HEIGHT, height + delta));
    setHeight(next);
    writeStorage(HEIGHT_KEY, String(next));
  };

  const running = instance !== null && instance.state !== "idle" && instance.state !== "failed";
  const last = instance?.alias ?? instance?.name ?? null;

  return (
    <section className="flex flex-col gap-2" aria-labelledby="dashboard-console">
      <div className="flex flex-wrap items-center gap-2">
        <h2 id="dashboard-console" className="flex items-center gap-2 text-sm font-semibold">
          <Terminal className="size-4 text-muted-foreground" />
          Server console
        </h2>
        {instance ? <StateBadge state={instance.state} /> : null}
        <span className="min-w-0 truncate text-xs text-muted-foreground">
          {running && instance?.pid
            ? `infernix-serve for ${last ?? "the model"} · pid ${instance.pid}${instance.port ? ` · port ${instance.port}` : ""}`
            : last
              ? `last run: ${last}`
              : "appears when a model loads"}
        </span>
        <Button variant="ghost" size="sm" className="ml-auto" asChild>
          <Link to="/logs">
            <Maximize2 />
            Full view
          </Link>
        </Button>
      </div>
      <div className="flex flex-col">
        <ConsolePanel height={height} className="rounded-b-none" />
        <div
          role="separator"
          aria-orientation="horizontal"
          aria-label="Resize the console"
          aria-valuemin={MIN_HEIGHT}
          aria-valuemax={MAX_HEIGHT}
          aria-valuenow={height}
          tabIndex={0}
          className="console-resize flex h-4 items-center justify-center rounded-b-xl border border-t-0 bg-card text-muted-foreground transition-colors outline-none hover:bg-muted/60 focus-visible:ring-[3px] focus-visible:ring-ring/50"
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={onPointerUp}
          onDoubleClick={() => nudge(DEFAULT_HEIGHT - height)}
          onKeyDown={(event) => {
            if (event.key === "ArrowDown") {
              event.preventDefault();
              nudge(40);
            } else if (event.key === "ArrowUp") {
              event.preventDefault();
              nudge(-40);
            }
          }}
          title="Drag to resize; double-click to reset"
        >
          <GripHorizontal className="size-3.5" />
        </div>
      </div>
    </section>
  );
}
