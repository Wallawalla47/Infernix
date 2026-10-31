// Per-configuration idle unload (`idle_unload_min`): null follows Settings.idle_unload_min, 0 never
// unloads, N unloads after N minutes without requests.

import { useState } from "react";

import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { CommitInput } from "@/pages/configure/controls";

type Mode = "global" | "never" | "after";

function modeOf(value: number | null): Mode {
  if (value === null) return "global";
  return value === 0 ? "never" : "after";
}

export function IdleUnloadField({
  value,
  globalMinutes,
  onChange,
}: {
  value: number | null;
  /** Settings.idle_unload_min, when known. */
  globalMinutes: number | null;
  onChange: (value: number | null) => void;
}) {
  const mode = modeOf(value);
  // The minutes to restore when "After" is picked again.
  const [lastMinutes, setLastMinutes] = useState(() =>
    value !== null && value > 0 ? value : globalMinutes !== null && globalMinutes > 0 ? globalMinutes : 30,
  );
  const globalText =
    globalMinutes === null ? "the global setting" : globalMinutes === 0 ? "never (global setting)" : `${globalMinutes} min (global setting)`;

  return (
    <div className="flex flex-col gap-3 py-2.5 sm:flex-row sm:items-center sm:justify-between">
      <div className="min-w-0 sm:max-w-[58%]">
        <div className="text-sm font-medium">Unload when idle</div>
        <div className="mt-0.5 text-xs text-muted-foreground">
          Unload after this many minutes without requests, freeing the GPU. Now: {mode === "global" ? globalText : mode === "never" ? "never" : `${value} min`}.
        </div>
      </div>
      <div className="flex flex-wrap items-center gap-2 sm:justify-end">
        <ToggleGroup
          type="single"
          value={mode}
          aria-label="Unload when idle"
          onValueChange={(next) => {
            if (next === "global") onChange(null);
            else if (next === "never") onChange(0);
            else if (next === "after") onChange(lastMinutes);
          }}
        >
          <ToggleGroupItem value="global">Global</ToggleGroupItem>
          <ToggleGroupItem value="never">Never</ToggleGroupItem>
          <ToggleGroupItem value="after">After</ToggleGroupItem>
        </ToggleGroup>
        {mode === "after" ? (
          <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
            <CommitInput
              type="number"
              className="h-8 w-20"
              value={String(value)}
              min={1}
              ariaLabel="Minutes without requests"
              onCommit={(text) => {
                const minutes = Math.round(Number(text));
                if (text.length === 0 || !Number.isFinite(minutes) || minutes < 0) return;
                if (minutes > 0) setLastMinutes(minutes);
                onChange(minutes);
              }}
            />
            min
          </div>
        ) : null}
      </div>
    </div>
  );
}
