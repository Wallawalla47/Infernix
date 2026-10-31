import { StateDot } from "@/components/StateBadge";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectSeparator,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Hint } from "@/components/ui/tooltip";
import type { InstanceInfo, InstanceState } from "@/lib/api-types";
import { useLive } from "@/lib/events";
import { STATE_LABELS } from "@/lib/format";
import { useLibrary } from "@/lib/hooks";
import { findConfig } from "@/lib/library";

/** The Select value of "Loaded model" (Radix items cannot have an empty value). */
const LOADED = "@loaded";

/** Every configuration of the library by model id, grouped by artifact, and "Loaded model"
 * (null: whatever the launcher has loaded). */
export function ModelPicker({ value, onChange }: { value: string | null; onChange: (configId: string | null) => void }) {
  const { data: library } = useLibrary();
  const instance = useLive((live) => live.instance);
  const loadedAlias = instance && instance.state !== "idle" ? instance.alias : null;
  const missing = value !== null && findConfig(library, value) === null;
  return (
    <Select value={value ?? LOADED} onValueChange={(next) => onChange(next === LOADED ? null : next)}>
      <SelectTrigger size="sm" className="w-56 max-w-full sm:w-64" aria-label="Model">
        <SelectValue />
      </SelectTrigger>
      <SelectContent className="max-h-96 min-w-64">
        <SelectItem value={LOADED}>{loadedAlias ? `Loaded model (${loadedAlias})` : "Loaded model"}</SelectItem>
        {missing && value !== null ? <SelectItem value={value}>{library ? "Unavailable model" : value}</SelectItem> : null}
        {library?.models.map((model) => (
          <SelectGroup key={model.key}>
            <SelectSeparator />
            <SelectLabel className="truncate">{model.name}</SelectLabel>
            {model.configs.map((config) => (
              <SelectItem key={config.id} value={config.id}>
                {config.alias}
              </SelectItem>
            ))}
          </SelectGroup>
        ))}
      </SelectContent>
    </Select>
  );
}

function stateOf(configId: string | null, instance: InstanceInfo | null): { state: InstanceState; label: string; hint: string } {
  const loaded = instance && instance.state !== "idle" ? instance : null;
  if (configId === null) {
    if (!loaded) {
      return { state: "idle", label: "Nothing loaded", hint: "The launcher loads its default model with the first message, if one is set" };
    }
    return {
      state: loaded.state,
      label: STATE_LABELS[loaded.state],
      hint: `${loaded.alias ?? "The model"}: ${loaded.phase || STATE_LABELS[loaded.state]}`,
    };
  }
  if (loaded && loaded.model_key === configId) {
    return { state: loaded.state, label: STATE_LABELS[loaded.state], hint: loaded.phase || "Loaded" };
  }
  if (instance?.pending_model === configId) {
    return { state: "starting", label: "Switching", hint: "Loads once the current model's requests finish" };
  }
  return { state: "idle", label: "Not loaded", hint: "The launcher loads it with the first message (the reply waits for the load)" };
}

/** Whether the chat's model is loaded (ready or loading), from the launcher's instance. */
export function ModelStatus({ configId }: { configId: string | null }) {
  const instance = useLive((live) => live.instance);
  const { state, label, hint } = stateOf(configId, instance);
  return (
    <Hint label={hint}>
      <span className="inline-flex min-w-0 cursor-default items-center gap-1.5 text-xs text-muted-foreground">
        <StateDot state={state} />
        <span className="truncate">{label}</span>
      </span>
    </Hint>
  );
}
