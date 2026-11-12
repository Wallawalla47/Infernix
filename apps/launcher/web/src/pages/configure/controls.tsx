// The controls of the structured launch-parameter form. Each reads and writes one flag of the args
// array, which stays the single source of truth (unknown flags pass through untouched). The
// options themselves, grouped into the form's sections, are in options.ts.

import { useEffect, useState, type ReactNode } from "react";
import { CircleHelp } from "lucide-react";

import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Hint } from "@/components/ui/tooltip";
import { getFlag, getFlagValues, setFlag, setFlagValues, type TakesValue } from "@/lib/args";
import type { ModelArtifact } from "@/lib/api-types";
import { cn } from "@/lib/utils";

export const UNSET = "__unset__";

interface Base {
  flag: string;
  label: string;
  help: string;
  /** Shown only for models this returns true for. */
  when?: (model: ModelArtifact) => boolean;
  /** A less common option: listed under the section's "More options" (open while one is set). */
  more?: boolean;
}

export type Control =
  | (Base & { kind: "switch" })
  | (Base & { kind: "number"; placeholder?: string; min?: number; max?: number; step?: number })
  | (Base & { kind: "text"; placeholder?: string; mono?: boolean; wide?: boolean })
  | (Base & { kind: "select"; options: { value: string; label: string }[]; unsetLabel: string })
  /** "auto" or a number. */
  | (Base & { kind: "auto"; placeholder?: string; autoDefault?: boolean })
  /** A repeatable option: names separated by spaces or commas, written once per name. */
  | (Base & { kind: "list"; placeholder?: string });

export interface FormContext {
  args: string[];
  setArgs: (args: string[]) => void;
  takesValue: TakesValue;
}

function Row({
  label,
  help,
  flag,
  children,
  wide = false,
}: {
  label: string;
  help: string;
  flag: string;
  children: ReactNode;
  wide?: boolean;
}) {
  return (
    <div className={cn("flex gap-3 py-2.5", wide ? "flex-col" : "flex-col sm:flex-row sm:items-center sm:justify-between")}>
      <div className="min-w-0 sm:max-w-[58%]">
        <div className="flex items-center gap-1.5 text-sm font-medium">
          {label}
          <Hint label={<span className="block max-w-xs leading-relaxed">{help}</span>}>
            <CircleHelp className="size-3.5 cursor-help text-muted-foreground/70" aria-label={help} />
          </Hint>
        </div>
        <code className="mt-0.5 block font-mono text-[11px] text-muted-foreground">{flag}</code>
      </div>
      <div className={cn("min-w-0", wide ? "w-full" : "sm:w-60 sm:shrink-0")}>{children}</div>
    </div>
  );
}

/** A text box that commits its value on blur or Enter, so typing does not rewrite args per key. */
export function CommitInput({
  value,
  onCommit,
  placeholder,
  mono = false,
  type = "text",
  min,
  max,
  step,
  className,
  ariaLabel,
  disabled = false,
}: {
  value: string;
  onCommit: (value: string) => void;
  placeholder?: string;
  mono?: boolean;
  type?: "text" | "number";
  min?: number;
  max?: number;
  step?: number;
  className?: string;
  ariaLabel?: string;
  disabled?: boolean;
}) {
  const [local, setLocal] = useState(value);
  useEffect(() => setLocal(value), [value]);
  return (
    <Input
      value={local}
      type={type}
      min={min}
      max={max}
      step={step}
      disabled={disabled}
      placeholder={placeholder}
      aria-label={ariaLabel}
      spellCheck={false}
      className={cn(mono && "font-mono text-xs", className)}
      onChange={(event) => setLocal(event.target.value)}
      onBlur={() => {
        if (local !== value) onCommit(local.trim());
      }}
      onKeyDown={(event) => {
        if (event.key === "Enter") {
          event.preventDefault();
          if (local !== value) onCommit(local.trim());
        }
      }}
    />
  );
}

export function ControlField({ control, form }: { control: Control; form: FormContext }) {
  const { args, setArgs, takesValue } = form;
  const current = getFlag(args, control.flag, takesValue);
  const set = (value: string | boolean | null) => setArgs(setFlag(args, control.flag, value, takesValue));

  switch (control.kind) {
    case "switch":
      return (
        <Row label={control.label} help={control.help} flag={control.flag}>
          <div className="flex sm:justify-end">
            <Switch checked={current !== undefined} onCheckedChange={(checked) => set(checked)} aria-label={control.label} />
          </div>
        </Row>
      );
    case "number":
      return (
        <Row label={control.label} help={control.help} flag={control.flag}>
          <CommitInput
            type="number"
            value={typeof current === "string" ? current : ""}
            placeholder={control.placeholder ?? "default"}
            min={control.min}
            max={control.max}
            step={control.step}
            ariaLabel={control.label}
            onCommit={(value) => set(value.length > 0 ? value : null)}
          />
        </Row>
      );
    case "text":
      return (
        <Row label={control.label} help={control.help} flag={control.flag} wide={control.wide}>
          <CommitInput
            value={typeof current === "string" ? current : ""}
            placeholder={control.placeholder ?? "unset"}
            mono={control.mono}
            ariaLabel={control.label}
            onCommit={(value) => set(value.length > 0 ? value : null)}
          />
        </Row>
      );
    case "select": {
      const value = typeof current === "string" ? current : UNSET;
      const known = control.options.some((option) => option.value === value);
      return (
        <Row label={control.label} help={control.help} flag={control.flag}>
          <Select value={value} onValueChange={(next) => set(next === UNSET ? null : next)}>
            <SelectTrigger aria-label={control.label}>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={UNSET}>{control.unsetLabel}</SelectItem>
              {control.options.map((option) => (
                <SelectItem key={option.value} value={option.value}>
                  {option.label}
                </SelectItem>
              ))}
              {!known && value !== UNSET ? <SelectItem value={value}>{value}</SelectItem> : null}
            </SelectContent>
          </Select>
        </Row>
      );
    }
    case "list": {
      const values = getFlagValues(args, control.flag, takesValue);
      return (
        <Row label={control.label} help={control.help} flag={control.flag} wide>
          <CommitInput
            value={values.join(" ")}
            placeholder={control.placeholder ?? "none"}
            mono
            ariaLabel={control.label}
            onCommit={(value) =>
              setArgs(setFlagValues(args, control.flag, value.split(/[\s,]+/).filter((name) => name.length > 0), takesValue))
            }
          />
        </Row>
      );
    }
    case "auto": {
      const text = typeof current === "string" ? current : "";
      const isAuto = text === "auto";
      return (
        <Row label={control.label} help={control.help} flag={control.flag}>
          <div className="flex items-center gap-2">
            <CommitInput
              value={isAuto ? "" : text}
              placeholder={isAuto ? "auto" : (control.placeholder ?? "default")}
              ariaLabel={control.label}
              className={cn(isAuto && "opacity-60")}
              onCommit={(value) => set(value.length > 0 ? value : isAuto ? "auto" : null)}
            />
            <label className="flex shrink-0 items-center gap-1.5 text-xs text-muted-foreground">
              <Switch
                checked={isAuto}
                onCheckedChange={(checked) => set(checked ? "auto" : null)}
                aria-label={`${control.label}: auto`}
              />
              auto
            </label>
          </div>
        </Row>
      );
    }
  }
}
