import { useEffect, useRef, useState, type ReactNode } from "react";
import { Check, LoaderCircle, Pencil, X } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Hint } from "@/components/ui/tooltip";
import { errorMessage } from "@/lib/api";
import { cn } from "@/lib/utils";

/** A value shown as text with an edit button. Editing swaps in a text box that saves on Enter or
 * the check button and cancels on Escape; `validate` blocks saving with its reason, and a failed
 * save (the launcher's 400) keeps the box open with the launcher's message. */
export function InlineEdit({
  value,
  onSave,
  validate,
  placeholder,
  label,
  textClassName,
  inputClassName,
  mono = false,
  extra,
  className,
}: {
  value: string;
  /** Resolves when saved; a rejection keeps the editor open. */
  onSave: (text: string) => Promise<unknown>;
  validate?: (text: string) => string | null;
  placeholder?: string;
  /** Accessible name of the value ("Model id"). */
  label: string;
  textClassName?: string;
  inputClassName?: string;
  mono?: boolean;
  /** Shown after the edit button while not editing (e.g. a reset button). */
  extra?: ReactNode;
  className?: string;
}) {
  const [editing, setEditing] = useState(false);
  const [text, setText] = useState(value);
  const [saving, setSaving] = useState(false);
  const [failure, setFailure] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!editing) setText(value);
  }, [value, editing]);

  useEffect(() => {
    if (editing) inputRef.current?.select();
  }, [editing]);

  const problem = editing && validate ? validate(text) : null;
  const message = problem ?? failure;

  const cancel = () => {
    setEditing(false);
    setFailure(null);
    setText(value);
  };

  const commit = async () => {
    if (saving || problem) return;
    if (text.trim() === value) {
      cancel();
      return;
    }
    setSaving(true);
    setFailure(null);
    try {
      await onSave(text.trim());
      setEditing(false);
    } catch (error) {
      setFailure(errorMessage(error));
    } finally {
      setSaving(false);
    }
  };

  if (!editing) {
    return (
      <div className={cn("flex min-w-0 items-center gap-1", className)}>
        <span className={cn("min-w-0 truncate", mono && "font-mono", textClassName)} title={value}>
          {value}
        </span>
        <Hint label={`Rename the ${label.toLowerCase()}`}>
          <Button
            variant="ghost"
            size="icon-xs"
            className="shrink-0 text-muted-foreground"
            aria-label={`Edit the ${label.toLowerCase()}`}
            onClick={() => setEditing(true)}
          >
            <Pencil />
          </Button>
        </Hint>
        {extra}
      </div>
    );
  }

  return (
    <div className={cn("flex min-w-0 flex-col gap-1", className)}>
      <div className="flex min-w-0 items-center gap-1">
        <Input
          ref={inputRef}
          value={text}
          placeholder={placeholder}
          spellCheck={false}
          autoComplete="off"
          aria-label={label}
          aria-invalid={message !== null}
          disabled={saving}
          className={cn("h-8 min-w-0", mono && "font-mono", inputClassName)}
          onChange={(event) => {
            setText(event.target.value);
            setFailure(null);
          }}
          onKeyDown={(event) => {
            if (event.key === "Enter") {
              event.preventDefault();
              void commit();
            } else if (event.key === "Escape") {
              event.preventDefault();
              cancel();
            }
          }}
        />
        <Button
          variant="ghost"
          size="icon-sm"
          aria-label="Save"
          disabled={saving || problem !== null}
          onClick={() => void commit()}
        >
          {saving ? <LoaderCircle className="animate-spin" /> : <Check />}
        </Button>
        <Button variant="ghost" size="icon-sm" aria-label="Cancel" disabled={saving} onClick={cancel}>
          <X />
        </Button>
      </div>
      {message ? <div className="text-xs text-critical">{message}</div> : null}
    </div>
  );
}
