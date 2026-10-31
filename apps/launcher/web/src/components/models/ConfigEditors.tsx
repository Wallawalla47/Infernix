// Inline editors of a configuration's model id (alias) and name, saved at once on their own
// (separately from the Configure page's launch-parameter draft).

import { Undo2 } from "lucide-react";
import { toast } from "sonner";

import { InlineEdit } from "@/components/models/InlineEdit";
import { Button } from "@/components/ui/button";
import { Hint } from "@/components/ui/tooltip";
import type { ModelArtifact, ModelConfig } from "@/lib/api-types";
import { aliasProblem, configNameProblem, useSaveConfig } from "@/lib/configs";
import { useLibrary } from "@/lib/hooks";
import { cn } from "@/lib/utils";

/** The model id coding tools see and send. An empty box saves the default id. */
export function ModelIdEditor({
  config,
  size = "sm",
  className,
}: {
  config: ModelConfig;
  size?: "sm" | "lg";
  className?: string;
}) {
  const { data: library } = useLibrary();
  const save = useSaveConfig();
  const resolve = (text: string) => (text.trim().length > 0 ? text.trim() : config.default_alias);
  const saveAlias = (alias: string) =>
    save.mutateAsync({ id: config.id, patch: { alias } }).then((saved) => {
      toast.success(`The model id is now ${saved.alias}`);
    });
  const custom = config.alias !== config.default_alias;
  return (
    <InlineEdit
      label="Model id"
      value={config.alias}
      placeholder={config.default_alias}
      validate={(text) => aliasProblem(resolve(text), library, config.id)}
      onSave={(text) => saveAlias(resolve(text))}
      className={className}
      textClassName={cn(size === "lg" ? "text-2xl font-semibold tracking-tight" : "text-sm font-semibold")}
      inputClassName={size === "lg" ? "h-9 text-base sm:w-96" : undefined}
      extra={
        custom ? (
          <Hint label={`Reset to ${config.default_alias}`}>
            <Button
              variant="ghost"
              size="icon-xs"
              className="shrink-0 text-muted-foreground"
              aria-label={`Reset the model id to ${config.default_alias}`}
              disabled={save.isPending}
              onClick={() => {
                const problem = aliasProblem(config.default_alias, library, config.id);
                if (problem) {
                  toast.error(problem);
                  return;
                }
                void saveAlias(config.default_alias).catch(() => undefined);
              }}
            >
              <Undo2 />
            </Button>
          </Hint>
        ) : null
      }
    />
  );
}

/** A named configuration's name; the default configuration's ("Default") is fixed. */
export function ConfigNameEditor({
  model,
  config,
  className,
  textClassName,
}: {
  model: ModelArtifact;
  config: ModelConfig;
  className?: string;
  textClassName?: string;
}) {
  const save = useSaveConfig();
  if (config.is_default) {
    return <span className={cn("truncate", textClassName, className)}>{config.name}</span>;
  }
  return (
    <InlineEdit
      label="Configuration name"
      value={config.name}
      placeholder="Name"
      validate={(text) => configNameProblem(text, model, config.id)}
      onSave={(text) =>
        save.mutateAsync({ id: config.id, patch: { name: text } }).then((saved) => {
          toast.success(`Renamed the configuration to ${saved.name}`);
        })
      }
      className={className}
      textClassName={textClassName}
      inputClassName="sm:w-56"
    />
  );
}
