import { useEffect, useState } from "react";
import { CopyPlus, LoaderCircle } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import type { EnvVar, ModelArtifact, ModelConfig } from "@/lib/api-types";
import { aliasProblem, configNameProblem, useCreateConfig } from "@/lib/configs";
import { useLibrary } from "@/lib/hooks";

const DEFAULTS = "__defaults__";

/** Creates a named configuration of an artifact: a copy of one of its configurations or of the
 * defaults, under a new name and (optionally) a chosen model id. */
export function NewConfigDialog({
  open,
  onOpenChange,
  model,
  from,
  unsaved,
  onCreated,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  model: ModelArtifact;
  /** The configuration proposed as the source; null proposes the defaults. */
  from: string | null;
  /** Unsaved edits of `from` the copy takes instead of its saved parameters. */
  unsaved?: { args: string[]; env: EnvVar[] } | null;
  /** The caller announces it (and may open it); `carriedUnsaved` says the copy took `unsaved`. */
  onCreated: (config: ModelConfig, carriedUnsaved: boolean) => void;
}) {
  const { data: library } = useLibrary();
  const create = useCreateConfig();
  const [name, setName] = useState("");
  const [alias, setAlias] = useState("");
  const [source, setSource] = useState<string>(from ?? DEFAULTS);
  const [touched, setTouched] = useState(false);

  useEffect(() => {
    if (!open) return;
    setName("");
    setAlias("");
    setSource(from ?? DEFAULTS);
    setTouched(false);
  }, [open, from]);

  const nameProblem = configNameProblem(name, model, null);
  const idProblem = alias.trim().length > 0 ? aliasProblem(alias, library, null) : null;
  const carried = unsaved && source === from ? unsaved : null;

  const submit = () => {
    setTouched(true);
    if (nameProblem || idProblem || create.isPending) return;
    create.mutate(
      {
        artifactKey: model.key,
        body: {
          name: name.trim(),
          from: source === DEFAULTS ? undefined : source,
          alias: alias.trim().length > 0 ? alias.trim() : undefined,
          args: carried?.args,
          env: carried?.env,
        },
      },
      {
        onSuccess: (config) => {
          onOpenChange(false);
          onCreated(config, carried !== null);
        },
      },
    );
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>New configuration</DialogTitle>
          <DialogDescription>
            Another set of launch parameters for {model.name}. Each configuration is a separate model in /v1/models
            under its own model id; loading it starts the artifact with its parameters.
          </DialogDescription>
        </DialogHeader>
        <form
          className="flex flex-col gap-4"
          onSubmit={(event) => {
            event.preventDefault();
            submit();
          }}
        >
          <div className="flex flex-col gap-1.5">
            <label htmlFor="new-config-name" className="text-sm font-medium">
              Name
            </label>
            <Input
              id="new-config-name"
              value={name}
              autoFocus
              placeholder="Long context, Fast, Agents…"
              aria-invalid={touched && nameProblem !== null}
              onChange={(event) => setName(event.target.value)}
            />
            {touched && nameProblem ? <div className="text-xs text-critical">{nameProblem}</div> : null}
          </div>
          <div className="flex flex-col gap-1.5">
            <div className="text-sm font-medium">Copy of</div>
            <Select value={source} onValueChange={setSource}>
              <SelectTrigger aria-label="Copy of">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value={DEFAULTS}>The defaults for this artifact</SelectItem>
                {model.configs.map((config) => (
                  <SelectItem key={config.id} value={config.id}>
                    {config.name} · {config.alias}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {carried ? (
              <div className="text-xs text-muted-foreground">Includes the unsaved changes on this page.</div>
            ) : null}
          </div>
          <div className="flex flex-col gap-1.5">
            <label htmlFor="new-config-alias" className="text-sm font-medium">
              Model id <span className="font-normal text-muted-foreground">(optional)</span>
            </label>
            <Input
              id="new-config-alias"
              value={alias}
              spellCheck={false}
              autoComplete="off"
              className="font-mono"
              placeholder="Chosen by the launcher"
              aria-invalid={idProblem !== null}
              onChange={(event) => setAlias(event.target.value)}
            />
            {idProblem ? (
              <div className="text-xs text-critical">{idProblem}</div>
            ) : (
              <div className="text-xs text-muted-foreground">What coding tools send as `model`; editable later.</div>
            )}
          </div>
          <DialogFooter>
            <Button variant="ghost" onClick={() => onOpenChange(false)}>
              Cancel
            </Button>
            <Button type="submit" disabled={create.isPending || (touched && nameProblem !== null) || idProblem !== null}>
              {create.isPending ? <LoaderCircle className="animate-spin" /> : <CopyPlus />}
              Create
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
