// Importing a launch script (.bat/.cmd/.sh): the launcher finds its infernix-serve commands and
// matches each to a library artifact; each matched command can replace a configuration's
// parameters or become a new configuration.

import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router";
import { Check, CopyPlus, FileUp, LoaderCircle, ScanSearch, TriangleAlert } from "lucide-react";
import { toast } from "sonner";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
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
import { Textarea } from "@/components/ui/textarea";
import type { ImportedCommand, ModelConfig } from "@/lib/api-types";
import { argsToText } from "@/lib/args";
import {
  aliasProblem,
  configNameProblem,
  configPath,
  useCreateConfig,
  useImportScript,
  useSaveConfig,
} from "@/lib/configs";
import { useLibrary, useTakesValue } from "@/lib/hooks";

/** "start-27b" from "C:\scripts\start-27b.bat". */
function scriptStem(fileName: string | null): string | null {
  if (!fileName) return null;
  const base = fileName.split(/[\\/]/).pop() ?? fileName;
  const dot = base.lastIndexOf(".");
  const stem = dot > 0 ? base.slice(0, dot) : base;
  return stem.length > 0 ? stem : null;
}

type Outcome = { kind: "applied" | "created"; config: ModelConfig };

function CommandCard({
  command,
  index,
  fileName,
  preferConfigId,
  onApplied,
}: {
  command: ImportedCommand;
  index: number;
  fileName: string | null;
  preferConfigId?: string;
  onApplied?: (config: ModelConfig, created: boolean) => void;
}) {
  const { data: library } = useLibrary();
  const takesValue = useTakesValue();
  const save = useSaveConfig();
  const create = useCreateConfig();
  const model = command.model ? (library?.models.find((item) => item.key === command.model?.key) ?? null) : null;
  const configs = model?.configs ?? [];
  const preferred = configs.find((config) => config.id === preferConfigId) ?? configs[0] ?? null;
  const [target, setTarget] = useState<string>(preferred?.id ?? "");
  const [alias, setAlias] = useState(command.alias ?? "");
  const [name, setName] = useState(() => scriptStem(fileName) ?? command.alias ?? `Imported ${index + 1}`);
  const [outcomes, setOutcomes] = useState<Outcome[]>([]);

  useEffect(() => {
    if (target === "" && preferred) setTarget(preferred.id);
  }, [target, preferred]);

  const targetConfig = configs.find((config) => config.id === target) ?? null;
  const aliasText = alias.trim();
  const applyAliasProblem = aliasText.length > 0 ? aliasProblem(aliasText, library, target || null) : null;
  const createAliasProblem = aliasText.length > 0 ? aliasProblem(aliasText, library, null) : null;
  const nameProblem = configNameProblem(name, model, null);
  const createProblem = nameProblem ?? createAliasProblem;
  const busy = save.isPending || create.isPending;
  const argsText = argsToText(command.args, takesValue);
  const navigate = useNavigate();
  const configureAction = (config: ModelConfig) => ({
    action: { label: "Configure", onClick: () => navigate(configPath(config.id)) },
  });

  const apply = () => {
    if (!targetConfig || applyAliasProblem) return;
    save.mutate(
      {
        id: targetConfig.id,
        patch: { args: command.args, env: command.env, ...(aliasText.length > 0 ? { alias: aliasText } : {}) },
      },
      {
        onSuccess: (config) => {
          setOutcomes((old) => [...old, { kind: "applied", config }]);
          toast.success(`Applied to ${config.name} (model id ${config.alias})`, configureAction(config));
          onApplied?.(config, false);
        },
      },
    );
  };

  const createConfig = () => {
    if (!command.model || nameProblem || createAliasProblem) return;
    create.mutate(
      {
        artifactKey: command.model.key,
        body: {
          name: name.trim(),
          args: command.args,
          env: command.env,
          alias: aliasText.length > 0 ? aliasText : undefined,
        },
      },
      {
        onSuccess: (config) => {
          setOutcomes((old) => [...old, { kind: "created", config }]);
          toast.success(`Created ${config.name} (model id ${config.alias})`, configureAction(config));
          onApplied?.(config, true);
        },
      },
    );
  };

  return (
    <li className="flex flex-col gap-3 rounded-lg border p-3">
      <div className="flex flex-wrap items-start gap-2">
        <div className="min-w-0 flex-1">
          <div className="text-[11px] font-medium tracking-wide text-muted-foreground uppercase">
            Command {index + 1}
          </div>
          <div className="font-mono text-xs break-all">{command.artifact}</div>
        </div>
        {command.model ? (
          <Badge variant="accent" className="max-w-full truncate">
            {command.model.name}
          </Badge>
        ) : (
          <Badge variant="destructive">not in the library</Badge>
        )}
      </div>

      <div className="grid gap-2 text-xs sm:grid-cols-[7rem_minmax(0,1fr)]">
        <div className="text-muted-foreground">Parameters</div>
        {command.args.length > 0 ? (
          <pre className="max-h-40 overflow-auto rounded-md border bg-muted/40 px-2 py-1.5 font-mono text-[11px] leading-5 whitespace-pre-wrap break-all">
            {argsText}
          </pre>
        ) : (
          <div className="text-muted-foreground">none (the defaults of infernix-serve)</div>
        )}
        <div className="text-muted-foreground">Environment</div>
        {command.env.length > 0 ? (
          <ul className="flex flex-col gap-0.5 font-mono text-[11px] break-all">
            {command.env.map((variable, position) => (
              <li key={`${position}-${variable.name}`}>
                {variable.name}={variable.value}
              </li>
            ))}
          </ul>
        ) : (
          <div className="text-muted-foreground">none</div>
        )}
        <div className="text-muted-foreground">Model id</div>
        <div className="font-mono">{command.alias ?? <span className="font-sans text-muted-foreground">not set</span>}</div>
      </div>

      {command.warnings.length > 0 ? (
        <Alert variant="warning">
          <TriangleAlert />
          <AlertTitle>
            {command.warnings.length} warning{command.warnings.length === 1 ? "" : "s"}
          </AlertTitle>
          <AlertDescription>
            <ul className="flex flex-col gap-0.5 text-xs">
              {command.warnings.map((warning, position) => (
                <li key={`${position}-${warning}`}>{warning}</li>
              ))}
            </ul>
          </AlertDescription>
        </Alert>
      ) : null}

      {outcomes.length > 0 ? (
        <ul className="flex flex-col gap-1 text-xs">
          {outcomes.map((outcome, position) => (
            <li key={`${outcome.config.id}-${position}`} className="flex items-center gap-1.5 text-success">
              <Check className="size-3.5" />
              <span className="text-foreground">
                {outcome.kind === "applied" ? "Applied to" : "Created"} {outcome.config.name} · model id{" "}
                <span className="font-mono">{outcome.config.alias}</span>
              </span>
            </li>
          ))}
        </ul>
      ) : null}

      {command.model && model ? (
        <div className="flex flex-col gap-3 border-t pt-3">
          <div className="flex flex-col gap-1">
            <label className="text-xs font-medium" htmlFor={`import-alias-${index}`}>
              Model id to set <span className="font-normal text-muted-foreground">(empty keeps the configuration's)</span>
            </label>
            <Input
              id={`import-alias-${index}`}
              value={alias}
              spellCheck={false}
              autoComplete="off"
              className="h-8 font-mono text-xs"
              placeholder="unchanged"
              onChange={(event) => setAlias(event.target.value)}
            />
          </div>
          <div className="flex flex-col gap-1">
            <div className="flex flex-wrap items-center gap-2">
              <Select value={target} onValueChange={setTarget}>
                <SelectTrigger size="sm" className="min-w-0 flex-1" aria-label="Configuration to replace">
                  <SelectValue placeholder="Choose a configuration" />
                </SelectTrigger>
                <SelectContent>
                  {configs.map((config) => (
                    <SelectItem key={config.id} value={config.id}>
                      {config.name} · {config.alias}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <Button size="sm" variant="outline" disabled={busy || !targetConfig || applyAliasProblem !== null} onClick={apply}>
                {save.isPending ? <LoaderCircle className="animate-spin" /> : <Check />}
                Apply to {targetConfig?.name ?? "configuration"}
              </Button>
            </div>
            {applyAliasProblem ? <div className="text-xs text-critical">{applyAliasProblem}</div> : null}
            <div className="text-xs text-muted-foreground">
              Replaces its launch parameters and environment variables.
            </div>
          </div>
          <div className="flex flex-col gap-1">
            <div className="flex flex-wrap items-center gap-2">
              <Input
                value={name}
                className="h-8 min-w-0 flex-1"
                aria-label="Name of the new configuration"
                aria-invalid={nameProblem !== null}
                onChange={(event) => setName(event.target.value)}
              />
              <Button
                size="sm"
                disabled={busy || nameProblem !== null || createAliasProblem !== null}
                onClick={createConfig}
              >
                {create.isPending ? <LoaderCircle className="animate-spin" /> : <CopyPlus />}
                Create configuration
              </Button>
            </div>
            {createProblem ? <div className="text-xs text-critical">{createProblem}</div> : null}
          </div>
        </div>
      ) : command.model ? (
        <div className="border-t pt-3 text-xs text-muted-foreground">
          {command.model.name} is not in the library any more; rescan the model folders.
        </div>
      ) : (
        <div className="border-t pt-3 text-xs text-muted-foreground">
          Add the folder holding <span className="font-mono">{command.artifact}</span> to your model folders, then import
          again.
        </div>
      )}
    </li>
  );
}

/** Reads a launch script (pasted, or a file read in the browser) and lists its commands. */
export function ImportScriptDialog({
  open,
  onOpenChange,
  preferConfigId,
  onApplied,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** The configuration "Apply to" proposes when a command names its artifact. */
  preferConfigId?: string;
  onApplied?: (config: ModelConfig, created: boolean) => void;
}) {
  const parse = useImportScript();
  const [text, setText] = useState("");
  const [fileName, setFileName] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const resetParse = parse.reset;
  useEffect(() => {
    if (!open) return;
    setText("");
    setFileName(null);
    resetParse();
  }, [open, resetParse]);

  const readFile = (file: File) => {
    const reader = new FileReader();
    reader.onload = () => {
      const content = typeof reader.result === "string" ? reader.result : "";
      setText(content);
      setFileName(file.name);
      parse.reset();
      if (content.trim().length > 0) parse.mutate({ text: content, fileName: file.name });
    };
    reader.onerror = () => toast.error(`Could not read ${file.name}`);
    reader.readAsText(file);
  };

  const result = parse.data ?? null;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-3xl">
        <DialogHeader>
          <DialogTitle>Import a launch script</DialogTitle>
          <DialogDescription>
            Paste a .bat, .cmd or .sh script that starts infernix-serve (or ninfer-serve), or pick the file. The
            launcher reads its commands and environment variables and matches each command to a model in your library.
          </DialogDescription>
        </DialogHeader>

        <div className="flex flex-col gap-2">
          <Textarea
            value={text}
            spellCheck={false}
            rows={result ? 5 : 10}
            placeholder={"set CUDA_VISIBLE_DEVICES=0\ninfernix-serve.exe D:\\models\\Qwen3.8-27B.infernix --max-context 131072 ^\n  --kv-dtype int8 --spec auto"}
            className="font-mono text-xs leading-5"
            aria-label="Launch script"
            onChange={(event) => {
              setText(event.target.value);
              if (event.target.value.length === 0) setFileName(null);
            }}
          />
          <div className="flex flex-wrap items-center gap-2">
            <input
              ref={fileRef}
              type="file"
              accept=".bat,.cmd,.sh,text/plain"
              className="hidden"
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (file) readFile(file);
                event.target.value = "";
              }}
            />
            <Button variant="outline" size="sm" onClick={() => fileRef.current?.click()}>
              <FileUp />
              Choose a file
            </Button>
            {fileName ? <span className="truncate font-mono text-xs text-muted-foreground">{fileName}</span> : null}
            <Button
              size="sm"
              className="ml-auto"
              disabled={text.trim().length === 0 || parse.isPending}
              onClick={() => parse.mutate({ text, fileName: fileName ?? undefined })}
            >
              {parse.isPending ? <LoaderCircle className="animate-spin" /> : <ScanSearch />}
              Read the commands
            </Button>
          </div>
        </div>

        {result ? (
          result.commands.length === 0 ? (
            <div className="rounded-lg border border-dashed p-6 text-center text-sm text-muted-foreground">
              No infernix-serve or ninfer-serve command found in this script.
            </div>
          ) : (
            <div className="flex flex-col gap-2">
              <div className="text-sm font-medium">
                {result.commands.length} command{result.commands.length === 1 ? "" : "s"} found
              </div>
              <ul className="flex flex-col gap-3">
                {result.commands.map((command, index) => (
                  <CommandCard
                    key={`${parse.submittedAt}-${index}`}
                    command={command}
                    index={index}
                    fileName={fileName}
                    preferConfigId={preferConfigId}
                    onApplied={onApplied}
                  />
                ))}
              </ul>
            </div>
          )
        ) : null}

        <DialogFooter>
          <Button variant="ghost" onClick={() => onOpenChange(false)}>
            Close
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
