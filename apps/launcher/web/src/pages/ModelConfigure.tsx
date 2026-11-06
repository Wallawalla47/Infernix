// The Configure page of one configuration (the route's `:key` is a configuration id: an artifact
// key for its default configuration, or `<artifact key>@<8 hex>`). The launch parameters,
// environment variables, listing and idle unload are a draft saved together; the model id and the
// configuration's name are edited inline and saved at once.

import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { Link, useNavigate, useParams } from "react-router";
import { keepPreviousData, useMutation, useQuery } from "@tanstack/react-query";
import {
  ArrowLeft,
  BookOpen,
  Brain,
  CopyPlus,
  Cpu,
  Database,
  EllipsisVertical,
  Eye,
  FileCode,
  FileDown,
  FileInput,
  Image as ImageIcon,
  Play,
  Plug,
  Plus,
  RotateCw,
  Save,
  Search,
  Sparkles,
  Trash2,
  TriangleAlert,
  Undo2,
  Variable,
  Wrench,
} from "lucide-react";
import { toast } from "sonner";

import { ConfirmDialog, EmptyState, Fact } from "@/components/common";
import { CopyButton } from "@/components/CopyButton";
import { ConfigNameEditor, ModelIdEditor } from "@/components/models/ConfigEditors";
import { ImportScriptDialog } from "@/components/models/ImportScriptDialog";
import { NewConfigDialog } from "@/components/models/NewConfigDialog";
import { StateBadge, StateDot } from "@/components/StateBadge";
import { useLoadModel } from "@/components/useModelActions";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { Skeleton } from "@/components/ui/skeleton";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import { Hint } from "@/components/ui/tooltip";
import { api } from "@/lib/api";
import type { EnvVar, Library, ModelArtifact, ModelConfig, ServeOption } from "@/lib/api-types";
import { argsToCommandLine, argsToText, getFlagString, hasFlag, sameArgs, setFlag, tokenize } from "@/lib/args";
import {
  cleanEnv,
  configPath,
  envProblem,
  freshestConfig,
  liveInstance,
  sameEnv,
  useDeleteConfig,
  useDownloadScript,
  useResetConfig,
  useSaveConfig,
} from "@/lib/configs";
import { useLive } from "@/lib/events";
import { NGRAM_FORMAT_LABELS, formatBytes, formatContext, formatInt, quantizationLabel } from "@/lib/format";
import { useDebounced, useLibrary, useOptionMap, useServeOptions, useSettings, useTakesValue } from "@/lib/hooks";
import { findConfig, findModel } from "@/lib/library";
import { baseName, cn } from "@/lib/utils";
import { EnvEditor } from "@/pages/configure/EnvEditor";
import { FitCard } from "@/pages/configure/FitPanel";
import { IdleUnloadField } from "@/pages/configure/IdleUnloadField";
import {
  ADVANCED_CONTROLS,
  CommitInput,
  ControlField,
  FORM_FLAGS,
  MEMORY_CONTROLS,
  NGRAM_CONTROLS,
  SAMPLING_CONTROLS,
  SPEC_CONTROLS,
  THINKING_CONTROLS,
  UNSET,
  VISION_CONTROLS,
  type Control,
  type FormContext,
} from "@/pages/configure/controls";

/** The unsaved edits; the model id and name are saved on their own. */
interface Draft {
  listed: boolean;
  args: string[];
  env: EnvVar[];
  /** idle_unload_min: null follows the global setting, 0 never unloads. */
  idle: number | null;
}

const CONTEXT_PRESETS = [32_768, 65_536, 131_072, 200_000, 240_000, 262_144, 524_288, 1_048_576];

function fromConfig(config: ModelConfig): Draft {
  return {
    listed: config.listed,
    args: [...config.args],
    env: config.env.map((row) => ({ ...row })),
    idle: config.idle_unload_min,
  };
}

function draftDiffers(draft: Draft, config: ModelConfig): boolean {
  return (
    draft.listed !== config.listed ||
    draft.idle !== config.idle_unload_min ||
    !sameArgs(draft.args, config.args) ||
    !sameEnv(cleanEnv(draft.env), config.env)
  );
}

function Section({
  icon: Icon,
  title,
  description,
  children,
}: {
  icon: typeof Cpu;
  title: string;
  description?: string;
  children: ReactNode;
}) {
  return (
    <Card className="gap-1">
      <CardHeader>
        <CardTitle>
          <Icon className="size-4 text-muted-foreground" />
          {title}
        </CardTitle>
        {description ? <CardDescription>{description}</CardDescription> : null}
      </CardHeader>
      <CardContent className="divide-y">{children}</CardContent>
    </Card>
  );
}

function Controls({
  controls,
  model,
  form,
  available,
}: {
  controls: Control[];
  model: ModelArtifact;
  form: FormContext;
  available: (flag: string) => boolean;
}) {
  return (
    <>
      {controls
        .filter((control) => (control.when ? control.when(model) : true) && available(control.flag))
        .map((control) => (
          <ControlField key={control.flag} control={control} form={form} />
        ))}
    </>
  );
}

function FieldShell({ label, flag, help, children }: { label: string; flag: string; help: string; children: ReactNode }) {
  return (
    <div className="flex flex-col gap-2 py-2.5">
      <div>
        <div className="text-sm font-medium">{label}</div>
        <div className="mt-0.5 text-xs text-muted-foreground">
          <code className="font-mono text-[11px]">{flag}</code> · {help}
        </div>
      </div>
      {children}
    </div>
  );
}

function ContextField({ model, form }: { model: ModelArtifact; form: FormContext }) {
  const raw = getFlagString(form.args, "--max-context", form.takesValue);
  const value = raw.length > 0 ? Number(raw) : 8192;
  const yarn = getFlagString(form.args, "--rope-yarn-factor", form.takesValue);
  const factor = yarn.length > 0 ? Number(yarn) : 1;
  const ceiling = model.native_context * (Number.isFinite(factor) ? factor : 1);
  const set = (next: number | null) =>
    form.setArgs(setFlag(form.args, "--max-context", next === null ? null : String(next), form.takesValue));
  return (
    <FieldShell
      label="Context length"
      flag="--max-context"
      help={`Tokens per request (prompt and output). Native ${formatInt(model.native_context)}; without the flag the server uses 8,192.`}
    >
      <div className="flex flex-wrap items-center gap-1.5">
        {CONTEXT_PRESETS.map((preset) => (
          <button
            key={preset}
            type="button"
            onClick={() => set(preset)}
            className={cn(
              "rounded-md border px-2 py-1 text-xs font-medium tabular-nums transition-colors",
              value === preset ? "border-primary/50 bg-primary/10 text-foreground" : "text-muted-foreground hover:text-foreground",
              preset > model.native_context && "border-dashed",
            )}
          >
            {formatContext(preset)}
          </button>
        ))}
        <CommitInput
          type="number"
          className="h-8 w-32"
          value={raw}
          placeholder="8192"
          min={1024}
          ariaLabel="Context length"
          onCommit={(text) => set(text.length > 0 ? Number(text) : null)}
        />
      </div>
      {value > ceiling ? (
        <div className="flex items-center gap-1.5 text-xs text-warning">
          <TriangleAlert className="size-3.5" />
          Longer than the model's native {formatContext(model.native_context)}: add --rope-yarn-factor{" "}
          {Math.min(4, Math.ceil((value / model.native_context) * 4) / 4)} (YaRN) or the server refuses it.
        </div>
      ) : null}
    </FieldShell>
  );
}

function ConcurrencyField({ form }: { form: FormContext }) {
  const raw = getFlagString(form.args, "--max-concurrency", form.takesValue);
  const value = raw.length > 0 ? Math.max(1, Math.min(8, Number(raw) || 1)) : 1;
  return (
    <FieldShell
      label="Concurrent requests"
      flag="--max-concurrency"
      help="Resident execution lanes (1-8). Two suits a coding agent with subagents; each lane holds its own context."
    >
      <div className="flex items-center gap-4">
        <Slider
          min={1}
          max={8}
          step={1}
          value={[value]}
          onValueChange={(next) =>
            form.setArgs(setFlag(form.args, "--max-concurrency", next[0] === 1 ? null : String(next[0]), form.takesValue))
          }
          aria-label="Concurrent requests"
          className="max-w-sm"
        />
        <span className="w-6 text-right text-sm font-semibold tabular-nums">{value}</span>
      </div>
    </FieldShell>
  );
}

/** `specArg` is the server's --spec placeholder ("off|mtp|...|auto"). A server that offers `off`
 * speculates when --spec is absent, so Off must be written out; an older one decodes plainly without
 * the flag, and refuses `off`. */
function SpecField({ model, form, specArg }: { model: ModelArtifact; form: FormContext; specArg: string | null | undefined }) {
  const current = getFlagString(form.args, "--spec", form.takesValue);
  const offIsExplicit = (specArg ?? "").split("|").includes("off");
  const options: { value: string; label: string }[] = [{ value: "auto", label: "Auto (fastest measured drafter)" }];
  if (model.has_mtp) options.push({ value: "mtp", label: "MTP" });
  if (model.has_dflash) options.push({ value: "dflash", label: "DFlash" });
  if (model.has_dflash2 || model.accepts_draft_model) options.push({ value: "dflash2", label: "DFlash2" });
  // UNSET is the Off item; without the flag the server's own default applies.
  const value = current === "off" ? UNSET : current.length > 0 ? current : offIsExplicit ? "auto" : UNSET;
  const choose = (next: string) =>
    form.setArgs(setFlag(form.args, "--spec", next === UNSET ? (offIsExplicit ? "off" : null) : next, form.takesValue));
  return (
    <div className="flex flex-col gap-3 py-2.5 sm:flex-row sm:items-center sm:justify-between">
      <div>
        <div className="text-sm font-medium">Speculative decoding</div>
        <div className="mt-0.5 text-xs text-muted-foreground">
          <code className="font-mono text-[11px]">--spec</code> · Auto picks the drafter measured fastest for this artifact.
        </div>
      </div>
      <div className="sm:w-60">
        <Select value={value} onValueChange={choose}>
          <SelectTrigger aria-label="Speculative decoding">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value={UNSET}>Off (plain decode)</SelectItem>
            {options.map((option) => (
              <SelectItem key={option.value} value={option.value}>
                {option.label}
              </SelectItem>
            ))}
            {value !== UNSET && !options.some((option) => option.value === value) ? (
              <SelectItem value={value}>{value}</SelectItem>
            ) : null}
          </SelectContent>
        </Select>
      </div>
    </div>
  );
}

function DrafterField({ model, form, library }: { model: ModelArtifact; form: FormContext; library: Library }) {
  const current = getFlagString(form.args, "--draft-model", form.takesValue);
  const compatible = library.drafters.filter((drafter) => drafter.compatible_models.includes(model.key));
  const known = compatible.some((drafter) => drafter.path === current);
  const value = current.length > 0 ? current : UNSET;
  return (
    <FieldShell
      label="DFlash2 drafter"
      flag="--draft-model"
      help="Inco AI's DFlash2 drafter is 8-11 % faster than the model's own MTP drafter; with --spec auto, naming it selects it."
    >
      <Select
        value={value}
        onValueChange={(next) => form.setArgs(setFlag(form.args, "--draft-model", next === UNSET ? null : next, form.takesValue))}
      >
        <SelectTrigger aria-label="DFlash2 drafter">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value={UNSET}>None (the model's MTP drafter)</SelectItem>
          {compatible.map((drafter) => (
            <SelectItem key={drafter.path} value={drafter.path}>
              {drafter.name} · {formatBytes(drafter.bytes)} · {drafter.dir}
            </SelectItem>
          ))}
          {!known && current.length > 0 ? <SelectItem value={current}>Other file: {current}</SelectItem> : null}
        </SelectContent>
      </Select>
      {compatible.length === 0 ? (
        <div className="text-xs text-muted-foreground">
          No compatible drafter in your folders.{" "}
          <Link to="/discover" className="font-medium text-foreground underline-offset-4 hover:underline">
            Download Qwen3.8-Flash-Next-DFlash2-Infernix
          </Link>{" "}
          (776 MB).
        </div>
      ) : null}
    </FieldShell>
  );
}

function NgramVolumeField({ model, form, library }: { model: ModelArtifact; form: FormContext; library: Library }) {
  const current = getFlagString(form.args, "--ngram-volume", form.takesValue);
  const volumes = library.ngram_volumes
    .filter((volume) => volume.compatible_models.includes(model.key))
    .sort((a, b) => Number(b.own_for.includes(model.key)) - Number(a.own_for.includes(model.key)));
  const known = volumes.some((volume) => volume.path === current);
  const value = current.length > 0 ? current : UNSET;
  const missing = current.length === 0 && model.own_ngram_volume === null;
  return (
    <FieldShell
      label="N-gram volume"
      flag="--ngram-volume"
      help="The model's PLE n-gram table, read from NVMe (or RAM). Any of the published tables runs any Flash-Next conversion."
    >
      <Select
        value={value}
        onValueChange={(next) => form.setArgs(setFlag(form.args, "--ngram-volume", next === UNSET ? null : next, form.takesValue))}
      >
        <SelectTrigger aria-label="N-gram volume">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value={UNSET}>
            {model.own_ngram_volume
              ? `The artifact's own (${baseName(model.own_ngram_volume)})`
              : "The artifact's own <entry>.ngram (not found)"}
          </SelectItem>
          {volumes.map((volume) => (
            <SelectItem key={volume.path} value={volume.path}>
              {volume.file_name} · {NGRAM_FORMAT_LABELS[volume.format]} · {formatBytes(volume.bytes)}
              {volume.own_for.includes(model.key) ? " · own table" : ""}
            </SelectItem>
          ))}
          {!known && current.length > 0 ? <SelectItem value={current}>Other file: {current}</SelectItem> : null}
        </SelectContent>
      </Select>
      {missing ? (
        <div className="flex items-center gap-1.5 text-xs text-warning">
          <TriangleAlert className="size-3.5 shrink-0" />
          <span>
            This model cannot start without an n-gram volume: pick one above or{" "}
            <Link to="/discover" className="font-medium text-foreground underline-offset-4 hover:underline">
              download one
            </Link>{" "}
            (the FP8 table is recommended).
          </span>
        </div>
      ) : null}
    </FieldShell>
  );
}

function HelpDrawer({
  open,
  onOpenChange,
  options,
  form,
  onInsert,
  error,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  options: ServeOption[];
  form: FormContext;
  onInsert: (option: ServeOption) => void;
  error: string | null;
}) {
  const [query, setQuery] = useState("");
  const groups = useMemo(() => {
    const needle = query.trim().toLowerCase();
    const map = new Map<string, ServeOption[]>();
    for (const option of options) {
      if (
        needle.length > 0 &&
        !option.flag.toLowerCase().includes(needle) &&
        !option.description.toLowerCase().includes(needle)
      ) {
        continue;
      }
      const list = map.get(option.section) ?? [];
      list.push(option);
      map.set(option.section, list);
    }
    return [...map.entries()];
  }, [options, query]);

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent side="right" className="w-[min(40rem,calc(100vw-1rem))] gap-0 p-0">
        <SheetHeader className="border-b">
          <SheetTitle>infernix-serve options</SheetTitle>
          <SheetDescription>From the server's own --help. Add inserts an option into the launch parameters.</SheetDescription>
          <div className="relative mt-2">
            <Search className="pointer-events-none absolute top-1/2 left-2.5 size-4 -translate-y-1/2 text-muted-foreground" />
            <Input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search options" className="pl-8" />
          </div>
        </SheetHeader>
        <div className="min-h-0 flex-1 overflow-y-auto p-4">
          {error ? <div className="text-sm text-critical">{error}</div> : null}
          {groups.length === 0 && !error ? <div className="text-sm text-muted-foreground">No option matches.</div> : null}
          {groups.map(([section, list]) => (
            <div key={section} className="mb-5">
              <div className="mb-2 text-[11px] font-semibold tracking-wide text-muted-foreground uppercase">{section}</div>
              <ul className="flex flex-col divide-y rounded-lg border">
                {list.map((option) => {
                  const used = hasFlag(form.args, option.flag, form.takesValue);
                  return (
                    <li key={option.flag} className="flex gap-3 px-3 py-2.5">
                      <div className="min-w-0 flex-1">
                        <div className="flex flex-wrap items-center gap-1.5">
                          <code className="font-mono text-xs font-semibold">{option.flag}</code>
                          {option.arg ? <code className="font-mono text-xs text-muted-foreground">{option.arg}</code> : null}
                          {used ? <Badge variant="accent">in use</Badge> : null}
                          {FORM_FLAGS.has(option.flag) ? <Badge variant="muted">in the form</Badge> : null}
                        </div>
                        <p className="mt-1 text-xs leading-relaxed whitespace-pre-line text-muted-foreground">{option.description}</p>
                      </div>
                      <div className="shrink-0">
                        {used ? (
                          <Button
                            size="xs"
                            variant="ghost"
                            onClick={() => form.setArgs(setFlag(form.args, option.flag, null, form.takesValue))}
                          >
                            Remove
                          </Button>
                        ) : (
                          <Button size="xs" variant="outline" onClick={() => onInsert(option)}>
                            Add
                          </Button>
                        )}
                      </div>
                    </li>
                  );
                })}
              </ul>
            </div>
          ))}
        </div>
      </SheetContent>
    </Sheet>
  );
}

/** The artifact's configurations as tabs; switching opens that configuration's page. */
function ConfigTabs({
  model,
  current,
  onSelect,
  onNew,
}: {
  model: ModelArtifact;
  current: string;
  onSelect: (configId: string) => void;
  onNew: () => void;
}) {
  const instance = useLive((live) => live.instance);
  const live = liveInstance(instance);
  return (
    <div className="flex min-w-0 flex-wrap items-center gap-2 border-t px-5 pt-4">
      <span className="text-[11px] font-medium tracking-wide text-muted-foreground uppercase">Configurations</span>
      <Tabs value={current} onValueChange={onSelect} className="min-w-0">
        <TabsList className="h-8">
          {model.configs.map((item) => (
            <TabsTrigger key={item.id} value={item.id} title={`Model id ${item.alias}`} className="max-w-56 text-xs">
              {live && live.model_key === item.id ? <StateDot state={live.state} /> : null}
              <span className="truncate">{item.name}</span>
            </TabsTrigger>
          ))}
        </TabsList>
      </Tabs>
      <Button size="xs" variant="ghost" onClick={onNew}>
        <Plus />
        New configuration
      </Button>
    </div>
  );
}

export function ModelConfigurePage() {
  // The route parameter is a configuration id.
  const { key: id = "" } = useParams();
  const navigate = useNavigate();
  const { data: library, isLoading: libraryLoading } = useLibrary();
  const { data: settings } = useSettings();
  const serveOptions = useServeOptions();
  const optionMap = useOptionMap();
  const takesValue = useTakesValue();
  const instance = useLive((live) => live.instance);
  const { load, dialog, pending: loading } = useLoadModel();

  const model = findModel(library, id);
  const configQuery = useQuery({
    queryKey: ["config", id],
    queryFn: () => api.modelConfig(id),
    enabled: id.length > 0,
  });
  // The query's and the library's copies of this configuration; the newest wins.
  const config = freshestConfig(configQuery.data, findConfig(library, id)?.config);

  const [draft, setDraft] = useState<Draft | null>(null);
  const dirtyRef = useRef(false);
  const [textEditing, setTextEditing] = useState(false);
  const [text, setText] = useState("");
  const [helpOpen, setHelpOpen] = useState(false);
  const [confirmReset, setConfirmReset] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [creating, setCreating] = useState<"duplicate" | "new" | null>(null);
  const [importing, setImporting] = useState(false);
  const [leaveTo, setLeaveTo] = useState<string | null>(null);
  const textRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    setDraft(null);
    setTextEditing(false);
  }, [id]);

  useEffect(() => {
    if (!config) return;
    setDraft((old) => (old === null || !dirtyRef.current ? fromConfig(config) : old));
  }, [config]);

  const dirty = draft !== null && config !== null && draftDiffers(draft, config);
  dirtyRef.current = dirty;

  const args = draft?.args ?? [];
  const setArgs = (next: string[]) => setDraft((old) => (old ? { ...old, args: next } : old));
  const form: FormContext = { args, setArgs, takesValue };
  const available = (flag: string) => optionMap === null || optionMap.has(flag);

  const debouncedArgs = useDebounced(args, 300);
  const preview = useQuery({
    queryKey: ["preview", id, debouncedArgs],
    queryFn: () => api.preview(id, debouncedArgs),
    enabled: draft !== null && config !== null,
    placeholderData: keepPreviousData,
    staleTime: 10_000,
  });

  const save = useSaveConfig();
  const reset = useResetConfig();
  const remove = useDeleteConfig();
  const downloadScript = useDownloadScript();
  const setDefault = useMutation({
    mutationFn: (value: string | null) => api.saveSettings({ default_model: value }),
  });
  const restart = useMutation({ mutationFn: () => api.restart() });

  /** Takes a configuration the launcher answered as the new starting point of the draft. */
  const adopt = (saved: ModelConfig) => {
    dirtyRef.current = false;
    setDraft(fromConfig(saved));
    setTextEditing(false);
  };

  /** Opens another page, asking first when there are unsaved changes. */
  const goTo = (path: string) => {
    if (dirtyRef.current) setLeaveTo(path);
    else navigate(path);
  };

  if (!library && libraryLoading) {
    return (
      <div className="flex flex-col gap-4">
        <Skeleton className="h-28 rounded-xl" />
        <Skeleton className="h-96 rounded-xl" />
      </div>
    );
  }
  if (!model) {
    return (
      <EmptyState
        icon={TriangleAlert}
        title="Model not found"
        description="It is not in the library any more (moved, deleted, or its folder was removed)."
        actions={<Button onClick={() => navigate("/models")}>Back to models</Button>}
      />
    );
  }
  if (!config) {
    return (
      <EmptyState
        icon={TriangleAlert}
        title="Configuration not found"
        description={`${model.name} has no configuration with this id any more (it was deleted).`}
        actions={
          <>
            <Button onClick={() => navigate(configPath(model.key))}>Open its default configuration</Button>
            <Button variant="outline" onClick={() => navigate("/models")}>
              Back to models
            </Button>
          </>
        }
      />
    );
  }

  const live = liveInstance(instance);
  const loadedHere = live !== null && live.model_key === config.id && (live.state === "ready" || live.state === "starting");
  const sibling =
    live !== null && live.model_key !== config.id
      ? (model.configs.find((item) => item.id === live.model_key) ?? null)
      : null;
  const isDefault = settings?.default_model === config.id;
  const commandLine = preview.data?.command_line ?? argsToCommandLine(args);
  const textValue = textEditing ? text : argsToText(args, takesValue);

  const onSave = (then?: "load" | "restart") => {
    if (!draft) return;
    const problem = envProblem(draft.env);
    if (problem) {
      toast.error(`Environment variables: ${problem}`);
      return;
    }
    save.mutate(
      {
        id: config.id,
        patch: { listed: draft.listed, args: draft.args, env: cleanEnv(draft.env), idle_unload_min: draft.idle },
      },
      {
        onSuccess: (saved) => {
          adopt(saved);
          if (then === "load") load(saved.id, saved.alias);
          else if (then === "restart") restart.mutate();
          else toast.success("Saved");
        },
      },
    );
  };

  const onDownload = (shell: "cmd" | "sh") => {
    if (dirty) toast.info("The script holds the saved parameters; save first to include your changes.");
    downloadScript.mutate({ id: config.id, shell });
  };

  const insertOption = (option: ServeOption) => {
    if (option.arg === null) {
      setArgs(setFlag(args, option.flag, true, takesValue));
      toast.success(`Added ${option.flag}`);
      return;
    }
    const base = argsToText(args, takesValue);
    const next = `${base}${base.length > 0 ? "\n" : ""}${option.flag} `;
    setText(next);
    setTextEditing(true);
    setHelpOpen(false);
    window.setTimeout(() => {
      const area = textRef.current;
      if (area) {
        area.focus();
        area.setSelectionRange(next.length, next.length);
      }
    }, 50);
  };

  return (
    <div className="flex flex-col gap-5 pb-24">
      <div className="flex flex-col gap-4">
        <Button variant="ghost" size="sm" className="w-fit" onClick={() => goTo("/models")}>
          <ArrowLeft />
          Models
        </Button>
        <Card className="hero-glow gap-4 py-5">
          <div className="flex flex-col gap-4 px-5 lg:flex-row lg:items-start lg:justify-between">
            <div className="min-w-0 flex-1">
              <div className="flex flex-wrap items-center gap-1.5">
                <Badge variant="outline">{model.family_label}</Badge>
                {model.quantization ? <Badge variant="secondary">{quantizationLabel(model.quantization)}</Badge> : null}
                {!config.is_default ? (
                  <Badge variant="accent">named configuration</Badge>
                ) : config.customized ? (
                  <Badge variant="muted">custom settings</Badge>
                ) : (
                  <Badge variant="muted">defaults</Badge>
                )}
                {config.vision ? <Badge variant="muted">vision on</Badge> : null}
                {loadedHere && live ? <StateBadge state={live.state} /> : null}
              </div>
              <div className="mt-3 text-[11px] font-medium tracking-wide text-muted-foreground uppercase">Model id</div>
              <ModelIdEditor config={config} size="lg" className="mt-0.5" />
              <div className="mt-1 flex min-w-0 flex-wrap items-center gap-x-1.5 text-sm text-muted-foreground">
                <span className="truncate">{model.name}</span>
                <span aria-hidden="true">·</span>
                <span>configuration</span>
                <ConfigNameEditor model={model} config={config} textClassName="font-medium text-foreground" />
              </div>
            </div>
            <div className="flex flex-col gap-2 lg:items-end">
              <div className="flex flex-wrap items-center gap-2 lg:justify-end">
                <Button variant="outline" size="sm" onClick={() => setCreating("duplicate")}>
                  <CopyPlus />
                  Duplicate
                </Button>
                <Button variant="outline" size="sm" onClick={() => setConfirmReset(true)} disabled={reset.isPending}>
                  <Undo2 />
                  Restore defaults
                </Button>
                <Button variant="outline" size="sm" onClick={() => setHelpOpen(true)}>
                  <BookOpen />
                  All options
                </Button>
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button variant="outline" size="icon-sm" aria-label="More actions">
                      <EllipsisVertical />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end" className="w-60">
                    <DropdownMenuItem onSelect={() => setImporting(true)}>
                      <FileInput />
                      Import launch script…
                    </DropdownMenuItem>
                    <DropdownMenuItem onSelect={() => setCreating("new")}>
                      <Plus />
                      New configuration…
                    </DropdownMenuItem>
                    {!config.is_default ? (
                      <>
                        <DropdownMenuSeparator />
                        <DropdownMenuItem variant="destructive" onSelect={() => setConfirmDelete(true)}>
                          <Trash2 />
                          Delete this configuration…
                        </DropdownMenuItem>
                      </>
                    ) : null}
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>
              <div className="flex flex-wrap items-center gap-2 lg:justify-end">
                <CopyButton text={commandLine} label="Copy command" variant="outline" size="sm" toastMessage="Command copied" />
                <Hint label="This configuration's saved parameters as a Windows launch script">
                  <Button variant="outline" size="sm" disabled={downloadScript.isPending} onClick={() => onDownload("cmd")}>
                    <FileDown />
                    Download .bat
                  </Button>
                </Hint>
                <Hint label="This configuration's saved parameters as a shell launch script">
                  <Button variant="outline" size="sm" disabled={downloadScript.isPending} onClick={() => onDownload("sh")}>
                    <FileDown />
                    Download .sh
                  </Button>
                </Hint>
              </div>
            </div>
          </div>
          <ConfigTabs
            model={model}
            current={config.id}
            onSelect={(next) => {
              if (next !== config.id) goTo(configPath(next));
            }}
            onNew={() => setCreating("new")}
          />
          <div className="grid grid-cols-2 gap-x-6 gap-y-3 border-t px-5 pt-4 sm:grid-cols-3 xl:grid-cols-6">
            <Fact label="Size">{formatBytes(model.bytes)}</Fact>
            <Fact label="Native context">{formatContext(model.native_context)}</Fact>
            <Fact label="Layers · hidden">
              {model.num_layers} · {formatInt(model.hidden_size)}
            </Fact>
            <Fact label="Experts">{model.num_experts ? formatInt(model.num_experts) : "dense"}</Fact>
            <Fact label="Components">{model.components.join(", ")}</Fact>
            <Fact label="Recipe">{model.recipe ?? "—"}</Fact>
          </div>
          <div className="flex min-w-0 items-center gap-1.5 px-5 text-xs text-muted-foreground">
            <FileCode className="size-3.5 shrink-0" />
            <span className="truncate font-mono">{model.path}</span>
            <CopyButton text={model.path} size="icon-xs" toastMessage="Path copied" />
          </div>
        </Card>
      </div>

      {draft === null ? (
        <Skeleton className="h-96 rounded-xl" />
      ) : (
        <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_minmax(0,26rem)]">
          <div className="flex min-w-0 flex-col gap-5">
            <Section
              icon={Plug}
              title="Serving"
              description="How coding tools see this configuration (its model id is the headline above) and when it unloads"
            >
              <div className="flex items-center justify-between gap-3 py-2.5">
                <div>
                  <div className="text-sm font-medium">Listed for coding tools</div>
                  <div className="mt-0.5 text-xs text-muted-foreground">Shown in /v1/models; a hidden model still answers its id.</div>
                </div>
                <Switch checked={draft.listed} onCheckedChange={(checked) => setDraft({ ...draft, listed: checked })} aria-label="Listed" />
              </div>
              <div className="flex items-center justify-between gap-3 py-2.5">
                <div>
                  <div className="text-sm font-medium">Default model</div>
                  <div className="mt-0.5 text-xs text-muted-foreground">
                    Loaded for requests that name no known model while nothing is loaded, and at startup with autoload.
                    Applies at once.
                  </div>
                </div>
                <Switch
                  checked={isDefault}
                  disabled={setDefault.isPending}
                  onCheckedChange={(checked) => setDefault.mutate(checked ? config.id : null)}
                  aria-label="Default model"
                />
              </div>
              <IdleUnloadField
                key={config.id}
                value={draft.idle}
                globalMinutes={settings?.idle_unload_min ?? null}
                onChange={(idle) => setDraft({ ...draft, idle })}
              />
            </Section>

            <Section icon={Cpu} title="Context and memory">
              {available("--max-context") ? <ContextField model={model} form={form} /> : null}
              {available("--max-concurrency") ? <ConcurrencyField form={form} /> : null}
              <Controls controls={MEMORY_CONTROLS} model={model} form={form} available={available} />
            </Section>

            <Section icon={Sparkles} title="Speculative decoding" description="Drafters propose tokens the model verifies; output is unchanged.">
              {available("--spec") ? <SpecField model={model} form={form} specArg={optionMap?.get("--spec")} /> : null}
              {model.accepts_draft_model && available("--draft-model") && library ? (
                <DrafterField model={model} form={form} library={library} />
              ) : null}
              <Controls controls={SPEC_CONTROLS} model={model} form={form} available={available} />
            </Section>

            {model.needs_ngram_volume ? (
              <Section icon={Database} title="N-gram table" description="Qwen3.8-Flash-Next's per-layer n-gram embeddings">
                {available("--ngram-volume") && library ? <NgramVolumeField model={model} form={form} library={library} /> : null}
                <Controls controls={NGRAM_CONTROLS} model={model} form={form} available={available} />
              </Section>
            ) : null}

            <Section icon={Brain} title="Thinking and tools">
              <Controls controls={THINKING_CONTROLS} model={model} form={form} available={available} />
            </Section>

            <Section icon={Wrench} title="Sampling overrides" description="Process-wide defaults; fields a request sends still win.">
              <Controls controls={SAMPLING_CONTROLS} model={model} form={form} available={available} />
            </Section>

            {model.has_vision ? (
              <Section icon={ImageIcon} title="Vision">
                <Controls controls={VISION_CONTROLS} model={model} form={form} available={available} />
              </Section>
            ) : null}

            <Section
              icon={Variable}
              title="Environment variables"
              description="Set for infernix-serve when this configuration starts, besides the launcher's own."
            >
              <EnvEditor env={draft.env} onChange={(env) => setDraft({ ...draft, env })} />
            </Section>

            <Section icon={Eye} title="Advanced" description="Numerical routes, diagnostics and approximate speed-ups">
              <Controls controls={ADVANCED_CONTROLS} model={model} form={form} available={available} />
            </Section>
          </div>

          <div className="min-w-0">
            <div className="flex flex-col gap-4 xl:sticky xl:top-20">
              <Card className="gap-3">
                <CardHeader>
                  <CardTitle>Launch parameters</CardTitle>
                  <CardDescription>
                    One option per line. Edit freely: the form follows when you leave the box. The artifact path and the
                    launcher's own flags are added at launch.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <Textarea
                    ref={textRef}
                    value={textValue}
                    spellCheck={false}
                    rows={Math.min(22, Math.max(8, textValue.split("\n").length + 1))}
                    className="font-mono text-xs leading-5"
                    onFocus={() => {
                      if (!textEditing) {
                        setText(argsToText(args, takesValue));
                        setTextEditing(true);
                      }
                    }}
                    onChange={(event) => setText(event.target.value)}
                    onBlur={() => {
                      setArgs(tokenize(text));
                      setTextEditing(false);
                    }}
                    aria-label="Launch parameters"
                  />
                </CardContent>
              </Card>

              <FitCard
                key={config.id}
                configId={config.id}
                args={args}
                env={draft.env}
                takesValue={takesValue}
                onSetArgs={(next) => {
                  setArgs(next);
                  setTextEditing(false);
                }}
              />

              <Card className="gap-3">
                <CardHeader>
                  <CardTitle>Command</CardTitle>
                  <CardDescription>What the launcher runs{preview.isFetching ? " (updating…)" : ""}</CardDescription>
                </CardHeader>
                <CardContent className="flex flex-col gap-3">
                  <pre className="max-h-56 overflow-auto rounded-lg border bg-muted/40 p-3 font-mono text-[11px] leading-5 break-all whitespace-pre-wrap">
                    {commandLine}
                  </pre>
                  {preview.data && preview.data.managed.length > 0 ? (
                    <div className="text-xs text-muted-foreground">
                      Set by the launcher:{" "}
                      {preview.data.managed.map((flag) => (
                        <code key={flag} className="mr-1.5 font-mono">
                          {flag}
                        </code>
                      ))}
                    </div>
                  ) : null}
                  {draft.env.some((row) => row.name.trim().length > 0) ? (
                    <div className="text-xs text-muted-foreground">
                      With{" "}
                      {cleanEnv(draft.env)
                        .filter((row) => row.name.length > 0)
                        .map((row) => (
                          <code key={row.name} className="mr-1.5 font-mono">
                            {row.name}
                          </code>
                        ))}
                      in its environment.
                    </div>
                  ) : null}
                  {preview.data && preview.data.warnings.length > 0 ? (
                    <Alert variant="warning">
                      <TriangleAlert />
                      <AlertTitle>Check before launching</AlertTitle>
                      <AlertDescription>
                        <ul className="flex flex-col gap-1 text-xs">
                          {preview.data.warnings.map((warning) => (
                            <li key={warning}>{warning}</li>
                          ))}
                        </ul>
                      </AlertDescription>
                    </Alert>
                  ) : null}
                  {preview.isError ? <div className="text-xs text-critical">Preview unavailable.</div> : null}
                </CardContent>
              </Card>
            </div>
          </div>
        </div>
      )}

      {/* Save bar: always reachable, highlighted when there are unsaved changes. */}
      {/* Starts after the sidebar: the icon rail below lg, the full sidebar from lg (AppShell). */}
      <div className="fixed inset-x-0 bottom-0 z-20 border-t bg-background/90 backdrop-blur-md md:left-16 lg:left-60">
        <div className="mx-auto flex max-w-[1600px] flex-wrap items-center gap-2 px-4 py-3 md:px-6">
          <span className={cn("text-sm", dirty ? "font-medium text-foreground" : "text-muted-foreground")}>
            {dirty ? "Unsaved changes" : "All changes saved"}
          </span>
          <Separator orientation="vertical" className="mx-1 hidden h-5 sm:block" />
          <span className="hidden min-w-0 truncate text-xs text-muted-foreground sm:inline">
            {loadedHere
              ? "This configuration is loaded: restart it to apply changes."
              : sibling
                ? `${sibling.name} of this model is loaded: Save and load switches to ${config.name}.`
                : "Saved settings apply at the next load."}
          </span>
          <div className="ml-auto flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              disabled={!dirty}
              onClick={() => {
                dirtyRef.current = false;
                setDraft(fromConfig(config));
                setTextEditing(false);
              }}
            >
              Discard
            </Button>
            <Button variant="outline" size="sm" disabled={!dirty || save.isPending} onClick={() => onSave()}>
              <Save />
              Save
            </Button>
            {loadedHere ? (
              <Button size="sm" disabled={save.isPending || restart.isPending} onClick={() => onSave("restart")}>
                <RotateCw />
                Save and restart
              </Button>
            ) : (
              <Button
                size="sm"
                disabled={save.isPending || loading || model.missing_files.length > 0}
                onClick={() => onSave("load")}
              >
                <Play />
                Save and load
              </Button>
            )}
          </div>
        </div>
      </div>

      <HelpDrawer
        open={helpOpen}
        onOpenChange={setHelpOpen}
        options={serveOptions.data?.options ?? []}
        form={form}
        onInsert={insertOption}
        error={
          serveOptions.data && !serveOptions.data.found
            ? serveOptions.data.error ?? "infernix-serve was not found; set its path under Settings."
            : serveOptions.isError
              ? "The option list is unavailable."
              : null
        }
      />
      <ConfirmDialog
        open={confirmReset}
        onOpenChange={setConfirmReset}
        title="Restore the default configuration?"
        description={
          config.is_default
            ? "The launch parameters, environment variables, idle unload, model id and listing return to the defaults the launcher computes for this artifact (with the drafter and n-gram volume found now). Your changes are lost."
            : `The launch parameters, environment variables, idle unload and listing of ${config.name} return to the defaults the launcher computes for this artifact (with the drafter and n-gram volume found now); it keeps its name and model id. Your changes are lost.`
        }
        confirmLabel="Restore defaults"
        destructive
        onConfirm={() =>
          reset.mutate(config.id, {
            onSuccess: (saved) => {
              adopt(saved);
              toast.success("Restored the defaults");
            },
          })
        }
      />
      <ConfirmDialog
        open={confirmDelete}
        onOpenChange={setConfirmDelete}
        title={`Delete the configuration ${config.name}?`}
        description={
          <span>
            Its model id <span className="font-mono text-foreground">{config.alias}</span> goes: requests naming it no
            longer find it. {model.name} and its other configurations stay.
            {loadedHere ? " It is loaded now." : ""}
          </span>
        }
        confirmLabel="Delete configuration"
        destructive
        onConfirm={() => {
          const name = config.name;
          remove.mutate(config.id, {
            onSuccess: () => {
              toast.success(`Deleted the configuration ${name}`);
              dirtyRef.current = false;
              navigate(configPath(model.key), { replace: true });
            },
          });
        }}
      />
      <ConfirmDialog
        open={leaveTo !== null}
        onOpenChange={(open) => {
          if (!open) setLeaveTo(null);
        }}
        title="Discard the unsaved changes?"
        description={`Your edits to ${config.name} are not saved. Save them first to keep them.`}
        confirmLabel="Discard and leave"
        destructive
        onConfirm={() => {
          const path = leaveTo;
          dirtyRef.current = false;
          if (path) navigate(path);
        }}
      />
      <NewConfigDialog
        open={creating !== null}
        onOpenChange={(open) => {
          if (!open) setCreating(null);
        }}
        model={model}
        from={creating === "duplicate" ? config.id : null}
        unsaved={creating === "duplicate" && dirty ? { args: draft?.args ?? [], env: cleanEnv(draft?.env ?? []) } : null}
        onCreated={(created, carriedUnsaved) => {
          toast.success(`Created ${created.name} (model id ${created.alias})`);
          if (carriedUnsaved || !dirty) {
            dirtyRef.current = false;
            navigate(configPath(created.id));
          } else {
            goTo(configPath(created.id));
          }
        }}
      />
      <ImportScriptDialog
        open={importing}
        onOpenChange={setImporting}
        preferConfigId={config.id}
        onApplied={(saved, created) => {
          if (!created && saved.id === config.id) adopt(saved);
        }}
      />
      {dialog}
    </div>
  );
}
