import { useMemo, useState } from "react";
import { Link, useNavigate } from "react-router";
import { useMutation } from "@tanstack/react-query";
import {
  Boxes,
  CloudDownload,
  Copy,
  CopyPlus,
  Database,
  EllipsisVertical,
  Eye,
  EyeOff,
  FileInput,
  FolderOpen,
  FolderPlus,
  HardDrive,
  LayoutGrid,
  List,
  Play,
  Plus,
  RefreshCw,
  Search,
  SlidersHorizontal,
  Sparkles,
  Star,
  Trash2,
  TriangleAlert,
  Zap,
} from "lucide-react";
import { toast } from "sonner";

import { ConfirmDialog, EmptyState, PageHeader } from "@/components/common";
import { FolderPicker } from "@/components/FolderPicker";
import { ModelIdEditor } from "@/components/models/ConfigEditors";
import {
  DeleteFilesDialog,
  drafterDeleteTarget,
  modelDeleteTarget,
  volumeDeleteTarget,
  type DeleteTarget,
} from "@/components/models/DeleteFilesDialog";
import { ImportScriptDialog } from "@/components/models/ImportScriptDialog";
import { NewConfigDialog } from "@/components/models/NewConfigDialog";
import { StateBadge } from "@/components/StateBadge";
import { useLoadModel } from "@/components/useModelActions";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { Hint } from "@/components/ui/tooltip";
import { api } from "@/lib/api";
import type {
  DrafterArtifact,
  InstanceInfo,
  Library,
  ModelArtifact,
  ModelConfig,
  ModelDir,
  ModelFamily,
  NgramVolume,
} from "@/lib/api-types";
import {
  configPath,
  describeConfig,
  liveInstance,
  useDeleteConfig,
  useSaveConfig,
  type FileKind,
} from "@/lib/configs";
import { useLive } from "@/lib/events";
import { FAMILY_ORDER, NGRAM_FORMAT_LABELS, formatBytes, formatContext, quantizationLabel } from "@/lib/format";
import { useLibrary, useSettings } from "@/lib/hooks";
import { artifactKeyOf, findConfig, findModel } from "@/lib/library";
import { cn, copyText, readStorage, writeStorage } from "@/lib/utils";

type View = "grid" | "list";
const VIEW_KEY = "infernix.models.view";

/** The page's dialogs work on keys and look the objects up each render, so a library update (or a
 * deletion) never leaves them holding stale copies. */
interface Actions {
  load: (config: ModelConfig) => void;
  loading: boolean;
  newConfig: (model: ModelArtifact, from: string | null) => void;
  deleteConfig: (config: ModelConfig) => void;
  deleteFiles: (kind: FileKind, key: string) => void;
}

function ComponentBadges({ model, className }: { model: ModelArtifact; className?: string }) {
  const badges: { label: string; hint: string }[] = [];
  if (model.has_vision) badges.push({ label: "Vision", hint: "Images and video (enable --vision)" });
  if (model.has_mtp) badges.push({ label: "MTP", hint: "Multi-token-prediction drafter" });
  if (model.has_dflash) badges.push({ label: "DFlash", hint: "Bundled DFlash drafter" });
  if (model.has_dflash2) badges.push({ label: "DFlash2", hint: "Bundled DFlash2 drafter" });
  if (model.accepts_draft_model) badges.push({ label: "DFlash2 companion", hint: "Takes a DFlash2 drafter artifact (--draft-model)" });
  if (model.has_proposal) badges.push({ label: "Proposal head", hint: "Optimized proposal head (--lm-head-draft)" });
  if (model.offloaded_experts) badges.push({ label: "Experts in RAM", hint: "Routed experts live in system RAM behind a VRAM expert cache" });
  if (model.needs_ngram_volume) badges.push({ label: "N-gram volume", hint: "Needs a PLE n-gram volume, ideally on NVMe" });
  return (
    <div className={cn("flex flex-wrap gap-1", className)}>
      {badges.map((badge) => (
        <Hint key={badge.label} label={badge.hint}>
          <Badge variant="muted" className="cursor-default">
            {badge.label}
          </Badge>
        </Hint>
      ))}
    </div>
  );
}

/** The instance state of this configuration, when it is the one a server holds. */
function configState(config: ModelConfig, instance: InstanceInfo | null) {
  if (!instance || instance.model_key !== config.id) return null;
  return instance.state === "idle" ? null : instance.state;
}

/** The instance state of any configuration of this artifact. */
function artifactState(model: ModelArtifact, instance: InstanceInfo | null) {
  if (!instance || instance.state === "idle" || !instance.model_key) return null;
  const key = instance.artifact_key ?? artifactKeyOf(instance.model_key);
  return key === model.key ? instance.state : null;
}

function copy(text: string, what: string) {
  void copyText(text).then((ok) => (ok ? toast.success(`${what} copied`) : toast.error("Copying failed")));
}

/** A configuration's own actions; `withArtifact` adds the artifact's (the list view has no card). */
function ConfigMenu({
  model,
  config,
  isDefault,
  actions,
  withArtifact = false,
}: {
  model: ModelArtifact;
  config: ModelConfig;
  isDefault: boolean;
  actions: Actions;
  withArtifact?: boolean;
}) {
  const saveConfig = useSaveConfig();
  const setDefault = useMutation({
    mutationFn: (id: string | null) => api.saveSettings({ default_model: id }),
    onSuccess: (_, id) => toast.success(id ? `${config.alias} is the default model` : "No default model"),
  });
  const setListed = (listed: boolean) =>
    saveConfig.mutate(
      { id: config.id, patch: { listed } },
      {
        onSuccess: (saved) =>
          toast.success(listed ? `${saved.alias} is listed in /v1/models` : `${saved.alias} is hidden from /v1/models`),
      },
    );
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" size="icon-sm" aria-label={`More actions for ${config.alias}`}>
          <EllipsisVertical />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-64">
        <DropdownMenuLabel className="truncate">{config.name}</DropdownMenuLabel>
        <DropdownMenuItem onSelect={() => copy(config.alias, "Model id")}>
          <Copy />
          Copy the model id
        </DropdownMenuItem>
        <DropdownMenuItem onSelect={() => setDefault.mutate(isDefault ? null : config.id)}>
          <Star />
          {isDefault ? "Clear the default model" : "Make the default model"}
        </DropdownMenuItem>
        <DropdownMenuItem onSelect={() => setListed(!config.listed)}>
          {config.listed ? <EyeOff /> : <Eye />}
          {config.listed ? "Hide from coding tools" : "List for coding tools"}
        </DropdownMenuItem>
        <DropdownMenuItem onSelect={() => actions.newConfig(model, config.id)}>
          <CopyPlus />
          Duplicate…
        </DropdownMenuItem>
        {!config.is_default ? (
          <DropdownMenuItem variant="destructive" onSelect={() => actions.deleteConfig(config)}>
            <Trash2 />
            Delete the configuration…
          </DropdownMenuItem>
        ) : null}
        {withArtifact ? (
          <>
            <DropdownMenuSeparator />
            <DropdownMenuLabel className="truncate">{model.name}</DropdownMenuLabel>
            <ArtifactItems model={model} actions={actions} />
          </>
        ) : null}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

function ArtifactItems({ model, actions }: { model: ModelArtifact; actions: Actions }) {
  return (
    <>
      <DropdownMenuItem onSelect={() => actions.newConfig(model, null)}>
        <Plus />
        New configuration…
      </DropdownMenuItem>
      <DropdownMenuItem onSelect={() => copy(model.path, "Path")}>
        <FolderOpen />
        Copy the artifact path
      </DropdownMenuItem>
      <DropdownMenuSeparator />
      <DropdownMenuItem variant="destructive" onSelect={() => actions.deleteFiles("model", model.key)}>
        <Trash2 />
        Delete files…
      </DropdownMenuItem>
    </>
  );
}

function ArtifactMenu({ model, actions }: { model: ModelArtifact; actions: Actions }) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" size="icon-sm" aria-label={`Actions for ${model.name}`}>
          <EllipsisVertical />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-60">
        <ArtifactItems model={model} actions={actions} />
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

function LoadButton({
  model,
  config,
  instance,
  actions,
}: {
  model: ModelArtifact;
  config: ModelConfig;
  instance: InstanceInfo | null;
  actions: Actions;
}) {
  const state = configState(config, instance);
  if (state === "ready") {
    return (
      <Button size="sm" variant="secondary" disabled>
        <Zap />
        Loaded
      </Button>
    );
  }
  if (state === "starting") {
    return (
      <Button size="sm" variant="secondary" disabled>
        Loading…
      </Button>
    );
  }
  const other = liveInstance(instance) !== null && instance?.model_key !== config.id;
  const incomplete = model.missing_files.length > 0;
  return (
    <Button size="sm" disabled={actions.loading || incomplete} onClick={() => actions.load(config)}>
      <Play />
      {other ? "Switch" : "Load"}
    </Button>
  );
}

function ConfigRow({
  model,
  config,
  instance,
  isDefault,
  actions,
}: {
  model: ModelArtifact;
  config: ModelConfig;
  instance: InstanceInfo | null;
  isDefault: boolean;
  actions: Actions;
}) {
  const navigate = useNavigate();
  const state = configState(config, instance);
  return (
    // The model id has the whole first line; the actions share the second with the details and
    // move below them when the card is too narrow for both.
    <li className={cn("flex flex-col gap-1 px-3 py-2", state && "bg-primary/5")}>
      <div className="flex min-w-0 items-center gap-1">
        {isDefault ? (
          <Hint label="The default model">
            <Star className="size-3.5 shrink-0 fill-current text-primary" aria-label="default model" />
          </Hint>
        ) : null}
        <ModelIdEditor config={config} className="min-w-0 flex-1" />
      </div>
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1.5">
        <div className="flex min-w-0 grow flex-wrap items-center gap-x-1.5 gap-y-1 text-xs text-muted-foreground">
          <span className="truncate font-medium text-foreground/80">{config.name}</span>
          <span aria-hidden="true">·</span>
          <span className="tabular-nums">{formatContext(config.context_length)} context</span>
          {config.is_default && config.customized ? <Badge variant="muted">custom</Badge> : null}
          {!config.listed ? <Badge variant="muted">hidden from API</Badge> : null}
          {config.vision ? <Badge variant="muted">vision on</Badge> : null}
          {state ? <StateBadge state={state} /> : null}
        </div>
        <div className="ml-auto flex shrink-0 items-center gap-1">
          <LoadButton model={model} config={config} instance={instance} actions={actions} />
          <Hint label="Configure">
            <Button size="icon-sm" variant="outline" aria-label={`Configure ${config.alias}`} onClick={() => navigate(configPath(config.id))}>
              <SlidersHorizontal />
            </Button>
          </Hint>
          <ConfigMenu model={model} config={config} isDefault={isDefault} actions={actions} />
        </div>
      </div>
    </li>
  );
}

function ModelCard({
  model,
  instance,
  defaultId,
  actions,
}: {
  model: ModelArtifact;
  instance: InstanceInfo | null;
  defaultId: string | null;
  actions: Actions;
}) {
  const state = artifactState(model, instance);
  return (
    <Card className={cn("gap-3 transition-colors", state !== null && "border-primary/50 ring-1 ring-primary/20")}>
      <CardHeader className="gap-2">
        <div className="flex flex-wrap items-center gap-1.5">
          <Badge variant="outline">{model.family_label}</Badge>
          {model.quantization ? <Badge variant="secondary">{quantizationLabel(model.quantization)}</Badge> : null}
          <div className="ml-auto flex items-center gap-1">
            {state ? <StateBadge state={state} /> : null}
            <ArtifactMenu model={model} actions={actions} />
          </div>
        </div>
        <div className="min-w-0">
          <div className="truncate text-base font-semibold" title={model.name}>
            {model.name}
          </div>
          <div className="truncate text-xs text-muted-foreground" title={model.file_name}>
            {model.file_name}
          </div>
        </div>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <div className="grid grid-cols-3 gap-2 rounded-lg bg-muted/50 px-3 py-2 text-xs">
          <div>
            <div className="text-muted-foreground">Size</div>
            <div className="font-medium tabular-nums">{formatBytes(model.bytes)}</div>
          </div>
          <div>
            <div className="text-muted-foreground">Native context</div>
            <div className="font-medium tabular-nums">{formatContext(model.native_context)}</div>
          </div>
          <div>
            <div className="text-muted-foreground">{model.num_experts ? "Experts" : "Layers"}</div>
            <div className="font-medium tabular-nums">{model.num_experts ?? model.num_layers}</div>
          </div>
        </div>
        <ComponentBadges model={model} />
        {model.missing_files.length > 0 ? (
          <div className="flex items-start gap-2 text-xs text-critical">
            <TriangleAlert className="mt-0.5 size-3.5 shrink-0" />
            <span>
              Missing {model.missing_files.length} part{model.missing_files.length === 1 ? "" : "s"}:{" "}
              {model.missing_files.slice(0, 2).join(", ")}
              {model.missing_files.length > 2 ? "…" : ""}
            </span>
          </div>
        ) : null}
        <div className="flex flex-col gap-1.5">
          <div className="flex items-center justify-between gap-2">
            <span className="text-[11px] font-medium tracking-wide text-muted-foreground uppercase">
              Configurations
            </span>
            <Button size="xs" variant="ghost" onClick={() => actions.newConfig(model, null)}>
              <Plus />
              New configuration
            </Button>
          </div>
          <ul className="divide-y rounded-lg border">
            {model.configs.map((config) => (
              <ConfigRow
                key={config.id}
                model={model}
                config={config}
                instance={instance}
                isDefault={defaultId === config.id}
                actions={actions}
              />
            ))}
          </ul>
        </div>
        <Hint label={model.path}>
          <div className="flex min-w-0 cursor-default items-center gap-1.5 text-xs text-muted-foreground">
            <FolderOpen className="size-3.5 shrink-0" />
            <span className="truncate">{model.dir}</span>
            {model.locations.length > 1 ? <span className="shrink-0">+{model.locations.length - 1} copies</span> : null}
          </div>
        </Hint>
      </CardContent>
    </Card>
  );
}

/** One row per configuration. */
function ConfigTable({
  models,
  instance,
  defaultId,
  actions,
}: {
  models: ModelArtifact[];
  instance: InstanceInfo | null;
  defaultId: string | null;
  actions: Actions;
}) {
  const navigate = useNavigate();
  return (
    <Card className="py-0">
      <Table>
        <TableHeader>
          <TableRow className="hover:bg-transparent">
            <TableHead className="pl-4">Model id</TableHead>
            <TableHead>Family</TableHead>
            <TableHead>Weights</TableHead>
            <TableHead className="text-right">Size</TableHead>
            <TableHead className="text-right">Context</TableHead>
            <TableHead>State</TableHead>
            <TableHead className="pr-4 text-right">Actions</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {models.flatMap((model) =>
            model.configs.map((config) => {
              const state = configState(config, instance);
              return (
                <TableRow key={config.id}>
                  <TableCell className="max-w-[24rem] pl-4">
                    <div className="flex min-w-0 items-center gap-1">
                      {defaultId === config.id ? <Star className="size-3.5 shrink-0 fill-current text-primary" aria-label="default model" /> : null}
                      <ModelIdEditor config={config} className="min-w-0 flex-1" />
                    </div>
                    <div className="truncate text-xs text-muted-foreground" title={model.path}>
                      {model.name} · {config.name}
                    </div>
                  </TableCell>
                  <TableCell>{model.family_label}</TableCell>
                  <TableCell>{quantizationLabel(model.quantization)}</TableCell>
                  <TableCell className="text-right tabular-nums">{formatBytes(model.bytes)}</TableCell>
                  <TableCell className="text-right tabular-nums">{formatContext(config.context_length)}</TableCell>
                  <TableCell>{state ? <StateBadge state={state} /> : <span className="text-muted-foreground">—</span>}</TableCell>
                  <TableCell className="pr-4">
                    <div className="flex items-center justify-end gap-1.5">
                      <LoadButton model={model} config={config} instance={instance} actions={actions} />
                      <Button
                        size="icon-sm"
                        variant="ghost"
                        aria-label={`Configure ${config.alias}`}
                        onClick={() => navigate(configPath(config.id))}
                      >
                        <SlidersHorizontal />
                      </Button>
                      <ConfigMenu model={model} config={config} isDefault={defaultId === config.id} actions={actions} withArtifact />
                    </div>
                  </TableCell>
                </TableRow>
              );
            }),
          )}
        </TableBody>
      </Table>
    </Card>
  );
}

function modelNames(library: Library, keys: readonly string[]): string {
  const names = keys
    .map((key) => library.models.find((model) => model.key === key)?.name)
    .filter((name): name is string => typeof name === "string");
  return names.length > 0 ? names.join(", ") : "no model in the library";
}

/** Library bytes, other used space and free space on a folder's drive. */
function FolderCard({ dir }: { dir: ModelDir }) {
  const known = dir.total_bytes !== null && dir.free_bytes !== null && dir.total_bytes > 0;
  const total = dir.total_bytes ?? 0;
  const free = dir.free_bytes ?? 0;
  const freeRatio = known ? free / total : null;
  const tone = freeRatio === null ? null : freeRatio < 0.05 ? "critical" : freeRatio < 0.15 ? "warning" : null;
  const pct = (bytes: number) => (known ? `${Math.max(0, Math.min(100, (bytes / total) * 100))}%` : "0%");
  const other = known ? Math.max(0, total - free - dir.library_bytes) : 0;
  return (
    <Card className="gap-2 py-3">
      <CardContent className="flex flex-col gap-2">
        <div className="flex min-w-0 items-center gap-2">
          <HardDrive className="size-4 shrink-0 text-muted-foreground" />
          <Hint label={dir.path}>
            <span className="cursor-default truncate font-mono text-xs">{dir.path}</span>
          </Hint>
        </div>
        <div className="text-xs text-muted-foreground">
          {dir.exists
            ? `${dir.models} model${dir.models === 1 ? "" : "s"} · ${dir.drafters} drafter${dir.drafters === 1 ? "" : "s"} · ${dir.ngram_volumes} n-gram volume${dir.ngram_volumes === 1 ? "" : "s"}`
            : "Folder not found"}
          {dir.error ? <span className="text-critical"> · {dir.error}</span> : null}
        </div>
        {known ? (
          <>
            <div
              className="flex h-2 w-full overflow-hidden rounded-full bg-muted"
              role="img"
              aria-label={`${formatBytes(dir.library_bytes)} in the library, ${formatBytes(other)} other files, ${formatBytes(free)} free of ${formatBytes(total)}`}
            >
              <div className="h-full bg-primary" style={{ width: pct(dir.library_bytes) }} />
              <div className="h-full bg-muted-foreground/35" style={{ width: pct(other) }} />
            </div>
            <div className="flex flex-wrap justify-between gap-x-3 gap-y-0.5 text-xs">
              <span className="inline-flex items-center gap-1.5">
                <span className="size-2 rounded-sm bg-primary" />
                <span className="font-medium tabular-nums">{formatBytes(dir.library_bytes)}</span>
                <span className="text-muted-foreground">in the library</span>
              </span>
              <span
                className={cn(
                  "tabular-nums",
                  tone === "critical" ? "font-medium text-critical" : tone === "warning" ? "font-medium text-warning" : "text-muted-foreground",
                )}
              >
                {formatBytes(free)} free of {formatBytes(total)}
              </span>
            </div>
          </>
        ) : (
          <div className="text-xs text-muted-foreground">
            <span className="font-medium text-foreground tabular-nums">{formatBytes(dir.library_bytes)}</span> in the library ·
            drive space unknown
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function Folders({ library }: { library: Library }) {
  if (library.dirs.length === 0) return null;
  return (
    <section className="flex flex-col gap-3">
      <div>
        <h2 className="text-sm font-semibold">Model folders</h2>
        <p className="text-xs text-muted-foreground">
          What the library holds in each folder and the space left on its drive. Manage the folders under Settings.
        </p>
      </div>
      <div className="grid gap-3 md:grid-cols-2 2xl:grid-cols-3">
        {library.dirs.map((dir) => (
          <FolderCard key={dir.path} dir={dir} />
        ))}
      </div>
    </section>
  );
}

function DeleteButton({ label, onClick }: { label: string; onClick: () => void }) {
  return (
    <Hint label="Delete files…">
      <Button variant="ghost" size="icon-xs" className="text-muted-foreground" aria-label={label} onClick={onClick}>
        <Trash2 />
      </Button>
    </Hint>
  );
}

function Companions({ library, actions }: { library: Library; actions: Actions }) {
  const { drafters, ngram_volumes: volumes } = library;
  if (drafters.length === 0 && volumes.length === 0) return null;
  return (
    <section className="flex flex-col gap-3">
      <div>
        <h2 className="text-sm font-semibold">Companions</h2>
        <p className="text-xs text-muted-foreground">
          Drafters and n-gram tables found beside your models; pick them per configuration under Configure.
        </p>
      </div>
      <div className="grid gap-4 lg:grid-cols-2">
        <Card className="gap-2">
          <CardHeader>
            <CardTitle>
              <Sparkles className="size-4 text-muted-foreground" />
              Drafters
            </CardTitle>
            <CardDescription>DFlash2 drafter artifacts for --draft-model</CardDescription>
          </CardHeader>
          <CardContent className="px-0">
            {drafters.length === 0 ? (
              <div className="px-4 text-sm text-muted-foreground">None found.</div>
            ) : (
              <ul className="divide-y">
                {drafters.map((drafter: DrafterArtifact) => (
                  <li key={drafter.key} className="flex items-center gap-3 py-2.5 pr-2 pl-4">
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-sm font-medium">{drafter.name}</div>
                      <Hint label={drafter.path}>
                        <div className="cursor-default truncate text-xs text-muted-foreground">{drafter.dir}</div>
                      </Hint>
                    </div>
                    <Badge variant="muted">{drafter.kind}</Badge>
                    <Hint label={`Drafts for: ${modelNames(library, drafter.compatible_models)}`}>
                      <span className="w-20 cursor-default text-right text-xs text-muted-foreground tabular-nums">
                        {drafter.compatible_models.length} model{drafter.compatible_models.length === 1 ? "" : "s"}
                      </span>
                    </Hint>
                    <span className="w-16 text-right text-xs tabular-nums">{formatBytes(drafter.bytes)}</span>
                    <DeleteButton
                      label={`Delete ${drafter.file_name}`}
                      onClick={() => actions.deleteFiles("drafter", drafter.key)}
                    />
                  </li>
                ))}
              </ul>
            )}
          </CardContent>
        </Card>
        <Card className="gap-2">
          <CardHeader>
            <CardTitle>
              <Database className="size-4 text-muted-foreground" />
              N-gram volumes
            </CardTitle>
            <CardDescription>PLE n-gram tables for Qwen3.8-Flash-Next (--ngram-volume)</CardDescription>
          </CardHeader>
          <CardContent className="px-0">
            {volumes.length === 0 ? (
              <div className="px-4 text-sm text-muted-foreground">None found.</div>
            ) : (
              <ul className="divide-y">
                {volumes.map((volume: NgramVolume) => (
                  <li key={`${volume.key}:${volume.path}`} className="flex items-center gap-3 py-2.5 pr-2 pl-4">
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-sm font-medium">{volume.file_name}</div>
                      <Hint label={volume.path}>
                        <div className="cursor-default truncate text-xs text-muted-foreground">{volume.dir}</div>
                      </Hint>
                    </div>
                    <Badge variant="muted">{NGRAM_FORMAT_LABELS[volume.format]}</Badge>
                    <Hint
                      label={
                        volume.own_for.length > 0
                          ? `Own table of ${modelNames(library, volume.own_for)}; readable by ${modelNames(library, volume.compatible_models)}`
                          : `Readable by ${modelNames(library, volume.compatible_models)}`
                      }
                    >
                      <span className="w-20 cursor-default text-right text-xs text-muted-foreground tabular-nums">
                        {volume.compatible_models.length} model{volume.compatible_models.length === 1 ? "" : "s"}
                      </span>
                    </Hint>
                    <span className="w-16 text-right text-xs tabular-nums">{formatBytes(volume.bytes)}</span>
                    <DeleteButton
                      label={`Delete ${volume.file_name}`}
                      onClick={() => actions.deleteFiles("ngram", volume.key)}
                    />
                  </li>
                ))}
              </ul>
            )}
          </CardContent>
        </Card>
      </div>
    </section>
  );
}

function matches(model: ModelArtifact, needle: string): boolean {
  if (needle.length === 0) return true;
  return (
    model.name.toLowerCase().includes(needle) ||
    model.file_name.toLowerCase().includes(needle) ||
    model.family_label.toLowerCase().includes(needle) ||
    model.configs.some(
      (config) => config.alias.toLowerCase().includes(needle) || config.name.toLowerCase().includes(needle),
    )
  );
}

export function ModelsPage() {
  const navigate = useNavigate();
  const { data: library, isLoading } = useLibrary();
  const { data: settings } = useSettings();
  const instance = useLive((live) => live.instance);
  const [query, setQuery] = useState("");
  const [family, setFamily] = useState<ModelFamily | "all">("all");
  const [view, setView] = useState<View>(() => (readStorage(VIEW_KEY) === "list" ? "list" : "grid"));
  const [picker, setPicker] = useState(false);
  const [importing, setImporting] = useState(false);
  const [creating, setCreating] = useState<{ key: string; from: string | null } | null>(null);
  const [deletingConfig, setDeletingConfig] = useState<string | null>(null);
  const [deletingFiles, setDeletingFiles] = useState<{ kind: FileKind; key: string } | null>(null);
  const { load, dialog, pending } = useLoadModel();
  const deleteConfig = useDeleteConfig();

  const rescan = useMutation({ mutationFn: api.rescan, onSuccess: () => toast.info("Rescanning the model folders") });
  const addDir = useMutation({
    mutationFn: (path: string) => api.addDir(path),
    onSuccess: (_, path) => toast.success(`Added ${path}; scanning it now`),
  });

  const families = useMemo(() => {
    const present = new Set((library?.models ?? []).map((model) => model.family));
    return FAMILY_ORDER.filter((item) => present.has(item));
  }, [library]);

  const models = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return (library?.models ?? [])
      .filter((model) => family === "all" || model.family === family)
      .filter((model) => matches(model, needle))
      .sort(
        (a, b) =>
          FAMILY_ORDER.indexOf(a.family) - FAMILY_ORDER.indexOf(b.family) || a.name.localeCompare(b.name),
      );
  }, [library, query, family]);

  const actions: Actions = {
    load: (config) => load(config.id, config.alias),
    loading: pending,
    newConfig: (model, from) => setCreating({ key: model.key, from }),
    deleteConfig: (config) => setDeletingConfig(config.id),
    deleteFiles: (kind, key) => setDeletingFiles({ kind, key }),
  };

  const creatingModel = creating ? findModel(library, creating.key) : null;
  const configToDelete = findConfig(library, deletingConfig);
  const filesTarget: DeleteTarget | null = (() => {
    if (!deletingFiles || !library) return null;
    if (deletingFiles.kind === "model") {
      const model = library.models.find((item) => item.key === deletingFiles.key);
      return model ? modelDeleteTarget(model, instance) : null;
    }
    if (deletingFiles.kind === "drafter") {
      const drafter = library.drafters.find((item) => item.key === deletingFiles.key);
      return drafter ? drafterDeleteTarget(drafter, instance) : null;
    }
    const volume = library.ngram_volumes.find((item) => item.key === deletingFiles.key);
    return volume ? volumeDeleteTarget(volume, instance) : null;
  })();

  const total = library?.models.length ?? 0;
  const configCount = library?.models.reduce((count, model) => count + model.configs.length, 0) ?? 0;
  const defaultId = settings?.default_model ?? null;

  return (
    <div className="flex flex-col gap-5">
      <PageHeader
        title="Models"
        description={
          library
            ? `${total} model${total === 1 ? "" : "s"} (${configCount} configuration${configCount === 1 ? "" : "s"}), ${library.drafters.length} drafter${library.drafters.length === 1 ? "" : "s"} and ${library.ngram_volumes.length} n-gram volume${library.ngram_volumes.length === 1 ? "" : "s"} in ${library.dirs.length} folder${library.dirs.length === 1 ? "" : "s"}`
            : "Artifacts found in your model folders"
        }
        actions={
          <>
            <Button variant="outline" size="sm" disabled={rescan.isPending || library?.scanning} onClick={() => rescan.mutate()}>
              <RefreshCw className={cn(library?.scanning && "animate-spin")} />
              {library?.scanning ? "Scanning…" : "Rescan"}
            </Button>
            <Button variant="outline" size="sm" onClick={() => setPicker(true)}>
              <FolderPlus />
              Add folder
            </Button>
            <Button variant="outline" size="sm" onClick={() => setImporting(true)}>
              <FileInput />
              Import launch script
            </Button>
            <Button size="sm" asChild>
              <Link to="/discover">
                <CloudDownload />
                Download models
              </Link>
            </Button>
          </>
        }
      />

      {library && library.issues.length > 0 ? (
        <Alert variant="warning">
          <TriangleAlert />
          <AlertTitle>
            {library.issues.length} file{library.issues.length === 1 ? "" : "s"} could not be used
          </AlertTitle>
          <AlertDescription>
            <ul className="flex flex-col gap-0.5 text-xs">
              {library.issues.slice(0, 6).map((issue) => (
                <li key={`${issue.path}:${issue.message}`} className="break-all">
                  <span className="font-mono">{issue.path}</span>: {issue.message}
                </li>
              ))}
              {library.issues.length > 6 ? <li>…and {library.issues.length - 6} more</li> : null}
            </ul>
          </AlertDescription>
        </Alert>
      ) : null}

      {total > 0 ? (
        <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
          <div className="relative w-full sm:max-w-xs">
            <Search className="pointer-events-none absolute top-1/2 left-2.5 size-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search models and model ids"
              className="pl-8"
              aria-label="Search models"
            />
          </div>
          <Select value={family} onValueChange={(value) => setFamily(value as ModelFamily | "all")}>
            <SelectTrigger className="sm:w-52" aria-label="Family">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All families</SelectItem>
              {families.map((item) => (
                <SelectItem key={item} value={item}>
                  {library?.models.find((model) => model.family === item)?.family_label ?? item}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <ToggleGroup
            type="single"
            value={view}
            className="sm:ml-auto"
            onValueChange={(value) => {
              if (value === "grid" || value === "list") {
                setView(value);
                writeStorage(VIEW_KEY, value);
              }
            }}
            aria-label="View"
          >
            <ToggleGroupItem value="grid" aria-label="Grid">
              <LayoutGrid />
            </ToggleGroupItem>
            <ToggleGroupItem value="list" aria-label="List">
              <List />
            </ToggleGroupItem>
          </ToggleGroup>
        </div>
      ) : null}

      {isLoading ? (
        <div className="grid gap-4 md:grid-cols-2 2xl:grid-cols-3">
          {[0, 1, 2].map((index) => (
            <Skeleton key={index} className="h-64 rounded-xl" />
          ))}
        </div>
      ) : total === 0 ? (
        <EmptyState
          icon={Boxes}
          title={library && library.dirs.length > 0 ? "No models in your folders yet" : "Add a folder with Infernix models"}
          description={
            library && library.dirs.length > 0
              ? "The launcher looks for .infernix and .ninfer artifacts, DFlash2 drafters and n-gram volumes in the folders listed under Settings. Download a model, or add the folder that holds yours."
              : "Point the launcher at the folders that hold .infernix or .ninfer artifacts, or download a supported model from Hugging Face."
          }
          actions={
            <>
              <Button onClick={() => setPicker(true)}>
                <FolderPlus />
                Add a folder
              </Button>
              <Button variant="outline" asChild>
                <Link to="/discover">
                  <CloudDownload />
                  Download a model
                </Link>
              </Button>
            </>
          }
        />
      ) : models.length === 0 ? (
        <div className="rounded-xl border border-dashed p-8 text-center text-sm text-muted-foreground">
          No model matches the search.
        </div>
      ) : view === "grid" ? (
        <div className="grid gap-4 md:grid-cols-2 2xl:grid-cols-3">
          {models.map((model) => (
            <ModelCard key={model.key} model={model} instance={instance} defaultId={defaultId} actions={actions} />
          ))}
        </div>
      ) : (
        <ConfigTable models={models} instance={instance} defaultId={defaultId} actions={actions} />
      )}

      {library ? <Folders library={library} /> : null}
      {library ? <Companions library={library} actions={actions} /> : null}

      <FolderPicker
        open={picker}
        onOpenChange={setPicker}
        title="Add a model folder"
        description="The launcher scans it (and its subfolders) for artifacts, drafters and n-gram volumes."
        confirmLabel="Add this folder"
        onSelect={(path) => addDir.mutate(path)}
      />
      <ImportScriptDialog open={importing} onOpenChange={setImporting} />
      {creatingModel && creating ? (
        <NewConfigDialog
          open
          onOpenChange={(open) => {
            if (!open) setCreating(null);
          }}
          model={creatingModel}
          from={creating.from}
          onCreated={(config) =>
            toast.success(`Created ${config.name} (model id ${config.alias})`, {
              action: { label: "Configure", onClick: () => navigate(configPath(config.id)) },
            })
          }
        />
      ) : null}
      <ConfirmDialog
        open={configToDelete !== null}
        onOpenChange={(open) => {
          if (!open) setDeletingConfig(null);
        }}
        title={`Delete the configuration ${configToDelete?.config.name ?? ""}?`}
        description={
          configToDelete ? (
            <span>
              {describeConfig(configToDelete.model, configToDelete.config)} and its model id{" "}
              <span className="font-mono text-foreground">{configToDelete.config.alias}</span> go; requests naming it no
              longer find it. The artifact and its other configurations stay.
              {liveInstance(instance)?.model_key === configToDelete.config.id ? " It is loaded now." : ""}
            </span>
          ) : (
            ""
          )
        }
        confirmLabel="Delete configuration"
        destructive
        onConfirm={() => {
          if (!configToDelete) return;
          const { config } = configToDelete;
          deleteConfig.mutate(config.id, { onSuccess: () => toast.success(`Deleted the configuration ${config.name}`) });
        }}
      />
      <DeleteFilesDialog
        target={filesTarget}
        onOpenChange={(open) => {
          if (!open) setDeletingFiles(null);
        }}
      />
      {dialog}
    </div>
  );
}
