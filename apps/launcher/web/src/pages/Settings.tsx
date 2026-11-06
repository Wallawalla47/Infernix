import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  ArrowRightLeft,
  CircleCheck,
  CircleX,
  Download,
  FolderOpen,
  FolderPlus,
  Info,
  Monitor,
  Moon,
  Palette,
  RefreshCw,
  RotateCcw,
  RotateCw,
  Save,
  ScrollText,
  Server,
  ShieldCheck,
  Sun,
  Terminal,
  Trash2,
  TriangleAlert,
} from "lucide-react";
import { toast } from "sonner";

import { ApiKeysCard } from "@/components/admin/ApiKeysCard";
import { ModelRoutesCard } from "@/components/admin/ModelRoutesCard";
import { UpdatesCard } from "@/components/admin/UpdatesCard";
import { FieldRow, PageHeader, SectionCardTitle } from "@/components/common";
import { CopyButton } from "@/components/CopyButton";
import { FolderPicker } from "@/components/FolderPicker";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Switch } from "@/components/ui/switch";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { api } from "@/lib/api";
import type { CpuTempSource, Settings, SettingsPatch, UnknownModelPolicy } from "@/lib/api-types";
import { formatBytes } from "@/lib/format";
import { useLauncherState, useLibrary, useOpenLogFolder, useServeOptions, useSettings } from "@/lib/hooks";
import { allConfigs, findConfig } from "@/lib/library";
import { useTheme, type Theme } from "@/lib/theme";
import { cn } from "@/lib/utils";
import { CommitInput } from "@/pages/configure/controls";

const NO_DEFAULT = "__none__";

function PathBox({ path }: { path: string | null }) {
  return (
    <div className="flex items-center gap-2 rounded-lg border bg-muted/40 px-3 py-2">
      <code className="min-w-0 flex-1 truncate font-mono text-xs" title={path ?? undefined}>
        {path ?? "—"}
      </code>
      {path ? <CopyButton text={path} size="icon-xs" toastMessage="Path copied" /> : null}
    </div>
  );
}

const CPU_TEMP_SOURCES: Record<CpuTempSource, string> = {
  hwinfo: "HWiNFO (CPU package)",
  coretemp: "Core Temp (CPU package)",
  "thermal-zone": "Windows ACPI thermal zone (a motherboard sensor)",
  hwmon: "Linux hwmon",
};

function SecretField({
  label,
  help,
  isSet,
  onSave,
  pending,
}: {
  label: string;
  help: string;
  isSet: boolean;
  onSave: (value: string) => void;
  pending: boolean;
}) {
  const [value, setValue] = useState("");
  return (
    <FieldRow
      label={
        <span className="flex items-center gap-2">
          {label}
          {isSet ? <Badge variant="accent">set</Badge> : <Badge variant="muted">not set</Badge>}
        </span>
      }
      help={help}
      stacked
    >
      <form
        className="flex gap-2"
        onSubmit={(event) => {
          event.preventDefault();
          if (value.trim().length === 0) return;
          onSave(value.trim());
          setValue("");
        }}
      >
        <Input
          type="password"
          value={value}
          autoComplete="off"
          placeholder={isSet ? "Enter a new value to replace it" : "Enter a value"}
          onChange={(event) => setValue(event.target.value)}
          aria-label={label}
        />
        <Button type="submit" variant="outline" disabled={pending || value.trim().length === 0}>
          Save
        </Button>
        {isSet ? (
          <Button variant="ghost" disabled={pending} onClick={() => onSave("")}>
            Clear
          </Button>
        ) : null}
      </form>
    </FieldRow>
  );
}

function NumberSetting({
  label,
  help,
  value,
  onSave,
  min,
  max,
  suffix,
  disabled = false,
}: {
  label: string;
  help: string;
  value: number;
  onSave: (value: number) => void;
  min?: number;
  max?: number;
  suffix?: string;
  disabled?: boolean;
}) {
  return (
    <FieldRow label={label} help={help} className={cn(disabled && "text-muted-foreground")}>
      <div className="flex items-center gap-2">
        <CommitInput
          type="number"
          value={String(value)}
          min={min}
          max={max}
          disabled={disabled}
          ariaLabel={label}
          onCommit={(text) => {
            const next = Number(text);
            if (text.length === 0 || !Number.isFinite(next)) {
              toast.error(`${label}: enter a number`);
              return;
            }
            onSave(next);
          }}
        />
        {suffix ? <span className="shrink-0 text-xs text-muted-foreground">{suffix}</span> : null}
      </div>
    </FieldRow>
  );
}

export function SettingsPage() {
  const queryClient = useQueryClient();
  const { data: settings, isLoading } = useSettings();
  const { data: state } = useLauncherState();
  const { data: library } = useLibrary();
  const serveOptions = useServeOptions();
  const { theme, setTheme } = useTheme();
  const [folderPicker, setFolderPicker] = useState(false);
  const [downloadPicker, setDownloadPicker] = useState(false);
  const [logPicker, setLogPicker] = useState(false);
  const openLogFolder = useOpenLogFolder();

  const save = useMutation({
    mutationFn: (patch: SettingsPatch) => api.saveSettings(patch),
    onSuccess: (next: Settings, patch) => {
      queryClient.setQueryData(["settings"], next);
      void queryClient.invalidateQueries({ queryKey: ["state"] });
      if ("serve_executable" in patch) void queryClient.invalidateQueries({ queryKey: ["serve-options"] });
      toast.success("Saved");
    },
  });
  const addDir = useMutation({
    mutationFn: (path: string) => api.addDir(path),
    onSuccess: (_, path) => {
      toast.success(`Added ${path}; scanning it now`);
      void queryClient.invalidateQueries({ queryKey: ["settings"] });
    },
  });
  const removeDir = useMutation({
    mutationFn: (path: string) => api.removeDir(path),
    onSuccess: (_, path) => {
      toast.success(`Removed ${path} (its files are untouched)`);
      void queryClient.invalidateQueries({ queryKey: ["settings"] });
    },
  });
  const rescan = useMutation({ mutationFn: api.rescan, onSuccess: () => toast.info("Rescanning the model folders") });

  if (isLoading || !settings) {
    return (
      <div className="flex flex-col gap-4">
        <Skeleton className="h-10 w-48" />
        <Skeleton className="h-64 rounded-xl" />
        <Skeleton className="h-64 rounded-xl" />
      </div>
    );
  }

  const patch = (value: SettingsPatch) => save.mutate(value);
  const dirInfo = (path: string) => library?.dirs.find((dir) => dir.path === path) ?? null;
  const system = state?.system ?? null;
  const logs = state?.logs ?? null;
  const watching = settings.watchdog;

  return (
    <div className="flex flex-col gap-5">
      <PageHeader title="Settings" description="Everything here persists between launcher runs." />

      {state?.restart_required ? (
        <Alert variant="warning">
          <RotateCw />
          <AlertTitle>Restart the launcher to apply the new address</AlertTitle>
          <AlertDescription>
            The listen address, port or remote access changed. Close infernix-launcher and start it again; your settings are kept.
          </AlertDescription>
        </Alert>
      ) : null}

      <div className="grid gap-5 xl:grid-cols-2">
        <Card className="gap-2 xl:col-span-2">
          <CardHeader>
            <CardTitle>
              <SectionCardTitle icon={FolderOpen}>Model folders</SectionCardTitle>
            </CardTitle>
            <CardDescription>
              Scanned for .infernix and .ninfer models, DFlash2 drafters and n-gram volumes, including subfolders.
            </CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-3">
            {settings.model_dirs.length === 0 ? (
              <div className="rounded-lg border border-dashed p-6 text-center text-sm text-muted-foreground">
                No folders yet. Add the folders that hold your artifacts, or download a model from Discover.
              </div>
            ) : (
              <ul className="divide-y rounded-lg border">
                {settings.model_dirs.map((path) => {
                  const info = dirInfo(path);
                  return (
                    <li key={path} className="flex flex-wrap items-center gap-3 px-3 py-2.5">
                      {info && !info.exists ? (
                        <CircleX className="size-4 shrink-0 text-critical" aria-label="missing" />
                      ) : (
                        <CircleCheck className="size-4 shrink-0 text-success" aria-label="found" />
                      )}
                      <div className="min-w-0 flex-1">
                        <div className="truncate font-mono text-xs">{path}</div>
                        <div className="text-xs text-muted-foreground">
                          {info
                            ? info.exists
                              ? `${info.models} model${info.models === 1 ? "" : "s"} · ${info.drafters} drafter${info.drafters === 1 ? "" : "s"} · ${info.ngram_volumes} n-gram volume${info.ngram_volumes === 1 ? "" : "s"}`
                              : "Folder not found"
                            : "Not scanned yet"}
                          {info?.error ? <span className="text-critical"> · {info.error}</span> : null}
                        </div>
                      </div>
                      <Button
                        variant="ghost"
                        size="icon-sm"
                        aria-label={`Remove ${path}`}
                        disabled={removeDir.isPending}
                        onClick={() => removeDir.mutate(path)}
                      >
                        <Trash2 />
                      </Button>
                    </li>
                  );
                })}
              </ul>
            )}
            <div className="flex flex-wrap items-center gap-2">
              <Button variant="outline" size="sm" onClick={() => setFolderPicker(true)}>
                <FolderPlus />
                Add folder
              </Button>
              <Button variant="outline" size="sm" disabled={rescan.isPending || library?.scanning} onClick={() => rescan.mutate()}>
                <RefreshCw className={library?.scanning ? "animate-spin" : undefined} />
                Rescan
              </Button>
              <div className="ml-auto flex items-center gap-2 text-xs text-muted-foreground">
                Scan depth
                <CommitInput
                  type="number"
                  className="h-8 w-16"
                  value={String(settings.scan_depth)}
                  min={0}
                  max={6}
                  ariaLabel="Scan depth"
                  onCommit={(text) => {
                    const next = Number(text);
                    if (Number.isInteger(next) && next >= 0) patch({ scan_depth: next });
                  }}
                />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="gap-1">
          <CardHeader>
            <CardTitle>
              <SectionCardTitle icon={Terminal}>infernix-serve</SectionCardTitle>
            </CardTitle>
            <CardDescription>The server the launcher starts for each model</CardDescription>
          </CardHeader>
          <CardContent className="divide-y">
            <FieldRow label="Executable" help="Leave empty to use the infernix-serve beside the launcher." stacked>
              <CommitInput
                mono
                value={settings.serve_executable}
                placeholder="infernix-serve beside the launcher"
                ariaLabel="infernix-serve executable"
                onCommit={(text) => patch({ serve_executable: text })}
              />
            </FieldRow>
            <div className="flex items-start gap-2.5 py-3 text-sm">
              {serveOptions.data?.found ? (
                <CircleCheck className="mt-0.5 size-4 shrink-0 text-success" />
              ) : (
                <CircleX className="mt-0.5 size-4 shrink-0 text-critical" />
              )}
              <div className="min-w-0">
                <div className="font-medium">
                  {serveOptions.isLoading
                    ? "Checking…"
                    : serveOptions.data?.found
                      ? `Found${serveOptions.data.build ? `: build ${serveOptions.data.build}` : ""}`
                      : "Not found"}
                </div>
                <div className="truncate font-mono text-xs text-muted-foreground">{serveOptions.data?.executable ?? ""}</div>
                {serveOptions.data?.error ? <div className="text-xs text-critical">{serveOptions.data.error}</div> : null}
                {serveOptions.data?.found ? (
                  <div className="text-xs text-muted-foreground">{serveOptions.data.options.length} options available</div>
                ) : null}
              </div>
            </div>
            <NumberSetting
              label="Server port"
              help="Loopback port of the managed infernix-serve; 0 picks a free one each time."
              value={settings.child_port}
              min={0}
              max={65535}
              onSave={(value) => patch({ child_port: value })}
            />
          </CardContent>
        </Card>

        <Card className="gap-1">
          <CardHeader>
            <CardTitle>
              <SectionCardTitle icon={Server}>Launcher server</SectionCardTitle>
            </CardTitle>
            <CardDescription>Where coding tools and this page reach the launcher</CardDescription>
          </CardHeader>
          <CardContent className="divide-y">
            <FieldRow label="Listen address" help="127.0.0.1 keeps it on this computer; 0.0.0.0 serves the network (create an API key first). Restart to apply.">
              <CommitInput mono value={settings.host} ariaLabel="Listen address" onCommit={(text) => text.length > 0 && patch({ host: text })} />
            </FieldRow>
            <NumberSetting
              label="Port"
              help="Restart to apply. LM Studio's default is 1234."
              value={settings.port}
              min={1}
              max={65535}
              onSave={(value) => patch({ port: value })}
            />
            <FieldRow label="Browser access from other origins (CORS)" help="Lets web apps on other origins call the API.">
              <div className="flex sm:justify-end">
                <Switch checked={settings.cors} onCheckedChange={(checked) => patch({ cors: checked })} aria-label="CORS" />
              </div>
            </FieldRow>
            <FieldRow
              label="llama.cpp endpoints"
              help="Answers llama-server's router API (/props, /models/load, /models/unload, /models/sse) for tools with a llama.cpp provider, such as Zed. Tools that tell servers apart by these paths (Hermes) then treat the launcher as llama.cpp. Applies at once."
            >
              <div className="flex sm:justify-end">
                <Switch
                  checked={settings.llamacpp_api}
                  onCheckedChange={(checked) => patch({ llamacpp_api: checked })}
                  aria-label="llama.cpp endpoints"
                />
              </div>
            </FieldRow>
            <FieldRow
              label="Manage from other computers"
              help="Allow this page and its management API from non-loopback addresses; once any API key exists, they sign in with an admin key."
            >
              <div className="flex sm:justify-end">
                <Switch
                  checked={settings.allow_remote_admin}
                  onCheckedChange={(checked) => patch({ allow_remote_admin: checked })}
                  aria-label="Remote management"
                />
              </div>
            </FieldRow>
          </CardContent>
        </Card>

        <ApiKeysCard settings={settings} className="gap-2 xl:col-span-2" />

        <Card className="gap-1">
          <CardHeader>
            <CardTitle>
              <SectionCardTitle icon={ArrowRightLeft}>Model switching</SectionCardTitle>
            </CardTitle>
            <CardDescription>How requests for different models share the GPU</CardDescription>
          </CardHeader>
          <CardContent className="divide-y">
            <FieldRow label="Unknown model ids" help="What a request naming a model the launcher does not know gets.">
              <Select value={settings.unknown_model} onValueChange={(value) => patch({ unknown_model: value as UnknownModelPolicy })}>
                <SelectTrigger aria-label="Unknown model ids">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="loaded">The loaded (or default) model</SelectItem>
                  <SelectItem value="reject">Rejected (model_not_found)</SelectItem>
                </SelectContent>
              </Select>
            </FieldRow>
            <FieldRow label="Default model" help="Loaded for unknown ids while nothing is loaded, and at startup with autoload.">
              <Select
                value={settings.default_model ?? NO_DEFAULT}
                onValueChange={(value) => patch({ default_model: value === NO_DEFAULT ? null : value })}
              >
                <SelectTrigger aria-label="Default model">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value={NO_DEFAULT}>None</SelectItem>
                  {allConfigs(library).map(({ config }) => (
                    <SelectItem key={config.id} value={config.id}>
                      {config.alias}
                    </SelectItem>
                  ))}
                  {settings.default_model && !findConfig(library, settings.default_model) ? (
                    <SelectItem value={settings.default_model}>Missing model</SelectItem>
                  ) : null}
                </SelectContent>
              </Select>
            </FieldRow>
            <FieldRow label="Load at startup" help="When the launcher starts, load the default model (else the last one used).">
              <div className="flex sm:justify-end">
                <Switch checked={settings.autoload} onCheckedChange={(checked) => patch({ autoload: checked })} aria-label="Load at startup" />
              </div>
            </FieldRow>
            <FieldRow
              label="Keep each model's prefix cache"
              help="Save the prefix cache when a model unloads and restore it at its next load (adds --prefix-cache-file in the launcher folder; several GB per model)."
            >
              <div className="flex sm:justify-end">
                <Switch
                  checked={settings.persist_prefix_cache}
                  onCheckedChange={(checked) => patch({ persist_prefix_cache: checked })}
                  aria-label="Keep the prefix cache"
                />
              </div>
            </FieldRow>
            <NumberSetting
              label="Drain timeout"
              help="How long running requests may finish before a switch cancels them."
              value={settings.drain_timeout_s}
              min={0}
              suffix="s"
              onSave={(value) => patch({ drain_timeout_s: value })}
            />
            <NumberSetting
              label="Load timeout"
              help="How long a model may take to become ready before the load fails."
              value={settings.load_timeout_s}
              min={10}
              suffix="s"
              onSave={(value) => patch({ load_timeout_s: value })}
            />
            <NumberSetting
              label="Stop timeout"
              help="How long an orderly stop (which saves caches) may take before the server is ended."
              value={settings.stop_timeout_s}
              min={5}
              suffix="s"
              onSave={(value) => patch({ stop_timeout_s: value })}
            />
            <NumberSetting
              label="Unload when idle"
              help="Unload the model after this many minutes without requests, freeing the GPU; 0 = never. Each configuration can override it (Configure model)."
              value={settings.idle_unload_min}
              min={0}
              suffix="min"
              onSave={(value) => patch({ idle_unload_min: value })}
            />
          </CardContent>
        </Card>

        <Card className="gap-1">
          <CardHeader>
            <CardTitle>
              <SectionCardTitle icon={Download}>Downloads</SectionCardTitle>
            </CardTitle>
            <CardDescription>Hugging Face downloads</CardDescription>
          </CardHeader>
          <CardContent className="divide-y">
            <FieldRow label="Default folder" help="Proposed by the download dialog." stacked>
              <div className="flex gap-2">
                <CommitInput
                  mono
                  value={settings.download_dir}
                  placeholder="Choose a folder"
                  ariaLabel="Default download folder"
                  onCommit={(text) => patch({ download_dir: text })}
                />
                <Button variant="outline" onClick={() => setDownloadPicker(true)}>
                  <FolderOpen />
                  Browse
                </Button>
              </div>
            </FieldRow>
            <NumberSetting
              label="Connections per file"
              help="Parallel ranges per file; more can fill a fast link."
              value={settings.download_connections}
              min={1}
              max={32}
              onSave={(value) => patch({ download_connections: value })}
            />
            <FieldRow label="Verify checksums" help="Check each finished file's SHA-256 against Hugging Face's record.">
              <div className="flex sm:justify-end">
                <Switch
                  checked={settings.verify_downloads}
                  onCheckedChange={(checked) => patch({ verify_downloads: checked })}
                  aria-label="Verify checksums"
                />
              </div>
            </FieldRow>
            <SecretField
              label="Hugging Face token"
              help="Only for gated or private repositories; the supported models need none."
              isSet={settings.hf_token_set}
              pending={save.isPending}
              onSave={(value) => patch({ hf_token: value })}
            />
          </CardContent>
        </Card>

        <ModelRoutesCard
          settings={settings}
          library={library}
          onSave={patch}
          saving={save.isPending}
          className="xl:col-span-2"
        />

        <Card className="gap-1">
          <CardHeader>
            <CardTitle>
              <SectionCardTitle icon={ScrollText}>Logging</SectionCardTitle>
            </CardTitle>
            <CardDescription>The launcher's event log, request contents and the server logs</CardDescription>
            <CardAction>
              <Button variant="outline" size="sm" disabled={openLogFolder.isPending} onClick={() => openLogFolder.mutate()}>
                <FolderOpen />
                Open log folder
              </Button>
            </CardAction>
          </CardHeader>
          <CardContent className="divide-y">
            <FieldRow
              label="Log folder"
              help="Leave empty for the launcher's own folder. A change applies at once; files already written stay where they are."
              stacked
            >
              <div className="flex flex-col gap-1.5">
                <div className="flex gap-2">
                  <CommitInput
                    mono
                    value={settings.log_dir}
                    placeholder={logs?.dir ?? "The launcher's folder"}
                    ariaLabel="Log folder"
                    onCommit={(text) => patch({ log_dir: text })}
                  />
                  <Button variant="outline" onClick={() => setLogPicker(true)}>
                    <FolderOpen />
                    Browse
                  </Button>
                </div>
                <div className="flex min-h-7 items-center gap-2 text-xs text-muted-foreground">
                  <span className="shrink-0">In use</span>
                  <code className="min-w-0 truncate font-mono text-foreground" title={logs?.dir}>
                    {logs?.dir ?? "—"}
                  </code>
                  {settings.log_dir.length > 0 ? (
                    <Button
                      variant="ghost"
                      size="xs"
                      className="ml-auto"
                      disabled={save.isPending}
                      onClick={() => patch({ log_dir: "" })}
                    >
                      <RotateCcw />
                      Use default
                    </Button>
                  ) : (
                    <Badge variant="muted" className="ml-auto">
                      default
                    </Badge>
                  )}
                </div>
              </div>
            </FieldRow>
            <FieldRow label="Event log" help="The launcher's events and one line per API request." stacked>
              <PathBox path={logs?.events_file ?? null} />
            </FieldRow>
            <FieldRow label="Request contents" help="Written only while request and response contents are recorded (below)." stacked>
              <PathBox path={logs?.requests_file ?? null} />
            </FieldRow>
            <FieldRow label="Server logs" help="One file per server run (serve-<time>-<model>.log); the newest 20 are kept." stacked>
              <PathBox path={logs?.serve_dir ?? null} />
            </FieldRow>
            <FieldRow
              label={
                <span className="flex items-center gap-2">
                  Record request and response contents
                  {settings.log_bodies ? <Badge variant="accent">recording</Badge> : null}
                </span>
              }
              help={
                <span className="flex items-start gap-1.5">
                  <TriangleAlert className="mt-0.5 size-3.5 shrink-0 text-warning" />
                  <span>
                    Stores every prompt and answer in full in infernix-launcher-requests.jsonl (rotated at 100 MiB, 5 files
                    kept); it can grow quickly.
                  </span>
                </span>
              }
            >
              <div className="flex sm:justify-end">
                <Switch
                  checked={settings.log_bodies}
                  onCheckedChange={(checked) => patch({ log_bodies: checked })}
                  aria-label="Record request and response contents"
                />
              </div>
            </FieldRow>
          </CardContent>
        </Card>

        <Card className="gap-1">
          <CardHeader>
            <CardTitle>
              <SectionCardTitle icon={ShieldCheck}>Watchdog</SectionCardTitle>
            </CardTitle>
            <CardDescription>Crashes, lock-ups, stalls and loads that run out of memory</CardDescription>
          </CardHeader>
          <CardContent className="divide-y">
            <FieldRow
              label="Watch the server for crashes, lock-ups and stalls"
              help="Ends a server that stops answering or stops making progress while requests wait on it, and reports every incident on the dashboard and in the event log. A server sitting idle is never ended."
            >
              <div className="flex sm:justify-end">
                <Switch checked={watching} onCheckedChange={(checked) => patch({ watchdog: checked })} aria-label="Watchdog" />
              </div>
            </FieldRow>
            <FieldRow
              label="Restart the model automatically"
              help="After a crash, lock-up or stall; at most 3 times in 15 minutes."
              className={cn(!watching && "text-muted-foreground")}
            >
              <div className="flex sm:justify-end">
                <Switch
                  checked={settings.restart_on_failure}
                  disabled={!watching}
                  onCheckedChange={(checked) => patch({ restart_on_failure: checked })}
                  aria-label="Restart the model automatically"
                />
              </div>
            </FieldRow>
            <NumberSetting
              label="Lock-up timeout"
              help="Seconds without an answer, with requests waiting on it, before a server counts as locked up (10 to 3600). An idle server that stops answering is only noted in the event log."
              value={settings.hang_timeout_s}
              min={10}
              max={3600}
              suffix="s"
              disabled={!watching}
              onSave={(value) => patch({ hang_timeout_s: value })}
            />
            <NumberSetting
              label="Stall timeout"
              help="Seconds with requests running but no tokens processed before the server counts as stalled (up to 86400); 0 turns this check off."
              value={settings.stall_timeout_s}
              min={0}
              max={86400}
              suffix="s"
              disabled={!watching}
              onSave={(value) => patch({ stall_timeout_s: value })}
            />
            <NumberSetting
              label="Stop a load when free RAM falls below"
              help="Ends a model load that would leave the computer short of memory; it would otherwise page heavily and can stop responding for minutes. 0 turns this off."
              value={settings.load_ram_floor_mib}
              min={0}
              max={262144}
              suffix="MiB"
              disabled={!watching}
              onSave={(value) => patch({ load_ram_floor_mib: value })}
            />
          </CardContent>
        </Card>

        <UpdatesCard className="xl:col-span-2" />

        <Card className="gap-1">
          <CardHeader>
            <CardTitle>
              <SectionCardTitle icon={Palette}>Appearance</SectionCardTitle>
            </CardTitle>
          </CardHeader>
          <CardContent className="divide-y">
            <FieldRow label="Theme" help="Stored in this browser.">
              <ToggleGroup
                type="single"
                value={theme}
                onValueChange={(value) => {
                  if (value === "dark" || value === "light" || value === "system") setTheme(value as Theme);
                }}
                aria-label="Theme"
              >
                <ToggleGroupItem value="dark">
                  <Moon />
                  Dark
                </ToggleGroupItem>
                <ToggleGroupItem value="light">
                  <Sun />
                  Light
                </ToggleGroupItem>
                <ToggleGroupItem value="system">
                  <Monitor />
                  System
                </ToggleGroupItem>
              </ToggleGroup>
            </FieldRow>
            <FieldRow label="Open the browser at startup" help="Open this page when the launcher starts.">
              <div className="flex sm:justify-end">
                <Switch
                  checked={settings.open_browser}
                  onCheckedChange={(checked) => patch({ open_browser: checked })}
                  aria-label="Open the browser at startup"
                />
              </div>
            </FieldRow>
          </CardContent>
        </Card>

        <Card className="gap-1">
          <CardHeader>
            <CardTitle>
              <SectionCardTitle icon={Info}>About</SectionCardTitle>
            </CardTitle>
          </CardHeader>
          <CardContent className="divide-y text-sm">
            <FieldRow label="Launcher version" help="">
              <span className="font-mono text-xs">{state?.version ?? "—"}</span>
            </FieldRow>
            {system ? (
              <>
                <FieldRow label="Computer" help={system.os}>
                  <span className="text-xs">{system.hostname}</span>
                </FieldRow>
                <FieldRow label="CPU temperature source" help="HWiNFO or Core Temp, when running, give the CPU package sensor.">
                  <span className="text-xs">{system.cpu_temp_source ? CPU_TEMP_SOURCES[system.cpu_temp_source] : "None available"}</span>
                </FieldRow>
                <FieldRow label="GPUs" help={system.driver_version ? `NVIDIA driver ${system.driver_version}` : "No NVIDIA driver found"}>
                  <span className="text-xs">
                    {system.gpus.length > 0 ? system.gpus.map((gpu) => `${gpu.name} (${formatBytes(gpu.vram_total, { binary: true })})`).join(", ") : "none"}
                  </span>
                </FieldRow>
              </>
            ) : null}
          </CardContent>
        </Card>

        <Card className="gap-1 xl:col-span-2">
          <CardHeader>
            <CardTitle>
              <SectionCardTitle icon={Save}>Memory</SectionCardTitle>
            </CardTitle>
            <CardDescription>
              Everything the launcher remembers is in this file; copy it to keep your settings when you move to a new release
              folder.
            </CardDescription>
          </CardHeader>
          <CardContent className="divide-y">
            <FieldRow
              label="Memory file"
              help="Settings, model configurations, the last model and the download queue (the API keys' hashes and the Hugging Face token too, never shown again)."
              stacked
            >
              <PathBox path={state?.memory_file ?? null} />
            </FieldRow>
            <FieldRow
              label="Launcher folder"
              help="Holds the memory file; the logs go here too unless Logging names another folder (server logs in its serve-logs subfolder)."
              stacked
            >
              <PathBox path={state?.data_dir ?? null} />
            </FieldRow>
          </CardContent>
        </Card>
      </div>

      <FolderPicker
        open={folderPicker}
        onOpenChange={setFolderPicker}
        title="Add a model folder"
        confirmLabel="Add this folder"
        onSelect={(path) => addDir.mutate(path)}
      />
      <FolderPicker
        open={downloadPicker}
        onOpenChange={setDownloadPicker}
        initialPath={settings.download_dir}
        title="Default download folder"
        confirmLabel="Use this folder"
        onSelect={(path) => patch({ download_dir: path })}
      />
      <FolderPicker
        open={logPicker}
        onOpenChange={setLogPicker}
        initialPath={settings.log_dir || logs?.dir}
        title="Log folder"
        description="The launcher's event log, request contents and server logs go here (on the computer running the launcher)."
        confirmLabel="Use this folder"
        onSelect={(path) => patch({ log_dir: path })}
      />
    </div>
  );
}
