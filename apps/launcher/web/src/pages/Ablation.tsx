import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { Link, useSearchParams } from "react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  CheckCircle2,
  CircleDashed,
  CloudDownload,
  ExternalLink,
  FileText,
  LoaderCircle,
  Play,
  RefreshCw,
  Scissors,
  Square,
  TerminalSquare,
  TriangleAlert,
  Wrench,
  XCircle,
} from "lucide-react";
import { toast } from "sonner";

import { PageHeader, SectionCardTitle } from "@/components/common";
import { CopyButton } from "@/components/CopyButton";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { api } from "@/lib/api";
import type {
  AblationJob,
  AblationPhase,
  AblationPreset,
  AblationState,
  AblationTask,
  Library,
  ModelArtifact,
  ModelConfig,
} from "@/lib/api-types";
import { setFlag } from "@/lib/args";
import { formatBytes, formatDateTime, formatEta } from "@/lib/format";
import { useLibrary, useTakesValue } from "@/lib/hooks";
import { allConfigs, findConfig } from "@/lib/library";

const PHASES: { phase: AblationPhase; label: string }[] = [
  { phase: "doctor", label: "Check" },
  { phase: "prompts", label: "Prompts" },
  { phase: "capture", label: "Record" },
  { phase: "direction", label: "Direction" },
  { phase: "fit", label: "Trials" },
  { phase: "pick", label: "Pick" },
  { phase: "grade", label: "Test" },
  { phase: "export", label: "Export" },
];

// The trial counts the page offers: the tool's default (300) and one shorter and one longer search.
const TRIALS = [
  { trials: 150, label: "Quick", help: "150 trials: a usable cut, less well tuned" },
  { trials: 300, label: "Standard", help: "300 trials: the tool's default (recommended)" },
  { trials: 600, label: "Thorough", help: "600 trials: a finer search, twice as long" },
];

const STATE_LABEL: Record<AblationJob["state"], string> = {
  waiting_gpu: "Unloading the model",
  running: "Running",
  stopping: "Stopping",
  done: "Sidecar made",
  no_sidecar: "No sidecar",
  stopped: "Stopped",
  failed: "Failed",
};

/** Folder names the launcher takes for a sidecar: letters, digits, '.', '_' and '-'. */
function sidecarName(text: string): string {
  return text
    .replace(/[^A-Za-z0-9._-]+/g, "-")
    .replace(/^[^A-Za-z0-9]+/, "")
    .slice(0, 64)
    .replace(/[.-]+$/, "");
}

function runnable(model: ModelArtifact): boolean {
  return (
    model.family !== "unknown" &&
    model.missing_files.length === 0 &&
    !model.option_support.unsupported.includes("--ablate")
  );
}


function Log({ lines, className }: { lines: string[]; className?: string }) {
  const box = useRef<HTMLPreElement>(null);
  const text = lines.join("\n");
  // Follows the newest line, unless the reader scrolled up to read an older one.
  const following = useRef(true);
  useEffect(() => {
    if (box.current && following.current) box.current.scrollTop = box.current.scrollHeight;
  }, [text]);
  if (lines.length === 0) return null;
  return (
    <pre
      ref={box}
      onScroll={(event) => {
        const el = event.currentTarget;
        following.current = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
      }}
      className={`max-h-64 overflow-auto rounded-md border bg-muted/40 p-2 font-mono text-xs leading-relaxed whitespace-pre-wrap break-all ${className ?? ""}`}
    >
      {text}
    </pre>
  );
}

function Status({ ok, children }: { ok: boolean | null; children: ReactNode }) {
  const Icon = ok === null ? LoaderCircle : ok ? CheckCircle2 : XCircle;
  return (
    <div className="flex items-start gap-2 text-sm">
      <Icon
        className={`mt-0.5 size-4 shrink-0 ${ok === null ? "animate-spin text-muted-foreground" : ok ? "text-success" : "text-critical"}`}
      />
      <div className="min-w-0">{children}</div>
    </div>
  );
}

function TaskLog({ task, kind }: { task: AblationTask | null; kind: AblationTask["kind"] }) {
  if (!task || task.kind !== kind) return null;
  return (
    <div className="space-y-1.5">
      <div className={`text-xs ${task.state === "failed" ? "text-critical" : "text-muted-foreground"}`}>
        {task.state === "running" ? <LoaderCircle className="mr-1 inline size-3 animate-spin" /> : null}
        {task.message}
      </div>
      <Log lines={task.log.slice(-40)} />
    </div>
  );
}

// ---------------------------------------------------------------------------------------------
// 1. The tool

function ToolCard({ state }: { state: AblationState }) {
  const qc = useQueryClient();
  const [path, setPath] = useState("");
  const apply = (next: AblationState) => qc.setQueryData(["ablation"], next);
  const detect = useMutation({ mutationFn: api.ablationDetectPython, onSuccess: apply });
  const choose = useMutation({ mutationFn: api.ablationPython, onSuccess: apply });
  const setup = useMutation({ mutationFn: api.ablationSetup, onSuccess: apply });
  const python = state.python.found;
  const installing = state.task?.kind === "setup" && state.task.state === "running";
  const ready = state.tool.found && python !== null && state.packages.ready;
  const running = state.job?.active ?? false;
  return (
    <Card>
      <CardHeader>
        <CardTitle>
          <SectionCardTitle icon={Wrench}>1. The tool</SectionCardTitle>
        </CardTitle>
        <CardDescription>
          The fitter (tools/ablate) runs on Python {state.python.minimum} or newer with two packages, numpy and optuna.
          {ready ? " Everything it needs is installed." : null}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <Status ok={state.tool.found}>
          {state.tool.found ? (
            <>
              Ablation tool <span className="font-mono text-xs break-all text-muted-foreground">{state.tool.dir}</span>
            </>
          ) : (
            <>
              The ablation tool was not found. A release keeps it in the <span className="font-mono">ablate</span> folder beside
              the launcher. The launcher looked in:
              <ul className="mt-1 list-disc pl-5 font-mono text-xs break-all text-muted-foreground">
                {state.tool.looked_in.map((place) => (
                  <li key={place}>{place}</li>
                ))}
              </ul>
            </>
          )}
        </Status>
        <Status ok={python !== null}>
          {python ? (
            <>
              Python {python.version} <span className="font-mono text-xs break-all text-muted-foreground">{python.executable}</span>
              {state.python.chosen ? <span className="text-xs text-muted-foreground"> (chosen on this page)</span> : null}
              {!state.python.chosen && state.python.portable?.installed && python.executable.startsWith(state.python.portable.dir) ? (
                <span className="text-xs text-muted-foreground"> (the portable Python Set up downloaded)</span>
              ) : null}
            </>
          ) : (
            state.python.portable ? (
              <>
                No Python {state.python.minimum} or newer with pip was found. Set up downloads a portable one, CPython{" "}
                {state.python.portable.version} from{" "}
                <a className="underline" href="https://github.com/astral-sh/python-build-standalone" target="_blank" rel="noreferrer">
                  python-build-standalone
                </a>{" "}
                ({formatBytes(state.python.portable.bytes)}, checked against its pinned SHA-256), into{" "}
                <span className="font-mono text-xs break-all">{state.python.portable.dir}</span>. Or give the path of a python
                executable below.
              </>
            ) : (
              <>
                No Python {state.python.minimum} or newer with pip was found. Install one from{" "}
                <a className="underline" href="https://www.python.org/downloads/" target="_blank" rel="noreferrer">
                  python.org
                </a>
                , then check again, or give the path of a python executable below.
              </>
            )
          )}
          {state.python.tried.length > 0 ? (
            <details className="mt-1 text-xs text-muted-foreground">
              <summary className="cursor-pointer">The Pythons tried</summary>
              <ul className="mt-1 space-y-0.5 font-mono break-all">
                {state.python.tried.map((tried) => (
                  <li key={tried.path}>
                    {tried.path}: {tried.result}
                  </li>
                ))}
              </ul>
            </details>
          ) : null}
        </Status>
        <div className="flex flex-wrap items-center gap-2">
          <Input
            className="min-w-0 flex-[1_1_18rem] font-mono text-xs"
            placeholder={state.python.chosen || "Another Python: the path of its python executable"}
            value={path}
            onChange={(event) => setPath(event.target.value)}
            aria-label="Python executable"
          />
          <Button variant="outline" size="sm" disabled={choose.isPending || path.trim() === ""} onClick={() => choose.mutate(path.trim())}>
            Use this Python
          </Button>
          {state.python.chosen ? (
            <Button variant="ghost" size="sm" disabled={choose.isPending} onClick={() => choose.mutate("")}>
              Find one again
            </Button>
          ) : (
            <Button variant="ghost" size="sm" disabled={detect.isPending} onClick={() => detect.mutate()}>
              <RefreshCw className={detect.isPending ? "animate-spin" : ""} /> Check again
            </Button>
          )}
        </div>
        <Status ok={installing ? null : state.packages.ready}>
          {state.packages.ready ? (
            <>
              numpy and optuna are installed <span className="font-mono text-xs break-all text-muted-foreground">{state.packages.dir}</span>
            </>
          ) : (
            <>The tool's packages are not installed for this Python. Set up downloads them from PyPI (about 40 MB).</>
          )}
        </Status>
        <div className="flex flex-wrap items-center gap-2">
          <Button
            size="sm"
            variant={state.packages.ready ? "outline" : "default"}
            disabled={
              !state.tool.found ||
              (python === null && (state.python.portable === null || state.python.chosen !== "")) ||
              installing ||
              running ||
              setup.isPending
            }
            onClick={() => setup.mutate()}
          >
            <CloudDownload /> {state.packages.ready ? "Install again" : "Set up"}
          </Button>
          {running ? <span className="text-xs text-muted-foreground">A run uses the packages.</span> : null}
        </div>
        <TaskLog task={state.task} kind="setup" />
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------------------------------------
// 2. Prompt sets

function PresetRow({ preset, state }: { preset: AblationPreset; state: AblationState }) {
  const qc = useQueryClient();
  const fetch = useMutation({
    mutationFn: () => api.ablationFetch(preset.name),
    onSuccess: (next) => qc.setQueryData(["ablation"], next),
  });
  const fetched = state.prompts[preset.name];
  const busy = state.task?.state === "running";
  const restricted = preset.sources.some((source) => source.licence.includes("NC"));
  return (
    <div className="space-y-2 rounded-lg border p-3">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div className="min-w-0 flex-[1_1_18rem]">
          <div className="text-sm font-medium">{preset.title}</div>
          <div className="text-xs text-muted-foreground">{preset.description}</div>
        </div>
        <div className="flex items-center gap-2">
          {fetched ? (
            <Badge variant="accent">
              {fetched.counts.harmful} + {fetched.counts.harmless} prompts, test {fetched.counts["test-harmful"] ?? 0} +{" "}
              {fetched.counts["test-benign"] ?? 0}
            </Badge>
          ) : null}
          <Button
            size="sm"
            variant={fetched ? "outline" : "default"}
            disabled={busy || fetch.isPending || state.python.found === null}
            onClick={() => fetch.mutate()}
          >
            <CloudDownload /> {fetched ? "Download again" : "Download"}
          </Button>
        </div>
      </div>
      <ul className="space-y-1 text-xs">
        {preset.sources.map((source) => (
          <li key={source.name} className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5">
            <a className="underline-offset-2 hover:underline" href={source.home} target="_blank" rel="noreferrer">
              {source.title}
              <ExternalLink className="ml-0.5 inline size-3" />
            </a>
            <Badge variant={source.licence.includes("NC") ? "warning" : "muted"}>{source.licence}</Badge>
            <span className="text-muted-foreground">{source.licence_note}</span>
          </li>
        ))}
      </ul>
      {restricted ? (
        <div className="flex items-start gap-1.5 text-xs text-warning">
          <TriangleAlert className="mt-0.5 size-3.5 shrink-0" />
          <span>
            Part of this set is licensed for non-commercial use only. A sidecar made from it records the licences in its
            source; check they fit how you will use it.
          </span>
        </div>
      ) : null}
      {state.task?.kind === "fetch" && state.task.preset === preset.name ? <TaskLog task={state.task} kind="fetch" /> : null}
    </div>
  );
}

function PromptsCard({ state }: { state: AblationState }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>
          <SectionCardTitle icon={FileText}>2. Prompt sets</SectionCardTitle>
        </CardTitle>
        <CardDescription>
          A fit compares the model's activations on harmful requests with those on harmless ones. Each set is downloaded from
          its publisher's pinned file, checked against its SHA-256 and converted to the tool's lists; a test pair the fit never
          sees grades the result. Licences are those of the publishers.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        {state.catalogue ? (
          state.catalogue.map((preset) => <PresetRow key={preset.name} preset={preset} state={state} />)
        ) : (
          <div className="text-sm text-muted-foreground">
            {state.catalogue_error ?? "The prompt sets are listed once the tool and a Python are found."}
          </div>
        )}
        <div className="text-xs text-muted-foreground">
          Downloaded into <span className="font-mono break-all">{state.prompts_dir}</span>
        </div>
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------------------------------------
// 3. The run

function RunCard({ state, library, preselect }: { state: AblationState; library: Library; preselect: string | null }) {
  const qc = useQueryClient();
  const configs = useMemo(() => allConfigs(library).filter(({ model }) => runnable(model)), [library]);
  const presets = Object.keys(state.prompts);
  const [configId, setConfigId] = useState<string>(() => preselect ?? "");
  const [preset, setPreset] = useState<string>("");
  const [trials, setTrials] = useState(300);
  const [name, setName] = useState("");
  const [named, setNamed] = useState(false);
  const chosen = findConfig(library, configId);
  // Defaults once their lists arrive: the page's ?model=, else the first model; the first downloaded set.
  useEffect(() => {
    if (!configId && configs.length > 0) setConfigId(configs[0].config.id);
  }, [configId, configs]);
  useEffect(() => {
    if ((!preset || !state.prompts[preset]) && presets.length > 0) setPreset(presets[0]);
  }, [preset, presets, state.prompts]);
  const suggested = chosen && preset ? sidecarName(`${chosen.config.alias}-${preset}`) : "";
  const sidecar = named ? name : suggested;
  useEffect(() => {
    if (!named) setName(suggested);
  }, [named, suggested]);
  const exists = library.sidecars.some((entry) => entry.name === sidecar);
  const active = state.job?.active ?? false;
  const start = useMutation({
    mutationFn: () => api.ablationStart({ config: configId, preset, trials, name: sidecar }),
    onSuccess: (next) => {
      qc.setQueryData(["ablation"], next);
      toast.info("The run starts: the loaded model is unloaded first");
    },
  });
  const missing: string[] = [];
  if (!state.tool.found || !state.python.found || !state.packages.ready) missing.push("set up the tool (1)");
  if (presets.length === 0) missing.push("download a prompt set (2)");
  if (!chosen) missing.push("choose a model");
  if (sidecar === "") missing.push("name the sidecar");
  return (
    <Card>
      <CardHeader>
        <CardTitle>
          <SectionCardTitle icon={Play}>3. Make a sidecar</SectionCardTitle>
        </CardTitle>
        <CardDescription>
          The tool records the model's hidden states on the prompts, computes the refusal direction, then searches for the
          layers and strengths whose cut removes the most refusals while keeping the model's answers to harmless prompts
          unchanged (first-token KL below 0.4 on held-out prompts), grades the pick on the test pair and exports it.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid gap-4 md:grid-cols-2">
          <div className="space-y-1.5">
            <div className="text-sm font-medium">Model</div>
            <Select value={configId} onValueChange={setConfigId}>
              <SelectTrigger aria-label="Model">
                <SelectValue placeholder="Choose a model" />
              </SelectTrigger>
              <SelectContent>
                {configs.map(({ model, config }) => (
                  <SelectItem key={config.id} value={config.id}>
                    {config.alias}
                    {config.is_default ? "" : ` (${config.name})`} · {model.family_label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <div className="text-xs text-muted-foreground">
              Its configuration's launch parameters and environment are used as they are (without its sidecar): a sidecar is
              fitted to the model as you run it.
            </div>
          </div>
          <div className="space-y-1.5">
            <div className="text-sm font-medium">Prompt set</div>
            <Select value={preset} onValueChange={setPreset} disabled={presets.length === 0}>
              <SelectTrigger aria-label="Prompt set">
                <SelectValue placeholder="Download one first" />
              </SelectTrigger>
              <SelectContent>
                {presets.map((key) => (
                  <SelectItem key={key} value={key}>
                    {state.prompts[key].title || key}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>
        <div className="space-y-1.5">
          <div className="text-sm font-medium">Search</div>
          <ToggleGroup
            type="single"
            value={String(trials)}
            onValueChange={(value) => {
              if (value) setTrials(Number(value));
            }}
            aria-label="Trials"
          >
            {TRIALS.map((option) => (
              <ToggleGroupItem key={option.trials} value={String(option.trials)}>
                {option.label} ({option.trials})
              </ToggleGroupItem>
            ))}
          </ToggleGroup>
          <div className="text-xs text-muted-foreground">{TRIALS.find((option) => option.trials === trials)?.help}</div>
        </div>
        <div className="space-y-1.5">
          <div className="text-sm font-medium">Sidecar name</div>
          <Input
            value={sidecar}
            onChange={(event) => {
              setNamed(true);
              setName(sidecarName(event.target.value));
            }}
            className="font-mono text-sm"
            aria-label="Sidecar name"
          />
          <div className={`text-xs ${exists ? "text-warning" : "text-muted-foreground"}`}>
            {exists
              ? "A sidecar of this name exists: choose another name (or delete it on the Models page)."
              : `Written to ${state.sidecars_dir}${state.sidecars_dir.includes("\\") ? "\\" : "/"}${sidecar || "…"}`}
          </div>
        </div>
        <Alert>
          <TriangleAlert />
          <AlertTitle>The run takes the whole GPU, for hours</AlertTitle>
          <AlertDescription>
            The loaded model is unloaded and no model loads until the run ends (requests are refused with the reason). Each
            trial starts the model twice; expect about half a minute to a minute and a half a trial on large models, so a
            standard run takes several hours. Stop keeps the finished work: starting the same model, prompt set and name
            again continues it.
          </AlertDescription>
        </Alert>
        <div className="flex flex-wrap items-center gap-3">
          <Button disabled={missing.length > 0 || exists || active || start.isPending} onClick={() => start.mutate()}>
            <Play /> Start the run
          </Button>
          {missing.length > 0 ? <span className="text-xs text-muted-foreground">First {missing.join(", ")}.</span> : null}
          {active ? <span className="text-xs text-muted-foreground">A run is in progress.</span> : null}
        </div>
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------------------------------------
// The run's progress and result

function Phases({ job }: { job: AblationJob }) {
  const progress = job.progress;
  return (
    <ol className="flex flex-wrap gap-1.5" aria-label="Phases">
      {PHASES.map(({ phase, label }) => {
        const done = progress?.phases_done.includes(phase) ?? false;
        const current = progress?.phase === phase && !done;
        const Icon = done ? CheckCircle2 : current && job.active ? LoaderCircle : CircleDashed;
        return (
          <li
            key={phase}
            className={`flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs ${
              done ? "border-success/40 text-foreground" : current ? "border-primary/50 text-foreground" : "text-muted-foreground"
            }`}
          >
            <Icon className={`size-3 ${done ? "text-success" : current && job.active ? "animate-spin text-primary" : ""}`} />
            {label}
          </li>
        );
      })}
    </ol>
  );
}

function Grade({ job }: { job: AblationJob }) {
  const grade = job.progress?.grade;
  if (!grade) return null;
  // The stock model first, as the baseline the sidecar's row is read against.
  const arms = Object.entries(grade).sort(([a], [b]) => Number(b === "stock") - Number(a === "stock"));
  const fraction = (value: { refusals: number; n: number } | null) => (value ? `${value.refusals} / ${value.n}` : "–");
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead className="text-xs text-muted-foreground">
          <tr className="text-left">
            <th className="py-1 pr-4 font-medium">On the test prompts</th>
            <th className="py-1 pr-4 font-medium">Harmful refused</th>
            <th className="py-1 pr-4 font-medium">Benign refused</th>
            <th className="py-1 font-medium">KL (where the stock model answered)</th>
          </tr>
        </thead>
        <tbody>
          {arms.map(([arm, row]) => (
            <tr key={arm} className="border-t">
              <td className="py-1 pr-4">{arm === "stock" ? "Stock model" : "With the sidecar"}</td>
              <td className="py-1 pr-4 tabular-nums">{fraction(row.harmful)}</td>
              <td className="py-1 pr-4 tabular-nums">{fraction(row.benign)}</td>
              <td className="py-1 tabular-nums">
                {row.kl === null || row.kl === undefined ? "–" : row.kl.toFixed(3)}
                {row.kl_stock_answered !== null && row.kl_stock_answered !== undefined && arm !== "stock"
                  ? ` (${row.kl_stock_answered.toFixed(3)})`
                  : ""}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function UseSidecar({ job, library }: { job: AblationJob; library: Library }) {
  const qc = useQueryClient();
  const takesValue = useTakesValue();
  const found = findConfig(library, job.run.config_id);
  const use = useMutation({
    mutationFn: ({ config }: { config: ModelConfig }) =>
      api.saveModelConfig(config.id, { args: setFlag(config.args, "--ablate", job.exported, takesValue) }),
    onSuccess: (config) => {
      void qc.invalidateQueries({ queryKey: ["library"] });
      toast.success(`${config.alias} loads with the sidecar from now on (Load it on the Models page)`);
    },
  });
  if (!job.exported) return null;
  return (
    <div className="space-y-2 rounded-lg border border-success/40 p-3">
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <CheckCircle2 className="size-4 text-success" /> The sidecar is ready
        <span className="font-mono text-xs break-all text-muted-foreground">{job.exported}</span>
        <CopyButton text={job.exported} label="Copy the path" />
      </div>
      <div className="flex flex-wrap gap-2">
        {found ? (
          <Button size="sm" disabled={use.isPending} onClick={() => use.mutate({ config: found.config })}>
            <Scissors /> Use it with {found.config.alias}
          </Button>
        ) : null}
        {found ? (
          <Button size="sm" variant="outline" asChild>
            <Link to={`/models/${encodeURIComponent(found.config.id)}`}>Open its configuration</Link>
          </Button>
        ) : null}
      </div>
      <div className="text-xs text-muted-foreground">
        It is listed on the Models page with the other sidecars, and in a configuration's Ablation sidecar field.
      </div>
    </div>
  );
}

function JobCard({ state, library }: { state: AblationState; library: Library }) {
  const qc = useQueryClient();
  const job = state.job;
  const apply = (next: AblationState) => qc.setQueryData(["ablation"], next);
  const stop = useMutation({ mutationFn: (now: boolean) => api.ablationStop(now), onSuccess: apply });
  const again = useMutation({
    mutationFn: () =>
      job
        ? api.ablationStart({ config: job.run.config_id, preset: job.run.preset, trials: job.run.trials, name: job.run.name })
        : Promise.reject(new Error("no run")),
    onSuccess: apply,
  });
  if (!job) return null;
  const progress = job.progress;
  const trials = progress?.trials;
  const percent = trials && trials.target > 0 ? (100 * trials.done) / trials.target : null;
  const eta = trials?.seconds && trials.done < trials.target ? trials.seconds * (trials.target - trials.done) : null;
  const tone = job.state === "failed" ? "critical" : job.state === "done" ? "success" : job.state === "no_sidecar" ? "warning" : "primary";
  const resumable = !job.active && (job.state === "stopped" || job.state === "failed");
  return (
    <Card>
      <CardHeader>
        <CardTitle>
          <SectionCardTitle icon={TerminalSquare}>
            {job.active ? "The run" : "The last run"}
            <Badge variant={job.state === "failed" ? "critical" : job.state === "no_sidecar" ? "warning" : job.active ? "accent" : "muted"}>
              {STATE_LABEL[job.state]}
            </Badge>
          </SectionCardTitle>
        </CardTitle>
        <CardDescription>
          {job.run.model_name} · {state.prompts[job.run.preset]?.title ?? job.run.preset} · {job.run.trials} trials · sidecar{" "}
          <span className="font-mono">{job.run.name}</span> · started {formatDateTime(job.started)}
          {job.ended ? `, ended ${formatDateTime(job.ended)}` : ""}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <Phases job={job} />
        <div className={`text-sm ${job.state === "failed" ? "text-critical" : ""} whitespace-pre-wrap`}>{job.message}</div>
        {trials && (progress?.phase === "fit" || (progress?.phases_done.includes("fit") ?? false)) ? (
          <div className="space-y-1">
            <Progress value={percent} tone={tone} />
            <div className="flex flex-wrap justify-between gap-2 text-xs text-muted-foreground">
              <span>
                {trials.done} of {trials.target} trials
                {trials.seconds ? ` · ${Math.round(trials.seconds)} s a trial` : ""}
              </span>
              {eta && job.active ? <span>about {formatEta(eta)} left</span> : null}
            </div>
          </div>
        ) : null}
        <div className="grid gap-2 text-sm sm:grid-cols-2">
          {progress?.control ? (
            <div>
              <div className="text-xs text-muted-foreground">Stock model, held-out harmful prompts</div>
              {progress.control.refusals} of {progress.control.n} refused
            </div>
          ) : null}
          {progress?.best ? (
            <div>
              <div className="text-xs text-muted-foreground">Best trial so far (KL below 0.4)</div>
              trial {progress.best.trial}: {progress.best.refusals} refused, KL {progress.best.kl.toFixed(3)}
            </div>
          ) : null}
        </div>
        <Grade job={job} />
        {job.state === "done" ? <UseSidecar job={job} library={library} /> : null}
        {job.state === "no_sidecar" ? (
          <div className="text-xs text-muted-foreground">
            No cut met the rule. More trials (Thorough), another prompt set, or a closer look with the tool's own commands
            (docs/ablate-fitting.md) may find one.
          </div>
        ) : null}
        <div className="flex flex-wrap gap-2">
          {job.active && job.state !== "stopping" ? (
            <Button variant="outline" size="sm" disabled={stop.isPending} onClick={() => stop.mutate(false)}>
              <Square /> Stop
            </Button>
          ) : null}
          {job.state === "stopping" ? (
            <Button variant="destructive" size="sm" disabled={stop.isPending} onClick={() => stop.mutate(true)}>
              <XCircle /> End now
            </Button>
          ) : null}
          {resumable ? (
            <Button size="sm" disabled={again.isPending} onClick={() => again.mutate()}>
              <Play /> Run again (continues it)
            </Button>
          ) : null}
        </div>
        {job.state === "stopping" ? (
          <div className="text-xs text-muted-foreground">
            The tool stops its engines and keeps its finished trials. End now ends it at once, losing the trial under way.
          </div>
        ) : null}
        {job.log.length > 0 ? (
          <details open={job.active || job.state === "failed"}>
            <summary className="cursor-pointer text-xs text-muted-foreground">The tool's output</summary>
            <Log lines={job.log.slice(-120)} className="mt-1.5" />
          </details>
        ) : null}
        <div className="text-xs text-muted-foreground">
          Run folder <span className="font-mono break-all">{job.run_dir}</span>
        </div>
      </CardContent>
    </Card>
  );
}

export function AblationPage() {
  const [params] = useSearchParams();
  const { data: library } = useLibrary();
  const query = useQuery({
    queryKey: ["ablation"],
    queryFn: api.ablation,
    // Read again on every visit (files outside the launcher's own changes: a Python installed meanwhile); the
    // launcher pushes its own changes, and a slow poll covers a missed event while a run goes on.
    refetchOnMount: "always",
    refetchInterval: (q) => (q.state.data?.job?.active || q.state.data?.task?.state === "running" ? 5000 : false),
  });
  const state = query.data;
  return (
    <div className="space-y-6">
      <PageHeader
        title="Ablation"
        description="Make a refusal-ablation sidecar for one of your models on this computer: a measured cut of the direction its activations use to refuse, which the model loads with (Ablation sidecar in its configuration, --ablate)."
      />
      {state?.job?.active ? (
        <Alert>
          <LoaderCircle className="animate-spin" />
          <AlertTitle>A run has the GPU</AlertTitle>
          <AlertDescription>No model loads until it ends or is stopped.</AlertDescription>
        </Alert>
      ) : null}
      {query.isError ? (
        <Alert variant="destructive">
          <TriangleAlert />
          <AlertTitle>The ablation state did not load</AlertTitle>
          <AlertDescription>{query.error instanceof Error ? query.error.message : String(query.error)}</AlertDescription>
        </Alert>
      ) : null}
      {!state || !library ? (
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <LoaderCircle className="size-4 animate-spin" /> Looking for the tool and a Python…
        </div>
      ) : (
        <>
          {state.job ? <JobCard state={state} library={library} /> : null}
          <ToolCard state={state} />
          <PromptsCard state={state} />
          <RunCard state={state} library={library} preselect={params.get("model")} />
        </>
      )}
    </div>
  );
}
