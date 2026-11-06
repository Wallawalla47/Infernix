// Settings: updates of Infernix itself (the latest GitHub release: download, then install and
// restart) and of the models downloaded through the launcher (Hugging Face repositories whose files
// changed). Both states also arrive on the event stream (infernix_release, model_updates).

import { useState, type ReactNode } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  ChevronDown,
  CircleCheck,
  CloudDownload,
  Download,
  ExternalLink,
  PackageCheck,
  RefreshCw,
  RotateCw,
  TriangleAlert,
} from "lucide-react";
import { toast } from "sonner";

import { ConfirmDialog, SectionCardTitle } from "@/components/common";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { Hint } from "@/components/ui/tooltip";
import { INFERNIX_RELEASE_KEY, MODEL_UPDATES_KEY, useInfernixRelease, useModelUpdates } from "@/lib/admin";
import { api, errorMessage } from "@/lib/api";
import type { InfernixRelease, ModelUpdate, ModelUpdates } from "@/lib/api-types";
import { formatBytes, formatRelative } from "@/lib/format";
import { useNow } from "@/lib/hooks";
import { cn } from "@/lib/utils";

function releaseDate(published: string): string {
  const date = new Date(published);
  return Number.isNaN(date.getTime())
    ? published
    : date.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
}

function shortRevision(revision: string): string {
  return /^[0-9a-f]{40}$/i.test(revision) ? revision.slice(0, 7) : revision;
}

function Collapsible({ label, children }: { label: string; children: ReactNode }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="w-full">
      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOpen((value) => !value)}
        className="inline-flex items-center gap-1 text-xs font-medium text-foreground hover:underline"
      >
        <ChevronDown className={cn("size-3.5 transition-transform", !open && "-rotate-90")} />
        {label}
      </button>
      {open ? <div className="mt-1.5">{children}</div> : null}
    </div>
  );
}

function InfernixSection({ release }: { release: InfernixRelease }) {
  const queryClient = useQueryClient();
  const now = useNow(30_000);
  const [confirmInstall, setConfirmInstall] = useState(false);
  const latest = release.latest;
  const store = (next: InfernixRelease) => queryClient.setQueryData(INFERNIX_RELEASE_KEY, next);

  const download = useMutation({ mutationFn: api.downloadInfernixUpdate, onSuccess: store });
  const install = useMutation({
    mutationFn: api.installInfernixUpdate,
    onSuccess: (next) => {
      store(next);
      toast.info("Installing: the launcher restarts, and this page reconnects when it is back");
    },
  });

  const progress = release.progress;
  const percent = progress && progress.total > 0 ? (progress.done / progress.total) * 100 : null;

  return (
    <section className="flex flex-col gap-3" aria-labelledby="updates-infernix">
      <div className="flex flex-wrap items-center gap-2">
        <h3 id="updates-infernix" className="text-sm font-medium">
          Infernix
        </h3>
        {release.newer && latest ? (
          <Badge variant="accent">update available</Badge>
        ) : latest && !release.error ? (
          <Badge variant="outline">
            <CircleCheck className="text-success" />
            up to date
          </Badge>
        ) : null}
        <span className="ml-auto text-xs text-muted-foreground">
          {release.state === "checking"
            ? "Checking GitHub…"
            : release.checked_at !== null
              ? `Checked ${formatRelative(release.checked_at, now)}`
              : "Not checked yet"}
        </span>
        <Hint label="Read the launcher's latest check again">
          <Button
            variant="ghost"
            size="icon-xs"
            aria-label="Refresh"
            onClick={() => void queryClient.invalidateQueries({ queryKey: INFERNIX_RELEASE_KEY })}
          >
            <RefreshCw />
          </Button>
        </Hint>
      </div>

      <dl className="grid gap-x-6 gap-y-2 text-sm sm:grid-cols-2">
        <div className="min-w-0">
          <dt className="text-[11px] font-medium tracking-wide text-muted-foreground uppercase">This build</dt>
          <dd className="mt-0.5 truncate font-mono text-xs" title={release.current}>
            {release.current_version ?? release.current}
            {release.current_version ? <span className="ml-2 text-muted-foreground">{release.current}</span> : null}
          </dd>
        </div>
        <div className="min-w-0">
          <dt className="text-[11px] font-medium tracking-wide text-muted-foreground uppercase">Latest release</dt>
          <dd className="mt-0.5 flex min-w-0 flex-wrap items-center gap-x-2 text-xs">
            {latest ? (
              <>
                <span className="font-mono">{latest.version}</span>
                <span className="text-muted-foreground">{releaseDate(latest.published_at)}</span>
                {latest.asset ? <span className="text-muted-foreground">{formatBytes(latest.asset.bytes)}</span> : null}
                <a
                  href={latest.url}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center gap-1 text-primary-text underline-offset-4 hover:underline"
                >
                  Release page
                  <ExternalLink className="size-3" />
                </a>
              </>
            ) : (
              <span className="text-muted-foreground">{release.state === "checking" ? "Checking…" : "Unknown"}</span>
            )}
          </dd>
        </div>
      </dl>

      {latest && latest.notes.trim().length > 0 ? (
        <Collapsible label={`What's new in ${latest.version}`}>
          <pre className="max-h-72 overflow-y-auto rounded-lg border bg-muted/40 p-3 font-sans text-xs leading-relaxed whitespace-pre-wrap">
            {latest.notes.trim()}
          </pre>
        </Collapsible>
      ) : null}

      {release.state === "downloading" ? (
        <div className="flex flex-col gap-1.5">
          <Progress value={percent} />
          <div className="flex justify-between gap-2 text-xs text-muted-foreground tabular-nums">
            <span>Downloading {latest?.asset?.name ?? "the update"}</span>
            <span>{progress ? `${formatBytes(progress.done)} of ${formatBytes(progress.total)}` : ""}</span>
          </div>
        </div>
      ) : null}
      {release.state === "installing" ? (
        <div className="flex flex-col gap-1.5">
          <Progress value={null} />
          <span className="text-xs text-muted-foreground">
            Installing: the model stops, the program files are replaced and the launcher restarts. This page reconnects
            when it is back.
          </span>
        </div>
      ) : null}

      {release.error ? (
        <div className="flex items-start gap-1.5 text-xs text-critical">
          <TriangleAlert className="mt-0.5 size-3.5 shrink-0" />
          <span className="break-words">{release.error}</span>
        </div>
      ) : null}

      {release.newer && latest ? (
        release.blocked ? (
          <div className="flex items-start gap-1.5 rounded-lg border px-3 py-2 text-xs text-muted-foreground">
            <TriangleAlert className="mt-0.5 size-3.5 shrink-0 text-warning" />
            <span>
              It cannot be installed from here: {release.blocked}. Download it from the release page instead.
            </span>
          </div>
        ) : release.state === "ready" ? (
          <div className="flex flex-wrap items-center gap-2">
            <Button size="sm" disabled={install.isPending} onClick={() => setConfirmInstall(true)}>
              <PackageCheck />
              Install and restart
            </Button>
            <span className="text-xs text-muted-foreground">Downloaded and checked; ready to install.</span>
          </div>
        ) : release.state === "idle" || release.state === "failed" || release.state === "checking" ? (
          <div className="flex flex-wrap items-center gap-2">
            <Button
              size="sm"
              variant="outline"
              disabled={release.state === "checking" || download.isPending}
              onClick={() => download.mutate()}
            >
              <Download />
              {release.state === "failed" ? "Download again" : "Download update"}
              {latest.asset ? <span className="text-muted-foreground">{formatBytes(latest.asset.bytes)}</span> : null}
            </Button>
          </div>
        ) : null
      ) : null}

      <ConfirmDialog
        open={confirmInstall}
        onOpenChange={setConfirmInstall}
        title={`Install Infernix ${latest?.version ?? ""} and restart?`}
        description={
          <span className="flex flex-col gap-2">
            <span>
              The launcher stops the loaded model in order (running requests are ended), replaces its program files and
              restarts. Your settings, model configurations, logs and chats stay.
            </span>
            <span>This page reconnects when the launcher is back.</span>
          </span>
        }
        confirmLabel="Install and restart"
        onConfirm={() => install.mutate()}
      />
    </section>
  );
}

function ModelUpdateRow({ update }: { update: ModelUpdate }) {
  const queryClient = useQueryClient();
  const changed = update.files.filter((file) => file.change === "changed").length;
  const added = update.files.length - changed;
  const apply = useMutation({
    mutationFn: () => api.applyModelUpdate(update.job_id),
    onSuccess: () => {
      toast.success(`Downloading the changed files of ${update.repo}`);
      void queryClient.invalidateQueries({ queryKey: ["downloads"] });
      void queryClient.invalidateQueries({ queryKey: MODEL_UPDATES_KEY });
    },
  });
  const counts = [changed > 0 ? `${changed} changed` : null, added > 0 ? `${added} new` : null].filter(Boolean).join(", ");

  return (
    <li className="flex flex-col gap-2 px-3 py-2.5">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
        <div className="min-w-0 flex-1">
          <a
            href={`https://huggingface.co/${update.repo}`}
            target="_blank"
            rel="noreferrer"
            className="inline-flex max-w-full items-center gap-1 truncate text-sm font-medium underline-offset-4 hover:underline"
          >
            {update.repo}
            <ExternalLink className="size-3 shrink-0 text-muted-foreground" />
          </a>
          <div className="truncate text-xs text-muted-foreground">
            {counts || "files changed"} · {formatBytes(update.bytes)} ·{" "}
            <span className="font-mono">
              {shortRevision(update.revision)} → {shortRevision(update.latest_revision)}
            </span>
          </div>
          <div className="truncate font-mono text-[11px] text-muted-foreground/80" title={update.target_dir}>
            {update.target_dir}
          </div>
        </div>
        <Button size="sm" variant="outline" className="self-start sm:self-auto" disabled={apply.isPending} onClick={() => apply.mutate()}>
          <CloudDownload />
          Update
        </Button>
      </div>
      {update.files.length > 0 ? (
        <Collapsible label={update.files.length === 1 ? "1 file" : `${update.files.length} files`}>
          <ul className="flex flex-col gap-0.5 rounded-lg border bg-muted/30 px-3 py-2 text-xs">
            {update.files.map((file) => (
              <li key={file.path} className="flex items-center gap-2">
                <Badge variant={file.change === "new" ? "accent" : "muted"}>{file.change}</Badge>
                <span className="min-w-0 flex-1 truncate font-mono" title={file.path}>
                  {file.path}
                </span>
                <span className="shrink-0 text-muted-foreground tabular-nums">{formatBytes(file.bytes)}</span>
              </li>
            ))}
          </ul>
        </Collapsible>
      ) : null}
    </li>
  );
}

function ModelsSection({ updates }: { updates: ModelUpdates }) {
  const queryClient = useQueryClient();
  const now = useNow(30_000);
  const check = useMutation({
    mutationFn: api.checkModelUpdates,
    onSuccess: (next) => queryClient.setQueryData(MODEL_UPDATES_KEY, next),
  });
  const checking = updates.checking || check.isPending;

  return (
    <section className="flex flex-col gap-3" aria-labelledby="updates-models">
      <div className="flex flex-wrap items-center gap-2">
        <h3 id="updates-models" className="text-sm font-medium">
          Models
        </h3>
        {updates.updates.length > 0 ? (
          <Badge variant="accent">
            {updates.updates.length} update{updates.updates.length === 1 ? "" : "s"}
          </Badge>
        ) : null}
        <span className="ml-auto text-xs text-muted-foreground">
          {checking
            ? "Checking Hugging Face…"
            : updates.checked_at !== null
              ? `Checked ${formatRelative(updates.checked_at, now)}; again every 6 hours`
              : "Not checked yet"}
        </span>
        <Button variant="outline" size="xs" disabled={checking} onClick={() => check.mutate()}>
          <RotateCw className={checking ? "animate-spin" : undefined} />
          Check now
        </Button>
      </div>
      {updates.error ? (
        <div className="flex items-start gap-1.5 text-xs text-critical">
          <TriangleAlert className="mt-0.5 size-3.5 shrink-0" />
          <span className="break-words">{updates.error}</span>
        </div>
      ) : null}
      {updates.updates.length === 0 ? (
        <div className="rounded-lg border border-dashed px-3 py-4 text-center text-xs text-muted-foreground">
          {updates.checked_at !== null
            ? "Every model downloaded through the launcher matches its Hugging Face repository."
            : "Models downloaded through the launcher are compared with their Hugging Face repositories."}
        </div>
      ) : (
        <>
          <ul className="divide-y rounded-lg border">
            {updates.updates.map((update) => (
              <ModelUpdateRow key={update.job_id} update={update} />
            ))}
          </ul>
          <div className="text-xs text-muted-foreground">
            Update downloads only the changed and new files into the same folder; it is refused while that model is
            loaded.
          </div>
        </>
      )}
    </section>
  );
}

export function UpdatesCard({ className }: { className?: string }) {
  const release = useInfernixRelease();
  const updates = useModelUpdates();
  return (
    <Card className={cn("gap-2", className)}>
      <CardHeader>
        <CardTitle>
          <SectionCardTitle icon={PackageCheck}>Updates</SectionCardTitle>
        </CardTitle>
        <CardDescription>New Infernix releases and changed model files</CardDescription>
      </CardHeader>
      <CardContent className="grid gap-6 lg:grid-cols-2">
        {release.data ? (
          <InfernixSection release={release.data} />
        ) : release.isError ? (
          <div className="text-xs text-critical">Infernix release: {errorMessage(release.error)}</div>
        ) : (
          <Skeleton className="h-28 rounded-lg" />
        )}
        {updates.data ? (
          <ModelsSection updates={updates.data} />
        ) : updates.isError ? (
          <div className="text-xs text-critical">Model updates: {errorMessage(updates.error)}</div>
        ) : (
          <Skeleton className="h-28 rounded-lg" />
        )}
      </CardContent>
    </Card>
  );
}
