import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import {
  ChevronDown,
  ChevronRight,
  CircleCheck,
  CircleX,
  Download,
  FolderOpen,
  LoaderCircle,
  Pause,
  Play,
  Trash2,
  X,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Hint } from "@/components/ui/tooltip";
import { api } from "@/lib/api";
import type { DownloadFile, DownloadJob, DownloadState } from "@/lib/api-types";
import { formatByteRate, formatBytes, formatEta, formatRelative } from "@/lib/format";
import { cn } from "@/lib/utils";

const ACTIVE: ReadonlySet<DownloadState> = new Set(["queued", "running", "verifying"]);

function JobState({ state }: { state: DownloadState }) {
  switch (state) {
    case "running":
      return (
        <Badge variant="accent">
          <LoaderCircle className="animate-spin" />
          downloading
        </Badge>
      );
    case "verifying":
      return (
        <Badge variant="accent">
          <LoaderCircle className="animate-spin" />
          verifying
        </Badge>
      );
    case "done":
      return (
        <Badge variant="outline">
          <CircleCheck className="text-success" />
          done
        </Badge>
      );
    case "failed":
      return (
        <Badge variant="destructive">
          <CircleX />
          failed
        </Badge>
      );
    default:
      return <Badge variant="muted">{state}</Badge>;
  }
}

function FileRow({ file }: { file: DownloadFile }) {
  const ratio = file.bytes > 0 ? (file.done / file.bytes) * 100 : file.state === "done" ? 100 : 0;
  return (
    <li className="flex flex-col gap-1 py-2">
      <div className="flex items-center gap-2 text-xs">
        <span className="min-w-0 flex-1 truncate font-mono">{file.path}</span>
        <span className="text-muted-foreground tabular-nums">
          {file.state === "done" || file.state === "skipped" ? formatBytes(file.bytes) : `${formatBytes(file.done)} / ${formatBytes(file.bytes)}`}
        </span>
        <Badge variant={file.state === "failed" ? "destructive" : "muted"}>{file.state}</Badge>
      </div>
      {file.state === "downloading" || file.state === "verifying" ? (
        <Progress value={file.state === "verifying" ? null : ratio} className="h-1" />
      ) : null}
      {file.error ? <div className="text-xs text-critical">{file.error}</div> : null}
    </li>
  );
}

function JobCard({ job }: { job: DownloadJob }) {
  const [open, setOpen] = useState(false);
  const pause = useMutation({ mutationFn: () => api.pauseDownload(job.id) });
  const resume = useMutation({ mutationFn: () => api.resumeDownload(job.id) });
  const cancel = useMutation({ mutationFn: () => api.cancelDownload(job.id) });
  const remove = useMutation({ mutationFn: () => api.removeDownload(job.id) });

  const ratio = job.total > 0 ? (job.done / job.total) * 100 : 0;
  const active = ACTIVE.has(job.state);
  const tone = job.state === "failed" ? "critical" : job.state === "done" ? "success" : job.state === "paused" || job.state === "cancelled" ? "muted" : "primary";
  const filesDone = job.files.filter((file) => file.state === "done" || file.state === "skipped").length;

  return (
    <li className="flex flex-col gap-2.5 px-4 py-3">
      <div className="flex flex-wrap items-center gap-2">
        <button
          type="button"
          className="flex min-w-0 flex-1 items-center gap-1.5 text-left"
          onClick={() => setOpen((value) => !value)}
          aria-expanded={open}
        >
          {open ? <ChevronDown className="size-4 shrink-0 text-muted-foreground" /> : <ChevronRight className="size-4 shrink-0 text-muted-foreground" />}
          <span className="truncate text-sm font-medium">{job.repo}</span>
        </button>
        <JobState state={job.state} />
        <div className="flex items-center gap-1">
          {job.state === "running" || job.state === "queued" ? (
            <Hint label="Pause">
              <Button size="icon-xs" variant="ghost" aria-label="Pause" disabled={pause.isPending} onClick={() => pause.mutate()}>
                <Pause />
              </Button>
            </Hint>
          ) : null}
          {job.state === "paused" || job.state === "failed" ? (
            <Hint label="Resume">
              <Button size="icon-xs" variant="ghost" aria-label="Resume" disabled={resume.isPending} onClick={() => resume.mutate()}>
                <Play />
              </Button>
            </Hint>
          ) : null}
          {active || job.state === "paused" ? (
            <Hint label="Cancel (keeps finished files)">
              <Button size="icon-xs" variant="ghost" aria-label="Cancel" disabled={cancel.isPending} onClick={() => cancel.mutate()}>
                <X />
              </Button>
            </Hint>
          ) : null}
          {job.state === "done" || job.state === "failed" || job.state === "cancelled" ? (
            <Hint label="Remove from the list">
              <Button size="icon-xs" variant="ghost" aria-label="Remove" disabled={remove.isPending} onClick={() => remove.mutate()}>
                <Trash2 />
              </Button>
            </Hint>
          ) : null}
        </div>
      </div>
      <Progress value={job.state === "verifying" ? null : ratio} tone={tone} />
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted-foreground tabular-nums">
        <span>
          <span className="font-medium text-foreground">{formatBytes(job.done)}</span> of {formatBytes(job.total)}
        </span>
        {job.state === "running" ? <span>{formatByteRate(job.speed_bps)}</span> : null}
        {job.state === "running" && job.eta_s !== null ? <span>{formatEta(job.eta_s)} left</span> : null}
        <span>
          {filesDone} of {job.files.length} file{job.files.length === 1 ? "" : "s"}
        </span>
        <span className="flex min-w-0 items-center gap-1">
          <FolderOpen className="size-3.5 shrink-0" />
          <span className="truncate">{job.target_dir}</span>
        </span>
        <span className="ml-auto">{job.finished_at ? `finished ${formatRelative(job.finished_at)}` : `started ${formatRelative(job.created_at)}`}</span>
      </div>
      {job.error ? <div className="text-xs text-critical">{job.error}</div> : null}
      {open ? (
        <ul className={cn("divide-y rounded-lg border px-3")}>
          {job.files.map((file) => (
            <FileRow key={file.path} file={file} />
          ))}
        </ul>
      ) : null}
    </li>
  );
}

export function DownloadsPanel() {
  const { data: jobs } = useQuery<DownloadJob[]>({ queryKey: ["downloads"], queryFn: api.downloads, staleTime: 5_000 });
  if (!jobs || jobs.length === 0) return null;
  const active = jobs.filter((job) => ACTIVE.has(job.state)).length;
  const ordered = [...jobs].sort((a, b) => {
    const rank = (job: DownloadJob) => (ACTIVE.has(job.state) ? 0 : job.state === "paused" || job.state === "failed" ? 1 : 2);
    return rank(a) - rank(b) || b.created_at - a.created_at;
  });
  return (
    <Card className="gap-2">
      <CardHeader>
        <CardTitle>
          <Download className="size-4 text-muted-foreground" />
          Downloads
        </CardTitle>
        <CardDescription>
          {active > 0 ? `${active} in progress` : "Nothing in progress"}; finished folders are added to the model folders and scanned.
        </CardDescription>
      </CardHeader>
      <CardContent className="px-0">
        <ul className="divide-y border-t">
          {ordered.map((job) => (
            <JobCard key={job.id} job={job} />
          ))}
        </ul>
      </CardContent>
    </Card>
  );
}
