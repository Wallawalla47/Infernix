import { useEffect, useMemo, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { ExternalLink, FolderOpen, HardDriveDownload, LoaderCircle, TriangleAlert } from "lucide-react";
import { toast } from "sonner";

import { FolderPicker } from "@/components/FolderPicker";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { api, errorMessage } from "@/lib/api";
import type { CatalogEntry, DownloadRequest, Library, RepoFile, RepoFileRole, RepoListing } from "@/lib/api-types";
import { NGRAM_FORMAT_LABELS, formatBytes } from "@/lib/format";
import { useSettings } from "@/lib/hooks";
import { baseName, cn, joinPath, sum } from "@/lib/utils";

const ROLE_LABEL: Record<RepoFileRole, string> = {
  model: "model",
  "model-part": "model part",
  drafter: "drafter",
  ngram: "n-gram volume",
  "expert-state": "expert state",
  report: "report",
  other: "other",
};

export function sameRepo(a: string, b: string): boolean {
  return a.toLowerCase() === b.toLowerCase();
}

export function repoName(repo: string): string {
  const parts = repo.split("/");
  return parts[parts.length - 1] || repo;
}

/** Whether the library already holds this catalog entry's artifact (matched by name). */
export function inLibrary(entry: CatalogEntry, library: Library | undefined): boolean {
  if (!library) return false;
  const name = repoName(entry.repo).toLowerCase();
  const match = (fileName: string, artifactName: string) =>
    fileName.toLowerCase().startsWith(name) || artifactName.toLowerCase() === name;
  if (entry.kind === "drafter") return library.drafters.some((drafter) => match(drafter.file_name, drafter.name));
  return library.models.some((model) => match(model.file_name, model.name));
}

function hasDrafterFor(entry: CatalogEntry, library: Library | undefined): boolean {
  if (!library || !entry.drafter_repo) return false;
  const name = repoName(entry.drafter_repo).toLowerCase();
  return library.drafters.some(
    (drafter) =>
      drafter.file_name.toLowerCase().startsWith(name) ||
      (entry.family === "qwen3.8-flash-next" && drafter.target_architecture === "Qwen4ExpForCausalLM"),
  );
}

function hasNgramVolume(entry: CatalogEntry, library: Library | undefined): boolean {
  if (!library) return false;
  const files = new Set(entry.ngram_options.map((option) => option.file.toLowerCase()));
  return library.ngram_volumes.some(
    (volume) => files.has(volume.file_name.toLowerCase()) || entry.family === "qwen3.8-flash-next",
  );
}

const NONE = "none";

function optionKey(repo: string, file: string): string {
  return `${repo.toLowerCase()}|${file}`;
}

export function DownloadDialog({
  open,
  onOpenChange,
  repo,
  entry,
  library,
  onQueued,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  repo: string;
  entry: CatalogEntry | null;
  library: Library | undefined;
  onQueued: () => void;
}) {
  const queryClient = useQueryClient();
  const { data: settings } = useSettings();
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [ngram, setNgram] = useState<string>(NONE);
  const [withDrafter, setWithDrafter] = useState(false);
  const [dest, setDest] = useState("");
  const [subfolder, setSubfolder] = useState(true);
  const [picker, setPicker] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  const listing = useQuery<RepoListing>({
    queryKey: ["repo", repo.toLowerCase()],
    queryFn: () => api.repoFiles(repo),
    enabled: open && repo.length > 0,
    staleTime: 5 * 60_000,
    retry: 0,
  });

  const drafterRepo = entry?.drafter_repo ?? null;
  const drafterListing = useQuery<RepoListing>({
    queryKey: ["repo", (drafterRepo ?? "").toLowerCase()],
    queryFn: () => api.repoFiles(drafterRepo ?? ""),
    enabled: open && withDrafter && drafterRepo !== null,
    staleTime: 5 * 60_000,
    retry: 0,
  });

  const ngramOptions = entry?.ngram_options ?? [];
  const chooser = ngramOptions.length > 0;

  // Defaults whenever the dialog opens for a repository.
  useEffect(() => {
    if (!open) return;
    setDest(settings?.download_dir ?? "");
    setSubfolder(true);
    if (entry) {
      setWithDrafter(entry.drafter_repo !== null && !hasDrafterFor(entry, library));
      const recommended = entry.ngram_options.find((option) => option.recommended) ?? entry.ngram_options[0];
      setNgram(recommended && !hasNgramVolume(entry, library) ? optionKey(recommended.repo, recommended.file) : NONE);
    } else {
      setWithDrafter(false);
      setNgram(NONE);
    }
    // Reset only when the dialog opens or the repository changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, repo]);

  useEffect(() => {
    if (!listing.data) return;
    setSelected(
      new Set(
        listing.data.files
          .filter((file) => file.default_selected && file.present_locally === null && !(chooser && file.role === "ngram"))
          .map((file) => file.path),
      ),
    );
  }, [listing.data, chooser]);

  const free = useQuery({
    queryKey: ["fs", dest],
    queryFn: () => api.listDir(dest),
    enabled: open && dest.length > 0,
    staleTime: 10_000,
    retry: 0,
  });

  const files: RepoFile[] = useMemo(
    () => (listing.data?.files ?? []).filter((file) => !(chooser && file.role === "ngram")),
    [listing.data, chooser],
  );

  const ngramChoice = ngramOptions.find((option) => optionKey(option.repo, option.file) === ngram) ?? null;
  const drafterFiles = (drafterListing.data?.files ?? []).filter((file) => file.default_selected);

  const mainBytes = sum(files.filter((file) => selected.has(file.path)).map((file) => file.bytes));
  const ngramBytes = ngramChoice
    ? (listing.data?.files.find((file) => sameRepo(ngramChoice.repo, repo) && file.path === ngramChoice.file)?.bytes ??
      ngramChoice.bytes)
    : 0;
  const drafterBytes = withDrafter ? sum(drafterFiles.map((file) => file.bytes)) : 0;
  const total = mainBytes + ngramBytes + drafterBytes;
  const freeBytes = free.data?.free_bytes ?? null;
  const tooBig = freeBytes !== null && total > freeBytes;

  const requests = (): DownloadRequest[] => {
    const out: DownloadRequest[] = [];
    const main = files.filter((file) => selected.has(file.path)).map((file) => file.path);
    if (ngramChoice && sameRepo(ngramChoice.repo, repo)) main.push(ngramChoice.file);
    if (main.length > 0) out.push({ repo, files: main, dest_dir: dest, subfolder });
    if (ngramChoice && !sameRepo(ngramChoice.repo, repo)) {
      out.push({ repo: ngramChoice.repo, files: [ngramChoice.file], dest_dir: dest, subfolder });
    }
    if (withDrafter && drafterRepo && drafterFiles.length > 0) {
      out.push({ repo: drafterRepo, files: drafterFiles.map((file) => file.path), dest_dir: dest, subfolder });
    }
    return out;
  };

  const submit = async () => {
    const list = requests();
    if (list.length === 0 || dest.length === 0) return;
    setSubmitting(true);
    try {
      for (const request of list) {
        await api.startDownload(request);
      }
      toast.success(list.length === 1 ? `Downloading ${repoName(list[0].repo)}` : `Queued ${list.length} downloads`);
      await queryClient.invalidateQueries({ queryKey: ["downloads"] });
      onQueued();
      onOpenChange(false);
    } catch (error) {
      toast.error(errorMessage(error));
    } finally {
      setSubmitting(false);
    }
  };

  const toggle = (path: string, checked: boolean) => {
    setSelected((old) => {
      const next = new Set(old);
      if (checked) next.add(path);
      else next.delete(path);
      return next;
    });
  };

  const target = dest.length > 0 ? (subfolder ? joinPath(dest, repoName(repo)) : dest) : "";

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="gap-4 sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle>Download {entry?.title ?? repoName(repo)}</DialogTitle>
          <DialogDescription asChild>
            <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
              <a
                href={`https://huggingface.co/${repo}`}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-1 font-mono text-xs hover:text-foreground"
              >
                {repo}
                <ExternalLink className="size-3" />
              </a>
              {listing.data ? <span className="text-xs">revision {listing.data.revision.slice(0, 10)}</span> : null}
            </div>
          </DialogDescription>
        </DialogHeader>

        <div className="flex flex-col gap-2">
          <div className="text-sm font-medium">Files</div>
          <div className="max-h-60 overflow-y-auto rounded-lg border">
            {listing.isLoading ? (
              <div className="flex items-center gap-2 p-4 text-sm text-muted-foreground">
                <LoaderCircle className="size-4 animate-spin" />
                Listing the repository…
              </div>
            ) : listing.isError ? (
              <div className="p-4 text-sm text-critical">{errorMessage(listing.error)}</div>
            ) : files.length === 0 ? (
              <div className="p-4 text-sm text-muted-foreground">No files to download.</div>
            ) : (
              <ul className="divide-y">
                {files.map((file) => (
                  <li key={file.path}>
                    <label className="flex cursor-pointer items-center gap-3 px-3 py-2 hover:bg-muted/40">
                      <Checkbox
                        checked={selected.has(file.path)}
                        onCheckedChange={(checked) => toggle(file.path, checked === true)}
                        aria-label={file.path}
                      />
                      <span className="min-w-0 flex-1">
                        <span className="block truncate font-mono text-xs">{file.path}</span>
                        {file.present_locally ? (
                          <span className="block truncate text-[11px] text-muted-foreground">
                            Already at {file.present_locally}
                          </span>
                        ) : null}
                      </span>
                      <Badge variant={file.role === "model" || file.role === "model-part" ? "accent" : "muted"}>
                        {ROLE_LABEL[file.role]}
                      </Badge>
                      <span className="w-20 text-right text-xs tabular-nums">{formatBytes(file.bytes)}</span>
                    </label>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>

        {chooser ? (
          <div className="flex flex-col gap-2">
            <div>
              <div className="text-sm font-medium">N-gram volume</div>
              <div className="text-xs text-muted-foreground">
                Flash-Next needs one on a fast drive (NVMe); any table runs any conversion.
                {entry && hasNgramVolume(entry, library) ? " You already have one in your folders." : ""}
              </div>
            </div>
            <div className="flex flex-col gap-1 rounded-lg border p-1" role="radiogroup" aria-label="N-gram volume">
              {[{ key: NONE, label: "None (use one I already have)", bytes: 0, recommended: false, format: null as string | null }, ...ngramOptions.map((option) => ({
                key: optionKey(option.repo, option.file),
                label: option.file,
                bytes: option.bytes,
                recommended: option.recommended,
                format: NGRAM_FORMAT_LABELS[option.format],
              }))].map((choice) => (
                <button
                  key={choice.key}
                  type="button"
                  role="radio"
                  aria-checked={ngram === choice.key}
                  onClick={() => setNgram(choice.key)}
                  className={cn(
                    "flex items-center gap-3 rounded-md px-2.5 py-2 text-left text-sm transition-colors",
                    ngram === choice.key ? "bg-primary/10" : "hover:bg-muted/50",
                  )}
                >
                  <span
                    className={cn(
                      "flex size-4 shrink-0 items-center justify-center rounded-full border",
                      ngram === choice.key ? "border-primary" : "border-input",
                    )}
                  >
                    {ngram === choice.key ? <span className="size-2 rounded-full bg-primary" /> : null}
                  </span>
                  <span className="min-w-0 flex-1 truncate font-mono text-xs">{choice.label}</span>
                  {choice.format ? <Badge variant="muted">{choice.format}</Badge> : null}
                  {choice.recommended ? <Badge variant="accent">recommended</Badge> : null}
                  {choice.bytes > 0 ? <span className="w-20 text-right text-xs tabular-nums">{formatBytes(choice.bytes)}</span> : null}
                </button>
              ))}
            </div>
          </div>
        ) : null}

        {drafterRepo ? (
          <label className="flex cursor-pointer items-center gap-3 rounded-lg border px-3 py-2.5">
            <Checkbox checked={withDrafter} onCheckedChange={(checked) => setWithDrafter(checked === true)} />
            <span className="min-w-0 flex-1">
              <span className="block text-sm font-medium">Also download the DFlash2 drafter</span>
              <span className="block truncate text-xs text-muted-foreground">
                {drafterRepo}
                {entry && hasDrafterFor(entry, library) ? " · you already have a compatible drafter" : " · 8-11 % faster decode than MTP"}
              </span>
            </span>
            <span className="text-xs tabular-nums">
              {withDrafter ? (drafterListing.isLoading ? "…" : formatBytes(drafterBytes)) : null}
            </span>
          </label>
        ) : null}

        <div className="flex flex-col gap-2">
          <div className="text-sm font-medium">Save to</div>
          <div className="flex gap-2">
            <Input
              value={dest}
              onChange={(event) => setDest(event.target.value)}
              placeholder="Choose a folder"
              className="font-mono text-xs"
              spellCheck={false}
            />
            <Button variant="outline" onClick={() => setPicker(true)}>
              <FolderOpen />
              Browse
            </Button>
          </div>
          <label className="flex items-center gap-2 text-sm">
            <Switch checked={subfolder} onCheckedChange={setSubfolder} />
            A subfolder per repository
          </label>
          {target ? (
            <div className="text-xs text-muted-foreground">
              Files land in <span className="font-mono">{target}</span>
              {ngramChoice && !sameRepo(ngramChoice.repo, repo) && subfolder
                ? ` (the n-gram volume in ${baseName(joinPath(dest, repoName(ngramChoice.repo)))})`
                : ""}
              ; the folder is added to your model folders.
            </div>
          ) : null}
        </div>

        {tooBig ? (
          <Alert variant="destructive">
            <TriangleAlert />
            <AlertDescription>
              {formatBytes(total)} does not fit in the {formatBytes(freeBytes)} free on that drive.
            </AlertDescription>
          </Alert>
        ) : null}

        <DialogFooter className="items-center sm:justify-between">
          <div className="text-sm">
            <span className="font-semibold tabular-nums">{formatBytes(total)}</span>
            <span className="text-muted-foreground">
              {freeBytes !== null ? ` of ${formatBytes(freeBytes)} free` : " to download"}
            </span>
          </div>
          <div className="flex gap-2">
            <Button variant="ghost" onClick={() => onOpenChange(false)}>
              Cancel
            </Button>
            <Button
              disabled={submitting || dest.length === 0 || total === 0 || tooBig || listing.isLoading}
              onClick={() => void submit()}
            >
              {submitting ? <LoaderCircle className="animate-spin" /> : <HardDriveDownload />}
              Download
            </Button>
          </div>
        </DialogFooter>

        <FolderPicker
          open={picker}
          onOpenChange={setPicker}
          initialPath={dest}
          title="Download to"
          confirmLabel="Download here"
          onSelect={setDest}
        />
      </DialogContent>
    </Dialog>
  );
}
