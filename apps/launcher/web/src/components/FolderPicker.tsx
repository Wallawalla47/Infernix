import { useEffect, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowUp, Check, ChevronRight, Folder, FolderPlus, HardDrive, LoaderCircle, RefreshCw } from "lucide-react";
import { toast } from "sonner";

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
import { api, errorMessage } from "@/lib/api";
import type { FsListing } from "@/lib/api-types";
import { formatBytes } from "@/lib/format";
import { cn, joinPath, pathSeparator } from "@/lib/utils";

/** Breadcrumb segments of a host path, each with the path up to it. */
function crumbs(path: string): { name: string; path: string }[] {
  if (path.length === 0) return [];
  const sep = pathSeparator(path);
  const out: { name: string; path: string }[] = [];
  if (sep === "/") {
    out.push({ name: "/", path: "/" });
    let current = "";
    for (const part of path.split("/").filter((p) => p.length > 0)) {
      current += `/${part}`;
      out.push({ name: part, path: current });
    }
    return out;
  }
  const parts = path.split("\\").filter((p) => p.length > 0);
  let current = "";
  parts.forEach((part, index) => {
    current = index === 0 ? `${part}\\` : joinPath(current, part);
    out.push({ name: part, path: current });
  });
  return out;
}

export function FolderPicker({
  open,
  onOpenChange,
  initialPath,
  title = "Choose a folder",
  description,
  confirmLabel = "Use this folder",
  onSelect,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  initialPath?: string;
  title?: string;
  description?: string;
  confirmLabel?: string;
  onSelect: (path: string) => void;
}) {
  const queryClient = useQueryClient();
  const [path, setPath] = useState(initialPath ?? "");
  const [typed, setTyped] = useState(initialPath ?? "");
  const [newFolder, setNewFolder] = useState<string | null>(null);

  useEffect(() => {
    if (open) {
      setPath(initialPath ?? "");
      setTyped(initialPath ?? "");
      setNewFolder(null);
    }
  }, [open, initialPath]);

  const listing = useQuery<FsListing>({
    queryKey: ["fs", path],
    queryFn: () => api.listDir(path),
    enabled: open,
    staleTime: 5_000,
  });

  const go = (next: string) => {
    setPath(next);
    setTyped(next);
    setNewFolder(null);
  };

  const createFolder = async () => {
    const name = (newFolder ?? "").trim();
    if (name.length === 0 || path.length === 0) return;
    const target = joinPath(path, name);
    try {
      await api.mkdir(target);
      await queryClient.invalidateQueries({ queryKey: ["fs", path] });
      go(target);
    } catch (error) {
      toast.error(errorMessage(error));
    }
  };

  const data = listing.data;
  const current = data?.path ?? path;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="gap-3 sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
          <DialogDescription>
            {description ?? "Folders on the computer running the launcher."}
          </DialogDescription>
        </DialogHeader>

        <form
          className="flex gap-2"
          onSubmit={(event) => {
            event.preventDefault();
            go(typed.trim());
          }}
        >
          <Input
            value={typed}
            onChange={(event) => setTyped(event.target.value)}
            placeholder="Type a path, or pick a drive below"
            className="font-mono text-xs"
            spellCheck={false}
          />
          <Button type="submit" variant="outline">
            Go
          </Button>
        </form>

        <div className="flex min-h-7 flex-wrap items-center gap-1 text-xs">
          <Button variant="ghost" size="xs" onClick={() => go("")}>
            <HardDrive />
            Drives
          </Button>
          {crumbs(current).map((crumb) => (
            <span key={crumb.path} className="flex items-center gap-1">
              <ChevronRight className="size-3 text-muted-foreground" />
              <button
                type="button"
                className="max-w-40 truncate rounded px-1 py-0.5 hover:bg-accent"
                onClick={() => go(crumb.path)}
              >
                {crumb.name}
              </button>
            </span>
          ))}
        </div>

        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" disabled={!data?.parent && current.length === 0} onClick={() => go(data?.parent ?? "")}>
            <ArrowUp />
            Up
          </Button>
          <Button variant="outline" size="sm" disabled={current.length === 0} onClick={() => setNewFolder("")}>
            <FolderPlus />
            New folder
          </Button>
          <Button
            variant="ghost"
            size="icon-sm"
            aria-label="Refresh"
            onClick={() => void queryClient.invalidateQueries({ queryKey: ["fs", path] })}
          >
            <RefreshCw />
          </Button>
          <div className="ml-auto text-xs text-muted-foreground tabular-nums">
            {data?.free_bytes != null ? `${formatBytes(data.free_bytes)} free` : null}
            {data?.free_bytes != null && data.total_bytes != null ? ` of ${formatBytes(data.total_bytes)}` : null}
          </div>
        </div>

        {newFolder !== null ? (
          <form
            className="flex gap-2"
            onSubmit={(event) => {
              event.preventDefault();
              void createFolder();
            }}
          >
            <Input
              autoFocus
              value={newFolder}
              onChange={(event) => setNewFolder(event.target.value)}
              placeholder="New folder name"
            />
            <Button type="submit" disabled={newFolder.trim().length === 0}>
              Create
            </Button>
            <Button variant="ghost" onClick={() => setNewFolder(null)}>
              Cancel
            </Button>
          </form>
        ) : null}

        <div className="h-72 overflow-y-auto rounded-lg border">
          {listing.isLoading ? (
            <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
              <LoaderCircle className="mr-2 size-4 animate-spin" />
              Reading…
            </div>
          ) : listing.isError ? (
            <div className="p-4 text-sm text-critical">{errorMessage(listing.error)}</div>
          ) : data?.error ? (
            <div className="p-4 text-sm text-critical">{data.error}</div>
          ) : data && data.entries.length === 0 ? (
            <div className="p-4 text-sm text-muted-foreground">No folders here.</div>
          ) : (
            <ul className="divide-y">
              {(data?.entries ?? []).map((entry) => (
                <li key={entry.path}>
                  <button
                    type="button"
                    className="flex w-full items-center gap-2.5 px-3 py-2 text-left text-sm hover:bg-accent/60"
                    onClick={() => go(entry.path)}
                  >
                    {entry.kind === "drive" ? (
                      <HardDrive className="size-4 shrink-0 text-muted-foreground" />
                    ) : (
                      <Folder className="size-4 shrink-0 text-primary/80" />
                    )}
                    <span className="truncate">{entry.name}</span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>

        <DialogFooter className="items-center sm:justify-between">
          <div className={cn("truncate font-mono text-xs text-muted-foreground", current.length === 0 && "invisible")}>
            {current}
          </div>
          <div className="flex gap-2">
            <Button variant="ghost" onClick={() => onOpenChange(false)}>
              Cancel
            </Button>
            <Button
              disabled={current.length === 0}
              onClick={() => {
                onSelect(current);
                onOpenChange(false);
              }}
            >
              <Check />
              {confirmLabel}
            </Button>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
