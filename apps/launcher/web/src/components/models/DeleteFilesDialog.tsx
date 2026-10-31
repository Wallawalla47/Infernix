// Deleting an artifact's, a drafter's or an n-gram volume's files from disk. The user types the file
// name to confirm (sent as `confirm`); files a running server holds are refused here as the
// launcher refuses them.

import { useEffect, useState, type ReactNode } from "react";
import { LoaderCircle, Lock, Trash2 } from "lucide-react";
import { toast } from "sonner";

import { Alert, AlertDescription } from "@/components/ui/alert";
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
import type { DrafterArtifact, InstanceInfo, ModelArtifact, NgramVolume } from "@/lib/api-types";
import { loadedArtifactKey, pathInUse, useDeleteFiles, volumeInUse, type FileKind } from "@/lib/configs";
import { formatBytes } from "@/lib/format";

export interface DeleteTarget {
  kind: FileKind;
  key: string;
  /** What the user types to confirm. */
  fileName: string;
  /** Shown in the title. */
  title: string;
  path: string;
  bytes: number;
  /** What goes, beyond the file itself. */
  details: ReactNode;
  /** Why it cannot be deleted now, or null. */
  blocked: string | null;
}

export function modelDeleteTarget(model: ModelArtifact, instance: InstanceInfo | null): DeleteTarget {
  const loaded = loadedArtifactKey(instance) === model.key;
  const named = model.configs.length - 1;
  return {
    kind: "model",
    key: model.key,
    fileName: model.file_name,
    title: model.name,
    path: model.path,
    bytes: model.bytes,
    details: (
      <>
        {model.file_count > 1 ? `All ${model.file_count} parts` : "The artifact"}, its .conversion.json and
        .expert-state sidecars
        {named > 0
          ? `, and its ${model.configs.length} configurations (the default and ${named} named)`
          : ", and its configuration"}
        .
        {model.locations.length > 1 ? (
          <span className="mt-1 block">
            The library lists {model.locations.length} copies of this artifact:{" "}
            <span className="font-mono text-xs break-all">{model.locations.join(", ")}</span>
          </span>
        ) : null}
      </>
    ),
    blocked: loaded ? "This model is loaded. Unload it (or switch to another) first." : null,
  };
}

export function drafterDeleteTarget(drafter: DrafterArtifact, instance: InstanceInfo | null): DeleteTarget {
  return {
    kind: "drafter",
    key: drafter.key,
    fileName: drafter.file_name,
    title: drafter.name,
    path: drafter.path,
    bytes: drafter.bytes,
    details: "The drafter artifact. Configurations that name it with --draft-model fail to start until you pick another.",
    blocked: pathInUse(instance, drafter.path) ? "The loaded model drafts with it. Unload it first." : null,
  };
}

export function volumeDeleteTarget(volume: NgramVolume, instance: InstanceInfo | null): DeleteTarget {
  return {
    kind: "ngram",
    key: volume.key,
    fileName: volume.file_name,
    title: volume.file_name,
    path: volume.path,
    bytes: volume.bytes,
    details:
      volume.own_for.length > 0
        ? "The n-gram table, which is a model's own <entry>.ngram: that model needs another volume (--ngram-volume) to start."
        : "The n-gram table. Configurations that name it with --ngram-volume fail to start until you pick another.",
    blocked: volumeInUse(instance, volume.path, volume.own_for) ? "The loaded model reads this table. Unload it first." : null,
  };
}

export function DeleteFilesDialog({
  target,
  onOpenChange,
}: {
  /** null closes the dialog. */
  target: DeleteTarget | null;
  onOpenChange: (open: boolean) => void;
}) {
  const remove = useDeleteFiles();
  const [typed, setTyped] = useState("");

  useEffect(() => {
    setTyped("");
  }, [target?.kind, target?.key]);

  const matches = target !== null && typed === target.fileName;
  const submit = () => {
    const current = target;
    if (!current || !matches || current.blocked || remove.isPending) return;
    remove.mutate(
      { kind: current.kind, key: current.key, confirm: typed },
      {
        onSuccess: () => {
          toast.success(`Deleted ${current.fileName} (${formatBytes(current.bytes)} freed)`);
          onOpenChange(false);
        },
      },
    );
  };

  return (
    <Dialog open={target !== null} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-lg">
        {target ? (
          <>
            <DialogHeader>
              <DialogTitle>Delete {target.title} from disk?</DialogTitle>
              <DialogDescription asChild>
                <div className="flex flex-col gap-2">
                  <span>
                    {target.details} {formatBytes(target.bytes)} are freed. This cannot be undone: download it again to
                    get it back.
                  </span>
                  <span className="font-mono text-xs break-all text-foreground">{target.path}</span>
                </div>
              </DialogDescription>
            </DialogHeader>
            {target.blocked ? (
              <Alert variant="warning">
                <Lock />
                <AlertDescription>{target.blocked}</AlertDescription>
              </Alert>
            ) : null}
            <form
              className="flex flex-col gap-1.5"
              onSubmit={(event) => {
                event.preventDefault();
                submit();
              }}
            >
              <label htmlFor="delete-confirm" className="text-sm">
                Type <span className="font-mono font-semibold select-all">{target.fileName}</span> to confirm
              </label>
              <Input
                id="delete-confirm"
                value={typed}
                spellCheck={false}
                autoComplete="off"
                className="font-mono text-xs"
                disabled={target.blocked !== null}
                onChange={(event) => setTyped(event.target.value)}
              />
              <DialogFooter className="mt-3">
                <Button variant="ghost" onClick={() => onOpenChange(false)}>
                  Cancel
                </Button>
                <Button
                  type="submit"
                  variant="destructive"
                  disabled={!matches || target.blocked !== null || remove.isPending}
                >
                  {remove.isPending ? <LoaderCircle className="animate-spin" /> : <Trash2 />}
                  Delete files
                </Button>
              </DialogFooter>
            </form>
          </>
        ) : null}
      </DialogContent>
    </Dialog>
  );
}
