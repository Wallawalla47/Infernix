import { useState, type ReactNode } from "react";
import { useMutation } from "@tanstack/react-query";
import { toast } from "sonner";

import { ConfirmDialog } from "@/components/common";
import { api } from "@/lib/api";
import { useLive } from "@/lib/events";

interface Target {
  /** Configuration id: the artifact key, or `<artifact key>@<8 hex>`. */
  key: string;
  name: string;
}

/** Load (or switch to) a configuration by its id. Another configuration of the same artifact is a
 * switch too (other parameters). While the loaded one has requests running, asks whether to wait
 * for them (drain, the default) or cancel them and switch at once. */
export function useLoadModel(): {
  load: (configId: string, name: string) => void;
  dialog: ReactNode;
  pending: boolean;
} {
  const instance = useLive((live) => live.instance);
  const [confirm, setConfirm] = useState<Target | null>(null);
  const mutation = useMutation({
    mutationFn: ({ key, force }: { key: string; force: boolean }) => api.load(key, force),
  });

  const run = (target: Target, force: boolean) => {
    mutation.mutate(
      { key: target.key, force },
      {
        onSuccess: () => {
          toast.info(force ? `Switching to ${target.name}` : `Loading ${target.name}`);
        },
      },
    );
  };

  const load = (configId: string, name: string) => {
    const busy =
      instance !== null &&
      instance.model_key !== null &&
      instance.model_key !== configId &&
      instance.in_flight > 0 &&
      (instance.state === "ready" || instance.state === "starting");
    if (busy) {
      setConfirm({ key: configId, name });
      return;
    }
    run({ key: configId, name }, false);
  };

  const running = instance?.in_flight ?? 0;
  const dialog = (
    <ConfirmDialog
      open={confirm !== null}
      onOpenChange={(open) => {
        if (!open) setConfirm(null);
      }}
      title={`Switch to ${confirm?.name ?? "this model"}?`}
      description={
        <span>
          {running} request{running === 1 ? " is" : "s are"} running on{" "}
          <span className="font-medium text-foreground">{instance?.alias ?? "the loaded model"}</span>. The launcher can
          let {running === 1 ? "it" : "them"} finish first (up to the drain timeout), or cancel{" "}
          {running === 1 ? "it" : "them"} and switch now.
        </span>
      }
      confirmLabel="Switch when done"
      onConfirm={() => {
        if (confirm) run(confirm, false);
      }}
      secondaryLabel="Switch now"
      onSecondary={() => {
        if (confirm) run(confirm, true);
      }}
    />
  );

  return { load, dialog, pending: mutation.isPending };
}
