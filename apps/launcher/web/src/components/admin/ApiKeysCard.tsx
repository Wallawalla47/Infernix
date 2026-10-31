// Settings: the launcher's named API keys. The secret of a new key is shown once (the launcher keeps
// only its SHA-256); revoking one stops every tool that sends it at once.

import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { KeyRound, Plus, ShieldCheck, ShieldOff, Trash2, TriangleAlert } from "lucide-react";
import { toast } from "sonner";

import { ConfirmDialog, SectionCardTitle } from "@/components/common";
import { CopyButton } from "@/components/CopyButton";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { Hint } from "@/components/ui/tooltip";
import { useSession } from "@/lib/admin";
import { api } from "@/lib/api";
import type { ApiKeyInfo, ApiKeyScope, CreatedApiKey, Settings } from "@/lib/api-types";
import { formatDateTime, formatRelative } from "@/lib/format";
import { useNow } from "@/lib/hooks";

const SCOPES: Record<ApiKeyScope, { label: string; help: string }> = {
  api: { label: "API", help: "The OpenAI and Anthropic endpoints (/v1): for coding tools." },
  admin: {
    label: "Admin",
    help: "The endpoints and the launcher's management: signs in to this page from another computer.",
  },
};

function ScopeBadge({ scope }: { scope: ApiKeyScope }) {
  return (
    <Hint label={SCOPES[scope]?.help ?? scope}>
      <Badge variant={scope === "admin" ? "accent" : "outline"} className="cursor-default">
        {scope === "admin" ? <ShieldCheck /> : null}
        {SCOPES[scope]?.label ?? scope}
      </Badge>
    </Hint>
  );
}

function CreateKeyDialog({
  open,
  onOpenChange,
  onCreated,
  existing,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onCreated: (created: CreatedApiKey) => void;
  existing: readonly ApiKeyInfo[];
}) {
  const [name, setName] = useState("");
  const [scope, setScope] = useState<ApiKeyScope>("api");
  const create = useMutation({
    mutationFn: () => api.createApiKey(name.trim(), scope),
    onSuccess: (created) => {
      setName("");
      setScope("api");
      onCreated(created);
    },
  });
  const trimmed = name.trim();
  const duplicate = existing.some((key) => key.name.toLowerCase() === trimmed.toLowerCase());

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Create an API key</DialogTitle>
          <DialogDescription>Name it after the tool or computer that will use it, so its requests are recognisable in Usage and the logs.</DialogDescription>
        </DialogHeader>
        <form
          className="flex flex-col gap-4"
          onSubmit={(event) => {
            event.preventDefault();
            if (trimmed.length > 0 && !duplicate && !create.isPending) create.mutate();
          }}
        >
          <div className="flex flex-col gap-1.5">
            <label htmlFor="api-key-name" className="text-sm font-medium">
              Name
            </label>
            <Input
              id="api-key-name"
              value={name}
              autoFocus
              autoComplete="off"
              placeholder="Claude Code on the laptop"
              aria-invalid={duplicate}
              onChange={(event) => setName(event.target.value)}
            />
            {duplicate ? <span className="text-xs text-critical">A key with this name exists already.</span> : null}
          </div>
          <div className="flex flex-col gap-1.5">
            <span className="text-sm font-medium">Access</span>
            <ToggleGroup
              type="single"
              value={scope}
              onValueChange={(value) => {
                if (value === "api" || value === "admin") setScope(value);
              }}
              aria-label="Access"
            >
              <ToggleGroupItem value="api">API</ToggleGroupItem>
              <ToggleGroupItem value="admin">
                <ShieldCheck />
                Admin
              </ToggleGroupItem>
            </ToggleGroup>
            <span className="text-xs text-muted-foreground">{SCOPES[scope].help}</span>
          </div>
          <DialogFooter>
            <Button variant="ghost" onClick={() => onOpenChange(false)}>
              Cancel
            </Button>
            <Button type="submit" disabled={trimmed.length === 0 || duplicate || create.isPending}>
              <KeyRound />
              {create.isPending ? "Creating…" : "Create key"}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}

function SecretDialog({ created, onClose }: { created: CreatedApiKey | null; onClose: () => void }) {
  return (
    <Dialog
      open={created !== null}
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <DialogContent className="sm:max-w-lg" onInteractOutside={(event) => event.preventDefault()}>
        <DialogHeader>
          <DialogTitle>Your new key: {created?.key.name}</DialogTitle>
          <DialogDescription>
            {created?.key.scope === "admin"
              ? "Tools send it as Authorization: Bearer, x-api-key or api-key; it also signs in to this page from another computer."
              : "Tools send it as Authorization: Bearer, x-api-key or api-key."}
          </DialogDescription>
        </DialogHeader>
        <div className="flex items-center gap-2 rounded-lg border bg-muted/40 py-1.5 pr-1.5 pl-3">
          <code className="min-w-0 flex-1 font-mono text-sm break-all select-all">{created?.secret ?? ""}</code>
          <CopyButton text={created?.secret ?? ""} label="Copy" variant="outline" size="sm" toastMessage="Key copied" />
        </div>
        <Alert variant="warning">
          <TriangleAlert />
          <AlertTitle>Copy it now: you won't see it again</AlertTitle>
          <AlertDescription>
            The launcher keeps only a hash of the key. If it is lost, revoke it and create another.
          </AlertDescription>
        </Alert>
        <DialogFooter>
          <Button onClick={onClose}>Done</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

export function ApiKeysCard({ settings, className }: { settings: Settings; className?: string }) {
  const queryClient = useQueryClient();
  const { data: session } = useSession();
  const now = useNow(30_000);
  const [creating, setCreating] = useState(false);
  const [created, setCreated] = useState<CreatedApiKey | null>(null);
  const [revoking, setRevoking] = useState<ApiKeyInfo | null>(null);

  const keys = settings.api_keys;
  const refresh = () => {
    void queryClient.invalidateQueries({ queryKey: ["settings"] });
    void queryClient.invalidateQueries({ queryKey: ["state"] });
  };
  const revoke = useMutation({
    mutationFn: (key: ApiKeyInfo) => api.deleteApiKey(key.id),
    onSuccess: (_, key) => {
      toast.success(`Revoked ${key.name}`);
      refresh();
    },
  });
  const ownKey = (key: ApiKeyInfo) => session?.signed_in === true && session.key_name === key.name;

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle>
          <SectionCardTitle icon={KeyRound}>API keys</SectionCardTitle>
        </CardTitle>
        <CardDescription>Who may use the models, and who may manage the launcher from elsewhere</CardDescription>
        <CardAction>
          <Button variant="outline" size="sm" onClick={() => setCreating(true)}>
            <Plus />
            Create key
          </Button>
        </CardAction>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <div className="flex items-start gap-2.5 rounded-lg border px-3 py-2.5 text-sm">
          {keys.length > 0 ? (
            <ShieldCheck className="mt-0.5 size-4 shrink-0 text-success" />
          ) : (
            <ShieldOff className="mt-0.5 size-4 shrink-0 text-muted-foreground" />
          )}
          <div className="min-w-0">
            <div className="font-medium">
              {keys.length > 0 ? "Tools must send a key" : "No keys: /v1 is open to anyone who can reach the launcher"}
            </div>
            <ul className="mt-1 flex list-disc flex-col gap-0.5 pl-4 text-xs leading-relaxed text-muted-foreground">
              <li>With no keys, the OpenAI and Anthropic endpoints (/v1) need none.</li>
              <li>With any key, every tool must send one (Authorization: Bearer, x-api-key or api-key).</li>
              <li>This page never needs a key on the computer running the launcher (Chat included).</li>
              <li>An admin key also signs in to manage the launcher from another computer (Manage from other computers, below).</li>
            </ul>
          </div>
        </div>

        {keys.length > 0 ? (
          <div className="rounded-lg border">
            <Table>
              <TableHeader>
                <TableRow className="hover:bg-transparent">
                  <TableHead>Name</TableHead>
                  <TableHead>Key</TableHead>
                  <TableHead>Access</TableHead>
                  <TableHead>Created</TableHead>
                  <TableHead>Last used</TableHead>
                  <TableHead className="w-10" />
                </TableRow>
              </TableHeader>
              <TableBody>
                {keys.map((key) => (
                  <TableRow key={key.id}>
                    <TableCell className="max-w-56 truncate font-medium" title={key.name}>
                      {key.name}
                      {ownKey(key) ? (
                        <Badge variant="muted" className="ml-2">
                          this browser
                        </Badge>
                      ) : null}
                    </TableCell>
                    <TableCell className="font-mono text-xs">{key.prefix}</TableCell>
                    <TableCell>
                      <ScopeBadge scope={key.scope} />
                    </TableCell>
                    <TableCell className="text-xs text-muted-foreground tabular-nums">{formatDateTime(key.created_at)}</TableCell>
                    <TableCell className="text-xs text-muted-foreground tabular-nums">
                      {key.last_used_at !== null ? (
                        <span title={new Date(key.last_used_at).toLocaleString()}>{formatRelative(key.last_used_at, now)}</span>
                      ) : (
                        "never"
                      )}
                    </TableCell>
                    <TableCell className="text-right">
                      <Button
                        variant="ghost"
                        size="icon-xs"
                        aria-label={`Revoke ${key.name}`}
                        disabled={revoke.isPending}
                        onClick={() => setRevoking(key)}
                      >
                        <Trash2 />
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        ) : null}
      </CardContent>

      <CreateKeyDialog
        open={creating}
        onOpenChange={setCreating}
        existing={keys}
        onCreated={(result) => {
          setCreating(false);
          setCreated(result);
          refresh();
        }}
      />
      <SecretDialog created={created} onClose={() => setCreated(null)} />
      <ConfirmDialog
        open={revoking !== null}
        onOpenChange={(open) => {
          if (!open) setRevoking(null);
        }}
        title={`Revoke ${revoking?.name ?? "this key"}?`}
        description={
          <span className="flex flex-col gap-2">
            <span>Every tool sending this key is refused from now on. This cannot be undone; create a new key instead.</span>
            {revoking && ownKey(revoking) ? (
              <span className="text-foreground">This browser is signed in with it and will have to sign in again.</span>
            ) : null}
            {revoking && keys.length === 1 ? (
              <span className="text-foreground">It is the last key: /v1 becomes open to anyone who can reach the launcher.</span>
            ) : null}
          </span>
        }
        confirmLabel="Revoke"
        destructive
        onConfirm={() => {
          if (revoking) revoke.mutate(revoking);
        }}
      />
    </Card>
  );
}
