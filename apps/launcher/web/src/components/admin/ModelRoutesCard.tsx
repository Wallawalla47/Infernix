// Settings: which model a request goes to. Pinning sends everything to the loaded model; the route
// list sends requests whose `model` matches a wildcard pattern to a configuration (or the loaded
// model), first match wins. The list is edited as a draft and saved as a whole.

import { useEffect, useMemo, useState } from "react";
import { ArrowDown, ArrowUp, FlaskConical, Pin, Plus, RotateCcw, Route as RouteIcon, Save, X } from "lucide-react";

import { FieldRow, SectionCardTitle } from "@/components/common";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectSeparator, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { findRoute } from "@/lib/admin";
import type { Library, ModelRoute, Settings, SettingsPatch } from "@/lib/api-types";
import { allConfigs, findConfig } from "@/lib/library";
import { cn } from "@/lib/utils";

const LOADED = "loaded";
const EXAMPLE: ModelRoute = { pattern: "claude-*haiku*", target: LOADED };

function sameRoutes(a: readonly ModelRoute[], b: readonly ModelRoute[]): boolean {
  return a.length === b.length && a.every((route, index) => route.pattern === b[index].pattern && route.target === b[index].target);
}

function targetLabel(library: Library | undefined, target: string): string {
  if (target === LOADED) return "the loaded model";
  return findConfig(library, target)?.config.alias ?? "a missing configuration";
}

export function ModelRoutesCard({
  settings,
  library,
  onSave,
  saving,
  className,
}: {
  settings: Settings;
  library: Library | undefined;
  onSave: (patch: SettingsPatch) => void;
  saving: boolean;
  className?: string;
}) {
  const saved = settings.model_routes;
  const [draft, setDraft] = useState<ModelRoute[]>(saved);
  const [probe, setProbe] = useState("");
  // A save (or another browser's change) arriving replaces the draft only when nothing is pending.
  const [base, setBase] = useState<ModelRoute[]>(saved);
  useEffect(() => {
    if (sameRoutes(saved, base)) return;
    setDraft((current) => (sameRoutes(current, base) ? saved : current));
    setBase(saved);
  }, [saved, base]);

  const configs = useMemo(() => allConfigs(library), [library]);
  const dirty = !sameRoutes(draft, saved);
  const invalid = draft.some((route) => route.pattern.trim().length === 0);
  const match = probe.trim().length > 0 ? findRoute(draft, probe.trim()) : null;

  const edit = (index: number, change: Partial<ModelRoute>) =>
    setDraft((routes) => routes.map((route, i) => (i === index ? { ...route, ...change } : route)));
  const move = (index: number, by: -1 | 1) =>
    setDraft((routes) => {
      const next = routes.slice();
      const [route] = next.splice(index, 1);
      next.splice(index + by, 0, route);
      return next;
    });
  const remove = (index: number) => setDraft((routes) => routes.filter((_, i) => i !== index));

  return (
    <Card className={cn("gap-2", className)}>
      <CardHeader>
        <CardTitle>
          <SectionCardTitle icon={RouteIcon}>Model routing</SectionCardTitle>
        </CardTitle>
        <CardDescription>Which model answers a request, before the launcher switches models for it</CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <FieldRow
          className="border-b"
          label={
            <span className="flex items-center gap-2">
              <Pin className="size-3.5 text-muted-foreground" />
              Send every request to the loaded model; never switch automatically
            </span>
          }
          help="Whatever model a request names, the loaded model answers it. Loading a model here still switches; with nothing loaded, requests resolve as usual."
        >
          <div className="flex sm:justify-end">
            <Switch
              checked={settings.pin_loaded}
              onCheckedChange={(checked) => onSave({ pin_loaded: checked })}
              aria-label="Send every request to the loaded model"
            />
          </div>
        </FieldRow>

        <div>
          <div className="text-sm font-medium">Routes</div>
          <div className="mt-0.5 text-xs leading-relaxed text-muted-foreground">
            A request whose model id matches a pattern goes to that rule's target, before anything else. Case does not
            matter, <code className="font-mono text-foreground">*</code> matches any run of characters, and the first
            matching rule wins.
          </div>
        </div>

        {draft.length === 0 ? (
          <div className="flex flex-col items-start gap-2 rounded-lg border border-dashed px-3 py-3 text-xs text-muted-foreground sm:flex-row sm:items-center">
            <span className="min-w-0 flex-1">
              No routes: every request goes to the model it names. For example,{" "}
              <code className="font-mono text-foreground">{EXAMPLE.pattern}</code> → Loaded model keeps Claude Code's
              background requests on the loaded model instead of switching.
            </span>
            <Button variant="outline" size="xs" onClick={() => setDraft([EXAMPLE])}>
              <Plus />
              Add this example
            </Button>
          </div>
        ) : (
          <ol className="flex flex-col gap-2">
            {draft.map((route, index) => {
              const missing = route.target !== LOADED && findConfig(library, route.target) === null;
              const empty = route.pattern.trim().length === 0;
              return (
                <li
                  key={index}
                  className={cn(
                    "flex flex-col gap-2 rounded-lg border px-2.5 py-2 sm:flex-row sm:items-center",
                    match?.index === index && "border-primary/50 bg-primary/5",
                  )}
                >
                  <span className="hidden w-5 shrink-0 text-center text-xs text-muted-foreground tabular-nums sm:block">{index + 1}</span>
                  <Input
                    value={route.pattern}
                    placeholder={EXAMPLE.pattern}
                    spellCheck={false}
                    aria-label={`Pattern of rule ${index + 1}`}
                    aria-invalid={empty}
                    className="h-8 font-mono text-xs sm:flex-1"
                    onChange={(event) => edit(index, { pattern: event.target.value })}
                  />
                  <span className="hidden shrink-0 text-xs text-muted-foreground sm:block">→</span>
                  <Select value={route.target} onValueChange={(value) => edit(index, { target: value })}>
                    <SelectTrigger size="sm" className="sm:w-64" aria-label={`Target of rule ${index + 1}`}>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value={LOADED}>Loaded model</SelectItem>
                      {configs.length > 0 ? <SelectSeparator /> : null}
                      {configs.map(({ config }) => (
                        <SelectItem key={config.id} value={config.id}>
                          {config.alias}
                        </SelectItem>
                      ))}
                      {missing ? <SelectItem value={route.target}>Missing configuration</SelectItem> : null}
                    </SelectContent>
                  </Select>
                  <div className="flex shrink-0 items-center gap-0.5 self-end sm:self-auto">
                    <Button
                      variant="ghost"
                      size="icon-xs"
                      aria-label={`Move rule ${index + 1} up`}
                      disabled={index === 0}
                      onClick={() => move(index, -1)}
                    >
                      <ArrowUp />
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon-xs"
                      aria-label={`Move rule ${index + 1} down`}
                      disabled={index === draft.length - 1}
                      onClick={() => move(index, 1)}
                    >
                      <ArrowDown />
                    </Button>
                    <Button variant="ghost" size="icon-xs" aria-label={`Remove rule ${index + 1}`} onClick={() => remove(index)}>
                      <X />
                    </Button>
                  </div>
                </li>
              );
            })}
          </ol>
        )}

        <div className="flex flex-wrap items-center gap-2">
          {draft.length > 0 ? (
            <Button variant="outline" size="sm" onClick={() => setDraft((routes) => routes.concat({ pattern: "", target: LOADED }))}>
              <Plus />
              Add rule
            </Button>
          ) : null}
          {dirty ? (
            <>
              <Badge variant="accent">unsaved</Badge>
              <Button variant="ghost" size="sm" className="ml-auto" disabled={saving} onClick={() => setDraft(saved)}>
                <RotateCcw />
                Discard
              </Button>
              <Button size="sm" disabled={saving || invalid} onClick={() => onSave({ model_routes: draft })}>
                <Save />
                Save routes
              </Button>
            </>
          ) : null}
        </div>
        {dirty && invalid ? <div className="text-xs text-critical">Give every rule a pattern (or remove it) before saving.</div> : null}

        <div className="flex flex-col gap-2 rounded-lg border bg-muted/30 px-3 py-2.5">
          <label htmlFor="route-probe" className="flex items-center gap-2 text-xs font-medium text-muted-foreground">
            <FlaskConical className="size-3.5" />
            Test a name{dirty ? " (against the rules above, unsaved changes included)" : ""}
          </label>
          <Input
            id="route-probe"
            value={probe}
            spellCheck={false}
            placeholder="claude-3-5-haiku-20241022"
            className="h-8 font-mono text-xs"
            onChange={(event) => setProbe(event.target.value)}
          />
          {probe.trim().length > 0 ? (
            <div className="text-xs">
              {match ? (
                <span>
                  Rule {match.index + 1} (<code className="font-mono">{match.route.pattern}</code>) matches: it goes to{" "}
                  <span className="font-medium">{targetLabel(library, match.route.target)}</span>.
                </span>
              ) : (
                <span className="text-muted-foreground">
                  No rule matches: the request goes to the model it names (an unknown name follows Unknown model ids).
                </span>
              )}
              {settings.pin_loaded ? (
                <span className="mt-1 block text-muted-foreground">
                  Pinning is on: while a model is loaded, it answers every request.
                </span>
              ) : null}
            </div>
          ) : null}
        </div>
      </CardContent>
    </Card>
  );
}
