// "Will it fit?": `infernix-serve --plan` sizes the edited (unsaved) parameters on this machine
// without loading the weights. Suggestions and the largest fitting context edit the draft only;
// nothing is saved until the page is.

import { useEffect, useRef, useState } from "react";
import { Check, CircleCheck, CircleHelp, CircleX, Gauge, LoaderCircle, MemoryStick, Gpu } from "lucide-react";
import { toast } from "sonner";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { errorMessage } from "@/lib/api";
import type { EnvVar, FitPlan, PlanBudget } from "@/lib/api-types";
import { getFlagString, sameArgs, setFlag, type TakesValue } from "@/lib/args";
import { cleanEnv, sameEnv, usePlan } from "@/lib/configs";
import { formatContext, formatGiB, formatInt } from "@/lib/format";
import { useNow } from "@/lib/hooks";
import { cn } from "@/lib/utils";

function FitBadge({ fits }: { fits: boolean | null }) {
  const Icon = fits === true ? CircleCheck : fits === false ? CircleX : CircleHelp;
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full border bg-background/60 px-2 py-0.5 text-xs font-medium">
      <Icon
        className={cn("size-3.5", fits === true ? "text-success" : fits === false ? "text-critical" : "text-muted-foreground")}
        aria-hidden="true"
      />
      {fits === true ? "Fits" : fits === false ? "Does not fit" : "Unknown"}
    </span>
  );
}

/** Planned memory against what is available for the model (the rest of the total is in use by
 * other programs, and by the loaded model unless it is the one planned). */
function BudgetBar({ label, icon: Icon, budget }: { label: string; icon: typeof Gpu; budget: PlanBudget }) {
  const total = Math.max(budget.total, 1);
  const others = Math.max(0, budget.total - budget.available);
  const room = Math.max(0, budget.total - others);
  const over = budget.planned > budget.available;
  const pct = (bytes: number) => `${Math.max(0, Math.min(100, (bytes / total) * 100))}%`;
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-0.5">
        <span className="flex items-center gap-1.5 text-sm font-medium">
          <Icon className="size-4 text-muted-foreground" />
          {label}
        </span>
        <span className="text-xs text-muted-foreground tabular-nums">
          <span className={cn("font-semibold", over ? "text-critical" : "text-foreground")}>{formatGiB(budget.planned)}</span> of{" "}
          {formatGiB(budget.available)} available · {formatGiB(budget.total)} total
        </span>
      </div>
      <div
        className="flex h-2.5 w-full overflow-hidden rounded-full bg-muted"
        role="img"
        aria-label={`${label}: ${formatGiB(budget.planned)} planned of ${formatGiB(budget.available)} available, ${formatGiB(budget.total)} total`}
      >
        <div className="h-full bg-muted-foreground/35" style={{ width: pct(others) }} />
        <div className={cn("h-full", over ? "bg-critical" : "bg-primary")} style={{ width: pct(Math.min(budget.planned, room)) }} />
      </div>
      <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-[11px] text-muted-foreground">
        <span className="inline-flex items-center gap-1">
          <span className="size-2 rounded-sm bg-muted-foreground/35" />
          In use elsewhere {formatGiB(others)}
        </span>
        <span className="inline-flex items-center gap-1">
          <span className={cn("size-2 rounded-sm", over ? "bg-critical" : "bg-primary")} />
          This configuration {formatGiB(budget.planned)}
        </span>
        <span className={cn(over && "font-medium text-critical")}>
          {over ? `Short by ${formatGiB(budget.planned - budget.available)}` : `${formatGiB(budget.available - budget.planned)} to spare`}
        </span>
      </div>
      {budget.parts.length > 0 ? (
        <ul className="flex flex-col gap-0.5 rounded-md border bg-muted/30 px-2.5 py-1.5 text-xs">
          {budget.parts.map((part, index) => (
            <li key={`${index}-${part.label}`} className="flex items-baseline justify-between gap-3">
              <span className="min-w-0 truncate text-muted-foreground" title={part.label}>
                {part.label}
              </span>
              <span className="shrink-0 tabular-nums">{formatGiB(part.bytes, 2)}</span>
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}

function Elapsed({ since }: { since: number }) {
  const now = useNow(250);
  return <span className="tabular-nums">{Math.max(0, (now - since) / 1000).toFixed(1)} s</span>;
}

function PlanResult({
  plan,
  args,
  takesValue,
  onSetArgs,
}: {
  plan: FitPlan;
  args: string[];
  takesValue: TakesValue;
  onSetArgs: (args: string[]) => void;
}) {
  const [applied, setApplied] = useState<ReadonlySet<number>>(() => new Set());
  useEffect(() => setApplied(new Set()), [plan]);

  if (!plan.supported) {
    return (
      <Alert variant="warning">
        <CircleHelp />
        <AlertTitle>This infernix-serve cannot plan</AlertTitle>
        <AlertDescription>
          It is older than --plan. Update Infernix to check a configuration before loading it; loading still works.
        </AlertDescription>
      </Alert>
    );
  }
  if (plan.error) {
    return (
      <Alert variant="destructive">
        <CircleX />
        <AlertTitle>The plan could not be made</AlertTitle>
        <AlertDescription>
          <p className="break-words">{plan.error}</p>
        </AlertDescription>
      </Alert>
    );
  }

  const currentContext = getFlagString(args, "--max-context", takesValue);
  const contextShown = currentContext.length > 0 ? Number(currentContext) : 8192;

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-col gap-1.5">
        <FitBadge fits={plan.fits} />
        {plan.summary ? <p className="text-sm">{plan.summary}</p> : null}
      </div>
      {plan.vram ? <BudgetBar label="GPU memory" icon={Gpu} budget={plan.vram} /> : null}
      {plan.ram ? <BudgetBar label="System RAM" icon={MemoryStick} budget={plan.ram} /> : null}
      {plan.max_context !== null || plan.kv_tokens !== null ? (
        <div className="flex flex-col gap-2 rounded-lg border px-3 py-2.5">
          {plan.max_context !== null ? (
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="min-w-0">
                <div className="text-xs text-muted-foreground">Largest context that fits</div>
                <div className="text-sm font-semibold tabular-nums">
                  {formatInt(plan.max_context)} tokens{" "}
                  <span className="font-normal text-muted-foreground">({formatContext(plan.max_context)})</span>
                </div>
              </div>
              {plan.max_context !== contextShown ? (
                <Button
                  size="xs"
                  variant="outline"
                  onClick={() => {
                    onSetArgs(setFlag(args, "--max-context", String(plan.max_context), takesValue));
                    toast.success(`--max-context ${plan.max_context} set; save to keep it`);
                  }}
                >
                  Use --max-context {formatContext(plan.max_context)}
                </Button>
              ) : (
                <Badge variant="accent">
                  <Check />
                  in use
                </Badge>
              )}
            </div>
          ) : null}
          {plan.kv_tokens !== null ? (
            <div className="text-xs text-muted-foreground">
              KV capacity across all requests: <span className="text-foreground tabular-nums">{formatInt(plan.kv_tokens)}</span>{" "}
              tokens
            </div>
          ) : null}
        </div>
      ) : null}
      {plan.fits === false && plan.suggestions.length > 0 ? (
        <div className="flex flex-col gap-1.5">
          <div className="text-xs font-medium">
            {plan.suggestions.length === 1 ? "A change to try" : "Changes to try, most effective first"}
          </div>
          <ul className="flex flex-col gap-1.5">
            {plan.suggestions.map((suggestion, index) => (
              <li
                key={`${index}-${suggestion.title}`}
                className="flex flex-col gap-2 rounded-md border bg-background/60 px-3 py-2 sm:flex-row sm:items-center"
              >
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-1.5 text-sm font-medium">
                    {suggestion.title}
                    {applied.has(index) ? (
                      <Badge variant="accent">
                        <Check />
                        applied
                      </Badge>
                    ) : null}
                  </div>
                  <div className="text-xs text-muted-foreground">{suggestion.detail}</div>
                </div>
                {suggestion.args ? (
                  <Button
                    size="xs"
                    variant="outline"
                    className="shrink-0"
                    onClick={() => {
                      if (!suggestion.args) return;
                      onSetArgs([...suggestion.args]);
                      setApplied((old) => new Set(old).add(index));
                      toast.success(`${suggestion.title}: applied to the launch parameters; save to keep it`);
                    }}
                  >
                    Apply
                  </Button>
                ) : null}
              </li>
            ))}
          </ul>
        </div>
      ) : null}
      {plan.report !== null && plan.report !== undefined ? (
        <details className="text-xs">
          <summary className="cursor-pointer text-muted-foreground hover:text-foreground">The server's report</summary>
          <pre className="mt-1.5 max-h-64 overflow-auto rounded-md border bg-muted/40 p-2 font-mono text-[11px] leading-4">
            {JSON.stringify(plan.report, null, 2)}
          </pre>
        </details>
      ) : null}
    </div>
  );
}

/** The "Will it fit?" card. Render it with `key={configId}` so another configuration starts clean. */
export function FitCard({
  configId,
  args,
  env,
  takesValue,
  onSetArgs,
}: {
  configId: string;
  /** The edited (unsaved) parameters. */
  args: string[];
  env: EnvVar[];
  takesValue: TakesValue;
  onSetArgs: (args: string[]) => void;
}) {
  const plan = usePlan();
  // The last answer and the parameters it was made for (kept while a new check runs).
  const [checked, setChecked] = useState<{ plan: FitPlan; args: string[]; env: EnvVar[] } | null>(null);
  const [startedAt, setStartedAt] = useState(0);

  const run = () => {
    const body = { args: [...args], env: cleanEnv(env) };
    setStartedAt(Date.now());
    plan.mutate({ id: configId, ...body }, { onSuccess: (answer) => setChecked({ plan: answer, ...body }) });
  };

  const stale = checked !== null && (!sameArgs(checked.args, args) || !sameEnv(checked.env, cleanEnv(env)));

  // The vision tower takes VRAM the default context was sized without (a 27B's default context no
  // longer fits beside it), so turning it on checks at once and offers the context that fits.
  const hadVision = useRef(args.includes("--vision"));
  useEffect(() => {
    const hasVision = args.includes("--vision");
    if (hasVision && !hadVision.current && !plan.isPending) run();
    hadVision.current = hasVision;
    // eslint-disable-next-line react-hooks/exhaustive-deps -- runs on edits to the parameters only
  }, [args]);

  return (
    <Card className="gap-3">
      <CardHeader>
        <CardTitle>
          <Gauge className="size-4 text-muted-foreground" />
          Will it fit?
        </CardTitle>
        <CardDescription>
          The Engine's own sizing of the edited parameters on this machine, without loading the weights.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <div className="flex flex-wrap items-center gap-2">
          <Button size="sm" variant={checked ? "outline" : "default"} disabled={plan.isPending} onClick={run}>
            {plan.isPending ? <LoaderCircle className="animate-spin" /> : <Gauge />}
            {checked ? "Check again" : "Check fit"}
          </Button>
          {stale && !plan.isPending ? (
            <span className="text-xs text-warning">The parameters changed since this check.</span>
          ) : null}
        </div>
        {plan.isPending ? (
          <div className="flex flex-col gap-1.5">
            <Progress value={null} />
            <div className="flex items-center justify-between text-xs text-muted-foreground">
              <span>Sizing with infernix-serve --plan…</span>
              <Elapsed since={startedAt} />
            </div>
          </div>
        ) : null}
        {plan.isError && !plan.isPending ? <div className="text-xs text-critical">{errorMessage(plan.error)}</div> : null}
        {checked ? (
          <div className={cn((stale || plan.isPending) && "opacity-60")}>
            <PlanResult plan={checked.plan} args={args} takesValue={takesValue} onSetArgs={onSetArgs} />
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}
