// Usage: requests and tokens through the launcher per day, configuration and client, from the
// launcher's own accounting (GET /launcher/api/usage). Totals on top, tokens per day stacked by
// model, then the same totals by model and by client.

import { useMemo, useState, type ReactNode } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ChartColumnBig, CircleX, TriangleAlert } from "lucide-react";

import { AXIS_TICK, type RechartsTooltipProps } from "@/components/charts";
import { EmptyState, PageHeader } from "@/components/common";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { daysBetween, localDay, parseDay, useUsage, type DayRange } from "@/lib/admin";
import { errorMessage } from "@/lib/api";
import type { UsageRow } from "@/lib/api-types";
import { formatCount, formatInt, formatRatio, formatTokS } from "@/lib/format";
import { useNow } from "@/lib/hooks";
import { useTheme } from "@/lib/theme";
import { cn } from "@/lib/utils";

type Preset = "7" | "30" | "90" | "month";
type Metric = "total" | "prompt" | "output";

const PRESETS: { value: Preset; label: string }[] = [
  { value: "7", label: "7 days" },
  { value: "30", label: "30 days" },
  { value: "90", label: "90 days" },
  { value: "month", label: "This month" },
];

const METRICS: Record<Metric, { label: string; description: string; of: (row: UsageRow) => number }> = {
  total: { label: "All tokens", description: "Prompt and output tokens", of: (row) => row.prompt_tokens + row.output_tokens },
  prompt: { label: "Prompt", description: "Prompt tokens (cached included)", of: (row) => row.prompt_tokens },
  output: { label: "Output", description: "Output tokens", of: (row) => row.output_tokens },
};

// The validated categorical palette in its fixed order: slots 1-3 are the theme's series tokens,
// 4-8 the same palette's further hues stepped per theme. A ninth model folds into "Other".
const PALETTE: Record<"light" | "dark", string[]> = {
  light: ["var(--series-1)", "var(--series-2)", "var(--series-3)", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
  dark: ["var(--series-1)", "var(--series-2)", "var(--series-3)", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
};
const OTHER_COLOR = "var(--chart-muted)";

function rangeOf(preset: Preset, today: string): DayRange {
  const date = parseDay(today);
  if (preset === "month") return { from: localDay(new Date(date.getFullYear(), date.getMonth(), 1)), to: today };
  return { from: localDay(new Date(date.getFullYear(), date.getMonth(), date.getDate() - (Number(preset) - 1))), to: today };
}

function shortDay(day: string): string {
  return parseDay(day).toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function longDay(day: string): string {
  return parseDay(day).toLocaleDateString(undefined, { weekday: "short", year: "numeric", month: "short", day: "numeric" });
}

interface Totals {
  requests: number;
  failed: number;
  prompt: number;
  cached: number;
  prefill: number;
  output: number;
  prefillS: number;
  decodeS: number;
}

function emptyTotals(): Totals {
  return { requests: 0, failed: 0, prompt: 0, cached: 0, prefill: 0, output: 0, prefillS: 0, decodeS: 0 };
}

function add(totals: Totals, row: UsageRow): void {
  totals.requests += row.requests;
  totals.failed += row.failed;
  totals.prompt += row.prompt_tokens;
  totals.cached += row.cached_tokens;
  totals.prefill += row.prefill_tokens;
  totals.output += row.output_tokens;
  totals.prefillS += row.prefill_s;
  totals.decodeS += row.decode_s;
}

const cachedShare = (t: Totals) => (t.prompt > 0 ? t.cached / t.prompt : null);
const prefillRate = (t: Totals) => (t.prefillS > 0 ? t.prefill / t.prefillS : null);
const decodeRate = (t: Totals) => (t.decodeS > 0 ? t.output / t.decodeS : null);

interface Group {
  id: string;
  label: string;
  /** For the model groups: the configuration id; for clients, nothing more. */
  title?: string;
  totals: Totals;
  rows: UsageRow[];
  /** Day of the newest row (the model id is taken from it). */
  lastDay: string;
}

function groupBy(rows: readonly UsageRow[], idOf: (row: UsageRow) => string, labelOf: (row: UsageRow) => string): Group[] {
  const groups = new Map<string, Group>();
  for (const row of rows) {
    const id = idOf(row);
    let group = groups.get(id);
    if (!group) {
      group = { id, label: labelOf(row), totals: emptyTotals(), rows: [], lastDay: row.day };
      groups.set(id, group);
    }
    add(group.totals, row);
    group.rows.push(row);
    if (row.day >= group.lastDay) {
      group.lastDay = row.day;
      group.label = labelOf(row);
    }
  }
  return Array.from(groups.values()).sort(
    (a, b) => b.totals.prompt + b.totals.output - (a.totals.prompt + a.totals.output) || a.label.localeCompare(b.label),
  );
}

interface Series {
  key: string;
  /** Group ids it covers (several for "Other"). */
  ids: string[];
  label: string;
  color: string;
}

/** The chart's series: every model while they fit the palette, else the seven largest and
 * "Other". Colours follow the models in name order, so a model keeps its colour when the metric
 * changes. Returned in stack order, largest at the bottom. */
function seriesOf(models: readonly Group[], palette: readonly string[]): Series[] {
  const kept = models.length > palette.length ? models.slice(0, palette.length - 1) : models.slice();
  const folded = models.slice(kept.length);
  const colorOrder = kept.slice().sort((a, b) => a.label.localeCompare(b.label));
  const series: Series[] = kept.map((group) => ({
    key: `s${colorOrder.indexOf(group)}`,
    ids: [group.id],
    label: group.label,
    color: palette[colorOrder.indexOf(group)],
  }));
  if (folded.length > 0) {
    series.push({ key: "other", ids: folded.map((group) => group.id), label: `Other (${folded.length})`, color: OTHER_COLOR });
  }
  return series;
}

function StatTile({ label, value, sub, icon }: { label: string; value: string; sub?: string; icon?: ReactNode }) {
  return (
    <Card className="gap-1 px-4 py-3">
      <div className="flex items-center gap-1.5 text-[11px] font-medium tracking-wide text-muted-foreground uppercase">
        {icon}
        {label}
      </div>
      <div className="text-2xl leading-tight font-semibold tracking-tight">{value}</div>
      {sub ? <div className="truncate text-xs text-muted-foreground">{sub}</div> : null}
    </Card>
  );
}

function DayTooltip({ active, payload, label, series }: RechartsTooltipProps & { series: readonly Series[] }) {
  if (!active || !payload || payload.length === 0) return null;
  const entries = series
    .map((spec) => {
      const value = payload.find((item) => item.dataKey === spec.key)?.value;
      return { spec, value: typeof value === "number" && Number.isFinite(value) ? value : 0 };
    })
    .filter((entry) => entry.value > 0)
    .reverse();
  const total = entries.reduce((sum, entry) => sum + entry.value, 0);
  return (
    <div className="min-w-44 rounded-lg border bg-popover px-3 py-2 text-xs shadow-lg">
      <div className="mb-1.5 text-muted-foreground">{typeof label === "string" ? longDay(label) : ""}</div>
      <div className="mb-1 flex items-center gap-2">
        <span className="font-semibold text-foreground tabular-nums">{formatInt(total)}</span>
        <span className="text-muted-foreground">tokens</span>
      </div>
      <div className="flex flex-col gap-1">
        {entries.map(({ spec, value }) => (
          <div key={spec.key} className="flex items-center gap-2">
            <span className="h-0.5 w-3 shrink-0 rounded-full" style={{ background: spec.color }} />
            <span className="font-semibold text-foreground tabular-nums">{formatInt(value)}</span>
            <span className="truncate text-muted-foreground">{spec.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function TokensChart({ range, rows, models }: { range: DayRange; rows: readonly UsageRow[]; models: readonly Group[] }) {
  const { resolved } = useTheme();
  const [metric, setMetric] = useState<Metric>("total");
  const series = useMemo(() => seriesOf(models, PALETTE[resolved]), [models, resolved]);

  const data = useMemo(() => {
    const keyOf = new Map<string, string>();
    for (const spec of series) for (const id of spec.ids) keyOf.set(id, spec.key);
    const byDay = new Map<string, Record<string, number | string>>();
    for (const day of daysBetween(range)) {
      const point: Record<string, number | string> = { day };
      for (const spec of series) point[spec.key] = 0;
      byDay.set(day, point);
    }
    const of = METRICS[metric].of;
    for (const row of rows) {
      const point = byDay.get(row.day);
      const key = keyOf.get(row.config_id);
      if (!point || !key) continue;
      point[key] = (point[key] as number) + of(row);
    }
    return Array.from(byDay.values());
  }, [range, rows, series, metric]);

  const totals = useMemo(() => {
    const of = METRICS[metric].of;
    const byKey = new Map<string, number>();
    const keyOf = new Map<string, string>();
    for (const spec of series) for (const id of spec.ids) keyOf.set(id, spec.key);
    for (const row of rows) {
      const key = keyOf.get(row.config_id);
      if (key) byKey.set(key, (byKey.get(key) ?? 0) + of(row));
    }
    return byKey;
  }, [rows, series, metric]);

  return (
    <Card className="gap-3">
      <CardHeader>
        <CardTitle>
          <ChartColumnBig className="size-4 text-muted-foreground" />
          Tokens per day
        </CardTitle>
        <CardDescription>{METRICS[metric].description}, stacked by model</CardDescription>
        <CardAction>
          <ToggleGroup
            type="single"
            value={metric}
            onValueChange={(value) => {
              if (value === "total" || value === "prompt" || value === "output") setMetric(value);
            }}
            aria-label="Tokens shown"
          >
            {(Object.keys(METRICS) as Metric[]).map((key) => (
              <ToggleGroupItem key={key} value={key}>
                {METRICS[key].label}
              </ToggleGroupItem>
            ))}
          </ToggleGroup>
        </CardAction>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <ul className="flex flex-wrap gap-x-4 gap-y-1.5 text-xs" aria-label="Models">
          {series.map((spec) => (
            <li key={spec.key} className="flex min-w-0 items-center gap-2">
              <span className="size-2.5 shrink-0 rounded-[3px]" style={{ background: spec.color }} />
              <span className="max-w-64 truncate text-foreground" title={spec.label}>
                {spec.label}
              </span>
              <span className="text-muted-foreground tabular-nums">{formatCount(totals.get(spec.key) ?? 0)}</span>
            </li>
          ))}
        </ul>
        <div className="h-[260px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} margin={{ top: 8, right: 4, bottom: 0, left: 0 }} barCategoryGap="18%">
              <CartesianGrid vertical={false} stroke="var(--chart-grid)" />
              <XAxis
                dataKey="day"
                tick={AXIS_TICK}
                tickLine={false}
                axisLine={{ stroke: "var(--chart-axis)" }}
                tickFormatter={(day: string) => shortDay(day)}
                interval="preserveStartEnd"
                minTickGap={24}
              />
              <YAxis
                tick={AXIS_TICK}
                tickLine={false}
                axisLine={false}
                width={44}
                allowDecimals={false}
                tickFormatter={(value: number) => formatCount(value)}
              />
              <Tooltip
                isAnimationActive={false}
                cursor={{ fill: "var(--muted)", opacity: 0.6 }}
                content={<DayTooltip series={series} />}
              />
              {series.map((spec, index) => (
                <Bar
                  key={spec.key}
                  dataKey={spec.key}
                  name={spec.label}
                  stackId="tokens"
                  fill={spec.color}
                  stroke="var(--card)"
                  strokeWidth={2}
                  radius={index === series.length - 1 ? [4, 4, 0, 0] : 0}
                  maxBarSize={28}
                  isAnimationActive={false}
                />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}

function GroupTable({
  title,
  description,
  nameHeader,
  groups,
  colors,
}: {
  title: string;
  description: string;
  nameHeader: string;
  groups: readonly Group[];
  /** Group id -> its colour in the chart (models). */
  colors?: ReadonlyMap<string, string>;
}) {
  return (
    <Card className="gap-3">
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent className="px-0">
        <Table>
          <TableHeader>
            <TableRow className="hover:bg-transparent">
              <TableHead className="pl-4">{nameHeader}</TableHead>
              <TableHead className="text-right">Requests</TableHead>
              <TableHead className="text-right">Failed</TableHead>
              <TableHead className="text-right">Prompt</TableHead>
              <TableHead className="text-right">Cached</TableHead>
              <TableHead className="text-right">Output</TableHead>
              <TableHead className="text-right">Prefill</TableHead>
              <TableHead className="pr-4 text-right">Decode</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {groups.map((group) => {
              const t = group.totals;
              const color = colors?.get(group.id);
              return (
                <TableRow key={group.id}>
                  <TableCell className="max-w-72 pl-4" title={group.title ?? group.label}>
                    <span className="flex min-w-0 items-center gap-2">
                      {colors ? (
                        <span className="size-2.5 shrink-0 rounded-[3px]" style={{ background: color ?? OTHER_COLOR }} />
                      ) : null}
                      <span className="truncate">{group.label}</span>
                    </span>
                  </TableCell>
                  <TableCell className="text-right tabular-nums">{formatInt(t.requests)}</TableCell>
                  <TableCell className={cn("text-right tabular-nums", t.failed === 0 && "text-muted-foreground")}>
                    {formatInt(t.failed)}
                  </TableCell>
                  <TableCell className="text-right tabular-nums">{formatCount(t.prompt)}</TableCell>
                  <TableCell className="text-right tabular-nums">{formatRatio(cachedShare(t), 0)}</TableCell>
                  <TableCell className="text-right tabular-nums">{formatCount(t.output)}</TableCell>
                  <TableCell className="text-right tabular-nums">{formatTokS(prefillRate(t))}</TableCell>
                  <TableCell className="pr-4 text-right tabular-nums">{formatTokS(decodeRate(t))}</TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
}

export function UsagePage() {
  const [preset, setPreset] = useState<Preset>("30");
  const now = useNow(60_000);
  const today = localDay(new Date(now));
  const range = useMemo(() => rangeOf(preset, today), [preset, today]);
  const usage = useUsage(range);
  const { resolved } = useTheme();

  const rows = useMemo(() => usage.data?.rows ?? [], [usage.data]);
  const totals = useMemo(() => {
    const result = emptyTotals();
    for (const row of rows) add(result, row);
    return result;
  }, [rows]);
  const models = useMemo(() => {
    const groups = groupBy(rows, (row) => row.config_id, (row) => row.model);
    for (const group of groups) group.title = group.id;
    return groups;
  }, [rows]);
  const clients = useMemo(() => groupBy(rows, (row) => row.client, (row) => row.client), [rows]);
  const modelColors = useMemo(() => {
    const colors = new Map<string, string>();
    for (const spec of seriesOf(models, PALETTE[resolved])) for (const id of spec.ids) colors.set(id, spec.color);
    return colors;
  }, [models, resolved]);

  const perDay = totals.requests / Math.max(1, daysBetween(range).length);
  const failedShare = totals.requests > 0 ? totals.failed / totals.requests : null;

  return (
    <div className="flex flex-col gap-5">
      <PageHeader
        title="Usage"
        description={`Requests and tokens through the launcher, ${shortDay(range.from)} to ${shortDay(range.to)}, by day, model and client.`}
        actions={
          <ToggleGroup
            type="single"
            value={preset}
            onValueChange={(value) => {
              if (PRESETS.some((item) => item.value === value)) setPreset(value as Preset);
            }}
            aria-label="Date range"
          >
            {PRESETS.map((item) => (
              <ToggleGroupItem key={item.value} value={item.value}>
                {item.label}
              </ToggleGroupItem>
            ))}
          </ToggleGroup>
        }
      />

      {usage.isError ? (
        <Alert variant="destructive">
          <TriangleAlert />
          <AlertTitle>Usage could not be read</AlertTitle>
          <AlertDescription>{errorMessage(usage.error)}</AlertDescription>
        </Alert>
      ) : usage.isLoading ? (
        <div className="flex flex-col gap-4">
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
            {Array.from({ length: 5 }, (_, index) => (
              <Skeleton key={index} className="h-[88px] rounded-xl" />
            ))}
          </div>
          <Skeleton className="h-[340px] rounded-xl" />
        </div>
      ) : rows.length === 0 ? (
        <EmptyState
          icon={ChartColumnBig}
          title="No usage in this range"
          description="Every request through the launcher's endpoints (tools and Chat) is counted here by day, model and client once it finishes. Try a longer range, or send a request."
        />
      ) : (
        <>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
            <StatTile
              label="Requests"
              value={formatInt(totals.requests)}
              sub={`${perDay < 10 ? perDay.toFixed(1) : formatCount(perDay)} a day on average`}
            />
            <StatTile label="Prompt tokens" value={formatCount(totals.prompt)} sub={`${formatCount(totals.prefill)} computed`} />
            <StatTile
              label="Cached share"
              value={formatRatio(cachedShare(totals))}
              sub={`${formatCount(totals.cached)} prompt tokens from the cache`}
            />
            <StatTile label="Output tokens" value={formatCount(totals.output)} sub={`Decode ${formatTokS(decodeRate(totals))}`} />
            <StatTile
              label="Failed"
              value={formatInt(totals.failed)}
              sub={failedShare !== null ? `${formatRatio(failedShare)} of requests` : undefined}
              icon={totals.failed > 0 ? <CircleX className="size-3.5 text-critical" aria-hidden="true" /> : undefined}
            />
          </div>

          <TokensChart range={range} rows={rows} models={models} />

          <div className="grid gap-5 2xl:grid-cols-2">
            <GroupTable
              title="By model"
              description="Each configuration under its model id; rates are averages over the range"
              nameHeader="Model"
              groups={models}
              colors={modelColors}
            />
            <GroupTable
              title="By client"
              description="The API key's name, else the client's address"
              nameHeader="Client"
              groups={clients}
            />
          </div>
        </>
      )}
    </div>
  );
}
