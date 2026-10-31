import { useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ChartColumn, Table2 } from "lucide-react";

import { AXIS_TICK, SeriesTooltip, type SeriesSpec } from "@/components/charts";
import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { useLive } from "@/lib/events";
import { formatInt, formatRateNumber, formatSeconds, formatTokS } from "@/lib/format";
import { isCompleted } from "@/lib/stats";
import { cn } from "@/lib/utils";

const WINDOW = 40;

interface Point {
  label: string;
  id: number;
  decode: number | null;
  ttft: number;
  prompt: number;
  output: number;
}

function compact(value: number): string {
  if (value >= 1000) return `${(value / 1000).toFixed(value >= 10_000 ? 0 : 1)}K`;
  return value >= 10 ? value.toFixed(0) : value.toFixed(1);
}

function MiniBars({
  data,
  dataKey,
  color,
  spec,
  height = 168,
}: {
  data: Point[];
  dataKey: "decode" | "ttft";
  color: string;
  spec: SeriesSpec;
  height?: number;
}) {
  return (
    <div style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 8, right: 4, bottom: 0, left: 0 }} barCategoryGap="12%">
          <CartesianGrid vertical={false} stroke="var(--chart-grid)" />
          <XAxis
            dataKey="label"
            tick={AXIS_TICK}
            tickLine={false}
            axisLine={{ stroke: "var(--chart-axis)" }}
            interval="preserveStartEnd"
            minTickGap={18}
          />
          <YAxis
            tick={AXIS_TICK}
            tickLine={false}
            axisLine={false}
            width={38}
            tickFormatter={(value: number) => compact(value)}
            allowDecimals={dataKey === "ttft"}
          />
          <Tooltip
            isAnimationActive={false}
            cursor={{ fill: "var(--muted)", opacity: 0.6 }}
            content={<SeriesTooltip series={[spec]} labelFormat={(label) => `Request ${String(label)}`} />}
          />
          <Bar
            dataKey={dataKey}
            name={spec.label}
            fill={color}
            radius={[4, 4, 0, 0]}
            maxBarSize={24}
            isAnimationActive={false}
            activeBar={{ fillOpacity: 0.8 }}
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function RequestCharts({ className }: { className?: string }) {
  const metrics = useLive((live) => live.metrics);
  const [view, setView] = useState<"chart" | "table">("chart");

  const data = useMemo<Point[]>(() => {
    if (!metrics) return [];
    return metrics.requests
      .filter(isCompleted)
      .slice(-WINDOW)
      .map((row) => ({
        label: `#${row.id}`,
        id: row.id,
        decode: row.decode_tok_s,
        ttft: row.ttft_ms / 1000,
        prompt: row.prompt_tokens,
        output: row.output_tokens,
      }));
  }, [metrics]);

  // Decode wears the accent, like the decode gauge; beside series-1 it passes the palette checks in
  // both themes (CVD Delta E 21.3 light, 17.4 dark).
  const decodeSpec: SeriesSpec = { key: "decode", label: "decode", color: "var(--primary)", format: (v) => formatTokS(v) };
  const ttftSpec: SeriesSpec = { key: "ttft", label: "time to first token", color: "var(--series-1)", format: (v) => formatSeconds(v) };
  const latest = data.length > 0 ? data[data.length - 1] : null;

  return (
    <Card className={cn("gap-3", className)}>
      <CardHeader>
        <CardTitle>
          <ChartColumn className="size-4 text-muted-foreground" />
          Per request
        </CardTitle>
        <CardDescription>
          {data.length > 0
            ? `The last ${data.length} completed requests, oldest first`
            : "Each completed request's decode rate and time to first token"}
        </CardDescription>
        <CardAction>
          <ToggleGroup
            type="single"
            value={view}
            onValueChange={(value) => {
              if (value === "chart" || value === "table") setView(value);
            }}
            aria-label="Chart or table"
          >
            <ToggleGroupItem value="chart" aria-label="Chart">
              <ChartColumn />
            </ToggleGroupItem>
            <ToggleGroupItem value="table" aria-label="Table">
              <Table2 />
            </ToggleGroupItem>
          </ToggleGroup>
        </CardAction>
      </CardHeader>
      <CardContent>
        {data.length === 0 ? (
          <div className="flex h-[200px] items-center justify-center rounded-lg border border-dashed text-sm text-muted-foreground">
            No completed requests yet
          </div>
        ) : view === "chart" ? (
          <div className="grid gap-5 lg:grid-cols-2">
            <div className="min-w-0">
              <div className="mb-2 flex items-baseline justify-between gap-2">
                <span className="flex items-center gap-2 text-xs font-medium text-muted-foreground">
                  <span className="size-2.5 rounded-[3px]" style={{ background: decodeSpec.color }} />
                  Decode, tok/s
                </span>
                <span className="text-xs text-muted-foreground tabular-nums">
                  latest <span className="font-medium text-foreground">{formatRateNumber(latest?.decode)}</span>
                </span>
              </div>
              <MiniBars data={data} dataKey="decode" color={decodeSpec.color} spec={decodeSpec} />
            </div>
            <div className="min-w-0">
              <div className="mb-2 flex items-baseline justify-between gap-2">
                <span className="flex items-center gap-2 text-xs font-medium text-muted-foreground">
                  <span className="size-2.5 rounded-[3px]" style={{ background: ttftSpec.color }} />
                  Time to first token, s
                </span>
                <span className="text-xs text-muted-foreground tabular-nums">
                  latest <span className="font-medium text-foreground">{formatSeconds(latest?.ttft)}</span>
                </span>
              </div>
              <MiniBars data={data} dataKey="ttft" color={ttftSpec.color} spec={ttftSpec} />
            </div>
          </div>
        ) : (
          <div className="max-h-[220px] overflow-y-auto rounded-lg border">
            <Table>
              <TableHeader className="sticky top-0 bg-card">
                <TableRow className="hover:bg-transparent">
                  <TableHead>Request</TableHead>
                  <TableHead className="text-right">Prompt</TableHead>
                  <TableHead className="text-right">Output</TableHead>
                  <TableHead className="text-right">TTFT</TableHead>
                  <TableHead className="text-right">Decode</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {[...data].reverse().map((point) => (
                  <TableRow key={point.id}>
                    <TableCell className="font-mono text-xs">{point.label}</TableCell>
                    <TableCell className="text-right tabular-nums">{formatInt(point.prompt)}</TableCell>
                    <TableCell className="text-right tabular-nums">{formatInt(point.output)}</TableCell>
                    <TableCell className="text-right tabular-nums">{formatSeconds(point.ttft)}</TableCell>
                    <TableCell className="text-right tabular-nums">{formatTokS(point.decode)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
