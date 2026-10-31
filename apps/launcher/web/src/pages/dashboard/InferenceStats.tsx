import { Gauge, Info } from "lucide-react";

import { EmptyState } from "@/components/common";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Hint } from "@/components/ui/tooltip";
import type { StatsRow } from "@/lib/api-types";
import { useLive } from "@/lib/events";
import { formatInt, formatRatio, formatSeconds, formatTokS } from "@/lib/format";
import { dashboardStats } from "@/lib/stats";
import { cn } from "@/lib/utils";

interface MetricRow {
  label: string;
  hint: string;
  cell: (row: StatsRow) => string;
}

function fixed(value: number | null, digits = 2): string {
  return value === null ? "—" : value.toFixed(digits);
}

export function InferenceStats({ className }: { className?: string }) {
  const metrics = useLive((live) => live.metrics);
  const instance = useLive((live) => live.instance);

  if (!metrics) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle>
            <Gauge className="size-4 text-muted-foreground" />
            Inference statistics
          </CardTitle>
        </CardHeader>
        <CardContent>
          <EmptyState
            icon={Gauge}
            title={instance?.state === "starting" ? "Waiting for the server" : "No server running"}
            description="Time to first token, prompt-cache hits, prefill and decode rates appear here as requests complete."
            className="py-8"
          />
        </CardContent>
      </Card>
    );
  }

  const stats = dashboardStats(metrics);
  const { session } = stats;
  const drafter = stats.drafter ? stats.drafter.toUpperCase() : "Drafter";
  const rows: MetricRow[] = [
    { label: "Requests", hint: "Completed requests in the window", cell: (row) => formatInt(row.requests) },
    {
      label: "Time to first token",
      hint: "Mean time from receiving a request to its first token, including queueing",
      cell: (row) => formatSeconds(row.ttft_s),
    },
    {
      label: "Prompt cache hit",
      hint: "Prompt tokens served from the prefix cache / all prompt tokens",
      cell: (row) => formatRatio(row.cache_hit_rate),
    },
    {
      label: "Prefill",
      hint: "Computed (uncached) prompt tokens per second of prefill",
      cell: (row) => formatTokS(row.prefill_tok_s),
    },
    {
      label: "Decode",
      hint: "Output tokens after the first per second of decode; batched requests add up",
      cell: (row) => formatTokS(row.decode_tok_s),
    },
  ];
  if (stats.exact) {
    rows.push({
      label: "Decode batch",
      hint: "Mean decode batch size, each round weighted by its duration",
      cell: (row) => fixed(row.batch),
    });
  }
  if (session.drafted > 0) {
    rows.push({
      label: `${drafter} acceptance`,
      hint: "Accepted / drafted tokens of the model drafter (n-gram rounds excluded)",
      cell: (row) => formatRatio(row.acceptance),
    });
    if (stats.exact) {
      rows.push({
        label: "Accepted per round",
        hint: "Accepted draft tokens per model-drafted round",
        cell: (row) => fixed(row.accepted_per_round),
      });
    }
  }
  if (stats.ngramEnabled || session.ngram_drafted > 0) {
    rows.push({
      label: "N-gram acceptance",
      hint: "Accepted / drafted n-gram copy tokens",
      cell: (row) => formatRatio(row.ngram_acceptance),
    });
  }
  if (session.archive_drafted > 0) {
    rows.push({
      label: "Archive acceptance",
      hint: "Accepted / drafted tokens from the n-gram archive",
      cell: (row) => formatRatio(row.archive_acceptance),
    });
  }
  if (session.expert_lookups > 0) {
    rows.push({
      label: "Expert VRAM cache hit",
      hint: "Routed experts served from the VRAM expert cache / all routed-expert lookups",
      cell: (row) => formatRatio(row.expert_hit_rate),
    });
  }

  const live = metrics.live;
  const busy = live.state === "generating" || live.state === "reading";
  const outcome = [`${formatInt(stats.completed)} done`];
  if (stats.failed > 0) outcome.push(`${formatInt(stats.failed)} failed`);
  if (stats.cancelled > 0) outcome.push(`${formatInt(stats.cancelled)} cancelled`);
  if (stats.rejected > 0) outcome.push(`${formatInt(stats.rejected)} rejected`);

  return (
    <Card className={cn("gap-3", className)}>
      <CardHeader>
        <CardTitle>
          <Gauge className="size-4 text-muted-foreground" />
          Inference statistics
        </CardTitle>
        <CardDescription>
          {outcome.join(" · ")} · {live.running} running, {live.waiting} waiting
        </CardDescription>
      </CardHeader>
      <CardContent className="px-0">
        <Table>
          <TableHeader>
            <TableRow className="hover:bg-transparent">
              <TableHead className="pl-4">Metric</TableHead>
              <TableHead className="text-right">{stats.recentLabel}</TableHead>
              <TableHead className="pr-4 text-right">{stats.sessionLabel}</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((row) => (
              <TableRow key={row.label}>
                <TableCell className="pl-4">
                  <Hint label={row.hint} side="right">
                    <span className="cursor-default text-muted-foreground decoration-dotted underline-offset-4 hover:underline">
                      {row.label}
                    </span>
                  </Hint>
                </TableCell>
                <TableCell className="text-right font-medium tabular-nums">{row.cell(stats.recent)}</TableCell>
                <TableCell className="pr-4 text-right font-medium tabular-nums">{row.cell(session)}</TableCell>
              </TableRow>
            ))}
            <TableRow className="hover:bg-transparent">
              <TableCell className="pl-4 text-muted-foreground">Now</TableCell>
              <TableCell colSpan={2} className="pr-4 text-right tabular-nums">
                {busy ? (
                  <span className="font-medium">
                    {formatTokS(live.tok_s)}
                    <span className="ml-1.5 font-normal text-muted-foreground">
                      {live.state === "reading" ? "· reading a prompt" : "· generating"}
                    </span>
                  </span>
                ) : (
                  <span className="text-muted-foreground">idle</span>
                )}
              </TableCell>
            </TableRow>
          </TableBody>
        </Table>
        {!stats.exact ? (
          <div className="mt-3 flex items-start gap-2 px-4 text-xs text-muted-foreground">
            <Info className="mt-0.5 size-3.5 shrink-0" />
            This infernix-serve build reports no session aggregates, so the rows are derived from its last{" "}
            {metrics.requests.length} requests; the batch size and drafts per round need a newer build.
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}
