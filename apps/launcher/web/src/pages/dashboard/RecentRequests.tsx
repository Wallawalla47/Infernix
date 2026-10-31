import { History } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import type { MetricsRequestRow } from "@/lib/api-types";
import { useLive } from "@/lib/events";
import { formatClock, formatCount, formatMs, formatRatio, formatTokS } from "@/lib/format";
import { cn } from "@/lib/utils";

const SHOWN = 25;

const PROTOCOL: Record<string, string> = {
  chat: "Chat",
  "chat.completions": "Chat",
  responses: "Responses",
  messages: "Messages",
  anthropic: "Messages",
  decide: "Decide",
};

function FinishBadge({ finish }: { finish: string }) {
  if (finish === "failed") return <Badge variant="destructive">failed</Badge>;
  if (finish === "cancelled") return <Badge variant="muted">cancelled</Badge>;
  if (finish === "length") return <Badge variant="accent">length</Badge>;
  return <Badge variant="outline">{finish}</Badge>;
}

function drafts(row: MetricsRequestRow): string {
  if (row.drafts_offered <= 0) return "—";
  return formatRatio(row.drafts_accepted / row.drafts_offered, 0);
}

export function RecentRequests({ className }: { className?: string }) {
  const metrics = useLive((live) => live.metrics);
  const rows = metrics ? metrics.requests.slice(-SHOWN).reverse() : [];
  const experts = rows.some((row) => row.expert_lookups > 0);

  return (
    <Card className={cn("gap-3", className)}>
      <CardHeader>
        <CardTitle>
          <History className="size-4 text-muted-foreground" />
          Recent requests
        </CardTitle>
        <CardDescription>The newest first; the server keeps its last 128.</CardDescription>
      </CardHeader>
      <CardContent className="px-0">
        {rows.length === 0 ? (
          <div className="mx-4 flex h-32 items-center justify-center rounded-lg border border-dashed text-sm text-muted-foreground">
            No requests yet
          </div>
        ) : (
          <Table containerClassName="max-h-[340px] overflow-y-auto">
            <TableHeader className="sticky top-0 z-10 bg-card">
              <TableRow className="hover:bg-transparent">
                <TableHead className="pl-4">Time</TableHead>
                <TableHead>API</TableHead>
                <TableHead>Finish</TableHead>
                <TableHead className="text-right">Prompt</TableHead>
                <TableHead className="text-right">Cached</TableHead>
                <TableHead className="text-right">Output</TableHead>
                <TableHead className="text-right">TTFT</TableHead>
                <TableHead className="text-right">Prefill</TableHead>
                <TableHead className="text-right">Decode</TableHead>
                <TableHead className="text-right">Drafts</TableHead>
                {experts ? <TableHead className="pr-4 text-right">Experts</TableHead> : null}
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map((row) => {
                const failed = row.finish === "failed" || row.finish === "cancelled";
                return (
                  <TableRow key={row.id} className={cn(failed && "text-muted-foreground")}>
                    <TableCell className="pl-4 tabular-nums">{formatClock(row.time * 1000)}</TableCell>
                    <TableCell>{PROTOCOL[row.protocol] ?? row.protocol}</TableCell>
                    <TableCell>
                      <FinishBadge finish={row.finish} />
                    </TableCell>
                    <TableCell className="text-right tabular-nums">{failed ? "—" : formatCount(row.prompt_tokens)}</TableCell>
                    <TableCell className="text-right tabular-nums">{failed ? "—" : formatRatio(row.hit_rate, 0)}</TableCell>
                    <TableCell className="text-right tabular-nums">{failed ? "—" : formatCount(row.output_tokens)}</TableCell>
                    <TableCell className="text-right tabular-nums">{failed ? "—" : formatMs(row.ttft_ms)}</TableCell>
                    <TableCell className="text-right tabular-nums">{failed ? "—" : formatTokS(row.prefill_tok_s)}</TableCell>
                    <TableCell className="text-right tabular-nums">{failed ? "—" : formatTokS(row.decode_tok_s)}</TableCell>
                    <TableCell className="text-right tabular-nums">{failed ? "—" : drafts(row)}</TableCell>
                    {experts ? (
                      <TableCell className="pr-4 text-right tabular-nums">
                        {row.expert_lookups > 0 ? formatRatio(row.expert_hits / row.expert_lookups, 0) : "—"}
                      </TableCell>
                    ) : null}
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        )}
      </CardContent>
    </Card>
  );
}
