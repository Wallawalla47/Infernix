import { useMemo } from "react";
import { Database, Gauge as GaugeIcon, Layers, Sparkles, Timer, Zap } from "lucide-react";

import { GaugeCard, accentZones, severityZones, useGrowingMax } from "@/components/Gauge";
import { Badge } from "@/components/ui/badge";
import { useLive } from "@/lib/events";
import { formatRateNumber, formatSeconds } from "@/lib/format";
import { dashboardStats, isCompleted } from "@/lib/stats";

function maxOf(values: (number | null)[]): number | null {
  let best: number | null = null;
  for (const value of values) {
    if (value !== null && Number.isFinite(value) && (best === null || value > best)) best = value;
  }
  return best;
}

const pct = (ratio: number | null) => (ratio === null ? null : ratio * 100);
const pctText = (value: number) => value.toFixed(1);
const secondsText = (value: number) => (value < 10 ? value.toFixed(2) : value < 100 ? value.toFixed(1) : value.toFixed(0));

/** The serving statistics as dials: rates and times over the last ten requests (the decode needle
 * follows the live rate while the server generates), with the session values beneath. */
export function InferenceGauges() {
  const metrics = useLive((live) => live.metrics);
  const instance = useLive((live) => live.instance);

  const stats = metrics ? dashboardStats(metrics) : null;
  const rows = useMemo(() => (metrics ? metrics.requests.filter(isCompleted) : []), [metrics]);
  const peaks = useMemo(
    () => ({
      decode: maxOf(rows.map((row) => row.decode_tok_s)),
      prefill: maxOf(rows.map((row) => row.prefill_tok_s)),
      ttft: maxOf(rows.map((row) => row.ttft_ms / 1000)),
      hit: maxOf(rows.map((row) => (row.hit_rate === null ? null : row.hit_rate * 100))),
    }),
    [rows],
  );

  const live = metrics?.live ?? null;
  const generating = live !== null && (live.state === "generating" || live.state === "reading");
  const recent = stats?.recent ?? null;
  const session = stats?.session ?? null;

  const decodeNow = generating && live.tok_s_mean !== null ? live.tok_s_mean : (recent?.decode_tok_s ?? null);
  const decodeMax = useGrowingMax(maxOf([decodeNow, peaks.decode]), 250);
  const prefillMax = useGrowingMax(maxOf([recent?.prefill_tok_s ?? null, peaks.prefill]), 12_000);
  const ttftMax = useGrowingMax(maxOf([recent?.ttft_s ?? null, peaks.ttft]), 10);

  const showExperts = session !== null && session.expert_lookups > 0;
  const showDrafter = session !== null && session.drafted > 0;
  const drafter = stats?.drafter ? stats.drafter.toUpperCase() : "Drafter";
  const empty = metrics === null;

  return (
    <section className="flex flex-col gap-3" aria-labelledby="inference-gauges">
      <div className="flex flex-wrap items-center gap-2">
        <h2 id="inference-gauges" className="flex items-center gap-2 text-sm font-semibold">
          <GaugeIcon className="size-4 text-muted-foreground" />
          Inference
        </h2>
        <span className="text-xs text-muted-foreground">
          {empty
            ? instance?.state === "starting"
              ? "waiting for the server to become ready"
              : "no model serving: the dials wake up with the first requests"
            : `${stats?.recentLabel ?? "Last 10"} requests; the decode needle follows the live rate while generating`}
        </span>
        {live ? (
          <div className="ml-auto flex items-center gap-1.5">
            <Badge variant={live.running > 0 ? "accent" : "outline"} className="tabular-nums">
              {live.running} running
            </Badge>
            <Badge variant={live.waiting > 0 ? "accent" : "outline"} className="tabular-nums">
              {live.waiting} waiting
            </Badge>
          </div>
        ) : null}
      </div>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-[repeat(auto-fit,minmax(170px,1fr))]">
        <GaugeCard
          title="Decode"
          icon={Zap}
          hint="Output tokens per second (after the first). While the server generates, the needle shows the live rate; otherwise the last ten requests' rate. Batched requests add up."
          value={decodeNow}
          max={decodeMax}
          unit="tok/s"
          format={formatRateNumber}
          zones={accentZones(decodeMax)}
          peak={peaks.decode}
          peakLabel="fastest request"
          aside={generating ? <Badge variant="accent">live</Badge> : null}
          sub={
            <>
              <span>Last 10 {formatRateNumber(recent?.decode_tok_s)}</span>
              <span>Session {formatRateNumber(session?.decode_tok_s)}</span>
            </>
          }
        />
        <GaugeCard
          title="Prefill"
          icon={Layers}
          hint="Computed (uncached) prompt tokens per second of prefill, over the last ten requests."
          value={recent?.prefill_tok_s ?? null}
          max={prefillMax}
          unit="tok/s"
          format={formatRateNumber}
          zones={accentZones(prefillMax)}
          peak={peaks.prefill}
          peakLabel="fastest request"
          sub={<span>Session {formatRateNumber(session?.prefill_tok_s)}</span>}
        />
        <GaugeCard
          title="Time to first token"
          icon={Timer}
          hint="Mean time from a request's arrival to its first token (queueing included), last ten requests. Lower is better."
          value={recent?.ttft_s ?? null}
          max={ttftMax}
          unit="s"
          format={secondsText}
          zones={severityZones(ttftMax, 2, 10)}
          valueColor="zone"
          peak={peaks.ttft}
          peakLabel="slowest request"
          sub={<span>Session {formatSeconds(session?.ttft_s)}</span>}
        />
        <GaugeCard
          title="Prompt cache hit"
          icon={Database}
          hint="Prompt tokens served from the prefix cache, as a share of all prompt tokens (last ten requests)."
          value={pct(recent?.cache_hit_rate ?? null)}
          max={100}
          unit="%"
          format={pctText}
          tickFormat={(value) => String(Math.round(value))}
          zones={accentZones(100)}
          peak={peaks.hit}
          peakLabel="best request"
          sub={<span>Session {session?.cache_hit_rate != null ? `${pctText(session.cache_hit_rate * 100)}%` : "—"}</span>}
        />
        {showExperts && session ? (
          <GaugeCard
            title="Expert VRAM cache hit"
            icon={GaugeIcon}
            hint="Routed experts served from the VRAM expert cache, as a share of all routed-expert lookups (last ten requests); the rest ran from RAM or the SSD tier."
            value={pct(recent?.expert_hit_rate ?? null)}
            max={100}
            unit="%"
            format={pctText}
            tickFormat={(value) => String(Math.round(value))}
            zones={accentZones(100)}
            sub={<span>Session {session.expert_hit_rate !== null ? `${pctText(session.expert_hit_rate * 100)}%` : "—"}</span>}
          />
        ) : null}
        {showDrafter && session ? (
          <GaugeCard
            title={`${drafter} acceptance`}
            icon={Sparkles}
            hint="Accepted / drafted tokens of the model drafter (n-gram rounds excluded), last ten requests."
            value={pct(recent?.acceptance ?? null)}
            max={100}
            unit="%"
            format={pctText}
            tickFormat={(value) => String(Math.round(value))}
            zones={accentZones(100)}
            sub={
              <>
                <span>Session {session.acceptance !== null ? `${pctText(session.acceptance * 100)}%` : "—"}</span>
                {session.accepted_per_round !== null ? <span>{session.accepted_per_round.toFixed(2)} per round</span> : null}
              </>
            }
          />
        ) : null}
      </div>
    </section>
  );
}
