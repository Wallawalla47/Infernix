import type { MetricsRequestRow, MetricsSnapshot, StatsRow } from "./api-types";

export interface DashboardStats {
  recent: StatsRow;
  session: StatsRow;
  recentLabel: string;
  sessionLabel: string;
  drafter: string | null;
  ngramEnabled: boolean;
  /** True when the server computed the rows (session_stats); false when derived here. */
  exact: boolean;
  completed: number;
  failed: number;
  cancelled: number;
  rejected: number;
}

const RECENT_WINDOW = 10;

function ratio(part: number, whole: number): number | null {
  return whole > 0 ? part / whole : null;
}

function rate(tokens: number, seconds: number): number | null {
  return seconds > 0 ? tokens / seconds : null;
}

export function isCompleted(row: MetricsRequestRow): boolean {
  return row.finish !== "failed" && row.finish !== "cancelled";
}

/** The console panel's aggregation, rebuilt from request rows for servers whose metrics carry no
 * session_stats. Ratios divide summed tokens by summed seconds; the decode batch and accepted
 * drafts per round are not in the rows and stay null. */
export function deriveStatsRow(rows: readonly MetricsRequestRow[]): StatsRow {
  let ttft = 0;
  let prompt = 0;
  let cached = 0;
  let prefillTokens = 0;
  let prefillSeconds = 0;
  let decodeTokens = 0;
  let decodeSeconds = 0;
  let drafted = 0;
  let accepted = 0;
  let ngramDrafted = 0;
  let ngramAccepted = 0;
  let expertLookups = 0;
  let expertHits = 0;
  for (const row of rows) {
    ttft += row.ttft_ms / 1000;
    prompt += row.prompt_tokens;
    cached += Math.min(row.reused, row.prompt_tokens);
    if (row.prompt_ms > 0) {
      prefillTokens += Math.max(0, row.prompt_tokens - row.reused);
      prefillSeconds += row.prompt_ms / 1000;
    }
    if (row.decode_ms > 0) {
      decodeTokens += Math.max(0, row.output_tokens - 1);
      decodeSeconds += row.decode_ms / 1000;
    }
    drafted += Math.max(0, row.drafts_offered - row.ngram_drafts);
    accepted += Math.max(0, row.drafts_accepted - row.ngram_accepted);
    ngramDrafted += row.ngram_drafts;
    ngramAccepted += row.ngram_accepted;
    expertLookups += row.expert_lookups;
    expertHits += row.expert_hits;
  }
  return {
    requests: rows.length,
    ttft_s: rows.length > 0 ? ttft / rows.length : null,
    prompt_tokens: prompt,
    cached_tokens: cached,
    cache_hit_rate: ratio(cached, prompt),
    prefill_tokens: prefillTokens,
    prefill_s: prefillSeconds,
    prefill_tok_s: rate(prefillTokens, prefillSeconds),
    decode_tokens: decodeTokens,
    decode_s: decodeSeconds,
    decode_tok_s: rate(decodeTokens, decodeSeconds),
    batch: null,
    drafted,
    accepted,
    acceptance: ratio(accepted, drafted),
    accepted_per_round: null,
    ngram_drafted: ngramDrafted,
    ngram_accepted: ngramAccepted,
    ngram_acceptance: ratio(ngramAccepted, ngramDrafted),
    ngram_rounds: 0,
    archive_drafted: 0,
    archive_accepted: 0,
    archive_acceptance: null,
    expert_lookups: expertLookups,
    expert_hits: expertHits,
    expert_hit_rate: ratio(expertHits, expertLookups),
  };
}

export function dashboardStats(metrics: MetricsSnapshot): DashboardStats {
  const server = metrics.session_stats;
  if (server) {
    return {
      recent: server.recent,
      session: server.session,
      recentLabel: `Last ${server.recent_window}`,
      sessionLabel: "Session",
      drafter: server.drafter,
      ngramEnabled: server.ngram_enabled,
      exact: true,
      completed: server.completed,
      failed: server.failed,
      cancelled: server.cancelled,
      rejected: server.rejected,
    };
  }
  const completed = metrics.requests.filter(isCompleted);
  const recent = completed.slice(-RECENT_WINDOW);
  const wholeSession = metrics.totals.completed <= completed.length;
  const backend = metrics.engine.speculative_backend;
  return {
    recent: deriveStatsRow(recent),
    session: deriveStatsRow(completed),
    recentLabel: `Last ${RECENT_WINDOW}`,
    sessionLabel: wholeSession ? "Session" : `Last ${completed.length}`,
    drafter: backend && backend !== "none" ? backend : null,
    ngramEnabled: completed.some((row) => row.ngram_drafts > 0),
    exact: false,
    completed: metrics.totals.completed,
    failed: metrics.totals.failed,
    cancelled: metrics.totals.cancelled,
    rejected: metrics.totals.rejected,
  };
}
