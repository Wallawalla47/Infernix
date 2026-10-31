// Queries of the administration pages (session, usage, updates) and the pure helpers they share.
// The event stream writes the update states under the same keys (see events.ts).

import { useQuery } from "@tanstack/react-query";

import { ApiFailure, api } from "./api";
import type { InfernixRelease, ModelRoute, ModelUpdates, Session, Usage } from "./api-types";

export const SESSION_KEY = ["session"] as const;
export const INFERNIX_RELEASE_KEY = ["infernix-release"] as const;
export const MODEL_UPDATES_KEY = ["model-updates"] as const;

/** A management call refused because this browser must sign in with an admin key. */
export function isSignInFailure(error: unknown): boolean {
  return error instanceof ApiFailure && error.code === "sign_in";
}

export function useSession() {
  return useQuery<Session>({ queryKey: SESSION_KEY, queryFn: api.session, staleTime: 60_000 });
}

export function useInfernixRelease() {
  return useQuery<InfernixRelease>({ queryKey: INFERNIX_RELEASE_KEY, queryFn: api.infernixRelease, staleTime: 5 * 60_000 });
}

export function useModelUpdates() {
  return useQuery<ModelUpdates>({ queryKey: MODEL_UPDATES_KEY, queryFn: api.modelUpdates, staleTime: 5 * 60_000 });
}

export interface DayRange {
  /** Local dates, inclusive (YYYY-MM-DD). */
  from: string;
  to: string;
}

/** Usage over a range of local days; re-read every minute while shown (requests keep adding). */
export function useUsage(range: DayRange) {
  return useQuery<Usage>({
    queryKey: ["usage", range.from, range.to],
    queryFn: () => api.usage(range),
    staleTime: 30_000,
    refetchInterval: 60_000,
  });
}

function pad(value: number): string {
  return String(value).padStart(2, "0");
}

/** A local date as YYYY-MM-DD (the usage API's day). */
export function localDay(date: Date): string {
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;
}

/** The local Date of a YYYY-MM-DD day (midnight). */
export function parseDay(day: string): Date {
  const [year, month, date] = day.split("-").map(Number);
  return new Date(year, month - 1, date);
}

/** Every day from `from` to `to`, inclusive. */
export function daysBetween(range: DayRange): string[] {
  const days: string[] = [];
  const end = parseDay(range.to).getTime();
  // Step by calendar day (not 24 h), so daylight-saving changes do not skip or repeat a day.
  for (let date = parseDay(range.from); date.getTime() <= end && days.length < 400; date.setDate(date.getDate() + 1)) {
    days.push(localDay(date));
  }
  return days;
}

/** Whether a request's `model` matches a route pattern: case-insensitive, `*` matches any run of
 * characters (none included), everything else literally. */
export function matchesPattern(pattern: string, name: string): boolean {
  const escaped = pattern
    .split("*")
    .map((part) => part.replace(/[.+?^${}()|[\]\\]/g, "\\$&"))
    .join(".*");
  return new RegExp(`^${escaped}$`, "is").test(name);
}

/** The first route whose pattern matches `name` (first match wins), with its position. */
export function findRoute(routes: readonly ModelRoute[], name: string): { route: ModelRoute; index: number } | null {
  for (let index = 0; index < routes.length; index += 1) {
    const route = routes[index];
    if (route.pattern.length > 0 && matchesPattern(route.pattern, name)) return { route, index };
  }
  return null;
}
