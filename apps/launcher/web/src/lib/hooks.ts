import { useMutation, useQuery } from "@tanstack/react-query";
import { useCallback, useEffect, useMemo, useState } from "react";
import { toast } from "sonner";

import { api } from "./api";
import { makeTakesValue, type TakesValue } from "./args";
import type { LauncherState, Library, ServeOptions, Settings } from "./api-types";
import { useLive } from "./events";
import { readStorage, writeStorage } from "./utils";

/** The current time, refreshed every `intervalMs`. */
export function useNow(intervalMs = 1000): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), intervalMs);
    return () => window.clearInterval(id);
  }, [intervalMs]);
  return now;
}

export function useDebounced<T>(value: T, delayMs: number): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const id = window.setTimeout(() => setDebounced(value), delayMs);
    return () => window.clearTimeout(id);
  }, [value, delayMs]);
  return debounced;
}

export function useLauncherState() {
  return useQuery<LauncherState>({ queryKey: ["state"], queryFn: api.state, staleTime: 30_000 });
}

export function useSettings() {
  return useQuery<Settings>({ queryKey: ["settings"], queryFn: api.settings, staleTime: 30_000 });
}

export function useLibrary() {
  return useQuery<Library>({ queryKey: ["library"], queryFn: api.library, staleTime: 60_000 });
}

export function useServeOptions() {
  return useQuery<ServeOptions>({
    queryKey: ["serve-options"],
    queryFn: api.serveOptions,
    staleTime: 5 * 60_000,
    retry: 1,
  });
}

/** Flags infernix-serve lists (flag -> argument placeholder), or null until known. */
export function useOptionMap(): ReadonlyMap<string, string | null> | null {
  const { data } = useServeOptions();
  return useMemo(() => {
    if (!data || !data.found || data.options.length === 0) return null;
    return new Map(data.options.map((option) => [option.flag, option.arg] as const));
  }, [data]);
}

export function useTakesValue(): TakesValue {
  const options = useOptionMap();
  return useMemo(() => makeTakesValue(options), [options]);
}

/** Opens the log folder in the file manager of the computer running the launcher. */
export function useOpenLogFolder() {
  return useMutation({
    mutationFn: api.openLogFolder,
    onSuccess: () => toast.success("Opened the log folder"),
  });
}

/** Endpoint URLs as this browser reaches the launcher (falls back to the page origin). */
export function useEndpoints(): { openai: string; anthropic: string } {
  const launcher = useLive((live) => live.launcher);
  const fallback = typeof window !== "undefined" ? window.location.origin : "http://127.0.0.1:1234";
  return {
    openai: launcher?.endpoints.openai ?? `${fallback}/v1`,
    anthropic: launcher?.endpoints.anthropic ?? fallback,
  };
}

/** An on/off choice this browser remembers (a section shown in full, a list expanded). */
export function useStoredFlag(key: string, initial: boolean): [boolean, (value: boolean) => void] {
  const [value, setValue] = useState(() => {
    const stored = readStorage(key);
    return stored === null ? initial : stored === "1";
  });
  const set = useCallback(
    (next: boolean) => {
      setValue(next);
      writeStorage(key, next ? "1" : "0");
    },
    [key],
  );
  return [value, set];
}

/** Matches a min-width media query (e.g. Tailwind's md = 48rem). */
export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(() =>
    typeof window !== "undefined" ? window.matchMedia(query).matches : false,
  );
  useEffect(() => {
    const media = window.matchMedia(query);
    const onChange = () => setMatches(media.matches);
    onChange();
    media.addEventListener("change", onChange);
    return () => media.removeEventListener("change", onChange);
  }, [query]);
  return matches;
}
