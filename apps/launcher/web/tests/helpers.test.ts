// The UI's pure helpers whose rules the launcher's server shares or whose output users read: elapsed-time
// formatting (U-3), the wildcard rule of the model routes (U-6, the server's routing.cpp wildcard_match) and what
// counts as a model repository (U-7). Run by `npm test` (scripts/test.mjs).
import { matchesPattern } from "@/lib/admin";
import { formatSeconds } from "@/lib/format";
import { parseRepo } from "@/lib/utils";

export function run(): string[] {
  const failures: string[] = [];
  const check = (ok: boolean, what: string) => {
    if (!ok) failures.push(what);
  };

  // U-3: rounded to whole seconds before minutes and hours are split, so 59.96 s is never "60.0 s" or "0m 60s".
  const seconds: [number, string][] = [
    [0.25, "250 ms"],
    [9.5, "9.50 s"],
    [59.9, "59.9 s"],
    [59.96, "1m 00s"],
    [60, "1m 00s"],
    [119.4, "1m 59s"],
    [119.6, "2m 00s"],
    [3599.4, "59m 59s"],
    [3599.6, "1h 00m"],
    [3600, "1h 00m"],
    [7170, "2h 00m"],
    [5399.9, "1h 30m"],
  ];
  for (const [value, expected] of seconds) {
    const got = formatSeconds(value);
    check(got === expected, `formatSeconds(${value}) = ${got}, expected ${expected}`);
  }

  // U-6: the server's wildcard rule (ASCII case folding, '*' only), linear on a pathological pattern.
  const patterns: [string, string, boolean][] = [
    ["qwen*", "Qwen3.8-27B", true],
    ["*27b*", "qwen3.8-27B-nvfp4", true],
    ["*", "", true],
    ["", "", true],
    ["", "a", false],
    ["a*b", "ab", true],
    ["a*b", "acb", true],
    ["a*b", "acbd", false],
    ["a.b", "axb", false],
    ["a(b)", "A(B)", true],
    ["*a*a*a*a*a*b", "a".repeat(40), false],
    ["*a*a*a*a*a*b", "a".repeat(40) + "b", true],
  ];
  for (const [pattern, name, expected] of patterns) {
    check(matchesPattern(pattern, name) === expected, `matchesPattern(${pattern}, ${name}) is not ${expected}`);
  }
  const start = performance.now();
  matchesPattern("*a".repeat(30) + "b", "a".repeat(4000));
  const elapsed = performance.now() - start;
  check(elapsed < 200, `a 61-character wildcard pattern over 4,000 characters took ${elapsed.toFixed(1)} ms`);

  // U-7: datasets, Spaces and models/ listing pages are no model repository.
  const repos: [string, string | null][] = [
    ["owner/name", "owner/name"],
    ["https://huggingface.co/owner/name/tree/main", "owner/name"],
    ["hf.co/owner/name", "owner/name"],
    ["https://huggingface.co/datasets/owner/name", null],
    ["https://huggingface.co/spaces/owner/name", null],
    ["https://huggingface.co/models/owner", null],
  ];
  for (const [input, expected] of repos) {
    check(parseRepo(input) === expected, `parseRepo(${input}) = ${parseRepo(input)}, expected ${expected}`);
  }
  return failures;
}
