// Launch parameters are kept as the argv array the launcher passes to infernix-serve. The form
// controls read and write single flags in that array; the text view tokenizes and quotes it.

/** Whether a flag takes a value: true/false when known, undefined when unknown. */
export type TakesValue = (flag: string) => boolean | undefined;

export interface ArgEntry {
  /** null for a stray token that follows no flag. */
  flag: string | null;
  /** null for a switch. */
  value: string | null;
}

/** Splits text into arguments. Double or single quotes group; inside double quotes `\"` is a
 * literal quote. Backslashes are otherwise literal, so Windows paths need no escaping. */
export function tokenize(text: string): string[] {
  const out: string[] = [];
  let current = "";
  let inToken = false;
  let quote: '"' | "'" | null = null;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    if (quote !== null) {
      if (ch === quote) {
        quote = null;
      } else if (ch === "\\" && quote === '"' && text[i + 1] === '"') {
        current += '"';
        i += 1;
      } else {
        current += ch;
      }
      continue;
    }
    if (ch === '"' || ch === "'") {
      quote = ch;
      inToken = true;
      continue;
    }
    if (/\s/.test(ch)) {
      if (inToken) {
        out.push(current);
        current = "";
        inToken = false;
      }
      continue;
    }
    current += ch;
    inToken = true;
  }
  if (inToken) out.push(current);
  return out;
}

export function quoteArg(arg: string): string {
  if (arg.length === 0) return '""';
  if (!/[\s"']/.test(arg)) return arg;
  return `"${arg.replace(/"/g, '\\"')}"`;
}

export function parseEntries(args: readonly string[], takesValue: TakesValue): ArgEntry[] {
  const out: ArgEntry[] = [];
  for (let i = 0; i < args.length; i += 1) {
    const token = args[i];
    if (token.startsWith("--")) {
      const eq = token.indexOf("=");
      if (eq > 2) {
        out.push({ flag: token.slice(0, eq), value: token.slice(eq + 1) });
        continue;
      }
      const known = takesValue(token);
      const next = args[i + 1];
      if (known !== false && next !== undefined && !next.startsWith("--")) {
        out.push({ flag: token, value: next });
        i += 1;
        continue;
      }
      out.push({ flag: token, value: null });
      continue;
    }
    out.push({ flag: null, value: token });
  }
  return out;
}

export function entriesToArgs(entries: readonly ArgEntry[]): string[] {
  const out: string[] = [];
  for (const entry of entries) {
    if (entry.flag !== null) out.push(entry.flag);
    if (entry.value !== null) out.push(entry.value);
  }
  return out;
}

/** One option per line, values quoted where needed. */
export function argsToText(args: readonly string[], takesValue: TakesValue): string {
  return parseEntries(args, takesValue)
    .map((entry) => {
      if (entry.flag === null) return quoteArg(entry.value ?? "");
      if (entry.value === null) return entry.flag;
      return `${entry.flag} ${quoteArg(entry.value)}`;
    })
    .join("\n");
}

/** A command line on one line (for copying). */
export function argsToCommandLine(args: readonly string[]): string {
  return args.map(quoteArg).join(" ");
}

/** The flag's last value; `true` for a switch; undefined when absent. */
export function getFlag(args: readonly string[], flag: string, takesValue: TakesValue): string | true | undefined {
  let found: string | true | undefined;
  for (const entry of parseEntries(args, takesValue)) {
    if (entry.flag === flag) found = entry.value ?? true;
  }
  return found;
}

export function getFlagString(args: readonly string[], flag: string, takesValue: TakesValue): string {
  const value = getFlag(args, flag, takesValue);
  return typeof value === "string" ? value : "";
}

export function hasFlag(args: readonly string[], flag: string, takesValue: TakesValue): boolean {
  return getFlag(args, flag, takesValue) !== undefined;
}

/** Sets (or with false/""/null/undefined removes) a flag, keeping its position when present. */
export function setFlag(
  args: readonly string[],
  flag: string,
  value: string | boolean | null | undefined,
  takesValue: TakesValue,
): string[] {
  const entries = parseEntries(args, takesValue);
  const first = entries.findIndex((entry) => entry.flag === flag);
  const rest = entries.filter((entry) => entry.flag !== flag);
  if (value === undefined || value === null || value === false || value === "") {
    return entriesToArgs(rest);
  }
  const entry: ArgEntry = { flag, value: value === true ? null : value };
  if (first >= 0) {
    rest.splice(Math.min(first, rest.length), 0, entry);
  } else {
    rest.push(entry);
  }
  return entriesToArgs(rest);
}

export function sameArgs(a: readonly string[], b: readonly string[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i += 1) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

/** Built-in knowledge of the flags the form edits, used until serve/options arrives and for
 * flags an older infernix-serve does not list. */
export const KNOWN_SWITCHES: ReadonlySet<string> = new Set([
  "--any-model-name",
  "--vram-past-budget",
  "--expert-csf",
  "--reproducible",
  "--lossless",
  "--expert-a4-activations",
  "--a8-activations",
  "--embedding-host",
  "--lm-head-draft",
  "--no-thinking-answer-room",
  "--vision",
  "--no-cuda-graph",
  "--no-prefix-reuse",
  "--no-queue-holds",
  "--no-thinking",
  "--preserve-thinking",
  "--tolerant-tool-calls",
  "--restore-tool-reasoning",
  "--cors",
  "--usage-chunk-choice",
  "--claude-code-usage-compat",
  "--greedy",
  "--ngram-native-sessions",
  "--use-original-int8-prefill-kernel",
  "--use-original-nvfp4-prefill-kernel",
]);

export function makeTakesValue(optionArgs: ReadonlyMap<string, string | null> | null): TakesValue {
  return (flag: string) => {
    if (optionArgs !== null && optionArgs.has(flag)) return optionArgs.get(flag) !== null;
    if (KNOWN_SWITCHES.has(flag)) return false;
    return undefined;
  };
}
