// The structured launch-parameter form. Each control reads and writes one flag of the args array,
// which stays the single source of truth (unknown flags pass through untouched).

import { useEffect, useState, type ReactNode } from "react";
import { CircleHelp } from "lucide-react";

import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Hint } from "@/components/ui/tooltip";
import { getFlag, setFlag, type TakesValue } from "@/lib/args";
import type { ModelArtifact } from "@/lib/api-types";
import { cn } from "@/lib/utils";

export const UNSET = "__unset__";

interface Base {
  flag: string;
  label: string;
  help: string;
  /** Shown only for models this returns true for. */
  when?: (model: ModelArtifact) => boolean;
}

export type Control =
  | (Base & { kind: "switch" })
  | (Base & { kind: "number"; placeholder?: string; min?: number; max?: number; step?: number })
  | (Base & { kind: "text"; placeholder?: string; mono?: boolean; wide?: boolean })
  | (Base & { kind: "select"; options: { value: string; label: string }[]; unsetLabel: string })
  /** "auto" or a number. */
  | (Base & { kind: "auto"; placeholder?: string; autoDefault?: boolean });

export interface FormContext {
  args: string[];
  setArgs: (args: string[]) => void;
  takesValue: TakesValue;
}

function Row({
  label,
  help,
  flag,
  children,
  wide = false,
}: {
  label: string;
  help: string;
  flag: string;
  children: ReactNode;
  wide?: boolean;
}) {
  return (
    <div className={cn("flex gap-3 py-2.5", wide ? "flex-col" : "flex-col sm:flex-row sm:items-center sm:justify-between")}>
      <div className="min-w-0 sm:max-w-[58%]">
        <div className="flex items-center gap-1.5 text-sm font-medium">
          {label}
          <Hint label={<span className="block max-w-xs leading-relaxed">{help}</span>}>
            <CircleHelp className="size-3.5 cursor-help text-muted-foreground/70" aria-label={help} />
          </Hint>
        </div>
        <code className="mt-0.5 block font-mono text-[11px] text-muted-foreground">{flag}</code>
      </div>
      <div className={cn("min-w-0", wide ? "w-full" : "sm:w-60 sm:shrink-0")}>{children}</div>
    </div>
  );
}

/** A text box that commits its value on blur or Enter, so typing does not rewrite args per key. */
export function CommitInput({
  value,
  onCommit,
  placeholder,
  mono = false,
  type = "text",
  min,
  max,
  step,
  className,
  ariaLabel,
  disabled = false,
}: {
  value: string;
  onCommit: (value: string) => void;
  placeholder?: string;
  mono?: boolean;
  type?: "text" | "number";
  min?: number;
  max?: number;
  step?: number;
  className?: string;
  ariaLabel?: string;
  disabled?: boolean;
}) {
  const [local, setLocal] = useState(value);
  useEffect(() => setLocal(value), [value]);
  return (
    <Input
      value={local}
      type={type}
      min={min}
      max={max}
      step={step}
      disabled={disabled}
      placeholder={placeholder}
      aria-label={ariaLabel}
      spellCheck={false}
      className={cn(mono && "font-mono text-xs", className)}
      onChange={(event) => setLocal(event.target.value)}
      onBlur={() => {
        if (local !== value) onCommit(local.trim());
      }}
      onKeyDown={(event) => {
        if (event.key === "Enter") {
          event.preventDefault();
          if (local !== value) onCommit(local.trim());
        }
      }}
    />
  );
}

export function ControlField({ control, form }: { control: Control; form: FormContext }) {
  const { args, setArgs, takesValue } = form;
  const current = getFlag(args, control.flag, takesValue);
  const set = (value: string | boolean | null) => setArgs(setFlag(args, control.flag, value, takesValue));

  switch (control.kind) {
    case "switch":
      return (
        <Row label={control.label} help={control.help} flag={control.flag}>
          <div className="flex sm:justify-end">
            <Switch checked={current !== undefined} onCheckedChange={(checked) => set(checked)} aria-label={control.label} />
          </div>
        </Row>
      );
    case "number":
      return (
        <Row label={control.label} help={control.help} flag={control.flag}>
          <CommitInput
            type="number"
            value={typeof current === "string" ? current : ""}
            placeholder={control.placeholder ?? "default"}
            min={control.min}
            max={control.max}
            step={control.step}
            ariaLabel={control.label}
            onCommit={(value) => set(value.length > 0 ? value : null)}
          />
        </Row>
      );
    case "text":
      return (
        <Row label={control.label} help={control.help} flag={control.flag} wide={control.wide}>
          <CommitInput
            value={typeof current === "string" ? current : ""}
            placeholder={control.placeholder ?? "unset"}
            mono={control.mono}
            ariaLabel={control.label}
            onCommit={(value) => set(value.length > 0 ? value : null)}
          />
        </Row>
      );
    case "select": {
      const value = typeof current === "string" ? current : UNSET;
      const known = control.options.some((option) => option.value === value);
      return (
        <Row label={control.label} help={control.help} flag={control.flag}>
          <Select value={value} onValueChange={(next) => set(next === UNSET ? null : next)}>
            <SelectTrigger aria-label={control.label}>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={UNSET}>{control.unsetLabel}</SelectItem>
              {control.options.map((option) => (
                <SelectItem key={option.value} value={option.value}>
                  {option.label}
                </SelectItem>
              ))}
              {!known && value !== UNSET ? <SelectItem value={value}>{value}</SelectItem> : null}
            </SelectContent>
          </Select>
        </Row>
      );
    }
    case "auto": {
      const text = typeof current === "string" ? current : "";
      const isAuto = text === "auto";
      return (
        <Row label={control.label} help={control.help} flag={control.flag}>
          <div className="flex items-center gap-2">
            <CommitInput
              value={isAuto ? "" : text}
              placeholder={isAuto ? "auto" : (control.placeholder ?? "default")}
              ariaLabel={control.label}
              className={cn(isAuto && "opacity-60")}
              onCommit={(value) => set(value.length > 0 ? value : isAuto ? "auto" : null)}
            />
            <label className="flex shrink-0 items-center gap-1.5 text-xs text-muted-foreground">
              <Switch
                checked={isAuto}
                onCheckedChange={(checked) => set(checked ? "auto" : null)}
                aria-label={`${control.label}: auto`}
              />
              auto
            </label>
          </div>
        </Row>
      );
    }
  }
}

const offloaded = (model: ModelArtifact) => model.offloaded_experts;
const flashNext = (model: ModelArtifact) => model.needs_ngram_volume;
const qwen35 = (model: ModelArtifact) => !model.needs_ngram_volume;

export const KV_OPTIONS = [
  { value: "int8", label: "INT8 (recommended)" },
  { value: "fp8", label: "FP8" },
  { value: "nvfp4", label: "NVFP4" },
  { value: "k8v4", label: "K8V4" },
  { value: "vq2", label: "VQ2 (2-bit, 3-4x context)" },
  { value: "k4v2", label: "K4V2 (4-bit K)" },
  { value: "bf16p", label: "BF16 packed (exact, 23 % smaller)" },
  { value: "bf16", label: "BF16" },
];

export const MEMORY_CONTROLS: Control[] = [
  {
    kind: "select",
    flag: "--kv-dtype",
    label: "KV cache format",
    help: "Storage of the attention KV cache. INT8 holds twice BF16's context at the measured quality; vq2 and k4v2 hold three to four times INT8's. BF16 packed keeps BF16's exact values in 23 % less memory (a request whose KV row cannot be packed fails).",
    options: KV_OPTIONS.filter((option) => option.value !== "bf16"),
    unsetLabel: "BF16 (default)",
  },
  {
    kind: "auto",
    flag: "--kv-capacity",
    label: "KV capacity",
    help: "Shared KV pool in tokens; auto fills the VRAM the model leaves free (with offloaded experts, max context x concurrency).",
    placeholder: "auto (default)",
  },
  {
    kind: "auto",
    flag: "--host-context-mib",
    label: "Host prefix cache (MiB)",
    help: "Pinned RAM for the prefix cache's Host tier (KV blocks and state snapshots). Flash-Next's auto sizes it after the experts.",
    placeholder: "8192 (default)",
  },
  {
    kind: "number",
    flag: "--prefill-chunk",
    label: "Prefill chunk",
    help: "Tokens per prefill step; rounded down to whole attention waves with INT8 and other quantized KV.",
    placeholder: "default",
    min: 128,
    step: 128,
  },
  {
    kind: "auto",
    flag: "--vram-headroom-mib",
    label: "VRAM headroom (MiB)",
    help: "VRAM left free after startup sizing for the display and other programs (auto: 1024 with a display on the GPU, 256 without).",
    placeholder: "auto (default)",
  },
  {
    kind: "switch",
    flag: "--vram-past-budget",
    label: "Use VRAM past the Windows budget",
    help: "Offloaded experts on Windows: also size the expert cache into the ~0.6 GiB the per-process budget withholds (~240 more cached experts).",
    when: offloaded,
  },
  {
    kind: "select",
    flag: "--expert-ram-layout",
    label: "Expert RAM layout",
    help: "exclusive keeps in RAM only the experts VRAM does not hold; full keeps a RAM copy of every expert (4-6 GiB more on Flash-Next, same output).",
    options: [
      { value: "exclusive", label: "exclusive" },
      { value: "full", label: "full" },
    ],
    unsetLabel: "Default for the model",
    when: offloaded,
  },
  {
    kind: "auto",
    flag: "--expert-ram-mib",
    label: "Expert RAM (MiB)",
    help: "RAM for the routed experts; less than all of them reads the rest from the artifact (SSD expert tier).",
    placeholder: "auto (default)",
    when: offloaded,
  },
  {
    kind: "number",
    flag: "--ram-free-mib",
    label: "RAM left free (MiB)",
    help: "Physical RAM everything Infernix keeps in RAM leaves free for other programs.",
    placeholder: "8192 (default)",
    min: 0,
  },
  {
    kind: "switch",
    flag: "--embedding-host",
    label: "Token embedding in RAM",
    help: "Keep the token embedding in pinned host memory so its VRAM (1.18 GiB on the 27B) goes to the KV and prefix caches. Same output.",
    when: (model) => qwen35(model) && !model.offloaded_experts,
  },
  {
    kind: "number",
    flag: "--rope-yarn-factor",
    label: "YaRN factor",
    help: "Startup-fixed YaRN factor in [1, 4]; extends the allowed context beyond the model's native length.",
    placeholder: "1 (default)",
    min: 1,
    max: 4,
    step: 0.25,
  },
];

export const SPEC_CONTROLS: Control[] = [
  {
    kind: "auto",
    flag: "--draft-tokens",
    label: "Draft tokens",
    help: "Drafts verified per round; auto lets the drafter's calibrated probabilities pick the fastest width each round. Speculative decoding Auto takes auto.",
    placeholder: "number",
  },
  {
    kind: "switch",
    flag: "--lm-head-draft",
    label: "Optimized proposal head",
    help: "Load the artifact's proposal head for drafting (--lm-head-draft).",
    when: (model) => model.has_proposal,
  },
  {
    kind: "number",
    flag: "--spec-min-p",
    label: "Draft minimum probability",
    help: "Verify drafts up to the first whose drafter probability is below P (0 verifies all). Refused with speculative decoding Auto, and with --draft-tokens auto on Flash-Next MTP.",
    placeholder: "0",
    min: 0,
    max: 1,
    step: 0.05,
  },
  {
    kind: "auto",
    flag: "--ngram-draft-tokens",
    label: "N-gram copy drafts",
    help: "Verified n-gram copy proposals per round beside the drafter (0 disables); auto takes the configuration's measured window.",
    placeholder: "0 (off)",
  },
  {
    kind: "text",
    flag: "--draft-tree-nodes",
    label: "Tree verification",
    help: "auto lets each round choose a chain or a tree of drafts; a list fixes the columns per batch size.",
    placeholder: "off",
    mono: true,
    when: qwen35,
  },
];

export const NGRAM_CONTROLS: Control[] = [
  {
    kind: "select",
    flag: "--ngram-residency",
    label: "N-gram table residency",
    help: "ram reads the whole table into RAM at startup; volume reads rows on demand; auto takes RAM only when every expert still fits.",
    options: [
      { value: "auto", label: "auto" },
      { value: "ram", label: "ram" },
      { value: "volume", label: "volume" },
    ],
    unsetLabel: "auto (default)",
    when: flashNext,
  },
];

export const THINKING_CONTROLS: Control[] = [
  {
    kind: "switch",
    flag: "--preserve-thinking",
    label: "Preserve thinking",
    help: "Keep closed-turn assistant reasoning in the prompt by default (better prefix-cache reuse for agents).",
  },
  { kind: "switch", flag: "--no-thinking", label: "Thinking off by default", help: "Requests think only when they ask to." },
  {
    kind: "number",
    flag: "--default-thinking-budget",
    label: "Default thinking budget",
    help: "Positive thinking-token cap for thinking requests that set none. The output limit counts the thinking, so the cap is lowered where it would leave the answer less than a quarter of it.",
    placeholder: "unlimited",
    min: 1,
  },
  {
    kind: "text",
    flag: "--thinking-budget-message",
    label: "Budget wrap-up message",
    help: "Text committed when a thinking budget ends thinking (replaces Qwen's guidance).",
    placeholder: "Qwen guidance (default)",
    wide: true,
  },
  {
    kind: "switch",
    flag: "--no-thinking-answer-room",
    label: "Let thinking use the whole output limit",
    help: "Thinking may use all of the output limit instead of leaving a quarter for the answer (with no budget, under the default thinking budget, or with a request budget at or above the limit).",
  },
  {
    kind: "select",
    flag: "--reasoning-loop",
    label: "Reasoning-loop guard",
    help: "stop or conclude a thinking request that starts repeating itself.",
    options: [
      { value: "stop", label: "stop" },
      { value: "conclude", label: "conclude" },
      { value: "off", label: "off" },
    ],
    unsetLabel: "off (default)",
  },
  {
    kind: "switch",
    flag: "--tolerant-tool-calls",
    label: "Tolerant tool calls",
    help: "Recover complete tool calls cut by a malformed wrapper, a trailing suffix or the output budget, and calls stranded in unclosed thinking.",
  },
  {
    kind: "switch",
    flag: "--restore-tool-reasoning",
    label: "Restore tool-call reasoning",
    help: "Give Chat Completions tool-call messages sent back without reasoning the reasoning this server generated for them.",
  },
  {
    kind: "number",
    flag: "--default-max-tokens",
    label: "Default output limit",
    help: "Output tokens when a request sets no limit.",
    placeholder: "8192 (default)",
    min: 1,
  },
  {
    kind: "number",
    flag: "--pending-timeout-ms",
    label: "Queue timeout (ms)",
    help: "Longest a request may wait for admission. Agents sending several requests at once benefit from minutes.",
    placeholder: "30000 (default)",
    min: 0,
  },
  {
    kind: "number",
    flag: "--max-pending-requests",
    label: "Queued requests",
    help: "Requests allowed to wait for a lane.",
    placeholder: "16 (default)",
    min: 0,
  },
  {
    kind: "select",
    flag: "--claude-code-prompt",
    label: "Claude Code system prompt",
    help: "Use Claude Code's system prompt as sent, or replace its Claude-specific parts with a short local-agent preamble.",
    options: [
      { value: "as-sent", label: "as sent" },
      { value: "local", label: "local preamble" },
    ],
    unsetLabel: "as sent (default)",
  },
  {
    kind: "text",
    flag: "--chat-template",
    label: "Chat template file",
    help: "Replace the artifact's chat template at startup (e.g. tools/chat_templates/qwen3_8.jinja).",
    placeholder: "artifact template",
    mono: true,
    wide: true,
  },
];

export const SAMPLING_CONTROLS: Control[] = [
  { kind: "number", flag: "--temperature", label: "Temperature", help: "Process-level override; request fields still win.", placeholder: "model preset", min: 0, step: 0.05 },
  { kind: "number", flag: "--top-p", label: "Top-p", help: "Process-level override.", placeholder: "model preset", min: 0, max: 1, step: 0.01 },
  { kind: "number", flag: "--top-k", label: "Top-k", help: "Process-level override (0..20; 0 = top-20 cap).", placeholder: "model preset", min: 0, max: 20 },
  { kind: "number", flag: "--min-p", label: "Min-p", help: "Process-level override.", placeholder: "model preset", min: 0, max: 1, step: 0.01 },
  { kind: "number", flag: "--presence-penalty", label: "Presence penalty", help: "Process-level override.", placeholder: "model preset", step: 0.1 },
  { kind: "number", flag: "--frequency-penalty", label: "Frequency penalty", help: "Process-level override.", placeholder: "0", step: 0.1 },
  { kind: "number", flag: "--seed", label: "Seed", help: "Fixed seed when a request omits one.", placeholder: "random" },
  { kind: "switch", flag: "--greedy", label: "Greedy", help: "Force exact argmax for every request." },
];

export const VISION_CONTROLS: Control[] = [
  {
    kind: "switch",
    flag: "--vision",
    label: "Accept images and video",
    help: "Load the vision encoder; without it media requests fail with vision_disabled.",
    when: (model) => model.has_vision,
  },
  {
    kind: "select",
    flag: "--vision-offload",
    label: "Vision encoder in RAM",
    help: "Keep the vision tower in pinned RAM and borrow VRAM only while encoding (auto: on for offloaded-expert models).",
    options: [
      { value: "auto", label: "auto" },
      { value: "on", label: "on" },
      { value: "off", label: "off" },
    ],
    unsetLabel: "auto (default)",
    when: (model) => model.has_vision,
  },
  {
    kind: "number",
    flag: "--vision-max-merged",
    label: "Max vision tokens per item",
    help: "Merged vision tokens per image or video (64..32768).",
    placeholder: "32768 (default)",
    min: 64,
    max: 32768,
    when: (model) => model.has_vision,
  },
];

export const ADVANCED_CONTROLS: Control[] = [
  {
    kind: "switch",
    flag: "--reproducible",
    label: "Reproducible route",
    help: "Deterministic and request-invariant output; turns the qualified fast paths off.",
  },
  {
    kind: "switch",
    flag: "--lossless",
    label: "Lossless route",
    help: "Every operation correctly rounded: output depends on the request and artifact alone. Needs BF16 KV and no speculation; slower.",
  },
  {
    kind: "select",
    flag: "--logits-dtype",
    label: "Logits",
    help: "fp32 sums without a final rounding, or bf16 as Hugging Face returns them.",
    options: [
      { value: "fp32", label: "fp32" },
      { value: "bf16", label: "bf16" },
    ],
    unsetLabel: "fp32 (default)",
  },
  {
    kind: "switch",
    flag: "--a8-activations",
    label: "8-bit activations (approximate)",
    help: "Faster on GGUF imports and some routes; changes the output slightly. Refused where it has no route.",
  },
  {
    kind: "switch",
    flag: "--expert-a4-activations",
    label: "4-bit expert activations (approximate)",
    help: "NVIDIA FP4-Local only: W4A4 experts; faster, changes the output.",
    when: (model) => /fp4[-_]?local/i.test(`${model.recipe ?? ""} ${model.name} ${model.file_name}`),
  },
  {
    kind: "number",
    flag: "--expert-tail-skip",
    label: "Expert tail skip (approximate)",
    help: "Skip missing experts every token of a round ranks at R or lower: faster decode, slightly worse output.",
    placeholder: "off",
    min: 1,
    when: offloaded,
  },
  {
    kind: "number",
    flag: "--cache-next-turn-ms",
    label: "Next-turn prefill (ms)",
    help: "After a reply, once idle this long, prefill the conversation's next turn (0 off).",
    placeholder: "model default",
    min: 0,
  },
  {
    kind: "number",
    flag: "--prefill-split-workspace-mib",
    label: "Prefill split workspace (MiB)",
    help: "Memory for splitting prompt attention across SMs (128-384 recommended).",
    placeholder: "256 (default)",
    min: 0,
  },
  { kind: "switch", flag: "--no-cuda-graph", label: "Disable CUDA graphs", help: "Debugging only: slower decode." },
  { kind: "switch", flag: "--no-prefix-reuse", label: "Disable the prefix cache", help: "Every request prefills from the root." },
  { kind: "switch", flag: "--usage-chunk-choice", label: "Usage chunk with a choice", help: "For strict client parsers that reject an empty choices array." },
  {
    kind: "select",
    flag: "--log-level",
    label: "Log level",
    help: "Console verbosity of the server.",
    options: ["trace", "debug", "info", "warning", "error"].map((value) => ({ value, label: value })),
    unsetLabel: "info (default)",
  },
  {
    kind: "text",
    flag: "--request-log-jsonl",
    label: "Request log file",
    help: "Append full-precision server and request records (JSON lines).",
    placeholder: "off",
    mono: true,
    wide: true,
  },
];

/** Every flag the form renders (the help drawer marks the rest as "other"). */
export const FORM_FLAGS: ReadonlySet<string> = new Set(
  [
    "--max-context",
    "--max-concurrency",
    "--spec",
    "--draft-model",
    "--ngram-volume",
    ...MEMORY_CONTROLS.map((c) => c.flag),
    ...SPEC_CONTROLS.map((c) => c.flag),
    ...NGRAM_CONTROLS.map((c) => c.flag),
    ...THINKING_CONTROLS.map((c) => c.flag),
    ...SAMPLING_CONTROLS.map((c) => c.flag),
    ...VISION_CONTROLS.map((c) => c.flag),
    ...ADVANCED_CONTROLS.map((c) => c.flag),
  ],
);
