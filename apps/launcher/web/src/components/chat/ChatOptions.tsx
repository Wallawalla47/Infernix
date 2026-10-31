import { useState } from "react";
import { Popover } from "radix-ui";
import { SlidersHorizontal } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { Hint } from "@/components/ui/tooltip";
import type { Chat, ChatOptions } from "@/lib/api-types";

export type ChatSettingsPatch = { system_prompt?: string; options?: ChatOptions };

/** A number field's value: null when empty, undefined when it is not a valid value. */
function parseField(text: string, { integer, min, max }: { integer: boolean; min: number; max: number }): number | null | undefined {
  const trimmed = text.trim();
  if (trimmed.length === 0) return null;
  const value = Number(trimmed);
  if (!Number.isFinite(value) || value < min || value > max || (integer && !Number.isInteger(value))) return undefined;
  return value;
}

const TEMPERATURE = { integer: false, min: 0, max: 2 };
const MAX_TOKENS = { integer: true, min: 1, max: 16_777_216 };

function OptionsForm({ chat, onChange }: { chat: Chat; onChange: (patch: ChatSettingsPatch) => void }) {
  const [system, setSystem] = useState(chat.system_prompt);
  const [temperature, setTemperature] = useState(chat.options.temperature === null ? "" : String(chat.options.temperature));
  const [maxTokens, setMaxTokens] = useState(chat.options.max_tokens === null ? "" : String(chat.options.max_tokens));
  const temperatureValue = parseField(temperature, TEMPERATURE);
  const maxTokensValue = parseField(maxTokens, MAX_TOKENS);

  return (
    <div className="grid gap-4">
      <div className="grid gap-1.5">
        <Label htmlFor="chat-system-prompt">System prompt</Label>
        <Textarea
          id="chat-system-prompt"
          rows={5}
          value={system}
          placeholder="None"
          className="max-h-64 min-h-24 resize-y"
          onChange={(event) => {
            setSystem(event.target.value);
            onChange({ system_prompt: event.target.value });
          }}
        />
      </div>
      <div className="grid grid-cols-2 gap-3">
        <div className="grid gap-1.5">
          <Label htmlFor="chat-temperature">Temperature</Label>
          <Input
            id="chat-temperature"
            type="number"
            inputMode="decimal"
            min={TEMPERATURE.min}
            max={TEMPERATURE.max}
            step={0.05}
            placeholder="Default"
            value={temperature}
            aria-invalid={temperatureValue === undefined}
            onChange={(event) => {
              setTemperature(event.target.value);
              const value = parseField(event.target.value, TEMPERATURE);
              if (value !== undefined) onChange({ options: { ...chat.options, temperature: value } });
            }}
          />
        </div>
        <div className="grid gap-1.5">
          <Label htmlFor="chat-max-tokens">Max tokens</Label>
          <Input
            id="chat-max-tokens"
            type="number"
            inputMode="numeric"
            min={MAX_TOKENS.min}
            step={1}
            placeholder="Default"
            value={maxTokens}
            aria-invalid={maxTokensValue === undefined}
            onChange={(event) => {
              setMaxTokens(event.target.value);
              const value = parseField(event.target.value, MAX_TOKENS);
              if (value !== undefined) onChange({ options: { ...chat.options, max_tokens: value } });
            }}
          />
        </div>
      </div>
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <Label htmlFor="chat-thinking">Thinking</Label>
          <div className="mt-1 text-xs leading-relaxed text-muted-foreground">
            Lets the model reason before it answers (enable_thinking).
          </div>
        </div>
        <Switch
          id="chat-thinking"
          checked={chat.options.thinking}
          onCheckedChange={(checked) => onChange({ options: { ...chat.options, thinking: checked } })}
        />
      </div>
      <p className="text-xs text-muted-foreground">Empty fields use the server's defaults. Changes apply to the next message.</p>
    </div>
  );
}

/** The chat's options in a popover: system prompt, temperature, max tokens, thinking. */
export function ChatOptionsButton({ chat, onChange }: { chat: Chat; onChange: (patch: ChatSettingsPatch) => void }) {
  const customized =
    chat.system_prompt.trim().length > 0 ||
    chat.options.temperature !== null ||
    chat.options.max_tokens !== null ||
    !chat.options.thinking;
  return (
    <Popover.Root>
      <Hint label="Chat options">
        <Popover.Trigger asChild>
          <Button variant="outline" size="icon-sm" aria-label="Chat options" className="relative">
            <SlidersHorizontal />
            {customized ? <span className="absolute top-1 right-1 size-1.5 rounded-full bg-primary" aria-hidden="true" /> : null}
          </Button>
        </Popover.Trigger>
      </Hint>
      <Popover.Portal>
        <Popover.Content
          data-slot="chat-popover"
          align="end"
          sideOffset={6}
          collisionPadding={12}
          className="z-50 w-[min(24rem,calc(100vw-2rem))] rounded-lg border bg-popover p-4 text-popover-foreground shadow-lg outline-none"
        >
          <OptionsForm chat={chat} onChange={onChange} />
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}
