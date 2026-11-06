import { memo, useEffect, useLayoutEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { Brain, ChevronRight, CircleAlert, LoaderCircle, Pencil, RotateCcw, Trash2 } from "lucide-react";

import { CopyButton } from "@/components/CopyButton";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Hint } from "@/components/ui/tooltip";
import type { ChatMessage } from "@/lib/api-types";
import { deleteMessage, editAndResend, regenerate, type LiveStats } from "@/lib/chat";
import { useLive } from "@/lib/events";
import { formatInt, formatPercent } from "@/lib/format";
import { cn } from "@/lib/utils";
import { Markdown } from "./Markdown";
import { MessageStats } from "./ReplyStats";

/** Message actions show on hover or focus (always on touch screens and on the last message). */
const ACTIONS =
  "flex items-center gap-0.5 opacity-0 transition-opacity group-hover:opacity-100 focus-within:opacity-100 pointer-coarse:opacity-100";

interface MessageViewProps {
  chatId: string;
  message: ChatMessage;
  /** The model id of the configuration that answered (assistant messages). */
  modelLabel: string;
  /** Set while the reply streams into this message. */
  live: LiveStats | null;
  /** A reply is streaming in this chat: messages cannot change meanwhile. */
  busy: boolean;
  /** The chat's last message (a reply there can be regenerated). */
  last: boolean;
  onOpenImage: (src: string) => void;
}

function EditBox({
  initial,
  allowEmpty,
  onCancel,
  onSend,
}: {
  initial: string;
  /** The message has images, so it may go without text. */
  allowEmpty: boolean;
  onCancel: () => void;
  onSend: (text: string) => void;
}) {
  const [text, setText] = useState(initial);
  const ref = useRef<HTMLTextAreaElement>(null);
  useEffect(() => {
    const area = ref.current;
    if (!area) return;
    area.focus();
    area.setSelectionRange(area.value.length, area.value.length);
  }, []);
  const ready = text.trim().length > 0 || allowEmpty;
  const send = () => {
    if (ready) onSend(text.trim());
  };
  return (
    <div className="w-full max-w-[85%] rounded-2xl border bg-card p-2 shadow-xs">
      <Textarea
        ref={ref}
        value={text}
        aria-label="Edit the message"
        rows={Math.min(12, Math.max(2, text.split("\n").length))}
        className="min-h-0 resize-none border-0 bg-transparent shadow-none focus-visible:ring-0 dark:bg-transparent"
        onChange={(event) => setText(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Escape") {
            event.preventDefault();
            onCancel();
          } else if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
            event.preventDefault();
            send();
          }
        }}
      />
      <div className="mt-1 flex items-center justify-end gap-2">
        <Button size="xs" variant="ghost" onClick={onCancel}>
          Cancel
        </Button>
        <Button size="xs" disabled={!ready} onClick={send}>
          Send
        </Button>
      </div>
    </div>
  );
}

function UserMessage({ chatId, message, busy, last, onOpenImage }: MessageViewProps) {
  const qc = useQueryClient();
  const [editing, setEditing] = useState(false);
  return (
    <div className="group flex flex-col items-end gap-1.5">
      {message.images.length > 0 ? (
        <div className="flex max-w-[85%] flex-wrap justify-end gap-2">
          {message.images.map((src, index) => (
            <button
              key={index}
              type="button"
              className="overflow-hidden rounded-lg border outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50"
              onClick={() => onOpenImage(src)}
              aria-label={`Show image ${index + 1}`}
            >
              <img src={src} alt="" className="h-28 max-w-56 object-cover" />
            </button>
          ))}
        </div>
      ) : null}
      {editing ? (
        <EditBox
          initial={message.content}
          allowEmpty={message.images.length > 0}
          onCancel={() => setEditing(false)}
          onSend={(text) => {
            setEditing(false);
            editAndResend(qc, chatId, message.id, text);
          }}
        />
      ) : message.content.length > 0 ? (
        <div className="max-w-[85%] rounded-2xl rounded-br-md bg-secondary px-4 py-2.5 text-sm leading-relaxed break-words whitespace-pre-wrap text-secondary-foreground">
          {message.content}
        </div>
      ) : null}
      {editing ? null : (
        <div className={cn(ACTIONS, last && "opacity-100")}>
          {message.content.length > 0 ? (
            <span>
              <CopyButton text={message.content} size="icon-xs" />
            </span>
          ) : null}
          <Hint label="Edit and send again">
            <Button variant="ghost" size="icon-xs" aria-label="Edit and send again" disabled={busy} onClick={() => setEditing(true)}>
              <Pencil />
            </Button>
          </Hint>
          <Hint label="Delete the message">
            <Button
              variant="ghost"
              size="icon-xs"
              aria-label="Delete the message"
              disabled={busy}
              onClick={() => deleteMessage(qc, chatId, message.id)}
            >
              <Trash2 />
            </Button>
          </Hint>
        </div>
      )}
    </div>
  );
}

/** The reply's thinking: open while it streams, collapsed once the answer starts (unless the
 * user opened or closed it); a reply that ended without an answer keeps it open. */
function ThinkingBlock({ text, live, answered }: { text: string; live: boolean; answered: boolean }) {
  const [choice, setChoice] = useState<boolean | null>(null);
  const open = choice ?? !answered;
  const bodyRef = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => {
    const body = bodyRef.current;
    if (live && open && body) body.scrollTop = body.scrollHeight;
  }, [text, live, open]);
  const words = text.trim().length > 0 ? text.trim().split(/\s+/).length : 0;
  return (
    <div className="rounded-lg border bg-muted/40">
      <button
        type="button"
        aria-expanded={open}
        className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-xs font-medium text-muted-foreground transition-colors outline-none hover:text-foreground focus-visible:ring-[3px] focus-visible:ring-ring/50"
        onClick={() => setChoice(!open)}
      >
        <ChevronRight className={cn("size-3.5 transition-transform", open && "rotate-90")} />
        <Brain className="size-3.5" />
        {live ? <span className="animate-pulse">Thinking…</span> : <span>Thinking</span>}
        <span className="ml-auto tabular-nums">{formatInt(words)} words</span>
      </button>
      {open ? (
        <div ref={bodyRef} className={cn("border-t px-3 py-2 text-muted-foreground", live && "max-h-64 overflow-y-auto")}>
          <Markdown text={text} className="text-[13px]" />
        </div>
      ) : null}
    </div>
  );
}

/** Before any text arrives: queued, a model loading, or the prompt being read. */
function WaitingLine({ live }: { live: LiveStats }) {
  const instance = useLive((state) => state.instance);
  let text = "Waiting for the model";
  const progress = live.progress;
  if (live.phase === "prefill" && progress && progress.total > 0) {
    text = `Reading the prompt: ${formatPercent((progress.processed / progress.total) * 100)} of ${formatInt(progress.total)} tokens`;
  } else if (instance && instance.state === "starting") {
    text = `Loading ${instance.alias ?? "the model"}`;
  }
  return (
    <div className="flex items-center gap-2 text-sm text-muted-foreground">
      <LoaderCircle className="size-4 animate-spin text-primary" />
      {text}
    </div>
  );
}

function AssistantMessage({ chatId, message, modelLabel, live, busy, last }: MessageViewProps) {
  const qc = useQueryClient();
  const reasoning = message.reasoning ?? "";
  const answering = message.content.length > 0;
  return (
    <div className="group flex min-w-0 flex-col gap-2">
      {reasoning.length > 0 ? (
        <ThinkingBlock text={reasoning} live={live !== null && !answering} answered={answering} />
      ) : null}
      {answering ? <Markdown text={message.content} /> : null}
      {live !== null && !answering && reasoning.length === 0 ? <WaitingLine live={live} /> : null}
      {message.error ? (
        <div role="alert" className="flex items-start gap-2 rounded-lg border border-critical/40 bg-critical/8 px-3 py-2 text-sm">
          <CircleAlert className="mt-0.5 size-4 shrink-0 text-critical" />
          <span className="min-w-0 break-words">{message.error}</span>
        </div>
      ) : null}
      {live === null ? (
        <div className="flex min-h-7 min-w-0 items-center gap-2">
          {message.stats ? (
            <MessageStats stats={message.stats} model={modelLabel} />
          ) : (
            <span className="truncate text-[11px] text-muted-foreground">{modelLabel}</span>
          )}
          <div className={cn(ACTIONS, "ml-auto", last && "opacity-100")}>
            {answering ? (
              <span>
                <CopyButton text={message.content} size="icon-xs" />
              </span>
            ) : null}
            {last ? (
              <Hint label="Regenerate">
                <Button variant="ghost" size="icon-xs" aria-label="Regenerate" disabled={busy} onClick={() => regenerate(qc, chatId)}>
                  <RotateCcw />
                </Button>
              </Hint>
            ) : null}
            <Hint label="Delete the reply">
              <Button
                variant="ghost"
                size="icon-xs"
                aria-label="Delete the reply"
                disabled={busy}
                onClick={() => deleteMessage(qc, chatId, message.id)}
              >
                <Trash2 />
              </Button>
            </Hint>
          </div>
        </div>
      ) : null}
    </div>
  );
}

export const MessageView = memo(function MessageView(props: MessageViewProps) {
  return props.message.role === "user" ? <UserMessage {...props} /> : <AssistantMessage {...props} />;
});
