import { useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { EllipsisVertical, LoaderCircle, MessageSquarePlus, Pencil, Trash2 } from "lucide-react";

import { ConfirmDialog } from "@/components/common";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { errorMessage } from "@/lib/api";
import type { ChatSummary } from "@/lib/api-types";
import { chatTitle, modelName, removeChat, renameChat, useChatSummaries, useStreamingChats, type ChatEntry } from "@/lib/chat";
import { formatRelative } from "@/lib/format";
import { useLibrary, useNow } from "@/lib/hooks";
import { cn } from "@/lib/utils";

function RenameField({ initial, onDone }: { initial: string; onDone: (title: string | null) => void }) {
  const [value, setValue] = useState(initial);
  const ref = useRef<HTMLInputElement>(null);
  const done = useRef(false);
  const finish = (title: string | null) => {
    if (done.current) return;
    done.current = true;
    onDone(title);
  };
  useEffect(() => {
    // After the menu that opened it has closed and handed focus back.
    const id = window.requestAnimationFrame(() => {
      ref.current?.focus();
      ref.current?.select();
    });
    return () => window.cancelAnimationFrame(id);
  }, []);
  return (
    <div className="px-1 py-1">
      <Input
        ref={ref}
        value={value}
        maxLength={200}
        aria-label="Chat title"
        className="h-8 text-sm"
        onChange={(event) => setValue(event.target.value)}
        onBlur={() => finish(value)}
        onKeyDown={(event) => {
          if (event.key === "Enter") {
            event.preventDefault();
            finish(value);
          } else if (event.key === "Escape") {
            event.preventDefault();
            finish(null);
          }
        }}
      />
    </div>
  );
}

function ChatRow({
  summary,
  selected,
  streaming,
  model,
  now,
  onSelect,
  onRename,
  onDelete,
}: {
  summary: ChatSummary;
  selected: boolean;
  streaming: boolean;
  model: string;
  now: number;
  onSelect: () => void;
  onRename: () => void;
  onDelete: () => void;
}) {
  const renameRequested = useRef(false);
  return (
    <div
      className={cn(
        "group relative flex items-center rounded-md transition-colors",
        selected ? "bg-accent text-accent-foreground" : "hover:bg-accent/60",
      )}
    >
      <button
        type="button"
        aria-current={selected ? "page" : undefined}
        className="min-w-0 flex-1 rounded-md py-2 pr-1 pl-2.5 text-left outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50"
        onClick={onSelect}
      >
        <div className="flex items-center gap-1.5">
          <span className="truncate text-sm font-medium">{chatTitle(summary)}</span>
          {streaming ? <LoaderCircle className="size-3 shrink-0 animate-spin text-primary" aria-label="Replying" /> : null}
        </div>
        <div className="mt-0.5 flex min-w-0 items-center gap-1 text-[11px] text-muted-foreground">
          <span className="truncate">{model}</span>
          <span className="shrink-0">· {formatRelative(summary.updated_at, now)}</span>
        </div>
      </button>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button
            variant="ghost"
            size="icon-xs"
            aria-label={`Actions for ${chatTitle(summary)}`}
            className="mr-1 shrink-0 opacity-0 group-hover:opacity-100 focus-visible:opacity-100 data-[state=open]:opacity-100 pointer-coarse:opacity-100"
          >
            <EllipsisVertical />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent
          align="end"
          className="w-40"
          onCloseAutoFocus={(event) => {
            // The rename field takes the focus instead of the menu button.
            if (renameRequested.current) {
              event.preventDefault();
              renameRequested.current = false;
            }
          }}
        >
          <DropdownMenuItem
            onSelect={() => {
              renameRequested.current = true;
              onRename();
            }}
          >
            <Pencil />
            Rename
          </DropdownMenuItem>
          <DropdownMenuItem variant="destructive" onSelect={onDelete}>
            <Trash2 className="text-destructive" />
            Delete
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  );
}

/** The stored chats, newest first, with the new chat being written at the top. */
export function ChatList({
  selectedId,
  draft,
  onSelect,
  onNew,
  onDeleted,
  className,
  headerClassName,
}: {
  selectedId: string | null;
  /** The open chat while it is not stored yet (not in the list). */
  draft: ChatEntry | null;
  onSelect: (id: string) => void;
  onNew: () => void;
  /** A chat was deleted (the page moves off it when it was open). */
  onDeleted: (id: string) => void;
  className?: string;
  headerClassName?: string;
}) {
  const qc = useQueryClient();
  const summaries = useChatSummaries();
  const { data: library } = useLibrary();
  const streaming = useStreamingChats();
  const now = useNow(30_000);
  const [renaming, setRenaming] = useState<string | null>(null);
  const [deleting, setDeleting] = useState<ChatSummary | null>(null);

  const list = summaries.data ?? [];

  return (
    <div className={cn("flex h-full min-h-0 flex-col", className)}>
      <div className={cn("flex items-center gap-2 border-b px-3 py-2.5", headerClassName)}>
        <div className="text-sm font-semibold">Chats</div>
        <Button size="xs" variant="outline" className="ml-auto" onClick={onNew}>
          <MessageSquarePlus />
          New chat
        </Button>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto p-2">
        <div className="flex flex-col gap-0.5">
          {draft ? (
            <div className="rounded-md bg-accent px-2.5 py-2 text-accent-foreground" aria-current="page">
              <div className="flex items-center gap-1.5">
                <span className="truncate text-sm font-medium">{chatTitle(draft.chat)}</span>
                {draft.reply ? <LoaderCircle className="size-3 shrink-0 animate-spin text-primary" aria-label="Replying" /> : null}
              </div>
              <div className="mt-0.5 truncate text-[11px] text-muted-foreground">{modelName(library, draft.chat.model)}</div>
            </div>
          ) : null}
          {summaries.isPending ? (
            Array.from({ length: 4 }, (_, index) => <Skeleton key={index} className="h-12 w-full rounded-md" />)
          ) : summaries.isError ? (
            <div className="px-2.5 py-3 text-xs text-muted-foreground">
              The chats could not be read: {errorMessage(summaries.error)}
              <Button variant="link" size="xs" className="h-auto px-1" onClick={() => void summaries.refetch()}>
                Retry
              </Button>
            </div>
          ) : list.length === 0 && !draft ? (
            <div className="px-2.5 py-3 text-xs text-muted-foreground">No chats yet.</div>
          ) : (
            list.map((summary) =>
              renaming === summary.id ? (
                <RenameField
                  key={summary.id}
                  initial={summary.title}
                  onDone={(title) => {
                    setRenaming(null);
                    if (title !== null && title.trim() !== summary.title) void renameChat(qc, summary.id, title);
                  }}
                />
              ) : (
                <ChatRow
                  key={summary.id}
                  summary={summary}
                  selected={summary.id === selectedId}
                  streaming={streaming.includes(summary.id)}
                  model={modelName(library, summary.model)}
                  now={now}
                  onSelect={() => onSelect(summary.id)}
                  onRename={() => setRenaming(summary.id)}
                  onDelete={() => setDeleting(summary)}
                />
              ),
            )
          )}
        </div>
      </div>
      <ConfirmDialog
        open={deleting !== null}
        onOpenChange={(open) => {
          if (!open) setDeleting(null);
        }}
        title="Delete this chat?"
        description={
          <>
            <span className="font-medium text-foreground">{deleting ? chatTitle(deleting) : ""}</span> and its{" "}
            {deleting?.messages ?? 0} messages are removed from the launcher's folder. This cannot be undone.
          </>
        }
        confirmLabel="Delete"
        destructive
        onConfirm={() => {
          if (!deleting) return;
          const id = deleting.id;
          void removeChat(qc, id);
          onDeleted(id);
        }}
      />
    </div>
  );
}
