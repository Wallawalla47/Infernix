import { useEffect, useMemo, useRef, useState, type DragEvent, type ReactNode } from "react";
import { useSearchParams } from "react-router";
import { useQueryClient } from "@tanstack/react-query";
import { ImagePlus, LoaderCircle, MessageSquarePlus, MessagesSquare, PanelLeft, RefreshCw } from "lucide-react";

import { ChatList } from "@/components/chat/ChatList";
import { ChatOptionsButton } from "@/components/chat/ChatOptions";
import { Composer, type ComposerHandle } from "@/components/chat/Composer";
import { Conversation } from "@/components/chat/Conversation";
import { ModelPicker, ModelStatus } from "@/components/chat/ModelPicker";
import { ReplyStatsBar } from "@/components/chat/ReplyStats";
import { EmptyState } from "@/components/common";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetDescription, SheetTitle } from "@/components/ui/sheet";
import {
  CHAT_ID_PATTERN,
  chatTitle,
  createDraft,
  discardDraft,
  imageSupport,
  lastOpenChat,
  latestStats,
  openChat,
  retryOpen,
  sendMessage,
  setChatModel,
  setChatSettings,
  stopReply,
  useChatEntry,
  useChatFailure,
  useChatSummaries,
} from "@/lib/chat";
import { useLive } from "@/lib/events";
import { useLibrary, useMediaQuery } from "@/lib/hooks";

function hasFiles(event: DragEvent<HTMLElement>): boolean {
  return Array.from(event.dataTransfer.types).includes("Files");
}

/** The chat playground: stored chats on the left, the open one on the right. `?id=` names the
 * open chat; without it the page reopens the last chat of this visit, else starts a new one. */
export function ChatPage() {
  const qc = useQueryClient();
  const [params, setParams] = useSearchParams();
  const requested = params.get("id");
  const id = requested !== null && CHAT_ID_PATTERN.test(requested) ? requested : null;
  const entry = useChatEntry(id);
  const failure = useChatFailure(id);
  const summaries = useChatSummaries();
  const { data: library } = useLibrary();
  const instance = useLive((live) => live.instance);
  const wide = useMediaQuery("(min-width: 64rem)");
  const [listOpen, setListOpen] = useState(false);
  const composerRef = useRef<ComposerHandle>(null);
  const [dragging, setDragging] = useState(false);
  const dragDepth = useRef(0);

  useEffect(() => {
    if (id === null) setParams({ id: lastOpenChat() ?? createDraft() }, { replace: true });
    else void openChat(id);
  }, [id, setParams]);

  const chatModel = entry ? entry.chat.model : null;
  const support = useMemo(() => imageSupport(library, chatModel, instance), [library, chatModel, instance]);

  const select = (next: string) => {
    setListOpen(false);
    if (next === id) return;
    if (id !== null) discardDraft(id);
    setParams({ id: next });
  };

  const newChat = () => {
    if (entry && !entry.saved && entry.chat.messages.length === 0) {
      setListOpen(false);
      return;
    }
    select(createDraft(entry ? { model: entry.chat.model, options: entry.chat.options } : null));
  };

  const onDeleted = (deleted: string) => {
    if (deleted !== id) return;
    const next = summaries.data?.find((summary) => summary.id !== deleted);
    setParams({ id: next ? next.id : createDraft(entry ? { model: entry.chat.model, options: entry.chat.options } : null) }, { replace: true });
  };

  const draft = entry && !entry.saved ? entry : null;
  const listButton = wide ? null : (
    <Button variant="ghost" size="icon-sm" aria-label="Chats" onClick={() => setListOpen(true)}>
      <PanelLeft />
    </Button>
  );

  let body: ReactNode;
  if (failure !== null && id !== null) {
    body = (
      <>
        {listButton ? <header className="flex items-center border-b px-3 py-2.5">{listButton}</header> : null}
        <div className="flex flex-1 items-center justify-center p-6">
          <EmptyState
            icon={MessagesSquare}
            title="This chat could not be opened"
            description={failure}
            actions={
              <>
                <Button variant="outline" onClick={() => void retryOpen(id)}>
                  <RefreshCw />
                  Retry
                </Button>
                <Button onClick={newChat}>
                  <MessageSquarePlus />
                  New chat
                </Button>
              </>
            }
          />
        </div>
      </>
    );
  } else if (entry === null) {
    body = (
      <div className="flex flex-1 items-center justify-center gap-2 text-sm text-muted-foreground">
        <LoaderCircle className="size-4 animate-spin" />
        Opening the chat…
      </div>
    );
  } else {
    const { chat, reply } = entry;
    body = (
      <>
        <header className="flex flex-wrap items-center gap-x-3 gap-y-2 border-b px-3 py-2.5">
          {listButton}
          <div className="min-w-0 flex-1 basis-32 truncate text-sm font-semibold" title={chatTitle(chat)}>
            {chatTitle(chat)}
          </div>
          <div className="flex min-w-0 items-center gap-2">
            <ModelStatus configId={chat.model} />
            <ModelPicker value={chat.model} onChange={(model) => setChatModel(qc, chat.id, model)} />
            <ChatOptionsButton key={chat.id} chat={chat} onChange={(patch) => setChatSettings(qc, chat.id, patch)} />
          </div>
        </header>
        <Conversation entry={entry} className="flex-1" />
        <div className="border-t px-3 pt-3 pb-2">
          <div className="mx-auto w-full max-w-3xl">
            <Composer
              key={chat.id}
              ref={composerRef}
              chatId={chat.id}
              busy={reply !== null}
              images={support}
              autoFocus={wide}
              onSend={(text, images) => sendMessage(qc, chat.id, text, images)}
              onStop={() => stopReply(chat.id)}
            />
            <div className="mt-2 px-1">
              <ReplyStatsBar stats={latestStats(chat.messages)} live={reply?.live ?? null} />
            </div>
          </div>
        </div>
      </>
    );
  }

  return (
    <div className="flex h-[calc(100dvh-7rem)] min-h-[26rem] gap-4">
      {wide ? (
        <aside className="flex w-64 shrink-0 flex-col overflow-hidden rounded-xl border bg-card">
          <ChatList selectedId={id} draft={draft} onSelect={select} onNew={newChat} onDeleted={onDeleted} />
        </aside>
      ) : (
        <Sheet open={listOpen} onOpenChange={setListOpen}>
          <SheetContent side="left" className="gap-0 bg-card p-0">
            <SheetTitle className="sr-only">Chats</SheetTitle>
            <SheetDescription className="sr-only">Chats with the launcher's models</SheetDescription>
            <ChatList
              selectedId={id}
              draft={draft}
              onSelect={select}
              onNew={newChat}
              onDeleted={onDeleted}
              headerClassName="pr-12"
            />
          </SheetContent>
        </Sheet>
      )}
      <section
        aria-label="Chat"
        className="relative flex min-w-0 flex-1 flex-col overflow-hidden rounded-xl border bg-card"
        onDragEnter={(event) => {
          if (!hasFiles(event)) return;
          event.preventDefault();
          dragDepth.current += 1;
          setDragging(true);
        }}
        onDragOver={(event) => {
          if (!hasFiles(event)) return;
          event.preventDefault();
          event.dataTransfer.dropEffect = "copy";
        }}
        onDragLeave={(event) => {
          if (!hasFiles(event)) return;
          dragDepth.current = Math.max(0, dragDepth.current - 1);
          if (dragDepth.current === 0) setDragging(false);
        }}
        onDrop={(event) => {
          if (!hasFiles(event)) return;
          event.preventDefault();
          dragDepth.current = 0;
          setDragging(false);
          composerRef.current?.addFiles(Array.from(event.dataTransfer.files));
        }}
      >
        {body}
        {dragging && entry !== null ? (
          <div className="pointer-events-none absolute inset-2 z-10 flex items-center justify-center gap-2 rounded-lg border-2 border-dashed border-primary/60 bg-background/85 px-6 text-center text-sm font-medium">
            <ImagePlus className="size-4 shrink-0 text-primary" />
            {support.allowed ? "Drop images to attach them" : (support.note ?? "This model takes no images")}
          </div>
        ) : null}
      </section>
    </div>
  );
}
