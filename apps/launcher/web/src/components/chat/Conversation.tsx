import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { ArrowDown, MessagesSquare } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog";
import { modelName, type ChatEntry } from "@/lib/chat";
import { useLibrary } from "@/lib/hooks";
import { cn } from "@/lib/utils";
import { MessageView } from "./MessageView";

/** Within this many pixels of the end counts as following the conversation. */
const STICK_PX = 48;

function ImageDialog({ src, onClose }: { src: string | null; onClose: () => void }) {
  return (
    <Dialog open={src !== null} onOpenChange={(open) => (open ? undefined : onClose())}>
      <DialogContent className="w-auto max-w-[calc(100vw-2rem)] p-2 sm:max-w-[min(64rem,calc(100vw-2rem))]">
        <DialogTitle className="sr-only">Attached image</DialogTitle>
        {src ? <img src={src} alt="" className="max-h-[calc(100dvh-5rem)] w-auto rounded-lg object-contain" /> : null}
      </DialogContent>
    </Dialog>
  );
}

/** The messages of a chat. Follows new text while the view is at the end; scrolling up stops
 * that until the view returns to the end. */
export function Conversation({ entry, className }: { entry: ChatEntry; className?: string }) {
  const { data: library } = useLibrary();
  const scrollRef = useRef<HTMLDivElement>(null);
  const contentRef = useRef<HTMLDivElement>(null);
  const follow = useRef(true);
  const [atEnd, setAtEnd] = useState(true);
  const [preview, setPreview] = useState<string | null>(null);

  const { chat, reply } = entry;
  const messages = chat.messages;
  const busy = reply !== null;

  const toEnd = useCallback((smooth = false) => {
    const scroller = scrollRef.current;
    if (!scroller) return;
    follow.current = true;
    setAtEnd(true);
    scroller.scrollTo({ top: scroller.scrollHeight, behavior: smooth ? "smooth" : "auto" });
  }, []);

  // Growth of the content (streamed text, images and math laying out) keeps the end in view.
  useEffect(() => {
    const scroller = scrollRef.current;
    const content = contentRef.current;
    if (!scroller || !content) return;
    const observer = new ResizeObserver(() => {
      if (follow.current) scroller.scrollTop = scroller.scrollHeight;
    });
    observer.observe(content);
    return () => observer.disconnect();
  }, []);

  // Another chat, or a new reply (the user just sent): show the end. A reply that ends leaves the
  // view where the user put it.
  const replyId = reply?.message_id ?? null;
  const shownReply = useRef<string | null>(null);
  useLayoutEffect(() => {
    toEnd();
  }, [chat.id, toEnd]);
  useLayoutEffect(() => {
    if (replyId !== null && replyId !== shownReply.current) toEnd();
    shownReply.current = replyId;
  }, [replyId, toEnd]);

  const onScroll = () => {
    const scroller = scrollRef.current;
    if (!scroller) return;
    const end = scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight <= STICK_PX;
    follow.current = end;
    setAtEnd(end);
  };

  const lastId = messages.length > 0 ? messages[messages.length - 1].id : null;

  return (
    <div className={cn("relative min-h-0", className)}>
      <div
        ref={scrollRef}
        className="h-full overflow-y-auto"
        onScroll={onScroll}
        onWheel={(event) => {
          // Scrolling up leaves the end at once, before the next streamed text pulls it back.
          if (event.deltaY < 0) follow.current = false;
        }}
      >
        <div ref={contentRef} className="mx-auto flex w-full max-w-3xl flex-col gap-6 px-4 py-6">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center gap-3 py-16 text-center">
              <div className="flex size-11 items-center justify-center rounded-full bg-primary/10 text-primary">
                <MessagesSquare className="size-5" />
              </div>
              <div className="max-w-md">
                <div className="font-medium">Start a conversation</div>
                <div className="mt-1 text-sm text-muted-foreground">
                  Messages go to {chat.model === null ? "the loaded model" : modelName(library, chat.model)} through the
                  launcher's OpenAI endpoint, like any client's. A model that is not loaded loads with the first message.
                </div>
              </div>
            </div>
          ) : (
            messages.map((message) => (
              <MessageView
                key={message.id}
                chatId={chat.id}
                message={message}
                modelLabel={modelName(library, message.model)}
                live={reply?.message_id === message.id ? reply.live : null}
                busy={busy}
                last={message.id === lastId}
                onOpenImage={setPreview}
              />
            ))
          )}
        </div>
      </div>
      {!atEnd ? (
        <Button
          variant="outline"
          size="xs"
          className="absolute bottom-3 left-1/2 -translate-x-1/2 rounded-full bg-background shadow-md"
          onClick={() => toEnd(true)}
        >
          <ArrowDown />
          Latest
        </Button>
      ) : null}
      <ImageDialog src={preview} onClose={() => setPreview(null)} />
    </div>
  );
}
