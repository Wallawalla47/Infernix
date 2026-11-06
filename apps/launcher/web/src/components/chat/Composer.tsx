import { useCallback, useEffect, useImperativeHandle, useLayoutEffect, useRef, useState, type Ref } from "react";
import { ArrowUp, ImagePlus, Square, X } from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Hint } from "@/components/ui/tooltip";
import { IMAGE_ACCEPT, readImages, type ImageSupport } from "@/lib/chat";
import { cn } from "@/lib/utils";

export interface ComposerHandle {
  /** Attaches dropped files (images only; refused when the model takes none). */
  addFiles: (files: File[]) => void;
}

/** Unsent text and images per chat, kept while the page shows another chat. */
const drafts = new Map<string, { text: string; images: string[] }>();

/** The highest the prompt box grows before it scrolls. */
const MAX_HEIGHT_PX = 240;

export function Composer({
  chatId,
  busy,
  images: support,
  autoFocus,
  onSend,
  onStop,
  ref,
}: {
  chatId: string;
  /** A reply is streaming: the button stops it, Enter does not send. */
  busy: boolean;
  images: ImageSupport;
  autoFocus: boolean;
  onSend: (text: string, images: string[]) => void;
  onStop: () => void;
  ref?: Ref<ComposerHandle>;
}) {
  const [text, setText] = useState(() => drafts.get(chatId)?.text ?? "");
  const [images, setImages] = useState<string[]>(() => drafts.get(chatId)?.images ?? []);
  const areaRef = useRef<HTMLTextAreaElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (text.length > 0 || images.length > 0) drafts.set(chatId, { text, images });
    else drafts.delete(chatId);
  }, [chatId, text, images]);

  useLayoutEffect(() => {
    const area = areaRef.current;
    if (!area) return;
    area.style.height = "auto";
    area.style.height = `${Math.min(area.scrollHeight, MAX_HEIGHT_PX)}px`;
    area.style.overflowY = area.scrollHeight > MAX_HEIGHT_PX ? "auto" : "hidden";
  }, [text]);

  const addFiles = useCallback(
    (files: File[]) => {
      if (files.length === 0) return;
      if (!support.allowed) {
        toast.warning(support.note ?? "This model takes no images");
        return;
      }
      void readImages(files).then(({ images: read, rejected }) => {
        for (const reason of rejected) toast.error(reason);
        if (read.length > 0) setImages((current) => [...current, ...read]);
      });
    },
    [support],
  );

  useImperativeHandle(ref, () => ({ addFiles }), [addFiles]);

  const canSend = !busy && (text.trim().length > 0 || images.length > 0);
  const send = () => {
    if (!canSend) return;
    onSend(text.trim(), images);
    setText("");
    setImages([]);
    areaRef.current?.focus();
  };

  return (
    <div className="rounded-xl border bg-background shadow-xs transition-[border-color,box-shadow] focus-within:border-ring focus-within:ring-[3px] focus-within:ring-ring/50 dark:bg-input/20">
      {images.length > 0 ? (
        <div className="flex flex-wrap gap-2 px-3 pt-3">
          {images.map((src, index) => (
            <div key={index} className="relative size-16 overflow-hidden rounded-lg border bg-muted">
              <img src={src} alt={`Attachment ${index + 1}`} className="size-full object-cover" />
              <button
                type="button"
                aria-label={`Remove attachment ${index + 1}`}
                className="absolute top-1 right-1 flex size-5 items-center justify-center rounded-full bg-background/90 text-foreground shadow-sm outline-none hover:bg-background focus-visible:ring-2 focus-visible:ring-ring/50"
                onClick={() => setImages((current) => current.filter((_, at) => at !== index))}
              >
                <X className="size-3" />
              </button>
            </div>
          ))}
        </div>
      ) : null}
      <textarea
        ref={areaRef}
        rows={1}
        value={text}
        autoFocus={autoFocus}
        placeholder="Message the model…"
        aria-label="Message"
        className="block min-h-11 w-full resize-none bg-transparent px-3.5 py-3 text-sm outline-none placeholder:text-muted-foreground"
        onChange={(event) => setText(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
            event.preventDefault();
            send();
          }
        }}
        onPaste={(event) => {
          const files = Array.from(event.clipboardData.files).filter((file) => file.type.startsWith("image/"));
          if (files.length === 0) return;
          // An image copied alone pastes as the image; with text beside it, the text pastes too.
          if (!event.clipboardData.types.includes("text/plain")) event.preventDefault();
          addFiles(files);
        }}
      />
      <div className="flex items-center gap-2 px-2 pb-2">
        <Hint label={support.allowed ? "Attach images (or paste or drop them)" : (support.note ?? "Images are not available")}>
          <span>
            <Button
              variant="ghost"
              size="icon-sm"
              aria-label="Attach images"
              disabled={!support.allowed}
              onClick={() => fileRef.current?.click()}
            >
              <ImagePlus />
            </Button>
          </span>
        </Hint>
        {!support.allowed && support.note ? (
          <span className="min-w-0 truncate text-[11px] text-muted-foreground">{support.note}</span>
        ) : null}
        <div className="ml-auto flex shrink-0 items-center gap-2">
          <span className={cn("hidden text-[11px] text-muted-foreground xl:inline", busy && "xl:hidden")}>
            Enter to send, Shift+Enter for a new line
          </span>
          {busy ? (
            <Button size="sm" variant="secondary" onClick={onStop}>
              <Square className="fill-current" />
              Stop
            </Button>
          ) : (
            <Button size="sm" disabled={!canSend} onClick={send}>
              <ArrowUp />
              Send
            </Button>
          )}
        </div>
      </div>
      <input
        ref={fileRef}
        type="file"
        accept={IMAGE_ACCEPT}
        multiple
        hidden
        onChange={(event) => {
          addFiles(Array.from(event.target.files ?? []));
          event.target.value = "";
        }}
      />
    </div>
  );
}
