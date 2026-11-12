import { useEffect, useRef, useState, type ReactNode, type RefObject } from "react";
import type { LucideIcon } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { cn } from "@/lib/utils";

export function PageHeader({
  title,
  description,
  actions,
  className,
}: {
  title: ReactNode;
  description?: ReactNode;
  actions?: ReactNode;
  className?: string;
}) {
  return (
    // The actions sit beside the title while the title keeps 20rem, and below it otherwise.
    <div className={cn("flex flex-wrap items-end justify-between gap-x-6 gap-y-3", className)}>
      <div className="min-w-0 flex-[1_1_20rem]">
        <h1 className="text-xl font-semibold tracking-tight">{title}</h1>
        {description ? <p className="mt-1 text-sm text-muted-foreground">{description}</p> : null}
      </div>
      {actions ? <div className="flex flex-wrap items-center gap-2">{actions}</div> : null}
    </div>
  );
}

export function EmptyState({
  icon: Icon,
  title,
  description,
  actions,
  className,
}: {
  icon: LucideIcon;
  title: string;
  description?: ReactNode;
  actions?: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center gap-3 rounded-xl border border-dashed px-6 py-12 text-center",
        className,
      )}
    >
      <div className="flex size-11 items-center justify-center rounded-full bg-primary/10 text-primary">
        <Icon className="size-5" />
      </div>
      <div className="max-w-md">
        <div className="font-medium">{title}</div>
        {description ? <div className="mt-1 text-sm text-muted-foreground">{description}</div> : null}
      </div>
      {actions ? <div className="mt-1 flex flex-wrap items-center justify-center gap-2">{actions}</div> : null}
    </div>
  );
}

/** A labelled fact: small muted label over a value. */
export function Fact({ label, children, className }: { label: string; children: ReactNode; className?: string }) {
  return (
    <div className={cn("min-w-0", className)}>
      <div className="text-[11px] font-medium tracking-wide text-muted-foreground uppercase">{label}</div>
      <div className="mt-0.5 truncate text-sm">{children}</div>
    </div>
  );
}

export function ConfirmDialog({
  open,
  onOpenChange,
  title,
  description,
  confirmLabel,
  onConfirm,
  destructive = false,
  secondaryLabel,
  onSecondary,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: string;
  description: ReactNode;
  confirmLabel: string;
  onConfirm: () => void;
  destructive?: boolean;
  /** An alternative action beside Cancel (e.g. "Switch now"). */
  secondaryLabel?: string;
  onSecondary?: () => void;
}) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
          <DialogDescription asChild>
            <div>{description}</div>
          </DialogDescription>
        </DialogHeader>
        <DialogFooter>
          <Button variant="ghost" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          {secondaryLabel && onSecondary ? (
            <Button
              variant="outline"
              onClick={() => {
                onSecondary();
                onOpenChange(false);
              }}
            >
              {secondaryLabel}
            </Button>
          ) : null}
          <Button
            variant={destructive ? "destructive" : "default"}
            onClick={() => {
              onConfirm();
              onOpenChange(false);
            }}
          >
            {confirmLabel}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

/** A row in a settings form: label and help on the left, the control on the right. */
export function FieldRow({
  label,
  help,
  htmlFor,
  children,
  className,
  stacked = false,
}: {
  label: ReactNode;
  help?: ReactNode;
  htmlFor?: string;
  children: ReactNode;
  className?: string;
  /** Put the control under the label (wide controls). */
  stacked?: boolean;
}) {
  return (
    <div
      className={cn(
        "flex gap-3 py-3",
        stacked ? "flex-col" : "flex-col sm:flex-row sm:items-center sm:justify-between",
        className,
      )}
    >
      <div className="min-w-0 sm:max-w-[60%]">
        <label htmlFor={htmlFor} className="text-sm font-medium">
          {label}
        </label>
        {help ? <div className="mt-0.5 text-xs leading-relaxed text-muted-foreground">{help}</div> : null}
      </div>
      <div className={cn("min-w-0", stacked ? "w-full" : "sm:w-64 sm:shrink-0")}>{children}</div>
    </div>
  );
}

/** The titles of the cards inside `container` as a row of links that stays under the page header;
 * a click scrolls to that card. For long pages of cards (Settings). It reads the titles from the
 * page, so a card added later is listed without being named here. */
export function CardIndex({ container }: { container: RefObject<HTMLElement | null> }) {
  const [titles, setTitles] = useState<string[]>([]);
  const nav = useRef<HTMLElement>(null);
  // After every render of the page: its cards appear once its data has loaded.
  useEffect(() => {
    const found = [...(container.current?.querySelectorAll<HTMLElement>('[data-slot="card-title"]') ?? [])]
      .map((title) => title.textContent?.trim() ?? "")
      .filter((title) => title.length > 0);
    setTitles((old) => (old.join("\n") === found.join("\n") ? old : found));
  });
  if (titles.length < 4) return null;
  const go = (index: number) => {
    const title = container.current?.querySelectorAll<HTMLElement>('[data-slot="card-title"]')[index];
    const card = title?.closest<HTMLElement>('[data-slot="card"]');
    if (!card) return;
    // Clear of the app header and of this row, whose height depends on how its entries wrap.
    const strip = nav.current;
    const stripBottom = strip ? (parseFloat(getComputedStyle(strip).top) || 0) + strip.offsetHeight : 0;
    card.style.scrollMarginTop = `${stripBottom + 24}px`;
    // At once: an animated scroll can be cut short by the page's once-a-second live updates.
    card.scrollIntoView({ block: "start" });
  };
  return (
    // One row that scrolls sideways on a phone, where wrapping would cover a sixth of the screen.
    <nav
      ref={nav}
      aria-label="Sections of this page"
      className="sticky top-14 z-20 -mx-1 flex gap-1 overflow-x-auto rounded-lg bg-background/85 px-1 py-2 backdrop-blur-md sm:flex-wrap sm:overflow-x-visible"
    >
      {titles.map((title, index) => (
        <button
          key={title}
          type="button"
          onClick={() => go(index)}
          className="shrink-0 rounded-md border px-2 py-1 text-xs font-medium whitespace-nowrap text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
        >
          {title}
        </button>
      ))}
    </nav>
  );
}

export function SectionCardTitle({ icon: Icon, children }: { icon: LucideIcon; children: ReactNode }) {
  return (
    <span className="flex items-center gap-2">
      <Icon className="size-4 text-muted-foreground" />
      {children}
    </span>
  );
}
