import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { BadgeCheck, CloudDownload, Database, Download, ExternalLink, Library as LibraryIcon, Search, Sparkles } from "lucide-react";
import { toast } from "sonner";

import { PageHeader } from "@/components/common";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { api } from "@/lib/api";
import type { Catalog, CatalogEntry, ModelFamily } from "@/lib/api-types";
import { FAMILY_LABELS, FAMILY_ORDER, NGRAM_FORMAT_LABELS, formatBytes } from "@/lib/format";
import { useLibrary } from "@/lib/hooks";
import { cn, parseRepo } from "@/lib/utils";
import { DownloadDialog, inLibrary, repoName, sameRepo } from "@/pages/discover/DownloadDialog";
import { DownloadsPanel } from "@/pages/discover/DownloadsPanel";

type Source = "all" | "infernix" | "ninfer";

const OFFICIAL_TAG = "ninfer official";

function isOfficial(entry: CatalogEntry): boolean {
  return entry.tags.some((tag) => tag.toLowerCase() === OFFICIAL_TAG) || entry.repo.toLowerCase().startsWith("neroued/");
}

function EntryCard({
  entry,
  present,
  onDownload,
}: {
  entry: CatalogEntry;
  present: boolean;
  onDownload: (entry: CatalogEntry) => void;
}) {
  const official = isOfficial(entry);
  const tags = entry.tags.filter((tag) => tag.toLowerCase() !== OFFICIAL_TAG);
  const recommendedNgram = entry.ngram_options.find((option) => option.recommended) ?? null;
  return (
    <Card className={cn("gap-3", entry.recommended && "border-primary/40")}>
      <CardHeader className="gap-2">
        <div className="flex flex-wrap items-center gap-1.5">
          {entry.recommended ? (
            <Badge variant="default">
              <Sparkles />
              Recommended
            </Badge>
          ) : null}
          {official ? (
            <Badge variant="outline">
              <BadgeCheck />
              NInfer official
            </Badge>
          ) : (
            <Badge variant="muted">Infernix conversion</Badge>
          )}
          {entry.kind !== "model" ? <Badge variant="secondary">{entry.kind === "drafter" ? "Drafter" : "N-gram volume"}</Badge> : null}
          {present ? (
            <Badge variant="accent">
              <LibraryIcon />
              In library
            </Badge>
          ) : null}
        </div>
        <div className="min-w-0">
          <div className="text-base leading-snug font-semibold">{entry.title}</div>
          <a
            href={`https://huggingface.co/${entry.repo}`}
            target="_blank"
            rel="noreferrer"
            className="inline-flex max-w-full items-center gap-1 truncate font-mono text-[11px] text-muted-foreground hover:text-foreground"
          >
            <span className="truncate">{entry.repo}</span>
            <ExternalLink className="size-3 shrink-0" />
          </a>
        </div>
      </CardHeader>
      <CardContent className="flex flex-1 flex-col gap-3">
        <p className="text-sm leading-relaxed text-muted-foreground">{entry.description}</p>
        {tags.length > 0 ? (
          <div className="flex flex-wrap gap-1">
            {tags.map((tag) => (
              <Badge key={tag} variant="muted">
                {tag}
              </Badge>
            ))}
          </div>
        ) : null}
        {entry.drafter_repo || recommendedNgram ? (
          <div className="flex flex-col gap-1 rounded-lg bg-muted/50 px-3 py-2 text-xs text-muted-foreground">
            {entry.drafter_repo ? (
              <span className="flex items-center gap-1.5">
                <Sparkles className="size-3.5 shrink-0" />
                Pairs with the DFlash2 drafter ({repoName(entry.drafter_repo)})
              </span>
            ) : null}
            {recommendedNgram ? (
              <span className="flex items-center gap-1.5">
                <Database className="size-3.5 shrink-0" />
                Needs an n-gram volume: {NGRAM_FORMAT_LABELS[recommendedNgram.format]} recommended ({formatBytes(recommendedNgram.bytes)})
              </span>
            ) : null}
          </div>
        ) : null}
        {entry.notes ? <p className="text-xs text-muted-foreground">{entry.notes}</p> : null}
        {/* The button moves below the size and source when the card is too narrow for all three. */}
        <div className="mt-auto flex flex-wrap items-center gap-2 border-t pt-3">
          <div className="flex min-w-0 flex-auto items-baseline gap-2">
            <span className="shrink-0 text-sm font-medium tabular-nums">
              {entry.approx_bytes > 0 ? `~${formatBytes(entry.approx_bytes)}` : "size on request"}
            </span>
            {entry.source_repo ? (
              <a
                href={`https://huggingface.co/${entry.source_repo}`}
                target="_blank"
                rel="noreferrer"
                className="hidden min-w-0 truncate text-xs text-muted-foreground hover:text-foreground sm:inline"
                title={`Converted from ${entry.source_repo}`}
              >
                from {entry.source_repo}
              </a>
            ) : null}
          </div>
          <Button size="sm" variant={present ? "outline" : "default"} className="ml-auto" onClick={() => onDownload(entry)}>
            <Download />
            {present ? "Download again" : "Download"}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

export function DiscoverPage() {
  const catalog = useQuery<Catalog>({ queryKey: ["catalog"], queryFn: api.catalog, staleTime: Infinity });
  const { data: library } = useLibrary();
  const [query, setQuery] = useState("");
  const [source, setSource] = useState<Source>("all");
  const [custom, setCustom] = useState("");
  const [dialog, setDialog] = useState<{ repo: string; entry: CatalogEntry | null } | null>(null);

  const groups = useMemo(() => {
    const needle = query.trim().toLowerCase();
    const entries = (catalog.data?.entries ?? []).filter((entry) => {
      if (source === "infernix" && isOfficial(entry)) return false;
      if (source === "ninfer" && !isOfficial(entry)) return false;
      if (needle.length === 0) return true;
      return [entry.title, entry.repo, entry.description, ...entry.tags].some((text) => text.toLowerCase().includes(needle));
    });
    const byFamily = new Map<ModelFamily, CatalogEntry[]>();
    for (const entry of entries) {
      const list = byFamily.get(entry.family) ?? [];
      list.push(entry);
      byFamily.set(entry.family, list);
    }
    return FAMILY_ORDER.filter((family) => byFamily.has(family)).map((family) => ({
      family,
      entries: (byFamily.get(family) ?? []).sort(
        (a, b) =>
          Number(b.recommended) - Number(a.recommended) ||
          Number(isOfficial(a)) - Number(isOfficial(b)) ||
          Number(a.kind !== "model") - Number(b.kind !== "model"),
      ),
    }));
  }, [catalog.data, query, source]);

  const officialCount = (catalog.data?.entries ?? []).filter(isOfficial).length;
  const total = catalog.data?.entries.length ?? 0;

  const openRepo = () => {
    const repo = parseRepo(custom);
    if (!repo) {
      toast.error("Enter a Hugging Face repository as owner/name or its URL");
      return;
    }
    const entry = (catalog.data?.entries ?? []).find((item) => sameRepo(item.repo, repo)) ?? null;
    setDialog({ repo: entry?.repo ?? repo, entry });
  };

  return (
    <div className="flex flex-col gap-5">
      <PageHeader
        title="Discover"
        description="Supported Infernix and NInfer artifacts on Hugging Face. Downloads resume after interruptions, and their folders join your model folders."
      />

      <DownloadsPanel />

      <Card className="gap-3 py-4">
        <CardContent className="flex flex-col gap-3 lg:flex-row lg:items-center">
          <form
            className="flex min-w-0 flex-1 gap-2"
            onSubmit={(event) => {
              event.preventDefault();
              openRepo();
            }}
          >
            <Input
              value={custom}
              onChange={(event) => setCustom(event.target.value)}
              placeholder="Any Hugging Face repository: owner/name or https://huggingface.co/owner/name"
              className="font-mono text-xs"
              spellCheck={false}
              aria-label="Hugging Face repository"
            />
            <Button type="submit" variant="outline" disabled={custom.trim().length === 0}>
              <CloudDownload />
              Open
            </Button>
          </form>
        </CardContent>
      </Card>

      <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
        <div className="relative w-full sm:max-w-xs">
          <Search className="pointer-events-none absolute top-1/2 left-2.5 size-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search the catalog"
            className="pl-8"
            aria-label="Search the catalog"
          />
        </div>
        <ToggleGroup
          type="single"
          value={source}
          onValueChange={(value) => {
            if (value === "all" || value === "infernix" || value === "ninfer") setSource(value);
          }}
          aria-label="Source"
          className="sm:ml-auto"
        >
          <ToggleGroupItem value="all">All ({total})</ToggleGroupItem>
          <ToggleGroupItem value="infernix">Infernix conversions ({total - officialCount})</ToggleGroupItem>
          <ToggleGroupItem value="ninfer">NInfer official ({officialCount})</ToggleGroupItem>
        </ToggleGroup>
      </div>

      {catalog.isLoading ? (
        <div className="grid gap-4 md:grid-cols-2 2xl:grid-cols-3">
          {[0, 1, 2, 3].map((index) => (
            <Skeleton key={index} className="h-72 rounded-xl" />
          ))}
        </div>
      ) : groups.length === 0 ? (
        <div className="rounded-xl border border-dashed p-8 text-center text-sm text-muted-foreground">Nothing matches.</div>
      ) : (
        groups.map(({ family, entries }) => (
          <section key={family} className="flex flex-col gap-3">
            <div className="flex items-baseline gap-2">
              <h2 className="text-sm font-semibold">{FAMILY_LABELS[family]}</h2>
              <span className="text-xs text-muted-foreground">
                {entries.length} item{entries.length === 1 ? "" : "s"}
              </span>
            </div>
            <div className="grid gap-4 md:grid-cols-2 2xl:grid-cols-3">
              {entries.map((entry) => (
                <EntryCard
                  key={entry.repo}
                  entry={entry}
                  present={inLibrary(entry, library)}
                  onDownload={(item) => setDialog({ repo: item.repo, entry: item })}
                />
              ))}
            </div>
          </section>
        ))
      )}

      <DownloadDialog
        open={dialog !== null}
        onOpenChange={(open) => {
          if (!open) setDialog(null);
        }}
        repo={dialog?.repo ?? ""}
        entry={dialog?.entry ?? null}
        library={library}
        onQueued={() => window.scrollTo({ top: 0, behavior: "smooth" })}
      />
    </div>
  );
}
