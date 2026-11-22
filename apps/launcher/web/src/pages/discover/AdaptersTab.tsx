// The Discover page's LoRA adapters: the Hugging Face PEFT adapters of a dense 27B base model that
// `--lora` reads (adapter_config.json and adapter_model.safetensors at the repository's root),
// Harness Hub's first. The launcher asks the Hub (GET /launcher/api/hf/adapters), so a Hugging Face
// token in Settings reaches gated adapters.

import { useEffect, useState } from "react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { Download, ExternalLink, Heart, Library as LibraryIcon, LoaderCircle, Lock, Puzzle, Search, Sparkles } from "lucide-react";

import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { Hint } from "@/components/ui/tooltip";
import { api, errorMessage } from "@/lib/api";
import type { HubAdapter, HubAdapterSearch, Library } from "@/lib/api-types";
import { formatCount } from "@/lib/format";
import { useLibrary } from "@/lib/hooks";
import { repoName } from "@/pages/discover/DownloadDialog";

/** The base models whose adapters --lora applies: the dense 27Bs the catalog offers. */
const BASES = [
  { repo: "Qwen/Qwen3.8-27B", label: "Qwen3.8-27B" },
  { repo: "Qwen/Qwen3.6-27B", label: "Qwen3.6-27B" },
] as const;

interface HarnessAdapter {
  title: string;
  text: string;
}

function specialist(harness: string): HarnessAdapter {
  return { title: `${harness} agent harness`, text: `Trained for running the model behind the ${harness} agent harness.` };
}

/** What a Harness Hub adapter was trained for, from its name: a specialist for one agent harness, or the super-adapter
 * (rank 96) for three. */
const HARNESS_ADAPTERS: [RegExp, HarnessAdapter][] = [
  [/-pi-adapted$/i, specialist("Pi")],
  [/-opencode-adapted$/i, specialist("OpenCode")],
  [/-hermes-adapted$/i, specialist("Hermes")],
  [/-claude-code-adapted$/i, specialist("Claude Code")],
  [
    /-super-adapter$/i,
    {
      title: "Pi, Hermes and OpenCode agent harnesses",
      text:
        "The super-adapter: one adapter for all three harnesses, merged from Harness Hub's specialists for them. Its " +
        "card recommends it with thinking on; with thinking off under one harness, that harness's specialist.",
    },
  ],
];

function harnessAdapterOf(repo: string): HarnessAdapter | null {
  return HARNESS_ADAPTERS.find(([pattern]) => pattern.test(repo))?.[1] ?? null;
}

function updated(iso: string): string {
  const time = Date.parse(iso);
  return Number.isFinite(time) ? new Date(time).toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" }) : "";
}

function Facts({ adapter }: { adapter: HubAdapter }) {
  return (
    <span className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground tabular-nums">
      <Hint label="Downloads in the last 30 days">
        <span className="inline-flex cursor-default items-center gap-1">
          <Download className="size-3" />
          {formatCount(adapter.downloads)}
        </span>
      </Hint>
      <span className="inline-flex items-center gap-1">
        <Heart className="size-3" />
        {formatCount(adapter.likes)}
      </span>
      {adapter.last_modified ? <span>updated {updated(adapter.last_modified)}</span> : null}
    </span>
  );
}

/** The library's copy of a Hub adapter: a folder of the repository's name (where a download puts one). */
function libraryCopy(library: Library | undefined, repo: string): string | null {
  const name = repoName(repo).toLowerCase();
  return library?.adapters.find((adapter) => adapter.name.toLowerCase() === name)?.path ?? null;
}

function Badges({ adapter, present }: { adapter: HubAdapter; present: string | null }) {
  return (
    <>
      {adapter.gated ? (
        <Hint label="Its files need a Hugging Face token (Settings) from an account that accepted its terms">
          <Badge variant="warning" className="cursor-default">
            <Lock />
            Gated
          </Badge>
        </Hint>
      ) : null}
      {present ? (
        <Hint label={present}>
          <Badge variant="accent" className="cursor-default">
            <LibraryIcon />
            In library
          </Badge>
        </Hint>
      ) : null}
    </>
  );
}

function RepoLink({ repo, className }: { repo: string; className?: string }) {
  return (
    <a
      href={`https://huggingface.co/${repo}`}
      target="_blank"
      rel="noreferrer"
      className={`inline-flex max-w-full items-center gap-1 truncate font-mono text-[11px] text-muted-foreground hover:text-foreground ${className ?? ""}`}
    >
      <span className="truncate">{repo}</span>
      <ExternalLink className="size-3 shrink-0" />
    </a>
  );
}

interface ItemProps {
  adapter: HubAdapter;
  present: string | null;
  onDownload: (repo: string) => void;
}

function HarnessCard({ adapter, present, onDownload }: ItemProps) {
  const harness = harnessAdapterOf(adapter.repo);
  return (
    <Card className="gap-3">
      <CardHeader className="gap-2">
        <div className="flex flex-wrap items-center gap-1.5">
          <Badge variant="default">
            <Sparkles />
            Harness Hub
          </Badge>
          <Badges adapter={adapter} present={present} />
        </div>
        <div className="min-w-0">
          <div className="text-base leading-snug font-semibold">{harness ? harness.title : repoName(adapter.repo)}</div>
          <RepoLink repo={adapter.repo} />
        </div>
      </CardHeader>
      <CardContent className="flex flex-1 flex-col gap-3">
        <p className="text-sm leading-relaxed text-muted-foreground">
          {harness ? harness.text : "Trained by Harness Hub for agent harnesses."}{" "}
          Its card on Hugging Face has the training details and conditions of use.
        </p>
        <div className="mt-auto flex flex-wrap items-center gap-2 border-t pt-3">
          <Facts adapter={adapter} />
          <Button size="sm" variant={present ? "outline" : "default"} className="ml-auto" onClick={() => onDownload(adapter.repo)}>
            <Download />
            {present ? "Download again" : "Download"}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

function AdapterRow({ adapter, present, onDownload }: ItemProps) {
  return (
    <li className="flex flex-col gap-2 px-4 py-3 sm:flex-row sm:items-center sm:gap-3">
      <div className="min-w-0 flex-1">
        <div className="flex min-w-0 flex-wrap items-center gap-1.5">
          <span className="truncate text-sm font-medium">{repoName(adapter.repo)}</span>
          <Badges adapter={adapter} present={present} />
        </div>
        <RepoLink repo={adapter.repo} />
      </div>
      <Facts adapter={adapter} />
      <Button size="sm" variant="outline" className="sm:ml-2" onClick={() => onDownload(adapter.repo)}>
        <Download />
        {present ? "Again" : "Download"}
      </Button>
    </li>
  );
}

export function AdaptersTab({ onDownload }: { onDownload: (repo: string) => void }) {
  const { data: library } = useLibrary();
  const [base, setBase] = useState<string>(BASES[0].repo);
  const [text, setText] = useState("");
  const [search, setSearch] = useState("");
  // The Hub is asked once the typing stops.
  useEffect(() => {
    const timer = window.setTimeout(() => setSearch(text.trim()), 400);
    return () => window.clearTimeout(timer);
  }, [text]);

  const found = useQuery<HubAdapterSearch>({
    queryKey: ["hub-adapters", base, search],
    queryFn: () => api.hubAdapters(base, search),
    staleTime: 5 * 60_000,
    retry: 0,
    placeholderData: keepPreviousData,
  });
  const adapters = found.data?.adapters ?? [];
  const featured = adapters.filter((adapter) => adapter.harness_hub);
  const others = adapters.filter((adapter) => !adapter.harness_hub);
  const baseLabel = BASES.find((item) => item.repo === base)?.label ?? base;

  return (
    <div className="flex flex-col gap-5">
      <Alert variant="info">
        <Puzzle />
        <AlertDescription>
          A LoRA adapter fine-tunes a model when it loads (--lora), without changing the artifact. Infernix applies
          plain LoRA on the attention, Gated DeltaNet and MLP projections, up to rank 128, to the dense 27B models.
          Download one here, then pick it on a 27B model's Configure page (LoRA adapter): its "Will it fit?" check
          asks the server whether it can apply it.
        </AlertDescription>
      </Alert>

      <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
        <div className="relative w-full sm:max-w-xs">
          <Search className="pointer-events-none absolute top-1/2 left-2.5 size-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            value={text}
            onChange={(event) => setText(event.target.value)}
            placeholder="Search adapters on Hugging Face"
            className="pl-8"
            aria-label="Search adapters on Hugging Face"
            maxLength={100}
          />
        </div>
        {found.isFetching ? <LoaderCircle className="size-4 animate-spin text-muted-foreground" aria-label="Searching" /> : null}
        <ToggleGroup
          type="single"
          value={base}
          onValueChange={(value) => {
            if (value) setBase(value);
          }}
          aria-label="Base model"
          className="sm:ml-auto"
        >
          {BASES.map((item) => (
            <ToggleGroupItem key={item.repo} value={item.repo}>
              {item.label}
            </ToggleGroupItem>
          ))}
        </ToggleGroup>
      </div>

      {found.isLoading ? (
        <div className="grid gap-4 md:grid-cols-2 2xl:grid-cols-3">
          {[0, 1, 2].map((index) => (
            <Skeleton key={index} className="h-56 rounded-xl" />
          ))}
        </div>
      ) : found.isError ? (
        <Alert variant="destructive">
          <AlertDescription>Could not ask Hugging Face: {errorMessage(found.error)}</AlertDescription>
        </Alert>
      ) : adapters.length === 0 ? (
        <div className="rounded-xl border border-dashed p-8 text-center text-sm text-muted-foreground">
          No adapter of {baseLabel} {search ? `matches "${search}"` : "found"}.
        </div>
      ) : (
        <>
          {featured.length > 0 ? (
            <section className="flex flex-col gap-3">
              <div>
                <h2 className="text-sm font-semibold">Harness Hub</h2>
                <p className="text-xs text-muted-foreground">
                  Adapters of {baseLabel} trained for agent harnesses by{" "}
                  <a href="https://huggingface.co/harness-hub" target="_blank" rel="noreferrer" className="underline-offset-4 hover:underline">
                    huggingface.co/harness-hub
                  </a>
                  .
                </p>
              </div>
              <div className="grid gap-4 md:grid-cols-2 2xl:grid-cols-3">
                {featured.map((adapter) => (
                  <HarnessCard
                    key={adapter.repo}
                    adapter={adapter}
                    present={libraryCopy(library, adapter.repo)}
                    onDownload={onDownload}
                  />
                ))}
              </div>
            </section>
          ) : null}
          {others.length > 0 ? (
            <section className="flex flex-col gap-3">
              <div>
                <h2 className="text-sm font-semibold">More adapters on Hugging Face</h2>
                <p className="text-xs text-muted-foreground">
                  PEFT adapters of {baseLabel}, most downloaded first{search ? `, named like "${search}"` : ""} (up to 60).
                  Anyone can publish one: read its card before you use it.
                </p>
              </div>
              <Card className="py-0">
                <ul className="divide-y">
                  {others.map((adapter) => (
                    <AdapterRow
                      key={adapter.repo}
                      adapter={adapter}
                      present={libraryCopy(library, adapter.repo)}
                      onDownload={onDownload}
                    />
                  ))}
                </ul>
              </Card>
            </section>
          ) : null}
        </>
      )}
    </div>
  );
}
