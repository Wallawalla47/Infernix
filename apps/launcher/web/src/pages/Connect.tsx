import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ArrowRightLeft, KeyRound, Plug, RefreshCw, ShieldCheck, ShieldOff } from "lucide-react";

import { PageHeader } from "@/components/common";
import { CopyButton } from "@/components/CopyButton";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { errorMessage, publicModels, type PublicModel } from "@/lib/api";
import { useLive } from "@/lib/events";
import { formatContext } from "@/lib/format";
import { useEndpoints, useSettings } from "@/lib/hooks";
import { readStorage, writeStorage } from "@/lib/utils";

const KEY_STORAGE = "infernix.connect.apiKey";

function CodeBlock({ code, label }: { code: string; label?: string }) {
  return (
    <div className="relative min-w-0">
      {label ? <div className="mb-1.5 text-xs font-medium text-muted-foreground">{label}</div> : null}
      <pre className="overflow-x-auto rounded-lg border bg-muted/40 p-3 pr-12 font-mono text-[12px] leading-5">{code}</pre>
      <div className="absolute right-1.5" style={{ top: label ? "1.85rem" : "0.375rem" }}>
        <CopyButton text={code} size="icon-xs" toastMessage="Copied" />
      </div>
    </div>
  );
}

interface Snippet {
  id: string;
  label: string;
  intro: string;
  blocks: { label?: string; code: string }[];
}

function snippets(
  openai: string,
  anthropic: string,
  model: string,
  context: number | null,
  key: string | null,
  llamacpp: boolean,
): Snippet[] {
  const token = key ?? "infernix";
  const keyNote = key ? "" : "  # any value: the launcher has no API keys";
  // Claude Code assumes 200K tokens for a model it does not know, Codex 272K.
  const claudeContext = context ? `\nexport CLAUDE_CODE_MAX_CONTEXT_TOKENS=${context}` : "";
  const claudeContextPs = context ? `\n$env:CLAUDE_CODE_MAX_CONTEXT_TOKENS = "${context}"` : "";
  const codexContext = context ? `\nmodel_context_window = ${context}` : "";
  return [
    {
      id: "claude-code",
      label: "Claude Code",
      intro:
        "Claude Code speaks the Anthropic Messages API. Point it at the launcher and name the model; setting the Haiku model to the same id keeps background tasks from switching models, and the context setting tells Claude Code the model's real window. With model discovery on, /model lists every launcher model (as <model id>:claude-code, the spelling Claude Code accepts).",
      blocks: [
        {
          label: "bash / zsh",
          code: `export ANTHROPIC_BASE_URL=${anthropic}
export ANTHROPIC_AUTH_TOKEN=${token}${keyNote}
export ANTHROPIC_MODEL=${model}
export ANTHROPIC_DEFAULT_HAIKU_MODEL=${model}${claudeContext}
export CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1
claude`,
        },
        {
          label: "PowerShell",
          code: `$env:ANTHROPIC_BASE_URL = "${anthropic}"
$env:ANTHROPIC_AUTH_TOKEN = "${token}"
$env:ANTHROPIC_MODEL = "${model}"
$env:ANTHROPIC_DEFAULT_HAIKU_MODEL = "${model}"${claudeContextPs}
$env:CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY = "1"
claude`,
        },
      ],
    },
    {
      id: "hermes",
      label: "Hermes",
      intro:
        "Add a custom OpenAI-compatible endpoint with this base URL. Hermes lists the launcher's models in its model picker and reads each one's context length from /v1/models; the first request to a model loads it.",
      blocks: [
        { label: "Base URL", code: openai },
        { label: "API key", code: token },
      ],
    },
    {
      id: "codex",
      label: "Codex CLI",
      intro:
        "Add a model provider to ~/.codex/config.toml (Codex uses the Responses API). Codex does not list a custom provider's models, so name the one to use; model_context_window gives it the model's real window.",
      blocks: [
        {
          label: "config.toml",
          code: `model = "${model}"
model_provider = "infernix"${codexContext}

[model_providers.infernix]
name = "Infernix"
base_url = "${openai}"
wire_api = "responses"${key ? '\nenv_key = "INFERNIX_API_KEY"' : ""}`,
        },
      ],
    },
    {
      id: "zed",
      label: "Zed",
      intro: `${
        llamacpp ? "" : "First turn on Settings → llama.cpp endpoints (off by default). "
      }Zed's llama.cpp provider then lists every launcher model with its context, vision and tool support, and follows loads as they happen. Add this to Zed's settings.json${
        key ? ", and set LLAMACPP_API_KEY to an API key (or enter one in the provider's settings)" : ""
      }.`,
      blocks: [
        {
          label: "settings.json",
          code: `{
  "language_models": {
    "llama.cpp": {
      "api_url": "${anthropic}"
    }
  }
}`,
        },
      ],
    },
    {
      id: "qwen-code",
      label: "Qwen Code",
      intro: "Qwen Code reads the OpenAI settings from the environment (or its settings.json).",
      blocks: [
        {
          label: "bash / zsh",
          code: `export OPENAI_BASE_URL=${openai}
export OPENAI_API_KEY=${token}
export OPENAI_MODEL=${model}
qwen`,
        },
      ],
    },
    {
      id: "openai-python",
      label: "OpenAI SDK",
      intro: "Any OpenAI client works with the /v1 base URL: Chat Completions and Responses, streaming included.",
      blocks: [
        {
          label: "Python",
          code: `from openai import OpenAI

client = OpenAI(base_url="${openai}", api_key="${token}")
stream = client.chat.completions.create(
    model="${model}",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)`,
        },
      ],
    },
    {
      id: "anthropic-python",
      label: "Anthropic SDK",
      intro: "Anthropic SDKs append /v1/messages themselves, so give them the bare launcher address.",
      blocks: [
        {
          label: "Python",
          code: `import anthropic

client = anthropic.Anthropic(base_url="${anthropic}", api_key="${token}")
message = client.messages.create(
    model="${model}",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}],
)
print(message.content[0].text)`,
        },
      ],
    },
    {
      id: "curl",
      label: "curl",
      intro: "List the models, then send a request (the first one loads the model).",
      blocks: [
        { label: "Models", code: `curl ${openai}/models${key ? ` \\\n  -H "Authorization: Bearer ${key}"` : ""}` },
        {
          label: "Chat",
          code: `curl ${openai}/chat/completions \\
  -H "Content-Type: application/json" \\${key ? `\n  -H "Authorization: Bearer ${key}" \\` : ""}
  -d '{"model": "${model}", "messages": [{"role": "user", "content": "Hello!"}]}'`,
        },
      ],
    },
  ];
}

export function ConnectPage() {
  const endpoints = useEndpoints();
  const { data: settings } = useSettings();
  const instance = useLive((live) => live.instance);
  const listen = useLive((live) => live.launcher?.listen ?? null);
  const [apiKey, setApiKey] = useState(() => readStorage(KEY_STORAGE) ?? "");

  const models = useQuery<PublicModel[]>({
    queryKey: ["public-models", apiKey],
    queryFn: () => publicModels(apiKey),
    staleTime: 10_000,
    retry: 0,
  });

  const defaultModel = useMemo(() => {
    const list = models.data ?? [];
    return instance?.alias ?? list.find((model) => model.state === "loaded")?.id ?? list[0]?.id ?? "MODEL_ID";
  }, [models.data, instance]);
  const [chosen, setChosen] = useState<string | null>(null);
  const model = chosen ?? defaultModel;

  const keyRequired = (settings?.api_keys.length ?? 0) > 0;
  const shownKey = keyRequired ? (apiKey.length > 0 ? apiKey : "YOUR_API_KEY") : null;
  const chosenEntry = models.data?.find((entry) => entry.id === model);
  const context = chosenEntry?.context_length ?? chosenEntry?.max_model_len ?? null;
  const list = snippets(endpoints.openai, endpoints.anthropic, model, context, shownKey, settings?.llamacpp_api ?? false);

  return (
    <div className="flex flex-col gap-5">
      <PageHeader
        title="Connect"
        description="Point coding tools at the launcher once. They see every listed model, and the launcher loads whichever one a request names."
      />

      <div className="grid gap-5 xl:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>
              <Plug className="size-4 text-muted-foreground" />
              Endpoints
            </CardTitle>
            <CardDescription>OpenAI-compatible and Anthropic-compatible, on the same port</CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-3">
            <div>
              <div className="mb-1 text-xs font-medium text-muted-foreground">OpenAI base URL (Chat Completions, Responses)</div>
              <div className="flex items-center gap-2 rounded-lg border bg-muted/40 px-3 py-2">
                <code className="min-w-0 flex-1 truncate font-mono text-sm">{endpoints.openai}</code>
                <CopyButton text={endpoints.openai} size="icon-xs" toastMessage="OpenAI base URL copied" />
              </div>
            </div>
            <div>
              <div className="mb-1 text-xs font-medium text-muted-foreground">Anthropic base URL (Messages)</div>
              <div className="flex items-center gap-2 rounded-lg border bg-muted/40 px-3 py-2">
                <code className="min-w-0 flex-1 truncate font-mono text-sm">{endpoints.anthropic}</code>
                <CopyButton text={endpoints.anthropic} size="icon-xs" toastMessage="Anthropic base URL copied" />
              </div>
            </div>
            <div className="flex items-start gap-2.5 rounded-lg border px-3 py-2.5 text-sm">
              {keyRequired ? (
                <ShieldCheck className="mt-0.5 size-4 shrink-0 text-success" />
              ) : (
                <ShieldOff className="mt-0.5 size-4 shrink-0 text-muted-foreground" />
              )}
              <div>
                <div className="font-medium">{keyRequired ? "API key required" : "No API key"}</div>
                <div className="text-xs text-muted-foreground">
                  {keyRequired
                    ? "Tools send one of the launcher's API keys as a bearer token, x-api-key or api-key. Create and revoke keys under Settings."
                    : `Anyone who can reach ${listen ? `${listen.host}:${listen.port}` : "the launcher"} can use the models; create an API key under Settings before listening beyond this computer.`}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>
              <ArrowRightLeft className="size-4 text-muted-foreground" />
              How switching works
            </CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col gap-2.5 text-sm text-muted-foreground">
            <p>
              Each request names a model. If it is the loaded one, the launcher forwards the request at once. If another
              model is loaded, the launcher lets that model's running requests finish (up to {settings?.drain_timeout_s ?? 120}{" "}
              s), stops it, starts the requested model with your saved configuration, and then serves every request that
              waited for it.
            </p>
            <p>
              {settings?.unknown_model === "reject"
                ? "A model id the launcher does not know is rejected with model_not_found."
                : "A model id the launcher does not know goes to the loaded model, or to the default model when none is loaded (tools with fixed model names keep working)."}
            </p>
            <p>
              Loading a large model takes from seconds to a couple of minutes. A streaming request gets a keep-alive every 5
              s meanwhile; a non-streaming one waits without a response, so give tools a generous timeout. A model whose load
              failed is refused at once for 2 minutes (with Retry-After) instead of being loaded again by every retry; Load
              in the UI, a configuration change or a Settings change ends that.
            </p>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>What tools see</CardTitle>
          <CardDescription>GET {endpoints.openai}/models, exactly as a coding tool requests it</CardDescription>
          <CardAction>
            <Button variant="ghost" size="icon-sm" aria-label="Refresh" onClick={() => void models.refetch()}>
              <RefreshCw className={models.isFetching ? "animate-spin" : undefined} />
            </Button>
          </CardAction>
        </CardHeader>
        <CardContent className="flex flex-col gap-3">
          {keyRequired ? (
            <div className="flex max-w-md items-center gap-2">
              <KeyRound className="size-4 shrink-0 text-muted-foreground" />
              <Input
                type="password"
                value={apiKey}
                placeholder="API key (kept in this browser only)"
                onChange={(event) => {
                  setApiKey(event.target.value);
                  writeStorage(KEY_STORAGE, event.target.value.length > 0 ? event.target.value : null);
                }}
                aria-label="API key"
              />
            </div>
          ) : null}
          {models.isError ? (
            <Alert variant="warning">
              <AlertTitle>The model list could not be read</AlertTitle>
              <AlertDescription>{errorMessage(models.error)}</AlertDescription>
            </Alert>
          ) : (models.data ?? []).length === 0 ? (
            <div className="rounded-lg border border-dashed p-6 text-center text-sm text-muted-foreground">
              {models.isLoading ? "Reading…" : "No model is listed yet: add a model folder or download a model."}
            </div>
          ) : (
            <div className="rounded-lg border">
              <Table>
                <TableHeader>
                  <TableRow className="hover:bg-transparent">
                    <TableHead>Model id</TableHead>
                    <TableHead className="text-right">Context</TableHead>
                    <TableHead className="text-right">Default output</TableHead>
                    <TableHead>State</TableHead>
                    <TableHead className="w-10" />
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {(models.data ?? []).map((item) => (
                    <TableRow key={item.id}>
                      <TableCell className="font-mono text-xs">{item.id}</TableCell>
                      <TableCell className="text-right tabular-nums">
                        {formatContext(item.context_length ?? item.max_model_len ?? null)}
                      </TableCell>
                      <TableCell className="text-right tabular-nums">{formatContext(item.max_tokens ?? null)}</TableCell>
                      <TableCell>
                        <Badge variant={item.state === "loaded" ? "accent" : "muted"}>{item.state ?? "—"}</Badge>
                      </TableCell>
                      <TableCell>
                        <CopyButton text={item.id} size="icon-xs" toastMessage="Model id copied" />
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Set up a tool</CardTitle>
          <CardDescription>Snippets filled in with this launcher's address and the model you pick</CardDescription>
          <CardAction>
            <Select value={model} onValueChange={setChosen}>
              <SelectTrigger size="sm" className="w-56" aria-label="Model for the snippets">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {(models.data ?? []).map((item) => (
                  <SelectItem key={item.id} value={item.id}>
                    {item.id}
                  </SelectItem>
                ))}
                {(models.data ?? []).some((item) => item.id === model) ? null : <SelectItem value={model}>{model}</SelectItem>}
              </SelectContent>
            </Select>
          </CardAction>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="claude-code">
            <TabsList className="h-auto flex-wrap justify-start">
              {list.map((snippet) => (
                <TabsTrigger key={snippet.id} value={snippet.id} className="flex-none">
                  {snippet.label}
                </TabsTrigger>
              ))}
            </TabsList>
            {list.map((snippet) => (
              <TabsContent key={snippet.id} value={snippet.id} className="flex flex-col gap-3">
                <p className="text-sm text-muted-foreground">{snippet.intro}</p>
                {snippet.blocks.map((block, index) => (
                  <CodeBlock key={index} code={block.code} label={block.label} />
                ))}
              </TabsContent>
            ))}
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
}
