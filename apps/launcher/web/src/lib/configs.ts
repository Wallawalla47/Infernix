// Named configurations: the query-cache updates after the configuration routes answer, the model
// id's validation, and the mutations the Models and Configure pages share. A configuration id is an
// artifact key (its default configuration) or `<artifact key>@<8 hex>` (see library.ts).

import { useMutation, useQueryClient, type QueryClient } from "@tanstack/react-query";
import { toast } from "sonner";

import { api } from "./api";
import type { EnvVar, InstanceInfo, Library, ModelArtifact, ModelConfig, ModelConfigPatch } from "./api-types";
import { allConfigs, artifactKeyOf } from "./library";
import { downloadText } from "./utils";

/** The Configure page of a configuration. */
export function configPath(configId: string): string {
  return `/models/${configId}`;
}

/** The newest of several copies of one configuration (the query cache, the library). */
export function freshestConfig(...candidates: (ModelConfig | null | undefined)[]): ModelConfig | null {
  let best: ModelConfig | null = null;
  for (const candidate of candidates) {
    if (!candidate) continue;
    if (best === null || (candidate.updated_at ?? 0) > (best.updated_at ?? 0)) best = candidate;
  }
  return best;
}

/** Writes a configuration the launcher answered into the cache: its own query and its artifact's
 * `configs` (and `config` for the default one) in the library. */
export function storeConfig(qc: QueryClient, config: ModelConfig): void {
  qc.setQueryData(["config", config.id], config);
  qc.setQueryData<Library>(["library"], (old) => {
    if (!old) return old;
    return {
      ...old,
      models: old.models.map((model) => {
        if (model.key !== config.key) return model;
        const known = model.configs.some((item) => item.id === config.id);
        const configs = known
          ? model.configs.map((item) => (item.id === config.id ? config : item))
          : [...model.configs, config];
        return { ...model, configs, config: config.is_default ? config : model.config };
      }),
    };
  });
}

function dropConfig(qc: QueryClient, configId: string): void {
  qc.removeQueries({ queryKey: ["config", configId], exact: true });
  const key = artifactKeyOf(configId);
  qc.setQueryData<Library>(["library"], (old) =>
    old
      ? {
          ...old,
          models: old.models.map((model) =>
            model.key === key ? { ...model, configs: model.configs.filter((item) => item.id !== configId) } : model,
          ),
        }
      : old,
  );
}

/** "<artifact name> (<configuration name>)", or the artifact name alone for its default. */
export function describeConfig(model: ModelArtifact, config: ModelConfig): string {
  return config.is_default ? model.name : `${model.name} (${config.name})`;
}

/** Why a model id cannot be used, or null. Ids are compared ignoring case, as requests match them;
 * the launcher checks again and answers 400 with its own reason. */
export function aliasProblem(alias: string, library: Library | null | undefined, configId: string | null): string | null {
  const text = alias.trim();
  if (text.length === 0) return "The model id cannot be empty";
  if (/\s/.test(text)) return "The model id cannot contain spaces";
  const lower = text.toLowerCase();
  const clash = allConfigs(library).find(
    ({ config }) => config.id !== configId && config.alias.toLowerCase() === lower,
  );
  return clash ? `${describeConfig(clash.model, clash.config)} already uses this model id` : null;
}

/** Why a configuration name cannot be used for this artifact, or null. */
export function configNameProblem(name: string, model: ModelArtifact | null, configId: string | null): string | null {
  const text = name.trim();
  if (text.length === 0) return "The name cannot be empty";
  const lower = text.toLowerCase();
  if (lower === "default") return "\"Default\" names the artifact's default configuration";
  const clash = model?.configs.find((config) => config.id !== configId && config.name.toLowerCase() === lower);
  return clash ? `This artifact already has a configuration named ${clash.name}` : null;
}

/** Variable rows as the launcher stores them: names trimmed, rows with neither name nor value
 * dropped. */
export function cleanEnv(env: readonly EnvVar[]): EnvVar[] {
  return env
    .map((row) => ({ name: row.name.trim(), value: row.value }))
    .filter((row) => row.name.length > 0 || row.value.length > 0);
}

/** Why a variable row cannot be saved, or null (`index` into `env`). Names are compared ignoring
 * case, as Windows does. */
export function envRowProblem(env: readonly EnvVar[], index: number): string | null {
  const row = env[index];
  const name = row.name.trim();
  if (name.length === 0) return row.value.length > 0 ? "Name the variable" : null;
  if (/[\s=]/.test(name)) return "No spaces or = in a name";
  const lower = name.toLowerCase();
  const first = env.findIndex((other) => other.name.trim().toLowerCase() === lower);
  return first !== index ? `${name} is set twice` : null;
}

export function envProblem(env: readonly EnvVar[]): string | null {
  for (let index = 0; index < env.length; index += 1) {
    const problem = envRowProblem(env, index);
    if (problem) return problem;
  }
  return null;
}

export function sameEnv(a: readonly EnvVar[], b: readonly EnvVar[]): boolean {
  if (a.length !== b.length) return false;
  return a.every((row, index) => row.name === b[index].name && row.value === b[index].value);
}

// ---------------------------------------------------------------------------------------------
// What a running server holds

/** The instance while a server holds a model (starting, ready or stopping), else null. */
export function liveInstance(instance: InstanceInfo | null): InstanceInfo | null {
  if (!instance) return null;
  return instance.state === "starting" || instance.state === "ready" || instance.state === "stopping" ? instance : null;
}

/** The artifact key of the model a server holds, or null. */
export function loadedArtifactKey(instance: InstanceInfo | null): string | null {
  const live = liveInstance(instance);
  if (!live || !live.model_key) return null;
  return live.artifact_key ?? artifactKeyOf(live.model_key);
}

function samePath(a: string, b: string): boolean {
  // Windows paths compare ignoring case and separator style.
  const norm = (path: string) => path.replace(/\\/g, "/").toLowerCase();
  return norm(a) === norm(b);
}

/** Whether the running server's command line names this file (`--flag path` or `--flag=path`). */
export function pathInUse(instance: InstanceInfo | null, path: string): boolean {
  const live = liveInstance(instance);
  if (!live) return false;
  return live.argv.some((arg) => {
    if (samePath(arg, path)) return true;
    const eq = arg.startsWith("--") ? arg.indexOf("=") : -1;
    return eq > 0 && samePath(arg.slice(eq + 1), path);
  });
}

/** Whether the running server reads this n-gram volume: named on its command line, or the loaded
 * artifact's own `<entry>.ngram` that serve opens when no --ngram-volume is given. */
export function volumeInUse(instance: InstanceInfo | null, path: string, ownFor: readonly string[]): boolean {
  if (pathInUse(instance, path)) return true;
  const loaded = loadedArtifactKey(instance);
  const live = liveInstance(instance);
  if (!live || loaded === null || !ownFor.includes(loaded)) return false;
  return !live.argv.some((arg) => arg === "--ngram-volume" || arg.startsWith("--ngram-volume="));
}

// ---------------------------------------------------------------------------------------------
// Mutations

export function useSaveConfig() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, patch }: { id: string; patch: ModelConfigPatch }) => api.saveModelConfig(id, patch),
    onSuccess: (config) => storeConfig(qc, config),
  });
}

export interface CreateConfigBody {
  name: string;
  /** Configuration id to copy; the defaults when absent. */
  from?: string;
  alias?: string;
  args?: string[];
  env?: EnvVar[];
}

export function useCreateConfig() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ artifactKey, body }: { artifactKey: string; body: CreateConfigBody }) =>
      api.createModelConfig(artifactKey, body),
    onSuccess: (config) => storeConfig(qc, config),
  });
}

export function useResetConfig() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => api.resetModelConfig(id),
    onSuccess: (config) => storeConfig(qc, config),
  });
}

export function useDeleteConfig() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => api.deleteModelConfig(id),
    onSuccess: (_answer, id) => {
      dropConfig(qc, id);
      void qc.invalidateQueries({ queryKey: ["library"] });
    },
  });
}

export type FileKind = "model" | "drafter" | "ngram";

/** Deletes an artifact's, a drafter's or an n-gram volume's files; the launcher answers the new
 * library. `confirm` is the file name the user typed. */
export function useDeleteFiles() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ kind, key, confirm }: { kind: FileKind; key: string; confirm: string }) =>
      kind === "model"
        ? api.deleteModelFiles(key, confirm)
        : kind === "drafter"
          ? api.deleteDrafter(key, confirm)
          : api.deleteNgramVolume(key, confirm),
    onSuccess: (library, { kind, key }) => {
      qc.setQueryData(["library"], library);
      if (kind === "model") {
        qc.removeQueries({
          predicate: (query) => {
            const configId = query.queryKey[1];
            return query.queryKey[0] === "config" && typeof configId === "string" && artifactKeyOf(configId) === key;
          },
        });
      }
    },
  });
}

export function useImportScript() {
  return useMutation({
    mutationFn: ({ text, fileName }: { text: string; fileName?: string }) => api.importScript(text, fileName),
  });
}

/** Saves the configuration as a launch script (the saved parameters, not unsaved edits). */
export function useDownloadScript() {
  return useMutation({
    mutationFn: ({ id, shell }: { id: string; shell: "cmd" | "sh" }) => api.script(id, shell),
    onSuccess: (script) => {
      downloadText(script.file_name, script.text);
      toast.success(`Saved ${script.file_name}`);
    },
  });
}

/** `infernix-serve --plan` for these parameters (a few seconds). */
export function usePlan() {
  return useMutation({
    mutationFn: ({ id, args, env }: { id: string; args: string[]; env: EnvVar[] }) => api.plan(id, { args, env }),
  });
}
