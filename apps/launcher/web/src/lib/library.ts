// Configuration ids and the library: a configuration id is an artifact key (its default
// configuration) or `<artifact key>@<8 hex>` (a named one). Instance, activity and incident
// `model_key` fields hold configuration ids.

import type { Library, ModelArtifact, ModelConfig } from "./api-types";

/** The artifact key of a configuration id. */
export function artifactKeyOf(configId: string): string {
  const at = configId.indexOf("@");
  return at < 0 ? configId : configId.slice(0, at);
}

export function isDefaultConfigId(configId: string): boolean {
  return !configId.includes("@");
}

/** The artifact a configuration id belongs to. */
export function findModel(library: Library | null | undefined, configId: string | null | undefined): ModelArtifact | null {
  if (!library || !configId) return null;
  const key = artifactKeyOf(configId);
  return library.models.find((model) => model.key === key) ?? null;
}

/** The configuration with this id, with its artifact. */
export function findConfig(
  library: Library | null | undefined,
  configId: string | null | undefined,
): { model: ModelArtifact; config: ModelConfig } | null {
  const model = findModel(library, configId);
  if (!model || !configId) return null;
  const config = model.configs.find((candidate) => candidate.id === configId);
  return config ? { model, config } : null;
}

/** Every configuration in the library, artifact by artifact (defaults first). */
export function allConfigs(library: Library | null | undefined): { model: ModelArtifact; config: ModelConfig }[] {
  if (!library) return [];
  return library.models.flatMap((model) => model.configs.map((config) => ({ model, config })));
}
