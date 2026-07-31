import { isRecord } from "../hf/client.ts";
import type { RepoSnapshot } from "../hf/types.ts";

/**
 * How the repo has to be read. This decides where the description comes from
 * and which numbers are trustworthy, so it is settled before anything else runs.
 */
export type RepoKind =
  /** Its own model: own config.json, own weights. */
  | "standard"
  /** Somebody's model re-encoded at lower precision. Description belongs to the base. */
  | "quantization"
  /** A delta on top of another model. Description belongs to the base. */
  | "adapter"
  /** Neither a config nor recognisable weights. Metadata and file sizes only. */
  | "metadata-only";

export interface RepoFlags {
  /** Needs an access token before any file can be read. */
  gated: boolean;
  /** Language-model fields live under `text_config` rather than at the top level. */
  multimodal: boolean;
  /** Ships GGUF weights and nothing else. */
  ggufOnly: boolean;
  /** Quantised by someone other than the author of the base model. */
  thirdParty: boolean;
  /** Calibrated k-quants. */
  imatrix: boolean;
  /** Produced by merging other models. */
  merge: boolean;
  hasConfig: boolean;
  hasWeights: boolean;
}

export interface Classification {
  kind: RepoKind;
  flags: RepoFlags;
  /** Repo to read the description from, when this repo is not its own subject. */
  baseModel: string | null;
  /** Fields this repo cannot supply, named so the output can say so instead of guessing. */
  unavailable: string[];
}

const QUANT_NAME_PATTERN = /(gguf|awq|gptq|exl2|exl3|mlx|bnb|[-_](int4|int8|4bit|8bit|w4a16|w8a8|fp8))/i;
const QUANT_TAGS = new Set(["gguf", "awq", "gptq", "exl2", "mlx", "bitsandbytes", "compressed-tensors"]);

export function classify(snapshot: RepoSnapshot): Classification {
  const { info, files, config } = snapshot;
  const paths = files.map((f) => f.path);
  const lowerPaths = paths.map((p) => p.toLowerCase());
  const tags = info.tags.map((t) => t.toLowerCase());

  const hasGguf = lowerPaths.some((p) => p.endsWith(".gguf"));
  const hasSafetensors = lowerPaths.some((p) => p.endsWith(".safetensors"));
  const hasPytorch = lowerPaths.some((p) => /(pytorch_model|consolidated).*\.(bin|pth)$/i.test(p));
  const hasWeights = hasGguf || hasSafetensors || hasPytorch;

  const adapterConfig = lowerPaths.some((p) => p.endsWith("adapter_config.json"));
  const isAdapter =
    adapterConfig ||
    lowerPaths.some((p) => /adapter_model\.(safetensors|bin)$/.test(p)) ||
    tags.includes("peft") ||
    tags.includes("lora");

  const quantConfig = config !== null && isRecord(config["quantization_config"]);
  const isQuantization =
    !isAdapter &&
    (hasGguf ||
      quantConfig ||
      QUANT_NAME_PATTERN.test(info.repoId.split("/")[1] ?? "") ||
      tags.some((t) => QUANT_TAGS.has(t)) ||
      info.tags.some((t) => t.startsWith("base_model:quantized:")));

  const baseModel = resolveBaseModel(snapshot, isQuantization || isAdapter);

  const flags: RepoFlags = {
    gated: info.gated,
    multimodal: hasNestedTextConfig(config),
    ggufOnly: hasGguf && !hasSafetensors && !hasPytorch,
    thirdParty: isThirdParty(info.repoId, baseModel),
    imatrix: lowerPaths.some((p) => /imat/i.test(p)),
    merge: tags.includes("merge") || tags.includes("mergekit"),
    hasConfig: config !== null,
    hasWeights,
  };

  let kind: RepoKind;
  if (isAdapter) kind = "adapter";
  else if (isQuantization) kind = "quantization";
  else if (config === null && !hasWeights) kind = "metadata-only";
  else if (config === null) kind = "metadata-only";
  else kind = "standard";

  return { kind, flags, baseModel, unavailable: unavailableFields(kind, flags, snapshot) };
}

function hasNestedTextConfig(config: Record<string, unknown> | null): boolean {
  if (config === null) return false;
  if (typeof config["num_hidden_layers"] === "number") return false;
  for (const key of ["text_config", "llm_config", "language_config"]) {
    const nested = config[key];
    if (isRecord(nested) && typeof nested["num_hidden_layers"] === "number") return true;
  }
  return false;
}

function resolveBaseModel(snapshot: RepoSnapshot, derivative: boolean): string | null {
  const declared = snapshot.info.baseModels[0] ?? null;
  if (declared !== null) return declared;
  if (!derivative) return null;

  // Third-party quantisers frequently skip the metadata. The name almost always
  // carries the base: "TheBloke/Qwen3-8B-GGUF" comes from something-Qwen3-8B.
  const name = snapshot.info.repoId.split("/")[1] ?? "";
  const stripped = name.replace(/[-_](gguf|awq|gptq|exl2|exl3|mlx|bnb|int4|int8|4bit|8bit|i1)$/i, "");
  return stripped !== name && stripped.length > 0 ? stripped : null;
}

/**
 * "Third party" means the quantiser is not the base model's author. It is worth
 * naming because the base model's vendor never reviewed the result.
 */
function isThirdParty(repoId: string, baseModel: string | null): boolean {
  if (baseModel === null || !baseModel.includes("/")) return false;
  const owner = repoId.split("/")[0]?.toLowerCase();
  const baseOwner = baseModel.split("/")[0]?.toLowerCase();
  return owner !== undefined && baseOwner !== undefined && owner !== baseOwner;
}

function unavailableFields(kind: RepoKind, flags: RepoFlags, snapshot: RepoSnapshot): string[] {
  const missing: string[] = [];
  if (!flags.hasConfig) {
    missing.push("layer count", "key/value head count", "head dimension", "maximum context length");
  }
  if (snapshot.info.safetensorsTotalParams === null && !flags.hasConfig) {
    missing.push("exact parameter count");
  }
  if (kind === "adapter") {
    missing.push("standalone memory footprint, an adapter is loaded on top of its base model");
  }
  if (!flags.hasWeights && kind !== "adapter") {
    missing.push("weight file sizes");
  }
  return [...new Set(missing)];
}
