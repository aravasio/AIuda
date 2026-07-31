import { isRecord } from "../hf/client.ts";

/**
 * Where a number came from. Printed next to the number so a reader can tell
 * a measured value from a derived one, and never confuse either with a guess.
 */
export type FieldSource =
  | "config.json"
  | "config.json:text_config"
  | "derived"
  | "safetensors metadata"
  | "repo file listing"
  | "model name"
  | "estimate";

export interface SourcedNumber {
  value: number;
  source: FieldSource;
}

export interface MoeInfo {
  /** Total number of routed experts. */
  experts: number | null;
  /** How many experts run per token. Drives speed, not memory. */
  expertsPerToken: number | null;
}

/**
 * Fields describing a recurrent (linear-attention) layer's state. Such a layer
 * keeps a fixed-size state instead of a cache that grows with every token, so
 * it must not be charged the per-token rate.
 */
export interface LinearAttentionInfo {
  numKeyHeads: number | null;
  keyHeadDim: number | null;
  numValueHeads: number | null;
  valueHeadDim: number | null;
  convKernelDim: number | null;
  /** Runtimes commonly hold the recurrent state at higher precision than the KV cache. */
  stateDtype: string | null;
}

export interface Architecture {
  /** e.g. "qwen3_5_moe". */
  modelType: string | null;
  architectures: string[];
  layers: SourcedNumber | null;
  hiddenSize: SourcedNumber | null;
  attentionHeads: SourcedNumber | null;
  /** Grouped-query models have fewer of these than attention heads; the KV cache scales with these. */
  kvHeads: SourcedNumber | null;
  headDim: SourcedNumber | null;
  /** The model's own maximum, which is not the same as what a runtime will honour. */
  maxContext: SourcedNumber | null;
  torchDtype: string | null;
  moe: MoeInfo | null;
  /**
   * Per-layer attention kind, when the repo publishes it. Modern architectures
   * mix full attention with sliding-window or recurrent layers, and each kind
   * charges a different amount of memory per token.
   */
  layerTypes: string[] | null;
  /** Window size for sliding-window attention layers. */
  slidingWindow: number | null;
  /** Some repos ship a window size but switch it off. */
  useSlidingWindow: boolean | null;
  linearAttention: LinearAttentionInfo | null;
  /** True when the architecture fields were read out of a nested sub-config. */
  nested: boolean;
  /** Fields we looked for and could not find. Reported to the user rather than filled in. */
  missing: string[];
}

/**
 * Multimodal repos put the language-model fields one level down. The spec names
 * `text_config`; the other two are the same idea under different vendor spellings
 * and are cheap to accept.
 */
const NESTED_KEYS = ["text_config", "llm_config", "language_config"] as const;

export function parseArchitecture(config: Record<string, unknown> | null): Architecture | null {
  if (config === null) return null;

  const nestedConfig = findNested(config);
  const missing: string[] = [];

  const layers = pick(config, nestedConfig, ["num_hidden_layers", "n_layer", "num_layers"]);
  const hiddenSize = pick(config, nestedConfig, ["hidden_size", "n_embd", "d_model"]);
  const attentionHeads = pick(config, nestedConfig, ["num_attention_heads", "n_head"]);
  const kvHeadsRaw = pick(config, nestedConfig, ["num_key_value_heads", "num_kv_heads"]);
  const headDimRaw = pick(config, nestedConfig, ["head_dim"]);
  const maxContext = pick(config, nestedConfig, [
    "max_position_embeddings",
    "n_positions",
    "max_sequence_length",
  ]);

  // No `num_key_value_heads` means multi-head attention, where every attention
  // head carries its own KV. Falling back is correct, not a guess.
  const kvHeads: SourcedNumber | null =
    kvHeadsRaw ?? (attentionHeads !== null ? { value: attentionHeads.value, source: "derived" } : null);

  // `head_dim` is only stated when it is not hidden_size / heads, so deriving it is exact.
  let headDim: SourcedNumber | null = headDimRaw;
  if (headDim === null && hiddenSize !== null && attentionHeads !== null && attentionHeads.value > 0) {
    headDim = { value: hiddenSize.value / attentionHeads.value, source: "derived" };
  }

  if (layers === null) missing.push("layer count");
  if (kvHeads === null) missing.push("key/value head count");
  if (headDim === null) missing.push("head dimension");
  if (maxContext === null) missing.push("maximum context length");

  const source = nestedConfig ?? config;

  return {
    modelType: typeof config["model_type"] === "string" ? config["model_type"] : null,
    architectures: Array.isArray(config["architectures"])
      ? config["architectures"].filter((a): a is string => typeof a === "string")
      : [],
    layers,
    hiddenSize,
    attentionHeads,
    kvHeads,
    headDim,
    maxContext,
    // Newer configs spell this "dtype"; older ones "torch_dtype". A nested
    // sub-config wins, since that is where the language model's own weights are
    // described on a multimodal repo.
    torchDtype: firstString(
      [source, config],
      ["torch_dtype", "dtype"],
    ),
    moe: parseMoe(config, nestedConfig),
    layerTypes: parseLayerTypes(config, nestedConfig),
    slidingWindow: pick(config, nestedConfig, ["sliding_window"])?.value ?? null,
    useSlidingWindow: parseBool(config, nestedConfig, "use_sliding_window"),
    linearAttention: parseLinearAttention(config, nestedConfig),
    nested: nestedConfig !== null,
    missing,
  };
}

/** First string found for any of `keys`, searching each source in order. */
function firstString(
  sources: Array<Record<string, unknown> | null>,
  keys: string[],
): string | null {
  for (const source of sources) {
    if (source === null) continue;
    for (const key of keys) {
      const value = source[key];
      if (typeof value === "string" && value !== "") return value;
    }
  }
  return null;
}

function parseLayerTypes(
  config: Record<string, unknown>,
  nested: Record<string, unknown> | null,
): string[] | null {
  for (const source of [config, nested]) {
    if (source === null) continue;
    const value = source["layer_types"];
    if (Array.isArray(value) && value.every((v) => typeof v === "string") && value.length > 0) {
      return value as string[];
    }
  }
  return null;
}

function parseBool(
  config: Record<string, unknown>,
  nested: Record<string, unknown> | null,
  key: string,
): boolean | null {
  for (const source of [config, nested]) {
    if (source === null) continue;
    const value = source[key];
    if (typeof value === "boolean") return value;
  }
  return null;
}

function parseLinearAttention(
  config: Record<string, unknown>,
  nested: Record<string, unknown> | null,
): LinearAttentionInfo | null {
  const numKeyHeads = pick(config, nested, ["linear_num_key_heads"]);
  const numValueHeads = pick(config, nested, ["linear_num_value_heads"]);
  if (numKeyHeads === null && numValueHeads === null) return null;
  const source = nested ?? config;
  return {
    numKeyHeads: numKeyHeads?.value ?? null,
    keyHeadDim: pick(config, nested, ["linear_key_head_dim"])?.value ?? null,
    numValueHeads: numValueHeads?.value ?? null,
    valueHeadDim: pick(config, nested, ["linear_value_head_dim"])?.value ?? null,
    convKernelDim: pick(config, nested, ["linear_conv_kernel_dim"])?.value ?? null,
    stateDtype:
      typeof source["mamba_ssm_dtype"] === "string"
        ? source["mamba_ssm_dtype"]
        : typeof config["mamba_ssm_dtype"] === "string"
          ? config["mamba_ssm_dtype"]
          : null,
  };
}

/**
 * Only treat a sub-config as the source of truth when the top level genuinely
 * lacks the language-model fields. A repo that states them at both levels keeps
 * the top-level values.
 */
function findNested(config: Record<string, unknown>): Record<string, unknown> | null {
  if (typeof config["num_hidden_layers"] === "number") return null;
  for (const key of NESTED_KEYS) {
    const candidate = config[key];
    if (isRecord(candidate) && typeof candidate["num_hidden_layers"] === "number") {
      return candidate;
    }
  }
  return null;
}

function pick(
  top: Record<string, unknown>,
  nested: Record<string, unknown> | null,
  keys: string[],
): SourcedNumber | null {
  for (const key of keys) {
    const value = top[key];
    if (typeof value === "number" && Number.isFinite(value)) {
      return { value, source: "config.json" };
    }
  }
  if (nested !== null) {
    for (const key of keys) {
      const value = nested[key];
      if (typeof value === "number" && Number.isFinite(value)) {
        return { value, source: "config.json:text_config" };
      }
    }
  }
  return null;
}

const EXPERT_KEYS = ["num_experts", "num_local_experts", "n_routed_experts", "num_routed_experts"];
const EXPERTS_PER_TOKEN_KEYS = ["num_experts_per_tok", "moe_topk", "num_selected_experts", "topk"];

function parseMoe(
  config: Record<string, unknown>,
  nested: Record<string, unknown> | null,
): MoeInfo | null {
  const experts = pick(config, nested, EXPERT_KEYS);
  const perToken = pick(config, nested, EXPERTS_PER_TOKEN_KEYS);
  if (experts === null && perToken === null) return null;
  return {
    experts: experts?.value ?? null,
    expertsPerToken: perToken?.value ?? null,
  };
}
