import type { Architecture } from "../model/architecture.ts";

/**
 * Attention kinds charge memory at different rates, so they cannot be counted
 * together. A full-attention layer stores every token it has seen. A
 * sliding-window layer stores at most its window. A recurrent layer stores a
 * fixed state no matter how long the context is.
 */
export type LayerKind = "full_attention" | "sliding_attention" | "recurrent" | "unknown";

export interface LayerClass {
  kind: LayerKind;
  count: number;
}

export interface LayerBreakdown {
  classes: LayerClass[];
  /** Where the split came from, so the output can say which layer count it used. */
  source: "layer_types" | "num_hidden_layers";
  totalLayers: number;
}

/**
 * Splits the model's layers by attention kind.
 *
 * `layer_types` is published as a plain array by the architectures that mix
 * kinds, so this is exact arithmetic on structured data rather than a guess.
 * When it is absent, every layer is treated as the same kind, which is correct
 * for the uniform architectures that omit it.
 */
export function classifyLayers(architecture: Architecture): LayerBreakdown | null {
  const declared = architecture.layerTypes;

  if (declared !== null) {
    const counts = new Map<LayerKind, number>();
    for (const raw of declared) {
      const kind = normaliseLayerKind(raw);
      counts.set(kind, (counts.get(kind) ?? 0) + 1);
    }
    return {
      classes: [...counts.entries()].map(([kind, count]) => ({ kind, count })),
      source: "layer_types",
      totalLayers: declared.length,
    };
  }

  const total = architecture.layers?.value ?? null;
  if (total === null) return null;

  // A window that the config explicitly switches off does not apply.
  const windowed =
    architecture.slidingWindow !== null &&
    architecture.slidingWindow > 0 &&
    architecture.useSlidingWindow === true;

  return {
    classes: [{ kind: windowed ? "sliding_attention" : "full_attention", count: total }],
    source: "num_hidden_layers",
    totalLayers: total,
  };
}

/**
 * Vendors spell these differently. Anything unrecognised is charged the full
 * per-token rate, which errs toward needing more memory rather than less.
 */
export function normaliseLayerKind(raw: string): LayerKind {
  const value = raw.toLowerCase();
  if (value.includes("linear") || value.includes("mamba") || value.includes("recurrent") || value.includes("ssm")) {
    return "recurrent";
  }
  if (value.includes("sliding") || value.includes("local")) return "sliding_attention";
  if (value.includes("full") || value.includes("attention")) return "full_attention";
  return "unknown";
}

/** How many tokens one layer of this kind holds in cache at the assumed context. */
export function cachedTokensFor(
  kind: LayerKind,
  contextTokens: number,
  slidingWindow: number | null,
): number {
  switch (kind) {
    case "recurrent":
      // The state is fixed size; it does not scale with context and is
      // accounted for separately.
      return 0;
    case "sliding_attention":
      return slidingWindow === null || slidingWindow <= 0
        ? contextTokens
        : Math.min(contextTokens, slidingWindow);
    case "full_attention":
    case "unknown":
    default:
      return contextTokens;
  }
}
