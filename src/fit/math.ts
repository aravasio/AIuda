import type { Architecture } from "../model/architecture.ts";
import { bytesPerParam } from "../model/quant.ts";
import { cachedTokensFor, classifyLayers, type LayerBreakdown, type LayerKind } from "./layers.ts";

/** The context the fit math assumes when the user does not say otherwise. */
export const DEFAULT_CONTEXT_TOKENS = 8192;

/**
 * Flat allowance for the runtime process, its GPU context and activation
 * buffers.
 *
 * This is an assumption, not a measurement, and it is the largest single source
 * of error in the whole calculation — larger than any of the components it sits
 * next to. It is roughly what llama.cpp and ollama occupy above the weights on
 * a mid-size model at a modest batch size. It is printed as its own line and
 * named as an assumption in the output, and it can be changed in the config
 * file (`runtimeOverheadBytes`) for a runtime that behaves differently.
 */
export const RUNTIME_OVERHEAD_BYTES = 1_000_000_000;

/** Fraction of fast memory that must remain free for a fit to count as comfortable. */
export const COMFORTABLE_HEADROOM = 0.2;

/** KV cache entries are held at 16-bit unless a runtime is told otherwise. */
export const DEFAULT_KV_BYTES_PER_ELEMENT = 2;

/**
 * A measured-but-small component is noted rather than added while it stays
 * under this share of the total. Above it, the component is added, because
 * silently dropping a meaningful fraction of the requirement is the kind of
 * quiet error this tool exists to prevent.
 */
export const NEGLIGIBLE_COMPONENT_FRACTION = 0.05;

export interface WeightsResult {
  bytes: number;
  source: "repo file listing" | "estimate";
  /** The format the number describes, e.g. "BF16" or "Q4_K_M". */
  quant: string | null;
}

/**
 * A real file size always wins. An estimate is only reached when the repo
 * publishes no usable listing, and the result says which one was used.
 */
export function computeWeightsBytes(input: {
  realBytes: number | null;
  paramsTotal: number | null;
  quant: string | null;
}): WeightsResult | null {
  if (input.realBytes !== null && input.realBytes > 0) {
    return { bytes: input.realBytes, source: "repo file listing", quant: input.quant };
  }
  if (input.paramsTotal === null || input.quant === null) return null;
  const perParam = bytesPerParam(input.quant);
  if (perParam === null) return null;
  return {
    bytes: Math.round(input.paramsTotal * perParam),
    source: "estimate",
    quant: input.quant,
  };
}

/** One group of layers and what they contribute to the cache. */
export interface KvComponent {
  kind: LayerKind;
  layers: number;
  /** Tokens each of these layers holds. Zero for recurrent layers. */
  cachedTokens: number;
  bytes: number;
}

/**
 * A component that was measured but is small enough to leave out of the total.
 * It is always reported, so the reader knows it exists and what it is worth.
 */
export interface SideComponent {
  label: string;
  bytes: number;
  /** True when it was large enough that it had to be added to the total. */
  counted: boolean;
  note: string;
}

export interface KvCacheResult {
  bytes: number;
  components: KvComponent[];
  layerSource: LayerBreakdown["source"];
  /** Layers that actually hold a per-token cache, which is what the bytes above scale with. */
  cachingLayers: number;
  totalLayers: number;
}

export interface KvCacheInput {
  contextTokens: number;
  layers: LayerBreakdown;
  kvHeads: number;
  headDim: number;
  slidingWindow?: number | null;
  bytesPerElement?: number;
}

/**
 * Every cached token costs one key and one value vector per layer, sized by the
 * number of key/value heads rather than attention heads. On long context this
 * routinely exceeds the weights, which is why it is never folded into them.
 */
export function computeKvCache(input: KvCacheInput): KvCacheResult {
  const bytesPerElement = input.bytesPerElement ?? DEFAULT_KV_BYTES_PER_ELEMENT;
  const K_AND_V = 2;
  const slidingWindow = input.slidingWindow ?? null;

  const components: KvComponent[] = [];
  let bytes = 0;
  let cachingLayers = 0;

  for (const layerClass of input.layers.classes) {
    const cachedTokens = cachedTokensFor(layerClass.kind, input.contextTokens, slidingWindow);
    const componentBytes =
      K_AND_V * cachedTokens * layerClass.count * input.kvHeads * input.headDim * bytesPerElement;
    components.push({
      kind: layerClass.kind,
      layers: layerClass.count,
      cachedTokens,
      bytes: componentBytes,
    });
    bytes += componentBytes;
    if (cachedTokens > 0) cachingLayers += layerClass.count;
  }

  return {
    bytes,
    components,
    layerSource: input.layers.source,
    cachingLayers,
    totalLayers: input.layers.totalLayers,
  };
}

/**
 * Size of the fixed state held by recurrent layers.
 *
 * The exact tensor layout is architecture-specific and is not published, so
 * this is a derived figure, not a read one. It exists to answer one question:
 * is the component big enough that leaving it out would matter? It is labelled
 * as derived wherever it appears.
 */
export function estimateRecurrentStateBytes(
  architecture: Architecture,
  recurrentLayers: number,
): number | null {
  const linear = architecture.linearAttention;
  if (linear === null || recurrentLayers <= 0) return null;
  const { numValueHeads, keyHeadDim, valueHeadDim, numKeyHeads, convKernelDim } = linear;
  if (numValueHeads === null || keyHeadDim === null || valueHeadDim === null) return null;

  const stateBytesPerElement = /float32|fp32/i.test(linear.stateDtype ?? "float32") ? 4 : 2;
  const recurrentState = numValueHeads * keyHeadDim * valueHeadDim * stateBytesPerElement;

  // Short convolution over the key, value and query projections.
  const convState =
    convKernelDim === null || numKeyHeads === null
      ? 0
      : convKernelDim *
        (numKeyHeads * keyHeadDim * 2 + numValueHeads * valueHeadDim) *
        stateBytesPerElement;

  return (recurrentState + convState) * recurrentLayers;
}

export interface Breakdown {
  weights: WeightsResult | null;
  kv: KvCacheResult | null;
  runtimeOverheadBytes: number;
  /** Measured extras, each either added to the total or reported as too small to matter. */
  sideComponents: SideComponent[];
  /** Null when a component is unknown: an incomplete total is worse than none. */
  totalBytes: number | null;
  contextTokens: number;
  /** Named components we could not compute, so the output can say so. */
  unknown: string[];
}

export function computeBreakdown(input: {
  weights: WeightsResult | null;
  architecture: Architecture | null;
  contextTokens: number;
  kvBytesPerElement?: number;
  runtimeOverheadBytes?: number;
}): Breakdown {
  const unknown: string[] = [];
  const runtimeOverheadBytes = input.runtimeOverheadBytes ?? RUNTIME_OVERHEAD_BYTES;
  const architecture = input.architecture;

  const layers = architecture === null ? null : classifyLayers(architecture);
  const kvHeads = architecture?.kvHeads?.value ?? null;
  const headDim = architecture?.headDim?.value ?? null;

  let kv: KvCacheResult | null = null;
  if (layers !== null && kvHeads !== null && headDim !== null) {
    kv = computeKvCache({
      contextTokens: input.contextTokens,
      layers,
      kvHeads,
      headDim,
      slidingWindow: architecture?.slidingWindow ?? null,
      ...(input.kvBytesPerElement !== undefined
        ? { bytesPerElement: input.kvBytesPerElement }
        : {}),
    });
  } else {
    unknown.push("KV cache size");
  }

  if (input.weights === null) unknown.push("weights size");

  const base =
    input.weights !== null && kv !== null ? input.weights.bytes + kv.bytes + runtimeOverheadBytes : null;

  const sideComponents = collectSideComponents(architecture, kv, base);
  const countedExtra = sideComponents
    .filter((component) => component.counted)
    .reduce((sum, component) => sum + component.bytes, 0);

  return {
    weights: input.weights,
    kv,
    runtimeOverheadBytes,
    sideComponents,
    totalBytes: base === null ? null : base + countedExtra,
    contextTokens: input.contextTokens,
    unknown,
  };
}

function collectSideComponents(
  architecture: Architecture | null,
  kv: KvCacheResult | null,
  base: number | null,
): SideComponent[] {
  if (architecture === null || kv === null) return [];

  const recurrentLayers = kv.components
    .filter((component) => component.kind === "recurrent")
    .reduce((sum, component) => sum + component.layers, 0);

  const bytes = estimateRecurrentStateBytes(architecture, recurrentLayers);
  if (bytes === null || bytes === 0) return [];

  // The threshold is on the share of the total, not on the layer kind: a state
  // that is negligible on one architecture can be substantial on another.
  const share = base === null || base === 0 ? 1 : bytes / base;
  const counted = share >= NEGLIGIBLE_COMPONENT_FRACTION;

  return [
    {
      label: `fixed state held by ${recurrentLayers} recurrent ${recurrentLayers === 1 ? "layer" : "layers"}`,
      bytes,
      counted,
      note: counted
        ? `derived from the config, added to the total because it is ${(share * 100).toFixed(0)}% of it`
        : `derived from the config, not added: under ${(NEGLIGIBLE_COMPONENT_FRACTION * 100).toFixed(0)}% of the total and already inside the runtime allowance`,
    },
  ];
}
