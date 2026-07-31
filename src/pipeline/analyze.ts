import { classify, type Classification } from "../classify/index.ts";
import type { ModelIndexResult, RepoSnapshot } from "../hf/types.ts";
import { parseArchitecture, type Architecture } from "../model/architecture.ts";
import { isMixtureOfExperts, parseSizeFromName, resolveParams, type ParamCount } from "../model/params.ts";
import { qualityBand, quantFromTorchDtype } from "../model/quant.ts";
import { collectWeightVariants, totalRepoBytes, type WeightVariant } from "../model/weights.ts";
import { computeBreakdown, computeWeightsBytes, DEFAULT_CONTEXT_TOKENS, type Breakdown } from "../fit/math.ts";
import { decideVerdict, type Verdict } from "../fit/verdict.ts";
import { formatParams } from "../render/format.ts";
import type { MachineSpecs } from "../specs/detect.ts";

/** Fit for one downloadable set of weights. GGUF repos produce several. */
export interface VariantFit {
  variant: WeightVariant | null;
  breakdown: Breakdown;
  verdict: Verdict;
  /** Plain-language quality band for the quantisation, never a percentage. */
  qualityBand: string | null;
}

/**
 * Branch A of the pipeline: everything that can be known without asking a
 * language model anything. Every number the tool prints originates here.
 */
export interface Analysis {
  repoId: string;
  /** Revision SHA. Also the cache key. */
  sha: string;
  author: string | null;
  license: string | null;
  downloads: number | null;
  likes: number | null;
  lastModified: string | null;
  pipelineTag: string | null;
  tags: string[];
  classification: Classification;
  /** Read from this repo's config.json, or from the base model's when this repo has none. */
  architecture: Architecture | null;
  architectureFrom: string | null;
  params: ParamCount;
  isMoe: boolean;
  weightVariants: WeightVariant[];
  repoTotalBytes: number;
  fits: VariantFit[];
  modelIndexBenchmarks: ModelIndexResult[];
  contextTokens: number;
  /** Things the user should know that came out of the deterministic pass. */
  notes: string[];
  /** Fields no source could supply. */
  unavailable: string[];
}

export interface AnalyzeOptions {
  contextTokens?: number;
  kvBytesPerElement?: number;
  runtimeOverheadBytes?: number;
}

export function analyze(
  snapshot: RepoSnapshot,
  /** The base model's snapshot when this repo is a quantisation or an adapter. */
  base: RepoSnapshot | null,
  specs: MachineSpecs,
  options: AnalyzeOptions = {},
): Analysis {
  const classification = classify(snapshot);
  const contextTokens = options.contextTokens ?? DEFAULT_CONTEXT_TOKENS;
  const notes: string[] = [];

  // A quantisation repo often ships no config.json. The base model's config
  // describes the same architecture, so the KV cache math stays exact.
  let architecture = parseArchitecture(snapshot.config);
  let architectureFrom: string | null = architecture !== null ? snapshot.info.repoId : null;
  if (architecture === null && base !== null) {
    architecture = parseArchitecture(base.config);
    if (architecture !== null) {
      architectureFrom = base.info.repoId;
      notes.push(`Architecture read from the base model, ${base.info.repoId}.`);
    }
  }

  const nameForHint = base?.info.repoId ?? snapshot.info.repoId;
  const shortName = nameForHint.split("/")[1] ?? nameForHint;
  const hint = parseSizeFromName(shortName);

  // Parameter counts must describe the model, not the quantisation. A GGUF repo
  // publishes no safetensors metadata, so the base repo's count is the right one.
  const paramsSource =
    snapshot.info.safetensorsTotalParams !== null || base === null ? snapshot.info : base.info;
  const params = resolveParams(paramsSource, architecture, shortName);
  const isMoe = isMixtureOfExperts(architecture, hint);

  const weightVariants = collectWeightVariants(snapshot.files, architecture?.torchDtype ?? null);
  const fallbackQuant =
    quantFromTorchDtype(architecture?.torchDtype ?? null) ??
    (classification.kind === "standard" ? "BF16" : null);

  const fitOptions = {
    contextTokens,
    isMoe,
    kvBytesPerElement: options.kvBytesPerElement,
    runtimeOverheadBytes: options.runtimeOverheadBytes,
  };

  const fits: VariantFit[] = [];
  if (weightVariants.length > 0) {
    for (const variant of weightVariants) {
      fits.push(
        buildFit(
          variant,
          variant.bytes,
          variant.quant ?? fallbackQuant,
          params,
          architecture,
          specs,
          fitOptions,
        ),
      );
    }
  } else {
    // No weight files to measure: fall back to the bytes-per-param estimate,
    // which the output will label as an estimate.
    fits.push(buildFit(null, null, fallbackQuant, params, architecture, specs, fitOptions));
  }

  if (isMoe && params.total !== null && params.active !== null && params.active < params.total) {
    notes.push(
      `Mixture of experts: it needs the memory of a ${billions(params.total)} model but runs at roughly the speed of a ${billions(params.active)} one. Memory follows the total, speed follows the ${billions(params.active)} that are active per token.`,
    );
  }

  // Mixed-attention architectures are the one place where "layers" in the KV
  // formula does not mean every layer, and the difference is large enough to
  // change the verdict, so it is stated rather than left implicit.
  const kv = fits[0]?.breakdown.kv ?? null;
  if (kv !== null && kv.cachingLayers < kv.totalLayers) {
    notes.push(
      `Only ${kv.cachingLayers} of this model's ${kv.totalLayers} layers keep a cache that grows with the context. The other ${kv.totalLayers - kv.cachingLayers} hold a fixed-size state instead, so the KV cache figure above counts ${kv.cachingLayers} layers, not ${kv.totalLayers}.`,
    );
  }

  // Asking for more context than the model supports produces a real number for
  // a configuration that cannot be run, so it is called out rather than printed
  // as though it were achievable.
  const maxContext = architecture?.maxContext?.value ?? null;
  if (maxContext !== null && contextTokens > maxContext) {
    notes.push(
      `You asked for ${contextTokens.toLocaleString("en-US")} tokens of context, but this model was trained for at most ${maxContext.toLocaleString("en-US")}. The memory figure above is what that would cost; going past the trained length usually needs a context-extension setting and costs quality.`,
    );
  }

  if (classification.flags.multimodal) {
    notes.push(
      "This repo handles more than text, so its architecture numbers come from the nested text_config rather than the top level.",
    );
  }
  if (classification.flags.thirdParty && classification.baseModel !== null) {
    notes.push(
      `Quantised by ${snapshot.info.author ?? "a third party"}, not by the authors of ${classification.baseModel}.`,
    );
  }
  if (classification.flags.imatrix) {
    notes.push("Built with an importance matrix, which usually gives a better result at the same size.");
  }
  if (classification.kind === "adapter") {
    notes.push(
      "This is an adapter, not a standalone model. It is loaded on top of its base model, so the memory it needs is the base model's plus a small amount.",
    );
  }

  return {
    repoId: snapshot.info.repoId,
    sha: snapshot.info.sha,
    author: snapshot.info.author,
    license: snapshot.info.license,
    downloads: snapshot.info.downloads,
    likes: snapshot.info.likes,
    lastModified: snapshot.info.lastModified,
    pipelineTag: snapshot.info.pipelineTag,
    tags: snapshot.info.tags,
    classification,
    architecture,
    architectureFrom,
    params,
    isMoe,
    weightVariants,
    repoTotalBytes: totalRepoBytes(snapshot.files),
    fits,
    modelIndexBenchmarks: snapshot.info.modelIndex,
    contextTokens,
    notes,
    unavailable: resolveUnavailable(classification.unavailable, architecture, params),
  };
}

/**
 * Classification lists what the repo itself does not publish. Anything the base
 * model supplied is no longer missing, and saying otherwise would send the user
 * looking for a number that is already on screen.
 */
function resolveUnavailable(
  fromClassification: string[],
  architecture: Architecture | null,
  params: ParamCount,
): string[] {
  const stillMissing = fromClassification.filter((field) => {
    if (architecture !== null && ARCHITECTURE_FIELDS.has(field)) {
      return architecture.missing.includes(field);
    }
    if (field === "exact parameter count") {
      return params.totalSource !== "safetensors metadata";
    }
    return true;
  });
  return [...new Set([...stillMissing, ...(architecture?.missing ?? [])])];
}

const ARCHITECTURE_FIELDS = new Set([
  "layer count",
  "key/value head count",
  "head dimension",
  "maximum context length",
]);

function buildFit(
  variant: WeightVariant | null,
  realBytes: number | null,
  quant: string | null,
  params: ParamCount,
  architecture: Architecture | null,
  specs: MachineSpecs,
  options: {
    contextTokens: number;
    isMoe: boolean;
    kvBytesPerElement?: number | undefined;
    runtimeOverheadBytes?: number | undefined;
  },
): VariantFit {
  const weights = computeWeightsBytes({ realBytes, paramsTotal: params.total, quant });
  const breakdown = computeBreakdown({
    weights,
    architecture,
    contextTokens: options.contextTokens,
    ...(options.kvBytesPerElement !== undefined
      ? { kvBytesPerElement: options.kvBytesPerElement }
      : {}),
    ...(options.runtimeOverheadBytes !== undefined
      ? { runtimeOverheadBytes: options.runtimeOverheadBytes }
      : {}),
  });
  // The band describes the weights actually in this variant. When a variant
  // carries no readable quantisation label, the answer is "unknown", never the
  // fallback used for the size estimate — that would claim a fidelity nobody
  // stated.
  const bandQuant = variant === null ? quant : variant.quant;

  return {
    variant,
    breakdown,
    verdict: decideVerdict(breakdown, specs, { isMoe: options.isMoe }),
    qualityBand: bandQuant === null ? null : qualityBand(bandQuant),
  };
}

function billions(params: number): string {
  return formatParams(params);
}
