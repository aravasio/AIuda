import { humanBytes } from "../fit/verdict.ts";
import type { Analysis, VariantFit } from "../pipeline/analyze.ts";
import type { MachineSpecs } from "../specs/detect.ts";
import { fastMemoryBytes } from "../specs/detect.ts";
import { formatParams, labelled, style, wrapText } from "./format.ts";

const WRAP = 72;

/**
 * Prints the memory arithmetic with every component visible. A single opaque
 * "needs 22 GB" cannot be checked by the reader; weights plus KV cache plus
 * overhead can.
 */
export function renderFit(analysis: Analysis, specs: MachineSpecs): string {
  const lines: string[] = [];
  lines.push("");
  lines.push(header(analysis));
  lines.push("");

  lines.push(labelled("Size", sizeLine(analysis)));
  if (analysis.architecture !== null) {
    lines.push(labelled("Shape", shapeLine(analysis)));
  }
  lines.push(
    labelled("Context", `${analysis.contextTokens.toLocaleString("en-US")} tokens assumed for this calculation, set it with --context`),
  );
  lines.push("");

  // A quantisation catalogue can hold two dozen variants. Listing every
  // breakdown in full would bury the answer, so they get one line each and the
  // one worth downloading gets the full arithmetic.
  if (analysis.fits.length > MAX_FULL_BREAKDOWNS) {
    lines.push(...renderVariantTable(analysis.fits));
    lines.push("");
    const pick = recommendedFit(analysis.fits);
    if (pick !== null) {
      lines.push(`  ${style.dim("The largest one that runs here")}`);
      lines.push("");
      lines.push(...renderVariant(pick, true));
      lines.push("");
    }
  } else {
    const multiple = analysis.fits.length > 1;
    for (const fit of analysis.fits) {
      lines.push(...renderVariant(fit, multiple));
      lines.push("");
    }
  }

  for (const note of analysis.notes) {
    lines.push(wrapText(note, WRAP, "  "));
    lines.push("");
  }

  if (analysis.unavailable.length > 0) {
    lines.push(
      wrapText(`Not published by this repo: ${analysis.unavailable.join("; ")}.`, WRAP, "  "),
    );
    lines.push("");
  }

  lines.push(renderMachine(specs));
  lines.push("");
  lines.push(
    wrapText(
      `Assumption: ${humanBytes(analysis.fits[0]?.breakdown.runtimeOverheadBytes ?? 0)} is allowed for the runtime itself. That is the least certain number here, and you can change it with runtimeOverheadBytes in the config file.`,
      WRAP,
      "  ",
    ),
  );
  lines.push("");
  lines.push(
    style.dim(
      "  Memory here is exact arithmetic. Speed is not, so nothing above guesses at it.",
    ),
  );
  lines.push("");
  return lines.join("\n");
}

function header(analysis: Analysis): string {
  const name = analysis.repoId.split("/")[1] ?? analysis.repoId;
  const right = [analysis.author, analysis.license ?? "license not stated"]
    .filter((v): v is string => v !== null)
    .join(", ");
  const left = style.bold(name);
  const pad = Math.max(2, WRAP - name.length - right.length);
  return `${left}${" ".repeat(pad)}${style.dim(right)}`;
}

function sizeLine(analysis: Analysis): string {
  const { params } = analysis;
  if (params.total === null) return "parameter count not published";
  const total = `${formatParams(params.total)} parameters`;
  const source = params.totalSource === null ? "" : ` (${params.totalSource})`;
  if (analysis.isMoe && params.active !== null && params.active < params.total) {
    return `${total}${source}, of which ${formatParams(params.active)} are active per token`;
  }
  return `${total}${source}`;
}

function shapeLine(analysis: Analysis): string {
  const a = analysis.architecture;
  if (a === null) return "unknown";
  const parts: string[] = [];
  if (a.layers !== null) parts.push(`${a.layers.value} layers`);
  if (a.kvHeads !== null) parts.push(`${a.kvHeads.value} key/value heads`);
  if (a.headDim !== null) parts.push(`head dimension ${a.headDim.value}`);
  if (a.maxContext !== null)
    parts.push(`up to ${a.maxContext.value.toLocaleString("en-US")} tokens of context`);
  const from =
    analysis.architectureFrom !== null && analysis.architectureFrom !== analysis.repoId
      ? ` (from ${analysis.architectureFrom})`
      : "";
  return parts.length === 0 ? "unknown" : `${parts.join(", ")}${from}`;
}

/** Above this many downloadable variants, the per-variant detail is summarised. */
const MAX_FULL_BREAKDOWNS = 3;

/** Best to worst, so "the largest that runs" is the first acceptable one. */
const VERDICT_RANK: Record<string, number> = {
  "runs-comfortably": 0,
  "runs-tight": 1,
  "split-possible": 2,
  "runs-slowly": 3,
  unknown: 4,
  "will-not-run": 5,
};

/**
 * The biggest variant that still runs, since size tracks quality within a
 * repo. Nothing is recommended when nothing runs.
 */
function recommendedFit(fits: VariantFit[]): VariantFit | null {
  const runnable = fits.filter((fit) => fit.verdict.kind !== "will-not-run" && fit.verdict.kind !== "unknown");
  if (runnable.length === 0) return null;
  return runnable.reduce((best, fit) =>
    (fit.breakdown.totalBytes ?? 0) > (best.breakdown.totalBytes ?? 0) ? fit : best,
  );
}

function renderVariantTable(fits: VariantFit[]): string[] {
  const rows = [...fits].sort(
    (a, b) => (b.breakdown.totalBytes ?? 0) - (a.breakdown.totalBytes ?? 0),
  );
  const labelWidth = Math.max(...rows.map((f) => (f.variant?.label ?? "weights").length)) + 2;

  const lines: string[] = [`  ${style.dim(`${fits.length} versions to choose from, largest first`)}`, ""];
  for (const fit of rows) {
    const label = (fit.variant?.label ?? "weights").padEnd(labelWidth);
    const total =
      fit.breakdown.totalBytes === null ? "unknown" : humanBytes(fit.breakdown.totalBytes);
    const verdict = verdictStyle(fit).padEnd(0);
    lines.push(
      `    ${label}${total.padStart(8)}   ${verdict}${fit.qualityBand === null ? "" : style.dim(`  ${fit.qualityBand}`)}`,
    );
  }
  lines.push("");
  lines.push(
    style.dim(`    Sizes include the KV cache and the runtime allowance, not just the download.`),
  );
  return lines;
}

function renderVariant(fit: VariantFit, showLabel: boolean): string[] {
  const lines: string[] = [];
  const label = fit.variant?.label ?? fit.breakdown.weights?.quant ?? "weights";

  if (showLabel) {
    const band = fit.qualityBand === null ? "" : style.dim(`  ${fit.qualityBand}`);
    lines.push(`  ${style.bold(label)}${band}`);
  }

  const indent = showLabel ? "    " : "  ";
  const { weights, kv } = fit.breakdown;

  lines.push(
    `${indent}Weights        ${weights === null ? "unknown" : `${humanBytes(weights.bytes)}  ${style.dim(`from the ${weights.source}`)}`}`,
  );

  if (kv === null) {
    lines.push(`${indent}KV cache       unknown`);
  } else {
    const scope =
      kv.cachingLayers === kv.totalLayers
        ? `${kv.totalLayers} layers`
        : `${kv.cachingLayers} of ${kv.totalLayers} layers`;
    lines.push(
      `${indent}KV cache       ${humanBytes(kv.bytes)}  ${style.dim(`${scope} at ${fit.breakdown.contextTokens.toLocaleString("en-US")} tokens`)}`,
    );
  }

  lines.push(
    `${indent}Runtime        ${humanBytes(fit.breakdown.runtimeOverheadBytes)}  ${style.dim("assumed allowance, not measured")}`,
  );

  for (const component of fit.breakdown.sideComponents) {
    if (!component.counted) continue;
    lines.push(
      `${indent}Layer state    ${humanBytes(component.bytes)}  ${style.dim(component.note)}`,
    );
  }

  lines.push(
    `${indent}Total          ${fit.breakdown.totalBytes === null ? "cannot be computed" : style.bold(humanBytes(fit.breakdown.totalBytes))}`,
  );

  for (const component of fit.breakdown.sideComponents) {
    if (component.counted) continue;
    lines.push(
      style.dim(
        `${indent}Not counted: ${component.label}, about ${humanBytes(component.bytes)} — ${component.note}.`,
      ),
    );
  }

  lines.push("");
  lines.push(`${indent}${verdictStyle(fit)}`);
  lines.push(wrapText(fit.verdict.detail, WRAP - indent.length, indent));
  return lines;
}

function verdictStyle(fit: VariantFit): string {
  const text = fit.verdict.summary;
  switch (fit.verdict.kind) {
    case "runs-comfortably":
      return style.green(text);
    case "runs-tight":
    case "split-possible":
    case "runs-slowly":
      return style.yellow(text);
    case "will-not-run":
      return style.red(text);
    default:
      return style.dim(text);
  }
}

export function renderMachine(specs: MachineSpecs): string {
  const fast = fastMemoryBytes(specs);
  const parts: string[] = [];
  if (specs.ramTotalBytes !== null) parts.push(`${humanBytes(specs.ramTotalBytes)} of memory`);
  if (fast !== null) {
    parts.push(
      specs.memoryKind === "unified"
        ? `${humanBytes(fast)} of it usable by the GPU`
        : `${humanBytes(fast)} of VRAM`,
    );
  } else if (specs.memoryKind === "cpu-only") {
    parts.push("no GPU detected");
  }
  return style.dim(`  Measured against this machine: ${parts.join(", ")}.`);
}
