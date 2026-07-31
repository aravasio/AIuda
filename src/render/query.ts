import { humanBytes } from "../fit/verdict.ts";
import { BaseModelWarning, type LabelledBenchmark } from "../llm/schema.ts";
import type { QueryResult } from "../pipeline/query.ts";
import type { MachineSpecs } from "../specs/detect.ts";
import { renderMachine } from "./fit.ts";
import { formatParams, labelled, style, wrapText } from "./format.ts";

const WRAP = 72;
const INDENT = "  ";

/**
 * The default answer is short on purpose. Everything a reader needs to decide
 * whether to download, and nothing else. A benchmark table is never printed
 * here; `--technical` is where that lives.
 */
export function renderQuery(
  result: QueryResult,
  specs: MachineSpecs,
  options: { technical: boolean },
): string {
  const { analysis, prose } = result;
  const lines: string[] = [];

  lines.push("");
  lines.push(header(analysis.repoId, analysis.author, analysis.license));
  lines.push("");
  lines.push(wrapText(prose.one_liner, WRAP - 2, `${INDENT}  `));
  lines.push("");

  if (prose.category.includes("base")) {
    lines.push(wrapText(style.yellow(BaseModelWarning), WRAP, `${INDENT}  `));
    lines.push("");
  }

  lines.push(labelled("Use it for", prose.use_for.join(", ")));
  lines.push(labelled("Not for", prose.not_for));
  if (prose.pairs_with !== null && prose.pairs_with.length > 0) {
    lines.push(labelled("Pairs with", prose.pairs_with.join(", ")));
  }
  lines.push("");

  lines.push(labelled("Run", `${prose.deployment.where}, ${prose.deployment.reason}`));
  lines.push(labelled("Your machine", verdictLine(result)));
  lines.push("");

  lines.push(
    labelled("Benchmarks", benchmarkLine(result.benchmarks, result.benchmarksUnreadable)),
  );

  if (prose.supersedes_note !== null) {
    lines.push("");
    lines.push(wrapText(style.yellow(prose.supersedes_note), WRAP, INDENT));
  }

  lines.push("");
  for (const note of analysis.notes) {
    lines.push(wrapText(note, WRAP, INDENT));
    lines.push("");
  }

  if (options.technical) {
    lines.push(renderTechnical(result, specs));
  }

  return lines.join("\n");
}

function header(repoId: string, author: string | null, license: string | null): string {
  const name = repoId.split("/")[1] ?? repoId;
  const right = [author, license ?? "license not stated"]
    .filter((v): v is string => v !== null)
    .join(", ");
  const pad = Math.max(2, WRAP - name.length - right.length);
  return `${style.bold(name)}${" ".repeat(pad)}${style.dim(right)}`;
}

/** One line: the verdict, the size it is based on, and where that size came from. */
function verdictLine(result: QueryResult): string {
  const fit = result.analysis.fits[0];
  if (fit === undefined) return "cannot say";

  const verdict = fit.verdict.summary.replace(/\.$/, "").toLowerCase();
  const coloured =
    fit.verdict.kind === "runs-comfortably"
      ? style.green(verdict)
      : fit.verdict.kind === "will-not-run"
        ? style.red(verdict)
        : style.yellow(verdict);

  const total = fit.breakdown.totalBytes;
  if (total === null) return coloured;

  // Only a real precision reads sensibly here. "70 GB at safetensors" names a
  // file format where the reader expects a number of bits.
  const label = fit.variant?.quant ?? fit.breakdown.weights?.quant ?? null;
  const at = label === null ? "" : ` at ${label}`;
  return `${coloured}. ${humanBytes(total)}${at}, at ${result.analysis.contextTokens.toLocaleString("en-US")} tokens of context`;
}

/**
 * Counts rather than scores. Which numbers exist and where they came from is
 * the useful part; the numbers themselves belong behind --technical, where the
 * caveat about them sits too.
 */
function benchmarkLine(benchmarks: LabelledBenchmark[], unreadable: boolean): string {
  if (benchmarks.length === 0) {
    return unreadable
      ? "this card has some, but they did not fit and were not read"
      : "none reported by the vendor";
  }

  const fromIndex = benchmarks.filter((b) => b.source === "model-index").length;
  const fromCard = benchmarks.filter((b) => b.source === "card-table").length;
  const parts: string[] = [];
  if (fromIndex > 0) parts.push(`${fromIndex} in the repo's metadata`);
  if (fromCard > 0) parts.push(`${fromCard} in the card's own tables`);
  return `${parts.join(", ")}, all self-reported. See --technical`;
}

function renderTechnical(result: QueryResult, specs: MachineSpecs): string {
  const lines: string[] = [];
  const { analysis } = result;

  lines.push(style.bold("  Detail"));
  lines.push("");
  lines.push(labelled("Revision", analysis.sha));
  lines.push(labelled("Repo type", analysis.classification.kind));
  lines.push(labelled("Size", sizeDetail(analysis)));
  if (analysis.architecture !== null) {
    const a = analysis.architecture;
    lines.push(
      labelled(
        "Shape",
        [
          a.modelType,
          a.layers === null ? null : `${a.layers.value} layers`,
          a.kvHeads === null ? null : `${a.kvHeads.value} key/value heads`,
          a.headDim === null ? null : `head dimension ${a.headDim.value}`,
        ]
          .filter((v) => v !== null)
          .join(", "),
      ),
    );
  }
  if (analysis.downloads !== null) {
    lines.push(labelled("Downloads", analysis.downloads.toLocaleString("en-US")));
  }
  lines.push("");

  lines.push(...renderBenchmarkTable(result.benchmarks));

  if (result.droppedClaims.length > 0) {
    lines.push(style.bold("  Claims dropped"));
    lines.push("");
    for (const claim of result.droppedClaims) {
      lines.push(wrapText(`${claim.field}: ${claim.reason}.`, WRAP - 2, `${INDENT}  `));
    }
    lines.push("");
    lines.push(
      wrapText(
        "The repository's own data always wins over the summary. Dropped claims are removed, never averaged in.",
        WRAP,
        INDENT,
      ),
    );
    lines.push("");
  }

  lines.push(style.bold("  How this was read"));
  lines.push("");
  lines.push(
    labelled(
      "Card",
      `trimmed to ${result.trimming.prose.tokens.toLocaleString("en-US")} of ${result.trimming.prose.budgetTokens.toLocaleString("en-US")} tokens available`,
    ),
  );
  if (result.trimming.prose.dropped.length > 0) {
    lines.push(labelled("Left out", result.trimming.prose.dropped.join(", ")));
  }
  lines.push(
    labelled(
      "Window",
      `${result.contextPlan.windowTokens.toLocaleString("en-US")} tokens, ${result.contextPlan.outputTokens.toLocaleString("en-US")} reserved for the reply`,
    ),
  );
  lines.push("");
  lines.push(renderMachine(specs));
  lines.push("");
  return lines.join("\n");
}

function sizeDetail(analysis: QueryResult["analysis"]): string {
  if (analysis.params.total === null) return "not published";
  const base = `${formatParams(analysis.params.total)} parameters (${analysis.params.totalSource})`;
  if (analysis.isMoe && analysis.params.active !== null && analysis.params.active < analysis.params.total) {
    return `${base}, ${formatParams(analysis.params.active)} active per token`;
  }
  return base;
}

function renderBenchmarkTable(benchmarks: LabelledBenchmark[]): string[] {
  if (benchmarks.length === 0) {
    return [style.bold("  Benchmarks"), "", `${INDENT}  None reported.`, ""];
  }

  const lines: string[] = [style.bold("  Benchmarks"), ""];
  const nameWidth = Math.min(38, Math.max(...benchmarks.map((b) => b.benchmark.length)) + 2);

  for (const entry of benchmarks) {
    const name = truncate(entry.benchmark, nameWidth).padEnd(nameWidth);
    const model = entry.model === null ? "" : style.dim(`  ${truncate(entry.model, 28)}`);
    lines.push(`${INDENT}  ${name}${entry.score.padStart(8)}${model}  ${sourceLabel(entry.source)}`);
  }

  lines.push("");
  lines.push(
    wrapText(
      "These are the vendor's own numbers, self-reported and self-selected. Numbers from different sources use different setups and are not strictly comparable.",
      WRAP,
      INDENT,
    ),
  );
  lines.push("");
  return lines;
}

function sourceLabel(source: LabelledBenchmark["source"]): string {
  switch (source) {
    case "model-index":
      return style.dim("repo metadata");
    case "card-table":
      return style.dim("card table");
    case "local":
      return style.dim("measured here");
  }
}

function truncate(text: string, width: number): string {
  return text.length <= width ? text : `${text.slice(0, width - 1)}…`;
}
