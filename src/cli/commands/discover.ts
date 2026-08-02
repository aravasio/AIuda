import { loadConfig } from "../../config.ts";
import { REGISTRY, runnableCandidates } from "../../discover/registry.ts";
import { rankEntries, tierOf, bestBenchmark } from "../../discover/ranker.ts";
import { fullAnalysis, type FullResult } from "../../discover/full.ts";
import { style } from "../../render/format.ts";
import { humanBytes } from "../../fit/verdict.ts";
import { detectMachine } from "../../specs/detect.ts";
import type { MachineSpecs } from "../../specs/detect.ts";
import { HfClient } from "../../hf/client.ts";
import { resolveRepo } from "../../pipeline/resolve.ts";
import { analyze } from "../../pipeline/analyze.ts";
import { createProgress } from "../../render/progress.ts";
import type { GlobalFlags } from "../args.ts";
import { UsageError } from "../../errors.ts";

const HF_TRENDING_URL =
  "https://huggingface.co/api/models?pipeline_tag=text-generation&sort=trending&direction=-1&limit=30";

export async function discoverCommand(positionals: string[], flags: GlobalFlags): Promise<number> {
  const config = loadConfig();
  const specs = detectMachine();
  const contextTokens = flags.context ?? config.defaultContextTokens;

  if (positionals[0] === "scan") {
    return scanCommand(flags, specs, contextTokens);
  }

  return recommendCommand(flags, specs, contextTokens);
}

// ── quick (pre-computed) / full (HF pipeline) recommendation ───────

async function recommendCommand(
  flags: GlobalFlags,
  specs: MachineSpecs,
  contextTokens: number,
): Promise<number> {
  if (flags.technical) {
    return fullRecommendCommand(flags, specs, contextTokens);
  }
  return quickRecommendCommand(flags, specs, contextTokens);
}

/** Uses pre-computed sizes. No network needed. */
function quickRecommendCommand(
  flags: GlobalFlags,
  specs: MachineSpecs,
  contextTokens: number,
): number {
  const ram = specs.ramTotalBytes ?? 0;
  const candidates = runnableCandidates();
  const ranked = rankEntries(candidates, specs, contextTokens);
  const fitting = ranked.filter((e) => e.fits);

  const lines: string[] = [];
  lines.push("");
  lines.push(`  ${style.bold("catalog discover")} — what runs on this machine`);
  lines.push("");
  lines.push(`  ${style.dim(`Checked ${candidates.length} models against ${humanBytes(ram)} of RAM at ${contextTokens.toLocaleString("en-US")} tokens of context.`)}`);
  lines.push("");

  if (fitting.length === 0) {
    lines.push(`  ${style.red("Nothing here runs on this machine.")}`);
    lines.push("");
    process.stdout.write(lines.join("\n"));
    return 0;
  }

  lines.push(`  ${style.green(`${fitting.length} model${fitting.length === 1 ? "" : "s"} fit.`)}`);
  lines.push("");

  const light = fitting.filter((e) => tierOf(e.sizeGb) === "light");
  const medium = fitting.filter((e) => tierOf(e.sizeGb) === "medium");
  const heavy = fitting.filter((e) => tierOf(e.sizeGb) === "heavy");
  const daily = medium[0] ?? light[0] ?? null;
  const heavyLifter = heavy.find((e) => e.name !== daily?.name) ?? null;
  const contextSpecialist = fitting.find((e) => e.name.includes("Coder:30b")) ?? null;

  lines.push(`  ${style.bold("Your team")}`);
  lines.push("");
  if (daily !== null) {
    lines.push(`    ${style.green(daily.name.padEnd(22))}${style.bold(humanBytes(daily.fitBytes).padStart(8))}  ${daily.note}`);
  }
  if (heavyLifter !== null) {
    lines.push(`    ${style.yellow(heavyLifter.name.padEnd(22))}${style.bold(humanBytes(heavyLifter.fitBytes).padStart(8))}  ${heavyLifter.note}`);
  }
  if (contextSpecialist !== null && contextSpecialist.name !== daily?.name && contextSpecialist.name !== heavyLifter?.name) {
    lines.push(`    ${style.cyan(contextSpecialist.name.padEnd(22))}${style.bold(humanBytes(contextSpecialist.fitBytes).padStart(8))}  ${contextSpecialist.note}`);
  }

  lines.push("");
  lines.push(`  ${style.dim("All models that fit, by size group")}`);
  lines.push("");
  for (const [label, group] of [["Light  (< 20 GB)", light], ["Medium (20-35 GB)", medium], ["Heavy  (35-60 GB)", heavy]] as const) {
    if (group.length === 0) continue;
    lines.push(`  ${style.dim(label)}`);
    for (const entry of group) {
      const color = entry.verdictKind === "runs-comfortably" ? style.green : style.yellow;
      const tag = entry.architecture === "moe" ? "MoE" : "dense";
      lines.push(`    ${color(entry.name.padEnd(22))}${humanBytes(entry.fitBytes).padStart(8)}  ${style.dim(`${tag} · ${entry.context}`)}`);
    }
    lines.push("");
  }

  const notFitting = ranked.filter((e) => !e.fits);
  if (notFitting.length > 0) {
    lines.push(`  ${style.dim("Notable models that don't fit")}`);
    for (const entry of notFitting.slice(0, 5)) {
      lines.push(`    ${style.red(entry.name.padEnd(22))}${humanBytes(entry.fitBytes).padStart(8)}  ${style.dim(`needs ${entry.sizeGb} GB weights alone`)}`);
    }
    lines.push("");
  }

  lines.push(`  ${style.dim("To download:")}  ollama pull <name>`);
  lines.push(`  ${style.dim("Exact numbers:")}  catalog discover --technical`);
  lines.push("");

  process.stdout.write(lines.join("\n"));
  return 0;
}

/** Runs the real HF pipeline for every candidate. Network-heavy. */
async function fullRecommendCommand(
  flags: GlobalFlags,
  specs: MachineSpecs,
  contextTokens: number,
): Promise<number> {
  const candidates = runnableCandidates();
  const results = await fullAnalysis(candidates, specs, contextTokens, flags.json);
  const fitting = results.filter((r) => r.fit.verdict.kind !== "will-not-run");
  fitting.sort((a, b) => bestBenchmark(b) - bestBenchmark(a));

  const medium = fitting.filter((e) => tierOf(e.sizeGb) === "medium");
  const heavy = fitting.filter((e) => tierOf(e.sizeGb) === "heavy");
  const light = fitting.filter((e) => tierOf(e.sizeGb) === "light");
  const daily = medium[0] ?? light[0] ?? null;
  const heavyLifter = heavy.find((e) => e.name !== daily?.name) ?? null;

  const lines: string[] = [];
  lines.push("");
  lines.push(`  ${style.bold("catalog discover --technical")}`);
  lines.push("");
  lines.push(`  ${style.green(`${fitting.length}/${candidates.length} model${fitting.length === 1 ? "" : "s"} fit.`)}`);
  lines.push("");

  lines.push(`  ${style.bold("Your team")}`);
  lines.push("");
  if (daily !== null) {
    lines.push(`    ${style.green(daily.name.padEnd(22))}${style.bold(humanBytes(daily.fit.breakdown.totalBytes ?? 0).padStart(8))}  ${style.dim(daily.fit.verdict.summary)}`);
  }
  if (heavyLifter !== null) {
    lines.push(`    ${style.yellow(heavyLifter.name.padEnd(22))}${style.bold(humanBytes(heavyLifter.fit.breakdown.totalBytes ?? 0).padStart(8))}  ${style.dim(heavyLifter.fit.verdict.summary)}`);
  }

  lines.push("");
  lines.push(`  ${style.dim("All checked models")}`);
  for (const entry of results) {
    const total = entry.fit.breakdown.totalBytes ?? 0;
    const color = entry.fit.verdict.kind === "will-not-run" ? style.red : style.green;
    const icon = entry.fit.verdict.kind === "will-not-run" ? "✗" : "✓";
    lines.push(`    ${color(`${icon} ${entry.name}`.padEnd(28))}${humanBytes(total).padStart(8)}  ${style.dim(entry.fit.verdict.summary)}`);
  }

  lines.push("");
  lines.push(`  ${style.dim("To download:")}  ollama pull <name>`);
  lines.push("");

  process.stdout.write(lines.join("\n"));
  return 0;
}

// ── scan: fetch trending models from HF ────────────────────────────

async function scanCommand(
  flags: GlobalFlags,
  specs: MachineSpecs,
  contextTokens: number,
): Promise<number> {
  const lines: string[] = [];
  lines.push("");
  lines.push(`  ${style.bold("catalog discover scan")} — scanning Hugging Face trending...`);
  lines.push("");

  let trendingIds: string[];
  try {
    trendingIds = await fetchTrendingModels();
  } catch (error) {
    throw new UsageError(
      `Could not fetch trending models: ${error instanceof Error ? error.message : String(error)}`,
      "Check your network connection.",
    );
  }

  if (trendingIds.length === 0) {
    lines.push(`  ${style.yellow("No trending models found.")}`);
    lines.push("");

    process.stdout.write(lines.join("\n"));
    return 0;
  }

  lines.push(`  ${style.dim(`Found ${trendingIds.length} trending text-generation models. Running fit on each...`)}`);
  lines.push("");

  const config = loadConfig();
  const client = new HfClient();
  const progress = createProgress({ json: flags.json });
  const fitting: Array<{ name: string; size: number; verdict: string }> = [];

  for (let i = 0; i < trendingIds.length; i++) {
    const id = trendingIds[i]!;
    const shortName = id.split("/")[1] ?? id;
    const label = `[${i + 1}/${trendingIds.length}] ${shortName}`;

    try {
      progress.start(label);
      const resolved = await resolveRepo(id, client);
      const analysis = analyze(resolved.primary, resolved.base, specs, {
        contextTokens,
        runtimeOverheadBytes: config.runtimeOverheadBytes,
      });
      const bestFit = analysis.fits[0];

      if (bestFit && bestFit.verdict.kind !== "will-not-run") {
        fitting.push({
          name: shortName,
          size: bestFit.breakdown.totalBytes ?? 0,
          verdict: bestFit.verdict.summary,
        });
        progress.done(style.green("✓"));
      } else {
        progress.done(style.dim("✗"));
      }
    } catch {
      progress.done(style.dim("error"));
    }
  }

  progress.stop();
  lines.push("");

  if (fitting.length === 0) {
    lines.push(`  ${style.yellow("No trending models fit your machine.")}`);
  } else {
    lines.push(`  ${style.bold("Trending models that fit")}`);
    lines.push("");
    for (const fit of fitting) {
      lines.push(`    ${style.green(fit.name.padEnd(40))}${humanBytes(fit.size).padStart(8)}  ${style.dim(fit.verdict)}`);
    }
    lines.push("");
    lines.push(`  ${style.dim("To add one to the permanent registry:")}  edit src/discover/registry.ts`);
  }
  lines.push("");

  process.stdout.write(lines.join("\n"));
  return 0;
}

async function fetchTrendingModels(): Promise<string[]> {
  const url = HF_TRENDING_URL;
  const response = await fetch(url, {
    headers: { "User-Agent": "catalog/0.1.0" },
    signal: AbortSignal.timeout(10000),
  });

  if (!response.ok) {
    throw new Error(`HF API returned ${response.status}`);
  }

  const models = (await response.json()) as Array<{ id: string }>;
  return models.filter((m) => m.id && m.id.includes("/")).map((m) => m.id);
}