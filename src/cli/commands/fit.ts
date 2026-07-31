import { RevisionCache } from "../../cache/index.ts";
import { loadConfig } from "../../config.ts";
import { UsageError } from "../../errors.ts";
import { HfClient } from "../../hf/client.ts";
import { parseModelRef } from "../../hf/url.ts";
import { analyze, type Analysis } from "../../pipeline/analyze.ts";
import { resolveRepo } from "../../pipeline/resolve.ts";
import { renderFit } from "../../render/fit.ts";
import { detectMachine } from "../../specs/detect.ts";
import type { GlobalFlags } from "../args.ts";

/** What the cache stores for a revision after the deterministic pass. */
interface CachedFit {
  analysis: Analysis;
}

export async function fitCommand(positionals: string[], flags: GlobalFlags): Promise<number> {
  const target = positionals[0];
  if (target === undefined) {
    throw new UsageError(
      "fit needs a model to look at.",
      "catalog fit https://huggingface.co/Qwen/QwQ-32B",
    );
  }

  const config = loadConfig();
  const ref = parseModelRef(target);
  const client = new HfClient();
  const specs = detectMachine();
  const contextTokens = flags.context ?? config.defaultContextTokens;

  const resolved = await resolveRepo(ref, client);
  const analysis = analyze(resolved.primary, resolved.base, specs, {
    contextTokens,
    runtimeOverheadBytes: config.runtimeOverheadBytes,
  });

  // The fetched snapshot is keyed by SHA so a later `query` on the same revision
  // reuses it. The fit numbers themselves depend on --context and on this
  // machine, so they are recomputed rather than read back.
  const cache = new RevisionCache();
  cache.set<CachedFit>(analysis.sha, analysis.repoId, { analysis });

  if (resolved.baseError !== null) {
    analysis.notes.push(resolved.baseError);
  }

  if (flags.json) {
    process.stdout.write(`${JSON.stringify({ analysis, machine: specs }, null, 2)}\n`);
    return 0;
  }

  process.stdout.write(renderFit(analysis, specs));
  if (flags.technical) {
    process.stdout.write(renderTechnical(analysis));
  }
  return 0;
}

function renderTechnical(analysis: Analysis): string {
  const lines: string[] = [];
  lines.push("  Where each number came from");
  lines.push("");
  lines.push(`    revision           ${analysis.sha}`);
  lines.push(`    repo type          ${analysis.classification.kind}`);
  if (analysis.classification.baseModel !== null) {
    lines.push(`    base model         ${analysis.classification.baseModel}`);
  }
  const a = analysis.architecture;
  if (a !== null) {
    lines.push(`    model_type         ${a.modelType ?? "not stated"}`);
    lines.push(`    architectures      ${a.architectures.join(", ") || "not stated"}`);
    for (const [label, field] of [
      ["num_hidden_layers", a.layers],
      ["hidden_size", a.hiddenSize],
      ["num_attention_heads", a.attentionHeads],
      ["num_key_value_heads", a.kvHeads],
      ["head_dim", a.headDim],
      ["max_position_embeddings", a.maxContext],
    ] as const) {
      if (field !== null) {
        lines.push(`    ${label.padEnd(19)}${field.value}  (${field.source})`);
      }
    }
    lines.push(`    torch_dtype        ${a.torchDtype ?? "not stated"}`);
    if (a.moe !== null) {
      lines.push(`    experts            ${a.moe.experts ?? "not stated"}, ${a.moe.expertsPerToken ?? "?"} used per token`);
    }
  }
  lines.push(`    parameters         ${analysis.params.total ?? "unknown"}  (${analysis.params.totalSource ?? "no source"})`);
  for (const variant of analysis.weightVariants) {
    lines.push(`    ${variant.label.padEnd(19)}${variant.bytes} bytes across ${variant.files.length} files`);
  }
  lines.push("");
  return lines.join("\n");
}
