import { defaultCacheDir, RevisionCache } from "../../cache/index.ts";
import { UsageError } from "../../errors.ts";
import { humanBytes } from "../../fit/verdict.ts";
import { style } from "../../render/format.ts";
import { nearestMatch, type GlobalFlags } from "../args.ts";

const ACTIONS = ["list", "clear"];

export function cacheCommand(positionals: string[], flags: GlobalFlags): number {
  const action = positionals[0];
  if (action === undefined) {
    throw new UsageError("cache needs to know what to do.", "catalog cache list");
  }
  if (!ACTIONS.includes(action)) {
    const suggestion = nearestMatch(action, ACTIONS);
    throw new UsageError(
      `cache has no "${action}".${suggestion === null ? "" : ` Did you mean ${suggestion}?`}`,
      "catalog cache list",
    );
  }

  const cache = new RevisionCache();

  if (action === "clear") {
    const removed = cache.clear();
    if (flags.json) {
      process.stdout.write(`${JSON.stringify({ removed })}\n`);
    } else {
      process.stdout.write(
        `\n  Deleted ${removed} saved ${removed === 1 ? "result" : "results"}.\n\n`,
      );
    }
    return 0;
  }

  const entries = cache.list();
  if (flags.json) {
    process.stdout.write(`${JSON.stringify(entries, null, 2)}\n`);
    return 0;
  }

  if (entries.length === 0) {
    process.stdout.write(`\n  Nothing saved yet. Results are stored in ${defaultCacheDir()}.\n\n`);
    return 0;
  }

  const lines: string[] = ["", `  ${style.bold("Saved results")}`, ""];
  const width = Math.max(...entries.map((e) => e.repoId.length)) + 3;
  for (const entry of entries) {
    lines.push(
      `    ${entry.repoId.padEnd(width)}${style.dim(`${entry.sha.slice(0, 10)}  ${entry.storedAt.slice(0, 10)}  ${humanBytes(entry.bytes)}`)}`,
    );
  }
  lines.push("");
  lines.push(style.dim(`  Each line is one revision of a repo. ${defaultCacheDir()}`));
  lines.push("");
  process.stdout.write(lines.join("\n"));
  return 0;
}
