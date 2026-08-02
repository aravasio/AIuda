import { readFileSync, existsSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { style } from "../../render/format.ts";

const TEAM_PATH = join(homedir(), ".config", "aiuda", "team.json");

export function teamCommand(): number {
  if (!existsSync(TEAM_PATH)) {
    process.stdout.write(`\n  No team file at ${TEAM_PATH}\n\n`);
    return 1;
  }

  let team: { models: Array<{ name: string; purpose: string }> };
  try {
    team = JSON.parse(readFileSync(TEAM_PATH, "utf8"));
  } catch {
    process.stdout.write(`\n  Could not parse ${TEAM_PATH}\n\n`);
    return 1;
  }

  const lines: string[] = [];
  lines.push("");
  lines.push(`  ${style.bold("AIuda Team — which model when")}`);
  lines.push("");

  for (const m of team.models ?? []) {
    lines.push(`    ${style.green(m.name.padEnd(22))}${m.purpose}`);
  }

  lines.push("");
  lines.push(`  ${style.bold("catalog commands")}`);
  lines.push(`    ${style.dim("discover       models that fit your machine")}`);
  lines.push(`    ${style.dim("team           this cheat sheet")}`);
  lines.push(`    ${style.dim("fit <url>      check if one model fits")}`);
  lines.push(`    ${style.dim("query <url>    explain a model")}`);

  lines.push("");
  lines.push(`  ${style.dim(`Edit: ${TEAM_PATH}`)}`);
  lines.push("");

  process.stdout.write(lines.join("\n"));
  return 0;
}