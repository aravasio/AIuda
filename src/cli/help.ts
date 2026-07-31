import { style } from "../render/format.ts";

/**
 * Help text follows the same plain-language rules as the output. A --help page
 * written in jargon would undo the point of the tool.
 */
export interface CommandHelp {
  name: string;
  /** One line, shown in the command list. */
  summary: string;
  usage: string;
  details: string[];
  flags: Array<{ flag: string; description: string }>;
}

export const COMMANDS: CommandHelp[] = [
  {
    name: "fit",
    summary: "Work out whether this machine can run a model, and show the arithmetic",
    usage: "catalog fit <url>",
    details: [
      "Takes a Hugging Face model page and reports how much memory the model needs,",
      "broken into the weights, the KV cache and an allowance for the runtime, then",
      "says whether that fits here. No language model is involved: every number comes",
      "from the repo's own config.json and file listing.",
      "",
      "Memory is exact arithmetic. Speed is not, so this command does not guess at",
      "tokens per second.",
    ],
    flags: [
      { flag: "--context <n>", description: "Assume n tokens of context. Longer context needs more memory." },
      { flag: "--technical", description: "Also show the architecture fields the numbers were read from." },
      { flag: "--json", description: "Print the result as JSON instead of text." },
      { flag: "--refresh", description: "Fetch the repo again instead of using the saved copy." },
    ],
  },
  {
    name: "specs",
    summary: "Show what the tool detected about this machine",
    usage: "catalog specs",
    details: [
      "Reports the processor, memory, graphics card and installed model runtimes that",
      "the fit calculations are measured against. Anything it could not read is listed",
      "as unknown, with the path of the file where you can fill it in by hand.",
    ],
    flags: [{ flag: "--json", description: "Print the result as JSON instead of text." }],
  },
  {
    name: "cache",
    summary: "List or delete the saved results",
    usage: "catalog cache <list|clear>",
    details: [
      "Results are saved per revision of a repo, so a model card edited by its authors",
      "is picked up as new work automatically and nothing goes stale.",
    ],
    flags: [{ flag: "--json", description: "Print the listing as JSON instead of text." }],
  },
];

export function renderMainHelp(): string {
  const lines: string[] = [];
  lines.push("");
  lines.push(`  ${style.bold("catalog")} — paste a Hugging Face model link, find out what it does`);
  lines.push(`             and whether this machine can run it.`);
  lines.push("");
  lines.push(`  ${style.dim("Usage")}  catalog <command> [options]`);
  lines.push("");
  lines.push(`  ${style.dim("Commands")}`);
  const width = Math.max(...COMMANDS.map((c) => c.name.length)) + 3;
  for (const command of COMMANDS) {
    lines.push(`    ${command.name.padEnd(width)}${command.summary}`);
  }
  lines.push("");
  lines.push(`  ${style.dim("Options that work everywhere")}`);
  lines.push(`    --json         Print machine-readable output.`);
  lines.push(`    --technical    Show the underlying detail as well as the summary.`);
  lines.push(`    --context <n>  Assume n tokens of context when working out memory.`);
  lines.push(`    --no-color     Plain text with no colour codes.`);
  lines.push(`    --refresh      Ignore anything saved and fetch the repo again.`);
  lines.push(`    --help         Show this, or the help for one command.`);
  lines.push("");
  lines.push(`  ${style.dim("Example")}  catalog fit https://huggingface.co/Qwen/QwQ-32B`);
  lines.push("");
  return lines.join("\n");
}

export function renderCommandHelp(command: CommandHelp): string {
  const lines: string[] = [];
  lines.push("");
  lines.push(`  ${style.bold(`catalog ${command.name}`)} — ${command.summary.toLowerCase()}`);
  lines.push("");
  lines.push(`  ${style.dim("Usage")}  ${command.usage}`);
  lines.push("");
  for (const line of command.details) {
    lines.push(line === "" ? "" : `  ${line}`);
  }
  lines.push("");
  if (command.flags.length > 0) {
    lines.push(`  ${style.dim("Options")}`);
    const width = Math.max(...command.flags.map((f) => f.flag.length)) + 3;
    for (const flag of command.flags) {
      lines.push(`    ${flag.flag.padEnd(width)}${flag.description}`);
    }
    lines.push("");
  }
  return lines.join("\n");
}

export function findCommandHelp(name: string): CommandHelp | null {
  return COMMANDS.find((c) => c.name === name) ?? null;
}
