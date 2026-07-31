#!/usr/bin/env node
import { CatalogError, ExitCode } from "../errors.ts";
import { colorSupported, setColorEnabled, style } from "../render/format.ts";
import { nearestMatch, parseArgs } from "./args.ts";
import { cacheCommand } from "./commands/cache.ts";
import { fitCommand } from "./commands/fit.ts";
import { specsCommand } from "./commands/specs.ts";
import { COMMANDS, findCommandHelp, renderCommandHelp, renderMainHelp } from "./help.ts";

const VERSION = "0.1.0";

export async function main(argv: string[]): Promise<number> {
  const parsed = parseArgs(argv);
  setColorEnabled(parsed.flags.color && colorSupported());

  if (parsed.flags.version) {
    process.stdout.write(`catalog ${VERSION}\n`);
    return ExitCode.Ok;
  }

  if (parsed.command === null) {
    process.stdout.write(renderMainHelp());
    return parsed.flags.help ? ExitCode.Ok : ExitCode.Usage;
  }

  if (parsed.flags.help) {
    const help = findCommandHelp(parsed.command);
    if (help === null) {
      process.stdout.write(renderMainHelp());
      return ExitCode.Usage;
    }
    process.stdout.write(renderCommandHelp(help));
    return ExitCode.Ok;
  }

  switch (parsed.command) {
    case "fit":
      return await fitCommand(parsed.positionals, parsed.flags);
    case "specs":
      return specsCommand(parsed.positionals, parsed.flags);
    case "cache":
      return cacheCommand(parsed.positionals, parsed.flags);
    default: {
      const suggestion = nearestMatch(
        parsed.command,
        COMMANDS.map((c) => c.name),
      );
      process.stderr.write(
        `\n  There is no "${parsed.command}" command.${suggestion === null ? "" : ` Did you mean ${style.bold(`catalog ${suggestion}`)}?`}\n\n  Run catalog --help to see what there is.\n\n`,
      );
      return ExitCode.Usage;
    }
  }
}

/** Turns a thrown error into a message a person can act on and a code a script can read. */
function report(error: unknown): number {
  if (error instanceof CatalogError) {
    process.stderr.write(`\n  ${style.red(error.message)}\n`);
    if (error.fix !== null) {
      process.stderr.write(`\n${error.fix.split("\n").map((l) => `  ${l}`).join("\n")}\n`);
    }
    process.stderr.write("\n");
    return error.exitCode;
  }
  const message = error instanceof Error ? error.message : String(error);
  process.stderr.write(`\n  ${style.red("Something went wrong:")} ${message}\n\n`);
  return ExitCode.Generic;
}

const invokedDirectly =
  process.argv[1] !== undefined &&
  (process.argv[1].endsWith("cli/index.ts") || process.argv[1].endsWith("cli/index.js"));

if (invokedDirectly) {
  main(process.argv.slice(2))
    .then((code) => {
      process.exitCode = code;
    })
    .catch((error: unknown) => {
      process.exitCode = report(error);
    });
}
