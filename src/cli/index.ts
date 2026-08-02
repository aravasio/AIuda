#!/usr/bin/env node
import { realpathSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { CatalogError, ExitCode } from "../errors.ts";
import { colorSupported, setColorEnabled, style } from "../render/format.ts";
import { nearestMatch, parseArgs } from "./args.ts";
import { cacheCommand } from "./commands/cache.ts";
import { discoverCommand } from "./commands/discover.ts";
import { fitCommand } from "./commands/fit.ts";
import { queryCommand } from "./commands/query.ts";
import { specsCommand } from "./commands/specs.ts";
import { teamCommand } from "./commands/team.ts";
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
    case "query":
      return await queryCommand(parsed.positionals, parsed.flags);
    case "fit":
      return await fitCommand(parsed.positionals, parsed.flags);
    case "discover":
      return await discoverCommand(parsed.positionals, parsed.flags);
    case "specs":
      return specsCommand(parsed.positionals, parsed.flags);
    case "team":
      return teamCommand();
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

/**
 * True when this file is the program, rather than something a test imported.
 *
 * Compared by resolved path rather than by filename: an installed `catalog` is a
 * symlink in a bin directory, so argv[1] is the link's own name and no filename
 * test can recognise it. Following the link is what makes the installed command
 * behave like the one run out of the source tree.
 */
function isInvokedDirectly(): boolean {
  const entry = process.argv[1];
  if (entry === undefined) return false;
  try {
    return realpathSync(entry) === fileURLToPath(import.meta.url);
  } catch {
    return false;
  }
}

const invokedDirectly = isInvokedDirectly();

if (invokedDirectly) {
  main(process.argv.slice(2))
    .then((code) => {
      process.exitCode = code;
    })
    .catch((error: unknown) => {
      process.exitCode = report(error);
    });
}
