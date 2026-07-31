import { UsageError } from "../errors.ts";

export interface GlobalFlags {
  json: boolean;
  technical: boolean;
  /** Assumed context length for fit math. Null means use the configured default. */
  context: number | null;
  color: boolean;
  /** Ignore any cached result for this revision and process it again. */
  refresh: boolean;
  help: boolean;
  version: boolean;
}

export interface ParsedArgs {
  command: string | null;
  positionals: string[];
  flags: GlobalFlags;
}

const KNOWN_FLAGS = new Set([
  "--json",
  "--technical",
  "--context",
  "--no-color",
  "--color",
  "--refresh",
  "--help",
  "-h",
  "--version",
  "-v",
]);

export function parseArgs(argv: string[]): ParsedArgs {
  const flags: GlobalFlags = {
    json: false,
    technical: false,
    context: null,
    color: true,
    refresh: false,
    help: false,
    version: false,
  };
  const positionals: string[] = [];

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === undefined) continue;

    if (!arg.startsWith("-")) {
      positionals.push(arg);
      continue;
    }

    // "--context=8192" and "--context 8192" are both accepted.
    const eq = arg.indexOf("=");
    const name = eq === -1 ? arg : arg.slice(0, eq);
    const inlineValue = eq === -1 ? null : arg.slice(eq + 1);

    switch (name) {
      case "--json":
        flags.json = true;
        break;
      case "--technical":
        flags.technical = true;
        break;
      case "--no-color":
        flags.color = false;
        break;
      case "--color":
        flags.color = true;
        break;
      case "--refresh":
        flags.refresh = true;
        break;
      case "--help":
      case "-h":
        flags.help = true;
        break;
      case "--version":
      case "-v":
        flags.version = true;
        break;
      case "--context": {
        const raw = inlineValue ?? argv[++i];
        if (raw === undefined) {
          throw new UsageError(
            "--context needs a number of tokens.",
            "catalog fit <url> --context 32768",
          );
        }
        const value = Number(raw.replace(/[_,]/g, ""));
        if (!Number.isFinite(value) || value <= 0 || !Number.isInteger(value)) {
          throw new UsageError(
            `--context needs a whole number of tokens, not "${raw}".`,
            "catalog fit <url> --context 32768",
          );
        }
        flags.context = value;
        break;
      }
      default:
        throw new UsageError(
          `Unknown option ${name}.${suggestFlag(name)}`,
          "catalog --help lists every option.",
        );
    }
  }

  const [command, ...rest] = positionals;
  return { command: command ?? null, positionals: rest, flags };
}

function suggestFlag(name: string): string {
  const nearest = nearestMatch(name, [...KNOWN_FLAGS]);
  return nearest === null ? "" : ` Did you mean ${nearest}?`;
}

/** Returns the closest candidate, or null when nothing is close enough to suggest. */
export function nearestMatch(input: string, candidates: string[]): string | null {
  let best: string | null = null;
  let bestDistance = Number.POSITIVE_INFINITY;
  for (const candidate of candidates) {
    const distance = editDistance(input.toLowerCase(), candidate.toLowerCase());
    if (distance < bestDistance) {
      bestDistance = distance;
      best = candidate;
    }
  }
  // Suggesting something three edits away is noise, not help.
  const limit = Math.max(2, Math.floor(input.length / 3));
  return bestDistance <= limit ? best : null;
}

function editDistance(a: string, b: string): number {
  const previous: number[] = Array.from({ length: b.length + 1 }, (_, i) => i);
  for (let i = 1; i <= a.length; i += 1) {
    let diagonal = previous[0] ?? 0;
    previous[0] = i;
    for (let j = 1; j <= b.length; j += 1) {
      const temp = previous[j] ?? 0;
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      previous[j] = Math.min((previous[j] ?? 0) + 1, (previous[j - 1] ?? 0) + 1, diagonal + cost);
      diagonal = temp;
    }
  }
  return previous[b.length] ?? 0;
}
