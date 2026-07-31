import { renderSpecs } from "../../render/specs.ts";
import { detectMachine } from "../../specs/detect.ts";
import type { GlobalFlags } from "../args.ts";

export function specsCommand(_positionals: string[], flags: GlobalFlags): number {
  const specs = detectMachine();
  if (flags.json) {
    process.stdout.write(`${JSON.stringify(specs, null, 2)}\n`);
    return 0;
  }
  process.stdout.write(renderSpecs(specs));
  return 0;
}
