import { humanBytes } from "../fit/verdict.ts";
import { fastMemoryBytes, overridePath, type MachineSpecs } from "../specs/detect.ts";
import { labelled, style, wrapText } from "./format.ts";

export function renderSpecs(specs: MachineSpecs): string {
  const lines: string[] = [];
  lines.push("");
  lines.push(style.bold("  This machine"));
  lines.push("");

  lines.push(labelled("System", `${specs.platform} ${specs.arch}${specs.osRelease === null ? "" : ` (${specs.osRelease})`}`));
  lines.push(labelled("Processor", cpuLine(specs)));
  lines.push(labelled("Memory", memoryLine(specs)));
  lines.push(labelled("Graphics", gpuLine(specs)));
  lines.push(labelled("Free disk", specs.diskAvailableBytes === null ? "unknown" : humanBytes(specs.diskAvailableBytes)));
  lines.push(labelled("Runtimes", runtimeLine(specs)));
  lines.push("");

  const fast = fastMemoryBytes(specs);
  lines.push(
    wrapText(
      fast === null
        ? "Models will be measured against system memory, because no graphics memory was found."
        : `Models are measured against ${humanBytes(fast)}, the memory a model has to fit into to run at full speed here.`,
      72,
      "  ",
    ),
  );
  lines.push("");

  if (specs.unknown.length > 0) {
    lines.push(wrapText(`Could not read: ${specs.unknown.join(", ")}.`, 72, "  "));
    lines.push(
      wrapText(
        `Correct any of these by hand in ${overridePath()} — a JSON object whose keys match the field names above, for example {"ramTotalBytes": 68719476736}.`,
        72,
        "  ",
      ),
    );
    lines.push("");
  }

  if (specs.overridden.length > 0) {
    lines.push(style.dim(`  Taken from your override file: ${specs.overridden.join(", ")}.`));
    lines.push("");
  }

  return lines.join("\n");
}

function cpuLine(specs: MachineSpecs): string {
  const model = specs.cpuModel ?? "unknown";
  const cores: string[] = [];
  if (specs.physicalCores !== null) cores.push(`${specs.physicalCores} cores`);
  if (specs.logicalCores !== null) cores.push(`${specs.logicalCores} threads`);
  return cores.length === 0 ? model : `${model}, ${cores.join(", ")}`;
}

function memoryLine(specs: MachineSpecs): string {
  if (specs.ramTotalBytes === null) return "unknown";
  const available =
    specs.ramAvailableBytes === null ? "" : `, ${humanBytes(specs.ramAvailableBytes)} free right now`;
  const kind =
    specs.memoryKind === "unified" ? " (shared with the GPU)" : "";
  return `${humanBytes(specs.ramTotalBytes)} total${available}${kind}`;
}

function gpuLine(specs: MachineSpecs): string {
  if (specs.gpus.length === 0) {
    return specs.memoryKind === "cpu-only" ? "none detected, models will run on the CPU" : "unknown";
  }
  return specs.gpus
    .map((gpu) => `${gpu.name}${gpu.vramBytes === null ? "" : ` (${humanBytes(gpu.vramBytes)})`}`)
    .join("; ");
}

function runtimeLine(specs: MachineSpecs): string {
  const present = specs.runtimes.filter((r) => r.present);
  if (present.length === 0) return "none found";
  return present.map((r) => (r.version === null ? r.name : `${r.name} ${r.version}`)).join(", ");
}
