import { execFileSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import os from "node:os";
import { join } from "node:path";

export interface GpuSpec {
  name: string;
  vramBytes: number | null;
}

export interface RuntimeSpec {
  name: string;
  present: boolean;
  version: string | null;
  endpoint: string | null;
}

export interface MachineSpecs {
  platform: string;
  arch: string;
  osRelease: string | null;
  cpuModel: string | null;
  physicalCores: number | null;
  logicalCores: number | null;
  ramTotalBytes: number | null;
  ramAvailableBytes: number | null;
  /** Unified memory is shared with the CPU; discrete VRAM is not. */
  memoryKind: "unified" | "discrete" | "cpu-only" | "unknown";
  gpus: GpuSpec[];
  diskAvailableBytes: number | null;
  runtimes: RuntimeSpec[];
  /** Probes that failed. Reported rather than filled in with a plausible number. */
  unknown: string[];
  /** Fields that came from the override file instead of a probe. */
  overridden: string[];
}

/**
 * How much of a unified-memory machine's RAM the GPU may actually wire down.
 * Apple caps this well below the installed total; using the full figure would
 * report fits that fail in practice.
 */
export const UNIFIED_MEMORY_GPU_FRACTION = 0.75;

export function overridePath(): string {
  return join(process.env["XDG_CONFIG_HOME"] ?? join(homedir(), ".config"), "catalog", "machine.json");
}

/** Probes the machine. No probe is allowed to throw; a failure degrades one field. */
export function detectMachine(): MachineSpecs {
  const unknown: string[] = [];

  const specs: MachineSpecs = {
    platform: os.platform(),
    arch: os.arch(),
    osRelease: safe(() => os.release(), unknown, "OS version"),
    cpuModel: safe(() => os.cpus()[0]?.model?.trim() ?? null, unknown, "CPU model"),
    logicalCores: safe(() => os.cpus().length || null, unknown, "logical core count"),
    physicalCores: safe(() => physicalCores(), unknown, "physical core count"),
    ramTotalBytes: safe(() => os.totalmem() || null, unknown, "total memory"),
    ramAvailableBytes: safe(() => availableMemory(), unknown, "available memory"),
    memoryKind: "unknown",
    gpus: safe(() => detectGpus(), unknown, "graphics hardware") ?? [],
    diskAvailableBytes: safe(() => availableDisk(), unknown, "free disk space"),
    runtimes: safe(() => detectRuntimes(), unknown, "installed runtimes") ?? [],
    unknown,
    overridden: [],
  };

  specs.memoryKind = classifyMemory(specs);
  return applyOverride(specs);
}

function classifyMemory(specs: MachineSpecs): MachineSpecs["memoryKind"] {
  if (specs.platform === "darwin" && specs.arch === "arm64") return "unified";
  if (specs.gpus.length > 0) return "discrete";
  if (specs.gpus.length === 0 && !specs.unknown.includes("graphics hardware")) return "cpu-only";
  return "unknown";
}

/**
 * The memory a model has to fit into for the fast verdicts. On a discrete-GPU
 * machine this is one card's VRAM: the tool has no way to confirm that
 * multi-GPU sharding is configured, and reporting a fit that needs it would be
 * optimistic in exactly the way this tool is meant to avoid.
 */
export function fastMemoryBytes(specs: MachineSpecs): number | null {
  if (specs.memoryKind === "unified") {
    return specs.ramTotalBytes === null
      ? null
      : Math.floor(specs.ramTotalBytes * UNIFIED_MEMORY_GPU_FRACTION);
  }
  const vrams = specs.gpus.map((g) => g.vramBytes).filter((v): v is number => v !== null);
  if (vrams.length === 0) return null;
  return Math.max(...vrams);
}

function physicalCores(): number | null {
  if (os.platform() === "linux" && existsSync("/proc/cpuinfo")) {
    const text = readFileSync("/proc/cpuinfo", "utf8");
    const seen = new Set<string>();
    let physicalId = "0";
    for (const line of text.split("\n")) {
      const [rawKey, rawValue] = line.split(":");
      if (rawKey === undefined || rawValue === undefined) continue;
      const key = rawKey.trim();
      const value = rawValue.trim();
      if (key === "physical id") physicalId = value;
      if (key === "core id") seen.add(`${physicalId}:${value}`);
    }
    if (seen.size > 0) return seen.size;
    // Containers and VMs often hide topology; the logical count is all there is.
    return os.cpus().length || null;
  }
  if (os.platform() === "darwin") {
    const out = run("sysctl", ["-n", "hw.physicalcpu"]);
    const parsed = out === null ? Number.NaN : Number(out.trim());
    return Number.isFinite(parsed) ? parsed : null;
  }
  return os.cpus().length || null;
}

function availableMemory(): number | null {
  if (os.platform() === "linux" && existsSync("/proc/meminfo")) {
    const text = readFileSync("/proc/meminfo", "utf8");
    const match = /^MemAvailable:\s+(\d+) kB$/m.exec(text);
    if (match?.[1] !== undefined) return Number(match[1]) * 1024;
  }
  return os.freemem() || null;
}

function detectGpus(): GpuSpec[] {
  if (os.platform() === "darwin") {
    if (os.arch() === "arm64") {
      const chip = run("sysctl", ["-n", "machdep.cpu.brand_string"])?.trim() ?? "Apple Silicon";
      const total = os.totalmem();
      return [{ name: `${chip} (unified memory)`, vramBytes: Math.floor(total * UNIFIED_MEMORY_GPU_FRACTION) }];
    }
    return [];
  }

  const nvidia = run("nvidia-smi", ["--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]);
  if (nvidia !== null) {
    const gpus: GpuSpec[] = [];
    for (const line of nvidia.trim().split("\n")) {
      const [name, mib] = line.split(",").map((s) => s.trim());
      if (name === undefined || name === "") continue;
      const megabytes = mib === undefined ? Number.NaN : Number(mib);
      gpus.push({
        name,
        vramBytes: Number.isFinite(megabytes) ? megabytes * 1024 * 1024 : null,
      });
    }
    if (gpus.length > 0) return gpus;
  }

  const rocm = run("rocm-smi", ["--showproductname", "--showmeminfo", "vram", "--csv"]);
  if (rocm !== null) {
    const gpus: GpuSpec[] = [];
    for (const line of rocm.trim().split("\n").slice(1)) {
      const cells = line.split(",");
      const name = cells[1]?.trim();
      const bytes = Number(cells[2]?.trim());
      if (name === undefined || name === "") continue;
      gpus.push({ name, vramBytes: Number.isFinite(bytes) ? bytes : null });
    }
    if (gpus.length > 0) return gpus;
  }

  return [];
}

function availableDisk(): number | null {
  if (os.platform() === "win32") return null;
  const out = run("df", ["-Pk", homedir()]);
  if (out === null) return null;
  const line = out.trim().split("\n")[1];
  if (line === undefined) return null;
  const available = line.split(/\s+/)[3];
  const kilobytes = available === undefined ? Number.NaN : Number(available);
  return Number.isFinite(kilobytes) ? kilobytes * 1024 : null;
}

function detectRuntimes(): RuntimeSpec[] {
  const runtimes: RuntimeSpec[] = [];

  const ollamaVersion = run("ollama", ["--version"]);
  runtimes.push({
    name: "ollama",
    present: ollamaVersion !== null,
    version: ollamaVersion?.trim().replace(/^.*version is\s*/i, "") ?? null,
    endpoint: process.env["OLLAMA_HOST"] ?? "http://127.0.0.1:11434",
  });

  for (const binary of ["llama-server", "llama-cli"]) {
    const found = run("which", [binary]);
    if (found !== null) {
      runtimes.push({ name: binary, present: true, version: null, endpoint: null });
    }
  }

  return runtimes;
}

/** Runs a probe. Any failure means "could not tell", never a crash. */
function run(command: string, args: string[]): string | null {
  try {
    return execFileSync(command, args, {
      encoding: "utf8",
      timeout: 4000,
      stdio: ["ignore", "pipe", "ignore"],
    });
  } catch {
    return null;
  }
}

function safe<T>(probe: () => T, unknown: string[], label: string): T | null {
  try {
    const value = probe();
    if (value === null || value === undefined) {
      unknown.push(label);
      return null;
    }
    return value;
  } catch {
    unknown.push(label);
    return null;
  }
}

/**
 * A hand-written override file wins over every probe, so a machine the tool
 * reads wrongly can be corrected once instead of producing bad fit answers.
 */
export function applyOverride(specs: MachineSpecs, path: string = overridePath()): MachineSpecs {
  if (!existsSync(path)) return specs;
  let parsed: unknown;
  try {
    parsed = JSON.parse(readFileSync(path, "utf8"));
  } catch {
    specs.unknown.push(`override file at ${path} is not valid JSON and was ignored`);
    return specs;
  }
  if (typeof parsed !== "object" || parsed === null) return specs;

  const overridden: string[] = [];
  const merged = { ...specs };
  for (const [key, value] of Object.entries(parsed as Record<string, unknown>)) {
    if (!(key in specs)) continue;
    if (key === "unknown" || key === "overridden") continue;
    (merged as Record<string, unknown>)[key] = value;
    overridden.push(key);
    const index = merged.unknown.indexOf(key);
    if (index !== -1) merged.unknown.splice(index, 1);
  }
  merged.overridden = overridden;
  return merged;
}
