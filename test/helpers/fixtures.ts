import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import type { RepoSnapshot } from "../../src/hf/types.ts";
import type { MachineSpecs } from "../../src/specs/detect.ts";

const ROOT = join(import.meta.dirname, "..", "fixtures");

export type FixtureSlug =
  | "standard-dense"
  | "moe-multimodal"
  | "world-model"
  | "audio-utility"
  | "embedding"
  | "third-party-gguf"
  | "gated"
  | "vision"
  | "adapter";

/** Loads a recorded Hugging Face response. Tests never touch the network. */
export function loadSnapshot(slug: FixtureSlug): RepoSnapshot {
  const path = join(ROOT, slug, "snapshot.json");
  if (!existsSync(path)) {
    throw new Error(`No snapshot fixture for "${slug}". Run: npm run fixtures:refresh`);
  }
  return JSON.parse(readFileSync(path, "utf8")) as RepoSnapshot;
}

/** Loads the recorded failure for a repo that cannot be read without a token. */
export function loadRecordedError(slug: FixtureSlug): { error: string; name: string; repoId: string } {
  const path = join(ROOT, slug, "error.json");
  if (!existsSync(path)) {
    throw new Error(`No error fixture for "${slug}". Run: npm run fixtures:refresh`);
  }
  return JSON.parse(readFileSync(path, "utf8")) as { error: string; name: string; repoId: string };
}

/**
 * A fixed machine, so a verdict assertion means the same thing on every
 * developer's laptop and in CI.
 */
export function fakeMachine(overrides: Partial<MachineSpecs> = {}): MachineSpecs {
  return {
    platform: "linux",
    arch: "x64",
    osRelease: "6.0.0",
    cpuModel: "Test CPU",
    physicalCores: 8,
    logicalCores: 16,
    ramTotalBytes: 64 * 1e9,
    ramAvailableBytes: 48 * 1e9,
    memoryKind: "discrete",
    gpus: [{ name: "Test GPU", vramBytes: 24 * 1e9 }],
    diskAvailableBytes: 500 * 1e9,
    runtimes: [],
    unknown: [],
    overridden: [],
    ...overrides,
  };
}
