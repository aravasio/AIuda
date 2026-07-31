import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { applyOverride, detectMachine, fastMemoryBytes } from "../src/specs/detect.ts";
import { fakeMachine } from "./helpers/fixtures.ts";

let dir: string;

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "catalog-specs-"));
});

afterEach(() => {
  rmSync(dir, { recursive: true, force: true });
});

describe("machine detection", () => {
  it("never throws, whatever the machine looks like", () => {
    const specs = detectMachine();
    expect(specs.platform).toBeTypeOf("string");
    expect(Array.isArray(specs.gpus)).toBe(true);
    expect(Array.isArray(specs.unknown)).toBe(true);
  });

  it("degrades a field it cannot read to unknown and names it", () => {
    const specs = fakeMachine({ ramTotalBytes: null, unknown: ["total memory"] });
    expect(specs.ramTotalBytes).toBeNull();
    expect(specs.unknown).toContain("total memory");
  });
});

describe("the override file", () => {
  it("wins over what was probed", () => {
    const path = join(dir, "machine.json");
    writeFileSync(path, JSON.stringify({ ramTotalBytes: 137e9 }));

    const specs = applyOverride(fakeMachine({ ramTotalBytes: 16e9 }), path);
    expect(specs.ramTotalBytes).toBe(137e9);
    expect(specs.overridden).toContain("ramTotalBytes");
  });

  it("supplies a field the probe could not read", () => {
    const path = join(dir, "machine.json");
    writeFileSync(path, JSON.stringify({ gpus: [{ name: "RTX 4090", vramBytes: 24e9 }] }));

    const specs = applyOverride(
      fakeMachine({ gpus: [], unknown: ["graphics hardware"] }),
      path,
    );
    expect(fastMemoryBytes(specs)).toBe(24e9);
  });

  it("ignores keys that are not machine fields", () => {
    const path = join(dir, "machine.json");
    writeFileSync(path, JSON.stringify({ ramTotalBytes: 137e9, somethingElse: "ignored" }));

    const specs = applyOverride(fakeMachine(), path);
    expect(specs.overridden).toEqual(["ramTotalBytes"]);
    expect(specs as unknown as Record<string, unknown>).not.toHaveProperty("somethingElse");
  });

  it("carries on with the probed values when the file is not valid JSON", () => {
    const path = join(dir, "machine.json");
    writeFileSync(path, "{ not json");

    const specs = applyOverride(fakeMachine({ ramTotalBytes: 16e9 }), path);
    expect(specs.ramTotalBytes).toBe(16e9);
    expect(specs.unknown.join(" ")).toContain("not valid JSON");
  });

  it("changes nothing when there is no file", () => {
    const specs = applyOverride(fakeMachine(), join(dir, "absent.json"));
    expect(specs.overridden).toEqual([]);
    expect(specs.ramTotalBytes).toBe(64e9);
  });
});
