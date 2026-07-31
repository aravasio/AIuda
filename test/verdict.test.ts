import { describe, expect, it } from "vitest";
import { COMFORTABLE_HEADROOM, type Breakdown } from "../src/fit/math.ts";
import { decideVerdict } from "../src/fit/verdict.ts";
import { fastMemoryBytes, UNIFIED_MEMORY_GPU_FRACTION } from "../src/specs/detect.ts";
import { fakeMachine } from "./helpers/fixtures.ts";

function breakdown(totalBytes: number | null, unknown: string[] = []): Breakdown {
  return {
    weights: totalBytes === null ? null : { bytes: totalBytes, source: "repo file listing", quant: "BF16" },
    kv: null,
    runtimeOverheadBytes: 1e9,
    sideComponents: [],
    totalBytes,
    contextTokens: 8192,
    unknown,
  };
}

const dense = { isMoe: false };
const moe = { isMoe: true };

describe("verdicts on a discrete-GPU machine", () => {
  const machine = fakeMachine({ gpus: [{ name: "GPU", vramBytes: 24e9 }], ramTotalBytes: 64e9 });

  it("calls it comfortable with the required headroom free", () => {
    const limit = 24e9 * (1 - COMFORTABLE_HEADROOM);
    expect(decideVerdict(breakdown(limit - 1), machine, dense).kind).toBe("runs-comfortably");
  });

  it("calls it tight once the headroom is gone", () => {
    const limit = 24e9 * (1 - COMFORTABLE_HEADROOM);
    expect(decideVerdict(breakdown(limit + 1), machine, dense).kind).toBe("runs-tight");
    expect(decideVerdict(breakdown(24e9), machine, dense).kind).toBe("runs-tight");
  });

  it("calls a dense model that overflows VRAM but fits RAM slow", () => {
    const verdict = decideVerdict(breakdown(30e9), machine, dense);
    expect(verdict.kind).toBe("runs-slowly");
    expect(verdict.detail).toContain("CPU");
  });

  it("offers the split for a mixture-of-experts model in the same position", () => {
    const verdict = decideVerdict(breakdown(30e9), machine, moe);
    expect(verdict.kind).toBe("split-possible");
    expect(verdict.detail).toMatch(/n-cpu-moe|KTransformers/);
  });

  it("refuses when nothing on the machine is large enough", () => {
    expect(decideVerdict(breakdown(200e9), machine, dense).kind).toBe("will-not-run");
    expect(decideVerdict(breakdown(200e9), machine, moe).kind).toBe("will-not-run");
  });
});

describe("verdicts without a GPU", () => {
  const machine = fakeMachine({ gpus: [], memoryKind: "cpu-only", ramTotalBytes: 32e9 });

  it("never claims full speed, only that it fits", () => {
    const verdict = decideVerdict(breakdown(8e9), machine, dense);
    expect(verdict.kind).toBe("runs-slowly");
    expect(verdict.fastMemoryBytes).toBeNull();
  });

  it("still refuses what does not fit in RAM", () => {
    expect(decideVerdict(breakdown(64e9), machine, dense).kind).toBe("will-not-run");
  });
});

describe("verdicts on unified memory", () => {
  const machine = fakeMachine({
    platform: "darwin",
    arch: "arm64",
    memoryKind: "unified",
    gpus: [{ name: "Apple M-series", vramBytes: 48e9 }],
    ramTotalBytes: 64e9,
  });

  it("measures against the share the GPU may actually use, not the installed total", () => {
    expect(fastMemoryBytes(machine)).toBe(Math.floor(64e9 * UNIFIED_MEMORY_GPU_FRACTION));
  });

  it("says unified memory rather than VRAM", () => {
    const verdict = decideVerdict(breakdown(20e9), machine, dense);
    expect(verdict.kind).toBe("runs-comfortably");
    expect(verdict.detail).toContain("unified memory");
  });
});

describe("multi-GPU machines", () => {
  it("measures against one card, since sharding cannot be assumed", () => {
    const machine = fakeMachine({
      gpus: [
        { name: "A", vramBytes: 24e9 },
        { name: "B", vramBytes: 24e9 },
      ],
    });
    expect(fastMemoryBytes(machine)).toBe(24e9);
  });

  it("ignores a card whose memory could not be read", () => {
    const machine = fakeMachine({
      gpus: [
        { name: "A", vramBytes: null },
        { name: "B", vramBytes: 16e9 },
      ],
    });
    expect(fastMemoryBytes(machine)).toBe(16e9);
  });
});

describe("when something is unknown", () => {
  it("says so rather than producing a verdict from a partial total", () => {
    const verdict = decideVerdict(breakdown(null, ["KV cache size"]), fakeMachine(), dense);
    expect(verdict.kind).toBe("unknown");
    expect(verdict.detail).toContain("KV cache size");
  });

  it("says so when the machine's memory could not be read", () => {
    const machine = fakeMachine({ gpus: [], ramTotalBytes: null, memoryKind: "unknown" });
    expect(decideVerdict(breakdown(8e9), machine, dense).kind).toBe("unknown");
  });
});
