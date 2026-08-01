import { describe, expect, it } from "vitest";
import { setColorEnabled } from "../src/render/format.ts";
import { createProgress, formatElapsed, SILENT, TtyProgress } from "../src/render/progress.ts";

setColorEnabled(false);

/** Stands in for a terminal, and records everything written to it. */
class FakeStream {
  isTTY: boolean;
  readonly writes: string[] = [];

  constructor(isTTY = true) {
    this.isTTY = isTTY;
  }

  write(chunk: string): boolean {
    this.writes.push(chunk);
    return true;
  }

  /** What a person would be left looking at, with the redraws taken out. */
  get lines(): string[] {
    return this.writes
      .join("")
      .split("\n")
      .map((line) => line.replace(/.*\x1b\[2K/, "").trim())
      .filter((line) => line !== "");
  }
}

function progressOn(stream: FakeStream): TtyProgress {
  return new TtyProgress(stream as unknown as NodeJS.WriteStream);
}

describe("saying what is happening while it happens", () => {
  it("keeps a finished phase on screen with how long it took", () => {
    const stream = new FakeStream();
    const progress = progressOn(stream);
    progress.start("Reading the repository");
    progress.done();
    progress.stop();

    const [line] = stream.lines;
    expect(line).toContain("Reading the repository");
    expect(line).toMatch(/\d+s$/);
  });

  it("starting a phase finishes the one before it", () => {
    const stream = new FakeStream();
    const progress = progressOn(stream);
    progress.start("Reading the repository");
    progress.start("Writing the description");
    progress.stop();

    // The first is settled and the second was abandoned by stop(), so only the
    // first is left on screen.
    expect(stream.lines.filter((l) => l.includes("Reading the repository"))).toHaveLength(1);
  });

  it("marks a phase that failed rather than ticking it", () => {
    const stream = new FakeStream();
    const progress = progressOn(stream);
    progress.start("Reading the benchmark tables");
    progress.fail();
    progress.stop();

    const [line] = stream.lines;
    expect(line).toContain("Reading the benchmark tables");
    expect(line).toContain("×");
    expect(line).not.toContain("✓");
  });

  it("clears to the end of the line, so a shorter detail leaves no tail", () => {
    const stream = new FakeStream();
    const progress = progressOn(stream);
    progress.start("Writing the description");
    progress.detail("attempt 2 of 3, 1200 characters");
    progress.detail("8 characters");
    progress.stop();

    expect(stream.writes.filter((w) => w.includes("\x1b[2K")).length).toBeGreaterThan(1);
  });

  it("survives being stopped twice, and ignores work after it stops", () => {
    const stream = new FakeStream();
    const progress = progressOn(stream);
    progress.start("Reading the repository");
    progress.stop();
    progress.stop();
    progress.start("Writing the description");
    progress.done();

    expect(stream.lines.some((l) => l.includes("Writing the description"))).toBe(false);
  });
});

describe("choosing whether to draw at all", () => {
  it("says nothing when the output is a pipe rather than a terminal", () => {
    // A spinner redrawn into a log file is thousands of lines of escape codes.
    const stream = new FakeStream(false);
    expect(createProgress({ json: false, stream: stream as unknown as NodeJS.WriteStream })).toBe(
      SILENT,
    );
  });

  it("says nothing when the caller asked for JSON", () => {
    const stream = new FakeStream(true);
    expect(createProgress({ json: true, stream: stream as unknown as NodeJS.WriteStream })).toBe(
      SILENT,
    );
  });

  it("draws on an interactive terminal", () => {
    const stream = new FakeStream(true);
    expect(createProgress({ json: false, stream: stream as unknown as NodeJS.WriteStream })).not.toBe(
      SILENT,
    );
  });
});

describe("elapsed time", () => {
  it("stays short enough to sit at the end of a redrawn line", () => {
    expect(formatElapsed(0)).toBe("0s");
    expect(formatElapsed(8_400)).toBe("8s");
    expect(formatElapsed(59_000)).toBe("59s");
    expect(formatElapsed(64_000)).toBe("1m 04s");
    expect(formatElapsed(605_000)).toBe("10m 05s");
  });
});
