import { style } from "./format.ts";

/**
 * Says what the tool is doing while it does it.
 *
 * A query makes several network calls and several passes of a language model,
 * and on a machine that is loading an 8B model from disk the gap between "the
 * card was fetched" and "the description is written" is minutes of silence. The
 * silence is indistinguishable from a hang, so the phases are named as they
 * happen and each one keeps its elapsed time on screen when it finishes.
 */
export interface Progress {
  /** Begins a phase, finishing the previous one. */
  start(label: string): void;
  /** Replaces the detail shown beside the running phase. */
  detail(text: string): void;
  /** Finishes the running phase, keeping it on screen with how long it took. */
  done(note?: string): void;
  /** Abandons the running phase without claiming it succeeded. */
  fail(): void;
  /** Releases the terminal. Safe to call more than once. */
  stop(): void;
}

/** Does nothing, for JSON output, pipes and tests. */
export const SILENT: Progress = {
  start: () => {},
  detail: () => {},
  done: () => {},
  fail: () => {},
  stop: () => {},
};

const FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
const FRAME_MS = 80;

/**
 * Draws to stderr, never stdout.
 *
 * The result is the product and has to stay pipeable, so nothing here may end
 * up in it: `catalog query --json | jq` must not see a spinner.
 */
export class TtyProgress implements Progress {
  private readonly stream: NodeJS.WriteStream;
  private timer: NodeJS.Timeout | null = null;
  private frame = 0;
  private label = "";
  private note = "";
  private startedAt = 0;
  private stopped = false;

  constructor(stream: NodeJS.WriteStream = process.stderr) {
    this.stream = stream;
  }

  start(label: string): void {
    if (this.stopped) return;
    if (this.label !== "") this.done();
    this.label = label;
    this.note = "";
    this.startedAt = Date.now();
    this.frame = 0;
    this.draw();
    if (this.timer === null) {
      this.timer = setInterval(() => {
        this.frame = (this.frame + 1) % FRAMES.length;
        this.draw();
      }, FRAME_MS);
      // The spinner must never be the reason a finished process stays alive.
      this.timer.unref?.();
    }
  }

  detail(text: string): void {
    if (this.stopped || this.label === "") return;
    this.note = text;
    this.draw();
  }

  done(note?: string): void {
    if (this.stopped || this.label === "") return;
    const finished = note ?? this.note;
    this.clear();
    this.stream.write(
      `  ${style.green("✓")} ${this.label}${finished === "" ? "" : ` ${style.dim(finished)}`} ${style.dim(this.elapsed())}\n`,
    );
    this.label = "";
    this.note = "";
  }

  fail(): void {
    if (this.stopped || this.label === "") return;
    this.clear();
    // Named rather than ticked: the phase is where the run stopped, and the
    // error printed after it means more with the phase still on screen.
    this.stream.write(`  ${style.red("×")} ${this.label} ${style.dim(this.elapsed())}\n`);
    this.label = "";
    this.note = "";
  }

  stop(): void {
    if (this.stopped) return;
    this.clear();
    if (this.timer !== null) {
      clearInterval(this.timer);
      this.timer = null;
    }
    this.stopped = true;
    this.label = "";
  }

  private elapsed(): string {
    return formatElapsed(Date.now() - this.startedAt);
  }

  private draw(): void {
    if (this.stopped || this.label === "") return;
    this.clear();
    const detail = this.note === "" ? "" : ` ${style.dim(this.note)}`;
    this.stream.write(
      `  ${style.cyan(FRAMES[this.frame] ?? "")} ${this.label}${detail} ${style.dim(this.elapsed())}`,
    );
  }

  private clear(): void {
    // Carriage return plus a clear-to-end-of-line, so a long detail replaced by
    // a short one does not leave its tail behind.
    this.stream.write("\r[2K");
  }
}

/** "8s", "1m 04s" — short enough to sit at the end of a line that is being redrawn. */
export function formatElapsed(ms: number): string {
  const seconds = Math.round(ms / 1000);
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes}m ${String(seconds % 60).padStart(2, "0")}s`;
}

/**
 * Picks a reporter for the situation.
 *
 * Anything that is not an interactive terminal gets silence: a spinner redrawn
 * into a log file is thousands of lines of escape codes, and a caller reading
 * JSON wants the machine output and nothing else.
 */
export function createProgress(options: { json: boolean; stream?: NodeJS.WriteStream }): Progress {
  const stream = options.stream ?? process.stderr;
  if (options.json) return SILENT;
  if (stream.isTTY !== true) return SILENT;
  if (process.env["NO_COLOR"] !== undefined) return SILENT;
  if (process.env["TERM"] === "dumb") return SILENT;
  return new TtyProgress(stream);
}
