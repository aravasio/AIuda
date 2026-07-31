let colorEnabled = true;

export function setColorEnabled(enabled: boolean): void {
  colorEnabled = enabled;
}

export function colorSupported(): boolean {
  if (process.env["NO_COLOR"] !== undefined) return false;
  if (process.env["TERM"] === "dumb") return false;
  return process.stdout.isTTY === true;
}

function wrap(code: string, text: string): string {
  return colorEnabled ? `[${code}m${text}[0m` : text;
}

export const style = {
  bold: (t: string) => wrap("1", t),
  dim: (t: string) => wrap("2", t),
  green: (t: string) => wrap("32", t),
  yellow: (t: string) => wrap("33", t),
  red: (t: string) => wrap("31", t),
  cyan: (t: string) => wrap("36", t),
};

/** Left column width for the "Label   value" layout used throughout the output. */
export const LABEL_WIDTH = 13;

export function labelled(label: string, value: string): string {
  return `  ${style.dim(label.padEnd(LABEL_WIDTH))}${value}`;
}

/** Wraps prose to a readable width and indents continuation lines to match. */
export function wrapText(text: string, width: number, indent: string): string {
  const words = text.split(/\s+/).filter((w) => w.length > 0);
  const lines: string[] = [];
  let current = "";
  for (const word of words) {
    if (current === "") {
      current = word;
    } else if (current.length + 1 + word.length <= width) {
      current += ` ${word}`;
    } else {
      lines.push(current);
      current = word;
    }
  }
  if (current !== "") lines.push(current);
  return lines.map((line) => indent + line).join("\n");
}

export function plural(count: number, singular: string, pluralForm?: string): string {
  return count === 1 ? singular : (pluralForm ?? `${singular}s`);
}

export function formatCount(value: number | null): string {
  if (value === null) return "unknown";
  return value.toLocaleString("en-US");
}

/**
 * "35.9B", "0.6B", "600M" — the form vendors use in model names.
 *
 * Truncated rather than rounded: a repo named 35B holds 35.95 billion
 * parameters, and rounding that to "36B" reads as a contradiction of the name
 * on the same screen.
 */
export function formatParams(params: number | null): string {
  if (params === null) return "unknown";
  if (params < 1_000_000_000) return `${Math.round(params / 1_000_000)}M`;
  const billions = Math.trunc((params / 1_000_000_000) * 10) / 10;
  return `${billions.toFixed(1).replace(/\.0$/, "")}B`;
}
