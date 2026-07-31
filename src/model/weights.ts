import type { RepoFile } from "../hf/types.ts";
import { quantFromFilename, quantFromTorchDtype } from "./quant.ts";

export type WeightFormat = "safetensors" | "gguf" | "pytorch" | "other";

/**
 * One complete set of weights in the repo. A plain model repo has exactly one.
 * A GGUF repo usually ships several, one per quantisation, and summing them all
 * would report a size nobody would ever download.
 */
export interface WeightVariant {
  /** What to call it in output, e.g. "Q4_K_M" or "BF16". */
  label: string;
  /** Table key for bytes-per-param lookups, when the format is a known one. */
  quant: string | null;
  format: WeightFormat;
  bytes: number;
  files: string[];
  /** Calibrated k-quant. Usually a better result at the same size. */
  imatrix: boolean;
}

const WEIGHT_EXTENSIONS: Array<{ ext: string; format: WeightFormat }> = [
  { ext: ".safetensors", format: "safetensors" },
  { ext: ".gguf", format: "gguf" },
  { ext: ".bin", format: "pytorch" },
  { ext: ".pth", format: "pytorch" },
];

function formatOf(path: string): WeightFormat | null {
  const lower = path.toLowerCase();
  for (const { ext, format } of WEIGHT_EXTENSIONS) {
    if (lower.endsWith(ext)) {
      // Repos carry plenty of .bin files that are not weights.
      if (format === "pytorch" && !/(^|\/)(pytorch_model|model|consolidated)/i.test(path)) {
        return null;
      }
      return format;
    }
  }
  return null;
}

/**
 * Groups the repo's weight files into the sets a user would actually download.
 *
 * Real sizes beat any estimate, so this is the preferred input to the fit math.
 * The grouping rules exist to stop obvious double counting: a repo holding both
 * safetensors and the legacy .bin copies of the same tensors would otherwise
 * report twice the true size.
 */
export function collectWeightVariants(
  files: RepoFile[],
  torchDtype: string | null = null,
): WeightVariant[] {
  const weightFiles = files
    .map((file) => ({ file, format: formatOf(file.path) }))
    .filter((entry): entry is { file: RepoFile; format: WeightFormat } => entry.format !== null);

  if (weightFiles.length === 0) return [];

  const gguf = weightFiles.filter((e) => e.format === "gguf");
  const safetensors = weightFiles.filter((e) => e.format === "safetensors");
  const pytorch = weightFiles.filter((e) => e.format === "pytorch");

  const variants: WeightVariant[] = [];

  // GGUF repos are quantisation catalogues. Each file is a complete model on
  // its own unless it is part of a numbered shard set, so the grouping key is
  // the shard set, never the quant label: a repo holding Q4_0, Q4_0_4_4 and
  // Q4_0_8_8 has three separate models, not one 13 GB download.
  if (gguf.length > 0) {
    const groups = new Map<string, RepoFile[]>();
    for (const { file } of gguf) {
      const key = shardSetKey(file.path);
      const existing = groups.get(key);
      if (existing === undefined) groups.set(key, [file]);
      else existing.push(file);
    }
    for (const [key, group] of groups) {
      const quant = quantFromFilename(key);
      variants.push({
        label: quant ?? describeUnlabelled(key),
        quant,
        format: "gguf",
        bytes: sumSizes(group),
        files: group.map((f) => f.path).sort(),
        imatrix: group.some((f) => /imat/i.test(f.path)),
      });
    }
  }

  // Prefer safetensors and drop the .bin duplicates; fall back to .bin only
  // when a repo predates safetensors entirely.
  const tensorFiles = safetensors.length > 0 ? safetensors : pytorch;
  if (tensorFiles.length > 0) {
    const chosen = dropDuplicateShardSets(tensorFiles.map((e) => e.file));
    const quant = quantFromTorchDtype(torchDtype) ?? quantFromFilename(chosen[0]?.path ?? "");
    variants.push({
      label: quant ?? (safetensors.length > 0 ? "safetensors" : "PyTorch"),
      quant,
      format: safetensors.length > 0 ? "safetensors" : "pytorch",
      bytes: sumSizes(chosen),
      files: chosen.map((f) => f.path).sort(),
      imatrix: false,
    });
  }

  return variants.sort((a, b) => b.bytes - a.bytes);
}

/**
 * Identity of the shard set a file belongs to. "Model-Q4_K_M-00001-of-00002.gguf"
 * and its sibling share one key; every other file is its own.
 */
function shardSetKey(path: string): string {
  return path.replace(/-\d{4,5}-of-\d{4,5}(\.[A-Za-z0-9]+)$/, "$1");
}

/** Last resort label: the part of the filename that distinguishes this file. */
function describeUnlabelled(path: string): string {
  const stem = basename(path).replace(/\.[A-Za-z0-9]+$/, "");
  const tail = stem.split(/[-_]/).pop();
  return tail === undefined || tail === "" ? "GGUF" : tail;
}

/**
 * A sharded set ("model-00001-of-00026.safetensors") and a single-file copy of
 * the same weights ("consolidated.safetensors") often live side by side.
 * When both are present, keep the sharded set and drop the standalone copies.
 */
function dropDuplicateShardSets(files: RepoFile[]): RepoFile[] {
  const sharded = files.filter((f) => /-\d{4,5}-of-\d{4,5}\./i.test(f.path));
  if (sharded.length === 0) return files;
  const standalone = files.filter((f) => !/-\d{4,5}-of-\d{4,5}\./i.test(f.path));
  // Keep standalone files only if they are clearly something else, such as an
  // adapter or a vision tower shipped alongside the sharded language model.
  const keep = standalone.filter((f) => !/(consolidated|^model\.safetensors$)/i.test(basename(f.path)));
  return [...sharded, ...keep];
}

function basename(path: string): string {
  const index = path.lastIndexOf("/");
  return index === -1 ? path : path.slice(index + 1);
}

function sumSizes(files: RepoFile[]): number {
  let total = 0;
  for (const file of files) {
    if (file.size !== null) total += file.size;
  }
  return total;
}

/** Total download size of the repo, which includes tokenizer and config files. */
export function totalRepoBytes(files: RepoFile[]): number {
  return sumSizes(files);
}
