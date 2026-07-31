/**
 * Bytes per parameter for each weight format. The values below the 8-bit line
 * are averages: k-quants mix bit widths across tensors, so the true size varies
 * a little by architecture. These are only ever used when the repo does not
 * publish real file sizes, and the output says so when they are.
 */
export const BYTES_PER_PARAM: Record<string, number> = {
  FP32: 4.0,
  FP16: 2.0,
  BF16: 2.0,
  FP8: 1.0,
  INT8: 1.0,
  Q8_0: 1.06,
  Q6_K: 0.82,
  Q5_K_M: 0.7,
  Q5_K_S: 0.68,
  Q4_K_M: 0.58,
  Q4_K_S: 0.55,
  Q4_0: 0.56,
  Q3_K_M: 0.45,
  Q3_K_S: 0.42,
  Q2_K: 0.33,
};

export type QuantFormat = keyof typeof BYTES_PER_PARAM;

export function bytesPerParam(format: string): number | null {
  return BYTES_PER_PARAM[format.toUpperCase()] ?? null;
}

/**
 * Plain-language quality bands, from section 9. Deliberately qualitative:
 * a "retains 97% accuracy" figure is model-specific, rarely published, and
 * inventing one is precisely the failure this tool exists to prevent.
 */
export const QUALITY_BANDS: Record<string, string> = {
  FP32: "full precision, the reference",
  FP16: "full precision in practice, the reference",
  BF16: "full precision in practice, the reference",
  FP8: "very close to the original",
  INT8: "effectively identical to the original",
  Q8_0: "effectively identical to the original",
  Q6_K: "near-lossless",
  Q5_K_M: "a small quality loss, hard to notice",
  Q5_K_S: "a small quality loss, hard to notice",
  Q4_K_M: "a noticeable quality loss, still usable",
  Q4_K_S: "a noticeable quality loss, still usable",
  Q4_0: "a noticeable quality loss, still usable",
  Q3_K_M: "degrades meaningfully",
  Q3_K_S: "degrades meaningfully",
  Q2_K: "degrades meaningfully",
};

/**
 * Bands for the labels the table does not list by name. Quantisers add suffixes
 * constantly (Q4_K_L, Q3_K_XL, IQ3_M), but the leading bit width is what decides
 * the band, so the general rule holds without pretending to more precision than
 * a bit width supports.
 */
const BAND_BY_BIT_WIDTH: Record<string, string> = {
  "8": "effectively identical to the original",
  "6": "near-lossless",
  "5": "a small quality loss, hard to notice",
  "4": "a noticeable quality loss, still usable",
  "3": "degrades meaningfully",
  "2": "degrades meaningfully",
  "1": "degrades severely",
};

export function qualityBand(format: string): string | null {
  const key = format.toUpperCase();
  const exact = QUALITY_BANDS[key];
  if (exact !== undefined) return exact;
  // "IQ" quants pack tighter than the plain "Q" of the same width, so they are
  // named as such rather than quietly folded into the same band.
  const match = /^(I?)Q(\d)/.exec(key);
  const width = match?.[2];
  if (width === undefined) return null;
  const band = BAND_BY_BIT_WIDTH[width];
  if (band === undefined) return null;
  return match?.[1] === "I" ? `${band}, packed tighter than a plain Q${width}` : band;
}

/**
 * Matches a quantisation label in a filename.
 *
 * Deliberately open-ended rather than a fixed list: quantisers keep inventing
 * suffixes (Q4_K_L, Q3_K_XL, Q4_0_4_8, IQ3_M), and a closed list silently
 * mislabels anything new. A label that is read but not in the bytes-per-param
 * table is still a perfectly good label, because GGUF repos publish real file
 * sizes and no estimate is needed.
 */
const QUANT_LABEL = /(^|[^A-Z0-9])(I?Q\d(?:_[A-Z0-9]+)*|BF16|FP16|F16|FP32|F32|FP8)([^A-Z0-9]|$)/;

/** Reads the quantisation out of a GGUF-style filename, e.g. "Model-Q4_K_M.gguf". */
export function quantFromFilename(path: string): string | null {
  const stem = path.toUpperCase().replace(/\.[A-Z0-9]+$/, "");
  const match = QUANT_LABEL.exec(stem);
  const token = match?.[2];
  if (token === undefined) return null;
  return NAME_ALIASES[token] ?? token;
}

const NAME_ALIASES: Record<string, string> = {
  F16: "FP16",
  F32: "FP32",
};

/** Normalises a torch_dtype string ("bfloat16") to a table key ("BF16"). */
export function quantFromTorchDtype(dtype: string | null): string | null {
  if (dtype === null) return null;
  switch (dtype.toLowerCase()) {
    case "bfloat16":
      return "BF16";
    case "float16":
    case "half":
      return "FP16";
    case "float32":
    case "float":
      return "FP32";
    case "float8_e4m3fn":
    case "float8_e5m2":
      return "FP8";
    case "int8":
      return "INT8";
    default:
      return null;
  }
}

/** True for the imatrix flavour of a k-quant, which is calibrated and usually better. */
export function isImatrix(paths: string[]): boolean {
  return paths.some((p) => /imat|imatrix/i.test(p));
}
