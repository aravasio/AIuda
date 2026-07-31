/** One file in the repo, with the size Hugging Face reports for it. */
export interface RepoFile {
  path: string;
  /** Bytes. LFS-backed files report their real size under `lfs`, which we prefer. */
  size: number | null;
}

/** A benchmark result taken from the `model-index` block in the card's front matter. */
export interface ModelIndexResult {
  task: string | null;
  dataset: string | null;
  metric: string;
  value: number | string;
}

/** The subset of the Hugging Face model API we actually depend on. */
export interface ModelInfo {
  repoId: string;
  /** Commit SHA of the resolved revision. This is the cache key. */
  sha: string;
  author: string | null;
  pipelineTag: string | null;
  libraryName: string | null;
  tags: string[];
  downloads: number | null;
  likes: number | null;
  lastModified: string | null;
  license: string | null;
  gated: boolean;
  /** Base model ids declared in card metadata or tags, e.g. for quantizations and adapters. */
  baseModels: string[];
  /** Exact parameter count from safetensors metadata when the repo publishes it. */
  safetensorsTotalParams: number | null;
  /** Per-dtype parameter counts, e.g. { BF16: 35951822704 }. */
  safetensorsParamsByDtype: Record<string, number>;
  modelIndex: ModelIndexResult[];
  cardData: Record<string, unknown>;
}

/** Everything fetched for one repo at one revision. */
export interface RepoSnapshot {
  ref: { repoId: string; revision: string | null };
  info: ModelInfo;
  files: RepoFile[];
  /** README.md contents, or null if the repo has none. */
  card: string | null;
  /** Parsed config.json, or null if the repo has none. */
  config: Record<string, unknown> | null;
}
