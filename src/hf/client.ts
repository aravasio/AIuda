import { GatedRepoError, NetworkError, NotFoundError } from "../errors.ts";
import type { ModelIndexResult, ModelInfo, RepoFile, RepoSnapshot } from "./types.ts";
import type { RepoRef } from "./url.ts";

const DEFAULT_ENDPOINT = "https://huggingface.co";

/** Metadata is public even for gated repos; files are not. They fail differently. */
type RequestKind = "metadata" | "file";

export interface HfClientOptions {
  endpoint?: string;
  token?: string | null;
  timeoutMs?: number;
  fetchImpl?: typeof fetch;
}

export class HfClient {
  private readonly endpoint: string;
  private readonly token: string | null;
  private readonly timeoutMs: number;
  private readonly fetchImpl: typeof fetch;

  constructor(options: HfClientOptions = {}) {
    this.endpoint = options.endpoint ?? process.env["HF_ENDPOINT"] ?? DEFAULT_ENDPOINT;
    this.token =
      options.token ?? process.env["HF_TOKEN"] ?? process.env["HUGGING_FACE_HUB_TOKEN"] ?? null;
    this.timeoutMs = options.timeoutMs ?? 30_000;
    this.fetchImpl = options.fetchImpl ?? fetch;
  }

  /** Fetches everything the pipeline needs for one repo, in as few round trips as possible. */
  async snapshot(ref: RepoRef): Promise<RepoSnapshot> {
    const info = await this.modelInfo(ref);
    // Every later request pins the resolved SHA, so a repo edited mid-run
    // cannot mix files from two different commits into one answer.
    const revision = info.sha;
    const [files, card, config] = await Promise.all([
      this.tree(ref.repoId, revision),
      this.textFile(ref.repoId, revision, "README.md"),
      this.jsonFile(ref.repoId, revision, "config.json"),
    ]);
    return { ref: { repoId: ref.repoId, revision: ref.revision }, info, files, card, config };
  }

  async modelInfo(ref: RepoRef): Promise<ModelInfo> {
    const base = `${this.endpoint}/api/models/${ref.repoId}`;
    const path = ref.revision ? `${base}/revision/${encodeURIComponent(ref.revision)}` : base;
    // `expand[]` returns only the fields asked for, so `sha` has to be named
    // explicitly: without it there is no revision to key the cache on.
    const fields = [
      "sha",
      "safetensors",
      "cardData",
      "gated",
      "downloads",
      "likes",
      "lastModified",
      "tags",
      "author",
      "pipeline_tag",
      "library_name",
    ];
    const url = `${path}?${fields.map((f) => `expand[]=${f}`).join("&")}`;
    const raw = await this.getJson(url, ref.repoId, "metadata");
    return normaliseModelInfo(ref.repoId, raw);
  }

  /**
   * Names that look like the one asked for, so an unresolvable id can point
   * somewhere instead of just failing. Never throws: a suggestion is a courtesy
   * and must not replace the original error.
   */
  async suggest(repoId: string, limit = 3): Promise<string[]> {
    const name = repoId.split("/")[1] ?? repoId;
    const segments = name.split(/[-_]/);

    // A misspelling rarely matches the whole name, so the query is shortened a
    // segment at a time until something comes back. "Qwen3-Embeding-0.6B" finds
    // nothing; "Qwen3" finds the family the user meant.
    for (let length = segments.length; length > 0; length -= 1) {
      const query = segments.slice(0, length).join("-");
      const found = await this.searchModels(query, limit);
      const useful = found.filter((id) => id !== repoId);
      if (useful.length > 0) return useful.slice(0, limit);
    }
    return [];
  }

  private async searchModels(query: string, limit: number): Promise<string[]> {
    try {
      const url = `${this.endpoint}/api/models?search=${encodeURIComponent(query)}&limit=${limit}&sort=downloads&direction=-1`;
      const response = await this.fetchImpl(url, {
        headers: { Accept: "application/json" },
        signal: AbortSignal.timeout(this.timeoutMs),
      });
      if (!response.ok) return [];
      const raw: unknown = await response.json();
      if (!Array.isArray(raw)) return [];
      return raw
        .map((entry) => (isRecord(entry) && typeof entry["id"] === "string" ? entry["id"] : null))
        .filter((id): id is string => id !== null);
    } catch {
      return [];
    }
  }

  async tree(repoId: string, revision: string): Promise<RepoFile[]> {
    const url = `${this.endpoint}/api/models/${repoId}/tree/${encodeURIComponent(revision)}?recursive=true`;
    const raw = await this.getJson(url, repoId, "metadata");
    if (!Array.isArray(raw)) return [];
    const files: RepoFile[] = [];
    for (const entry of raw) {
      if (!isRecord(entry)) continue;
      if (entry["type"] !== "file") continue;
      const path = typeof entry["path"] === "string" ? entry["path"] : null;
      if (path === null) continue;
      files.push({ path, size: fileSize(entry) });
    }
    return files;
  }

  async textFile(repoId: string, revision: string, path: string): Promise<string | null> {
    const url = `${this.endpoint}/${repoId}/resolve/${encodeURIComponent(revision)}/${path}`;
    const response = await this.request(url, repoId, "file");
    if (response === null) return null;
    return await response.text();
  }

  async jsonFile(
    repoId: string,
    revision: string,
    path: string,
  ): Promise<Record<string, unknown> | null> {
    const text = await this.textFile(repoId, revision, path);
    if (text === null) return null;
    try {
      const parsed: unknown = JSON.parse(text);
      return isRecord(parsed) ? parsed : null;
    } catch {
      // A malformed config.json is treated the same as a missing one: the
      // deterministic branch degrades and says which fields are unavailable.
      return null;
    }
  }

  private async getJson(url: string, repoId: string, kind: RequestKind): Promise<unknown> {
    const response = await this.request(url, repoId, kind);
    if (response === null) {
      throw new NotFoundError(`No model found at ${repoId}.`);
    }
    return (await response.json()) as unknown;
  }

  /** Returns null for a clean 404 so callers can treat "no such file" as a normal case. */
  private async request(url: string, repoId: string, kind: RequestKind): Promise<Response | null> {
    const headers: Record<string, string> = { Accept: "application/json" };
    if (this.token !== null) headers["Authorization"] = `Bearer ${this.token}`;

    let response: Response;
    try {
      response = await this.fetchImpl(url, {
        headers,
        signal: AbortSignal.timeout(this.timeoutMs),
        redirect: "follow",
      });
    } catch (cause) {
      throw new NetworkError(
        `Could not reach ${new URL(url).host}. The first look at a model needs network access; after that it is served from the cache.`,
        "Check your connection, then run the same command again.",
      );
    }

    if (response.ok) return response;
    if (response.status === 404) return null;
    if (response.status === 401 || response.status === 403) {
      // Metadata is public even for gated repos, so a refusal there means the
      // repo is not visible at all. The gate only bites on the files, which is
      // where the "accept the terms" instruction belongs.
      if (kind === "metadata") {
        throw new NotFoundError(
          `No model found at ${repoId}. It may not exist, or it may be private.`,
        );
      }
      throw new GatedRepoError(repoId);
    }
    if (response.status === 429) {
      throw new NetworkError(
        "Hugging Face is rate-limiting this machine.",
        "Wait a minute and try again, or set HF_TOKEN to raise the limit.",
      );
    }
    throw new NetworkError(`Hugging Face replied ${response.status} for ${repoId}.`);
  }
}

/** Hugging Face reports a plain size for small files and an `lfs.size` for weights. */
function fileSize(entry: Record<string, unknown>): number | null {
  const lfs = entry["lfs"];
  if (isRecord(lfs) && typeof lfs["size"] === "number") return lfs["size"];
  if (typeof entry["size"] === "number") return entry["size"];
  return null;
}

export function normaliseModelInfo(repoId: string, raw: unknown): ModelInfo {
  if (!isRecord(raw)) {
    throw new NotFoundError(`Hugging Face returned nothing usable for ${repoId}.`);
  }
  const sha = typeof raw["sha"] === "string" ? raw["sha"] : null;
  if (sha === null) {
    throw new NotFoundError(
      `Hugging Face did not report a revision for ${repoId}, so the result could not be cached safely.`,
    );
  }

  const tags = Array.isArray(raw["tags"]) ? raw["tags"].filter(isString) : [];
  const cardData = isRecord(raw["cardData"]) ? raw["cardData"] : {};
  const safetensors = isRecord(raw["safetensors"]) ? raw["safetensors"] : null;
  const paramsByDtype: Record<string, number> = {};
  if (safetensors !== null && isRecord(safetensors["parameters"])) {
    for (const [dtype, count] of Object.entries(safetensors["parameters"])) {
      if (typeof count === "number") paramsByDtype[dtype] = count;
    }
  }

  return {
    repoId: typeof raw["id"] === "string" ? raw["id"] : repoId,
    sha,
    author: typeof raw["author"] === "string" ? raw["author"] : repoId.split("/")[0] ?? null,
    pipelineTag: typeof raw["pipeline_tag"] === "string" ? raw["pipeline_tag"] : null,
    libraryName: typeof raw["library_name"] === "string" ? raw["library_name"] : null,
    tags,
    downloads: typeof raw["downloads"] === "number" ? raw["downloads"] : null,
    likes: typeof raw["likes"] === "number" ? raw["likes"] : null,
    lastModified: typeof raw["lastModified"] === "string" ? raw["lastModified"] : null,
    license: licenseFrom(cardData, tags),
    gated: raw["gated"] !== false && raw["gated"] !== undefined && raw["gated"] !== null,
    baseModels: baseModelsFrom(cardData, tags),
    safetensorsTotalParams:
      safetensors !== null && typeof safetensors["total"] === "number" ? safetensors["total"] : null,
    safetensorsParamsByDtype: paramsByDtype,
    modelIndex: modelIndexFrom(cardData),
  cardData,
  };
}

function licenseFrom(cardData: Record<string, unknown>, tags: string[]): string | null {
  const declared = cardData["license"];
  if (typeof declared === "string") return declared;
  if (Array.isArray(declared)) {
    const first = declared.find(isString);
    if (first !== undefined) return first;
  }
  const tagged = tags.find((tag) => tag.startsWith("license:"));
  return tagged !== undefined ? tagged.slice("license:".length) : null;
}

function baseModelsFrom(cardData: Record<string, unknown>, tags: string[]): string[] {
  const found = new Set<string>();
  const declared = cardData["base_model"];
  if (typeof declared === "string") found.add(declared);
  if (Array.isArray(declared)) for (const entry of declared) if (isString(entry)) found.add(entry);

  // Tags look like "base_model:Qwen/Qwen2.5-32B" or "base_model:quantized:Qwen/Qwen3-8B".
  for (const tag of tags) {
    if (!tag.startsWith("base_model:")) continue;
    const rest = tag.slice("base_model:".length);
    const parts = rest.split(":");
    const candidate = parts[parts.length - 1];
    if (candidate !== undefined && candidate.includes("/")) found.add(candidate);
  }
  return [...found];
}

/**
 * `model-index` is the structured, machine-written half of the benchmark story.
 * Flattened here into one row per metric so the renderer never walks the tree.
 */
function modelIndexFrom(cardData: Record<string, unknown>): ModelIndexResult[] {
  const index = cardData["model-index"];
  if (!Array.isArray(index)) return [];
  const out: ModelIndexResult[] = [];
  for (const model of index) {
    if (!isRecord(model) || !Array.isArray(model["results"])) continue;
    for (const result of model["results"]) {
      if (!isRecord(result)) continue;
      const task = isRecord(result["task"]) ? stringOrNull(result["task"]["name"] ?? result["task"]["type"]) : null;
      const dataset = isRecord(result["dataset"])
        ? stringOrNull(result["dataset"]["name"] ?? result["dataset"]["type"])
        : null;
      const metrics = Array.isArray(result["metrics"]) ? result["metrics"] : [];
      for (const metric of metrics) {
        if (!isRecord(metric)) continue;
        const name = stringOrNull(metric["name"] ?? metric["type"]);
        const value = metric["value"];
        if (name === null) continue;
        if (typeof value !== "number" && typeof value !== "string") continue;
        out.push({ task, dataset, metric: name, value });
      }
    }
  }
  return out;
}

function stringOrNull(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

function isString(value: unknown): value is string {
  return typeof value === "string";
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
