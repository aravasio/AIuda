import { existsSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

/**
 * Which language model to use is a setting, not a constant. Nothing in the code
 * may assume a size or a hardware class: swapping in a smaller model has to be
 * a one-line config change.
 */
export interface CatalogConfig {
  /** Runtime that hosts the model. Only ollama is implemented; the shape allows others. */
  runtime: "ollama";
  runtimeEndpoint: string;
  /** Model tag as the runtime knows it. */
  model: string;
  /** Upper bound on the JSON the model may produce, reserved before the card is trimmed. */
  responseTokenBudget: number;
  /**
   * Window to ask the runtime for, capped by what the model supports.
   *
   * Asked for explicitly because runtimes start far below what a model can do:
   * ollama defaults to 4096 tokens for a model that supports 262144. Reading
   * the model's own maximum instead would give a number the runtime ignores.
   */
  llmContextTokens: number;
  /** Assumed context for fit math when --context is not given. */
  defaultContextTokens: number;
  /**
   * Allowance added on top of weights and KV cache for the runtime process.
   * An assumption rather than a measurement, and the largest source of error in
   * the fit calculation, so it is exposed here and named in the output.
   */
  runtimeOverheadBytes: number;
  /** How many times a malformed response is retried before the run fails. */
  maxSchemaRetries: number;
  /**
   * How long a single call to the runtime may take.
   *
   * Sized for the slow case rather than the fast one: on a machine without a
   * GPU the runtime has to load the model from disk and then read the whole
   * prompt before it writes anything, which takes minutes. A timeout tuned for
   * GPU speeds turns a working setup into an intermittent failure.
   */
  requestTimeoutMs: number;
}

export const DEFAULT_CONFIG: CatalogConfig = {
  runtime: "ollama",
  runtimeEndpoint: "http://127.0.0.1:11434",
  model: "qwen3:8b",
  responseTokenBudget: 700,
  llmContextTokens: 16384,
  defaultContextTokens: 8192,
  runtimeOverheadBytes: 1_000_000_000,
  maxSchemaRetries: 2,
  requestTimeoutMs: 1_200_000,
};

export function configPath(): string {
  return join(
    process.env["XDG_CONFIG_HOME"] ?? join(homedir(), ".config"),
    "catalog",
    "config.json",
  );
}

/** File first, then environment, then the defaults. */
export function loadConfig(path: string = configPath()): CatalogConfig {
  let config: CatalogConfig = { ...DEFAULT_CONFIG };

  if (existsSync(path)) {
    try {
      const parsed: unknown = JSON.parse(readFileSync(path, "utf8"));
      if (typeof parsed === "object" && parsed !== null) {
        config = { ...config, ...(parsed as Partial<CatalogConfig>) };
      }
    } catch {
      // A broken config file falls back to defaults rather than blocking the run.
    }
  }

  const envModel = process.env["CATALOG_MODEL"];
  if (envModel !== undefined && envModel !== "") config.model = envModel;

  const envEndpoint = process.env["CATALOG_RUNTIME_ENDPOINT"] ?? process.env["OLLAMA_HOST"];
  if (envEndpoint !== undefined && envEndpoint !== "") {
    config.runtimeEndpoint = envEndpoint.startsWith("http") ? envEndpoint : `http://${envEndpoint}`;
  }

  return config;
}
