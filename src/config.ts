import { existsSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

/**
 * Which language model to use is a setting, not a constant. Nothing in the code
 * may assume a size or a hardware class: swapping in a smaller model has to be
 * a one-line config change.
 */
/**
 * Where the model that writes the prose runs.
 *
 * "ollama" keeps the model card on this machine. "openrouter" posts it to a
 * third party, which is a choice about privacy as much as about speed, so it is
 * only ever used when it has been asked for by name.
 */
export type RuntimeName = "ollama" | "openrouter";

export interface CatalogConfig {
  /** Runtime that hosts the model. */
  runtime: RuntimeName;
  runtimeEndpoint: string;
  /**
   * Credential for a runtime that needs one. Read from the environment only,
   * never from the config file: a key in a file is a key that gets committed,
   * pasted into a bug report, or read by anything else running as this user.
   */
  apiKey: string | null;
  /** Model tag as the runtime knows it. */
  model: string;
  /** Upper bound on the JSON the model may produce, reserved before the card is trimmed. */
  responseTokenBudget: number;
  /**
   * Room for the benchmark reply, which is a list rather than a paragraph.
   *
   * A vendor comparison table with five columns and a dozen rows becomes sixty
   * JSON entries, so this is several times the prose allowance. Too small and
   * the reply is cut off mid-list.
   */
  benchmarkTokenBudget: number;
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

/** Where each runtime lives when the config file does not say. */
export const DEFAULT_ENDPOINTS: Record<RuntimeName, string> = {
  ollama: "http://127.0.0.1:11434",
  openrouter: "https://openrouter.ai/api/v1",
};

/** The model each runtime uses when the config file does not say. */
export const DEFAULT_MODELS: Record<RuntimeName, string> = {
  ollama: "qwen3:8b",
  openrouter: "deepseek/deepseek-v4-flash-0731",
};

export const DEFAULT_CONFIG: CatalogConfig = {
  runtime: "ollama",
  runtimeEndpoint: DEFAULT_ENDPOINTS.ollama,
  apiKey: null,
  model: DEFAULT_MODELS.ollama,
  responseTokenBudget: 700,
  benchmarkTokenBudget: 3000,
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
export function loadConfig(
  path: string = configPath(),
  env: NodeJS.ProcessEnv = process.env,
): CatalogConfig {
  let config: CatalogConfig = { ...DEFAULT_CONFIG };
  /** Settings the user named themselves, which the runtime's defaults must not overwrite. */
  const named = new Set<string>();

  if (existsSync(path)) {
    try {
      const parsed: unknown = JSON.parse(readFileSync(path, "utf8"));
      if (typeof parsed === "object" && parsed !== null) {
        const fromFile = parsed as Partial<CatalogConfig>;
        // A key in the file cannot set the credential, whatever it is called.
        delete fromFile.apiKey;
        for (const key of Object.keys(fromFile)) named.add(key);
        config = { ...config, ...fromFile };
      }
    } catch {
      // A broken config file falls back to defaults rather than blocking the run.
    }
  }

  const envRuntime = env["CATALOG_RUNTIME"];
  if (envRuntime !== undefined && envRuntime !== "") {
    // Validated where the runtime is built, so an unknown name reaches the user
    // as the name they typed rather than as a silent fall back to ollama.
    config.runtime = envRuntime as RuntimeName;
    named.add("runtime");
  }

  const envModel = env["CATALOG_MODEL"];
  if (envModel !== undefined && envModel !== "") {
    config.model = envModel;
    named.add("model");
  }

  // OLLAMA_HOST is ollama's own variable and says nothing about anywhere else,
  // so it only applies when ollama is the runtime.
  const envEndpoint =
    env["CATALOG_RUNTIME_ENDPOINT"] ?? (config.runtime === "ollama" ? env["OLLAMA_HOST"] : undefined);
  if (envEndpoint !== undefined && envEndpoint !== "") {
    config.runtimeEndpoint = envEndpoint.startsWith("http") ? envEndpoint : `http://${envEndpoint}`;
    named.add("runtimeEndpoint");
  }

  // The defaults in DEFAULT_CONFIG are ollama's. Switching runtime without
  // saying where it lives or which model to use has to move both, or the tool
  // would ask openrouter for a model named "qwen3:8b" at a local address.
  if (config.runtime in DEFAULT_ENDPOINTS) {
    if (!named.has("runtimeEndpoint")) config.runtimeEndpoint = DEFAULT_ENDPOINTS[config.runtime];
    if (!named.has("model")) config.model = DEFAULT_MODELS[config.runtime];
  }

  const key = env["OPENROUTER_API_KEY"] ?? env["CATALOG_API_KEY"];
  if (key !== undefined && key.trim() !== "") config.apiKey = key.trim();

  return config;
}
