import { RuntimeUnavailableError } from "../errors.ts";
import { isRecord } from "../hf/client.ts";
import type { CatalogConfig } from "../config.ts";

export interface GenerateRequest {
  system: string;
  prompt: string;
  /** Tokens the model is allowed to produce. Reserved out of the window first. */
  maxOutputTokens: number;
  /** Size of the window to ask the runtime for. */
  contextTokens: number;
}

export interface RuntimeStatus {
  reachable: boolean;
  /** True when the configured model has been pulled. */
  modelPresent: boolean;
  /** Largest window the model itself supports. */
  modelMaxContext: number | null;
  version: string | null;
  /** What went wrong, when something did. */
  problem: string | null;
}

/** What a runtime has to provide. Only ollama is implemented; the shape allows others. */
export interface LlmRuntime {
  readonly name: string;
  readonly model: string;
  status(): Promise<RuntimeStatus>;
  /** Throws RuntimeUnavailableError with an actionable fix when anything is missing. */
  requireReady(): Promise<RuntimeStatus>;
  generate(request: GenerateRequest): Promise<string>;
}

/**
 * Runs the configured model through an already-installed ollama. The tool never
 * implements inference; it asks a runtime that is already there.
 */
export class OllamaRuntime implements LlmRuntime {
  readonly name = "ollama";
  readonly model: string;
  private readonly endpoint: string;
  private readonly timeoutMs: number;
  private readonly fetchImpl: typeof fetch;

  constructor(config: CatalogConfig, fetchImpl: typeof fetch = fetch) {
    this.model = config.model;
    this.endpoint = config.runtimeEndpoint.replace(/\/$/, "");
    this.timeoutMs = config.requestTimeoutMs;
    this.fetchImpl = fetchImpl;
  }

  async status(): Promise<RuntimeStatus> {
    const version = await this.version();
    if (version === null) {
      return {
        reachable: false,
        modelPresent: false,
        modelMaxContext: null,
        version: null,
        problem: `Nothing is answering at ${this.endpoint}.`,
      };
    }

    const shown = await this.show();
    if (shown === null) {
      return {
        reachable: true,
        modelPresent: false,
        modelMaxContext: null,
        version,
        problem: `ollama is running but "${this.model}" has not been pulled.`,
      };
    }

    return {
      reachable: true,
      modelPresent: true,
      modelMaxContext: modelMaxContext(shown),
      version,
      problem: null,
    };
  }

  /**
   * The plain-English explanation is the product. If the runtime cannot produce
   * it, the run fails here rather than quietly returning numbers only, because a
   * user does not notice what is missing from an answer that still looks whole.
   */
  async requireReady(): Promise<RuntimeStatus> {
    const status = await this.status();
    if (!status.reachable) {
      throw new RuntimeUnavailableError(
        `${status.problem} The plain-English explanation needs a model runtime, so there is nothing useful to print without it.`,
        `Start it with:\n  ollama serve\n\nIf ollama is not installed, get it from https://ollama.com/download`,
      );
    }
    if (!status.modelPresent) {
      throw new RuntimeUnavailableError(
        `${status.problem} The plain-English explanation needs it.`,
        `Download it with:\n  ollama pull ${this.model}\n\nOr point the tool at a model you already have, with CATALOG_MODEL=<name> or the "model" setting in the config file.`,
      );
    }
    return status;
  }

  async generate(request: GenerateRequest): Promise<string> {
    const body = {
      model: this.model,
      system: request.system,
      prompt: request.prompt,
      stream: false,
      format: "json",
      options: {
        // Asked for explicitly: ollama starts at a window far below what the
        // model supports, and a prompt over that limit is silently cut off.
        num_ctx: request.contextTokens,
        num_predict: request.maxOutputTokens,
        // The job is transcription of a card into fixed fields, not invention.
        temperature: 0,
      },
      think: false,
    };

    let response: Response;
    try {
      response = await this.fetchImpl(`${this.endpoint}/api/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(this.timeoutMs),
      });
    } catch {
      throw new RuntimeUnavailableError(
        `Lost contact with ollama at ${this.endpoint} while it was working.`,
        "Check that it is still running, then try again.",
      );
    }

    if (!response.ok) {
      throw new RuntimeUnavailableError(
        `ollama replied ${response.status} when asked to run ${this.model}.`,
        `Check the model is usable with:\n  ollama run ${this.model} "hello"`,
      );
    }

    const raw: unknown = await response.json();
    if (!isRecord(raw) || typeof raw["response"] !== "string") {
      throw new RuntimeUnavailableError("ollama returned something that was not a reply.");
    }
    return raw["response"];
  }

  private async version(): Promise<string | null> {
    try {
      const response = await this.fetchImpl(`${this.endpoint}/api/version`, {
        signal: AbortSignal.timeout(5000),
      });
      if (!response.ok) return null;
      const raw: unknown = await response.json();
      return isRecord(raw) && typeof raw["version"] === "string" ? raw["version"] : "unknown";
    } catch {
      return null;
    }
  }

  private async show(): Promise<Record<string, unknown> | null> {
    try {
      const response = await this.fetchImpl(`${this.endpoint}/api/show`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: this.model }),
        signal: AbortSignal.timeout(10_000),
      });
      if (!response.ok) return null;
      const raw: unknown = await response.json();
      return isRecord(raw) ? raw : null;
    } catch {
      return null;
    }
  }
}

/**
 * Reads the model's own maximum from what the runtime reports. The key is
 * namespaced by architecture ("qwen3.context_length"), so it is found by suffix.
 */
export function modelMaxContext(shown: Record<string, unknown>): number | null {
  const info = shown["model_info"];
  if (!isRecord(info)) return null;
  for (const [key, value] of Object.entries(info)) {
    if (key.endsWith(".context_length") && typeof value === "number") return value;
  }
  return null;
}

export interface ContextPlan {
  /** Window the runtime will be asked for. */
  windowTokens: number;
  /** Reserved for the reply before the card is measured against what is left. */
  outputTokens: number;
  /** What the system prompt and scaffolding cost. */
  overheadTokens: number;
  /** What is left for the card itself. */
  cardBudgetTokens: number;
  /** True when the model's own maximum forced the window below what was asked for. */
  cappedByModel: boolean;
}

/**
 * Works out how much of the window the card may occupy.
 *
 * The reply shares the window with the prompt, so its allowance is taken out
 * first and the card is trimmed to whatever remains. Doing it the other way
 * round produces a prompt that fits and a reply that gets cut off mid-JSON.
 */
export function planContext(input: {
  requestedWindow: number;
  modelMaxContext: number | null;
  outputTokens: number;
  overheadTokens: number;
}): ContextPlan {
  const cappedByModel =
    input.modelMaxContext !== null && input.modelMaxContext < input.requestedWindow;
  const windowTokens = cappedByModel
    ? (input.modelMaxContext as number)
    : input.requestedWindow;

  return {
    windowTokens,
    outputTokens: input.outputTokens,
    overheadTokens: input.overheadTokens,
    cardBudgetTokens: Math.max(0, windowTokens - input.outputTokens - input.overheadTokens),
    cappedByModel,
  };
}
