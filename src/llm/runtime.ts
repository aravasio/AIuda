import http from "node:http";
import https from "node:https";
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
      // Streamed so the reply arrives as it is written rather than in one
      // block at the end.
      stream: true,
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

    // Deliberately not fetch.
    //
    // Node's fetch abandons a request when response headers do not arrive
    // within five minutes, and that limit cannot be raised through AbortSignal
    // or any option fetch accepts. Streaming was not enough on its own: ollama
    // sends no headers at all until it has finished loading the model, so a
    // cold start on a slow machine still hit the limit and reported a working
    // runtime as unreachable. node:http imposes no such deadline, which leaves
    // the configured timeout as the only one.
    try {
      return await postForStreamedText(`${this.endpoint}/api/generate`, body, this.timeoutMs);
    } catch (cause) {
      if (cause instanceof RuntimeUnavailableError) throw cause;
      if (cause instanceof Error && cause.name === "TimeoutError") throw this.timeoutError();
      throw new RuntimeUnavailableError(
        `Lost contact with ollama at ${this.endpoint} while it was working.`,
        "Check that it is still running, then try again:\n  ollama serve",
      );
    }
  }

  private timeoutError(): RuntimeUnavailableError {
    const minutes = Math.round(this.timeoutMs / 60_000);
    return new RuntimeUnavailableError(
      `${this.model} did not answer within ${minutes} minutes. On a machine without a graphics card the runtime has to read the whole model card before it writes anything, which can take a while.`,
      `Give it longer by raising "requestTimeoutMs" in the config file, or use a smaller model:\n  CATALOG_MODEL=<a smaller model>`,
    );
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
 * Posts JSON and reassembles a newline-delimited streamed reply.
 *
 * ollama sends one JSON object per line, each carrying the next fragment. The
 * only deadline is `timeoutMs`, measured across the whole exchange.
 */
export function postForStreamedText(
  url: string,
  body: unknown,
  timeoutMs: number,
): Promise<string> {
  const target = new URL(url);
  const transport = target.protocol === "https:" ? https : http;
  const payload = JSON.stringify(body);

  return new Promise<string>((resolve, reject) => {
    let settled = false;
    const finish = (fn: () => void): void => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      fn();
    };

    const timer = setTimeout(() => {
      const error = new Error("timed out");
      error.name = "TimeoutError";
      finish(() => {
        request.destroy();
        reject(error);
      });
    }, timeoutMs);

    const request = transport.request(
      target,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Content-Length": Buffer.byteLength(payload),
        },
      },
      (response) => {
        const status = response.statusCode ?? 0;
        if (status < 200 || status >= 300) {
          response.resume();
          finish(() =>
            reject(
              new RuntimeUnavailableError(
                `ollama replied ${status} when asked to generate.`,
                `Check the model is usable with:\n  ollama run <model> "hello"`,
              ),
            ),
          );
          return;
        }

        response.setEncoding("utf8");
        let buffer = "";
        let reply = "";
        let failure: RuntimeUnavailableError | null = null;

        const consume = (line: string): void => {
          const text = line.trim();
          if (text === "") return;
          let parsed: unknown;
          try {
            parsed = JSON.parse(text);
          } catch {
            // A partial line at a chunk boundary is normal; the next chunk
            // completes it.
            return;
          }
          if (!isRecord(parsed)) return;
          if (typeof parsed["error"] === "string") {
            failure = new RuntimeUnavailableError(`ollama reported: ${parsed["error"]}`);
            return;
          }
          if (typeof parsed["response"] === "string") reply += parsed["response"];
        };

        response.on("data", (chunk: string) => {
          buffer += chunk;
          const lines = buffer.split("\n");
          buffer = lines.pop() ?? "";
          for (const line of lines) consume(line);
        });

        response.on("end", () => {
          consume(buffer);
          finish(() => {
            if (failure !== null) reject(failure);
            else if (reply === "") reject(new RuntimeUnavailableError("ollama produced an empty reply."));
            else resolve(reply);
          });
        });

        response.on("error", (cause: Error) => finish(() => reject(cause)));
      },
    );

    request.on("error", (cause: Error) => finish(() => reject(cause)));
    request.write(payload);
    request.end();
  });
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
