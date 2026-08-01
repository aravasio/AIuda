import http from "node:http";
import https from "node:https";
import { RuntimeUnavailableError, UsageError } from "../errors.ts";
import { isRecord } from "../hf/client.ts";
import type { CatalogConfig, RuntimeName } from "../config.ts";

export interface GenerateRequest {
  system: string;
  prompt: string;
  /** Tokens the model is allowed to produce. Reserved out of the window first. */
  maxOutputTokens: number;
  /** Size of the window to ask the runtime for. */
  contextTokens: number;
  /**
   * Called with each fragment as it arrives, so a caller can show that the
   * reply is being written. The reply is already streamed; this only exposes
   * what was arriving anyway.
   */
  onFragment?: (text: string) => void;
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

/** What a runtime has to provide. */
export interface LlmRuntime {
  readonly name: string;
  readonly model: string;
  status(): Promise<RuntimeStatus>;
  /** Throws RuntimeUnavailableError with an actionable fix when anything is missing. */
  requireReady(): Promise<RuntimeStatus>;
  generate(request: GenerateRequest): Promise<GeneratedReply>;
}

/** A reply, and whether the runtime stopped it because it ran out of room. */
export interface GeneratedReply {
  text: string;
  /** True when the output limit cut the reply off rather than the model finishing. */
  truncated: boolean;
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

  async generate(request: GenerateRequest): Promise<GeneratedReply> {
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

    try {
      return await postForStreamedText(
        `${this.endpoint}/api/generate`,
        body,
        this.timeoutMs,
        OLLAMA_STREAM,
        request.onFragment,
      );
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
 * Runs the configured model through OpenRouter.
 *
 * The trade against ollama is not speed, it is where the model card goes. A
 * local runtime reads the card on this machine; this one posts it to a third
 * party, which is why the runtime has to be asked for by name and is never
 * fallen back to.
 */
export class OpenRouterRuntime implements LlmRuntime {
  readonly name = "openrouter";
  readonly model: string;
  private readonly endpoint: string;
  private readonly timeoutMs: number;
  private readonly apiKey: string | null;
  private readonly fetchImpl: typeof fetch;

  constructor(config: CatalogConfig, fetchImpl: typeof fetch = fetch) {
    this.model = config.model;
    this.endpoint = config.runtimeEndpoint.replace(/\/$/, "");
    this.timeoutMs = config.requestTimeoutMs;
    this.apiKey = config.apiKey;
    this.fetchImpl = fetchImpl;
  }

  async status(): Promise<RuntimeStatus> {
    // The catalogue is public, so reachability and whether the model exists can
    // both be settled before the key is ever needed. That keeps "your key is
    // missing" and "that model does not exist" as separate answers.
    const catalogue = await this.catalogue();
    if (catalogue === null) {
      return {
        reachable: false,
        modelPresent: false,
        modelMaxContext: null,
        version: null,
        problem: `Nothing is answering at ${this.endpoint}.`,
      };
    }

    const entry = catalogue.get(this.model);
    if (entry === undefined) {
      return {
        reachable: true,
        modelPresent: false,
        modelMaxContext: null,
        version: null,
        problem: `OpenRouter has no model called "${this.model}".`,
      };
    }

    return {
      reachable: true,
      modelPresent: true,
      modelMaxContext: entry,
      version: null,
      problem: null,
    };
  }

  async requireReady(): Promise<RuntimeStatus> {
    if (this.apiKey === null) {
      throw new RuntimeUnavailableError(
        "OpenRouter needs an API key, and none is set. The plain-English explanation cannot be written without one.",
        "Create a key at https://openrouter.ai/keys, then:\n  export OPENROUTER_API_KEY=sk-or-...\n\nThe key is read from the environment and never from the config file, so it does not end up in a file you might share.",
      );
    }

    const status = await this.status();
    if (!status.reachable) {
      throw new RuntimeUnavailableError(
        `${status.problem} The plain-English explanation needs a model runtime, so there is nothing useful to print without it.`,
        "Check that this machine is online. To use a model on this machine instead:\n  CATALOG_RUNTIME=ollama catalog query <url>",
      );
    }
    if (!status.modelPresent) {
      throw new RuntimeUnavailableError(
        `${status.problem} The plain-English explanation needs it.`,
        `Pick one from https://openrouter.ai/models and name it in full, owner and all:\n  CATALOG_MODEL=deepseek/deepseek-v4-flash-0731`,
      );
    }
    return status;
  }

  async generate(request: GenerateRequest): Promise<GeneratedReply> {
    const body = {
      model: this.model,
      messages: [
        { role: "system", content: request.system },
        { role: "user", content: request.prompt },
      ],
      stream: true,
      max_tokens: request.maxOutputTokens,
      // The job is transcription of a card into fixed fields, not invention.
      temperature: 0,
      // Asked for, not relied on. Models that cannot do it have the parameter
      // ignored rather than rejected, and the extraction pass already copes
      // with prose wrapped around the JSON.
      response_format: { type: "json_object" },
    };

    // Unlike ollama, the window is not ours to set: the context length belongs
    // to the model as the provider serves it, and is read back in status().

    try {
      return await postForStreamedText(
        `${this.endpoint}/chat/completions`,
        body,
        this.timeoutMs,
        this.protocol(),
        request.onFragment,
      );
    } catch (cause) {
      if (cause instanceof RuntimeUnavailableError) throw cause;
      if (cause instanceof Error && cause.name === "TimeoutError") {
        const minutes = Math.round(this.timeoutMs / 60_000);
        throw new RuntimeUnavailableError(
          `${this.model} did not answer within ${minutes} minutes.`,
          `Give it longer by raising "requestTimeoutMs" in the config file, or pick a faster model:\n  CATALOG_MODEL=<another model>`,
        );
      }
      throw new RuntimeUnavailableError(
        `Lost contact with OpenRouter at ${this.endpoint} while it was working.`,
        "Check that this machine is online, then try again.",
      );
    }
  }

  /** Server-sent events: `data:` lines, comment lines, and a `[DONE]` sentinel. */
  private protocol(): StreamProtocol {
    const model = this.model;
    return {
      headers: {
        Authorization: `Bearer ${this.apiKey ?? ""}`,
        // Identifies the caller on OpenRouter's side. Not authentication.
        "X-Title": "catalog",
      },
      readLine: readOpenRouterLine,
      onHttpError: (status, body) => openRouterHttpError(status, body, model),
      emptyReply: `${this.model} returned an empty reply through OpenRouter.`,
    };
  }

  /** Model id to the context length OpenRouter serves it at. Public, no key needed. */
  private async catalogue(): Promise<Map<string, number | null> | null> {
    try {
      const response = await this.fetchImpl(`${this.endpoint}/models`, {
        signal: AbortSignal.timeout(20_000),
      });
      if (!response.ok) return null;
      const raw: unknown = await response.json();
      if (!isRecord(raw) || !Array.isArray(raw["data"])) return null;
      const models = new Map<string, number | null>();
      for (const entry of raw["data"]) {
        if (!isRecord(entry) || typeof entry["id"] !== "string") continue;
        const context = entry["context_length"];
        models.set(entry["id"], typeof context === "number" ? context : null);
      }
      return models;
    } catch {
      return null;
    }
  }
}

/**
 * Reads one line of an OpenRouter stream.
 *
 * Server-sent events, not ollama's one-object-per-line: fragments arrive as
 * `delta.content` rather than `response`, and a stream can carry a failure after
 * the headers already said 200, when a provider gives out partway through.
 */
export function readOpenRouterLine(line: string, sink: StreamSink): void {
  // `: OPENROUTER PROCESSING` arrives while a provider is still thinking.
  // Comments keep the connection open and carry no reply.
  if (line.startsWith(":")) return;
  if (!line.startsWith("data:")) return;
  const payload = line.slice("data:".length).trim();
  if (payload === "" || payload === "[DONE]") return;

  let parsed: unknown;
  try {
    parsed = JSON.parse(payload);
  } catch {
    return;
  }
  if (!isRecord(parsed)) return;

  const error = parsed["error"];
  if (isRecord(error) && typeof error["message"] === "string") {
    sink.fail(new RuntimeUnavailableError(`OpenRouter reported: ${error["message"]}`));
    return;
  }

  const choices = parsed["choices"];
  if (!Array.isArray(choices)) return;
  for (const choice of choices) {
    if (!isRecord(choice)) continue;
    const delta = choice["delta"];
    if (isRecord(delta) && typeof delta["content"] === "string") sink.append(delta["content"]);
    // Cut off at the output limit, which is a different problem from a model
    // that wrote something malformed.
    if (choice["finish_reason"] === "length") sink.markTruncated();
  }
}

/**
 * Turns a refusal into something the user can act on.
 *
 * The four cases are separated because the fix differs: a dead key, an empty
 * balance and a rate limit all look identical as "it did not work", and only one
 * of them is worth waiting out.
 */
export function openRouterHttpError(
  status: number,
  body: string,
  model: string,
): RuntimeUnavailableError {
  const reported = messageFromErrorBody(body);
  const detail = reported === null ? "" : ` It said: ${reported}`;
  if (status === 401 || status === 403) {
    return new RuntimeUnavailableError(
      `OpenRouter rejected the API key.${detail}`,
      "Check the key is current at https://openrouter.ai/keys, then:\n  export OPENROUTER_API_KEY=sk-or-...",
    );
  }
  if (status === 402) {
    return new RuntimeUnavailableError(
      `This OpenRouter account cannot pay for ${model}.${detail}`,
      "Add credit at https://openrouter.ai/credits, or pick a cheaper model with CATALOG_MODEL, or run it on this machine instead:\n  CATALOG_RUNTIME=ollama catalog query <url>",
    );
  }
  if (status === 429) {
    return new RuntimeUnavailableError(
      `OpenRouter is rate limiting this key.${detail}`,
      "Wait and try again, or run the model on this machine instead:\n  CATALOG_RUNTIME=ollama catalog query <url>",
    );
  }
  return new RuntimeUnavailableError(
    `OpenRouter replied ${status} when asked to generate.${detail}`,
    `Check that ${model} is currently served at https://openrouter.ai/models`,
  );
}

/** Pulls the human-readable reason out of an error body, when there is one. */
function messageFromErrorBody(body: string): string | null {
  if (body.trim() === "") return null;
  try {
    const parsed: unknown = JSON.parse(body);
    if (isRecord(parsed)) {
      const error = parsed["error"];
      if (isRecord(error) && typeof error["message"] === "string") return error["message"];
      if (typeof parsed["message"] === "string") return parsed["message"];
    }
  } catch {
    // Not JSON. A short plain-text body is still worth repeating.
  }
  const flat = body.replace(/\s+/g, " ").trim();
  return flat.length > 200 ? `${flat.slice(0, 200)}...` : flat;
}

/**
 * Builds the runtime the config asks for.
 *
 * An unknown name fails here rather than falling back to a working default: a
 * typo that silently sends a model card to a different place than the one asked
 * for is worse than a run that stops.
 */
export function createRuntime(config: CatalogConfig, fetchImpl: typeof fetch = fetch): LlmRuntime {
  switch (config.runtime) {
    case "ollama":
      return new OllamaRuntime(config, fetchImpl);
    case "openrouter":
      return new OpenRouterRuntime(config, fetchImpl);
    default:
      throw new UsageError(
        `There is no "${String(config.runtime)}" runtime.`,
        `The runtimes this tool knows are:\n  ollama       a model running on this machine\n  openrouter   a model served over the network\n\nSet it with CATALOG_RUNTIME, or "runtime" in the config file.`,
      );
  }
}

/** Runtimes this build can talk to, for help text and error messages. */
export const RUNTIME_NAMES: RuntimeName[] = ["ollama", "openrouter"];

/** ollama writes one JSON object per line, each carrying the next fragment. */
const OLLAMA_STREAM: StreamProtocol = {
  readLine(line, sink) {
    let parsed: unknown;
    try {
      parsed = JSON.parse(line);
    } catch {
      // A partial line at a chunk boundary is normal; the next chunk completes it.
      return;
    }
    if (!isRecord(parsed)) return;
    if (typeof parsed["error"] === "string") {
      sink.fail(new RuntimeUnavailableError(`ollama reported: ${parsed["error"]}`));
      return;
    }
    if (typeof parsed["response"] === "string") sink.append(parsed["response"]);
    // The runtime says why it stopped. "length" means the reply was cut off at
    // the output limit, which is a different problem from a model that wrote
    // something malformed.
    if (parsed["done_reason"] === "length") sink.markTruncated();
  },
  onHttpError(status) {
    return new RuntimeUnavailableError(
      `ollama replied ${status} when asked to generate.`,
      `Check the model is usable with:\n  ollama run <model> "hello"`,
    );
  },
  emptyReply: "ollama produced an empty reply.",
};

/** How much of a failed reply's body is worth quoting back at the user. */
const ERROR_BODY_LIMIT = 2000;

/** Where a streamed reply is assembled, one line at a time. */
export interface StreamSink {
  /** Adds the next fragment of the reply. */
  append(text: string): void;
  /** Records that the runtime stopped at the output limit rather than finishing. */
  markTruncated(): void;
  /** Records a failure the runtime reported inside the stream. */
  fail(error: RuntimeUnavailableError): void;
}

/** How one runtime's stream differs from another's. The transport does not care. */
export interface StreamProtocol {
  headers?: Record<string, string>;
  /** Reads one line of the response body into the sink. */
  readLine(line: string, sink: StreamSink): void;
  /** Turns a non-2xx reply into the error the user should see. */
  onHttpError(status: number, body: string): RuntimeUnavailableError;
  /** Said when the stream ended without a single fragment of reply. */
  emptyReply: string;
}

/**
 * Posts JSON and reassembles a streamed reply, line by line.
 *
 * Deliberately not fetch.
 *
 * Node's fetch abandons a request when response headers do not arrive within
 * five minutes, and that limit cannot be raised through AbortSignal or any
 * option fetch accepts. Streaming was not enough on its own: ollama sends no
 * headers at all until it has finished loading the model, so a cold start on a
 * slow machine still hit the limit and reported a working runtime as
 * unreachable. node:http imposes no such deadline, which leaves the configured
 * timeout as the only one.
 *
 * What counts as a line, and what a line means, belongs to the protocol: ollama
 * writes one JSON object per line, OpenRouter writes server-sent events.
 */
export function postForStreamedText(
  url: string,
  body: unknown,
  timeoutMs: number,
  protocol: StreamProtocol,
  onFragment?: (text: string) => void,
): Promise<GeneratedReply> {
  const target = new URL(url);
  const transport = target.protocol === "https:" ? https : http;
  const payload = JSON.stringify(body);

  return new Promise<GeneratedReply>((resolve, reject) => {
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
          ...protocol.headers,
        },
      },
      (response) => {
        response.setEncoding("utf8");
        const status = response.statusCode ?? 0;

        if (status < 200 || status >= 300) {
          // The body of a refusal is where the reason lives: which model was
          // rejected, or that the key is out of credit. Read enough of it to
          // say so, and no more.
          let detail = "";
          response.on("data", (chunk: string) => {
            if (detail.length < ERROR_BODY_LIMIT) detail += chunk;
          });
          response.on("end", () =>
            finish(() => reject(protocol.onHttpError(status, detail.slice(0, ERROR_BODY_LIMIT)))),
          );
          response.on("error", () => finish(() => reject(protocol.onHttpError(status, ""))));
          return;
        }

        let buffer = "";
        let reply = "";
        let truncated = false;
        let failure: RuntimeUnavailableError | null = null;

        const sink: StreamSink = {
          append: (text) => {
            reply += text;
            onFragment?.(text);
          },
          markTruncated: () => {
            truncated = true;
          },
          fail: (error) => {
            failure = error;
          },
        };

        const consume = (line: string): void => {
          const text = line.trim();
          if (text === "") return;
          protocol.readLine(text, sink);
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
            else if (reply === "") reject(new RuntimeUnavailableError(protocol.emptyReply));
            else resolve({ text: reply, truncated });
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
