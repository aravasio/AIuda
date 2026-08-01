import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { DEFAULT_CONFIG, loadConfig, type CatalogConfig } from "../src/config.ts";
import { RuntimeUnavailableError, UsageError } from "../src/errors.ts";
import {
  createRuntime,
  openRouterHttpError,
  OpenRouterRuntime,
  readOpenRouterLine,
  type StreamSink,
} from "../src/llm/runtime.ts";

/** Collects what a stream reader produced, so a line can be asserted on directly. */
function collect(lines: string[]): { text: string; truncated: boolean; failure: string | null } {
  let text = "";
  let truncated = false;
  let failure: string | null = null;
  const sink: StreamSink = {
    append: (fragment) => {
      text += fragment;
    },
    markTruncated: () => {
      truncated = true;
    },
    fail: (error) => {
      failure = error.message;
    },
  };
  for (const line of lines) readOpenRouterLine(line.trim(), sink);
  return { text, truncated, failure };
}

function sse(payload: unknown): string {
  return `data: ${JSON.stringify(payload)}`;
}

function delta(content: string, finish: string | null = null): string {
  return sse({ choices: [{ delta: { content }, finish_reason: finish }] });
}

describe("reading an OpenRouter stream", () => {
  it("reassembles the reply from the fragments", () => {
    const result = collect([delta('{"one_liner"'), delta(': "a model"'), delta("}"), "data: [DONE]"]);
    expect(result.text).toBe('{"one_liner": "a model"}');
    expect(result.truncated).toBe(false);
    expect(result.failure).toBeNull();
  });

  it("ignores the keep-alive comments sent while a provider is thinking", () => {
    const result = collect([": OPENROUTER PROCESSING", delta("hello"), ": OPENROUTER PROCESSING"]);
    expect(result.text).toBe("hello");
  });

  it("tells a reply cut off at the output limit apart from a finished one", () => {
    expect(collect([delta("{", null), delta('"a"', "length")]).truncated).toBe(true);
    expect(collect([delta("{}", "stop")]).truncated).toBe(false);
  });

  it("reports an error that arrives after the headers already said 200", () => {
    const result = collect([delta("partial"), sse({ error: { message: "provider gave out" } })]);
    expect(result.failure).toContain("provider gave out");
  });

  it("survives a half-written line at a chunk boundary", () => {
    const result = collect([delta("kept"), 'data: {"choices": [{"delta": {"cont']);
    expect(result.text).toBe("kept");
    expect(result.failure).toBeNull();
  });

  it("skips fragments that carry no content, including the opening role event", () => {
    const result = collect([
      sse({ choices: [{ delta: { role: "assistant" } }] }),
      delta("text"),
      sse({ id: "gen-1" }),
    ]);
    expect(result.text).toBe("text");
  });
});

describe("what OpenRouter says when it refuses", () => {
  it("separates a dead key, an empty balance and a rate limit", () => {
    const key = openRouterHttpError(401, "", "a/model");
    const paid = openRouterHttpError(402, "", "a/model");
    const limited = openRouterHttpError(429, "", "a/model");

    expect(key.message).toContain("rejected the API key");
    expect(key.fix).toContain("openrouter.ai/keys");

    expect(paid.message).toContain("cannot pay");
    expect(paid.fix).toContain("credits");

    expect(limited.message).toContain("rate limiting");
    // Waiting is the fix here and nowhere else.
    expect(limited.fix).toContain("Wait");
  });

  it("repeats the reason the service gave, when it gave one", () => {
    const body = JSON.stringify({ error: { message: "No endpoints found for that model." } });
    expect(openRouterHttpError(404, body, "a/model").message).toContain("No endpoints found");
  });

  it("says something useful when the body is not JSON at all", () => {
    const error = openRouterHttpError(502, "<html>Bad Gateway</html>", "a/model");
    expect(error.message).toContain("502");
    expect(error.message).toContain("Bad Gateway");
  });
});

describe("choosing a runtime", () => {
  it("builds each runtime the config names", () => {
    expect(createRuntime({ ...DEFAULT_CONFIG, runtime: "ollama" }).name).toBe("ollama");
    expect(createRuntime({ ...DEFAULT_CONFIG, runtime: "openrouter" }).name).toBe("openrouter");
  });

  it("refuses a name it does not know rather than falling back to a working one", () => {
    // Falling back would send the card somewhere the user did not ask for.
    const config = { ...DEFAULT_CONFIG, runtime: "openrouter-typo" } as unknown as CatalogConfig;
    expect(() => createRuntime(config)).toThrow(UsageError);
    expect(() => createRuntime(config)).toThrow(/openrouter-typo/);
  });
});

describe("OpenRouter readiness", () => {
  const config: CatalogConfig = {
    ...DEFAULT_CONFIG,
    runtime: "openrouter",
    runtimeEndpoint: "https://openrouter.example/api/v1",
    model: "vendor/model",
  };

  /** Answers the catalogue call with a fixed model list. Never touches the network. */
  function catalogue(models: Array<{ id: string; context_length?: number }>): typeof fetch {
    return (async () =>
      new Response(JSON.stringify({ data: models }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      })) as unknown as typeof fetch;
  }

  it("asks for a key before anything else, since the catalogue is public", async () => {
    const runtime = new OpenRouterRuntime(
      { ...config, apiKey: null },
      catalogue([{ id: "vendor/model", context_length: 1000 }]),
    );
    await expect(runtime.requireReady()).rejects.toThrow(RuntimeUnavailableError);
    await expect(runtime.requireReady()).rejects.toThrow(/needs an API key/);
  });

  it("reads the context length the provider actually serves", async () => {
    const runtime = new OpenRouterRuntime(
      { ...config, apiKey: "sk-or-test" },
      catalogue([{ id: "vendor/model", context_length: 1_048_576 }]),
    );
    const status = await runtime.requireReady();
    expect(status.modelPresent).toBe(true);
    expect(status.modelMaxContext).toBe(1_048_576);
  });

  it("names a model that is not served, rather than failing at generate time", async () => {
    const runtime = new OpenRouterRuntime(
      { ...config, apiKey: "sk-or-test" },
      catalogue([{ id: "someone/else", context_length: 8192 }]),
    );
    await expect(runtime.requireReady()).rejects.toThrow(/no model called "vendor\/model"/);
  });

  it("reports an unreachable service as unreachable, not as a missing model", async () => {
    const dead = (async () => {
      throw new Error("ENOTFOUND");
    }) as unknown as typeof fetch;
    const status = await new OpenRouterRuntime({ ...config, apiKey: "sk-or-test" }, dead).status();
    expect(status.reachable).toBe(false);
    expect(status.problem).toContain("Nothing is answering");
  });
});

describe("configuring which runtime to use", () => {
  const made: string[] = [];

  afterEach(() => {
    for (const dir of made.splice(0)) rmSync(dir, { recursive: true, force: true });
  });

  function configFile(contents: unknown): string {
    const dir = mkdtempSync(join(tmpdir(), "catalog-config-"));
    made.push(dir);
    const path = join(dir, "config.json");
    writeFileSync(path, JSON.stringify(contents));
    return path;
  }

  function missingFile(): string {
    const dir = mkdtempSync(join(tmpdir(), "catalog-config-"));
    made.push(dir);
    return join(dir, "config.json");
  }

  it("moves the endpoint and the model when the runtime changes", () => {
    // The defaults are ollama's. Carrying them over would ask OpenRouter for a
    // model called "qwen3:8b" at a local address.
    const config = loadConfig(missingFile(), { CATALOG_RUNTIME: "openrouter" });
    expect(config.runtime).toBe("openrouter");
    expect(config.runtimeEndpoint).toBe("https://openrouter.ai/api/v1");
    expect(config.model).toContain("/");
  });

  it("keeps a model the user named across a runtime change", () => {
    const config = loadConfig(missingFile(), {
      CATALOG_RUNTIME: "openrouter",
      CATALOG_MODEL: "vendor/something",
    });
    expect(config.model).toBe("vendor/something");
  });

  it("keeps an endpoint the config file named", () => {
    const path = configFile({ runtime: "openrouter", runtimeEndpoint: "https://proxy.example/v1" });
    expect(loadConfig(path, {}).runtimeEndpoint).toBe("https://proxy.example/v1");
  });

  it("ignores OLLAMA_HOST when ollama is not the runtime", () => {
    // It is ollama's own variable and says nothing about anywhere else.
    const config = loadConfig(missingFile(), {
      CATALOG_RUNTIME: "openrouter",
      OLLAMA_HOST: "127.0.0.1:11434",
    });
    expect(config.runtimeEndpoint).toBe("https://openrouter.ai/api/v1");
    expect(loadConfig(missingFile(), { OLLAMA_HOST: "box:11434" }).runtimeEndpoint).toBe(
      "http://box:11434",
    );
  });

  it("reads the key from the environment and never from the file", () => {
    const path = configFile({ runtime: "openrouter", apiKey: "sk-or-from-a-file" });
    expect(loadConfig(path, {}).apiKey).toBeNull();
    expect(loadConfig(path, { OPENROUTER_API_KEY: "sk-or-from-the-env" }).apiKey).toBe(
      "sk-or-from-the-env",
    );
  });

  it("leaves an unknown runtime name alone, for the runtime builder to reject", () => {
    // Correcting it here would hide the typo behind a working default.
    expect(loadConfig(missingFile(), { CATALOG_RUNTIME: "openroutr" }).runtime).toBe("openroutr");
  });

  it("still defaults to a model on this machine", () => {
    const config = loadConfig(missingFile(), {});
    expect(config.runtime).toBe("ollama");
    expect(config.runtimeEndpoint).toBe("http://127.0.0.1:11434");
  });
});
