/**
 * Records real Hugging Face responses to disk so the test suite never touches
 * the network. Run by hand when a fixture needs updating:
 *
 *   npm run fixtures:refresh
 *
 * Tests that depend on a live API fail for reasons unrelated to the code and
 * get ignored within a week, so this is deliberately a separate, manual step.
 */
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { HfClient } from "../src/hf/client.ts";
import { parseModelRef } from "../src/hf/url.ts";

/** One repo per case the pipeline has to route differently. */
const FIXTURES: Array<{ slug: string; repoId: string; why: string }> = [
  { slug: "standard-dense", repoId: "Qwen/QwQ-32B", why: "plain dense repo, superseded, own config.json" },
  { slug: "moe-multimodal", repoId: "Qwen/Qwen3.6-35B-A3B", why: "MoE with nested text_config and hybrid attention" },
  { slug: "world-model", repoId: "Qwen/Qwen-AgentWorld-35B-A3B", why: "world model, name says nothing about it" },
  { slug: "audio-utility", repoId: "Qwen/Qwen3-ForcedAligner-0.6B-hf", why: "labelled token-classification on the Hub" },
  { slug: "embedding", repoId: "Qwen/Qwen3-Embedding-0.6B", why: "embedding model" },
  { slug: "third-party-gguf", repoId: "bartowski/Qwen2.5-7B-Instruct-GGUF", why: "third-party quantisation, no config.json, many variants" },
  { slug: "gated", repoId: "meta-llama/Llama-3.1-8B-Instruct", why: "gated repo, files unreadable without a token" },
  { slug: "vision", repoId: "Qwen/Qwen2.5-VL-7B-Instruct", why: "multimodal with top-level text fields" },
  { slug: "adapter", repoId: "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised", why: "LoRA adapter, loaded on top of a base model" },
];

const ROOT = join(import.meta.dirname, "..", "test", "fixtures");

async function main(): Promise<void> {
  const client = new HfClient();

  for (const fixture of FIXTURES) {
    const dir = join(ROOT, fixture.slug);
    mkdirSync(dir, { recursive: true });
    process.stdout.write(`${fixture.repoId} … `);

    const ref = parseModelRef(fixture.repoId);
    try {
      const snapshot = await client.snapshot(ref);
      writeFileSync(
        join(dir, "snapshot.json"),
        `${JSON.stringify({ recordedAt: new Date().toISOString(), why: fixture.why, ...snapshot }, null, 2)}\n`,
      );
      process.stdout.write(`ok (${snapshot.files.length} files, sha ${snapshot.info.sha.slice(0, 8)})\n`);
    } catch (error) {
      // A gated repo cannot be fully read without a token. Record the failure
      // itself, because "this errors in a specific way" is the behaviour under
      // test for that case.
      const message = error instanceof Error ? error.message : String(error);
      writeFileSync(
        join(dir, "error.json"),
        `${JSON.stringify({ recordedAt: new Date().toISOString(), why: fixture.why, repoId: fixture.repoId, error: message, name: error instanceof Error ? error.name : "Error" }, null, 2)}\n`,
      );
      process.stdout.write(`recorded failure: ${message.slice(0, 60)}\n`);
    }
  }
}

await main();
