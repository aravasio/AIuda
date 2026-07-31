import { describe, expect, it } from "vitest";
import { CardTooLargeError } from "../src/errors.ts";
import {
  estimateTokens,
  parseSections,
  Priority,
  looksLikeScore,
  stripCodeBlocks,
  stripTables,
  trimCard,
} from "../src/llm/trim.ts";
import { loadSnapshot } from "./helpers/fixtures.ts";

/** The real cards, which is the only content that exercises the hard cases. */
const CARDS = (["standard-dense", "moe-multimodal", "world-model", "audio-utility", "embedding"] as const).map(
  (slug) => ({ slug, card: loadSnapshot(slug).card ?? "" }),
);

const SYNTHETIC = `---
license: apache-2.0
---

# Test-Model-7B

Test-Model-7B turns a question into an answer.

## Overview

It reads text and writes text back. It was built for answering questions
about documents.

## Quickstart

Install it first.

\`\`\`python
from transformers import AutoModel
model = AutoModel.from_pretrained("test/Test-Model-7B")
# a long code block that is mostly noise
\`\`\`

## Deployment

Run it with vLLM.

\`\`\`bash
vllm serve test/Test-Model-7B --max-model-len 32768
\`\`\`

## Evaluation

| Benchmark | Test-Model-7B | Rival-7B |
|---|---|---|
| MMLU | 74.2 | 71.0 |
| GSM8K | 82.1 | 79.5 |

## Some other detail

Extra prose that is neither the introduction nor pure instructions.

## Citation

\`\`\`bibtex
@article{test2026}
\`\`\`
`;

describe("reading a card's structure", () => {
  it("drops the YAML front matter, which is already parsed structurally", () => {
    const sections = parseSections(SYNTHETIC);
    expect(sections.map((s) => s.body).join(" ")).not.toContain("license: apache-2.0");
  });

  it("scores the introduction and overview as essential", () => {
    const sections = parseSections(SYNTHETIC);
    const overview = sections.find((s) => s.heading === "Overview");
    expect(overview?.priority).toBe(Priority.Essential);
  });

  it("scores instructions as the first thing to go", () => {
    const sections = parseSections(SYNTHETIC);
    for (const heading of ["Quickstart", "Deployment", "Citation"]) {
      expect(sections.find((s) => s.heading === heading)?.priority).toBe(Priority.Noise);
    }
  });

  it("does not mistake a heading inside a code block for structure", () => {
    const card = "# Real\n\ntext\n\n```\n# not a heading\n```\n";
    expect(parseSections(card).map((s) => s.heading)).toEqual(["Real"]);
  });
});

describe("stripping", () => {
  it("removes fenced code blocks", () => {
    const stripped = stripCodeBlocks(SYNTHETIC);
    expect(stripped).not.toContain("AutoModel.from_pretrained");
    expect(stripped).not.toContain("vllm serve");
    expect(stripped).toContain("turns a question into an answer");
  });

  it("removes an unterminated code block rather than leaving it open", () => {
    const stripped = stripCodeBlocks("intro\n\n```python\nnever closed\nmore\n");
    expect(stripped).not.toContain("never closed");
    expect(stripped).toContain("intro");
  });

  it("removes tables for the prose pass", () => {
    const stripped = stripTables(SYNTHETIC);
    expect(stripped).not.toContain("| MMLU |");
    expect(stripped).toContain("Overview");
  });
});

describe("trimming, as properties", () => {
  const budgets = [200, 500, 1000, 4000];

  it.each(budgets)("never exceeds a budget of %i tokens", (budget) => {
    const result = trimCard(SYNTHETIC, { budgetTokens: budget, pass: "prose" });
    expect(result.tokens).toBeLessThanOrEqual(budget);
    expect(estimateTokens(result.text)).toBeLessThanOrEqual(budget);
  });

  it.each(budgets)("is never empty at a budget of %i tokens", (budget) => {
    const result = trimCard(SYNTHETIC, { budgetTokens: budget, pass: "prose" });
    expect(result.text.trim()).not.toBe("");
  });

  it.each(budgets)("keeps the introduction at a budget of %i tokens", (budget) => {
    const result = trimCard(SYNTHETIC, { budgetTokens: budget, pass: "prose" });
    expect(result.text).toContain("turns a question into an answer");
  });

  it("drops code before it drops prose", () => {
    // A budget that fits the prose but not the prose plus its code blocks.
    const result = trimCard(SYNTHETIC, { budgetTokens: 200, pass: "prose" });
    expect(result.text).not.toContain("AutoModel");
    expect(result.text).toContain("It reads text and writes text back");
  });

  it("drops instructions before it drops ordinary prose", () => {
    const tight = trimCard(SYNTHETIC, { budgetTokens: 60, pass: "prose" });
    expect(tight.text).not.toContain("Run it with vLLM");
    expect(tight.text).toContain("turns a question into an answer");
  });

  it("gives the same output for the same input and budget, every time", () => {
    for (const budget of budgets) {
      const first = trimCard(SYNTHETIC, { budgetTokens: budget, pass: "prose" });
      const second = trimCard(SYNTHETIC, { budgetTokens: budget, pass: "prose" });
      expect(first.text).toBe(second.text);
      expect(first.dropped).toEqual(second.dropped);
    }
  });

  it("errors rather than returning a card cut down to nothing", () => {
    // Ten tokens cannot hold any real introduction.
    expect(() => trimCard(SYNTHETIC, { budgetTokens: 10, pass: "prose" })).toThrow(CardTooLargeError);
  });

  it("says what it could not fit, and names the setting that would fix it", () => {
    try {
      trimCard(SYNTHETIC, { budgetTokens: 10, pass: "prose" });
      expect.unreachable("should have thrown");
    } catch (error) {
      expect(error).toBeInstanceOf(CardTooLargeError);
      const message = (error as CardTooLargeError).message;
      // The point of failing is that the user learns nothing was produced.
      expect(message).toContain("No summary was produced");
      expect((error as CardTooLargeError).fix).toContain("llmContextTokens");
    }
  });

  it("reports which sections it left out", () => {
    const result = trimCard(SYNTHETIC, { budgetTokens: 60, pass: "prose" });
    expect(result.dropped.length).toBeGreaterThan(0);
  });
});

describe("trimming real cards", () => {
  // A realistic budget: a 16k window less the reply allowance and the prompt
  // scaffolding leaves roughly this much for the card.
  const REALISTIC_BUDGETS = [1500, 3500, 15000];

  it.each(CARDS)("keeps $slug within budget and non-empty", ({ card }) => {
    if (card === "") return;
    for (const budget of REALISTIC_BUDGETS) {
      const result = trimCard(card, { budgetTokens: budget, pass: "prose" });
      expect(result.tokens).toBeLessThanOrEqual(budget);
      expect(result.text.trim().length).toBeGreaterThan(0);
    }
  });

  it.each(CARDS)("refuses $slug rather than half-reading it at an impossible budget", ({ card }) => {
    if (card === "") return;
    // Whatever a card's introduction weighs, 50 tokens cannot hold it. The
    // required behaviour is an explicit failure, never a truncated card.
    expect(() => trimCard(card, { budgetTokens: 50, pass: "prose" })).toThrow(CardTooLargeError);
  });

  it("removes every code fence from every real card", () => {
    for (const { card } of CARDS) {
      if (card === "") continue;
      const result = trimCard(card, { budgetTokens: 3500, pass: "prose" });
      expect(result.text).not.toContain("```");
    }
  });

  it("keeps the opening line of the longest real card", () => {
    const longest = CARDS.reduce((a, b) => (b.card.length > a.card.length ? b : a));
    const result = trimCard(longest.card, { budgetTokens: 3500, pass: "prose" });
    expect(result.text.trim().length).toBeGreaterThan(50);
  });
});

describe("the two passes get different content", () => {
  it("gives the prose pass no tables", () => {
    const result = trimCard(SYNTHETIC, { budgetTokens: 4000, pass: "prose" });
    expect(result.text).not.toContain("| MMLU |");
  });

  it("gives the benchmark pass only the tables", () => {
    const result = trimCard(SYNTHETIC, { budgetTokens: 4000, pass: "benchmarks" });
    expect(result.text).toContain("MMLU");
    expect(result.text).not.toContain("It reads text and writes text back");
  });

  it("errors on the benchmark pass when a card has no tables", () => {
    const noTables = "# Model\n\nIt does a thing.\n\n## Overview\n\nMore about the thing.\n";
    expect(() => trimCard(noTables, { budgetTokens: 4000, pass: "benchmarks" })).toThrow(
      CardTooLargeError,
    );
  });

  it("leaves the benchmark pass a smaller job than the whole card", () => {
    const prose = trimCard(SYNTHETIC, { budgetTokens: 4000, pass: "prose" });
    const benchmarks = trimCard(SYNTHETIC, { budgetTokens: 4000, pass: "benchmarks" });
    expect(benchmarks.tokens + prose.tokens).toBeLessThan(estimateTokens(SYNTHETIC));
  });
});

describe("telling a results table from every other kind", () => {
  const RESULTS = `## Evaluation

| Benchmark | Ours | Rival |
|---|---|---|
| MMLU | 74.2 | 71.0 |
| GSM8K | 82.1 | 79.5 |
`;

  const FEATURES = `## Available Checkpoints

| Model | Languages | Mode |
|---|---|---|
| Qwen3-ASR-1.7B | Chinese, English, Cantonese | Offline / Streaming |
| Qwen3-ForcedAligner-0.6B | Chinese, English | Speech |
`;

  it("keeps a table of scores", () => {
    const result = trimCard(RESULTS, { budgetTokens: 4000, pass: "benchmarks" });
    expect(result.text).toContain("MMLU");
  });

  it("rejects a feature table, which is where fake benchmarks come from", () => {
    // This is the real Qwen3-ForcedAligner table that produced eight
    // "benchmarks" whose scores were language names.
    expect(() => trimCard(FEATURES, { budgetTokens: 4000, pass: "benchmarks" })).toThrow(
      CardTooLargeError,
    );
  });

  it("keeps the scores when both kinds of table are present", () => {
    const result = trimCard(`${FEATURES}\n${RESULTS}`, { budgetTokens: 4000, pass: "benchmarks" });
    expect(result.text).toContain("MMLU");
    expect(result.text).not.toContain("Offline / Streaming");
  });

  it.each(["74.2", "74.2%", "62.3 / 71.0", "0.87", "41.6±0.3", "1,234"])(
    "accepts %s as a score",
    (score) => {
      expect(looksLikeScore(score)).toBe(true);
    },
  );

  it.each(["Offline / Streaming", "Speech", "Chinese, English", "NAR", "—", ""])(
    "rejects %s as a score",
    (score) => {
      expect(looksLikeScore(score)).toBe(false);
    },
  );
});
