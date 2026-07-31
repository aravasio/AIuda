import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { CACHE_VERSION, RevisionCache } from "../src/cache/index.ts";

let dir: string;
let cache: RevisionCache;

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "catalog-cache-"));
  cache = new RevisionCache(dir);
});

afterEach(() => {
  rmSync(dir, { recursive: true, force: true });
});

const SHA_A = "976055f8c83f394f35dbd3ab09a285a984907bd0";
const SHA_B = "995ad96eacd98c81ed38be0c5b274b04031597b0";

describe("cache keyed by revision", () => {
  it("returns what was stored for the same SHA", () => {
    cache.set(SHA_A, "Qwen/QwQ-32B", { one_liner: "first" });
    expect(cache.get(SHA_A)).toEqual({ one_liner: "first" });
  });

  it("misses on a different SHA, so an edited card reprocesses", () => {
    cache.set(SHA_A, "Qwen/QwQ-32B", { one_liner: "first" });
    // Same repo, new commit: the vendor edited the card.
    expect(cache.get(SHA_B)).toBeNull();
  });

  it("hits for the same SHA regardless of the repo name recorded with it", () => {
    // The name is stored for display only. It is never part of the key.
    cache.set(SHA_A, "Qwen/QwQ-32B", { one_liner: "shared" });
    expect(cache.get(SHA_A)).toEqual({ one_liner: "shared" });
    cache.set(SHA_A, "someone-else/mirror-of-QwQ-32B", { one_liner: "rewritten" });
    expect(cache.get(SHA_A)).toEqual({ one_liner: "rewritten" });
    expect(cache.list()).toHaveLength(1);
  });

  it("keeps two revisions of the same repo side by side", () => {
    cache.set(SHA_A, "Qwen/QwQ-32B", { one_liner: "old" });
    cache.set(SHA_B, "Qwen/QwQ-32B", { one_liner: "new" });
    expect(cache.get(SHA_A)).toEqual({ one_liner: "old" });
    expect(cache.get(SHA_B)).toEqual({ one_liner: "new" });
    expect(cache.list()).toHaveLength(2);
  });

  it("misses when the stored shape predates the current code", () => {
    writeFileSync(
      join(dir, `${SHA_A}.json`),
      JSON.stringify({
        version: CACHE_VERSION - 1,
        sha: SHA_A,
        repoId: "Qwen/QwQ-32B",
        storedAt: new Date().toISOString(),
        payload: { one_liner: "stale shape" },
      }),
    );
    expect(cache.get(SHA_A)).toBeNull();
  });

  it("misses rather than throwing on a corrupted entry", () => {
    writeFileSync(join(dir, `${SHA_A}.json`), "{ not json");
    expect(cache.get(SHA_A)).toBeNull();
    expect(cache.list()).toEqual([]);
  });

  it("refuses to let a SHA from the network escape its directory", () => {
    const hostile = "../../etc/passwd";
    cache.set(hostile, "attacker/repo", { one_liner: "nope" });
    // Sanitised to a flat name, so nothing was written outside the cache dir.
    expect(cache.list()).toHaveLength(1);
    expect(cache.list()[0]?.repoId).toBe("attacker/repo");
  });

  it("lists what it holds and clears it on request", () => {
    cache.set(SHA_A, "Qwen/QwQ-32B", { one_liner: "a" });
    cache.set(SHA_B, "Qwen/Qwen3.6-35B-A3B", { one_liner: "b" });

    const listed = cache.list();
    expect(listed.map((e) => e.repoId).sort()).toEqual(["Qwen/QwQ-32B", "Qwen/Qwen3.6-35B-A3B"]);
    expect(listed.every((e) => e.bytes > 0)).toBe(true);

    expect(cache.clear()).toBe(2);
    expect(cache.list()).toEqual([]);
    expect(cache.get(SHA_A)).toBeNull();
  });

  it("reports an empty listing for a directory that does not exist yet", () => {
    const missing = new RevisionCache(join(dir, "not-created"));
    expect(missing.list()).toEqual([]);
    expect(missing.clear()).toBe(0);
  });
});
