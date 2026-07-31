import { existsSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

/**
 * Bump when the cached shape changes, so old entries are ignored rather than
 * rendered through code that no longer understands them.
 */
export const CACHE_VERSION = 1;

export interface CacheEntry<T> {
  version: number;
  /** Revision SHA. This is the key, and it is also stored for `cache list`. */
  sha: string;
  /** Recorded for display only. It is never part of the key. */
  repoId: string;
  storedAt: string;
  payload: T;
}

export interface CacheListing {
  sha: string;
  repoId: string;
  storedAt: string;
  bytes: number;
}

export function defaultCacheDir(): string {
  const base = process.env["XDG_CACHE_HOME"] ?? join(homedir(), ".cache");
  return join(base, "catalog", "revisions");
}

/**
 * Content cache keyed by the repo's revision SHA.
 *
 * Keying on the SHA rather than the repo name is what makes invalidation free:
 * vendors edit model cards in place, and a name-keyed cache would pin the first
 * summary forever. A new commit is a new key, so an edited card reprocesses on
 * its own with no staleness check anywhere.
 */
export class RevisionCache {
  private readonly dir: string;

  constructor(dir: string = defaultCacheDir()) {
    this.dir = dir;
  }

  get<T>(sha: string): T | null {
    const path = this.pathFor(sha);
    if (!existsSync(path)) return null;
    try {
      const entry = JSON.parse(readFileSync(path, "utf8")) as CacheEntry<T>;
      if (entry.version !== CACHE_VERSION) return null;
      if (entry.sha !== sha) return null;
      return entry.payload;
    } catch {
      return null;
    }
  }

  set<T>(sha: string, repoId: string, payload: T): void {
    mkdirSync(this.dir, { recursive: true });
    const entry: CacheEntry<T> = {
      version: CACHE_VERSION,
      sha,
      repoId,
      storedAt: new Date().toISOString(),
      payload,
    };
    writeFileSync(this.pathFor(sha), JSON.stringify(entry, null, 2), "utf8");
  }

  list(): CacheListing[] {
    if (!existsSync(this.dir)) return [];
    const listings: CacheListing[] = [];
    for (const file of readdirSync(this.dir)) {
      if (!file.endsWith(".json")) continue;
      try {
        const raw = readFileSync(join(this.dir, file), "utf8");
        const entry = JSON.parse(raw) as CacheEntry<unknown>;
        listings.push({
          sha: entry.sha,
          repoId: entry.repoId,
          storedAt: entry.storedAt,
          bytes: Buffer.byteLength(raw),
        });
      } catch {
        // A corrupted entry is skipped rather than crashing the listing.
      }
    }
    return listings.sort((a, b) => b.storedAt.localeCompare(a.storedAt));
  }

  /** Removes every entry and reports how many were removed. */
  clear(): number {
    if (!existsSync(this.dir)) return 0;
    const files = readdirSync(this.dir).filter((f) => f.endsWith(".json"));
    for (const file of files) rmSync(join(this.dir, file), { force: true });
    return files.length;
  }

  private pathFor(sha: string): string {
    // A SHA is hex, but the value arrives from the network, so it is sanitised
    // before it is ever used as a path.
    const safe = sha.replace(/[^a-zA-Z0-9]/g, "");
    return join(this.dir, `${safe}.json`);
  }
}
