import { classify } from "../classify/index.ts";
import { NotFoundError } from "../errors.ts";
import type { HfClient } from "../hf/client.ts";
import type { RepoSnapshot } from "../hf/types.ts";
import { parseModelRef, type RepoRef } from "../hf/url.ts";

export interface ResolvedRepo {
  primary: RepoSnapshot;
  /** The base model, fetched when the primary repo is a quantisation or an adapter. */
  base: RepoSnapshot | null;
  /** Why the base could not be fetched, when it was wanted but unavailable. */
  baseError: string | null;
}

/**
 * Fetches the repo and, when the repo is only a re-encoding or a delta of
 * another model, the model it actually describes. A GGUF repo's own card is
 * usually three lines of download instructions; the description lives upstream.
 */
export async function resolveRepo(input: string | RepoRef, client: HfClient): Promise<ResolvedRepo> {
  const ref = typeof input === "string" ? parseModelRef(input) : input;

  let primary: RepoSnapshot;
  try {
    primary = await client.snapshot(ref);
  } catch (error) {
    throw await withSuggestions(error, ref.repoId, client);
  }
  const classification = classify(primary);

  const wantsBase = classification.kind === "quantization" || classification.kind === "adapter";
  const baseId = classification.baseModel;
  if (!wantsBase || baseId === null || !baseId.includes("/")) {
    return { primary, base: null, baseError: null };
  }

  try {
    const base = await client.snapshot(parseModelRef(baseId));
    return { primary, base, baseError: null };
  } catch (error: unknown) {
    // A missing or gated base is not fatal: the deterministic numbers from the
    // primary repo, including real file sizes, still stand.
    return {
      primary,
      base: null,
      baseError: `Could not read the base model ${baseId}: ${error instanceof Error ? error.message : String(error)}`,
    };
  }
}

/**
 * An unresolvable model id points at the nearest real names rather than failing
 * flatly. Looking them up is best effort: if the search also fails, the original
 * error stands unchanged.
 */
async function withSuggestions(
  error: unknown,
  repoId: string,
  client: HfClient,
): Promise<unknown> {
  if (!(error instanceof NotFoundError)) return error;
  const candidates = await client.suggest(repoId);
  if (candidates.length === 0) return error;
  return new NotFoundError(
    error.message,
    `Did you mean one of these?\n${candidates.map((id) => `  ${id}`).join("\n")}`,
  );
}
