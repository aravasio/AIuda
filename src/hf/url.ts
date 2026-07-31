import { UsageError } from "../errors.ts";

export interface RepoRef {
  /** Owner or organisation, e.g. "Qwen". */
  owner: string;
  /** Repo name without the owner, e.g. "QwQ-32B". */
  name: string;
  /** "owner/name", the form the Hugging Face API wants. */
  repoId: string;
  /** Branch, tag or commit the user asked for. Null means the default branch. */
  revision: string | null;
}

const HOSTS = new Set(["huggingface.co", "www.huggingface.co", "hf.co", "www.hf.co"]);

/** Path segments that follow the repo id on the website but are not part of it. */
const TRAILING_SECTIONS = new Set([
  "tree",
  "blob",
  "resolve",
  "commit",
  "commits",
  "discussions",
  "settings",
  "raw",
]);

const SEGMENT = /^[A-Za-z0-9._-]+$/;

/**
 * Accepts anything a user is likely to paste: a full model page URL, a URL
 * pointing at a file or a branch inside the repo, a bare "owner/name", or
 * either of those with an explicit revision.
 */
export function parseModelRef(input: string): RepoRef {
  const raw = input.trim();
  if (raw === "") {
    throw new UsageError("No model given.", "catalog query https://huggingface.co/Qwen/QwQ-32B");
  }

  // "owner/name@revision" is a convenient shorthand that no URL form provides.
  let atRevision: string | null = null;
  let body = raw;
  const at = raw.lastIndexOf("@");
  if (at > 0 && !raw.slice(0, at).includes("@") && !raw.includes("://")) {
    atRevision = raw.slice(at + 1);
    body = raw.slice(0, at);
  }

  const segments = extractPathSegments(body);

  if (segments.length < 2) {
    throw new UsageError(
      `"${input}" does not look like a Hugging Face model. Expected a URL or an owner/name pair.`,
      "catalog query https://huggingface.co/Qwen/QwQ-32B",
    );
  }

  const [owner, name, ...rest] = segments;
  if (!owner || !name || !SEGMENT.test(owner) || !SEGMENT.test(name)) {
    throw new UsageError(
      `"${input}" does not look like a Hugging Face model. Expected a URL or an owner/name pair.`,
      "catalog query https://huggingface.co/Qwen/QwQ-32B",
    );
  }

  return {
    owner,
    name,
    repoId: `${owner}/${name}`,
    revision: atRevision ?? revisionFromRest(rest),
  };
}

function extractPathSegments(body: string): string[] {
  let path = body;

  if (/^[a-z][a-z0-9+.-]*:\/\//i.test(path)) {
    let url: URL;
    try {
      url = new URL(path);
    } catch {
      throw new UsageError(`"${body}" is not a URL this tool can read.`);
    }
    if (!HOSTS.has(url.hostname.toLowerCase())) {
      throw new UsageError(
        `${url.hostname} is not Hugging Face. This tool only reads huggingface.co model pages.`,
      );
    }
    path = url.pathname;
  } else if (path.toLowerCase().startsWith("huggingface.co/") || path.toLowerCase().startsWith("hf.co/")) {
    path = path.slice(path.indexOf("/"));
  }

  const segments = path.split("/").filter((s) => s.length > 0);

  // Model pages live at the root; datasets and spaces are prefixed and are not models.
  const first = segments[0]?.toLowerCase();
  if (first === "datasets" || first === "spaces") {
    throw new UsageError(
      `That is a ${first === "datasets" ? "dataset" : "space"}, not a model. This tool reads models only.`,
    );
  }
  if (first === "models") {
    return segments.slice(1);
  }
  return segments;
}

/** "tree/main", "blob/abc123/config.json" and friends carry the revision in slot 1. */
function revisionFromRest(rest: string[]): string | null {
  const section = rest[0]?.toLowerCase();
  if (section === undefined) return null;
  if (!TRAILING_SECTIONS.has(section)) return null;
  const revision = rest[1];
  if (revision === undefined || revision === "") return null;
  return revision;
}
