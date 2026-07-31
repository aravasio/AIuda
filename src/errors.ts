/**
 * Exit codes are part of the contract: the tool is meant to be scriptable,
 * so a caller must be able to tell "your machine is too small" apart from
 * "the network is down" without parsing prose.
 */
export const ExitCode = {
  Ok: 0,
  Generic: 1,
  Usage: 2,
  NotFound: 3,
  Network: 4,
  Gated: 5,
  RuntimeUnavailable: 6,
  CardTooLarge: 7,
  InvalidLlmOutput: 8,
} as const;

export type ExitCodeValue = (typeof ExitCode)[keyof typeof ExitCode];

/** An error we chose to raise, with a message meant for a human and a code meant for a script. */
export class CatalogError extends Error {
  readonly exitCode: ExitCodeValue;
  /** The exact command the user should run to fix this, when there is one. */
  readonly fix: string | null;

  constructor(message: string, exitCode: ExitCodeValue, fix: string | null = null) {
    super(message);
    this.name = "CatalogError";
    this.exitCode = exitCode;
    this.fix = fix;
  }
}

export class UsageError extends CatalogError {
  constructor(message: string, fix: string | null = null) {
    super(message, ExitCode.Usage, fix);
    this.name = "UsageError";
  }
}

export class NotFoundError extends CatalogError {
  constructor(message: string, fix: string | null = null) {
    super(message, ExitCode.NotFound, fix);
    this.name = "NotFoundError";
  }
}

export class NetworkError extends CatalogError {
  constructor(message: string, fix: string | null = null) {
    super(message, ExitCode.Network, fix);
    this.name = "NetworkError";
  }
}

export class GatedRepoError extends CatalogError {
  constructor(repoId: string) {
    super(
      `${repoId} is gated. The owner requires you to accept their terms and use an access token before the files can be read.`,
      ExitCode.Gated,
      `Open https://huggingface.co/${repoId} and accept the terms, then create a token at https://huggingface.co/settings/tokens and run:\n  export HF_TOKEN=hf_your_token_here`,
    );
    this.name = "GatedRepoError";
  }
}

/**
 * The LLM runtime is missing or unreachable. This is deliberately fatal:
 * the plain-English explanation is the product, and a run that quietly
 * drops it looks successful while being useless.
 */
export class RuntimeUnavailableError extends CatalogError {
  constructor(message: string, fix: string | null = null) {
    super(message, ExitCode.RuntimeUnavailable, fix);
    this.name = "RuntimeUnavailableError";
  }
}

/**
 * The model card could not be trimmed down to the space available.
 * Raised instead of sending a half-read card, which would produce a
 * confident summary of a different model.
 */
export class CardTooLargeError extends CatalogError {
  constructor(message: string, fix: string | null = null) {
    super(message, ExitCode.CardTooLarge, fix);
    this.name = "CardTooLargeError";
  }
}

/**
 * The runtime stopped writing because it hit the output limit. Distinct from
 * malformed output: the model was writing correctly and was cut off.
 */
export class TruncatedReplyError extends CatalogError {
  constructor(message: string, fix: string | null = null) {
    super(message, ExitCode.InvalidLlmOutput, fix);
    this.name = "TruncatedReplyError";
  }
}

export class InvalidLlmOutputError extends CatalogError {
  constructor(message: string, fix: string | null = null) {
    super(message, ExitCode.InvalidLlmOutput, fix);
    this.name = "InvalidLlmOutputError";
  }
}
