export interface Benchmarks {
  sweBenchVerified?: number;
  liveCodeBench?: number;
  terminalBench?: number;
  mmluPro?: number;
}

export type ModelCategory = "coding" | "general" | "reasoning" | "light";

export type ModelTier = "light" | "medium" | "heavy";

export interface RegistryEntry {
  repoId: string;
  name: string;
  hfUrl: string;
  sizeGb: number;
  paramsTotal: string;
  paramsActive: string | null;
  context: string;
  architecture: "dense" | "moe";
  category: ModelCategory;
  tags: string[];
  benchmarks: Benchmarks;
  note: string;
}

export interface ScoredEntry extends RegistryEntry {
  fitBytes: number;
  fits: boolean;
  verdictKind: string;
  rank: number;
}

export interface TeamRecommendation {
  daily: ScoredEntry | null;
  heavy: ScoredEntry | null;
  context: ScoredEntry | null;
  overflow: ScoredEntry[];
}