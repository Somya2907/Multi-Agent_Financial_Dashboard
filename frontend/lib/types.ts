export interface RiskCategory {
  severity: "HIGH" | "MEDIUM" | "LOW";
  evidence: string[];
  score: number;
}

export interface RiskSynthesis {
  market_risk: RiskCategory;
  credit_risk: RiskCategory;
  liquidity_risk: RiskCategory;
  regulatory_risk: RiskCategory;
  macroeconomic_risk: RiskCategory;
  overall_risk_level: "HIGH" | "MEDIUM" | "LOW";
  overall_risk_score: number;
  summary: string;
  error?: string;
}

export interface QualitativeAnalysis {
  overall_tone: string;
  sentiment_score: number;
  key_themes: string[];
  forward_looking: string[];
  risk_themes: string[];
  management_confidence: string;
  error?: string;
}

export interface MetricData {
  ticker: string;
  current_ratio: number | null;
  debt_to_equity: number | null;
  raw: {
    current_assets?: number;
    current_liabilities?: number;
    total_assets?: number;
    total_liabilities?: number;
    shareholder_equity?: number;
  };
}

export interface CritiqueIssue {
  claim: string;
  issue: string;
}

export interface ClaimResult {
  text: string;
  status: "verified" | "flagged";
  reason: string;
}

export interface Critique {
  total_claims: number;
  verified_claims: number;
  flagged_claims: number;
  verification_rate: number;
  issues: CritiqueIssue[];
  claims?: ClaimResult[];
  overall_confidence: "HIGH" | "MEDIUM" | "LOW";
  error?: string;
}

export interface Citation {
  ticker: string;
  source_type: string;
  section_name: string;
  score: number;
}

export interface GraphState {
  query: string;
  query_type: string;
  tickers: string[];
  requires_metrics: boolean;
  requires_qualitative: boolean;
  metrics_only_mode: boolean;
  retrieved_chunks: Record<string, unknown>[];
  context: string;
  citations: Citation[];
  metrics: Record<string, MetricData>;
  qualitative_analysis: QualitativeAnalysis | null;
  risk_synthesis: RiskSynthesis | null;
  critique: Critique | null;
  final_answer: string;
}

export interface Company {
  ticker: string;
  name: string;
  sector: string;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface FollowupResponse {
  answer: string;
  citations: Citation[];
}
