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

  // Liquidity & capital structure
  current_ratio: number | null;
  debt_to_equity: number | null;

  // Profitability
  gross_margin: number | null;
  operating_margin: number | null;
  net_margin: number | null;
  roe: number | null;
  roa: number | null;

  // Cash flow
  free_cash_flow: number | null;
  fcf_margin: number | null;

  // Per-share
  eps_diluted: number | null;

  raw: {
    // Balance sheet
    current_assets?: number | null;
    current_liabilities?: number | null;
    total_assets?: number | null;
    total_liabilities?: number | null;
    shareholder_equity?: number | null;
    cash_and_equivalents?: number | null;
    long_term_debt?: number | null;
    // Income statement
    revenue?: number | null;
    cost_of_revenue?: number | null;
    gross_profit?: number | null;
    operating_income?: number | null;
    net_income?: number | null;
    eps_diluted?: number | null;
    // Cash flow
    operating_cash_flow?: number | null;
    capex?: number | null;
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
