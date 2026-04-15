import type { GraphState, Company, FollowupResponse } from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function unwrap<T>(res: Response, fallbackMsg: string): Promise<T> {
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? fallbackMsg);
  }
  return res.json();
}

export async function submitQuery(query: string, ticker?: string): Promise<GraphState> {
  const res = await fetch(`${API_BASE}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, ticker: ticker || null }),
  });
  return unwrap<GraphState>(res, "Query failed");
}

export async function analyzeCompany(ticker: string, refresh = false): Promise<GraphState> {
  const res = await fetch(`${API_BASE}/analyze_company`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ticker, refresh }),
  });
  return unwrap<GraphState>(res, "Analysis failed");
}

export async function askFollowup(query: string, priorState: GraphState): Promise<FollowupResponse> {
  const res = await fetch(`${API_BASE}/followup`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, prior_state: priorState }),
  });
  return unwrap<FollowupResponse>(res, "Follow-up failed");
}

export async function fetchCompanies(): Promise<Company[]> {
  const res = await fetch(`${API_BASE}/companies`);
  return unwrap<Company[]>(res, "Failed to load companies");
}
