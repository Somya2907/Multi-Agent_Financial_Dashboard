"use client";

import type { RiskSynthesis } from "@/lib/types";
import { severityColor } from "@/lib/utils";
import { ShieldAlert } from "lucide-react";

interface Props {
  riskSynthesis: RiskSynthesis | null;
  tickers: string[];
  queryType: string;
}

export function ExecutiveSummary({ riskSynthesis, tickers, queryType }: Props) {
  if (!riskSynthesis || riskSynthesis.error) {
    const msg = !riskSynthesis
      ? "Submit a query to generate the risk assessment."
      : riskSynthesis.error === "insufficient_data"
      ? "Not enough data to assess risk for this query."
      : `Risk synthesis unavailable: ${riskSynthesis.error}`;
    return (
      <div className="rounded-xl border border-gray-800 bg-gray-900 p-5">
        <h2 className="text-xs font-semibold uppercase tracking-widest text-gray-500 mb-3">
          Executive Summary
        </h2>
        <p className="text-sm text-gray-500">{msg}</p>
      </div>
    );
  }

  const colors = severityColor(riskSynthesis.overall_risk_level);
  const score = riskSynthesis.overall_risk_score;

  return (
    <div className="rounded-xl border border-gray-800 bg-gray-900 p-5">
      <div className="flex items-start justify-between mb-3">
        <h2 className="text-xs font-semibold uppercase tracking-widest text-gray-500">
          Executive Summary
        </h2>
        <span
          className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-semibold ${colors.badge}`}
        >
          <ShieldAlert className="h-3 w-3" />
          {riskSynthesis.overall_risk_level} RISK
        </span>
      </div>

      {tickers.length > 0 && (
        <p className="text-xs text-gray-500 mb-2">
          {queryType === "comparison" ? "Comparing:" : "Analysis for:"}{" "}
          <span className="text-gray-300 font-medium">{tickers.join(", ")}</span>
        </p>
      )}

      <p className="text-sm text-gray-200 leading-relaxed mb-4">
        {riskSynthesis.summary}
      </p>

      {/* Risk score bar */}
      <div className="space-y-1.5">
        <div className="flex justify-between text-xs text-gray-500">
          <span>Overall Risk Score</span>
          <span className={colors.text}>{(score * 100).toFixed(0)} / 100</span>
        </div>
        <div className="h-1.5 w-full rounded-full bg-gray-800">
          <div
            className={`h-1.5 rounded-full transition-all duration-700 ${
              score >= 0.7
                ? "bg-red-500"
                : score >= 0.45
                ? "bg-yellow-400"
                : "bg-green-500"
            }`}
            style={{ width: `${Math.round(score * 100)}%` }}
          />
        </div>
      </div>
    </div>
  );
}
