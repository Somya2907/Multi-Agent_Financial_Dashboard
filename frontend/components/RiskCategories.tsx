"use client";

import { useState } from "react";
import type { RiskSynthesis, RiskCategory } from "@/lib/types";
import { severityColor } from "@/lib/utils";
import { ChevronDown, ChevronRight } from "lucide-react";

const CATEGORY_LABELS: Record<string, string> = {
  market_risk: "Market Risk",
  credit_risk: "Credit Risk",
  liquidity_risk: "Liquidity Risk",
  regulatory_risk: "Regulatory Risk",
  macroeconomic_risk: "Macro Risk",
};

function RiskCard({
  label,
  category,
}: {
  label: string;
  category: RiskCategory;
}) {
  const [open, setOpen] = useState(false);
  const colors = severityColor(category.severity);

  return (
    <div className="rounded-lg border border-gray-800 bg-gray-850 overflow-hidden">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between px-4 py-3 hover:bg-gray-800 transition-colors"
      >
        <div className="flex items-center gap-3">
          <div className={`h-2 w-2 rounded-full ${colors.dot}`} />
          <span className="text-sm font-medium text-gray-200">{label}</span>
        </div>
        <div className="flex items-center gap-2">
          <span
            className={`rounded-full border px-2.5 py-0.5 text-xs font-semibold ${colors.badge}`}
          >
            {category.severity}
          </span>
          <span className="text-xs text-gray-600 font-mono">
            {(category.score * 100).toFixed(0)}
          </span>
          {open ? (
            <ChevronDown className="h-4 w-4 text-gray-500" />
          ) : (
            <ChevronRight className="h-4 w-4 text-gray-500" />
          )}
        </div>
      </button>

      {open && category.evidence?.length > 0 && (
        <div className="border-t border-gray-800 px-4 py-3">
          <ul className="space-y-1.5">
            {category.evidence.map((ev, i) => (
              <li key={i} className="flex items-start gap-2 text-xs text-gray-400">
                <span className="mt-1 h-1 w-1 shrink-0 rounded-full bg-gray-600" />
                {ev}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

interface Props {
  riskSynthesis: RiskSynthesis | null;
}

export function RiskCategories({ riskSynthesis }: Props) {
  if (!riskSynthesis || riskSynthesis.error) {
    const msg = !riskSynthesis
      ? "Submit a query to see risk categorization."
      : riskSynthesis.error === "insufficient_data"
      ? "Not enough data to categorize risks for this query."
      : `Risk categorization unavailable: ${riskSynthesis.error}`;
    return (
      <div className="rounded-xl border border-gray-800 bg-gray-900 p-5">
        <h2 className="text-xs font-semibold uppercase tracking-widest text-gray-500 mb-3">
          Risk Categorization
        </h2>
        <p className="text-sm text-gray-500">{msg}</p>
      </div>
    );
  }

  const categories = [
    "market_risk",
    "credit_risk",
    "liquidity_risk",
    "regulatory_risk",
    "macroeconomic_risk",
  ] as const;

  return (
    <div className="rounded-xl border border-gray-800 bg-gray-900 p-5">
      <h2 className="text-xs font-semibold uppercase tracking-widest text-gray-500 mb-4">
        Risk Categorization
      </h2>
      <div className="space-y-2">
        {categories.map((cat) => {
          const data = riskSynthesis[cat];
          if (!data) return null;
          return (
            <RiskCard
              key={cat}
              label={CATEGORY_LABELS[cat]}
              category={data}
            />
          );
        })}
      </div>
    </div>
  );
}
