"use client";

import type { MetricData } from "@/lib/types";
import { formatNumber, formatLargeNumber } from "@/lib/utils";
import { TrendingDown, TrendingUp, Minus } from "lucide-react";

interface Props {
  metrics: Record<string, MetricData>;
}

function RatioRow({
  label,
  value,
  threshold,
  inverse = false,
  format = "ratio",
}: {
  label: string;
  value: number | null;
  threshold: number;
  inverse?: boolean;
  format?: "ratio" | "currency";
}) {
  if (value === null)
    return (
      <tr className="border-b border-gray-800">
        <td className="py-2.5 text-sm text-gray-400">{label}</td>
        <td className="py-2.5 text-sm text-gray-600 text-right">N/A</td>
        <td className="py-2.5 text-right">
          <Minus className="h-3.5 w-3.5 text-gray-700 ml-auto" />
        </td>
      </tr>
    );

  const isGood = inverse ? value < threshold : value >= threshold;
  const TrendIcon = isGood ? TrendingUp : TrendingDown;
  const trendColor = isGood ? "text-green-400" : "text-red-400";

  return (
    <tr className="border-b border-gray-800 last:border-0">
      <td className="py-2.5 text-sm text-gray-300">{label}</td>
      <td className={`py-2.5 text-sm font-mono text-right ${isGood ? "text-green-300" : "text-red-300"}`}>
        {format === "currency" ? formatLargeNumber(value) : formatNumber(value)}
      </td>
      <td className="py-2.5 text-right">
        <TrendIcon className={`h-3.5 w-3.5 ml-auto ${trendColor}`} />
      </td>
    </tr>
  );
}

export function MetricsTable({ metrics }: Props) {
  const tickers = Object.keys(metrics);

  if (tickers.length === 0) {
    return (
      <div className="rounded-xl border border-gray-800 bg-gray-900 p-5">
        <h2 className="text-xs font-semibold uppercase tracking-widest text-gray-500 mb-3">
          Key Financial Metrics
        </h2>
        <p className="text-sm text-gray-500">No metrics available for this query.</p>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-gray-800 bg-gray-900 p-5">
      <h2 className="text-xs font-semibold uppercase tracking-widest text-gray-500 mb-4">
        Key Financial Metrics
      </h2>

      {tickers.map((ticker) => {
        const m = metrics[ticker];
        const raw = m.raw ?? {};
        return (
          <div key={ticker} className="mb-5 last:mb-0">
            {tickers.length > 1 && (
              <p className="text-xs font-semibold text-blue-400 mb-2">{ticker}</p>
            )}
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-800">
                  <th className="pb-2 text-left text-xs text-gray-600 font-medium">Metric</th>
                  <th className="pb-2 text-right text-xs text-gray-600 font-medium">Value</th>
                  <th className="pb-2 text-right text-xs text-gray-600 font-medium">Trend</th>
                </tr>
              </thead>
              <tbody>
                <RatioRow label="Current Ratio" value={m.current_ratio} threshold={1.0} />
                <RatioRow
                  label="Debt-to-Equity"
                  value={m.debt_to_equity}
                  threshold={2.0}
                  inverse
                />
                {raw.current_assets !== undefined && (
                  <RatioRow
                    label="Current Assets"
                    value={raw.current_assets}
                    threshold={0}
                    format="currency"
                  />
                )}
                {raw.current_liabilities !== undefined && (
                  <RatioRow
                    label="Current Liabilities"
                    value={raw.current_liabilities}
                    threshold={0}
                    inverse
                    format="currency"
                  />
                )}
                {raw.total_assets !== undefined && (
                  <RatioRow
                    label="Total Assets"
                    value={raw.total_assets}
                    threshold={0}
                    format="currency"
                  />
                )}
                {raw.shareholder_equity !== undefined && (
                  <RatioRow
                    label="Shareholder Equity"
                    value={raw.shareholder_equity}
                    threshold={0}
                    format="currency"
                  />
                )}
              </tbody>
            </table>
          </div>
        );
      })}
    </div>
  );
}
