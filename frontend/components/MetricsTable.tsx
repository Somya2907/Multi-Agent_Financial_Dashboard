"use client";

import type { MetricData } from "@/lib/types";
import { formatNumber, formatLargeNumber, formatPercent } from "@/lib/utils";
import { TrendingDown, TrendingUp, Minus } from "lucide-react";

interface Props {
  metrics: Record<string, MetricData>;
}

type MetricFormat = "ratio" | "currency" | "percent";

interface Row {
  label: string;
  value: number | null | undefined;
  threshold: number | null;
  format: MetricFormat;
  inverse?: boolean;
}

function formatValue(value: number | null | undefined, format: MetricFormat): string {
  if (value === null || value === undefined) return "N/A";
  if (format === "currency") return formatLargeNumber(value);
  if (format === "percent") return formatPercent(value);
  return formatNumber(value);
}

function MetricRow({ row }: { row: Row }) {
  const value = row.value ?? null;

  if (value === null) {
    return (
      <tr className="border-b border-gray-800 last:border-0">
        <td className="py-2 text-sm text-gray-400">{row.label}</td>
        <td className="py-2 text-sm text-gray-600 text-right">N/A</td>
        <td className="py-2 text-right">
          <Minus className="h-3.5 w-3.5 text-gray-700 ml-auto" />
        </td>
      </tr>
    );
  }

  const showTrend = row.threshold !== null;
  const isGood = showTrend
    ? row.inverse
      ? value < (row.threshold as number)
      : value >= (row.threshold as number)
    : true;
  const TrendIcon = isGood ? TrendingUp : TrendingDown;
  const valueClass = showTrend
    ? isGood
      ? "text-green-300"
      : "text-red-300"
    : "text-gray-200";
  const trendClass = isGood ? "text-green-400" : "text-red-400";

  return (
    <tr className="border-b border-gray-800 last:border-0">
      <td className="py-2 text-sm text-gray-300">{row.label}</td>
      <td className={`py-2 text-sm font-mono text-right ${valueClass}`}>
        {formatValue(value, row.format)}
      </td>
      <td className="py-2 text-right">
        {showTrend ? (
          <TrendIcon className={`h-3.5 w-3.5 ml-auto ${trendClass}`} />
        ) : (
          <Minus className="h-3.5 w-3.5 text-gray-700 ml-auto" />
        )}
      </td>
    </tr>
  );
}

function Section({ title, rows }: { title: string; rows: Row[] }) {
  const hasAny = rows.some((r) => r.value !== null && r.value !== undefined);
  if (!hasAny) return null;
  return (
    <div
      className="
        relative rounded-xl border border-gray-700/60
        bg-gradient-to-br from-gray-800 to-gray-900
        p-4
        shadow-[0_6px_20px_rgba(0,0,0,0.4),inset_0_1px_0_rgba(255,255,255,0.04)]
        transition-all duration-150
        hover:-translate-y-0.5
        hover:shadow-[0_10px_28px_rgba(0,0,0,0.55),inset_0_1px_0_rgba(255,255,255,0.06)]
      "
    >
      <h4 className="text-[10px] font-semibold uppercase tracking-widest text-blue-400 mb-2 pb-1.5 border-b border-gray-700/60">
        {title}
      </h4>
      <table className="w-full">
        <tbody>
          {rows.map((row) => (
            <MetricRow key={row.label} row={row} />
          ))}
        </tbody>
      </table>
    </div>
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

        const liquidity: Row[] = [
          { label: "Current Ratio", value: m.current_ratio, threshold: 1.0, format: "ratio" },
          { label: "Debt-to-Equity", value: m.debt_to_equity, threshold: 2.0, format: "ratio", inverse: true },
          { label: "Cash & Equivalents", value: raw.cash_and_equivalents, threshold: null, format: "currency" },
          { label: "Long-Term Debt", value: raw.long_term_debt, threshold: null, format: "currency" },
        ];

        const profitability: Row[] = [
          { label: "Gross Margin", value: m.gross_margin, threshold: 0.35, format: "percent" },
          { label: "Operating Margin", value: m.operating_margin, threshold: 0.15, format: "percent" },
          { label: "Net Margin", value: m.net_margin, threshold: 0.10, format: "percent" },
          { label: "ROE", value: m.roe, threshold: 0.15, format: "percent" },
          { label: "ROA", value: m.roa, threshold: 0.05, format: "percent" },
          { label: "EPS (Diluted)", value: m.eps_diluted, threshold: null, format: "ratio" },
        ];

        const cashflow: Row[] = [
          { label: "Operating Cash Flow", value: raw.operating_cash_flow, threshold: null, format: "currency" },
          { label: "Capex", value: raw.capex, threshold: null, format: "currency" },
          { label: "Free Cash Flow", value: m.free_cash_flow, threshold: 0, format: "currency" },
          { label: "FCF Margin", value: m.fcf_margin, threshold: 0.10, format: "percent" },
        ];

        const balance: Row[] = [
          { label: "Revenue", value: raw.revenue, threshold: null, format: "currency" },
          { label: "Net Income", value: raw.net_income, threshold: null, format: "currency" },
          { label: "Total Assets", value: raw.total_assets, threshold: null, format: "currency" },
          { label: "Total Liabilities", value: raw.total_liabilities, threshold: null, format: "currency" },
          { label: "Shareholder Equity", value: raw.shareholder_equity, threshold: null, format: "currency" },
        ];

        return (
          <div key={ticker} className="mb-5 last:mb-0">
            {tickers.length > 1 && (
              <p className="text-xs font-semibold text-blue-400 mb-2">{ticker}</p>
            )}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <Section title="Liquidity & Capital Structure" rows={liquidity} />
              <Section title="Profitability" rows={profitability} />
              <Section title="Cash Flow" rows={cashflow} />
              <Section title="Income & Balance Sheet" rows={balance} />
            </div>
          </div>
        );
      })}
    </div>
  );
}
