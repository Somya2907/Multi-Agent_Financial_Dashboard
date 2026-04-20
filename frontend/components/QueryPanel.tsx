"use client";

import { useState, useEffect } from "react";
import type { Company } from "@/lib/types";
import { fetchCompanies } from "@/lib/api";
import { ChevronDown, RefreshCw, GitCompare, X } from "lucide-react";

interface Props {
  onAnalyzeCompany: (ticker: string, refresh?: boolean) => void;
  onCompare: (tickerA: string, tickerB: string) => void;
  selectedTicker: string;
  compareTickers: [string, string] | null;
  isLoading: boolean;
  onExitCompare: () => void;
}

function CompanyDropdown({
  companies,
  selected,
  disabled,
  placeholder,
  excludeTicker,
  onPick,
}: {
  companies: Company[];
  selected: string;
  disabled: boolean;
  placeholder: string;
  excludeTicker?: string;
  onPick: (ticker: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const selectedCompany = companies.find((c) => c.ticker === selected);

  return (
    <div className="relative flex-1">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        disabled={disabled}
        className="flex h-11 w-full items-center gap-2 rounded-lg border border-gray-700 bg-gray-800 px-4 text-sm text-gray-200 hover:border-blue-600 transition-colors disabled:opacity-60"
      >
        <span className="flex-1 text-left truncate">
          {selectedCompany ? (
            <span>
              <span className="font-semibold text-blue-400">{selectedCompany.ticker}</span>
              <span className="text-gray-400 ml-2">{selectedCompany.name}</span>
              <span className="text-gray-600 ml-2 text-xs">· {selectedCompany.sector}</span>
            </span>
          ) : (
            <span className="text-gray-500">{placeholder}</span>
          )}
        </span>
        <ChevronDown className="h-4 w-4 text-gray-500 shrink-0" />
      </button>

      {open && (
        <div className="absolute top-full left-0 mt-1 z-50 w-full rounded-lg border border-gray-700 bg-gray-800 shadow-xl overflow-hidden">
          <div className="max-h-72 overflow-y-auto">
            {companies
              .filter((c) => c.ticker !== excludeTicker)
              .map((c) => (
                <button
                  key={c.ticker}
                  type="button"
                  onClick={() => {
                    setOpen(false);
                    onPick(c.ticker);
                  }}
                  className="w-full flex items-center justify-between px-4 py-2.5 text-left text-sm hover:bg-gray-700 transition-colors"
                >
                  <span>
                    <span className="font-semibold text-blue-400">{c.ticker}</span>
                    <span className="ml-2 text-gray-300">{c.name}</span>
                  </span>
                  <span className="text-xs text-gray-600 ml-2">{c.sector}</span>
                </button>
              ))}
          </div>
        </div>
      )}
    </div>
  );
}

export function QueryPanel({
  onAnalyzeCompany,
  onCompare,
  selectedTicker,
  compareTickers,
  isLoading,
  onExitCompare,
}: Props) {
  const [companies, setCompanies] = useState<Company[]>([]);
  const [compareMode, setCompareMode] = useState<boolean>(false);
  const [tickerA, setTickerA] = useState<string>("");
  const [tickerB, setTickerB] = useState<string>("");

  useEffect(() => {
    fetchCompanies()
      .then(setCompanies)
      .catch(() => {});
  }, []);

  // Sync external compareTickers back into local state
  useEffect(() => {
    if (compareTickers) {
      setCompareMode(true);
      setTickerA(compareTickers[0]);
      setTickerB(compareTickers[1]);
    }
  }, [compareTickers]);

  const enterCompareMode = () => {
    setCompareMode(true);
    setTickerA(selectedTicker || "");
    setTickerB("");
  };

  const leaveCompareMode = () => {
    setCompareMode(false);
    setTickerA("");
    setTickerB("");
    onExitCompare();
  };

  const canRunCompare =
    !!tickerA && !!tickerB && tickerA !== tickerB && !isLoading;

  const runCompare = () => {
    if (!canRunCompare) return;
    onCompare(tickerA, tickerB);
  };

  // ── Compare mode UI ──
  if (compareMode) {
    return (
      <div className="rounded-xl border border-gray-800 bg-gray-900 p-5">
        <div className="flex items-center gap-3">
          <GitCompare className="h-4 w-4 text-blue-400 shrink-0" />
          <span className="text-xs font-semibold uppercase tracking-widest text-blue-400 shrink-0">
            Compare
          </span>
          <CompanyDropdown
            companies={companies}
            selected={tickerA}
            disabled={isLoading}
            placeholder="Company A…"
            excludeTicker={tickerB}
            onPick={setTickerA}
          />
          <span className="text-xs text-gray-600 shrink-0">vs</span>
          <CompanyDropdown
            companies={companies}
            selected={tickerB}
            disabled={isLoading}
            placeholder="Company B…"
            excludeTicker={tickerA}
            onPick={setTickerB}
          />
          <button
            type="button"
            onClick={runCompare}
            disabled={!canRunCompare}
            className="h-11 rounded-lg bg-blue-600 px-5 text-sm font-semibold text-white hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed transition-colors shrink-0"
          >
            {isLoading ? "Running…" : "Analyze"}
          </button>
          <button
            type="button"
            onClick={leaveCompareMode}
            title="Exit compare mode"
            className="h-11 w-11 rounded-lg border border-gray-800 bg-gray-850 text-gray-500 hover:text-gray-200 hover:border-gray-700 transition-colors flex items-center justify-center shrink-0"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
      </div>
    );
  }

  // ── Single-company mode UI ──
  const selectedCompany = companies.find((c) => c.ticker === selectedTicker);

  return (
    <div className="rounded-xl border border-gray-800 bg-gray-900 p-5">
      <div className="flex items-center gap-3">
        <CompanyDropdown
          companies={companies}
          selected={selectedTicker}
          disabled={isLoading}
          placeholder="Select a company to analyze…"
          onPick={(t) => onAnalyzeCompany(t, false)}
        />

        {selectedTicker && (
          <button
            type="button"
            onClick={() => onAnalyzeCompany(selectedTicker, true)}
            disabled={isLoading}
            title="Regenerate analysis (bypass cache)"
            className="h-11 rounded-lg border border-gray-700 bg-gray-800 px-3 text-gray-400 hover:border-gray-600 hover:text-gray-200 disabled:opacity-40 transition-colors"
          >
            <RefreshCw className={`h-4 w-4 ${isLoading ? "animate-spin" : ""}`} />
          </button>
        )}

        <button
          type="button"
          onClick={enterCompareMode}
          className="h-11 rounded-lg border border-blue-800 bg-blue-950/40 px-4 text-xs font-semibold uppercase tracking-widest text-blue-400 hover:bg-blue-900/40 hover:border-blue-700 transition-colors flex items-center gap-2"
        >
          <GitCompare className="h-3.5 w-3.5" />
          Compare
        </button>
      </div>

      {selectedCompany && (
        <p className="mt-2 text-xs text-gray-600">
          Showing analysis for <span className="text-gray-400">{selectedCompany.name}</span>.
        </p>
      )}
    </div>
  );
}
