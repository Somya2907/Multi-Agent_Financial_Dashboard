"use client";

import { useState, useEffect } from "react";
import type { Company } from "@/lib/types";
import { fetchCompanies } from "@/lib/api";
import { Search, ChevronDown, RefreshCw } from "lucide-react";

interface Props {
  onAnalyzeCompany: (ticker: string, refresh?: boolean) => void;
  onCustomQuery: (query: string, ticker?: string) => void;
  selectedTicker: string;
  isLoading: boolean;
}

export function QueryPanel({
  onAnalyzeCompany,
  onCustomQuery,
  selectedTicker,
  isLoading,
}: Props) {
  const [query, setQuery] = useState("");
  const [companies, setCompanies] = useState<Company[]>([]);
  const [showDropdown, setShowDropdown] = useState(false);
  const [showCustom, setShowCustom] = useState(false);

  useEffect(() => {
    fetchCompanies()
      .then(setCompanies)
      .catch(() => {});
  }, []);

  function handlePickCompany(ticker: string) {
    setShowDropdown(false);
    if (ticker) onAnalyzeCompany(ticker, false);
  }

  const submitCustom = () => {
    if (!query.trim() || isLoading) return;
    onCustomQuery(query.trim(), selectedTicker || undefined);
  };

  const selectedCompany = companies.find((c) => c.ticker === selectedTicker);

  return (
    <div className="rounded-xl border border-gray-800 bg-gray-900 p-5">
      <div className="flex items-center gap-3">
        {/* Company selector — primary action */}
        <div className="relative flex-1">
          <button
            type="button"
            onClick={() => setShowDropdown((v) => !v)}
            disabled={isLoading}
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
                <span className="text-gray-500">Select a company to analyze…</span>
              )}
            </span>
            <ChevronDown className="h-4 w-4 text-gray-500 shrink-0" />
          </button>

          {showDropdown && (
            <div className="absolute top-full left-0 mt-1 z-50 w-full rounded-lg border border-gray-700 bg-gray-800 shadow-xl overflow-hidden">
              <div className="max-h-72 overflow-y-auto">
                {companies.map((c) => (
                  <button
                    key={c.ticker}
                    type="button"
                    onClick={() => handlePickCompany(c.ticker)}
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

        {/* Refresh cached analysis */}
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

        {/* Advanced toggle */}
        <button
          type="button"
          onClick={() => setShowCustom((v) => !v)}
          className="h-11 rounded-lg border border-gray-800 bg-gray-850 px-4 text-xs text-gray-500 hover:text-gray-300 hover:border-gray-700 transition-colors"
        >
          {showCustom ? "Hide" : "Advanced"}
        </button>
      </div>

      {/* Advanced: custom free-form query (legacy mode) */}
      {showCustom && (
        <div className="mt-3 flex gap-2">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-500 pointer-events-none" />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") submitCustom();
              }}
              placeholder="Ask a custom analysis question across any company…"
              className="h-10 w-full rounded-lg border border-gray-700 bg-gray-800 pl-9 pr-3 text-sm text-gray-200 placeholder-gray-600 focus:border-blue-600 focus:outline-none transition-colors"
            />
          </div>
          <button
            type="button"
            onClick={submitCustom}
            disabled={!query.trim() || isLoading}
            className="h-10 rounded-lg bg-blue-600 px-5 text-sm font-semibold text-white hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            Run
          </button>
        </div>
      )}
    </div>
  );
}
