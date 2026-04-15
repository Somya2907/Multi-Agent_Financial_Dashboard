"use client";

import { useState } from "react";
import type { ChatMessage, GraphState } from "@/lib/types";
import { analyzeCompany, submitQuery } from "@/lib/api";
import { QueryPanel } from "@/components/QueryPanel";
import { ExecutiveSummary } from "@/components/ExecutiveSummary";
import { MetricsTable } from "@/components/MetricsTable";
import { SentimentPanel } from "@/components/SentimentPanel";
import { RiskCategories } from "@/components/RiskCategories";
import { VerificationStatus } from "@/components/VerificationStatus";
import { ChatPanel } from "@/components/ChatPanel";
import { BarChart3 } from "lucide-react";

export default function Home() {
  const [result, setResult] = useState<GraphState | null>(null);
  const [selectedTicker, setSelectedTicker] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);

  async function handleAnalyzeCompany(ticker: string, refresh = false) {
    setIsLoading(true);
    setError(null);
    setSelectedTicker(ticker);
    setMessages([]); // fresh conversation per company
    try {
      const data = await analyzeCompany(ticker, refresh);
      setResult(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "An unexpected error occurred");
      setResult(null);
    } finally {
      setIsLoading(false);
    }
  }

  async function handleCustomQuery(query: string, ticker?: string) {
    setIsLoading(true);
    setError(null);
    setSelectedTicker(ticker || "");
    setMessages([]);
    try {
      const data = await submitQuery(query, ticker);
      setResult(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "An unexpected error occurred");
      setResult(null);
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      {/* Header */}
      <header className="border-b border-gray-800 bg-gray-900/80 backdrop-blur-sm sticky top-0 z-40">
        <div className="mx-auto max-w-screen-2xl px-6 py-3 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <BarChart3 className="h-5 w-5 text-blue-400" />
            <span className="text-sm font-semibold text-gray-100">
              Financial Risk Analyst AI
            </span>
            <span className="rounded-full bg-blue-950 border border-blue-800 px-2 py-0.5 text-xs text-blue-400">
              CMU 11-766
            </span>
          </div>
          <span className="text-xs text-gray-600">
            50 companies · SEC EDGAR · XBRL · Multi-HyDE RAG
          </span>
        </div>
      </header>

      <main className="mx-auto max-w-screen-2xl px-6 py-6 space-y-5">
        {/* Company selector / advanced query */}
        <QueryPanel
          onAnalyzeCompany={handleAnalyzeCompany}
          onCustomQuery={handleCustomQuery}
          selectedTicker={selectedTicker}
          isLoading={isLoading}
        />

        {/* Error state */}
        {error && (
          <div className="rounded-xl border border-red-800 bg-red-950/30 px-5 py-4 text-sm text-red-300">
            {error}
          </div>
        )}

        {/* Dashboard grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
          {/* Left column — structured panels */}
          <div className="lg:col-span-2 space-y-5">
            <ExecutiveSummary
              riskSynthesis={result?.risk_synthesis ?? null}
              tickers={result?.tickers ?? []}
              queryType={result?.query_type ?? "general"}
            />
            <MetricsTable metrics={result?.metrics ?? {}} />
            <SentimentPanel qualitative={result?.qualitative_analysis ?? null} />
            <RiskCategories riskSynthesis={result?.risk_synthesis ?? null} />
            <VerificationStatus critique={result?.critique ?? null} />
          </div>

          {/* Right column — AI chat + citations (sticky sidebar) */}
          <div className="lg:col-span-1">
            <div className="sticky top-[4.5rem] h-[calc(100vh-6rem)]">
              <ChatPanel
                priorState={result}
                messages={messages}
                setMessages={setMessages}
                isInitialLoading={isLoading}
              />
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
