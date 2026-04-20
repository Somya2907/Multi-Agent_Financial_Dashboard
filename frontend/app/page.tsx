"use client";

import { useMemo, useState } from "react";
import type { ChatMessage, Citation, GraphState } from "@/lib/types";
import { analyzeCompany, askFollowup } from "@/lib/api";
import { QueryPanel } from "@/components/QueryPanel";
import { ExecutiveSummary } from "@/components/ExecutiveSummary";
import { MetricsTable } from "@/components/MetricsTable";
import { SentimentPanel } from "@/components/SentimentPanel";
import { RiskCategories } from "@/components/RiskCategories";
import { VerificationStatus } from "@/components/VerificationStatus";
import { ChatPanel } from "@/components/ChatPanel";
import { BarChart3 } from "lucide-react";

function CompanyColumn({ data, ticker }: { data: GraphState | null; ticker: string }) {
  return (
    <div className="space-y-5">
      <div className="flex items-center gap-2 px-1">
        <span className="text-lg font-semibold text-blue-400">{ticker}</span>
        {data?.tickers?.[0] && data.tickers[0] !== ticker && (
          <span className="text-xs text-gray-600">({data.tickers[0]})</span>
        )}
      </div>
      <ExecutiveSummary
        riskSynthesis={data?.risk_synthesis ?? null}
        tickers={data?.tickers ?? [ticker]}
        queryType={data?.query_type ?? "general"}
      />
      <MetricsTable metrics={data?.metrics ?? {}} />
      <SentimentPanel qualitative={data?.qualitative_analysis ?? null} />
      <RiskCategories riskSynthesis={data?.risk_synthesis ?? null} />
      <VerificationStatus critique={data?.critique ?? null} />
    </div>
  );
}

function mergeStates(a: GraphState, b: GraphState, finalAnswer: string): GraphState {
  const mergedCitations: Citation[] = [...(a.citations ?? []), ...(b.citations ?? [])];
  const mergedMetrics = { ...(a.metrics ?? {}), ...(b.metrics ?? {}) };
  const mergedChunks = [
    ...(a.retrieved_chunks ?? []),
    ...(b.retrieved_chunks ?? []),
  ];
  const tickerA = a.tickers?.[0] ?? "A";
  const tickerB = b.tickers?.[0] ?? "B";

  return {
    ...a,
    query: `Compare ${tickerA} vs ${tickerB}`,
    query_type: "comparison",
    tickers: [tickerA, tickerB],
    metrics: mergedMetrics,
    retrieved_chunks: mergedChunks,
    citations: mergedCitations,
    final_answer: finalAnswer,
    qualitative_analysis: a.qualitative_analysis,
    risk_synthesis: a.risk_synthesis,
    critique: a.critique,
  };
}

const COMPARISON_PROMPT =
  "Produce a single, integrated side-by-side comparison of these two companies. " +
  "Do NOT write two separate reports. For each dimension — liquidity, leverage, " +
  "profitability, cash flow, management tone, and overall risk — state which " +
  "company looks stronger or weaker and by how much, citing specific numbers " +
  "from each. End with a 'Trader Takeaway' on relative positioning.";

export default function Home() {
  const [result, setResult] = useState<GraphState | null>(null);
  const [resultB, setResultB] = useState<GraphState | null>(null);
  const [comparisonAnswer, setComparisonAnswer] = useState<string | null>(null);
  const [selectedTicker, setSelectedTicker] = useState("");
  const [compareTickers, setCompareTickers] = useState<[string, string] | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);

  async function handleAnalyzeCompany(ticker: string, refresh = false) {
    setIsLoading(true);
    setError(null);
    setSelectedTicker(ticker);
    setCompareTickers(null);
    setResultB(null);
    setComparisonAnswer(null);
    setMessages([]);
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

  async function handleCompare(tickerA: string, tickerB: string) {
    setIsLoading(true);
    setError(null);
    setSelectedTicker(tickerA);
    setCompareTickers([tickerA, tickerB]);
    setMessages([]);
    setResult(null);
    setResultB(null);
    setComparisonAnswer(null);
    try {
      const [dataA, dataB] = await Promise.all([
        analyzeCompany(tickerA, false),
        analyzeCompany(tickerB, false),
      ]);
      setResult(dataA);
      setResultB(dataB);

      // Seed the prior_analysis with both reports so the LLM sees each
      // company's full context (risk/qualitative blocks in followup only
      // format a single ticker, so we rely on final_answer for B's details).
      const priorAnalysis = [
        `## ${tickerA}`,
        dataA.final_answer?.trim() || "_(no analysis)_",
        "",
        `## ${tickerB}`,
        dataB.final_answer?.trim() || "_(no analysis)_",
      ].join("\n\n");
      const merged = mergeStates(dataA, dataB, priorAnalysis);
      try {
        // gpt-5 burns reasoning tokens; comparison needs a larger budget.
        const res = await askFollowup(COMPARISON_PROMPT, merged, 20000);
        setComparisonAnswer(res.answer || "_Comparison could not be generated._");
      } catch (e) {
        setComparisonAnswer(
          `_Comparison generation failed: ${e instanceof Error ? e.message : "unknown error"}_`,
        );
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Comparison failed");
      setResult(null);
      setResultB(null);
    } finally {
      setIsLoading(false);
    }
  }

  function handleExitCompare() {
    setCompareTickers(null);
    setResultB(null);
    setComparisonAnswer(null);
    setMessages([]);
  }

  const chatState = useMemo<GraphState | null>(() => {
    if (compareTickers && result && resultB) {
      return mergeStates(result, resultB, comparisonAnswer ?? "");
    }
    return result;
  }, [compareTickers, result, resultB, comparisonAnswer]);

  const isCompareMode = !!compareTickers;

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
        <QueryPanel
          onAnalyzeCompany={handleAnalyzeCompany}
          onCompare={handleCompare}
          selectedTicker={selectedTicker}
          compareTickers={compareTickers}
          isLoading={isLoading}
          onExitCompare={handleExitCompare}
        />

        {error && (
          <div className="rounded-xl border border-red-800 bg-red-950/30 px-5 py-4 text-sm text-red-300">
            {error}
          </div>
        )}

        {/* Dashboard grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
          {/* Left — structured panels (split into 2 sub-cols in compare mode) */}
          <div className="lg:col-span-2">
            {isCompareMode ? (
              <div className="grid grid-cols-1 xl:grid-cols-2 gap-5">
                <CompanyColumn data={result} ticker={compareTickers![0]} />
                <CompanyColumn data={resultB} ticker={compareTickers![1]} />
              </div>
            ) : (
              <div className="space-y-5">
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
            )}
          </div>

          {/* Right — AI chat (sticky sidebar) */}
          <div className="lg:col-span-1">
            <div className="sticky top-[4.5rem] h-[calc(100vh-6rem)]">
              <ChatPanel
                priorState={chatState}
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
