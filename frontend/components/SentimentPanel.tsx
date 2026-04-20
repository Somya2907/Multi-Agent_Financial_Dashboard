"use client";

import type { QualitativeAnalysis } from "@/lib/types";
import { toneColor } from "@/lib/utils";
import { MessageSquare } from "lucide-react";

interface Props {
  qualitative: QualitativeAnalysis | null;
}

export function SentimentPanel({ qualitative }: Props) {
  if (!qualitative || qualitative.error) {
    return (
      <div className="rounded-xl border border-gray-800 bg-gray-900 p-5">
        <h2 className="text-xs font-semibold uppercase tracking-widest text-gray-500 mb-3">
          Management &amp; Sentiment Analysis
        </h2>
        <p className="text-sm text-gray-500">
          No qualitative data available — query may not include earnings transcripts.
        </p>
      </div>
    );
  }

  const tone = qualitative.overall_tone ?? "neutral";
  const badgeClass = toneColor(tone);
  const sentimentPct = Math.round((((qualitative.sentiment_score ?? 0) + 1) / 2) * 100);

  return (
    <div className="rounded-xl border border-gray-800 bg-gray-900 p-5">
      <div className="flex items-start justify-between mb-4">
        <h2 className="text-xs font-semibold uppercase tracking-widest text-gray-500">
          Management &amp; Sentiment Analysis
        </h2>
        <span
          className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-semibold ${badgeClass}`}
        >
          <MessageSquare className="h-3 w-3" />
          {tone.toUpperCase()}
        </span>
      </div>

      {/* Sentiment score */}
      <div className="mb-4 space-y-1.5">
        <div className="flex justify-between text-xs text-gray-500">
          <span>Sentiment Score</span>
          <span>{(qualitative.sentiment_score ?? 0) > 0 ? "+" : ""}{(qualitative.sentiment_score ?? 0).toFixed(2)}</span>
        </div>
        <div className="h-1.5 w-full rounded-full bg-gray-800">
          <div
            className={`h-1.5 rounded-full transition-all duration-700 ${
              sentimentPct >= 60 ? "bg-green-500" : sentimentPct >= 40 ? "bg-yellow-400" : "bg-red-500"
            }`}
            style={{ width: `${sentimentPct}%` }}
          />
        </div>
      </div>

      {/* Key themes */}
      {qualitative.key_themes?.length > 0 && (
        <div className="mb-4">
          <p className="text-xs text-gray-500 mb-2">Key Themes</p>
          <ul className="space-y-1">
            {qualitative.key_themes.map((theme, i) => (
              <li key={i} className="flex items-start gap-2 text-sm text-gray-300">
                <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-blue-400" />
                {theme}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Risk themes */}
      {qualitative.risk_themes?.length > 0 && (
        <div className="mb-4">
          <p className="text-xs text-gray-500 mb-2">Risk Themes Mentioned</p>
          <ul className="space-y-1">
            {qualitative.risk_themes.map((theme, i) => (
              <li key={i} className="flex items-start gap-2 text-sm text-gray-300">
                <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-red-400" />
                {theme}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Forward-looking statements */}
      {qualitative.forward_looking?.length > 0 && (
        <div>
          <p className="text-xs text-gray-500 mb-2">Forward-Looking Statements</p>
          <ul className="space-y-2">
            {qualitative.forward_looking.map((stmt, i) => (
              <li
                key={i}
                className="rounded-lg border border-gray-800 bg-gray-850 px-3 py-2 text-xs text-gray-400 italic"
              >
                &ldquo;{stmt}&rdquo;
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
