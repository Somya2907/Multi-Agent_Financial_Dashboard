"use client";

import { useEffect, useRef, useState } from "react";
import type { ChatMessage, Citation, GraphState } from "@/lib/types";
import { askFollowup } from "@/lib/api";
import { FileText, ExternalLink, Send, Loader2, Sparkles } from "lucide-react";

interface Props {
  priorState: GraphState | null;
  messages: ChatMessage[];
  setMessages: (updater: (prev: ChatMessage[]) => ChatMessage[]) => void;
  isInitialLoading: boolean;
}

const SOURCE_LABELS: Record<string, string> = {
  "10-K": "10-K",
  "10-Q": "10-Q",
  transcript: "Earnings Call",
  earnings_release: "Earnings Release",
  "8-K": "8-K",
  "8-k": "8-K",
  news: "News",
};

// Inline renderer: **bold**, `code`, and [SOURCE ...] highlighting.
function renderInline(text: string, keyPrefix: string) {
  const tokens: React.ReactNode[] = [];
  const regex = /(\*\*[^*]+\*\*|`[^`]+`|\[SOURCE[^\]]*\])/g;
  let last = 0;
  let m: RegExpExecArray | null;
  let i = 0;
  while ((m = regex.exec(text)) !== null) {
    if (m.index > last) tokens.push(text.slice(last, m.index));
    const t = m[0];
    if (t.startsWith("**")) {
      tokens.push(
        <strong key={`${keyPrefix}-b-${i++}`} className="font-semibold text-gray-50">
          {t.slice(2, -2)}
        </strong>
      );
    } else if (t.startsWith("`")) {
      tokens.push(
        <code
          key={`${keyPrefix}-c-${i++}`}
          className="rounded bg-gray-800 px-1 py-0.5 text-xs font-mono text-blue-300"
        >
          {t.slice(1, -1)}
        </code>
      );
    } else {
      tokens.push(
        <span
          key={`${keyPrefix}-s-${i++}`}
          className="rounded bg-blue-950/50 px-1 py-0.5 text-xs font-mono text-blue-300"
        >
          {t}
        </span>
      );
    }
    last = m.index + t.length;
  }
  if (last < text.length) tokens.push(text.slice(last));
  return tokens;
}

// Analyst-report renderer: headings (##, ###), bullets (- / *), paragraphs.
function renderReport(text: string) {
  const lines = text.replace(/\r\n/g, "\n").split("\n");
  const blocks: React.ReactNode[] = [];
  let bullets: string[] = [];
  let para: string[] = [];
  let idx = 0;

  const flushBullets = () => {
    if (!bullets.length) return;
    const localBullets = bullets;
    const k = `ul-${idx++}`;
    blocks.push(
      <ul key={k} className="list-disc pl-5 space-y-1 mb-3 text-sm text-gray-200">
        {localBullets.map((b, i) => (
          <li key={`${k}-${i}`} className="leading-relaxed">
            {renderInline(b, `${k}-${i}`)}
          </li>
        ))}
      </ul>
    );
    bullets = [];
  };

  const flushPara = () => {
    if (!para.length) return;
    const joined = para.join(" ");
    const k = `p-${idx++}`;
    blocks.push(
      <p key={k} className="text-sm leading-relaxed text-gray-200 mb-3 last:mb-0">
        {renderInline(joined, k)}
      </p>
    );
    para = [];
  };

  const flushAll = () => {
    flushPara();
    flushBullets();
  };

  for (const raw of lines) {
    const line = raw.trim();
    if (!line) {
      flushAll();
      continue;
    }
    if (line.startsWith("## ")) {
      flushAll();
      blocks.push(
        <h3
          key={`h2-${idx++}`}
          className="text-xs font-semibold uppercase tracking-widest text-blue-400 mt-4 first:mt-0 mb-2 pb-1 border-b border-gray-800"
        >
          {line.slice(3)}
        </h3>
      );
      continue;
    }
    if (line.startsWith("### ")) {
      flushAll();
      blocks.push(
        <h4 key={`h3-${idx++}`} className="text-sm font-semibold text-gray-100 mt-3 mb-1.5">
          {line.slice(4)}
        </h4>
      );
      continue;
    }
    if (line.startsWith("# ")) {
      flushAll();
      blocks.push(
        <h2 key={`h1-${idx++}`} className="text-sm font-bold text-gray-50 mt-4 first:mt-0 mb-2">
          {line.slice(2)}
        </h2>
      );
      continue;
    }
    if (line.startsWith("- ") || line.startsWith("* ")) {
      flushPara();
      bullets.push(line.slice(2));
      continue;
    }
    flushBullets();
    para.push(line);
  }
  flushAll();
  return blocks;
}

export function ChatPanel({ priorState, messages, setMessages, isInitialLoading }: Props) {
  const [input, setInput] = useState("");
  const [isAsking, setIsAsking] = useState(false);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  // Auto-scroll on new messages
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages.length, isAsking]);

  const canSend = !!priorState && input.trim().length > 0 && !isAsking && !isInitialLoading;

  const send = async () => {
    if (!canSend || !priorState) return;
    const question = input.trim();
    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: question }]);
    setIsAsking(true);
    try {
      const res = await askFollowup(question, priorState);
      const answer = (res.answer || "").trim()
        || "_No answer was returned. The model may have exhausted its reasoning budget — try a narrower question._";
      setMessages((prev) => [...prev, { role: "assistant", content: answer }]);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Follow-up failed";
      setMessages((prev) => [...prev, { role: "assistant", content: `⚠︎ ${msg}` }]);
    } finally {
      setIsAsking(false);
    }
  };

  // ── Initial loading skeleton ──
  if (isInitialLoading && messages.length === 0) {
    return (
      <div className="flex h-full flex-col rounded-xl border border-gray-800 bg-gray-900 p-5">
        <h2 className="text-xs font-semibold uppercase tracking-widest text-gray-500 mb-4">
          AI Assistant
        </h2>
        <div className="flex-1 flex flex-col gap-3 animate-pulse">
          <div className="h-3 w-3/4 rounded bg-gray-800" />
          <div className="h-3 w-full rounded bg-gray-800" />
          <div className="h-3 w-5/6 rounded bg-gray-800" />
          <div className="h-3 w-full rounded bg-gray-800" />
          <div className="h-3 w-2/3 rounded bg-gray-800" />
        </div>
        <p className="mt-4 text-xs text-gray-600 text-center">
          Running multi-agent analysis… ~30–60s
        </p>
      </div>
    );
  }

  // ── Empty state ──
  if (!priorState) {
    return (
      <div className="flex h-full flex-col rounded-xl border border-gray-800 bg-gray-900 p-5">
        <h2 className="text-xs font-semibold uppercase tracking-widest text-gray-500 mb-4">
          AI Assistant
        </h2>
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center max-w-[220px]">
            <FileText className="h-10 w-10 text-gray-700 mx-auto mb-3" />
            <p className="text-sm text-gray-500">
              Select a company to generate an analysis, then ask follow-up questions here.
            </p>
          </div>
        </div>
      </div>
    );
  }

  const initialAnswer = priorState.final_answer ?? "";
  const citations: Citation[] = priorState.citations ?? [];

  return (
    <div className="flex h-full flex-col rounded-xl border border-gray-800 bg-gray-900">
      {/* Header */}
      <div className="border-b border-gray-800 px-5 py-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Sparkles className="h-3.5 w-3.5 text-blue-400" />
          <h2 className="text-xs font-semibold uppercase tracking-widest text-gray-500">
            AI Assistant
          </h2>
        </div>
        {priorState.tickers?.length > 0 && (
          <span className="text-xs text-gray-600">
            Context: <span className="text-blue-400 font-medium">{priorState.tickers.join(", ")}</span>
          </span>
        )}
      </div>

      {/* Scrollable message area */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-5 py-4 space-y-4">
        {/* Initial analysis as first assistant message */}
        {initialAnswer && (
          <div className="rounded-lg bg-gray-850 border border-gray-800 px-4 py-3 text-gray-200">
            {renderReport(initialAnswer)}
          </div>
        )}

        {/* Follow-up Q&A thread */}
        {messages.map((m, i) =>
          m.role === "user" ? (
            <div key={i} className="flex justify-end">
              <div className="rounded-lg bg-blue-600 text-white px-3 py-2 text-sm max-w-[85%]">
                {m.content}
              </div>
            </div>
          ) : (
            <div key={i} className="rounded-lg bg-gray-850 border border-gray-800 px-4 py-3 text-gray-200">
              {renderReport(m.content)}
            </div>
          )
        )}

        {isAsking && (
          <div className="flex items-center gap-2 text-xs text-gray-500 px-1">
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
            Thinking…
          </div>
        )}
      </div>

      {/* Citations strip */}
      {citations.length > 0 && (
        <div className="border-t border-gray-800 px-5 py-3 max-h-32 overflow-y-auto">
          <p className="text-xs font-semibold uppercase tracking-widest text-gray-500 mb-2">
            Sources ({citations.length})
          </p>
          <div className="space-y-1">
            {citations.slice(0, 5).map((cit, i) => (
              <div
                key={i}
                className="flex items-center justify-between rounded border border-gray-800 bg-gray-850 px-2.5 py-1.5"
              >
                <div className="flex items-center gap-2 min-w-0">
                  <ExternalLink className="h-3 w-3 shrink-0 text-blue-500" />
                  <span className="text-xs font-medium text-blue-400">{cit.ticker}</span>
                  <span className="text-xs text-gray-500">
                    {SOURCE_LABELS[cit.source_type] ?? cit.source_type}
                  </span>
                  {cit.section_name && (
                    <span className="text-xs text-gray-600 truncate">{cit.section_name}</span>
                  )}
                </div>
                <span className="text-xs font-mono text-gray-700 ml-2">
                  {(cit.score * 100).toFixed(0)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Follow-up input */}
      <div className="border-t border-gray-800 px-4 py-3">
        <div className="flex items-center gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") send();
            }}
            placeholder="Ask a follow-up question…"
            disabled={isAsking}
            className="h-10 flex-1 rounded-lg border border-gray-700 bg-gray-800 px-3 text-sm text-gray-200 placeholder-gray-600 focus:border-blue-600 focus:outline-none transition-colors disabled:opacity-60"
          />
          <button
            type="button"
            onClick={send}
            disabled={!canSend}
            className="h-10 w-10 rounded-lg bg-blue-600 text-white flex items-center justify-center hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            {isAsking ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
          </button>
        </div>
      </div>
    </div>
  );
}
