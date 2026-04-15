"use client";

import { useState } from "react";
import type { Critique, ClaimResult } from "@/lib/types";
import { confidenceColor } from "@/lib/utils";
import { CheckCircle, AlertTriangle, XCircle } from "lucide-react";

interface Props {
  critique: Critique | null;
}

type Tab = "verified" | "flagged";

export function VerificationStatus({ critique }: Props) {
  const [tab, setTab] = useState<Tab>("verified");

  if (!critique || critique.error) {
    const msg = !critique
      ? "Submit a query to see verification status."
      : critique.error === "no_answer_to_verify"
      ? "No answer was generated to verify."
      : `Verification unavailable: ${critique.error}`;
    return (
      <div className="rounded-xl border border-gray-800 bg-gray-900 p-5">
        <h2 className="text-xs font-semibold uppercase tracking-widest text-gray-500 mb-3">
          Verification Status
        </h2>
        <p className="text-sm text-gray-500">{msg}</p>
      </div>
    );
  }

  const pct = Math.round((critique.verification_rate ?? 0) * 100);
  const confidenceClass = confidenceColor(critique.overall_confidence);
  const claims: ClaimResult[] = critique.claims ?? [];
  const verified = claims.filter((c) => c.status === "verified");
  const flagged = claims.filter((c) => c.status === "flagged");
  const activeList = tab === "verified" ? verified : flagged;

  return (
    <div className="rounded-xl border border-gray-800 bg-gray-900 p-5">
      <div className="flex items-start justify-between mb-4">
        <h2 className="text-xs font-semibold uppercase tracking-widest text-gray-500">
          Verification Status
        </h2>
        <span className={`text-xs font-semibold ${confidenceClass}`}>
          {critique.overall_confidence} CONFIDENCE
        </span>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-3 gap-3 mb-4">
        <div className="rounded-lg border border-gray-800 bg-gray-850 px-3 py-2 text-center">
          <p className="text-lg font-bold text-gray-100">{critique.total_claims}</p>
          <p className="text-xs text-gray-500">Total Claims</p>
        </div>
        <div className="rounded-lg border border-green-900 bg-green-950 px-3 py-2 text-center">
          <p className="text-lg font-bold text-green-300">{critique.verified_claims}</p>
          <p className="text-xs text-green-600">Verified</p>
        </div>
        <div className="rounded-lg border border-red-900 bg-red-950 px-3 py-2 text-center">
          <p className="text-lg font-bold text-red-300">{critique.flagged_claims}</p>
          <p className="text-xs text-red-600">Flagged</p>
        </div>
      </div>

      {/* Progress bar */}
      <div className="mb-4 space-y-1.5">
        <div className="flex justify-between text-xs text-gray-500">
          <span>Verification Rate</span>
          <span>{pct}%</span>
        </div>
        <div className="h-1.5 w-full rounded-full bg-gray-800">
          <div
            className={`h-1.5 rounded-full transition-all duration-700 ${
              pct >= 85 ? "bg-green-500" : pct >= 60 ? "bg-yellow-400" : "bg-red-500"
            }`}
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>

      {/* Tabs */}
      {claims.length > 0 && (
        <>
          <div className="flex gap-2 border-b border-gray-800 mb-3">
            <button
              type="button"
              onClick={() => setTab("verified")}
              className={`flex items-center gap-1.5 px-3 py-2 text-xs font-medium transition-colors border-b-2 -mb-px ${
                tab === "verified"
                  ? "border-green-500 text-green-300"
                  : "border-transparent text-gray-500 hover:text-gray-300"
              }`}
            >
              <CheckCircle className="h-3.5 w-3.5" />
              Verified ({verified.length})
            </button>
            <button
              type="button"
              onClick={() => setTab("flagged")}
              className={`flex items-center gap-1.5 px-3 py-2 text-xs font-medium transition-colors border-b-2 -mb-px ${
                tab === "flagged"
                  ? "border-red-500 text-red-300"
                  : "border-transparent text-gray-500 hover:text-gray-300"
              }`}
            >
              <AlertTriangle className="h-3.5 w-3.5" />
              Flagged ({flagged.length})
            </button>
          </div>

          <div className="space-y-2 max-h-64 overflow-y-auto">
            {activeList.length === 0 ? (
              <p className="text-xs text-gray-600 italic px-1 py-2">
                {tab === "verified"
                  ? "No claims were verified against source evidence."
                  : "No flagged claims — every assertion is grounded."}
              </p>
            ) : (
              activeList.map((c, i) => (
                <div
                  key={i}
                  className={`rounded-lg border px-3 py-2 ${
                    c.status === "verified"
                      ? "border-green-900 bg-green-950/30"
                      : "border-yellow-900 bg-yellow-950/30"
                  }`}
                >
                  <div className="flex items-start gap-2">
                    {c.status === "verified" ? (
                      <CheckCircle className="h-3.5 w-3.5 text-green-400 shrink-0 mt-0.5" />
                    ) : (
                      <XCircle className="h-3.5 w-3.5 text-yellow-400 shrink-0 mt-0.5" />
                    )}
                    <div className="min-w-0">
                      <p
                        className={`text-xs font-medium mb-1 ${
                          c.status === "verified" ? "text-green-200" : "text-yellow-200"
                        }`}
                      >
                        {c.text}
                      </p>
                      <p
                        className={`text-xs ${
                          c.status === "verified" ? "text-green-700" : "text-yellow-700"
                        }`}
                      >
                        {c.reason}
                      </p>
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </>
      )}
    </div>
  );
}
