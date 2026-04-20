# Multi-Agent Financial RAG System

A multi-agent retrieval-augmented generation system for financial risk analysis over 50 large-cap US public companies. Built for **CMU 11-766 (Advanced NLP)** as an end-semester project.

The system combines a **LangGraph multi-agent orchestrator**, a **Multi-HyDE hybrid retriever** (FAISS + BM25 with a BGE cross-encoder reranker), structured **financial-metric computation** from SEC XBRL, **qualitative sentiment analysis** of earnings calls, **risk synthesis** across five categories, and a **critic / fact-checker** — all exposed through a **FastAPI** backend and a **Next.js 14 dashboard**.

---

## Table of Contents

1. [What the system does](#what-the-system-does)
2. [Architecture](#architecture)
3. [Data pipeline](#data-pipeline)
4. [Retrieval: Multi-HyDE + hybrid + reranker](#retrieval-multi-hyde--hybrid--reranker)
5. [Multi-agent graph (LangGraph)](#multi-agent-graph-langgraph)
6. [API backend (FastAPI)](#api-backend-fastapi)
7. [Frontend dashboard (Next.js)](#frontend-dashboard-nextjs)
8. [Models used](#models-used)
9. [Evaluation results](#evaluation-results)
10. [Repository layout](#repository-layout)
11. [Setup & running](#setup--running)
12. [Configuration](#configuration)
13. [Known limitations / trade-offs](#known-limitations--trade-offs)

---

## What the system does

Given a single ticker (or a pair for comparison), the system produces a desk-ready equity-research style report containing:

- **Executive summary** — 4–5 sentence risk narrative with an overall risk level.
- **Financial metrics** — current ratio, debt-to-equity, margins, ROE, ROA, FCF margin, EPS (computed from XBRL filings; not scraped text).
- **Management & sentiment analysis** — overall tone, sentiment score, key themes, risk themes, forward-looking statements (extracted from earnings-call transcripts).
- **Risk categorisation** — severity (HIGH/MEDIUM/LOW), score, and 2–3 pieces of evidence for each of: market, credit, liquidity, regulatory, macroeconomic risk.
- **Verification status** — claim-level fact-checking against the underlying evidence by a separate Critic agent (verified / flagged counts and confidence).
- **Interactive AI Assistant** — follow-up Q&A grounded in the prior analysis, plus an integrated side-by-side analysis for two-company compares.
- **Citations** — ticker / source type (10-K, 10-Q, transcript, earnings release) / section name / relevance score.

Scope: **50 companies** (Technology, Financial Services, Healthcare, Consumer, Industrials, Energy, Communications — see [config/companies.py](config/companies.py)). Index covers 2024–2026 filings and transcripts (**38,689 chunks**).

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                         Next.js 14 Dashboard                            │
│   QueryPanel · ExecutiveSummary · MetricsTable · SentimentPanel        │
│   RiskCategories · VerificationStatus · ChatPanel                      │
└───────────────────────────────┬────────────────────────────────────────┘
                                │ HTTP (JSON)
┌───────────────────────────────▼────────────────────────────────────────┐
│                           FastAPI backend                               │
│  /analyze_company  /followup  /query  /companies  /health              │
└───────────────────────────────┬────────────────────────────────────────┘
                                │
┌───────────────────────────────▼────────────────────────────────────────┐
│                     LangGraph multi-agent orchestrator                  │
│                                                                         │
│   planner → retrieval → ┬─ financial_metrics ─┐                        │
│                         └─ qualitative ───────┴─► risk_synthesis       │
│                                                         │               │
│                                                  synthesize_answer     │
│                                                         │               │
│                                                       critic           │
└───────────────────────────────┬────────────────────────────────────────┘
                                │
          ┌─────────────────────┼─────────────────────┐
          ▼                     ▼                     ▼
   Multi-HyDE retriever   XBRL metrics        Earnings-call
   (FAISS + BM25 + BGE)    (live from cache)   transcripts
```

---

## Data pipeline

Raw documents are ingested, parsed, chunked, embedded, and indexed. All of this is regenerable from the scripts in [scripts/](scripts/).

### 1. Ingestion ([pipeline/fetchers/](pipeline/fetchers/))

- `sec_edgar.py` — SEC EDGAR filings via the official API (10-K, 10-Q, 8-K). Rate-limited to 10 req/s with a registered user-agent.
- `xbrl_fetcher.py` — structured balance-sheet / income-statement / cash-flow data from XBRL companyfacts (single JSON per ticker). This is the **source of truth for financial metrics** — we never parse dollar figures out of prose.
- `transcripts.py` — earnings-call transcripts.
- `earnings_releases.py` — 8-K earnings release exhibits.
- `news.py` — news articles.
- `financials.py` — wrapper tying XBRL data back to the balance-sheet store.

Raw artefacts live under `data/raw/` (gitignored). XBRL balance sheets are cached per-ticker under `data/financials/` (50 companies).

### 2. Parsing ([pipeline/parsers/](pipeline/parsers/))

- `sec_parser.py` — extracts section-aware text from 10-K/10-Q HTML.
- `transcript_parser.py` — splits transcripts into "prepared remarks" and "Q&A".
- `news_parser.py` — cleans news article bodies.
- `balance_sheet.py` / `balance_sheet_store.py` — loads the XBRL balance sheet into a typed dict used by the financial-metrics agent.

### 3. Chunking ([pipeline/chunking/chunker.py](pipeline/chunking/chunker.py))

- `cl100k_base` tokeniser (tiktoken).
- 512 tokens per chunk, 64-token overlap.
- Section-aware: each chunk is tagged with `ticker`, `source_type`, `section_name`, `fiscal_period`, `filing_date`.

Output: per-ticker JSONL under `data/processed/chunks/` (gitignored).

### 4. Embedding ([pipeline/embedding/embedder.py](pipeline/embedding/embedder.py))

- **Amazon Titan Text Embeddings V2** via AWS Bedrock.
- 1024-dimensional, L2-normalised vectors (compatible with FAISS inner-product search).
- Batched with retry/backoff; latency is the dominant cost.

### 5. Indexing ([pipeline/indexing/](pipeline/indexing/))

- `faiss_manager.py` — `IndexFlatIP` (inner product ≡ cosine on normalised vectors). Metadata-aware: keeps a `by_ticker` reverse index for ticker-filtered search without scanning the whole index.
- `bm25_index.py` — classical BM25 over the same chunk corpus for lexical / keyword matching (financial-statement terms often fail dense retrieval alone).

Artefacts: `faiss_index.bin`, `chunk_store.jsonl`, `metadata_index.json`, `bm25_index.pkl` in `data/indexes/` (gitignored — 38,689 chunks).

---

## Retrieval: Multi-HyDE + hybrid + reranker

Implemented in [pipeline/reasoning/hyde.py](pipeline/reasoning/hyde.py). This is the core retrieval loop used by every agent that needs document context.

```
   original query
       │
       ▼  classify_query()  →  GENERAL | NUMERICAL
       │
       ▼  LLM: generate_diverse_queries      (type-aware prompt)
   [q1, q2, q3, q4]
       │
       ▼  LLM, parallel: generate_hypothetical_doc × N
   [doc1, doc2, doc3, doc4]
       │
       ▼  Titan V2 embed_batch
   [vec1 … vec4]
       │
       ├── 4 × FAISS dense search       (dense_weight per query type)
       ├── 4 × BM25 sparse search       (sparse_weight per query type)
       │   + decomposed sub-queries     (NUMERICAL only, BM25 only)
       │
       ▼  weighted multi-list Reciprocal Rank Fusion
   merged candidate pool
       │
       ▼  BGE-reranker-v2-m3 against the ORIGINAL query
       │   + intent-based section / source bonus
   final ranked chunks
```

**Key design choices:**

- The **original query is never embedded**. Only the hypothetical documents produced by the LLM are embedded for dense search — this is the HyDE insight. The original query is used verbatim for BM25 (keyword match) and for the cross-encoder reranker.
- **Query-type adaptation**:
  - `GENERAL` queries (risk, narrative): `dense=0.5 / sparse=0.5`.
  - `NUMERICAL` queries (ratios, balance-sheet items): `dense=0.4 / sparse=0.6`. Keyword terms dominate. Hypothesis prompts produce balance-sheet-style passages with explicit line items and plausible dollar figures. The query is also **decomposed** into component BM25 sub-queries (e.g. "current ratio" → "current assets current liabilities balance sheet", "current assets total", "current liabilities total").
- **Intent classification** for rerank biasing: `metric | risk | sentiment | general`. The reranker gives a small additive bonus (+0.08) to chunks whose section or source type matches the intent — e.g. "what did management say about X" favours `transcript` chunks; "X risk factors" favours the Risk Factors section.
- **Weighted multi-list RRF**: `rrf_score(d) = dense_w × Σ 1/(rank+k) + sparse_w × Σ 1/(rank+k)` over all dense and sparse lists.
- **Reranker**: `BAAI/bge-reranker-v2-m3` cross-encoder. Scores are sigmoid'd to `[0,1]` for display as "relevance". Top-k from rerank (default 24) is what flows downstream.
- **Fallback robustness**: if the LLM returns an empty completion (reasoning-budget exhaustion in gpt-5), `generate_diverse_queries` and `generate_hypothetical_doc` fall back to deterministic templates so retrieval never degrades to zero.

---

## Multi-agent graph (LangGraph)

Compiled in [pipeline/agents/graph.py](pipeline/agents/graph.py). State is a `TypedDict` defined in [pipeline/agents/state.py](pipeline/agents/state.py).

```
  START
    │
  planner ── metrics_only_mode=True ─► financial_metrics ──► risk_synthesis
    │                                                                   ▲
    │ (default)                                                          │
    ▼                                                                    │
  retrieval ──► fan-out ──► [financial_metrics, qualitative_analysis] ──┤
                            (parallel, converge at risk_synthesis)      │
                                                                         │
                                                                  risk_synthesis
                                                                         │
                                                                 synthesize_answer
                                                                         │
                                                                      critic
                                                                         │
                                                                        END
```

### Agents

| Agent | File | Purpose |
|---|---|---|
| **Planner** | [pipeline/agents/planner.py](pipeline/agents/planner.py) | Classifies query type (`single_company` / `comparison` / `general`), extracts tickers, decides whether metrics and/or qualitative analysis are needed, and whether to skip retrieval entirely (`metrics_only_mode`). |
| **Retrieval** | [pipeline/agents/retrieval.py](pipeline/agents/retrieval.py) | Runs the Multi-HyDE retriever per ticker, deduplicates across tickers, builds the formatted `context` and `citations` for downstream agents. Applies the optional `year_filter`. |
| **Financial Metrics** | [pipeline/agents/financial_metrics.py](pipeline/agents/financial_metrics.py) | Computes ratios deterministically from the cached XBRL balance sheet (never from retrieved text). Returns per-ticker: current ratio, debt-to-equity, gross/operating/net margin, ROE, ROA, FCF margin, EPS, plus the raw line items. |
| **Qualitative** | [pipeline/agents/qualitative.py](pipeline/agents/qualitative.py) | LLM-driven extraction from transcript chunks → overall tone, sentiment score (-1…+1), key themes, risk themes, forward-looking statements, management confidence. |
| **Risk Synthesis** | [pipeline/agents/risk_synthesis.py](pipeline/agents/risk_synthesis.py) | Combines metrics + qualitative + top chunks to score 5 risk categories (market / credit / liquidity / regulatory / macroeconomic). Produces an `overall_risk_level`, `overall_risk_score`, and a rule-based 4–5 sentence summary written for a trader audience. |
| **Synthesize Answer** | `_synthesize_answer` in [graph.py](pipeline/agents/graph.py) | Injects metrics, qualitative, and risk blocks into the context, then calls the answer generator ([pipeline/reasoning/answer_generator.py](pipeline/reasoning/answer_generator.py)) to produce the final markdown-formatted analyst report. |
| **Critic** | [pipeline/agents/critic.py](pipeline/agents/critic.py) | Extracts every factual claim from the final answer, verifies each against the evidence (metrics / qualitative / chunks), and returns `{total_claims, verified_claims, flagged_claims, verification_rate, issues, overall_confidence}`. |

### Follow-up path ([pipeline/reasoning/followup.py](pipeline/reasoning/followup.py))

The `/followup` endpoint bypasses the full graph. It re-uses the prior `GraphState` (metrics + risk + qualitative + retrieved chunks + prior answer) and makes **one** LLM call to answer a grounded follow-up question — ~3–8 s latency vs ~30–60 s for the full graph.

The frontend's side-by-side **Compare** mode calls this endpoint with a synthetic "produce a single integrated comparison" prompt, merging both companies' states. Because gpt-5 spends most of its completion budget on reasoning tokens, the comparison path uses `max_completion_tokens=20000` instead of the default `6000`.

---

## API backend (FastAPI)

Entry point: [api/server.py](api/server.py). File-based response cache (24 h TTL) in [api/cache.py](api/cache.py).

| Endpoint | Method | Behaviour |
|---|---|---|
| `/analyze_company` | POST | `{ticker, refresh?}` → full GraphState. Checks the 24 h file cache first; on miss, runs the graph with a default comprehensive-risk-analysis prompt and `year_filter = (current_year-2, current_year)`. |
| `/followup` | POST | `{query, prior_state, max_tokens?}` → `{answer, citations}`. Single LLM call grounded in the prior state. Used for the chat panel and for the merged compare analysis. |
| `/query` | POST | `{query, ticker?}` → GraphState. Free-form query against the full graph (no cache). |
| `/companies` | GET | Returns all 50 configured companies (`ticker`, `name`, `sector`). |
| `/health` | GET | Liveness check. |

CORS is enabled for `localhost:3000`/`3001`. FAISS is **not fork-safe** — always run uvicorn with `--workers 1`.

---

## Frontend dashboard (Next.js)

Next.js 14 (App Router) + Tailwind CSS + lucide-react icons. Dark theme, single-page dashboard.

**Layout:**

```
┌──────────────────────────────────────────────────────────────────────┐
│ Financial Risk Analyst AI        · CMU 11-766 · 50 companies · ...   │
├──────────────────────────────────────────────────────────────────────┤
│  Company ▼   [↻ refresh]   [⟷ Compare]                              │
├─────────────────────────────────────────────┬────────────────────────┤
│ Executive Summary                [MED risk] │  AI Assistant          │
│ Key Financial Metrics                       │                        │
│ Management & Sentiment Analysis             │  [initial analysis]    │
│ Risk Categorisation (5 cards)               │  [follow-up thread]    │
│ Verification Status                         │                        │
│                                             │  Sources (top 5)       │
│                                             │  [ ask follow-up ► ]   │
└─────────────────────────────────────────────┴────────────────────────┘
```

**Compare mode** splits the left column into two sub-columns (one per ticker) so all five panels render side-by-side. The right-column AI Assistant receives a merged GraphState and displays **one** integrated comparison analysis generated via `/followup`.

**Components** ([frontend/components/](frontend/components/)):

| File | Purpose |
|---|---|
| `QueryPanel.tsx` | Company selector, refresh (cache-bypass), Compare toggle with two-dropdown picker (mutual exclusion). |
| `ExecutiveSummary.tsx` | Overall risk badge + narrative. |
| `MetricsTable.tsx` | Computed ratios + raw balance-sheet line items. |
| `SentimentPanel.tsx` | Tone badge, sentiment bar, key + risk themes, forward-looking statements. |
| `RiskCategories.tsx` | 5 collapsible cards, one per risk category, with severity and evidence. |
| `VerificationStatus.tsx` | Critic stats (claims verified / flagged / rate), flagged-claim list. |
| `ChatPanel.tsx` | Initial analysis rendered as markdown + follow-up thread with citations strip. |

---

## Models used

| Role | Model | Provider | Notes |
|---|---|---|---|
| **Main LLM** (planner, qualitative, risk, answer, critic, followup) | `gpt-5` | CMU AI Gateway (LiteLLM proxy at `ai-gateway.andrew.cmu.edu`) | Reasoning model — budgets adjusted to account for CoT tokens. |
| **HyDE generation** (eval runs) | `gpt-4.1-mini` | Same gateway | Non-reasoning, ~10× cheaper, ~3× faster. Reranker is what determines retrieval quality, so cheaper HyDE does not degrade final ranking. |
| **Embeddings** | `amazon.titan-embed-text-v2:0` (1024-d) | AWS Bedrock | L2-normalised → cosine ≡ inner product in FAISS. |
| **Reranker** | `BAAI/bge-reranker-v2-m3` | Local (HuggingFace `sentence-transformers`) | Cross-encoder; CPU-OK but preferred on GPU. |

The default LLM is set via `OPENAI_LLM_MODEL` in `.env` (falls back to `gpt-4o` in `config/settings.py`).

---

## Evaluation results

### Retrieval-only evaluation on ConvFinQA

**Setup:**

- ConvFinQA dev + train bundles each question with a (`pre_text`, `table`, `post_text`) passage drawn from a specific 10-K page, identified by a `filename` field like `AAPL/2023/page_42.pdf`.
- We build a **separate retrieval corpus** from ConvFinQA's own passages (1,806 unique passages after dedup across train+dev by filename) — scripts: [scripts/build_convfinqa_corpus.py](scripts/build_convfinqa_corpus.py), [scripts/eval_convfinqa_retrieval.py](scripts/eval_convfinqa_retrieval.py).
- The full Multi-HyDE retriever runs on this ConvFinQA corpus. Gold evidence = the passage whose `chunk_id` equals the question's `filename`. This isolates the retrieval *algorithm* from our 50-ticker domain corpus (which is time-shifted; ConvFinQA questions reference 2007–2018).

**Configuration:** `--hyde-model gpt-4.1-mini`, `--k 20`, dev set (421 questions), 15-query sample run.

**Results (15-query sample):**

| Question type | N | R@5 | R@10 | R@20 | MRR@10 |
|---|---:|---:|---:|---:|---:|
| Type I (`qa.question`)       | 10 | 80.00 % | 80.00 % | 80.00 % | 0.667 |
| Type II (`qa_0.question`)    | 5  | 100.00 % | 100.00 % | 100.00 % | 0.740 |
| **Overall**                  | **15** | **86.67 %** | **86.67 %** | **86.67 %** | **0.691** |

- 13/15 questions retrieve the gold passage in the top 5.
- MRR@10 ≈ 0.69 → the gold passage is on average at rank ~1.4 when found.
- The two failures (ranks = 0) are both Type I questions asking for return computations over specific dollar amounts — the gold passage and several distractor passages share nearly identical balance-sheet language.
- Latency per query (full Multi-HyDE with BGE rerank, median ≈ 73 s; range 50–293 s). gpt-4.1-mini HyDE vs gpt-5 HyDE cuts median per-query latency by ~30 %.

Per-query results: [data/eval/convfinqa_results_15.csv](data/eval/convfinqa_results_15.csv).

### Ad-hoc retrieval probe on the 50-ticker domain corpus

- 28-query probe across `metric_lookup`, `risk_factors`, `sentiment`, `multi-hop` categories, with category-appropriate relevance rubrics in [data/eval/retrieval_eval.jsonl](data/eval/retrieval_eval.jsonl).
- Used to iteratively tune the rerank section/source boost, the NUMERICAL-query decomposition dictionary, and the intent classifier.

### End-to-end smoke tests

- `python scripts/test_agents.py` exercises each agent in isolation.
- `python scripts/test_retrieval.py` runs a handful of natural-language queries end-to-end and prints the retrieved chunks.
- Response caches for AAPL and AMZN live in `data/cache/` so the UI can demo without re-running the full 30–60 s pipeline.

---

## Repository layout

```
Multi-Agent_Financial_Dashboard/
├── api/                              FastAPI backend
│   ├── server.py                     endpoints
│   └── cache.py                      24 h file cache for /analyze_company
├── config/
│   ├── companies.py                  50 tickers + CIK + sector
│   └── settings.py                   pydantic-settings config (reads .env)
├── pipeline/
│   ├── agents/                       LangGraph agents + state + graph
│   │   ├── graph.py                  orchestrator
│   │   ├── state.py                  TypedDict shared state
│   │   ├── planner.py
│   │   ├── retrieval.py
│   │   ├── financial_metrics.py
│   │   ├── qualitative.py
│   │   ├── risk_synthesis.py
│   │   └── critic.py
│   ├── reasoning/
│   │   ├── hyde.py                   Multi-HyDE retriever
│   │   ├── reranker.py               BGE cross-encoder + section boost
│   │   ├── answer_generator.py       final-answer LLM call
│   │   ├── followup.py               lightweight /followup handler
│   │   ├── metrics_computer.py       ratio calculations (XBRL-sourced)
│   │   ├── noise_filter.py           chunk post-processing
│   │   ├── aggregator.py             context assembly helpers
│   │   └── query_engine.py           legacy single-shot query helper
│   ├── fetchers/                     SEC, XBRL, transcripts, news
│   ├── parsers/                      10-K / transcript / news parsers
│   ├── chunking/chunker.py           tiktoken chunker
│   ├── embedding/embedder.py         Titan V2 Bedrock client
│   ├── indexing/
│   │   ├── faiss_manager.py          FAISS IndexFlatIP + metadata index
│   │   └── bm25_index.py             BM25 sparse index
│   └── orchestrator.py               legacy non-graph orchestrator
├── frontend/                         Next.js 14 dashboard
│   ├── app/{layout,page,globals}     root
│   ├── components/                   7 panels (see table above)
│   └── lib/{api,types,utils}.ts
├── scripts/
│   ├── run_pipeline.py               build indexes from scratch
│   ├── refresh_xbrl.py               refresh cached XBRL balance sheets
│   ├── test_agents.py                per-agent smoke tests
│   ├── test_retrieval.py             retrieval smoke tests
│   ├── build_convfinqa_corpus.py     eval corpus builder
│   ├── eval_convfinqa_retrieval.py   ConvFinQA R@k / MRR evaluator
│   └── eval_retrieval.py             in-domain rubric evaluator
├── data/                             (gitignored except eval/)
│   ├── raw/                          raw SEC / transcript downloads
│   ├── processed/chunks/             per-ticker chunk JSONL
│   ├── indexes/                      FAISS + BM25 indexes (main + convfinqa)
│   ├── financials/                   XBRL balance sheets (50 JSONs)
│   ├── cache/                        /analyze_company response cache
│   └── eval/                         evaluation datasets + results
├── tests/                            pytest suite (non-critical)
├── pyproject.toml                    Python package config
├── Project_Proposal.pdf              original proposal
└── README.md                         this file
```

---

## Setup & running

### Prerequisites

- Python ≥ 3.10
- Node.js ≥ 18
- AWS credentials with Bedrock access (for Titan embeddings)
- An OpenAI-compatible LLM endpoint (we use the CMU AI Gateway; any LiteLLM proxy or the OpenAI API works)

### 1. Install Python deps

```bash
pip install -e ".[dev]"
```

### 2. Configure `.env` at the repo root

```bash
# LLM (CMU AI Gateway — or plain OpenAI)
OPENAI_API_KEY=sk-...
OPENAI_LLM_MODEL=gpt-5
OPENAI_BASE_URL=https://ai-gateway.andrew.cmu.edu

# AWS Bedrock (Titan embeddings)
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1

# SEC EDGAR (required by EDGAR ToS)
SEC_USER_AGENT=Your Name you@example.com
```

### 3. Build the index (one-time, ~hours — or unpack an existing `data/` directory)

```bash
python scripts/run_pipeline.py            # ingest → chunk → embed → index
python scripts/refresh_xbrl.py            # fetch XBRL balance sheets
```

### 4. Run the backend

```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload --workers 1
# FAISS is not fork-safe — always --workers 1
```

### 5. Run the frontend

```bash
cd frontend
npm install
npm run dev          # http://localhost:3000
```

### 6. Run evaluations

```bash
# ConvFinQA retrieval-only eval
python scripts/build_convfinqa_corpus.py                      # 1,806 passages
python scripts/eval_convfinqa_retrieval.py --hyde-model gpt-4.1-mini --limit 15

# In-domain rubric eval
python scripts/eval_retrieval.py
```

---

## Configuration

All settings live in [config/settings.py](config/settings.py) (pydantic-settings; env-overridable):

| Setting | Default | Notes |
|---|---|---|
| `openai_llm_model` | `gpt-5` | Override via `OPENAI_LLM_MODEL` in `.env`. |
| `openai_base_url` | `https://ai-gateway.andrew.cmu.edu` | CMU gateway; use `https://api.openai.com/v1` for plain OpenAI. |
| `embedding_model_id` | `amazon.titan-embed-text-v2:0` | Must match the vectors in the built index. |
| `embedding_dim` | `1024` | |
| `chunk_size_tokens` | `512` | |
| `chunk_overlap_tokens` | `64` | |
| `tokenizer_name` | `cl100k_base` | Matches GPT-4 family. |
| `faiss_index_type` | `FlatIP` | 1024-d × 38,689 vectors is small — `Flat` is fine. |
| `sec_rate_limit` | `0.11` | Seconds between SEC requests (≈ 10 req/s). |

---

## Known limitations / trade-offs

- **Latency vs quality**: gpt-5 is a reasoning model; a full `/analyze_company` run takes ~30–60 s because every agent makes one or more gpt-5 calls. `/followup` is ~3–8 s because it skips the graph. We cache `/analyze_company` responses for 24 h per ticker to keep the UI snappy.
- **Reasoning-token budgeting**: gpt-5 can consume its entire `max_completion_tokens` on internal reasoning and emit an empty visible answer. Each agent sets a per-call budget that accounts for this (e.g. HyDE = 4000; final answer = large; comparison followup = 20,000). Template fallbacks keep HyDE working even on empty completions.
- **Time window**: the indexed corpus covers 2024–2026. ConvFinQA questions reference 2007–2018, which is why we evaluate retrieval against ConvFinQA's *own* passages rather than our domain index.
- **FAISS fork-unsafety**: always run the API with `--workers 1`. The index is loaded once into the parent process; a forked worker would segfault on first FAISS query.
- **50-ticker scope**: adding a ticker requires updating `config/companies.py` and re-running the ingest pipeline for that ticker.
- **UI scope**: dashboard is read-only — no PDF export, no saved reports, no auth. Everything lives in browser state.
