Multi-Agent Financial RAG

This project implements a Multi-Agent Retrieval-Augmented Generation (RAG) system for financial analysis using a planner–executor architecture built with LangGraph. The system combines hybrid retrieval (BM25 + dense embeddings via FAISS) with Multi-HyDE query expansion to improve recall and semantic coverage, achieving high retrieval relevance across diverse financial queries.

A central planner agent decomposes complex user queries into structured sub-tasks, while specialized executor agents handle retrieval, reasoning, and synthesis. The pipeline integrates financial data sources and produces context-aware, explainable responses through a coordinated multi-agent workflow. This modular design enables scalability, better interpretability, and improved performance on multi-hop financial question answering.
