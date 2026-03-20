"""Aggregate filtered chunks into a structured context string for the LLM."""


def _format_source_label(chunk: dict) -> str:
    """Format a readable source citation for a chunk."""
    ticker = chunk.get("ticker", "?")
    source_type = chunk.get("source_type", "?")
    section = chunk.get("section_name", "?")
    date = chunk.get("filing_date", "")
    score = chunk.get("score", 0)

    label = f"{ticker} | {source_type}"
    if date:
        label += f" ({date})"
    label += f" | {section}"
    label += f" [relevance: {score:.3f}]"
    return label


def aggregate_chunks(chunks: list[dict]) -> tuple[str, list[dict]]:
    """Build a context string from filtered chunks and collect source citations.

    Groups chunks by (ticker, source_type, section_name) so related passages
    appear together in the context, making it easier for the LLM to reason
    across them.

    Returns:
        context_str: Formatted multi-section context string.
        citations: List of source dicts for inclusion in the final answer.
    """
    # Group by source
    from collections import defaultdict
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for chunk in chunks:
        key = (
            chunk.get("ticker"),
            chunk.get("source_type"),
            chunk.get("section_name"),
            chunk.get("filing_date"),
        )
        groups[key].append(chunk)

    context_parts = []
    citations = []

    for (ticker, source_type, section, date), group_chunks in groups.items():
        header = f"[SOURCE: {ticker} — {source_type}"
        if date:
            header += f" ({date})"
        header += f" — {section}]"

        passages = "\n\n".join(c["text"] for c in group_chunks)
        context_parts.append(f"{header}\n{passages}")

        for c in group_chunks:
            citations.append({
                "chunk_id": c.get("chunk_id"),
                "ticker": ticker,
                "source_type": source_type,
                "section_name": section,
                "filing_date": date,
                "score": c.get("score"),
                "source_url": c.get("source_url", ""),
            })

    context_str = "\n\n---\n\n".join(context_parts)
    return context_str, citations
