"""Filter low-quality and redundant chunks from retrieval results."""


def filter_noise(
    chunks: list[dict],
    min_score: float = 0.45,
    max_chunks: int = 12,
) -> list[dict]:
    """Remove low-relevance and near-duplicate chunks.

    Steps:
    1. Drop chunks below the score threshold.
    2. Deduplicate: for chunks from the same section with adjacent chunk_index,
       keep only the highest-scoring one.
    3. Cap at max_chunks.

    Args:
        chunks: Retrieved chunks, each with 'score', 'chunk_id', 'chunk_index',
                'ticker', 'source_type', 'section_name', 'text'.
        min_score: Minimum cosine similarity to keep a chunk.
        max_chunks: Maximum number of chunks to return.

    Returns:
        Filtered, deduplicated list sorted by descending score.
    """
    # Step 1: Score threshold
    filtered = [c for c in chunks if c.get("score", 0) >= min_score]

    # Step 2: Deduplicate adjacent chunks from the same section
    # Key = (ticker, source_type, section_name) — group chunks from the same section
    seen: dict[tuple, dict] = {}
    deduped = []

    for chunk in filtered:
        group_key = (
            chunk.get("ticker"),
            chunk.get("source_type"),
            chunk.get("section_name"),
            chunk.get("filing_date"),
        )
        if group_key not in seen:
            seen[group_key] = chunk
            deduped.append(chunk)
        else:
            # Keep the higher-scoring chunk from this section
            existing = seen[group_key]
            if chunk.get("score", 0) > existing.get("score", 0):
                deduped.remove(existing)
                seen[group_key] = chunk
                deduped.append(chunk)

    # Step 3: Sort and cap
    deduped.sort(key=lambda c: c.get("score", 0), reverse=True)
    return deduped[:max_chunks]
