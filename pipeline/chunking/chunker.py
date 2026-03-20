"""Token-based text chunker with overlap and sentence-boundary snapping."""

import logging

import tiktoken

from config.settings import settings

logger = logging.getLogger(__name__)


def _get_tokenizer():
    return tiktoken.get_encoding(settings.tokenizer_name)


def _find_sentence_boundary(tokens: list[int], enc, start: int, window: int = 32) -> int:
    """Find the nearest sentence boundary (period/newline) within a window around `start`.

    Returns the best split position (token index).
    """
    best = start
    best_dist = window + 1

    search_start = max(0, start - window)
    search_end = min(len(tokens), start + window)

    # Decode tokens in the window to look for sentence endings
    for i in range(search_start, search_end):
        try:
            token_text = enc.decode([tokens[i]])
        except Exception:
            continue
        if token_text.rstrip().endswith((".","!","?","\n")):
            dist = abs(i - start)
            if dist < best_dist:
                best = i + 1  # Split after the sentence-ending token
                best_dist = dist

    return best if best_dist <= window else start


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[dict]:
    """Split text into overlapping token-based chunks.

    Returns a list of dicts with 'text' and 'token_count' keys.
    """
    chunk_size = chunk_size or settings.chunk_size_tokens
    chunk_overlap = chunk_overlap or settings.chunk_overlap_tokens
    min_trailing = 128  # Merge trailing chunks smaller than this

    enc = _get_tokenizer()
    tokens = enc.encode(text)
    total = len(tokens)

    if total == 0:
        return []

    if total <= chunk_size:
        return [{"text": text.strip(), "token_count": total}]

    chunks = []
    pos = 0

    while pos < total:
        end = min(pos + chunk_size, total)

        # Snap to sentence boundary if not at the very end
        if end < total:
            end = _find_sentence_boundary(tokens, enc, end)

        chunk_tokens = tokens[pos:end]
        chunk_text_str = enc.decode(chunk_tokens).strip()

        if chunk_text_str:
            chunks.append({
                "text": chunk_text_str,
                "token_count": len(chunk_tokens),
            })

        # Advance by (chunk_size - overlap), snapped to sentence boundary
        step = chunk_size - chunk_overlap
        pos = pos + step
        if pos >= total:
            break

        # If remaining tokens are too few, merge with last chunk
        remaining = total - pos
        if remaining < min_trailing and chunks:
            extra_text = enc.decode(tokens[pos:total]).strip()
            if extra_text:
                last = chunks[-1]
                merged_text = last["text"] + " " + extra_text
                merged_tokens = enc.encode(merged_text)
                chunks[-1] = {
                    "text": merged_text,
                    "token_count": len(merged_tokens),
                }
            break

    return chunks


def chunk_document(document: dict) -> list[dict]:
    """Chunk a parsed document and propagate metadata.

    Input: a dict with 'text' and metadata fields (ticker, source_type, etc.)
    Output: list of chunk dicts with inherited metadata + chunk_id, chunk_index, token_count.
    """
    text = document.get("text", "")
    if not text.strip():
        return []

    raw_chunks = chunk_text(text)

    ticker = document.get("ticker", "unknown")
    source_type = document.get("source_type", "unknown")
    filing_date = document.get("filing_date", "")
    section_name = document.get("section_name", "unknown")
    fiscal_period = document.get("fiscal_period", filing_date)

    results = []
    for i, chunk in enumerate(raw_chunks):
        # Create safe section name for chunk_id
        safe_section = section_name.lower().replace(" ", "_").replace("&", "and")
        chunk_id = f"{ticker}_{source_type}_{fiscal_period}_{safe_section}_{i:03d}"

        results.append({
            "chunk_id": chunk_id,
            "ticker": ticker,
            "source_type": source_type,
            "filing_date": filing_date,
            "fiscal_period": fiscal_period,
            "section_name": section_name,
            "chunk_index": i,
            "text": chunk["text"],
            "token_count": chunk["token_count"],
            "source_url": document.get("source_url", ""),
        })

    return results
