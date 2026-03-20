"""Parse earnings call transcripts into speaker-segmented sections."""

import logging
import re

logger = logging.getLogger(__name__)

EXECUTIVE_TITLES = [
    "ceo", "cfo", "coo", "cto", "president", "chairman",
    "chief executive", "chief financial", "chief operating",
    "vice president", "vp", "director", "head of",
]

QA_MARKERS = [
    r"(?i)question.and.answer",
    r"(?i)q\s*&\s*a\s*session",
    r"(?i)we.will.now.begin.the.question",
    r"(?i)open.the.line.for.questions",
    r"(?i)operator.*question",
]


def _classify_speaker(speaker: str) -> str:
    """Classify a speaker as 'executive', 'analyst', or 'operator'."""
    speaker_lower = speaker.lower()
    if "operator" in speaker_lower:
        return "operator"
    for title in EXECUTIVE_TITLES:
        if title in speaker_lower:
            return "executive"
    return "analyst"


def _find_qa_boundary(text: str) -> int | None:
    """Find the position where Q&A session begins."""
    for pattern in QA_MARKERS:
        match = re.search(pattern, text)
        if match:
            return match.start()
    return None


def parse_transcript(
    content: str,
    ticker: str,
    year: int,
    quarter: int,
    date: str = "",
) -> list[dict]:
    """Parse an earnings call transcript into structured sections.

    Returns a list of parsed section dicts ready for chunking.
    """
    fiscal_period = f"Q{quarter}_{year}"

    # Split into prepared remarks and Q&A
    qa_pos = _find_qa_boundary(content)

    sections = []
    if qa_pos:
        prepared = content[:qa_pos].strip()
        qa = content[qa_pos:].strip()

        if prepared:
            sections.append(
                {
                    "ticker": ticker,
                    "source_type": "transcript",
                    "filing_date": date,
                    "fiscal_period": fiscal_period,
                    "section_name": "Prepared Remarks",
                    "text": prepared,
                    "source_url": "",
                }
            )
        if qa:
            sections.append(
                {
                    "ticker": ticker,
                    "source_type": "transcript",
                    "filing_date": date,
                    "fiscal_period": fiscal_period,
                    "section_name": "Q&A Session",
                    "text": qa,
                    "source_url": "",
                }
            )
    else:
        # No Q&A boundary found — treat as single document
        sections.append(
            {
                "ticker": ticker,
                "source_type": "transcript",
                "filing_date": date,
                "fiscal_period": fiscal_period,
                "section_name": "Full Transcript",
                "text": content.strip(),
                "source_url": "",
            }
        )

    logger.info(f"Parsed {ticker} {fiscal_period} transcript: {len(sections)} sections")
    return sections
