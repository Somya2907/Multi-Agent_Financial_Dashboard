"""Parse SEC 10-K/10-Q HTML filings into section-segmented text."""

import logging
import re
from dataclasses import dataclass, asdict

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Standard 10-K section patterns
SECTION_PATTERNS = [
    (r"(?i)item\s*1a[\.\s\-:]+risk\s*factors", "Risk Factors"),
    (r"(?i)item\s*1b[\.\s\-:]+unresolved\s*staff", "Unresolved Staff Comments"),
    (r"(?i)item\s*1[\.\s\-:]+business", "Business"),
    (r"(?i)item\s*2[\.\s\-:]+properties", "Properties"),
    (r"(?i)item\s*3[\.\s\-:]+legal\s*proceedings", "Legal Proceedings"),
    (r"(?i)item\s*5[\.\s\-:]+market", "Market Information"),
    (r"(?i)item\s*6[\.\s\-:]+", "Reserved"),
    (r"(?i)item\s*7a[\.\s\-:]+quantitative", "Quantitative Disclosures"),
    (r"(?i)item\s*7[\.\s\-:]+management", "MD&A"),
    (r"(?i)item\s*8[\.\s\-:]+financial\s*statements", "Financial Statements"),
    (r"(?i)item\s*9a[\.\s\-:]+controls", "Controls and Procedures"),
    (r"(?i)item\s*9[\.\s\-:]+changes", "Disagreements with Accountants"),
]


@dataclass
class ParsedSection:
    ticker: str
    source_type: str
    filing_date: str
    section_name: str
    text: str
    source_url: str


def _clean_html(html: str) -> str:
    """Strip HTML tags, scripts, styles, and XBRL inline tags."""
    soup = BeautifulSoup(html, "lxml")

    # Remove script, style, and XBRL tags
    for tag in soup.find_all(["script", "style"]):
        tag.decompose()
    for tag in soup.find_all(re.compile(r"^ix:", re.IGNORECASE)):
        tag.unwrap()

    # Convert tables to text
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [td.get_text(separator=" ", strip=True) for td in tr.find_all(["td", "th"])]
            if any(cells):
                rows.append(" | ".join(cells))
        table.replace_with(soup.new_string("\n".join(rows) + "\n"))

    text = soup.get_text(separator="\n")

    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines)


def _segment_sections(text: str) -> list[tuple[str, str]]:
    """Split cleaned filing text into named sections.

    Returns list of (section_name, section_text) tuples.
    """
    # Find all section header positions
    matches = []
    for pattern, name in SECTION_PATTERNS:
        for m in re.finditer(pattern, text):
            matches.append((m.start(), name))

    if not matches:
        return [("Full Document", text)]

    # Sort by position in document
    matches.sort(key=lambda x: x[0])

    sections = []
    for i, (start, name) in enumerate(matches):
        end = matches[i + 1][0] if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        if len(section_text) > 100:  # Skip trivially small sections
            sections.append((name, section_text))

    return sections


def extract_balance_sheet_html(html: str) -> str | None:
    """Find the Consolidated Balance Sheets table in a raw filing HTML.

    Searches for any element containing "balance sheet" and returns the
    raw HTML string of the nearest following table so the downstream
    extractor can parse it structurally (row-by-row, cell-by-cell).
    Returns None if not found.
    """
    soup = BeautifulSoup(html, "lxml")

    # Walk every element looking for "balance sheet" in its text node
    for tag in soup.find_all(string=re.compile(r"balance\s+sheet", re.IGNORECASE)):
        parent = tag.parent
        # Walk up a few levels then look for the next table
        for _ in range(5):
            if parent is None:
                break
            table = parent.find_next("table")
            if table:
                row_count = len(table.find_all("tr"))
                if row_count > 0:
                    logger.info(f"Balance sheet table found ({row_count} rows)")
                    # Return raw HTML — balance_sheet.py parses it structurally
                    return str(table)
            parent = parent.parent

    logger.debug("No balance sheet table found in filing HTML")
    return None


def parse_filing(
    html: str,
    ticker: str,
    form_type: str,
    filing_date: str,
    source_url: str,
) -> list[dict]:
    """Parse a single SEC filing HTML into section-segmented documents.

    Returns a list of ParsedSection dicts ready for chunking.
    Also sets 'balance_sheet_text' on the first section if a balance
    sheet table is detected in the HTML.
    """
    # Extract balance sheet from raw HTML before cleaning destroys table structure
    balance_sheet_text = extract_balance_sheet_html(html)

    clean_text = _clean_html(html)
    sections = _segment_sections(clean_text)

    results = []
    first = True
    for section_name, section_text in sections:
        parsed = ParsedSection(
            ticker=ticker,
            source_type=form_type,
            filing_date=filing_date,
            section_name=section_name,
            text=section_text,
            source_url=source_url,
        )
        section_dict = asdict(parsed)
        # Attach balance sheet text to the first section only (avoids duplication)
        if first and balance_sheet_text:
            section_dict["balance_sheet_text"] = balance_sheet_text
            first = False
        results.append(section_dict)

    logger.info(
        f"Parsed {ticker} {form_type} ({filing_date}): {len(results)} sections"
    )
    return results
