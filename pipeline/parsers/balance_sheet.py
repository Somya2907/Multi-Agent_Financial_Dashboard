"""Extract balance sheet values and compute financial metrics from SEC filing text."""

import logging
import re

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ── Line-item patterns (most-specific first) ─────────────────────────────────
# Each tuple: (field_name, list_of_regex_alternatives)
_ITEM_PATTERNS: list[tuple[str, list[str]]] = [
    ("current_assets", [
        r"(?i)total\s+current\s+assets",
    ]),
    ("current_liabilities", [
        r"(?i)total\s+current\s+liabilities",
    ]),
    ("total_assets", [
        r"(?i)total\s+assets(?!\s+and)",  # avoid "total assets and liabilities"
    ]),
    ("total_liabilities", [
        r"(?i)total\s+liabilities(?!\s+and\s+stockholders)",
    ]),
    ("shareholder_equity", [
        r"(?i)total\s+(?:stockholders|shareholders)['\u2019\s]*\s*equity",
        r"(?i)total\s+equity",
    ]),
]


def _parse_dollar(token: str) -> float | None:
    """Convert a balance sheet cell string to a float.

    Handles:
      "143,566"    →  143566.0
      "(12,345)"   → -12345.0   (parentheses = negative)
      "$ 143,566"  →  143566.0
      "143566"     →  143566.0
      Multiple numbers in one cell → takes the FIRST match (most-recent period)
      Footnote markers / dashes / empty cells → None
    """
    token = token.strip()
    if not token or token in ("-", "—", "–", "*", ""):
        return None

    # Parenthesised negative: (1,234) or (  1,234  )
    neg_m = re.fullmatch(r"\(\s*([\d,]+)\s*\)", token)
    if neg_m:
        return -float(neg_m.group(1).replace(",", ""))

    # First comma-grouped number OR a run of 3+ digits
    num_m = re.search(r"([\d]{1,3}(?:,\d{3})+|\d{3,})", token)
    if num_m:
        return float(num_m.group(1).replace(",", ""))

    return None


# ── Primary: structured HTML table parsing ────────────────────────────────────

def _extract_from_html_table(html: str) -> dict[str, float]:
    """Parse a balance sheet HTML <table> directly for maximum fidelity.

    For each <tr>:
      1. Extract all <td>/<th> cell texts; collapse internal whitespace.
      2. Treat the first non-empty cell as the row label.
      3. Match the label against _ITEM_PATTERNS (case-insensitive).
      4. Take the first numeric value from cells AFTER the label column
         (leftmost = most-recent period in standard US filings).
      5. If no value found in subsequent cells, scan within the label cell
         itself (handles compact iXBRL rows that merge label + value in one <td>).
      6. Log every match for debugging.

    Returns an empty dict when nothing is found.
    """
    soup = BeautifulSoup(html, "lxml")

    # Unwrap inline XBRL tags so get_text() is not polluted by namespace text
    for tag in soup.find_all(re.compile(r"^ix:", re.IGNORECASE)):
        tag.unwrap()

    table = soup.find("table")
    if not table:
        logger.debug("[BalanceSheet] No <table> element found in HTML snippet")
        return {}

    values: dict[str, float] = {}

    for tr in table.find_all("tr"):
        raw_cells = tr.find_all(["td", "th"])
        if not raw_cells:
            continue

        # Normalize each cell: collapse internal whitespace
        cells = [" ".join(td.get_text(separator=" ").split()) for td in raw_cells]

        # First non-empty cell is the row label
        label = ""
        label_idx = 0
        for i, cell in enumerate(cells):
            if cell.strip():
                label = cell.strip()
                label_idx = i
                break

        if not label:
            continue

        for field, patterns in _ITEM_PATTERNS:
            if field in values:
                continue
            for pat in patterns:
                if not re.search(pat, label):
                    continue

                # Search cells after the label for the first numeric value
                value_found = None
                for cell in cells[label_idx + 1:]:
                    v = _parse_dollar(cell)
                    if v is not None:
                        value_found = v
                        break

                # Fallback: the label cell itself may contain the number
                # (e.g. "Total current assets 143,566" in a single <td>)
                if value_found is None:
                    after_label = re.sub(pat, "", label, flags=re.IGNORECASE).strip()
                    if after_label:
                        v = _parse_dollar(after_label.split()[-1])
                        if v is not None:
                            value_found = v

                if value_found is not None:
                    values[field] = value_found
                    logger.info(
                        f'[BalanceSheet] Matched "{label}" → {value_found:,.0f}'
                    )
                else:
                    logger.debug(
                        f'[BalanceSheet] Label matched for {field} ("{label}") '
                        f"but no numeric value in cells: {cells}"
                    )
                break  # only one pattern per field per row

    return values


# ── Secondary: pipe-delimited table text ─────────────────────────────────────

def _extract_from_table_text(text: str) -> dict[str, float]:
    """Parse pipe-delimited balance sheet text (output of _clean_html tables).

    Takes the first numeric value found on a matching row as the most-recent period.
    """
    values: dict[str, float] = {}
    lines = text.split("\n")

    for line in lines:
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 2:
            continue
        label = parts[0]

        for field, patterns in _ITEM_PATTERNS:
            if field in values:
                continue
            for pat in patterns:
                if re.search(pat, label):
                    for cell in parts[1:]:
                        v = _parse_dollar(cell)
                        if v is not None:
                            values[field] = v
                            logger.info(
                                f"  [BalanceSheet] Extracted {field} = {v:,.0f} from table text"
                            )
                            break
                    break

    return values


# ── Tertiary: plain-text regex fallback ──────────────────────────────────────

def _extract_with_regex(text: str) -> dict[str, float]:
    """Regex fallback: scan plain text for labelled dollar amounts."""
    values: dict[str, float] = {}

    for field, patterns in _ITEM_PATTERNS:
        if field in values:
            continue
        for pat in patterns:
            m = re.search(pat + r"[^\n]{0,60}?([\d,]{3,})", text)
            if m:
                v = _parse_dollar(m.group(1))
                if v is not None:
                    values[field] = v
                    logger.info(
                        f"  [BalanceSheet] Extracted {field} = {v:,.0f} via regex fallback"
                    )
                    break

    return values


# ── Public entry point ────────────────────────────────────────────────────────

def extract_balance_sheet_values(section_text: str) -> dict[str, float]:
    """Extract key balance sheet line items from an HTML table or plain text.

    Source routing (in order of reliability):
      1. HTML table parsing  — used when section_text is raw HTML (from sec_parser)
      2. Pipe-delimited text — used for cleaned-text representations
      3. Plain-text regex    — last resort

    Returns a dict with any subset of:
        current_assets, current_liabilities, total_assets,
        total_liabilities, shareholder_equity
    Missing items are simply absent (no None values).
    """
    logger.info("Extracting balance sheet values...")

    stripped = section_text.strip()

    # ── Primary: HTML table parser ────────────────────────────────────────────
    if stripped.startswith("<"):
        logger.info("  [BalanceSheet] Using HTML table parser (primary)")
        values = _extract_from_html_table(stripped)
        if values:
            logger.info(
                f"Balance sheet extraction complete (HTML). Found: {list(values.keys())}"
            )
            return values
        logger.info(
            "  [BalanceSheet] HTML parsing yielded no values — falling back to text"
        )

    # ── Secondary: pipe-delimited ─────────────────────────────────────────────
    values = _extract_from_table_text(section_text)

    # ── Tertiary: regex over any remaining gaps ───────────────────────────────
    missing = [f for f, _ in _ITEM_PATTERNS if f not in values]
    if missing:
        logger.info(f"  [BalanceSheet] Regex fallback for: {missing}")
        regex_values = _extract_with_regex(section_text)
        for field in missing:
            if field in regex_values:
                values[field] = regex_values[field]

    logger.info(f"Balance sheet extraction complete. Found: {list(values.keys())}")
    return values


# ── Metrics computation ───────────────────────────────────────────────────────

def compute_financial_metrics(balance_sheet: dict[str, float]) -> dict:
    """Compute current_ratio and debt_to_equity from extracted balance sheet values.

    Safe division — returns None for a ratio when a required input is missing or zero.
    Returns a dict with 'current_ratio', 'debt_to_equity', 'source', plus all raw values.
    """
    metrics: dict = {"source": "balance_sheet_extraction"}

    # Current Ratio = Current Assets / Current Liabilities
    ca = balance_sheet.get("current_assets")
    cl = balance_sheet.get("current_liabilities")
    if ca is not None and cl is not None and cl != 0:
        metrics["current_ratio"] = round(ca / cl, 3)
        logger.info(f"Computed current_ratio = {metrics['current_ratio']}")
    else:
        logger.info(
            f"Skipping current_ratio: current_assets={ca}, current_liabilities={cl}"
        )

    # Debt-to-Equity = Total Liabilities / Shareholder Equity
    tl = balance_sheet.get("total_liabilities")
    eq = balance_sheet.get("shareholder_equity")
    if tl is not None and eq is not None and eq != 0:
        metrics["debt_to_equity"] = round(tl / eq, 3)
        logger.info(f"Computed debt_to_equity = {metrics['debt_to_equity']}")
    else:
        logger.info(
            f"Skipping debt_to_equity: total_liabilities={tl}, shareholder_equity={eq}"
        )

    # Pass through raw values for transparency
    for k, v in balance_sheet.items():
        metrics[k] = v

    return metrics
