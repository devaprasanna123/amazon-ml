"""
src/preprocessing.py
====================
Text normalization for the Amazon Business Entity Resolution pipeline.

Design principles:
- NEVER modify original values; always produce a new normalized field.
- Normalization is lossy-but-reversible: keep enough information for matching.
- Handle Devanagari, Tamil, Kannada and other Indic scripts gracefully
  (lowercase ASCII parts; preserve non-ASCII script characters as-is).
- Observed patterns in actual data:
    * S2 records: ALL-CAPS names and addresses
    * S3 records: Title Case names, mixed addresses
    * S1 records: mixed case
    * Legal suffixes: LLC, Inc, Ltd, Limited, Pvt, Private, LLP, Corp,
                      Corporation, Co, Company, PC, PLC, LP, LTD, PVT
    * Address components often reordered across sources
    * Indic script names (Hindi, Kannada, Tamil) in both S2 and S3
    * URLs in names (heassociates.com, wilfordhancock.com)
    * Noise: |, --, ##, PO Box, Unit markers
    * Postal patterns: US 5-digit ZIP, India 6-digit PIN
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Constants built from observed data
# ─────────────────────────────────────────────────────────────────────────────

# Legal suffix variants → canonical form (applied after lowercasing)
LEGAL_SUFFIXES: dict[str, str] = {
    # LLC variants
    "llc": "llc",
    "l.l.c.": "llc",
    "l.l.c": "llc",
    # Inc variants
    "inc": "inc",
    "inc.": "inc",
    "incorporated": "inc",
    # Ltd variants
    "ltd": "ltd",
    "ltd.": "ltd",
    "limited": "ltd",
    # Pvt variants
    "pvt": "pvt",
    "pvt.": "pvt",
    "private": "pvt",
    # LLP
    "llp": "llp",
    "l.l.p.": "llp",
    # Corp variants
    "corp": "corp",
    "corp.": "corp",
    "corporation": "corp",
    # Company variants
    "co": "co",
    "co.": "co",
    "company": "co",
    # PC (professional corp)
    "pc": "pc",
    "p.c.": "pc",
    # PLC
    "plc": "plc",
    "p.l.c.": "plc",
    # LP
    "lp": "lp",
    "l.p.": "lp",
}

# Build regex that matches any legal suffix token at word boundary
_SUFFIX_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(LEGAL_SUFFIXES, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

# Address abbreviation expansions (applied after lowercasing)
ADDRESS_ABBREV: dict[str, str] = {
    r"\bst\b": "street",
    r"\bst\.\b": "street",
    r"\bave\b": "avenue",
    r"\bave\.\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\brd\b": "road",
    r"\brd\.\b": "road",
    r"\bdr\b": "drive",
    r"\bdr\.\b": "drive",
    r"\bln\b": "lane",
    r"\bln\.\b": "lane",
    r"\bct\b": "court",
    r"\bct\.\b": "court",
    r"\bpl\b": "place",
    r"\bpl\.\b": "place",
    r"\bcir\b": "circle",
    r"\bfwy\b": "freeway",
    r"\bhwy\b": "highway",
    r"\bpkwy\b": "parkway",
    r"\bpky\b": "parkway",
    r"\bsq\b": "square",
    r"\bfte\b": "suite",
    r"\bste\b": "suite",
    r"\bste\.\b": "suite",
    r"\bapt\b": "apartment",
    r"\bapt\.\b": "apartment",
    r"\bfl\b": "floor",
    r"\bflr\b": "floor",
    r"\bn\b": "north",
    r"\bs\b": "south",
    r"\be\b": "east",
    r"\bw\b": "west",
    r"\bne\b": "northeast",
    r"\bnw\b": "northwest",
    r"\bse\b": "southeast",
    r"\bsw\b": "southwest",
    # India-specific
    r"\bnagar\b": "nagar",
    r"\bh\.no\b": "house no",
    r"\bh\.no\.\b": "house no",
    r"\bhn\b": "house no",
    r"\bno\.\b": "no",
    r"\bno\b": "no",
    r"\bopp\.\b": "opposite",
    r"\bopp\b": "opposite",
    r"\bnr\b": "near",
}

# Build combined compiled regex dict for address abbrevs
_ADDR_ABBREV_PATTERNS = [
    (re.compile(pat, re.IGNORECASE), repl)
    for pat, repl in ADDRESS_ABBREV.items()
]

# Patterns to strip from names (noise)
_NAME_NOISE_RE = re.compile(
    r"(\|.*$)"            # everything after a pipe
    r"|(--+\s*)"          # leading dashes
    r"|(\s*\|\s*.*)"      # pipe + rest
    r"|(https?://\S+)"    # URLs
    r"|(\.\w{2,4}\b(?!\.))"  # .com .org .net etc (loose)
    r"|(#+\s*\w+)"        # hashtag-style (#centraleducation)
    ,
    re.IGNORECASE,
)

# Patterns to strip from addresses
_ADDR_NOISE_RE = re.compile(
    r"(#+\s*)"            # ## markers
    r"|(po\s+box\s*\S+)"  # PO Box (keep number separately if needed)
    r"|(p\.o\.\s*box\s*\S+)"
    ,
    re.IGNORECASE,
)

# Postal code patterns
_US_ZIP_RE = re.compile(r"\b(\d{5})(?:-\d{4})?\b")
_INDIA_PIN_RE = re.compile(r"\b(\d{6})\b")
_GENERIC_NUMERIC_RE = re.compile(r"\b\d+(?:[/\-]\d+)*\b")

# Tokenization: split on whitespace + common punctuation but keep alphanumerics
_TOKEN_RE = re.compile(r"[a-zA-Z0-9\u0900-\u097F\u0C00-\u0C7F\u0B80-\u0BFF"
                        r"\u0980-\u09FF\u0A00-\u0A7F\u0A80-\u0AFF\u0C80-\u0CFF"
                        r"\u0D00-\u0D7F]+", re.UNICODE)


# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _unicode_normalize(text: str) -> str:
    """NFKC normalize — collapses ligatures, width variants, etc."""
    return unicodedata.normalize("NFKC", text)


def _is_ascii_char(c: str) -> bool:
    return ord(c) < 128


def _safe_lower(text: str) -> str:
    """Lowercase ASCII letters only; preserve non-ASCII (Indic scripts)."""
    return "".join(c.lower() if _is_ascii_char(c) else c for c in text)


def _remove_punct_ascii(text: str) -> str:
    """
    Replace ASCII punctuation with spaces, keep non-ASCII characters intact.
    Preserves digits and letters (including Indic).
    """
    result = []
    for c in text:
        if _is_ascii_char(c):
            if c.isalnum() or c.isspace():
                result.append(c)
            else:
                result.append(" ")
        else:
            result.append(c)
    return "".join(result)


def _collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> List[str]:
    """
    Extract meaningful tokens including Indic script characters.
    Returns lowercase ASCII tokens + original non-ASCII tokens.
    """
    if not text:
        return []
    tokens = _TOKEN_RE.findall(text)
    return [_safe_lower(t) for t in tokens if t]


# Precompiled dotted legal forms
_DOTTED_LEGAL = [
    (re.compile(r"\bl\.l\.c\.?\b", re.IGNORECASE), "llc"),
    (re.compile(r"\bl\.l\.p\.?\b", re.IGNORECASE), "llp"),
    (re.compile(r"\bp\.c\.?\b", re.IGNORECASE), "pc"),
    (re.compile(r"\bp\.l\.c\.?\b", re.IGNORECASE), "plc"),
    (re.compile(r"\bl\.p\.?\b", re.IGNORECASE), "lp"),
    (re.compile(r"\bpvt\.?\b", re.IGNORECASE), "pvt"),
    (re.compile(r"\bltd\.?\b", re.IGNORECASE), "ltd"),
    (re.compile(r"\binc\.?\b", re.IGNORECASE), "inc"),
    (re.compile(r"\bcorp\.?\b", re.IGNORECASE), "corp"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Name normalization
# ─────────────────────────────────────────────────────────────────────────────

def normalize_name(name: str) -> str:
    """
    Normalize a business name for matching.

    Steps:
    1. NFKC Unicode normalize
    2. Strip leading noise (-- markers, pipe-delimited extras)
    3. Lowercase ASCII
    4. Normalize '&' → 'and'
    5. Remove ASCII punctuation
    6. Collapse whitespace
    7. Normalize legal suffixes to canonical short form

    Non-ASCII characters (Indic scripts) are preserved after lowercasing.
    """
    if not name or not name.strip():
        return ""

    t = _unicode_normalize(name)

    # Strip pipe-separated alternate names and URL suffixes
    t = re.sub(r"\s*\|.*", "", t)
    t = re.sub(r"^--+\s*", "", t)
    # Strip .com / website suffixes if the whole name isn't just that
    t = re.sub(r"(?<!\w)\.(?:com|org|net|in|co)\b", "", t, flags=re.IGNORECASE)
    # Strip #hashtag prefix
    t = re.sub(r"^#+\s*", "", t)

    # Lowercase ASCII only
    t = _safe_lower(t)

    # Normalize ampersand
    t = re.sub(r"\s*&\s*", " and ", t)

    # ── Collapse dotted legal forms ──
    for _pat, _repl in _DOTTED_LEGAL:
        t = _pat.sub(_repl, t)

    # ── Normalize legal suffixes BEFORE removing punctuation ──

    # This ensures "L.L.C." "L.L.P." etc. are matched before dots are stripped.
    def _replace_suffix(m: re.Match) -> str:
        token = m.group(0).lower()
        return LEGAL_SUFFIXES.get(token, token)

    t = _SUFFIX_RE.sub(_replace_suffix, t)

    # Remove ASCII punctuation (keep alphanumeric + spaces + non-ASCII)
    t = _remove_punct_ascii(t)

    # Collapse whitespace
    t = _collapse_whitespace(t)

    return t


def name_tokens(name: str) -> List[str]:
    """Return deduplicated tokens of normalized name."""
    norm = normalize_name(name)
    return list(dict.fromkeys(tokenize(norm)))  # preserve order, remove dups


# ─────────────────────────────────────────────────────────────────────────────
# Address normalization
# ─────────────────────────────────────────────────────────────────────────────

def normalize_address(addr: str) -> str:
    """
    Normalize a business address for matching.

    Steps:
    1. NFKC Unicode normalize
    2. Strip ## noise markers
    3. Lowercase ASCII
    4. Expand common abbreviations (Street, Avenue, Road, etc.)
    5. Remove ASCII punctuation
    6. Collapse whitespace

    Does NOT reorder components — component order is part of the evidence.
    """
    if not addr or not addr.strip():
        return ""

    t = _unicode_normalize(addr)

    # Strip ## noise
    t = re.sub(r"#+\s*", "", t)

    # Lowercase ASCII only
    t = _safe_lower(t)

    # Expand address abbreviations
    for pat, repl in _ADDR_ABBREV_PATTERNS:
        t = pat.sub(repl, t)

    # Remove ASCII punctuation (keep digits, letters, non-ASCII, spaces)
    t = _remove_punct_ascii(t)

    # Collapse whitespace
    t = _collapse_whitespace(t)

    return t


def extract_numerics(addr: str) -> List[str]:
    """
    Extract all numeric tokens from an address (building numbers, PIN codes, etc.).
    Returns list of numeric strings found.
    """
    if not addr:
        return []
    return _GENERIC_NUMERIC_RE.findall(addr)


def extract_postal_code(addr: str) -> Optional[str]:
    """
    Extract the most likely postal code from an address.
    Tries US 5-digit ZIP first, then India 6-digit PIN.
    Returns string or None.
    """
    if not addr:
        return None
    m = _US_ZIP_RE.search(addr)
    if m:
        return m.group(1)
    m = _INDIA_PIN_RE.search(addr)
    if m:
        return m.group(1)
    return None


_ADDR_STOPWORDS = {
    "no", "and", "the", "of", "in", "at", "near", "to", "a", "an",
    "for", "by", "with", "is", "on", "or", "as", "be", "it",
    "1st", "2nd", "3rd", "4th",  # floor markers handled separately
}


def address_tokens(addr: str) -> List[str]:
    """Return meaningful tokens from normalized address (no stopwords)."""
    norm = normalize_address(addr)
    toks = tokenize(norm)
    return [t for t in toks if t not in _ADDR_STOPWORDS and len(t) > 1]


def address_numeric_tokens(addr: str) -> List[str]:
    """Numeric tokens that appear in the address (for number-specific blocking)."""
    return [t for t in tokenize(addr) if t.isdigit()]


# ─────────────────────────────────────────────────────────────────────────────
# Country normalization
# ─────────────────────────────────────────────────────────────────────────────

_COUNTRY_MAP: dict[str, str] = {
    "us": "US",
    "usa": "US",
    "united states": "US",
    "united states of america": "US",
    "india": "India",
    "in": "India",
    "bharat": "India",
    "france": "France",
    "fr": "France",
}


def normalize_country(country: str) -> str:
    """Map country string to canonical label. Unknown → original stripped."""
    if not country:
        return ""
    key = country.strip().lower()
    return _COUNTRY_MAP.get(key, country.strip())


# ─────────────────────────────────────────────────────────────────────────────
# Row-level normalizer (for DataFrame apply)
# ─────────────────────────────────────────────────────────────────────────────

def normalize_record(row: dict) -> dict:
    """
    Normalize a single source record dict.
    Input keys: entity_id, business_name, business_address, country
    Returns dict with added keys:
      norm_name, norm_address, norm_country,
      name_toks, addr_toks, addr_numerics, postal_code
    """
    name = row.get("business_name", "") or ""
    addr = row.get("business_address", "") or ""
    country = row.get("country", "") or ""

    norm_name = normalize_name(name)
    norm_addr = normalize_address(addr)
    norm_country = normalize_country(country)
    n_toks = list(dict.fromkeys(tokenize(norm_name)))
    a_toks = [t for t in tokenize(norm_addr) if t not in _ADDR_STOPWORDS and len(t) > 1]
    a_nums = [t for t in tokenize(norm_addr) if t.isdigit()]
    postal = extract_postal_code(addr)

    return {
        **row,
        "norm_name": norm_name,
        "norm_address": norm_addr,
        "norm_country": norm_country,
        "name_toks": " ".join(n_toks),
        "addr_toks": " ".join(a_toks),
        "addr_numerics": " ".join(a_nums),
        "postal_code": postal or "",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Batch normalization (chunked, memory-efficient)
# ─────────────────────────────────────────────────────────────────────────────

def normalize_dataframe(df: "pd.DataFrame") -> "pd.DataFrame":
    """
    Add normalized columns to a DataFrame in-place.
    Reuses already-normalized name and address to avoid redundant re-normalization.
    """
    df = df.copy()

    # Base normalizations (computed once)
    df["norm_name"] = df["business_name"].apply(normalize_name)
    df["norm_address"] = df["business_address"].apply(normalize_address)
    df["norm_country"] = df["country"].apply(normalize_country)

    # Token and feature extractions derived from already normalized text
    df["name_toks"] = df["norm_name"].apply(lambda s: " ".join(dict.fromkeys(tokenize(s))))
    df["addr_toks"] = df["norm_address"].apply(
        lambda s: " ".join(t for t in tokenize(s) if t not in _ADDR_STOPWORDS and len(t) > 1)
    )
    df["addr_numerics"] = df["norm_address"].apply(
        lambda s: " ".join(t for t in tokenize(s) if t.isdigit())
    )
    df["postal_code"] = df["business_address"].apply(extract_postal_code)

    return df


def normalize_file_chunked(
    input_path: "Path",
    output_path: "Path",
    chunk_size: int = 50_000,
    show_progress: bool = True,
) -> None:
    """
    Stream-normalize a source TSV file, writing results to a Parquet (or CSV) file.
    Memory: only one chunk in RAM at a time.
    """
    import time
    import pandas as pd
    from pathlib import Path

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    total = 0
    chunks_out = []

    for i, chunk in enumerate(
        pd.read_csv(input_path, sep="\t", dtype=str,
                    keep_default_na=False, chunksize=chunk_size)
    ):
        normed = normalize_dataframe(chunk)
        chunks_out.append(normed)
        total += len(chunk)
        if show_progress and (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  chunk {i+1}: {total:,} rows  ({elapsed:.1f}s)")

    df_full = pd.concat(chunks_out, ignore_index=True)

    if output_path.suffix == ".parquet":
        try:
            df_full.to_parquet(str(output_path), index=False)
        except Exception:
            csv_path = output_path.with_suffix(".csv")
            df_full.to_csv(str(csv_path), index=False)
            output_path = csv_path
    else:
        df_full.to_csv(str(output_path), index=False, sep="\t")

    elapsed = time.time() - t0
    if show_progress:
        print(f"  Done: {total:,} rows in {elapsed:.1f}s → {output_path}")
