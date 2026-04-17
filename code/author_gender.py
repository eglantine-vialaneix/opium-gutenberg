"""
gutenberg_author_scraper.py
===========================
Scrapes author metadata (nationality, gender, birth/death dates) from:
  1. Wikipedia  (via MediaWiki REST API  – fast, rich prose)
  2. Wikidata   (via SPARQL + Entity API – structured, machine-readable)
  3. Open Library (via /search/authors + /authors endpoints)

Usage
-----
    from gutenberg_author_scraper import scrape_authors
    import pandas as pd

    authors = pd.Index([...])          # your Gutenberg author index
    df, not_found = scrape_authors(authors, delay=1.0)

    df.to_csv("author_metadata.csv", index=False)
    pd.Series(list(not_found)).to_csv("authors_not_found.csv", index=False)
"""

from __future__ import annotations

import re
import time
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional

import requests

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── result dataclass ──────────────────────────────────────────────────────────

@dataclass
class AuthorRecord:
    raw_name: str                       # original Gutenberg string
    normalised_name: str                # "First Last" used for queries
    gender: Optional[str]       = None
    nationality: Optional[str]  = None
    birth_year: Optional[int]   = None
    death_year: Optional[int]   = None
    source: Optional[str]       = None  # which API answered first
    found: bool                 = False


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  NAME NORMALISATION
# ═══════════════════════════════════════════════════════════════════════════════

# Titles/honorifics we strip from Gutenberg names
_NOISE = re.compile(
    r"\b(Mrs?\.?|Sir|Lord|Baron|Lady|Dr\.?|Prof\.?|Rev\.?|Saint|St\.?)\b",
    re.IGNORECASE,
)
# Parenthetical expansions   e.g.  "Whitman, Vic (Victor Sargent)"
_PAREN = re.compile(r"\(.*?\)")
# Multiple authors joined by semicolons  "Hackett, Walter ; Megrue, Roi Cooper"
_SEMI  = re.compile(r"\s*;\s*")


def _parse_gutenberg_name(raw: str) -> list[str]:
    """
    Return a list of normalised "Firstname Lastname" strings.
    Handles:
      - "Last, First"
      - "Last, First (Real Name)"
      - "Last, Title First"            (Mrs., Baron, …)
      - "Author1 ; Author2"            (collaborative works)
    """
    parts = _SEMI.split(raw.strip())
    names: list[str] = []
    for part in parts:
        part = _PAREN.sub("", part).strip()     # drop parentheticals
        part = _NOISE.sub("", part).strip()     # drop honorifics

        if "," in part:
            last, *rest = part.split(",", 1)
            first = rest[0].strip() if rest else ""
            name  = f"{first} {last}".strip()
        else:
            name = part.strip()

        # collapse extra whitespace
        name = re.sub(r"\s{2,}", " ", name)
        if name:
            names.append(name)
    return names or [raw]


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  SHARED HTTP HELPER
# ═══════════════════════════════════════════════════════════════════════════════

_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": "GutenbergAuthorScraper/1.0 (research project; python-requests)"
})


def _get(url: str, params: dict | None = None, timeout: int = 10) -> dict | None:
    try:
        r = _SESSION.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        log.debug("GET %s failed: %s", url, exc)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  SOURCE A — WIKIPEDIA (MediaWiki REST + Action API)
# ═══════════════════════════════════════════════════════════════════════════════

_WP_SEARCH = "https://en.wikipedia.org/w/api.php"
_WP_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"


def _year(text: str) -> Optional[int]:
    """Extract the first 4-digit year from a string."""
    m = re.search(r"\b(1[0-9]{3}|20[0-2][0-9])\b", text)
    return int(m.group()) if m else None


def _scrape_wikipedia(name: str) -> AuthorRecord | None:
    """
    Search Wikipedia for `name`, pull the summary page, and parse
    birth/death years and nationality from the extract text.
    """
    # Step 1 – search
    search = _get(_WP_SEARCH, params={
        "action": "query", "list": "search",
        "srsearch": name, "srlimit": 3,
        "format": "json",
    })
    if not search:
        return None

    results = search.get("query", {}).get("search", [])
    if not results:
        return None

    # Pick the first hit that looks like a person (heuristic: title matches name)
    page_title = None
    name_tokens = set(name.lower().split())
    for hit in results:
        title_tokens = set(hit["title"].lower().split())
        if name_tokens & title_tokens:          # at least one token overlaps
            page_title = hit["title"]
            break
    if not page_title:
        page_title = results[0]["title"]        # fall back to top result

    # Step 2 – summary
    summary = _get(_WP_SUMMARY.format(requests.utils.quote(page_title)))
    if not summary or summary.get("type") == "disambiguation":
        return None

    desc    = (summary.get("description") or "").lower()
    extract = summary.get("extract") or ""

    # Discard if clearly not a person
    person_signals = ("author", "writer", "novelist", "poet",
                      "playwright", "journalist", "born", "died")
    if not any(s in desc + extract.lower() for s in person_signals):
        return None

    # ── birth / death ──────────────────────────────────────────────────────
    birth = death = None
    # Infobox dates often appear as "(born 1812)" or "(1812–1870)"
    range_m = re.search(r"\((\d{4})\s*[–\-]\s*(\d{4})\)", extract)
    born_m  = re.search(r"\bborn\b[^;,\.]{0,40}?(\d{4})", extract, re.I)
    died_m  = re.search(r"\bdied\b[^;,\.]{0,40}?(\d{4})", extract, re.I)

    if range_m:
        birth, death = int(range_m.group(1)), int(range_m.group(2))
    else:
        if born_m:  birth = int(born_m.group(1))
        if died_m:  death = int(died_m.group(1))

    # ── nationality ────────────────────────────────────────────────────────
    nationality = None
    nat_m = re.search(
        r"\b(American|British|English|Scottish|Irish|Welsh|French|German|"
        r"Canadian|Australian|Russian|Italian|Spanish|Portuguese|Dutch|"
        r"Polish|Swedish|Norwegian|Danish|Finnish|Belgian|Swiss|Austrian|"
        r"Indian|Japanese|Chinese|Mexican|Brazilian|Argentine|South African|"
        r"New Zealand|Czech|Hungarian|Greek|Romanian|Ukrainian|Israeli|"
        r"Turkish|Egyptian|Nigerian|Kenyan|Ghanaian|Jamaican|Cuban|"
        r"Colombian|Venezuelan|Chilean|Peruvian|Bangladeshi|Pakistani|"
        r"Sri Lankan|Filipino|Indonesian|Malaysian|Thai|Vietnamese|Korean)\b",
        extract, re.I,
    )
    if nat_m:
        nationality = nat_m.group(1).title()

    # ── gender from description or extract ────────────────────────────────
    gender = None
    text_lower = (desc + " " + extract).lower()
    female_signals = ("she ", "her ", " woman ", " actress ", " authoress ",
                      "female writer", "women's")
    male_signals   = ("he ",  "his ", " man ",  " actor ",
                      "male writer", "men's")
    if any(s in text_lower for s in female_signals):
        gender = "female"
    elif any(s in text_lower for s in male_signals):
        gender = "male"

    if not any([birth, death, nationality, gender]):
        return None

    rec = AuthorRecord(raw_name="", normalised_name=name, found=True,
                       source="Wikipedia")
    rec.birth_year   = birth
    rec.death_year   = death
    rec.nationality  = nationality
    rec.gender       = gender
    return rec


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  SOURCE B — WIKIDATA (SPARQL)
# ═══════════════════════════════════════════════════════════════════════════════

_SPARQL = "https://query.wikidata.org/sparql"

_SPARQL_QUERY = """
SELECT ?item ?genderLabel ?citizenshipLabel ?birth ?death WHERE {{
  ?item wikibase:sitelinks ?links .
  ?item rdfs:label "{name}"@en .
  ?item wdt:P31 wd:Q5 .                      # instance of: human
  OPTIONAL {{ ?item wdt:P21 ?gender . }}
  OPTIONAL {{ ?item wdt:P27 ?citizenship . }}
  OPTIONAL {{ ?item wdt:P569 ?birth . }}
  OPTIONAL {{ ?item wdt:P570 ?death . }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
LIMIT 3
"""


def _scrape_wikidata(name: str) -> AuthorRecord | None:
    query = _SPARQL_QUERY.format(name=name.replace('"', '\\"'))
    data  = _get(_SPARQL, params={"query": query, "format": "json"},
                 timeout=15)
    if not data:
        return None

    bindings = data.get("results", {}).get("bindings", [])
    if not bindings:
        return None

    b = bindings[0]

    def val(key):
        return b.get(key, {}).get("value")

    birth = death = None
    raw_b = val("birth")
    raw_d = val("death")
    if raw_b:
        birth = _year(raw_b)
    if raw_d:
        death = _year(raw_d)

    gender_raw      = (val("genderLabel") or "").lower()
    nationality_raw = val("citizenshipLabel")

    gender = None
    if "female" in gender_raw or "woman" in gender_raw:
        gender = "female"
    elif "male" in gender_raw or "man" in gender_raw:
        gender = "male"

    if not any([birth, death, nationality_raw, gender]):
        return None

    rec = AuthorRecord(raw_name="", normalised_name=name, found=True,
                       source="Wikidata")
    rec.birth_year  = birth
    rec.death_year  = death
    rec.nationality = nationality_raw
    rec.gender      = gender
    return rec


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  SOURCE C — OPEN LIBRARY
# ═══════════════════════════════════════════════════════════════════════════════

_OL_SEARCH  = "https://openlibrary.org/search/authors.json"
_OL_AUTHOR  = "https://openlibrary.org/authors/{}.json"


def _scrape_openlibrary(name: str) -> AuthorRecord | None:
    search = _get(_OL_SEARCH, params={"q": name, "limit": 3})
    if not search or search.get("numFound", 0) == 0:
        return None

    docs = search.get("docs", [])
    if not docs:
        return None

    # Find the best-matching doc (most works)
    docs_sorted = sorted(docs, key=lambda d: d.get("work_count", 0), reverse=True)
    doc = docs_sorted[0]
    ol_key = doc.get("key", "")      # e.g. "OL23919A"

    birth = death = gender = nationality = None

    # Top-level fields sometimes present directly in search results
    birth = _year(str(doc.get("birth_date", "")))
    death = _year(str(doc.get("death_date", "")))

    # Fetch full author record for more fields
    if ol_key:
        author = _get(_OL_AUTHOR.format(ol_key))
        if author:
            if not birth:
                birth = _year(str(author.get("birth_date", "")))
            if not death:
                death = _year(str(author.get("death_date", "")))

            bio = str(author.get("bio", author.get("bio", {}))).lower()
            # Gender heuristic from bio
            if any(s in bio for s in ("she ", "her ", "woman ", "female")):
                gender = "female"
            elif any(s in bio for s in ("he ", "his ", "man ", "male")):
                gender = "male"

    if not any([birth, death, nationality, gender]):
        return None

    rec = AuthorRecord(raw_name="", normalised_name=name, found=True,
                       source="OpenLibrary")
    rec.birth_year  = birth
    rec.death_year  = death
    rec.nationality = nationality
    rec.gender      = gender
    return rec


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  MERGE  (fill gaps across sources)
# ═══════════════════════════════════════════════════════════════════════════════

def _merge(primary: AuthorRecord, secondary: AuthorRecord) -> AuthorRecord:
    """Fill None fields in `primary` from `secondary`."""
    for attr in ("gender", "nationality", "birth_year", "death_year"):
        if getattr(primary, attr) is None:
            setattr(primary, attr, getattr(secondary, attr))
    return primary


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  PER-AUTHOR ORCHESTRATION
# ═══════════════════════════════════════════════════════════════════════════════

_SCRAPERS = [
    ("Wikipedia",   _scrape_wikipedia),
    ("Wikidata",    _scrape_wikidata),
    ("OpenLibrary", _scrape_openlibrary),
]


def scrape_author(raw_name: str) -> AuthorRecord:
    """
    Try all sources for a single Gutenberg author string.
    Returns an AuthorRecord (found=False if nothing was found).
    """
    normalised_names = _parse_gutenberg_name(raw_name)
    primary_name     = normalised_names[0]

    combined: AuthorRecord | None = None

    for name in normalised_names:
        for source_label, scraper in _SCRAPERS:
            try:
                result = scraper(name)
            except Exception as exc:
                log.debug("[%s] %s error: %s", source_label, name, exc)
                result = None

            if result:
                result.raw_name        = raw_name
                result.normalised_name = primary_name
                if combined is None:
                    combined = result
                else:
                    combined = _merge(combined, result)

        # If we have a complete record after this name variant, stop early
        if combined and all([
            combined.gender, combined.nationality,
            combined.birth_year, combined.death_year,
        ]):
            break

    if combined is None:
        return AuthorRecord(raw_name=raw_name, normalised_name=primary_name,
                            found=False)

    combined.found = True
    return combined


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  BATCH SCRAPER
# ═══════════════════════════════════════════════════════════════════════════════

def scrape_authors(
    authors,                    # any iterable of raw Gutenberg author strings
    delay: float = 1.0,         # seconds between requests (be polite!)
    log_every: int = 50,
) -> tuple:
    """
    Scrape metadata for every author in `authors`.

    Returns
    -------
    records    : list[dict]   – one dict per author (all fields, including found=False)
    not_found  : set[str]     – raw names for which no information was found
    """
    import pandas as pd

    records: list[dict]   = []
    not_found: set[str]   = set()

    for i, raw in enumerate(authors, 1):
        if i % log_every == 0:
            log.info("Progress: %d / %d", i, len(list(authors)) if hasattr(authors, '__len__') else "?")

        rec = scrape_author(raw)
        records.append(asdict(rec))

        if not rec.found:
            not_found.add(raw)

        time.sleep(delay)

    df = pd.DataFrame(records)
    log.info("Done. Found: %d  |  Not found: %d", len(df) - len(not_found), len(not_found))
    return df, not_found


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  QUICK CLI TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    test_authors = [
        "Twain, Mark",
        "Dickens, Charles",
        "Doyle, Arthur Conan",
        "Oliphant, Mrs. (Margaret)",
        "Oppenheim, E. Phillips (Edward Phillips)",
        "Hackett, Walter ; Megrue, Roi Cooper",
        "Strachan, A. A.",                         # likely not found
        "Trench, Herbert",                          # likely not found
    ]

    for raw in test_authors:
        rec = scrape_author(raw)
        status = "✓" if rec.found else "✗"
        print(
            f"{status}  {rec.normalised_name:<35}"
            f"  gender={rec.gender or '?':<8}"
            f"  nat={rec.nationality or '?':<15}"
            f"  {rec.birth_year or '?'}–{rec.death_year or '?'}"
            f"  [{rec.source or 'none'}]"
        )
        time.sleep(1.0)