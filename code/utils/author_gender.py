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
from dataclasses import dataclass, asdict
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
    rate_limited_sources: Optional[str] = None


class RateLimitedError(Exception):
    """Raised when an API asks us to slow down."""

    def __init__(self, url: str, retry_after: int):
        super().__init__(f"Rate limited by {url}; retry after {retry_after}s")
        self.url = url
        self.retry_after = retry_after


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  NAME NORMALISATION
# ═══════════════════════════════════════════════════════════════════════════════

# Titles/honorifics we leverage to infer gender
_HONOR = re.compile(r"\b(Mrs?\.?|Sir|Lord|Lady|Baroness|Baron)\s*", re.I)

# Gender associated with honorifics
_HONOR_GENDER = {
    "mrs": "female",
    "lady": "female",
    "baroness": "female",
    "mr": "male",
    "lord": "male",
    "baron": "male",
    "sir": "male",
}

# Parenthetical expansions of the First Name e.g.  "Oppenheim, E. Phillips (Edward Phillips)"
_PAREN = re.compile(r"\(([^)]+)\)")

# Single nationalities relevant to English-speaking authors, 1870-1920
_NATIONALITIES = (
    "American", "British", "English", "Scottish", "Irish", "Welsh",
    "Canadian", "Australian", "New Zealand",
    # Common immigrant/expatriate authors of the period
    "Polish", "Russian", "German", "French", "Italian",
)

# Common dual-nationality patterns for the period
_DUAL = (
    r"Anglo-Irish", r"Anglo-Indian", r"Anglo-American",
    r"Scots-Irish", r"Scottish-American", r"Irish-American",
    r"British-American", r"American-British", r"Canadian-American",
    r"Australian-American",
)

_NAT_PATTERN = re.compile(
    r"\b(" + "|".join(_DUAL + _NATIONALITIES) + r")\b",
    re.I,
)



def _clean_spaces(text: str) -> str:
    return re.sub(r"\s{2,}", " ", text).strip()


def _name_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower()))


def _name_match_score(query: str, candidate: str) -> tuple[int, float]:
    query_tokens = _name_tokens(query)
    candidate_tokens = _name_tokens(candidate)
    if not query_tokens or not candidate_tokens:
        return 0, 0.0

    overlap = query_tokens & candidate_tokens
    return len(overlap), len(overlap) / len(query_tokens)


def _parse_gutenberg_name(raw: str) -> tuple[str, Optional[str]]:
    """
    Return a normalised "Firstname Lastname" string and leverage honorific mentions to infer gender.
    Handles:
      - "Last, First"
      - "Last, First (Real Name)"
      - "Last, Title First"            (Mrs., Baron, …)
      - already-normalised names without commas
    
    Return:
      - ("First Last", inferred_gender)
    """
    raw = _clean_spaces(str(raw))
    if not raw:
        return "", None
    if ";" in raw:
        raw = raw.split(";", 1)[0].strip()

    if "," in raw:
        last, first = raw.split(",", 1)
        last, first = last.strip(), first.strip()
    else:
        last, first = "", raw
    
    match = _HONOR.search(first)
    gender = None
    if match:
        honorific = match.group(1).rstrip(".").lower()
        gender = _HONOR_GENDER.get(honorific)
        first = _HONOR.sub("", first, count=1).strip()
        
    first_p = re.search(_PAREN, first)
    first = first_p.group(1).strip() if first_p else first

    norm_name = _clean_spaces(f"{first} {last}" if last else first)
            
    return norm_name, gender


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  SHARED HTTP HELPER
# ═══════════════════════════════════════════════════════════════════════════════

_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": "ProjectGutenbergAuthorScraper/1.0 (student research project; contact: eglantine.vialaneix@epfl.ch)"
})


def _get(
    url: str,
    params: dict | None = None,
    timeout: int = 10,
) -> dict | None:
    try:
        r = _SESSION.get(url, params=params, timeout=timeout)
        if r.status_code == 429:
            retry_after = r.headers.get("Retry-After")
            delay = int(retry_after) if retry_after and retry_after.isdigit() else 30
            raise RateLimitedError(url, delay)

        r.raise_for_status()
        return r.json()
    except RateLimitedError:
        raise
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


def _date_years_from_text(text: str) -> tuple[Optional[int], Optional[int]]:
    """Extract likely birth/death years from compact encyclopedia prose."""
    birth = death = None

    range_m = re.search(
        r"\((?:[^)]{0,80}?)(1[0-9]{3}|20[0-2][0-9])\s*[–-]\s*"
        r"(?:[^)]{0,80}?)(1[0-9]{3}|20[0-2][0-9])(?:[^)]{0,80}?)\)",
        text,
    )
    born_m = re.search(r"\bborn\b[^;.]{0,120}?\b(1[0-9]{3}|20[0-2][0-9])\b", text, re.I)
    died_m = re.search(r"\bdied\b[^;.]{0,120}?\b(1[0-9]{3}|20[0-2][0-9])\b", text, re.I)

    if range_m:
        birth, death = int(range_m.group(1)), int(range_m.group(2))
    if born_m:
        birth = int(born_m.group(1))
    if died_m:
        death = int(died_m.group(1))

    return birth, death


def _extract_nationality(text: str) -> Optional[str]:
    nationalities: list[str] = []
    for m in _NAT_PATTERN.finditer(text):
        nat = m.group(1).title()
        if nat not in nationalities:
            nationalities.append(nat)
    return ", ".join(nationalities) if nationalities else None


def _infer_gender_from_text(text: str, fallback: Optional[str] = None) -> Optional[str]:
    """Infer binary gender from pronouns/titles when no structured value exists."""
    if fallback:
        return fallback

    text = f" {text.lower()} "
    female_patterns = (
        r"\bshe\b", r"\bher\b", r"\bhers\b", r"\bwoman\b",
        r"\bfemale\b", r"\bmrs\b", r"\bmiss\b", r"\blady\b",
    )
    male_patterns = (
        r"\bhe\b", r"\bhim\b", r"\bhis\b", r"\bman\b",
        r"\bmale\b", r"\bmr\b", r"\bsir\b", r"\blord\b",
    )
    female_count = sum(len(re.findall(pattern, text)) for pattern in female_patterns)
    male_count = sum(len(re.findall(pattern, text)) for pattern in male_patterns)

    if female_count > male_count:
        return "female"
    if male_count > female_count:
        return "male"
    return None


def _scrape_wikipedia(name: str, inferred_gender: Optional[str] = None) -> AuthorRecord | None:
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
    for hit in results:
        _, ratio = _name_match_score(name, hit["title"])
        if ratio >= 0.5:
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
    text = f"{desc} {extract}"

    # Discard if clearly not a person
    person_signals = ("author", "writer", "novelist", "poet",
                      "playwright", "journalist", "born", "died")
    if not any(s in text.lower() for s in person_signals):
        return None

    # ── birth / death ──────────────────────────────────────────────────────
    birth, death = _date_years_from_text(text)

    # ── nationality ────────────────────────────────────────────────────────
    nationality = _extract_nationality(text)

    # ── gender from description or extract ────────────────────────────────
    gender = _infer_gender_from_text(text, fallback=inferred_gender)

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
SELECT ?item ?genderLabel ?citizenshipLabel ?birth ?death ?links WHERE {{
  ?item wikibase:sitelinks ?links .
  ?item wdt:P31 wd:Q5 .                      # instance of: human
  {{
    ?item rdfs:label "{name}"@en .
  }}
  UNION
  {{
    ?item skos:altLabel "{name}"@en .
  }}
  OPTIONAL {{ ?item wdt:P21 ?gender . }}
  OPTIONAL {{ ?item wdt:P27 ?citizenship . }}
  OPTIONAL {{ ?item wdt:P569 ?birth . }}
  OPTIONAL {{ ?item wdt:P570 ?death . }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
ORDER BY DESC(?links)
LIMIT 10
"""


def _scrape_wikidata(name: str, inferred_gender: Optional[str] = None) -> AuthorRecord | None:
    query = _SPARQL_QUERY.format(name=name.replace('"', '\\"'))
    data  = _get(_SPARQL, params={"query": query, "format": "json"},
                 timeout=15)
    if not data:
        return None

    bindings = data.get("results", {}).get("bindings", [])
    if not bindings:
        return None

    birth = death = None
    nationalities: list[str] = []
    gender_raw = ""

    for binding in bindings:
        item = binding.get("item", {}).get("value")
        if item != bindings[0].get("item", {}).get("value"):
            break

        raw_b = binding.get("birth", {}).get("value")
        raw_d = binding.get("death", {}).get("value")
        if raw_b and birth is None:
            birth = _year(raw_b)
        if raw_d and death is None:
            death = _year(raw_d)

        gender_raw = gender_raw or (binding.get("genderLabel", {}).get("value") or "").lower()
        nationality_raw = binding.get("citizenshipLabel", {}).get("value")
        if nationality_raw and nationality_raw not in nationalities:
            nationalities.append(nationality_raw)

    gender = inferred_gender
    if "female" in gender_raw or "woman" in gender_raw:
        gender = "female"
    elif "male" in gender_raw or "man" in gender_raw:
        gender = "male"

    nationality = ", ".join(nationalities) if nationalities else None

    if not any([birth, death, nationality, gender]):
        return None

    rec = AuthorRecord(raw_name="", normalised_name=name, found=True,
                       source="Wikidata")
    rec.birth_year  = birth
    rec.death_year  = death
    rec.nationality = nationality
    rec.gender      = gender
    return rec


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  SOURCE C — OPEN LIBRARY
# ═══════════════════════════════════════════════════════════════════════════════

_OL_SEARCH  = "https://openlibrary.org/search/authors.json"
_OL_AUTHOR  = "https://openlibrary.org/authors/{}.json"


def _openlibrary_text(value) -> str:
    if isinstance(value, dict):
        return str(value.get("value") or "")
    return str(value or "")


def _openlibrary_doc_score(name: str, doc: dict) -> tuple[float, int, int, int]:
    _, ratio = _name_match_score(name, doc.get("name", ""))
    exact_name = int(_clean_spaces(doc.get("name", "")).lower() == name.lower())
    has_life_dates = int(bool(doc.get("birth_date") or doc.get("death_date")))
    return ratio, exact_name, has_life_dates, doc.get("work_count", 0)


def _scrape_openlibrary(name: str, inferred_gender: Optional[str] = None) -> AuthorRecord | None:
    search = _get(_OL_SEARCH, params={"q": name, "limit": 3})
    if not search or search.get("numFound", 0) == 0:
        return None

    docs = search.get("docs", [])
    if not docs:
        return None

    # Find the best-matching docs. Open Library often has high-work duplicate
    # records with sparse metadata, so usable dates outrank work_count.
    docs_sorted = sorted(
        docs,
        key=lambda d: _openlibrary_doc_score(name, d),
        reverse=True,
    )

    for doc in docs_sorted:
        ol_key = doc.get("key", "")      # e.g. "OL23919A"

        birth = death = gender = nationality = None

        # Top-level fields sometimes present directly in search results
        birth = _year(str(doc.get("birth_date", "")))
        death = _year(str(doc.get("death_date", "")))
        bio_parts = [
            doc.get("name", ""),
            doc.get("top_work", ""),
        ]

        # Fetch full author record for more fields
        if ol_key:
            author = _get(_OL_AUTHOR.format(ol_key))
            if author:
                if not birth:
                    birth = _year(str(author.get("birth_date", "")))
                if not death:
                    death = _year(str(author.get("death_date", "")))

                bio_parts.extend([
                    _openlibrary_text(author.get("bio")),
                    _openlibrary_text(author.get("personal_name")),
                    _openlibrary_text(author.get("alternate_names")),
                    _openlibrary_text(author.get("wikipedia")),
                ])

        bio = " ".join(bio_parts)
        nationality = _extract_nationality(bio)
        gender = _infer_gender_from_text(bio, fallback=inferred_gender)

        if not any([birth, death, nationality, gender]):
            continue

        rec = AuthorRecord(raw_name="", normalised_name=name, found=True,
                           source="OpenLibrary")
        rec.birth_year  = birth
        rec.death_year  = death
        rec.nationality = nationality
        rec.gender      = gender
        return rec

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  MERGE  (fill gaps across sources)
# ═══════════════════════════════════════════════════════════════════════════════

def _merge(primary: AuthorRecord, secondary: AuthorRecord) -> AuthorRecord:
    """Fill None fields in `primary` from `secondary`."""
    filled_from_secondary = False
    for attr in ("gender", "nationality", "birth_year", "death_year"):
        if getattr(primary, attr) is None:
            value = getattr(secondary, attr)
            if value is not None:
                setattr(primary, attr, value)
                filled_from_secondary = True

    if filled_from_secondary and secondary.source:
        sources = [s.strip() for s in (primary.source or "").split(",") if s.strip()]
        if secondary.source not in sources:
            sources.append(secondary.source)
        primary.source = ", ".join(sources)

    return primary


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  PER-AUTHOR ORCHESTRATION
# ═══════════════════════════════════════════════════════════════════════════════

_SCRAPERS = [
    ("Wikipedia",   _scrape_wikipedia),
    ("OpenLibrary", _scrape_openlibrary),
    ("Wikidata",    _scrape_wikidata),
]


def scrape_author(raw_name: str) -> AuthorRecord:
    """
    Try all sources for a single Gutenberg author string.
    Returns an AuthorRecord (found=False if nothing was found).
    """
    normalised_name, inferred_gender = _parse_gutenberg_name(raw_name)
    if not normalised_name:
        return AuthorRecord(raw_name=raw_name, normalised_name="", found=False)

    combined: AuthorRecord | None = None
    rate_limited_sources: list[str] = []

    for source_label, scraper in _SCRAPERS:
        try:
            result = scraper(normalised_name, inferred_gender)
        except RateLimitedError as exc:
            rate_limited_sources.append(source_label)
            log.warning(
                "[%s] rate limited for %s; skipping this source for now "
                "(retry after %ss)",
                source_label,
                normalised_name,
                exc.retry_after,
            )
            result = None
        except Exception as exc:
            log.debug("[%s] %s error: %s", source_label, normalised_name, exc)
            result = None

        if result:
            result.raw_name        = raw_name
            result.normalised_name = normalised_name
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
        rate_limited = ", ".join(rate_limited_sources) or None
        return AuthorRecord(raw_name=raw_name, normalised_name=normalised_name,
                            found=False, rate_limited_sources=rate_limited)

    combined.found = True
    combined.rate_limited_sources = ", ".join(rate_limited_sources) or None
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
    authors = list(authors)

    for i, raw in enumerate(authors, 1):
        if i % log_every == 0:
            log.info("Progress: %d / %d", i, len(authors))

        rec = scrape_author(raw)
        records.append(asdict(rec))

        if not rec.found:
            not_found.add(raw)

        time.sleep(delay)

    df = pd.DataFrame(records)
    log.info("Done. Found: %d  |  Not found: %d", len(df) - len(not_found), len(not_found))
    return df, not_found


# # ═══════════════════════════════════════════════════════════════════════════════
# # 9.  QUICK CLI TEST
# # ═══════════════════════════════════════════════════════════════════════════════

# if __name__ == "__main__":
#     test_authors = [
#         "Twain, Mark",
#         "Dickens, Charles",
#         "Doyle, Arthur Conan",
#         "Oliphant, Mrs. (Margaret)",
#         "Oppenheim, E. Phillips (Edward Phillips)",
#         "Hackett, Walter ; Megrue, Roi Cooper",
#         "Strachan, A. A.",                         # likely not found
#         "Trench, Herbert",                          # likely not found
#     ]

#     for raw in test_authors:
#         rec = scrape_author(raw)
#         status = "✓" if rec.found else "✗"
#         print(
#             f"{status}  {rec.normalised_name:<35}"
#             f"  gender={rec.gender or '?':<8}"
#             f"  nat={rec.nationality or '?':<15}"
#             f"  {rec.birth_year or '?'}–{rec.death_year or '?'}"
#             f"  [{rec.source or 'none'}]"
#         )
#         time.sleep(1.0)
