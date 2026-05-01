from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path


OPENALEX_URL = "https://api.openalex.org/works"

DEFAULT_QUERIES = [
    "protein conformational ensemble generation",
    "protein dynamics generative model conformational ensemble",
    "diffusion model protein conformations molecular dynamics",
    "flow matching protein conformational dynamics",
    "BioEmu protein ensemble",
]


def abstract_from_inverted_index(index: object) -> str:
    if not isinstance(index, dict):
        return ""
    positions: list[tuple[int, str]] = []
    for word, offsets in index.items():
        if not isinstance(word, str) or not isinstance(offsets, list):
            continue
        for offset in offsets:
            if isinstance(offset, int):
                positions.append((offset, word))
    return " ".join(word for _, word in sorted(positions))


def fetch_openalex(query: str, per_page: int, api_key: str | None, mailto: str | None) -> list[dict[str, object]]:
    params = {
        "search": query,
        "per-page": str(per_page),
        "select": "id,doi,title,display_name,publication_year,authorships,primary_location,abstract_inverted_index,cited_by_count",
    }
    if mailto:
        params["mailto"] = mailto
    if api_key:
        params["api_key"] = api_key
    url = OPENALEX_URL + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "protein-agent-tiny/0.1"})
    with urllib.request.urlopen(req, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))
    results = payload.get("results", [])
    return results if isinstance(results, list) else []


def normalize_work(work: dict[str, object], query: str) -> dict[str, object]:
    authors = []
    for authorship in work.get("authorships", []) or []:
        if not isinstance(authorship, dict):
            continue
        author = authorship.get("author")
        if isinstance(author, dict) and author.get("display_name"):
            authors.append(author["display_name"])
    location = work.get("primary_location")
    landing_page = ""
    if isinstance(location, dict):
        source = location.get("source")
        landing_page = str(location.get("landing_page_url") or "")
        if not landing_page and isinstance(source, dict):
            landing_page = str(source.get("homepage_url") or "")
    title = str(work.get("title") or work.get("display_name") or "")
    abstract = abstract_from_inverted_index(work.get("abstract_inverted_index"))
    return {
        "query": query,
        "title": title,
        "year": work.get("publication_year"),
        "authors": authors[:6],
        "doi": work.get("doi"),
        "openalex_id": work.get("id"),
        "url": landing_page or work.get("doi") or work.get("id"),
        "cited_by_count": work.get("cited_by_count"),
        "abstract": abstract[:1200],
    }


def is_relevant(paper: dict[str, object]) -> bool:
    text = f"{paper.get('title') or ''} {paper.get('abstract') or ''}".lower()
    if "protein" not in text:
        return False
    signals = ("conformation", "conformational", "ensemble", "dynamics", "diffusion", "generative", "molecular dynamics")
    return any(signal in text for signal in signals)


def collect_literature(
    workspace: Path,
    queries: list[str] | None = None,
    per_query: int = 3,
) -> dict[str, object]:
    api_key = os.environ.get("OPENALEX_API_KEY") or os.environ.get("OPENALEX_KEY")
    mailto = os.environ.get("OPENALEX_EMAIL")
    queries = queries or DEFAULT_QUERIES
    papers: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    seen: set[str] = set()

    for query in queries:
        try:
            for work in fetch_openalex(query, per_query, api_key, mailto):
                if not isinstance(work, dict):
                    continue
                paper = normalize_work(work, query)
                if not is_relevant(paper):
                    continue
                key = str(paper.get("doi") or paper.get("openalex_id") or paper.get("title"))
                if not key or key in seen:
                    continue
                seen.add(key)
                papers.append(paper)
        except Exception as exc:
            errors.append({"query": query, "error": str(exc)})

    payload = {
        "source": "OpenAlex",
        "timestamp_unix": int(time.time()),
        "queries": queries,
        "paper_count": len(papers),
        "papers": papers,
        "errors": errors,
    }
    (workspace / "literature_sources.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (workspace / "literature_review.md").write_text(render_literature_review(payload), encoding="utf-8")
    return payload


def render_literature_review(payload: dict[str, object]) -> str:
    lines = [
        "# Literature Review",
        "",
        "Source: OpenAlex",
        "",
        "Use these papers only for high-level architecture inspiration. Do not use competition MD trajectories, crystal structures, or NMR ensembles as inputs.",
        "",
        "## Retrieved Papers",
        "",
    ]
    papers = payload.get("papers", [])
    if not isinstance(papers, list) or not papers:
        lines.extend([
            "No papers were retrieved. Continue with the sequence-only fallback and record this retrieval failure in the audit log.",
            "",
        ])
    else:
        for idx, paper in enumerate(papers, start=1):
            if not isinstance(paper, dict):
                continue
            authors = ", ".join(str(a) for a in paper.get("authors", [])[:3])
            abstract = str(paper.get("abstract") or "").strip()
            lines.extend([
                f"### {idx}. {paper.get('title')}",
                "",
                f"- Year: `{paper.get('year')}`",
                f"- Query: `{paper.get('query')}`",
                f"- Authors: {authors or 'unknown'}",
                f"- URL/DOI: {paper.get('url') or paper.get('doi') or paper.get('openalex_id')}",
                f"- Cited by: `{paper.get('cited_by_count')}`",
            ])
            if abstract:
                lines.append(f"- Abstract summary input: {abstract[:700]}")
            lines.append("")
    errors = payload.get("errors", [])
    if isinstance(errors, list) and errors:
        lines.extend(["## Retrieval Errors", ""])
        for error in errors:
            if isinstance(error, dict):
                lines.append(f"- `{error.get('query')}`: {error.get('error')}")
        lines.append("")
    return "\n".join(lines)
