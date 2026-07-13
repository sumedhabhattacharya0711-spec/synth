"""
mcp_fallback.py — fallback paper search when librarian_v2's primary search
comes up empty for a hypothesis.

ARCHITECTURE NOTE: Originally this spawned backend/mcp_server/research_server.py
as a subprocess over MCP stdio transport. On 512MB hosts (Render free tier),
the second Python process OOM-kills the service. This version keeps the exact
same ReAct agent architecture and the same three tools — search_arxiv,
search_semantic_scholar, get_paper_citations — but defines them as in-process
LangChain tools instead of MCP tools. Same agent behavior, same search
capability, zero subprocess.

The MCP server (research_server.py) still exists for the demo/standalone use
case; this module just doesn't spawn it.
"""

import os
import json
import time
from pathlib import Path

import arxiv
import requests
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from .config import big, safe_parse_json, S2_API_KEY, S2_FIELDS

# ── S2 helpers (mirrors research_server.py's auth/cache/rate-limit) ─────────

S2_BASE = "https://api.semanticscholar.org/graph/v1"
S2_CACHE = Path(os.environ.get("DATA_DIR", "store")) / "cache" / "s2"
S2_CACHE.mkdir(parents=True, exist_ok=True)

_last_call = [0.0]

def _rate_limit(delay=0.2):
    elapsed = time.time() - _last_call[0]
    if elapsed < delay:
        time.sleep(delay - elapsed)
    _last_call[0] = time.time()

def _s2_get(endpoint, params=None):
    cache_key = endpoint.replace("/", "_") + "_" + json.dumps(params or {}, sort_keys=True)
    cache_key = "".join(c for c in cache_key if c.isalnum() or c in "_-")[:150]
    cache_path = S2_CACHE / f"{cache_key}.json"
    if cache_path.exists():
        raw = cache_path.read_text()
        if raw.strip() == "null":
            return None
        return json.loads(raw)
    _rate_limit()
    try:
        headers = {"x-api-key": S2_API_KEY} if S2_API_KEY else {}
        r = requests.get(f"{S2_BASE}{endpoint}", params=params,
                         headers=headers, timeout=15)
        if r.status_code == 429:
            time.sleep(3)
            r = requests.get(f"{S2_BASE}{endpoint}", params=params,
                             headers=headers, timeout=15)
        if r.status_code != 200:
            cache_path.write_text("null")
            return None
        data = r.json()
        cache_path.write_text(json.dumps(data))
        return data
    except Exception:
        cache_path.write_text("null")
        return None


# ── the same three tools, in-process ─────────────────────────────────────────

@tool
def search_arxiv(query: str, max_results: int = 5) -> list:
    """Search arXiv for research papers matching a query.
    Returns titles, abstracts, years, and PDF URLs."""
    try:
        results = list(arxiv.Client().results(
            arxiv.Search(query=query, max_results=max_results,
                         sort_by=arxiv.SortCriterion.Relevance)))
        return [{
            "title": r.title,
            "abstract": r.summary[:500],
            "year": r.published.year,
            "pdf_url": r.pdf_url,
            "arxiv_id": r.entry_id.split("/")[-1],
        } for r in results]
    except Exception as e:
        return [{"error": str(e)}]


@tool
def search_semantic_scholar(query: str, limit: int = 5) -> list:
    """Search Semantic Scholar for papers with citation data."""
    try:
        data = _s2_get("/paper/search",
                       {"query": query, "limit": limit, "fields": S2_FIELDS})
        papers = data.get("data") or [] if data else []
        return [{
            "title": p.get("title"),
            "abstract": (p.get("abstract") or "")[:500],
            "year": p.get("year"),
            "citations": p.get("citationCount", 0),
        } for p in papers]
    except Exception as e:
        return [{"error": str(e)}]


@tool
def get_paper_citations(s2_paper_id: str, limit: int = 10) -> list:
    """Get papers that cite a given paper (forward citation chase)."""
    try:
        data = _s2_get(f"/paper/{s2_paper_id}/citations",
                       {"fields": S2_FIELDS, "limit": limit})
        citations = data.get("data") or [] if data else []
        return [{"title": c["citingPaper"].get("title"),
                 "year": c["citingPaper"].get("year")}
                for c in citations if c.get("citingPaper", {}).get("title")]
    except Exception as e:
        return [{"error": str(e)}]


TOOLS = [search_arxiv, search_semantic_scholar, get_paper_citations]

FALLBACK_PROMPT = """You are searching for research papers relevant to this hypothesis,
which the primary literature search failed to find enough for.

Problem statement context: {ps_context}
Hypothesis: {hypothesis}

Use the available tools to search arXiv and Semantic Scholar. If a direct
search on the hypothesis wording comes back empty, try broader or adjacent
terms — don't just repeat the same query. Stop once you find 3-5 genuinely
relevant papers, or after a reasonable number of attempts if nothing turns up.

Return ONLY JSON in this exact shape (no other text):
{{"papers": [
  {{"title": "...", "abstract": "...", "year": 2023, "pdf_url": "..."}}
]}}
If you find nothing relevant, return {{"papers": []}}."""


def run_mcp_fallback(hypothesis_text: str, ps_context: str = "") -> list:
    """
    Synchronous entry point — same signature and return shape as before.
    ReAct agent with in-process tools, no subprocess, no extra RAM.

    Returns [] on any failure rather than raising, since this is itself
    a fallback path.
    """
    try:
        agent = create_react_agent(big, TOOLS)

        prompt = FALLBACK_PROMPT.format(
            ps_context=ps_context[:300],
            hypothesis=hypothesis_text)

        result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
        final_message = result["messages"][-1].content

        parsed = safe_parse_json(final_message)
        papers = parsed.get("papers", []) if parsed else []

        for p in papers:
            p["source"] = "mcp_fallback"
            p["source_hypothesis"] = hypothesis_text[:80]

        return papers
    except Exception as e:
        print(f"  fallback agent error: {e}")
        return []
