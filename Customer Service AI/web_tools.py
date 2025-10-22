# app/web_tools.py
import httpx
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from typing import Tuple, List

async def web_answer(query: str) -> Tuple[str, List[str]]:
    """
    Try to fetch a couple of snippets for a query.
    On ANY error (incl. rate limits), return empty context & no citations.
    """
    try:
        urls: List[str] = []
        # keep it small to avoid rate limits
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=1, backend="api"):
                if "href" in r:
                    urls.append(r["href"])
                if len(urls) >= 1:
                    break

        if not urls:
            return "", []

        contexts, citations = [], []
        async with httpx.AsyncClient(timeout=15) as client:
            for u in urls:
                try:
                    resp = await client.get(u, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0"})
                    soup = BeautifulSoup(resp.text, "html.parser")
                    text = soup.get_text(" ", strip=True)
                    if text:
                        contexts.append(text[:1200])
                        citations.append(u)
                except Exception:
                    continue

        return ("\n\n".join(contexts), citations) if contexts else ("", [])
    except Exception:
        return "", []
