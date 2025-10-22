import httpx
from bs4 import BeautifulSoup

async def fetch_page_text(url: str, timeout: int = 25) -> str:
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        r = await client.get(url, headers={"User-Agent":"Mozilla/5.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        # Keep basic text; caller can pass it to RAG
        return soup.get_text(" ", strip=True)
