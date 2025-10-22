import os, re, hashlib
from typing import List, Optional, Dict, Any
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from pypdf import PdfReader
import docx
import markdown
from bs4 import BeautifulSoup

EMB = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

def _file_to_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if ext in [".txt", ".md"]:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    if ext in [".docx"]:
        d = docx.Document(path)
        return "\n".join(p.text for p in d.paragraphs)
    if ext in [".html", ".htm"]:
        html = open(path, "r", encoding="utf-8", errors="ignore").read()
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text(" ")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def _chunk(text: str, size: int = 800, overlap: int = 120) -> List[str]:
    words = re.split(r"\s+", text)
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i:i+size])
        chunks.append(chunk.strip())
        i += size - overlap
    return [c for c in chunks if c]

class RAG:
    def __init__(self, persist_dir: str = "./chroma"):
        self.client = chromadb.PersistentClient(path=persist_dir)

    def _collection(self, namespace: Optional[str] = None):
        name = f"kb_{(namespace or 'default')}"
        return self.client.get_or_create_collection(
            name=name, embedding_function=EMB
        )

    def ingest(self, paths: List[str], namespace: Optional[str] = None) -> int:
        col = self._collection(namespace)
        total = 0
        for p in paths:
            text = _file_to_text(p)
            for idx, ch in enumerate(_chunk(text)):
                uid = hashlib.sha1(f"{p}::{idx}".encode()).hexdigest()
                col.add(ids=[uid], documents=[ch], metadatas=[{"source": os.path.basename(p)}])
                total += 1
        return total

    def retrieve(self, query: str, top_k: int = 5, namespace: Optional[str] = None) -> List[Dict[str, Any]]:
        col = self._collection(namespace)
        res = col.query(query_texts=[query], n_results=top_k, include=["documents", "metadatas", "distances"])
        docs = []
        if res and res.get("documents"):
            docs_raw = res["documents"][0]
            metas = res["metadatas"][0]
            dists = res.get("distances", [[0]*len(docs_raw)])[0]
            sims = [1/(1+d) for d in dists]
            for ch, meta, sim in zip(docs_raw, metas, sims):
                docs.append({"text": ch, "source": meta.get("source", "kb"), "score": sim})
        return docs
