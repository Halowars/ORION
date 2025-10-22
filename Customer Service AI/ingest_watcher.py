import os, json, time, hashlib, asyncio, pathlib, re
from typing import Dict, Tuple, List
import yaml

from .rag import RAG
from .web_utils import fetch_page_text

CONFIG = yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))
ING = CONFIG["ingest"]
STATE_FILE = os.path.join(ING["root"], ".ingested_state.json")

DOC_EXTS = {".pdf",".docx",".txt",".md",".html",".htm"}

def _load_state() -> Dict[str, Dict]:
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        return json.load(open(STATE_FILE, "r", encoding="utf-8"))
    except Exception:
        return {}

def _save_state(state: Dict[str, Dict]):
    os.makedirs(ING["root"], exist_ok=True)
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_FILE)

def _hash_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024*1024)
            if not b: break
            h.update(b)
    return h.hexdigest()

async def _ingest_files_in_dir(rag: RAG, dir_path: str, namespace: str, state: Dict[str, Dict]) -> int:
    count = 0
    for root, _, files in os.walk(dir_path):
        for name in files:
            p = os.path.join(root, name)
            ext = os.path.splitext(name)[1].lower()
            if ext not in DOC_EXTS:
                continue
            key = f"file::{namespace}::{os.path.abspath(p)}"
            sha = _hash_file(p)
            entry = state.get(key)
            if entry and entry.get("sha") == sha:
                continue
            rag.ingest([p], namespace=namespace)
            state[key] = {"sha": sha, "ts": time.time()}
            count += 1
    return count

async def _ingest_links_in_dir(rag: RAG, dir_path: str, namespace: str, state: Dict[str, Dict]) -> int:
    count = 0
    cache_dir = ING["links_cache_dir"]
    os.makedirs(cache_dir, exist_ok=True)
    for root, _, files in os.walk(dir_path):
        for name in files:
            if not name.lower().endswith(".txt"):
                continue
            p = os.path.join(root, name)
            key_file = f"linksfile::{namespace}::{os.path.abspath(p)}"
            mtime = os.path.getmtime(p)
            entry = state.get(key_file)
            # Reprocess file if modified
            if not entry or entry.get("mtime") != mtime:
                # process each line URL
                try:
                    urls = [ln.strip() for ln in open(p, "r", encoding="utf-8").read().splitlines() if ln.strip()]
                except Exception:
                    urls = []
                for u in urls:
                    if not re.match(r"^https?://", u): 
                        continue
                    key = f"url::{namespace}::{u}"
                    if key in state and state[key].get("ok"):
                        continue
                    try:
                        text = await fetch_page_text(u)
                        sha = hashlib.sha1(u.encode()).hexdigest()
                        cache_path = os.path.join(cache_dir, f"{sha}.txt")
                        open(cache_path, "w", encoding="utf-8").write(text)
                        rag.ingest([cache_path], namespace=namespace)
                        state[key] = {"ok": True, "ts": time.time(), "cache": cache_path}
                        count += 1
                    except Exception as e:
                        state[key] = {"ok": False, "error": str(e), "ts": time.time()}
                state[key_file] = {"mtime": mtime, "ts": time.time()}
    return count

async def scan_all(rag: RAG) -> Dict[str, int]:
    state = _load_state()
    totals = {"docs":0, "links":0}
    uploads_dir = ING["uploads_dir"]
    links_dir = ING["links_dir"]
    messages_dir = ING["messages_dir"]

    # Namespaces = immediate subfolders; default namespace if files placed directly
    def ns_dirs(base_dir):
        if not os.path.exists(base_dir):
            return []
        items = []
        for entry in os.scandir(base_dir):
            if entry.is_dir():
                items.append((entry.path, os.path.basename(entry.path)))
        # also include base dir as default namespace bucket
        items.append((base_dir, ING.get("namespace_default","default")))
        return items

    for dir_path, ns in ns_dirs(uploads_dir):
        totals["docs"] += await _ingest_files_in_dir(rag, dir_path, ns, state)

    for dir_path, ns in ns_dirs(messages_dir):
        totals["docs"] += await _ingest_files_in_dir(rag, dir_path, ns, state)

    for dir_path, ns in ns_dirs(links_dir):
        totals["links"] += await _ingest_links_in_dir(rag, dir_path, ns, state)

    _save_state(state)
    return totals

async def run_watcher(rag: RAG, interval: int):
    while True:
        try:
            await scan_all(rag)
        except Exception:
            pass
        await asyncio.sleep(max(3, interval))
