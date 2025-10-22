from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel
from typing import List, Optional, Iterable, Tuple
import os, tempfile, asyncio, yaml, sys
from pathlib import Path

# --- FS SAFETY GUARDS (skip systemd-private and Proton z:; don't follow symlinks) ---
def _is_bad_path(p: str) -> bool:
    # Normalize separators; keep case-sensitive since we're on Linux
    s = p
    # Skip systemd PrivateTmp dirs and Proton/Wine z: mapping to /
    if "systemd-private-" in s:
        return True
    if "/steamapps/compatdata/" in s and "/pfx/dosdevices/z:" in s:
        return True
    return False

def _safe_walk(top: str,
               topdown: bool = True,
               onerror=None,
               followlinks: bool = False) -> Iterable[Tuple[str, list, list]]:
    """
    Drop-in replacement for os.walk that:
      - never follows symlinks (overrides followlinks=False)
      - prunes 'systemd-private-*' and Proton '.../pfx/dosdevices/z:' paths
      - ignores PermissionError / FileNotFoundError during traversal
    """
    # Hard-disable link following (critical for Proton's z:)
    followlinks = False

    try:
        top = os.fspath(top)
    except Exception:
        top = str(top)

    if _is_bad_path(top):
        return

    try:
        with os.scandir(top) as it:
            dirs = []
            nondirs = []
            for entry in it:
                name = entry.name
                path = os.path.join(top, name)

                # Skip broken/blocked things early without stat
                if _is_bad_path(path):
                    continue
                try:
                    # Never follow symlinks
                    if entry.is_symlink():
                        # treat as file-ish; do not descend
                        try:
                            if entry.is_file(follow_symlinks=False):
                                nondirs.append(name)
                            # else ignore symlinked dirs entirely
                        except (PermissionError, FileNotFoundError):
                            continue
                    elif entry.is_dir(follow_symlinks=False):
                        dirs.append(name)
                    else:
                        nondirs.append(name)
                except (PermissionError, FileNotFoundError):
                    # Can't stat; skip
                    continue
    except (NotADirectoryError, FileNotFoundError, PermissionError) as e:
        # Not a dir or inaccessible — behave like os.walk (i.e., yield nothing)
        if onerror:
            try:
                onerror(e)
            except Exception:
                pass
        return

    if topdown:
        yield top, dirs, nondirs

    for d in list(dirs):  # iterate over a snapshot
        new_path = os.path.join(top, d)
        if _is_bad_path(new_path):
            continue
        # Recurse safely
        for x in _safe_walk(new_path, topdown, onerror, False):
            yield x

    if not topdown:
        yield top, dirs, nondirs

# Monkey-patch os.walk globally so any module (ingester, watcher, etc.) benefits.
_os_walk_original = os.walk
os.walk = _safe_walk  # type: ignore[attr-defined]

# Also provide a safe rglob for Path-heavy code paths.
def _safe_rglob(self: Path, pattern: str):
    root = str(self)
    for root_dir, _dirs, files in _safe_walk(root, topdown=True):
        # Files
        for fname in files:
            p = os.path.join(root_dir, fname)
            try:
                if Path(p).match(pattern):
                    yield Path(p)
            except Exception:
                continue
        # Dirs are handled by walk recursion

# Monkey-patch Path.rglob cautiously (keep original accessible)
_Path_rglob_original = Path.rglob
Path.rglob = _safe_rglob  # type: ignore[assignment]
# --- END FS SAFETY GUARDS ---

from .orchestrator import Orchestrator
from .rag import RAG
from .web_utils import fetch_page_text
from .ingest_watcher import run_watcher, scan_all
from fastapi.middleware.cors import CORSMiddleware


CONFIG = yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))
ING = CONFIG["ingest"]

app = FastAPI(title="Customer Service AI — Local (Ollama)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = RAG()
orch = Orchestrator(rag=rag)

# ---------- Startup / Shutdown ----------
bg_task: Optional[asyncio.Task] = None

@app.on_event("startup")
async def _startup():
    # Optional: log once so you know the guards are active
    print("[fs-guard] os.walk patched; skipping systemd-private-* and Proton z:; not following symlinks.", file=sys.stderr)

    for key in ["uploads_dir","links_dir","messages_dir","links_cache_dir","root"]:
        os.makedirs(ING[key], exist_ok=True)
    global bg_task
    bg_task = asyncio.create_task(run_watcher(rag, ING.get("scan_interval_seconds", 10)))

@app.on_event("shutdown")
async def _shutdown():
    global bg_task
    if bg_task and not bg_task.done():
        bg_task.cancel()
        try:
            await bg_task
        except asyncio.CancelledError:
            pass

# ---------- Schemas ----------
class ChatIn(BaseModel):
    user_id: str
    message: str
    namespace: Optional[str] = None  # we’ll send null; backend defaults to "default"

class TeachIn(BaseModel):
    text: str
    namespace: Optional[str] = None

class LinksIn(BaseModel):
    urls: List[str]
    namespace: Optional[str] = None

# ---------- Minimal Chat UI ----------
@app.get("/ui", response_class=HTMLResponse)
async def ui():
    return """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Chat</title>
<style>
  :root { --bg:#0b0c10; --panel:#14161b; --text:#e6e6e6; --muted:#9aa0a6; --accent:#6aa0ff; }
  *{ box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text); font:16px/1.45 system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif; }
  header { padding:14px 16px; background:var(--panel); border-bottom:1px solid #22262d; position:sticky; top:0; z-index:1; }
  header h1 { margin:0; font-size:16px; font-weight:600; }
  #chat { max-width:900px; margin:0 auto; padding:16px; min-height:calc(100vh - 140px); }
  .bubble { max-width:78%; padding:10px 12px; border-radius:14px; margin:8px 0; white-space:pre-wrap; word-break:break-word; }
  .u { background:#1f232b; margin-left:auto; border-bottom-right-radius:6px; }
  .a { background:#11151b; border:1px solid #22262d; border-bottom-left-radius:6px; }
  .sys { color: var(--muted); text-align:center; margin:12px 0; font-size:13px; }
  footer { position:sticky; bottom:0; background:var(--panel); border-top:1px solid #22262d; }
  .row { display:flex; gap:8px; padding:12px; max-width:900px; margin:0 auto; }
  #msg { flex:1; padding:10px 12px; border-radius:12px; background:#0e1015; color:var(--text); border:1px solid #22262d; }
  button { padding:10px 14px; border-radius:12px; border:1px solid #2a313b; background:#1b212a; color:#fff; cursor:pointer; }
  button:hover { background:#212833; }
</style>
</head>
<body>
  <header><h1>Customer Support Assistant</h1></header>
  <div id="chat"></div>
  <footer>
    <div class="row">
      <input id="msg" type="text" placeholder="Type a message and press Enter..." autofocus />
      <button onclick="sendMsg()">Send</button>
    </div>
  </footer>

<script>
const chat = document.getElementById('chat');
const msg = document.getElementById('msg');

function addBubble(text, who){
  const div = document.createElement('div');
  div.className = 'bubble ' + (who === 'u' ? 'u' : 'a');
  div.textContent = text;
  chat.appendChild(div);
  window.scrollTo({top: document.body.scrollHeight, behavior: 'smooth'});
}

function addSystem(text){
  const d = document.createElement('div');
  d.className = 'sys';
  d.textContent = text;
  chat.appendChild(d);
  window.scrollTo({top: document.body.scrollHeight, behavior: 'smooth'});
}

async function sendMsg(){
  const text = msg.value.trim();
  if(!text) return;
  addBubble(text, 'u');
  msg.value='';
  addSystem('…thinking…');

  try {
    const r = await fetch('/chat', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({user_id:'ui', message: text, namespace: null})
    });
    const raw = await r.text();
    const last = chat.querySelector('.sys:last-child');
    if(last && last.textContent.includes('…thinking…')) last.remove();

    try {
      const json = JSON.parse(raw);
      addBubble(json?.answer ?? '(no answer)', 'a');
    } catch(e){
      addBubble(`Non-JSON response (status ${r.status}):\\n\\n` + raw, 'a');
    }
  } catch (e) {
    const last = chat.querySelector('.sys:last-child');
    if(last && last.textContent.includes('…thinking…')) last.remove();
    addBubble('Request failed: ' + (e?.message || e), 'a');
  }
}

msg.addEventListener('keydown', (ev)=>{
  if(ev.key === 'Enter' && !ev.shiftKey){
    ev.preventDefault();
    sendMsg();
  }
});
</script>
</body>
</html>
"""

# ---------- Convenience routes ----------
@app.get("/", include_in_schema=False)
async def root_redirect():
    return RedirectResponse(url="/ui", status_code=307)

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return HTMLResponse(status_code=204)

# ---------- (Optional) Keep simple add-knowledge endpoints for you ----------
@app.post("/teach")
async def teach(payload: TeachIn):
    ns = payload.namespace or ING.get("namespace_default","default")
    ns_dir = os.path.join(ING["messages_dir"], ns)
    os.makedirs(ns_dir, exist_ok=True)
    path = os.path.join(ns_dir, "note-ui.txt")
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n" + payload.text.strip() + "\n")
    return {"status":"ok","saved": path, "namespace": ns}

@app.post("/upload")
async def upload(files: List[UploadFile] = File(...), namespace: Optional[str] = Form(None)):
    ns = namespace or ING.get("namespace_default","default")
    ns_dir = os.path.join(ING["uploads_dir"], ns)
    os.makedirs(ns_dir, exist_ok=True)
    saved = []
    for f in files:
        pth = os.path.join(ns_dir, f.filename)
        with open(pth, "wb") as out:
            out.write(await f.read())
        saved.append(pth)
    return {"status":"ok","saved": saved, "namespace": ns}

@app.post("/links")
async def links(payload: LinksIn):
    ns = payload.namespace or ING.get("namespace_default","default")
    ns_dir = os.path.join(ING["links_dir"], ns)
    os.makedirs(ns_dir, exist_ok=True)
    path = os.path.join(ns_dir, "links-ui.txt")
    with open(path, "a", encoding="utf-8") as f:
        for u in payload.urls:
            f.write(u.strip()+"\n")
    return {"status":"ok","saved": path, "count": len(payload.urls), "namespace": ns}

@app.post("/ingest/scan")
async def ingest_scan():
    totals = await scan_all(rag)
    return {"status":"ok","totals": totals}

# ---------- Chat ----------
@app.post("/chat")
async def chat(payload: ChatIn):
    result = await orch.handle_chat(user_id=payload.user_id, message=payload.message, namespace=payload.namespace)
    return result
