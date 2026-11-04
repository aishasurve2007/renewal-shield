"""
Renewal Shield — web app (Phase 5).

Upload a photo of a Dubai tenancy contract, get the red / yellow / fine report in the browser.
The page is static/index.html; this file serves it and wraps check_contract.check() in an API.

Run locally:
  uvicorn app:app --reload
  -> open http://127.0.0.1:8000

API:
  POST /api/check          form field "file": contract photo (jpg / png / webp)  -> report JSON
  POST /api/check-clause   JSON {"text": "..."}                                  -> report JSON
  GET  /api/health

Settings (environment variables, or .env locally; on Hugging Face: Space settings -> secrets/variables):
  OPENAI_API_KEY         the key (secret!)
  RS_PROVIDER            openai (default) | gemini | anthropic | ollama
  RS_MODEL               override the provider's default model
  RS_PHOTOS_PER_HOUR     photo checks per visitor per hour      (default 5)
  RS_CLAUSES_PER_HOUR    single-clause checks per visitor/hour  (default 20)
  RS_DAILY_LIMIT         all checks, all visitors, per day      (default 150)
  RS_MAX_CLAUSES         clauses checked per photo              (default 15)
  RS_MAX_MB              largest upload in MB                   (default 10)
  RS_TURNSTILE_SITE_KEY  Cloudflare Turnstile site key (public)
  RS_TURNSTILE_SECRET    Cloudflare Turnstile secret   (secret!)  both unset = no human check
  (limits, Turnstile and the upload size check live in demo_guard.py)

Why the limits: on a public link every check is paid from my OpenAI key. One photo is about
6 clauses x ~2.5k tokens on gpt-4.1-mini, well under 1 US cent, so 150 checks/day stays below
about $1/day even if someone hammers it. Set a monthly budget in the OpenAI dashboard as well.

Privacy: uploads go to a temporary file that is deleted straight after the check.
Nothing is stored except the LLM answer cache (clause text + answer), as in the CLI.
"""

import os
import tempfile
import threading

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import demo_guard
from check_contract import check          # also loads .env (via classify_llm)

PROVIDER = os.environ.get("RS_PROVIDER", "openai")
MODEL = os.environ.get("RS_MODEL") or None
MAX_CLAUSES = int(os.environ.get("RS_MAX_CLAUSES", 15))
MAX_MB = demo_guard.MAX_MB
ALLOWED = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
HERE = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="Renewal Shield")
app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")

# One check at a time: OCR is CPU-heavy and the answer cache is a plain file.
_lock = threading.Lock()


@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    """Refuse an oversized upload from its Content-Length, before FastAPI reads the body."""
    if request.method == "POST" and request.url.path == "/api/check":
        try:
            demo_guard.check_size(request)
        except HTTPException as e:
            return JSONResponse({"detail": e.detail}, status_code=e.status_code)
    return await call_next(request)


def _run(**kwargs):
    """Call check(). Errors (missing key, API down) are logged in full but shown to visitors
    without details: an OpenAI error message can contain part of the API key."""
    with _lock:
        try:
            report = check(provider=PROVIDER, model=MODEL, max_clauses=MAX_CLAUSES, **kwargs)
        except SystemExit as e:                      # check() / classify() use SystemExit for clear messages
            print(f"[check failed] {e}", flush=True)
            raise HTTPException(503, "The checking service is temporarily unavailable. Please try again later.")
    report.pop("name", None)
    return report


# ----------------------------------------------------------------- routes
@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "static", "index.html"))


@app.get("/api/health")
def health():
    return {"ok": True, "provider": PROVIDER, "checks_today": demo_guard.checks_today(),
            "daily_limit": demo_guard.DAILY_LIMIT}


@app.get("/api/config")
def config():
    """What the page needs to know: the public Turnstile site key (empty = no widget)."""
    return {"turnstile_site_key": demo_guard.TURNSTILE_SITE_KEY if demo_guard.TURNSTILE_ON else ""}


# Plain `def` (not async): FastAPI runs it in a worker thread, so a 20 s OCR doesn't block the server.
@app.post("/api/check")
def check_photo(request: Request, file: UploadFile = File(...), cf_token: str = Form("")):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext in (".heic", ".heif"):
        raise HTTPException(400, "iPhone HEIC photos aren't supported yet. Export as JPG, or take a screenshot of the photo.")
    if ext not in ALLOWED:
        raise HTTPException(400, f"Please upload a photo (JPG or PNG), not '{ext or 'unknown'}'.")
    data = file.file.read(MAX_MB * 1024 * 1024 + 1)
    if len(data) > MAX_MB * 1024 * 1024:
        raise HTTPException(413, f"File is over {MAX_MB} MB.")
    demo_guard.guard(request, "photo", cf_token)   # after validation: a wrong file type doesn't use a check

    fd, path = tempfile.mkstemp(suffix=ext)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        report = _run(image=path)
    finally:
        os.remove(path)
    report["filename"] = file.filename
    return report


class ClauseIn(BaseModel):
    text: str
    cf_token: str = ""


@app.post("/api/check-clause")
def check_clause(request: Request, body: ClauseIn):
    text = " ".join(body.text.split())
    if len(text) < 10:
        raise HTTPException(400, "Paste a full clause (at least a sentence).")
    if len(text) > 2000:
        raise HTTPException(400, "That's too long for one clause. Paste clauses one at a time.")
    demo_guard.guard(request, "clause", body.cf_token)
    return _run(clause=text)
