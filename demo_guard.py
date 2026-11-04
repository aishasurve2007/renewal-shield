"""
Demo guard: keeps a public demo from running up the LLM bill.

  - Cloudflare Turnstile check (proves a browser with a human in it sent the request)
  - per-visitor hourly limits
  - one daily cap for all visitors (resets at midnight Dubai time)
  - upload size limit, checked from Content-Length before the body is read

Settings (environment variables):
  RS_TURNSTILE_SITE_KEY  public site key, sent to the page        (unset = Turnstile off)
  RS_TURNSTILE_SECRET    secret key, server side only             (unset = Turnstile off)
  RS_PHOTOS_PER_HOUR     photo checks per visitor per hour        (default 5)
  RS_CLAUSES_PER_HOUR    single-clause checks per visitor/hour    (default 20)
  RS_DAILY_LIMIT         all checks, all visitors, per day        (default 150)
  RS_MAX_MB              largest upload in MB                     (default 10)

Turnstile is off when either key is missing, so local runs work without Cloudflare.
On the deployed app set both. Limits live in memory and reset when the container restarts.
"""

import os
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone

import httpx
from fastapi import HTTPException, Request

TURNSTILE_SITE_KEY = os.environ.get("RS_TURNSTILE_SITE_KEY", "").strip()
TURNSTILE_SECRET = os.environ.get("RS_TURNSTILE_SECRET", "").strip()
TURNSTILE_ON = bool(TURNSTILE_SITE_KEY and TURNSTILE_SECRET)
VERIFY_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"

PHOTOS_PER_HOUR = int(os.environ.get("RS_PHOTOS_PER_HOUR", 5))
CLAUSES_PER_HOUR = int(os.environ.get("RS_CLAUSES_PER_HOUR", 20))
DAILY_LIMIT = int(os.environ.get("RS_DAILY_LIMIT", 150))
MAX_MB = int(os.environ.get("RS_MAX_MB", 10))
MAX_BYTES = MAX_MB * 1024 * 1024

DUBAI = timezone(timedelta(hours=4))          # fixed offset: no DST in the UAE, no tzdata needed

_hits = defaultdict(deque)                    # (visitor, kind) -> timestamps in the last hour
_day = {"date": None, "count": 0}
_lock = threading.Lock()


def visitor(request: Request) -> str:
    """Visitor IP. Behind Azure Container Apps the real IP is the first X-Forwarded-For entry."""
    fwd = request.headers.get("x-forwarded-for", "")
    return fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "?")


def checks_today() -> int:
    return _day["count"] if _day["date"] == datetime.now(DUBAI).date() else 0


def check_size(request: Request):
    """Refuse a too-big upload from its Content-Length header, before reading the body."""
    try:
        length = int(request.headers.get("content-length", 0))
    except ValueError:
        length = 0
    if length > MAX_BYTES + 64 * 1024:        # small allowance for the multipart wrapping
        raise HTTPException(413, f"File is over {MAX_MB} MB.")


def verify_turnstile(request: Request, token: str | None):
    """Ask Cloudflare whether the widget token is real. Each token works once."""
    if not TURNSTILE_ON:
        return
    if not token:
        raise HTTPException(403, "Please complete the 'verify you are human' check, then try again.")
    try:
        r = httpx.post(VERIFY_URL, timeout=10, data={
            "secret": TURNSTILE_SECRET, "response": token, "remoteip": visitor(request)})
        ok = r.json().get("success", False)
    except Exception as e:                    # Cloudflare unreachable: fail closed, log without details
        print(f"[turnstile] verify failed: {type(e).__name__}", flush=True)
        raise HTTPException(503, "Couldn't verify the human check. Please try again in a moment.")
    if not ok:
        raise HTTPException(403, "The human check expired or failed. Please try again.")


def take(request: Request, kind: str):
    """Count one check against the visitor's hourly limit and the daily cap, or refuse."""
    per_hour = PHOTOS_PER_HOUR if kind == "photo" else CLAUSES_PER_HOUR
    now, who = time.time(), visitor(request)
    with _lock:
        today = datetime.now(DUBAI).date()
        if _day["date"] != today:
            _day["date"], _day["count"] = today, 0
            _hits.clear()                     # keeps memory bounded: old visitors drop out daily
        if _day["count"] >= DAILY_LIMIT:
            raise HTTPException(429, "The demo has reached its daily limit. Please try again tomorrow.")
        q = _hits[(who, kind)]
        while q and now - q[0] > 3600:
            q.popleft()
        if len(q) >= per_hour:
            wait = int((3600 - (now - q[0])) / 60) + 1
            raise HTTPException(429, f"You've reached the demo limit ({per_hour} per hour). "
                                     f"Try again in about {wait} minutes.")
        q.append(now)
        _day["count"] += 1


def guard(request: Request, kind: str, token: str | None):
    """Everything a paid check must pass, cheapest first. Call after input validation,
    so a wrong file type doesn't use up a check."""
    verify_turnstile(request, token)
    take(request, kind)
