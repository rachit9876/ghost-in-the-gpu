"""Ghost in the GPU — Multi-Domain API & Web Server."""
import base64
import hashlib
import ipaddress
import json
import logging
import math
import mimetypes
import random
import re
import secrets
import socket
import threading
import time
import urllib.error
import urllib.request
import warnings
from collections import deque
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from statistics import NormalDist
from typing import Optional, List
from urllib.parse import urlparse

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from ghost.config import (
    DB_PATH, IMAGES_DIR, EMBEDDINGS_DIR,
    get_domain_model_paths, HOST, PORT
)
from ghost.rater import db
from ghost.rater.scorer import GhostScorer
from ghost.sensory.ingest import ingest_bytes

logger = logging.getLogger("ghost.mirror")


# ═══════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════
_STATIC_DIR = Path(__file__).parent / "static"
_RECENT_SHOWN_MAX = 500
_MIRROR_RATE = 0.20
_MIRROR_MIN_AGE_S = 48 * 3600
_MAX_UPLOAD_BYTES = 50_000_000
_MAX_REMOTE_BYTES = 25_000_000
_REMOTE_TIMEOUT_S = 12
_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
_MAX_B64_CHARS = (_MAX_UPLOAD_BYTES * 4) // 3 + 4  # cap on base64 length
_ALLOWED_ORIGIN_EXACT = ("http://127.0.0.1", "http://localhost")
_ALLOWED_ORIGIN_PREFIXES = (
    "http://127.0.0.1:",
    "http://localhost:",
    "chrome-extension://",
    "moz-extension://",
)


# ═══════════════════════════════════════════════════
# STATE
# ═══════════════════════════════════════════════════
_RECENT_SHOWN: dict = {}
_SCORERS: dict = {}
_COMPARED_PAIRS_CACHE: dict = {}
_PERCENTILE_CACHE: dict = {}
_PERCENTILE_LOCK = threading.Lock()
_SCORER_LOCK = threading.Lock()
_TRAIN_PROCS: dict = {}
_TRAIN_LOCK = threading.Lock()
_SESSION_MAP: dict = {}
_SESSION_MAP_LOCK = threading.Lock()


# ═══════════════════════════════════════════════════
# DB CONTEXT MANAGER
# ═══════════════════════════════════════════════════
@contextmanager
def db_conn():
    c = db.connect()
    try:
        yield c
    finally:
        c.close()


# ═══════════════════════════════════════════════════
# SESSION HELPERS
# ═══════════════════════════════════════════════════
def _request_session_id(request: Request) -> Optional[int]:
    if request is None:
        return None
    return getattr(request.state, "db_session_id", None)


# ═══════════════════════════════════════════════════
# URL SAFETY (SSRF guard)
# ═══════════════════════════════════════════════════
def _is_safe_url(url: str) -> bool:
    try:
        p = urlparse(url)
        if p.scheme not in ("http", "https"):
            return False
        host = p.hostname
        if not host:
            return False
        try:
            infos = socket.getaddrinfo(host, None)
        except (socket.gaierror, OSError):
            return False
        for info in infos:
            addr = info[4][0]
            try:
                ip = ipaddress.ip_address(addr)
            except ValueError:
                return False
            if (ip.is_private or ip.is_loopback or ip.is_link_local
                    or ip.is_reserved or ip.is_multicast or ip.is_unspecified):
                return False
        return True
    except Exception:
        return False


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not _is_safe_url(newurl):
            raise urllib.error.URLError(f"Blocked redirect to {newurl}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_SAFE_OPENER = urllib.request.build_opener(_SafeRedirectHandler())


def _safe_download(url: str, max_bytes: int = _MAX_REMOTE_BYTES,
                   timeout: int = _REMOTE_TIMEOUT_S) -> bytes:
    if not _is_safe_url(url):
        raise HTTPException(400, f"URL is not allowed: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with _SAFE_OPENER.open(req, timeout=timeout) as resp:
            data = resp.read(max_bytes + 1)
    except urllib.error.HTTPError as e:
        raise HTTPException(400, f"Failed to download image: HTTP {e.code}")
    except Exception as e:
        raise HTTPException(400, f"Failed to download image URL: {e}")
    if len(data) > max_bytes:
        raise HTTPException(400, "Remote image is too large")
    if len(data) < 100:
        raise HTTPException(400, "Remote payload is empty or invalid")
    return data


# ═══════════════════════════════════════════════════
# MISC HELPERS
# ═══════════════════════════════════════════════════
def _safe_json_list(s) -> list:
    if not s:
        return []
    try:
        v = json.loads(s)
        return v if isinstance(v, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def _stable_score(iid: int) -> float:
    h = int(hashlib.sha256(str(iid).encode("utf-8")).hexdigest()[:8], 16)
    return (h / 0xFFFFFFFF) - 0.5


def _invalidate_compared_pairs(domain_id: int) -> None:
    _COMPARED_PAIRS_CACHE.pop(domain_id, None)


def get_compared_pairs(domain_id: int) -> set:
    with db_conn() as c:
        row = c.execute(
            "SELECT MAX(id) FROM comparisons WHERE domain_id=?", (domain_id,)
        ).fetchone()
        max_id = row[0] if (row and row[0] is not None) else 0

        cached = _COMPARED_PAIRS_CACHE.get(domain_id)
        if cached is not None and cached["max_id"] == max_id:
            return cached["pairs"]

        cmp_rows = c.execute(
            "SELECT winner_id, losers_json FROM comparisons WHERE domain_id=?",
            (domain_id,)
        ).fetchall()

    pairs = set()
    for cr in cmp_rows:
        try:
            w = cr["winner_id"]
            losers = _safe_json_list(cr["losers_json"])
            for l in losers:
                pairs.add((w, l) if w < l else (l, w))
        except Exception:
            continue

    _COMPARED_PAIRS_CACHE[domain_id] = {"max_id": max_id, "pairs": pairs}
    return pairs


def get_recent_shown(domain_id: int) -> deque:
    if domain_id not in _RECENT_SHOWN:
        _RECENT_SHOWN[domain_id] = deque(maxlen=_RECENT_SHOWN_MAX)
    return _RECENT_SHOWN[domain_id]


def get_scorer(domain_id: int = 1) -> GhostScorer:
    s = _SCORERS.get(domain_id)
    if s is not None:
        s.reload_if_needed()
        return s
    with _SCORER_LOCK:
        s = _SCORERS.get(domain_id)
        if s is None:
            s = GhostScorer(domain_id)
            _SCORERS[domain_id] = s
        else:
            s.reload_if_needed()
        return s


def resolve_image_path(image_id: int) -> Optional[Path]:
    with db_conn() as c:
        row = c.execute(
            "SELECT filename FROM images WHERE id=?", (image_id,)
        ).fetchone()
    if row and (IMAGES_DIR / row[0]).is_file():
        return IMAGES_DIR / row[0]
    return next(IMAGES_DIR.glob(f"{image_id}.*"), None)


def _prune_comparisons_for_images(c, removed_ids: set,
                                  domain_id: Optional[int] = None):
    if not removed_ids:
        return

    if domain_id is not None:
        cmps = c.execute(
            "SELECT id, winner_id, losers_json, shown_order_json "
            "FROM comparisons WHERE domain_id=?",
            (domain_id,)
        ).fetchall()
    else:
        cmps = c.execute(
            "SELECT id, winner_id, losers_json, shown_order_json "
            "FROM comparisons"
        ).fetchall()

    for cmp_row in cmps:
        try:
            w = cmp_row["winner_id"]
            orig_losers = _safe_json_list(cmp_row["losers_json"])
            orig_shown = _safe_json_list(cmp_row["shown_order_json"])
        except Exception as e:
            logger.warning("prune: skipping malformed comparison %s: %s",
                           cmp_row["id"], e)
            continue

        if w not in removed_ids and not any(x in removed_ids for x in orig_losers):
            continue

        losers = [x for x in orig_losers if x not in removed_ids]
        shown = [x for x in orig_shown if x not in removed_ids]

        if w in removed_ids or len(losers) == 0:
            c.execute("DELETE FROM comparisons WHERE id=?", (cmp_row["id"],))
        else:
            c.execute(
                "UPDATE comparisons SET losers_json=?, shown_order_json=?, "
                "group_size=? WHERE id=?",
                (json.dumps(losers), json.dumps(shown), len(shown),
                 cmp_row["id"])
            )

    if domain_id is not None:
        _invalidate_compared_pairs(domain_id)
    else:
        _COMPARED_PAIRS_CACHE.clear()


def _get_sorted_scores(domain_id: int, scorer: GhostScorer):
    """Return sorted ascending array of every domain score. Cached on model mtime."""
    if not scorer.loaded:
        return None
    mtime = getattr(scorer, "_mtime", 0.0)
    cached = _PERCENTILE_CACHE.get(domain_id)
    if cached is not None and cached["mtime"] == mtime:
        return cached["sorted"]

    with _PERCENTILE_LOCK:
        cached = _PERCENTILE_CACHE.get(domain_id)
        if cached is not None and cached["mtime"] == mtime:
            return cached["sorted"]

        with db_conn() as c:
            dom_ids = [r[0] for r in c.execute(
                "SELECT id FROM images WHERE domain_id=?", (domain_id,)
            ).fetchall()]
        scores = [scorer.score(x) for x in dom_ids]
        scores = [s for s in scores if s is not None]
        arr = np.sort(np.asarray(scores, dtype=np.float32)) if scores else np.empty(0)

        _PERCENTILE_CACHE[domain_id] = {"mtime": mtime, "sorted": arr}
        return arr


def _percentile_and_tier(domain_id: int, score: Optional[float],
                         scorer: GhostScorer) -> dict:
    if score is None:
        return {"percentile": 50.0, "tier": "approved", "badge": "Approved"}

    arr = _get_sorted_scores(domain_id, scorer)
    if arr is None or arr.size == 0:
        percentile = 50.0
    else:
        rank = int(np.searchsorted(arr, score, side="right"))
        percentile = round((rank / arr.size) * 100.0, 1)

    if percentile >= 95:
        return {"percentile": percentile, "tier": "elite",
                "badge": f"Top {max(1, round(100 - percentile))}% - Elite"}
    if percentile >= 80:
        return {"percentile": percentile, "tier": "favorite",
                "badge": f"Top {round(100 - percentile)}% - Favorite"}
    if percentile >= 50:
        return {"percentile": percentile, "tier": "approved",
                "badge": f"Approved ({round(percentile)}%)"}
    return {"percentile": percentile, "tier": "reject",
            "badge": f"Below Threshold ({round(percentile)}%)"}


# ═══════════════════════════════════════════════════
# PYDANTIC MODELS
# ═══════════════════════════════════════════════════
class DomainCreate(BaseModel):
    name: str
    description: str = ""


class DomainUpdate(BaseModel):
    name: str
    description: Optional[str] = None


class BulkDeleteBody(BaseModel):
    domain_id: Optional[int] = None
    image_ids: List[int] = Field(..., min_length=1)


class BulkMoveBody(BaseModel):
    target_domain_id: int
    image_ids: List[int] = Field(..., min_length=1)


class MergeImagesBody(BaseModel):
    keep_id: int
    retire_id: int
    domain_id: int = 1


class FetchUrlBody(BaseModel):
    url: str


class ImageEditBody(BaseModel):
    image_id: int
    image_base64: str
    domain_id: Optional[int] = None


class ImportRunBody(BaseModel):
    urls: List[str] = Field(default_factory=list)
    domain_id: int = 1
    batch_tag: Optional[str] = None


class ImportRollbackBody(BaseModel):
    batch_id: str
    domain_id: Optional[int] = None


class RateBody(BaseModel):
    domain_id: int = 1
    winner_id: int
    shown_order: List[int] = Field(..., min_length=2)
    sampler: str = "active"
    is_retest: int = 0
    retest_of: Optional[int] = None


class UndoBody(BaseModel):
    domain_id: Optional[int] = None


class AuditFlipBody(BaseModel):
    comparison_id: int
    new_winner_id: int


class AuditDeleteBody(BaseModel):
    comparison_id: int


class TrainBody(BaseModel):
    domain_id: int = 1


class BulkDeleteComparisonsBody(BaseModel):
    comparison_ids: List[int] = Field(..., min_length=1)
    domain_id: int = 1


class LiteLogBody(BaseModel):
    domain_id: int = 1
    image_base64: Optional[str] = None
    image_url: Optional[str] = None
    page_url: Optional[str] = None


class LiteTestBody(BaseModel):
    domain_id: int = 1
    image_base64: Optional[str] = None
    image_url: Optional[str] = None
    page_url: Optional[str] = None


# ═══════════════════════════════════════════════════
# LIFESPAN
# ═══════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init()
    logger.info("Ghost Multi-Domain Engine active at http://%s:%s", HOST, PORT)
    # Warm the vision backbone in a background thread so uvicorn starts
    # accepting TCP connections in ~1s. The first real image request
    # blocks on get_vision_model() if warmup hasn't finished yet — same
    # effective latency as before, but startup feels instant instead of
    # a 10-20s black screen.
    def _warm_vision():
        time.sleep(0.5)
        try:
            from ghost.sensory.backbone import get_vision_model
            get_vision_model()
            logger.info("Vision backbone pre-warmed.")
        except Exception as e:
            logger.warning("Could not pre-warm Vision Backbone on startup: %s",
                           e, exc_info=True)
    import threading as _threading
    _warm_thread = _threading.Thread(
        target=_warm_vision, daemon=True, name="vision-warmup"
    )
    _warm_thread.start()
    yield


app = FastAPI(title="Ghost in the GPU — Multi-Domain Neural Hub",
              lifespan=lifespan)


# ═══════════════════════════════════════════════════
# MIDDLEWARE
# ═══════════════════════════════════════════════════
class GhostSessionMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        sid = request.cookies.get("ghost_session")
        is_new = not sid
        if not sid:
            sid = "s_" + secrets.token_urlsafe(16)

        with _SESSION_MAP_LOCK:
            if sid not in _SESSION_MAP:
                try:
                    _SESSION_MAP[sid] = db.start_session()
                except Exception as e:
                    logger.warning("Failed to start DB session: %s", e)
                    _SESSION_MAP[sid] = None

        request.state.session_id = sid
        request.state.db_session_id = _SESSION_MAP.get(sid)

        response = await call_next(request)
        if is_new:
            response.set_cookie(
                "ghost_session", sid,
                max_age=60 * 60 * 24 * 30,
                httponly=True, samesite="lax",
            )
        return response


def _origin_allowed(origin):
    """True if the Origin header is missing or from a trusted local source."""
    if not origin:
        return True  # programmatic / same-origin (browser omits Origin)
    if origin in _ALLOWED_ORIGIN_EXACT:
        return True
    return any(origin.startswith(p) for p in _ALLOWED_ORIGIN_PREFIXES)


class OriginGuardMiddleware(BaseHTTPMiddleware):
    """Reject mutating requests whose Origin header is not local.

    Blocks drive-by POST /api/undo, /api/train, /api/images/upload
    from any website the user happens to visit. Simple (no-preflight)
    requests still get here, so CORS alone is not enough.
    """
    async def dispatch(self, request, call_next):
        if request.method in ("POST", "PUT", "DELETE", "PATCH"):
            origin = request.headers.get("origin")
            if not _origin_allowed(origin):
                logger.warning(
                    "Blocked cross-origin %s %s from %s",
                    request.method, request.url.path, origin,
                )
                return JSONResponse(
                    {"detail": "Cross-origin request blocked"},
                    status_code=403,
                )
        return await call_next(request)


app.add_middleware(GhostSessionMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=(
        r"^(https?://(localhost|127\.0\.0\.1)(:\d+)?"
        r"|chrome-extension://[a-z]+"
        r"|moz-extension://[a-z-]+)$"
    ),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(OriginGuardMiddleware)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["127.0.0.1", "localhost"],
    www_redirect=False,
)

app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


# ═══════════════════════════════════════════════════
# UI & HEALTH ROUTES
# ═══════════════════════════════════════════════════
@app.get("/api/health")
def api_health():
    db_ok = True
    try:
        with db_conn() as c:
            c.execute("SELECT 1").fetchone()
    except Exception:
        db_ok = False
    import torch
    return {
        "ok": True,
        "status": "running",
        "db_ok": db_ok,
        "gpu": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


@app.get("/")
def hub_page():
    return FileResponse(_STATIC_DIR / "hub.html")


@app.get("/domain")
def domain_workspace_page():
    return FileResponse(_STATIC_DIR / "domain.html")


@app.get("/rate")
def rater_page():
    return FileResponse(
        _STATIC_DIR / "index.html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate",
                 "Pragma": "no-cache", "Expires": "0"}
    )


@app.get("/flex")
def flex_page():
    return FileResponse(_STATIC_DIR / "flex.html")


@app.get("/import")
def import_page():
    return FileResponse(_STATIC_DIR / "import.html")


@app.get("/duplicates")
def duplicates_page():
    return FileResponse(_STATIC_DIR / "duplicates.html")


@app.get("/audit")
def audit_page():
    return FileResponse(_STATIC_DIR / "audit.html")


@app.get("/history")
def history_page():
    return FileResponse(_STATIC_DIR / "image_history.html")


# ═══════════════════════════════════════════════════
# DOMAIN MANAGEMENT
# ═══════════════════════════════════════════════════
@app.get("/api/domains")
def api_list_domains():
    with db_conn() as c:
        rows = c.execute("""
            SELECT d.id, d.name, d.slug, d.description, d.created_at,
                   d.last_trained_at, d.accuracy,
                   (SELECT COUNT(*) FROM images WHERE domain_id = d.id) AS img_cnt,
                   (SELECT COUNT(*) FROM comparisons WHERE domain_id = d.id) AS cmp_cnt,
                   (SELECT id FROM images WHERE domain_id = d.id
                      ORDER BY id DESC LIMIT 1) AS cover_id
            FROM domains d
            ORDER BY d.id
        """).fetchall()

    out = []
    for d in rows:
        did = d["id"]
        brain_p, _ = get_domain_model_paths(did)
        out.append({
            "id": did,
            "name": d["name"],
            "slug": d["slug"],
            "description": d["description"] or "",
            "image_count": d["img_cnt"] or 0,
            "comparison_count": d["cmp_cnt"] or 0,
            "cover_image_id": d["cover_id"],
            "is_trained": brain_p.exists(),
            "accuracy": d["accuracy"],
            "last_trained_at": d["last_trained_at"],
        })
    return {"ok": True, "domains": out}


@app.post("/api/domains")
def api_create_domain(payload: DomainCreate):
    name = payload.name.strip()
    desc = (payload.description or "").strip()
    if not name:
        raise HTTPException(400, "Domain name is required")
    base_slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "domain"

    with db_conn() as c:
        slug = base_slug
        idx = 1
        while c.execute("SELECT 1 FROM domains WHERE slug=?", (slug,)).fetchone():
            idx += 1
            slug = f"{base_slug}-{idx}"

        try:
            cur = c.execute(
                "INSERT INTO domains (name, slug, description, created_at) "
                "VALUES (?, ?, ?, ?)",
                (name, slug, desc, time.time())
            )
            new_id = cur.lastrowid
            c.commit()
        except Exception as e:
            raise HTTPException(
                400, f"Could not create domain (name might already exist): {e}"
            )

    return {"ok": True, "id": new_id, "name": name, "slug": slug}


@app.put("/api/domains/{domain_id}")
def api_update_domain(domain_id: int, payload: DomainUpdate):
    name = payload.name.strip()
    if not name:
        raise HTTPException(400, "Domain name cannot be empty")
    base_slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or f"domain-{domain_id}"

    with db_conn() as c:
        row = c.execute(
            "SELECT id, description FROM domains WHERE id=?", (domain_id,)
        ).fetchone()
        if not row:
            raise HTTPException(404, "Domain not found")

        slug = base_slug
        idx = 1
        while c.execute(
            "SELECT 1 FROM domains WHERE slug=? AND id!=?", (slug, domain_id)
        ).fetchone():
            idx += 1
            slug = f"{base_slug}-{idx}"

        if payload.description is not None:
            desc = payload.description.strip()
        else:
            desc = row["description"] or ""

        try:
            c.execute(
                "UPDATE domains SET name=?, slug=?, description=? WHERE id=?",
                (name, slug, desc, domain_id)
            )
            c.commit()
        except Exception as e:
            raise HTTPException(400, f"Could not rename domain: {e}")

    return {"ok": True, "id": domain_id, "name": name,
            "slug": slug, "description": desc}


@app.delete("/api/domains/{domain_id}")
def api_delete_domain(domain_id: int):
    with db_conn() as c:
        cnt = c.execute("SELECT COUNT(*) FROM domains").fetchone()[0]
        if cnt <= 1:
            raise HTTPException(400, "Cannot delete the last remaining domain.")

        imgs = c.execute(
            "SELECT id, filename FROM images WHERE domain_id=?", (domain_id,)
        ).fetchall()
        for r in imgs:
            p = IMAGES_DIR / r["filename"]
            if p.is_file():
                p.unlink(missing_ok=True)
            ep = EMBEDDINGS_DIR / f"{r['id']}.npy"
            if ep.is_file():
                ep.unlink(missing_ok=True)

        c.execute("DELETE FROM comparisons WHERE domain_id=?", (domain_id,))
        c.execute("DELETE FROM images WHERE domain_id=?", (domain_id,))
        c.execute("DELETE FROM sessions WHERE domain_id=?", (domain_id,))
        c.execute("DELETE FROM domains WHERE id=?", (domain_id,))
        c.commit()

    brain_p, pca_p = get_domain_model_paths(domain_id)
    brain_p.unlink(missing_ok=True)
    pca_p.unlink(missing_ok=True)

    with _SCORER_LOCK:
        _SCORERS.pop(domain_id, None)
    _RECENT_SHOWN.pop(domain_id, None)
    _invalidate_compared_pairs(domain_id)
    _PERCENTILE_CACHE.pop(domain_id, None)
    return {"ok": True}


# ═══════════════════════════════════════════════════
# PAGINATED GALLERY & BULK ACTIONS
# ═══════════════════════════════════════════════════
@app.get("/api/images")
def api_list_images(domain_id: int = 1, page: int = 1, per_page: int = 40,
                    sort: str = "newest", batch: Optional[str] = None,
                    image_id: Optional[int] = None):
    page = max(1, page)
    per_page = max(1, min(200, per_page))
    offset = (page - 1) * per_page

    brain_p, _ = get_domain_model_paths(domain_id)
    is_trained = brain_p.exists()

    batch_filter = "AND batch_id=?" if batch else ""
    id_filter = "AND id=?" if image_id else ""
    where_extra = f"{batch_filter} {id_filter}".strip()
    extra_params = []
    if batch:
        extra_params.append(batch)
    if image_id:
        extra_params.append(image_id)
    base_params = (domain_id, *extra_params)

    with db_conn() as c:
        dom = c.execute(
            "SELECT id, name, slug, description, accuracy FROM domains WHERE id=?",
            (domain_id,)
        ).fetchone()
        if not dom:
            raise HTTPException(404, "Domain not found")

        total = c.execute(
            f"SELECT COUNT(*) FROM images WHERE domain_id=? {where_extra}",
            base_params
        ).fetchone()[0]
        cmp_cnt = c.execute(
            "SELECT COUNT(*) FROM comparisons WHERE domain_id=?", (domain_id,)
        ).fetchone()[0]

        use_scores = sort in ("score_high", "score_low") and is_trained

        if use_scores:
            scorer = get_scorer(domain_id)
            rows = c.execute(
                f"SELECT id, filename, sha256, added_at, batch_id FROM images "
                f"WHERE domain_id=? {where_extra} ORDER BY id DESC",
                base_params
            ).fetchall()
            items = []
            for r in rows:
                iid = r["id"]
                sc = scorer.score(iid)
                items.append({
                    "id": iid,
                    "url": f"/api/image/{iid}",
                    "sha256": r["sha256"],
                    "added_at": r["added_at"],
                    "batch_id": r["batch_id"],
                    "score": round(sc, 3) if sc is not None else None,
                })
            if sort == "score_high":
                items.sort(
                    key=lambda x: (x["score"] is not None,
                                   x["score"] if x["score"] is not None else -1e9),
                    reverse=True,
                )
            else:
                items.sort(
                    key=lambda x: (x["score"] is None,
                                   x["score"] if x["score"] is not None else 1e9),
                )
            paged = items[offset: offset + per_page]
        else:
            order_dir = "ASC" if sort == "oldest" else "DESC"
            rows = c.execute(
                f"SELECT id, filename, sha256, added_at, batch_id FROM images "
                f"WHERE domain_id=? {where_extra} "
                f"ORDER BY id {order_dir} LIMIT ? OFFSET ?",
                base_params + (per_page, offset)
            ).fetchall()
            scorer = get_scorer(domain_id) if is_trained else None
            paged = []
            for r in rows:
                iid = r["id"]
                sc = scorer.score(iid) if (scorer and scorer.loaded) else None
                paged.append({
                    "id": iid,
                    "url": f"/api/image/{iid}",
                    "sha256": r["sha256"],
                    "added_at": r["added_at"],
                    "batch_id": r["batch_id"],
                    "score": round(sc, 3) if sc is not None else None,
                })

    total_pages = max(1, (total + per_page - 1) // per_page)
    return {
        "ok": True,
        "domain": dict(dom),
        "total": total,
        "comparisons": cmp_cnt,
        "is_trained": is_trained,
        "page": page,
        "per_page": per_page,
        "total_pages": total_pages,
        "images": paged,
    }


@app.post("/api/images/bulk-delete")
def api_bulk_delete(payload: BulkDeleteBody):
    with db_conn() as c:
        if payload.domain_id is not None:
            dom = c.execute(
                "SELECT id FROM domains WHERE id=?", (payload.domain_id,)
            ).fetchone()
            if not dom:
                raise HTTPException(404, "Domain not found")

        placeholders = ",".join("?" for _ in payload.image_ids)
        if payload.domain_id is not None:
            rows = c.execute(
                f"SELECT id, filename, domain_id FROM images "
                f"WHERE domain_id=? AND id IN ({placeholders})",
                (payload.domain_id, *payload.image_ids)
            ).fetchall()
        else:
            rows = c.execute(
                f"SELECT id, filename, domain_id FROM images "
                f"WHERE id IN ({placeholders})",
                tuple(payload.image_ids)
            ).fetchall()

        deleted_ids = set()
        affected_domains = set()
        for r in rows:
            iid = r["id"]
            (IMAGES_DIR / r["filename"]).unlink(missing_ok=True)
            (EMBEDDINGS_DIR / f"{iid}.npy").unlink(missing_ok=True)
            c.execute("DELETE FROM images WHERE id=?", (iid,))
            deleted_ids.add(iid)
            if r["domain_id"] is not None:
                affected_domains.add(r["domain_id"])

        for d_id in affected_domains:
            _prune_comparisons_for_images(c, deleted_ids, domain_id=d_id)
        c.commit()

    with _SCORER_LOCK:
        for scorer in _SCORERS.values():
            for iid in deleted_ids:
                try:
                    scorer._cache.pop(iid, None)
                except Exception:
                    pass

    for _did in affected_domains:
        _PERCENTILE_CACHE.pop(_did, None)

    return {"ok": True, "deleted": len(deleted_ids)}


@app.get("/api/duplicates")
def api_find_duplicates(domain_id: int = 1, threshold: float = 0.95, limit: int = 50):
    """Finds near-duplicate image pairs using cosine similarity of 1024D PE-Core embeddings."""
    threshold = max(0.85, min(0.999, threshold))
    limit = max(1, min(200, limit))

    with db_conn() as c:
        imgs = c.execute(
            "SELECT id, filename, sha256, added_at, batch_id FROM images WHERE domain_id=? ORDER BY id",
            (domain_id,)
        ).fetchall()
        if len(imgs) < 2:
            return {"ok": True, "pairs": [], "total_images": len(imgs)}

        # Load comparison battle counts
        c_stats = {}
        rows = c.execute(
            "SELECT winner_id, COUNT(*) as w_cnt FROM comparisons WHERE domain_id=? AND is_retest=0 GROUP BY winner_id",
            (domain_id,)
        ).fetchall()
        for r in rows:
            c_stats[r["winner_id"]] = {"wins": r["w_cnt"], "battles": r["w_cnt"]}

        all_comps = c.execute(
            "SELECT losers_json FROM comparisons WHERE domain_id=? AND is_retest=0",
            (domain_id,)
        ).fetchall()
        for r in all_comps:
            try:
                for l in json.loads(r["losers_json"]):
                    st = c_stats.setdefault(l, {"wins": 0, "battles": 0})
                    st["battles"] += 1
            except Exception:
                pass

    img_map = {r["id"]: dict(r) for r in imgs}
    valid_ids = []
    vec_list = []
    for r in imgs:
        iid = r["id"]
        ep = EMBEDDINGS_DIR / f"{iid}.npy"
        if ep.is_file():
            try:
                v = np.load(ep).astype(np.float32).flatten()
                norm = float(np.linalg.norm(v))
                if norm > 0:
                    vec_list.append(v / norm)
                    valid_ids.append(iid)
            except Exception:
                pass

    if len(valid_ids) < 2:
        return {"ok": True, "pairs": [], "total_images": len(imgs)}

    X = np.stack(vec_list)
    sim_mat = X @ X.T
    np.fill_diagonal(sim_mat, 0)

    # Scorer for current quality scores
    scorer = get_scorer(domain_id)

    pairs = []
    iu_i, iu_j = np.triu_indices(len(valid_ids), k=1)
    high_sim_indices = np.where(sim_mat[iu_i, iu_j] >= threshold)[0]

    for idx in high_sim_indices:
        i, j = iu_i[idx], iu_j[idx]
        id_a, id_b = valid_ids[i], valid_ids[j]
        sim = float(sim_mat[i, j])

        info_a = img_map[id_a]
        info_b = img_map[id_b]

        # Get file dimensions and size
        pa = IMAGES_DIR / info_a["filename"]
        pb = IMAGES_DIR / info_b["filename"]
        size_a = pa.stat().st_size if pa.is_file() else 0
        size_b = pb.stat().st_size if pb.is_file() else 0

        dim_a = dim_b = None
        try:
            from PIL import Image
            if pa.is_file():
                with Image.open(pa) as im_a:
                    dim_a = f"{im_a.width}×{im_a.height}"
            if pb.is_file():
                with Image.open(pb) as im_b:
                    dim_b = f"{im_b.width}×{im_b.height}"
        except Exception:
            pass

        stat_a = c_stats.get(id_a, {"wins": 0, "battles": 0})
        stat_b = c_stats.get(id_b, {"wins": 0, "battles": 0})

        sc_a = scorer.score(id_a) if (scorer and scorer.loaded) else None
        sc_b = scorer.score(id_b) if (scorer and scorer.loaded) else None

        pairs.append({
            "similarity": round(sim, 4),
            "image_a": {
                "id": id_a,
                "url": f"/api/image/{id_a}",
                "filename": info_a["filename"],
                "file_size": size_a,
                "dimensions": dim_a,
                "score": round(sc_a, 3) if sc_a is not None else None,
                "battles": stat_a["battles"],
                "wins": stat_a["wins"],
            },
            "image_b": {
                "id": id_b,
                "url": f"/api/image/{id_b}",
                "filename": info_b["filename"],
                "file_size": size_b,
                "dimensions": dim_b,
                "score": round(sc_b, 3) if sc_b is not None else None,
                "battles": stat_b["battles"],
                "wins": stat_b["wins"],
            }
        })

    pairs.sort(key=lambda x: x["similarity"], reverse=True)
    return {
        "ok": True,
        "pairs": pairs[:limit],
        "total_pairs_found": len(pairs),
        "total_images": len(imgs),
        "threshold": threshold
    }


@app.post("/api/images/merge")
def api_merge_images(payload: MergeImagesBody):
    """Safely merges a duplicate image (retire_id) into the keeper (keep_id).
    Redirects comparison history without contradictions and removes the retired image.
    """
    keep_id = payload.keep_id
    retire_id = payload.retire_id
    domain_id = payload.domain_id

    if keep_id == retire_id:
        raise HTTPException(400, "Cannot merge an image into itself")

    with db_conn() as c:
        row_keep = c.execute("SELECT id, filename, domain_id FROM images WHERE id=?", (keep_id,)).fetchone()
        row_retire = c.execute("SELECT id, filename, domain_id FROM images WHERE id=?", (retire_id,)).fetchone()

        if not row_keep or not row_retire:
            raise HTTPException(404, "One or both images not found")
        if row_keep["domain_id"] != domain_id or row_retire["domain_id"] != domain_id:
            raise HTTPException(400, "Both images must belong to the specified domain")

        # 1. Prune direct battles between keep_id and retire_id (JSON-safe, retests excluded)
        c.execute("""
            DELETE FROM comparisons
            WHERE domain_id=?
              AND is_retest=0
              AND (
                (winner_id=? AND EXISTS (SELECT 1 FROM json_each(losers_json) WHERE value=?))
                OR
                (winner_id=? AND EXISTS (SELECT 1 FROM json_each(losers_json) WHERE value=?))
              )
        """, (domain_id, keep_id, retire_id, retire_id, keep_id))

        # 2. Update comparisons where retire_id won (skip retests)
        win_rows = c.execute(
            "SELECT id, shown_order_json, losers_json FROM comparisons "
            "WHERE domain_id=? AND winner_id=? AND is_retest=0",
            (domain_id, retire_id)
        ).fetchall()

        for wr in win_rows:
            try:
                so = json.loads(wr["shown_order_json"])
                so = [keep_id if x == retire_id else x for x in so]
                # Ensure unique shown_order
                so_dedup = []
                for x in so:
                    if x not in so_dedup:
                        so_dedup.append(x)
                if len(so_dedup) >= 2:
                    c.execute(
                        "UPDATE comparisons SET winner_id=?, shown_order_json=?, "
                        "group_size=? WHERE id=?",
                        (keep_id, json.dumps(so_dedup), len(so_dedup), wr["id"])
                    )
                else:
                    c.execute("DELETE FROM comparisons WHERE id=?", (wr["id"],))
            except Exception:
                pass

        # 3. Update comparisons where retire_id lost (JSON-safe, skip retests)
        lose_rows = c.execute("""
            SELECT id, winner_id, losers_json, shown_order_json
            FROM comparisons
            WHERE domain_id=?
              AND is_retest=0
              AND EXISTS (SELECT 1 FROM json_each(losers_json) WHERE value=?)
        """, (domain_id, retire_id)).fetchall()

        for lr in lose_rows:
            try:
                w_id = lr["winner_id"]
                losers = json.loads(lr["losers_json"])
                so = json.loads(lr["shown_order_json"])

                # Replace retire_id with keep_id in losers list
                losers = [keep_id if x == retire_id else x for x in losers]
                losers = [x for x in losers if x != w_id] # Can't lose to yourself

                so = [keep_id if x == retire_id else x for x in so]
                so_dedup = []
                for x in so:
                    if x not in so_dedup:
                        so_dedup.append(x)

                if len(losers) >= 1 and len(so_dedup) >= 2 and w_id in so_dedup:
                    c.execute(
                        "UPDATE comparisons SET losers_json=?, shown_order_json=?, "
                        "group_size=? WHERE id=?",
                        (json.dumps(list(set(losers))), json.dumps(so_dedup),
                         len(so_dedup), lr["id"])
                    )
                else:
                    c.execute("DELETE FROM comparisons WHERE id=?", (lr["id"],))
            except Exception:
                pass

        # 3.5 B2: delete retest comparisons that either directly involve
        # retire_id OR whose retest_of was orphaned by step 1 (FK cascade
        # set retest_of=NULL). Without this, retests referencing the retired
        # image survive with broken thumbnails and skew mirror agreement.
        retest_rows = c.execute("""
            SELECT id, session_id FROM comparisons
            WHERE domain_id=?
              AND is_retest=1
              AND (
                retest_of IS NULL
                OR winner_id=?
                OR EXISTS (SELECT 1 FROM json_each(losers_json) WHERE value=?)
                OR EXISTS (SELECT 1 FROM json_each(shown_order_json) WHERE value=?)
              )
        """, (domain_id, retire_id, retire_id, retire_id)).fetchall()
        for r in retest_rows:
            c.execute("DELETE FROM comparisons WHERE id=?", (r["id"],))
            if r["session_id"]:
                c.execute(
                    "UPDATE sessions SET comparison_count = "
                    "MAX(0, comparison_count - 1) WHERE id=?",
                    (r["session_id"],)
                )

        # 4. Remove retired image record from DB first (transactional)
        c.execute("DELETE FROM images WHERE id=?", (retire_id,))
        c.commit()

        # 5. Then clean up files (idempotent; safe if crash leaves orphans)
        (IMAGES_DIR / row_retire["filename"]).unlink(missing_ok=True)
        (EMBEDDINGS_DIR / f"{retire_id}.npy").unlink(missing_ok=True)

    # 6. Invalidate caches
    with _SCORER_LOCK:
        for scorer in _SCORERS.values():
            try:
                scorer._cache.pop(retire_id, None)
                scorer._cache.pop(keep_id, None)
            except Exception:
                pass

    _COMPARED_PAIRS_CACHE.pop(domain_id, None)
    _PERCENTILE_CACHE.pop(domain_id, None)

    return {
        "ok": True,
        "kept_id": keep_id,
        "retired_id": retire_id,
        "message": f"Successfully merged #{retire_id} into #{keep_id} with rating history preserved."
    }


@app.post("/api/images/bulk-move")
def api_bulk_move(payload: BulkMoveBody):
    if not payload.image_ids:
        return {"ok": True, "moved": 0, "skipped_duplicates": 0,
                "target_domain_id": payload.target_domain_id}

    moved_ids = []
    skipped_dupes = 0
    with db_conn() as c:
        dom = c.execute(
            "SELECT id FROM domains WHERE id=?", (payload.target_domain_id,)
        ).fetchone()
        if not dom:
            raise HTTPException(404, "Target domain does not exist")

        placeholders = ",".join("?" for _ in payload.image_ids)
        src_rows = c.execute(
            f"SELECT id, domain_id, sha256 FROM images WHERE id IN ({placeholders})",
            payload.image_ids
        ).fetchall()

        src_domains = set()
        for r in src_rows:
            if r["domain_id"] == payload.target_domain_id:
                continue
            if r["sha256"]:
                dup = c.execute(
                    "SELECT id FROM images WHERE domain_id=? AND sha256=?",
                    (payload.target_domain_id, r["sha256"])
                ).fetchone()
                if dup:
                    skipped_dupes += 1
                    continue
            c.execute(
                "UPDATE images SET domain_id=? WHERE id=?",
                (payload.target_domain_id, r["id"])
            )
            moved_ids.append(r["id"])
            src_domains.add(r["domain_id"])

        if moved_ids:
            moved_set = set(moved_ids)
            for sd in src_domains:
                _prune_comparisons_for_images(c, moved_set, domain_id=sd)

        c.commit()

    with _SCORER_LOCK:
        for scorer in _SCORERS.values():
            for iid in moved_ids:
                try:
                    scorer._cache.pop(iid, None)
                except Exception:
                    pass

    for _did in src_domains:
        _PERCENTILE_CACHE.pop(_did, None)
    _PERCENTILE_CACHE.pop(payload.target_domain_id, None)

    return {
        "ok": True,
        "moved": len(moved_ids),
        "skipped_duplicates": skipped_dupes,
        "target_domain_id": payload.target_domain_id,
    }


@app.post("/api/images/edit")
def api_edit_image(payload: ImageEditBody):
    # 1. Decode base64 with a size guard
    raw_bytes, ext = _decode_b64_image(payload.image_base64)
    if not raw_bytes or len(raw_bytes) < 100:
        raise HTTPException(400, "Invalid image data")
    if len(raw_bytes) > _MAX_UPLOAD_BYTES:
        raise HTTPException(400, "Image too large")

    # 2. Validate + normalize BEFORE touching disk. Failure here leaves
    #    the original image untouched.
    try:
        import io
        from PIL import Image, ImageOps
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            pil_img = Image.open(io.BytesIO(raw_bytes))
            pil_img.load()
        pil_img = ImageOps.exif_transpose(pil_img) or pil_img
        pil_img = pil_img.convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Cannot decode image: {e}")

    # 3. Look up target + check for sha256 collisions
    with db_conn() as c:
        row = c.execute(
            "SELECT id, filename, domain_id FROM images WHERE id=?",
            (payload.image_id,)
        ).fetchone()
        if not row:
            raise HTTPException(404, "Image not found")

        domain_id = row["domain_id"]
        filename = row["filename"]
        new_sha = hashlib.sha256(raw_bytes).hexdigest()

        dup = c.execute(
            "SELECT id FROM images WHERE domain_id=? AND sha256=? AND id!=?",
            (domain_id, new_sha, payload.image_id)
        ).fetchone()
        if dup:
            raise HTTPException(
                409,
                f"Edited image matches existing image #{dup['id']} in this domain"
            )

    # 4. Embed the new image (still before disk write)
    try:
        from ghost.sensory.backbone import embed_image
        new_vec = embed_image(pil_img)
    except Exception as e:
        raise HTTPException(500, f"Failed to compute embedding: {e}")

    # 5. All validations passed — write files atomically
    target_path = IMAGES_DIR / filename
    emb_path = EMBEDDINGS_DIR / f"{payload.image_id}.npy"
    tmp_img = target_path.with_name(target_path.name + ".tmp")
    tmp_emb = emb_path.with_name(emb_path.name + ".tmp")
    try:
        tmp_img.write_bytes(raw_bytes)
        with open(tmp_emb, "wb") as f:
            np.save(f, new_vec)
        import os as _os
        _os.replace(tmp_img, target_path)
        _os.replace(tmp_emb, emb_path)
    except Exception as e:
        tmp_img.unlink(missing_ok=True)
        tmp_emb.unlink(missing_ok=True)
        raise HTTPException(500, f"Failed to save edited image: {e}")

    # 6. Update sha256 in DB
    with db_conn() as c:
        c.execute(
            "UPDATE images SET sha256=? WHERE id=?",
            (new_sha, payload.image_id)
        )
        c.commit()

    # 7. Invalidate caches — OUTSIDE the scorer lock to avoid deadlock
    with _SCORER_LOCK:
        for scorer in _SCORERS.values():
            try:
                scorer._cache.pop(payload.image_id, None)
            except Exception:
                pass
    _COMPARED_PAIRS_CACHE.pop(domain_id, None)
    _PERCENTILE_CACHE.pop(domain_id, None)

    # 8. Compute new score (get_scorer acquires lock itself if needed)
    new_score = None
    try:
        scorer = get_scorer(domain_id)
        if scorer.loaded:
            new_score = scorer.score(payload.image_id)
    except Exception as e:
        logger.warning("edit: post-score failed for #%s: %s",
                       payload.image_id, e)

    return {
        "ok": True,
        "image_id": payload.image_id,
        "domain_id": domain_id,
        "score": round(new_score, 3) if new_score is not None else None,
        "url": f"/api/image/{payload.image_id}?t={int(time.time())}",
    }


@app.post("/api/images/fetch-url")
def api_fetch_image_url(payload: FetchUrlBody):
    """Safely downloads an image from a URL and returns base64 data for client-side preview."""
    u = payload.url.strip()
    if not u:
        raise HTTPException(400, "URL cannot be empty")
    try:
        raw = _safe_download(u)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Failed to download image: {e}")

    try:
        import io
        from PIL import Image, ImageOps
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            pil_img = Image.open(io.BytesIO(raw))
            pil_img.load()
        pil_img = ImageOps.exif_transpose(pil_img) or pil_img
        pil_img = pil_img.convert("RGB")
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=95)
        b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
        return {
            "ok": True,
            "data_url": b64,
            "width": pil_img.width,
            "height": pil_img.height,
            "size_bytes": len(raw)
        }
    except Exception as e:
        raise HTTPException(400, f"Cannot process image from URL: {e}")


# ═══════════════════════════════════════════════════
# UPLOAD & URL IMPORT
# ═══════════════════════════════════════════════════
@app.post("/api/images/upload")
async def api_upload_images(
    domain_id: int = Form(1),
    batch_tag: Optional[str] = Form(None),
    files: List[UploadFile] = File(...),
):
    added = 0
    existed = 0
    errors = 0
    clean_tag = (batch_tag or "").strip()
    batch_id = clean_tag if clean_tag else f"upload_{int(time.time())}"

    for uf in files:
        size = getattr(uf, "size", None)
        if size is not None and size > _MAX_UPLOAD_BYTES:
            errors += 1
            continue
        raw = await uf.read()
        if len(raw) > _MAX_UPLOAD_BYTES:
            errors += 1
            continue
        ext = Path(uf.filename or "img.jpg").suffix.lower() or ".jpg"
        try:
            res = ingest_bytes(raw, domain_id=domain_id, ext=ext,
                               batch_id=batch_id)
        except Exception as e:
            logger.warning("upload: ingest failed: %s", e)
            errors += 1
            continue
        status = res.get("status")
        if status == "added":
            added += 1
        elif status == "exists":
            existed += 1
        else:
            errors += 1

    if added > 0:
        _PERCENTILE_CACHE.pop(domain_id, None)

    return {"ok": True, "added": added, "existed": existed,
            "errors": errors, "batch_id": batch_id}


@app.post("/api/importer/run")
def api_importer_run(payload: ImportRunBody):
    batch_tag = payload.batch_tag or f"batch_{int(time.time())}"
    added = existed = errors = 0
    new_ids = []

    for u in payload.urls:
        u = u.strip()
        if not u:
            continue
        try:
            raw = _safe_download(u)
            ext = ".jpg"
            pl = u.lower().split("?")[0]
            if pl.endswith(".png"):
                ext = ".png"
            elif pl.endswith(".webp"):
                ext = ".webp"
            elif pl.endswith(".gif"):
                ext = ".gif"
            res = ingest_bytes(raw, domain_id=payload.domain_id, ext=ext,
                               source_url=u, batch_id=batch_tag)
        except HTTPException as e:
            logger.warning("importer: %s -> %s", u, e.detail)
            errors += 1
            continue
        except Exception as e:
            logger.warning("importer: %s -> %s", u, e)
            errors += 1
            continue

        status = res.get("status")
        if status == "added":
            added += 1
            if res.get("image_id"):
                new_ids.append(res["image_id"])
        elif status == "exists":
            existed += 1
        else:
            errors += 1

    scorer = get_scorer(payload.domain_id)
    high_likes = moderate_likes = dislikes = 0
    for iid in new_ids:
        sc = scorer.score(iid) if scorer.loaded else None
        if sc is None:
            moderate_likes += 1
        elif sc >= 0.15:
            high_likes += 1
        elif sc >= -0.05:
            moderate_likes += 1
        else:
            dislikes += 1

    if added > 0:
        _PERCENTILE_CACHE.pop(payload.domain_id, None)

    return {
        "ok": True,
        "added": added,
        "existed": existed,
        "errors": errors,
        "batch_id": batch_tag,
        "downloaded": added + existed,
        "imported": added,
        "skipped_main_dupes": 0,
        "skipped_ext_dupes": existed,
        "corrupted_or_failed": errors,
        "high_likes": high_likes,
        "moderate_likes": moderate_likes,
        "dislikes": dislikes,
    }


@app.post("/api/importer/rollback")
def api_importer_rollback(payload: ImportRollbackBody):
    batch_id = (payload.batch_id or "").strip()
    if not batch_id or batch_id == "unbatched":
        raise HTTPException(400, "Valid batch_id is required for rollback")

    removed_ids = []
    with db_conn() as c:
        if payload.domain_id is not None:
            rows = c.execute(
                "SELECT id, filename FROM images WHERE batch_id=? AND domain_id=?",
                (batch_id, payload.domain_id)
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT id, filename FROM images WHERE batch_id=?",
                (batch_id,)
            ).fetchall()

        for r in rows:
            iid = r["id"]
            (IMAGES_DIR / r["filename"]).unlink(missing_ok=True)
            (EMBEDDINGS_DIR / f"{iid}.npy").unlink(missing_ok=True)
            c.execute("DELETE FROM images WHERE id=?", (iid,))
            removed_ids.append(iid)

        if removed_ids:
            _prune_comparisons_for_images(
                c, set(removed_ids), domain_id=payload.domain_id
            )
        c.commit()

    if payload.domain_id is not None:
        _PERCENTILE_CACHE.pop(payload.domain_id, None)
    else:
        _PERCENTILE_CACHE.clear()

    if removed_ids:
        with _SCORER_LOCK:
            for scorer in _SCORERS.values():
                for iid in removed_ids:
                    try:
                        scorer._cache.pop(iid, None)
                    except Exception:
                        pass
        _COMPARED_PAIRS_CACHE.pop(payload.domain_id, None) if payload.domain_id else _COMPARED_PAIRS_CACHE.clear()

    return {"ok": True, "deleted": len(removed_ids), "batch_id": batch_id}


@app.get("/api/importer/batches")
def api_importer_batches(domain_id: Optional[int] = None):
    with db_conn() as c:
        if domain_id:
            rows = c.execute("""
                SELECT COALESCE(batch_id, 'unbatched') as b_id,
                       COUNT(*) as count, MIN(added_at) as first_added
                FROM images WHERE domain_id=?
                GROUP BY b_id ORDER BY first_added DESC
            """, (domain_id,)).fetchall()
        else:
            rows = c.execute("""
                SELECT COALESCE(batch_id, 'unbatched') as b_id,
                       COUNT(*) as count, MIN(added_at) as first_added
                FROM images GROUP BY b_id ORDER BY first_added DESC
            """).fetchall()

    batches = []
    for r in rows:
        d_str = (time.strftime("%Y-%m-%d %H:%M:%S",
                               time.localtime(r["first_added"]))
                 if r["first_added"] else "N/A")
        batches.append({"batch_id": r["b_id"], "count": r["count"],
                        "date": d_str})
    return {"ok": True, "batches": batches}


# ═══════════════════════════════════════════════════
# IMAGE SERVING
# ═══════════════════════════════════════════════════
@app.get("/api/image/{image_id}")
def serve_image(image_id: int):
    p = resolve_image_path(image_id)
    if not p or not p.is_file():
        raise HTTPException(404, "Image not found")
    media_type, _ = mimetypes.guess_type(p.name)
    return FileResponse(p, media_type=media_type or "image/jpeg")


# ═══════════════════════════════════════════════════
# ACTIVE LEARNING RATER
# ═══════════════════════════════════════════════════
@app.get("/api/next")
def api_next(request: Request, size: int = 3, n: int = 0, domain_id: int = 1,
             mode: str = "active", mirror: int = 0):
    sz = size if size else (n if n else 3)
    sz = max(2, min(5, sz))

    with db_conn() as c:
        d_row = c.execute(
            "SELECT id FROM domains WHERE id=?", (domain_id,)
        ).fetchone()
        if not d_row:
            raise HTTPException(404, f"Domain #{domain_id} not found")

        rows = c.execute(
            "SELECT id FROM images WHERE domain_id=? ORDER BY id", (domain_id,)
        ).fetchall()
        cmp_count = c.execute(
            "SELECT COUNT(*) FROM comparisons WHERE domain_id=?", (domain_id,)
        ).fetchone()[0]

        all_ids = [r[0] for r in rows
                   if (EMBEDDINGS_DIR / f"{r[0]}.npy").is_file()]
        if len(all_ids) < sz:
            raise HTTPException(
                400,
                f"This domain has {len(all_ids)} images, need at least {sz} "
                f"to compare. Import images first!"
            )

        retest_of_id = None
        retest_drawn = False
        chosen = None
        strategy = None

        if mirror and random.random() < _MIRROR_RATE:
            cutoff = time.time() - _MIRROR_MIN_AGE_S
            retest_cand = c.execute("""
                SELECT id, shown_order_json, group_size, created_at
                FROM comparisons
                WHERE domain_id = ?
                  AND id > 0
                  AND is_retest = 0
                  AND created_at <= ?
                  AND sampler IN ('active', 'elite', 'contrastive',
                                  'top_bracket', 'uncertainty', 'exploration')
                  AND group_size = ?
                  AND id NOT IN (
                      SELECT retest_of FROM comparisons
                      WHERE retest_of IS NOT NULL AND domain_id = ?
                  )
                ORDER BY RANDOM()
                LIMIT 1
            """, (domain_id, cutoff, sz, domain_id)).fetchone()

            if retest_cand:
                try:
                    hist_shown = _safe_json_list(retest_cand["shown_order_json"])
                    if len(hist_shown) == sz and all(i in all_ids for i in hist_shown):
                        chosen = list(hist_shown)
                        random.shuffle(chosen)
                        strategy = "mirror"
                        retest_of_id = retest_cand["id"]
                        retest_drawn = True
                        age_h = round(
                            (time.time() - retest_cand["created_at"]) / 3600.0, 1
                        )
                        logger.info(
                            "mirror: retesting cid=%s (age=%sh, domain=%s)",
                            retest_of_id, age_h, domain_id
                        )
                except Exception as e:
                    logger.warning("mirror: failed to parse candidate %s: %s",
                                   retest_cand["id"], e)

    if not retest_drawn:
        scorer = get_scorer(domain_id)
        recent_shown = get_recent_shown(domain_id)
        candidates = [i for i in all_ids if i not in recent_shown]
        if len(candidates) < sz:
            candidates = all_ids

        if scorer.loaded and mode == "contrastive":
            compared_pairs = get_compared_pairs(domain_id)
            chosen = scorer.sample_contrastive_group(
                all_ids, sz, recent_shown, compared_pairs
            )
            if chosen and len(chosen) >= 2:
                strategy = "contrastive"
            else:
                return {
                    "ok": True,
                    "domain_id": domain_id,
                    "strategy": "contrastive_empty",
                    "cards": [],
                    "image_ids": [],
                    "size": sz,
                    "total_corpus": len(all_ids),
                    "msg": ("Lookalike contrastive pairs are exhausted for "
                            "this domain. Switch to Active Exploration or "
                            "Elite mode!"),
                }
        elif scorer.loaded and mode == "elite":
            scored = []
            for i in all_ids:
                s = scorer.score(i)
                scored.append((s if s is not None else -99.0, i))
            scored.sort(key=lambda x: x[0], reverse=True)
            pool = [x[1] for x in scored[: max(sz * 10, 30)]]
            chosen = random.sample(pool, min(sz, len(pool)))
            strategy = "elite"
        elif scorer.loaded and mode == "top_bracket":
            scored = []
            for i in all_ids:
                s = scorer.score(i)
                if s is not None:
                    scored.append((s, i))
            scored.sort(key=lambda x: -x[0])
            top_pool = [x[1] for x in scored[:16]]

            if len(top_pool) >= 2:
                compared_pairs = get_compared_pairs(domain_id)
                import itertools as _it
                unfought = []
                for a, b in _it.combinations(top_pool, 2):
                    key = (a, b) if a < b else (b, a)
                    if key not in compared_pairs:
                        unfought.append((a, b))

                if unfought:
                    a, b = random.choice(unfought)
                    chosen = [a, b]
                    extra_pool = [i for i in top_pool if i not in chosen]
                    if sz > 2 and extra_pool:
                        chosen.extend(
                            random.sample(extra_pool, min(sz - 2, len(extra_pool)))
                        )
                    strategy = "top_bracket"
                else:
                    chosen = random.sample(top_pool, min(sz, len(top_pool)))
                    strategy = "top_bracket_exhausted"
            else:
                chosen = random.sample(candidates, sz)
                strategy = "exploration"
        elif scorer.loaded and mode != "random":
            scored = []
            for i in candidates:
                s = scorer.score(i)
                if s is not None:
                    scored.append((abs(s), i))
            if len(scored) >= sz:
                scored.sort(key=lambda x: x[0])
                pool = [x[1] for x in scored[: max(sz * 4, 16)]]
                chosen = random.sample(pool, min(sz, len(pool)))
                strategy = "uncertainty"
            else:
                chosen = random.sample(candidates, sz)
                strategy = "exploration"
        else:
            chosen = random.sample(candidates, sz)
            strategy = "exploration"

    recent_shown = get_recent_shown(domain_id)
    for cid in chosen:
        recent_shown.append(cid)

    cards = [{"id": cid, "url": f"/api/image/{cid}"} for cid in chosen]
    return {
        "ok": True,
        "domain_id": domain_id,
        "size": sz,
        "image_ids": chosen,
        "strategy": strategy,
        "retest_of": retest_of_id,
        "is_retest": 1 if retest_of_id else 0,
        "group": cards,
        "images": cards,
        "comparisons": cmp_count,
        "session_id": _request_session_id(request),
    }


@app.get("/api/comparisons")
def api_list_comparisons(
    domain_id: int = 1,
    image_id: Optional[int] = None,
    page: int = 1,
    per_page: int = 50,
    include_retests: bool = True,
):
    """List comparisons, optionally filtered to those involving one image.

    Used by /history to browse and bulk-delete comparison records.
    """
    page = max(1, page)
    per_page = max(1, min(200, per_page))
    offset = (page - 1) * per_page

    with db_conn() as c:
        dom = c.execute(
            "SELECT id, name FROM domains WHERE id=?", (domain_id,)
        ).fetchone()
        if not dom:
            raise HTTPException(404, "Domain not found")

        if image_id is not None:
            where = (
                "WHERE domain_id=? AND ("
                "  winner_id=? "
                "  OR EXISTS (SELECT 1 FROM json_each(losers_json) WHERE value=?) "
                "  OR EXISTS (SELECT 1 FROM json_each(shown_order_json) WHERE value=?)"
                ")"
            )
            base_params = (domain_id, image_id, image_id, image_id)
            if not include_retests:
                where += " AND is_retest=0"

            total = c.execute(
                f"SELECT COUNT(*) FROM comparisons {where}", base_params
            ).fetchone()[0]
            rows = c.execute(
                f"SELECT id, created_at, winner_id, losers_json, shown_order_json, "
                f"       sampler, is_retest, retest_of, group_size "
                f"FROM comparisons {where} ORDER BY id DESC LIMIT ? OFFSET ?",
                base_params + (per_page, offset)
            ).fetchall()
        else:
            if include_retests:
                where = "WHERE domain_id=?"
            else:
                where = "WHERE domain_id=? AND is_retest=0"
            total = c.execute(
                f"SELECT COUNT(*) FROM comparisons {where}", (domain_id,)
            ).fetchone()[0]
            rows = c.execute(
                f"SELECT id, created_at, winner_id, losers_json, shown_order_json, "
                f"       sampler, is_retest, retest_of, group_size "
                f"FROM comparisons {where} ORDER BY id DESC LIMIT ? OFFSET ?",
                (domain_id, per_page, offset)
            ).fetchall()

    comps = []
    for r in rows:
        comps.append({
            "id": r["id"],
            "created_at": r["created_at"],
            "winner_id": r["winner_id"],
            "losers": _safe_json_list(r["losers_json"]),
            "shown_order": _safe_json_list(r["shown_order_json"]),
            "sampler": r["sampler"] or "unknown",
            "is_retest": r["is_retest"],
            "retest_of": r["retest_of"],
            "group_size": r["group_size"],
        })

    return {
        "ok": True,
        "domain_id": domain_id,
        "domain_name": dom["name"],
        "image_id": image_id,
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": max(1, (total + per_page - 1) // per_page),
        "comparisons": comps,
    }


@app.post("/api/comparisons/bulk-delete")
def api_bulk_delete_comparisons(payload: BulkDeleteComparisonsBody):
    """Delete multiple comparisons by ID. Also deletes dependent retests."""
    with db_conn() as c:
        ph = ",".join("?" for _ in payload.comparison_ids)
        rows = c.execute(
            f"SELECT id, domain_id, session_id FROM comparisons WHERE id IN ({ph})",
            tuple(payload.comparison_ids)
        ).fetchall()

        valid_ids = list({
            r["id"] for r in rows if r["domain_id"] == payload.domain_id
        })
        if not valid_ids:
            raise HTTPException(400, "No valid comparisons found in this domain")

        ph_valid = ",".join("?" for _ in valid_ids)
        dep_rows = c.execute(
            f"SELECT id, session_id FROM comparisons WHERE retest_of IN ({ph_valid})",
            tuple(valid_ids)
        ).fetchall()
        dep_ids = [r["id"] for r in dep_rows if r["id"] not in valid_ids]

        # Session counter decrements
        session_dec = {}
        for r in rows:
            if r["domain_id"] == payload.domain_id and r["session_id"]:
                session_dec[r["session_id"]] = session_dec.get(r["session_id"], 0) + 1
        for r in dep_rows:
            if r["id"] in dep_ids and r["session_id"]:
                session_dec[r["session_id"]] = session_dec.get(r["session_id"], 0) + 1

        if dep_ids:
            ph_dep = ",".join("?" for _ in dep_ids)
            c.execute(f"DELETE FROM comparisons WHERE id IN ({ph_dep})", tuple(dep_ids))

        c.execute(f"DELETE FROM comparisons WHERE id IN ({ph_valid})", tuple(valid_ids))

        for sid, cnt in session_dec.items():
            c.execute(
                "UPDATE sessions SET comparison_count = MAX(0, comparison_count - ?) "
                "WHERE id=?",
                (cnt, sid)
            )
        c.commit()

    _invalidate_compared_pairs(payload.domain_id)
    _PERCENTILE_CACHE.pop(payload.domain_id, None)

    return {
        "ok": True,
        "domain_id": payload.domain_id,
        "deleted_direct": len(valid_ids),
        "deleted_retests": len(dep_ids),
        "total_deleted": len(valid_ids) + len(dep_ids),
    }


@app.post("/api/compare")
@app.post("/api/choose")
@app.post("/api/rate")
def api_rate(request: Request, payload: RateBody):
    if not payload.winner_id or not payload.shown_order:
        raise HTTPException(400, "Invalid rating submission")
    if payload.winner_id not in payload.shown_order:
        raise HTTPException(400, "winner_id must be in shown_order")

    with db_conn() as c:
        placeholders = ",".join("?" for _ in payload.shown_order)
        rows = c.execute(
            f"SELECT id FROM images WHERE domain_id=? AND id IN ({placeholders})",
            (payload.domain_id, *payload.shown_order)
        ).fetchall()
        valid = {r["id"] for r in rows}
        if valid != set(payload.shown_order):
            raise HTTPException(
                400, "shown_order contains image IDs not belonging to this domain"
            )

        # B1: validate retest linkage (client can no longer forge
        # is_retest / retest_of since /api/next is the only source)
        is_retest = 1 if payload.is_retest else 0
        retest_of = payload.retest_of
        if is_retest:
            if not retest_of:
                raise HTTPException(400, "is_retest requires retest_of")
            orig = c.execute(
                "SELECT id, domain_id, shown_order_json, group_size, is_retest "
                "FROM comparisons WHERE id=?",
                (retest_of,)
            ).fetchone()
            if not orig:
                raise HTTPException(400, f"retest_of #{retest_of} not found")
            if orig["domain_id"] != payload.domain_id:
                raise HTTPException(400, "retest_of belongs to a different domain")
            if orig["is_retest"]:
                raise HTTPException(400, "Cannot retest a retest")
            if orig["group_size"] != len(payload.shown_order):
                raise HTTPException(400, "retest group_size mismatch")
            if set(_safe_json_list(orig["shown_order_json"])) != set(payload.shown_order):
                raise HTTPException(400, "retest shown_order does not match original")
            existing = c.execute(
                "SELECT id FROM comparisons WHERE retest_of=? AND domain_id=? LIMIT 1",
                (retest_of, payload.domain_id)
            ).fetchone()
            if existing:
                raise HTTPException(409, f"comparison #{retest_of} already retested")
        else:
            retest_of = None

    losers = [x for x in payload.shown_order if x != payload.winner_id]
    session_id = _request_session_id(request)
    db.log_comparison(
        domain_id=payload.domain_id,
        winner_id=payload.winner_id,
        losers=losers,
        shown_order=payload.shown_order,
        session_id=session_id,
        sampler=payload.sampler,
        is_retest=is_retest,
        retest_of=retest_of,
    )
    return {"ok": True}


@app.post("/api/undo")
def api_undo(request: Request,
             payload: Optional[UndoBody] = None,
             domain_id: int = Query(1)):
    did = (payload.domain_id if (payload and payload.domain_id is not None)
           else domain_id)

    with db_conn() as c:
        row = c.execute(
            "SELECT id, session_id FROM comparisons WHERE domain_id=? "
            "ORDER BY id DESC LIMIT 1",
            (did,)
        ).fetchone()
        if not row:
            return {"ok": False, "msg": "Nothing to undo"}
        c.execute("DELETE FROM comparisons WHERE id=?", (row["id"],))
        if row["session_id"]:
            c.execute(
                "UPDATE sessions SET comparison_count = "
                "MAX(0, comparison_count - 1) WHERE id=?",
                (row["session_id"],)
            )
        c.commit()

    _invalidate_compared_pairs(did)
    return {"ok": True}


@app.get("/api/stats")
def api_stats(request: Request, domain_id: int = 1):
    with db_conn() as c:
        cmp_cnt = c.execute(
            "SELECT COUNT(*) FROM comparisons WHERE domain_id=?", (domain_id,)
        ).fetchone()[0]
        img_cnt = c.execute(
            "SELECT COUNT(*) FROM images WHERE domain_id=?", (domain_id,)
        ).fetchone()[0]

        shown_rows = c.execute(
            "SELECT DISTINCT shown_order_json FROM comparisons "
            "WHERE domain_id=? ORDER BY id DESC LIMIT 2000",
            (domain_id,)
        ).fetchall()

        seen = set()
        for r in shown_rows:
            for iid in _safe_json_list(r["shown_order_json"]):
                seen.add(iid)
        unique_shown = len(seen)

        session_cmp = 0
        sid = _request_session_id(request)
        if sid is not None:
            s_row = c.execute(
                "SELECT comparison_count FROM sessions WHERE id=?", (sid,)
            ).fetchone()
            if s_row:
                session_cmp = s_row[0]

    coverage_pct = round((unique_shown / img_cnt) * 100, 1) if img_cnt else 0.0
    scorer = get_scorer(domain_id)
    return {
        "ok": True,
        "domain_id": domain_id,
        "total_comparisons": cmp_cnt,
        "current_session_count": session_cmp,
        "total_corpus": img_cnt,
        "unique_images_shown": unique_shown,
        "coverage_pct": coverage_pct,
        "comparisons": cmp_cnt,
        "images": img_cnt,
        "is_trained": scorer.loaded,
        "accuracy": scorer.meta.get("accuracy") if scorer.loaded else None,
    }


# ═══════════════════════════════════════════════════
# MIRROR (TEST-RETEST)
# ═══════════════════════════════════════════════════
def _wilson_score_interval(successes: int, trials: int,
                           confidence: float = 0.95) -> dict:
    if trials <= 0:
        return {"low": 0.0, "high": 0.0, "center": 0.0}
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    p_hat = max(0.0, min(1.0, successes / trials))
    denom = 1 + (z ** 2) / trials
    center = (p_hat + (z ** 2) / (2 * trials)) / denom
    margin = (z / denom) * math.sqrt(
        (p_hat * (1 - p_hat) / trials) + ((z ** 2) / (4 * (trials ** 2)))
    )
    return {
        "center": round(center * 100, 2),
        "low": round(max(0.0, center - margin) * 100, 2),
        "high": round(min(1.0, center + margin) * 100, 2),
    }


@app.get("/api/mirror/stats")
def api_mirror_stats(domain_id: int = 1):
    with db_conn() as c:
        rows = c.execute("""
            SELECT r.id AS r_id, r.winner_id AS r_winner,
                   r.shown_order_json AS r_shown, r.created_at AS r_created,
                   o.id AS o_id, o.winner_id AS o_winner,
                   o.shown_order_json AS o_shown, o.sampler AS o_sampler,
                   o.created_at AS o_created
            FROM comparisons r
            JOIN comparisons o ON r.retest_of = o.id
            WHERE r.domain_id = ? AND r.is_retest = 1
            ORDER BY r.id DESC
        """, (domain_id,)).fetchall()

    n_retested = len(rows)
    if n_retested == 0:
        return {
            "ok": True,
            "domain_id": domain_id,
            "n_retested": 0,
            "overall_agreement_pct": 0.0,
            "wilson_95_ci": {"center": 0.0, "low": 0.0, "high": 0.0},
            "per_pair_agreement": {},
            "per_image_agreement": {},
            "sampler_breakdown": {},
            "recent_retests": [],
        }

    agreed_count = 0
    per_pair = {}
    per_image = {}
    sampler_counts = {}
    recent_retests = []

    for r in rows:
        o_cid = r["o_id"]
        o_w = r["o_winner"]
        r_w = r["r_winner"]
        sampler = r["o_sampler"] or "active"

        is_agree = (o_w == r_w)
        if is_agree:
            agreed_count += 1

        sampler_counts.setdefault(sampler, {"total": 0, "agreed": 0})
        sampler_counts[sampler]["total"] += 1
        if is_agree:
            sampler_counts[sampler]["agreed"] += 1

        per_pair[str(o_cid)] = {
            "retest_id": r["r_id"],
            "original_cid": o_cid,
            "agreement": is_agree,
            "n_flips": 0 if is_agree else 1,
            "winner_orig": o_w,
            "winner_retest": r_w,
            "sampler": sampler,
        }

        shown_imgs = _safe_json_list(r["r_shown"]) or [r_w]
        for img in shown_imgs:
            per_image.setdefault(img, {"wins": 0, "appearances": 0, "flips": 0})
            per_image[img]["appearances"] += 1
            if img == r_w:
                per_image[img]["wins"] += 1
            if not is_agree and (img == o_w or img == r_w):
                per_image[img]["flips"] += 1

        if len(recent_retests) < 15:
            recent_retests.append({
                "retest_id": r["r_id"],
                "original_id": o_cid,
                "agreed": is_agree,
                "winner_orig": o_w,
                "winner_retest": r_w,
                "sampler": sampler,
                "date": time.strftime(
                    "%Y-%m-%d %H:%M", time.localtime(r["r_created"])
                ),
            })

    overall_pct = round((agreed_count / n_retested) * 100, 2)
    wilson = _wilson_score_interval(agreed_count, n_retested)

    img_report = {}
    for img_id, stats in per_image.items():
        app = stats["appearances"]
        img_report[str(img_id)] = {
            "win_rate": round(stats["wins"] / app, 3) if app else 0.0,
            "flip_rate": round(stats["flips"] / app, 3) if app else 0.0,
            "n": app,
        }

    breakdown = {}
    for smp, counts in sampler_counts.items():
        tot = counts["total"]
        agr = counts["agreed"]
        breakdown[smp] = {
            "total": tot,
            "agreed": agr,
            "agreement_pct": round((agr / tot) * 100, 1) if tot else 0.0,
        }

    return {
        "ok": True,
        "domain_id": domain_id,
        "n_retested": n_retested,
        "agreed_count": agreed_count,
        "overall_agreement_pct": overall_pct,
        "wilson_95_ci": wilson,
        "per_pair_agreement": per_pair,
        "per_image_agreement": img_report,
        "sampler_breakdown": breakdown,
        "recent_retests": recent_retests,
    }


# ═══════════════════════════════════════════════════
# AUDIT & MISCLICK ANOMALY
# ═══════════════════════════════════════════════════
@app.get("/api/audit")
def api_audit(threshold: float = 0.35, domain_id: int = 1):
    with db_conn() as c:
        rows = c.execute(
            "SELECT id, created_at, winner_id, losers_json, shown_order_json "
            "FROM comparisons WHERE domain_id=? AND is_retest=0 ORDER BY id DESC",
            (domain_id,)
        ).fetchall()

    scorer = get_scorer(domain_id)
    if not scorer.loaded:
        return {
            "ok": False,
            "msg": (f"Domain #{domain_id} does not have a trained Ghost brain "
                    "yet. Train the domain first to audit misclicks!"),
            "total_comparisons": len(rows),
            "upsets": [],
        }

    upsets = []
    for r in rows:
        w_id = r["winner_id"]
        w_sc = scorer.score(w_id)
        if w_sc is None:
            continue
        losers_ids = _safe_json_list(r["losers_json"])
        shown_ids = _safe_json_list(r["shown_order_json"])

        losers_info = []
        best_loser_id = None
        best_loser_sc = -999.0
        for lid in losers_ids:
            l_sc = scorer.score(lid)
            losers_info.append(
                {"id": lid, "score": round(l_sc, 3) if l_sc is not None else None}
            )
            if l_sc is not None and l_sc > best_loser_sc:
                best_loser_sc = l_sc
                best_loser_id = lid

        if best_loser_id is not None:
            delta = best_loser_sc - w_sc
            if delta >= threshold:
                upsets.append({
                    "comparison_id": r["id"],
                    "created_at": r["created_at"],
                    "winner_id": w_id,
                    "winner_score": round(w_sc, 3),
                    "expected_winner_id": best_loser_id,
                    "expected_winner_score": round(best_loser_sc, 3),
                    "max_delta": round(delta, 3),
                    "shown_order": shown_ids,
                    "losers": losers_info,
                })

    upsets.sort(key=lambda x: x["max_delta"], reverse=True)
    return {
        "ok": True,
        "domain_id": domain_id,
        "total_comparisons": len(rows),
        "upsets": upsets,
    }


@app.post("/api/audit/flip")
def api_audit_flip(payload: AuditFlipBody):
    if not payload.comparison_id or not payload.new_winner_id:
        raise HTTPException(400, "comparison_id and new_winner_id are required")

    with db_conn() as c:
        row = c.execute(
            "SELECT id, domain_id, shown_order_json FROM comparisons WHERE id=?",
            (payload.comparison_id,)
        ).fetchone()
        if not row:
            raise HTTPException(404, "Comparison not found")

        shown = _safe_json_list(row["shown_order_json"])
        if payload.new_winner_id not in shown:
            raise HTTPException(
                400, "new_winner_id was not part of this comparison group"
            )

        new_losers = [x for x in shown if x != payload.new_winner_id]
        c.execute(
            "UPDATE comparisons SET winner_id=?, losers_json=? WHERE id=?",
            (payload.new_winner_id, json.dumps(new_losers),
             payload.comparison_id)
        )
        c.commit()
        did = row["domain_id"]

    _invalidate_compared_pairs(did)
    return {"ok": True}


@app.post("/api/audit/delete")
def api_audit_delete(payload: AuditDeleteBody):
    if not payload.comparison_id:
        raise HTTPException(400, "comparison_id is required")

    with db_conn() as c:
        row = c.execute(
            "SELECT session_id, domain_id FROM comparisons WHERE id=?",
            (payload.comparison_id,)
        ).fetchone()

        # B7: also delete any retest rows that reference this comparison
        dep_rows = c.execute(
            "SELECT id, session_id FROM comparisons WHERE retest_of=?",
            (payload.comparison_id,)
        ).fetchall()
        for dr in dep_rows:
            c.execute("DELETE FROM comparisons WHERE id=?", (dr["id"],))
            if dr["session_id"]:
                c.execute(
                    "UPDATE sessions SET comparison_count = "
                    "MAX(0, comparison_count - 1) WHERE id=?",
                    (dr["session_id"],)
                )

        c.execute("DELETE FROM comparisons WHERE id=?", (payload.comparison_id,))
        if row and row["session_id"]:
            c.execute(
                "UPDATE sessions SET comparison_count = "
                "MAX(0, comparison_count - 1) WHERE id=?",
                (row["session_id"],)
            )
        c.commit()
        did = row["domain_id"] if row else None

    if did is not None:
        _invalidate_compared_pairs(did)
    return {"ok": True}


# ═══════════════════════════════════════════════════
# TRAIN
# ═══════════════════════════════════════════════════
@app.post("/api/train")
def api_train(payload: Optional[TrainBody] = None,
              domain_id: int = Query(1)):
    did = payload.domain_id if payload else domain_id

    with db_conn() as c:
        cmp_cnt = c.execute(
            "SELECT COUNT(*) FROM comparisons WHERE domain_id=? AND is_retest=0",
            (did,)
        ).fetchone()[0]
    if cmp_cnt < 6:
        raise HTTPException(
            400,
            f"Need at least 6 rated comparisons in this domain to train "
            f"(currently have {cmp_cnt})."
        )

    import os
    import subprocess
    import sys
    from ghost.config import ROOT_DIR

    with _TRAIN_LOCK:
        existing = _TRAIN_PROCS.get(did)
        if existing is not None and existing.poll() is None:
            raise HTTPException(
                409, f"Training already running for Domain #{did}"
            )

        if os.name == "nt":
            py_exe = ROOT_DIR / ".venv" / "Scripts" / "python.exe"
            py_cmd = str(py_exe) if py_exe.is_file() else sys.executable
            # CREATE_NEW_CONSOLE gives the child its own terminal window AND
            # keeps Popen's handle tied to the actual Python process (unlike
            # shell=True + start "..." which detaches immediately). Ctrl+C in
            # the server terminal cannot kill training; closing the training
            # window does kill it.
            proc = subprocess.Popen(
                [py_cmd, "-m", "ghost.sensory.tournament", str(did)],
                cwd=str(ROOT_DIR),
                creationflags=subprocess.CREATE_NEW_CONSOLE,
            )
        else:
            py_exe = ROOT_DIR / ".venv" / "bin" / "python"
            py_cmd = str(py_exe) if py_exe.is_file() else sys.executable
            # start_new_session detaches the child from the server's process
            # group so Ctrl+C in the server doesn't kill training.
            proc = subprocess.Popen(
                [py_cmd, "-m", "ghost.sensory.tournament", str(did)],
                cwd=str(ROOT_DIR),
                start_new_session=True,
            )
        _TRAIN_PROCS[did] = proc

    return {
        "ok": True,
        "status": "launched",
        "domain_id": did,
        "pid": proc.pid,
        "message": f"Tournament training launched for Domain #{did}!",
    }


# ═══════════════════════════════════════════════════
# FLEX NEURAL SORTER
# ═══════════════════════════════════════════════════
@app.get("/api/flex/cards")
def api_flex_cards(domain_id: int = 1, limit: int = 0):
    with db_conn() as c:
        rows = c.execute(
            "SELECT id, filename FROM images WHERE domain_id=? ORDER BY id",
            (domain_id,)
        ).fetchall()
        dom = c.execute(
            "SELECT name FROM domains WHERE id=?", (domain_id,)
        ).fetchone()

    if not rows:
        raise HTTPException(404, "No images in this domain yet. Import images first!")

    scorer = get_scorer(domain_id)
    scored = []
    for r in rows:
        iid = r["id"]
        fname = r["filename"]
        if not (IMAGES_DIR / fname).is_file() and not list(
            IMAGES_DIR.glob(f"{iid}.*")
        ):
            continue
        if scorer.loaded:
            sc = scorer.score(iid)
            if sc is None:
                sc = 0.0
            reasons = ["Model-derived aesthetic score"]
            trained = True
        else:
            sc = _stable_score(iid)
            reasons = ["Untrained preview — train the domain for real scores"]
            trained = False
        scored.append({
            "id": int(iid),
            "score": round(float(sc), 3),
            "_reasons": reasons,
            "_trained": trained,
        })

    if not scored:
        raise HTTPException(404, "No valid image files found in this domain.")

    scored.sort(key=lambda item: item["score"])
    med_score = float(np.median([x["score"] for x in scored]))

    likes, dislikes = [], []
    for item in scored:
        if item["score"] >= med_score:
            item["decision"] = "like"
            item["confidence"] = round(
                min(99.8, 70.0 + (item["score"] - med_score) * 18.0), 1
            )
        else:
            item["decision"] = "dislike"
            item["confidence"] = round(
                min(99.8, 70.0 + (med_score - item["score"]) * 18.0), 1
            )
        item["reasons"] = item.pop("_reasons")
        item.pop("_trained", None)
        (likes if item["decision"] == "like" else dislikes).append(item)

    combined = []
    i, j = 0, 0
    while i < len(likes) or j < len(dislikes):
        if (random.random() < 0.5 and i < len(likes)) or j >= len(dislikes):
            combined.append(likes[i])
            i += 1
        else:
            combined.append(dislikes[j])
            j += 1

    for idx in range(0, len(combined) - 4, 4):
        chunk = combined[idx: idx + 4]
        random.shuffle(chunk)
        combined[idx: idx + 4] = chunk

    if limit and limit > 0:
        combined = combined[:limit]

    return {
        "ok": True,
        "domain_id": domain_id,
        "domain_name": dom["name"] if dom else f"Domain #{domain_id}",
        "cards": combined,
        "stats": {
            "total": len(scored),
            "model": (scorer.meta.get("algo", "Untrained Preview")
                      if scorer.loaded else "Untrained Preview"),
            "dim": scorer.meta.get("dim", 1024) if scorer.loaded else 1024,
            "accuracy": scorer.meta.get("accuracy", 0.0) if scorer.loaded else 0.0,
        },
    }


# ═══════════════════════════════════════════════════
# BROWSER EXTENSION
# ═══════════════════════════════════════════════════
@app.get("/api/lite/health")
@app.get("/api/ext/health")
def api_ext_health(domain_id: int = 1):
    with db_conn() as c:
        doms = c.execute("SELECT id, name FROM domains ORDER BY id").fetchall()
        cnt = c.execute(
            "SELECT COUNT(*) FROM images WHERE domain_id=?", (domain_id,)
        ).fetchone()[0]
    scorer = get_scorer(domain_id)
    return {
        "ok": True,
        "service": "Ghost in the GPU",
        "domain_id": domain_id,
        "logged_count": cnt,
        "is_trained": scorer.loaded,
        "domains": [{"id": d["id"], "name": d["name"]} for d in doms],
    }


def _decode_b64_image(b64: str):
    ext = ".jpg"
    if "," in b64:
        header, b64 = b64.split(",", 1)
        if "png" in header:
            ext = ".png"
        elif "webp" in header:
            ext = ".webp"
    if len(b64) > _MAX_B64_CHARS:
        raise HTTPException(
            413,
            f"Base64 payload too large ({len(b64)} chars, max {_MAX_B64_CHARS})",
        )
    try:
        return base64.b64decode(b64), ext
    except Exception as e:
        raise HTTPException(400, f"Invalid base64 payload: {e}")


@app.post("/api/lite/log")
@app.post("/api/ext/log")
def api_ext_log(payload: LiteLogBody):
    b64 = payload.image_base64 or ""
    img_url = payload.image_url or payload.page_url

    raw_bytes = None
    ext = ".jpg"
    if b64:
        raw_bytes, ext = _decode_b64_image(b64)

    if raw_bytes is None and img_url and img_url.startswith(("http://", "https://")):
        raw_bytes = _safe_download(img_url)

    if raw_bytes is None:
        raise HTTPException(400, "No valid image data provided")

    try:
        res = ingest_bytes(raw_bytes, domain_id=payload.domain_id, ext=ext,
                           source_url=img_url, batch_id="extension")
    except Exception as e:
        raise HTTPException(400, f"Failed to ingest image: {e}")

    if not res.get("ok"):
        raise HTTPException(400, res.get("error", "Failed to log image"))

    iid = res["image_id"]
    if res.get("status") == "added":
        _PERCENTILE_CACHE.pop(payload.domain_id, None)
    scorer = get_scorer(payload.domain_id)
    score = scorer.score(iid) if scorer.loaded else None
    tier_info = _percentile_and_tier(payload.domain_id, score, scorer)

    return {
        "ok": True,
        "status": res["status"],
        "id": iid,
        "image_id": iid,
        "domain_id": payload.domain_id,
        "score": round(score, 2) if score is not None else 0.0,
        "percentile": tier_info["percentile"],
        "tier": tier_info["tier"],
        "badge": tier_info["badge"],
        "tier_label": tier_info["badge"],
    }


@app.post("/api/lite/test")
@app.post("/api/ext/test")
def api_ext_test(payload: LiteTestBody):
    """Temporary test score: evaluates in-memory without touching DB or disk."""
    b64 = payload.image_base64 or ""
    img_url = payload.image_url or payload.page_url

    raw_bytes = None
    if b64:
        raw_bytes, _ = _decode_b64_image(b64)
    elif img_url and img_url.startswith(("http://", "https://")):
        raw_bytes = _safe_download(img_url)

    if not raw_bytes or len(raw_bytes) < 100:
        raise HTTPException(400, "Image payload is empty or invalid")

    try:
        import io
        from PIL import Image, ImageOps
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            img = Image.open(io.BytesIO(raw_bytes))
            img.load()
            img = ImageOps.exif_transpose(img) or img
            img = img.convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Cannot decode image: {e}")

    from ghost.sensory.backbone import embed_image
    emb = embed_image(img)

    scorer = get_scorer(payload.domain_id)
    score = None

    if scorer.loaded and scorer.pca is not None and scorer.w is not None:
        try:
            v = scorer.pca.transform(emb.reshape(1, -1))[0]
            score = float(np.dot(scorer.w, v) + scorer.b)
        except Exception as e:
            logger.warning("lite/test: scoring failed: %s", e, exc_info=True)
            score = None

    tier_info = _percentile_and_tier(payload.domain_id, score, scorer)

    return {
        "ok": True,
        "status": "tested",
        "domain_id": payload.domain_id,
        "score": round(score, 2) if score is not None else 0.0,
        "percentile": tier_info["percentile"],
        "tier": tier_info["tier"],
        "badge": tier_info["badge"],
        "tier_label": tier_info["badge"],
    }

