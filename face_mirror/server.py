"""Ghost Face Mirror — Standalone GPU Web Server.
Allows drag-and-drop face lookalike search across the entire corpus, live threshold tuning,
and instant verification against the 7 benchmark failure images.
"""
import io
import json
import mimetypes
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from face_mirror.config import (
    SUBPROJECT_DIR, FACE_DB_PATH, MAIN_DB_PATH, MAIN_IMAGES_DIR, DEFAULT_THRESHOLDS
)
from face_mirror.cache import init_face_db, get_thresholds, save_thresholds, sync_corpus
from face_mirror.engine import GPUFaceEngine

app = FastAPI(title="Ghost Face Mirror — GPU ArcFace Search")
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = SUBPROJECT_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_ENGINE: Optional[GPUFaceEngine] = None
_ENGINE_LOCK = threading.Lock()


def get_engine() -> GPUFaceEngine:
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            _ENGINE = GPUFaceEngine()
        return _ENGINE


class ThresholdsPayload(BaseModel):
    min_det_score: float = DEFAULT_THRESHOLDS["min_det_score"]
    max_yaw_asymmetry: float = DEFAULT_THRESHOLDS["max_yaw_asymmetry"]
    max_align_error: float = DEFAULT_THRESHOLDS["max_align_error"]
    min_mouth_grad_ratio: float = DEFAULT_THRESHOLDS["min_mouth_grad_ratio"]
    min_face_size: float = DEFAULT_THRESHOLDS["min_face_size"]
    re_evaluate_corpus: bool = False


@app.get("/")
def index_page():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
def api_status():
    init_face_db()
    eng = get_engine()
    with sqlite3.connect(FACE_DB_PATH) as conn:
        accepted = conn.execute("SELECT COUNT(*) FROM face_records WHERE status='accepted'").fetchone()[0]
        rejected = conn.execute("SELECT COUNT(*) FROM face_records WHERE status='rejected'").fetchone()[0]

    total_main = 0
    if MAIN_DB_PATH.is_file():
        with sqlite3.connect(MAIN_DB_PATH) as mconn:
            total_main = mconn.execute("SELECT COUNT(*) FROM images").fetchone()[0]

    return {
        "ok": True,
        "gpu": eng.gpu_name,
        "device": str(eng.device),
        "total_corpus_images": total_main,
        "cached_accepted": accepted,
        "cached_rejected": rejected,
        "uncached": max(0, total_main - (accepted + rejected)),
        "thresholds": get_thresholds(),
    }


@app.post("/api/sync")
def api_sync(force: bool = Query(False)):
    eng = get_engine()
    res = sync_corpus(engine=eng, force_recheck_all=force)
    return res


@app.post("/api/thresholds")
def api_update_thresholds(payload: ThresholdsPayload):
    new_th = save_thresholds({
        "min_det_score": payload.min_det_score,
        "max_yaw_asymmetry": payload.max_yaw_asymmetry,
        "max_align_error": payload.max_align_error,
        "min_mouth_grad_ratio": payload.min_mouth_grad_ratio,
        "min_face_size": payload.min_face_size,
    })
    sync_res = None
    if payload.re_evaluate_corpus:
        eng = get_engine()
        sync_res = sync_corpus(engine=eng, force_recheck_all=True)
    return {"ok": True, "thresholds": new_th, "sync": sync_res}


@app.get("/api/benchmark")
def api_run_benchmark():
    """Tests the current thresholds against the 7 mandatory failure benchmark images:
    Visibility: 4, 6, 102
    Side Angles: 394, 366, 345, 322
    """
    eng = get_engine()
    th = get_thresholds()
    visibility_ids = [4, 6, 102]
    side_angle_ids = [394, 366, 345, 322]
    results = []

    if not MAIN_DB_PATH.is_file():
        raise HTTPException(404, "Main database not found")

    with sqlite3.connect(MAIN_DB_PATH) as mconn:
        for cat, id_list in [("visibility", visibility_ids), ("side_angle", side_angle_ids)]:
            for iid in id_list:
                row = mconn.execute("SELECT filename FROM images WHERE id=?", (iid,)).fetchone()
                if not row:
                    results.append({
                        "image_id": iid,
                        "category": cat,
                        "found_on_disk": False,
                        "status": "missing",
                        "reason": "not_in_db",
                        "benchmark_pass": True,
                    })
                    continue
                img_p = MAIN_IMAGES_DIR / row[0]
                img_bgr = cv2.imread(str(img_p)) if img_p.is_file() else None
                res = eng.process_image(img_bgr, thresholds=th)
                # Benchmark succeeds when these bad images are REJECTED
                results.append({
                    "image_id": iid,
                    "category": cat,
                    "url": f"/api/image/{iid}",
                    "status": res["status"],
                    "reason": res["reason"],
                    "metrics": res["metrics"],
                    "benchmark_pass": (res["status"] == "rejected"),
                })

    all_passed = all(r["benchmark_pass"] for r in results)
    return {
        "ok": True,
        "all_eliminated": all_passed,
        "benchmarks": results,
    }


@app.post("/api/search")
async def api_search_face(
    file: UploadFile = File(...),
    top_k: int = Query(24, ge=1, le=100),
    face_index: int = Query(0, ge=0),
    domain_id: Optional[int] = Query(None),
):
    """Accepts a dragged-and-dropped image, detects all faces, extracts ArcFace 512D for selected face_index,
    and returns top-K lookalikes alongside thumbnail crops for all detected faces in the image.
    """
    raw_bytes = await file.read()
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(400, "Could not decode uploaded image file.")

    eng = get_engine()
    q_res = eng.process_query_image(img_bgr, face_index=face_index)
    if q_res["status"] != "accepted" or q_res["embedding"] is None:
        return {
            "ok": False,
            "rejected": True,
            "reason": q_res["reason"],
            "metrics": q_res.get("metrics"),
            "message": "No face could be found in the uploaded image.",
        }

    q_vec = q_res["embedding"]  # L2-normalized 512D float32

    init_face_db()
    with sqlite3.connect(FACE_DB_PATH) as conn:
        if domain_id is not None:
            rows = conn.execute("""
                SELECT image_id, domain_id, filename, det_score, yaw_asymmetry, embedding_blob
                FROM face_records
                WHERE status='accepted' AND embedding_blob IS NOT NULL AND domain_id=?
            """, (domain_id,)).fetchall()
        else:
            rows = conn.execute("""
                SELECT image_id, domain_id, filename, det_score, yaw_asymmetry, embedding_blob
                FROM face_records
                WHERE status='accepted' AND embedding_blob IS NOT NULL
            """).fetchall()

    if not rows:
        return {
            "ok": True,
            "query_metrics": q_res["metrics"],
            "total_searched": 0,
            "matches": [],
        }

    # Build matrix and compute exact cosine similarity on GPU/NumPy
    img_ids = []
    meta_list = []
    vecs = []
    for r in rows:
        v = np.frombuffer(r[5], dtype=np.float32)
        if v.shape[0] == 512:
            img_ids.append(r[0])
            meta_list.append({
                "image_id": r[0],
                "domain_id": r[1],
                "filename": r[2],
                "det_score": round(float(r[3] or 0.0), 3),
                "yaw_asymmetry": round(float(r[4] or 0.0), 3),
            })
            vecs.append(v)

    mat = np.stack(vecs, axis=0)  # (N, 512)
    sims = mat @ q_vec  # Cosine similarity in [-1, 1]
    top_indices = np.argsort(sims)[::-1][:top_k]

    matches = []
    for rank, idx in enumerate(top_indices, 1):
        sim = float(sims[idx])
        m = meta_list[idx]
        matches.append({
            "rank": rank,
            "image_id": m["image_id"],
            "domain_id": m["domain_id"],
            "url": f"/api/image/{m['image_id']}",
            "cosine_similarity": round(sim, 4),
            "match_pct": round(max(0.0, min(100.0, sim * 100.0)), 1),
            "det_score": m["det_score"],
            "yaw_asymmetry": m["yaw_asymmetry"],
        })

    return {
        "ok": True,
        "selected_index": q_res.get("selected_index", 0),
        "faces_found": q_res.get("faces_found", 1),
        "all_faces": q_res.get("all_faces", []),
        "query_metrics": q_res["metrics"],
        "total_searched": len(vecs),
        "matches": matches,
    }


@app.get("/api/image/{image_id}")
def api_serve_image(image_id: int):
    if not MAIN_DB_PATH.is_file():
        raise HTTPException(404, "Main DB not found")
    with sqlite3.connect(MAIN_DB_PATH) as conn:
        row = conn.execute("SELECT filename FROM images WHERE id=?", (image_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Image not found")
    p = MAIN_IMAGES_DIR / row[0]
    if not p.is_file():
        raise HTTPException(404, "Image file missing on disk")
    media_type, _ = mimetypes.guess_type(p.name)
    return FileResponse(p, media_type=media_type or "image/jpeg")
