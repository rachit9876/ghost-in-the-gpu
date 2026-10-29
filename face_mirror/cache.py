"""Ghost Face Mirror — Isolated SQLite Database & Dynamic Incremental GPU Sync.
Reads image list from main corpus (taste.db), runs GPU Face Engine on any uncached images,
and stores 512D ArcFace vectors + quality metrics in face_cache.db.
"""
import json
import sqlite3
import sys
import time
from pathlib import Path
import cv2
import numpy as np

from face_mirror.config import (
    FACE_DB_PATH, MAIN_DB_PATH, MAIN_IMAGES_DIR, DEFAULT_THRESHOLDS
)

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def init_face_db():
    with sqlite3.connect(FACE_DB_PATH) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS face_records (
                image_id INTEGER PRIMARY KEY,
                domain_id INTEGER,
                filename TEXT,
                sha256 TEXT,
                status TEXT,             -- 'accepted' or 'rejected'
                reason TEXT,
                det_score REAL,
                yaw_asymmetry REAL,
                align_error REAL,
                mouth_grad_ratio REAL,
                bbox_json TEXT,
                embedding_blob BLOB,     -- 512 float32 bytes (2048 bytes) if accepted
                processed_at REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS face_settings (
                key TEXT PRIMARY KEY,
                value_json TEXT
            )
        """)
        # Initialize default thresholds if not set
        row = conn.execute("SELECT value_json FROM face_settings WHERE key='thresholds'").fetchone()
        if not row:
            conn.execute(
                "INSERT INTO face_settings (key, value_json) VALUES ('thresholds', ?)",
                (json.dumps(DEFAULT_THRESHOLDS),)
            )
        conn.commit()


def get_thresholds() -> dict:
    init_face_db()
    with sqlite3.connect(FACE_DB_PATH) as conn:
        row = conn.execute("SELECT value_json FROM face_settings WHERE key='thresholds'").fetchone()
        if row:
            try:
                return {**DEFAULT_THRESHOLDS, **json.loads(row[0])}
            except Exception:
                pass
    return dict(DEFAULT_THRESHOLDS)


def save_thresholds(new_th: dict) -> dict:
    init_face_db()
    merged = {**DEFAULT_THRESHOLDS, **new_th}
    with sqlite3.connect(FACE_DB_PATH) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO face_settings (key, value_json) VALUES ('thresholds', ?)",
            (json.dumps(merged),)
        )
        conn.commit()
    return merged


def sync_corpus(engine=None, force_recheck_all: bool = False):
    """Incrementally scans main corpus (taste.db) and caches any new or modified images on GPU."""
    init_face_db()
    if not MAIN_DB_PATH.is_file():
        print("[!] Main database not found at:", MAIN_DB_PATH)
        return {"ok": False, "error": "Main database not found"}

    with sqlite3.connect(MAIN_DB_PATH) as main_conn:
        main_rows = main_conn.execute(
            "SELECT id, domain_id, filename, sha256 FROM images ORDER BY id ASC"
        ).fetchall()

    with sqlite3.connect(FACE_DB_PATH) as face_conn:
        if force_recheck_all:
            face_conn.execute("DELETE FROM face_records")
            face_conn.commit()
            cached_map = {}
        else:
            cached_rows = face_conn.execute(
                "SELECT image_id, sha256 FROM face_records"
            ).fetchall()
            cached_map = {r[0]: r[1] for r in cached_rows}

        # Prune deleted images that no longer exist in main corpus
        main_ids = {r[0] for r in main_rows}
        orphans = set(cached_map.keys()) - main_ids
        if orphans:
            placeholders = ",".join("?" for _ in orphans)
            face_conn.execute(f"DELETE FROM face_records WHERE image_id IN ({placeholders})", tuple(orphans))
            face_conn.commit()

    # Determine which images need GPU processing
    to_process = []
    for iid, dom_id, fn, sha in main_rows:
        if iid not in cached_map or cached_map[iid] != sha:
            to_process.append((iid, dom_id, fn, sha))

    if not to_process:
        with sqlite3.connect(FACE_DB_PATH) as face_conn:
            acc = face_conn.execute("SELECT COUNT(*) FROM face_records WHERE status='accepted'").fetchone()[0]
            rej = face_conn.execute("SELECT COUNT(*) FROM face_records WHERE status='rejected'").fetchone()[0]
        print(f"[✓] Face Cache is 100% up-to-date! ({acc} valid faces, {rej} filtered out)")
        return {"ok": True, "newly_processed": 0, "accepted": acc, "rejected": rej}

    if engine is None:
        from face_mirror.engine import GPUFaceEngine
        engine = GPUFaceEngine()

    thresholds = get_thresholds()
    total = len(to_process)
    print(f"[*] Syncing {total} new/updated corpus image(s) on GPU ({engine.gpu_name})...")

    t0 = time.time()
    new_acc = 0
    new_rej = 0

    with sqlite3.connect(FACE_DB_PATH) as face_conn:
        for idx, (iid, dom_id, fn, sha) in enumerate(to_process, 1):
            img_p = MAIN_IMAGES_DIR / fn
            img_bgr = cv2.imread(str(img_p)) if img_p.is_file() else None
            if img_bgr is None:
                res = {
                    "status": "rejected",
                    "reason": "file_missing_or_unreadable",
                    "embedding": None,
                    "metrics": {"score": 0.0, "yaw_asymmetry": 0.0, "align_error": 0.0, "mouth_grad_ratio": 0.0},
                }
            else:
                res = engine.process_image(img_bgr, thresholds=thresholds)

            status = res["status"]
            if status == "accepted":
                new_acc += 1
                emb_blob = res["embedding"].astype(np.float32).tobytes()
                bbox_json = json.dumps(res.get("bbox", []))
            else:
                new_rej += 1
                emb_blob = None
                bbox_json = None

            m = res.get("metrics") or {}
            face_conn.execute("""
                INSERT OR REPLACE INTO face_records (
                    image_id, domain_id, filename, sha256, status, reason,
                    det_score, yaw_asymmetry, align_error, mouth_grad_ratio,
                    bbox_json, embedding_blob, processed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                iid, dom_id, fn, sha, status, res["reason"],
                m.get("score", 0.0), m.get("yaw_asymmetry", 0.0),
                m.get("align_error", 0.0), m.get("mouth_grad_ratio", 0.0),
                bbox_json, emb_blob, time.time()
            ))

            if idx % 25 == 0 or idx == total:
                face_conn.commit()
                elapsed = time.time() - t0
                rate = idx / max(elapsed, 1e-3)
                print(f"    [{idx:4d}/{total:4d}] {rate:.1f} img/s | Accepted: {new_acc} | Filtered: {new_rej}")

        total_acc = face_conn.execute("SELECT COUNT(*) FROM face_records WHERE status='accepted'").fetchone()[0]
        total_rej = face_conn.execute("SELECT COUNT(*) FROM face_records WHERE status='rejected'").fetchone()[0]

    print(f"[✓] GPU Face Sync complete in {time.time() - t0:.1f}s! Active face bank: {total_acc} faces.")
    return {
        "ok": True,
        "newly_processed": total,
        "accepted": total_acc,
        "rejected": total_rej,
    }


if __name__ == "__main__":
    force = "--force" in sys.argv
    sync_corpus(force_recheck_all=force)
