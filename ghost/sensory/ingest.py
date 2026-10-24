"""Ghost in the GPU — Domain-Aware Atomic Ingestion & Embedding Pipeline."""
import hashlib
import io
import ipaddress
import logging
import os
import socket
import sqlite3
import time
import urllib.error
import urllib.request
import warnings
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from PIL import Image, ImageOps

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass

try:
    import pillow_avif  # noqa: F401
except Exception:
    pass

from ghost.config import IMAGES_DIR, EMBEDDINGS_DIR
from ghost.rater import db

logger = logging.getLogger("ghost.ingest")

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".avif", ".heic", ".bmp"}

# Decompression-bomb guard. 80 MP comfortably covers 8K ultrawide (~33 MP)
# and modern phone sensors (~50 MP). Anything above is rejected.
# PIL raises DecompressionBombWarning above this and DecompressionBombError
# above 2x this. We treat the warning as an error to hard-cap at 80 MP.
Image.MAX_IMAGE_PIXELS = 80_000_000

# PIL format string -> canonical file extension.
_FORMAT_TO_EXT = {
    "JPEG": ".jpg",
    "MPO": ".jpg",   # multi-picture JPEG (common in phone shots)
    "PNG": ".png",
    "WEBP": ".webp",
    "HEIF": ".heic",
    "AVIF": ".avif",
    "BMP": ".bmp",
    "GIF": ".gif",
    "TIFF": ".tiff",
}

_REMOTE_TIMEOUT_S = 15
_REMOTE_MAX_BYTES = 25_000_000
_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"


# ═══════════════════════════════════════════════════
# ATOMIC FILE WRITES
# ═══════════════════════════════════════════════════
def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "wb") as f:
            np.save(f, arr)
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


# ═══════════════════════════════════════════════════
# IMAGE DECODE + NORMALIZE
# ═══════════════════════════════════════════════════
def _decode_and_normalize(raw_bytes: bytes):
    """Decode, bomb-check, EXIF-correct, and return (RGB image, real_ext).

    Raises ValueError on bad payloads or decompression bombs.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            pil_img = Image.open(io.BytesIO(raw_bytes))
            pil_img.load()  # force full decode (triggers bomb check)
    except Image.DecompressionBombError as e:
        raise ValueError(f"Image exceeds pixel limit: {e}")
    except Image.DecompressionBombWarning as e:
        raise ValueError(f"Image exceeds pixel limit: {e}")
    except Exception as e:
        raise ValueError(f"Cannot decode image: {e}")

    real_ext = _FORMAT_TO_EXT.get((pil_img.format or "").upper(), ".jpg")

    try:
        transposed = ImageOps.exif_transpose(pil_img)
        if transposed is not None:
            pil_img = transposed
    except Exception:
        pass

    return pil_img.convert("RGB"), real_ext


# ═══════════════════════════════════════════════════
# INGEST (bytes)
# ═══════════════════════════════════════════════════
def ingest_bytes(raw_bytes: bytes, domain_id: int = 1, ext: str = ".jpg",
                 source_url: str = None, batch_id: str = "import") -> dict:
    """Atomically deduplicate per-domain, embed, persist image + .npy + DB row.

    `ext` is advisory — the on-disk extension is derived from the true format.
    """
    if not raw_bytes or len(raw_bytes) < 200:
        return {"ok": False, "status": "error", "error": "Image too small"}

    sha256 = hashlib.sha256(raw_bytes).hexdigest()

    # 1. Cheap dedup check (per-domain) + cross-domain embedding reuse probe.
    with db.connection() as c:
        existing = c.execute(
            "SELECT id, domain_id FROM images WHERE domain_id=? AND sha256=?",
            (domain_id, sha256),
        ).fetchone()
        if existing:
            return {
                "ok": True,
                "status": "exists",
                "image_id": existing["id"],
                "domain_id": existing["domain_id"],
            }
        other_row = c.execute(
            "SELECT id FROM images WHERE sha256=? LIMIT 1", (sha256,)
        ).fetchone()

    # 2. Try to reuse an embedding from another domain.
    vec = None
    if other_row:
        existing_emb = EMBEDDINGS_DIR / f"{other_row['id']}.npy"
        if existing_emb.is_file():
            try:
                vec = np.load(existing_emb).astype(np.float32).flatten()
            except Exception:
                vec = None

    # 3. Decode + validate + EXIF-correct + derive true extension.
    try:
        pil_img, real_ext = _decode_and_normalize(raw_bytes)
    except ValueError as e:
        return {"ok": False, "status": "error", "error": str(e)}

    # 4. Embed (unless reused).
    if vec is None:
        try:
            from ghost.sensory.backbone import embed_image
            vec = embed_image(pil_img)
        except Exception as e:
            return {"ok": False, "status": "error",
                    "error": f"Failed to embed image: {e}"}

    # 5. Persist image + embedding + DB row atomically.
    now = time.time()
    image_id = None
    dest_path = None
    emb_path = None

    try:
        with db.connection() as c:
            cur = c.execute(
                "INSERT INTO images "
                "(domain_id, sha256, filename, source_url, added_at, batch_id) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (domain_id, sha256, "pending", source_url, now, batch_id),
            )
            image_id = cur.lastrowid

            filename = f"{image_id}{real_ext}"
            dest_path = IMAGES_DIR / filename
            emb_path = EMBEDDINGS_DIR / f"{image_id}.npy"

            _atomic_write_bytes(dest_path, raw_bytes)
            _atomic_save_npy(emb_path, vec)

            c.execute("UPDATE images SET filename=? WHERE id=?",
                      (filename, image_id))
            c.commit()

    except sqlite3.IntegrityError as e:
        # Deduplication race: another ingest inserted (domain_id, sha256)
        # between our check and our INSERT.
        if dest_path is not None:
            dest_path.unlink(missing_ok=True)
        if emb_path is not None:
            emb_path.unlink(missing_ok=True)

        if "UNIQUE" not in str(e).upper():
            logger.warning("ingest[%s]: integrity error: %s", domain_id, e,
                           exc_info=True)
            return {"ok": False, "status": "error", "error": str(e)}

        with db.connection() as c2:
            row = c2.execute(
                "SELECT id, domain_id FROM images "
                "WHERE domain_id=? AND sha256=?",
                (domain_id, sha256),
            ).fetchone()
        if row:
            return {
                "ok": True,
                "status": "exists",
                "image_id": row["id"],
                "domain_id": row["domain_id"],
            }
        return {"ok": False, "status": "error",
                "error": "UNIQUE collision but existing row not found"}

    except Exception as e:
        if dest_path is not None:
            dest_path.unlink(missing_ok=True)
        if emb_path is not None:
            emb_path.unlink(missing_ok=True)
        logger.warning("ingest[%s]: failed to persist: %s", domain_id, e,
                       exc_info=True)
        return {"ok": False, "status": "error", "error": str(e)}

    return {
        "ok": True,
        "status": "added",
        "image_id": image_id,
        "domain_id": domain_id,
    }


# ═══════════════════════════════════════════════════
# SSRF GUARD (used by ingest_url)
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


# ═══════════════════════════════════════════════════
# INGEST (URL)
# ═══════════════════════════════════════════════════
def ingest_url(url: str, domain_id: int = 1, batch_id: str = "url_import") -> dict:
    url = (url or "").strip()
    if not url:
        return {"ok": False, "status": "error", "error": "Empty URL"}
    if not _is_safe_url(url):
        return {"ok": False, "status": "error", "error": "URL is not allowed"}

    try:
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with _SAFE_OPENER.open(req, timeout=_REMOTE_TIMEOUT_S) as resp:
            raw = resp.read(_REMOTE_MAX_BYTES + 1)
    except Exception as e:
        return {"ok": False, "status": "error", "error": str(e)}

    if len(raw) > _REMOTE_MAX_BYTES:
        return {"ok": False, "status": "error", "error": "Remote image too large"}

    ext = Path(url.split("?")[0]).suffix.lower() or ".jpg"
    return ingest_bytes(raw, domain_id=domain_id, ext=ext,
                        source_url=url, batch_id=batch_id)


# ═══════════════════════════════════════════════════
# INGEST (folder, CLI convenience)
# ═══════════════════════════════════════════════════
def ingest_folder(folder_path: str, domain_id: int = 1,
                  batch_id: str = "folder_import") -> None:
    folder = Path(folder_path)
    if not folder.is_dir():
        logger.error("Folder not found: %s", folder_path)
        return

    db.init()
    files = [f for f in folder.rglob("*")
             if f.suffix.lower() in VALID_EXTENSIONS]
    logger.info("Found %d images in %s for Domain #%s",
                len(files), folder_path, domain_id)

    added = existed = errors = 0
    for idx, f in enumerate(files, 1):
        try:
            res = ingest_bytes(f.read_bytes(), domain_id=domain_id,
                               ext=f.suffix, batch_id=batch_id)
        except Exception as e:
            logger.warning("ingest_folder: %s -> %s", f, e)
            errors += 1
            continue

        status = res.get("status")
        if status == "added":
            added += 1
        elif status == "exists":
            existed += 1
        else:
            errors += 1

        if idx % 10 == 0 or idx == len(files):
            logger.info("[%d/%d] added=%d existed=%d errors=%d",
                        idx, len(files), added, existed, errors)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    if len(sys.argv) > 1:
        dom = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        ingest_folder(sys.argv[1], domain_id=dom)
    else:
        print("Usage: python -m ghost.sensory.ingest <folder_path> [domain_id]")

