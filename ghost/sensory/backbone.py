"""Ghost in the GPU — Unified Vision Backbone.
Extracts 1024D semantic perception embeddings using Meta PE-Core L14-336.
"""
import logging
import sys
import threading
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

log = logging.getLogger(__name__)

VENDOR_DIR = Path(__file__).resolve().parent.parent / "vendor" / "perception_models"
if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms

_FEATURE_DIM = 1024
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_MODEL_LOCK = threading.Lock()
_BUNDLE = None  # (model, transform, device) — published atomically


def _load_backbone():
    """Build model + transform + device, warm up, and return the bundle."""
    log.info("Booting PE-Core Vision Backbone on %s", _DEVICE.upper())

    model = pe.CLIP.from_config("PE-Core-L14-336", pretrained=True).to(_DEVICE).eval()
    transform = transforms.get_image_transform(model.image_size)

    try:
        sz = model.image_size
        dummy = torch.zeros((1, 3, sz, sz), dtype=torch.float32, device=_DEVICE)
        with torch.inference_mode():
            feat = model.encode_image(dummy)
            if _DEVICE == "cuda":
                torch.cuda.synchronize()
        if feat.shape[-1] != _FEATURE_DIM:
            raise RuntimeError(
                f"Backbone produced {feat.shape[-1]}-D features, expected {_FEATURE_DIM}"
            )
        log.info("PE-Core Vision Backbone warmed up on %s", _DEVICE.upper())
    except Exception:
        log.warning("Backbone warmup skipped", exc_info=True)

    return model, transform, _DEVICE


def get_vision_model():
    """Return (model, transform, device), initializing once under a lock."""
    global _BUNDLE
    bundle = _BUNDLE
    if bundle is not None:
        return bundle
    with _MODEL_LOCK:
        if _BUNDLE is None:
            _BUNDLE = _load_backbone()
        return _BUNDLE


def _embed_image_single(image_input: Union[str, Path, Image.Image]) -> np.ndarray:
    """Embed a PIL Image or path into a 1024D normalized perception vector."""
    model, transform, device = get_vision_model()

    if isinstance(image_input, (str, Path)):
        with Image.open(image_input) as f:
            img = f.convert("RGB")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
    else:
        raise TypeError(
            f"embed_image expects str, Path, or PIL.Image; got {type(image_input).__name__}"
        )

    tensor = transform(img).unsqueeze(0).to(device)
    with torch.inference_mode():
        features = model.encode_image(tensor)
        features = F.normalize(features, dim=-1, eps=1e-12)

    return features.cpu().numpy()[0].astype(np.float32)

def embed_image(pil_img, n_crops: int = 6) -> np.ndarray:
    """CANONICAL embedding API. Smart multi-crop, aspect-aware, always 1:1 crops.

    Every crop is 1:1 square → encoder gets pure resize (no squash distortion).
    Number of crops adapts to aspect ratio so wide images get proper coverage.

    `_embed_image_single` is kept only for diagnostic comparison and should not
    be used in production paths. The `n_crops` argument is ignored (aspect
    determines N) — kept for backward compatibility with existing callers.
    """
    model, transform, device = get_vision_model()
    W, H = pil_img.size
    
    if W < 32 or H < 32:
        views = [pil_img]
    else:
        views = _generate_smart_crops(pil_img)
    
    embs = []
    for v in views:
        with torch.inference_mode():
            t = transform(v).unsqueeze(0).to(device)
            emb = model.encode_image(t)
            emb = F.normalize(emb, dim=-1, eps=1e-12)
        embs.append(emb.cpu().numpy().astype(np.float32)[0])
    
    avg = np.mean(embs, axis=0)
    avg = avg / np.linalg.norm(avg)
    return avg.astype(np.float32)


def _generate_smart_crops(pil_img, min_n: int = 3, max_n: int = 8):
    """Generate N aspect-aware strictly-1:1 crops covering the image essence.
    
    - Square/near-square (aspect <= 1.2): 3 center crops at scales 100/85/70%
      (all strictly 1:1 — no squash)
    - Non-square: N sliding-window crops along the longer axis
      N = clamp(round(aspect * 2), 3, 8)
    
    Every crop is guaranteed square → encoder gets pure resize, no distortion.
    """
    W, H = pil_img.size
    aspect = max(W, H) / min(W, H)
    side = min(W, H)
    
    crops = []
    
    if aspect <= 1.2:
        # Near-square: 3 strictly 1:1 center crops at scales 100/85/70%
        off_x = (W - side) // 2
        off_y = (H - side) // 2
        crops.append(pil_img.crop((off_x, off_y, off_x + side, off_y + side)))
        for frac in (0.85, 0.7):
            s = int(side * frac)
            cx = (W - s) // 2
            cy = (H - s) // 2
            crops.append(pil_img.crop((cx, cy, cx + s, cy + s)))
    else:
        # Non-square: sliding window along longer axis.
        # Adaptive cap: n_crops >= aspect guarantees full coverage
        # with no blind gaps. Capped at 32 crops to bound compute
        # for pathological inputs (>32:1) where full coverage would
        # be impractical.
        effective_max = max(8, min(32, int(aspect) + 1))
        n_crops = max(min_n, min(int(round(aspect * 2)), effective_max))
        if W > H:
            shift = W - side
            for i in range(n_crops):
                pos = int(shift * i / (n_crops - 1)) if n_crops > 1 else shift // 2
                crops.append(pil_img.crop((pos, 0, pos + side, side)))
        else:
            shift = H - side
            for i in range(n_crops):
                pos = int(shift * i / (n_crops - 1)) if n_crops > 1 else shift // 2
                crops.append(pil_img.crop((0, pos, side, pos + side)))
    
    return crops


# Backward compat alias — new code should use `embed_image` directly
embed_image_multicrop = embed_image
