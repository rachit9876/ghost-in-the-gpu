"""Ghost Face Mirror — Isolated Configuration & Model Bootstrapper.
Runs 100% on PyTorch CUDA GPU (or falls back only if no NVIDIA GPU exists).
Auto-downloads and converts buffalo_l models if missing so any GitHub clone works on 1 click.
"""
import os
import urllib.request
import zipfile
from pathlib import Path

SUBPROJECT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SUBPROJECT_DIR.parent

# Shared read-only access to main corpus
MAIN_DB_PATH = ROOT_DIR / "ghost" / "data" / "taste.db"
MAIN_IMAGES_DIR = ROOT_DIR / "ghost" / "data" / "images"

# Isolated Face Mirror database and models
FACE_DB_PATH = SUBPROJECT_DIR / "face_cache.db"
MODELS_DIR = SUBPROJECT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

SCRFD_PT_PATH = MODELS_DIR / "scrfd_10g.pt"
ARCFACE_PT_PATH = MODELS_DIR / "arcface_r50.pt"

# Default thresholds calibrated to strictly eliminate occluded & side-profile faces:
# - Visibility benchmark failures: #4 (mouth occluded: mouth_grad_ratio=0.34), #6 (det_score=0.31), #102 (no face)
# - Side-angle benchmark failures: #394 (yaw=0.60, 0.98), #366 (yaw=1.89), #345 (yaw=1.23), #322 (yaw=10.53)
DEFAULT_THRESHOLDS = {
    "min_det_score": 0.55,          # Eliminates #6 (0.31) and weak ghost detections
    "max_yaw_asymmetry": 0.56,      # Eliminates #394 (0.60, 0.98), #366 (1.89), #345 (1.23), #322 (10.53)
    "max_align_error": 7.5,         # Accommodates natural human pitch/upward angles while eliminating unnatural distortions
    "min_mouth_grad_ratio": 0.45,   # Eliminates #4 (0.34) where lower face/mouth is covered
    "min_face_size": 45,            # Minimum pixel width/height of face bounding box
}

FACE_HOST = "127.0.0.1"
FACE_PORT = 8775

BUFFALO_L_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"


def ensure_gpu_models():
    """Ensures scrfd_10g.pt and arcface_r50.pt exist in face_mirror/models/.
    1. If already converted in face_mirror/models/, returns immediately (<0.01s).
    2. If ~/.insightface/models/buffalo_l exists locally, converts from there.
    3. Otherwise, downloads buffalo_l.zip from GitHub release and converts automatically.
    """
    if SCRFD_PT_PATH.is_file() and ARCFACE_PT_PATH.is_file():
        return

    import onnx
    import torch
    from onnx2torch import convert

    user_buffalo = Path.home() / ".insightface" / "models" / "buffalo_l"
    det_onnx_path = user_buffalo / "det_10g.onnx"
    rec_onnx_path = user_buffalo / "w600k_r50.onnx"

    if not (det_onnx_path.is_file() and rec_onnx_path.is_file()):
        local_onnx_dir = MODELS_DIR / "buffalo_l_onnx"
        local_onnx_dir.mkdir(parents=True, exist_ok=True)
        det_onnx_path = local_onnx_dir / "det_10g.onnx"
        rec_onnx_path = local_onnx_dir / "w600k_r50.onnx"

        if not (det_onnx_path.is_file() and rec_onnx_path.is_file()):
            zip_path = MODELS_DIR / "buffalo_l.zip"
            print(f"[*] Downloading InsightFace buffalo_l models from {BUFFALO_L_URL} ...")
            urllib.request.urlretrieve(BUFFALO_L_URL, str(zip_path))
            print("[*] Extracting ONNX weights...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(local_onnx_dir)
            zip_path.unlink(missing_ok=True)
            # Handle nested folder if present
            if not det_onnx_path.is_file() and (local_onnx_dir / "buffalo_l" / "det_10g.onnx").is_file():
                det_onnx_path = local_onnx_dir / "buffalo_l" / "det_10g.onnx"
                rec_onnx_path = local_onnx_dir / "buffalo_l" / "w600k_r50.onnx"

    if not SCRFD_PT_PATH.is_file():
        print("[*] Compiling SCRFD 10G Face Detector to native PyTorch GPU graph...")
        det_model = convert(onnx.load(str(det_onnx_path))).eval()
        torch.save(det_model, SCRFD_PT_PATH)

    if not ARCFACE_PT_PATH.is_file():
        print("[*] Compiling ArcFace R50 (w600k) to native PyTorch GPU graph...")
        rec_model = convert(onnx.load(str(rec_onnx_path))).eval()
        torch.save(rec_model, ARCFACE_PT_PATH)

    print("[+] GPU Face Models ready!")
