"""Ghost in the GPU — Core Configuration.
Multi-domain standalone architecture.
"""
from pathlib import Path

ROOT_DIR   = Path(__file__).resolve().parent.parent
GHOST_DIR  = ROOT_DIR / "ghost"
DATA_DIR   = GHOST_DIR / "data"
MODELS_DIR = GHOST_DIR / "models"

DB_PATH        = DATA_DIR / "taste.db"
IMAGES_DIR     = DATA_DIR / "images"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

HOST = "127.0.0.1"
PORT = 8770

for d in (DATA_DIR, MODELS_DIR, IMAGES_DIR, EMBEDDINGS_DIR):
    d.mkdir(parents=True, exist_ok=True)


def get_domain_model_paths(domain_id: int):
    """Returns (brain_path, pca_path) for a given domain_id."""
    brain_p = MODELS_DIR / f"domain_{domain_id}_brain.pkl"
    pca_p   = MODELS_DIR / f"domain_{domain_id}_pca.pkl"
    return brain_p, pca_p
