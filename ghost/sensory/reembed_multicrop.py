"""Re-embed all images in a domain using multi-crop averaging."""
import sys
import shutil
from pathlib import Path
from PIL import Image
import numpy as np

from ghost.config import IMAGES_DIR, EMBEDDINGS_DIR
from ghost.sensory.backbone import get_vision_model, embed_image_multicrop
from ghost.rater import db


def reembed_domain(domain_id: int = 1, backup: bool = True, dry_run: bool = False):
    model, transform, device = get_vision_model()
    
    c = db.connect()
    rows = c.execute("SELECT id, filename FROM images WHERE domain_id=? ORDER BY id", (domain_id,)).fetchall()
    c.close()
    
    print(f"Domain {domain_id}: {len(rows)} images to re-embed")
    
    backup_dir = EMBEDDINGS_DIR / f"backup_pre_multicrop"
    if backup and not dry_run:
        backup_dir.mkdir(exist_ok=True)
        print(f"Backing up existing embeddings to {backup_dir}")
        for r in rows:
            src = EMBEDDINGS_DIR / f"{r['id']}.npy"
            dst = backup_dir / f"{r['id']}.npy"
            if src.is_file() and not dst.exists():
                shutil.copy2(src, dst)
    
    if dry_run:
        print("DRY RUN — no files will be modified")
        return
    
    total = len(rows)
    for i, r in enumerate(rows, 1):
        p = IMAGES_DIR / r["filename"]
        if not p.is_file():
            print(f"  [{i}/{total}] #{r['id']}: file missing, skip")
            continue
        try:
            with Image.open(p) as im:
                pil_img = im.convert("RGB")
            vec = embed_image_multicrop(pil_img, n_crops=6)
            np.save(EMBEDDINGS_DIR / f"{r['id']}.npy", vec)
            if i % 100 == 0 or i == total:
                print(f"  [{i}/{total}] #{r['id']}: done")
        except Exception as e:
            print(f"  [{i}/{total}] #{r['id']}: FAILED: {e}")
    
    print(f"\nDone. Re-embedded {total} images in domain {domain_id}.")
    print("Next: retrain to re-fit PCA and weights on new embeddings.")


if __name__ == "__main__":
    dom = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    dry = "--dry-run" in sys.argv
    reembed_domain(dom, backup=True, dry_run=dry)
