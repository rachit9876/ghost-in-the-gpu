"""Ghost in the GPU — Domain-Aware Neural Tournament Trainer.
Trains a specific domain's visual twin using Zero-Leakage 5-Fold Cross-Validation.
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import pickle
import sys
import time
import warnings

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ConvergenceWarning

from ghost.config import DB_PATH, EMBEDDINGS_DIR, get_domain_model_paths
from ghost.rater import db

# Silence sklearn convergence warnings only — keep everything else visible.
warnings.filterwarnings("ignore", category=ConvergenceWarning)
# Silence numpy "future" warnings and sklearn parameter deprecation warnings.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

SEED = 42
N_FOLDS = 5

CANDIDATE_DIMS = [32, 48, 64, 96, 128, 160, 192, 256, 384, 512, 768, 896, 1024]

ALGO_GRID = [
    # Standard Logistic Regression with varying regularizations (Smooth Bradley-Terry)
    {"name": "Logistic (C=0.1)",  "make": lambda: LogisticRegression(C=0.1, max_iter=1000, solver="lbfgs")},
    {"name": "Logistic (C=0.5)",  "make": lambda: LogisticRegression(C=0.5, max_iter=1000, solver="lbfgs")},
    {"name": "Logistic (C=1.0)",  "make": lambda: LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")},
    {"name": "Logistic (C=2.0)",  "make": lambda: LogisticRegression(C=2.0, max_iter=1000, solver="lbfgs")},
    {"name": "Logistic (C=4.0)",  "make": lambda: LogisticRegression(C=4.0, max_iter=1000, solver="lbfgs")},

    # Linear Support Vector Machines (Squared Hinge Max-Margin)
    {"name": "Linear SVM (C=0.1)", "make": lambda: LinearSVC(C=0.1, max_iter=2500, random_state=SEED)},
    {"name": "Linear SVM (C=0.5)", "make": lambda: LinearSVC(C=0.5, max_iter=2500, random_state=SEED)},
    {"name": "Linear SVM (C=1.0)", "make": lambda: LinearSVC(C=1.0, max_iter=2500, random_state=SEED)},
    {"name": "Linear SVM (C=2.0)", "make": lambda: LinearSVC(C=2.0, max_iter=2500, random_state=SEED)},
    {"name": "Linear SVM (C=3.0)", "make": lambda: LinearSVC(C=3.0, max_iter=2500, random_state=SEED)},

    # L1-Hinge Support Vector Machines (Robust to non-transitive loops & outliers)
    {"name": "Hinge SVM (C=0.5)",  "make": lambda: LinearSVC(C=0.5, loss="hinge", max_iter=3000, random_state=SEED)},
    {"name": "Hinge SVM (C=1.0)",  "make": lambda: LinearSVC(C=1.0, loss="hinge", max_iter=3000, random_state=SEED)},
    {"name": "Hinge SVM (C=2.0)",  "make": lambda: LinearSVC(C=2.0, loss="hinge", max_iter=3000, random_state=SEED)},

    # Closed-Form L2 Ridge Rankers (Fast, analytical global optimum)
    {"name": "Ridge Ranker (a=0.5)", "make": lambda: RidgeClassifier(alpha=0.5, random_state=SEED)},
    {"name": "Ridge Ranker (a=1.0)", "make": lambda: RidgeClassifier(alpha=1.0, random_state=SEED)},
    {"name": "Ridge Ranker (a=2.0)", "make": lambda: RidgeClassifier(alpha=2.0, random_state=SEED)},

    # ElasticNet / Modified Huber (Outlier-tolerant + L1/L2 hybrid with sample_weight support)
    {"name": "Huber Elastic (a=1e-3)", "make": lambda: SGDClassifier(loss="modified_huber", penalty="elasticnet", l1_ratio=0.3, alpha=1e-3, max_iter=1500, random_state=SEED)},

    # Sparse L1 regularization
    {"name": "Sparse L1 (C=0.5)",  "make": lambda: LogisticRegression(l1_ratio=1.0, C=0.5, solver="liblinear", random_state=SEED, max_iter=1000)},
    {"name": "Sparse L1 (C=1.0)",  "make": lambda: LogisticRegression(l1_ratio=1.0, C=1.0, solver="liblinear", random_state=SEED, max_iter=1000)},
    {"name": "Sparse L1 (C=2.0)",  "make": lambda: LogisticRegression(l1_ratio=1.0, C=2.0, solver="liblinear", random_state=SEED, max_iter=1000)},
    {"name": "Sparse L1 (C=3.0)",  "make": lambda: LogisticRegression(l1_ratio=1.0, C=3.0, solver="liblinear", random_state=SEED, max_iter=1000)},
    {"name": "Sparse L1 (C=4.0)",  "make": lambda: LogisticRegression(l1_ratio=1.0, C=4.0, solver="liblinear", random_state=SEED, max_iter=1000)},
]


def get_pair_weights(c, domain_id: int) -> dict:
    """Calculates pair weights based on test-retest consistency.
    weight = max(0.20, 1.0 - 0.8 * flip_rate(pair))
    """
    retests = c.execute("""
        SELECT r.winner_id AS r_win, r.losers_json AS r_losers,
               o.winner_id AS o_win, o.losers_json AS o_losers
        FROM comparisons r
        JOIN comparisons o ON r.retest_of = o.id
        WHERE r.domain_id = ? AND r.is_retest = 1
    """, (domain_id,)).fetchall()

    pair_counts = {}
    for row in retests:
        r_w, o_w = row["r_win"], row["o_win"]
        try:
            o_losers = json.loads(row["o_losers"])
        except Exception:
            continue

        for l in o_losers:
            pkey = (min(o_w, l), max(o_w, l))
            if pkey not in pair_counts:
                pair_counts[pkey] = {"agreements": 0, "flips": 0}
            if r_w == o_w:
                pair_counts[pkey]["agreements"] += 1
            elif l == r_w:
                pair_counts[pkey]["flips"] += 1

    weights = {}
    for pkey, stats in pair_counts.items():
        total = stats["agreements"] + stats["flips"]
        if total > 0:
            flip_rate = stats["flips"] / total
            weights[pkey] = max(0.20, 1.0 - 0.8 * flip_rate)
    return weights


def load_domain_comparisons(domain_id: int):
    if not DB_PATH.exists():
        return [], set(), {}
    with db.connection() as c:
        rows = c.execute(
            "SELECT id, winner_id, losers_json FROM comparisons "
            "WHERE domain_id=? AND is_retest=0 ORDER BY id",
            (domain_id,)
        ).fetchall()
        dom_imgs = {
            r[0] for r in c.execute(
                "SELECT id FROM images WHERE domain_id=?", (domain_id,)
            ).fetchall()
        }
        pair_weights = get_pair_weights(c, domain_id)
    return rows, dom_imgs, pair_weights


def train_ghost(domain_id: int = 1) -> dict:
    print("=" * 60)
    print(f"  GHOST IN THE GPU — TRAINING DOMAIN #{domain_id}")
    print("=" * 60)

    comps, dom_img_ids, pair_weights = load_domain_comparisons(domain_id)
    if len(comps) < 6:
        return {
            "ok": False,
            "error": f"Need at least 6 comparisons in this domain to train "
                     f"(currently have {len(comps)})."
        }

    all_img_ids = set()
    for c in comps:
        if not dom_img_ids or c["winner_id"] in dom_img_ids:
            all_img_ids.add(c["winner_id"])
        for l in json.loads(c["losers_json"]):
            if not dom_img_ids or l in dom_img_ids:
                all_img_ids.add(l)

    X_raw = {}
    for iid in sorted(all_img_ids):
        p = EMBEDDINGS_DIR / f"{iid}.npy"
        if p.is_file():
            X_raw[iid] = np.load(p).astype(np.float32).flatten()

    valid_images = sorted(X_raw.keys())
    if len(valid_images) < 6:
        return {"ok": False,
                "error": f"Need at least 6 embedded images "
                         f"(found {len(valid_images)})."}

    # Zero-leakage CV
    n_folds = min(N_FOLDS, max(2, len(valid_images) // 3))
    rng = np.random.default_rng(SEED)
    shuffled = list(valid_images)
    rng.shuffle(shuffled)
    folds = [[] for _ in range(n_folds)]
    for i, img in enumerate(shuffled):
        folds[i % n_folds].append(img)

    img_list = sorted(valid_images)
    img_to_idx = {img: idx for idx, img in enumerate(img_list)}
    all_mat = np.stack([X_raw[img] for img in img_list])

    parsed_comps = []
    for c in comps:
        w = c["winner_id"]
        if w not in img_to_idx:
            continue
        losers = json.loads(c["losers_json"])
        valid_l = [l for l in losers if l in img_to_idx]
        if valid_l:
            parsed_comps.append((w, valid_l))

    def precompute_pair_indices(allowed_set):
        w_idxs = []
        l_idxs = []
        pws = []
        for w, losers in parsed_comps:
            if w not in allowed_set:
                continue
            w_idx = img_to_idx[w]
            for l in losers:
                if l in allowed_set:
                    l_idx = img_to_idx[l]
                    pkey = (min(w, l), max(w, l))
                    pw = pair_weights.get(pkey, 1.0)
                    w_idxs.append(w_idx)
                    l_idxs.append(l_idx)
                    pws.append(pw)
        if not w_idxs:
            return (np.empty(0, dtype=np.int32),
                    np.empty(0, dtype=np.int32),
                    np.empty(0, dtype=np.int8),
                    np.empty(0, dtype=np.float32))
        n = len(w_idxs)
        w_arr = np.array(w_idxs, dtype=np.int32)
        l_arr = np.array(l_idxs, dtype=np.int32)
        pw_arr = np.array(pws, dtype=np.float32)
        labels = np.concatenate([np.ones(n, dtype=np.int8), np.zeros(n, dtype=np.int8)])
        weights = np.concatenate([pw_arr, pw_arr])
        return w_arr, l_arr, labels, weights

    def build_pair_diffs_from_indices(X_reduced_mat, w_arr, l_arr):
        if len(w_arr) == 0:
            return np.empty((0, 0), dtype=np.float32)
        d = X_reduced_mat[w_arr] - X_reduced_mat[l_arr]
        return np.vstack([d, -d]).astype(np.float32, copy=False)

    def fast_pair_diffs(X_reduced_mat, allowed_set):
        w_arr, l_arr, labels, weights = precompute_pair_indices(allowed_set)
        diffs = build_pair_diffs_from_indices(X_reduced_mat, w_arr, l_arr)
        return diffs, labels, weights

    best_combo = None
    best_score = -1.0
    leaderboard = []

    min_train_fold_size = min(len(valid_images) - len(f) for f in folds)
    max_cv_dim = min(min_train_fold_size - 1, all_mat.shape[1])
    candidate_dims = [d for d in CANDIDATE_DIMS if d <= max_cv_dim]
    if not candidate_dims:
        fallback_d = max(2, max_cv_dim)
        candidate_dims = [fallback_d]

    total_contenders = len(candidate_dims) * len(ALGO_GRID)
    print(f"[*] Running {n_folds}-Fold Neural Tournament across "
          f"{len(candidate_dims)} dimension(s) x {len(ALGO_GRID)} algorithms "
          f"= {total_contenders} contenders...")

    max_needed_dim = max(candidate_dims)
    fold_cache = []
    for k in range(n_folds):
        test_imgs = set(folds[k])
        train_imgs = set(valid_images) - test_imgs
        if len(train_imgs) < 3:
            continue

        train_indices = [img_to_idx[img] for img in sorted(train_imgs)]
        train_mat = all_mat[train_indices]

        n_c_max = min(max_needed_dim, train_mat.shape[0] - 1, train_mat.shape[1])
        if n_c_max < 2:
            continue

        # Fit full SVD once per fold and slice down for all smaller dimensions
        pca = PCA(n_components=n_c_max, svd_solver="full", random_state=SEED)
        pca.fit(train_mat)
        X_reduced_max = pca.transform(all_mat)

        tr_w, tr_l, y_tr, w_tr = precompute_pair_indices(train_imgs)
        te_w, te_l, y_te, _ = precompute_pair_indices(test_imgs)

        if len(y_tr) < 4 or len(y_te) == 0:
            continue

        X_tr_max = build_pair_diffs_from_indices(X_reduced_max, tr_w, tr_l)
        X_te_max = build_pair_diffs_from_indices(X_reduced_max, te_w, te_l)

        fold_cache.append({
            "X_tr_max": X_tr_max,
            "X_te_max": X_te_max,
            "n_c_max": n_c_max,
            "y_tr": y_tr, "w_tr": w_tr, "y_te": y_te,
        })

    def _eval_dim_algo(dim, a_idx):
        algo = ALGO_GRID[a_idx]
        scores = []
        for fc in fold_cache:
            n_c = min(dim, fc["n_c_max"])
            if n_c < 2:
                continue
            X_tr = fc["X_tr_max"][:, :n_c]
            X_te = fc["X_te_max"][:, :n_c]
            y_tr, w_tr, y_te = fc["y_tr"], fc["w_tr"], fc["y_te"]

            clf = algo["make"]()
            try:
                clf.fit(X_tr, y_tr, sample_weight=w_tr)
            except Exception:
                clf.fit(X_tr, y_tr)
            preds = clf.predict(X_te)
            scores.append(accuracy_score(y_te, preds))
        return dim, a_idx, scores

    tasks = [(dim, a_idx) for dim in candidate_dims for a_idx in range(len(ALGO_GRID))]
    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_eval_dim_algo)(dim, a_idx) for dim, a_idx in tasks
    )

    for dim, a_idx, scores in results:
        if scores:
            algo = ALGO_GRID[a_idx]
            mean_acc = float(np.mean(scores))
            leaderboard.append({
                "dim": dim,
                "algo": algo["name"],
                "accuracy": mean_acc,
                "scores": scores,
            })
            if mean_acc > best_score:
                best_score = mean_acc
                best_combo = (dim, algo, mean_acc)

    leaderboard.sort(key=lambda x: x["accuracy"], reverse=True)

    # Fallback path: CV never produced a valid result (e.g. very small data).
    # Train a conservative default model, but DO NOT fabricate an accuracy.
    fallback_used = best_combo is None
    if fallback_used:
        champ_dim = max(2, min(32, len(valid_images) - 1))
        champ_algo = ALGO_GRID[1]
        champ_acc = 0.0  # honest: no CV was measured
    else:
        champ_dim, champ_algo, champ_acc = best_combo

    final_n_c = max(2, min(champ_dim, all_mat.shape[0] - 1, all_mat.shape[1]))
    final_pca = PCA(n_components=final_n_c, random_state=SEED)
    final_reduced = final_pca.fit_transform(all_mat)

    X_full, y_full, w_full = fast_pair_diffs(final_reduced, set(valid_images))
    if len(X_full) < 2:
        return {"ok": False, "error": "Not enough valid pairs to fit final classifier."}

    final_clf = champ_algo["make"]()
    try:
        final_clf.fit(X_full, y_full, sample_weight=w_full)
    except Exception:
        final_clf.fit(X_full, y_full)

    coef = np.atleast_2d(final_clf.coef_)[0].tolist() if hasattr(final_clf, "coef_") else []
    intercept = (float(np.atleast_1d(final_clf.intercept_)[0])
                 if hasattr(final_clf, "intercept_") else 0.0)
    acc_pct = round(champ_acc * 100, 2)

    brain_path, pca_path = get_domain_model_paths(domain_id)
    tmp_brain_path = brain_path.with_suffix(".tmp")
    tmp_pca_path = pca_path.with_suffix(".tmp")

    with open(tmp_pca_path, "wb") as f:
        pickle.dump(final_pca, f)
    with open(tmp_brain_path, "wb") as f:
        pickle.dump({
            "domain_id": domain_id,
            "coef": coef,
            "intercept": intercept,
            "dim": final_n_c,
            "algo": champ_algo["name"],
            "accuracy": acc_pct,
            "fallback": fallback_used,
            "trained_at": time.time(),
        }, f)

    # Order matters: replace PCA first, then brain. If the process dies
    # between the two, scorer.py's dimension-mismatch check keeps the
    # previous consistent model until the next successful train.
    os.replace(tmp_pca_path, pca_path)
    os.replace(tmp_brain_path, brain_path)

    with db.connection() as c:
        c.execute(
            "UPDATE domains SET last_trained_at=?, accuracy=? WHERE id=?",
            (time.time(), acc_pct, domain_id),
        )
        c.commit()

    _render_tournament_summary(
        domain_id=domain_id,
        leaderboard=leaderboard,
        champ_algo=champ_algo["name"],
        champ_dim=final_n_c,
        champ_acc=acc_pct,
        total_comps=len(comps),
        total_imgs=len(valid_images),
        candidate_dims=candidate_dims,
        fallback=fallback_used,
    )

    return {
        "ok": True,
        "domain_id": domain_id,
        "algo": champ_algo["name"],
        "dim": final_n_c,
        "accuracy": acc_pct,
        "fallback": fallback_used,
        "comparisons": len(comps),
        "images": len(valid_images),
    }


def _render_tournament_summary(
    domain_id: int,
    leaderboard: list,
    champ_algo: str,
    champ_dim: int,
    champ_acc: float,
    total_comps: int,
    total_imgs: int,
    candidate_dims: list,
    fallback: bool = False,
):
    """Renders a terminal dashboard and leaderboard using Rich if available."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich.text import Text
        from rich import box
    except ImportError:
        print("\n" + "=" * 60)
        if fallback:
            print(f"[*] FALLBACK MODEL: {champ_algo} @ {champ_dim}D "
                  f"(CV produced no winner; training corpus too small)")
        else:
            print(f"[*] CHAMPION: {champ_algo} @ {champ_dim}D ({champ_acc:.2f}%)")
        print("=" * 60)
        return

    console = Console()

    champ_text = Text()
    if fallback:
        champ_text.append("⚠  FALLBACK MODEL WRITTEN\n", style="bold yellow")
        champ_text.append(
            "Not enough comparisons were usable for cross-validation.\n"
            "The default algorithm was fitted on the full dataset.\n"
            "Accuracy is reported as 0% because it was not measured.\n"
            "Rate more comparisons and retrain for a real score.\n\n",
            style="white",
        )
    else:
        champ_text.append("🏆 CHAMPION ARCHITECTURE FOUND!\n", style="bold yellow")
        champ_text.append(
            f"Domain #{domain_id} Visual Twin is live & synced.\n\n",
            style="white",
        )

    champ_text.append("  • Winning Model:     ", style="dim")
    champ_text.append(f"{champ_algo}\n", style="bold cyan")
    champ_text.append("  • Optimal Dimension: ", style="dim")
    champ_text.append(f"{champ_dim}D PCA\n", style="bold magenta")
    champ_text.append("  • 5-Fold Validation: ", style="dim")
    if fallback:
        champ_text.append("not measured (fallback)\n", style="bold yellow")
    else:
        champ_text.append(f"{champ_acc:.2f}%\n", style="bold green")
    champ_text.append("  • Training Corpus:   ", style="dim")
    champ_text.append(
        f"{total_comps:,} pairwise comparisons · "
        f"{total_imgs:,} active images\n",
        style="dim white",
    )

    console.print()
    border = "yellow" if fallback else "green"
    console.print(Panel(
        champ_text,
        title=f"[bold {border}] Ghost Neural Tournament — Domain #{domain_id} [/]",
        border_style=border,
        box=box.ROUNDED,
    ))

    if leaderboard:
        table = Table(
            title=f"👑 TOURNAMENT LEADERBOARD "
                  f"(Top {min(15, len(leaderboard))} of {len(leaderboard)} Contenders)",
            box=box.HEAVY_EDGE,
            header_style="bold bright_white on dark_blue",
            border_style="cyan",
        )
        table.add_column("Rank", justify="center", style="bold yellow", width=6)
        table.add_column("Algorithm", style="bold white", width=22)
        table.add_column("Dimension", justify="center", style="magenta", width=12)
        table.add_column("5-Fold CV Acc", justify="right", width=16)
        table.add_column("Fold Breakdown (Min / Max)", justify="center",
                         style="dim", width=26)
        table.add_column("Status", justify="center", width=12)

        for rank, entry in enumerate(leaderboard[:15], 1):
            acc = entry["accuracy"] * 100
            scores = [s * 100 for s in entry.get("scores", [])]
            min_s = min(scores) if scores else acc
            max_s = max(scores) if scores else acc
            is_champ = (rank == 1) and not fallback

            acc_str = (f"[bold green]{acc:.2f}%[/]"
                       if acc >= 75.0 else f"[yellow]{acc:.2f}%[/]")
            breakdown_str = f"{min_s:.1f}% - {max_s:.1f}% (±{(max_s - min_s)/2:.1f}%)"
            status_str = ("[bold bright_green]👑 CHAMPION[/]"
                          if is_champ else f"#{rank}")
            rank_badge = ("🥇" if rank == 1 else
                          "🥈" if rank == 2 else
                          "🥉" if rank == 3 else f" {rank} ")

            table.add_row(rank_badge, entry["algo"], f"{entry['dim']}D",
                          acc_str, breakdown_str, status_str)
        console.print(table)

        dim_matrix = Table(title="📊 ACCURACY HEATMAP BY DIMENSION",
                           box=box.SIMPLE_HEAVY, header_style="bold cyan")
        dim_matrix.add_column("Dimension", justify="center",
                              style="bold magenta")
        dim_matrix.add_column("Top Algorithm", style="white")
        dim_matrix.add_column("Best Accuracy", justify="right",
                              style="bold green")
        dim_matrix.add_column("Contenders Tested", justify="center", style="dim")

        by_dim = {}
        for entry in leaderboard:
            d = entry["dim"]
            if d not in by_dim or entry["accuracy"] > by_dim[d]["accuracy"]:
                by_dim[d] = entry

        for d in candidate_dims:
            if d in by_dim:
                best_in_d = by_dim[d]
                mark = (" 🏆"
                        if not fallback
                        and best_in_d["dim"] == champ_dim
                        and best_in_d["algo"] == champ_algo
                        else "")
                dim_matrix.add_row(
                    f"{d}D",
                    f"{best_in_d['algo']}{mark}",
                    f"{best_in_d['accuracy']*100:.2f}%",
                    f"{len(ALGO_GRID)}",
                )
        console.print(dim_matrix)
        console.print()


if __name__ == "__main__":
    dom_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    res = train_ghost(dom_id)

    if not res.get("ok"):
        print("\n" + "=" * 65, file=sys.stderr)
        print(f"  ✗ Training FAILED: {res.get('error', 'unknown error')}",
              file=sys.stderr)
        print("=" * 65, file=sys.stderr)
        sys.exit(1)

    print("\n" + "=" * 65)
    print("  ✓ Training complete! Weights updated in database and models/.")
    print("=" * 65)

    # Keep terminal window open so the user can admire the leaderboard.
    # Safe under EOF / non-interactive spawns.
    try:
        input("\n[Press Enter to close this tournament window...]")
    except (EOFError, KeyboardInterrupt):
        pass