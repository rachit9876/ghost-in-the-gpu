"""Ghost Scorer — Per-domain instant scoring via Ghost Brain & PCA projection."""
import logging
import pickle
import threading

import numpy as np

from ghost.config import get_domain_model_paths, EMBEDDINGS_DIR

logger = logging.getLogger("ghost")

# Above this many images, the contrastive N×N cosine similarity matrix
# becomes too large to materialize in memory. We deterministically sample
# this many candidates for contrastive mode.
_CONTRASTIVE_FULL_MATRIX_MAX = 5000


class GhostScorer:
    def __init__(self, domain_id: int = 1):
        self.domain_id = domain_id
        self.brain_path, self.pca_path = get_domain_model_paths(domain_id)
        self._cache = {}
        self._max_cache = 8192
        self._mtime = 0
        self._contrastive_cache = None
        self._reload_lock = threading.Lock()
        self.loaded = False
        self.w = None
        self.b = 0.0
        self.pca = None
        self.meta = {}
        self.reload_if_needed()

    def reload_if_needed(self):
        if not self.brain_path.exists() or not self.pca_path.exists():
            self.loaded = False
            return

        try:
            mtime = max(
                self.brain_path.stat().st_mtime,
                self.pca_path.stat().st_mtime,
            )
        except OSError:
            self.loaded = False
            return

        if self.loaded and mtime == self._mtime:
            return

        with self._reload_lock:
            # Double-checked under lock so concurrent reloads serialize.
            if self.loaded and mtime == self._mtime:
                return
            try:
                with open(self.brain_path, "rb") as f:
                    d = pickle.load(f)
                w = np.asarray(d["coef"], dtype=np.float32)
                b = float(d["intercept"])

                with open(self.pca_path, "rb") as f:
                    pca = pickle.load(f)

                n_pca = (getattr(pca, "n_components_", None)
                         or getattr(pca, "n_components", None))
                if w.shape[0] != n_pca:
                    logger.warning(
                        "scorer[%s]: brain coef dim=%s != pca n_components=%s; "
                        "keeping previous model (mtimes frozen until next change)",
                        self.domain_id, w.shape[0], n_pca,
                    )
                    self._mtime = mtime  # don't retry the same broken files
                    return

                # Commit under the reload lock. score() reads w/b/pca without
                # the lock; the lock only serializes concurrent reloads.
                self.w = w
                self.b = b
                self.pca = pca
                self.meta = d
                self._cache.clear()
                self._contrastive_cache = None
                self._mtime = mtime
                self.loaded = True
            except Exception as e:
                logger.warning(
                    "scorer[%s]: reload failed (%s); keeping previous model",
                    self.domain_id, e,
                )
                self._mtime = mtime  # don't retry the same broken files

    def invalidate_contrastive_cache(self):
        self._contrastive_cache = None

    def _embedding_path(self, image_id: int):
        p = EMBEDDINGS_DIR / f"{image_id}.npy"
        return p if p.is_file() else None

    def score(self, image_id: int):
        self.reload_if_needed()
        if not self.loaded:
            return None
        if image_id in self._cache:
            return self._cache[image_id]

        p = self._embedding_path(image_id)
        if p is None:
            return None

        try:
            emb = np.load(p).reshape(1, -1)
            v = self.pca.transform(emb)[0]
            s = float(np.dot(self.w, v) + self.b)
        except Exception:
            s = None

        if len(self._cache) >= self._max_cache:
            self._cache.clear()
        self._cache[image_id] = s
        return s

    def _get_contrastive_index(self, all_ids: list):
        """Builds or returns cached 1024D cosine similarity clash index for the domain."""
        self.reload_if_needed()
        if not self.loaded or len(all_ids) < 4:
            return None

        try:
            brain_mtime = self.brain_path.stat().st_mtime if self.brain_path.exists() else 0
            pca_mtime = self.pca_path.stat().st_mtime if self.pca_path.exists() else 0
        except OSError:
            return None

        # B8: stat ALL embedding mtimes, not just the first 64.
        # Edits to any image (especially beyond position 64 in the
        # sorted id list) must invalidate this cache so stale
        # contrastive pairs are not served. Cost: O(N) stat calls
        # per contrastive rebuild-check, ~30-80ms warm on SSD.
        # Contrastive is user-driven, not in a hot loop.
        latest_emb_mtime = 0.0
        for iid in all_ids:
            try:
                m = (EMBEDDINGS_DIR / f"{iid}.npy").stat().st_mtime
                if m > latest_emb_mtime:
                    latest_emb_mtime = m
            except OSError:
                pass

        # Also fold max(all_ids) into the key so a delete-highest +
        # add-new pair (same len, different content) still invalidates.
        max_id = max(all_ids) if all_ids else 0
        key = (brain_mtime, pca_mtime, len(all_ids), max_id, latest_emb_mtime)
        cached = self._contrastive_cache
        if cached is not None and cached.get("key") == key:
            return cached

        valid_ids = []
        raw_list = []
        for iid in all_ids:
            p = self._embedding_path(iid)
            if p is not None:
                try:
                    vec = np.load(p).astype(np.float32).flatten()
                    raw_list.append(vec)
                    valid_ids.append(iid)
                except Exception:
                    pass

        if len(valid_ids) < 4:
            return None

        X = np.stack(raw_list)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X_norm = X / norms

        # Batch-score all valid images in one vectorized PCA + dot call
        V = self.pca.transform(X)
        scores = (V @ self.w + self.b).astype(np.float32)
        for iid, sc in zip(valid_ids, scores):
            self._cache[iid] = float(sc)

        n = len(valid_ids)

        # Cap N to avoid OOM on huge domains; deterministic sampling.
        if n > _CONTRASTIVE_FULL_MATRIX_MAX:
            logger.warning(
                "scorer[%s]: N=%d exceeds contrastive matrix cap (%d); "
                "sampling %d candidates (deterministic)",
                self.domain_id, n, _CONTRASTIVE_FULL_MATRIX_MAX,
                _CONTRASTIVE_FULL_MATRIX_MAX,
            )
            rng = np.random.default_rng(seed=42)
            sample_idx = np.sort(
                rng.choice(n, size=_CONTRASTIVE_FULL_MATRIX_MAX, replace=False)
            )
            valid_ids = [valid_ids[i] for i in sample_idx]
            X_norm = X_norm[sample_idx]
            scores = scores[sample_idx]
            n = len(valid_ids)

        S = X_norm @ X_norm.T
        iu_i, iu_j = np.triu_indices(n, k=1)
        sim_vals = S[iu_i, iu_j]

        raw_p98 = float(np.percentile(sim_vals, 98.0))
        tau = float(np.clip(raw_p98, 0.76, 0.85))
        if raw_p98 < 0.76 or raw_p98 > 0.85:
            logger.warning(
                "[Contrastive] raw_p98 (%.4f) clamped to [0.76, 0.85] -> tau=%.4f",
                raw_p98, tau,
            )
        mask = sim_vals >= tau
        if mask.sum() < 20:
            tau = float(np.percentile(sim_vals, 90.0))
            mask = sim_vals >= tau

        logger.info(
            "[Contrastive] Cache refresh: tau=%.4f (raw_p98=%.4f), pairs_above=%d",
            tau, raw_p98, int(mask.sum()),
        )

        cand_i = iu_i[mask]
        cand_j = iu_j[mask]
        cand_ds = np.abs(scores[cand_i] - scores[cand_j])

        order = np.argsort(cand_ds)[::-1][:10000]
        top_pairs = [
            (valid_ids[int(cand_i[idx])], valid_ids[int(cand_j[idx])],
             float(cand_ds[idx]))
            for idx in order
            if cand_ds[idx] > 0.05
        ]

        cached = {
            "key": key,
            "tau": tau,
            "valid_ids": valid_ids,
            "id_to_idx": {iid: idx for idx, iid in enumerate(valid_ids)},
            "S": S,
            "scores": scores,
            "top_pairs": top_pairs,
        }
        self._contrastive_cache = cached
        return cached

    def sample_contrastive_group(self, all_ids: list, sz: int, recent_shown, compared_pairs: set):
        """Stochastically samples a lookalike group with high score divergence (|Δs|),
        excluding pairs already compared in `compared_pairs`.
        """
        idx_data = self._get_contrastive_index(all_ids)
        if not idx_data or not idx_data["top_pairs"]:
            return None

        recent_set = set(recent_shown)
        eligible = []
        weights = []

        # 1st pass: exclude both compared pairs AND recently shown images
        for a, b, ds in idx_data["top_pairs"]:
            pair_key = (a, b) if a < b else (b, a)
            if pair_key in compared_pairs:
                continue
            if a in recent_set or b in recent_set:
                continue
            eligible.append((a, b, ds))
            weights.append(ds * ds)
            if len(eligible) >= 64:
                break

        # 2nd pass fallback: relax recent_set, still exclude compared_pairs
        if not eligible:
            for a, b, ds in idx_data["top_pairs"]:
                pair_key = (a, b) if a < b else (b, a)
                if pair_key in compared_pairs:
                    continue
                eligible.append((a, b, ds))
                weights.append(ds * ds)
                if len(eligible) >= 64:
                    break

        if not eligible:
            return None

        probs = np.array(weights, dtype=np.float64)
        probs /= probs.sum()
        pick_idx = int(np.random.choice(len(eligible), p=probs))
        anchor_a, anchor_b, _ = eligible[pick_idx]
        chosen = [anchor_a, anchor_b]

        # If group size > 2, fill remaining slots with visually similar
        # neighbors of anchor_a that maximize score diversity.
        if sz > 2:
            id_to_idx = idx_data["id_to_idx"]
            S = idx_data["S"]
            scores = idx_data["scores"]
            valid_ids = idx_data["valid_ids"]
            a_idx = id_to_idx[anchor_a]
            sims = S[a_idx]

            tau = idx_data["tau"]
            neighbor_order = np.argsort(sims)[::-1]
            extra_pool = []
            for n_idx in neighbor_order[1:80]:
                nid = valid_ids[int(n_idx)]
                if nid in chosen or nid in recent_set:
                    continue
                if float(sims[n_idx]) < tau:
                    break
                already_compared = False
                for ch_id in chosen:
                    pk = (ch_id, nid) if ch_id < nid else (nid, ch_id)
                    if pk in compared_pairs:
                        already_compared = True
                        break
                if already_compared:
                    continue
                extra_pool.append(
                    (abs(float(scores[a_idx] - scores[int(n_idx)])), nid)
                )

            extra_pool.sort(key=lambda x: x[0], reverse=True)
            top_extra = [x[1] for x in extra_pool[: max((sz - 2) * 3, 6)]]
            if len(top_extra) >= sz - 2:
                chosen.extend(
                    np.random.choice(top_extra, size=sz - 2, replace=False).tolist()
                )
            else:
                logger.warning(
                    "[Contrastive] Insufficient lookalike neighbors "
                    "(sim >= %.3f) for anchor %s; returning pair or smaller group",
                    tau, anchor_a,
                )
                if len(chosen) < 2:
                    return None

        np.random.shuffle(chosen)
        return [int(x) for x in chosen]