import pytest
import numpy as np
import tempfile
import pickle
from pathlib import Path

from ghost.rater.scorer import GhostScorer


class DummyPCA:
    def __init__(self, n_components=16):
        self.n_components_ = n_components
        self.n_components = n_components
        self.components_ = np.random.randn(n_components, 1024).astype(np.float32)

    def transform(self, X):
        return (X @ self.components_.T).astype(np.float32)


def test_contrastive_threshold_computation_synthetic():
    """Unit test: 100-image synthetic 1024D corpus and threshold computation."""
    np.random.seed(42)
    N = 100
    dim = 1024
    embeddings = np.random.randn(N, dim).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    # Cosine similarity matrix
    S = embeddings @ embeddings.T
    iu_i, iu_j = np.triu_indices(N, k=1)
    sim_vals = S[iu_i, iu_j]

    raw_p98 = float(np.percentile(sim_vals, 98.0))
    tau = float(np.clip(raw_p98, 0.76, 0.85))

    # For random Gaussian vectors in 1024D, raw_p98 will be ~0.06 - 0.10, so it clamps to 0.76
    assert 0.76 <= tau <= 0.85
    assert tau == 0.76 or tau == raw_p98

    # Now test with highly correlated cluster
    clustered = np.repeat(embeddings[:1], N, axis=0) + np.random.randn(N, dim).astype(np.float32) * 0.001
    clustered /= np.linalg.norm(clustered, axis=1, keepdims=True)
    S_clustered = clustered @ clustered.T
    sim_vals_clustered = S_clustered[iu_i, iu_j]
    raw_p98_high = float(np.percentile(sim_vals_clustered, 98.0))
    tau_high = float(np.clip(raw_p98_high, 0.76, 0.85))

    # For almost identical vectors, raw_p98_high > 0.99, so it clamps to 0.85
    assert 0.76 <= tau_high <= 0.85
    assert tau_high == 0.85


def test_sample_contrastive_group_never_returns_compared_pair():
    """Integration test: sample_contrastive_group never returns a pair in compared_pairs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        brain_path = tmp_path / "domain_99_brain.pkl"
        pca_path = tmp_path / "domain_99_pca.pkl"

        n_c = 16
        pca = DummyPCA(n_components=n_c)
        with open(pca_path, "wb") as f:
            pickle.dump(pca, f)

        coef = np.random.randn(n_c).astype(np.float32)
        with open(brain_path, "wb") as f:
            pickle.dump({
                "domain_id": 99,
                "coef": coef.tolist(),
                "intercept": 0.0,
                "dim": n_c,
                "accuracy": 88.0,
                "trained_at": 1000.0
            }, f)

        scorer = GhostScorer(domain_id=99)
        scorer.brain_path = brain_path
        scorer.pca_path = pca_path
        scorer.reload_if_needed()
        assert scorer.loaded is True

        # Mock embeddings for 30 fake images
        fake_ids = list(range(1001, 1031))
        base_vec = np.random.randn(1024).astype(np.float32)
        base_vec /= np.linalg.norm(base_vec)
        fake_embs = {}
        for iid in fake_ids:
            # High cosine similarity lookalikes
            v = base_vec + np.random.randn(1024).astype(np.float32) * 0.1
            v /= np.linalg.norm(v)
            fake_embs[iid] = v

        # Monkey-patch _embedding_path and np.load
        original_embedding_path = scorer._embedding_path
        scorer._embedding_path = lambda iid: tmp_path / f"{iid}.npy"
        for iid, v in fake_embs.items():
            np.save(tmp_path / f"{iid}.npy", v)

        compared = set()

        # Iteratively sample 20 groups and ensure none returned pair was in compared
        for _ in range(20):
            group = scorer.sample_contrastive_group(fake_ids, sz=3, recent_shown=[], compared_pairs=compared)
            if group is None:
                break
            # Check all internal pairs
            for idx1 in range(len(group)):
                for idx2 in range(idx1 + 1, len(group)):
                    a, b = group[idx1], group[idx2]
                    pk = (a, b) if a < b else (b, a)
                    assert pk not in compared, f"Pair {pk} was returned despite being in compared_pairs!"
                    compared.add(pk)
