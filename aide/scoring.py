"""The explainable AIDE scoring formula (report section 2.2).

S(b) = alpha*Texture + beta*Entropy + gamma*Variance + delta*EdgeStrength

Weights are learned during training (see ``aide.train``) as the trained
decision tree's normalised feature importances, so the formula stays
auditable: you can always point at *why* a block scored the way it did.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ScoringWeights:
    alpha: float = 0.25  # texture
    beta: float = 0.25   # entropy
    gamma: float = 0.25  # variance
    delta: float = 0.25  # edge strength

    def as_array(self) -> np.ndarray:
        return np.array([self.alpha, self.beta, self.gamma, self.delta], dtype=np.float64)

    def as_dict(self) -> dict:
        return asdict(self)


def compute_scores(features: np.ndarray, weights: ScoringWeights) -> np.ndarray:
    """S(b) for a batch of blocks. `features` is (N, 4) = [texture, entropy,
    variance, edge_strength], each already normalised to [0, 1].

    Deliberately avoids `@`/np.dot: on macOS's Accelerate BLAS backend,
    matmul on tiny arrays like this can raise spurious divide-by-zero /
    invalid-value FloatingPointWarnings that don't affect the (correct)
    result, but do trip up strict numpy error settings. A plain weighted
    sum is just as fast at this size and sidesteps BLAS entirely."""
    w = weights.as_array()
    return (features[..., 0] * w[0] + features[..., 1] * w[1]
            + features[..., 2] * w[2] + features[..., 3] * w[3])


def allocate_bpp(scores: np.ndarray, min_bpp: int = 1, max_bpp: int = 4,
                  s_max: float = 1.0) -> np.ndarray:
    """bpp(b) = floor(S(b) / S_max * 4), clamped to [1, 4] (report 2.2)."""
    bpp = np.floor(scores / s_max * max_bpp).astype(int)
    return np.clip(bpp, min_bpp, max_bpp)


def select_method_fallback(texture: np.ndarray, edge: np.ndarray,
                            t_low: float, e_high: float) -> np.ndarray:
    """if texture(b) < T_low -> LSB; elif edge(b) > E_high -> DWT; else DCT
    (report 2.2). Used when no trained classifier is available."""
    return np.where(texture < t_low, "LSB", np.where(edge > e_high, "DWT", "DCT"))
