"""AIDEModel: the trained decision engine (report section 2.2).

Wraps the learned scoring weights, thresholds, and the two small sklearn
models (a method classifier, a bpp regressor) that "refine" the raw S(b)
formula into the four simultaneous per-block decisions: WHERE (score rank +
confidence gate), HOW MUCH (bpp), WHICH METHOD, and CONFIDENCE (skip gate).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np

from .features import FEATURE_NAMES
from .scoring import ScoringWeights, allocate_bpp, compute_scores, select_method_fallback


@dataclass(frozen=True)
class AIDEThresholds:
    tau_min: float  # confidence gate: S(b) below this -> block skipped entirely
    t_low: float    # texture(b) < t_low -> LSB
    e_high: float   # edge(b) > e_high -> DWT


@dataclass
class BlockDecision:
    block_index: int
    score: float
    keep: bool
    method: str | None
    bpp: int | None
    rank: int | None  # embedding priority among kept blocks; 0 = highest score


class AIDEModel:
    def __init__(self, weights: ScoringWeights, thresholds: AIDEThresholds,
                 method_clf=None, bpp_reg=None, meta: dict | None = None):
        self.weights = weights
        self.thresholds = thresholds
        self.method_clf = method_clf
        self.bpp_reg = bpp_reg
        self.meta = meta or {}

    def score(self, features: np.ndarray) -> np.ndarray:
        return compute_scores(features, self.weights)

    def decide(self, features: np.ndarray) -> list[BlockDecision]:
        n = features.shape[0]
        scores = self.score(features)

        if self.method_clf is not None:
            pred_labels = np.asarray(self.method_clf.predict(features), dtype=object)
        else:
            pred_labels = select_method_fallback(
                features[:, 0], features[:, 3], self.thresholds.t_low, self.thresholds.e_high
            )

        if self.bpp_reg is not None:
            pred_bpp = np.clip(np.round(self.bpp_reg.predict(features)).astype(int), 1, 4)
        else:
            pred_bpp = allocate_bpp(scores)

        clf_says_skip = pred_labels == "SKIP"
        keep_mask = (scores >= self.thresholds.tau_min) & (~clf_says_skip)

        kept_indices = np.where(keep_mask)[0]
        order = kept_indices[np.argsort(-scores[kept_indices])]
        rank_of = {int(idx): r for r, idx in enumerate(order)}

        decisions: list[BlockDecision] = []
        for i in range(n):
            if keep_mask[i]:
                decisions.append(BlockDecision(
                    block_index=i, score=float(scores[i]), keep=True,
                    method=str(pred_labels[i]), bpp=int(pred_bpp[i]), rank=rank_of[i],
                ))
            else:
                decisions.append(BlockDecision(
                    block_index=i, score=float(scores[i]), keep=False,
                    method=None, bpp=None, rank=None,
                ))
        return decisions

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump({"method_clf": self.method_clf, "bpp_reg": self.bpp_reg},
                    directory / "aide_model.joblib")
        payload = {
            "weights": self.weights.as_dict(),
            "thresholds": asdict(self.thresholds),
            "meta": self.meta,
            "feature_names": list(FEATURE_NAMES),
        }
        (directory / "weights.json").write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, directory: Path) -> "AIDEModel":
        directory = Path(directory)
        payload = json.loads((directory / "weights.json").read_text())
        blob = joblib.load(directory / "aide_model.joblib")
        return cls(
            weights=ScoringWeights(**payload["weights"]),
            thresholds=AIDEThresholds(**payload["thresholds"]),
            method_clf=blob["method_clf"],
            bpp_reg=blob["bpp_reg"],
            meta=payload.get("meta", {}),
        )

    @classmethod
    def default(cls) -> "AIDEModel":
        """Formula-only fallback (no trained classifier): fixed equal
        weights and reasonable thresholds. Used until `aide train` has been
        run; the confidence gate and rule-based method selection from the
        report still apply."""
        return cls(ScoringWeights(), AIDEThresholds(tau_min=0.15, t_low=0.35, e_high=0.55))
