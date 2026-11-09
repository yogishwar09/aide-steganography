"""Robustness profiles: named, measured operating points on the
quality <-> robustness <-> capacity trade-off.

Every knob that moves that trade-off lives here, in one place, instead of
as scattered module constants:

  dct_delta / dwt_delta  coefficient-pair separation (aide/embedding/_common.py).
                         Larger = survives more noise, costs image quality.
  nsym                   Reed-Solomon parity bytes per full-length codeword
                         (aide/ecc.py). Larger = more errors corrected,
                         costs capacity.
  fragile_budgets        cap on the share of the payload carried by the
                         methods measured fragile under attack (LSB, DWT).
  min_codewords          short messages are split into at least this many
                         shortened RS codewords so interleaving can spread
                         localized damage (aide/ecc.py's `plan`).

The chosen profile is written into the sidecar (and the self-contained
header in blind mode), so the receiver never has to guess it.

The per-profile numbers in each `measured` note come from
`scripts/benchmark.py` on real BOSSbase covers - rerun it after changing
anything here. They are a record of what was measured, not a guarantee
for every possible cover image.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class RobustnessProfile:
    name: str
    description: str
    dct_delta: float
    dwt_delta: float
    nsym: int
    fragile_budgets: dict = field(default_factory=dict)
    min_codewords: int = 4
    max_bpp: int = 2  # preferred cap on per-block bpp; relaxed automatically if the message won't fit

    def delta_for(self, method: str) -> float | None:
        return {"DCT": self.dct_delta, "DWT": self.dwt_delta}.get(method)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> RobustnessProfile:
        return cls(**d)


# Why every profile sets the LSB/DWT payload budget to zero: measured on
# 30 BOSSbase covers (scripts/benchmark.py), LSB and DWT carried ~2% of the
# payload bits but caused ~40% of the bit errors under 3x3 median
# filtering (LSB flips ~29-54% of its bits under any attack, DWT ~8-13%,
# DCT ~0.5%). Zeroing their budgets *raised* 10%-crop recovery from 52% to
# 88% and median-filter recovery from 80% to 88% at equal or better PSNR.
# The AIDE engine still decides WHERE (score rank + confidence gate),
# HOW MUCH (bpp) and WHICH METHOD per block; with these budgets, blocks it
# assigns to LSB/DWT are simply left untouched. Pass explicit
# `fragile_method_budgets` to `pipeline.embed` to re-enable them.
_NO_FRAGILE = {"LSB": 0.0, "DWT": 0.0}

PROFILES: dict[str, RobustnessProfile] = {
    "quality": RobustnessProfile(
        name="quality",
        description="Highest fidelity (~43 dB PSNR, SSIM ~0.986 - meets the report's >42 dB "
                    "target). Survives JPEG Q>=50 and small splices; not for noisy channels.",
        dct_delta=20.0, dwt_delta=20.0, nsym=64, fragile_budgets=_NO_FRAGILE,
    ),
    "balanced": RobustnessProfile(
        name="balanced",
        description="Default (~40 dB PSNR). JPEG Q>=30, noise sigma<=10, median/Gaussian "
                    "filtering, 0.75x scaling, 5% splicing.",
        dct_delta=36.0, dwt_delta=36.0, nsym=64, fragile_budgets=_NO_FRAGILE,
    ),
    "robust": RobustnessProfile(
        name="robust",
        description="Meets report O4 end-to-end (~35 dB PSNR): Gaussian noise sigma=20, "
                    "JPEG Q=30-50, salt & pepper, median filter, 10% cropping.",
        dct_delta=64.0, dwt_delta=64.0, nsym=104, fragile_budgets=_NO_FRAGILE,
    ),
    "fortress": RobustnessProfile(
        name="fortress",
        description="Hostile channels (~30 dB PSNR, visible cost): additionally survives "
                    "0.5x down-and-up rescaling. Short messages only.",
        dct_delta=80.0, dwt_delta=80.0, nsym=176, fragile_budgets=_NO_FRAGILE,
    ),
}

DEFAULT_PROFILE = "balanced"


def get_profile(name_or_profile: str | RobustnessProfile | None) -> RobustnessProfile:
    if name_or_profile is None:
        return PROFILES[DEFAULT_PROFILE]
    if isinstance(name_or_profile, RobustnessProfile):
        return name_or_profile
    try:
        return PROFILES[name_or_profile]
    except KeyError:
        raise ValueError(f"unknown profile {name_or_profile!r}; choose from {sorted(PROFILES)}") from None


# The parameters every pre-profile (aide-0.1) sidecar was produced with -
# used to decode those files exactly as they were encoded.
LEGACY_PROFILE = RobustnessProfile(
    name="legacy-0.1", description="Fixed parameters of AIDE 0.1 sidecars.",
    dct_delta=40.0, dwt_delta=40.0, nsym=32,
    fragile_budgets={"LSB": 0.005, "DWT": 0.02}, min_codewords=1, max_bpp=4,
)
