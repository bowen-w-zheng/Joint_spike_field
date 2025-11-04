"""
params.py  –  Container for CT‑SSMT parameters (OU + observation noise)

Shapes (single source of truth)
--------------------------------
    lam, sig_v : (Jf, M)   # one OU process per (frequency, taper)
    sig_eps    : (M,)      # observation noise for each taper
    db         : float     # block length in seconds  (same for all)

All downstream modules must abide by these shapes.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

__all__ = ["CTParams", "OUParams", "BetaParams"]

@dataclass
class CTParams:
    """Dataclass holding model parameters and offering useful transforms."""
    lam: NDArray        # shape (Jf, M)  – OU decay rates
    sig_v: NDArray      # shape (Jf, M)  – OU innovation std‐dev
    sig_eps: NDArray    # shape (M,)     – measurement noise std‐dev
    db: float                       # block length (seconds)

    # ────────────────────────────────────────────────────────────────────
    #  Post‑init validation
    # ────────────────────────────────────────────────────────────────────
    def __post_init__(self) -> None:
        jf_m = self.lam.shape
        if self.sig_v.shape != jf_m:
            raise ValueError(
                f"`sig_v` must have the same shape as `lam` "
                f"(got {self.sig_v.shape} vs {jf_m})"
            )
        if self.sig_eps.ndim != 1 or self.sig_eps.shape[0] != jf_m[1]:
            raise ValueError(
                "`sig_eps` must be 1‑D with length M = lam.shape[1] "
                f"(got {self.sig_eps.shape} vs M = {jf_m[1]})"
            )
        if not (self.lam > 0).all():
            raise ValueError("All elements of `lam` must be positive.")
        if not (self.sig_v > 0).all():
            raise ValueError("All elements of `sig_v` must be positive.")
        if not (self.sig_eps > 0).all():
            raise ValueError("All elements of `sig_eps` must be positive.")
        if self.db <= 0:
            raise ValueError("`db` (block length) must be positive.")

    # ────────────────────────────────────────────────────────────────────
    #  Convenience properties
    # ────────────────────────────────────────────────────────────────────
    @property
    def phi(self) -> NDArray:
        """Discrete‑time AR(1) coefficient for each (frequency, taper)."""
        return np.exp(-self.lam * self.db)

    @property
    def q(self) -> NDArray:
        r"""Discrete‑time process‑noise variance.

        q = σᵥ² · (1 − e^(−2λΔt)) ⁄ (2λ)
        """
        lam, sig_v, dt = self.lam, self.sig_v, self.db
        return (sig_v ** 2) * (1.0 - np.exp(-2.0 * lam * dt)) / (2.0 * lam)

    # ────────────────────────────────────────────────────────────────────
    #  Pretty representation
    # ────────────────────────────────────────────────────────────────────
    def __repr__(self) -> str:  # pragma: no cover
        j, m = self.lam.shape
        return (
            f"CTParams(Jf={j}, M={m}, db={self.db:.3g}, "
            f"<lam min={self.lam.min():.3g}, max={self.lam.max():.3g}>, "
            f"<sig_v min={self.sig_v.min():.3g}, max={self.sig_v.max():.3g}>, "
            f"<sig_eps min={self.sig_eps.min():.3g}, max={self.sig_eps.max():.3g}>)"
        )



@dataclass
class OUParams:
    """
    Ornstein–Uhlenbeck dynamics, one set of scalars per (band, taper).

    Shapes
    ------
    lam     : (J, M)    decay rates   λ  (s⁻¹)
    sig_v   : (J, M)    innovation σ_v
    sig_eps : (J, M)    observation σ_ε  (multitaper FFT noise)
    """
    lam: np.ndarray       # (J,M)
    sig_v: np.ndarray     # (J,M)
    sig_eps: np.ndarray   # (J,M)

    def copy(self) -> "OUParams":
        """Deep copy – avoids accidental in-place modification."""
        return OUParams(
            lam=self.lam.copy(),
            sig_v=self.sig_v.copy(),
            sig_eps=self.sig_eps.copy(),
        )


@dataclass
class BetaParams:
    """
    Regression coefficients that couple latent spectral coefficients to
    spike log-odds via

        logit p_t = β_0  +
                    Σ_b [ β_{R,b} Re{Z̄_b(t)} + β_{I,b} Im{Z̄_b(t)} ].

    Shape
    -----
    beta : (1 + 2*B,)       [ β_0 , β_R,1 , β_I,1 , … , β_R,B , β_I,B ]
    """
    beta: np.ndarray        # 1-D vector

    def copy(self) -> "BetaParams":
        return BetaParams(beta=self.beta.copy())

