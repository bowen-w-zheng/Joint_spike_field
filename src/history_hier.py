# src/history_strict.py
from __future__ import annotations
import numpy as np
from typing import Sequence, Optional, Literal

def _require_scalar(x, name: str) -> float:
    arr = np.asarray(x)
    if arr.ndim == 0 or arr.size == 1:
        return float(arr.reshape(()))
    raise TypeError(f"{name} must be a scalar or size-1 array, got shape {arr.shape}")

def lag_bins_from_seconds_strict(lags_sec: Sequence[float], delta_spk) -> np.ndarray:
    """
    Convert lags (sec) → integer bins, failing if not exact multiples of delta_spk.
    Returns unique, sorted, positive ints (>=1).
    """
    dt = _require_scalar(delta_spk, "delta_spk")
    l = np.asarray(lags_sec, float).ravel()
    if l.size == 0: raise ValueError("lags_sec is empty")
    if np.any(l <= 0): raise ValueError("lags_sec must be > 0")
    bins_f = l / dt
    bins_r = np.rint(bins_f)
    err = np.max(np.abs(bins_f - bins_r))
    if err > 1e-10:
        raise ValueError(
            f"lags_sec must be integer multiples of delta_spk={dt:g}; "
            f"max fractional-bin error={err:.3e}."
        )
    bins = bins_r.astype(int)
    bins[bins < 1] = 1
    return np.unique(bins)

def assert_spikes_R_S_T(spikes) -> np.ndarray:
    """
    Validate spikes is a dense 3D array (R,S,T) with numeric type (no ragged object arrays).
    Returns float32 view (no copying if possible).
    """
    if not isinstance(spikes, np.ndarray) or spikes.ndim != 3:
        raise TypeError(f"spikes must be ndarray with shape (R,S,T); got {type(spikes)} with ndim={getattr(spikes,'ndim',None)}")
    if spikes.dtype == object:
        raise TypeError("spikes has dtype=object (ragged). Make T identical across trials before calling this builder.")
    return np.asarray(spikes, dtype=np.float32, order="C")

def build_history_lags_hier_strict(
    spikes: np.ndarray,          # (R,S,T) float/bool/uint8, validated
    lag_bins: Sequence[int],     # positive ints, exact bins
    mode: Literal["self","self+pop","all"] = "self",
) -> np.ndarray:
    """
    Construct per-trial spike-history design WITHOUT cross-trial leakage.
    Shapes:
      spikes : (R,S,T)
      lag_bins: e.g. array([1,2,4,8]) in bins
      mode='self'      → H: (R,S,T, L)
      mode='self+pop'  → H: (R,S,T, 2L)  [self | sum(others)]
      mode='all'       → H: (R,S,T, S·L) [all presyn units for every target]
    """
    X = assert_spikes_R_S_T(spikes)    # (R,S,T)
    R, S, T = X.shape
    lags = np.asarray(lag_bins, int).ravel()
    if lags.ndim != 1 or np.any(lags <= 0):
        raise ValueError("lag_bins must be a 1D array of positive integers")
    L = int(lags.size)

    if mode == "self":
        H = np.zeros((R, S, T, L), dtype=np.float32)
        for i, Lg in enumerate(lags):
            if Lg >= T: continue
            H[..., Lg:, i] = X[..., :-Lg]
        return H

    if mode == "self+pop":
        H = np.zeros((R, S, T, 2*L), dtype=np.float32)
        pop = X.sum(axis=1, keepdims=True) - X    # (R,S,T)
        for i, Lg in enumerate(lags):
            if Lg >= T: continue
            H[..., Lg:, i]      = X[..., :-Lg]         # self
            H[..., Lg:, L + i]  = pop[..., :-Lg]       # ∑(others)
        return H

    if mode == "all":
        H = np.zeros((R, S, T, S*L), dtype=np.float32)
        for u in range(S):
            Xu = X[:, u, :]                          # (R,T)
            for i, Lg in enumerate(lags):
                if Lg >= T: continue
                # broadcast presynaptic unit u to all targets s
                H[:, :, Lg:, u*L + i] = Xu[:, None, :-Lg]
        return H

    raise ValueError("mode must be 'self', 'self+pop', or 'all'")
