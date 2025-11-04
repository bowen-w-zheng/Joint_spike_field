"""
ct_gibbs/ou_fine.py
-------------------

Fine-resolution Kalman filter, RTS smoother, and FFBS sampler for the
continuous-time SSMT model, using *only* multitaper observations.

Prediction step   : every Δ_spk bin
Observation update: bins that coincide with window centres
                    t_c = offset_sec + k · win_sec

Returned `FineOut` is a NumPy named-tuple
    mu   – smoothed means  (T_f, d)
    var  – smoothed variances (diag) (T_f, d)
    z    – FFBS trajectory  (T_f, d)
"""
from __future__ import annotations
from typing import NamedTuple, Tuple

import numpy as np
from numpy.random import Generator
from .state_index import StateIndex


class KalmanOut(NamedTuple):
    m_f: np.ndarray
    P_f: np.ndarray
    m_s: np.ndarray
    P_s: np.ndarray
    z_draw: np.ndarray | None

# ----------------------------------------------------------------------
def _build_F_Q(lam: np.ndarray, sig_v: np.ndarray, dt: float
               ) -> Tuple[np.ndarray, np.ndarray]:
    exp = np.exp(-lam * dt)
    F   = np.repeat(exp[..., None], 2, -1).reshape(-1)
    Qsc = sig_v**2 * (1 - np.exp(-2 * lam * dt)) / (2 * lam)
    Q   = np.repeat(Qsc[..., None], 2, -1).reshape(-1)
    return F, Q

# def _build_R(sig_eps: np.ndarray) -> np.ndarray:
#     return np.repeat(sig_eps[..., None], 2, -1).reshape(-1)

def _build_R(sig_eps: np.ndarray, J: int | None = None) -> np.ndarray:
    # Expect sig_eps as *std dev*. Use variance and expand across J if needed.
    sig2 = sig_eps**2
    if sig2.ndim == 1:
        if J is None:
            raise ValueError("Provide J when sig_eps is (M,) to tile across J")
        sig2 = np.tile(sig2[None, :], (J, 1))  # (J,M)
    return np.repeat(sig2[..., None], 2, -1).reshape(-1)


# ────────────────────────────────────────────────────────────────────────
class FineOut(NamedTuple):
    mu:  np.ndarray      # (T_f, d)  smoothed means
    var: np.ndarray      # (T_f, d)  smoothed diag variances
    z:   np.ndarray      # (T_f, d)  FFBS sample


# -----------------------------------------------------------------------


def kalman_filter_rts_ffbs_fine(
    Y_cube: np.ndarray,          # (J, M, K) complex
    theta,                       # OUParams  (.lam, .sig_v, .sig_eps)
    delta_spk: float,            # fine-grid step (s)
    win_sec: float,              # multitaper window length / spacing
    offset_sec: float,           # first window centre (s)
    rng: Generator | None = None,
) -> FineOut:
    """
    Returns
    -------
    FineOut(mu, var, z)
    """
    if rng is None:
        rng = np.random.default_rng()

    J, M, K = Y_cube.shape
    d = StateIndex(J, M).dim

    # Indices of coarse centres in fine grid
    step   = int(round(win_sec / delta_spk))
    offset = int(round(offset_sec / delta_spk))
    idx_c  = offset + np.arange(K+1) * step            # (K,)
    T_f    = idx_c[-1] + 1

    # Pre-flatten observations to (K, d)
    y_flat = np.empty((K, d))
    for k in range(K):
        block = np.stack([Y_cube[..., k].real, Y_cube[..., k].imag], -1)
        y_flat[k] = block.reshape(-1)

    F, Q = _build_F_Q(theta.lam, theta.sig_v, delta_spk)
    R    = _build_R(theta.sig_eps)                   # (d,)

    # ------------------------------------------------------------------
    # Forward Kalman filter
    # ------------------------------------------------------------------
    m_f  = np.empty((T_f, d))
    P_f  = np.empty((T_f, d))
    P_pred_arr = np.empty((T_f, d))   # cache for RTS/FFBS

    m_prev = np.zeros(d)
    P_prev = np.ones(d) * 10.0        # diffuse prior

    centre_ptr = 0
    for t in range(T_f):
        # prediction
        if t > 0:
            m_pred = F * m_prev
            P_pred = (F**2) * P_prev + Q
        else:
            m_pred, P_pred = m_prev, P_prev
        P_pred_arr[t] = P_pred

        # observation update if this bin is a centre
        if centre_ptr < K and t == idx_c[centre_ptr]:
            S       = P_pred + R
            K_gain  = P_pred / S          # diag
            innov   = y_flat[centre_ptr] - m_pred
            m_post  = m_pred + K_gain * innov
            P_post  = (1 - K_gain) * P_pred
            centre_ptr += 1
        else:
            m_post, P_post = m_pred, P_pred

        m_f[t], P_f[t] = m_post, P_post
        m_prev, P_prev = m_post, P_post

    # ------------------------------------------------------------------
    # RTS backward smoother
    # ------------------------------------------------------------------
    m_s, P_s = m_f.copy(), P_f.copy()
    for t in range(T_f - 2, -1, -1):
        A = F * P_f[t] / P_pred_arr[t + 1]
        m_s[t] = m_f[t] + A * (m_s[t + 1] - F * m_f[t])
        P_s[t] = P_f[t] + A * (P_s[t + 1] - P_pred_arr[t + 1]) * A

    # ------------------------------------------------------------------
    # FFBS sample
    # ------------------------------------------------------------------
    z = np.empty_like(m_s)
    z[-1] = rng.normal(m_s[-1], np.sqrt(P_s[-1]))
    for t in range(T_f - 2, -1, -1):
        A = F * P_s[t] / P_pred_arr[t + 1]
        mean = m_s[t] + A * (z[t + 1] - F * m_s[t])
        var  = P_s[t] - A * P_pred_arr[t + 1] * A
        z[t] = rng.normal(mean, np.sqrt(var))

    return FineOut(mu=m_s, var=P_s, z=z)
