"""
Expectation–Maximisation for the continuous‑time OU‑SSMT model
==============================================================

This module provides a numerically‑stable, unit‑consistent EM algorithm for the
continuous‑time, complex‑valued Ornstein‑Uhlenbeck (OU) latent dynamics that
underpin the CT‑SSMT estimator.

Key points
----------
1. **Unitary‑FFT assumption** – All spectra passed to `em_ct` must come from a
   *unitary* FFT, i.e. NumPy’s default output divided by ``sqrt(nw)``.  If you
   follow the test‑script template this is already the case.
2. **Taper gain removed** – Each taper’s complex DC gain ``S_m`` must be
   divided out of the FFT cube *before* calling `em_ct`.  This leaves a unit
   observation matrix and avoids power biases.
3. **Monotone Q monitor** – The driver now prints the *true* expected
   complete‑data log‑likelihood (up to an irrelevant constant).  It increases
   monotonically with every EM iteration when the M‑step is applied exactly.

All loops are JIT‑compiled with Numba; the code runs in nopython mode.
"""
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from numba import njit, prange
from .ou import _phi_q

EPS = 1e-12
__all__ = ["em_ct"]

# ────────────────────────────────────────────────────────────────
# 0. Rauch‑Tung‑Striebel smoother  + moments  (Numba)
# ────────────────────────────────────────────────────────────────

@njit(parallel=True, cache=True, fastmath=True)
def _rtss_ou(Y: np.ndarray,
             lam: np.ndarray,
             sig_v: np.ndarray,
             sig_eps: np.ndarray,
             db: float):
    """
    Rauch‑Tung‑Striebel smoother + EM statistics for the complex OU model
    treating **each taper m as its own latent chain**.

    Parameters
    ----------
    Y       : complex ndarray, shape (Jf, M, K)
              multitaper FFT cube  (freq × taper × block)
    lam     : (Jf,)  OU decay rates λ_j  [s⁻¹]
    sig_v   : (Jf,)  OU process noise σ_{v,j}
    sig_eps : (M,)   taper–specific measurement noise σ_{ε,m}
    db      : scalar block length [s]

    Returns
    -------
    xs, Ps      : ndarray (Jf, M, K)  – smoothed mean / variance
    Csum        : (Jf,)               – Σ_k ⟨Z_k Z*_{k‑1}⟩
    Rprev,Rnext : (Jf,)               – Σ_k ⟨|Z_{k‑1}|²⟩ / ⟨|Z_k|²⟩
    xp, Pp      : ndarray (Jf, M, K)  – one‑step prediction mean / var
    E_YY, E_YZ,
    E_ZZ, E_dZ  : scalars             – global EM moments
    """
    Jf, M, K = Y.shape
    phi, q = _phi_q(lam, sig_v, db)

    # ─────────── Forward Kalman filter ───────────
    xp = np.zeros((Jf, M, K), Y.dtype)
    Pp = np.zeros((Jf, M, K))
    xf = np.zeros_like(xp)
    Pf = np.zeros_like(Pp)

    for j in prange(Jf):
        for m in range(M):
            Phi, Q = phi[j,m], q[j,m]
            z, P = 0.0 + 0.0j, 1e3          # diffuse prior
            for k in range(K):
                # predict
                z_pred = Phi * z
                P_pred = Phi*Phi * P + Q
                xp[j, m, k] = z_pred
                Pp[j, m, k] = P_pred

                # update with that taper’s observation
                R = sig_eps[m]**2
                S = P_pred + R
                Kg = P_pred / S
                y = Y[j, m, k]
                z_pred += Kg * (y - z_pred)
                P_pred  = (1.0 - Kg) * P_pred

                xf[j, m, k] = z_pred
                Pf[j, m, k] = P_pred
                z, P = z_pred, P_pred

    # ─────────── RTS backward smoother ───────────
    xs = xf.copy()
    Ps = Pf.copy()

    Csum  = np.zeros((Jf, M), Y.dtype)
    Rprev = np.zeros((Jf, M))
    Rnext = np.zeros((Jf, M))

    E_YY = np.sum(np.abs(Y)**2)
    E_YZ = 0.0
    E_ZZ = 0.0
    E_dZ = 0.0

    for j in prange(Jf):
        for m in range(M):
            Phi, Q = phi[j,m], q[j,m]
            # backward pass
            for k in range(K-2, -1, -1):
                P_pred = Phi*Phi*Pf[j, m, k] + Q
                G = Pf[j, m, k] * Phi / P_pred
                xs[j, m, k] += G * (xs[j, m, k+1] - Phi*xf[j, m, k])
                Ps[j, m, k] += G*G * (Ps[j, m, k+1] - P_pred)

            # EM moment accumulation
            Csum[j, m]  += np.sum(xs[j, m, 1:] * np.conj(xs[j, m, :-1]))
            Rprev[j, m] += np.sum(np.abs(xs[j, m, :-1])**2 + Ps[j, m, :-1])
            Rnext[j, m] += np.sum(np.abs(xs[j, m, 1: ])**2 + Ps[j, m, 1: ])
            E_ZZ     += np.sum(np.abs(xs[j, m])**2     + Ps[j, m])
            E_YZ     += np.real(np.sum(Y[j, m] * np.conj(xs[j, m])))

            diff = xs[j, m, 1:] - Phi * xs[j, m, :-1]
            E_dZ += np.sum(
                np.abs(diff)**2
                + Ps[j, m, 1:]
                + (Phi**2) * Ps[j, m, :-1]
                - 2*Phi*np.real(xs[j, m, 1:] * np.conj(xs[j, m, :-1]))
            )

    return (xs, Ps, Csum, Rprev, Rnext,
            xp, Pp, E_YY, E_YZ, E_ZZ, E_dZ)

EPS = 1e-12  # keep at top of module

def em_ct(Y: NDArray[np.complex128],
          db: float,
          *,
          max_iter: int = 500,
          tol: float   = 1e-3,
          sig_eps_init: float = 5.0,
          verbose: bool = False,
          obs_cut: int | None = None,
          return_moments: bool = False,
          freeze_lam_iters: int = 50):

    Jf, M, K = Y.shape
    lam      = np.full((Jf, M), 0.1)
    sig_v    = np.full((Jf, M), 1.0)
    sig_eps  = np.full(M, sig_eps_init)
    ll_hist: list[float] = []

    # Initialize the sig_eps while keep lam constant for a few iterations to avoid collapsing


    for it in range(max_iter):
        # ─── E‑step ───
        (xs, Ps, Csum, Rprev, Rnext,
         xp, Pp, E_YY, E_YZ, E_ZZ, E_dZ) = _rtss_ou(
            Y, lam, sig_v, sig_eps, db
        )

        # ─── M‑step : dynamics ───
        Phi   = np.clip(np.abs(Csum)/(Rprev + EPS), EPS, 1.0-EPS)
        if it == freeze_lam_iters:
            print(f"Unfreeze lam at {it}")
        if it >= freeze_lam_iters:               # update λ after warm‑up
            lam = -np.log(Phi) / db
        Qhat  = (Rnext - Phi*Csum.conj()).real / K 
        Qhat  = np.maximum(Qhat, EPS)
        sig_v = np.sqrt(2*lam*Qhat / np.maximum(1 - Phi**2, EPS))

        # ─── M‑step : measurement noise ───
        resid2 = np.abs(Y - xs)**2 + Ps
        # ----- pySSMT‑style thresholding ---------------------------------
        wnw = int(obs_cut) if obs_cut is not None else Jf
        if wnw < 1 or wnw > Jf:
            raise ValueError("obs_cut must be in [1, Jf]")
        # print(resid2.shape)
        sig_eps = np.sqrt(resid2[:wnw].mean(axis=(0, 2)) + EPS)
        # ─── expected complete‑data log‑likelihood (up to const) ───
        Rbar = np.mean(sig_eps**2)                         # ← mean of variances
        Qbar = np.mean(Qhat)
        ll   = -(E_YY - 2*E_YZ + E_ZZ) / (Rbar + EPS) \
               -  E_dZ / (Qbar + EPS) \
               -  K * Jf * M * np.log(Rbar + EPS) \
               -  K * Jf       * np.log(Qbar + EPS)
        ll_hist.append(ll)

        if verbose and (it % 200 == 0 or it == max_iter - 1):
            print(f"[EM‑CT]  iter {it:4d}  Q = {ll: .4e}")

        # convergence
        if it > 0 and abs(ll_hist[-1] - ll_hist[-2]) < tol:
            if verbose:
                print(f"[EM‑CT]  converged at iter {it}  Q = {ll: .4e}")
            break

    out = (lam, sig_v, sig_eps, np.asarray(ll_hist))
    if return_moments:
        return (*out, xs, Ps)
    return out
