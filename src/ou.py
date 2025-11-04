"""
ct_ssmt.ou
==========

Low–level utilities for the Ornstein–Uhlenbeck (OU) latent dynamics used
in the continuous-time state-space multitaper (CT-SSMT) model.

Contents
--------
phi_q(...)                   – Φ , Q  transition matrices
kalman_filter_ou(...)        – public wrapper; chooses JAX or Numba
_kalman_filter_ou_numba(...) – CPU fast-path (Numba -O3)
_kalman_filter_ou_jax(...)   – optional JAX GPU/CPU implementation
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from numba import njit, prange

__all__ = [
    "phi_q",
    "kalman_filter_ou",
]

# ---------------------------------------------------------------------
# 0.  Optional JAX back-end
# ---------------------------------------------------------------------
try:
    import jax
    import jax.numpy as jnp
    _JAX_AVAILABLE = True
except Exception:                    # pragma: no cover
    _JAX_AVAILABLE = False


# ---------------------------------------------------------------------
# 1.  OU transition helper  Φ, Q
# ---------------------------------------------------------------------
@njit(cache=True, fastmath=True, inline="always")
def _phi_q(lam: NDArray[np.float64],
           sig_v: NDArray[np.float64],
           db: float):
    phi = np.exp(-lam * db)
    q   = (sig_v**2) * (1.0 - phi**2) / (2.0 * lam + 1e-12)
    return phi, q
# ---------------------------------------------------------------------
@njit(parallel=True, cache=True, fastmath=True)
def kalman_filter_ou_numba(Y: NDArray[np.complex128],
                           lam: NDArray[np.float64],
                           sig_v: NDArray[np.float64],
                           sig_eps: NDArray[np.float64],
                           db: float):
    """
    Forward–backward Kalman/RTS filter for M independent OU chains per
    frequency.

    Parameters
    ----------
    Y       : complex (Jf, M, K)
    lam     : (Jf,)
    sig_v   : (Jf,)
    sig_eps : (M,)
    db      : block length [s]

    Returns
    -------
    xp, Pp  : prediction mean / var      (Jf, M, K)
    xs, Ps  : smoothed   mean / var      (Jf, M, K)
    """
    Jf, M, K = Y.shape
    phi, q = _phi_q(lam, sig_v, db)

    xp = np.zeros((Jf, M, K), Y.dtype)
    Pp = np.zeros((Jf, M, K))
    xf = np.zeros_like(xp)
    Pf = np.zeros_like(Pp)

    # ─── forward filter ───
    for j in prange(Jf):
        for m in range(M):
            Phi, Q = phi[j,m], q[j,m] 
            z, P = 0.0 + 0.0j, 1e6
            for k in range(K):
                # predict
                z_pred = Phi * z
                P_pred = Phi * Phi * P + Q
                xp[j, m, k] = z_pred
                Pp[j, m, k] = P_pred

                # update
                R = sig_eps[m] ** 2
                S = P_pred + R
                Kg = P_pred / S
                y  = Y[j, m, k]
                z_pred += Kg * (y - z_pred)
                P_pred  = (1.0 - Kg) * P_pred

                xf[j, m, k] = z_pred
                Pf[j, m, k] = P_pred
                z, P = z_pred, P_pred

    # ─── RTS smoother ───
    xs = xf.copy()
    Ps = Pf.copy()

    for j in prange(Jf):
        for m in range(M):
            Phi, Q = phi[j,m], q[j,m]
            for k in range(K - 2, -1, -1):
                P_pred = Phi * Phi * Pf[j, m, k] + Q
                G = Pf[j, m, k] * Phi / P_pred
                xs[j, m, k] += G * (xs[j, m, k + 1] - Phi * xf[j, m, k])
                Ps[j, m, k] += G * G * (Ps[j, m, k + 1] - P_pred)

    return xp, Pp, xs, Ps
# ---------------------------------------------------------------------
# 3.  Optional JAX Kalman filter  (GPU / CPU XLA)
# ---------------------------------------------------------------------
if _JAX_AVAILABLE:                                   # pragma: no cover
    # jit once; keep outside wrapper for reuse
# ---------------------------------------------------------------------
    def kalman_filter_ou_jax(Y: jnp.ndarray,
                            lam: jnp.ndarray,
                            sig_v: jnp.ndarray,
                            sig_eps: jnp.ndarray,
                            db: float):
        """
        Same contract as the Numba version but fully vectorised with JAX.

        All arrays are immutable; retval is a tuple (xp, Pp, xs, Ps).
        """
        Jf, M, K = Y.shape
        phi = jnp.exp(-lam * db)                       # (Jf,)
        q   = (sig_v**2) * (1.0 - phi**2) / (2.0*lam) # (Jf,)

        # Broadcast to (Jf,M)
        Phi = phi[:, None]
        Q   = q[:, None]
        R   = sig_eps[None, :] ** 2                   # (1,M)

        def _filter_single(y_jm, phi_j, q_j, r_m):
            """
            y_jm : (K,)
            Returns (xp, Pp, xs, Ps) each shape (K,)
            """
            def fwd(carry, y_t):
                z, P = carry
                # predict
                z_pred = phi_j * z
                P_pred = phi_j**2 * P + q_j
                # update
                S  = P_pred + r_m
                Kg = P_pred / S
                z   = z_pred + Kg * (y_t - z_pred)
                P   = (1.0 - Kg) * P_pred
                return (z, P), (z_pred, P_pred, z, P)

            (_, _), hist = jax.lax.scan(fwd, (0.0j, 1e6), y_jm)
            xp, Pp, xs, Ps = hist
            # backward RTS
            def bwd(carry, k):
                xs_next, Ps_next = carry
                P_pred = phi_j**2 * Pp[k] + q_j
                G = Pp[k] * phi_j / P_pred
                xs_k = xs[k] + G * (xs_next - phi_j * xp[k+1])
                Ps_k = Ps[k] + G*G * (Ps_next - P_pred)
                return (xs_k, Ps_k), (xs_k, Ps_k)

            init = (xs[-1], Ps[-1])
            (_, _), rev = jax.lax.scan(bwd, init, jnp.arange(K-2, -1, -1))
            xs_full = jnp.concatenate([jnp.flip(rev[0], 0), xs[-1:]], axis=0)
            Ps_full = jnp.concatenate([jnp.flip(rev[1], 0), Ps[-1:]], axis=0)
            return xp, Pp, xs_full, Ps_full

        # vmap over (j,m)
        xp, Pp, xs, Ps = jax.vmap(
            jax.vmap(_filter_single, in_axes=(1, 0, 0, None), out_axes=1),
            in_axes=(0, 0, 0, None),                    # broadcast r_m later
            out_axes=0
        )(Y, phi, q, R)

        return xp, Pp, xs, Ps

# ---------------------------------------------------------------------
# 4.  Public wrapper
# ---------------------------------------------------------------------
def kalman_filter_ou(Y, lam, sig_v, sig_eps, db, *, use_jax=False):
    if use_jax:
        return kalman_filter_ou_jax(
            jnp.asarray(Y), jnp.asarray(lam), jnp.asarray(sig_v),
            jnp.asarray(sig_eps), db
        )
    else:
        return kalman_filter_ou_numba(
            np.asarray(Y), np.asarray(lam), np.asarray(sig_v),
            np.asarray(sig_eps), db
        )
