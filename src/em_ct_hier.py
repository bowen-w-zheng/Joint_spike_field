# src/em_ct_hier.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

from src.em_ct import _rtss_ou
EPS = 1e-10
print("em_ct_hier.py loaded")
@dataclass
class EMHierResult:
    lam_X: np.ndarray        # (J, M)     OU rate for shared X
    sigv_X: np.ndarray       # (J, M)     CT-OU diffusion for X
    lam_D: np.ndarray        # (J, M)     OU rate for trial deltas δ
    sigv_D: np.ndarray       # (J, M)     CT-OU diffusion for δ
    sig_eps_jmr: np.ndarray  # (J, M, R)  OBS NOISE STD (band-specific)
    sig_eps_mr: np.ndarray   # (M, R)     OBS NOISE STD (RMS over bands; for compatibility)
    Q_hist: np.ndarray
    X_mean: np.ndarray       # (J, M, K)  complex (smoothed means)
    D_mean: np.ndarray       # (R, J, M, K) complex (smoothed means)

def _normalize_to_RMJK(Y: np.ndarray) -> np.ndarray:
    """
    Accepts Y with shape (R, M, J, K) or (R, J, M, K) and returns (R, M, J, K).
    """
    if Y.ndim != 4:
        raise ValueError("Y must have 4 dims: (R,M,J,K) or (R,J,M,K)")
    if Y.shape[1] == Y.shape[2]:
        return Y
    a, b = Y.shape[1], Y.shape[2]
    if a <= 16 and b > a:
        return Y  # (R, M, J, K)
    else:
        return np.swapaxes(Y, 1, 2)  # (R, J, M, K) -> (R, M, J, K)

def em_ct_hier(
    Y_trials: np.ndarray,
    db: float,
    *,
    max_iter: int = 200,
    tol: float = 1e-3,
    sig_eps_init: float = 5.0,   # initial OBS NOISE std (will be squared internally)
    lam_X_init: float = 0.1,
    sigv_X_init: float = 1.0,
    lam_D_init: float = 0.2,
    sigv_D_init: float = 0.5,
    verbose: bool = False,
    noise_floor: float = 1e-8,   # floor on variance (σ²)
) -> EMHierResult:
    """
    Hierarchical CT-SSMT (LFP only; no spikes): EM with shared X and trial-specific δ_r.

    Observation model:
        Y[r,m,j,k] = X[j,m,k] + δ[r,j,m,k] + ε[r,m,j,k],
        ε ~ NC(0, σ²_{ε,j,m,r})   ← band-specific observation noise

    Parameters
    ----------
    Y_trials : complex ndarray, shape (R, M, J, K) or (R, J, M, K)
        Trial-wise multitaper FFT cubes (derotated & scaled; decimated to K frames).
    db : float
        Block length (seconds).

    Returns
    -------
    EMHierResult with OU params, per-(j,m,r) noise (std reported), Q history, and smoothed means.
    """
    # ---- Shapes and layout normalization ----
    Y = _normalize_to_RMJK(np.asarray(Y_trials))
    R, M, J, K = Y.shape  # (R, M, J, K)

    # ---- Initial per-(j,m,r) observation noise: keep as VARIANCE internally ----
    sig2_eps_jmr = np.full((J, M, R), float(sig_eps_init)**2, dtype=np.float64)

    # ---- OU params (per (J,M)) for shared X and deltas δ ----
    lam_X  = np.full((J, M), lam_X_init,  dtype=np.float64)
    sigv_X = np.full((J, M), sigv_X_init, dtype=np.float64)
    lam_D  = np.full((J, M), lam_D_init,  dtype=np.float64)
    sigv_D = np.full((J, M), sigv_D_init, dtype=np.float64)

    # ---- Storage for smoothed means (latest E-step) ----
    X_mean = np.zeros((J, M, K), dtype=np.complex128)
    D_mean = np.zeros((R, J, M, K), dtype=np.complex128)

    # Also keep one-step-ahead predictions for unbiased σ² update
    xp_X = np.zeros_like(X_mean)
    Pp_X = np.zeros((J, M, K), dtype=np.float64)
    xp_D_all = np.zeros_like(D_mean)
    Pp_D_all = np.zeros((R, J, M, K), dtype=np.float64)

    Q_hist: list[float] = []

    # ===== Helper: precision-pooled observation for X (vectorized over trials; band-specific) =====
    def build_pooled_X_obs(D_mean_local: np.ndarray,
                           sig2_eps_jmr_local: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        D_mean_local : (R, J, M, K)  (smoothed δ means)
        sig2_eps_jmr_local : (J, M, R)   (per (j,m,r) variances)
        Returns:
            Y_pool : (J, M, K)         pooled observation for X
            sig2_pool_jm : (J, M)      effective pooled variance per (j,m)
        """
        # weights w_{j,m,r} = 1 / σ²_{ε,j,m,r}
        W_jmr = 1.0 / np.maximum(sig2_eps_jmr_local, EPS)   # (J, M, R)
        Wsum_jm = np.sum(W_jmr, axis=2)                     # (J, M)

        # Y - D at (R, M, J, K)
        D_rmjk = np.swapaxes(D_mean_local, 1, 2)            # (R, J, M, K) -> (R, M, J, K)
        Y_minus_D = Y - D_rmjk                               # (R, M, J, K)

        # Broadcast weights to (R, M, J, 1) and sum over r
        W_rmj = np.transpose(W_jmr, (2, 1, 0))              # (R, M, J)
        num_mjk = (W_rmj[:, :, :, None] * Y_minus_D).sum(axis=0)  # (M, J, K)

        # Divide by sum of weights per (j,m); move to (J, M, K)
        Wsum_mj = np.transpose(np.maximum(Wsum_jm, EPS), (1, 0))  # (M, J)
        Y_pool_mjk = num_mjk / (Wsum_mj[:, :, None])               # (M, J, K)
        Y_pool = np.swapaxes(Y_pool_mjk, 0, 1)                     # (J, M, K)

        sig2_pool_jm = 1.0 / np.maximum(Wsum_jm, EPS)              # (J, M)
        return Y_pool, sig2_pool_jm

    # ===== EM loop =====
    for it in range(max_iter):

        # ---------- E-step (δ | X): one RTS per trial AND per band (to allow band-specific noise) ----------
        # Accumulate δ EM moments (per (J,M)) across trials for M-step
        Csum_D  = np.zeros((J, M), dtype=np.complex128)
        Rprev_D = np.zeros((J, M), dtype=np.float64)
        Rnext_D = np.zeros((J, M), dtype=np.float64)
        E_ZZ_D  = 0.0
        E_dZ_D  = 0.0

        for r in range(R):
            # Residual for δ smoother per trial, shaped (J,M,K)
            Y_r_jmk = np.swapaxes(Y[r], 0, 1)  # (M,J,K) -> (J,M,K)

            for j in range(J):
                Y_res_j = Y_r_jmk[j] - X_mean[j]  # (M, K)

                xs_D_j, Ps_D_j, Csum_r_j, Rprev_r_j, Rnext_r_j, xp_D_j, Pp_D_j, \
                E_YY_j, E_YZ_j, E_ZZ_j, E_dZ_j = _rtss_ou(
                    Y_res_j[None, :, :],              # (1, M, K)
                    lam=lam_D[j:j+1, :],              # (1, M)
                    sig_v=sigv_D[j:j+1, :],           # (1, M)
                    sig_eps=np.sqrt(np.maximum(sig2_eps_jmr[j, :, r], EPS)),  # (M,)
                    db=db
                )

                # Store smoothed and predicted moments
                D_mean[r, j]   = xs_D_j[0]                   # (M, K)
                xp_D_all[r, j] = xp_D_j[0]                   # (M, K)
                Pp_D_all[r, j] = Pp_D_j.real[0]              # (M, K)

                # Accumulate EM sums (place into the j-th row)
                Csum_D[j]  += Csum_r_j[0]
                Rprev_D[j] += Rprev_r_j[0]
                Rnext_D[j] += Rnext_r_j[0]
                E_ZZ_D     += E_ZZ_j
                E_dZ_D     += E_dZ_j

        # ---------- E-step (X | δ): precision-pooled RTS per band ----------
        Y_pool, sig2_pool_jm = build_pooled_X_obs(D_mean, sig2_eps_jmr)

        # Run RTS per band to allow band-specific pooled noise
        xs_X = np.zeros_like(X_mean)         # (J, M, K)
        Ps_X = np.zeros((J, M, K), dtype=np.float64)
        Csum_X = np.zeros((J, M), dtype=np.complex128)
        Rprev_X = np.zeros((J, M), dtype=np.float64)
        Rnext_X = np.zeros((J, M), dtype=np.float64)
        E_YY_X = 0.0
        E_YZ_X = 0.0
        E_ZZ_X = 0.0
        E_dZ_X = 0.0

        for j in range(J):
            xs_X_j, Ps_X_j, Csum_X_j, Rprev_X_j, Rnext_X_j, xp_X_j, Pp_X_j, \
            E_YY_X_j, E_YZ_X_j, E_ZZ_X_j, E_dZ_X_j = _rtss_ou(
                Y_pool[j][None, :, :],                   # (1, M, K)
                lam=lam_X[j:j+1, :],                     # (1, M)
                sig_v=sigv_X[j:j+1, :],                  # (1, M)
                sig_eps=np.sqrt(np.maximum(sig2_pool_jm[j], EPS)),  # (M,)
                db=db
            )
            xs_X[j] = xs_X_j[0]
            Ps_X[j] = Ps_X_j.real[0]
            xp_X[j] = xp_X_j[0]
            Pp_X[j] = Pp_X_j.real[0]

            Csum_X[j]  = Csum_X_j[0]
            Rprev_X[j] = Rprev_X_j[0]
            Rnext_X[j] = Rnext_X_j[0]
            E_YY_X += E_YY_X_j
            E_YZ_X += E_YZ_X_j
            E_ZZ_X += E_ZZ_X_j
            E_dZ_X += E_dZ_X_j

        X_mean = xs_X  # (J,M,K)

        # ---------- M-step: update OU params for X ----------
        Phi_X = np.clip(np.abs(Csum_X) / (Rprev_X + EPS), EPS, 1.0 - EPS)
        lam_X = -np.log(Phi_X) / db
        Qhat_X = (Rnext_X - Phi_X * np.conj(Csum_X)).real / K
        Qhat_X = np.maximum(Qhat_X, EPS)
        sigv_X = np.sqrt(2.0 * lam_X * Qhat_X / np.maximum(1.0 - Phi_X**2, EPS))

        # ---------- M-step: update OU params for δ (pooled across trials) ----------
        Phi_D = np.clip(np.abs(Csum_D) / (Rprev_D + EPS), EPS, 1.0 - EPS)
        lam_D = -np.log(Phi_D) / db
        Qhat_D = (Rnext_D - Phi_D * np.conj(Csum_D)).real / (K * max(R, 1))
        Qhat_D = np.maximum(Qhat_D, EPS)
        sigv_D = np.sqrt(2.0 * lam_D * Qhat_D / np.maximum(1.0 - Phi_D**2, EPS))

        # ---------- M-step: unbiased update of obs noise VARIANCE σ²_{ε,j,m,r} ----------
        # Use one-step-ahead predictions: Z_pred = xp_X + xp_D, Var_pred = Pp_X + Pp_D
        for r in range(R):
            Z_pred = xp_X + xp_D_all[r]                 # (J,M,K)
            Var_pred = (Pp_X + Pp_D_all[r]).real        # (J,M,K)

            # Convert to (M,J,K) to align with Y[r]
            Z_pred_mjk = np.swapaxes(Z_pred, 0, 1)      # (M,J,K)
            Var_mjk    = np.swapaxes(Var_pred, 0, 1)    # (M,J,K)

            resid_mjk = Y[r] - Z_pred_mjk               # (M,J,K)
            resid_jmk = np.swapaxes(resid_mjk, 0, 1)    # (J,M,K)

            # E|Y-Z|^2 = |Y - E[Z]|^2 + Var[Z], average over frames K
            sig2_eps_jmr[:, :, r] = np.mean(np.abs(resid_jmk)**2 + Var_pred, axis=2)  # (J,M)

        # Numerical floor to avoid degeneracy (applied elementwise)
        sig2_eps_jmr = np.maximum(sig2_eps_jmr, noise_floor)

        # ---------- Q monitor (coarse) ----------
        # This is a rough scalar; useful only for convergence checks
        Rbar = float(np.mean(sig2_eps_jmr))
        Qbar_X = float(np.mean(Qhat_X))
        Qbar_D = float(np.mean(Qhat_D))
        Q_val = -(E_ZZ_X - 2*E_YZ_X + E_YY_X) / (Rbar + EPS) \
                -  E_dZ_X / (Qbar_X + EPS) \
                -  E_dZ_D / (Qbar_D + EPS)
        Q_hist.append(Q_val)

        if verbose and (it % 20 == 0 or it == max_iter - 1):
            print(f"[EM-CT-HIER] iter {it:4d}  Q≈{Q_val: .4e}")

        # ---------- Convergence ----------
        if it > 0 and abs(Q_hist[-1] - Q_hist[-2]) < tol:
            if verbose:
                print(f"[EM-CT-HIER] converged at iter {it}  Q≈{Q_val: .4e}")
            break

    # Return std for convenience/logging
    sig_eps_jmr = np.sqrt(sig2_eps_jmr)                        # (J,M,R)
    sig_eps_mr  = np.sqrt(np.mean(sig2_eps_jmr, axis=0))       # (M,R) RMS over bands for compatibility

    return EMHierResult(
        lam_X=lam_X, sigv_X=sigv_X,
        lam_D=lam_D, sigv_D=sigv_D,
        sig_eps_jmr=sig_eps_jmr,           # band-specific STD
        sig_eps_mr=sig_eps_mr,             # summarized STD (RMS over J)
        Q_hist=np.asarray(Q_hist, dtype=np.float64),
        X_mean=X_mean,                      # (J,M,K)
        D_mean=D_mean                       # (R,J,M,K)
    )
