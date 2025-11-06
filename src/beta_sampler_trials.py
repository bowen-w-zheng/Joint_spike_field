# beta_sampler_trials.py
# Trial-aware β–γ PG–Gaussian update with shared β across trials (and optional per-trial β mode).
# Matches numerics/style of gibbs_update_beta_robust, adds correct pooling & EIV across trials.

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr


# ──────────────────────────────────────────────────────────────────────
def build_design(latent_reim: np.ndarray) -> np.ndarray:
    """
    Prepend intercept column.
    latent_reim: (T, 2B)
    returns:     (T, 1 + 2B)
    """
    latent_reim = np.asarray(latent_reim, dtype=np.float64)
    T = latent_reim.shape[0]
    return np.column_stack([np.ones((T, 1), dtype=np.float64), latent_reim])


@dataclass
class TrialBetaConfig:
    omega_floor: float = 1e-6
    tau2_intercept: float = 100.0**2
    tau2_gamma: float = 25.0**2
    a0_ard: float = 1e-2
    b0_ard: float = 1e-2
    use_exact_cov: bool = False
    chol_jitter0: float = 1e-12
    chol_backoff: int = 6


# ──────────────────────────────────────────────────────────────────────
def _broadcast_gamma_priors(
    R: int,
    Rlags: int,
    Sigma_gamma: Optional[np.ndarray],
    mu_gamma: Optional[np.ndarray],
    tau2_gamma: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create per-trial gamma prior parameters.
    Returns:
        Prec_gamma: (R, Rlags, Rlags)
        mu_gamma:   (R, Rlags)
    """
    if Sigma_gamma is None:
        Prec_gamma = np.broadcast_to((1.0 / max(tau2_gamma, 1e-12)) * np.eye(Rlags), (R, Rlags, Rlags))
    else:
        Sg = np.asarray(Sigma_gamma, dtype=np.float64)
        if Sg.ndim == 2:
            Sg_inv = np.linalg.pinv(Sg)
            Prec_gamma = np.broadcast_to(Sg_inv, (R, Rlags, Rlags))
        elif Sg.ndim == 3:
            Prec_gamma = np.empty_like(Sg)
            for r in range(R):
                try:
                    Prec_gamma[r] = np.linalg.inv(Sg[r])
                except np.linalg.LinAlgError:
                    Prec_gamma[r] = np.linalg.pinv(Sg[r])
        else:
            raise ValueError("Sigma_gamma must be (Rlags,Rlags) or (R,Rlags,Rlags)")

    if mu_gamma is None:
        mu_g = np.zeros((R, Rlags), dtype=np.float64)
    else:
        mg = np.asarray(mu_gamma, dtype=np.float64)
        if mg.ndim == 1:
            mu_g = np.broadcast_to(mg, (R, Rlags))
        elif mg.ndim == 2:
            assert mg.shape == (R, Rlags), "mu_gamma shape mismatch"
            mu_g = mg
        else:
            raise ValueError("mu_gamma must be (Rlags,) or (R,Rlags)")

    return Prec_gamma, mu_g


def _design_per_trial(
    latent_reim: np.ndarray
) -> Tuple[bool, np.ndarray, Optional[np.ndarray]]:
    """
    Accepts latent_reim with shape:
      - (T, 2B): shared predictors across trials
      - (R, T, 2B): trial-specific predictors

    Returns:
      shared: bool
      X_shared: (T, 1+2B) if shared else None
      X_trials: (R, T, 1+2B) if not shared else None
    """
    L = np.asarray(latent_reim)
    if L.ndim == 2:
        X = build_design(L)  # (T, 1+2B)
        return True, X, None
    elif L.ndim == 3:
        R = L.shape[0]
        X_trials = np.empty((R, L.shape[1], 1 + L.shape[2]), dtype=np.float64)
        for r in range(R):
            X_trials[r] = build_design(L[r])
        return False, None, X_trials
    else:
        raise ValueError("latent_reim must be (T,2B) or (R,T,2B)")


# ──────────────────────────────────────────────────────────────────────
def gibbs_update_beta_trials_shared(
    key: "jax.Array",
    latent_reim: np.ndarray,      # (T,2B) shared OR (R,T,2B) trial-specific predictors
    spikes: np.ndarray,           # (R, T)
    omega: np.ndarray,            # (R, T)
    *,
    H_hist: Optional[np.ndarray] = None,          # (R, T, Rlags) or None
    Sigma_gamma: Optional[np.ndarray] = None,     # (Rlags,Rlags) or (R,Rlags,Rlags)
    mu_gamma: Optional[np.ndarray] = None,        # (Rlags,) or (R,Rlags)
    var_latent_reim: Optional[np.ndarray] = None, # (T,2B) or (R,T,2B)
    tau2_lat: Optional[np.ndarray] = None,        # (2B,)
    config: TrialBetaConfig = TrialBetaConfig(),
) -> Tuple["jax.Array", jnp.ndarray, Optional[jnp.ndarray], np.ndarray]:
    """
    Shared-β PG–Gaussian update across R trials with optional per-trial γ.

    Shapes:
      latent_reim: (T,2B) or (R,T,2B)
      spikes:      (R,T), values in {0,1}
      omega:       (R,T), PG weights (will be floored)
      H_hist:      None or (R,T,Rlags)
      Sigma_gamma: None or (Rlags,Rlags) or (R,Rlags,Rlags)
      mu_gamma:    None or (Rlags,) or (R,Rlags)
      var_latent_reim: None or (T,2B) or (R,T,2B)

    Returns:
      key:        PRNG key (advanced)
      beta_out:   (1+2B,)
      gamma_out:  None if no H_hist else (R, Rlags)
      tau2_lat_new: (2B,)
    """
    # Coerce inputs
    spikes = np.asarray(spikes, dtype=np.float64)
    omega  = np.maximum(np.asarray(omega, dtype=np.float64), config.omega_floor)
    R, T = spikes.shape

    # Design matrices
    shared, X_shared, X_trials = _design_per_trial(latent_reim)
    if shared:
        T2 = X_shared.shape[0]
        if T2 != T:
            T = min(T, T2)
            X_shared = X_shared[:T]
            spikes   = spikes[:, :T]
            omega    = omega[:, :T]
        P = X_shared.shape[1]   # 1 + 2B
    else:
        assert X_trials is not None and X_trials.shape[0] == R
        T2 = X_trials.shape[1]
        if T2 != T:
            T = min(T, T2)
            X_trials = X_trials[:, :T, :]
            spikes   = spikes[:, :T]
            omega    = omega[:, :T]
        P = X_trials.shape[2]

    # History design
    has_hist = H_hist is not None
    if has_hist:
        H_hist = np.asarray(H_hist, dtype=np.float64)
        assert H_hist.shape[:2] == (R, T), "H_hist shape mismatch"
        Rlags = H_hist.shape[2]
        Prec_gamma, mu_g = _broadcast_gamma_priors(R, Rlags, Sigma_gamma, mu_gamma, config.tau2_gamma)
    else:
        Rlags = 0
        Prec_gamma = None
        mu_g = None

    # Priors for β
    twoB = P - 1
    if tau2_lat is None or tau2_lat.shape[0] != twoB:
        tau2_lat = np.ones(twoB, dtype=np.float64)

    Prec_beta_prior = np.zeros((P, P), dtype=np.float64)
    Prec_beta_prior[0, 0] = 1.0 / max(config.tau2_intercept, 1e-12)
    Prec_beta_prior[1:P, 1:P] = np.diag(1.0 / np.maximum(tau2_lat, 1e-12))
    mu_beta_prior = np.zeros((P,), dtype=np.float64)

    # Normal equations blocks
    A_bb = np.zeros((P, P), dtype=np.float64)
    b_b  = np.zeros((P,), dtype=np.float64)

    # Cross and gamma blocks (block diagonal over trials)
    if has_hist:
        A_bg = [np.zeros((P, Rlags), dtype=np.float64) for _ in range(R)]
        A_gg = [np.zeros((Rlags, Rlags), dtype=np.float64) for _ in range(R)]
        b_g  = [np.zeros((Rlags,), dtype=np.float64) for _ in range(R)]

    # Accumulate per trial
    for r in range(R):
        X_r = (X_shared if shared else X_trials[r])              # (T,P)
        ω_r = omega[r]                                           # (T,)
        κ_r = spikes[r] - 0.5                                    # (T,)

        # A_bb += Xᵀ Ω X ; b_b += Xᵀ κ
        A_bb += X_r.T @ (ω_r[:, None] * X_r)
        b_b  += X_r.T @ κ_r

        if has_hist:
            H_r = H_hist[r]                                      # (T,Rlags)
            A_bg[r] = X_r.T @ (ω_r[:, None] * H_r)               # (P,Rlags)
            A_gg[r] = H_r.T @ (ω_r[:, None] * H_r) + Prec_gamma[r]
            b_g[r]  = H_r.T @ κ_r + Prec_gamma[r] @ mu_g[r]

    # Errors-in-variables correction for β-latent features
    # diag_add = sum_r (V_r^T @ ω_r)  where V_r is (T,2B)
    if var_latent_reim is not None:
        V = np.asarray(var_latent_reim, dtype=np.float64)
        if V.ndim == 2:
            # shared predictors: (T,2B)
            V_T = V[:T].T                                         # (2B,T)
            diag_add = V_T @ omega.sum(axis=0)                    # (2B,)
        elif V.ndim == 3:
            # per-trial predictors: (R,T,2B)
            diag_add = np.zeros((twoB,), dtype=np.float64)
            for r in range(R):
                V_T_r = V[r, :T].T                                # (2B,T)
                diag_add += V_T_r @ omega[r]
        else:
            raise ValueError("var_latent_reim must be (T,2B) or (R,T,2B)")

        # Add to β block (skip intercept)
        A_bb[1:P, 1:P] += np.diag(diag_add)

    # Add β prior
    A_bb += Prec_beta_prior
    b_b  += Prec_beta_prior @ mu_beta_prior

    # Assemble full precision & RHS: θ = [β , γ_1, ..., γ_R]
    dim = P + (R * Rlags if has_hist else 0)
    Prec = np.zeros((dim, dim), dtype=np.float64)
    RHS  = np.zeros((dim,), dtype=np.float64)

    # β–β
    Prec[:P, :P] = 0.5 * (A_bb + A_bb.T)  # symmetrize just in case
    RHS[:P] = b_b

    # Cross & γ blocks
    if has_hist:
        # Fill β–γ_r and γ_r–β
        for r in range(R):
            i0 = P + r * Rlags
            i1 = i0 + Rlags
            Prec[:P, i0:i1] = A_bg[r]
            Prec[i0:i1, :P] = A_bg[r].T
            Prec[i0:i1, i0:i1] = 0.5 * (A_gg[r] + A_gg[r].T)
            RHS[i0:i1] = b_g[r]

    # Solve / sample
    Prec = 0.5 * (Prec + Prec.T)
    p = Prec.shape[0]

    # Try Cholesky first with jitter backoff
    def _sample_from_prec(_key, _Prec, _RHS, use_exact_cov, jitter0, backoff):
        I = np.eye(p)
        try:
            mean = np.linalg.solve(_Prec, _RHS)
            chol_ok = True
        except np.linalg.LinAlgError:
            U, s, VT = np.linalg.svd(_Prec, hermitian=True)
            s = np.maximum(s, 1e-20)
            mean = VT.T @ ((U.T @ _RHS) / s)
            chol_ok = False

        if use_exact_cov or not chol_ok:
            U, s, VT = np.linalg.svd(_Prec, hermitian=True)
            s = np.maximum(s, 1e-20)
            _key, subk = jr.split(_key)
            eps = jr.normal(subk, (p,), dtype=jnp.float64)
            theta = mean + VT.T @ (np.asarray(eps) / np.sqrt(s))
            return _key, theta

        jitter = jitter0
        for _ in range(backoff):
            try:
                L = np.linalg.cholesky(_Prec + jitter * I)
                break
            except np.linalg.LinAlgError:
                jitter *= 10.0
        else:
            # Fall back to SVD sampling
            U, s, VT = np.linalg.svd(_Prec, hermitian=True)
            s = np.maximum(s, 1e-20)
            _key, subk = jr.split(_key)
            eps = jr.normal(subk, (p,), dtype=jnp.float64)
            theta = mean + VT.T @ (np.asarray(eps) / np.sqrt(s))
            return _key, theta

        _key, subk = jr.split(_key)
        eps = jr.normal(subk, (p,), dtype=jnp.float64)
        theta = mean + np.linalg.solve(L.T, np.asarray(eps))
        return _key, theta

    key, theta = _sample_from_prec(
        key, Prec, RHS, config.use_exact_cov, config.chol_jitter0, config.chol_backoff
    )

    beta_full = theta[:P]
    if has_hist:
        gamma_out = np.empty((R, Rlags), dtype=np.float64)
        for r in range(R):
            i0 = P + r * Rlags
            i1 = i0 + Rlags
            gamma_out[r] = theta[i0:i1]
    else:
        gamma_out = None

    # ARD update on β_lat (shared across trials)
    beta_lat = beta_full[1:]
    alpha_post = config.a0_ard + 0.5
    beta_post  = config.b0_ard + 0.5 * (beta_lat**2)
    # IG sample: tau2 ~ IG(alpha_post, beta_post)
    # Sample via Gamma on precision then invert:
    rng_np = np.random.default_rng()
    tau2_lat_new = 1.0 / rng_np.gamma(shape=alpha_post, scale=1.0 / beta_post)

    return key, jnp.asarray(beta_full), (jnp.asarray(gamma_out) if gamma_out is not None else None), tau2_lat_new


# ──────────────────────────────────────────────────────────────────────
def gibbs_update_beta_trials_per_trial(
    key: "jax.Array",
    latent_reim: np.ndarray,      # (T,2B) shared OR (R,T,2B)
    spikes: np.ndarray,           # (R,T)
    omega: np.ndarray,            # (R,T)
    *,
    H_hist: Optional[np.ndarray] = None,          # (R,T,Rlags) or None
    Sigma_gamma: Optional[np.ndarray] = None,     # (Rlags,Rlags) or (R,Rlags,Rlags)
    mu_gamma: Optional[np.ndarray] = None,        # (Rlags,) or (R,Rlags)
    var_latent_reim: Optional[np.ndarray] = None, # (T,2B) or (R,T,2B)
    tau2_lat: Optional[np.ndarray] = None,        # (R,2B) or None -> init to ones
    config: TrialBetaConfig = TrialBetaConfig(),
):
    """
    Convenience baseline: independent β per trial by calling your single-trial sampler.
    Returns:
      key, beta_r: (R, 1+2B), gamma_r: None or (R, Rlags), tau2_lat_r: (R, 2B)
    """
    from .beta_sampler import gibbs_update_beta_robust as _single  # your existing routine

    spikes = np.asarray(spikes, dtype=np.float64)
    omega  = np.maximum(np.asarray(omega, dtype=np.float64), config.omega_floor)
    R, T = spikes.shape

    # Prepare per-trial design & var
    shared, X_shared, X_trials = _design_per_trial(latent_reim)
    if shared:
        X_trials = np.broadcast_to(X_shared, (R, X_shared.shape[0], X_shared.shape[1]))
    if var_latent_reim is None:
        V_trials = [None] * R
    else:
        V = np.asarray(var_latent_reim, dtype=np.float64)
        if V.ndim == 2:
            V_trials = [V] * R
        elif V.ndim == 3:
            assert V.shape[0] == R
            V_trials = [V[r] for r in range(R)]
        else:
            raise ValueError("var_latent_reim must be (T,2B) or (R,T,2B)")

    # Per-trial gamma priors
    if H_hist is not None:
        H_hist = np.asarray(H_hist, dtype=np.float64)
        Rlags = H_hist.shape[2]
        Prec_gamma, mu_g = _broadcast_gamma_priors(R, Rlags, Sigma_gamma, mu_gamma, config.tau2_gamma)
    else:
        Prec_gamma = mu_g = None
        Rlags = 0

    # Tau2 per trial
    twoB = X_trials.shape[2] - 1
    if tau2_lat is None or tau2_lat.shape != (R, twoB):
        tau2_lat = np.ones((R, twoB), dtype=np.float64)

    beta_out = np.zeros((R, X_trials.shape[2]), dtype=np.float64)
    gamma_out = (np.zeros((R, Rlags), dtype=np.float64) if H_hist is not None else None)
    tau2_out = np.zeros((R, twoB), dtype=np.float64)

    for r in range(R):
        key, beta_r, gamma_r, tau2_r = _single(
            key,
            latent_reim=X_trials[r][:, 1:],  # strip intercept (function re-adds)
            spikes=spikes[r],
            omega=omega[r],
            H_hist=(H_hist[r] if H_hist is not None else None),
            Sigma_gamma=(np.linalg.pinv(Prec_gamma[r]) if Prec_gamma is not None else None),
            mu_gamma=(mu_g[r] if mu_g is not None else None),
            var_latent_reim=V_trials[r],
            a0_ard=config.a0_ard, b0_ard=config.b0_ard,
            tau2_lat=tau2_lat[r],
            tau2_intercept=config.tau2_intercept,
            tau2_gamma=config.tau2_gamma,
            chol_jitter0=config.chol_jitter0, chol_backoff=config.chol_backoff,
            omega_floor=config.omega_floor, use_exact_cov=config.use_exact_cov,
        )
        beta_out[r] = np.asarray(beta_r)
        if gamma_out is not None and gamma_r is not None:
            gamma_out[r] = np.asarray(gamma_r)
        tau2_out[r] = tau2_r

    return key, jnp.asarray(beta_out), (jnp.asarray(gamma_out) if gamma_out is not None else None), tau2_out
