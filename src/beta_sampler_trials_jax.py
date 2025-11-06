# src/beta_sampler_trials_jax.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import jax, jax.numpy as jnp, jax.random as jr
from jax import vmap

@dataclass
class TrialBetaJAXConfig:
    omega_floor: float = 1e-6
    tau2_intercept: float = 100.0**2
    a0_ard: float = 1e-2
    b0_ard: float = 1e-2

@jax.jit
def build_design_jax(latent_reim: jnp.ndarray) -> jnp.ndarray:
    """latent_reim: (T,2B) → (T,1+2B) with leading intercept."""
    T = latent_reim.shape[0]
    return jnp.concatenate([jnp.ones((T,1), latent_reim.dtype), latent_reim], axis=1)

def _psd(A: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    return 0.5*(A+A.T) + eps*jnp.eye(A.shape[0], dtype=A.dtype)

def _beta_gamma_shared_gamma_unit(
    key: jr.KeyArray,
    X: jnp.ndarray,                 # (T,P), shared across trials
    H_rtl: jnp.ndarray,             # (R,T,L)
    spikes_rt: jnp.ndarray,         # (R,T) in {0,1}
    omega_rt: jnp.ndarray,          # (R,T)
    V_rtb: jnp.ndarray,             # (R,T,2B)  (tile shared V if needed)
    Prec_gamma_ll: jnp.ndarray,     # (L,L)     (per-unit prior precision)
    mu_gamma_l: jnp.ndarray,        # (L,)      (per-unit prior mean)
    tau2_lat_b: jnp.ndarray,        # (2B,)
    cfg: TrialBetaJAXConfig,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Exact pooled info with shared β (P) and shared γ (L) across trials (per unit)."""
    key1, key2 = jr.split(key)
    R, T = spikes_rt.shape
    P    = X.shape[1]
    twoB = P - 1
    L    = H_rtl.shape[2]

    omega_rt = jnp.maximum(omega_rt, cfg.omega_floor)
    kappa_rt = spikes_rt - 0.5

    # Σ_r ω_r(t), Σ_r κ_r(t), Σ_r ω_r(t)H_r(t,:)
    omega_sum_t  = omega_rt.sum(axis=0)                                # (T,)
    kappa_sum_t  = kappa_rt.sum(axis=0)                                # (T,)
    H_agg_tl     = jnp.einsum('rt,rtl->tl', omega_rt, H_rtl)          # (T,L)

    # Blocks
    # A_bb = Xᵀ(ΣΩ)X + diag(tau^-2) + diag(Σ_r V_rᵀ ω_r)
    Prec_beta_diag = jnp.concatenate([jnp.array([1.0/cfg.tau2_intercept], X.dtype),
                                      1.0/jnp.maximum(tau2_lat_b,1e-12)])
    A_bb = X.T @ (omega_sum_t[:,None]*X) + jnp.diag(Prec_beta_diag)
    diag_add = jnp.einsum('rtb,rt->b', V_rtb, omega_rt)                # (2B,)
    A_bb = A_bb.at[1:,1:].add(jnp.diag(diag_add))

    # A_bg = Xᵀ (ΣΩH),   A_gg = Σ(HᵀΩH) + Prec_γ
    A_bg = X.T @ H_agg_tl                                             # (P,L)
    A_gg = jnp.einsum('rtl,rt,rtm->lm', H_rtl, omega_rt, H_rtl) + Prec_gamma_ll  # (L,L)

    # b_b = Xᵀ(Σκ),   b_g = Σ(Hᵀ κ) + Prec_γ μ_γ
    b_b = X.T @ kappa_sum_t                                           # (P,)
    b_g = jnp.einsum('rtl,rt->l', H_rtl, kappa_rt) + Prec_gamma_ll @ mu_gamma_l  # (L,)

    # Solve θ = [β; γ]
    Prec = jnp.block([[A_bb, A_bg],[A_bg.T, A_gg]])
    rhs  = jnp.concatenate([b_b, b_g])
    Prec = _psd(Prec, 1e-8)
    Lc   = jnp.linalg.cholesky(Prec)
    v    = jax.scipy.linalg.solve_triangular(Lc, rhs, lower=True)
    mean = jax.scipy.linalg.solve_triangular(Lc.T, v, lower=False)
    eps  = jr.normal(key1, (P+L,), dtype=X.dtype)
    theta= mean + jax.scipy.linalg.solve_triangular(Lc.T, eps, lower=False)

    beta  = theta[:P]
    gamma = theta[P:]                                                 # (L,)

    # ARD on β_lat
    beta_lat = beta[1:]
    alpha = cfg.a0_ard + 0.5
    scale = 1.0/(cfg.b0_ard + 0.5*(beta_lat**2))
    tau2_new = 1.0 / jr.gamma(key2, alpha, shape=(twoB,)) * (1.0/scale)
    return beta, gamma, tau2_new

def gibbs_update_beta_trials_shared_gamma_jax(
    key: jr.KeyArray,
    X: jnp.ndarray,                     # (T,P), shared across trials
    H_S_rtl: jnp.ndarray,               # (S,R,T,L)
    spikes_S_rt: jnp.ndarray,           # (S,R,T)
    omega_S_rt: jnp.ndarray,            # (S,R,T)
    V_S_rtb: jnp.ndarray,               # (S,R,T,2B)  (tile shared V to this shape upstream)
    Prec_gamma_S_ll: jnp.ndarray,       # (S,L,L) or (L,L)
    mu_gamma_S_l: jnp.ndarray,          # (S,L)   or (L,)
    tau2_lat_S_b: jnp.ndarray,          # (S,2B)
    cfg: TrialBetaJAXConfig = TrialBetaJAXConfig(),
) -> Tuple[jr.KeyArray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Vectorized over units (S). Returns:
      key_next, beta_SP (S,P), gamma_SL (S,L), tau2_S2B (S,2B)
    """
    S = spikes_S_rt.shape[0]

    # broadcast γ priors if shared
    if Prec_gamma_S_ll.ndim == 2:
        Prec_gamma_S_ll = jnp.tile(Prec_gamma_S_ll[None, ...], (S,1,1))
    if mu_gamma_S_l.ndim == 1:
        mu_gamma_S_l = jnp.tile(mu_gamma_S_l[None, ...], (S,1))

    keys = jr.split(key, S)
    beta_S, gamma_S, tau2_S = vmap(_beta_gamma_shared_gamma_unit)(
        keys, X, H_S_rtl, spikes_S_rt, omega_S_rt, V_S_rtb,
        Prec_gamma_S_ll, mu_gamma_S_l, tau2_lat_S_b, cfg
    )
    return jr.fold_in(key, 1), beta_S, gamma_S, tau2_S
