# beta_sampler_trials_jax.py
# JAX-native β–γ PG–Gaussian sampler for trial-aware inference
# β shared across trials, γ shared across trials per unit
# Fully vectorized, JIT-compiled, device-resident

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap
from functools import partial


# ============================================================================
# Config (registered as JAX pytree for passing through transformations)
# ============================================================================

@dataclass
class TrialBetaJAXConfig:
    """Configuration for trial-aware β/γ sampler (JAX-compatible pytree)."""
    omega_floor: float = 1e-3
    tau2_intercept: float = 1e4
    a0_ard: float = 1e-2
    b0_ard: float = 1e-2

    def _tree_flatten(self):
        """Flatten config into (values, aux_data) for JAX pytree."""
        children = (self.omega_floor, self.tau2_intercept, self.a0_ard, self.b0_ard)
        aux_data = None
        return children, aux_data

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        """Reconstruct config from flattened representation."""
        return cls(*children)


# Register as pytree
jax.tree_util.register_pytree_node(
    TrialBetaJAXConfig,
    TrialBetaJAXConfig._tree_flatten,
    TrialBetaJAXConfig._tree_unflatten
)


# ============================================================================
# JAX utility functions (all JIT-compiled)
# ============================================================================

@jax.jit
def build_design_jax(latent_reim: jnp.ndarray) -> jnp.ndarray:
    """
    Prepend intercept column - pure JAX.
    latent_reim: (T, 2B)
    returns:     (T, P) where P = 1 + 2B
    """
    T = latent_reim.shape[0]
    return jnp.concatenate([jnp.ones((T, 1), dtype=latent_reim.dtype), latent_reim], axis=1)


# ============================================================================
# Single-unit β/γ sampler (shared across trials)
# ============================================================================

def _beta_gamma_shared_gamma_unit(
    key: jr.KeyArray,
    X: jnp.ndarray,           # (T, P) shared design across trials
    H_rtl: jnp.ndarray,       # (R, T, L) history design per trial
    spikes_rt: jnp.ndarray,   # (R, T) spikes
    omega_rt: jnp.ndarray,    # (R, T) PG weights
    V_rtb: jnp.ndarray,       # (R, T, 2B) latent variances per trial
    Prec_gamma_ll: jnp.ndarray,  # (L, L) gamma precision
    mu_gamma_l: jnp.ndarray,     # (L,) gamma prior mean
    tau2_lat_b: jnp.ndarray,     # (2B,) ARD variances for latent features
    omega_floor: float,
    tau2_intercept: float,
    a0_ard: float,
    b0_ard: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Gibbs update for single unit with β/γ shared across R trials.

    Args:
        key: PRNG key
        X: (T, P) design matrix with intercept
        H_rtl: (R, T, L) history design
        spikes_rt: (R, T) spike counts
        omega_rt: (R, T) PG auxiliary variables
        V_rtb: (R, T, 2B) latent posterior variances
        Prec_gamma_ll: (L, L) gamma prior precision
        mu_gamma_l: (L,) gamma prior mean
        tau2_lat_b: (2B,) current ARD variances
        omega_floor: minimum omega value
        tau2_intercept: intercept prior variance
        a0_ard: ARD prior shape
        b0_ard: ARD prior rate

    Returns:
        beta: (P,) shared β
        gamma: (L,) shared γ
        tau2_lat_new: (2B,) updated ARD variances
    """
    key1, key2 = jr.split(key)

    R, T, L = H_rtl.shape
    T_x, P = X.shape
    assert T_x == T, "Design matrix and spikes time dimension mismatch"
    twoB = P - 1

    # Floor omega
    omega_rt = jnp.maximum(omega_rt, omega_floor)

    # κ = spikes - 0.5
    kappa_rt = spikes_rt - 0.5  # (R, T)

    # Build prior precision for β
    Prec_beta_diag = jnp.zeros(P, dtype=X.dtype)
    Prec_beta_diag = Prec_beta_diag.at[0].set(1.0 / tau2_intercept)
    Prec_beta_diag = Prec_beta_diag.at[1:].set(1.0 / jnp.maximum(tau2_lat_b, 1e-12))

    # Accumulate normal equations across trials
    # β block: A_bb = sum_r X^T Ω_r X
    # γ block: A_gg = sum_r H_r^T Ω_r H_r + Prec_gamma
    # cross:   A_bg = sum_r X^T Ω_r H_r
    # RHS:     b_b  = sum_r X^T κ_r
    #          b_g  = sum_r H_r^T κ_r + Prec_gamma @ mu_gamma

    def accumulate_trial(r):
        """Accumulate normal equations for trial r."""
        omega_r = omega_rt[r]  # (T,)
        kappa_r = kappa_rt[r]  # (T,)
        H_r = H_rtl[r]         # (T, L)

        # Weighted matrices
        omega_sqrt = jnp.sqrt(omega_r)[:, None]
        Xw = omega_sqrt * X       # (T, P)
        Hw = omega_sqrt * H_r     # (T, L)

        # Normal equation blocks for this trial
        A_bb_r = Xw.T @ Xw        # (P, P)
        A_gg_r = Hw.T @ Hw        # (L, L)
        A_bg_r = Xw.T @ Hw        # (P, L)
        b_b_r = X.T @ kappa_r     # (P,)
        b_g_r = H_r.T @ kappa_r   # (L,)

        return A_bb_r, A_gg_r, A_bg_r, b_b_r, b_g_r

    # Vectorize over trials
    A_bb_trials, A_gg_trials, A_bg_trials, b_b_trials, b_g_trials = vmap(accumulate_trial)(jnp.arange(R))

    # Sum across trials
    A_bb = A_bb_trials.sum(axis=0)  # (P, P)
    A_gg = A_gg_trials.sum(axis=0)  # (L, L)
    A_bg = A_bg_trials.sum(axis=0)  # (P, L)
    b_b = b_b_trials.sum(axis=0)    # (P,)
    b_g = b_g_trials.sum(axis=0)    # (L,)

    # Add priors
    A_bb = A_bb + jnp.diag(Prec_beta_diag)
    A_gg = A_gg + Prec_gamma_ll
    b_g = b_g + Prec_gamma_ll @ mu_gamma_l

    # EIV correction for β latent features (sum over trials)
    # diag_add = sum_r V_r^T @ omega_r where V_r is (T, 2B)
    def compute_eiv_correction(r):
        V_r = V_rtb[r]        # (T, 2B)
        omega_r = omega_rt[r]  # (T,)
        return V_r.T @ omega_r  # (2B,)

    diag_add = vmap(compute_eiv_correction)(jnp.arange(R)).sum(axis=0)  # (2B,)
    A_bb = A_bb.at[1:, 1:].add(jnp.diag(diag_add))

    # Assemble full precision matrix: [β; γ]
    dim = P + L
    Prec = jnp.zeros((dim, dim), dtype=X.dtype)
    Prec = Prec.at[:P, :P].set(A_bb)
    Prec = Prec.at[:P, P:].set(A_bg)
    Prec = Prec.at[P:, :P].set(A_bg.T)
    Prec = Prec.at[P:, P:].set(A_gg)

    # RHS
    h = jnp.concatenate([b_b, b_g])

    # Symmetrize and regularize
    Prec = 0.5 * (Prec + Prec.T) + 1e-8 * jnp.eye(dim, dtype=Prec.dtype)

    # Cholesky solve for mean
    L_chol = jnp.linalg.cholesky(Prec)
    v = jax.scipy.linalg.solve_triangular(L_chol, h, lower=True)
    mean = jax.scipy.linalg.solve_triangular(L_chol.T, v, lower=False)

    # Sample from posterior
    eps = jr.normal(key1, shape=(dim,), dtype=X.dtype)
    theta = mean + jax.scipy.linalg.solve_triangular(L_chol.T, eps, lower=False)

    beta = theta[:P]
    gamma = theta[P:]

    # ARD update for latent features
    beta_lat = beta[1:]  # (2B,)
    alpha_post = a0_ard + 0.5
    beta_post = b0_ard + 0.5 * (beta_lat ** 2)
    tau2_lat_new = 1.0 / jr.gamma(key2, alpha_post, shape=(twoB,)) * beta_post

    return beta, gamma, tau2_lat_new


# JIT-compile the single-unit sampler with static config params
_beta_gamma_shared_gamma_unit_jit = jax.jit(
    _beta_gamma_shared_gamma_unit,
    static_argnames=('omega_floor', 'tau2_intercept', 'a0_ard', 'b0_ard')
)


# ============================================================================
# Vectorized sampler across units
# ============================================================================

def gibbs_update_beta_trials_shared_gamma_jax(
    key: jr.KeyArray,
    X: jnp.ndarray,            # (T, P) shared design
    H_S_rtl: jnp.ndarray,      # (S, R, T, L) history per unit/trial
    spikes_S_rt: jnp.ndarray,  # (S, R, T) spikes per unit/trial
    omega_S_rt: jnp.ndarray,   # (S, R, T) PG weights
    V_S_rtb: jnp.ndarray,      # (S, R, T, 2B) latent variances
    Prec_gamma_S_ll: jnp.ndarray,  # (S, L, L) gamma precision per unit
    mu_gamma_S_l: jnp.ndarray,     # (S, L) gamma prior mean per unit
    tau2_lat_S_b: jnp.ndarray,     # (S, 2B) ARD variances per unit
    cfg: TrialBetaJAXConfig,
) -> Tuple[jr.KeyArray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Vectorized Gibbs update across S units, each with β/γ shared across R trials.

    Args:
        key: PRNG key
        X: (T, P) shared design matrix (with intercept)
        H_S_rtl: (S, R, T, L) history design per unit
        spikes_S_rt: (S, R, T) spike data
        omega_S_rt: (S, R, T) PG auxiliary variables
        V_S_rtb: (S, R, T, 2B) latent variances
        Prec_gamma_S_ll: (S, L, L) gamma priors
        mu_gamma_S_l: (S, L) gamma prior means
        tau2_lat_S_b: (S, 2B) ARD variances
        cfg: TrialBetaJAXConfig

    Returns:
        key: updated PRNG key
        beta_S: (S, P) β coefficients
        gamma_S: (S, L) γ coefficients
        tau2_S: (S, 2B) updated ARD variances
    """
    S = H_S_rtl.shape[0]

    # Handle broadcasting for priors if needed
    if Prec_gamma_S_ll.ndim == 2:
        # (L, L) -> broadcast to (S, L, L)
        L = Prec_gamma_S_ll.shape[0]
        Prec_gamma_S_ll = jnp.tile(Prec_gamma_S_ll[None, ...], (S, 1, 1))

    if mu_gamma_S_l.ndim == 1:
        # (L,) -> broadcast to (S, L)
        L = mu_gamma_S_l.shape[0]
        mu_gamma_S_l = jnp.tile(mu_gamma_S_l[None, ...], (S, 1))

    # Split keys for each unit
    keys = jr.split(key, S)

    # Vectorize over units - extract config values as static args
    beta_S, gamma_S, tau2_S = vmap(
        lambda k, H_rtl, spk_rt, om_rt, V_rtb, Prec_g, mu_g, tau2_b:
            _beta_gamma_shared_gamma_unit_jit(
                k, X, H_rtl, spk_rt, om_rt, V_rtb, Prec_g, mu_g, tau2_b,
                cfg.omega_floor, cfg.tau2_intercept, cfg.a0_ard, cfg.b0_ard
            )
    )(keys, H_S_rtl, spikes_S_rt, omega_S_rt, V_S_rtb,
      Prec_gamma_S_ll, mu_gamma_S_l, tau2_lat_S_b)

    return jr.fold_in(key, 1), beta_S, gamma_S, tau2_S


# JIT-compile the main function
gibbs_update_beta_trials_shared_gamma_jax = jax.jit(gibbs_update_beta_trials_shared_gamma_jax)
