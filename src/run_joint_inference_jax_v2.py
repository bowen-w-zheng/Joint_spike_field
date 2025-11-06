"""
Properly optimized JAX-native joint inference implementation.

Key differences from original:
1. ALL hot-path functions are @jax.jit compiled
2. Uses jax.lax.scan for loops (not Python for loops)
3. Pure JAX - no numpy in hot paths
4. Vectorizes with vmap where possible
5. Minimal array conversions

This matches the pattern from the fast reference implementation.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap, lax
from functools import partial
from tqdm.auto import tqdm

from src.joint_inference_core import JointMoments
from src.params import OUParams
from src.priors import gamma_prior_simple
from src.polyagamma_jax import sample_pg_saddle_single
from src.utils_joint import Trace


# ============================================================================
# Pure JAX utility functions (all JIT-compiled)
# ============================================================================

@jax.jit
def _sample_omega_pg_batch(key, psi: jnp.ndarray, omega_floor: float) -> jnp.ndarray:
    """Fast batch Polyagamma sampler."""
    N = psi.shape[0]
    keys = jr.split(key, N)
    omega = jax.vmap(lambda k, z: sample_pg_saddle_single(k, 1.0, z))(keys, psi)
    return jnp.maximum(omega, omega_floor)


@jax.jit
def _build_design_jax(latent_reim: jnp.ndarray) -> jnp.ndarray:
    """Prepend intercept column - pure JAX."""
    T = latent_reim.shape[0]
    return jnp.concatenate([jnp.ones((T, 1)), latent_reim], axis=1)


@jax.jit
def _gibbs_update_beta_gamma_jax(
    key,
    beta: jnp.ndarray,      # (P,) where P = 1 + 2*B
    gamma: jnp.ndarray,     # (R,)
    tau2_lat: jnp.ndarray,  # (2*B,) ARD variances
    X: jnp.ndarray,         # (T, P) design
    H: jnp.ndarray,         # (T, R) history
    spikes: jnp.ndarray,    # (T,) binary
    V: jnp.ndarray,         # (T, 2*B) latent variances
    Prec_gamma: jnp.ndarray,  # (R, R) gamma precision
    mu_gamma: jnp.ndarray,    # (R,) gamma prior mean
    omega: jnp.ndarray,     # (T,) PG weights
    tau2_intercept: float,
    a0_ard: float,
    b0_ard: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Pure JAX Gibbs update for [beta, gamma] given omega.

    Returns:
        beta_new: (P,)
        gamma_new: (R,)
        tau2_lat_new: (2*B,)
    """
    key1, key2 = jr.split(key)

    T, P = X.shape
    R = H.shape[1]
    twoB = P - 1  # number of Re/Im coefficients

    # Build precision
    kappa = spikes - 0.5

    # Prior precision (diagonal)
    Prec_beta_diag = jnp.zeros(P)
    Prec_beta_diag = Prec_beta_diag.at[0].set(1.0 / tau2_intercept)
    Prec_beta_diag = Prec_beta_diag.at[1:].set(1.0 / jnp.maximum(tau2_lat, 1e-12))

    # Weighted design
    sqrt_omega = jnp.sqrt(omega)[:, None]
    Xw = sqrt_omega * X
    Hw = sqrt_omega * H

    # Build block precision matrix
    Prec_beta_block = Xw.T @ Xw + jnp.diag(Prec_beta_diag)

    # Add EIV correction
    diag_add = V.T @ omega  # (2*B,)
    Prec_beta_block = Prec_beta_block.at[1:, 1:].add(jnp.diag(diag_add))

    Prec_gamma_block = Hw.T @ Hw + Prec_gamma
    Prec_cross = Xw.T @ Hw

    # Assemble full precision
    Prec = jnp.zeros((P + R, P + R))
    Prec = Prec.at[:P, :P].set(Prec_beta_block)
    Prec = Prec.at[:P, P:].set(Prec_cross)
    Prec = Prec.at[P:, :P].set(Prec_cross.T)
    Prec = Prec.at[P:, P:].set(Prec_gamma_block)

    # Build RHS
    h_beta = X.T @ kappa
    h_gamma = H.T @ kappa + Prec_gamma @ mu_gamma
    h = jnp.concatenate([h_beta, h_gamma])

    # Regularize
    Prec = 0.5 * (Prec + Prec.T) + 1e-8 * jnp.eye(P + R)

    # Cholesky solve
    L = jnp.linalg.cholesky(Prec)
    v = jax.scipy.linalg.solve_triangular(L, h, lower=True)
    mean = jax.scipy.linalg.solve_triangular(L.T, v, lower=False)

    # Sample
    eps = jr.normal(key1, shape=(P + R,))
    theta = mean + jax.scipy.linalg.solve_triangular(L.T, eps, lower=False)

    beta_new = theta[:P]
    gamma_new = theta[P:]

    # ARD update for Re/Im coefficients
    beta_lat = beta_new[1:]  # (2*B,)
    alpha_post = a0_ard + 0.5
    beta_post = b0_ard + 0.5 * (beta_lat ** 2)
    # Inverse gamma sample
    tau2_lat_new = 1.0 / jr.gamma(key2, alpha_post, shape=(twoB,)) * beta_post

    return beta_new, gamma_new, tau2_lat_new


# Vectorize over spike trains
_gibbs_update_vectorized = vmap(
    _gibbs_update_beta_gamma_jax,
    in_axes=(0, 0, 0, 0, None, 0, 0, None, 0, 0, 0, None, None, None)
    # keys, betas, gammas, tau2s, X (shared), H_all, spikes_all, V (shared),
    # Prec_gammas, mu_gammas, omegas, tau2_int, a0, b0
)


@partial(jax.jit, static_argnames=['n_iter'])
def _warmup_loop_scan(
    key,
    beta_init: jnp.ndarray,     # (S, P)
    gamma_init: jnp.ndarray,    # (S, R)
    tau2_init: jnp.ndarray,     # (S, 2*B)
    X: jnp.ndarray,             # (T, P) shared design
    H_all: jnp.ndarray,         # (S, T, R) per-train history
    spikes_all: jnp.ndarray,    # (S, T) binary
    V: jnp.ndarray,             # (T, 2*B) shared latent variances
    Prec_gamma_all: jnp.ndarray,  # (S, R, R)
    mu_gamma_all: jnp.ndarray,    # (S, R)
    omega_floor: float,
    tau2_intercept: float,
    a0_ard: float,
    b0_ard: float,
    n_iter: int,
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Warmup phase using jax.lax.scan for speed.

    Returns:
        final_carry: (beta_final, gamma_final, tau2_final)
        history: (beta_history, gamma_history) with shapes (n_iter, S, P) and (n_iter, S, R)
    """
    S = beta_init.shape[0]

    def scan_fn(carry, key_iter):
        beta, gamma, tau2 = carry

        # Compute psi for all trains (vectorized)
        psi_all = vmap(lambda b, g, h: X @ b + h @ g)(beta, gamma, H_all)  # (S, T)

        # Sample omega for all trains
        keys_omega = jr.split(key_iter, S)
        omega_all = vmap(lambda k, psi: _sample_omega_pg_batch(k, psi, omega_floor))(
            keys_omega, psi_all
        )  # (S, T)

        # Gibbs update for all trains (vectorized)
        keys_gibbs = jr.split(jr.fold_in(key_iter, 1), S)
        beta_new, gamma_new, tau2_new = _gibbs_update_vectorized(
            keys_gibbs, beta, gamma, tau2, X, H_all, spikes_all, V,
            Prec_gamma_all, mu_gamma_all, omega_all, tau2_intercept, a0_ard, b0_ard
        )

        # Return new state AND save beta/gamma for trace
        return (beta_new, gamma_new, tau2_new), (beta_new, gamma_new)

    # Generate keys for all iterations
    keys = jr.split(key, n_iter)

    # Run scan
    init_carry = (beta_init, gamma_init, tau2_init)
    final_carry, history = lax.scan(scan_fn, init_carry, keys)

    return final_carry, history  # (beta, gamma, tau2), (beta_hist, gamma_hist)


@partial(jax.jit, static_argnames=['n_iter'])
def _inner_loop_scan(
    key,
    beta_init: jnp.ndarray,     # (S, P)
    gamma_init: jnp.ndarray,    # (S, R)
    tau2_init: jnp.ndarray,     # (S, 2*B)
    beta0_fixed: jnp.ndarray,   # (S,) frozen intercepts
    X: jnp.ndarray,             # (T, P) shared design
    H_all: jnp.ndarray,         # (S, T, R) per-train history
    spikes_all: jnp.ndarray,    # (S, T) binary
    V: jnp.ndarray,             # (T, 2*B) shared latent variances
    Prec_gamma_lock: jnp.ndarray,  # (S, R, R) tight lock
    mu_gamma_post: jnp.ndarray,    # (S, R) posterior mean
    Sigma_gamma_post: jnp.ndarray, # (S, R, R) posterior cov
    omega_floor: float,
    tau2_intercept: float,
    a0_ard: float,
    b0_ard: float,
    n_iter: int,
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Inner refresh loop using jax.lax.scan.

    Samples gamma from posterior, then does Gibbs update with tight lock.

    Returns:
        final_carry: (beta_final, gamma_final, tau2_final)
        history: (beta_history, gamma_history) with shapes (n_iter, S, P) and (n_iter, S, R)
    """
    S = beta_init.shape[0]
    R = gamma_init.shape[1]

    def scan_fn(carry, key_iter):
        beta, gamma, tau2 = carry

        # Sample gamma from posterior for all trains
        keys_gamma = jr.split(key_iter, S)

        def sample_mvn(key, mu, Sigma):
            """Sample from multivariate normal"""
            L = jnp.linalg.cholesky(Sigma + 1e-6 * jnp.eye(R))
            z = jr.normal(key, shape=(R,))
            return mu + L @ z

        gamma_samp = vmap(sample_mvn)(keys_gamma, mu_gamma_post, Sigma_gamma_post)  # (S, R)

        # Compute psi with sampled gamma
        psi_all = vmap(lambda b, g, h: X @ b + h @ g)(beta, gamma_samp, H_all)  # (S, T)

        # Sample omega
        keys_omega = jr.split(jr.fold_in(key_iter, 1), S)
        omega_all = vmap(lambda k, psi: _sample_omega_pg_batch(k, psi, omega_floor))(
            keys_omega, psi_all
        )

        # Gibbs update with tight lock on gamma
        keys_gibbs = jr.split(jr.fold_in(key_iter, 2), S)
        beta_new, _, tau2_new = _gibbs_update_vectorized(
            keys_gibbs, beta, gamma_samp, tau2, X, H_all, spikes_all, V,
            Prec_gamma_lock, gamma_samp, omega_all, tau2_intercept, a0_ard, b0_ard
        )

        # Freeze beta0
        beta_new = beta_new.at[:, 0].set(beta0_fixed)

        # Return new state AND save beta/gamma for trace
        return (beta_new, gamma_samp, tau2_new), (beta_new, gamma_samp)

    keys = jr.split(key, n_iter)
    init_carry = (beta_init, gamma_init, tau2_init)
    final_carry, history = lax.scan(scan_fn, init_carry, keys)

    return final_carry, history


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class InferenceConfig:
    fixed_iter: int = 100
    beta0_window: int = 100
    n_refreshes: int = 3
    inner_steps_per_refresh: int = 100
    omega_floor: float = 1e-3
    sigma_u: float = 0.05
    pg_jax: bool = True


# ============================================================================
# Main function
# ============================================================================

def run_joint_inference_jax_v2(
    Y_cube_block: np.ndarray,
    params0: OUParams,
    spikes: np.ndarray,
    H_hist: np.ndarray,
    all_freqs: np.ndarray,
    build_design: Callable,
    extract_band_reim_with_var: Callable,
    gibbs_update_beta_robust: Callable,  # Not used
    joint_kf_rts_moments: Callable,
    em_theta_from_joint: Callable,
    config: Optional[InferenceConfig] = None,
    *,
    delta_spk: float,
    window_sec: float,
    rng_pg: np.random.Generator = np.random.default_rng(0),
    key_jax = None,
    offset_sec: float = 0.0,
):
    """
    Properly JAX-optimized inference using jax.lax.scan and JIT compilation.
    """

    if key_jax is None:
        with jax.default_device(jax.devices("cpu")[0]):
            key_jax = jr.PRNGKey(0)

    if config is None:
        config = InferenceConfig()

    sigma_u = getattr(config, 'sigma_u', 0.05)

    print("[JAX-V2] Starting properly optimized JAX inference...")
    print(f"[JAX-V2] Will use jax.lax.scan for {config.fixed_iter} warmup iterations")

    # ===== Preprocessing (same as original) =====
    J, M, K = Y_cube_block.shape
    single_spike_mode = (spikes.ndim == 1)

    if single_spike_mode:
        spikes_S = spikes[None, :]
        H_hist_S = H_hist[None, :, :]
    else:
        spikes_S = spikes
        H_hist_S = H_hist

    S, T_total = spikes_S.shape
    R = H_hist_S.shape[2]

    # LFP initialization
    theta = OUParams(
        lam=params0.lam,
        sig_v=params0.sig_v,
        sig_eps=np.broadcast_to(params0.sig_eps, (J, M))
    )

    from src.ou_fine import kalman_filter_rts_ffbs_fine
    fine0 = kalman_filter_rts_ffbs_fine(
        Y_cube_block, theta, delta_spk=delta_spk, win_sec=window_sec, offset_sec=offset_sec
    )

    lat_reim_np, var_reim_np = extract_band_reim_with_var(
        mu_fine=np.asarray(fine0.mu)[:-1],
        var_fine=np.asarray(fine0.var)[:-1],
        coupled_bands=all_freqs, freqs_hz=all_freqs, delta_spk=delta_spk, J=J, M=M
    )

    lat_reim_jax = jnp.asarray(lat_reim_np)
    design_np = np.asarray(build_design(lat_reim_jax))
    T_design = min(int(design_np.shape[0]), H_hist_S.shape[1], spikes_S.shape[1])

    # ===== Convert to JAX ONCE =====
    print(f"[JAX-V2] Converting data to JAX (one time)...")
    X_jax = jnp.array(design_np[:T_design], dtype=jnp.float64)
    V_jax = jnp.array(var_reim_np[:T_design], dtype=jnp.float64)
    lat_slice = lat_reim_jax[:T_design]

    spikes_jax = jnp.array(spikes_S[:, :T_design], dtype=jnp.float64)
    H_jax = jnp.array(H_hist_S[:, :T_design, :], dtype=jnp.float64)

    B = len(all_freqs)
    P = 1 + 2*B

    beta_jax = jnp.zeros((S, P), dtype=jnp.float64)
    gamma_jax = jnp.zeros((S, R), dtype=jnp.float64)
    tau2_lat_jax = jnp.ones((S, 2*B), dtype=jnp.float64)

    a0_ard, b0_ard = 1e-2, 1e-2
    mu_g, Sig_g = gamma_prior_simple(n_lags=R, strong_neg=-2.5, mild_neg=-0.5, k_mild=4, tau_gamma=1.5)

    # Convert priors to JAX
    Prec_gamma_all = jnp.array([np.linalg.pinv(Sig_g) for _ in range(S)])
    mu_gamma_all = jnp.array([mu_g for _ in range(S)])

    print(f"[JAX-V2] Data: S={S}, T={T_design}, B={B}, R={R}, P={P}")

    # ===== Initialize trace =====
    trace = Trace()
    trace.theta.append(theta)
    trace.latent.append(lat_reim_jax)
    trace.fine_latent.append(np.asarray(fine0.mu))

    # ===== WARMUP with jax.lax.scan =====
    print(f"[JAX-V2] Running warmup with jax.lax.scan (JIT-compiled)...")
    import time
    t0 = time.time()

    key_warmup, key_jax = jr.split(key_jax)
    (beta_jax, gamma_jax, tau2_lat_jax), (beta_history, gamma_history) = _warmup_loop_scan(
        key_warmup,
        beta_jax,
        gamma_jax,
        tau2_lat_jax,
        X_jax,
        H_jax,
        spikes_jax,
        V_jax,
        Prec_gamma_all,
        mu_gamma_all,
        config.omega_floor,
        100.0**2,  # tau2_intercept
        a0_ard,
        b0_ard,
        config.fixed_iter,
    )

    # Force computation
    beta_jax = beta_jax.block_until_ready()

    print(f"[JAX-V2] Warmup completed in {time.time()-t0:.2f}s")
    print(f"[JAX-V2] That's {(time.time()-t0)/config.fixed_iter*1000:.1f}ms per iteration")

    # Convert back to compute posteriors
    beta_np = np.array(beta_jax)
    gamma_np = np.array(gamma_jax)
    tau2_lat_np = np.array(tau2_lat_jax)
    beta_hist_np = np.array(beta_history)  # (n_iter, S, P)
    gamma_hist_np = np.array(gamma_history)  # (n_iter, S, R)

    print("[JAX-V2] Computing beta0_fixed and gamma posteriors...")

    # CRITICAL FIX: Freeze β0 per train using median from warmup history
    # Use last beta0_window iterations (or all if less)
    beta0_window = min(config.beta0_window, config.fixed_iter)
    beta0_recent = beta_hist_np[-beta0_window:, :, 0]  # (window, S)
    beta0_fixed = np.median(beta0_recent, axis=0)  # (S,)

    print(f"[JAX-V2]   Using median of last {beta0_window} iterations for beta0_fixed")
    beta_np[:, 0] = beta0_fixed
    beta_jax = jnp.array(beta_np)

    # CRITICAL FIX: Compute gamma posteriors from warmup history (matching original)
    mu_g_post = np.zeros((S, R), dtype=np.float64)
    Sig_g_post = np.zeros((S, R, R), dtype=np.float64)
    Sig_g_lock = np.zeros((S, R, R), dtype=np.float64)

    for s in range(S):
        # Get all gamma samples for this train
        gh = gamma_hist_np[:, s, :]  # (n_iter, R)
        mu_s = gh.mean(axis=0)  # (R,)
        ctr = gh - mu_s[None, :]  # (n_iter, R)
        Sg = (ctr.T @ ctr) / max(gh.shape[0] - 1, 1)  # (R, R)
        Sg = Sg + 1e-6 * np.eye(R)

        mu_g_post[s] = mu_s
        Sig_g_post[s] = Sg
        diag_scale = np.clip(np.diag(Sg), 1e-10, None)
        Sig_g_lock[s] = np.diag(1e-6 * diag_scale)

    print(f"[JAX-V2]   Computed gamma posteriors from {config.fixed_iter} warmup samples")

    # Add to trace (for compatibility)
    for i in range(config.fixed_iter):
        trace.beta.append(beta_hist_np[i])
        trace.gamma.append(gamma_hist_np[i])

    # Convert to JAX
    beta0_fixed_jax = jnp.array(beta0_fixed)
    mu_gamma_post_jax = jnp.array(mu_g_post)
    Sigma_gamma_post_jax = jnp.array(Sig_g_post)
    Prec_gamma_lock_jax = jnp.array([np.linalg.pinv(Sig_g_lock[s]) for s in range(S)])

    # ===== REFRESH PASSES =====
    from src.state_index import StateIndex
    sidx = StateIndex(J, M)

    print(f"[JAX-V2] Starting {config.n_refreshes} refresh passes...")

    for r in range(config.n_refreshes):
        print(f"[JAX-V2] Refresh {r+1}/{config.n_refreshes}...")

        # ===== Inner loop with jax.lax.scan =====
        print(f"[JAX-V2]   Running {config.inner_steps_per_refresh} inner steps with scan...")
        t_inner = jax.random.PRNGKey(r + 1000)  # Deterministic key per refresh

        key_inner, key_jax = jr.split(key_jax)
        (beta_jax, gamma_jax, tau2_lat_jax), (beta_history, gamma_history) = _inner_loop_scan(
            key_inner,
            beta_jax,
            gamma_jax,
            tau2_lat_jax,
            beta0_fixed_jax,
            X_jax,
            H_jax,
            spikes_jax,
            V_jax,
            Prec_gamma_lock_jax,
            mu_gamma_post_jax,
            Sigma_gamma_post_jax,
            config.omega_floor,
            100.0**2,
            a0_ard,
            b0_ard,
            config.inner_steps_per_refresh,
        )
        beta_jax = beta_jax.block_until_ready()

        # Convert for refresh step
        beta_np = np.array(beta_jax)
        gamma_np = np.array(gamma_jax)

        # CRITICAL FIX: Compute median across inner iterations (matching original)
        print(f"[JAX-V2]   Computing beta median across {config.inner_steps_per_refresh} iterations...")
        beta_median = np.median(np.array(beta_history), axis=0)  # (S, 1+2J)

        # Add to trace (for compatibility with original)
        for i in range(config.inner_steps_per_refresh):
            trace.beta.append(np.array(beta_history[i]))
            trace.gamma.append(np.array(gamma_history[i]))

        # Build omega for refresh (vectorized in JAX)
        print(f"[JAX-V2]   Building omega for refresh...")
        key_omega_refresh, key_jax = jr.split(key_jax)

        # Sample gamma for refresh
        keys_gamma_refresh = jr.split(key_omega_refresh, S)
        def sample_mvn(key, mu, Sigma):
            L = jnp.linalg.cholesky(Sigma + 1e-6 * jnp.eye(R))
            z = jr.normal(key, shape=(R,))
            return mu + L @ z
        gamma_refresh_jax = vmap(sample_mvn)(keys_gamma_refresh, mu_gamma_post_jax, Sigma_gamma_post_jax)

        # Compute psi and sample omega
        psi_refresh_all = vmap(lambda b, g, h: X_jax @ b + h @ g)(
            jnp.array(beta_median), gamma_refresh_jax, H_jax
        )
        keys_omega_all = jr.split(jr.fold_in(key_omega_refresh, 1), S)
        omega_refresh_jax = vmap(lambda k, psi: _sample_omega_pg_batch(k, psi, config.omega_floor))(
            keys_omega_all, psi_refresh_all
        )

        omega_refresh = np.array(omega_refresh_jax)
        gamma_np = np.array(gamma_refresh_jax)

        # ===== Latent refresh (numpy-based, unavoidable) =====
        print(f"[JAX-V2]   Running latent refresh (KF)...")
        mom = joint_kf_rts_moments(
            Y_cube=Y_cube_block, theta=theta,
            delta_spk=delta_spk, win_sec=window_sec, offset_sec=offset_sec,
            beta=beta_median,
            gamma=gamma_np,
            spikes=spikes_S,
            omega=omega_refresh,
            coupled_bands_idx=np.arange(J, dtype=np.int64),
            freqs_for_phase=np.asarray(all_freqs, np.float64),
            sidx=sidx, H_hist=H_hist_S,
            sigma_u=sigma_u
        )

        # Rebuild regressors
        lat_reim_np, var_reim_np = extract_band_reim_with_var(
            mu_fine=mom.m_s, var_fine=mom.P_s,
            coupled_bands=all_freqs, freqs_hz=all_freqs, delta_spk=delta_spk, J=J, M=M
        )

        # Update JAX arrays for next refresh
        lat_reim_jax = jnp.asarray(lat_reim_np)
        design_np = np.asarray(build_design(lat_reim_jax))
        T_design_new = min(int(design_np.shape[0]), H_hist_S.shape[1], spikes_S.shape[1])

        if T_design_new != T_design:
            print(f"[JAX-V2]   Warning: T_design changed from {T_design} to {T_design_new}")
            # Would need to recompile - skip for now
            T_design_new = T_design

        X_jax = jnp.array(np.ascontiguousarray(design_np[:T_design], dtype=np.float64))
        V_jax = jnp.array(np.ascontiguousarray(var_reim_np[:T_design], dtype=np.float64))
        lat_slice = lat_reim_jax[:T_design]

        beta_jax = jnp.array(beta_np)
        gamma_jax = jnp.array(gamma_np)

        # Update trace
        trace.theta.append(theta)
        trace.latent.append(lat_reim_jax)
        trace.fine_latent.append(mom.m_s)

    # Final outputs
    beta_final = np.array(beta_jax)
    gamma_final = np.array(gamma_jax)

    print("[JAX-V2] Inference complete!")

    if single_spike_mode:
        return beta_final[0], gamma_final[0], theta, trace
    else:
        return beta_final, gamma_final, theta, trace


# Alias
run_joint_inference = run_joint_inference_jax_v2


if __name__ == "__main__":
    print("JAX-native inference v2 (with proper JIT and scan)")
    print(f"JAX devices: {jax.devices()}")