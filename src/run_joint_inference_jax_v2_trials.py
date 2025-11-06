"""
Joint inference for trial-structured data using shared-trajectory (precision-pooling) approach.

This module implements the algorithm described in section A of the plan:
- Shared latent trajectory X across trials
- Per-(unit, trial) coupling coefficients β and γ
- Precision-pooled observations for Kalman filter updates
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Tuple
import numpy as np
import jax.numpy as jnp
import jax.random as jr
from tqdm.auto import tqdm

from src.joint_inference_core import JointMoments
from src.params import OUParams
from src.priors import gamma_prior_simple
from src.pg_utils import sample_polya_gamma
from src.polyagamma_jax import sample_pg_batch
from src.utils_joint import Trace


# ─────────────────────────────────────────────────────────────────────────────
# A.1 Pre-Kalman pooling utilities
# ─────────────────────────────────────────────────────────────────────────────

def pool_lfp_trials(Y_rm: np.ndarray, sig_eps_mr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Precision-pool LFP observations across trials (eq. 11 in plan).

    Parameters
    ----------
    Y_rm : (R, M) complex multitaper coefficients across trials
    sig_eps_mr : (R, M) noise variances for those rows

    Returns
    -------
    Y_tilde_m : (M,) complex pooled coefficients
    sig_tilde_m : (M,) pooled variances
    """
    w = 1.0 / np.asarray(sig_eps_mr, float)       # (R, M) precision weights
    Y_tilde = (w * Y_rm).sum(axis=0) / w.sum(axis=0)
    sig_tilde = 1.0 / w.sum(axis=0)
    return Y_tilde, sig_tilde


def pool_spike_pseudo(omega_nr: np.ndarray, kappa_nr: np.ndarray) -> Tuple[float, float]:
    """
    Pool spike pseudo-observations across trials (eq. 12 in plan).

    Parameters
    ----------
    omega_nr : (R,) PG weights ω_{n,r} across trials
    kappa_nr : (R,) κ_{n,r} = N_{n,r} - 1/2 across trials

    Returns
    -------
    y_tilde_n : scalar pooled pseudo-observation
    R_tilde_n : scalar pooled noise variance
    """
    wsum = np.asarray(omega_nr, float).sum()
    y_tilde = np.asarray(kappa_nr, float).sum() / wsum
    R_tilde = 1.0 / wsum
    return y_tilde, R_tilde


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class InferenceConfigTrials:
    """Configuration for trial-structured joint inference."""
    # Warm-up before latent refresh; freeze β0 after warm-up
    fixed_iter: int = 100
    beta0_window: int = 100

    # Latent refresh passes
    n_refreshes: int = 3
    inner_steps_per_refresh: int = 100

    # Numerical stability
    omega_floor: float = 1e-3
    sigma_u: float = 0.05

    # Sampler choice
    pg_jax: bool = False

    # Trial pooling mode
    mode: str = "shared"  # "shared" (precision-pooling) or "hier" (X + δr, future)


# ─────────────────────────────────────────────────────────────────────────────
# Main inference function
# ─────────────────────────────────────────────────────────────────────────────

def run_joint_inference_trials(
    Y_cube_block: np.ndarray,            # (R, J, M, K) trial-structured complex TFR
    params0: "OUParams",                 # initial OU params (LFP-only warm start)
    spikes: np.ndarray,                  # (R, S, T_f) OR (S, R, T_f) trials × units × time
    H_hist: np.ndarray,                  # (S, R, T_f, Rlags) history per unit/trial
    all_freqs: np.ndarray,               # (J,) band frequencies (Hz)
    build_design: Callable,              # X = [1, ReZ̃..., ImZ̃...] from latent_reim
    extract_band_reim_with_var: Callable[..., Tuple[np.ndarray, np.ndarray]],
    gibbs_update_beta_robust: Callable[..., Tuple["jax.random.KeyArray",
                                                  jnp.ndarray,
                                                  Optional[jnp.ndarray],
                                                  np.ndarray]],
    joint_kf_rts_moments: Callable[..., "JointMoments"],
    config: "InferenceConfigTrials" = None,
    *,
    delta_spk: float,
    window_sec: float,
    offset_sec: float = 0.0,
    rng_pg: np.random.Generator = np.random.default_rng(0),
    key_jax: "jr.KeyArray | None" = None,
) -> Tuple[np.ndarray, np.ndarray, "OUParams", Trace]:
    """
    Joint inference for trial-structured data with shared latent trajectory.

    This implements the precision-pooling approach (section A) where:
    - Latent spectral trajectory X is shared across trials
    - Coupling coefficients β and γ are per (unit, trial)
    - LFP and spike observations are precision-pooled for Kalman updates

    Parameters
    ----------
    Y_cube_block : (R, J, M, K)
        Complex TFR per trial, frequency band, taper, and time block
    spikes : (R, S, T_f) or (S, R, T_f)
        Spike trains per trial and unit
    H_hist : (S, R, T_f, Rlags)
        History features per unit, trial, time, and lag

    Returns
    -------
    beta : (S, R, 1+2J) coupling coefficients per unit and trial
    gamma : (S, R, Rlags) history coefficients per unit and trial
    theta : OUParams with updated OU parameters (shared across trials)
    trace : Trace object with sampling history
    """
    if key_jax is None:
        import jax
        with jax.default_device(jax.devices("cpu")[0]):
            key_jax = jr.PRNGKey(0)

    if config is None:
        config = InferenceConfigTrials()

    # ── Shapes & normalization ──────────────────────────────────────────────
    # Normalize spikes to (R, S, T_f)
    spikes = np.asarray(spikes)
    if spikes.ndim != 3:
        raise ValueError(f"spikes must be 3D: (R,S,T) or (S,R,T), got shape {spikes.shape}")

    # Check which convention: (R,S,T) or (S,R,T)
    if spikes.shape[0] < spikes.shape[1]:  # likely (R,S,T) already
        R, S, T_total = spikes.shape
    else:  # likely (S,R,T) -> transpose
        spikes = np.transpose(spikes, (1, 0, 2))
        R, S, T_total = spikes.shape

    # Normalize Y_cube_block to (R, J, M, K)
    Y_cube_block = np.asarray(Y_cube_block, np.complex128)
    if Y_cube_block.ndim != 4:
        raise ValueError(f"Y_cube_block must be 4D, got shape {Y_cube_block.shape}")

    # Infer J, M, K from Y_cube_block
    if Y_cube_block.shape[0] == R:  # (R, J, M, K)
        _, J, M, K = Y_cube_block.shape
    else:
        raise ValueError(f"Y_cube_block first dim must match R={R}, got {Y_cube_block.shape}")

    # Normalize H_hist to (S, R, T, Rlags)
    H_hist = np.asarray(H_hist)
    if H_hist.ndim != 4:
        raise ValueError(f"H_hist must be 4D: (S,R,T,Rlags), got {H_hist.shape}")
    assert H_hist.shape[0] == S and H_hist.shape[1] == R
    Rlags = H_hist.shape[3]

    # Initialize JAX PG sampler if requested
    key_pg_jax = None
    if config.pg_jax:
        import jax
        with jax.default_device(jax.devices("cpu")[0]):
            key_pg_jax = jr.PRNGKey(42)

    def sample_pg_wrapper(psi: np.ndarray) -> np.ndarray:
        """Sample from Polya-Gamma distribution using either numpy or JAX backend."""
        nonlocal key_pg_jax
        if config.pg_jax:
            key_pg_jax, subkey = jr.split(key_pg_jax)
            psi_jax = jnp.asarray(psi)
            samples = sample_pg_batch(subkey, psi_jax, h=1.0)
            return np.asarray(samples)
        else:
            return sample_polya_gamma(np.asarray(psi), rng_pg)

    # ── θ from LFP-only warm start (shared across trials) ───────────────────
    # Pool trials for initial LFP-only smoothing
    Y_pooled = np.zeros((J, M, K), dtype=np.complex128)
    sig_eps_pooled = np.zeros((J, M), dtype=np.float64)

    # Simple pooling: average across trials (can be improved with precision weighting)
    for j in range(J):
        for k in range(K):
            Y_rm = Y_cube_block[:, j, :, k]  # (R, M)
            sig_rm = np.ones((R, M)) * params0.sig_eps  # uniform noise initially
            Y_pooled[j, :, k], sig_pooled_m = pool_lfp_trials(Y_rm, sig_rm)
            if k == 0:  # store once per band
                sig_eps_pooled[j, :] = sig_pooled_m

    theta = OUParams(
        lam=params0.lam,
        sig_v=params0.sig_v,
        sig_eps=sig_eps_pooled
    )

    # ── LFP-only fine smoother → regressors (+ variances) ───────────────────
    from src.ou_fine import kalman_filter_rts_ffbs_fine
    fine0 = kalman_filter_rts_ffbs_fine(
        Y_pooled, theta, delta_spk=delta_spk, win_sec=window_sec, offset_sec=offset_sec
    )

    # Extract band-averaged predictors with variances (aligned to spike grid)
    lat_reim_np, var_reim_np = extract_band_reim_with_var(
        mu_fine=np.asarray(fine0.mu)[:-1],      # (T_f, d)
        var_fine=np.asarray(fine0.var)[:-1],
        coupled_bands=all_freqs, freqs_hz=all_freqs, delta_spk=delta_spk, J=J, M=M
    )
    lat_reim_jax = jnp.asarray(lat_reim_np)
    var_reim_np_current = var_reim_np
    design_np = np.asarray(build_design(lat_reim_jax))      # (T_f, 1+2J)
    T_design = min(int(design_np.shape[0]), H_hist.shape[2], T_total)

    # ── A.3 Vectorize β/γ over (unit, trial) ────────────────────────────────
    # Flatten (S, R) into S' = S*R "pseudo-units"
    Sprime = S * R
    B = len(all_freqs)
    P = 1 + 2*B  # design dimension

    # Reshape spikes and history: (R, S, T) -> (S', T)
    spikes_SR = spikes.transpose(1, 0, 2).reshape(Sprime, T_total)     # (S', T)
    H_SR = H_hist.transpose(0, 1, 2, 3).reshape(Sprime, T_total, Rlags)  # (S', T, Rlags)

    # Stable slices (for Gibbs cache hits)
    X_slice = np.ascontiguousarray(design_np[:T_design], dtype=np.float64)      # (T, 1+2J)
    V_slice = np.ascontiguousarray(var_reim_np_current[:T_design], dtype=np.float64)  # (T, 2J)
    lat_slice = lat_reim_jax[:T_design]
    spikes_slice = [jnp.asarray(spikes_SR[sp, :T_design]) for sp in range(Sprime)]
    H_slice = [np.asarray(H_SR[sp, :T_design]) for sp in range(Sprime)]

    # ── Priors & per-pseudo-unit init for β/γ/ARD ───────────────────────────
    beta_SR = np.zeros((Sprime, P), dtype=np.float64)
    gamma_SR = np.zeros((Sprime, Rlags), dtype=np.float64)
    a0_ard, b0_ard = 1e-2, 1e-2
    tau2_lat_SR = np.ones((Sprime, 2*B), dtype=np.float64)

    # Gamma prior (shared structure, broadcast to all pseudo-units)
    mu_g, Sig_g = gamma_prior_simple(n_lags=Rlags, strong_neg=-2.5, mild_neg=-0.5, k_mild=4, tau_gamma=1.5)

    # Initial ω per pseudo-unit
    omega_SR = np.empty((Sprime, T_design), dtype=np.float64)
    for sp in range(Sprime):
        psi0 = X_slice @ beta_SR[sp] + H_slice[sp] @ gamma_SR[sp]
        omega_SR[sp] = np.maximum(sample_pg_wrapper(np.asarray(psi0)), config.omega_floor)

    # ── Trace bookkeeping ────────────────────────────────────────────────────
    trace = Trace()
    trace.theta.append(theta)
    trace.latent.append(lat_reim_jax)
    trace.fine_latent.append(np.asarray(fine0.mu))

    # ====================== (1) WARM-UP: per-pseudo-unit β/γ ================
    print(f"[Trial inference] R={R} trials, S={S} units, S'={Sprime} pseudo-units")
    print(f"[Trial inference] T_design={T_design}, J={J} bands, M={M} tapers")

    beta0_history_SR = [[] for _ in range(Sprime)]
    gamma_hist_SR = [[] for _ in range(Sprime)]

    pbar_warm = tqdm(range(config.fixed_iter), unit="it", desc="Warm-up (β/γ per pseudo-unit)", mininterval=0.3)
    for _ in pbar_warm:
        for sp in range(Sprime):
            psi = X_slice @ beta_SR[sp] + H_slice[sp] @ gamma_SR[sp]
            omega = np.maximum(sample_pg_wrapper(np.asarray(psi)), config.omega_floor)

            key_jax, b_new, g_new, t2_new = gibbs_update_beta_robust(
                key_jax,
                lat_slice,
                spikes_slice[sp],
                jnp.asarray(omega),
                H_hist=H_slice[sp],
                Sigma_gamma=Sig_g, mu_gamma=mu_g,
                var_latent_reim=V_slice,
                a0_ard=a0_ard, b0_ard=b0_ard, tau2_lat=tau2_lat_SR[sp],
                tau2_intercept=100.0**2, tau2_gamma=25.0**2,
                omega_floor=config.omega_floor
            )
            beta_SR[sp] = np.asarray(b_new)
            gamma_SR[sp] = np.asarray(g_new)
            tau2_lat_SR[sp] = t2_new

            beta0_history_SR[sp].append(float(beta_SR[sp, 0]))
            if len(beta0_history_SR[sp]) > config.beta0_window:
                beta0_history_SR[sp].pop(0)
            gamma_hist_SR[sp].append(gamma_SR[sp].copy())

        # Store (reshape back to (S, R, P) for bookkeeping)
        trace.beta.append(beta_SR.reshape(S, R, P).copy())
        trace.gamma.append(gamma_SR.reshape(S, R, Rlags).copy())

    # Freeze β0 per pseudo-unit (robust median)
    beta0_fixed_SR = np.array([
        np.median(h) if len(h) else beta_SR[sp, 0]
        for sp, h in enumerate(beta0_history_SR)
    ], dtype=np.float64)
    beta_SR[:, 0] = beta0_fixed_SR

    # γ posterior per pseudo-unit (mean/cov + tight locks)
    mu_g_post_SR = np.zeros((Sprime, Rlags), dtype=np.float64)
    Sig_g_post_SR = np.zeros((Sprime, Rlags, Rlags), dtype=np.float64)
    Sig_g_lock_SR = np.zeros((Sprime, Rlags, Rlags), dtype=np.float64)

    for sp in range(Sprime):
        gh = np.stack(gamma_hist_SR[sp], axis=0) if len(gamma_hist_SR[sp]) else np.zeros((1, Rlags))
        mu_s = gh.mean(axis=0)
        ctr = gh - mu_s[None, :]
        Sg = (ctr.T @ ctr) / max(gh.shape[0]-1, 1)
        Sg = Sg + 1e-6 * np.eye(Rlags)
        mu_g_post_SR[sp] = mu_s
        Sig_g_post_SR[sp] = Sg
        diag_scale = np.clip(np.diag(Sg), 1e-10, None)
        Sig_g_lock_SR[sp] = np.diag(1e-6 * diag_scale)

    # ====================== (2) REFRESH PASSES ===============================
    from src.state_index import StateIndex
    sidx = StateIndex(J, M)
    sigma_u = config.sigma_u

    pbar_pass = tqdm(range(config.n_refreshes), unit="pass", desc="Latent refresh passes", mininterval=0.3)
    for refresh_idx in pbar_pass:
        # ---- inner PG steps per pseudo-unit (latents fixed) ----
        for _ in range(config.inner_steps_per_refresh):
            for sp in range(Sprime):
                gamma_samp = rng_pg.multivariate_normal(mean=mu_g_post_SR[sp], cov=Sig_g_post_SR[sp])
                psi = X_slice @ beta_SR[sp] + H_slice[sp] @ gamma_samp
                omega = np.maximum(sample_pg_wrapper(np.asarray(psi)), config.omega_floor)

                key_jax, b_new, g_new, t2_new = gibbs_update_beta_robust(
                    key_jax,
                    lat_slice,
                    spikes_slice[sp],
                    jnp.asarray(omega),
                    H_hist=H_slice[sp],
                    Sigma_gamma=Sig_g_lock_SR[sp],
                    mu_gamma=gamma_samp,
                    var_latent_reim=V_slice,
                    a0_ard=a0_ard, b0_ard=b0_ard, tau2_lat=tau2_lat_SR[sp],
                    tau2_intercept=100.0**2, tau2_gamma=25.0**2,
                    omega_floor=config.omega_floor
                )
                beta_SR[sp] = np.asarray(b_new)
                beta_SR[sp, 0] = beta0_fixed_SR[sp]  # enforce freeze
                tau2_lat_SR[sp] = t2_new
                gamma_SR[sp] = gamma_samp

            trace.beta.append(beta_SR.reshape(S, R, P).copy())
            trace.gamma.append(gamma_SR.reshape(S, R, Rlags).copy())

        # ---- robust β for refresh (median across last inner block) ----
        recent = np.stack([trace.beta[-i] for i in range(1, config.inner_steps_per_refresh + 1)], axis=0)
        beta_median_SR = np.median(recent, axis=0).reshape(Sprime, P)  # (S', P)

        # ---- build ω for refresh per pseudo-unit ----
        omega_refresh_SR = np.empty((Sprime, T_design), dtype=np.float64)
        gamma_refresh_SR = np.empty((Sprime, Rlags), dtype=np.float64)
        for sp in range(Sprime):
            gamma_samp = rng_pg.multivariate_normal(mean=mu_g_post_SR[sp], cov=Sig_g_post_SR[sp])
            psi_refresh = X_slice @ beta_median_SR[sp] + H_slice[sp] @ gamma_samp
            omega_refresh_SR[sp] = np.maximum(sample_pg_wrapper(np.asarray(psi_refresh)), config.omega_floor)
            gamma_refresh_SR[sp] = gamma_samp

        # ---- A.4 & A.5: Pool observations for LATENT REFRESH ----
        # For the Kalman filter, we need:
        # (1) Pooled LFP observations across trials
        # (2) Pooled spike pseudo-observations across trials (per unit, per time)

        # (1) Pool LFP: for each (j, k), pool across trials
        Y_pooled_refresh = np.zeros((J, M, K), dtype=np.complex128)
        sig_pooled_refresh = np.zeros((J, M), dtype=np.float64)

        for j in range(J):
            for k in range(K):
                Y_rm = Y_cube_block[:, j, :, k]  # (R, M)
                sig_rm = np.broadcast_to(theta.sig_eps[j, :], (R, M))  # (R, M)
                Y_pooled_refresh[j, :, k], sig_m = pool_lfp_trials(Y_rm, sig_rm)
                if k == 0:
                    sig_pooled_refresh[j, :] = sig_m

        # (2) Pool spike pseudo-obs: reshape back to (S, R, T) and pool per (unit, time)
        omega_SRT = omega_refresh_SR.reshape(S, R, T_design)  # (S, R, T)
        gamma_SRT = gamma_refresh_SR.reshape(S, R, Rlags)     # (S, R, Rlags)
        spikes_SRT = spikes_SR.reshape(S, R, T_total)[:, :, :T_design]  # (S, R, T)
        kappa_SRT = spikes_SRT.astype(np.float64) - 0.5      # (S, R, T)

        # Build single pooled spike pseudo-row per unit (average across units for shared latent)
        # NOTE: In the shared-trajectory model, we pool ACROSS TRIALS to get a single pseudo-row per time
        # But we have S units, each with R trials. We need to pool trials within each unit first,
        # then average/sum across units to get the final pseudo-rows for the Kalman filter.

        # Pool within each unit across trials
        omega_pooled_ST = np.zeros((S, T_design), dtype=np.float64)
        y_spk_pooled_ST = np.zeros((S, T_design), dtype=np.float64)
        R_spk_pooled_ST = np.zeros((S, T_design), dtype=np.float64)

        for s in range(S):
            for t in range(T_design):
                y_tilde, R_tilde = pool_spike_pseudo(omega_SRT[s, :, t], kappa_SRT[s, :, t])
                omega_pooled_ST[s, t] = 1.0 / R_tilde  # back to precision
                y_spk_pooled_ST[s, t] = y_tilde
                R_spk_pooled_ST[s, t] = R_tilde

        # Now we have pooled observations per unit. For multi-spike KF, we'll pass all S rows
        # Reshape back to format expected by joint_kf_rts_moments: (S, T)
        beta_median = beta_median_SR.reshape(S, R, P)  # (S, R, P)
        gamma_refresh = gamma_refresh_SR.reshape(S, R, Rlags)  # (S, R, Rlags)

        # For the KF, we use the median β and γ. Since we have per-trial coefficients,
        # we need to either:
        # (a) Average β/γ across trials per unit (simplest for shared latent)
        # (b) Use trial-averaged latent predictors
        # Let's use (a) for now
        beta_refresh = beta_median.mean(axis=1)  # (S, P) average across trials
        gamma_refresh_avg = gamma_refresh.mean(axis=1)  # (S, Rlags)

        # ---- LATENT REFRESH (multi-spike filter with pooled obs) ----
        mom = joint_kf_rts_moments(
            Y_cube=Y_pooled_refresh, theta=theta,
            delta_spk=delta_spk, win_sec=window_sec, offset_sec=offset_sec,
            beta=beta_refresh.astype(np.float64),      # (S, 1+2J)
            gamma=gamma_refresh_avg.astype(np.float64),  # (S, R)
            spikes=spikes_SRT.mean(axis=1).astype(np.float64),  # (S, T) averaged across trials
            omega=omega_pooled_ST.astype(np.float64),   # (S, T) pooled precision
            coupled_bands_idx=np.arange(J, dtype=np.int64),
            freqs_for_phase=np.asarray(all_freqs, np.float64),
            sidx=sidx,
            H_hist=H_hist.mean(axis=1).astype(np.float64),  # (S, T, Rlags) avg across trials
            sigma_u=sigma_u
        )

        # ---- rebuild regressors (+ variances) from refreshed latents ----
        lat_reim_np, var_reim_np = extract_band_reim_with_var(
            mu_fine=mom.m_s, var_fine=mom.P_s,
            coupled_bands=all_freqs, freqs_hz=all_freqs, delta_spk=delta_spk, J=J, M=M
        )
        lat_reim_jax = jnp.asarray(lat_reim_np)
        var_reim_np_current = var_reim_np
        design_np = np.asarray(build_design(lat_reim_jax))
        T_design = min(int(design_np.shape[0]), H_hist.shape[2], T_total)

        # refresh stable slices for next inner block
        X_slice = np.ascontiguousarray(design_np[:T_design], dtype=np.float64)
        V_slice = np.ascontiguousarray(var_reim_np_current[:T_design], dtype=np.float64)
        lat_slice = lat_reim_jax[:T_design]
        spikes_slice = [jnp.asarray(spikes_SR[sp, :T_design]) for sp in range(Sprime)]
        H_slice = [np.asarray(H_SR[sp, :T_design]) for sp in range(Sprime)]

        # prep ω for next block
        for sp in range(Sprime):
            psi = X_slice @ beta_SR[sp] + H_slice[sp] @ gamma_SR[sp]
            omega_SR[sp] = np.maximum(sample_pg_wrapper(np.asarray(psi)), config.omega_floor)

        # bookkeeping
        trace.theta.append(theta)
        trace.latent.append(lat_reim_jax)
        trace.fine_latent.append(mom.m_s)

    # ── Return shapes: (S, R, P) and (S, R, Rlags) ──────────────────────────
    beta_out = beta_SR.reshape(S, R, P)
    gamma_out = gamma_SR.reshape(S, R, Rlags)

    return beta_out, gamma_out, theta, trace
