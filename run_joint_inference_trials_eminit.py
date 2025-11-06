# run_joint_inference_trials_eminit.py
# Trial-aware runner with EM (hierarchical) warm start for θ and fine-grid latents.
# Follows the pattern of run_joint_inference_jax_v2_trials.py for maximum performance.

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Dict, Any, Callable
import numpy as np
import jax.numpy as jnp
import jax.random as jr
from tqdm.auto import tqdm

from src.params import OUParams
from src.utils_joint import Trace
from src.state_index import StateIndex
from src.joint_inference_core import joint_kf_rts_moments, JointMoments
from src.beta_sampler import gibbs_update_beta_robust
from src.priors import gamma_prior_simple
from src.pg_utils import sample_polya_gamma
from src.polyagamma_jax import sample_pg_batch


# ========================= Pooling utilities (from run_joint_inference_jax_v2_trials.py) =========================

def pool_lfp_trials(Y_rm: np.ndarray, sig_eps_mr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Precision-pool LFP observations across trials."""
    w = 1.0 / np.asarray(sig_eps_mr, float)       # (R, M) precision weights
    Y_tilde = (w * Y_rm).sum(axis=0) / w.sum(axis=0)
    sig_tilde = 1.0 / w.sum(axis=0)
    return Y_tilde, sig_tilde


def pool_spike_pseudo(omega_nr: np.ndarray, kappa_nr: np.ndarray) -> Tuple[float, float]:
    """Pool spike pseudo-observations across trials."""
    wsum = np.asarray(omega_nr, float).sum()
    y_tilde = np.asarray(kappa_nr, float).sum() / wsum
    R_tilde = 1.0 / wsum
    return y_tilde, R_tilde


# ========================= Config =========================

@dataclass
class InferenceTrialsEMConfig:
    fixed_iter: int = 150
    beta0_window: int = 100
    n_refreshes: int = 3
    inner_steps_per_refresh: int = 100
    omega_floor: float = 1e-3
    sigma_u: float = 0.05
    # Sampler choice
    pg_jax: bool = False                        # if True, use JAX-based PG sampler (faster)
    # EM settings
    em_kwargs: Dict[str, Any] = None
    # RNG
    key_jax: Optional["jr.KeyArray"] = None


# ========================= EM → θ, latents adapters =========================

def _theta_from_em(res, J: int, M: int, Rtr: int) -> Tuple[OUParams, np.ndarray]:
    """Build OUParams (pooled σ_ε for the core) and per-trial σ_ε from EM output."""
    lam = np.asarray(getattr(res, "lam_X"), float).reshape(J, 1)
    sigv = np.asarray(getattr(res, "sigv_X"), float).reshape(J, 1)

    if hasattr(res, "sig_eps_jmr"):
        sig_eps_jmr = np.asarray(res.sig_eps_jmr, float)
        if sig_eps_jmr.shape[0] == 1:
            sig_eps_jmr = np.broadcast_to(sig_eps_jmr, (J, sig_eps_jmr.shape[1], sig_eps_jmr.shape[2]))
        sig_eps_trials = np.moveaxis(sig_eps_jmr, 2, 0)
    elif hasattr(res, "sig_eps_mr"):
        sig_eps_mr = np.asarray(res.sig_eps_mr, float)
        sig_eps_trials = np.broadcast_to(sig_eps_mr.T[:, None, :], (Rtr, J, M))
    else:
        sig_eps_trials = np.full((Rtr, J, M), 5.0, float)

    var_rm = sig_eps_trials ** 2
    w_rm = 1.0 / np.maximum(var_rm, 1e-20)
    var_pool = 1.0 / np.maximum(w_rm.sum(axis=0), 1e-20)
    sig_eps_pool = np.sqrt(var_pool)

    theta0 = OUParams(lam=lam, sig_v=sigv, sig_eps=sig_eps_pool)
    return theta0, sig_eps_trials


def _extract_from_upsampled(upsampled, J: int, M: int) -> Tuple[np.ndarray, np.ndarray]:
    """Convert UpsampleResult -> (mu_fine, var_fine) with shapes (T_f, 2*J*M)."""
    mean_cplx = np.asarray(upsampled.X_mean)   # (J, M, T_f) complex
    var_real  = np.asarray(upsampled.X_var)    # (J, M, T_f) real

    T_f = mean_cplx.shape[2]
    mu_fine  = np.zeros((T_f, 2 * J * M), dtype=np.float64)
    var_fine = np.zeros((T_f, 2 * J * M), dtype=np.float64)

    Re = mean_cplx.real.transpose(2, 0, 1)
    Im = mean_cplx.imag.transpose(2, 0, 1)
    Vr = var_real.transpose(2, 0, 1)

    idx = 0
    for j in range(J):
        for m in range(M):
            mu_fine[:, idx]   = Re[:, j, m]
            var_fine[:, idx]  = Vr[:, j, m]
            mu_fine[:, idx+1] = Im[:, j, m]
            var_fine[:, idx+1]= Vr[:, j, m]
            idx += 2

    return mu_fine, var_fine


def _reim_from_fine(mu_fine: np.ndarray, var_fine: np.ndarray, J: int, M: int) -> Tuple[np.ndarray, np.ndarray]:
    """Build (T, 2J) taper-AVERAGED regressors and their variances from (T, 2JM)."""
    T, d = mu_fine.shape
    tmp = mu_fine.reshape(T, J, M, 2)
    vmp = var_fine.reshape(T, J, M, 2)

    mu_re = tmp[..., 0].mean(axis=2)
    mu_im = tmp[..., 1].mean(axis=2)
    var_re = (vmp[..., 0].mean(axis=2)) / M
    var_im = (vmp[..., 1].mean(axis=2)) / M

    latent_reim = np.concatenate([mu_re, mu_im], axis=1)
    var_lat     = np.concatenate([var_re, var_im], axis=1)
    return latent_reim, var_lat


# ========================= Main runner =========================

def run_joint_inference_trials(
    Y_trials: np.ndarray,            # (R, J, M, K) complex
    spikes: np.ndarray,              # (R, S, T_f) or (S, R, T_f)
    H_hist: np.ndarray,              # (S, R, T_f, Rlags)
    all_freqs: Sequence[float],      # (J,)
    *,
    delta_spk: float,
    window_sec: float,
    offset_sec: float = 0.0,
    config: Optional[InferenceTrialsEMConfig] = None,
    rng_pg: np.random.Generator = np.random.default_rng(0),
) -> Tuple[np.ndarray, np.ndarray, OUParams, Trace]:
    """
    EM-initialised trial-aware joint inference following run_joint_inference_jax_v2_trials.py pattern.

    Returns:
      beta:  (S, R, 1+2J) per unit, per trial
      gamma: (S, R, Rlags) per unit, per trial
      theta: OUParams (pooled σ_ε)
      trace: Trace
    """
    if config is None:
        config = InferenceTrialsEMConfig()
    key = config.key_jax or jr.PRNGKey(0)

    # Normalize spikes to (R, S, T_f)
    spikes = np.asarray(spikes)
    if spikes.shape[0] < spikes.shape[1]:
        R, S, T_total = spikes.shape
    else:
        spikes = np.transpose(spikes, (1, 0, 2))
        R, S, T_total = spikes.shape

    Rtr, J, M, K = Y_trials.shape
    assert Rtr == R
    Rlags = H_hist.shape[-1]

    # Initialize PG sampler wrapper
    key_pg_jax = None
    if config.pg_jax:
        import jax
        with jax.default_device(jax.devices("cpu")[0]):
            key_pg_jax = jr.PRNGKey(42)

    def sample_pg_wrapper(psi: np.ndarray) -> np.ndarray:
        nonlocal key_pg_jax
        if config.pg_jax:
            key_pg_jax, subkey = jr.split(key_pg_jax)
            samples = sample_pg_batch(subkey, jnp.asarray(psi), h=1.0)
            return np.asarray(samples)
        else:
            return sample_polya_gamma(np.asarray(psi), rng_pg)

    print(f"[EM-init Trial inference] R={R} trials, S={S} units, J={J} bands, M={M} tapers")
    print(f"[EM-init Trial inference] Running EM initialization...")

    # ---------- EM on trials ----------
    from src.em_ct_hier_jax import em_ct_hier_jax
    em_kwargs = dict(max_iter=5000, tol=1e-3, sig_eps_init=5.0, log_every=1000)
    if config.em_kwargs:
        em_kwargs.update(config.em_kwargs)

    res = em_ct_hier_jax(Y_trials=Y_trials, db=window_sec, **em_kwargs)
    theta, sig_eps_trials = _theta_from_em(res, J=J, M=M, Rtr=Rtr)

    print(f"[EM-init Trial inference] Upsampling latents...")

    # ---------- Upsample ----------
    from src.upsample_ct_hier_fine import upsample_ct_hier_fine
    ups = upsample_ct_hier_fine(
        Y_trials=Y_trials, res=res,
        delta_spk=delta_spk, win_sec=window_sec, offset_sec=offset_sec, T_f=None,
    )
    mu_fine0, var_fine0 = _extract_from_upsampled(ups, J=J, M=M)
    lat_reim_np, var_reim_np = _reim_from_fine(mu_fine0, var_fine0, J=J, M=M)
    T0 = min(T_total, lat_reim_np.shape[0])
    lat_reim_np = lat_reim_np[:T0]
    var_reim_np = var_reim_np[:T0]

    print(f"[EM-init Trial inference] Starting Gibbs sampling (following jax_v2_trials pattern)...")
    print(f"[EM-init Trial inference] Warm-up: {config.fixed_iter} iterations")
    print(f"[EM-init Trial inference] Refresh: {config.n_refreshes} passes × {config.inner_steps_per_refresh} steps")

    # ---------- Flatten (S, R) → Sprime pseudo-units (KEY PATTERN from reference) ----------
    Sprime = S * R
    B = len(all_freqs)
    P = 1 + 2*B

    spikes_SR = spikes.transpose(1, 0, 2).reshape(Sprime, T_total)
    H_SR = H_hist.transpose(0, 1, 2, 3).reshape(Sprime, T_total, Rlags)

    # Stable slices (for Gibbs cache hits - CRITICAL!)
    lat_slice = jnp.asarray(lat_reim_np[:T0])
    V_slice = np.ascontiguousarray(var_reim_np[:T0], dtype=np.float64)
    X_slice = np.ascontiguousarray(
        np.concatenate([np.ones((T0, 1)), np.asarray(lat_slice)], axis=1),
        dtype=np.float64
    )
    spikes_slice = [jnp.asarray(spikes_SR[sp, :T0]) for sp in range(Sprime)]
    H_slice = [np.asarray(H_SR[sp, :T0]) for sp in range(Sprime)]

    # ---------- Priors & init ----------
    beta_SR = np.zeros((Sprime, P), dtype=np.float64)
    gamma_SR = np.zeros((Sprime, Rlags), dtype=np.float64)
    a0_ard, b0_ard = 1e-2, 1e-2
    tau2_lat_SR = np.ones((Sprime, 2*B), dtype=np.float64)

    mu_g, Sig_g = gamma_prior_simple(n_lags=Rlags, strong_neg=-2.5, mild_neg=-0.5, k_mild=4, tau_gamma=1.5)

    # Initial ω
    omega_SR = np.empty((Sprime, T0), dtype=np.float64)
    for sp in range(Sprime):
        psi0 = X_slice @ beta_SR[sp] + H_slice[sp] @ gamma_SR[sp]
        omega_SR[sp] = np.maximum(sample_pg_wrapper(np.asarray(psi0)), config.omega_floor)

    # ---------- Trace ----------
    trace = Trace()
    trace.theta.append(theta)
    trace.latent.append(lat_slice)
    trace.fine_latent.append(mu_fine0)

    # ---------- WARM-UP ----------
    beta0_history_SR = [[] for _ in range(Sprime)]
    gamma_hist_SR = [[] for _ in range(Sprime)]

    pbar_warm = tqdm(range(config.fixed_iter), unit="it", desc="Warm-up (β/γ per pseudo-unit)", mininterval=0.3)
    for _ in pbar_warm:
        for sp in range(Sprime):
            psi = X_slice @ beta_SR[sp] + H_slice[sp] @ gamma_SR[sp]
            omega = np.maximum(sample_pg_wrapper(np.asarray(psi)), config.omega_floor)

            key, b_new, g_new, t2_new = gibbs_update_beta_robust(
                key,
                lat_slice,           # STABLE JAX array
                spikes_slice[sp],    # STABLE JAX array
                jnp.asarray(omega),
                H_hist=H_slice[sp],  # STABLE array
                Sigma_gamma=Sig_g, mu_gamma=mu_g,
                var_latent_reim=V_slice,  # STABLE array
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

        trace.beta.append(beta_SR.reshape(S, R, P).copy())
        trace.gamma.append(gamma_SR.reshape(S, R, Rlags).copy())

    # Freeze β0
    beta0_fixed_SR = np.array([
        np.median(h) if len(h) else beta_SR[sp, 0]
        for sp, h in enumerate(beta0_history_SR)
    ], dtype=np.float64)
    beta_SR[:, 0] = beta0_fixed_SR

    print(f"[EM-init Trial inference] Warm-up completed.")

    # γ posteriors
    mu_g_post_SR = np.zeros((Sprime, Rlags), dtype=np.float64)
    Sig_g_post_SR = np.zeros((Sprime, Rlags, Rlags), dtype=np.float64)
    Sig_g_lock_SR = np.zeros((Sprime, Rlags, Rlags), dtype=np.float64)

    for sp in range(Sprime):
        gh = np.stack(gamma_hist_SR[sp], axis=0) if len(gamma_hist_SR[sp]) else np.zeros((1, Rlags))
        mu_s = gh.mean(axis=0)
        ctr = gh - mu_s[None, :]
        Sg = (ctr.T @ ctr) / max(gh.shape[0]-1, 1) + 1e-6 * np.eye(Rlags)
        mu_g_post_SR[sp] = mu_s
        Sig_g_post_SR[sp] = Sg
        diag_scale = np.clip(np.diag(Sg), 1e-10, None)
        Sig_g_lock_SR[sp] = np.diag(1e-6 * diag_scale)

    # ---------- REFRESH PASSES ----------
    sidx = StateIndex(J, M)

    pbar_pass = tqdm(range(config.n_refreshes), unit="pass", desc="Latent refresh passes", mininterval=0.3)
    for refresh_idx in pbar_pass:
        # Inner steps
        for _ in range(config.inner_steps_per_refresh):
            for sp in range(Sprime):
                gamma_samp = rng_pg.multivariate_normal(mean=mu_g_post_SR[sp], cov=Sig_g_post_SR[sp])
                psi = X_slice @ beta_SR[sp] + H_slice[sp] @ gamma_samp
                omega = np.maximum(sample_pg_wrapper(np.asarray(psi)), config.omega_floor)

                key, b_new, g_new, t2_new = gibbs_update_beta_robust(
                    key,
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
                beta_SR[sp, 0] = beta0_fixed_SR[sp]
                tau2_lat_SR[sp] = t2_new
                gamma_SR[sp] = gamma_samp

            trace.beta.append(beta_SR.reshape(S, R, P).copy())
            trace.gamma.append(gamma_SR.reshape(S, R, Rlags).copy())

        # Robust β
        recent = np.stack([trace.beta[-i] for i in range(1, config.inner_steps_per_refresh + 1)], axis=0)
        beta_median_SR = np.median(recent, axis=0).reshape(Sprime, P)

        # Build ω for refresh
        omega_refresh_SR = np.empty((Sprime, T0), dtype=np.float64)
        gamma_refresh_SR = np.empty((Sprime, Rlags), dtype=np.float64)
        for sp in range(Sprime):
            gamma_samp = rng_pg.multivariate_normal(mean=mu_g_post_SR[sp], cov=Sig_g_post_SR[sp])
            psi_refresh = X_slice @ beta_median_SR[sp] + H_slice[sp] @ gamma_samp
            omega_refresh_SR[sp] = np.maximum(sample_pg_wrapper(np.asarray(psi_refresh)), config.omega_floor)
            gamma_refresh_SR[sp] = gamma_samp

        # Pool observations for LATENT REFRESH
        Y_pooled_refresh = np.zeros((J, M, K), dtype=np.complex128)
        sig_pooled_refresh = np.zeros((J, M), dtype=np.float64)

        for j in range(J):
            for k in range(K):
                Y_rm = Y_trials[:, j, :, k]
                sig_rm = np.broadcast_to(theta.sig_eps[j, :], (R, M))
                Y_pooled_refresh[j, :, k], sig_m = pool_lfp_trials(Y_rm, sig_rm)
                if k == 0:
                    sig_pooled_refresh[j, :] = sig_m

        # Pool spike pseudo-obs
        omega_SRT = omega_refresh_SR.reshape(S, R, T0)
        gamma_SRT = gamma_refresh_SR.reshape(S, R, Rlags)
        spikes_SRT = spikes_SR.reshape(S, R, T_total)[:, :, :T0]
        kappa_SRT = spikes_SRT.astype(np.float64) - 0.5

        omega_pooled_ST = np.zeros((S, T0), dtype=np.float64)
        y_spk_pooled_ST = np.zeros((S, T0), dtype=np.float64)
        R_spk_pooled_ST = np.zeros((S, T0), dtype=np.float64)

        for s in range(S):
            for t in range(T0):
                y_tilde, R_tilde = pool_spike_pseudo(omega_SRT[s, :, t], kappa_SRT[s, :, t])
                omega_pooled_ST[s, t] = 1.0 / R_tilde
                y_spk_pooled_ST[s, t] = y_tilde
                R_spk_pooled_ST[s, t] = R_tilde

        beta_median = beta_median_SR.reshape(S, R, P)
        gamma_refresh = gamma_refresh_SR.reshape(S, R, Rlags)
        beta_refresh = beta_median.mean(axis=1)
        gamma_refresh_avg = gamma_refresh.mean(axis=1)

        # LATENT REFRESH
        mom = joint_kf_rts_moments(
            Y_cube=Y_pooled_refresh, theta=theta,
            delta_spk=delta_spk, win_sec=window_sec, offset_sec=offset_sec,
            beta=beta_refresh.astype(np.float64),
            gamma=gamma_refresh_avg.astype(np.float64),
            spikes=spikes_SRT.mean(axis=1).astype(np.float64),
            omega=omega_pooled_ST.astype(np.float64),
            coupled_bands_idx=np.arange(J, dtype=np.int64),
            freqs_for_phase=np.asarray(all_freqs, np.float64),
            sidx=sidx,
            H_hist=H_hist.mean(axis=1).astype(np.float64),
            sigma_u=config.sigma_u
        )

        # Rebuild regressors
        lat_reim_np, var_reim_np = _reim_from_fine(mom.m_s, mom.P_s, J=J, M=M)
        T0 = min(T0, lat_reim_np.shape[0])

        # Rebuild STABLE slices
        lat_slice = jnp.asarray(lat_reim_np[:T0])
        V_slice = np.ascontiguousarray(var_reim_np[:T0], dtype=np.float64)
        X_slice = np.ascontiguousarray(
            np.concatenate([np.ones((T0, 1)), np.asarray(lat_slice)], axis=1),
            dtype=np.float64
        )

        # Prep ω for next block
        for sp in range(Sprime):
            psi = X_slice @ beta_SR[sp] + H_slice[sp] @ gamma_SR[sp]
            omega_SR[sp] = np.maximum(sample_pg_wrapper(np.asarray(psi)), config.omega_floor)

        trace.theta.append(theta)
        trace.latent.append(lat_slice)
        trace.fine_latent.append(mom.m_s)

    print(f"[EM-init Trial inference] Inference completed!")

    beta_out = beta_SR.reshape(S, R, P)
    gamma_out = gamma_SR.reshape(S, R, Rlags)

    return beta_out, gamma_out, theta, trace
