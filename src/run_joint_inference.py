from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Tuple, List
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from tqdm.auto import tqdm
from src.joint_inference_core import JointMoments
from src.params import OUParams
from src.priors import gamma_prior_simple
from src.pg_utils import sample_polya_gamma
from src.polyagamma_jax import sample_pg_saddle_single  # Direct JAX saddle point sampler
from src.utils_joint import Trace

# Pre-compiled JAX Polyagamma batch sampler (matches fast reference implementation)
# Defined at module level for proper JIT compilation
@jax.jit
def _sample_omega_pg_batch(key: "jr.KeyArray", psi: "jnp.ndarray", omega_floor: float) -> "jnp.ndarray":
    """
    Fast batch Polyagamma sampler using saddle point method.
    Matches the pattern from the fast reference implementation.

    Args:
        key: JAX random key
        psi: (N,) array of log-odds parameters
        omega_floor: minimum omega value

    Returns:
        (N,) array of PG(1, psi) samples
    """
    N = psi.shape[0]
    keys = jr.split(key, N)
    omega = jax.vmap(lambda k, z: sample_pg_saddle_single(k, 1.0, z))(keys, psi)
    return jnp.maximum(omega, omega_floor)

@dataclass
class InferenceConfig:
    # NEW: do a warm-up before any latent refresh; freeze β0 after warm-up
    fixed_iter: int = 100              # number of PG steps before any latent sampling
    beta0_window: int = 100            # robust window to estimate β0 (median of last W)
    # Existing controls
    n_refreshes: int = 3              # how many latent refresh passes (after warm-up)
    inner_steps_per_refresh: int = 100 # β/γ/ω updates between refreshes
    omega_floor: float = 1e-3          # keep PG rows well-behaved
    sigma_u: float = 0.05              # sigma_u for the OU process
    pg_jax: bool = False               # if True, use JAX-based Polyagamma sampler (faster, GPU-capable)

def run_joint_inference(
    Y_cube_block: np.ndarray,            # (J, M, K) complex TFR (derotated+scaled)
    params0: "OUParams",                 # initial OU params (LFP-only warm start)
    spikes: np.ndarray,                  # (T_f,)  OR (S, T_f)
    H_hist: np.ndarray,                  # (T_f, R) OR (S, T_f, R)
    all_freqs: np.ndarray,               # (J,) band frequencies (Hz) (order matches Y_cube_block)
    build_design: Callable,              # X = [1, ReZ̃..., ImZ̃...] from latent_reim
    extract_band_reim_with_var: Callable[..., Tuple[np.ndarray, np.ndarray]],
    gibbs_update_beta_robust: Callable[..., Tuple["jax.random.KeyArray",
                                                  jnp.ndarray,
                                                  Optional[jnp.ndarray],
                                                  np.ndarray]],
    joint_kf_rts_moments: Callable[..., "JointMoments"],   # (multi-spike aware version)
    em_theta_from_joint: Callable[..., "OUParams"],        # optional; left unused as requested
    config: "InferenceConfig" = None,
    *,
    delta_spk: float,
    window_sec: float,
    rng_pg: np.random.Generator = np.random.default_rng(0),
    key_jax: "jr.KeyArray | None" = None,
    offset_sec: float = 0.0,
):
    """
    Multi-spike orchestration, backward-compatible with single-spike:
      • LFP-only init → rotated taper-avg regressors (+ variances)
      • Warm-up PG steps: per-train β/γ (freeze β0 per train)
      • Build γ posteriors per train
      • For each refresh pass:
          – per-train inner PG β/γ updates (shared design/latents)
          – latent refresh with *all* spike trains (multi-spike filter)
          – rebuild regressors/variances for next block
    Returns
    -------
    beta, gamma, theta, trace
      beta  : (1+2J,)        in single-spike   | (S, 1+2J) in multi-spike
      gamma : (R,)           in single-spike   | (S, R)    in multi-spike
    """
    if key_jax is None:
        # create on CPU to avoid early cuDNN init
        import jax
        with jax.default_device(jax.devices("cpu")[0]):
            key_jax = jr.PRNGKey(0)
    if config is None:
        config = InferenceConfig()
    # if sigma_u is not in config, set it to 0.05 if absent
    if not hasattr(config, 'sigma_u'):
        config.sigma_u = 0.05
    else:
        sigma_u = config.sigma_u

    # Initialize JAX RNG key if using JAX backend
    key_pg_jax = None
    if config.pg_jax:
        import jax
        with jax.default_device(jax.devices("cpu")[0]):
            key_pg_jax = jr.PRNGKey(42)  # separate seed for PG sampling

    # Wrapper function to switch between numpy and JAX Polyagamma samplers
    def sample_pg_wrapper(psi: np.ndarray) -> np.ndarray:
        """
        Sample from Polya-Gamma distribution using either numpy or JAX backend.

        JAX version uses _sample_omega_pg_batch which is JIT-compiled at module level.
        This matches the fast reference implementation pattern.
        """
        nonlocal key_pg_jax

        if config.pg_jax:
            # Use JAX sampler (direct vmap of saddle point method)
            key_pg_jax, subkey = jr.split(key_pg_jax)
            psi_jax = jnp.asarray(psi, dtype=jnp.float64)
            samples = _sample_omega_pg_batch(subkey, psi_jax, config.omega_floor)
            return np.asarray(samples)
        else:
            # Use original numpy-based sampler
            return sample_polya_gamma(np.asarray(psi), rng_pg)
    # ── Shapes & normalization to multi-spike ──────────────────────────────
    J, M, K = Y_cube_block.shape
    single_spike_mode = (spikes.ndim == 1)

    if single_spike_mode:
        spikes_S = spikes[None, :]                 # (S=1, T)
        H_hist_S = H_hist[None, :, :]              # (1, T, R)
    else:
        spikes_S = spikes                           # (S, T)
        H_hist_S = H_hist                           # (S, T, R)

    S, T_total = spikes_S.shape
    R = H_hist_S.shape[2]

    # ── θ from LFP-only warm start (broadcast σ_ε to (J,M)) ─────────────────
    theta = OUParams(lam=params0.lam,
                     sig_v=params0.sig_v,
                     sig_eps=np.broadcast_to(params0.sig_eps, (J, M)))


    # ── LFP-only fine smoother → regressors (+ variances) ───────────────────
    from src.ou_fine import kalman_filter_rts_ffbs_fine
    fine0 = kalman_filter_rts_ffbs_fine(
        Y_cube_block, theta, delta_spk=delta_spk, win_sec=window_sec, offset_sec=offset_sec
    )

    # Use ([:-1]) as in your working pipeline to align T_f
    lat_reim_np, var_reim_np = extract_band_reim_with_var(
        mu_fine=np.asarray(fine0.mu)[:-1],      # (T_f, d)
        var_fine=np.asarray(fine0.var)[:-1],
        coupled_bands=all_freqs, freqs_hz=all_freqs, delta_spk=delta_spk, J=J, M=M
    )
    lat_reim_jax = jnp.asarray(lat_reim_np)
    print("lat_reim_jax device:", (lat_reim_jax.device() if callable(getattr(lat_reim_jax,"device",None)) else lat_reim_jax.device))
    var_reim_np_current = var_reim_np
    design_np = np.asarray(build_design(lat_reim_jax))      # (T_f, 1+2J)
    T_design = min(int(design_np.shape[0]), H_hist_S.shape[1], spikes_S.shape[1])

    # Stable slices (so gibbs cache hits every iteration)
    X_slice   = np.ascontiguousarray(design_np[:T_design], dtype=np.float64)      # (T, 1+2J)
    V_slice   = np.ascontiguousarray(var_reim_np_current[:T_design], dtype=np.float64)  # (T, 2J)
    lat_slice = lat_reim_jax[:T_design]                                           # JAX array (stable id)
    spikes_slice = [jnp.asarray(spikes_S[s, :T_design]) for s in range(S)]
    H_slice      = [np.asarray(H_hist_S[s, :T_design])  for s in range(S)]

    # ── Priors & per-train init for β/γ/ARD ─────────────────────────────────
    B = len(all_freqs)
    beta  = np.zeros((S, 1 + 2*B), dtype=np.float64)      # one β per train
    gamma = np.zeros((S, R),        dtype=np.float64)     # one γ per train
    a0_ard, b0_ard = 1e-2, 1e-2
    tau2_lat = np.ones((S, 2*B), dtype=np.float64)

    mu_g, Sig_g = gamma_prior_simple(n_lags=R, strong_neg=-2.5, mild_neg=-0.5, k_mild=4, tau_gamma=1.5)

    # Initial ω per train
    omega_S = np.empty((S, T_design), dtype=np.float64)
    for s in range(S):
        psi0 = X_slice @ beta[s] + H_slice[s] @ gamma[s]
        omega_S[s] = np.maximum(sample_pg_wrapper(np.asarray(psi0)), config.omega_floor)

    # ── Trace bookkeeping (store arrays per iteration) ──────────────────────
    trace = Trace()
    trace.theta.append(theta)
    trace.latent.append(lat_reim_jax)
    trace.fine_latent.append(np.asarray(fine0.mu))

    # ====================== (1) WARM-UP: per-train β/γ ======================
    beta0_history: List[List[float]] = [[] for _ in range(S)]
    gamma_hist:    List[List[np.ndarray]] = [[] for _ in range(S)]

    pbar_warm = tqdm(range(config.fixed_iter), unit="it", desc="Warm-up (β/γ per train)", mininterval=0.3)
    for _ in pbar_warm:
        for s in range(S):
            psi   = X_slice @ beta[s] + H_slice[s] @ gamma[s]
            omega = np.maximum(sample_pg_wrapper(np.asarray(psi)), config.omega_floor)
            key_jax, b_new, g_new, t2_new = gibbs_update_beta_robust(
                key_jax,
                lat_slice,
                spikes_slice[s],
                jnp.asarray(omega),
                H_hist=H_slice[s],
                Sigma_gamma=Sig_g, mu_gamma=mu_g,
                var_latent_reim=V_slice,
                a0_ard=a0_ard, b0_ard=b0_ard, tau2_lat=tau2_lat[s],
                tau2_intercept=100.0**2, tau2_gamma=25.0**2,
                omega_floor=config.omega_floor
            )
            beta[s]     = np.asarray(b_new)
            gamma[s]    = np.asarray(g_new)
            tau2_lat[s] = t2_new

            beta0_history[s].append(float(beta[s, 0]))
            if len(beta0_history[s]) > config.beta0_window:
                beta0_history[s].pop(0)
            gamma_hist[s].append(gamma[s].copy())

        trace.beta.append(beta.copy())
        trace.gamma.append(gamma.copy())

    # Freeze β0 per train (robust median)
    beta0_fixed = np.array([np.median(h) if len(h) else beta[s,0] for s,h in enumerate(beta0_history)], dtype=np.float64)
    beta[:, 0] = beta0_fixed

    # γ posterior per train (mean/cov + tight locks)
    mu_g_post  = np.zeros((S, R), dtype=np.float64)
    Sig_g_post = np.zeros((S, R, R), dtype=np.float64)
    Sig_g_lock = np.zeros((S, R, R), dtype=np.float64)
    for s in range(S):
        gh = np.stack(gamma_hist[s], axis=0) if len(gamma_hist[s]) else np.zeros((1, R))
        mu_s = gh.mean(axis=0)
        ctr  = gh - mu_s[None, :]
        Sg   = (ctr.T @ ctr) / max(gh.shape[0]-1, 1)
        Sg   = Sg + 1e-6 * np.eye(R)
        mu_g_post[s]  = mu_s
        Sig_g_post[s] = Sg
        diag_scale = np.clip(np.diag(Sg), 1e-10, None)
        Sig_g_lock[s] = np.diag(1e-6 * diag_scale)

    # ====================== (2) REFRESH PASSES ==============================
    from src.state_index import StateIndex
    sidx = StateIndex(J, M)

    pbar_pass = tqdm(range(config.n_refreshes), unit="pass", desc="Latent refresh passes", mininterval=0.3)
    for r in pbar_pass:
        # ---- inner PG steps per train (latents fixed) ----
        for _ in range(config.inner_steps_per_refresh):
            for s in range(S):
                gamma_samp = rng_pg.multivariate_normal(mean=mu_g_post[s], cov=Sig_g_post[s])
                psi   = X_slice @ beta[s] + H_slice[s] @ gamma_samp
                omega = np.maximum(sample_pg_wrapper(np.asarray(psi)), config.omega_floor)

                key_jax, b_new, g_new, t2_new = gibbs_update_beta_robust(
                    key_jax,
                    lat_slice,
                    spikes_slice[s],
                    jnp.asarray(omega),
                    H_hist=H_slice[s],
                    Sigma_gamma=Sig_g_lock[s],     # tight lock around sampled γ
                    mu_gamma=gamma_samp,
                    var_latent_reim=V_slice,
                    a0_ard=a0_ard, b0_ard=b0_ard, tau2_lat=tau2_lat[s],
                    tau2_intercept=100.0**2, tau2_gamma=25.0**2,
                    omega_floor=config.omega_floor
                )
                beta[s]     = np.asarray(b_new); beta[s, 0] = beta0_fixed[s]
                tau2_lat[s] = t2_new
                gamma[s]    = gamma_samp

            trace.beta.append(beta.copy())
            trace.gamma.append(gamma.copy())

        # ---- robust β for refresh (median across last inner block) ----
        recent = np.stack(trace.beta[-config.inner_steps_per_refresh:], axis=0)   # (inner, S, 1+2J)
        beta_median = np.median(recent, axis=0)                                   # (S, 1+2J)

        # ---- build ω for refresh per train ----
        omega_refresh = np.empty((S, T_design), dtype=np.float64)
        for s in range(S):
            gamma_samp = rng_pg.multivariate_normal(mean=mu_g_post[s], cov=Sig_g_post[s])
            psi_refresh = X_slice @ beta_median[s] + H_slice[s] @ gamma_samp
            omega_refresh[s] = np.maximum(sample_pg_wrapper(np.asarray(psi_refresh)), config.omega_floor)
            gamma[s] = gamma_samp   # keep the draw used at refresh

        
        # ---- LATENT REFRESH (multi-spike filter) ----
        mom = joint_kf_rts_moments(
            Y_cube=Y_cube_block, theta=theta,
            delta_spk=delta_spk, win_sec=window_sec, offset_sec=offset_sec,
            beta=beta_median.astype(np.float64),      # (S, 1+2J)
            gamma=gamma.astype(np.float64),           # (S, R)
            spikes=spikes_S.astype(np.float64),       # (S, T)
            omega=omega_refresh.astype(np.float64),   # (S, T)
            coupled_bands_idx=np.arange(J, dtype=np.int64),
            freqs_for_phase=np.asarray(all_freqs, np.float64),
            sidx=sidx, H_hist=H_hist_S,
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
        T_design = min(int(design_np.shape[0]), H_hist_S.shape[1], spikes_S.shape[1])

        # refresh stable slices for next inner block
        X_slice   = np.ascontiguousarray(design_np[:T_design], dtype=np.float64)
        V_slice   = np.ascontiguousarray(var_reim_np_current[:T_design], dtype=np.float64)
        lat_slice = lat_reim_jax[:T_design]
        spikes_slice = [jnp.asarray(spikes_S[s, :T_design]) for s in range(S)]
        H_slice      = [np.asarray(H_hist_S[s, :T_design])  for s in range(S)]

        # prep ω for next block (per train)
        for s in range(S):
            psi = X_slice @ beta[s] + H_slice[s] @ gamma[s]
            omega_S[s] = np.maximum(sample_pg_wrapper(np.asarray(psi)), config.omega_floor)

        # bookkeeping
        trace.theta.append(theta)
        trace.latent.append(lat_reim_jax)
        trace.fine_latent.append(mom.m_s)

    # ── Return shapes consistent with input mode ────────────────────────────
    if single_spike_mode:
        return beta[0], gamma[0], theta, trace
    else:
        return beta, gamma, theta, trace
