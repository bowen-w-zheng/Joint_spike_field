# run_joint_inference_trials_eminit.py
# Trial-aware runner with EM (hierarchical) warm start for θ and fine-grid latents.

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Dict, Any
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from tqdm.auto import tqdm

from src.params import OUParams
from src.utils_joint import Trace
from src.state_index import StateIndex

# trial-aware core (pooling wrapper → calls your untouched core)
from src.joint_inference_core_trial import joint_kf_rts_moments_trials
# trial-aware beta sampler
from src.beta_sampler_trials import gibbs_update_beta_trials_shared, TrialBetaConfig
# PG samplers
from src.pg_utils import sample_polya_gamma
from src.polyagamma_jax import sample_pg_batch


# ========================= Config =========================

@dataclass
class InferenceTrialsEMConfig:
    fixed_iter: int = 150
    beta0_window: int = 100
    n_refreshes: int = 3
    inner_steps_per_refresh: int = 100
    omega_floor: float = 1e-3
    sigma_u: float = 0.05
    # β/γ priors for the trial β sampler
    tau2_intercept: float = 100.0 ** 2
    tau2_gamma: float = 25.0 ** 2
    a0_ard: float = 1e-2
    b0_ard: float = 1e-2
    use_exact_cov: bool = False
    # pooling flags in KF refresh
    pool_lfp_trials: bool = True
    pool_spike_trials: bool = True
    # EM settings
    em_kwargs: Dict[str, Any] = None            # e.g. dict(max_iter=20000, tol=1e-3, sig_eps_init=5.0, log_every=1000)
    # Sampler choice
    pg_jax: bool = False                        # if True, use JAX-based PG sampler (faster)
    # RNG
    key_jax: Optional["jr.KeyArray"] = None


# ========================= EM → θ, latents adapters =========================

def _theta_from_em(res, J: int, M: int, Rtr: int) -> Tuple[OUParams, np.ndarray]:
    """
    Build OUParams (pooled σ_ε for the core) and per-trial σ_ε from EM output.
    Expects fields like: lam_X (J,), sigv_X (J,), sig_eps_jmr (J,M,R) or sig_eps_mr (M,R).
    """
    # λ, σ_v for the shared trajectory X
    lam = np.asarray(getattr(res, "lam_X"), float).reshape(J, 1)          # (J,1)
    sigv = np.asarray(getattr(res, "sigv_X"), float).reshape(J, 1)        # (J,1)

    # σ_ε per trial/taper → pool to a single (J,M) for the core; also return per-trial (R,J,M)
    if hasattr(res, "sig_eps_jmr"):
        sig_eps_jmr = np.asarray(res.sig_eps_jmr, float)                  # (J,M,R) or (1,M,R)
        if sig_eps_jmr.shape[0] == 1:  # shared across bands? broadcast
            sig_eps_jmr = np.broadcast_to(sig_eps_jmr, (J, sig_eps_jmr.shape[1], sig_eps_jmr.shape[2]))
        sig_eps_trials = np.moveaxis(sig_eps_jmr, 2, 0)                   # (R,J,M)
    elif hasattr(res, "sig_eps_mr"):
        sig_eps_mr = np.asarray(res.sig_eps_mr, float)                    # (M,R)
        sig_eps_trials = np.broadcast_to(sig_eps_mr.T[:, None, :], (Rtr, J, M))
    else:
        # fallback: use a mild constant (shouldn't happen if EM ran)
        sig_eps_trials = np.full((Rtr, J, M), 5.0, float)

    # precision-pooled σ_ε over trials (per j,m): (∑ 1/σ²)^(-1/2)
    var_rm = sig_eps_trials ** 2
    w_rm = 1.0 / np.maximum(var_rm, 1e-20)
    var_pool = 1.0 / np.maximum(w_rm.sum(axis=0), 1e-20)                  # (J,M)
    sig_eps_pool = np.sqrt(var_pool)

    theta0 = OUParams(lam=lam, sig_v=sigv, sig_eps=sig_eps_pool)          # for the pooled core
    return theta0, sig_eps_trials                                        # (OUParams, (R,J,M))

def _extract_from_upsampled(upsampled, J: int, M: int, *, which: str = "X"):
    """
    Convert UpsampleResult -> (mu_fine, var_fine) with shapes (T_f, 2*J*M),
    ordered to match joint_kf_rts_moments (base=(j*M+m)*2 then [Re, Im]).

    which: "X" uses condition-level latent (shared across trials);
           "Z" uses per-trial combined latent averaged across trials.
    """
    # Choose source arrays
    if which.upper() == "X":
        mean_cplx = np.asarray(upsampled.X_mean)   # (J, M, T_f) complex
        var_real  = np.asarray(upsampled.X_var)    # (J, M, T_f) real (variance of complex)
    elif which.upper() == "Z":
        # Z_mean has trials: (R, J, M, T_f) → average across trials for a shared predictor
        Zm = np.asarray(upsampled.Z_mean)          # (R, J, M, T_f) complex
        Zv = np.asarray(upsampled.Z_var)           # (R, J, M, T_f) real
        mean_cplx = Zm.mean(axis=0)                # (J, M, T_f)
        var_real  = Zv.mean(axis=0)                # (J, M, T_f)
    else:
        raise ValueError("which must be 'X' or 'Z'")

    assert mean_cplx.shape == (J, M, mean_cplx.shape[2]), "Unexpected X/Z shapes"
    T_f = mean_cplx.shape[2]

    # Build (T_f, 2*J*M) in the core's [Re,Im] per-(j,m) C-order
    mu_fine  = np.zeros((T_f, 2 * J * M), dtype=np.float64)
    var_fine = np.zeros((T_f, 2 * J * M), dtype=np.float64)

    # Rearrange: (J, M, T) -> (T, J, M)
    Re = mean_cplx.real.transpose(2, 0, 1)  # (T, J, M)
    Im = mean_cplx.imag.transpose(2, 0, 1)  # (T, J, M)
    Vr = var_real.transpose(2, 0, 1)        # (T, J, M)  (scalar complex variance)

    # Fill flat arrays
    idx = 0
    for j in range(J):
        for m in range(M):
            mu_fine[:, idx]   = Re[:, j, m]
            var_fine[:, idx]  = Vr[:, j, m]
            mu_fine[:, idx+1] = Im[:, j, m]
            var_fine[:, idx+1]= Vr[:, j, m]
            idx += 2

    return mu_fine, var_fine  # (T_f, 2*J*M)

def _reim_from_fine(mu_fine: np.ndarray, var_fine: np.ndarray, J: int, M: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (T, 2J) taper-AVERAGED regressors and their variances from (T, 2JM).
    Matches the /M scaling used in spike rows inside the core.
    """
    T, d = mu_fine.shape
    tmp = mu_fine.reshape(T, J, M, 2)
    vmp = var_fine.reshape(T, J, M, 2)

    mu_re = tmp[..., 0].mean(axis=2)           # (T, J)
    mu_im = tmp[..., 1].mean(axis=2)           # (T, J)
    var_re = (vmp[..., 0].mean(axis=2)) / M    # conservative
    var_im = (vmp[..., 1].mean(axis=2)) / M

    latent_reim = np.concatenate([mu_re, mu_im], axis=1)   # (T, 2J)
    var_lat     = np.concatenate([var_re, var_im], axis=1) # (T, 2J)
    return latent_reim, var_lat


# ========================= Main runner =========================

def run_joint_inference_trials(
    Y_trials: np.ndarray,            # (R, J, M, K) complex; already subselected to coupled bands
    spikes_SRT: np.ndarray,          # (S, R, T_f)
    H_SRTL: np.ndarray,              # (S, R, T_f, Rlags)
    all_freqs: Sequence[float],      # (J,)
    *,
    delta_spk: float,
    window_sec: float,
    offset_sec: float = 0.0,
    beta_init: Optional[np.ndarray] = None,  # (S, 1+2J)
    gamma_prior_mu: Optional[np.ndarray] = None,     # (Rlags,) or (S,Rlags) or (S,R,Rlags)
    gamma_prior_Sigma: Optional[np.ndarray] = None,  # (Rlags,Rlags) or (S,...) or (S,R,...)
    config: Optional[InferenceTrialsEMConfig] = None,
    rng_pg: np.random.Generator = np.random.default_rng(0),
) -> Tuple[np.ndarray, np.ndarray, OUParams, Trace]:
    """
    EM-initialised trial-aware joint inference:
      1) em_ct_hier_jax → θ (lam_X, sigv_X, σ_ε per trial) on (R,J,M,K)
      2) upsample_ct_hier_fine → fine-grid (mu_fine, var_fine)
      3) PG–Gibbs warmup (shared β per unit) + refresh passes with pooled KF

    Returns:
      beta:  (S, 1+2J)
      gamma: (S, R, Rlags)
      theta: OUParams (pooled σ_ε), used by KF refresh
      trace: Trace (latents & params per refresh)
    """
    if config is None:
        config = InferenceTrialsEMConfig()
    key = config.key_jax or jr.PRNGKey(0)

    Rtr, J, M, K = Y_trials.shape
    S, R, T_f = spikes_SRT.shape
    assert Rtr == R, "Y_trials and spikes must have the same trial count"
    Rlags = H_SRTL.shape[-1]
    P = 1 + 2 * J

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
            # Use JAX-based sampler
            key_pg_jax, subkey = jr.split(key_pg_jax)
            psi_jax = jnp.asarray(psi)
            samples = sample_pg_batch(subkey, psi_jax, h=1.0)
            return np.asarray(samples)
        else:
            # Use original numpy-based sampler
            return sample_polya_gamma(np.asarray(psi), rng_pg)

    # Print startup info
    print(f"[EM-init Trial inference] R={R} trials, S={S} units, J={J} bands, M={M} tapers")
    print(f"[EM-init Trial inference] T_f={T_f}, Rlags={Rlags}")
    print(f"[EM-init Trial inference] Running EM initialization...")

    # ---------- 0) EM on trials → θ and σ_ε per trial ----------
    from src.em_ct_hier_jax import em_ct_hier_jax
    em_kwargs = dict(max_iter=5000, tol=1e-3, sig_eps_init=5.0, log_every=1000)
    if config.em_kwargs:
        em_kwargs.update(config.em_kwargs)

    res = em_ct_hier_jax(Y_trials=Y_trials, db=window_sec, **em_kwargs)

    theta, sig_eps_trials = _theta_from_em(res, J=J, M=M, Rtr=Rtr)   # OUParams, (R,J,M)
    print(f"[EM-init Trial inference] EM completed. Starting upsampling...")

    # ---------- 1) Upsample EM latents to fine grid ----------
    from src.upsample_ct_hier_fine import upsample_ct_hier_fine
    ups = upsample_ct_hier_fine(
        Y_trials=Y_trials, res=res,
        delta_spk=delta_spk, win_sec=window_sec, offset_sec=offset_sec, T_f=None,
    )
    mu_fine0, var_fine0 = _extract_from_upsampled(ups, J=J, M=M)     # (T_f, 2JM)
    lat_reim_np, var_reim_np = _reim_from_fine(mu_fine0, var_fine0, J=J, M=M)
    T0 = min(T_f, lat_reim_np.shape[0])
    lat_reim_np = lat_reim_np[:T0]
    var_reim_np = var_reim_np[:T0]
    spikes_SRT  = spikes_SRT[:, :, :T0]
    H_SRTL      = H_SRTL[:, :, :T0, :]

    print(f"[EM-init Trial inference] Upsampling completed. Starting Gibbs sampling...")
    print(f"[EM-init Trial inference] Warm-up: {config.fixed_iter} iterations")
    print(f"[EM-init Trial inference] Refresh: {config.n_refreshes} passes × {config.inner_steps_per_refresh} steps")

    # Stable slices (for Gibbs cache hits - critical for JAX performance!)
    lat_slice = jnp.asarray(lat_reim_np[:T0])                                       # (T0, 2J) JAX array
    var_slice = np.ascontiguousarray(var_reim_np[:T0], dtype=np.float64)           # (T0, 2J)
    X_slice = np.ascontiguousarray(
        np.concatenate([np.ones((T0, 1)), np.asarray(lat_slice)], axis=1),
        dtype=np.float64
    )  # (T0, 1+2J)
    spikes_slices = [jnp.asarray(spikes_SRT[s, :, :T0]) for s in range(S)]         # list of (R, T0)
    H_slices = [np.asarray(H_SRTL[s, :, :T0, :]) for s in range(S)]                # list of (R, T0, Rlags)

    # ---------- 2) Initialise β, γ ----------
    if beta_init is None:
        beta = np.zeros((S, P), float)
    else:
        beta = np.asarray(beta_init, float)
        assert beta.shape == (S, P)

    if gamma_prior_mu is None:
        gamma = np.zeros((S, R, Rlags), float)
    else:
        if gamma_prior_mu.ndim == 1:
            gamma = np.broadcast_to(gamma_prior_mu[None, None, :], (S, R, Rlags))
        elif gamma_prior_mu.ndim == 2:
            gamma = np.broadcast_to(gamma_prior_mu[:, None, :], (S, R, Rlags))
        else:
            gamma = np.asarray(gamma_prior_mu, float)

    # ---------- 3) Trace (initialized after stable slices created) ----------
    trace = Trace()
    trace.theta.append(theta)
    trace.latent.append(lat_slice)      # Use stable JAX array
    trace.fine_latent.append(mu_fine0)  # store full (2JM) fine latent

    # ---------- 4) WARMUP: PG–Gibbs with shared β across trials ----------
    tb_cfg = TrialBetaConfig(
        omega_floor=config.omega_floor,
        tau2_intercept=config.tau2_intercept,
        tau2_gamma=config.tau2_gamma,
        a0_ard=config.a0_ard,
        b0_ard=config.b0_ard,
        use_exact_cov=config.use_exact_cov,
    )
    beta_hist = []; gamma_hist = []

    pbar_warm = tqdm(range(config.fixed_iter), unit="it", desc="Warm-up (β/γ shared per unit)", mininterval=0.3)
    for it in pbar_warm:
        # sample ω_{s,r,t} - use stable arrays for cache hits
        omega_SRT = np.zeros((S, R, T0), float)
        for s in range(S):
            for r in range(R):
                psi = X_slice @ beta[s] + H_slices[s][r] @ gamma[s, r]
                omega_SRT[s, r] = np.maximum(sample_pg_wrapper(psi), config.omega_floor)

        # β/γ update (shared β across trials) per unit - use STABLE slices
        for s in range(S):
            _, beta_s, gamma_sr, _ = gibbs_update_beta_trials_shared(
                key,
                latent_reim=lat_slice,            # STABLE JAX array (T0, 2J)
                spikes=spikes_slices[s],          # STABLE JAX array (R, T0)
                omega=jnp.asarray(omega_SRT[s]),  # (R, T0)
                H_hist=H_slices[s],               # STABLE array (R, T0, Rlags)
                Sigma_gamma=(gamma_prior_Sigma[s] if (gamma_prior_Sigma is not None and gamma_prior_Sigma.ndim >= 2) else None),
                mu_gamma=(gamma_prior_mu[s] if (gamma_prior_mu is not None and gamma_prior_mu.ndim >= 2) else None),
                var_latent_reim=var_slice,        # STABLE array (T0, 2J)
                tau2_lat=None,
                config=tb_cfg,
            )
            beta[s]  = np.asarray(beta_s)
            gamma[s] = np.asarray(gamma_sr)

        beta_hist.append(beta.copy())
        gamma_hist.append(gamma.copy())

    beta_hist = np.stack(beta_hist, axis=0)      # (fixed_iter, S, P)
    gamma_hist = np.stack(gamma_hist, axis=0)    # (fixed_iter, S, R, Rlags)

    # Freeze β0 per unit
    w = min(config.beta0_window, config.fixed_iter)
    beta0_fixed = np.median(beta_hist[-w:, :, 0], axis=0)    # (S,)
    beta[:, 0] = beta0_fixed

    print(f"[EM-init Trial inference] Warm-up completed. β0 frozen per unit.")

    # γ posteriors per (s,r)
    mu_g_post = gamma_hist.mean(axis=0)                      # (S,R,Rlags)
    Sig_g_post = np.zeros((S, R, Rlags, Rlags), float)
    for s in range(S):
        for r in range(R):
            gh = gamma_hist[:, s, r, :]
            ctr = gh - gh.mean(axis=0, keepdims=True)
            Sg  = (ctr.T @ ctr) / max(gh.shape[0]-1, 1)
            Sig_g_post[s, r] = Sg + 1e-6 * np.eye(Rlags)

    # tight lock for inner steps
    Sig_g_lock = np.zeros_like(Sig_g_post)
    for s in range(S):
        for r in range(R):
            d = np.clip(np.diag(Sig_g_post[s, r]), 1e-10, None)
            Sig_g_lock[s, r] = np.diag(1e-6 * d)

    # ---------- 5) Refresh passes ----------
    sidx = StateIndex(J, M)

    pbar_pass = tqdm(range(config.n_refreshes), unit="pass", desc="Latent refresh passes", mininterval=0.3)
    for rr in pbar_pass:
        inner_beta_hist = []
        inner_gamma_hist = []

        # Inner steps with simplified progress (only show refresh pass)
        for it in range(config.inner_steps_per_refresh):
            # sample γ from posterior
            gamma_samp = np.zeros_like(gamma)
            for s in range(S):
                for r in range(R):
                    L = np.linalg.cholesky(Sig_g_post[s, r])
                    gamma_samp[s, r] = mu_g_post[s, r] + L @ np.random.randn(Rlags)

            # sample ω given γ_samp - use stable arrays
            omega_SRT = np.zeros((S, R, T0), float)
            for s in range(S):
                for r in range(R):
                    psi = X_slice @ beta[s] + H_slices[s][r] @ gamma_samp[s, r]
                    omega_SRT[s, r] = np.maximum(sample_pg_wrapper(psi), config.omega_floor)

            # β update with tight γ lock - use STABLE slices
            for s in range(S):
                _, beta_s, gamma_sr, _ = gibbs_update_beta_trials_shared(
                    key,
                    latent_reim=lat_slice,            # STABLE JAX array
                    spikes=spikes_slices[s],          # STABLE JAX array
                    omega=jnp.asarray(omega_SRT[s]),  # (R, T0)
                    H_hist=H_slices[s],               # STABLE array
                    Sigma_gamma=Sig_g_lock[s],
                    mu_gamma=gamma_samp[s],
                    var_latent_reim=var_slice,        # STABLE array
                    tau2_lat=None,
                    config=tb_cfg,
                )
                beta[s]  = np.asarray(beta_s)
                gamma[s] = np.asarray(gamma_sr)

            beta[:, 0] = beta0_fixed
            inner_beta_hist.append(beta.copy())
            inner_gamma_hist.append(gamma.copy())

        inner_beta_hist = np.stack(inner_beta_hist, axis=0)
        inner_gamma_hist = np.stack(inner_gamma_hist, axis=0)
        beta_median = np.median(inner_beta_hist, axis=0)     # (S,P)

        # KF refresh on pooled trials
        mom = joint_kf_rts_moments_trials(
            Y_cube=Y_trials, theta=theta,
            delta_spk=delta_spk, win_sec=window_sec, offset_sec=offset_sec,
            beta=beta_median, gamma=gamma, spikes=spikes_SRT, omega=omega_SRT,
            coupled_bands_idx=np.arange(J, dtype=np.int64),
            freqs_for_phase=np.asarray(all_freqs, float),
            sidx=sidx, H_hist=H_SRTL,
            sigma_u=config.sigma_u,
            sig_eps_trials=sig_eps_trials,     # use EM per-trial σ_ε
            pool_lfp_trials=config.pool_lfp_trials,
            pool_spike_trials=config.pool_spike_trials,
        )

        # rebuild regressors from smoothed latents
        lat_reim_np, var_reim_np = _reim_from_fine(mom.m_s, mom.P_s, J=J, M=M)
        T0 = min(T0, lat_reim_np.shape[0])
        lat_reim_np = lat_reim_np[:T0]
        var_reim_np = var_reim_np[:T0]

        # Rebuild STABLE slices for next iteration (critical for JAX cache!)
        lat_slice = jnp.asarray(lat_reim_np)                                        # (T0, 2J) JAX array
        var_slice = np.ascontiguousarray(var_reim_np, dtype=np.float64)            # (T0, 2J)
        X_slice = np.ascontiguousarray(
            np.concatenate([np.ones((T0, 1)), np.asarray(lat_slice)], axis=1),
            dtype=np.float64
        )  # (T0, 1+2J)
        # Note: spikes_slices and H_slices remain unchanged

        # trace bookkeeping
        trace.theta.append(theta)
        trace.latent.append(lat_slice)  # Use stable slice instead of creating new JAX array
        trace.fine_latent.append(mom.m_s)
        for i in range(config.inner_steps_per_refresh):
            trace.beta.append(inner_beta_hist[i])
            trace.gamma.append(inner_gamma_hist[i])

    print(f"[EM-init Trial inference] Inference completed!")
    return beta, gamma, theta, trace
