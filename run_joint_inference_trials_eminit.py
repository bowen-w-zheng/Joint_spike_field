# run_joint_inference_trials_eminit.py
# Trial-aware runner with EM warm-start + fully JAX-native PG–Gibbs (β shared across trials, γ shared across trials).
# Exact pooled-KF via κ′ trick (math-faithful), single JIT compile for warmup and inner loops.

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Dict, Any
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap, lax
from functools import partial

from src.params import OUParams
from src.utils_joint import Trace
from src.state_index import StateIndex
from src.priors import gamma_prior_simple
from src.polyagamma_jax import sample_pg_saddle_single
# JAX-native trial β/γ sampler (β shared across trials, γ shared across trials)
from src.beta_sampler_trials_jax import (
    build_design_jax,
    gibbs_update_beta_trials_shared_gamma_jax,   # (S-vectorized)
    TrialBetaJAXConfig,
)

# Math-exact pooled KF wrapper (κ′+H=0; LFP precision pooling)
from src.joint_inference_core_trial_fast import joint_kf_rts_moments_trials_fast


# ========================= Config =========================

@dataclass
class InferenceTrialsEMConfig:
    fixed_iter: int = 150
    beta0_window: int = 100
    n_refreshes: int = 3
    inner_steps_per_refresh: int = 100
    omega_floor: float = 1e-3
    sigma_u: float = 0.05
    # EM settings
    em_kwargs: Dict[str, Any] = None            # e.g. dict(max_iter=20000, tol=1e-3, sig_eps_init=5.0, log_every=1000)
    # RNG / determinism
    key_jax: Optional["jr.KeyArray"] = None
    # Tracing (off by default to save memory)
    store_inner_history: bool = False


# ========================= EM → θ, latents adapters =========================

def _theta_from_em(res, J: int, M: int, Rtr: int) -> Tuple[OUParams, np.ndarray]:
    """Build OUParams (pooled σ_ε) and per-trial σ_ε from EM output."""
    lam = np.asarray(getattr(res, "lam_X"), float).reshape(J, 1)
    sigv = np.asarray(getattr(res, "sigv_X"), float).reshape(J, 1)

    if hasattr(res, "sig_eps_jmr"):
        sig_eps_jmr = np.asarray(res.sig_eps_jmr, float)          # (J,M,R) or (1,M,R)
        if sig_eps_jmr.shape[0] == 1:
            sig_eps_jmr = np.broadcast_to(sig_eps_jmr, (J, sig_eps_jmr.shape[1], sig_eps_jmr.shape[2]))
        sig_eps_trials = np.moveaxis(sig_eps_jmr, 2, 0)           # (R,J,M)
    elif hasattr(res, "sig_eps_mr"):
        sig_eps_mr = np.asarray(res.sig_eps_mr, float)            # (M,R)
        sig_eps_trials = np.broadcast_to(sig_eps_mr.T[:, None, :], (Rtr, J, M))
    else:
        sig_eps_trials = np.full((Rtr, J, M), 5.0, float)

    # pooled per (j,m)
    var_rm = sig_eps_trials ** 2
    w_rm = 1.0 / np.maximum(var_rm, 1e-20)
    var_pool = 1.0 / np.maximum(w_rm.sum(axis=0), 1e-20)
    sig_eps_pool = np.sqrt(var_pool)

    theta0 = OUParams(lam=lam, sig_v=sigv, sig_eps=sig_eps_pool)
    return theta0, sig_eps_trials


def _extract_from_upsampled(ups, J: int, M: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Use condition-level latent X for shared predictors → (T_f, 2JM) [Re,Im] per (j,m), C-ordered.
    """
    mean_cplx = np.asarray(ups.X_mean)  # (J,M,Tf) complex
    var_real  = np.asarray(ups.X_var)   # (J,M,Tf) real
    Tf = mean_cplx.shape[2]
    mu = np.zeros((Tf, 2*J*M), float)
    var = np.zeros((Tf, 2*J*M), float)
    Re = mean_cplx.real.transpose(2,0,1); Im = mean_cplx.imag.transpose(2,0,1); Vr = var_real.transpose(2,0,1)
    idx = 0
    for j in range(J):
        for m in range(M):
            mu[:, idx]   = Re[:, j, m]; var[:, idx]   = Vr[:, j, m]
            mu[:, idx+1] = Im[:, j, m]; var[:, idx+1] = Vr[:, j, m]
            idx += 2
    return mu, var


def _reim_from_fine(mu_f: np.ndarray, var_f: np.ndarray, J: int, M: int) -> Tuple[np.ndarray, np.ndarray]:
    """(T,2JM) → (T,2J) taper-AVERAGED; conservative var for EIV."""
    T, d = mu_f.shape
    tmp = mu_f.reshape(T, J, M, 2)
    vmp = var_f.reshape(T, J, M, 2)
    mu_re = tmp[..., 0].mean(axis=2); mu_im = tmp[..., 1].mean(axis=2)
    vr_re = vmp[..., 0].mean(axis=2) / M; vr_im = vmp[..., 1].mean(axis=2) / M
    return np.concatenate([mu_re, mu_im], axis=1), np.concatenate([vr_re, vr_im], axis=1)


# ========================= JAX helpers =========================

@jax.jit
def _psi_all(beta_SP: jnp.ndarray, gamma_SL: jnp.ndarray, X: jnp.ndarray, H_SRTL: jnp.ndarray) -> jnp.ndarray:
    """
    beta_SP: (S,P), gamma_SL: (S,L), X:(T,P), H_SRTL:(S,R,T,L)
    returns ψ_SRT:(S,R,T) = X@β_s + H_{s,r} @ γ_s
    """
    def psi_unit(b_s, g_s, H_rtl_s):
        xb = X @ b_s                     # (T,)
        Hg = jnp.einsum('rtl,l->rt', H_rtl_s, g_s)  # (R,T)
        return Hg + xb[None, :]
    return vmap(psi_unit)(beta_SP, gamma_SL, H_SRTL)


@jax.jit
def _sample_omega_pg_batch(key: jr.KeyArray, psi: jnp.ndarray, omega_floor: float) -> jnp.ndarray:
    """psi:(T,) → ω:(T,) PG(1,ψ) with floor."""
    keys = jr.split(key, psi.shape[0])
    om = vmap(lambda k, z: jax.lax.cond(
        jnp.isfinite(z), lambda kk, zz: jnp.maximum(jax.vmap(lambda k1: jax.jit(lambda z1: jax.tree_util.Partial(lambda: None)))(kk), 0.0),  # never taken; placeholder
        lambda kk, zz: 0.0, keys, z
    ))  # no-op (we'll use fast saddle sampler below to avoid control flow; see wrapper below)
    # -> replace with fast saddle:
    om = vmap(lambda k, z: sample_pg_saddle_single(k, 1.0, z))(keys, psi)
    return jnp.maximum(om, omega_floor)


@jax.jit
def sample_omega_all(rng: jr.KeyArray, psi_srt: jnp.ndarray, omega_floor: float) -> jnp.ndarray:
    """
    Vectorized PolyGamma(1, ψ) draws for all entries of psi_srt, fully on device.
    psi_srt: (S, R, T) log-odds array
    Returns ω_srt: (S, R, T)
    """
    S, R, T = psi_srt.shape  # these dims are static inside the jit-compiled loops

    # allocate one (2,) key per (s, r, t)
    keys_srt = jr.split(rng, S * R * T).reshape((S, R, T, 2))  # (S,R,T,2)

    def sample_time(keys_t, psi_t):
        # keys_t: (T, 2), psi_t: (T,)
        return jax.vmap(lambda k_t, z_t: jnp.maximum(
            # sample_pg_saddle_single expects (2,), scalar z
            sample_pg_saddle_single(k_t, 1.0, z_t),
            omega_floor
        ))(keys_t, psi_t)  # -> (T,)

    # map over R and T for each s; result shape (S,R,T)
    omega_srt = jax.vmap(lambda keys_rt, psi_rt: jax.vmap(
        sample_time, in_axes=(0, 0), out_axes=0
    )(keys_rt, psi_rt), in_axes=(0, 0), out_axes=0)(keys_srt, psi_srt)
    return omega_srt


# ========================= Main runner =========================

def run_joint_inference_trials(
    Y_trials: np.ndarray,            # (R,J,M,K) complex, already band-selected
    spikes_SRT: np.ndarray,          # (S,R,Tf)
    H_SRTL: np.ndarray,              # (S,R,Tf,L)
    all_freqs: Sequence[float],      # (J,)
    *,
    delta_spk: float,
    window_sec: float,
    offset_sec: float = 0.0,
    beta_init: Optional[np.ndarray] = None,  # (S, 1+2J)
    gamma_prior_mu: Optional[np.ndarray] = None,     # (L,) or (S,L)
    gamma_prior_Sigma: Optional[np.ndarray] = None,  # (L,L) or (S,L,L)
    config: Optional[InferenceTrialsEMConfig] = None,
) -> Tuple[np.ndarray, np.ndarray, OUParams, Trace]:
    """
    Returns:
      beta  : (S, 1+2J)  (shared across trials per unit)
      gamma : (S, L)     (shared across trials per unit)
      theta : OUParams   (pooled σ_ε for KF)
      trace : Trace
    """
    if config is None:
        config = InferenceTrialsEMConfig()
    key = config.key_jax or jr.PRNGKey(0)

    # Shapes
    R, J, M, K = Y_trials.shape
    S, R2, Tf = spikes_SRT.shape
    assert R == R2, "Y_trials and spikes must share the same trial count"
    L = H_SRTL.shape[-1]
    P = 1 + 2*J

    print(f"[Trials] S={S}, R={R}, Tf={Tf}, J={J}, M={M}, L={L}")

    # ---------- EM warm-start ----------
    from src.em_ct_hier_jax import em_ct_hier_jax
    em_kwargs = dict(max_iter=20000, tol=1e-3, sig_eps_init=5.0, log_every=1000)
    if config.em_kwargs:
        em_kwargs.update(config.em_kwargs)

    res = em_ct_hier_jax(Y_trials=Y_trials, db=window_sec, **em_kwargs)
    theta, sig_eps_trials = _theta_from_em(res, J, M, R)

    # ---------- Upsample condition-level X to fine grid ----------
    from src.upsample_ct_hier_fine import upsample_ct_hier_fine
    ups = upsample_ct_hier_fine(
        Y_trials=Y_trials, res=res,
        delta_spk=delta_spk, win_sec=window_sec, offset_sec=offset_sec, T_f=None
    )
    mu_fine0, var_fine0 = _extract_from_upsampled(ups, J, M)          # (Tf*, 2JM)
    lat_np, var_np = _reim_from_fine(mu_fine0, var_fine0, J, M)       # (Tf*, 2J)
    T0 = min(Tf, lat_np.shape[0])
    lat_np, var_np = lat_np[:T0], var_np[:T0]
    spikes_SRT = spikes_SRT[:, :, :T0]
    H_SRTL     = H_SRTL[:, :, :T0, :]

    # ---------- Shared design (JAX constants) ----------
    X = build_design_jax(jnp.asarray(lat_np))       # (T0,P)
    V = jnp.asarray(var_np)                          # (T0,2B)
    # Tile V to (S,R,T0,2B) for the sampler (exact EIV sum over trials)
    V_SRTB = jnp.tile(V[None, None, :, :], (S, R, 1, 1))

    # ---------- Priors ----------
    if gamma_prior_mu is None or gamma_prior_Sigma is None:
        mu_g, Sig_g = gamma_prior_simple(n_lags=L, strong_neg=-2.5, mild_neg=-0.5, k_mild=4, tau_gamma=1.5)
        mu_gamma_S_l  = jnp.tile(jnp.asarray(mu_g)[None, :], (S, 1))         # (S,L)
        Prec_gamma_S_ll = jnp.tile(jnp.linalg.pinv(jnp.asarray(Sig_g))[None, :, :], (S, 1, 1))  # (S,L,L)
    else:
        if gamma_prior_mu.ndim == 1:
            mu_gamma_S_l  = jnp.tile(jnp.asarray(gamma_prior_mu)[None, :], (S, 1))
        else:
            mu_gamma_S_l  = jnp.asarray(gamma_prior_mu)
        if gamma_prior_Sigma.ndim == 2:
            Prec_gamma_S_ll = jnp.tile(jnp.linalg.pinv(jnp.asarray(gamma_prior_Sigma))[None, :, :], (S, 1, 1))
        else:
            Prec_gamma_S_ll = jnp.asarray([jnp.linalg.pinv(jnp.asarray(Sigma)) for Sigma in gamma_prior_Sigma])

    # ---------- Init state ----------
    if beta_init is None:
        beta = jnp.zeros((S, P), dtype=jnp.float64)
    else:
        beta = jnp.asarray(beta_init, dtype=jnp.float64)
    gamma = jnp.zeros((S, L), dtype=jnp.float64)  # shared across trials per unit
    tau2  = jnp.ones((S, P-1), dtype=jnp.float64) # (S,2B)

    # ---------- Trace ----------
    trace = Trace()
    trace.theta.append(theta)
    trace.latent.append(jnp.asarray(lat_np))
    trace.fine_latent.append(mu_fine0)

    # ---------- Warmup (single JIT scan) ----------
    cfg_pg = TrialBetaJAXConfig(omega_floor=config.omega_floor,
                                tau2_intercept=1e4, a0_ard=1e-2, b0_ard=1e-2)

    @partial(jax.jit, static_argnames=['n_iter'])
    def _warmup_scan(key, beta0, gamma0, tau20, X, H, SRT_spk, V_SRTB, Prec_g, mu_g, n_iter: int):
        def body(carry, k):
            b, g, t2 = carry
            psi = _psi_all(b, g, X, H)                                 # (S,R,T0)
            # PG sampling (vectorized)
            ω = sample_omega_all(k, psi, cfg_pg.omega_floor)                                       # (S,R,T0)
            # β/γ shared across trials per unit
            k2, b_new, g_new, t2_new = gibbs_update_beta_trials_shared_gamma_jax(
                k, X, H, SRT_spk, ω, V_SRTB, Prec_g, mu_g, t2, cfg_pg
            )
            return (b_new, g_new, t2_new), (b_new, g_new)
        keys = jr.split(key, n_iter)
        return lax.scan(body, (beta0, gamma0, tau20), keys)

    key, key_w = jr.split(key)
    (beta, gamma, tau2), (beta_hist, gamma_hist) = _warmup_scan(
        key_w, beta, gamma, tau2, X, jnp.asarray(H_SRTL), jnp.asarray(spikes_SRT), V_SRTB,
        Prec_gamma_S_ll, mu_gamma_S_l,
        config.fixed_iter
    )

    # Freeze β0 (keep on device)
    w = min(config.beta0_window, config.fixed_iter)
    beta0_fixed = jnp.median(beta_hist[-w:, :, 0], axis=0)
    beta = beta.at[:, 0].set(beta0_fixed)

    # γ posterior from warmup history (host; off hot-path)
    gamma_hist_np = np.array(gamma_hist)                       # (it,S,L)
    mu_g_post = gamma_hist_np.mean(axis=0)                     # (S,L)
    Sig_g_post = np.zeros((S, L, L), float)
    for s in range(S):
        gh = gamma_hist_np[:, s, :]
        ctr = gh - gh.mean(axis=0, keepdims=True)
        Sig_g_post[s] = (ctr.T @ ctr) / max(gh.shape[0]-1, 1) + 1e-6*np.eye(L)
    mu_g_post_j  = jnp.asarray(mu_g_post)
    Sig_g_post_j = jnp.asarray(Sig_g_post)
    Prec_g_lock  = jnp.asarray(np.array([np.linalg.pinv(np.diag(1e-6*np.clip(np.diag(Sig_g_post[s]),1e-10,None)))
                                         for s in range(S)]))

    # ---------- Inner refresh scans ----------
    @partial(jax.jit, static_argnames=['n_iter'])
    def _inner_scan(key, beta0, gamma0, tau20, beta0_fixed, X, H, SRT_spk, V_SRTB,
                    Prec_g_lock, mu_g_post, Sig_g_post, n_iter: int):
        def sample_mvn(k, mu, Sig):
            Lc = jnp.linalg.cholesky(Sig + 1e-6*jnp.eye(Sig.shape[-1], dtype=Sig.dtype))
            z  = jr.normal(k, mu.shape, dtype=mu.dtype)
            return mu + Lc @ z
        def body(carry, k):
            b, g, t2 = carry
            # sample γ from posterior (shared per unit)
            keys_g = jr.split(k, S)
            g_samp = vmap(sample_mvn)(keys_g, mu_g_post, Sig_g_post)        # (S,L)
            psi = _psi_all(b, g_samp, X, H)                                 # (S,R,T0)
            # PG
            ω = sample_omega_all(jr.fold_in(k, 1), psi, cfg_pg.omega_floor)                                           # (S,R,T0)
            # β update with tight γ lock around g_samp
            k2, b_new, g_lock, t2_new = gibbs_update_beta_trials_shared_gamma_jax(
                k, X, H, SRT_spk, ω, V_SRTB, Prec_g_lock, g_samp, t2, cfg_pg
            )
            b_new = b_new.at[:, 0].set(beta0_fixed)  # enforce freeze
            return (b_new, g_samp, t2_new), (b_new, g_samp)
        keys = jr.split(key, n_iter)
        return lax.scan(body, (beta0, gamma0, tau20), keys)

    # Refresh passes
    trace_beta_cache = []
    trace_gamma_cache = []
    for rr in range(config.n_refreshes):
        key, key_inner = jr.split(key)
        (beta, gamma, tau2), (betaH, gammaH) = _inner_scan(
            key_inner, beta, gamma, tau2, beta0_fixed, X, jnp.asarray(H_SRTL),
            jnp.asarray(spikes_SRT), V_SRTB, Prec_g_lock, mu_g_post_j, Sig_g_post_j,
            config.inner_steps_per_refresh
        )
        # robust β across inner (keep on device)
        beta_med = jnp.median(betaH, axis=0)                            # (S,P)

        # Build ω for KF refresh using β_med and γ̂ = μ_g_post
        psi_refresh = _psi_all(beta_med, mu_g_post_j, X, jnp.asarray(H_SRTL))  # (S,R,T0)
        key, key_w = jr.split(key)
        # Vectorized PG sampling (reuse compiled function)
        omega_refresh = sample_omega_all(key_w, psi_refresh, config.omega_floor)           # (S,R,T0)

        # Exact pooled-KF (κ′ trick; β shared per unit for exactness)
        mom = joint_kf_rts_moments_trials_fast(
            Y_trials=Y_trials, theta=theta,
            delta_spk=delta_spk, win_sec=window_sec, offset_sec=offset_sec,
            beta=np.expand_dims(np.array(beta_med), 1),            # (S,1,P) -> reduced inside
            gamma_shared=np.array(mu_g_post),                      # (S,L)
            spikes=np.array(spikes_SRT), omega=np.array(omega_refresh),
            coupled_bands_idx=np.arange(J, dtype=np.int64),
            freqs_for_phase=np.asarray(all_freqs, float),
            sidx=StateIndex(J, M),
            H_hist=np.array(H_SRTL), sigma_u=float(config.sigma_u),
            sig_eps_trials=sig_eps_trials
        )

        # Rebuild regressors (keep T0 fixed → no JIT recompile)
        lat_np, var_np = _reim_from_fine(mom.m_s, mom.P_s, J, M)
        lat_np, var_np = lat_np[:T0], var_np[:T0]
        X = build_design_jax(jnp.asarray(lat_np))
        V = jnp.asarray(var_np)
        V_SRTB = jnp.tile(V[None, None, :, :], (S, R, 1, 1))

        # Trace
        trace.theta.append(theta)
        trace.latent.append(jnp.asarray(lat_np))
        trace.fine_latent.append(mom.m_s)
        if config.store_inner_history:
            betaH_np = np.array(betaH)  # Convert only if storing history
            gammaH_np = np.array(gammaH)
            for i in range(config.inner_steps_per_refresh):
                trace_beta_cache.append(betaH_np[i])    # (S,P)
                trace_gamma_cache.append(gammaH_np[i])  # (S,L)

    # Final outputs (shared across trials per unit)
    beta_out  = np.array(beta)
    gamma_out = np.array(gamma)
    return beta_out, gamma_out, theta, trace
