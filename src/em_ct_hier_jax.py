# src/em_ct_hier_jax.py
from __future__ import annotations
import jax
import jax.numpy as jnp
from jax import vmap, lax
from dataclasses import dataclass
from typing import Tuple
from functools import partial

# 64-bit
jax.config.update("jax_enable_x64", True)
EPS = 1e-10


@dataclass
class EMHierResult:
    lam_X: jnp.ndarray
    sigv_X: jnp.ndarray
    lam_D: jnp.ndarray
    sigv_D: jnp.ndarray
    sig_eps_jmr: jnp.ndarray  # (J,M,R) or (1,M,R) if shared
    sig_eps_mr: jnp.ndarray   # (M,R)
    Q_hist: jnp.ndarray       # (max_iter,)
    X_mean: jnp.ndarray       # (J,M,K)
    D_mean: jnp.ndarray       # (R,J,M,K)
    # NEW: inferred initial states (smoothed at t=0)
    x0_X: jnp.ndarray         # (J,M) complex
    P0_X: jnp.ndarray         # (J,M) real
    x0_D: jnp.ndarray         # (R,J,M) complex
    P0_D: jnp.ndarray         # (R,J,M) real


def _normalize_to_RMJK(Y: jnp.ndarray) -> jnp.ndarray:
    if Y.ndim != 4:
        raise ValueError("Y must have 4 dims: (R,M,J,K) or (R,J,M,K)")
    if Y.shape[1] > 16 or Y.shape[2] < Y.shape[1]:
        return jnp.swapaxes(Y, 1, 2)
    return Y


@jax.jit
def build_pooled_X_obs(
    Y: jnp.ndarray,                   # (R,M,J,K)
    D_mean_local: jnp.ndarray,        # (R,J,M,K)
    sig2_eps_jmr_local: jnp.ndarray   # (J,M,R) or (1,M,R)
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    W_jmr = 1.0 / jnp.maximum(sig2_eps_jmr_local, EPS)
    Wsum_jm = jnp.sum(W_jmr, axis=2)

    D_rmjk = jnp.swapaxes(D_mean_local, 1, 2)            # (R,M,J,K)
    Y_minus_D = Y - D_rmjk

    W_rmj1 = jnp.transpose(W_jmr, (2, 1, 0))[:, :, :, None]  # (R,M,J,1)
    num_mjk = jnp.sum(W_rmj1 * Y_minus_D, axis=0)            # (M,J,K)

    Wsum_mj = jnp.transpose(jnp.maximum(Wsum_jm, EPS), (1, 0))   # (M,J)
    Y_pool_mjk = num_mjk / Wsum_mj[:, :, None]                   # (M,J,K)
    Y_pool = jnp.swapaxes(Y_pool_mjk, 0, 1)                      # (J,M,K)

    sig2_pool_jm = 1.0 / jnp.maximum(Wsum_jm, EPS)               # (J,M) or (1,M)
    if sig2_pool_jm.shape[0] == 1:
        sig2_pool_jm = jnp.tile(sig2_pool_jm, (Y_pool.shape[0], 1))
    return Y_pool, sig2_pool_jm


@jax.jit
def _rtss_ou_single_taper_jax(Y_m, phi_val, q_val, R_val, z0, P0):
    """Scalar complex OU RTS for one taper, with user-supplied z0,P0."""
    K = Y_m.shape[0]

    def filter_step(carry, y_k):
        z, P = carry
        z_pred = phi_val * z
        P_pred = phi_val * phi_val * P + q_val
        S = P_pred + R_val
        K_gain = P_pred / jnp.maximum(S, EPS)
        innov = y_k - z_pred
        z_filt = z_pred + K_gain * innov
        P_filt = (1.0 - K_gain) * P_pred
        return (z_filt, P_filt), (z_filt, P_filt, z_pred, P_pred)

    # use provided z0,P0
    (_, _), (xf, Pf, xp, Pp) = lax.scan(filter_step, (z0, P0), Y_m)

    def smooth_step(carry, inputs):
        xs_next, Ps_next = carry
        xf_k, Pf_k, P_pred_next = inputs
        J_k = Pf_k * phi_val / jnp.maximum(P_pred_next, EPS)
        xs_k = xf_k + J_k * (xs_next - phi_val * xf_k)
        Ps_k = Pf_k + J_k * J_k * (Ps_next - P_pred_next)
        return (xs_k, Ps_k), (xs_k, Ps_k, J_k)

    rev_inputs = (xf[:-1], Pf[:-1], Pp[1:])
    (_, _), (xs_smooth, Ps_smooth, J_all) = lax.scan(
        smooth_step, (xf[-1], Pf[-1]), rev_inputs, reverse=True
    )

    xs = jnp.concatenate([xs_smooth, xf[-1:]])   # (K,)
    Ps = jnp.concatenate([Ps_smooth, Pf[-1:]])   # (K,)
    Pcs = J_all * Ps[1:]

    Csum = jnp.sum(xs[1:] * jnp.conj(xs[:-1]) + Pcs)
    Rprev = jnp.sum(jnp.abs(xs[:-1])**2 + Ps[:-1])
    Rnext = jnp.sum(jnp.abs(xs[1:])**2 + Ps[1:])

    E_YY = jnp.sum(jnp.abs(Y_m)**2).real
    E_YZ = jnp.real(jnp.sum(Y_m * jnp.conj(xs)))
    E_ZZ = jnp.sum(jnp.abs(xs)**2 + Ps)
    E_dZ_dummy = jnp.float64(0.0)

    return xs, Ps, Csum, Rprev, Rnext, xp, Pp, E_YY, E_YZ, E_ZZ, E_dZ_dummy


# vmap over tapers: add z0, P0 per taper
_rtss_ou_all_tapers_jax = vmap(
    _rtss_ou_single_taper_jax,
    in_axes=(0, None, None, 0, 0, 0),
    out_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
)


@jax.jit
def _rtss_ou_single_freq_jax(Y_j, phi_j, q_j, R_j, z0_m, P0_m):
    """One band (J), all tapers (M) with user z0,P0 per taper."""
    xs, Ps, Csum_m, Rprev_m, Rnext_m, xp, Pp, E_YY_m, E_YZ_m, E_ZZ_m, _ = \
        _rtss_ou_all_tapers_jax(Y_j, phi_j, q_j, R_j, z0_m, P0_m)

    Csum = jnp.sum(Csum_m)
    Rprev = jnp.sum(Rprev_m)
    Rnext = jnp.sum(Rnext_m)
    E_YY = jnp.sum(E_YY_m)
    E_YZ = jnp.sum(E_YZ_m)
    E_ZZ = jnp.sum(E_ZZ_m)
    E_dZ_dummy = jnp.float64(0.0)
    return xs, Ps, Csum, Rprev, Rnext, xp, Pp, E_YY, E_YZ, E_ZZ, E_dZ_dummy


# vmap over bands: add z0_jm, P0_jm
_rtss_ou_all_freqs_jax = vmap(
    _rtss_ou_single_freq_jax,
    in_axes=(0, 0, 0, 0, 0, 0),
    out_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
)


@jax.jit
def _rtss_ou_jax(Y, lam, sig_v, sig_eps, db, z0_jm, P0_jm):
    """
    RTS smoother with user initial state per chain.
    Y: (J,M,K), lam/sig_v: (J,M), sig_eps: (J,M) or (1,M), z0_jm/P0_jm: (J,M)
    """
    J, M, K = Y.shape
    phi = jnp.exp(-lam[:, 0] * db)
    q = sig_v[:, 0] ** 2 * (1.0 - phi**2) / (2.0 * jnp.maximum(lam[:, 0], EPS))

    if sig_eps.shape[0] == 1:
        R = jnp.tile(sig_eps**2, (J, 1))
    else:
        R = sig_eps**2

    xs, Ps, Csum, Rprev, Rnext, xp, Pp, E_YY_all, E_YZ_all, E_ZZ_all, _ = \
        _rtss_ou_all_freqs_jax(Y, phi, q, R, z0_jm, P0_jm)

    E_YY = jnp.sum(E_YY_all)
    E_YZ = jnp.sum(E_YZ_all)
    E_ZZ = jnp.sum(E_ZZ_all)
    E_dZ_dummy = jnp.float64(0.0)
    return xs, Ps, Csum, Rprev, Rnext, xp, Pp, E_YY, E_YZ, E_ZZ, E_dZ_dummy


@jax.jit
def m_step_ou_params(Csum, Rprev, Rnext, K, db, n_chains):
    Phi = jnp.clip(jnp.real(Csum) / jnp.maximum(Rprev, EPS), EPS, 1.0 - EPS)
    lam_j = -jnp.log(Phi) / db
    num = Rnext - 2.0 * Phi * jnp.real(Csum) + (Phi**2) * Rprev
    Qhat = jnp.maximum(num, EPS) / ((K - 1) * n_chains)
    sigv_j = jnp.sqrt(2.0 * lam_j * Qhat / jnp.maximum(1.0 - Phi**2, EPS))
    return lam_j, sigv_j


@jax.jit
def m_step_obs_noise_shared(Y, xp_X, Pp_X, xp_D_all, Pp_D_all, noise_floor):
    Z_pred = xp_X[None, :, :, :] + xp_D_all
    Var_pred = (Pp_X[None, :, :, :] + Pp_D_all).real
    Z_pred_rmjk = jnp.swapaxes(Z_pred, 1, 2)
    Var_rmjk = jnp.swapaxes(Var_pred, 1, 2)

    resid = Y - Z_pred_rmjk
    moment = jnp.abs(resid)**2 + Var_rmjk
    S_rm = jnp.sum(moment, axis=(2, 3))
    S_mr = jnp.transpose(S_rm, (1, 0))
    JK = Y.shape[2] * Y.shape[3]
    sig2_eps_mr = jnp.maximum(S_mr / jnp.maximum(JK, 1), noise_floor)
    return sig2_eps_mr[None, :, :]


@jax.jit
def m_step_obs_noise_band_specific(Y, xp_X, Pp_X, xp_D_all, Pp_D_all, noise_floor):
    Z_pred = xp_X[None, :, :, :] + xp_D_all
    Var_pred = (Pp_X[None, :, :, :] + Pp_D_all).real
    Z_pred_rmjk = jnp.swapaxes(Z_pred, 1, 2)
    Var_rmjk = jnp.swapaxes(Var_pred, 1, 2)

    resid = Y - Z_pred_rmjk
    moment = jnp.abs(resid)**2 + Var_rmjk
    S_rmj = jnp.sum(moment, axis=3)
    S_jmr = jnp.transpose(S_rmj, (2, 1, 0))
    K = Y.shape[3]
    return jnp.maximum(S_jmr / jnp.maximum(K, 1), noise_floor)


def _compute_Q_monitor(
    Y, xp_X, Pp_X, xp_D_all, Pp_D_all, sig2_eps_jmr,
    Csum_X, Rprev_X, Rnext_X, lam_X_j, sigv_X_j,
    Csum_D, Rprev_D, Rnext_D, lam_D_j, sigv_D_j,
    db, M, R, K, obs_noise_shared
):
    Phi_X = jnp.exp(-lam_X_j * db)
    Qproc_X = sigv_X_j**2 * (1.0 - Phi_X**2) / (2.0 * jnp.maximum(lam_X_j, EPS))
    EdZ_X = Rnext_X - 2.0 * Phi_X * jnp.real(Csum_X) + (Phi_X**2) * Rprev_X
    Qterm_X = -0.5 * ((K - 1) * M) * jnp.sum(jnp.log(jnp.maximum(Qproc_X, EPS))) \
              -0.5 * jnp.sum(EdZ_X / jnp.maximum(Qproc_X, EPS))

    Phi_D = jnp.exp(-lam_D_j * db)
    Qproc_D = sigv_D_j**2 * (1.0 - Phi_D**2) / (2.0 * jnp.maximum(lam_D_j, EPS))
    EdZ_D = Rnext_D - 2.0 * Phi_D * jnp.real(Csum_D) + (Phi_D**2) * Rprev_D
    Qterm_D = -0.5 * ((K - 1) * M * R) * jnp.sum(jnp.log(jnp.maximum(Qproc_D, EPS))) \
              -0.5 * jnp.sum(EdZ_D / jnp.maximum(Qproc_D, EPS))

    Z_pred = xp_X[None, :, :, :] + xp_D_all
    Var_pred = (Pp_X[None, :, :, :] + Pp_D_all).real
    Z_pred_rmjk = jnp.swapaxes(Z_pred, 1, 2)
    Var_rmjk = jnp.swapaxes(Var_pred, 1, 2)
    resid = Y - Z_pred_rmjk
    moment = jnp.abs(resid)**2 + Var_rmjk

    if obs_noise_shared:
        S_rm = jnp.sum(moment, axis=(2, 3))
        S_mr = jnp.transpose(S_rm, (1, 0))
        sig2_mr = jnp.maximum(sig2_eps_jmr[0], EPS)
        Rterm = -0.5 * (Y.shape[2] * K) * jnp.sum(jnp.log(sig2_mr)) - 0.5 * jnp.sum(S_mr / sig2_mr)
    else:
        S_rmj = jnp.sum(moment, axis=3)
        S_jmr = jnp.transpose(S_rmj, (2, 1, 0))
        sig2_jmr = jnp.maximum(sig2_eps_jmr, EPS)
        Rterm = -0.5 * K * jnp.sum(jnp.log(sig2_jmr)) - 0.5 * jnp.sum(S_jmr / sig2_jmr)

    return jnp.asarray(Qterm_X + Qterm_D + Rterm, dtype=jnp.float64)


def _moments_phi_Q_from_Ypool(Y: jnp.ndarray, db: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    Ybar_mjk = jnp.mean(Y, axis=0)               # (M,J,K)
    Ybar_jmk = jnp.swapaxes(Ybar_mjk, 0, 1)      # (J,M,K)

    Csum = jnp.sum(Ybar_jmk[:, :, 1:] * jnp.conj(Ybar_jmk[:, :, :-1]), axis=(1, 2))
    Rprev = jnp.sum(jnp.abs(Ybar_jmk[:, :, :-1])**2, axis=(1, 2))
    Rnext = jnp.sum(jnp.abs(Ybar_jmk[:, :, 1:])**2, axis=(1, 2))

    Phi0 = jnp.clip(jnp.real(Csum) / jnp.maximum(Rprev, EPS), 0.2, 0.9995)
    lam0 = -jnp.log(Phi0) / db
    num = Rnext - 2.0 * Phi0 * jnp.real(Csum) + (Phi0**2) * Rprev
    M = Y.shape[1]
    Qhat0 = jnp.maximum(num, EPS) / ((Y.shape[3] - 1) * M)
    sigv0 = jnp.sqrt(2.0 * lam0 * Qhat0 / jnp.maximum(1.0 - Phi0**2, EPS))
    return lam0, sigv0


@partial(jax.jit,
         static_argnames=("max_iter","obs_noise_shared","no_pool_iters",
                          "enable_logging","print_every"))
def _em_ct_hier_loop(
    Y: jnp.ndarray, db: float,
    max_iter: int, tol: float, noise_floor: float, obs_noise_shared: bool,
    sig2_eps_init: float,
    lam_X0: jnp.ndarray, sigv_X0: jnp.ndarray,   # (J,), (J,)
    lam_D0: jnp.ndarray, sigv_D0: jnp.ndarray,   # (J,), (J,)
    no_pool_iters: int,
    enable_logging: bool,
    print_every: int
):
    R, M, J, K = Y.shape

    # broadcast OU params across tapers
    lam_X = lam_X0[:, None] * jnp.ones((1, M))
    sigv_X = sigv_X0[:, None] * jnp.ones((1, M))
    lam_D = lam_D0[:, None] * jnp.ones((1, M))
    sigv_D = sigv_D0[:, None] * jnp.ones((1, M))

    if obs_noise_shared:
        sig2_eps_jmr = jnp.full((1, M, R), sig2_eps_init)
    else:
        sig2_eps_jmr = jnp.full((J, M, R), sig2_eps_init)

    # INITIAL priors: z0=0, P0 = stationary variance = sigv^2/(2*lam)
    x0_X = jnp.zeros((J, M), dtype=Y.dtype)
    P0_X = (sigv_X**2 / jnp.maximum(2.0 * lam_X, EPS)).real
    x0_D = jnp.zeros((R, J, M), dtype=Y.dtype)
    P0_D = jnp.broadcast_to((sigv_D**2 / jnp.maximum(2.0 * lam_D, EPS)).real[None, :, :], (R, J, M))

    X_mean = jnp.zeros((J, M, K), dtype=Y.dtype)
    D_mean = jnp.zeros((R, J, M, K), dtype=Y.dtype)
    Q_hist = jnp.zeros((max_iter,), dtype=jnp.float64)
    prev_Q = jnp.float64(-jnp.inf)
    it0 = jnp.int32(0)
    go0 = jnp.bool_(True)

    Y_rjmk = jnp.swapaxes(Y, 1, 2)  # (R,J,M,K)

    def cond_fun(st):
        return jnp.logical_and(st["it"] < max_iter, st["go"])

    def body_fun(st):
        lam_X_s, sigv_X_s = st["lam_X"], st["sigv_X"]
        lam_D_s, sigv_D_s = st["lam_D"], st["sigv_D"]
        sig2_eps = st["sig2_eps"]
        X_mean_s = st["X_mean"]

        # E-step for D: per trial, with its own z0,P0
        if obs_noise_shared:
            sig_eps_trials = jnp.sqrt(jnp.maximum(sig2_eps[0], EPS)).T[:, None, :]  # (R,1,M)
        else:
            sig_eps_trials = jnp.sqrt(jnp.maximum(jnp.transpose(sig2_eps, (2, 0, 1)), EPS))  # (R,J,M)

        def run_one(Y_rjmK, sig_eps_r, z0_rjm, P0_rjm):
            Y_res = Y_rjmK - X_mean_s
            return _rtss_ou_jax(Y_res, lam_D_s, sigv_D_s, sig_eps_r, db, z0_rjm, P0_rjm)

        xs_D_r, Ps_D_r, Csum_D_r, Rprev_D_r, Rnext_D_r, xp_D_r, Pp_D_r, *_ = \
            vmap(run_one, in_axes=(0, 0, 0, 0))(Y_rjmk, sig_eps_trials, st["x0_D"], st["P0_D"])

        D_mean_new = xs_D_r
        xp_D_all = xp_D_r
        Pp_D_all = Pp_D_r
        x0_D_new = xs_D_r[:, :, :, 0]
        P0_D_new = Ps_D_r[:, :, :, 0].real

        Csum_D_acc = jnp.sum(Csum_D_r, axis=0)
        Rprev_D_acc = jnp.sum(Rprev_D_r, axis=0)
        Rnext_D_acc = jnp.sum(Rnext_D_r, axis=0)

        # E-step for X: pooled across trials, with z0_X,P0_X
        use_pool = st["it"] >= no_pool_iters
        sig2_for_pool = lax.select(use_pool, sig2_eps, jnp.ones_like(sig2_eps))
        Y_pool, sig2_pool_jm = build_pooled_X_obs(Y, D_mean_new, sig2_for_pool)
        sig_pool = jnp.sqrt(jnp.maximum(sig2_pool_jm, EPS))

        xs_X, Ps_X, Csum_X, Rprev_X, Rnext_X, xp_X, Pp_X, *_ = \
            _rtss_ou_jax(Y_pool, lam_X_s, sigv_X_s, sig_pool, db, st["x0_X"], st["P0_X"])
        X_mean_new = xs_X
        x0_X_new = xs_X[:, :, 0]
        P0_X_new = Ps_X[:, :, 0].real

        # M-steps
        lam_X_j, sigv_X_j = m_step_ou_params(Csum_X, Rprev_X, Rnext_X, K, db, n_chains=M)
        lam_D_j, sigv_D_j = m_step_ou_params(Csum_D_acc, Rprev_D_acc, Rnext_D_acc, K, db, n_chains=M * R)

        lam_X_new = lam_X_j[:, None] * jnp.ones((1, M))
        sigv_X_new = sigv_X_j[:, None] * jnp.ones((1, M))
        lam_D_new = jnp.maximum(lam_D_j[:, None] * jnp.ones((1, M)), lam_X_new + 1e-6)
        sigv_D_new = sigv_D_j[:, None] * jnp.ones((1, M))

        # Noise update
        if obs_noise_shared:
            sig2_new = m_step_obs_noise_shared(Y, xp_X, Pp_X, xp_D_all, Pp_D_all, noise_floor)
        else:
            sig2_new = m_step_obs_noise_band_specific(Y, xp_X, Pp_X, xp_D_all, Pp_D_all, noise_floor)

        # Q monitor
        Q_val = _compute_Q_monitor(
            Y, xp_X, Pp_X, xp_D_all, Pp_D_all, sig2_new,
            Csum_X, Rprev_X, Rnext_X, lam_X_j, sigv_X_j,
            Csum_D_acc, Rprev_D_acc, Rnext_D_acc, lam_D_j, sigv_D_j,
            db, M, R, K, obs_noise_shared
        )

        # Optional logging
        if enable_logging:
            do_print = (st["it"] % jnp.int32(print_every)) == 0
            def _p(_):
                jax.debug.print("[EM-CT-HIER-JAX] iter {i}  Q={q:.6e}", i=st["it"], q=Q_val)
                return ()
            lax.cond(do_print, _p, lambda _: (), operand=None)

        it_next = st["it"] + 1
        Q_hist_next = st["Q_hist"].at[st["it"]].set(Q_val)
        delta = jnp.abs(Q_val - st["prev_Q"])
        go_next = jnp.logical_and(it_next < max_iter, delta > tol)

        return {
            "lam_X": lam_X_new, "sigv_X": sigv_X_new,
            "lam_D": lam_D_new, "sigv_D": sigv_D_new,
            "sig2_eps": sig2_new,
            "X_mean": X_mean_new, "D_mean": D_mean_new,
            "x0_X": x0_X_new, "P0_X": P0_X_new,
            "x0_D": x0_D_new, "P0_D": P0_D_new,
            "Q_hist": Q_hist_next, "prev_Q": Q_val,
            "it": it_next, "go": go_next,
        }

    state0 = {
        "lam_X": lam_X, "sigv_X": sigv_X,
        "lam_D": lam_D, "sigv_D": sigv_D,
        "sig2_eps": sig2_eps_jmr,
        "X_mean": X_mean, "D_mean": D_mean,
        "x0_X": x0_X, "P0_X": P0_X,
        "x0_D": x0_D, "P0_D": P0_D,
        "Q_hist": Q_hist, "prev_Q": prev_Q,
        "it": it0, "go": go0,
    }

    out = lax.while_loop(cond_fun, body_fun, state0)

    return (out["lam_X"], out["sigv_X"], out["lam_D"], out["sigv_D"],
            out["sig2_eps"], out["X_mean"], out["D_mean"], out["Q_hist"],
            out["x0_X"], out["P0_X"], out["x0_D"], out["P0_D"])


def em_ct_hier_jax(
    Y_trials: jnp.ndarray,
    db: float,
    *,
    max_iter: int = 200,
    tol: float = 1e-3,
    sig_eps_init: float = 5.0,
    lam_X_init: float | None = None,
    sigv_X_init: float | None = None,
    lam_D_init: float | None = None,
    sigv_D_init: float | None = None,
    verbose: bool = False,
    noise_floor: float = 1e-8,
    obs_noise_shared: bool = True,
    no_pool_iters: int = 3,
    log_every: int | None = None,
) -> EMHierResult:
    enable_logging = bool(verbose) or (log_every is not None)
    print_every = int(log_every if log_every is not None else (20 if verbose else 10**9))

    Y = _normalize_to_RMJK(jnp.asarray(Y_trials))
    R, M, J, K = Y.shape

    lam0_mom, sigv0_mom = _moments_phi_Q_from_Ypool(Y, db)
    lam_X0 = lam0_mom if lam_X_init is None else jnp.full((J,), float(lam_X_init))
    sigv_X0 = sigv0_mom if sigv_X_init is None else jnp.full((J,), float(sigv_X_init))
    lam_D0 = jnp.maximum(lam_X0 * 3.0, lam_X0 + 1e-3) if lam_D_init is None else jnp.full((J,), float(lam_D_init))
    sigv_D0 = 0.7 * sigv_X0 if sigv_D_init is None else jnp.full((J,), float(sigv_D_init))

    (lam_X, sigv_X, lam_D, sigv_D, sig2_eps_jmr,
     X_mean, D_mean, Q_hist,
     x0_X, P0_X, x0_D, P0_D) = _em_ct_hier_loop(
        Y=Y, db=float(db),
        max_iter=int(max_iter), tol=float(tol),
        noise_floor=float(noise_floor),
        obs_noise_shared=bool(obs_noise_shared),
        sig2_eps_init=float(sig_eps_init) ** 2,
        lam_X0=lam_X0, sigv_X0=sigv_X0,
        lam_D0=lam_D0, sigv_D0=sigv_D0,
        no_pool_iters=int(no_pool_iters),
        enable_logging=enable_logging,
        print_every=int(print_every),
    )

    sig_eps_jmr = jnp.sqrt(sig2_eps_jmr)
    if obs_noise_shared:
        sig_eps_mr = sig_eps_jmr[0]
    else:
        sig_eps_mr = jnp.sqrt(jnp.mean(sig_eps_jmr**2, axis=0))

    return EMHierResult(
        lam_X=lam_X, sigv_X=sigv_X,
        lam_D=lam_D, sigv_D=sigv_D,
        sig_eps_jmr=sig_eps_jmr,
        sig_eps_mr=sig_eps_mr,
        Q_hist=Q_hist,
        X_mean=X_mean, D_mean=D_mean,
        x0_X=x0_X, P0_X=P0_X,
        x0_D=x0_D, P0_D=P0_D
    )
