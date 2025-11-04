# src/hierarchical_jax.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp

# ────────────────────────────────────────────────────────────────────────
# PG weights: prefer your CPU sampler in src.pg_utils; fall back to IRLS mean
_HAS_PG_SAMPLER = False
try:
    from src.pg_utils import sample_polya_gamma as _pg_sample  # ω ~ PG(1, ψ)
    _HAS_PG_SAMPLER = True
except Exception:
    _HAS_PG_SAMPLER = False


def _pg_mean(psi: jnp.ndarray) -> jnp.ndarray:
    """E[ω | ψ] for PG(1, ψ); numerically stable near 0."""
    x = jnp.abs(psi)
    return jnp.where(x < 1e-8, 0.25, 0.5 * jnp.tanh(jnp.clip(x, 0., 50.) / 2.) / x)


def _ou_FQ(lam: float, sigv2: float, dt: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Exact OU discretization for one real coordinate: Z_{t+1} = F Z_t + η, η~N(0,Q)."""
    F = jnp.exp(-lam * dt)
    Q = jnp.where(lam > 0.0, sigv2 * (1.0 - F * F) / (2.0 * lam), sigv2 * dt)
    return F, Q


def _build_t2k(block_idx: jnp.ndarray, T: int) -> jnp.ndarray:
    """
    For each fine-bin t, list the frame indices k that anchor at t (padded with -1).
    Returns (T, Kmax) int32.
    """
    counts = jnp.bincount(block_idx, length=T)
    Kmax = jnp.max(counts).astype(jnp.int32)

    def row(t):
        ks = jnp.nonzero(block_idx == t, size=Kmax, fill_value=-1)[0]
        return ks

    return jax.vmap(row)(jnp.arange(T, dtype=jnp.int32))


# ────────────────────────────────────────────────────────────────────────
# Public API

@dataclass
class HierConfig:
    n_outer: int = 3                   # number of outer refreshes
    use_pg_mean: bool = True           # True → IRLS (ω = E[ω|ψ]); False → sample ω if sampler present
    omega_floor: float = 1e-6
    ridge_beta: float = 1e-6
    verbose: bool = False


@dataclass
class HierResult:
    beta: np.ndarray                   # (S, 1 + 2*B)
    X_reim: np.ndarray                 # (J, T, 2M) shared means (Re[0..M-1], Im[0..M-1])
    D_reim: np.ndarray                 # (J, R, T, 2M) trial deviations


def run_hierarchical_trials_jax(
    Y_trials: np.ndarray,             # (R, M, J, K), complex (derotated + scaled + decimated)
    spikes: np.ndarray,               # (R, S, T) fine-bin spikes (0/1)
    freqs_hz: np.ndarray,             # (J,)
    delta_spk: float,                 # fine-bin step (seconds)
    centres_sec: np.ndarray,          # (K,) block centers (seconds)
    *,
    # LFP noise options:
    # - If sigma_eps_jmr is provided, it is used directly (shape (J, M, R)).
    # - Else if sigma_eps_mr is provided:
    #     (M, R) → broadcast across J,
    #     (J, M, R) → used directly.
    # - Else we estimate a robust per-band default (J, M, R) from the median across frames K.
    sigma_eps_mr: Optional[np.ndarray] = None,        # (M, R) or (J, M, R)
    sigma_eps_jmr: Optional[np.ndarray] = None,       # (J, M, R)
    coupled_bands_idx: Optional[Sequence[int]] = None,
    lam_X: Optional[np.ndarray] = None,               # (J,)
    sigv_X2: Optional[np.ndarray] = None,             # (J,)
    lam_D: Optional[np.ndarray] = None,               # (J,)
    sigv_D2: Optional[np.ndarray] = None,             # (J,)
    beta_init: Optional[np.ndarray] = None,           # (S, 1+2*B)
    cfg: HierConfig = HierConfig(),
) -> HierResult:

    # Shapes & arrays
    Y = jnp.asarray(Y_trials)        # (R,M,J,K), complex
    R, M, J, K = Y.shape
    spikes = jnp.asarray(spikes)     # (R,S,T)
    R2, S, T = spikes.shape
    if int(R2) != int(R):
        raise ValueError(f"spikes R={R2} must equal LFP trials R={R}")
    freqs = jnp.asarray(freqs_hz, dtype=jnp.float32)

    if coupled_bands_idx is None:
        coupled_bands_idx = list(range(int(J)))
    J_idx = jnp.asarray(coupled_bands_idx, dtype=jnp.int32)
    B = int(J_idx.shape[0])

    # Map frame centers to nearest fine-bin indices
    block_idx = jnp.clip(jnp.round(jnp.asarray(centres_sec) / delta_spk).astype(jnp.int32), 0, T - 1)
    t2k = _build_t2k(block_idx, int(T))  # (T, Kmax)

    # ---------- LFP observation noise: allow per-band (J, M, R) ----------
    # Priority: explicit sigma_eps_jmr -> sigma_eps_mr (J,M,R or (M,R)) -> default estimate
    if sigma_eps_jmr is not None:
        sig_eps_jmr = jnp.asarray(sigma_eps_jmr, dtype=jnp.float32)
        if sig_eps_jmr.shape != (int(J), int(M), int(R)):
            raise ValueError(f"sigma_eps_jmr must have shape (J, M, R); got {sig_eps_jmr.shape}")
    elif sigma_eps_mr is not None:
        se = jnp.asarray(sigma_eps_mr, dtype=jnp.float32)
        if se.shape == (int(M), int(R)):
            # broadcast across J
            sig_eps_jmr = jnp.broadcast_to(se[None, :, :], (int(J), int(M), int(R)))
        elif se.shape == (int(J), int(M), int(R)):
            sig_eps_jmr = se
        else:
            raise ValueError(f"sigma_eps_mr must be (M,R) or (J,M,R); got {se.shape}")
    else:
        # Robust per-band default: median |Y|^2 over frames K → (R,M,J), then transpose to (J,M,R)
        mag2 = jnp.median(jnp.abs(Y) ** 2, axis=3)   # (R, M, J)
        sig_eps_jmr = jnp.transpose(mag2, (2, 1, 0)).astype(jnp.float32)  # (J, M, R)

    # OU hyperparams
    lam_X   = jnp.asarray(lam_X   if lam_X   is not None else np.full(int(J), 5.0),  dtype=jnp.float32)
    sigv_X2 = jnp.asarray(sigv_X2 if sigv_X2 is not None else np.full(int(J), 1.0),  dtype=jnp.float32)
    lam_D   = jnp.asarray(lam_D   if lam_D   is not None else np.full(int(J),10.0),  dtype=jnp.float32)
    sigv_D2 = jnp.asarray(sigv_D2 if sigv_D2 is not None else np.full(int(J),0.5),  dtype=jnp.float32)

    # β init
    if beta_init is None:
        beta = jnp.zeros((S, 1 + 2*B), dtype=jnp.float32)
    else:
        beta = jnp.asarray(beta_init, dtype=jnp.float32)
        if beta.shape != (S, 1 + 2*B):
            raise ValueError(f"beta_init must have shape (S, {1+2*B}), got {beta.shape}")

    # Phases per (j,t)
    t_sec = jnp.arange(T, dtype=jnp.float32) * float(delta_spk)   # (T,)
    phi   = (2.0 * jnp.pi) * freqs[:, None] * t_sec[None, :]      # (J,T)
    cJT, sJT = jnp.cos(phi), jnp.sin(phi)

    # OU (map lam,sig to T step dt = delta_spk)
    FX, QX = jax.vmap(_ou_FQ, in_axes=(0, 0, None))(lam_X, sigv_X2, delta_spk)
    FD, QD = jax.vmap(_ou_FQ, in_axes=(0, 0, None))(lam_D, sigv_D2, delta_spk)

    twoM = 2 * int(M)

    # ---------- per-band JIT kernel that consumes omega[r,s,t] ----------
    def band_kernel(j: int,
                    beta_in: jnp.ndarray,      # (S, 1+2B)
                    omega: jnp.ndarray         # (R,S,T)
                   ):
        F_x, Q_x = FX[j], QX[j]
        F_d, Q_d = FD[j], QD[j]
        c_t, s_t = cJT[j], sJT[j]      # (T,)

        # state means/vars (diag) for shared X and all δ_r (per taper)
        muX_re = jnp.zeros((M,), dtype=jnp.float32)
        muX_im = jnp.zeros((M,), dtype=jnp.float32)
        PX_re  = jnp.ones((M,), dtype=jnp.float32) * 10.0
        PX_im  = jnp.ones((M,), dtype=jnp.float32) * 10.0

        muD_re = jnp.zeros((R, M), dtype=jnp.float32)
        muD_im = jnp.zeros((R, M), dtype=jnp.float32)
        PD_re  = jnp.ones((R, M), dtype=jnp.float32) * 10.0
        PD_im  = jnp.ones((R, M), dtype=jnp.float32) * 10.0

        Xre = jnp.zeros((T, M), dtype=jnp.float32)
        Xim = jnp.zeros((T, M), dtype=jnp.float32)
        Dre = jnp.zeros((T, R, M), dtype=jnp.float32)
        Dim = jnp.zeros((T, R, M), dtype=jnp.float32)

        def step(carry, t):
            muX_re, muX_im, PX_re, PX_im, muD_re, muD_im, PD_re, PD_im, Xre, Xim, Dre, Dim = carry

            # predict
            muX_re = muX_re * F_x; muX_im = muX_im * F_x
            PX_re  = PX_re  * (F_x**2) + Q_x
            PX_im  = PX_im  * (F_x**2) + Q_x

            muD_re = muD_re * F_d; muD_im = muD_im * F_d
            PD_re  = PD_re  * (F_d**2) + Q_d
            PD_im  = PD_im  * (F_d**2) + Q_d

            # LFP snapshots at t
            ks = t2k[t]  # (Kmax,)

            def upd_lfp(state, k):
                muX_re, muX_im, PX_re, PX_im, muD_re, muD_im, PD_re, PD_im = state

                def one_trial(r, st):
                    muX_re, muX_im, PX_re, PX_im, muD_re, muD_im, PD_re, PD_im = st
                    y = Y[r, :, j, k]                      # (M,) complex
                    s2 = sig_eps_jmr[j, :, r]              # (M,)  ← per-band noise

                    Sre = PX_re + PD_re[r] + s2
                    inov = jnp.real(y) - (muX_re + muD_re[r])
                    Kx = jnp.where(Sre > 1e-12, PX_re / Sre, 0.0)
                    Kd = jnp.where(Sre > 1e-12, PD_re[r] / Sre, 0.0)
                    muX_re = muX_re + Kx * inov
                    muD_re = muD_re.at[r].set(muD_re[r] + Kd * inov)
                    PX_re  = PX_re  - Kx * PX_re
                    PD_re  = PD_re.at[r].set(PD_re[r] - Kd * PD_re[r])

                    Sim = PX_im + PD_im[r] + s2
                    inov = jnp.imag(y) - (muX_im + muD_im[r])
                    Kx = jnp.where(Sim > 1e-12, PX_im / Sim, 0.0)
                    Kd = jnp.where(Sim > 1e-12, PD_im[r] / Sim, 0.0)
                    muX_im = muX_im + Kx * inov
                    muD_im = muD_im.at[r].set(muD_im[r] + Kd * inov)
                    PX_im  = PX_im  - Kx * PX_im
                    PD_im  = PD_im.at[r].set(PD_im[r] - Kd * PD_im[r])
                    return (muX_re, muX_im, PX_re, PX_im, muD_re, muD_im, PD_re, PD_im)

                return jax.lax.cond(
                    k >= 0,
                    lambda st: jax.lax.fori_loop(0, R, lambda r, s: one_trial(r, s), st),
                    lambda st: st,
                    (muX_re, muX_im, PX_re, PX_im, muD_re, muD_im, PD_re, PD_im)
                )

            (muX_re, muX_im, PX_re, PX_im, muD_re, muD_im, PD_re, PD_im) = \
                jax.lax.fori_loop(0, ks.shape[0], lambda i, st: upd_lfp(st, ks[i]),
                                  (muX_re, muX_im, PX_re, PX_im, muD_re, muD_im, PD_re, PD_im))

            # spike pseudo-rows at t
            sumX_re = jnp.sum(muX_re); sumX_im = jnp.sum(muX_im)

            def trial_spike(r, st):
                muX_re, muX_im, PX_re, PX_im, muD_re, muD_im, PD_re, PD_im = st
                sumD_re = jnp.sum(muD_re[r]); sumD_im = jnp.sum(muD_im[r])

                def unit_upd(s, st2):
                    muX_re, muX_im, PX_re, PX_im, muD_re, muD_im, PD_re, PD_im = st2
                    om = jnp.maximum(omega[r, s, t], cfg.omega_floor)
                    ytil = (spikes[r, s, t] - 0.5) / om - beta_in[s, 0]

                    # scalar weights per band/time (averaged across M tapers)
                    a = 0.0; b = 0.0
                    for bi in range(B):
                        a += ( beta_in[s, 1+2*bi]   * c_t[t] + beta_in[s, 1+2*bi+1] * s_t[t] ) / M
                        b += (-beta_in[s, 1+2*bi]   * s_t[t] + beta_in[s, 1+2*bi+1] * c_t[t] ) / M

                    # predictor averages across M tapers
                    zR = (sumX_re + sumD_re) / M
                    zI = (sumX_im + sumD_im) / M

                    yhat = a * zR + b * zI
                    inov = ytil - yhat

                    # correct variance for averaged predictors: var(mean) = (sum diag)/M^2
                    varR = (jnp.sum(PX_re) + jnp.sum(PD_re[r])) / (M * M)
                    varI = (jnp.sum(PX_im) + jnp.sum(PD_im[r])) / (M * M)
                    S_spk = jnp.maximum(1e-12, 1.0/om + a*a*varR + b*b*varI)

                    # gains & updates
                    Kgx_re = (PX_re * (a / (M * S_spk)))   # reflect derivative wrt mean(·)
                    Kgx_im = (PX_im * (b / (M * S_spk)))
                    muX_re = muX_re + Kgx_re * inov
                    muX_im = muX_im + Kgx_im * inov
                    PX_re  = PX_re  - (PX_re * (a / (M * jnp.sqrt(S_spk))))**2
                    PX_im  = PX_im  - (PX_im * (b / (M * jnp.sqrt(S_spk))))**2

                    Kgdr_re = (PD_re[r] * (a / (M * S_spk)))
                    Kgdr_im = (PD_im[r] * (b / (M * S_spk)))
                    muD_re  = muD_re.at[r].set(muD_re[r] + Kgdr_re * inov)
                    muD_im  = muD_im.at[r].set(muD_im[r] + Kgdr_im * inov)
                    PD_re   = PD_re.at[r].set(PD_re[r] - (PD_re[r] * (a / (M * jnp.sqrt(S_spk))))**2)
                    PD_im   = PD_im.at[r].set(PD_im[r] - (PD_im[r] * (b / (M * jnp.sqrt(S_spk))))**2)

                    return (muX_re, muX_im, PX_re, PX_im, muD_re, muD_im, PD_re, PD_im)

                return jax.lax.fori_loop(0, S, unit_upd,
                                         (muX_re, muX_im, PX_re, PX_im, muD_re, muD_im, PD_re, PD_im))

            (muX_re, muX_im, PX_re, PX_im, muD_re, muD_im, PD_re, PD_im) = \
                jax.lax.fori_loop(0, R, trial_spike,
                                  (muX_re, muX_im, PX_re, PX_im, muD_re, muD_im, PD_re, PD_im))

            # store filtered-as-smoothed
            Xre = Xre.at[t].set(muX_re); Xim = Xim.at[t].set(muX_im)
            Dre = Dre.at[t].set(muD_re); Dim = Dim.at[t].set(muD_im)

            return (muX_re, muX_im, PX_re, PX_im, muD_re, muD_im, PD_re, PD_im,
                    Xre, Xim, Dre, Dim), None

        carry0 = (muX_re, muX_im, PX_re, PX_im, muD_re, muD_im, PD_re, PD_im,
                  Xre, Xim, Dre, Dim)
        (muX_re, muX_im, PX_re, PX_im, muD_re, muD_im, PD_re, PD_im,
         Xre, Xim, Dre, Dim), _ = jax.lax.scan(step, carry0, jnp.arange(T, dtype=jnp.int32))

        # band-level predictors for β (R,T)
        zR = (jnp.mean(Xre, axis=1)[None, :] + jnp.mean(Dre, axis=2).T)  # (R,T)
        zI = (jnp.mean(Xim, axis=1)[None, :] + jnp.mean(Dim, axis=2).T)  # (R,T)

        muZR = c_t[None, :] * zR - s_t[None, :] * zI
        muZI = s_t[None, :] * zR + c_t[None, :] * zI

        return Xre, Xim, Dre, Dim, muZR, muZI

    # ---------- Outer loop: build ω outside JIT; per-band kernel consumes ω ----------
    X_reim = np.zeros((int(J), int(T), twoM), dtype=np.float32)
    D_reim = np.zeros((int(J), int(R), int(T), twoM), dtype=np.float32)
    omega = np.full((int(R), int(S), int(T)), 0.25, dtype=np.float32)  # init at E[ω|ψ=0]

    for it in range(cfg.n_outer):
        if cfg.verbose:
            print(f"[hier] outer {it+1}/{cfg.n_outer}")

        # run all bands with current ω
        omega_jax = jnp.asarray(omega)
        outs = [jax.jit(band_kernel)(j, beta, omega_jax) for j in range(int(J))]
        Xre_list, Xim_list, Dre_list, Dim_list, muZR_list, muZI_list = zip(*outs)

        for j in range(int(J)):
            X_reim[j, :, :int(M)]     = np.array(Xre_list[j])
            X_reim[j, :,  int(M):]    = np.array(Xim_list[j])
            D_reim[j, :, :, :int(M)]  = np.array(Dre_list[j])
            D_reim[j, :, :,  int(M):] = np.array(Dim_list[j])

        muZR_full = np.stack(muZR_list, axis=2)  # (R,T,J)
        muZI_full = np.stack(muZI_list, axis=2)  # (R,T,J)

        # ---- β ridge update (per unit)
        beta_np = np.array(beta)
        p = 1 + 2*B
        for s in range(int(S)):
            A = np.eye(p, dtype=np.float32) * cfg.ridge_beta
            bvec = np.zeros(p, dtype=np.float32)
            for r in range(int(R)):
                for t in range(int(T)):
                    w = omega[r, s, t]; k = float(spikes[r, s, t] - 0.5)
                    x = np.zeros(p, dtype=np.float32); x[0] = 1.0
                    for bi, jj in enumerate(np.array(J_idx)):
                        x[1+2*bi]   = muZR_full[r, t, jj]
                        x[1+2*bi+1] = muZI_full[r, t, jj]
                    A += w * np.outer(x, x)
                    bvec += k * x
            beta_np[s] = np.linalg.solve(A, bvec).astype(np.float32)
        beta = jnp.asarray(beta_np)

        # ---- ω refresh outside JIT: IRLS mean or true samples via src.pg_utils
        psi = np.zeros((int(R), int(S), int(T)), dtype=np.float64)
        for s in range(int(S)):
            psi[:, s, :] = beta_np[s, 0]
            for bi, jj in enumerate(np.array(J_idx)):
                psi[:, s, :] += beta_np[s, 1+2*bi]   * np.array(muZR_full[:, :, jj])
                psi[:, s, :] += beta_np[s, 1+2*bi+1] * np.array(muZI_full[:, :, jj])

        if cfg.use_pg_mean or not _HAS_PG_SAMPLER:
            abspsi = np.maximum(np.abs(psi), 1e-8)
            omega = (0.5 * np.tanh(np.clip(abspsi, 0, 50.) / 2.) / abspsi).astype(np.float32)
            omega = np.maximum(omega, cfg.omega_floor)
        else:
            omega = _pg_sample(psi.astype(np.float64), np.random.default_rng()).astype(np.float32)
            omega = np.maximum(omega, cfg.omega_floor)

        if cfg.verbose:
            print("  beta[0]:", np.round(beta_np[0], 4), "  ω[min,max]:",
                  float(omega.min()), float(omega.max()))

    return HierResult(beta=np.array(beta), X_reim=X_reim, D_reim=D_reim)
