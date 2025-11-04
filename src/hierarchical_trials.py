# src/hierarchical_trials.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

@dataclass
class HConfig:
    n_outer: int = 3
    use_pg_mean: bool = True
    omega_floor: float = 1e-6
    ridge_beta: float = 1e-6
    verbose: bool = False

@dataclass
class HResult:
    beta: np.ndarray           # (S, 1+2*B)
    X_reim: np.ndarray         # (J, T_f, 2M)  shared means (per freq, time, [Re(M), Im(M)])
    D_reim: np.ndarray         # (J, R, T_f, 2M) trial deviations

def _ou_discretize(lam: float, sigv2: float, dt: float) -> Tuple[float, float]:
    F = np.exp(-lam * dt)
    Q = sigv2 * (1.0 - F*F) / (2.0 * lam) if lam > 0 else sigv2 * dt
    return F, Q

def _expect_pg_mean(psi: np.ndarray) -> np.ndarray:
    # E[ω | ψ] = (1/(2ψ)) tanh(ψ/2); stable at 0 → 1/4
    x = np.abs(psi)
    out = np.empty_like(x)
    small = x < 1e-8
    out[small] = 0.25
    z = x[~small]
    out[~small] = 0.5 * np.tanh(np.clip(z, 0, 50.0) / 2.0) / z
    return out

def _build_t2k(block_center_idx: np.ndarray, T_f: int) -> Tuple[np.ndarray, np.ndarray]:
    # maps fine time t to the list of block indices k that anchor at t
    t2k = -np.ones((T_f, 8), dtype=np.int32)
    kcount = np.zeros((T_f,), dtype=np.int32)
    for k, t in enumerate(block_center_idx):
        t = int(t)
        c = kcount[t]
        if c >= t2k.shape[1]:
            extra = np.full((T_f, t2k.shape[1]), -1, dtype=np.int32)
            t2k = np.concatenate([t2k, extra], axis=1)
        t2k[t, c] = k
        kcount[t] += 1
    return t2k, kcount

def _compute_ab(beta: np.ndarray, phi_B_T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    beta: (S, 1+2B);  phi_B_T: (B, T)
    returns a,b: (S,B,T)
    """
    S, p = beta.shape
    B = (p - 1) // 2
    pairs = beta[:, 1:].reshape(S, B, 2)
    betaR = pairs[..., 0]
    betaI = pairs[..., 1]
    c = np.cos(phi_B_T)  # (B,T)
    s = np.sin(phi_B_T)
    a = betaR[:, :, None] * c[None, :, :] + betaI[:, :, None] * s[None, :, :]
    b = -betaR[:, :, None] * s[None, :, :] + betaI[:, :, None] * c[None, :, :]
    return a, b

def run_hierarchical_multi(
    Y_trials: np.ndarray,           # (R, M, J, K) complex, derotated & scaled per trial
    spikes: np.ndarray,             # (R, S, T_f)
    freqs_hz: Sequence[float],      # (J,)
    delta_spk: float,
    block_centers_sec: Optional[np.ndarray] = None,  # (K,)
    block_center_idx: Optional[np.ndarray] = None,   # (K,)
    sigma_eps_mr: Optional[np.ndarray] = None,       # (M, R)
    coupled_bands_idx: Optional[Sequence[int]] = None,
    beta_init: Optional[np.ndarray] = None,          # (S, 1+2B)
    lam_X: Optional[np.ndarray] = None,              # (J,)
    sigv_X2: Optional[np.ndarray] = None,            # (J,)
    lam_D: Optional[np.ndarray] = None,              # (J,)
    sigv_D2: Optional[np.ndarray] = None,            # (J,)
    cfg: HConfig = HConfig(),
) -> HResult:
    R, M, J, K = Y_trials.shape
    assert spikes.ndim == 3, "spikes must be (R,S,T_f)"
    R2, S, T_f = spikes.shape
    assert R2 == R
    freqs_hz = np.asarray(freqs_hz, dtype=np.float64)
    if coupled_bands_idx is None:
        coupled_bands_idx = list(range(J))
    J_idx = np.asarray(coupled_bands_idx, dtype=int)
    B = len(J_idx)

    # map block centers to fine bins
    if block_center_idx is None:
        assert block_centers_sec is not None
        block_center_idx = np.clip(np.round(block_centers_sec / delta_spk).astype(int), 0, T_f-1)
    t2k, kcount = _build_t2k(block_center_idx, T_f)

    # per-(m,r) observation variance (if not provided, use a robust median)
    if sigma_eps_mr is None:
        mag2 = np.median(np.abs(Y_trials)**2, axis=2)  # (R,M,K) → (R,M)
        sigma_eps_mr = np.median(mag2, axis=2).T       # (M,R)
    else:
        sigma_eps_mr = np.asarray(sigma_eps_mr, dtype=np.float64)

    # OU params (shared vs delta)
    if lam_X is None:   lam_X   = np.full(J, 5.0)
    if sigv_X2 is None: sigv_X2 = np.full(J, 1.0)
    if lam_D is None:   lam_D   = np.full(J, 10.0)
    if sigv_D2 is None: sigv_D2 = np.full(J, 0.5)

    # β initial
    if beta_init is None:
        beta = np.zeros((S, 1 + 2*B), dtype=np.float64)
    else:
        beta = np.asarray(beta_init, dtype=np.float64)
        assert beta.shape == (S, 1+2*B)

    # PG arrays
    kappa = spikes.astype(np.float64) - 0.5              # (R,S,T)
    omega = np.full((R, S, T_f), 0.25, dtype=np.float64)

    # OU discretization (per J)
    FX = np.zeros(J); QX = np.zeros(J)
    FD = np.zeros(J); QD = np.zeros(J)
    for j in range(J):
        FX[j], QX[j] = _ou_discretize(lam_X[j],  sigv_X2[j],  delta_spk)
        FD[j], QD[j] = _ou_discretize(lam_D[j],  sigv_D2[j],  delta_spk)

    # phase tables
    t_sec = np.arange(T_f, dtype=np.float64) * delta_spk
    phi_J_T = 2.0*np.pi*freqs_hz.reshape(J,1) * t_sec.reshape(1,T_f)  # (J,T)
    phi_B_T = phi_J_T[J_idx]  # (B,T)

    # outputs: filtered means (Re/Im stacked per M)
    twoM = 2 * M
    X_reim = np.zeros((J, T_f, twoM), dtype=np.float64)
    D_reim = np.zeros((J, R, T_f, twoM), dtype=np.float64)

    for it in range(cfg.n_outer):
        if cfg.verbose:
            print(f"[hier] outer {it+1}/{cfg.n_outer}")

        # means & diag variances for X and deltas, per (j,m)
        muX_re = np.zeros((J, M));  muX_im = np.zeros((J, M))
        PX_re  = np.ones((J, M)) * 10.0; PX_im = np.ones((J, M)) * 10.0

        muD_re = np.zeros((R, J, M)); muD_im = np.zeros((R, J, M))
        PD_re  = np.ones((R, J, M)) * 10.0; PD_im = np.ones((R, J, M)) * 10.0

        # precompute a_s(t), b_s(t) for coupled bands
        a_tbl, b_tbl = _compute_ab(beta, phi_B_T)   # (S,B,T)

        for t in range(T_f):
            # predict
            if t > 0:
                muX_re *= FX[:, None];  muX_im *= FX[:, None]
                PX_re   = FX[:, None]**2 * PX_re + QX[:, None]
                PX_im   = FX[:, None]**2 * PX_im + QX[:, None]
                muD_re *= FD[None, :, None]; muD_im *= FD[None, :, None]
                PD_re   = FD[None, :, None]**2 * PD_re + QD[None, :, None]
                PD_im   = FD[None, :, None]**2 * PD_im + QD[None, :, None]

            # LFP snapshots at this fine bin (apply per trial & taper)
            kc = int(kcount[t])
            for i in range(kc):
                k = t2k[t, i]
                if k < 0: break
                for r in range(R):
                    for j in range(J):
                        for m in range(M):
                            y = Y_trials[r, m, j, k]
                            s2 = float(sigma_eps_mr[m, r])
                            # Re: y = (X + D_r) + eps
                            S_re = PX_re[j, m] + PD_re[r, j, m] + s2
                            if S_re > 1e-14:
                                inov = np.real(y) - (muX_re[j, m] + muD_re[r, j, m])
                                Kx = PX_re[j, m] / S_re
                                Kd = PD_re[r, j, m] / S_re
                                muX_re[j, m] += Kx * inov
                                muD_re[r, j, m] += Kd * inov
                                PX_re[j, m]  -= Kx * PX_re[j, m]
                                PD_re[r, j, m] -= Kd * PD_re[r, j, m]
                            # Im
                            S_im = PX_im[j, m] + PD_im[r, j, m] + s2
                            if S_im > 1e-14:
                                inov = np.imag(y) - (muX_im[j, m] + muD_im[r, j, m])
                                Kx = PX_im[j, m] / S_im
                                Kd = PD_im[r, j, m] / S_im
                                muX_im[j, m] += Kx * inov
                                muD_im[r, j, m] += Kd * inov
                                PX_im[j, m]  -= Kx * PX_im[j, m]
                                PD_im[r, j, m] -= Kd * PD_im[r, j, m]

            # spike pseudo-rows: loop trials & units; distribute innovation to X and δ_r
            sumX_re = muX_re.sum(axis=1); sumX_im = muX_im.sum(axis=1)
            sumPX_re = PX_re.sum(axis=1); sumPX_im = PX_im.sum(axis=1)
            for r in range(R):
                sumD_re = muD_re[r].sum(axis=1); sumD_im = muD_im[r].sum(axis=1)
                sumPD_re = PD_re[r].sum(axis=1);  sumPD_im = PD_im[r].sum(axis=1)
                for s in range(S):
                    om = float(max(omega[r, s, t], cfg.omega_floor))
                    y_pseudo = (float(kappa[r, s, t]) / om) - beta[s, 0]
                    # predicted mean from current latents
                    yhat = 0.0
                    for bi, j in enumerate(J_idx):
                        a = a_tbl[s, bi, t] / M
                        b = b_tbl[s, bi, t] / M
                        yhat += a * (sumX_re[j] + sumD_re[j]) + b * (sumX_im[j] + sumD_im[j])
                    inov = y_pseudo - yhat
                    # row variance
                    S_spk = 1.0 / om
                    for bi, j in enumerate(J_idx):
                        a = a_tbl[s, bi, t] / M
                        b = b_tbl[s, bi, t] / M
                        S_spk += a*a * (sumPX_re[j] + sumPD_re[j]) + b*b * (sumPX_im[j] + sumPD_im[j])
                    if S_spk <= 1e-16:
                        continue
                    # Kalman gains for X and δ_r along Re/Im blocks
                    for bi, j in enumerate(J_idx):
                        a = a_tbl[s, bi, t] / M
                        b = b_tbl[s, bi, t] / M
                        if a != 0.0:
                            Kgx = (PX_re[j, :] * a) / S_spk
                            muX_re[j, :] += Kgx * inov
                            PX_re[j, :]  -= (PX_re[j, :] * a) * Kgx
                            Kgdr = (PD_re[r, j, :] * a) / S_spk
                            muD_re[r, j, :] += Kgdr * inov
                            PD_re[r, j, :]  -= (PD_re[r, j, :] * a) * Kgdr
                        if b != 0.0:
                            Kgx = (PX_im[j, :] * b) / S_spk
                            muX_im[j, :] += Kgx * inov
                            PX_im[j, :]  -= (PX_im[j, :] * b) * Kgx
                            Kgdr = (PD_im[r, j, :] * b) / S_spk
                            muD_im[r, j, :] += Kgdr * inov
                            PD_im[r, j, :]  -= (PD_im[r, j, :] * b) * Kgdr

            # store filtered means
            for j in range(J):
                X_reim[j, t, :M]    = muX_re[j]
                X_reim[j, t, M:]    = muX_im[j]
                for r in range(R):
                    D_reim[j, r, t, :M] = muD_re[r, j]
                    D_reim[j, r, t, M:] = muD_im[r, j]

        # ---- PG means (IRLS) using current predictor means ----
        if cfg.use_pg_mean:
            muZR = np.zeros((R, T_f, J)); muZI = np.zeros((R, T_f, J))
            for j in range(J):
                c = np.cos(phi_J_T[j]); s = np.sin(phi_J_T[j])
                reX = X_reim[j, :, :M].mean(axis=1)
                imX = X_reim[j, :, M:].mean(axis=1)
                for r in range(R):
                    re = reX + D_reim[j, r, :, :M].mean(axis=1)
                    im = imX + D_reim[j, r, :, M:].mean(axis=1)
                    muZR[r, :, j] = c*re - s*im
                    muZI[r, :, j] = s*re + c*im
            for s in range(S):
                psi = beta[s, 0]
                for bi, j in enumerate(J_idx):
                    psi += beta[s, 1+2*bi]   * muZR[:, :, j] + \
                           beta[s, 1+2*bi+1] * muZI[:, :, j]
                omega[:, s, :] = np.maximum(_expect_pg_mean(psi), cfg.omega_floor)

        # ---- β ridge update across all (r,t) ----
        for s in range(S):
            A = np.eye(1 + 2*B) * cfg.ridge_beta
            bvec = np.zeros(1 + 2*B, dtype=np.float64)
            for r in range(R):
                for t in range(T_f):
                    w = float(omega[r, s, t]); k = float(kappa[r, s, t])
                    x = np.zeros(1 + 2*B); x[0] = 1.0
                    for bi, j in enumerate(J_idx):
                        c = np.cos(phi_J_T[j, t]); sgn = np.sin(phi_J_T[j, t])
                        zR = (X_reim[j, t, :M].mean() + D_reim[j, r, t, :M].mean())
                        zI = (X_reim[j, t, M:].mean() + D_reim[j, r, t, M:].mean())
                        x[1+2*bi]   = c*zR - sgn*zI
                        x[1+2*bi+1] = sgn*zR + c*zI
                    A += w * np.outer(x, x)
                    bvec += k * x
            beta[s] = np.linalg.solve(A, bvec)

        if cfg.verbose:
            print("  beta row0:", np.round(beta[0], 4))

    return HResult(beta=beta, X_reim=X_reim, D_reim=D_reim)
