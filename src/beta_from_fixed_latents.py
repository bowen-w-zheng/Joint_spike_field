# src/beta_gamma_pg_fixed_latents_joint.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Dict, Any

# Optional PG sampler; falls back to PG mean if unavailable
_HAS_PG = False
try:
    from src.pg_utils import sample_polya_gamma as _pg_sample  # ω ~ PG(1, ψ)
    _HAS_PG = True
except Exception:
    _HAS_PG = False


# ─────────────────────────────────────────────────────────────────────────────
# Predictors from fixed latents: taper-average, nearest block, physical phase
# ─────────────────────────────────────────────────────────────────────────────
def _nearest_centres_idx(centres_sec: np.ndarray, t_sec: np.ndarray) -> np.ndarray:
    c = np.asarray(centres_sec, float).ravel()
    t = np.asarray(t_sec, float).ravel()
    ir = np.searchsorted(c, t, side="left")
    il = np.clip(ir - 1, 0, c.size - 1)
    ir = np.clip(ir, 0, c.size - 1)
    use_r = np.abs(c[ir] - t) < np.abs(c[il] - t)
    return np.where(use_r, ir, il).astype(np.int32)

def build_rotated_predictors_from_em(
    *,
    X_mean: np.ndarray,        # (J,M,K) complex
    D_mean: np.ndarray,        # (R,J,M,K) complex
    freqs_hz: np.ndarray,      # (J,)
    delta_spk: float,          # spike bin step (s)
    centres_sec: np.ndarray,   # (K,)
    T: int,                    # # spike bins
    bands_idx: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (ZR, ZI, bands_idx):
      ZR, ZI: (R, T, B) = Re/Im of e^{i2π f_j t} * mean_m ( X + D_r )
    """
    X = np.asarray(X_mean)
    D = np.asarray(D_mean)
    R = int(D.shape[0]); J, M, K = X.shape
    freqs = np.asarray(freqs_hz, float).reshape(J,)

    if bands_idx is None:
        bands_idx = np.arange(J, dtype=np.int32)
    else:
        bands_idx = np.asarray(bands_idx, dtype=np.int32)
    B = int(bands_idx.size)

    Z_bar = (X[None, ...] + D).mean(axis=2)                 # (R,J,K)
    t_sec = np.arange(T, dtype=np.float64) * float(delta_spk)
    k_idx = _nearest_centres_idx(centres_sec, t_sec)        # (T,)

    Z_t = Z_bar[:, :, k_idx]                                # (R,J,T)
    phase = 2.0 * np.pi * freqs[:, None] * t_sec[None, :]   # (J,T)
    rot = np.exp(1j * phase, dtype=np.complex128)           # (J,T)

    Ztilt = Z_t * rot[None, :, :]                           # (R,J,T)
    Ztilt = np.transpose(Ztilt, (0, 2, 1))                  # (R,T,J)

    ZR = Ztilt.real[:, :, bands_idx].astype(np.float32)     # (R,T,B)
    ZI = Ztilt.imag[:, :, bands_idx].astype(np.float32)     # (R,T,B)
    return ZR, ZI, bands_idx


# ─────────────────────────────────────────────────────────────────────────────
# Config + Trace
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class JointSamplerConfig:
    n_warmup: int = 1000
    n_samples: int = 1000
    thin: int = 1
    use_pg_sampler: bool = True          # else PG mean (IRLS-style)
    omega_floor: float = 1e-6

    # Priors
    tau2_intercept: float = 100.0**2     # Var(β0)
    tau2_beta: float = 10.0**2           # Var(each βR/βI) if not using ARD
    use_ard_beta: bool = False
    ard_a0_beta: float = 1e-2
    ard_b0_beta: float = 1e-2

    # Spike-history prior per unit: γ ~ N(mu_gamma, Sigma_gamma)
    mu_gamma: Optional[np.ndarray] = None        # (R_h,) or None→0
    Sigma_gamma: Optional[np.ndarray] = None     # (R_h,R_h) or None→tau2_gamma I
    tau2_gamma: float = 25.0**2

    # Feature scaling
    standardize_reim: bool = False     # z-score [ZR, ZI] (shared across units)
    standardize_hist: bool = False     # z-score H per column (shared across units)

    rng: np.random.Generator = np.random.default_rng(0)
    verbose: bool = False


@dataclass
class JointTrace:
    beta: np.ndarray             # (K_keep, S, p)
    gamma: Optional[np.ndarray]  # (K_keep, S, R_h) or None if R_h=0
    bands_idx: np.ndarray        # (B,)
    feat_mean_reim: Optional[np.ndarray]  # (2B,) if standardized
    feat_std_reim: Optional[np.ndarray]   # (2B,)
    feat_mean_hist: Optional[np.ndarray]  # (R_h,) if standardized
    feat_std_hist: Optional[np.ndarray]
    meta: Dict[str, Any]


# ─────────────────────────────────────────────────────────────────────────────
# Main joint sampler (all units at once; includes γ)
# ─────────────────────────────────────────────────────────────────────────────
def sample_beta_gamma_from_fixed_latents_joint(
    *,
    spikes: np.ndarray,               # (R, S, T) 0/1
    H_hist: Optional[np.ndarray],     # (S,T,R_h) or (R,S,T,R_h) or None
    X_mean: np.ndarray,               # (J,M,K) complex
    D_mean: np.ndarray,               # (R,J,M,K) complex
    freqs_hz: np.ndarray,             # (J,)
    centres_sec: np.ndarray,          # (K,)
    delta_spk: float,
    bands_idx: Optional[Sequence[int]] = None,
    cfg: JointSamplerConfig = JointSamplerConfig(),
) -> JointTrace:

    # ----- shapes -----
    spikes = np.asarray(spikes, np.uint8)
    R, S, T = spikes.shape
    ZR, ZI, bidx = build_rotated_predictors_from_em(
        X_mean=X_mean, D_mean=D_mean, freqs_hz=freqs_hz,
        delta_spk=delta_spk, centres_sec=centres_sec, T=T, bands_idx=bands_idx
    )  # (R,T,B)

    B = int(ZR.shape[2])
    N = R * T                        # rows after flattening trials × time
    p = 1 + 2 * B                    # β length per unit

    # ----- shared design X (N × p) -----
    X = np.empty((N, p), dtype=np.float64)
    X[:, 0] = 1.0
    X[:, 1:1+B]     = ZR.reshape(N, B)
    X[:, 1+B:1+2*B] = ZI.reshape(N, B)

    feat_m_reim = feat_s_reim = None
    if cfg.standardize_reim:
        feat = X[:, 1:]
        feat_m_reim = feat.mean(axis=0).astype(np.float32)
        feat_s_reim = (feat.std(axis=0) + 1e-8).astype(np.float32)
        X[:, 1:] = (feat - feat_m_reim) / feat_s_reim

    # ----- per-unit spike history H_s (S × N × R_h) -----
    if H_hist is None:
        R_h = 0
        H_all = np.zeros((S, N, 0), dtype=np.float64)
    else:
        H_hist = np.asarray(H_hist)
        if H_hist.ndim == 3:      # (S, T, R_h) → repeat across trials
            S2, T2, R_h = H_hist.shape
            assert S2 == S and T2 == T, "H_hist (S,T,R_h) must match spikes (R,S,T)"
            H_all = np.repeat(H_hist[:, None, :, :], R, axis=1).reshape(S, N, R_h)
        elif H_hist.ndim == 4:    # (R, S, T, R_h)
            R2, S2, T2, R_h = H_hist.shape
            assert R2 == R and S2 == S and T2 == T, "H_hist (R,S,T,R_h) must match spikes"
            H_all = np.moveaxis(H_hist, 1, 0).reshape(S, N, R_h)   # (S,N,R_h)
        else:
            raise ValueError("H_hist must be (S,T,R_h) or (R,S,T,R_h)")

    feat_m_hist = feat_s_hist = None
    if cfg.standardize_hist and R_h > 0:
        # z-score across rows (same transform for all units)
        H_stack = H_all.reshape(S*N, R_h)
        feat_m_hist = H_stack.mean(axis=0).astype(np.float32)
        feat_s_hist = (H_stack.std(axis=0) + 1e-8).astype(np.float32)
        H_all = (H_all - feat_m_hist[None, None, :]) / feat_s_hist[None, None, :]

    # ----- responses by unit: Y (S × N), κ = Y-1/2 -----
    Y_all = spikes.transpose(1, 0, 2).reshape(S, N).astype(np.float64)  # (S,N)
    kappa = Y_all - 0.5                                                 # (S,N)

    # ----- priors -----
    # β prior precision per unit: diag([1/τ0^2, 1/τ^2,...])
    Prec_beta = np.zeros((p,), dtype=np.float64)
    Prec_beta[0] = 1.0 / max(cfg.tau2_intercept, 1e-12)
    if cfg.use_ard_beta:
        tau2_lat = np.ones((S, 2*B), dtype=np.float64) * float(cfg.tau2_beta)   # per-unit ARD
    else:
        Prec_beta[1:] = 1.0 / max(cfg.tau2_beta, 1e-12)

    # γ prior precision (shared across units unless array of shape (S,R_h,R_h) provided)
    if R_h > 0 and cfg.Sigma_gamma is not None:
        Sg = np.asarray(cfg.Sigma_gamma, float)
        if Sg.ndim == 2:
            Prec_gamma = np.linalg.pinv(Sg)
            Prec_gamma = np.broadcast_to(Prec_gamma, (S, R_h, R_h)).copy()
        elif Sg.ndim == 3:
            assert Sg.shape[0] == S and Sg.shape[1] == Sg.shape[2] == R_h
            Prec_gamma = np.stack([np.linalg.pinv(Sg[s]) for s in range(S)], axis=0)
        else:
            raise ValueError("Sigma_gamma must be (R_h,R_h) or (S,R_h,R_h)")
    else:
        Prec_gamma = np.broadcast_to(
            np.eye(R_h, dtype=np.float64) * (0.0 if R_h==0 else 1.0 / max(cfg.tau2_gamma, 1e-12)),
            (S, R_h, R_h)
        ).copy()

    mu_gamma = None
    if R_h > 0 and cfg.mu_gamma is not None:
        mu_gamma = np.broadcast_to(np.asarray(cfg.mu_gamma, float).reshape(1, R_h), (S, R_h)).copy()

    # ----- storage -----
    K_keep = int(cfg.n_samples)
    beta_draws  = np.zeros((K_keep, S, p), dtype=np.float32)
    gamma_draws = None if R_h == 0 else np.zeros((K_keep, S, R_h), dtype=np.float32)

    # init
    beta = np.zeros((S, p), dtype=np.float64)
    # sensible β0 init from per-unit rate
    pbar = Y_all.mean(axis=1)
    ok   = (pbar > 0.0) & (pbar < 1.0)
    beta[ok, 0] = np.log(pbar[ok] / (1.0 - pbar[ok]))
    gamma = np.zeros((S, R_h), dtype=np.float64) if R_h > 0 else None

    # precompute Xᵀ κ (same each iteration)
    XT_kappa = np.einsum('ni,sn->si', X, kappa)                     # (S,p)
    HT_kappa = None if R_h == 0 else np.einsum('snr,sn->sr', H_all, kappa)  # (S,R_h)

    total = int(cfg.n_warmup + cfg.n_samples * cfg.thin)
    keep = 0
    for it in range(total):
        if cfg.verbose and it % 10 == 0:
            print(f"[β,γ|fixed] iteration {it}/{total}")
        # ---- linear predictor ψ_s = Xβ_s + Hγ_s (all S at once) ----
        psi = (X @ beta.T).T                                          # (S,N)
        if R_h > 0:
            psi += np.einsum('snr,sr->sn', H_all, gamma)              # (S,N)

        # ---- PG weights (vectorized) ----
        if cfg.use_pg_sampler and _HAS_PG:
            omega = _pg_sample(psi.reshape(-1), cfg.rng).reshape(S, N).astype(np.float64)
            omega = np.maximum(omega, float(cfg.omega_floor))
        else:
            # E[ω|ψ] = 0.5 tanh(|ψ|/2)/|ψ|
            abspsi = np.maximum(np.abs(psi, dtype=np.float64), 1e-12)
            omega = (0.5 * np.tanh(np.clip(abspsi, 0.0, 50.0) / 2.0) / abspsi).astype(np.float64)
            omega = np.maximum(omega, float(cfg.omega_floor))

        # ---- Normal equations per unit (batched) ----
        # A11 = Xᵀ Ω X        (S,p,p)
        A11 = np.einsum('ni,sn,nj->sij', X, omega, X)
        # A12 = Xᵀ Ω H_s      (S,p,R_h)
        A12 = None if R_h == 0 else np.einsum('ni,sn,snr->sir', X, omega, H_all)
        # A22 = Hᵀ Ω H        (S,R_h,R_h)
        A22 = None if R_h == 0 else np.einsum('snr,sn,snk->srk', H_all, omega, H_all)

        # RHS
        b1 = XT_kappa.copy()                                         # (S,p)
        b2 = None if R_h == 0 else HT_kappa.copy()                   # (S,R_h)

        # Priors
        # β block
        if cfg.use_ard_beta:
            Prec_beta_all = np.zeros((S, p), dtype=np.float64)
            Prec_beta_all[:, 0] = Prec_beta[0]
            Prec_beta_all[:, 1:] = 1.0 / np.maximum(tau2_lat, 1e-12)
            for s in range(S):
                A11[s] += np.diag(Prec_beta_all[s])
        else:
            A11 += np.eye(p, dtype=np.float64) * Prec_beta           # broadcast add diag
        # γ block
        if R_h > 0:
            for s in range(S):
                A22[s] += Prec_gamma[s]
                if mu_gamma is not None:
                    b2[s] += Prec_gamma[s] @ mu_gamma[s]

        # Solve & sample θ_s ~ N(A^{-1} b, A^{-1}) for all s
        # We’ll do a per-unit Cholesky (fast; still jointly updated within this iteration)
        for s in range(S):
            if R_h == 0:
                A = A11[s]
                b = b1[s]
                d = p
            else:
                # assemble block matrix
                A = np.zeros((p + R_h, p + R_h), dtype=np.float64)
                A[:p, :p]       = A11[s]
                A[:p, p:]       = A12[s]
                A[p:, :p]       = A12[s].T
                A[p:, p:]       = A22[s]
                b = np.concatenate([b1[s], b2[s]], axis=0)
                d = p + R_h

            # draw θ
            A = 0.5 * (A + A.T)
            I = np.eye(d)
            jitter = 1e-12
            for _ in range(7):
                try:
                    L = np.linalg.cholesky(A + jitter * I)
                    mu = np.linalg.solve(A, b)
                    eps = cfg.rng.standard_normal(d)
                    theta = mu + np.linalg.solve(L.T, eps)
                    break
                except np.linalg.LinAlgError:
                    jitter *= 10.0
            else:
                # SVD fallback
                U, svals, VT = np.linalg.svd(A, hermitian=True)
                svals = np.maximum(svals, 1e-20)
                mu = VT.T @ ((U.T @ b) / svals)
                eps = cfg.rng.standard_normal(d)
                theta = mu + VT.T @ (eps / np.sqrt(svals))

            beta[s] = theta[:p]
            if R_h > 0:
                gamma[s] = theta[p:]

            # ARD update (per-unit) on β lat parts
            if cfg.use_ard_beta:
                b_lat = beta[s, 1:]
                a_post = cfg.ard_a0_beta + 0.5
                b_post = cfg.ard_b0_beta + 0.5 * (b_lat ** 2)
                tau2_lat[s] = 1.0 / cfg.rng.gamma(shape=a_post, scale=1.0 / b_post)

        # store
        if it >= cfg.n_warmup and ((it - cfg.n_warmup) % cfg.thin == 0):
            beta_draws[keep] = beta.astype(np.float32)
            if R_h > 0:
                gamma_draws[keep] = gamma.astype(np.float32)
            keep += 1
            if cfg.verbose and keep % 10 == 0:
                print(f"[β,γ|fixed] kept {keep}/{K_keep}")

    meta = dict(
        n_warmup=cfg.n_warmup, n_samples=cfg.n_samples, thin=cfg.thin,
        use_pg_sampler=(cfg.use_pg_sampler and _HAS_PG),
        B=B, p=p, R_h=R_h
    )
    return JointTrace(
        beta=beta_draws, gamma=gamma_draws, bands_idx=bidx,
        feat_mean_reim=feat_m_reim, feat_std_reim=feat_s_reim,
        feat_mean_hist=feat_m_hist, feat_std_hist=feat_s_hist,
        meta=meta
    )
