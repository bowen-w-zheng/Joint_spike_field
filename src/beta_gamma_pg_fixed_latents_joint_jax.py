from __future__ import annotations
import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
import numpy as np
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Dict, Any
from functools import partial
from jax import debug as jdebug

print("jax version:", jax.__version__)
# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)

# JAX Polya-Gamma sampler
_HAS_PG = False
try:
    from src.polyagamma_jax import sample_pg_saddle_single
    _HAS_PG = True
except Exception:
    _HAS_PG = False


# ─────────────────────────────────────────────────────────────────────────────
# Predictors (NumPy pre-processing)
# ─────────────────────────────────────────────────────────────────────────────
def _nearest_centres_idx(centres_sec: np.ndarray, t_sec: np.ndarray) -> np.ndarray:
    c = np.asarray(centres_sec, float).ravel()
    t = np.asarray(t_sec, float).ravel()
    ir = np.searchsorted(c, t, side="left")
    il = np.clip(ir - 1, 0, c.size - 1)
    ir = np.clip(ir, 0, c.size - 1)
    use_r = np.abs(c[ir] - t) < np.abs(c[il] - t)
    return np.where(use_r, ir, il).astype(np.int32)

# ========================= PATCHED PIECES ONLY =========================
# add this new builder next to your existing block-grid builder
def build_rotated_predictors_from_fine(
    *,
    Z_fine: np.ndarray,          # (R, J, M, T) complex on the spike grid (from upsampler: ups.Z_mean)
    freqs_hz: np.ndarray,        # (J,)
    delta_spk: float,
    T: int,                      # spike length; should equal Z_fine.shape[-1]
    bands_idx: Optional[Sequence[int]] = None,
    rotation_sign: float = -1.0, # -1.0 to match your current inference; set +1.0 if you standardize on +iωt
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build per-trial Re/Im predictors directly from fine-grid latents.
    Z̄_rj(t) = mean_m Z_fine[r,j,m,t] (complex baseband)
    Z̃_rj(t) = exp(rotation_sign * i * 2π f_j t) * Z̄_rj(t)
    Returns:
      ZR, ZI: (R, T, B) real/imag predictors
      bands_idx: (B,)
    """
    Z = np.asarray(Z_fine, np.complex128)
    R, J, M, Tf = Z.shape
    assert Tf == T, f"Z_fine last dim {Tf} must equal spikes length T={T}"
    freqs = np.asarray(freqs_hz, float).reshape(J,)
    if bands_idx is None:
        bands_idx = np.arange(J, dtype=np.int32)
    else:
        bands_idx = np.asarray(bands_idx, dtype=np.int32)
    B = int(bands_idx.size)

    # taper average on fine grid: (R,J,T)
    Zbar_RJT = Z.mean(axis=2)

    # rotate on fine grid
    t_sec = np.arange(T, dtype=np.float64) * float(delta_spk)     # (T,)
    phase = 2.0 * np.pi * freqs[:, None] * t_sec[None, :]         # (J,T)
    rot = np.exp(1j * rotation_sign * phase, dtype=np.complex128) # (J,T)

    Ztilt_RJT = Zbar_RJT * rot[None, :, :]   # (R,J,T)
    Ztilt_RTJ = np.transpose(Ztilt_RJT, (0, 2, 1))  # (R,T,J)

    ZR = Ztilt_RTJ.real[:, :, bands_idx].astype(np.float64)  # (R,T,B)
    ZI = Ztilt_RTJ.imag[:, :, bands_idx].astype(np.float64)  # (R,T,B)
    return ZR, ZI, bands_idx


# ─────────────────────────────────────────────────────────────────────────────
# Config + Trace
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class JointSamplerConfig:
    n_warmup: int = 1000
    n_samples: int = 1000
    thin: int = 1
    use_pg_sampler: bool = False   # keep default off; enable for calibrated β
    omega_floor: float = 1e-6

    tau2_intercept: float = 100.0**2
    tau2_beta: float = 10.0**2         # prior var for Re/Im β weights
    use_ard_beta: bool = False
    ard_a0_beta: float = 1e-2
    ard_b0_beta: float = 1e-2

    mu_gamma: Optional[np.ndarray] = None
    Sigma_gamma: Optional[np.ndarray] = None
    tau2_gamma: float = 25.0**2

    standardize_reim: bool = False     # stays OFF for interpretability
    standardize_hist: bool = False

    rng: np.random.Generator = np.random.default_rng(0)
    verbose: bool = False


@dataclass
class JointTrace:
    beta: np.ndarray
    gamma: Optional[np.ndarray]
    bands_idx: np.ndarray
    feat_mean_reim: Optional[np.ndarray]
    feat_std_reim: Optional[np.ndarray]
    feat_mean_hist: Optional[np.ndarray]
    feat_std_hist: Optional[np.ndarray]
    meta: Dict[str, Any]


# ─────────────────────────────────────────────────────────────────────────────
# JAX-optimized sampling kernels
# ─────────────────────────────────────────────────────────────────────────────
@jit
def compute_psi_vectorized(X: jnp.ndarray, H_all: jnp.ndarray,
                           beta: jnp.ndarray, gamma: jnp.ndarray) -> jnp.ndarray:
    """
    X: (N, p_full) with columns [1 | 2B Re/Im | (R-1) trial dummies]
    H_all: (S, N, R_h)
    beta: (S, p_full)
    gamma: (S, R_h)
    Returns: (S, N)
    """
    psi = beta @ X.T                             # (S, N)
    if gamma.shape[1] > 0:
        psi += jnp.einsum('snr,sr->sn', H_all, gamma)
    return psi


@jit
def compute_omega_mean(psi: jnp.ndarray, omega_floor: float) -> jnp.ndarray:
    abspsi = jnp.maximum(jnp.abs(psi), 1e-12)
    omega = 0.5 * jnp.tanh(jnp.clip(abspsi, 0.0, 50.0) / 2.0) / abspsi
    return jnp.maximum(omega, omega_floor)


@jit
def sample_omega_pg(key: jax.random.PRNGKey, psi: jnp.ndarray,
                    omega_floor: float) -> jnp.ndarray:
    S, N = psi.shape
    total = S * N
    keys = jax.random.split(key, total)
    psi_flat = psi.ravel()
    omega_flat = vmap(lambda k, z: sample_pg_saddle_single(k, 1.0, z))(keys, psi_flat)
    omega = omega_flat.reshape(S, N)
    return jnp.maximum(omega, omega_floor)


@jit
def build_normal_equations_vectorized(
    X: jnp.ndarray,            # (N, p_full)
    H_all: jnp.ndarray,        # (S, N, R_h)
    omega: jnp.ndarray,        # (S, N)
    XT_kappa: jnp.ndarray,     # (S, p_full)
    HT_kappa: jnp.ndarray,     # (S, R_h) or (S,0)
    Prec_beta_all: jnp.ndarray,# (S, p_full)
    Prec_gamma: jnp.ndarray,   # (S, R_h, R_h) or (S,0,0)
    mu_gamma: Optional[jnp.ndarray],  # (S, R_h) or None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    S, N = omega.shape
    p_full = X.shape[1]
    R_h = H_all.shape[2]
    d = p_full + R_h

    dtype = X.dtype
    eye_p = jnp.eye(p_full, dtype=dtype)

    sqrt_omega = jnp.sqrt(omega)[:, :, None]                # (S,N,1)
    Xw = sqrt_omega * X[None, :, :]                         # (S,N,p_full)
    A11 = jnp.einsum('snp,snq->spq', Xw, Xw)                # (S,p,p)
    A11 = A11 + Prec_beta_all[:, :, None] * eye_p[None, :, :]
    b1 = XT_kappa.astype(dtype)                             # (S,p)

    if R_h == 0:
        A = jnp.zeros((S, d, d), dtype=dtype)
        A = A.at[:, :p_full, :p_full].set(A11)
        b = b1
        return A, b

    Hw = sqrt_omega * H_all                                  # (S,N,R_h)
    A12 = jnp.einsum('snp,snr->spr', Xw, Hw)                # (S,p,R_h)
    A22 = jnp.einsum('snr,snk->srk', Hw, Hw) + Prec_gamma   # (S,R_h,R_h)
    b2 = HT_kappa.astype(dtype)                             # (S,R_h)
    if mu_gamma is not None:
        b2 = b2 + jnp.einsum('srk,sk->sr', Prec_gamma, mu_gamma)

    A = jnp.zeros((S, d, d), dtype=dtype)
    A = A.at[:, :p_full, :p_full].set(A11)
    A = A.at[:, :p_full, p_full:].set(A12)
    A = A.at[:, p_full:, :p_full].set(jnp.swapaxes(A12, 1, 2))
    A = A.at[:, p_full:, p_full:].set(A22)

    b = jnp.concatenate([b1, b2], axis=1).astype(dtype)
    return A, b


@partial(jit, static_argnames=['d'])
def sample_theta_single_unit(key: jax.random.PRNGKey, A: jnp.ndarray,
                             b: jnp.ndarray, d: int) -> jnp.ndarray:
    A_sym = 0.5 * (A + A.T)
    A_reg = A_sym + 1e-8 * jnp.eye(d, dtype=A.dtype)
    L = jnp.linalg.cholesky(A_reg)
    v = jax.scipy.linalg.solve_triangular(L, b, lower=True)
    mu = jax.scipy.linalg.solve_triangular(L.T, v, lower=False)
    eps = jax.random.normal(key, shape=(d,), dtype=A.dtype)
    theta = mu + jax.scipy.linalg.solve_triangular(L.T, eps, lower=False)
    return theta


sample_theta_all_units = vmap(sample_theta_single_unit, in_axes=(0, 0, 0, None))


@partial(jit, static_argnames=['n_reim'])
def update_ard_tau2(key: jax.random.PRNGKey, beta: jnp.ndarray,
                    a0: float, b0: float, n_reim: int) -> jnp.ndarray:
    """
    ARD for only the 2B Re/Im coefficients.
    beta: (S, p_full) with columns [β0 | 2B re/im | (R-1) dummies]
    Returns: (S, n_reim)
    """
    b_lat = beta[:, 1:1+n_reim]                   # (S, 2B)
    a_post = a0 + 0.5
    b_post = b0 + 0.5 * (b_lat ** 2)
    S, C = b_lat.shape
    keys = jax.random.split(key, S * C).reshape(S, C, 2)

    def inv_gamma(kpair, bval):
        return 1.0 / jax.random.gamma(kpair, a_post) * bval

    tau2 = vmap(vmap(inv_gamma))(keys, b_post)    # (S, 2B)
    return tau2


@partial(jit, static_argnames=['d', 'p_full', 'R_h', 'n_reim', 'use_ard', 'use_pg'])
def gibbs_iteration(
    beta: jnp.ndarray,             # (S, p_full)
    gamma: jnp.ndarray,            # (S, R_h)
    Prec_beta_all: jnp.ndarray,    # (S, p_full)
    key: jax.random.PRNGKey,
    X: jnp.ndarray,                # (N, p_full)
    H_all: jnp.ndarray,            # (S, N, R_h)
    XT_kappa: jnp.ndarray,         # (S, p_full)
    HT_kappa: jnp.ndarray,         # (S, R_h)
    Prec_gamma: jnp.ndarray,       # (S, R_h, R_h)
    mu_gamma: Optional[jnp.ndarray],  # (S, R_h) or None
    omega_floor: float,
    a0_ard: float,
    b0_ard: float,
    d: int,
    p_full: int,
    R_h: int,
    n_reim: int,                   # 2*B
    use_ard: bool,
    use_pg: bool,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    # 1) PG weights
    psi = compute_psi_vectorized(X, H_all, beta, gamma)
    key_omega, key_rest = jax.random.split(key)
    omega = lax.cond(use_pg,
                     lambda kp: sample_omega_pg(kp, psi, omega_floor),
                     lambda kp: compute_omega_mean(psi, omega_floor),
                     key_omega)
    jdebug.print("iter omega mean per unit: {m}", m=jnp.mean(omega, axis=1))

    # 2) Normal equations
    A, b = build_normal_equations_vectorized(
        X, H_all, omega, XT_kappa, HT_kappa, Prec_beta_all, Prec_gamma, mu_gamma
    )

    # 3) Sample θ = [β, γ]
    S_units = beta.shape[0]
    key_theta, key_rest = jax.random.split(key_rest)
    thetas = sample_theta_all_units(jax.random.split(key_theta, S_units), A, b, d)
    beta_new = thetas[:, :p_full]
    gamma_new = thetas[:, p_full:] if R_h > 0 else gamma

    # 4) ARD on Re/Im only
    def do_ard(arg):
        k, bval, prec = arg
        tau2 = update_ard_tau2(k, bval, a0_ard, b0_ard, n_reim)      # (S, 2B)
        new_prec = prec.at[:, 1:1+n_reim].set(1.0 / jnp.maximum(tau2, 1e-12))
        return new_prec

    key_ard, _ = jax.random.split(key_rest)
    Prec_beta_all_new = lax.cond(use_ard,
                                 do_ard,
                                 lambda arg: arg[2],
                                 (key_ard, beta_new, Prec_beta_all))

    return beta_new, gamma_new, Prec_beta_all_new


# ─────────────────────────────────────────────────────────────────────────────
# Main sampler
# ─────────────────────────────────────────────────────────────────────────────
def sample_beta_gamma_from_fixed_latents_joint(
    *,
    spikes: np.ndarray,            # (R, S, T) 0/1
    H_hist: Optional[np.ndarray],  # (S,T,R_h) or (R,S,T,R_h) or None
    Z_fine: np.ndarray,            # (R,J,M,K) complex on the spike grid (from upsampler: ups.Z_mean)
    freqs_hz: np.ndarray,          # (J,)
    delta_spk: float,
    bands_idx: Optional[Sequence[int]] = None,
    cfg: JointSamplerConfig = JointSamplerConfig(),
) -> Tuple[jnp.ndarray, jnp.ndarray, JointTrace]:

    if cfg.use_pg_sampler and not _HAS_PG:
        raise RuntimeError("use_pg_sampler=True but polyagamma_jax is unavailable.")

    if cfg.verbose:
        print(f"[JAX sampler] Using {'PG' if cfg.use_pg_sampler else 'E[ω|ψ]'} for ω")
        print("[JAX sampler] Preprocessing data...")

    # Shapes
    spikes = np.asarray(spikes, np.uint8)
    R, S, T = spikes.shape

    # Build demodulated predictors
    ZR, ZI, bidx = build_rotated_predictors_from_fine(
        Z_fine=np.asarray(Z_fine, np.complex128),
        freqs_hz=freqs_hz,
        delta_spk=delta_spk,
        T=T,
        bands_idx=bands_idx,
        rotation_sign=-1.0,  # <-- flip to +1.0 if your simulator and inference use +iωt
    )
    B = int(ZR.shape[2])      # #bands used
    N = R * T                 # rows per unit across all trials

    # === (A) Re/Im feature block (no standardization by default) ===
    X_reim = np.empty((N, 2*B), dtype=np.float64)
    X_reim[:, :B]  = ZR.reshape(N, B, order='C')   # (R,T,B)->(N,B)
    X_reim[:, B: ] = ZI.reshape(N, B, order='C')

    feat_m_reim = feat_s_reim = None
    if cfg.standardize_reim:
        feat_m_reim = X_reim.mean(axis=0).astype(np.float32)
        feat_s_reim = (X_reim.std(axis=0) + 1e-8).astype(np.float32)
        X_reim = (X_reim - feat_m_reim) / feat_s_reim

    # === (B) Per-trial intercept dummies (R-1), first trial is reference ===
    n_dum = max(R - 1, 0)
    X_dum = np.zeros((N, n_dum), dtype=np.float64)
    if n_dum > 0:
        for r in range(1, R):
            X_dum[r*T:(r+1)*T, r-1] = 1.0

    # === (C) Assemble full design: [1 | 2B Re/Im | (R-1) dummies] ===
    p_full = 1 + 2*B + n_dum
    X = np.empty((N, p_full), dtype=np.float64)
    X[:, 0] = 1.0
    X[:, 1:1+2*B] = X_reim
    if n_dum > 0:
        X[:, 1+2*B:] = X_dum

    # === (D) History features aligned to the same (r,t)->n map ===
    if H_hist is None:
        R_h = 0
        H_all = np.zeros((S, N, 0), dtype=np.float64)
    else:
        H_hist = np.asarray(H_hist)
        if H_hist.ndim == 3:                # (S,T,R_h) shared across trials
            S2, T2, R_h = H_hist.shape
            assert S2 == S and T2 == T
            H_all = np.empty((S, N, R_h), dtype=np.float64)
            for r in range(R):
                sl = slice(r*T, (r+1)*T)
                H_all[:, sl, :] = H_hist
        elif H_hist.ndim == 4:              # (R,S,T,R_h) per-trial
            R2, S2, T2, R_h = H_hist.shape
            assert R2 == R and S2 == S and T2 == T
            H_all = np.empty((S, N, R_h), dtype=np.float64)
            for r in range(R):
                sl = slice(r*T, (r+1)*T)
                H_all[:, sl, :] = H_hist[r]
        else:
            raise ValueError("H_hist must be (S,T,R_h) or (R,S,T,R_h) or None")

    feat_m_hist = feat_s_hist = None
    if cfg.standardize_hist and R_h > 0:
        H_stack = H_all.reshape(S*N, R_h)
        feat_m_hist = H_stack.mean(axis=0).astype(np.float32)
        feat_s_hist = (H_stack.std(axis=0) + 1e-8).astype(np.float32)
        H_all = (H_all - feat_m_hist[None, None, :]) / feat_s_hist[None, None, :]

    # === (E) Responses with the SAME (r,t)->n mapping ===
    Y_all = np.empty((S, N), dtype=np.float64)
    for r in range(R):
        sl = slice(r*T, (r+1)*T)
        Y_all[:, sl] = spikes[r].astype(np.float64)        # (S,T)
    kappa = Y_all - 0.5                                    # (S,N)

    if cfg.verbose:
        print(f"[JAX sampler] Shapes: R={R}, S={S}, T={T}, N={N}, B={B}, p_full={p_full}, R_h={R_h}")

    # Convert to JAX
    X_jax = jnp.array(X, dtype=jnp.float64)
    H_all_jax = jnp.array(H_all, dtype=jnp.float64)
    kappa_jax = jnp.array(kappa, dtype=jnp.float64)

    # Precompute X^T κ and H^T κ
    XT_kappa = jnp.einsum('np,sn->sp', X_jax, kappa_jax)                 # (S,p_full)
    HT_kappa = (jnp.zeros((S, 0), dtype=X_jax.dtype) if R_h == 0 else
                jnp.einsum('snr,sn->sr', H_all_jax, kappa_jax).astype(X_jax.dtype))

    # Priors: global intercept, Re/Im block, dummy intercepts
    Prec_beta_base = np.zeros(p_full, dtype=np.float64)
    Prec_beta_base[0] = 1.0 / max(cfg.tau2_intercept, 1e-12)             # β0
    Prec_beta_base[1:1+2*B] = 1.0 / max(cfg.tau2_beta, 1e-12)            # 2B
    if n_dum > 0:
        Prec_beta_base[1+2*B:] = 1.0 / max(cfg.tau2_intercept, 1e-12)    # dummy intercepts

    Prec_beta_all = jnp.broadcast_to(Prec_beta_base, (S, p_full))

    # Gamma prior
    if R_h > 0 and cfg.Sigma_gamma is not None:
        Sg = np.asarray(cfg.Sigma_gamma, float)
        if Sg.ndim == 2:
            Prec_gamma = np.linalg.pinv(Sg)
            Prec_gamma = np.broadcast_to(Prec_gamma, (S, R_h, R_h)).copy()
        else:
            Prec_gamma = np.stack([np.linalg.pinv(Sg[s]) for s in range(S)], axis=0)
    else:
        var = 0.0 if R_h == 0 else 1.0 / max(cfg.tau2_gamma, 1e-12)
        Prec_gamma = np.broadcast_to(np.eye(max(R_h, 1)) * var, (S, max(R_h, 1), max(R_h, 1))).copy()
        if R_h == 0:
            Prec_gamma = Prec_gamma[:, :0, :0]
    Prec_gamma = jnp.array(Prec_gamma, dtype=X_jax.dtype)

    mu_gamma_jax = None
    if R_h > 0 and cfg.mu_gamma is not None:
        mu_gamma_jax = jnp.broadcast_to(jnp.array(cfg.mu_gamma).reshape(1, R_h), (S, R_h))

    # Initialize β with logit of per-unit spike rate; γ zeros
    pbar = Y_all.mean(axis=1)                            # (S,)
    ok = (pbar > 0.0) & (pbar < 1.0)
    beta = jnp.zeros((S, p_full), dtype=jnp.float64)
    beta = beta.at[ok, 0].set(jnp.log(pbar[ok] / (1.0 - pbar[ok])))
    gamma = jnp.zeros((S, max(R_h, 1)), dtype=jnp.float64)
    if R_h == 0:
        gamma = gamma[:, :0]

    # Sampler
    total = int(cfg.n_warmup + cfg.n_samples * cfg.thin)
    d = p_full + R_h
    n_reim = int(2 * B)   # Python int so it's static for JIT

    key = jax.random.PRNGKey(cfg.rng.integers(0, 2**31))

    gibbs_fn = partial(
        gibbs_iteration,
        X=X_jax,
        H_all=H_all_jax,
        XT_kappa=XT_kappa,
        HT_kappa=HT_kappa,
        Prec_gamma=Prec_gamma,
        mu_gamma=mu_gamma_jax,
        omega_floor=cfg.omega_floor,
        a0_ard=cfg.ard_a0_beta,
        b0_ard=cfg.ard_b0_beta,
        d=d,
        p_full=p_full,
        R_h=R_h,
        n_reim=n_reim,                 # static for jit
        use_ard=cfg.use_ard_beta,
        use_pg=cfg.use_pg_sampler,
    )

    beta_draws = []
    gamma_draws = [] if R_h > 0 else None

    for i in range(total):
        key, subkey = jax.random.split(key)
        beta, gamma, Prec_beta_all = gibbs_fn(beta, gamma, Prec_beta_all, subkey)

        # store with thinning
        if i >= cfg.n_warmup and (i - cfg.n_warmup) % cfg.thin == 0:
            beta_draws.append(np.array(beta, dtype=np.float32))
            if R_h > 0:
                gamma_draws.append(np.array(gamma, dtype=np.float32))

        if cfg.verbose and (i + 1) % 10 == 0:
            phase = "Warmup" if i < cfg.n_warmup else "Sampling"
            iter_in_phase = i + 1 if i < cfg.n_warmup else i + 1 - cfg.n_warmup
            print(f"[JAX sampler] {phase} iteration {iter_in_phase}/"
                  f"{cfg.n_warmup if i < cfg.n_warmup else cfg.n_samples * cfg.thin}")

    beta_draws = np.array(beta_draws)
    gamma_draws = None if R_h == 0 else np.array(gamma_draws)

    meta = dict(
        n_warmup=cfg.n_warmup,
        n_samples=cfg.n_samples,
        thin=cfg.thin,
        use_pg_sampler=cfg.use_pg_sampler,
        B=B, p_full=p_full, R_h=R_h, n_reim=n_reim,
        n_dummies=n_dum
    )

    return X_jax, H_all_jax, JointTrace(
        beta=beta_draws,
        gamma=gamma_draws,
        bands_idx=bidx,
        feat_mean_reim=feat_m_reim,
        feat_std_reim=feat_s_reim,
        feat_mean_hist=feat_m_hist,
        feat_std_hist=feat_s_hist,
        meta=meta
    )
