from scipy.linalg import block_diag
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.linalg import cho_solve, solve_triangular
import numpy as np
import jax.numpy as jnp
import jax.random as jr
from jax import jit


# ──────────────────────────────────────────────────────────────────────
def build_design(latent_reim: jnp.ndarray) -> jnp.ndarray:
    """Prepend intercept column."""
    T, _ = latent_reim.shape
    return jnp.hstack([jnp.ones((T, 1)), latent_reim])


def pg_gaussian_update(X, kappa, omega, mu0, Sigma0):
    # Omega = diag(omega)
    # Solve with linear algebra that’s numerically stable
    A = X.T * omega  @ X   # (p,p)
    b = X.T @ kappa        # (p,)
    Sigma0_inv = np.linalg.inv(Sigma0)
    Sigma_star = np.linalg.inv(A + Sigma0_inv)
    mu_star    = Sigma_star @ (b + Sigma0_inv @ mu0)
    return 


def gibbs_update_beta_robust(
    key,
    latent_reim: jnp.ndarray,   # (T, 2B)
    spikes: jnp.ndarray,        # (T,)
    omega: jnp.ndarray,         # (T,)
    tau2_beta: float = 10.0,
    *,
    H_hist: np.ndarray | None = None,
    Sigma_gamma: np.ndarray | None = None,
    mu_gamma: np.ndarray | None = None,
    var_latent_reim: np.ndarray | None = None,   # (T, 2B) variance of latent regressors
    a0_ard: float = 1e-2, b0_ard: float = 1e-2, tau2_lat: np.ndarray | None = None,
    tau2_intercept: float = 100.0**2, tau2_gamma: float = 25.0**2,
    chol_jitter0: float = 1e-12, chol_backoff: int = 6,
    omega_floor: float = 1e-6, use_exact_cov: bool = False
):
    # initialize cache
    if not hasattr(gibbs_update_beta_robust, "_cache"):
        gibbs_update_beta_robust._cache = {}
    cache = gibbs_update_beta_robust._cache

    ## ---- stable slices (caller should pass the SAME objects during a block) ----
    y_full = np.asarray(spikes, dtype=np.float64).reshape(-1)
    ω_full = np.asarray(omega,  dtype=np.float64).reshape(-1)

    # --- cache F_beta_full = build_design(latent_reim) once per latent object ---
    key_F = ("F_beta_full", id(latent_reim))
    ent_F = cache.get(key_F)
    if ent_F is None or ent_F["shape"] != tuple(latent_reim.shape):
        F_beta_full = np.asarray(build_design(latent_reim), dtype=np.float64)
        cache[key_F] = {"F_beta_full": F_beta_full, "shape": tuple(latent_reim.shape)}
    else:
        F_beta_full = ent_F["F_beta_full"]

    T_full = F_beta_full.shape[0]
    T_h    = (H_hist.shape[0] if H_hist is not None else T_full)
    T_eff  = int(min(T_full, y_full.shape[0], ω_full.shape[0], T_h))
    if T_eff <= 0:
        raise ValueError("Empty overlap.")

    F_beta = F_beta_full[:T_eff, :]
    y      = y_full[:T_eff]
    ω      = np.maximum(ω_full[:T_eff], omega_floor)

    if H_hist is not None:
        H = np.asarray(H_hist[:T_eff], dtype=np.float64)
        key_FH = ("F_with_H", id(F_beta_full), id(H_hist), T_eff)
        ent_FH = cache.get(key_FH)
        if ent_FH is None:
            F = np.column_stack([F_beta, H])
            cache[key_FH] = {"F": F}
        else:
            F = ent_FH["F"]
        R = H.shape[1]
    else:
        F, R = F_beta, 0

    κ = y - 0.5
    p = F.shape[1]
    p_beta = F_beta.shape[1]              # = 1 + 2B
    twoB   = p_beta - 1

    # ARD state
    if (tau2_lat is None) or (tau2_lat.shape[0] != twoB):
        tau2_lat = np.ones(twoB, dtype=np.float64)

    # ---- cache FT_kappa per (F, κ) object identities ----
    key_FTk = ("FT_kappa", id(F), id(spikes), T_eff)
    ent_FTk = cache.get(key_FTk)
    if ent_FTk is None:
        FT_kappa = F.T @ κ
        cache[key_FTk] = {"FT_kappa": FT_kappa}
    else:
        FT_kappa = ent_FTk["FT_kappa"]

    # ---- prior block (cache Σγ^{-1} per Sigma_gamma object) ----
    Prec_prior = np.zeros((p, p), dtype=np.float64)
    Prec_prior[0, 0] = 1.0 / max(tau2_intercept, 1e-12)
    Prec_prior[1:p_beta, 1:p_beta] = np.diag(1.0 / np.maximum(tau2_lat, 1e-12))

    mu_prior = np.zeros(p, dtype=np.float64)
    if R > 0:
        if Sigma_gamma is not None:
            key_Sinv = ("Sig_g_inv", id(Sigma_gamma), R)
            ent_Sinv = cache.get(key_Sinv)
            if ent_Sinv is None:
                Sg = np.asarray(Sigma_gamma, dtype=np.float64)
                try:
                    Sg_inv = np.linalg.inv(Sg)
                except np.linalg.LinAlgError:
                    Sg_inv = np.linalg.pinv(Sg)
                cache[key_Sinv] = {"Sg_inv": Sg_inv}
            else:
                Sg_inv = ent_Sinv["Sg_inv"]
        else:
            Sg_inv = (1.0 / max(tau2_gamma, 1e-12)) * np.eye(R, dtype=np.float64)
        Prec_prior[p_beta:, p_beta:] = Sg_inv
        if mu_gamma is not None:
            mu_prior[p_beta:] = np.asarray(mu_gamma, dtype=np.float64)

    # ---- PG normal equations without forming W ----
    Prec = F.T @ (ω[:, None] * F) + Prec_prior
    RHS  = FT_kappa + Prec_prior @ mu_prior

    # ---- EIV variance correction (cache V^T per var_latent object) ----
    if var_latent_reim is not None:
        key_VT = ("V_T", id(var_latent_reim), T_eff)
        ent_VT = cache.get(key_VT)
        if ent_VT is None:
            V_T = np.asarray(var_latent_reim[:T_eff], np.float64).T  # (2B, T_eff)
            cache[key_VT] = {"V_T": V_T}
        else:
            V_T = ent_VT["V_T"]
        diag_add = V_T @ ω                                   # (2B,)
        Prec[1:p_beta, 1:p_beta] += np.diag(diag_add)

    # ---- solve & sample (same as before) ----
    Prec = 0.5 * (Prec + Prec.T)
    try:
        mean = np.linalg.solve(Prec, RHS); chol_ok = True
    except np.linalg.LinAlgError:
        U, s, VT = np.linalg.svd(Prec, hermitian=True); s = np.maximum(s, 1e-20)
        mean = VT.T @ ((U.T @ RHS) / s); chol_ok = False

    if use_exact_cov or not chol_ok:
        U, s, VT = np.linalg.svd(Prec, hermitian=True); s = np.maximum(s, 1e-20)
        key, subk = jr.split(key); eps = jr.normal(subk, (p,), dtype=jnp.float64)
        theta = mean + VT.T @ (np.asarray(eps) / np.sqrt(s))
    else:
        jitter = chol_jitter0; I = np.eye(p)
        for _ in range(chol_backoff):
            try:
                L = np.linalg.cholesky(Prec + jitter * I); break
            except np.linalg.LinAlgError:
                jitter *= 10.0
        else:
            U, s, VT = np.linalg.svd(Prec, hermitian=True); s = np.maximum(s, 1e-20)
            key, subk = jr.split(key); eps = jr.normal(subk, (p,), dtype=jnp.float64)
            theta = mean + VT.T @ (np.asarray(eps) / np.sqrt(s))
            beta_out  = jnp.asarray(theta[:p_beta])
            gamma_out = (jnp.asarray(theta[p_beta:]) if R > 0 else None)
            return key, beta_out, gamma_out, tau2_lat

        key, subk = jr.split(key); eps = jr.normal(subk, (p,), dtype=jnp.float64)
        theta = mean + np.linalg.solve(L.T, np.asarray(eps))

    beta_full = theta[:p_beta]
    gamma_out = (theta[p_beta:] if R > 0 else None)

    # ARD update (unchanged)
    beta_lat  = beta_full[1:]
    alpha_post = a0_ard + 0.5
    beta_post  = b0_ard + 0.5 * (beta_lat**2)
    tau2_lat_new = 1.0 / np.random.default_rng().gamma(shape=alpha_post, scale=1.0 / beta_post)

    return key, jnp.asarray(beta_full), (jnp.asarray(gamma_out) if R > 0 else None), tau2_lat_new

