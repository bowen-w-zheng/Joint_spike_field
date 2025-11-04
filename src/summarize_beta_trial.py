import numpy as np
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Dict, Any
from scipy.stats import chi2
from scipy.stats import circmean, circstd

# ---------------------- BH-FDR & utils ----------------------
def bh_fdr(pvals, alpha=0.05):
    p = np.asarray(pvals, float)
    m = p.size
    order  = np.argsort(p)
    ranked = p[order]
    thresh = alpha * (np.arange(1, m+1) / m)
    passed = ranked <= thresh
    reject = np.zeros(m, dtype=bool)
    if np.any(passed):
        k = np.max(np.where(passed)[0])
        reject[order[:k+1]] = True
    return reject

def winsorize_cols(X, lo_q, hi_q):
    lo = np.percentile(X, 100*lo_q, axis=0)
    hi = np.percentile(X, 100*hi_q, axis=0)
    return np.clip(X, lo, hi)

def _stack_trace_beta(blist):
    """
    Stack a list of beta arrays into shape (Tdraws, S, P).
    Accepts elements shaped (S,P) or (P,) or (1,P).
    """
    buf = []
    S = P = None
    for b in blist:
        if b is None:
            continue
        b_arr = np.asarray(b)
        try:
            b_arr = b_arr.astype(np.float64, copy=False)
        except Exception:
            continue
        if b_arr.ndim == 1:               # (P,) -> (1,P)
            b_arr = b_arr[None, :]
        if b_arr.ndim != 2:
            continue
        if S is None:
            S, P = b_arr.shape
        if b_arr.shape != (S, P):
            continue
        buf.append(b_arr)
    if not buf:
        raise ValueError("trace.beta contains no usable numeric arrays")
    return np.stack(buf, axis=0)  # (Tdraws, S, P)

# ---------------------- core extraction ----------------------
def _extract_reim_blocks(postB: np.ndarray, B: int) -> Tuple[np.ndarray, np.ndarray]:
    # contiguous halves, not alternating!
    beta_R = postB[:, :, 1:1+B]
    beta_I = postB[:, :, 1+B:1+2*B]
    return beta_R, beta_I

# ---------------------- per-train summary ----------------------
def _summarize_per_train(
    beta_R: np.ndarray,    # (Nsamp, S, B)
    beta_I: np.ndarray,    # (Nsamp, S, B)
    B: int,
    trim: float,
    ridge: float,
    alpha_fdr: float,
    band_freqs: np.ndarray,
    verbose: bool,
):
    Nsamp, S_b, _ = beta_R.shape

    mag_mean_all = np.zeros((S_b, B))
    mag_sd_all   = np.zeros((S_b, B))
    phi_mean_all = np.zeros((S_b, B))
    phi_std_all  = np.zeros((S_b, B))
    mag_trace_all = np.zeros((S_b, Nsamp, B))
    phi_trace_all = np.zeros((S_b, Nsamp, B))

    for s in range(S_b):
        pvals = np.ones(B)
        W     = np.zeros(B)

        for b in range(B):
            Xb = np.column_stack([beta_R[:, s, b], beta_I[:, s, b]])  # (Nsamp, 2)
            Xw = winsorize_cols(Xb, trim, 1 - trim) if trim > 0 else Xb

            # Wald test with centered covariance
            mu_b  = Xw.mean(axis=0)                         # (2,)
            Sig_b = np.cov(Xw, rowvar=False, bias=True)     # (2,2)
            Sig_b = Sig_b + ridge * np.eye(2)
            xb    = np.linalg.solve(Sig_b, mu_b)
            W[b]  = float(mu_b @ xb)
            pvals[b] = 1.0 - chi2.cdf(W[b], df=2)

            # Magnitude & phase traces
            mag = np.linalg.norm(Xw, axis=1)
            mag_trace_all[s, :, b] = mag
            mag_mean_all[s, b] = mag.mean()
            mag_sd_all[s, b]   = mag.std(ddof=0)

            phi = np.arctan2(Xw[:, 1], Xw[:, 0])            # (Nsamp,)
            phi_full = phi.copy()
            phi_full[~np.isfinite(phi_full)] = np.nan
            phi_trace_all[s, :, b] = phi_full

            if np.isfinite(phi).any():
                phi_mean_all[s, b] = circmean(phi, low=-np.pi, high=np.pi)
                phi_std_all [s, b] = circstd (phi, low=-np.pi, high=np.pi)
            else:
                phi_mean_all[s, b] = np.nan
                phi_std_all [s, b] = np.nan

        reject = bh_fdr(pvals, alpha_fdr)
        if verbose:
            print(f"[Train {s}] BH-FDR @ α={alpha_fdr}: {int(reject.sum())} significant bands")
            for b in np.where(reject)[0]:
                mu_b = np.array([beta_R[:, s, b], beta_I[:, s, b]]).T
                mu_b = winsorize_cols(mu_b, trim, 1 - trim) if trim > 0 else mu_b
                m    = mu_b.mean(axis=0)
                A    = float(np.linalg.norm(m))
                ph   = float(np.arctan2(m[1], m[0]))
                print(f"  b={b:3d} | f={band_freqs[b]:7.3f} Hz | p={pvals[b]:.3g} "
                      f"| δ=√W={np.sqrt(W[b]):.2f} | |β|≈{A:.3g} | φ={ph:+.3f} rad")

    return mag_mean_all, mag_sd_all, phi_mean_all, phi_std_all, mag_trace_all, phi_trace_all

# ---------------------- public API ----------------------
def summarize_beta_trace(
    trace,
    all_freqs: Sequence[float],
    coupled_freq_idx: Optional[Sequence[int]] = None,
    *,
    fix_iter: int = 0,
    thin_beta: int = 1,
    burnin_frac: float = 0.1,
    trim: float = 0.01,
    alpha_fdr: float = 0.05,
    ridge: float = 1e-9,
    verbose: bool = True,
    return_intercepts: bool = False,
) -> Dict[str, Any]:
    """
    Summarize posterior β (phasor) samples per train and band for the
    trial-structured β layout:
        [β0, (βR, βI) x B, (R-1) trial-dummy intercepts]

    - Uses trace.meta['B'], trace.meta['p_full'], trace.meta['n_dummies'] if present.
    - Ignores dummy columns by slicing [1 : 1+2B] only.
    - Reports per-band |β|, phase, Wald p-values with BH-FDR.

    Returns dict with keys:
      mag_mean_all, mag_sd_all, phi_mean_all, phi_std_all,
      mag_trace_all, phi_trace_all, band_freqs, postB
      (and optionally: intercepts, trial_dummies)
    """
    # --- Stack draws ---
    beta_list = [b for b in getattr(trace, "beta", []) if b is not None]
    BETA = _stack_trace_beta(beta_list)                     # (Tdraws, S, p_full)
    n_draws, S, P = BETA.shape

    # --- Layout from meta (preferred) or fallback ---
    meta = getattr(trace, "meta", {}) or {}
    B_meta = meta.get("B", None)
    p_full = int(meta.get("p_full", P))
    n_dum  = int(meta.get("n_dummies", max(0, P - 1) - 2 * ((P - 1) // 2)))

    if B_meta is not None:
        B = int(B_meta)
        # sanity: ensure consistency
        if 1 + 2 * B + n_dum != p_full:
            # fall back to strict slicing
            n_dum = max(0, p_full - (1 + 2 * B))
    else:
        # Fallback if meta is missing: infer B by excluding any trailing dummies
        # Assume dummies are at the end; try to find the largest B with 1+2B <= P
        B = (P - 1) // 2
        # No guarantee when dummies present; warn via verbose
        if meta == {} and verbose:
            print("[summarize] meta.B missing; inferring B=(P-1)//2. "
                  "If you used trial dummies, prefer trace.meta['B'].")

    # --- Burn-in & thinning ---
    joint_draws = n_draws - int(fix_iter)
    burnin = int(burnin_frac * max(joint_draws, 0)) + int(fix_iter)
    burnin = max(0, min(burnin, n_draws))
    if verbose:
        print("Burn in samples", burnin)

    postB = BETA[burnin::max(1, thin_beta)].astype(np.float64, copy=False)  # (Nsamp,S,P)
    if postB.size == 0:
        raise ValueError("No β samples after burn-in/thinning.")

    # --- Extract Re/Im blocks only (ignore dummy columns entirely) ---
    print(B)
    print(postB.shape)
    beta_R, beta_I = _extract_reim_blocks(postB, B)         # (Nsamp,S,B) each

    # --- Frequencies to report ---
    all_freqs = np.asarray(all_freqs, float)
    if "bands_idx" in meta and meta["bands_idx"] is not None:
        band_idx = np.asarray(meta["bands_idx"], int)
        band_freqs = all_freqs[band_idx]
    else:
        # fallback: assume the first B freqs correspond to bands used
        band_freqs = all_freqs[:B]
    if coupled_freq_idx is not None:
        # optional extra restriction (must be subset of used bands)
        sel = np.asarray(coupled_freq_idx, int)
        band_freqs = band_freqs[sel]
        beta_R = beta_R[:, :, sel]
        beta_I = beta_I[:, :, sel]
        B = beta_R.shape[2]

    # --- Summarize per train ---
    (mag_mean_all, mag_sd_all, phi_mean_all, phi_std_all,
     mag_trace_all, phi_trace_all) = _summarize_per_train(
        beta_R, beta_I, B, trim, ridge, alpha_fdr, band_freqs, verbose
    )

    out = dict(
        mag_mean_all=mag_mean_all,
        mag_sd_all=mag_sd_all,
        phi_mean_all=phi_mean_all,
        phi_std_all=phi_std_all,
        mag_trace_all=mag_trace_all,
        phi_trace_all=phi_trace_all,
        band_freqs=band_freqs,
        postB=postB,
        B=B,
    )

    # --- Optional: return intercepts and per-trial dummy offsets for diagnosis ---
    if return_intercepts:
        beta0 = postB[:, :, 0]  # (Nsamp, S)
        out["beta0_mean"] = beta0.mean(axis=0)
        out["beta0_sd"]   = beta0.std(axis=0, ddof=0)
        if n_dum > 0:
            dummies = postB[:, :, 1 + 2*B : 1 + 2*B + n_dum]  # (Nsamp, S, n_dum)
            out["dummy_mean"] = dummies.mean(axis=0)          # (S, n_dum)
            out["dummy_sd"]   = dummies.std(axis=0, ddof=0)   # (S, n_dum)
            out["dummy_names"]= [f"trial_dummy_{i+1}" for i in range(n_dum)]

    return out
