import numpy as np
from scipy.stats import chi2, circmean, circstd
import matplotlib.pyplot as plt

def bh_fdr(pvals, alpha=0.05):
    """
    Benjamini-Hochberg FDR procedure.
    """
    p = np.asarray(pvals)
    m = len(p)
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
    """
    Winsorize columns of X to the lo_q and hi_q quantiles.
    """
    lo = np.percentile(X, 100*lo_q, axis=0)
    hi = np.percentile(X, 100*hi_q, axis=0)
    return np.clip(X, lo, hi)

def _stack_trace_beta(blist):
    """
    Stack a list of beta arrays into a single array of shape (Tdraws, S, 1+2B).
    """
    buf = []
    S = P = None
    for b in blist:
        if b is None:
            continue
        b_arr = np.asarray(b)
        try:
            b_arr = b_arr.astype(np.float64)
        except Exception:
            continue
        if b_arr.ndim == 1:               # (1+2B,) -> (1, 1+2B)
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
    BETA = np.stack(buf, axis=0)         # (Tdraws, S, 1+2B)
    return BETA

def _summarize_per_train(postB, B, trim, ridge, alpha_fdr, band_freqs, verbose):
    """
    Summarize posterior samples for each train.
    """
    S_b = postB.shape[1]
    mag_mean_all = []
    mag_sd_all   = []
    phi_mean_all = []
    phi_std_all  = []
    mag_trace_all = []
    phi_trace_all = []

    for s in range(S_b):
        beta_R = postB[:, s, 1::2][:, :B]            # (Nsamp, B)
        beta_I = postB[:, s, 2::2][:, :B]

        mu  = np.zeros((B, 2))
        Sig = np.zeros((B, 2, 2))
        W   = np.zeros(B)
        p   = np.ones(B)
        mag_mean = np.zeros(B)
        mag_sd   = np.zeros(B)
        phi_mean = np.zeros(B)
        phi_std  = np.zeros(B)

        mag_trace = np.zeros((beta_R.shape[0], B))
        phi_trace = np.zeros((beta_R.shape[0], B))

        for b in range(B):
            Xb = np.column_stack([beta_R[:, b], beta_I[:, b]])
            Xw = winsorize_cols(Xb, trim, 1 - trim)

            mu[b]  = Xw.mean(axis=0)
            Sig[b] = np.cov(Xw, rowvar=False, bias=True)
            Sigi   = np.linalg.inv(Sig[b] + ridge*np.eye(2))

            W[b]   = float(mu[b].T @ Sigi @ mu[b])
            p[b]   = 1.0 - chi2.cdf(W[b], df=2)

            mag = np.linalg.norm(Xw, axis=1)
            mag_mean[b] = mag.mean()
            mag_sd[b]   = mag.std(ddof=0)
            mag_trace[:, b] = mag

            phi = np.arctan2(Xw[:, 1], Xw[:, 0])
            phi = phi[np.isfinite(phi)]
            phi_full = np.arctan2(Xw[:, 1], Xw[:, 0])
            phi_full[~np.isfinite(phi_full)] = np.nan
            phi_trace[:, b] = phi_full

            if phi.size:
                phi_mean[b] = circmean(phi, low=-np.pi, high=np.pi)
                phi_std[b]  = circstd (phi, low=-np.pi, high=np.pi)
            else:
                phi_mean[b] = np.nan
                phi_std[b]  = np.nan

        mag_mean_all.append(mag_mean)
        mag_sd_all.append(mag_sd)
        phi_mean_all.append(phi_mean)
        phi_std_all.append(phi_std)
        mag_trace_all.append(mag_trace)
        phi_trace_all.append(phi_trace)

        reject = bh_fdr(p, alpha_fdr)
        sig_idx = np.where(reject)[0]
        delta   = np.sqrt(W)

        if verbose:
            print(f"[Train {s}] BH-FDR @ α={alpha_fdr}: {len(sig_idx)} significant bands")
            for b in sig_idx:
                A  = np.linalg.norm(mu[b])
                ph = np.arctan2(mu[b, 1], mu[b, 0])
                print(f"  b={b:3d} | f={band_freqs[b]:7.3f} Hz | p={p[b]:.3g} | δ={delta[b]:.2f} | |β|≈{A:.3g} | φ={ph:+.3f} rad")

    mag_mean_all = np.asarray(mag_mean_all)
    mag_sd_all   = np.asarray(mag_sd_all)
    phi_mean_all = np.asarray(phi_mean_all)
    phi_std_all  = np.asarray(phi_std_all)
    mag_trace_all = np.asarray(mag_trace_all)
    phi_trace_all = np.asarray(phi_trace_all)

    return mag_mean_all, mag_sd_all, phi_mean_all, phi_std_all, mag_trace_all, phi_trace_all

def summarize_beta_trace(
    trace,
    all_freqs,
    coupled_freq_idx=None,
    fix_iter=500,
    thin_beta=1,
    burnin_frac=0.1,
    trim=0.01,
    alpha_fdr=0.2,
    ridge=1e-9,
    verbose=True,
):
    """
    Summarize posterior β (phasor) samples per train and band.

    Parameters
    ----------
    trace : object
        Must have attribute `beta`, a list of arrays (Tdraws, S, 1+2B) or (S,1+2B) or (1+2B,).
    all_freqs : array-like
        All frequency labels.
    coupled_freq_idx : array-like or None
        Indices of coupled frequencies (optional).
    fix_iter : int
        Number of initial iterations to skip (default 500).
    thin_beta : int
        Thinning for β trace (default 1).
    burnin_frac : float
        Fraction of burn-in (default 0.1, but burnin is set to 0 here).
    trim : float
        Winsorize tails for robust β stats (default 0.01).
    alpha_fdr : float
        BH-FDR level (default 0.2).
    ridge : float
        Small ridge for 2x2 inverses (default 1e-9).
    verbose : bool
        If True, print per-train significance summaries.

    Returns
    -------
    dict with keys:
        mag_mean_all, mag_sd_all, phi_mean_all, phi_std_all,
        mag_trace_all, phi_trace_all, band_freqs
    """

    # --- Stack and validate beta trace ---
    beta_list = [b for b in trace.beta if b is not None]
    BETA = _stack_trace_beta(beta_list)              # (Tdraws, S, 1+2B)
    n_draws_b, S_b, P = BETA.shape
    assert (P - 1) % 2 == 0, "β layout must be [β0, (βR, βI) per band]"
    B = (P - 1) // 2
    

    # --- Burn-in, thinning, and finite mask ---
    joint_draws = n_draws_b - fix_iter
    burnin = int(burnin_frac * joint_draws) + int(fix_iter)
    print("Burn in samples", burnin)
    postB     = BETA[burnin::thin_beta].astype(np.float64, copy=False)
    maskB     = np.isfinite(postB).all(axis=(1,2))
    postB    = postB[maskB]
    if postB.size == 0:
        raise ValueError("No finite β samples after burn-in; check sampler stability.")

    # --- Band frequency labels ---
    freqs = np.asarray(all_freqs)
    if coupled_freq_idx is not None:
        band_freqs = freqs[np.asarray(coupled_freq_idx, dtype=int)[:B]]
    else:
        band_freqs = freqs[:B]

    # --- Summarize per train ---
    mag_mean_all, mag_sd_all, phi_mean_all, phi_std_all, mag_trace_all, phi_trace_all = \
        _summarize_per_train(postB, B, trim, ridge, alpha_fdr, band_freqs, verbose)

    return dict(
        mag_mean_all=mag_mean_all,
        mag_sd_all=mag_sd_all,
        phi_mean_all=phi_mean_all,
        phi_std_all=phi_std_all,
        mag_trace_all=mag_trace_all,
        phi_trace_all=phi_trace_all,
        band_freqs=band_freqs,
        postB=postB
    )

# ---- Plotting modules ----

def plot_beta_magnitude_vs_null(
    postB, all_freqs, s_plot=1, trim=0.01, alpha_fdr=0.2, bins_per=40, ridge=1e-9, rng_seed=0
):
    """
    Plot per-band |β| vs null |z|, z ~ N(0, Σ̂) for a given train.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import chi2

    rng = np.random.default_rng(rng_seed)

    # Determine B from postB shape
    B = (postB.shape[2] - 1) // 2

    # Pull posterior draws for the selected train
    beta_R = postB[:, s_plot, 1::2][:, :B]   # (N, B)
    beta_I = postB[:, s_plot, 2::2][:, :B]
    N      = beta_R.shape[0]

    # per-band labels
    labels = np.asarray(all_freqs)[:B]

    # Per-band μ̂, Σ̂ (centered), Wald stats
    mu   = np.zeros((B, 2))
    Sig  = np.zeros((B, 2, 2))
    W    = np.zeros(B)
    pval = np.ones(B)

    for b in range(B):
        Xb = np.column_stack([beta_R[:, b], beta_I[:, b]])
        Xw = winsorize_cols(Xb, trim, 1 - trim) if trim > 0 else Xb

        mu_b  = Xw.mean(axis=0)
        Sig_b = np.cov(Xw, rowvar=False, bias=True) + ridge*np.eye(2)  # centered covariance only

        x = np.linalg.solve(Sig_b, mu_b)
        W[b]    = float(mu_b @ x)
        pval[b] = 1.0 - chi2.cdf(W[b], df=2)

        mu[b]  = mu_b
        Sig[b] = Sig_b

    reject = bh_fdr(pval, alpha_fdr)
    print(f"[Train {s_plot}] BH-FDR @ α={alpha_fdr}: {int(reject.sum())} significant bands")
    for b in np.where(reject)[0]:
        A  = float(np.linalg.norm(mu[b]))
        ph = float(np.arctan2(mu[b,1], mu[b,0]))
        print(f"  b={b:3d} | f={labels[b]:7.3f} Hz | p={pval[b]:.3g} | δ=√W={np.sqrt(W[b]):.2f} | |β|≈{A:.3g} | φ={ph:+.3f} rad")

    # Plot |β| vs proper null: z ~ N(0, Σ̂)
    max_cols = 5
    cols = min(max_cols, B)
    rows = int(np.ceil(B / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 2.4*rows), sharex=False, sharey=False)
    axes = np.atleast_1d(axes).ravel()

    for b in range(B):
        ax = axes[b]
        Xb = np.column_stack([beta_R[:, b], beta_I[:, b]])
        Xw = winsorize_cols(Xb, trim, 1 - trim) if trim > 0 else Xb

        # use centered covariance Sig[b] for the null
        Z   = rng.multivariate_normal(mean=[0.0, 0.0], cov=Sig[b], size=20000)
        mag_obs  = np.linalg.norm(Xw, axis=1)
        mag_null = np.linalg.norm(Z,  axis=1)

        mmax = np.nanmax([mag_obs.max(), mag_null.max()])
        bins = np.linspace(0, mmax, bins_per+1)

        ax.hist(mag_null, bins=bins, density=True, alpha=0.45, color="gray", label="Null |z| (μ=0, Σ=Σ̂)")
        ax.hist(mag_obs,  bins=bins, density=True, alpha=0.75, color="C0",   label="Posterior |β|")

        # annotate Wald decision
        tag = "sig" if reject[b] else "ns"
        ax.set_title(f"{labels[b]:.1f} Hz  (W={W[b]:.2f}, p={pval[b]:.3g}, {tag})", fontsize=10)
        if b == 0:
            ax.legend(frameon=False, fontsize=9)
        ax.set_xlabel("|β|"); ax.set_ylabel("Density"); ax.grid(alpha=0.15)

    for k in range(B, rows*cols):
        axes[k].set_axis_off()
    fig.suptitle("Per-band |β| vs null |z|, z ~ N(0, Σ̂)", y=1.02, fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()
    


