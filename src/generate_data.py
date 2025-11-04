import numpy as np
from typing import Optional
from scipy.interpolate import interp1d
from src.utils_spikes import build_history_lags_binary
from src.priors import gamma_prior_simple


def generate_spikes_from_latent(
    Z_true: np.ndarray,             # (J, T_raw) complex latent per band
    time: np.ndarray,               # (T_raw,)
    freqs_hz: np.ndarray,           # (J,)
    beta_true: np.ndarray,          # legacy: [β0, βR, βI] OR multi: [β0, βR_1, βI_1, ..., βR_J, βI_J]
    delta_spk: float,               # fine bin (s)
    n_lags: int,
    gamma_true: Optional[np.ndarray] = None,
    rng: np.random.Generator = np.random.default_rng(0),
):
    """
    Returns
    -------
    spikes         : (T_f,) 0/1
    H_hist         : (T_f, n_lags)
    t_fine         : (T_f,)
    Z_til_by_band  : (J, T_f)  real(Z̃_b,t) for convenience (optional plotting)

    Notes
    -----
    • Backward-compatible:
        - If beta_true.shape == (3,), uses only freqs_hz[1] with [β0, βR, βI] (old behavior).
        - If beta_true.shape == (1 + 2*J,), uses all bands: β0 plus [βR_b, βI_b] per band.
    """
    T_f    = int(round(time[-1] / delta_spk)) + 1
    t_fine = np.arange(T_f) * delta_spk

    from scipy.interpolate import interp1d

    J = len(freqs_hz)

    # carrier-rotated per-band signals (store complex for logit; real for plotting)
    Z_til_complex = np.zeros((J, T_f), dtype=np.complex128)
    Z_til_by_band = np.zeros((J, T_f), dtype=float)  # real part only (convenience/plots)
    Z_bar = np.zeros((J, T_f), dtype=np.complex128)
    for i, f in enumerate(freqs_hz):
        ω = 2*np.pi*float(f)
        Z_bar_i = interp1d(time, Z_true[i], kind="linear", fill_value="extrapolate")(t_fine)  # complex OK
        Z_til_i = np.exp(1j*ω*t_fine) * Z_bar_i
        Z_til_complex[i] = Z_til_i
        Z_til_by_band[i] = Z_til_i.real
        Z_bar[i] = Z_bar_i
    # ----- parse β coefficients (legacy 1-band OR multi-band) ----------------
    if beta_true.ndim == 1 and beta_true.size == 3:
        # legacy: only band index 1 contributes
        beta0 = float(beta_true[0])
        beta_pairs = np.zeros((J, 2), dtype=float)       # [βR_b, βI_b] per band
        if J < 2:
            raise ValueError("Legacy 3-param β expects at least two bands (uses freqs_hz[1]).")
        beta_pairs[1, 0] = float(beta_true[1])  # βR for band 1
        beta_pairs[1, 1] = float(beta_true[2])  # βI for band 1
    elif beta_true.ndim == 1 and beta_true.size == 1 + 2*J:
        beta0 = float(beta_true[0])
        beta_pairs = beta_true[1:].reshape(J, 2).astype(float)  # [[βR_1,βI_1], ..., [βR_J,βI_J]]
    else:
        raise ValueError(
            f"beta_true must be length 3 (legacy) or 1+2*J={1+2*J}. Got shape {beta_true.shape}."
        )

    # spike history prior mean if none provided
    if gamma_true is None:
        mu_g, _ = gamma_prior_simple(n_lags=n_lags,
                                     strong_neg=-2.5, mild_neg=-0.5, k_mild=4, tau_gamma=1.5)
        gamma_true = mu_g.copy()

    spikes = np.zeros(T_f, dtype=np.int8)

    # pre-extract real/imag arrays for speed
    ZR = Z_til_complex.real   # (J, T_f)
    ZI = Z_til_complex.imag   # (J, T_f)

    for t in range(T_f):
        # history vector h_t = [spike(t-1),..., spike(t-n_lags)]
        start = max(0, t - n_lags)
        h = spikes[start:t][::-1]
        if h.size < n_lags:
            h = np.pad(h, (0, n_lags - h.size))

        # multi-band linear term: sum_b (βR_b * Re Z̃_b,t + βI_b * Im Z̃_b,t)
        lin_multiband = np.sum(beta_pairs[:, 0] * ZR[:, t] + beta_pairs[:, 1] * ZI[:, t])

        logit = beta0 + lin_multiband + float(np.dot(gamma_true, h))
        # numerically stable sigmoid
        if logit >= 0:
            z = np.exp(-logit)
            p = 1.0 / (1.0 + z)
        else:
            z = np.exp(logit)
            p = z / (1.0 + z)

        spikes[t] = (rng.random() < p)

    H_hist = build_history_lags_binary(spikes, n_lags=n_lags)
    return spikes, H_hist, t_fine, Z_til_by_band, Z_bar

