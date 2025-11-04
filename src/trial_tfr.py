# src/trial_tfr.py
from __future__ import annotations
from typing import Tuple
import numpy as np
import spynal.spectra as spy_spec

# cache κ(fs) once per sampling rate
_SPYNAL_KAPPA = {}

def _estimate_kappa(fs: float, window_ref: float = 0.5, NW_ref: float = 3.0, f_ref: float = 30.0) -> float:
    M_ref = int(round(window_ref * fs))
    df = fs / M_ref
    f_ref_grid = round(f_ref / df) * df  # snap to grid
    t = np.arange(4 * M_ref, dtype=float) / fs
    x = np.cos(2 * np.pi * f_ref_grid * t)
    W_ref = NW_ref / window_ref
    S_raw, freqs, t_cent = spy_spec.spectrogram(
        data=x[None, :], smp_rate=fs, axis=1,
        method="multitaper", spec_type="complex",
        removeDC=False, keep_tapers=True,
        time_width=window_ref, spacing=window_ref,
        freq_width=W_ref, n_tapers=int(max(1, np.floor(2 * NW_ref - 1))), pad=False,
    )
    S = S_raw[0] * np.exp(1j * 2 * np.pi * freqs[:, None] * t_cent[None, :])[:, None, :]  # (F,K,T)
    C_ref = S.mean(axis=1)  # (F,T)
    j = int(np.argmin(np.abs(np.asarray(freqs, float) - f_ref_grid)))
    mean_mag = float(np.mean(np.abs(C_ref[j])))
    kappa = (1.0 / max(mean_mag, 1e-20))
    return kappa


def compute_trial_tfr_multitaper(
    lfp_trials: np.ndarray,   # (R, T)
    fs: float,
    window_sec: float,
    NW: float,
    *,
    f_max_hz: float = 60.0,
    apply_amplitude_scale: bool = True,  # γ(T,NW;fs) = κ(fs)*sqrt(NW/T)
    remove_dc: bool = False,
    pad: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Spynal multitaper spectrogram (complex), all tapers kept, center-demodulated.
    Returns (Y_trials, centres_sec, freqs_sel, M) with Y_trials shape (R, K, J, K_frames).
    """
    R, T = lfp_trials.shape
    M = int(round(window_sec * fs))
    if M < 2:
        raise ValueError(f"window_sec too small for fs={fs}: M={M} < 2")
    K_req = int(max(1, np.floor(2 * NW - 1)))
    W = NW / window_sec  # half-bandwidth (Hz)

    # Spynal MT (handles trials intrinsically)
    S_raw, freqs, t_cent = spy_spec.spectrogram(
        data=lfp_trials, smp_rate=fs, axis=1,
        method="multitaper", spec_type="complex",
        removeDC=bool(remove_dc), keep_tapers=True,
        time_width=window_sec, spacing=window_sec,
        freq_width=W, n_tapers=K_req, pad=bool(pad),
    )  # (R, F, K, K_frames)

    freqs = np.asarray(freqs, float)
    keep = (freqs > 0) & (freqs <= float(f_max_hz))
    S_raw = S_raw[:, keep, :, :]                  # (R, J, K, T_frames)
    freqs_sel = freqs[keep]
    _, J, K_eff, K_frames = S_raw.shape

    # Center-demodulate: rot shape (1, J, 1, K_frames) → broadcasts over (R, J, K, K_frames)
    t_cent = np.asarray(t_cent, float)
    rot = np.exp(1j * 2 * np.pi * freqs_sel[None, :, None, None] * t_cent[None, None, None, :])
    S_demod = S_raw * rot

    # Global amplitude scale γ(T,NW;fs) that preserves correlation
    if apply_amplitude_scale:
        kappa = _estimate_kappa(fs)                  # once per fs
        S_demod = S_demod * kappa

    # Reorder to (R, K, J, K_frames)
    Y_trials = np.transpose(S_demod, (0, 2, 1, 3))
    centres_sec = t_cent.copy()
    return Y_trials, centres_sec, freqs_sel, M
