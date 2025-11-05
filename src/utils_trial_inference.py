"""
Utility functions for trial-structured joint inference.

This module provides helper functions for:
- Building predictors from upsampled hierarchical latents
- Pooling observations across trials
- Formatting data for trial-structured inference
"""
from __future__ import annotations
from typing import Tuple
import numpy as np
import jax.numpy as jnp


def build_predictors_from_upsampled_hier(
    ups_result,              # UpsampleResult from upsample_ct_hier_fine
    freqs_hz: np.ndarray,    # (J,) frequencies
    delta_spk: float,
    pooling: str = "trial_average",  # "trial_average", "per_trial", or "shared_only"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build rotated Re/Im predictors from upsampled hierarchical latents.

    Parameters
    ----------
    ups_result : UpsampleResult
        Output from upsample_ct_hier_fine with X_mean, D_mean, Z_mean
    freqs_hz : (J,) array
        Frequency bands (Hz)
    delta_spk : float
        Spike time resolution (sec)
    pooling : str
        "trial_average": average Z across trials -> (T, 2J)
        "per_trial": keep per-trial Z -> (R, T, 2J)
        "shared_only": use only shared X -> (T, 2J)

    Returns
    -------
    lat_reim : (T, 2J) or (R, T, 2J)
        Rotated Re/Im predictors
    var_reim : (T, 2J) or (R, T, 2J)
        Corresponding variances
    """
    freqs = np.asarray(freqs_hz, float)

    if pooling == "shared_only":
        # Use only shared component X
        X_mean = ups_result.X_mean  # (J, M, T)
        X_var = ups_result.X_var    # (J, M, T)
        J, M, T = X_mean.shape

        # Average across tapers
        Z_avg = X_mean.mean(axis=1)  # (J, T)
        V_avg = X_var.mean(axis=1)   # (J, T)

        # Rotate by carrier: exp(-i * 2π * f * t)
        t_sec = np.arange(T, dtype=float) * float(delta_spk)
        phase = 2.0 * np.pi * freqs[:, None] * t_sec[None, :]
        rot = np.exp(-1j * phase, dtype=np.complex128)
        Z_rot = Z_avg * rot  # (J, T)

        # Stack Re/Im
        lat_reim = np.column_stack([Z_rot.real.T, Z_rot.imag.T])  # (T, 2J)
        var_reim = np.column_stack([V_avg.T, V_avg.T])            # (T, 2J)

        return lat_reim, var_reim

    elif pooling == "trial_average":
        # Average Z = X + δ across trials
        Z_mean = ups_result.Z_mean  # (R, J, M, T)
        Z_var = ups_result.Z_var    # (R, J, M, T)
        R, J, M, T = Z_mean.shape

        # Average across trials and tapers
        Z_avg = Z_mean.mean(axis=(0, 2))  # (J, T)
        V_avg = Z_var.mean(axis=(0, 2))   # (J, T)

        # Rotate
        t_sec = np.arange(T, dtype=float) * float(delta_spk)
        phase = 2.0 * np.pi * freqs[:, None] * t_sec[None, :]
        rot = np.exp(-1j * phase, dtype=np.complex128)
        Z_rot = Z_avg * rot

        lat_reim = np.column_stack([Z_rot.real.T, Z_rot.imag.T])
        var_reim = np.column_stack([V_avg.T, V_avg.T])

        return lat_reim, var_reim

    elif pooling == "per_trial":
        # Keep per-trial predictors
        Z_mean = ups_result.Z_mean  # (R, J, M, T)
        Z_var = ups_result.Z_var    # (R, J, M, T)
        R, J, M, T = Z_mean.shape

        # Average across tapers only
        Z_avg = Z_mean.mean(axis=2)  # (R, J, T)
        V_avg = Z_var.mean(axis=2)   # (R, J, T)

        # Rotate per trial
        t_sec = np.arange(T, dtype=float) * float(delta_spk)
        phase = 2.0 * np.pi * freqs[None, :, None] * t_sec[None, None, :]  # (1, J, T)
        rot = np.exp(-1j * phase, dtype=np.complex128)
        Z_rot = Z_avg * rot  # (R, J, T)

        # Stack Re/Im per trial
        lat_reim = np.stack([
            np.column_stack([Z_rot[r].real.T, Z_rot[r].imag.T])
            for r in range(R)
        ], axis=0)  # (R, T, 2J)

        var_reim = np.stack([
            np.column_stack([V_avg[r].T, V_avg[r].T])
            for r in range(R)
        ], axis=0)  # (R, T, 2J)

        return lat_reim, var_reim

    else:
        raise ValueError(f"Unknown pooling mode: {pooling}")


def normalize_trial_shapes(
    spikes: np.ndarray,
    H_hist: np.ndarray,
    Y_cube: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Normalize trial-structured data to canonical shapes.

    Parameters
    ----------
    spikes : array
        Spike trains, any of:
        - (R, S, T): trials × units × time
        - (S, R, T): units × trials × time
    H_hist : array
        History features, any of:
        - (S, R, T, L): units × trials × time × lags
        - (S, T, L): units × time × lags (shared across trials)
        - (R, S, T, L): trials × units × time × lags
    Y_cube : array
        LFP observations, any of:
        - (R, J, M, K): trials × bands × tapers × blocks
        - (J, M, K): bands × tapers × blocks (single trial)

    Returns
    -------
    spikes_norm : (R, S, T)
    H_hist_norm : (S, R, T, L)
    Y_cube_norm : (R, J, M, K)
    info : dict with R, S, T, J, M, K, L
    """
    spikes = np.asarray(spikes)
    H_hist = np.asarray(H_hist)
    Y_cube = np.asarray(Y_cube)

    # ── Normalize spikes to (R, S, T) ──
    if spikes.ndim == 3:
        dim0, dim1, T = spikes.shape
        # Heuristic: R (trials) is typically smaller than S (units)
        if dim0 < dim1:
            R, S = dim0, dim1
            spikes_norm = spikes
        else:
            S, R = dim0, dim1
            spikes_norm = np.transpose(spikes, (1, 0, 2))
    else:
        raise ValueError(f"spikes must be 3D, got shape {spikes.shape}")

    # ── Normalize Y_cube to (R, J, M, K) ──
    if Y_cube.ndim == 4:
        if Y_cube.shape[0] == R:
            Y_cube_norm = Y_cube
            _, J, M, K = Y_cube.shape
        else:
            # Assume single-trial: (J, M, K) broadcast to (1, J, M, K)
            raise ValueError(f"Y_cube shape {Y_cube.shape} does not match R={R}")
    elif Y_cube.ndim == 3:
        # Single trial case: (J, M, K) -> (1, J, M, K)
        if R != 1:
            raise ValueError(f"Y_cube is 3D but R={R} trials expected")
        Y_cube_norm = Y_cube[None, :, :, :]
        J, M, K = Y_cube.shape
    else:
        raise ValueError(f"Y_cube must be 3D or 4D, got shape {Y_cube.shape}")

    # ── Normalize H_hist to (S, R, T, L) ──
    if H_hist.ndim == 4:
        if H_hist.shape[0] == S and H_hist.shape[1] == R:
            H_hist_norm = H_hist
            L = H_hist.shape[3]
        elif H_hist.shape[0] == R and H_hist.shape[1] == S:
            # (R, S, T, L) -> (S, R, T, L)
            H_hist_norm = np.transpose(H_hist, (1, 0, 2, 3))
            L = H_hist.shape[3]
        else:
            raise ValueError(f"H_hist shape {H_hist.shape} does not match S={S}, R={R}")
    elif H_hist.ndim == 3:
        # Shared history: (S, T, L) -> (S, R, T, L) by broadcasting
        if H_hist.shape[0] == S:
            S_, T_, L = H_hist.shape
            H_hist_norm = np.tile(H_hist[:, None, :, :], (1, R, 1, 1))
        else:
            raise ValueError(f"H_hist shape {H_hist.shape} does not match S={S}")
    else:
        raise ValueError(f"H_hist must be 3D or 4D, got shape {H_hist.shape}")

    info = dict(R=R, S=S, T=T, J=J, M=M, K=K, L=L)
    return spikes_norm, H_hist_norm, Y_cube_norm, info


def pool_observations_across_trials(
    Y_cube: np.ndarray,        # (R, J, M, K) complex
    sig_eps: np.ndarray,       # (J, M) or (R, J, M) noise std
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Precision-pool LFP observations across trials.

    Parameters
    ----------
    Y_cube : (R, J, M, K) complex TFR
    sig_eps : (J, M) or (R, J, M) noise standard deviations

    Returns
    -------
    Y_pooled : (J, M, K) complex pooled observations
    sig_pooled : (J, M) pooled noise standard deviations
    """
    R, J, M, K = Y_cube.shape

    # Expand sig_eps to (R, J, M) if needed
    sig_eps = np.asarray(sig_eps, float)
    if sig_eps.ndim == 2:
        sig_eps = np.tile(sig_eps[None, :, :], (R, 1, 1))  # (R, J, M)
    elif sig_eps.ndim == 3:
        assert sig_eps.shape == (R, J, M)
    else:
        raise ValueError(f"sig_eps must be (J,M) or (R,J,M), got {sig_eps.shape}")

    # Precision weights
    w = 1.0 / (sig_eps**2 + 1e-12)  # (R, J, M)

    # Pool across trials
    Y_pooled = np.zeros((J, M, K), dtype=Y_cube.dtype)
    sig_pooled = np.zeros((J, M), dtype=float)

    for j in range(J):
        for m in range(M):
            w_jm = w[:, j, m]  # (R,)
            for k in range(K):
                Y_jmk = Y_cube[:, j, m, k]  # (R,) complex
                Y_pooled[j, m, k] = (w_jm * Y_jmk).sum() / w_jm.sum()

            sig_pooled[j, m] = 1.0 / np.sqrt(w_jm.sum())

    return Y_pooled, sig_pooled


def build_spike_history_matrix(
    spikes: np.ndarray,        # (R, S, T) binary
    n_lags: int,
    dt: float = 1.0,
) -> np.ndarray:
    """
    Build spike history design matrix.

    Parameters
    ----------
    spikes : (R, S, T) binary spike trains
    n_lags : int
        Number of history lags
    dt : float
        Time bin width (for normalization)

    Returns
    -------
    H : (S, R, T, n_lags)
        History features per unit, trial, time, and lag
    """
    R, S, T = spikes.shape
    H = np.zeros((S, R, T, n_lags), dtype=float)

    for s in range(S):
        for r in range(R):
            for lag in range(n_lags):
                if lag == 0:
                    # Lag 0: use a small constant or skip
                    continue
                else:
                    # Lag τ: shift by τ bins
                    H[s, r, lag:, lag] = spikes[r, s, :-lag].astype(float)

    return H


def summarize_beta_per_trial(
    beta: np.ndarray,          # (S, R, P) or samples × (S, R, P)
    band_names: list[str] = None,
) -> dict:
    """
    Summarize β estimates across trials and units.

    Parameters
    ----------
    beta : (S, R, P) or (n_samples, S, R, P)
        Coupling coefficients
    band_names : list of str
        Names for each frequency band (for reporting)

    Returns
    -------
    summary : dict
        Statistics per unit and band
    """
    beta = np.asarray(beta)

    if beta.ndim == 3:
        # Single estimate
        S, R, P = beta.shape
        beta_samples = beta[None, :, :, :]
    elif beta.ndim == 4:
        # Multiple samples
        n_samples, S, R, P = beta.shape
        beta_samples = beta
    else:
        raise ValueError(f"beta must be 3D or 4D, got shape {beta.shape}")

    n_samples, S, R, P = beta_samples.shape
    B = (P - 1) // 2  # number of bands

    if band_names is None:
        band_names = [f"band_{j}" for j in range(B)]

    summary = {}

    # Per-unit, per-band, across trials
    for s in range(S):
        summary[f"unit_{s}"] = {}
        for j in range(B):
            # Real and imaginary parts
            beta_R = beta_samples[:, s, :, 1 + j]       # (n_samples, R)
            beta_I = beta_samples[:, s, :, 1 + B + j]   # (n_samples, R)

            # Magnitude
            beta_mag = np.sqrt(beta_R**2 + beta_I**2)   # (n_samples, R)

            summary[f"unit_{s}"][band_names[j]] = {
                "beta_R_mean": beta_R.mean(axis=0),    # (R,)
                "beta_R_std": beta_R.std(axis=0),
                "beta_I_mean": beta_I.mean(axis=0),
                "beta_I_std": beta_I.std(axis=0),
                "mag_mean": beta_mag.mean(axis=0),
                "mag_std": beta_mag.std(axis=0),
            }

    return summary
