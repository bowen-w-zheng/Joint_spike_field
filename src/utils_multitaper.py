import numpy as np
def derotate_tfr(Y_cplx: np.ndarray,
                 freqs_hz: np.ndarray,
                 sfreq: float,
                 hop_samples: int,
                 win_samples: int,
                 t0: float = 0.0):
    """
    Y_cplx: (..., n_freqs, n_tapers, n_frames) from tfr_array_multitaper
    freqs_hz: frequency axis used in TFR (length = n_freqs)
    sfreq: sampling rate (Hz)
    hop_samples: stride between consecutive frames (samples); with decim=M, hop=M
    win_samples: window length (samples); with n_cycles=f*window_sec, win≈M
    t0: optional time offset (s)
    Supports Y_cplx of shape (n_freqs, n_tapers, n_frames) or
    (n_trials, n_freqs, n_tapers, n_frames)
    """
    if Y_cplx.ndim == 3:
        F, K, T = Y_cplx.shape
        centers_sec = t0 + (np.arange(T) * hop_samples + (win_samples - 1) / 2.0) / sfreq
        rot = np.exp(-1j * 2 * np.pi * freqs_hz[:, None, None] * centers_sec[None, None, :])
        return Y_cplx * rot
    elif Y_cplx.ndim == 4:
        R, F, K, T = Y_cplx.shape
        centers_sec = t0 + (np.arange(T) * hop_samples + (win_samples - 1) / 2.0) / sfreq
        # Broadcast trial axis
        rot = np.exp(-1j * 2 * np.pi * freqs_hz[None, :, None, None] * centers_sec[None, None, None, :])
        return Y_cplx * rot
    else:
        raise ValueError(f"Y_cplx with ndim={Y_cplx.ndim} unsupported for derotation")

def fine_to_cube_complex(fine_latent: np.ndarray, J: int, M: int) -> np.ndarray:
    """
    fine_latent: (T_f, d) with d == 2*J*M, flattened as (J, M, 2) in C-order
    returns:     (J, M, T_f) complex array
    """
    T_f, d = fine_latent.shape
    assert d == 2 * J * M, f"expected d=2*J*M={2*J*M}, got {d}"
    # reshape back to (T_f, J, M, 2), then split Re/Im and move time last
    tmp = fine_latent.reshape(T_f, J, M, 2, order="C")
    Z = (tmp[..., 0] + 1j * tmp[..., 1]).transpose(1, 2, 0)  # (J, M, T_f)
    return Z