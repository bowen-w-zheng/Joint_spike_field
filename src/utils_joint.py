import numpy as np
from typing import NamedTuple, Sequence
from dataclasses import dataclass, field
from typing import List, Tuple
from src.params import OUParams
from scipy.linalg import block_diag


def build_design_with_history(ZR: np.ndarray, ZI: np.ndarray, H_hist: np.ndarray):
    """
    ZR, ZI: (T,) real/imag carrier-rotated taper-avg latent
    H_hist: (T, R) history features
    Returns X: (T, 3+R) = [1, ZR, ZI, H_hist...]
    """
    T = ZR.shape[0]
    return np.column_stack([np.ones(T), ZR, ZI, H_hist])

def make_beta_gamma_prior(n_hist: int,
                          mu_gamma, Sigma_gamma,
                          tau_beta: float = 5.0):
    # Priors: β0, βR, βI ~ N(0, tau_beta^2)
    mu0 = np.concatenate([np.zeros(3), mu_gamma])
    Sigma0 = block_diag((tau_beta**2) * np.eye(3), Sigma_gamma)
    return mu0, Sigma0

def _chol_solve(L, b):
    # Solve (L L^T) x = b   with two triangular solves
    y = np.linalg.solve(L, b)
    return np.linalg.solve(L.T, y)

@dataclass
class Trace:
    beta:        List[np.ndarray] = field(default_factory=list)
    gamma:       List[np.ndarray] = field(default_factory=list)  # keep γ trace
    theta:       List[OUParams]   = field(default_factory=list)
    latent:      List[np.ndarray] = field(default_factory=list)
    fine_latent: List[np.ndarray] = field(default_factory=list)

def extract_band_reim_with_var(
    mu_fine: np.ndarray,    # (T_f, d) smoothed means
    var_fine: np.ndarray,   # (T_f, d) smoothed diag variances
    coupled_bands: Sequence[float],
    freqs_hz: Sequence[float],
    delta_spk: float,
    J: int, M: int,
    *, _cache={}
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      X_lat   : (T_f, 2B)  = [Re Z̃_b, Im Z̃_b] per band, taper-averaged (×1/M)
      Var_lat: (T_f, 2B)  = Var[Re Z̃_b], Var[Im Z̃_b] propagated from var_fine
    Assumes independence across tapers and Re/Im (consistent with your diag smoother).
    """
    T_f, _ = mu_fine.shape
    B      = len(coupled_bands)
    X_lat  = np.empty((T_f, 2*B), dtype=np.float64)
    V_lat  = np.empty((T_f, 2*B), dtype=np.float64)

    from src.state_index import StateIndex
    idx = StateIndex(n_bands=J, n_tapers=M)

    for j, b in enumerate(coupled_bands):
        band_idx = int(np.where(freqs_hz == b)[0][0])

        key = (band_idx, T_f, delta_spk)
        c, s = _cache.get(key, (None, None))
        if c is None:
            t = np.arange(T_f) * delta_spk
            phi = 2.0 * np.pi * freqs_hz[band_idx] * t
            c, s = np.cos(phi), np.sin(phi)
            _cache[key] = (c, s)

        real_offs = [idx.offset(band=band_idx, taper=m, comp="real") for m in range(M)]
        imag_offs = [idx.offset(band=band_idx, taper=m, comp="imag") for m in range(M)]

        # taper-averaged mean
        re_mu = mu_fine[:, real_offs].mean(axis=1)
        im_mu = mu_fine[:, imag_offs].mean(axis=1)

        # taper-averaged variance: Var(mean) = (1/M^2) sum_m Var(Z_m)
        re_var = var_fine[:, real_offs].sum(axis=1) / (M**2)
        im_var = var_fine[:, imag_offs].sum(axis=1) / (M**2)

        # rotate (ignore Re/Im covariance which is zero in your diag smoother)
        X_lat[:, 2*j    ] =  c * re_mu - s * im_mu
        X_lat[:, 2*j + 1] =  s * re_mu + c * im_mu

        V_lat[:, 2*j    ] = (c**2) * re_var + (s**2) * im_var
        V_lat[:, 2*j + 1] = (s**2) * re_var + (c**2) * im_var

    return X_lat, V_lat