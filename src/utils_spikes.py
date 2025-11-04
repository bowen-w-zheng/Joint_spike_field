import numpy as np
from typing import NamedTuple, Sequence
from dataclasses import dataclass
from typing import Callable

@dataclass
class StateIndex(NamedTuple):
    offset: Callable[[int, int, str], int]

def _build_H_spike_dense(
    T_f: int, d: int,
    coupled_bands_idx: np.ndarray,    # (B,) 0-based
    beta: np.ndarray,                 # (1+2B,)
    freqs_for_phase: np.ndarray,      # (B,) Hz aligned
    delta: float, J: int, M: int, sidx: StateIndex,
) -> np.ndarray:
    H = np.zeros((T_f, d), dtype=np.float64)
    t = np.arange(T_f) * delta
    for bi in np.asarray(coupled_bands_idx, np.int64):
        f_hz = float(freqs_for_phase[bi])
        βR   = float(beta[1 + 2*bi]); βI = float(beta[2 + 2*bi])
        c, s = np.cos(2*np.pi*f_hz*t), np.sin(2*np.pi*f_hz*t)
        a     = (βR * c + βI * s) / M
        bcoef = (-βR * s +  βI * c) / M
        for m in range(M):
            ri = sidx.offset(band=int(bi), taper=m, comp="real")
            ii = sidx.offset(band=int(bi), taper=m, comp="imag")
            H[:, ri] = a; H[:, ii] = bcoef
    return H

def build_history_lags_binary(spikes: np.ndarray, n_lags: int = 25) -> np.ndarray:
    """
    spikes: (T,) in {0,1}, time bin = 4 ms
    Returns H: (T, n_lags) with H[n, k-1] = spikes[n-k] (0 if n-k<0).
    """
    T = spikes.shape[0]
    H = np.zeros((T, n_lags), dtype=spikes.dtype)
    for k in range(1, n_lags + 1):
        H[k:, k-1] = spikes[:-k]
    return H

def check_spike_shape(spike_all):
    if spike_all.ndim == 1:
        spike_all = spike_all[None, :]                    # one train -> (1, T)
    elif spike_all.shape[0] < spike_all.shape[1]:
    # likely already (S, T)
        pass
    else:
        # probably (T, S) -> transpose
        spike_all = spike_all.T
    return spike_all

# OR-downsample within non-overlapping blocks of length `downsample_factor`
def downsample_binary_or(x_2d, q):
    # x_2d: (S, T); returns (S, T//q) with OR in each block
    S, T = x_2d.shape
    L = (T // q) * q
    if L != T:
        x_2d = x_2d[:, :L]
    x_blk = x_2d.reshape(S, -1, q)
    return (x_blk.sum(axis=2) > 0).astype(np.int8)

def build_history_lags_binary_multi(spikes_S, n_lags):
    S, T = spikes_S.shape
    H = np.zeros((S, T, n_lags), dtype=np.int8)
    for s in range(S):
        H[s] = build_history_lags_binary(spikes_S[s], n_lags=n_lags)
    return H


import numpy as np

def lag_bins_from_seconds(lags_sec, delta_spk):
    """
    Convert lag times (sec) → integer bins (>=1).
    Ensures uniqueness and sorting.
    """
    l = np.asarray(lags_sec, float).ravel()
    bins = np.maximum(1, np.round(l / float(delta_spk)).astype(int))
    return np.unique(bins)

def build_history_lags_hier(spikes, lag_bins, mode='self'):
    """
    Hierarchical spike history, **per trial** (no cross-trial leakage).

    Parameters
    ----------
    spikes  : (R, S, T) uint8/bool — trials × units × time (0/1)
    lag_bins: 1D array of positive ints (in bins)
    mode    : 'self' | 'self+pop' | 'all'
      - 'self'     → features = [x_s(t-ℓ) for ℓ in lag_bins]
      - 'self+pop' → features = [self_lags | sum_{u≠s} x_u(t-ℓ)]
      - 'all'      → features = [x_u(t-ℓ) for u=0..S-1]  (for every target s)

    Returns
    -------
    H_hist : np.ndarray, shape (R, S, T, R_h)
      R_h = len(lags)               if mode='self'
          = 2*len(lags)             if mode='self+pop'
          = S*len(lags)             if mode='all'
    """
    X = np.asarray(spikes, dtype=np.float32)       # (R,S,T)
    R, S, T = X.shape
    lags = np.asarray(lag_bins, int)
    if lags.ndim != 1 or np.any(lags <= 0):
        raise ValueError("lag_bins must be 1D positive integers")

    L = int(lags.size)
    if mode == 'self':
        R_h = L
        H = np.zeros((R, S, T, R_h), dtype=np.float32)
        for i, Lg in enumerate(lags):
            if Lg < T:
                H[..., Lg:, i] = X[..., :-Lg]
        return H

    if mode == 'self+pop':
        R_h = 2 * L
        H = np.zeros((R, S, T, R_h), dtype=np.float32)
        # population = sum of others per target unit
        pop = X.sum(axis=1, keepdims=True) - X     # (R,S,T)
        for i, Lg in enumerate(lags):
            if Lg < T:
                # self-lag
                H[..., Lg:, i]      = X[..., :-Lg]
                # population (others) lag
                H[..., Lg:, L + i]  = pop[..., :-Lg]
        return H

    if mode == 'all':
        # pairwise: presynaptic u for every target s
        R_h = S * L
        H = np.zeros((R, S, T, R_h), dtype=np.float32)
        for i, Lg in enumerate(lags):
            if Lg < T:
                # For each presynaptic unit u, broadcast to all targets s
                for u in range(S):
                    # feat for presyn u at lag Lg, shape (R,1,T-Lg) → broadcast to (R,S,T-Lg)
                    feat = X[:, u, :-Lg][:, None, :]
                    H[:, :, Lg:, u*L + i] = feat
        return H

    raise ValueError("mode must be one of {'self','self+pop','all'}")
