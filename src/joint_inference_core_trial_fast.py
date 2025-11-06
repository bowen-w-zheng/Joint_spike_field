# src/joint_inference_core_trial_fast.py
from __future__ import annotations
from typing import Optional, Sequence, Tuple
import numpy as np
from src.params import OUParams
from src.joint_inference_core import joint_kf_rts_moments, JointMoments

def _pool_lfp_trials(Y_trials: np.ndarray, sig_eps_trials: np.ndarray, eps: float = 1e-20) -> Tuple[np.ndarray, np.ndarray]:
    var = np.asarray(sig_eps_trials, float)**2
    w   = 1.0/np.maximum(var, eps)                         # (R,J,M)
    wsum= w.sum(axis=0)                                    # (J,M)
    Ynum= (w[...,None]*Y_trials).sum(axis=0)               # (J,M,K)
    Y_pool = Ynum/np.maximum(wsum[...,None], eps)
    sig_pool = np.sqrt(1.0/np.maximum(wsum, eps))          # (J,M)
    return Y_pool, sig_pool

def _pool_spike_pg_exact(
    spikes_SRT: np.ndarray, omega_SRT: np.ndarray,
    H_SRTL: Optional[np.ndarray], gamma_shared: Optional[np.ndarray],    # gamma_shared: (S,L) or None
    omega_floor: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    S,R,T = spikes_SRT.shape
    ω = np.maximum(np.asarray(omega_SRT, float), omega_floor)            # (S,R,T)
    κ = np.asarray(spikes_SRT, float) - 0.5                              # (S,R,T)
    ωST = np.maximum(ω.sum(axis=1), omega_floor)                         # (S,T)

    if (H_SRTL is not None) and (gamma_shared is not None):
        H = np.asarray(H_SRTL, float)                                    # (S,R,T,L)
        g = np.asarray(gamma_shared, float)                              # (S,L)
        g = g[:, None, :]                                                # (S,1,L) -> broadcast to (S,R,L) in einsum
        h_SRT = np.einsum('srtl,ssl->srt', H, np.broadcast_to(g,(S,S,g.shape[-1])))  # trick to broadcast g over R
        # simpler & clearer:
        # h_SRT = np.einsum('srtl,sl->srt', H, gamma_shared)
        h_SRT = np.einsum('srtl,sl->srt', H, gamma_shared)
        hbar_ST = (ω*h_SRT).sum(axis=1) / ωST                            # (S,T)
    else:
        hbar_ST = np.zeros_like(ωST)

    κ_sum_ST = κ.sum(axis=1)
    κ_eff_prime_ST = κ_sum_ST - ωST * hbar_ST
    spikes_eff_ST = κ_eff_prime_ST + 0.5

    H_eff_STL    = np.zeros((S, T, 1), float)
    gamma_eff_SL = np.zeros((S, 1), float)
    return spikes_eff_ST, ωST, H_eff_STL, gamma_eff_SL

def joint_kf_rts_moments_trials_fast(
    Y_trials: np.ndarray, theta: OUParams,
    delta_spk: float, win_sec: float, offset_sec: float,
    beta: np.ndarray,                     # (S,P) or (S,R,P) -> reduced to (S,P) via median
    gamma_shared: Optional[np.ndarray],   # (S,L) shared across trials (per unit)
    spikes: np.ndarray, omega: np.ndarray,# (S,R,T)
    coupled_bands_idx: Sequence[int], freqs_for_phase: Sequence[float], sidx,
    H_hist: Optional[np.ndarray],         # (S,R,T,L)
    *, sigma_u: float = 0.0, omega_floor: float = 1e-6, sig_eps_trials: Optional[np.ndarray] = None
) -> JointMoments:
    # LFP pooling
    if Y_trials.ndim == 4:
        R,J,M,K = Y_trials.shape
        if sig_eps_trials is None:
            se = np.asarray(theta.sig_eps, float); sig_eps_trials = np.broadcast_to(se[None,...], (R, se.shape[0], se.shape[1]))
        Y_use, sig_pool = _pool_lfp_trials(Y_trials, sig_eps_trials)
        theta_use = OUParams(lam=theta.lam, sig_v=theta.sig_v, sig_eps=sig_pool)
    else:
        Y_use = Y_trials; theta_use = theta
        J,M,K = Y_use.shape

    # Spikes pooling (exact)
    spikes_eff, omega_eff, H_eff, gamma_eff = _pool_spike_pg_exact(
        spikes_SRT=spikes, omega_SRT=omega, H_SRTL=H_hist, gamma_shared=gamma_shared, omega_floor=omega_floor
    )

    # β shared per unit for exact pooled row
    beta_use = np.median(beta, axis=1) if beta.ndim==3 else beta

    return joint_kf_rts_moments(
        Y_cube=Y_use, theta=theta_use,
        delta_spk=delta_spk, win_sec=win_sec, offset_sec=offset_sec,
        beta=beta_use, gamma=gamma_eff, spikes=spikes_eff, omega=omega_eff,
        coupled_bands_idx=coupled_bands_idx, freqs_for_phase=freqs_for_phase,
        sidx=sidx, H_hist=H_eff, sigma_u=sigma_u, omega_floor=omega_floor
    )
