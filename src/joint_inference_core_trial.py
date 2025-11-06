# joint_inference_core_trial.py
# Precision-pooled trial wrapper around your existing joint_inference_core.joint_kf_rts_moments
# No changes to joint_inference_core.py are required.

from __future__ import annotations
from typing import Optional, Sequence, Tuple
import numpy as np

from src.params import OUParams
from src.joint_inference_core import joint_kf_rts_moments, JointMoments


# ----------------------------- LFP pooling -----------------------------
def _pool_lfp_trials(Y_trials: np.ndarray,
                     sig_eps_trials: np.ndarray,
                     eps: float = 1e-20) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inverse-variance pool LFP complex rows across trials at fixed (j,m,k).

    Parameters
    ----------
    Y_trials        : (R,J,M,K) complex
    sig_eps_trials  : (R,J,M)   real (per-trial observation std)
    eps             : small numerical guard

    Returns
    -------
    Y_pooled  : (J,M,K) complex       (precision-weighted mean across trials)
    sig_eps_p : (J,M)   real (std)    (effective pooled observation std)
    """
    R, J, M, K = Y_trials.shape
    var_rm = np.asarray(sig_eps_trials, float) ** 2                        # (R,J,M)
    w_rm = 1.0 / np.maximum(var_rm, eps)                                   # (R,J,M)
    wsum_jm = w_rm.sum(axis=0)                                             # (J,M)
    var_pool = 1.0 / np.maximum(wsum_jm, eps)                              # (J,M)

    # Weighted average across trials for each (j,m,k)
    Y_num = (w_rm[..., None] * Y_trials).sum(axis=0)                       # (J,M,K)
    Y_den = np.maximum(wsum_jm[..., None], eps)                             # (J,M,1)
    Y_pooled = Y_num / Y_den
    sig_eps_pool = np.sqrt(var_pool)
    return Y_pooled, sig_eps_pool


# --------------------------- Spike PG pooling --------------------------
def _pool_spike_pg(
    spikes_SRT: np.ndarray,      # (S,R,T), values in {0,1} (but treated as floats)
    omega_SRT:  np.ndarray,      # (S,R,T), PG weights
    H_SRTL: Optional[np.ndarray],# (S,R,T,L) or None
    gamma_SRL: Optional[np.ndarray], # (S,R,L) or None
    omega_floor: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Pool PG pseudo-rows across trials to a single row per unit and time.

    Target is to produce arrays compatible with joint_kf_rts_moments (S,T) and (S,T,L):
        y_spk(s,t) = (sum_r κ_{srt} / sum_r ω_{srt}) - β0_s - h̄_{s,t}
        R_spk(s,t) = 1 / sum_r ω_{srt}
    Since joint_kf_rts_moments re-creates y_spk internally as (κ/ω) - β0 - Hγ,
    we encode h̄ into an adjusted κ_eff':
        κ_eff' = (sum_r κ) - (sum_r ω) * h̄
    and pass H=0, γ=0 so that hist=0 there.

    Returns
    -------
    spikes_eff : (S,T)  (float)   with κ_eff' = spikes_eff - 0.5
    omega_eff  : (S,T)
    H_eff      : (S,T,1) all zeros (so hist=0)
    gamma_eff  : (S,1)   all zeros
    """
    S, R, T = spikes_SRT.shape
    ω = np.maximum(np.asarray(omega_SRT, float), omega_floor)              # (S,R,T)
    κ = np.asarray(spikes_SRT, float) - 0.5                                # (S,R,T)

    # Weighted history mean h̄_{s,t}
    if (H_SRTL is not None) and (gamma_SRL is not None):
        H = np.asarray(H_SRTL, float)                                      # (S,R,T,L)
        G = np.asarray(gamma_SRL, float)                                   # (S,R,L)
        # h_{s,r,t} = H_{s,r,t,:} · gamma_{s,r,:}
        h_srt = np.einsum('srtl,srl->srt', H, G, optimize=True)            # (S,R,T)
        ωsum = ω.sum(axis=1)                                               # (S,T)
        ωsum = np.maximum(ωsum, omega_floor)
        h_bar = (ω * h_srt).sum(axis=1) / ωsum                             # (S,T)
    else:
        ωsum = ω.sum(axis=1)                                               # (S,T)
        ωsum = np.maximum(ωsum, omega_floor)
        h_bar = np.zeros_like(ωsum)

    # Adjusted kappa to bake in h̄, keep β0 subtraction in core:
    κ_eff_prime = κ.sum(axis=1) - ωsum * h_bar                             # (S,T)
    spikes_eff = κ_eff_prime + 0.5                                         # (S,T)
    omega_eff  = ωsum                                                      # (S,T)

    # Pass zero history so hist term in core is 0
    H_eff = np.zeros((S, T, 1), dtype=np.float64)
    gamma_eff = np.zeros((S, 1), dtype=np.float64)
    return spikes_eff, omega_eff, H_eff, gamma_eff


# ---------------------------- Public entrypoint ----------------------------
def joint_kf_rts_moments_trials(
    Y_cube: np.ndarray,                  # (R,J,M,K) OR (J,M,K) complex, derotated & scaled
    theta: OUParams,                     # OUParams(.lam, .sig_v, .sig_eps) (J,M)
    delta_spk: float,
    win_sec: float,
    offset_sec: float,
    beta: np.ndarray,                    # (P,) OR (S,P) OR (S,R,P)  — we'll reduce over trials if needed
    gamma: Optional[np.ndarray],         # (L,) OR (S,L) OR (S,R,L)  — optional
    spikes: np.ndarray,                  # (T,) OR (S,T) OR (S,R,T)
    omega: np.ndarray,                   # (T,) OR (S,T) OR (S,R,T)
    coupled_bands_idx: Sequence[int],
    freqs_for_phase: Sequence[float],
    sidx,                                # StateIndex (unused here; kept for API mirror)
    H_hist: Optional[np.ndarray],        # (T,L) OR (S,T,L) OR (S,R,T,L) — optional
    *,
    sigma_u: float = 0.0,
    omega_floor: float = 1e-6,
    # Optional per-trial LFP noise for pooling; if None, uses theta.sig_eps for all trials
    sig_eps_trials: Optional[np.ndarray] = None,   # (R,J,M)
    pool_lfp_trials: bool = True,
    pool_spike_trials: bool = True,
) -> JointMoments:
    """
    Trial-aware wrapper (precision pooling) that reduces trial-structured inputs to
    the shapes expected by joint_inference_core.joint_kf_rts_moments, then calls it.

    Notes
    -----
    * LFP trials are inverse-variance pooled per (j,m,k) if Y_cube is (R,J,M,K).
    * Spike PG pseudo-rows are pooled per unit/time if spikes are (S,R,T):
        (ω_eff, κ_eff') are constructed so the core computes y_spk equal to the
        pooled row; history is set to zero in the core by passing H=0, γ=0.
    * β per trial (S,R,P) is reduced to (S,P) via median across trials.
    """
    # ---------------- LFP: pool across trials if needed ----------------
    if Y_cube.ndim == 4:
        if not pool_lfp_trials:
            raise NotImplementedError(
                "Per-trial LFP without pooling is not supported by this wrapper. "
                "Set pool_lfp_trials=True."
            )
        Rtr, J, M, K = Y_cube.shape
        if sig_eps_trials is None:
            # broadcast shared sig_eps to all trials
            se = np.asarray(theta.sig_eps, float)           # (J,M)
            sig_eps_trials = np.broadcast_to(se[None, ...], (Rtr, J, M))
        else:
            assert sig_eps_trials.shape == (Rtr, J, M), "sig_eps_trials must be (R,J,M)"
        Y_use, sig_eps_pool = _pool_lfp_trials(Y_cube, sig_eps_trials)  # (J,M,K), (J,M)
        theta_use = OUParams(lam=theta.lam, sig_v=theta.sig_v, sig_eps=sig_eps_pool)
    elif Y_cube.ndim == 3:
        Y_use = Y_cube
        theta_use = theta
        J, M, K = Y_use.shape
    else:
        raise ValueError("Y_cube must be (J,M,K) or (R,J,M,K)")

    # ---------------- Spikes/PG/history: to (S,T) after pooling ----------
    # Normalize to at least 2D on spikes/omega
    if spikes.ndim == 1:
        spikes = spikes[None, :]
        omega  = omega[None, :]
    S_like = spikes.shape[0]

    if spikes.ndim == 3:
        # (S,R,T)
        if not pool_spike_trials:
            # Flatten trials into pseudo-units (S*R,T): allowed, but β must match (S*R,P)
            S, R, T = spikes.shape
            spikes_eff = spikes.reshape(S*R, T)
            omega_eff  = omega.reshape(S*R, T)
            # History: choose matching shape or zero
            if (H_hist is not None) and (gamma is not None):
                if H_hist.ndim == 4 and gamma.ndim == 3:
                    H_eff = H_hist.reshape(S*R, T, H_hist.shape[-1])
                    G_eff = gamma.reshape(S*R, gamma.shape[-1])
                else:
                    H_eff = np.zeros((S*R, T, 1), dtype=np.float64)
                    G_eff = np.zeros((S*R, 1), dtype=np.float64)
            else:
                H_eff = np.zeros((S*R, T, 1), dtype=np.float64)
                G_eff = np.zeros((S*R, 1), dtype=np.float64)
            S_out = S * R
        else:
            # Precision pool to a single row per unit/time, bake history into κ
            spikes_eff, omega_eff, H_eff, G_eff = _pool_spike_pg(
                spikes_SRT=spikes, omega_SRT=omega,
                H_SRTL=(H_hist if (H_hist is not None and H_hist.ndim == 4) else None),
                gamma_SRL=(gamma if (gamma is not None and gamma.ndim == 3) else None),
                omega_floor=omega_floor
            )
            S_out, T = spikes_eff.shape
        spikes_ST, omega_ST, H_STL, gamma_SL = spikes_eff, omega_eff, H_eff, G_eff

    elif spikes.ndim == 2:
        # (S,T)
        S_out, T = spikes.shape
        spikes_ST = np.asarray(spikes, float)
        omega_ST  = np.maximum(np.asarray(omega, float), omega_floor)
        if H_hist is not None:
            if H_hist.ndim == 3:    # (S,T,L)
                H_STL = H_hist
            elif H_hist.ndim == 2:  # (T,L) → broadcast over S
                H_STL = np.broadcast_to(H_hist[None, ...], (S_out, T, H_hist.shape[1]))
            else:
                raise ValueError("H_hist must be (T,L) or (S,T,L) when spikes are (S,T)")
        else:
            H_STL = np.zeros((S_out, T, 1), dtype=np.float64)

        if gamma is not None:
            if gamma.ndim == 2:     # (S,L)
                gamma_SL = gamma
            elif gamma.ndim == 1:   # (L,) → broadcast over S
                gamma_SL = np.broadcast_to(gamma[None, :], (S_out, gamma.shape[0]))
            else:
                raise ValueError("gamma must be (L,) or (S,L) when spikes are (S,T)")
        else:
            gamma_SL = np.zeros((S_out, H_STL.shape[2]), dtype=np.float64)
    else:
        raise ValueError("spikes must be (T,), (S,T) or (S,R,T)")

    # ---------------- β: reduce per-trial β if provided -------------------
    # β shapes allowed here: (P,), (S_out,P), or (S,R,P) with R-reduction
    if beta.ndim == 1:
        beta_SP = np.broadcast_to(beta[None, :], (S_out, beta.shape[0]))
    elif beta.ndim == 2:
        if beta.shape[0] != S_out:
            # try broadcasting a single β across units
            if beta.shape[0] == 1:
                beta_SP = np.broadcast_to(beta, (S_out, beta.shape[1]))
            else:
                raise ValueError(f"beta shape {beta.shape} incompatible with S={S_out}")
        else:
            beta_SP = beta
    elif beta.ndim == 3:
        # (S,R,P) → reduce across R (median is robust)
        beta_SP = np.median(beta, axis=1)
        if beta_SP.shape[0] != S_out:
            raise ValueError(f"Reduced beta shape {beta_SP.shape} incompatible with S={S_out}")
    else:
        raise ValueError("beta must be (P,), (S,P) or (S,R,P)")

    # ---------------- Delegate to the original core ----------------------
    # joint_kf_rts_moments expects:
    #   Y_cube: (J,M,K) complex
    #   beta  : (S,P) or (P,) — we pass (S_out,P) for multi-unit
    #   gamma : (S,L)
    #   spikes: (S,T)
    #   omega : (S,T)
    #   H_hist: (S,T,L)
    out = joint_kf_rts_moments(
        Y_cube=Y_use,
        theta=theta_use,
        delta_spk=delta_spk,
        win_sec=win_sec,
        offset_sec=offset_sec,
        beta=beta_SP,
        gamma=gamma_SL,
        spikes=spikes_ST,
        omega=omega_ST,
        coupled_bands_idx=coupled_bands_idx,
        freqs_for_phase=freqs_for_phase,
        sidx=sidx,
        H_hist=H_STL,
        sigma_u=sigma_u,
        omega_floor=omega_floor,
    )
    return out
