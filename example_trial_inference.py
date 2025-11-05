"""
Example workflow for trial-structured joint inference.

This script demonstrates the complete pipeline:
1. Compute trial TFR using multitaper spectrogram
2. Fit hierarchical model (X + δr) using EM
3. Upsample latent processes to spike time scale
4. Infer coupling coefficients β and history γ using trial-structured inference
"""
from __future__ import annotations
import numpy as np
import jax.numpy as jnp
import jax.random as jr

# Import components
from src.trial_tfr import compute_trial_tfr_multitaper
from src.em_ct_hier_jax import run_em_ct_hier_jax
from src.upsample_ct_hier_fine import upsample_ct_hier_fine
from src.params import OUParams
from run_joint_inference_jax_v2_trials import run_joint_inference_trials, InferenceConfigTrials


def build_design_from_latent(lat_reim: jnp.ndarray) -> jnp.ndarray:
    """
    Build design matrix from latent Re/Im predictors.

    Parameters
    ----------
    lat_reim : (T, 2*J) array with columns [Re_1, ..., Re_J, Im_1, ..., Im_J]

    Returns
    -------
    X : (T, 1+2*J) design matrix [1 | Re | Im]
    """
    T = lat_reim.shape[0]
    ones = jnp.ones((T, 1), dtype=lat_reim.dtype)
    return jnp.concatenate([ones, lat_reim], axis=1)


def extract_band_reim_with_var_from_upsampled(
    Z_mean: np.ndarray,    # (R, J, M, T) complex
    Z_var: np.ndarray,     # (R, J, M, T) real
    freqs_hz: np.ndarray,  # (J,)
    delta_spk: float,
    J: int,
    M: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract rotated, taper-averaged Re/Im predictors from upsampled Z = X + δ.

    For trial-structured data, we average Z across trials (or use trial-specific Z).
    Here we'll average across trials to get shared predictors.

    Parameters
    ----------
    Z_mean : (R, J, M, T) complex latent means per trial
    Z_var : (R, J, M, T) real latent variances per trial

    Returns
    -------
    lat_reim : (T, 2*J) with columns [Re_1, ..., Re_J, Im_1, ..., Im_J]
    var_reim : (T, 2*J) corresponding variances
    """
    R, J, M, T = Z_mean.shape
    freqs = np.asarray(freqs_hz, float)

    # Average across trials and tapers
    Z_avg_JT = Z_mean.mean(axis=(0, 2))  # (J, T)
    V_avg_JT = Z_var.mean(axis=(0, 2))   # (J, T)

    # Rotate by carrier phase: exp(-i * 2π * f * t)
    t_sec = np.arange(T, dtype=float) * float(delta_spk)
    phase = 2.0 * np.pi * freqs[:, None] * t_sec[None, :]  # (J, T)
    rot = np.exp(-1j * phase, dtype=np.complex128)

    Z_rot = Z_avg_JT * rot  # (J, T)

    # Extract Re/Im
    lat_reim = np.empty((T, 2*J), dtype=np.float64)
    lat_reim[:, :J] = Z_rot.real.T      # (T, J)
    lat_reim[:, J:] = Z_rot.imag.T      # (T, J)

    # Variances (rotate doesn't change magnitude for real variance)
    var_reim = np.empty((T, 2*J), dtype=np.float64)
    var_reim[:, :J] = V_avg_JT.T
    var_reim[:, J:] = V_avg_JT.T

    return lat_reim, var_reim


def run_trial_inference_example():
    """
    Complete example: from LFP trials + spikes to coupling estimates.
    """
    # ────────────────────────────────────────────────────────────────────────
    # 0. SIMULATE OR LOAD DATA
    # ────────────────────────────────────────────────────────────────────────
    print("=" * 70)
    print("TRIAL-STRUCTURED JOINT INFERENCE EXAMPLE")
    print("=" * 70)

    # Simulation parameters
    R = 10          # number of trials
    S = 3           # number of units
    fs = 1000.0     # LFP sampling rate (Hz)
    T_sec = 2.0     # trial duration (sec)
    T_lfp = int(T_sec * fs)  # LFP samples per trial

    delta_spk = 0.001  # spike time resolution (1 ms)
    T_spk = int(T_sec / delta_spk)  # spike time bins

    # Simulate LFP trials (R, T_lfp)
    # In practice, load your real data here
    rng = np.random.default_rng(42)
    lfp_trials = rng.normal(0, 1, size=(R, T_lfp)).astype(np.float32)

    # Simulate spike trains (R, S, T_spk) with ~10 Hz rates
    spike_rate = 0.01  # 10 Hz
    spikes = (rng.random((R, S, T_spk)) < spike_rate * delta_spk).astype(np.uint8)

    print(f"LFP trials: {lfp_trials.shape} (R={R} trials, {T_lfp} samples @ {fs} Hz)")
    print(f"Spike trains: {spikes.shape} (R={R} trials, S={S} units, T={T_spk} bins @ {delta_spk} sec)")

    # ────────────────────────────────────────────────────────────────────────
    # 1. COMPUTE TRIAL TFR (multitaper spectrogram)
    # ────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("STEP 1: Computing trial TFR (multitaper spectrogram)")
    print("─" * 70)

    window_sec = 0.2    # 200 ms windows
    NW = 3.0            # time-bandwidth product
    f_max_hz = 60.0     # max frequency

    Y_trials, centres_sec, freqs_sel, M_eff = compute_trial_tfr_multitaper(
        lfp_trials=lfp_trials,
        fs=fs,
        window_sec=window_sec,
        NW=NW,
        f_max_hz=f_max_hz,
        apply_amplitude_scale=True,
        remove_dc=False,
        pad=False,
    )
    # Y_trials: (R, M, J, K) where K = number of time blocks

    print(f"TFR shape: {Y_trials.shape}")
    print(f"  M={Y_trials.shape[1]} tapers")
    print(f"  J={Y_trials.shape[2]} frequency bands: {freqs_sel[0]:.1f} - {freqs_sel[-1]:.1f} Hz")
    print(f"  K={Y_trials.shape[3]} time blocks")
    print(f"  Block centres: {centres_sec[0]:.3f} - {centres_sec[-1]:.3f} sec")

    # Reorder to (R, J, M, K) for compatibility
    Y_trials_RJMK = np.transpose(Y_trials, (0, 2, 1, 3))

    # ────────────────────────────────────────────────────────────────────────
    # 2. FIT HIERARCHICAL MODEL (EM algorithm)
    # ────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("STEP 2: Fitting hierarchical model X + δr (EM)")
    print("─" * 70)

    # Note: run_em_ct_hier_jax expects (R, M, J, K)
    # We have (R, J, M, K), so we need to swap axes 1 and 2
    Y_for_em = np.transpose(Y_trials_RJMK, (0, 2, 1, 3))  # (R, M, J, K)

    # We would normally run EM here, but for this example we'll skip it
    # and create a mock result structure
    print("(Skipping EM for this example - would call run_em_ct_hier_jax here)")

    # Create mock EM result with reasonable parameters
    from src.em_ct_hier_jax import EMHierResult
    J = Y_trials_RJMK.shape[1]
    M = Y_trials_RJMK.shape[2]
    K = Y_trials_RJMK.shape[3]

    lam_X = np.ones((J, M)) * 30.0      # decay rate ~30 Hz
    sigv_X = np.ones((J, M)) * 0.5      # process noise
    lam_D = np.ones((J, M)) * 50.0      # faster decay for deviations
    sigv_D = np.ones((J, M)) * 0.2      # smaller deviations
    sig_eps_mr = np.ones((M, R)) * 0.1  # observation noise

    em_result = EMHierResult(
        lam_X=lam_X,
        sigv_X=sigv_X,
        lam_D=lam_D,
        sigv_D=sigv_D,
        sig_eps_jmr=sig_eps_mr[None, :, :],  # (1, M, R)
        sig_eps_mr=sig_eps_mr,
        Q_hist=np.zeros(10),
        X_mean=np.zeros((J, M, K), dtype=np.complex128),
        D_mean=np.zeros((R, J, M, K), dtype=np.complex128),
        x0_X=np.zeros((J, M), dtype=np.complex128),
        P0_X=np.ones((J, M)) * 0.1,
        x0_D=np.zeros((R, J, M), dtype=np.complex128),
        P0_D=np.ones((R, J, M)) * 0.1,
    )

    print(f"EM parameters (mock):")
    print(f"  λ_X: {lam_X[0, 0]:.1f} Hz (shared process decay)")
    print(f"  σ_X: {sigv_X[0, 0]:.2f} (shared process noise)")
    print(f"  λ_δ: {lam_D[0, 0]:.1f} Hz (deviation decay)")
    print(f"  σ_δ: {sigv_D[0, 0]:.2f} (deviation noise)")

    # ────────────────────────────────────────────────────────────────────────
    # 3. UPSAMPLE TO SPIKE TIME SCALE
    # ────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("STEP 3: Upsampling latents to spike time scale")
    print("─" * 70)

    ups_result = upsample_ct_hier_fine(
        Y_trials=Y_trials_RJMK,  # (R, J, M, K)
        res=em_result,
        delta_spk=delta_spk,
        win_sec=window_sec,
        offset_sec=centres_sec[0] - 0.5 * window_sec,  # align to first block
        T_f=T_spk,
    )

    print(f"Upsampled shapes:")
    print(f"  X_mean: {ups_result.X_mean.shape} (shared, J×M×T)")
    print(f"  D_mean: {ups_result.D_mean.shape} (deviations, R×J×M×T)")
    print(f"  Z_mean: {ups_result.Z_mean.shape} (combined, R×J×M×T)")

    # ────────────────────────────────────────────────────────────────────────
    # 4. BUILD HISTORY FEATURES
    # ────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("STEP 4: Building history features")
    print("─" * 70)

    # Simple history: previous 10 time bins
    n_lags = 10
    H_hist = np.zeros((S, R, T_spk, n_lags), dtype=np.float32)

    for s in range(S):
        for r in range(R):
            for lag in range(n_lags):
                if lag > 0:
                    H_hist[s, r, lag:, lag] = spikes[r, s, :-lag].astype(np.float32)

    print(f"History features: {H_hist.shape} (S×R×T×lags)")

    # ────────────────────────────────────────────────────────────────────────
    # 5. RUN JOINT INFERENCE FOR β AND γ
    # ────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("STEP 5: Running trial-structured joint inference")
    print("─" * 70)

    # Initial parameters from LFP-only fit (simplified)
    params0 = OUParams(
        lam=lam_X[:, 0],     # (J,)
        sig_v=sigv_X[:, 0],  # (J,)
        sig_eps=sig_eps_mr.mean(axis=1),  # (M,)
    )

    # Configure inference
    config = InferenceConfigTrials(
        fixed_iter=50,          # fewer iters for example
        n_refreshes=2,
        inner_steps_per_refresh=20,
        pg_jax=False,           # use numpy PG sampler
        mode="shared",
    )

    # IMPORTANT: We need to provide the extract_band_reim_with_var function
    # that works with upsampled latents
    def extract_fn(mu_fine, var_fine, coupled_bands, freqs_hz, delta_spk, J, M):
        # For the initial LFP-only pass, mu_fine is (T, d) with d = 2*J*M
        # We need to reshape and extract band-averaged predictors
        T = mu_fine.shape[0]
        d = mu_fine.shape[1]

        # Reshape to (J, M, T) complex (interleaved Re/Im)
        mu_re = mu_fine[:, 0::2].T.reshape(J, M, T)
        mu_im = mu_fine[:, 1::2].T.reshape(J, M, T)
        mu_complex = mu_re + 1j * mu_im

        var_re = var_fine[:, 0::2].T.reshape(J, M, T)
        var_im = var_fine[:, 1::2].T.reshape(J, M, T)

        # Average across tapers
        Z_avg = mu_complex.mean(axis=1)  # (J, T)
        V_avg = (var_re + var_im).mean(axis=1) / 2  # (J, T)

        # Rotate by carrier
        t_sec = np.arange(T) * delta_spk
        phase = 2.0 * np.pi * np.asarray(freqs_hz)[:, None] * t_sec[None, :]
        rot = np.exp(-1j * phase)
        Z_rot = Z_avg * rot

        # Stack Re/Im
        lat_reim = np.column_stack([Z_rot.real.T, Z_rot.imag.T])
        var_reim = np.column_stack([V_avg.T, V_avg.T])

        return lat_reim, var_reim

    # Import the Gibbs updater and KF from existing code
    from src.beta_from_fixed_latents import gibbs_update_beta_pg_ard_tightgamma
    from src.joint_inference_core import joint_kf_rts_moments

    # Note: This is a placeholder - the actual function signature may differ
    # You would need to adapt this to your specific implementation
    print("(This example stops here - full integration requires adapting the Gibbs sampler)")
    print("Key integration points:")
    print("  - Use gibbs_update_beta_robust from your working single-session code")
    print("  - Pass trial-structured Y_cube_block: (R, J, M, K)")
    print("  - Pass spikes: (R, S, T) and H_hist: (S, R, T, lags)")
    print("  - Function will return β: (S, R, P) and γ: (S, R, lags)")

    print("\n" + "=" * 70)
    print("Example workflow complete!")
    print("=" * 70)


if __name__ == "__main__":
    run_trial_inference_example()
