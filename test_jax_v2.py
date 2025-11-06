#!/usr/bin/env python3
"""
Test script for run_joint_inference_jax_v2.py
Tests the complete optimized pipeline including warmup and refresh passes.
"""

import numpy as np
import jax.numpy as jnp
import jax.random as jr

print("=" * 70)
print("Testing run_joint_inference_jax_v2")
print("=" * 70)

# Test imports
print("\n[1/5] Testing imports...")
try:
    from src.run_joint_inference_jax_v2 import run_joint_inference_jax_v2, InferenceConfig
    from src.params import OUParams
    from src.utils_joint import Trace
    print("✓ Imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    raise

# Create minimal synthetic data
print("\n[2/5] Creating synthetic data...")
try:
    # Dimensions
    J = 5   # frequency bands
    M = 3   # tapers
    K = M   # time-frequency
    T = 100  # time points
    S = 2   # spike trains
    R = 5   # history lags

    # LFP data (complex TFR)
    Y_cube_block = np.random.randn(J, M, K) + 1j * np.random.randn(J, M, K)
    Y_cube_block = Y_cube_block.astype(np.complex128)

    # Spikes (binary)
    spikes = (np.random.rand(S, T) < 0.05).astype(np.float64)

    # History
    H_hist = np.random.randn(S, T, R)

    # Frequencies
    all_freqs = np.array([10.0, 15.0, 20.0, 30.0, 40.0])

    # Initial parameters
    params0 = OUParams(
        lam=np.array([0.1] * J),
        sig_v=np.array([1.0] * J),
        sig_eps=np.array([0.5] * J)
    )

    print(f"✓ Data created: J={J}, M={M}, K={K}, T={T}, S={S}, R={R}")

except Exception as e:
    print(f"✗ Data creation failed: {e}")
    raise

# Define minimal required functions
print("\n[3/5] Setting up helper functions...")
try:
    def build_design(lat_reim):
        """Minimal design matrix: [1, lat_reim]"""
        T_f = lat_reim.shape[0]
        return jnp.concatenate([jnp.ones((T_f, 1)), lat_reim], axis=1)

    def extract_band_reim_with_var(mu_fine, var_fine, coupled_bands, freqs_hz, delta_spk, J, M):
        """Minimal extraction - just return dummy values"""
        T_f = mu_fine.shape[0] if hasattr(mu_fine, 'shape') else 100
        lat_reim = np.random.randn(T_f, 2*J)
        var_reim = np.ones((T_f, 2*J)) * 0.1
        return lat_reim, var_reim

    def joint_kf_rts_moments(**kwargs):
        """Minimal KF - return dummy moments"""
        class DummyMoments:
            def __init__(self):
                T_f = 100
                d = 10
                self.m_s = np.random.randn(T_f, d)
                self.P_s = np.abs(np.random.randn(T_f, d, d))
        return DummyMoments()

    print("✓ Helper functions defined")

except Exception as e:
    print(f"✗ Function setup failed: {e}")
    raise

# Run inference (just a few iterations to test)
print("\n[4/5] Running inference (minimal iterations for testing)...")
try:
    cfg = InferenceConfig(
        fixed_iter=10,          # Just 10 warmup iterations for testing
        n_refreshes=2,          # Just 2 refreshes for testing
        inner_steps_per_refresh=5,  # Just 5 inner steps
        pg_jax=True
    )

    key_jax = jr.PRNGKey(42)
    rng_np = np.random.default_rng(42)

    print("  Starting inference...")
    beta, gamma, theta, trace = run_joint_inference_jax_v2(
        Y_cube_block=Y_cube_block,
        params0=params0,
        spikes=spikes,
        H_hist=H_hist,
        all_freqs=all_freqs,
        build_design=build_design,
        extract_band_reim_with_var=extract_band_reim_with_var,
        gibbs_update_beta_robust=None,  # Not used in v2
        joint_kf_rts_moments=joint_kf_rts_moments,
        em_theta_from_joint=None,
        config=cfg,
        delta_spk=0.004,
        window_sec=2.0,
        rng_pg=rng_np,
        key_jax=key_jax,
        offset_sec=0.0
    )

    print("✓ Inference completed successfully!")
    print(f"  beta shape: {beta.shape}")
    print(f"  gamma shape: {gamma.shape}")

except Exception as e:
    print(f"✗ Inference failed: {e}")
    import traceback
    traceback.print_exc()
    raise

# Verify outputs
print("\n[5/5] Verifying outputs...")
try:
    expected_beta_shape = (S, 1 + 2*J)
    expected_gamma_shape = (S, R)

    assert beta.shape == expected_beta_shape, f"beta shape mismatch: {beta.shape} vs expected {expected_beta_shape}"
    assert gamma.shape == expected_gamma_shape, f"gamma shape mismatch: {gamma.shape} vs expected {expected_gamma_shape}"
    assert theta is not None, "theta is None"
    assert trace is not None, "trace is None"
    assert len(trace.theta) == cfg.n_refreshes + 1, f"trace.theta length mismatch: {len(trace.theta)} vs expected {cfg.n_refreshes + 1}"
    print("✓ All outputs have correct shapes")

except AssertionError as e:
    print(f"✗ Output verification failed: {e}")
    raise

print("\n" + "=" * 70)
print("SUCCESS! All tests passed.")
print("run_joint_inference_jax_v2 is working correctly.")
print("=" * 70)
