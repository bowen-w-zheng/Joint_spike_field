# Trial-Structured Joint Inference

This document describes the implementation of joint LFP-spike inference for trial-structured data.

## Overview

The trial-structured inference algorithm extends the single-session joint inference to handle multiple trials with:
- **Shared latent spectral trajectory** X(t) across trials (or hierarchical X + δᵣ)
- **Per-trial, per-unit coupling coefficients** β and γ
- **Precision-pooled observations** for efficient Kalman filtering

## Implementation

### Files

1. **`run_joint_inference_jax_v2_trials.py`**
   - Main inference function: `run_joint_inference_trials()`
   - Implements precision-pooling approach (shared trajectory)
   - Supports multi-trial, multi-unit spike trains
   - Uses existing Gibbs samplers with flattened (unit × trial) structure

2. **`src/utils_trial_inference.py`**
   - Helper functions for trial data:
     - `build_predictors_from_upsampled_hier()`: Extract Re/Im predictors from upsampled hierarchical latents
     - `normalize_trial_shapes()`: Normalize data to canonical shapes
     - `pool_observations_across_trials()`: Precision-pool LFP across trials
     - `build_spike_history_matrix()`: Build spike history features
     - `summarize_beta_per_trial()`: Summarize β estimates

3. **`example_trial_inference.py`**
   - Complete workflow example
   - Demonstrates pipeline from raw data to β/γ estimates

### Algorithm: Shared-Trajectory Precision Pooling

The algorithm follows section **A** of the plan:

#### Key Steps

1. **LFP Processing**
   ```
   For each trial r:
       Compute multitaper spectrogram Yᵣ(j,m,k)
   ```

2. **EM Fitting** (hierarchical model X + δᵣ)
   ```
   Fit: Zᵣ(t) = X(t) + δᵣ(t)
   Where:
       X ~ OU(λ_X, σ_X)     # shared across trials
       δᵣ ~ OU(λ_δ, σ_δ)    # per-trial deviation
   ```

3. **Upsample to Spike Time Scale**
   ```
   Upsample X and δᵣ to fine grid (1 ms bins)
   Z_fine = X_fine + δ_fine
   ```

4. **Flatten (Unit × Trial) Structure**
   ```
   S units × R trials → S' = S×R "pseudo-units"
   Each pseudo-unit (s,r) has its own β and γ
   ```

5. **Warm-Up: Per-Pseudo-Unit β/γ Sampling**
   ```
   For each iteration:
       For each pseudo-unit (s,r):
           Sample ω ~ PG(1, ψ) where ψ = X'β + H'γ
           Update (β, γ) ~ N(μ_post, Σ_post)

   Freeze β₀ using robust median
   Build γ posterior per pseudo-unit
   ```

6. **Refresh Passes with Precision Pooling**
   ```
   For each refresh:
       # Inner sampling (fixed latents)
       For inner_steps:
           For each pseudo-unit:
               Sample (β, γ) with current latents

       # Latent refresh
       Compute median β across inner steps

       # Pool LFP observations
       For each (j, k):
           Y_pooled[j,:,k] = Σᵣ wᵣ Yᵣ[j,:,k] / Σᵣ wᵣ
           where wᵣ = 1/σ²ᵣ (precision weights)

       # Pool spike pseudo-observations
       For each unit s, time t:
           For trials r:
               κᵣ = Nₛᵣ(t) - 1/2
               ωᵣ = PG weight
           y_pooled = Σᵣ κᵣ / Σᵣ ωᵣ
           R_pooled = 1 / Σᵣ ωᵣ

       # Run Kalman filter with pooled obs
       X_new ~ Kalman(Y_pooled, y_spike_pooled)

       # Rebuild regressors for next iteration
   ```

7. **Output**
   ```
   β: (S, R, 1+2J)  # coupling per unit and trial
   γ: (S, R, L)     # history per unit and trial
   ```

## Usage

### Basic Workflow

```python
from run_joint_inference_jax_v2_trials import run_joint_inference_trials, InferenceConfigTrials
from src.trial_tfr import compute_trial_tfr_multitaper
from src.em_ct_hier_jax import run_em_ct_hier_jax
from src.upsample_ct_hier_fine import upsample_ct_hier_fine

# 1. Compute trial spectrograms
Y_trials, centres, freqs, M = compute_trial_tfr_multitaper(
    lfp_trials,  # (R, T_lfp)
    fs=1000.0,
    window_sec=0.2,
    NW=3.0
)

# 2. Fit hierarchical model
em_result = run_em_ct_hier_jax(
    Y_trials,  # (R, M, J, K)
    db=window_sec,
    n_iter=50
)

# 3. Upsample to spike time scale
ups = upsample_ct_hier_fine(
    Y_trials=Y_trials,  # (R, J, M, K)
    res=em_result,
    delta_spk=0.001,
    win_sec=window_sec,
    offset_sec=0.0
)

# 4. Run joint inference
config = InferenceConfigTrials(
    fixed_iter=100,
    n_refreshes=3,
    inner_steps_per_refresh=100
)

beta, gamma, theta, trace = run_joint_inference_trials(
    Y_cube_block=Y_trials,    # (R, J, M, K)
    params0=initial_params,
    spikes=spikes,            # (R, S, T)
    H_hist=history,           # (S, R, T, lags)
    all_freqs=freqs,
    build_design=build_design_fn,
    extract_band_reim_with_var=extract_fn,
    gibbs_update_beta_robust=gibbs_fn,
    joint_kf_rts_moments=kf_fn,
    config=config,
    delta_spk=0.001,
    window_sec=0.2
)

# β: (S, R, 1+2J), γ: (S, R, lags)
```

### Data Format Requirements

- **LFP trials**: `(R, T_lfp)` where R = number of trials
- **Spikes**: `(R, S, T)` where S = number of units, T = spike time bins
- **History**: `(S, R, T, L)` where L = number of history lags
- **TFR**: `(R, J, M, K)` where J = bands, M = tapers, K = time blocks

## Mathematical Details

### Precision Pooling (eq. 11)

For LFP observations at block k:
```
Y_pooled[j,m,k] = (Σᵣ Yᵣ[j,m,k]/σ²ᵣ) / (Σᵣ 1/σ²ᵣ)
σ²_pooled[j,m] = 1 / (Σᵣ 1/σ²ᵣ)
```

### Spike Pseudo-Observation Pooling (eq. 12)

For spike train at time t:
```
y_pooled = (Σᵣ κᵣ) / (Σᵣ ωᵣ)
R_pooled = 1 / (Σᵣ ωᵣ)
```
where:
- κᵣ = Nᵣ(t) - 1/2 (spike count minus 1/2)
- ωᵣ ~ PG(1, ψᵣ) (Pólya-Gamma weight)

## Future Extensions

### Hierarchical Per-Trial Deviations (Section B)

The current implementation uses shared-trajectory pooling. Future work will add:

1. **Augmented State Space**
   ```
   x = [X^(1:M), δ₁^(1:M), ..., δᵣ^(1:M)]
   dim = 2M × (1 + R)
   ```

2. **Per-Trial Observation Rows**
   - Each trial r observes X + δᵣ
   - Separate δᵣ processes per trial

3. **Errors-in-Variables β Update**
   ```
   Include predictor uncertainty in coupling estimation:
   E[x x'] = [1, μ'; μ, μμ' + Σ]
   ```

4. **Mode Flag**
   ```python
   config = InferenceConfigTrials(mode="hier")  # vs "shared"
   ```

## Key Differences from Single-Session Code

| Aspect | Single Session | Trial-Structured |
|--------|---------------|------------------|
| Latent | One trajectory X(t) | Shared X(t) across trials |
| β/γ | Per unit: (S, P) | Per (unit, trial): (S, R, P) |
| Observations | One Y(j,m,k) | Pool Yᵣ(j,m,k) across r |
| Pseudo-obs | Per unit ω(t) | Pool across trials |
| Output | β: (S, P) | β: (S, R, P) |

## Notes

- The current implementation assumes **fixed trial structure** (same T across trials)
- For varying trial lengths, pad to max length and use masking
- The `beta_gamma_pg_fixed_latents_joint_jax.py` file should be **ignored** (has known issues)
- Use the existing `gibbs_update_beta_robust` from the working single-session code

## References

- See the detailed plan in the initial task description (sections A and B)
- Related files:
  - `src/run_joint_inference.py` - single-session implementation
  - `src/joint_inference_core.py` - Kalman filter/RTS core
  - `src/em_ct_hier_jax.py` - EM for hierarchical models
  - `src/upsample_ct_hier_fine.py` - upsampling to spike time scale
