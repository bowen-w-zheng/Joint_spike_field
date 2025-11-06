# Comparison: Original vs JAX V2 Implementation

## Critical Differences Found

### 1. **Design Matrix Construction**

**Original (`run_joint_inference.py:144`)**:
```python
X_slice = np.ascontiguousarray(design_np[:T_design], dtype=np.float64)  # (T, 1+2J)
```
This is the FULL design matrix [1, latent_reim] - intercept already included from `build_design()`

**JAX V2 (`run_joint_inference_jax_v2.py:372`)**:
```python
X_jax = jnp.array(design_np[:T_design], dtype=jnp.float64)
```
Same - should be correct.

---

### 2. **Gibbs Update Logic**

**Original (`beta_sampler.py:80-90`)**: Concatenates design matrices
```python
F = np.column_stack([F_beta, H])  # Combined [X, H] matrix
κ = y - 0.5
p = F.shape[1]  # Total parameters: (1+2B) + R
```

**Original (`beta_sampler.py:132-133`)**: Single precision matrix
```python
Prec = F.T @ (ω[:, None] * F) + Prec_prior  # (p, p) where p = 1+2B+R
RHS  = FT_kappa + Prec_prior @ mu_prior
```

**JAX V2 (`run_joint_inference_jax_v2.py:82-118`)**: Block structure
```python
# Separate blocks
Prec_beta_block = Xw.T @ Xw + jnp.diag(Prec_beta_diag)     # (P, P)
Prec_gamma_block = Hw.T @ Hw + Prec_gamma                  # (R, R)
Prec_cross = Xw.T @ Hw                                     # (P, R)

# Assemble
Prec[:P, :P] = Prec_beta_block
Prec[:P, P:] = Prec_cross
Prec[P:, :P] = Prec_cross.T
Prec[P:, P:] = Prec_gamma_block
```

**Analysis**: Mathematically equivalent - block form is just reorganized.

---

### 3. **ARD Update - POTENTIAL ISSUE**

**Original (`beta_sampler.py:184`)**:
```python
tau2_lat_new = 1.0 / np.random.default_rng().gamma(shape=alpha_post, scale=1.0 / beta_post)
```
- Uses numpy's parameterization: `Gamma(shape, scale)` with mean = `shape * scale`
- Samples from `Gamma(alpha_post, 1/beta_post)`
- Takes reciprocal → InverseGamma(alpha_post, beta_post)
- **IMPORTANT**: Creates a NEW RNG each time! This means it uses system entropy/time-based seed

**JAX V2 (`run_joint_inference_jax_v2.py:137`)**:
```python
tau2_lat_new = 1.0 / jr.gamma(key2, alpha_post, shape=(twoB,)) * beta_post
```
- JAX parameterization: `jr.gamma(key, a)` samples from `Gamma(a, 1)` with rate=1
- This computes: `beta_post / jr.gamma(key, alpha_post)`
- **Mathematical equivalence**: ✓ CORRECT

**Issue**: The numpy version creates a new RNG each call, so it gets different random state. This might cause drift over iterations.

---

### 4. **Warmup Loop - ORDER OF OPERATIONS**

**Original (`run_joint_inference.py:176-200`)**:
```python
for _ in range(fixed_iter):
    for s in range(S):
        psi = X_slice @ beta[s] + H_slice[s] @ gamma[s]
        omega = np.maximum(sample_pg_wrapper(psi), omega_floor)
        key_jax, b_new, g_new, t2_new = gibbs_update_beta_robust(...)
        beta[s] = np.asarray(b_new)
        gamma[s] = np.asarray(g_new)
        tau2_lat[s] = t2_new

        beta0_history[s].append(float(beta[s, 0]))
        gamma_hist[s].append(gamma[s].copy())
```

**JAX V2 (`run_joint_inference_jax_v2.py:179-198`)**:
```python
def scan_fn(carry, key_iter):
    beta, gamma, tau2 = carry

    # Compute psi for ALL trains at once (vectorized)
    psi_all = vmap(lambda b, g, h: X @ b + h @ g)(beta, gamma, H_all)

    # Sample omega for ALL trains at once
    keys_omega = jr.split(key_iter, S)
    omega_all = vmap(lambda k, psi: _sample_omega_pg_batch(k, psi, omega_floor))(
        keys_omega, psi_all
    )

    # Gibbs update for ALL trains at once (vectorized)
    keys_gibbs = jr.split(jr.fold_in(key_iter, 1), S)
    beta_new, gamma_new, tau2_new = _gibbs_update_vectorized(...)

    return (beta_new, gamma_new, tau2_new), None
```

**CRITICAL DIFFERENCE**:
- **Original**: Updates trains SEQUENTIALLY within each iteration (train 0, then train 1, then train 2...)
- **JAX V2**: Updates ALL trains IN PARALLEL within each iteration

**Impact**: This changes the random number stream! In the original, train 1's update might depend on the RNG state after train 0's update.

---

### 5. **Random Number Generation**

**Original**:
- Uses `np.random.default_rng(0)` for PG sampling
- Uses `key_jax` (JAX) for Gibbs update
- **ARD uses a FRESH RNG each time**: `np.random.default_rng().gamma(...)` - NO SEED!

**JAX V2**:
- Uses single `key_jax` for everything
- Splits keys deterministically

**Impact**: Random number streams are completely different.

---

### 6. **Trace Bookkeeping**

**Original (`run_joint_inference.py:201-202`)**:
```python
trace.beta.append(beta.copy())
trace.gamma.append(gamma.copy())
```
Saves beta/gamma at EVERY warmup iteration.

**JAX V2 (`run_joint_inference_jax_v2.py:452-456`)**:
```python
trace = Trace()
trace.theta.append(theta)
trace.latent.append(lat_reim_jax)
trace.fine_latent.append(np.asarray(fine0.mu))
```
Does NOT save beta/gamma during warmup - only at refresh passes.

**Impact**: Different trace contents, but shouldn't affect final results.

---

## KEY ISSUES IDENTIFIED

### Issue #1: ARD Update Uses Unseeded RNG (Original)
Line 184 in `beta_sampler.py`:
```python
tau2_lat_new = 1.0 / np.random.default_rng().gamma(...)
```
This creates a NEW unseeded RNG each call! This makes the original implementation non-reproducible and potentially unstable.

### Issue #2: Sequential vs Parallel Updates
Original processes trains sequentially; JAX processes them in parallel. This affects the random number stream and could lead to slight numerical differences.

### Issue #3: Missing Trace Bookkeeping
JAX v2 doesn't save beta/gamma during warmup - only during refresh passes.

---

---

### Issue #4: **CRITICAL BUG - Beta Median Not Computed** ⚠️

**Original (`run_joint_inference.py:257-258`)**:
```python
recent = np.stack(trace.beta[-config.inner_steps_per_refresh:], axis=0)  # (inner, S, 1+2J)
beta_median = np.median(recent, axis=0)                                   # (S, 1+2J)
```
Takes the MEDIAN across the last `config.inner_steps_per_refresh` iterations (typically 100).

**JAX V2 (`run_joint_inference_jax_v2.py:491`)**:
```python
beta_median = beta_np.copy()  # Use final values as "median"
```
Just uses the FINAL value from inner loop - does NOT compute median!

**Impact**: ⚠️ **THIS IS THE BUG!** The latent refresh uses incorrect beta values, causing the entire algorithm to diverge from the original.

---

### Issue #5: Missing Trace During Inner Loops

**Original (`run_joint_inference.py:253-254`)**:
```python
trace.beta.append(beta.copy())
trace.gamma.append(gamma.copy())
```
Saves beta/gamma at EVERY inner iteration.

**JAX V2**:
Does NOT save intermediate values during inner loop.

**Impact**: Can't compute median without intermediate values. Need to modify `_inner_loop_scan` to return all intermediate beta values.

---

## ROOT CAUSE

The JAX v2 implementation uses `jax.lax.scan` which by default only returns the final state. The original implementation:
1. Runs 100 inner iterations
2. Saves beta after each iteration to `trace.beta`
3. Computes `beta_median = median(last 100 iterations)`
4. Uses this median for latent refresh

The JAX v2 implementation:
1. Runs 100 inner iterations with `lax.scan` (fast!)
2. Only gets the FINAL beta value
3. Uses this final value (incorrectly labeled as "median")
4. Uses wrong beta for latent refresh → algorithm diverges

---

## FIX REQUIRED

1. **Modify `_inner_loop_scan`** to return intermediate beta values:
   ```python
   # Change from:
   return (beta_new, gamma_samp, tau2_new), None

   # To:
   return (beta_new, gamma_samp, tau2_new), beta_new  # Save beta at each iteration
   ```

2. **Compute median properly**:
   ```python
   # After _inner_loop_scan returns:
   (beta_jax, gamma_jax, tau2_lat_jax), beta_history = _inner_loop_scan(...)
   beta_median = jnp.median(beta_history, axis=0)  # (S, 1+2J)
   ```

3. **Add trace bookkeeping** (optional, for compatibility):
   ```python
   for beta_iter in beta_history:
       trace.beta.append(np.array(beta_iter))
   ```

---

## RECOMMENDATIONS

### Priority 1 (Critical):
1. ✅ **Fix beta_median computation** - return intermediate values from scan and compute median

### Priority 2 (Important):
2. **Add trace bookkeeping** - save beta/gamma during inner loops for full compatibility

### Priority 3 (Nice to have):
3. **Consider RNG differences** - document that results won't be bitwise identical due to parallel updates vs sequential
