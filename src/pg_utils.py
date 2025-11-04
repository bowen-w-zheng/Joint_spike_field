# ct_gibbs/pg_utils.py
"""
Pólya–Gamma ω-sampler wrapper.

We call the fast C routine from `polyagamma` on the **host**; the result is
then viewed as a JAX array without an extra device transfer.  Works for any
shape psi.

Usage
-----
>>> rng_np = np.random.default_rng(123)
>>> psi    = np.linspace(-10, 10, 1000)
>>> omega  = sample_polya_gamma(psi, rng_np)     # JAX DeviceArray
"""
from __future__ import annotations
import numpy as np
import jax.numpy as jnp
from numpy.random import Generator
from polyagamma import random_polyagamma

# ────────────────────────────────────────────────────────────────────────
_CLIP = 20.0            # numerical safety for large |ψ|


def sample_polya_gamma(psi: np.ndarray | jnp.ndarray,
                       rng: Generator) -> jnp.ndarray:
    """
    Draw ω ∼ PG(1, ψ) element-wise for arbitrary-shaped ψ.

    Parameters
    ----------
    psi : array_like
        Real-valued log-odds.
    rng : numpy.random.Generator
        Host RNG used by `polyagamma`.

    Returns
    -------
    ω : jax.numpy.DeviceArray
        Same shape as `psi`.
    """
    psi_host = np.asarray(psi, dtype=np.float64)
    psi_host = np.clip(psi_host, -_CLIP, _CLIP)

    out = np.empty_like(psi_host)
    random_polyagamma(1, psi_host, out=out, random_state=rng)
    out = np.clip(out, 1.0e-8, None, out=out)
    # zero-copy view on the default JAX device
    return jnp.asarray(out)
