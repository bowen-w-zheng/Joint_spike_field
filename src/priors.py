from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


# ─────────────────────────────────────────────────────────────
#  Hyper‑parameter container
# ─────────────────────────────────────────────────────────────
@dataclass
class Hyper:
    # Inverse‑Gamma( a, b )  ⇒  p(x) ∝ x^{−a−1} exp(−b/x)
    a_v:   float = 1e-3
    b_v:   float = 1e-3
    a_eps: float = 1e-3
    b_eps: float = 1e-3

    # Normal( μ, τ² ) prior for Φ before truncation to (0,1)
    mu_phi:   float = 0.0
    tau2_phi: float = 1.0


# ─────────────────────────────────────────────────────────────
#  Sampling utilities
# ─────────────────────────────────────────────────────────────
def invgamma_sample(shape: NDArray[np.float64] | float,
                    scale: NDArray[np.float64] | float,
                    rng: np.random.Generator) -> NDArray[np.float64]:
    """
    Draw from InvGamma(shape, scale)  by sampling Gamma and inverting.
    """
    gamma = rng.gamma(shape, 1.0 / scale)
    return 1.0 / gamma


def trunc_norm_01(mu: NDArray[np.float64],
                  sigma: NDArray[np.float64],
                  rng: np.random.Generator) -> NDArray[np.float64]:
    """
    Draw from N(mu, sigma²) **truncated to (0, 1)**, element‑wise.

    Works for scalar or array `mu`, `sigma` broadcasting together.
    Uses simple rejection (efficient because most mass lies inside).
    """
    out = rng.normal(mu, sigma)
    bad = (out <= 0.0) | (out >= 1.0)
    while np.any(bad):
        redraw = rng.normal(mu[bad], sigma[bad])
        out[bad] = redraw
        bad = (out <= 0.0) | (out >= 1.0)
    return out


def gamma_prior_simple(n_lags: int = 25,
                       strong_neg: float = -10,
                       mild_neg: float = -0.5,
                       k_mild: int = 4,
                       tau_gamma: float = 1.5):
    """
    Prior mean: γ_1 very negative (refractory), next k_mild-1 mildly negative,
                rest neutral (0).
    Prior cov : diagonal (tau_gamma^2) I  (simple ridge)
    """
    mu = np.zeros(n_lags)
    mu[0] = strong_neg
    if k_mild > 1:
        mu[1:k_mild] = mild_neg
    Sigma = (tau_gamma ** 2) * np.eye(n_lags)
    return mu, Sigma
