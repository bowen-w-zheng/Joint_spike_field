"""
warmstart.py  – deterministic initialisation for the Gibbs sampler
------------------------------------------------------------------
• Runs EM (ct_ssmt.em_ct) → θ̂
• Immediately runs kalman_filter_ou to obtain xs, Ps
• Packages everything in CTParams

This version works whether `em_ct` returns 3 + ll_hist or 3 + xs + Ps.
"""

from __future__ import annotations
from typing import Any, Tuple
import inspect
import numpy as np
from numpy.typing import NDArray
from src.params import CTParams
from src.em_ct import em_ct
from src.ou import kalman_filter_ou


def em_warmstart(
    Y: NDArray[np.complex128],      # (Jf, M, K)
    db: float,
    *,
    use_jax: bool = False,
    em_kwargs: dict[str, Any] | None = None,
) -> Tuple[CTParams,
           NDArray[np.complex128],  # xs
           NDArray[np.float64]]:    # Ps
    """
    Deterministic warm‑start for Gibbs.

    Returns
    -------
    params : CTParams
    xs, Ps : RTS‑smoothed means / variances (Jf, M, K)
    """
    if em_kwargs is None:
        em_kwargs = {}

    # ---- Run EM (handle presence/absence of use_jax) -----------------
    sig = inspect.signature(em_ct).parameters
    if "use_jax" in sig:
        em_out = em_ct(Y, db=db, use_jax=use_jax, **em_kwargs)
    else:
        em_out = em_ct(Y, db=db, **em_kwargs)

    # em_ct may give 3 or 4 outputs.  We only need the first 3
    lam, sig_v, sig_eps, *rest = em_out

    # ---- Compute xs, Ps via Kalman smoother --------------------------
    _, _, xs, Ps = kalman_filter_ou(
        Y, lam, sig_v, sig_eps, db, use_jax=False
    )

    # ---- Package parameters -----------------------------------------
    params = CTParams(lam=lam, sig_v=sig_v, sig_eps=sig_eps, db=db)
    return params, xs, Ps
