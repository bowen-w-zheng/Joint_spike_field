# ct_gibbs/state_index.py
"""
Utility for laying out the big latent state vector Z_t that contains

    • J  frequency bands
    • M  DPSS tapers   (per band)
    • 2  components    (real & imag)

The convention below is the same one already used in `ou_fine.py`
and `pg_multi_joint_sampler.py`, so existing Kalman code works
unchanged:

    flat_index =                       (component)
                  2 * (taper + m * J)  +  comp

where
    comp = 0  →  real part
           1  →  imag part

Hence the overall state dimension is `2 * J * M`.

Example
-------
>>> idx = StateIndex(n_bands=3, n_tapers=5)
>>> idx.slice(band=1, taper=2, comp='imag')   # band-1, taper-2, imag
slice(16, 17, None)                           # (zero-based)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal


@dataclass(slots=True, frozen=True)
class StateIndex:
    """
    Parameters
    ----------
    n_bands  : int
        Number of frequency bands J.
    n_tapers : int
        Number of tapers M (same for every band).
    """
    n_bands: int
    n_tapers: int

    # ─────────────────── public helpers ────────────────────────────
    @property
    def dim(self) -> int:
        """Total state dimension = 2 * J * M."""
        return 2 * self.n_bands * self.n_tapers

    def offset(self, *, band: int, taper: int, comp: Literal["real", "imag"]):
        """Return scalar offset (0-based) for the requested component."""
        if not (0 <= band < self.n_bands):
            raise IndexError("band out of range")
        if not (0 <= taper < self.n_tapers):
            raise IndexError("taper out of range")
        if comp not in ("real", "imag"):
            raise ValueError("comp must be 'real' or 'imag'")
        comp_int = 0 if comp == "real" else 1
        return 2 * (taper + band * self.n_tapers) + comp_int

    def slice(
        self,
        *,
        band: int,
        taper: int,
        comp: Literal["real", "imag"],
    ):
        """
        Return a Python slice addressing that single scalar within
        a flattened (dim,) state vector.
        """
        k = self.offset(band=band, taper=taper, comp=comp)
        return slice(k, k + 1)

    # ───── convenience: slices for an entire band or taper ─────────
    def band_slice(self, band: int):
        """
        Slice covering *both* components of *all* tapers in one band.
        Length = 2 * M.
        """
        if not (0 <= band < self.n_bands):
            raise IndexError("band out of range")
        start = 2 * band * self.n_tapers
        end   = start + 2 * self.n_tapers
        return slice(start, end)

    def taper_slice(self, taper: int):
        """
        Slice covering *both* components of one taper across *all* bands.
        Length = 2 * J.
        """
        if not (0 <= taper < self.n_tapers):
            raise IndexError("taper out of range")
        starts = [
            2 * (taper + b * self.n_tapers) for b in range(self.n_bands)
        ]
        # Build list of scalar slices because the elements are not contiguous
        return [slice(k, k + 2) for k in starts]
