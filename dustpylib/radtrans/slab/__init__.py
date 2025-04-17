"""
This package contains a simple slab model to estimate the continuum
emission from ``DustPy`` models (see Birnstiel et al. 2018).
"""

from dustpylib.radtrans.slab.slab import Opacity
from dustpylib.radtrans.slab.slab import get_observables
from dustpylib.radtrans.slab.slab import get_all_observables

__all__ = [
    "Opacity",
    "get_observables",
    "get_all_observables",
]
