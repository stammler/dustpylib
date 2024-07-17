"""
``DustPyLib`` is a package with auxiliary tools and extensions for the
dust evolution software ``DustPy``.
"""

from dustpylib import dynamics
from dustpylib import grid
from dustpylib import planetesimals
from dustpylib import radtrans
from dustpylib import substructures

from importlib import metadata as _md

__name__ = 'simframe'
__version__ = _md.version('simframe')

__all__ = [
    "dynamics",
    "grid",
    "planetesimals",
    "radtrans",
    "substructures",
]
