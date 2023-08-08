"""
This package contains methods to implement the dust backreaction coefficients.
The setup_backreaction(sim) function automatically implements all the required modifications to the Simulation object.
"""

from dustpylib.dynamics.backreaction.functions_backreaction import BackreactionCoefficients
from dustpylib.dynamics.backreaction.functions_backreaction import BackreactionCoefficients_VerticalStructure
from dustpylib.dynamics.backreaction.functions_backreaction import vrad_dust_BackreactionVerticalStructure
from dustpylib.dynamics.backreaction.functions_backreaction import dustDiffusivity_Backreaction

from dustpylib.dynamics.backreaction.setup_backreaction import setup_backreaction


__all__ = [
    "BackreactionCoefficients",
    "BackreactionCoefficients_VerticalStructure",
    "dustDiffusivity_Backreaction",
    "setup_backreaction",
    "vrad_dust_BackreactionVerticalStructure",
]
