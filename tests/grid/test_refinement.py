from dustpylib.grid.refinement import refine_radial_local
import numpy as np
import pytest

def test_refine_radial_local():
    assert refine_radial_local(0., 0., num=0) == 0.
    ri = np.array([1., 4., 8., 16.])
    r0 = 5.
    rexp = np.array([1., 2., 4., np.sqrt(4.*np.sqrt(4.*8.)), np.sqrt(4.*8.), 8., np.sqrt(8.*16.), 16.])
    assert np.allclose(refine_radial_local(ri, r0, num=2), rexp)