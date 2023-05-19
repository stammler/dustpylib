from dustpylib.planetesimals.formation import drazkowska2016
from dustpylib.planetesimals.formation import miller2021
from dustpylib.planetesimals.formation import schoonenberg2018
import numpy as np
import pytest

def test_drazkowska2016():
    Nr = 3
    Nm = 4
    OmegaK = np.ones(Nr)
    rhod = np.ones((Nr, Nm))
    rhog = np.ones(Nr)
    Sigmad = np.ones((Nr, Nm))
    St = np.zeros((Nr, Nm))
    St[:, -2:] = 1.

    p2g_crit = 1.
    St_crit = 0.01
    zeta = 1.e-2  

    dS = drazkowska2016(OmegaK, rhod, rhog, Sigmad, St, p2g_crit=p2g_crit, St_crit=St_crit, zeta=zeta)
    dSexp = np.zeros_like(rhod)
    dSexp[1, -2:] = -zeta

    assert np.allclose(dS, dSexp)


def test_miller2021():
    Nr = 3
    Nm = 4
    OmegaK = np.ones(Nr)
    rhod = np.ones((Nr, Nm))
    rhog = np.ones(Nr)
    Sigmad = np.ones((Nr, Nm))
    St = np.zeros((Nr, Nm))
    St[:, -2:] = 1.

    d2g_crit = 1.
    n = 0.03
    zeta = 1.e-2  

    dS = miller2021(OmegaK, rhod, rhog, Sigmad, St, d2g_crit=d2g_crit, n=n, zeta=zeta)
    dSexp = np.zeros_like(rhod)
    dSexp[1, -2:] = -zeta

    assert np.allclose(dS, dSexp)


def test_schoonenberg2018():
    Nr = 3
    Nm = 4
    OmegaK = np.ones(Nr)
    rhod = np.ones((Nr, Nm))
    rhog = np.ones(Nr)
    Sigmad = np.ones((Nr, Nm))
    St = np.zeros((Nr, Nm))
    St[:, -2:] = 1.

    d2g_crit = 1.
    zeta = 1.e-2  

    dS = schoonenberg2018(OmegaK, rhod, rhog, Sigmad, St, d2g_crit=d2g_crit, zeta=zeta)
    dSexp = np.zeros_like(rhod)
    dSexp[1, -2:] = -zeta

    assert np.allclose(dS, dSexp)