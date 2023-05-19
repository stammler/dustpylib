from dustpylib.substructures.gaps import duffell2020
from dustpylib.substructures.gaps import kanagawa2017
import numpy as np
import pytest

def test_duffell2020():
    assert np.allclose(duffell2020(1., 1., 1., 1., 1.), 1./(1.+(0.45/(3.*np.pi))))
    assert np.allclose(duffell2020(1., 1., 0., 1., 1.), 1.)


def test_kanagawa2017():
    assert np.allclose(kanagawa2017(1., 1., 1., 1., 1.), 1./(1.+0.04))
    assert np.allclose(kanagawa2017(1., 1., 0., 1., 1.), 1.)