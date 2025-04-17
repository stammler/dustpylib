from dustpy import Simulation
from dustpylib.radtrans import slab
import numpy as np
import pytest
from types import SimpleNamespace

test_string = \
"""{}
{}
1e4 1.0000000000000000e+00 2.000000000000e+00 1.000000000000e-01
2e4 2.0000000000000000e+00 3.000000000000e+00 2.000000000000e-01
"""


def get_data():
    """
    Creates a duspty output and reads the output data of time step 0
    as well as all available time steps (which has one dimension more).
    Returns:
        tuple: A tuple containing:
            - data_0: The output data at time step 0.
            - data_t: The output data for all time steps.
    """
    sim = Simulation()
    sim.initialize()
    sim.writer.overwrite = True
    sim.writeoutput(0)
    data_0 = sim.writer.read.output(0)
    data_t = sim.writer.read.all()

    opac = slab.Opacity()

    return data_0, data_t, opac

def test_get_observables():

    # get a single name space and one with time dimension
    data_0, _, opac = get_data()

    # set wavelength array and get opacity
    lam = np.geomspace(1e-4, 1e-1, 5)
    
    # call get_observables for a 1D and a 2D size grid
    obs00 = slab.get_observables(data_0.grid.r, data_0.dust.Sigma, data_0.gas.T, data_0.dust.a[0], lam, opac)
    obs01 = slab.get_observables(data_0.grid.r, data_0.dust.Sigma, data_0.gas.T, data_0.dust.a, lam, opac)
    assert np.allclose(obs00.I_nu, obs01.I_nu)


    # call get_observables for a 1D and a 2D size grid
    obs00 = slab.get_observables(data_0.grid.r, data_0.dust.Sigma, data_0.gas.T, data_0.dust.a[0], lam, opac, scattering=False)
    obs01 = slab.get_observables(data_0.grid.r, data_0.dust.Sigma, data_0.gas.T, data_0.dust.a, lam, opac, scattering=False)
    assert np.allclose(obs00.I_nu, obs01.I_nu)


def test_get_observables_all():

    # get a single name space and one with time dimension
    data_0, data_t, opac = get_data()

    lam = np.geomspace(1e-4, 1e-1, 5)

    obs1 = slab.get_all_observables(data_t, opac, lam)
    obs2 = slab.get_all_observables(data_0, opac, lam)
    obs3 = slab.get_all_observables('data', opac, lam)
    assert np.allclose(obs1.I_nu, obs2.I_nu, obs3.I_nu)


def test_get_observables_all_namespace():
    _, data_t, opac = get_data()
    
    data_ns = SimpleNamespace(
        r=data_t.grid.r[0],
        sig_da=data_t.dust.Sigma,
        T=data_t.gas.T,
        a=data_t.dust.a,
        t=data_t.t
        )
    
    lam = np.geomspace(1e-4, 1e-1, 5)

    obs0 = slab.get_all_observables(data_ns, opac, lam)


def test_opacity_load_from_dict():

    na = 5
    nlam = 10

    test_dict = dict(
        a=np.logspace(1e-4, 1e-1, na),
        lam=np.logspace(1e-4, 1e-1, nlam),
        k_abs=np.ones((na, nlam)),
        k_sca=np.zeros((na, nlam)) + 1e-10,
        g=None,
        test=None
    )

    opac = slab.Opacity(test_dict)

    assert np.allclose(opac.get_opacities(test_dict['a'][0], test_dict['lam'][0]), [1, 1e-10])

def test_opacity_error():
    with pytest.raises(ValueError):
        opac = slab.Opacity('abc')

def test_opacities_bounds():
    opac = slab.Opacity()

    with pytest.raises(ValueError):
        opac.get_opacities(np.array([100.0]), np.array([100]))

def test_opacity_load_from_dustkappa():


    # first we use a format string of 3 (k_abs, k_sca, g are present)
    # but the number of wavelength doesn't match. This triggers a warning
    with open('dustkappa.inp', 'w') as f:
        f.write(test_string.format(3, 3))
        
    with pytest.warns(UserWarning):
        opac = slab.Opacity('dustkappa.inp')

    res = opac.get_opacities(np.array([1.0]), np.array([1.5]))

    y0 = 1.5
    y1 = 10.**(np.log10(2.00) + np.log10(1.5) / np.log10(2) * np.log10(3.00/2.00) )

    assert np.allclose([y0, y1], np.squeeze(res))
    assert opac.rho_s is None

    # this should now give g=0.2, so res = 0.8 * y1 + y0
    res = opac.get_k_ext_eff(np.array([1.0]), np.array([1.5]))
    assert np.allclose(res, 0.8 * y1 + y0)

    
    # ----- next, we test the case where we have only k_abs and k_sca -----
    with open('dustkappa.inp', 'w') as f:
        f.write(test_string.format(2, 2))
    opac = slab.Opacity('dustkappa.inp')

    # g should now be zero
    res = opac.get_g(np.array([1.0]), np.array([1.5]))
    assert res == 0.0

    # ----- next, we test the case where we have only k_abs ----- 
    with open('dustkappa.inp', 'w') as f:
        f.write(test_string.format(1, 2))
    opac = slab.Opacity('dustkappa.inp')

    # k_sca should now be zero
    res = opac.get_opacities(np.array([1.0]), np.array([1.5]))
    assert res[1][0] == 0.0

    # ----- next, we test the case where iformat is 0 -----
    with open('dustkappa.inp', 'w') as f:
        f.write(test_string.format(0, 2))

    # this should raise an error
    with pytest.raises(ValueError):
        opac = slab.Opacity('dustkappa.inp')


def test_I_over_B():
    assert np.allclose(slab.slab.I_over_B(1.0, 1.0), 1.0 - np.exp(-1.0))
    assert np.allclose(slab.slab.I_over_B(1.0, 0.0), 0.0)


def test_J_over_B():
    assert np.allclose(
        slab.slab.J_over_B(1, 1, 1),
        1 - 0.5 * (1 + np.exp(-np.sqrt(3)))
        )
    
def test_Bplanck_limits():
    assert slab.slab.bplanck(2e13, 1.0) > 0
    assert slab.slab.bplanck(1e7, 1.0) > 0