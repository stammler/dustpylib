from dustpy import Simulation
from dustpylib.radtrans import radmc3d
import numpy as np
import os
import pytest
from simframe.io import writers
import dsharp_opac as do


def test_import_from_simulation():

    sim = Simulation()
    sim.initialize()

    rt = radmc3d.Model(sim)

    assert np.allclose(rt.ri_grid_, sim.grid.ri[:-1])
    assert np.allclose(rt.rc_grid_, sim.grid.r[:-1])

    assert np.allclose(rt.a_dust_, sim.dust.a[:-1, :])
    assert np.allclose(rt.H_dust_, sim.dust.H[:-1, :])
    assert np.allclose(rt.Sigma_dust_, sim.dust.Sigma[:-1, :])
    assert np.allclose(rt.T_gas_, sim.gas.T[:-1])

    assert np.allclose(rt.M_star_, sim.star.M)
    assert np.allclose(rt.R_star_, sim.star.R)
    assert np.allclose(rt.T_star_, sim.star.T)

    rt = radmc3d.Model(sim, ignore_last=False)

    assert np.allclose(rt.ri_grid_, sim.grid.ri)
    assert np.allclose(rt.rc_grid_, sim.grid.r)

    assert np.allclose(rt.a_dust_, sim.dust.a)
    assert np.allclose(rt.H_dust_, sim.dust.H)
    assert np.allclose(rt.Sigma_dust_, sim.dust.Sigma)
    assert np.allclose(rt.T_gas_, sim.gas.T)

    assert np.allclose(rt.M_star_, sim.star.M)
    assert np.allclose(rt.R_star_, sim.star.R)
    assert np.allclose(rt.T_star_, sim.star.T)


def test_import_from_file():

    sim = Simulation()
    sim.initialize()
    sim.writer = writers.namespacewriter()
    sim.writeoutput(0)

    data = sim.writer.read.output(0)

    rt = radmc3d.Model(data)

    assert np.allclose(rt.ri_grid_, sim.grid.ri[:-1])
    assert np.allclose(rt.rc_grid_, sim.grid.r[:-1])

    assert np.allclose(rt.a_dust_, sim.dust.a[:-1, :])
    assert np.allclose(rt.H_dust_, sim.dust.H[:-1, :])
    assert np.allclose(rt.Sigma_dust_, sim.dust.Sigma[:-1, :])
    assert np.allclose(rt.T_gas_, sim.gas.T[:-1])

    assert np.allclose(rt.M_star_, sim.star.M)
    assert np.allclose(rt.R_star_, sim.star.R)
    assert np.allclose(rt.T_star_, sim.star.T)

    rt = radmc3d.Model(data, ignore_last=False)

    assert np.allclose(rt.ri_grid_, sim.grid.ri)
    assert np.allclose(rt.rc_grid_, sim.grid.r)

    assert np.allclose(rt.a_dust_, sim.dust.a)
    assert np.allclose(rt.H_dust_, sim.dust.H)
    assert np.allclose(rt.Sigma_dust_, sim.dust.Sigma)
    assert np.allclose(rt.T_gas_, sim.gas.T)

    assert np.allclose(rt.M_star_, sim.star.M)
    assert np.allclose(rt.R_star_, sim.star.R)
    assert np.allclose(rt.T_star_, sim.star.T)


def test_import_from_unknown():
    with pytest.raises(RuntimeError):
        radmc3d.Model("")


def test_derived_data():

    sim = Simulation()
    sim.initialize()

    rt = radmc3d.Model(sim, ignore_last=False)

    assert rt.ai_grid[0] == sim.dust.a.min()
    assert rt.ai_grid[-1] == sim.dust.a.max()
    assert np.allclose(rt.ac_grid, 0.5*(rt.ai_grid[1:] + rt.ai_grid[:-1]))

    assert rt.ri_grid[0] == sim.grid.ri.min()
    assert rt.ri_grid[-1] == sim.grid.ri.max()
    assert np.allclose(rt.rc_grid, 0.5*(rt.ri_grid[1:] + rt.ri_grid[:-1]))

    assert rt.thetai_grid[0] == 0.
    assert rt.thetai_grid[-1] == np.pi
    assert np.allclose(rt.thetac_grid, 0.5 *
                       (rt.thetai_grid[1:] + rt.thetai_grid[:-1]))

    assert rt.phii_grid[0] == 0.
    assert rt.phii_grid[-1] == 2.*np.pi
    assert np.allclose(rt.phic_grid, 0.5 *
                       (rt.phii_grid[1:] + rt.phii_grid[:-1]))

    with pytest.raises(RuntimeError):
        rt.ac_grid = 1.
    with pytest.raises(RuntimeError):
        rt.rc_grid = 1.
    with pytest.raises(RuntimeError):
        rt.thetac_grid = 1.
    with pytest.raises(RuntimeError):
        rt.phic_grid = 1.


def test_write_read_files():

    sim = Simulation()
    sim.initialize()

    rt = radmc3d.Model(sim, ignore_last=False)
    rt.datadir = "temp"

    rt.ai_grid = np.array([rt.ai_grid[0], rt.ai_grid[-1]])

    rt.write_files()

    model = radmc3d.read_model(rt.datadir)
    assert np.allclose(model.grid.r, rt.rc_grid)
    assert np.allclose(model.grid.theta, rt.thetac_grid)
    assert np.allclose(model.grid.phi, rt.phic_grid)

    Nr, Nt, Np, Ns = rt.rc_grid.shape[0], rt.thetac_grid.shape[0], rt.phic_grid.shape[0], rt.ac_grid.shape[0]
    assert model.grid.r.shape[0] == Nr
    assert model.grid.theta.shape[0] == Nt
    assert model.grid.phi.shape[0] == Np
    assert model.rho.shape == (Nr, Nt, Np, Ns)
    assert model.T.shape == (Nr, Nt, Np, Ns)


def test_write_opacities():
    # There is no good way to test for the validity of the files.
    # So this is only testing if it runs without error.

    sim = Simulation()
    sim.initialize()

    rt = radmc3d.Model(sim, ignore_last=False)
    rt.datadir = "temp"

    rt.ai_grid = np.array([rt.ai_grid[0], rt.ai_grid[-1]])

    rt.write_opacity_files(opacity="ricci2010")

    mix = do.diel_henning('olivine')
    rt.write_opacity_files(opacity=mix)
    del mix.rho
    with pytest.raises(ValueError):
        rt.write_opacity_files(opacity=mix)

    with pytest.raises(RuntimeError):
        rt.write_opacity_files(opacity="_")


def test_read_image_iformat12():
    Nx, Ny = 10, 10
    pix_x, pix_y = 20., 20.
    lam = np.array([1.e-4, 1.])
    Nlam = lam.shape[0]
    img = np.random.rand(Nx, Ny, Nlam)

    path = os.path.join("temp", "image.out")

    lines = []
    lines.append("{:d}".format(1))
    lines.append("{:d} {:d}".format(Nx, Ny))
    lines.append("{:d}".format(Nlam))
    lines.append("{:e} {:e}".format(pix_x, pix_y))
    s = ""
    for l in lam:
        s += "{:e} ".format(l*1.e4)
    lines.append(s)
    lines.append("")
    for il in range(Nlam):
        for iy in range(Ny):
            for ix in range(Nx):
                lines.append("{:e}".format(img[ix, iy, il]))
        lines.append("")

    # Addinge line seperator
    lines = [l+os.linesep for l in lines]

    with open(path, "w") as f:
        f.writelines(lines)

    Wx = Nx*pix_x
    Wy = Ny*pix_y
    xi = np.linspace(-Wx/2., Wx/2., Nx+1)
    x = 0.5*(xi[1:]+xi[:-1])
    yi = np.linspace(-Wy/2., Wy/2., Ny+1)
    y = 0.5*(yi[1:]+yi[:-1])

    image = radmc3d.read_image(path)
    assert np.allclose(image["I"], img)
    assert np.allclose(image["Q"], np.zeros_like(img))
    assert np.allclose(image["U"], np.zeros_like(img))
    assert np.allclose(image["V"], np.zeros_like(img))
    assert np.allclose(image["lambda"], lam)
    assert np.allclose(image["x"], x)
    assert np.allclose(image["y"], y)


def test_read_image_iformat3():
    Nx, Ny = 10, 10
    pix_x, pix_y = 20., 20.
    lam = np.array([1.e-4, 1.])
    Nlam = lam.shape[0]
    img = np.random.rand(Nx, Ny, 4, Nlam)

    path = os.path.join("temp", "image.out")

    lines = []
    lines.append("{:d}".format(3))
    lines.append("{:d} {:d}".format(Nx, Ny))
    lines.append("{:d}".format(Nlam))
    lines.append("{:e} {:e}".format(pix_x, pix_y))
    s = ""
    for l in lam:
        s += "{:e} ".format(l*1.e4)
    lines.append(s)
    lines.append("")
    for il in range(Nlam):
        for iy in range(Ny):
            for ix in range(Nx):
                lines.append("{:e} {:e} {:e} {:e}".format(
                    img[ix, iy, 0, il], img[ix, iy, 1, il], img[ix, iy, 2, il], img[ix, iy, 3, il]))
        lines.append("")

    # Addinge line seperator
    lines = [l+os.linesep for l in lines]

    with open(path, "w") as f:
        f.writelines(lines)

    Wx = Nx*pix_x
    Wy = Ny*pix_y
    xi = np.linspace(-Wx/2., Wx/2., Nx+1)
    x = 0.5*(xi[1:]+xi[:-1])
    yi = np.linspace(-Wy/2., Wy/2., Ny+1)
    y = 0.5*(yi[1:]+yi[:-1])

    image = radmc3d.read_image(path)
    assert np.allclose(image["I"], img[..., 0, :])
    assert np.allclose(image["Q"], img[..., 1, :])
    assert np.allclose(image["U"], img[..., 2, :])
    assert np.allclose(image["V"], img[..., 3, :])
    assert np.allclose(image["lambda"], lam)
    assert np.allclose(image["x"], x)
    assert np.allclose(image["y"], y)


def test_read_image_iformat_unkown():
    Nx, Ny = 10, 10
    pix_x, pix_y = 20., 20.
    lam = np.array([1.e-4, 1.])
    Nlam = lam.shape[0]
    img = np.random.rand(Nx, Ny, Nlam)

    path = os.path.join("temp", "image.out")

    lines = []
    lines.append("{:d}".format(0))
    lines.append("{:d} {:d}".format(Nx, Ny))
    lines.append("{:d}".format(Nlam))
    lines.append("{:e} {:e}".format(pix_x, pix_y))
    s = ""
    for l in lam:
        s += "{:e} ".format(l*1.e4)
    lines.append(s)
    lines.append("")
    for il in range(Nlam):
        for iy in range(Ny):
            for ix in range(Nx):
                lines.append("{:e}".format(img[ix, iy, il]))
        lines.append("")

    # Addinge line seperator
    lines = [l+os.linesep for l in lines]

    with open(path, "w") as f:
        f.writelines(lines)

    with pytest.raises(RuntimeError):
        radmc3d.read_image(path)


def test_read_spectrum():
    Nlam = 100
    lam = np.geomspace(1.e-1, 1.e4, Nlam)
    F = np.random.rand(Nlam)

    path = os.path.join("temp", "spectrum.out")

    lines = []
    lines.append("{:d}".format(1))
    lines.append("{:d}".format(Nlam))
    lines.append("")
    for i in range(Nlam):
        lines.append("{:e} {:e}".format(lam[i]*1.e4, F[i]))

    # Addinge line seperator
    lines = [l+os.linesep for l in lines]

    with open(path, "w") as f:
        f.writelines(lines)

    spectrum = radmc3d.read_spectrum(path)
    assert np.allclose(spectrum["lambda"], lam)
    assert np.allclose(spectrum["flux"], F)


def test_mass_conservation_failure():
    sim = Simulation()
    sim.initialize()
    rt = radmc3d.Model(sim)
    rt.thetai_grid = np.linspace(0., 0.5*np.pi, 10)
    rt.write_files(write_opacities=False)


def test_smooth_opacities():
    sim = Simulation()
    sim.initialize()
    rt = radmc3d.Model(sim)
    rt.ai_grid = np.linspace(1.e4, 1.e5, 3)
    rt.write_opacity_files(smoothing=True)


def test_metadata_mismatch():
    sim = Simulation()
    sim.initialize()
    rt = radmc3d.Model(sim)
    rt.write_files(write_opacities=False)
    rt.ai_grid = rt.ai_grid[:5]
    rt._write_metadata()
    radmc3d.read_model(datadir=rt.datadir)
