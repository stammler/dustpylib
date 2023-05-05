"""
This module can be used to create simple, axisymmetric ``RADMC-3D`` input files
from ``DustPy`` models. It furthermore contains simple methods to read ``RADMC-3D``
images and spectra and methods to inspect the model files. These are only
meant for models created by this module. For more general models, use the
``Radmc3dPy` module within ``RADMC-3D``.
"""

import dsharp_opac as do
from dustpy import Simulation
from dustpylib.grid.refinement import refine_radial_local
import numpy as np
import os
from pathlib import Path
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
from types import SimpleNamespace


class Model():
    """
    Main model class that can read in ``DustPy`` models and can create ``RADMC-3D`` input files.
    Attributes with trailing underscore are imported from ``DustPy``, while the other attributes
    will be used to create ``RADMC-3D`` input files.

    Methods
    -------
    read_image :
        Reads ``RADMC-3D`` image file
    read_spectrum :
        Reads ``RADMC_3d`` spectrum file
    write_files :
        Writes all required ``RADMC-3D`` input files into the specified directory
    write_opacity_files :
        Writes only the required ``RADMC-3D`` opacity into files into the specified directory
    """

    def __init__(self, sim):
        """
        Class to create a simple axisymmetric ``RADMC-3D`` model from ``DustPy`` simulation data.

        Parameters
        ----------
        sim : namespace or DustPy.Simulation
            The ``DustPy`` model data. Can either be a ``Simulation`` object or
            a namespace returned from ``Writer.read.output()``.
        """

        self._ai_grid = None
        self._ac_grid = None

        #: Wavelength grid for ``RADMC-3D`` in cm
        self.lam_grid = None

        self._ri_grid = None
        self._rc_grid = None
        self._thetai_grid = None
        self._thetac_grid = None
        self._phii_grid = None
        self._phic_grid = None

        #: Radial grid cell interfaces from ``DustPy`` model
        self.ri_grid_ = None

        #: Radial grid cell centers from ``DustPy`` model
        self.rc_grid_ = None

        #: Stellar mass in g
        self.M_star_ = None

        #: Stellar radius in cm
        self.R_star_ = None

        #: Stellar effective temperature in K
        self.T_star_ = None

        #: Temperature profile in K from ``DustPy``
        self.T_gas_ = None

        #: Particle size array in cm from ``DustPy``
        self.a_dust_ = None

        #: Dust scale heights array in cm from ``DustPy``
        self.H_dust_ = None

        #: Dust midplane density profile in g/cm³ from ``DustPy``
        self.rho_dust_ = None

        #: Directory to store the ``RADMC-3D input files``
        self.datadir = "radmc3d"

        #: Opacity model. "birnstiel2018" (default) or "ricci2010"
        self.opacity = "birnstiel2018"

        #: ``RADMC-3D`` options for radmc3d.inp file
        self.radmc3d_options = {
            "modified_random_walk": 1,
            "iranfreqmode": 1,
            "istar_sphere": 1,
        }

        if isinstance(sim, Simulation):
            self._init_from_dustpy(sim)
        elif isinstance(sim, SimpleNamespace):
            self._init_from_namespace(sim)
        else:
            raise RuntimeError("Unknown data type of 'sim'.")

        #: Radial grid cell interfaces for ``RADMC-3D`` model
        self.ri_grid = refine_radial_local(self.ri_grid_, 0., num=3)

        lam1 = np.geomspace(0.1e-4, 7.e-4, 20, endpoint=False)
        lam2 = np.geomspace(7.e-4, 25.e-4, 100, endpoint=False)
        lam3 = np.geomspace(25.e-4, 1., 30, endpoint=True)
        self.lam_grid = np.concatenate([lam1, lam2, lam3])

        Ntheta = 64
        theta_threshold = 0.25*np.pi
        theta1 = np.linspace(0., theta_threshold, 5, endpoint=False)
        theta2 = np.linspace(theta_threshold,
                             0.5*np.pi, Ntheta+1-theta1.shape[0])
        theta_up = np.concatenate([theta1, theta2])
        theta_down = (np.pi - theta_up[:-1])[::-1]
        self.thetai_grid = np.concatenate([theta_up, theta_down])

        Nphi = 16
        self.phii_grid = np.linspace(0., 2.*np.pi, Nphi+1)

        Nspec = 16
        self.ai_grid = np.geomspace(
            self.a_dust_.min(),
            self.a_dust_.max(),
            Nspec+1)

    @property
    def ai_grid(self):
        """
        Particle size bin interfaces in cm for ``RADMC-3D`` model
        """
        return self._ai_grid

    @ai_grid.setter
    def ai_grid(self, value):
        self._ai_grid = value
        self._ac_grid = 0.5*(value[1:]+value[:-1])

    @property
    def ac_grid(self):
        """
        Particle size bin centers in cm for ``RADMC-3D`` model.
        Do not set manually. Only use size bin interfaces.
        """
        return self._ac_grid

    @ac_grid.setter
    def ac_grid(self, value):
        raise RuntimeError("Do not set this manually.")

    @property
    def ri_grid(self):
        """
        Radial grid cell interfaces in cm for ``RADMC-3D`` model.
        """
        return self._ri_grid

    @ri_grid.setter
    def ri_grid(self, value):
        self._ri_grid = value
        self._rc_grid = 0.5*(value[1:]+value[:-1])

    @property
    def rc_grid(self):
        """
        Radial grid cell centers in cm for ``RADMC-3D`` model.
        Do not set manually. Only use cell interfaces.
        """
        return self._rc_grid

    @rc_grid.setter
    def rc_grid(self, value):
        raise RuntimeError("Do not set this manually.")

    @property
    def thetai_grid(self):
        """
        Polar grid cell interfaces in rad for ``RADMC-3D`` model.
        """
        return self._thetai_grid

    @thetai_grid.setter
    def thetai_grid(self, value):
        self._thetai_grid = value
        self._thetac_grid = 0.5*(value[1:]+value[:-1])

    @property
    def thetac_grid(self):
        """
        Polar grid cell centers in rad for ``RADMC-3D`` model.
        Do not set manually. Only use cell interfaces.
        """
        return self._thetac_grid

    @thetac_grid.setter
    def thetac_grid(self, value):
        raise RuntimeError("Do not set this manually.")

    @property
    def phii_grid(self):
        """
        Azimuthal grid cell interfaces in rad for ``RADMC-3D`` model.
        """
        return self._phii_grid

    @phii_grid.setter
    def phii_grid(self, value):
        self._phii_grid = value
        self._phic_grid = 0.5*(value[1:]+value[:-1])

    @property
    def phic_grid(self):
        """
        Azimuthal grid cell centers in rad for ``RADMC-3D`` model.
        Do not set manually. Only use cell interfaces.
        """
        return self._phic_grid

    @phic_grid.setter
    def phic_grid(self, value):
        raise RuntimeError("Do not set this manually.")

    def _init_from_dustpy(self, sim):
        """
        This function initializes the model from a ``DustPy`` simulation object.
        """

        self.M_star_ = sim.star.M
        self.R_star_ = sim.star.R
        self.T_star_ = sim.star.T

        self.rc_grid_ = sim.grid.r
        self.ri_grid_ = sim.grid.ri

        self.T_gas_ = sim.gas.T

        self.a_dust_ = sim.dust.a
        self.H_dust_ = sim.dust.H
        self.rho_dust_ = sim.dust.rho

    def _init_from_namespace(self, sim):
        """
        This function inizializes the model from a namespace as returned by
        ``Writer.read.output()`` method.
        """

        self.M_star_ = sim.star.M[0]
        self.R_star_ = sim.star.R[0]
        self.T_star_ = sim.star.T[0]

        self.rc_grid_ = sim.grid.r
        self.ri_grid_ = sim.grid.ri

        self.T_gas_ = sim.gas.T

        self.a_dust_ = sim.dust.a
        self.H_dust_ = sim.dust.H
        self.rho_dust_ = sim.dust.rho

    def write_files(self, datadir=None, opacity=None):
        """
        Function writes all required ``RADMC-3D`` input files.

        Parameters
        ----------
        datadir : str, optional, default: None
            Data directory in which the files are written. None defaults to
            the datadir attribute of the parent class.
        """
        datadir = self.datadir if datadir is None else datadir
        opacity = opacity or self.opacity or 'birnstiel2018'
        self._write_radmc3d_inp(datadir=datadir)
        self._write_stars_inp(datadir=datadir)
        self._write_wavelength_micron_inp(datadir=datadir)
        self._write_amr_grid_inp(datadir=datadir)
        self._write_dust_density_inp(datadir=datadir)
        self._write_dust_temperature_dat(datadir=datadir)
        self.write_opacity_files(datadir=datadir, opacity=opacity)

    def write_opacity_files(self, datadir=None, opacity=None):
        """
        Function writes the required opacity files.

        Parameters
        ----------
        datadir : str, optional, default: None
            Data directory in which the files are written. None defaults to
            the datadir attribute of the parent class.
        """
        datadir = self.datadir if datadir is None else datadir
        opacity = self.opacity if opacity is None else opacity
        self._write_dustopac_inp(datadir=datadir)
        self._write_dustkapscatmat_inp(datadir=datadir, opacity=opacity)

    def _write_radmc3d_inp(self, datadir=None):
        """
        Function writes the 'radmc3d.inp' input file.

        Parameters
        ----------
        datadir : str, optional, default: None
            Data directory in which the files are written. None defaults to
            the datadir attribute of the parent class.
        """

        filename = "radmc3d.inp"
        datadir = self.datadir if datadir is None else datadir
        Path(datadir).mkdir(parents=True, exist_ok=True)
        path = os.path.join(datadir, filename)

        print("Writing {}.....".format(path), end="")
        with open(path, "w") as f:
            for key in self.radmc3d_options:
                f.write("{} = {}\n".format(key, self.radmc3d_options[key]))
        print("done.")

    def _write_stars_inp(self, datadir=None):
        """
        Function writes the 'stars.inp' input file.

        Parameters
        ----------
        datadir : str, optional, default: None
            Data directory in which the files are written. None defaults to
            the datadir attribute of the parent class.
        """

        filename = "stars.inp"
        datadir = self.datadir if datadir is None else datadir
        Path(datadir).mkdir(parents=True, exist_ok=True)
        path = os.path.join(datadir, filename)

        print("Writing {}.....".format(path), end="")
        with open(path, "w") as f:
            f.write("2\n")
            f.write("1 {:d}\n".format(self.lam_grid.shape[0]))
            f.write("{:.6e} {:.6e} {:.6e} {:.6e} {:.6e}\n".format(
                self.R_star_, self.M_star_, 0., 0., 0.))
            for val in self.lam_grid:
                f.write("{:.6e}\n".format(val*1.e4))
            f.write("-{:.6e}\n".format(self.T_star_))
        print("done.")

    def _write_wavelength_micron_inp(self, datadir=None):
        """
        Function writes the 'wavelength_micron.inp' input file.

        Parameters
        ----------
        datadir : str, optional, default: None
            Data directory in which the files are written. None defaults to
            the datadir attribute of the parent class.
        """

        filename = "wavelength_micron.inp"
        datadir = self.datadir if datadir is None else datadir
        Path(datadir).mkdir(parents=True, exist_ok=True)
        path = os.path.join(datadir, filename)

        print("Writing {}.....".format(path), end="")
        with open(path, "w") as f:
            f.write("{:d}\n".format(self.lam_grid.shape[0]))
            for val in self.lam_grid:
                f.write("{:.6e}\n".format(val*1.e4))
        print("done.")

    def _write_amr_grid_inp(self, datadir=None):
        """
        Function writes the 'amr_grid.inp' input file.

        Parameters
        ----------
        datadir : str, optional, default: None
            Data directory in which the files are written. None defaults to
            the datadir attribute of the parent class.
        """

        filename = "amr_grid.inp"
        datadir = self.datadir if datadir is None else datadir
        Path(datadir).mkdir(parents=True, exist_ok=True)
        path = os.path.join(datadir, filename)

        print("Writing {}.....".format(path), end="")
        Nr = self.rc_grid.shape[0]
        Ntheta = self.thetac_grid.shape[0]
        Nphi = self.phic_grid.shape[0]
        with open(path, "w") as f:
            f.write("1\n")
            f.write("0\n")
            f.write("100\n")
            f.write("0\n")
            f.write(
                "{:d} {:d} {:d}\n".format(
                    1 if Nr > 1 else 0,
                    1 if Ntheta > 1 else 0,
                    1 if Nphi > 1 else 0,
                )
            )
            f.write(
                "{:d} {:d} {:d}\n".format(
                    Nr,
                    Ntheta,
                    Nphi
                )
            )
            for val in self.ri_grid:
                f.write("{:.6e}\n".format(val))
            for val in self.thetai_grid:
                f.write("{:.6e}\n".format(val))
            for val in self.phii_grid:
                f.write("{:.6e}\n".format(val))
        print("done.")

    def _write_dust_density_inp(self, datadir=None):
        """
        Function writes the 'dust_density.inp' input file.

        Parameters
        ----------
        datadir : str, optional, default: None
            Data directory in which the files are written. None defaults to
            the datadir attribute of the parent class.
        """

        filename = "dust_density.inp"
        datadir = self.datadir if datadir is None else datadir
        Path(datadir).mkdir(parents=True, exist_ok=True)
        path = os.path.join(datadir, filename)

        print("Writing {}.....".format(path), end="")
        R_grid, theta_grid, _ = np.meshgrid(
            self.rc_grid, self.thetac_grid, self.phic_grid,
            indexing="ij"
        )
        r_grid = R_grid * np.sin(theta_grid)
        z_grid = R_grid * np.cos(theta_grid)

        rho_grid = np.empty(
            (self.rc_grid.shape[0],
             self.thetac_grid.shape[0],
             self.phic_grid.shape[0],
             self.ac_grid.shape[0])
        )

        # Interpolate dust scale heights on size grid
        x = np.repeat(self.rc_grid_, self.a_dust_.shape[1])
        y = self.a_dust_.flatten()
        z = self.H_dust_.flatten()
        xi = self.rc_grid_
        yi = self.ac_grid
        H_grid_ = griddata((x, y), z, (xi[:, None], yi[None, :]), method="linear", rescale=True)

        for i in range(self.ac_grid.shape[0]):
            rho = np.where(
                (self.a_dust_ >= self.ai_grid[i]) & (self.a_dust_ < self.ai_grid[i+1]),
                self.rho_dust_,
                0.
            ).sum(-1)
            f_H = interp1d(self.rc_grid_, H_grid_[:, i], fill_value="extrapolate")
            f_rho = interp1d(self.rc_grid_, rho, fill_value="extrapolate")
            rho_grid[..., i] = np.maximum(1.e-100, f_rho(r_grid) * np.exp(-0.5*(z_grid/f_H(r_grid))**2))

        with open(path, "w") as f:
            f.write("1\n")
            f.write("{:d}\n".format(rho_grid[..., 0].flatten().shape[0]))
            f.write("{:d}\n".format(rho_grid[0, 0, 0, :].flatten().shape[0]))
            for val in rho_grid.ravel(order="F"):
                f.write("{:.6e}\n".format(val))
        print("done.")

    def _write_dust_temperature_dat(self, datadir=None):
        """
        Function writes the 'dust_temperature.dat' input file.

        Parameters
        ----------
        datadir : str, optional, default: None
            Data directory in which the files are written. None defaults to
            the datadir attribute of the parent class.
        """

        filename = "dust_temperature.dat"
        datadir = self.datadir if datadir is None else datadir
        Path(datadir).mkdir(parents=True, exist_ok=True)
        path = os.path.join(datadir, filename)

        print("Writing {}.....".format(path), end="")

        R_grid, theta_grid, _ = np.meshgrid(
            self.rc_grid, self.thetac_grid, self.phic_grid,
            indexing="ij"
        )
        r_grid = R_grid * np.sin(theta_grid)

        T_grid = np.empty(
            (self.rc_grid.shape[0],
             self.thetac_grid.shape[0],
             self.phic_grid.shape[0],
             self.ac_grid.shape[0])
        )

        for i in range(self.ac_grid.shape[0]):
            f_T = interp1d(self.rc_grid_, self.T_gas_, fill_value="extrapolate")
            T_grid[..., i] = f_T(r_grid)

        with open(path, "w") as f:
            f.write("1\n")
            f.write("{:d}\n".format(T_grid[..., 0].flatten().shape[0]))
            f.write("{:d}\n".format(T_grid[0, 0, 0, :].flatten().shape[0]))
            for val in T_grid.ravel(order="F"):
                f.write("{:.6e}\n".format(val))
        print("done.")

    def _write_dustopac_inp(self, datadir=None):
        """
        Function writes the 'dustopac.inp' input file.

        Parameters
        ----------
        datadir : str, optional, default: None
            Data directory in which the files are written. None defaults to
            the datadir attribute of the parent class.
        """

        filename = "dustopac.inp"
        datadir = self.datadir if datadir is None else datadir
        Path(datadir).mkdir(parents=True, exist_ok=True)
        path = os.path.join(datadir, filename)

        print("Writing {}.....".format(path), end="")
        Nspec = self.ac_grid.shape[0]
        mag = int(np.ceil(np.log10(Nspec)))
        with open(path, "w") as f:
            f.write("2\n")
            f.write("{}\n".format(self.ac_grid.shape[0]))
            for i in range(self.ac_grid.shape[0]):
                f.write("--------------------\n")
                f.write("10\n")
                f.write("0\n")
                f.write("{}".format(i).zfill(mag)+"\n")
        print("done.")

    def _write_dustkapscatmat_inp(self, datadir=None, opacity=None):
        """
        Function writes the 'dustkapscatmat_*.inp' input files.

        Parameters
        ----------
        datadir : str, optional, default: None
            Data directory in which the files are written. None defaults to
            the datadir attribute of the parent class.
        """

        datadir = self.datadir if datadir is None else datadir
        opacity = opacity or self.opacity or 'birnstiel2018'
        Path(datadir).mkdir(parents=True, exist_ok=True)

        Nangle = 181
        Nlam = self.lam_grid.shape[0]
        Nspec = self.ac_grid.shape[0]
        mag = int(np.ceil(np.log10(Nspec)))

        print()
        print("Computing opacities...")
        print("Using dsharp_opac. Please cite Birnstiel et al. (2018).")
        if opacity == "birnstiel2018":
            print("Using DSHARP mix. Please cite Birnstiel et al. (2018).")
            mix, rho_s = do.get_dsharp_mix()
        elif opacity == "ricci2010":
            print("Using Ricci mix. Please cite Ricci et al. (2010).")
            mix, rho_s = do.get_ricci_mix(lmax=self.lam_grid[-1], extrapol=True)
        else:
            raise RuntimeError("Unknown opacity '{}'".format(opacity))
        opac_dict = do.get_opacities(self.ac_grid, self.lam_grid, rho_s, mix, extrapolate_large_grains=True, n_angle=int((Nangle-1)/2+1))
        zscat, _, k_sca, g = do.chop_forward_scattering(opac_dict)
        opac_dict["k_sca"] = k_sca
        opac_dict["g"] = g
        opac_dict["zscat"] = zscat
        print()

        for ia in range(Nspec):
            filename = "dustkapscatmat_{}.inp".format("{:d}".format(ia).zfill(mag))
            path = os.path.join(datadir, filename)
            print("Writing {}.....".format(path), end="")
            with open(path, "w") as f:
                f.write("1\n")
                f.write("{:d}\n".format(Nlam))
                f.write("{:d}\n".format(Nangle))
                f.write("\n")
                for ilam in range(Nlam):
                    f.write(
                        "{:.6e} {:.6e} {:.6e} {:.6e}\n".format(
                            self.lam_grid[ilam]*1.e4,
                            opac_dict["k_abs"][ia, ilam],
                            opac_dict["k_sca"][ia, ilam],
                            opac_dict["g"][ia, ilam],
                        )
                    )
                f.write("\n")
                for theta in opac_dict["theta"]:
                    f.write("{:.2f}\n".format(theta))
                f.write("\n")
                for ilam in range(Nlam):
                    for iang in range(Nangle):
                        f.write(
                            "{:.6e} {:.6e} {:.6e} {:.6e} {:.6e} {:.6e}\n".format(
                                zscat[ia, ilam, iang, 0],
                                zscat[ia, ilam, iang, 1],
                                zscat[ia, ilam, iang, 2],
                                zscat[ia, ilam, iang, 3],
                                zscat[ia, ilam, iang, 4],
                                zscat[ia, ilam, iang, 5],
                            )
                        )
            print("done.")


def read_model(datadir=""):
    """
    This functions reads the ``RADMC-3D`` model files and returns a namespace with the data.
    It should only be used for models created by ``dustpylib``. For more complex models
    use ``Radmc3dPy``.

    Parameters
    ----------
    datadir : str, optional, default: ""
        The path of the directory with the ``RADMC-3D`` input files

    Returns
    -------
    data : namespace
        Namespace with the model data
    """

    d = {}
    d["grid"] = _read_amr_grid_inp(datadir=datadir)
    d["rho"] = _read_dust_density_inp(datadir=datadir)

    path = os.path.join(datadir, "dust_temperature.dat")
    if os.path.isfile(path):
        d["T"] = _read_dust_temperature_dat(datadir=datadir)

    return SimpleNamespace(**d)


def _read_amr_grid_inp(datadir=""):
    """
    This functions reads the ``RADMC-3D`` model files and returns a namespace with the grid.
    It should only be used for models created by ``dustpylib``. For more complex models
    use ``Radmc3dPy``.

    Parameters
    ----------
    datadir : str, optiona, default: ""
        The path of the directory with the ``RADMC-3D`` input files

    Returns
    -------
    grid : namespace
        Namespace with the grid data
    """

    filename = "amr_grid.inp"
    path = os.path.join(datadir, filename)

    head = 10
    header = np.fromfile(path, dtype=int, count=head, sep=" ")
    Nr, Ntheta, Nphi = header[-3], header[-2], header[-1]

    grid = np.fromfile(path, dtype=float, count=-1, sep=" ")
    ri_grid = grid[head:head+Nr+1]
    thetai_grid = grid[head+Nr+1:head+Nr+1+Ntheta+1]
    phii_grid = grid[head+Nr+1+Ntheta+1:head+Nr+1+Ntheta+1+Nphi+1]

    rc_grid = 0.5*(ri_grid[1:]+ri_grid[:-1])
    thetac_grid = 0.5*(thetai_grid[1:]+thetai_grid[:-1])
    phic_grid = 0.5*(phii_grid[1:]+phii_grid[:-1])

    d = {
        "r": rc_grid,
        "theta": thetac_grid,
        "phi": phic_grid
    }

    return SimpleNamespace(**d)


def _read_dust_density_inp(datadir=""):
    """
    This functions reads the ``RADMC-3D`` model files and returns the dust density.
    It should only be used for models created by ``dustpylib``. For more complex models
    use ``Radmc3dPy``.

    Parameters
    ----------
    datadir : str, optiona, default: ""
        The path of the directory with the ``RADMC-3D`` input files

    Returns
    -------
    rho : array-like
        The dust density
    """

    grid = _read_amr_grid_inp(datadir=datadir)
    Nr, Ntheta, Nphi = grid.r.shape[0], grid.theta.shape[0], grid.phi.shape[0]

    filename = "dust_density.inp"
    path = os.path.join(datadir, filename)

    head = 3
    header = np.fromfile(path, dtype=int, count=head, sep=" ")
    Nspec = header[-1]

    rho = np.fromfile(
        path,
        dtype=float,
        count=-1,
        sep=" "
    )[head:].reshape((Nspec, Nphi, Ntheta, Nr))
    rho = rho.swapaxes(3, 0)
    rho = rho.swapaxes(2, 1)

    return rho


def _read_dust_temperature_dat(datadir=""):
    """
    This functions reads the ``RADMC-3D`` model files and returns the dust temperature.
    It should only be used for models created by ``dustpylib``. For more complex models
    use ``Radmc3dPy``.

    Parameters
    ----------
    datadir : str, optiona, default: ""
        The path of the directory with the ``RADMC-3D`` input files

    Returns
    -------
    T : array-like
        The dust temperature
    """

    grid = _read_amr_grid_inp(datadir=datadir)
    Nr, Ntheta, Nphi = grid.r.shape[0], grid.theta.shape[0], grid.phi.shape[0]

    filename = "dust_temperature.dat"
    path = os.path.join(datadir, filename)

    head = 3
    header = np.fromfile(path, dtype=int, count=head, sep=" ")
    Nspec = header[-1]

    T = np.fromfile(
        path,
        dtype=float,
        count=-1,
        sep=" "
    )[head:].reshape((Nspec, Nphi, Ntheta, Nr))
    T = T.swapaxes(3, 0)
    T = T.swapaxes(2, 1)

    return T


def read_image(path):
    """
    This functions reads an image file created by ``RADMC-3D`` and returns a dictionary
    with the image data.

    Parameters
    ----------
    path : str
        Path to the image data file

    Returns
    -------
    d : dict
        Dictionary with the image data
    """

    head = 4
    header = np.fromfile(path, dtype=int, count=head, sep=" ")
    iformat = header[0]
    Nx, Ny = header[1], header[2]
    Nlam = header[3]

    image = np.fromfile(path, dtype=float, count=-1, sep=" ")
    pix_x, pix_y = image[4], image[5]

    Wx = Nx*pix_x
    Wy = Ny*pix_y
    xi = np.linspace(-Wx/2., Wx/2., Nx+1)
    x = 0.5*(xi[1:]+xi[:-1])
    yi = np.linspace(-Wy/2., Wy/2., Ny+1)
    y = 0.5*(yi[1:]+yi[:-1])

    lam = image[6:6+Nlam]*1.e-4

    if iformat == 1:
        I = image[6+Nlam:].reshape(
            (Nlam, Ny, Nx)
        ).swapaxes(2, 0)
        Q = np.zeros_like(I)
        U = np.zeros_like(I)
        V = np.zeros_like(I)
    elif iformat == 3:
        image = image[6+Nlam:].reshape((-1, 4))
        I = image[:, 0].reshape((Nlam, Ny, Nx)).swapaxes(2, 0)
        Q = image[:, 1].reshape((Nlam, Ny, Nx)).swapaxes(2, 0)
        U = image[:, 2].reshape((Nlam, Ny, Nx)).swapaxes(2, 0)
        V = image[:, 3].reshape((Nlam, Ny, Nx)).swapaxes(2, 0)
    else:
        raise RuntimeError("Invalid file iformat: '{}'. Only '1' or '3' supported.".format(iformat))

    d = {
        "x": x,
        "y": y,
        "lambda": lam,
        "I": I,
        "Q": Q,
        "U": U,
        "V": V,
    }

    return d


def read_spectrum(path):
    """
    This functions reads a spectrum file created by ``RADMC-3D`` and returns a dictionary
    with the SED data.

    Parameters
    ----------
    path : str
        Path to the spectrum data file

    Returns
    -------
    d : dict
        Dictionary with the SED data
    """

    sed = np.fromfile(path, dtype=float, count=-1, sep=" ")
    sed = sed[2:].reshape((-1, 2))

    lam = sed[:, 0]*1.e-4
    flux = sed[:, 1]

    d = {
        "lambda": lam,
        "flux": flux,
    }

    return d
