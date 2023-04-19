from dustpy import Simulation
import os


class RADMC3D():

    def __init__(self, sim):

        self.a_grid = None
        self.lam_grid = None

        self.M_star = None
        self.R_star = None
        self.T_star = None

        self.datadir = "."

        if isinstance(sim, Simulation):
            self._init_from_dustpy(sim)

    def _init_from_dustpy(self, sim):

        self.M_star = sim.star.M
        self.R_star = sim.star.R
        self.T_star = sim.star.T

    def write_files(self, datadir=None):
        """
        Function writes all required RADMC3D input files.

        Parameters
        ----------
        datadir : str, optional, default: None
            Data directory in which the files are written. None defaults to the datadir attribute of the parent class.
        """
        if datadir is None:
            datadir = self.datadir
        self._write_stars_inp(datadir=datadir)

    def _write_stars_inp(self, datadir=None):
        """
        Function writes the 'stars.inp' input file for the central star.

        Parameters
        ----------
        datadir : str, optional, default: None
            Data directory in which the files are written. None defaults to the datadir attribute of the parent class.
        """

        filename = "stars.inp"
        if datadir is None:
            datadir = self.datadir
        path = os.path.join(datadir, filename)

        with open(path, "w", ) as f:
            f.write("2\n")
            f.write("1 {:d}\n".format(self.lam_grid.shape[0]))
            f.write("{:.6e} {:.6e} {:.6e} {:.6e} {:.6e}\n".format(self.R_star, self.M_star, 0., 0., 0.))
            for lam in self.lam_grid:
                f.write("{:.6e}\n".format(lam*1.e4))
            f.write("-{:.6e}\n".format(self.T_star))
        