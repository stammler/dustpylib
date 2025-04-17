from pathlib import Path
from types import SimpleNamespace
import warnings
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import dsharp_opac
import dustpy

c_light = 29979245800.0
pc = 3.0856775814913674e+18
jy_sas = 4.25451702961522e-13
year = 31557600.0
k_B = 1.380649e-16
h = 6.62607015e-27


def J_over_B(tauz_in, eps_e, tau):
    """Calculate the mean intensity in units of the Planck function value.

    Note: This follows Eq. 15 of Birnstiel et al. 2018. We later found
    that this was already solved in Miyake & Nakagawa 1993 (and used
    in Sierra et al. 2017). The equations are written slightly differently
    but are equivalent.

    Parameters
    ----------
    tauz_in : float | array
        optical depth at which the mean intensity should be returned
    eps_e : float
        effective absorption probability (= 1 - effective albedo)
    tau : float
        total optical depth

    Returns
    -------
    float | array
        mean intensity evaluated at `tauz_in`
    """
    # our tauz goes from 0 to tau
    # while in the paper it goes from -tau/2 to +tau/2
    if isinstance(tauz_in, np.ndarray):
        tauz = tauz_in.copy() - tau / 2
    else:
        tauz = tauz_in - tau / 2

    b = 1.0 / (
        (1.0 - np.sqrt(eps_e)) * np.exp(-np.sqrt(3 * eps_e) * tau) + 1 + np.sqrt(eps_e))

    J = 1.0 - b * (
        np.exp(-np.sqrt(3.0 * eps_e) * (0.5 * tau - tauz)) +
        np.exp(-np.sqrt(3.0 * eps_e) * (0.5 * tau + tauz)))

    return J


def S_over_B(tauz, eps_e, tau):
    """Calculate the source function in units of the Planck function value.

    Note: This follows Eq. 19 of Birnstiel et al. 2018.

    Parameters
    ----------
    tauz : float | array
        optical depth at which the mean source function should be returned
    eps_e : float
        effective absorption probability (= 1 - effective albedo)
    tau : float
        total optical depth

    Returns
    -------
    float | array
        source function evaluated at `tauz`
    """
    return eps_e + (1.0 - eps_e) * J_over_B(tauz, eps_e, tau)


def I_over_B(tau_total, eps_e, mu=1, ntau=300):
    """Integrates the scattering solution of Birnstiel 2018 numerically.

    This integrates Eq. 17 of Birnstiel 2018. See note for `J_over_B`.

    Parameters
    ----------
    tau_total : float
        total optical depth
    eps_e : float
        effective extinction probablility (1-albedo)
    mu : float, optional
        cosine of the incidence angle, by default 1
    ntau : int, optional
        number of grid points, by default 300

    Returns
    -------
    float
        outgoing intensity in units of the planck function.
    """
    tau = np.linspace(0, tau_total, ntau)
    Inu = np.zeros(ntau)
    Jnu = J_over_B(tau, eps_e, tau_total)
    # the first 1.0 here is a placeholder for Bnu, just for reference
    Snu = eps_e * 1.0 + (1.0 - eps_e) * Jnu
    for i in range(1, ntau):
        dtau = (tau[i] - tau[i - 1]) / mu
        expdtray = np.exp(-dtau)
        srcav = 0.5 * (Snu[i] + Snu[i - 1])
        Inu[i] = expdtray * Inu[i - 1] + (1 - expdtray) * srcav
    return Inu[-1]


def I_over_B_EB(tau, eps_e, mu=1):
    """"same as I_over_B but using the Eddington-Barbier approximation.

    This solves Eq. 19 of Birnstiel et al. 2018, but see also notes
    in `J_over_B`.

    Parameters
    ----------
    tau : float
        total optical depth
    eps_e : float
        effective extinction probablility (1-albedo)
    mu : float, optional
        cosine of the incidence angle, by default 1

    Returns
    -------
    float
        outgoing intensity in units of the planck function.
    """
    arg = np.where(tau > 2. / 3. * mu, tau - 2. / 3. * mu, tau)
    dummy = np.where(tau / mu > 1e-15,
                     1.0 - np.exp(-tau / mu),
                     tau / mu)
    return dummy * S_over_B(arg, eps_e, tau)


class Opacity(object):
    "a simple opacity object that stores and interpolates opacity tables for k_abs, k_sca, and g."
    _filename = None
    _lam = None
    _a = None
    _k_abs = None
    _k_sca = None
    _g = None
    _rho_s = None
    _extrapol = False

    @property
    def rho_s(self):
        "material density in g/cm^3"
        return self._rho_s

    def __init__(self, input=None, **kwargs):
        """Object to read and interpolate opacities.

        input : str | path
            the name of the opacity file to be read. If it doesn't
            exist at the given position, it will try to get it from
            dsharp.

            The opacity file can be a .npz file as used in dsharp_opac
            or it can be a radmc3d opacity file (not the scattering matrix!).

        kwargs : keyword dict
            they are passed to the `RegularGridInterpolator`. This way
            it is possible to turn off or change the interpolation method
            (in log-log space for the opacities, in log-linear space
            for g), e.g. by passing keywords like `bounds_error=True`.

            For g, the extrapolation will be set to nearest neighbor.
        """

        kwargs['fill_value'] = kwargs.get('fill_value', None)
        kwargs['bounds_error'] = kwargs.get('bounds_error', False)

        # set default opacities

        if input is None:
            input = 'default_opacities_smooth.npz'

        if type(input) is dict:
            # if a dict is passed, we can import it directly
            self._load_from_dict_like(input)
        else:
            # else, it should be a file or something that dsharp_opac accepts.
            if type(input) is str:
                input = Path(input)

            # if it is a .inp file, we assume radmc3d.
            if input.suffix == '.inp':
                self._load_from_dustkappa(input)
            else:
                if not input.is_file():
                    # if the input is not a file: try to get it from dsharp
                    input = Path(dsharp_opac.get_datafile(str(input)))
                    if not input.is_file():
                        # if it is still not a file, raise an error
                        raise ValueError('unknown input')

                # now we can read the dsharp_opac-like file
                self._filename = str(input)
                with np.load(input) as f:
                    self._load_from_dict_like(f)

        self._interp_k_abs = RegularGridInterpolator(
            (np.log10(self._lam), np.log10(self._a)), np.log10(self._k_abs).T, **kwargs)
        if self._k_sca is not None:
            self._interp_k_sca = RegularGridInterpolator(
                (np.log10(self._lam), np.log10(self._a)), np.log10(self._k_sca).T, **kwargs)
        else:
            self._interp_k_sca = None

        if self._g is not None:
            kwargs['method'] = 'nearest'
            self._interp_g = RegularGridInterpolator(
                (np.log10(self._lam), np.log10(self._a)), self._g.T, **kwargs)

    def _load_from_dict_like(self, input):
        for attr in ['a', 'lam', 'k_abs', 'k_sca', 'g', 'rho_s']:
            if attr in input:
                setattr(self, '_' + attr, input[attr])
            else:
                print(f'{attr} not in input')
        if self._rho_s is not None:
            self._rho_s = float(self._rho_s)

    def _load_from_dustkappa(self, file):
        with open(file, 'r') as f:
            iformat = np.fromfile(f, count=1, sep=' ', dtype=int).squeeze()[()]
            nlam = np.fromfile(f, count=1, sep=' ', dtype=int).squeeze()[()]
            data = np.loadtxt(f, dtype=float)
            self._lam = data[:, 0] * 1e-4
            self._a = [1e0]  # arbitrary, but needed for the interpolation
            self._rho_s = None

            if nlam != len(self._lam):
                warnings.warn(
                    'number of wavelength does not match the number ' + 
                    'of data rows. Will still use all rows')
                nlam = len(self._lam)
            if iformat > 0:
                self._k_abs = data[:, 1].reshape(1, nlam)
            else:
                raise ValueError('iformat needs to be larger than 0')
            if iformat > 1:
                self._k_sca = data[:, 2].reshape(1, nlam)
            else:
                self._k_sca = None
            if iformat > 2:
                self._g = data[:, 3].reshape(1, nlam)
            else:
                self._g = None


    def _check_input(self, a, lam):
        """Checks if the input is asking for extrapolation in a reasonable range

        Parameters
        ----------
        a : float | array
            particle size in cm

        lam : float | array
            wavelength in cm
        """
        # either we are in the grid of known opacities
        # OR we are at large enough optical sizes to be
        # properly extrapolating
        mask = \
            (
                ((a <= self._a[-1]) & (a >= self._a[0]))[:, None] &
                ((lam <= self._lam[-1]) & (lam >= self._lam[0]))[None, :]
            ) | (
                a[:, None] > 100 * lam[None, :] / (2 * np.pi)
            )
        if not np.all(mask):
            raise ValueError(
                'extrapolating too close to the interference part of the opacities')

    def get_opacities(self, a, lam):
        """
        Returns the absorption and scattering opacities for the given particle
        size a and wavelength lam.

        Arguments:
        ----------

        a : float | array
            particle size in cm

        lam : float | array
            wavelength in cm

        Returns:
        --------
        k_abs, k_sca : arrays
            absorption and scattering opacities, each of shape (len(a), len(lam))
        """
        out_shape = (*a.shape, *lam.shape)
        a = a.ravel()
        lam = lam.ravel()
        self._check_input(a, lam)
        k_abs = 10.**self._interp_k_abs(tuple(np.meshgrid(np.log10(lam), np.log10(a))))
        if self._interp_k_sca is not None:
            k_sca = 10.**self._interp_k_sca(tuple(np.meshgrid(np.log10(lam), np.log10(a))))
        else:
            k_sca = np.zeros_like(k_abs)
        return  k_abs.reshape(out_shape), k_sca.reshape(out_shape)

    def get_g(self, a, lam):
        """
        Returns the asymmetry parameter for the given particle
        size a and wavelength lam.

        Arguments:
        ----------

        a : float | array
            particle size in cm

        lam : float | array
            wavelength in cm

        Returns:
        --------
        g : arrays
            asymmetry parameter, array of shape (len(a), len(lam))
        """
        out_shape = (*a.shape, *lam.shape)
        a = a.ravel()
        lam = lam.ravel()
        if self._g is None:
            return np.zeros(out_shape).squeeze()
        else:
            self._check_input(a, lam)
            return self._interp_g(tuple(np.meshgrid(np.log10(lam), np.log10(a)))).reshape(out_shape)

    def get_k_ext_eff(self, a, lam):
        """
        Returns the effective extinction opacity for the given particle
        size `a` and wavelength `lam`.

            k_ext_eff = k_abs + (1.0 - g) * k_sca

        Arguments:
        ----------

        a : float | array
            particle size in cm

        lam : float | array
            wavelength in cm

        Returns:
        --------
        k_ext_eff : arrays
            effective extinction opacity, array of shape (len(a), len(lam))
        """
        k_a, k_s = self.get_opacities(a, lam)
        g = self.get_g(a, lam)

        k_se = (1.0 - g) * k_s
        k_ext_eff = k_a + k_se
        return k_ext_eff


def get_observables(r, sig_da, T, a, lam, opacity: Opacity, scattering=True, inc=0.0,
                    distance=140 * pc, flux_fraction=0.68):
    """
    Calculates the radial profiles of the (vertical) optical depth and the intensity
    as well as the integrated flux and the effective radius.

    Arguments:
    ----------

    r : array
        the radial grid [cm]

    sig_da : array
        the dust surface density [g/cm^2], shape (len(r), len(a))

    T : array
        the gas temperature [K], shape (len(r))

    a : array
        the dust particle sizes [cm], either shape (N_radii, N_sizes)
        or just shape (N_sizes) if the size grid is the same at every radius.

    lam : array
        the wavelengths at which to calculate the observables [cm]

    opacity : instance of dustpylib.radtrans.slab.Opacity
        the opacity to use for the calculation

    distance : float
        distance to source [cm]

    flux_fraction : float
        at which fraction of the total flux the effective radius is defined [-]

    Keywords:
    ---------

    scattering : bool
        if True, use the scattering solution, else just absorption

    inc : float
        inclination, default is 0.0 = face-on. This is just treated as
        increasing the path length across the slab model.


    Output:
    -------
    SimpleNamespace with the following attributes:

    rf : array
        effective radii for every wavelength [cm]

    flux_t : array
        integrated flux for every wavelength [Jy]

    tau, Inu : arrays
        optical depth and intensity profiles at every wavelength [-, Jy/arcsec**2]
    """

    from scipy.integrate import cumulative_trapezoid as cumtrapz

    # Total dust surface density
    sig_d_tot = sig_da.sum(-1)
    # Make sure wavelengths is one-dimensional
    lam = np.array(lam, ndmin=1)

    nr, na = sig_da.shape
    nlam = len(lam)

    # Get opacities at our wavelength and particle sizes
    # this should return k_a, k_s of shape (Nr, Na, Nlam) or (Nr, Nlam)
    # same for k_ext and g below
    k_a, k_s = opacity.get_opacities(a, lam)
    if scattering:
        g = opacity.get_g(a, lam)
        k_se = (1. - g) * k_s
        k_ext = k_a + k_se
    else:
        k_ext = k_a

    # Frequency and Planck function at every radius
    # this should be shape (nr, nlam)
    freq = c_light / lam
    B_nu = bplanck(freq[None, :], T[:, None])

    # Optical depth
    # sig_da should be shape (nr, na)
    # k_ext should be shape (na, nlam), (nr, na, nlam)
    tau = np.minimum(
        100., (sig_da[..., None] * k_ext.reshape(-1, na, nlam)).sum(axis=1) / np.cos(inc))

    # Mean opacities and intensity
    if scattering:
        k_a_mean = (sig_da[..., None] * k_a.reshape(-1, na, nlam)
                    ).sum(axis=1) / sig_d_tot[..., None]
        k_s_mean = (sig_da[..., None] * k_se.reshape(-1, na, nlam)
                    ).sum(axis=1) / sig_d_tot[..., None]
        eps_avg = k_a_mean / (k_a_mean + k_s_mean)
        I_nu = B_nu * I_over_B_EB(tau, eps_avg)  # (nr, nlam)
    else:
        dummy = np.where(tau > 1e-15, (1.0-np.exp(-tau)), tau)
        I_nu = B_nu * dummy

    # Fluxes (nr, nlam)
    flux = np.cos(inc) / distance**2 * cumtrapz(
        2. * np.pi * r[:, None] * I_nu, x=r, axis=0, initial=0)
    flux_t = flux[-1, :] / 1.e-23

    # Intensity in Jy per arcsec
    I_nu /= jy_sas

    # Luminosity size of disk
    rf = np.array([np.interp(flux_fraction, _f / _f[-1], r) for _f in flux.T])

    return SimpleNamespace(
        rf=rf,
        flux_t=flux_t,
        tau=tau,
        I_nu=I_nu,
        sig_da=sig_da)


def bplanck(freq, temp):
    """
    This function computes the Planck function

                   2 h nu^3 / c^2
       B_nu(T)  = ------------------    [ erg / cm^2 s ster Hz ]
                  exp(h nu / kT) - 1

    Arguments:
     freq  [Hz]            = Frequency in Herz (can be array)
     temp  [K]             = Temperature in Kelvin (can be array)
    """
    const1 = h / k_B
    const2 = 2 * h / c_light**2
    const3 = 2 * k_B / c_light**2
    x = const1 * freq / (temp + 1e-99)
    if np.isscalar(x):
        if x > 500.:
            x = 500.
    else:
        x[np.where(x > 500.)] = 500.
    bpl = const2 * (freq**3) / ((np.exp(x) - 1.e0) + 1e-99)
    bplrj = const3 * (freq**2) * temp
    if np.isscalar(x):
        if x < 1.e-3:
            bpl = bplrj
    else:
        ii = x < 1.e-3
        bpl[ii] = bplrj[ii]
    return bpl


def get_all_observables(data, opacity, lam, **kwargs):
    """
    This function calculates the observables for a list of simulations or a single simulation.

    Arguments:
    ----------
    data : list | str | SimpleNamespace
        the data to be analyzed. This can be a list of SimpleNamespace objects, a path to a
        directory containing the data, or a single SimpleNamespace object.

    opacity : instance of dustpylib.radtrans.slab.Opacity

    lam : array
        the wavelengths at which to calculate the observables [cm]

    Keywords:
    ---------
    distance : float
        distance to source [cm]

    flux_fraction : float
        at which fraction of the total flux the effective radius is defined [-]

    scattering : bool
        if True, use the scattering solution, else just absorption

    inc : float
        inclination, default is 0.0 = face-on. This is just treated as
        increasing the path length across the slab model.

    Returns:
    --------
    SimpleNamespace with the following attributes:

    rf : array
        effective radii for every wavelength [cm]

    flux_t : array
        integrated flux for every wavelength [Jy]

    tau, Inu : arrays
        optical depth and intensity profiles at every wavelength [-, Jy/arcsec**2]
    """

    if isinstance(data, str):
        reader = dustpy.hdf5writer()
        reader.datadir = str(data)
        data = reader.read.all()

    if isinstance(data, SimpleNamespace):
        if 'sig_da' in data.__dict__:
            # this assumes that data contains these specific entries
            # which have time as first index

            n_t = data.sig_da.shape[0]

            # loop over times to make a list of observables
            obs = []
            for it in range(n_t):

                obs += [
                    
                    get_observables(
                        data.r,
                        data.sig_da[it],
                        data.T[it],
                        data.a[it],
                        lam, opacity, **kwargs)
                    for it in range(len(data.t))
                ]

        elif 'dust' in data.__dict__:

            if data.dust.Sigma.ndim == 3:
                n_t = data.dust.Sigma.shape[0]
            elif data.dust.Sigma.ndim == 2:
                n_t = 1
                sli = np.s_[...]

            obs = []
            for it in range(n_t):
                if data.dust.Sigma.ndim == 3:
                    sli = np.s_[it, ...]
                obs += [
                    get_observables(
                        data.grid.r[sli],
                        data.dust.Sigma[sli],
                        data.gas.T[sli],
                        data.dust.a[sli],
                        lam, opacity, **kwargs
                    )
                ]

    # in any case: concatenate the observables
    obs = SimpleNamespace(
        rf=np.array([_o.rf for _o in obs]),
        flux_t=np.array([_o.flux_t for _o in obs]),
        tau=np.array([_o.tau for _o in obs]),
        I_nu=np.array([_o.I_nu for _o in obs]),
        sig_da=np.array([_o.sig_da for _o in obs])
    )

    return obs
