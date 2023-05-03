import numpy as np

def drazkowska2016(OmegaK, rho_dust, rho_gas, Sigma_dust, St, p2g_crit=1., St_crit=0.01, zeta=0.01):
    """
    Function calculates the dust source term due to planetesimal
    formation of Darzkowska et al. (2016).

    Parameters
    ----------
    OmegaK : array-like, (Nr,)
        Keplerian frequency
    rho_dust : array-like, (Nr, Nm)
        Midplane dust volume density
    rho_gas : array-like, (Nr,)
        Midplane gas volume density
    Sigma_dust : array-like, (Nr, Nm)
        Dust surface density
    St : array-like, (Nr, Nm)
        Stokes numbers
    p2g_crit : float, optional, default: 1.
        Critical midplane pebbles-to-gas ratio of particles above
        St_crit above which planetesimal formation is triggered
    St_crit : float, optional, default: 0.01
        Critical Stokes number above which dust particles
        contribute to trigger planetesimal formation
    zeta : float, optional, default: 0.1
        Planetesimal formation efficiency

    Returns
    -------
    S : array-like, (Nr, Nm)
        Dust source terms due to planetesimal formation
    """
    mask = St>=St_crit
    p2g_mid = np.where(mask, rho_dust, 0.).sum(-1)/rho_gas
    trigger = np.where(p2g_mid>=p2g_crit, True, False)

    Sigma_pebbles = np.where(mask, Sigma_dust, 0.)

    S = np.where(trigger[:, None], -zeta*Sigma_pebbles*OmegaK[:, None], 0.)
    S[0, :] = 0.
    S[-1, :] = 0.

    return S


def miller2021(OmegaK, rho_dust, rho_gas, Sigma_dust, St, d2g_crit=1., n=0.03, zeta=0.1):
    """
    Function calculates the dust source term due to planetesimal
    formation of Miller et al. (2021).

    Parameters
    ----------
    OmegaK : array-like, (Nr,)
        Keplerian frequency
    rho_dust : array-like, (Nr, Nm)
        Midplane dust volume density
    rho_gas : array-like, (Nr,)
        Midplane gas volume density
    Sigma_dust : array-like, (Nr, Nm)
        Dust surface density
    St : array-like, (Nr, Nm)
        Stokes numbers
    d2g_crit : float, optional, default: 1.
        Critical midplane dust-to-gas ratio above which
        planetesimal formation is triggered
    n : float, optional, default: 0.03
        Smoothness parameter of dust-to-gas ratio transition
    zeta : float, optional, default: 0.1
        Planetesimal formation efficiency

    Returns
    -------
    S : array-like, (Nr, Nm)
        Dust source terms due to planetesimal formation
    """
    d2g_mid = rho_dust.sum(-1)/rho_gas
    P = 0.5*(1.+np.tanh((np.log10(d2g_mid)-np.log10(d2g_crit))/n))

    S = -P[:, None]*zeta*Sigma_dust*St*OmegaK[:, None]
    S[0, :] = 0.
    S[-1, :] = 0.

    return S


def schoonenberg2018(OmegaK, rho_dust, rho_gas, Sigma_dust, St, d2g_crit=1., zeta=0.1):
    """
    Function calculates the dust source term due to planetesimal
    formation of Schoonenberg et al. (2018).

    Parameters
    ----------
    OmegaK : array-like, (Nr,)
        Keplerian frequency
    rho_dust : array-like, (Nr, Nm)
        Midplane dust volume density
    rho_gas : array-like, (Nr,)
        Midplane gas volume density
    Sigma_dust : array-like, (Nr, Nm)
        Dust surface density
    St : array-like, (Nr, Nm)
        Stokes numbers
    d2g_crit : float, optional, default: 1.
        Critical midplane dust-to-gas ratio above which
        planetesimal formation is triggered
    zeta : float, optional, default: 0.1
        Planetesimal formation efficiency

    Returns
    -------
    S : array-like, (Nr, Nm)
        Dust source terms due to planetesimal formation
    """
    d2g_mid = rho_dust.sum(-1)/rho_gas
    trigger = np.where(d2g_mid>=d2g_crit, True, False)

    S = np.where(trigger[:, None], -zeta*Sigma_dust*St*OmegaK[:, None], 0.)
    S[0, :] = 0.
    S[-1, :] = 0.

    return S