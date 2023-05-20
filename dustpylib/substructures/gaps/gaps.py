import numpy as np


def duffell2020(r, a, q, h, alpha0):
    """
    Function calculates the planetary gap profile according Duffell (2020).

    Parameters
    ----------
    r : array-like, (Nr,)
        Radial grid
    a : float
        Semi-major axis of planet
    q : float
        Planet-star mass ratio
    h : float
        Aspect ratio at planet location
    alpha0 : float
        Unperturbed alpha viscosity parameter

    Returns
    -------
    f : array-like, (Nr,)
        Pertubation of surface density due to planet
    """

    # Mach number
    M = 1./h

    # Add small value to avoid division by zero
    qp = q + 1.e-100

    # qtilde from equation (18) has shape (Nr,)
    D = 7*M**1.5/alpha0**0.25
    qtilde = qp/(1+D**3*((r/a)**(1./6.)-1)**6)**(1./3.)

    # delta from equation (9)
    # Note: there is a typo in the original publication
    # (q/qw)**3 is added in both cases
    qnl = 1.04/M**3
    qw = 34. * qnl * np.sqrt(alpha0*M)
    delta = np.where(qtilde > qnl, np.sqrt(qnl/qtilde), 1.) + (qtilde/qw)**3

    # Gap shape
    ret = 1. / (1. + 0.45/(3.*np.pi) * qtilde**2 * M**5 * delta/alpha0)

    return ret


def kanagawa2017(r, a, q, h, alpha0):
    """
    Function calculates the planetary gap profile according
    Kanagawa et al. (2017).

    Parameters
    ----------
    r : array-like, (Nr,)
        Radial grid
    a : float
        Semi-major axis of planet
    q : float
        Planet-star mass ratio
    h : float
        Aspect ratio at planet location
    alpha0 : float
        Unperturbed alpha viscosity parameter

    Returns
    -------
    f : array-like, (Nr,)
        Pertubation of surface density due to planet
    """

    # Distance to planet (normalized)
    dist = np.abs(r-a)/a

    # Add small value to avoid division by zero
    qp = q + 1.e-100

    K = qp**2 / (h**5 * alpha0)
    Kp = qp**2 / (h**3 * alpha0)
    Kp4 = Kp**(0.25)
    SigMin = 1. / (1 + 0.04*K)
    SigGap = 4 / Kp4 * dist - 0.32
    dr1 = (0.25*SigMin + 0.08) * Kp**0.25
    dr2 = 0.33 * Kp**0.25
    # Gap edges
    ret = np.where((dr1 < dist) & (dist < dr2), SigGap, 1.)
    # Gap center
    ret = np.where(dist < dr1, SigMin, ret)

    return ret
