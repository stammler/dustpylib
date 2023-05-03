import numpy as np

def kanagawa2017(r, a, q, h, alpha0):
    """
    Function calculates the planetary gap profile according Kanagawa et al. (2017).

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

    # Unperturbed return value
    ret = np.ones_like(r)

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
    mask1 = np.logical_and(dr1 < dist, dist < dr2)
    ret = np.where(mask1, SigGap, ret)
    # Gap center
    mask2 = dist < dr1
    ret = np.where(mask2, SigMin, ret)

    return ret