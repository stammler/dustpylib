import numpy as np

def refine_radial_local(ri, r0, num=3):
    """
    Function refines the radial grid locally bysplitting grid cells
    recursively at a specific location.

    Parameters
    ----------
    ri : array-like, (Nr,)
        Radial grid cell interfaces
    r0 : float
        Radial location to be refined
    num : int, optional, default: 3
        Number of refinement steps

    Returns
    -------
    ri_fine : array-like, (Nr+)
        Refined radial grid cell interfaces
    """
    # Break recursion
    if num == 0:
        return ri

    # Closest index to location
    i0 = np.abs(ri-r0).argmin()
    # Boundary indices of refinement region
    il = np.maximum(0, i0-num)
    ir = np.minimum(i0+num, ri.shape[0]-1)

    # Left and right unmodified regions
    ril = ri[:il]
    rir = ri[ir:]

    # Initialize refined region
    N = ir-il
    rim = np.empty(2*N)

    # Refined grid boundary is geometric mean
    for i in range(0, N):
        j = il+i
        rim[2*i] = ri[j]
        rim[2*i+1] = np.sqrt(ri[j]*ri[j+1])

    # New refined grid
    ri_fine = np.hstack((ril, rim, rir))

    # Next level of recursion
    return refine_radial_local(ri_fine, r0, num=num-1)