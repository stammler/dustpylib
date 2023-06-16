from dustpy.std.dust import D as dustDiffusivity
import numpy as np

#########################################################################################
#
# Backreaction Coefficients (simplified)
#
#########################################################################################


# Compute backreaction coefficients AB assuming vertically uniform dust-to-gas ratio per species
def BackreactionCoefficients(sim):
    '''
    Updater of the dust.backreaction Group.

    Obtain the backreaction coefficients considering the contribution of each dust species.
    For more information check Garate et al. (2019), equations 23 - 26 in Appendix.
    This implementation does not consider the vertical structure.
    Hence, all the dust species and the gas feel the same backreaction.

    ------------------------------

    Assigns the backreaction coefficients are returned to:
    sim.dust.backreaction.A
    sim.dust.backreaction.B

    '''
    # Additional Parameters
    # Set the last cell to the default values (A=1, B=0) for stability.
    OmitLastCell = True

    # Gas and Dust surface densities
    Sigma_g = sim.gas.Sigma
    Sigma_d = sim.dust.Sigma

    # Dust-to-Gas ratio (of each dust species)
    d2g_ratio = Sigma_d / Sigma_g[:, None]
    St = sim.dust.St                        # Stokes number

    # X, Y integrals (at each radius).
    factor_xy = 1.0 + np.square(St)
    integral_X = (1.0 / factor_xy) * d2g_ratio
    integral_Y = (St / factor_xy) * d2g_ratio

    # Sum over the mass axis
    X = np.sum(integral_X, axis=1)
    Y = np.sum(integral_Y, axis=1)

    # Backreaction Coefficients A, B (at each radius).
    factor_AB = np.square(Y) + np.square(1.0 + X)
    A = (X + 1) / factor_AB
    B = Y / factor_AB

    # Recomended to turn off backreactions at the last cell. Observed mass loss to happen in some cases.
    if OmitLastCell:
        A[-1] = 1.0
        B[-1] = 0.0

    # Assign the backreaction coefficients
    sim.dust.backreaction.A = A
    sim.dust.backreaction.B = B


#########################################################################################
#
# Damped diffusivity by the dust-to-gas ratio
#
#########################################################################################


def dustDiffusivity_Backreaction(sim):
    '''
    Reduces the dust diffusivity, accounts for the effect of the local dust-to-gas ratio.
    '''
    d2g_ratio = np.sum(sim.dust.Sigma, axis=-1) / sim.gas.Sigma
    return dustDiffusivity(sim) / (1. + d2g_ratio[:, None])


#########################################################################################
#########################################################################################
#
# Advanced Backreaction Coefficients (Considering the vertical structure of the gas and dust)
#
#########################################################################################
#########################################################################################
def BackreactionCoefficients_VerticalStructure(sim):
    '''
    Updater of the dust.backreaction Group.

    Obtain the backreaction coefficients considering the vertical structure.
    For more information check Garate et al. (2019), equations 23 - 26 in Appendix.

    Considers that the vertical distribution is gaussian for the gas and the dust.
    The final velocity is the mass flux vertical average at each location.
    For more information check Garate et al. (2019), equations 31 - 35 in Appendix.

    ------------------------------

    This updater assigns:
    - the backreaction coefficients used for the gas calculations, accounting for dust vertical settling
    sim.dust.backreaction.A
    sim.dust.backreaction.B

    - the backreaction coefficients used for the dust calculations, accounting for dust vertical settling
    sim.dust.backreaction.A_dust_settling
    sim.dust.backreaction.B_dust_settling
    '''

    Nr = sim.grid.Nr[0]
    Nm = sim.grid.Nm[0]
    # Number of grid points for the vertical grid (locally defined)
    Nz = 300
    # Height of the first vertical gridcell (after the midplane). In Gas Scale Heights
    zmin = 1.e-5
    zmax = 10.0             # Height of the last vertical gridcell. In Gas Scale Heights
    # Set the last cell to the default values (A=1, B=0) for stability.
    OmitLastCell = True

    # Gas and dust scale heights
    h_g = sim.gas.Hp
    h_d = sim.dust.H

    # Midplane densities of the dust and gas
    rho_gas_midplane = sim.gas.rho[:, None]
    rho_dust_midplane = sim.dust.rho[:, :, None]

    # Stokes number
    St = sim.dust.St[:, :, None]

    # The vertical grid. Notice is defined locally.
    z = np.concatenate(([0.0], np.logspace(np.log10(zmin), np.log10(
        zmax), Nz-1, 10.)))[None, :] * h_g[:, None]  # dim: nr, nz

    # Vertical distribution for the gas and the dust
    exp_z_g = np.exp(-z**2. / (2.0 * h_g[:, None]**2.0))  # nr, nz
    exp_z_d = np.exp(-z[:, None, :]**2. /
                     (2.0 * h_d[:, :, None]**2.0))  # nr, nm, nz

    # Dust-to-Gas ratio at each radius, for every mass bin, at every height (nr, nm, nz)
    d2g_ratio = (rho_dust_midplane * exp_z_d) / \
        (rho_gas_midplane * exp_z_g)[:, None, :]

    # X, Y integral argument (at each radius and height) (nr, nm, nz).
    factor_xy = 1.0 + np.square(St)
    integral_X = (1.0 / factor_xy) * d2g_ratio
    integral_Y = (St / factor_xy) * d2g_ratio

    # Integral result X, Y obtained by summing over the mass axis (nr, nz)
    X = np.sum(integral_X, axis=1)
    Y = np.sum(integral_Y, axis=1)

    # Backreaction Coefficients A, B (nr, nz).
    factor_AB = np.square(Y) + np.square(1.0 + X)
    A_rz = (X + 1) / factor_AB
    B_rz = Y / factor_AB

    # At this point we have the backreaction coefficients A, B
    # Now we obtain the vertically averaged mass flux velocity for the gas and each dust species

    # Integrate over the vertical axis for the gas structure
    # Ag, Bg have dimension (nr)
    Ag = np.trapz(A_rz * exp_z_g, z, axis=1) * np.sqrt(2. / np.pi) / h_g
    Bg = np.trapz(B_rz * exp_z_g, z, axis=1) * np.sqrt(2. / np.pi) / h_g

    # Integrate over the vertical axis for each dust species structure
    # Ad, Bd have dimension (nr, nm)
    Ad = np.trapz(A_rz[:, None, :] * exp_z_d, z[:, None, :],
                  axis=2) * np.sqrt(2. / np.pi) / h_d
    Bd = np.trapz(B_rz[:, None, :] * exp_z_d, z[:, None, :],
                  axis=2) * np.sqrt(2. / np.pi) / h_d

    if OmitLastCell:
        Ag[-1] = 1.0
        Bg[-1] = 0.0
        Ad[-1, :] = 1.0
        Bd[-1, :] = 0.0

    # With the default parameters the integral slightly overestimates the coefficients.
    # For this reason is a good idea to the A coefficient to its maximum value of 1.0
    Ag[Ag > 1.0] = 1.0
    Ad[Ad > 1.0] = 1.0

    # Assign the backreaction coefficients for the gas and dust calculations
    sim.dust.backreaction.A = Ag            # Dimension (Nr)
    sim.dust.backreaction.B = Bg            # Dimension (Nr)

    sim.dust.backreaction.A_dust_settling = Ad   # Dimension (Nr, Nm)
    sim.dust.backreaction.B_dust_settling = Bd   # Dimension (Nr, Nm)


# We also need to ammend the functions for dust.v.rad, since they need a local-per-species value for gas.v.rad and dust.v.driftmax
def vrad_dust_BackreactionVerticalStructure(sim):
    St = sim.dust.St

    A = sim.dust.backreaction.A_dust_settling
    B = sim.dust.backreaction.B_dust_settling

    # Viscous velocity and pressure velocity
    vvisc = sim.gas.v.visc[:, None]
    vpres = (sim.gas.eta * sim.grid.r * sim.grid.OmegaK)[:, None]

    # Radial gas velocity and the maximum drift velocity, following (Garate et al., 2020. Eqs. 14, 15)
    vgas_rad = A * vvisc + 2. * B * vpres
    vdrift_max = 0.5 * B * vvisc - A * vpres

    return (vgas_rad + 2. * vdrift_max * St) / (1. + St**2.)
