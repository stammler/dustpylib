import numpy as np


from dustpylib.dynamics.backreaction.functions_backreaction import BackreactionCoefficients
from dustpylib.dynamics.backreaction.functions_backreaction import BackreactionCoefficients_VerticalStructure
from dustpylib.dynamics.backreaction.functions_backreaction import vrad_dust_BackreactionVerticalStructure
from dustpylib.dynamics.backreaction.functions_backreaction import dustDiffusivity_Backreaction

################################
# Helper routine to add backreaction to your Simulation object in one line.
################################
def setup_backreaction(sim, vertical_setup = False):
    '''
    Add the backreaction setup to your simulation object.
    Call the backreaction setup function after the initialization and then run, as follows:

    sim.initialize()
    setup_backreaction(sim)
    sim.run()

    ----------------------------------------------
    vertical_setup [Bool]: If true, the vertical structure of the gas and dust component is considered to weight the effect of the backreaction coefficients.
    '''

    print("Setting up the backreaction module.")
    print("Please cite the work of Garate(2019, 2020).")

    if vertical_setup:
        # Include additional back-reaction coefficients for the dust accounting for vertical settling
        sim.dust.backreaction.addfield("A_vertical", np.ones_like(sim.dust.a) ,  description = "Backreaction Coefficient A, considering dust vertical settling")
        sim.dust.backreaction.addfield("B_vertical", np.zeros_like(sim.dust.a),  description = "Backreaction Coefficient B, considering dust vertical settling")


        # Instead of assigning an update order to the backreaction Group
        # we perform and assign the backreaction coefficient calculations and assign them within the Group updater
        sim.dust.backreaction.updater = BackreactionCoefficients_VerticalStructure

        # Redefine the radial dust velocity to consider one pair of backreaction coefficients per dust species
        sim.dust.v.rad.updater = vrad_dust_BackreactionVerticalStructure

    else:
        # Set the backreaction coefficients with the standard setup
        sim.dust.backreaction.updater = BackreactionCoefficients


    # Update the dust diffusivity to account for high dust-to-gas ratios
    sim.dust.D.updater = dustDiffusivity_Backreaction


    # Update all
    sim.update()
    sim.gas.v.rad.update()
    sim.dust.v.rad.update()
