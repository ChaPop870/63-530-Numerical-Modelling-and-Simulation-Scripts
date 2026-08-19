# Description of `aej.py`

This module creates a simple model of the African Easterly Jet. It takes the temperature field in y and z and evolves it
with various sources, sinks, and diffusion. The module outputs
- Plot of the evolved temperature field after the given number of days.
- The meridional temperature gradient.
- The thermal wind shear.
- The geostrophic wind diagnosed from the temperature gradient.
- The difference between the diffusion case and no diffusion case if the last part of main is uncommented.

## The default model parameters are:
n_days = 60: 

- Sets the number of days for the model to run.

tau = 20 * 24 * 3600 # Seconds (Characteristic forcing timescale).
- Define the timescale for Newtonian Cooling.

tau_convection = 3 * 24 * 3600
- Simulates an African Easterly Wave with convection over 3-days. 

day_number = -1      
- Can be used to plot a given day of the simulation.
- The default value sets the last day which is 60 by default.

lapse_rate = 0.0065

### Forcing parameters.
sahara_heating = 7.0
- Amplitude of Sahara heating.

comp_cooling = -2.5
- Amplitude of compensating cooling over Sahara.

gulf_cooling = -0.5
- Amplitude of cooling over Gulf and rainforest.

convection = 0.025      # Default 0.0
- Use the given value to simulate convection.
- The default value is 0.


cooling = 0.017         # Default 0.0
- Use the given value to simulate evaporative cooling from lower troposphere during convection.
- The default value is 0.

sahara_start = 15.0     
- This is the location where the sigmoid function used to define the Saharan Heating has the largest gradient.

gulf_cooling_start = 5.0
- This sets the location of the Gulf and rainforest cooling.


kappa_y = 200.0  # m² s⁻¹  # Default 200.0
- To turn off diffusion, set this value to 0.


### Define the temporal grid
tmin = 0.0                  # Initial time

tmax = n_days * 86_400.0    # Final time

tcheck = 24 * 3600          # Time interval to checkpoint data


## Additional Info
If you wish to create a plot of the difference in diffusion, uncomment the code in the bottom of the `main()` function.


# Description of `validation.py`
This module plots and evaluates the performance of the numerical schemes used in `aej.py`.

# Description of `perturbation_exp.py`
This module was created to quickly test the jet adjustment experiment detailed in the 5th question from the report.

# Description of `total_derivatives.py`
This modules contains the spatial discretizations developed over the semester. It includes biased finite-differencing schemes,
periodic finite differencing schemes and von Neumann schemes.

# Description of `time_marching_schemes.py`
This module contains the time marching schemes developed across the semester and also a solver.

# Description of `partial_derivatives.py`
This module defines a class which can calculate partial derivatives of functions of 2 variables using
the discretizations developed in `total_derivatives.py`.

# Description of `checkpointing.py`
This module essentially a clone of `checkpointing.py` from Juan Pedro's repo.