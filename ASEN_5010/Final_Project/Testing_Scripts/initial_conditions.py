import numpy as np

"""
This file simply stores the initial conditions used for testing the functions in the project.

It is not meant to be run as a script, but rather to be imported by the testing scripts to access
the initial conditions. The initial conditions are defined as variables that can be easily accessed
by the testing scripts.
"""

# Define gravitational parameter for Mars
mu = 42828.3

# Define initial conditions for LMO and GMO
r_LMO = 400 + 3396.19
raan_LMO = 20
inc_LMO = 30
ta_0_LMO = 60
ta_dot_LMO = np.rad2deg(np.sqrt(mu / r_LMO**3))

r_GMO = 20424.2
raan_GMO = 0
inc_GMO = 0
ta_0_GMO = 250
ta_dot_GMO = np.rad2deg(np.sqrt(mu / r_GMO**3))