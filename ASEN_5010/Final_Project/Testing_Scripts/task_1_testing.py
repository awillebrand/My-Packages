import numpy as np
from Project_Tools.get_orbit_state import get_orbit_state

# Define gravitational parameter for Mars
mu_mars = 42828.3

# Define times to test the function at
t_LMO = 450
t_GMO = 1150

# Define initial conditions for LMO and GMO
r_LMO = 400 + 3396.19
raan_LMO = 20
inc_LMO = 30
ta_LMO = 60
ta_dot_LMO = np.sqrt(mu_mars / r_LMO**3)

r_GMO = 20424.2
raan_GMO = 0
inc_GMO = 0
ta_GMO = 250
ta_dot_GMO = np.sqrt(mu_mars / r_GMO**3)

# Get the state for LMO and GMO at the specified times
pos_LMO, vel_LMO = get_orbit_state(r_LMO, raan_LMO, inc_LMO, ta_LMO + ta_dot_LMO * t_LMO)
pos_GMO, vel_GMO = get_orbit_state(r_GMO, raan_GMO, inc_GMO, ta_GMO + ta_dot_GMO * t_GMO)

# Print the results
print("LMO State at t =", t_LMO, "seconds:")
print("Position (km):", pos_LMO)
print("Velocity (km/s):", vel_LMO)

print("\nGMO State at t =", t_GMO, "seconds:")
print("Position (km):", pos_GMO)
print("Velocity (km/s):", vel_GMO)

