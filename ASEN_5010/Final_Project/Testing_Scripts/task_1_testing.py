import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from Project_Tools.get_orbit_state import get_orbit_state
from initial_conditions import mu, r_LMO, raan_LMO, inc_LMO, ta_0_LMO, ta_dot_LMO, r_GMO, raan_GMO, inc_GMO, ta_0_GMO, ta_dot_GMO

# Define times to test the function at
t_LMO = 450
t_GMO = 1150

# Get the state for LMO and GMO at the specified times
pos_LMO, vel_LMO = get_orbit_state(r_LMO, raan_LMO, inc_LMO, ta_0_LMO + ta_dot_LMO * t_LMO)
pos_GMO, vel_GMO = get_orbit_state(r_GMO, raan_GMO, inc_GMO, ta_0_GMO + ta_dot_GMO * t_GMO)

# Print the results
print("LMO State at t =", t_LMO, "seconds:")
print("Position (km):", pos_LMO)
print("Velocity (km/s):", vel_LMO)

print("\nGMO State at t =", t_GMO, "seconds:")
print("Position (km):", pos_GMO)
print("Velocity (km/s):", vel_GMO)

# write each result to a text file in the coursera_validation_files directory
with open(os.path.join(os.path.dirname(__file__), '..', 'coursera_validation_files', 'LMO_pos.txt'), 'w') as f:
    f.write(str(pos_LMO))
with open(os.path.join(os.path.dirname(__file__), '..', 'coursera_validation_files', 'LMO_vel.txt'), 'w') as f:
    f.write(str(vel_LMO))
with open(os.path.join(os.path.dirname(__file__), '..', 'coursera_validation_files', 'GMO_pos.txt'), 'w') as f:
    f.write(str(pos_GMO))
with open(os.path.join(os.path.dirname(__file__), '..', 'coursera_validation_files', 'GMO_vel.txt'), 'w') as f:
    f.write(str(vel_GMO))