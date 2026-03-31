import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from Project_Tools.nadir_frame_dcm import nadir_frame_dcm
from Project_Tools.hill_frame_funcs import hill_frame_dcm, hill_frame_angular_velocity
from initial_conditions import ta_dot_LMO
import numpy as np

# Define time to test the function at
t = 330

# Test the function at the specified time
DCM = nadir_frame_dcm(t)

# Print the results
print("DCM from inertial frame to nadir frame at t =", t, "seconds:")
print(DCM)

# Define the angular velocity of the frame relative to the inertial frame in the nadir frame
# The angular velocity of the nadir frame is directly related to the angular velocity of the Hill frame, which is equal to the rate of change of the true anomaly (ta_dot_LMO) in this case
omega_inertial = hill_frame_angular_velocity(t)

# print the angular velocity
print("Angular velocity of the nadir frame relative to the inertial frame (in nadir frame):", omega_inertial)

# Write the result to a text file in the coursera_validation_files directory
with open(os.path.join(os.path.dirname(__file__), '..', 'coursera_validation_files', 'nadir_frame_dcm.txt'), 'w') as f:
    f.write(' '.join(str(x) for x in DCM.flatten()))

with open(os.path.join(os.path.dirname(__file__), '..', 'coursera_validation_files', 'nadir_frame_angular_velocity.txt'), 'w') as f:
    f.write(' '.join(str(x) for x in omega_inertial.flatten()))