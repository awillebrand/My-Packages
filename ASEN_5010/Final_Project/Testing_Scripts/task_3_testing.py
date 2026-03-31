import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from Project_Tools.sun_frame_funcs import sun_frame_dcm, sun_frame_angular_velocity

# Define time to test the function at
t = 0

# Test the function at the specified time
DCM = sun_frame_dcm(t)

# Print the results
print("DCM from inertial frame to Sun frame at t =", t, "seconds:")
print(DCM)

# Define the angular velocity of the frame relative to the inertial frame in the Sun frame
omega = sun_frame_angular_velocity(t)

# print the angular velocity
print("Angular velocity of the Sun frame relative to the inertial frame (in Sun frame):", omega)

# Write the result to a text file in the coursera_validation_files directory
with open(os.path.join(os.path.dirname(__file__), '..', 'coursera_validation_files', 'sun_frame_dcm.txt'), 'w') as f:
        f.write(' '.join(str(x) for x in DCM.flatten()))

with open(os.path.join(os.path.dirname(__file__), '..', 'coursera_validation_files', 'sun_frame_angular_velocity.txt'), 'w') as f:
        f.write(' '.join(str(x) for x in omega.flatten()))