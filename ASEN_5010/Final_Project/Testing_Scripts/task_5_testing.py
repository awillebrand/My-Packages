import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from Project_Tools.GMO_pointing_frame_funcs import GMO_pointing_frame_dcm, GMO_pointing_frame_angular_velocity
import numpy as np

# Define time to test the function at
t = 330

# Test the functions at the specified time
DCM = GMO_pointing_frame_dcm(t)
omega = GMO_pointing_frame_angular_velocity(t)

# Print the results
print("DCM from inertial frame to GMO pointing frame at t =", t, "seconds:")
print(DCM)
print("Angular velocity of the GMO pointing frame relative to the inertial frame (in pointing frame):", omega)

# Write the results to text files in the coursera_validation_files directory
with open(os.path.join(os.path.dirname(__file__), '..', 'coursera_validation_files', 'GMO_pointing_frame_dcm.txt'), 'w') as f:
    f.write(' '.join(str(x) for x in DCM.flatten()))

with open(os.path.join(os.path.dirname(__file__), '..', 'coursera_validation_files', 'GMO_pointing_frame_angular_velocity.txt'), 'w') as f:
    f.write(' '.join(str(x) for x in omega.flatten()))