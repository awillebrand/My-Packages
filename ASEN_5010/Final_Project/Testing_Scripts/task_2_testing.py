import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from ASEN_5010.Final_Project.Project_Tools.hill_frame_funcs import hill_frame_dcm
from initial_conditions import mu, r_LMO, raan_LMO, inc_LMO, ta_0_LMO, ta_dot_LMO, r_GMO, raan_GMO, inc_GMO, ta_0_GMO, ta_dot_GMO

# Define time to test the function at
t = 300

# Test the function at the specified time
DCM = hill_frame_dcm(t)

# Print the results
print("DCM from inertial frame to Hill frame at t =", t, "seconds:")
print(DCM)

# Write the result to a text file in the coursera_validation_files directory
with open(os.path.join(os.path.dirname(__file__), '..', 'coursera_validation_files', 'hill_frame_dcm.txt'), 'w') as f:
        f.write(' '.join(str(x) for x in DCM.flatten()))