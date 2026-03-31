import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from Project_Tools.attitude_error_eval import attitude_error_eval
from Project_Tools.nadir_frame_dcm import nadir_frame_dcm, nadir_frame_angular_velocity
from Project_Tools.GMO_pointing_frame_funcs import GMO_pointing_frame_dcm, GMO_pointing_frame_angular_velocity
from Project_Tools.sun_frame_funcs import sun_frame_dcm, sun_frame_angular_velocity
from Testing_Scripts.initial_conditions import sigma_0_LMO, omega_0_LMO
import numpy as np

# Convert the initial angular velocity from degrees per second to radians per second for use in the attitude error evaluation function
omega_0_LMO_rad = np.deg2rad(omega_0_LMO)

# Define time to test the function at
t = 0

# Get the DCM from inertial to each frame at time t
DCM_sun = sun_frame_dcm(t)
breakpoint()
DCM_nadir = nadir_frame_dcm(t)
DCM_GMO = GMO_pointing_frame_dcm(t)

# Get the angular velocity of each frame at time t
omega_sun = sun_frame_angular_velocity(t)
omega_nadir = nadir_frame_angular_velocity(t)
omega_GMO = GMO_pointing_frame_angular_velocity(t)

# Evaluate the attitude error of the spacecraft relative to the sun frame, nadir frame, and GMO pointing frame at time t
attitude_error_sun_RN, angular_rate_error_sun_RN = attitude_error_eval(t, sigma_0_LMO, omega_0_LMO_rad, DCM_sun, omega_sun)
attitude_error_nadir_RN, angular_rate_error_nadir_RN = attitude_error_eval(t, sigma_0_LMO, omega_0_LMO_rad, DCM_nadir, omega_nadir)
attitude_error_GMO_RN, angular_rate_error_GMO_RN = attitude_error_eval(t, sigma_0_LMO, omega_0_LMO_rad, DCM_GMO, omega_GMO)

# Print the results
print("Attitude error relative to sun frame at t =", t, "seconds:", attitude_error_sun_RN)
print("Angular velocity error relative to sun frame at t =", t, "seconds (rad per second):", angular_rate_error_sun_RN)
print("Attitude error relative to nadir frame at t =", t, "seconds:", attitude_error_nadir_RN)
print("Angular velocity error relative to nadir frame at t =", t, "seconds (rad per second):", angular_rate_error_nadir_RN)
print("Attitude error relative to GMO pointing frame at t =", t, "seconds:", attitude_error_GMO_RN)
print("Angular velocity error relative to GMO pointing frame at t =", t, "seconds (rad per second):", angular_rate_error_GMO_RN)

# Write the results to text files in the coursera_validation_files directory
with open(os.path.join(os.path.dirname(__file__), '..', 'coursera_validation_files', 'attitude_error_sun_RN.txt'), 'w') as f:
    f.write(' '.join(str(x) for x in attitude_error_sun_RN.flatten()))
with open(os.path.join(os.path.dirname(__file__), '..', 'coursera_validation_files', 'angular_rate_error_sun_RN.txt'), 'w') as f:
    f.write(' '.join(str(x) for x in angular_rate_error_sun_RN.flatten()))
with open(os.path.join(os.path.dirname(__file__), '..', 'coursera_validation_files', 'attitude_error_nadir_RN.txt'), 'w') as f:
    f.write(' '.join(str(x) for x in attitude_error_nadir_RN.flatten()))
with open(os.path.join(os.path.dirname(__file__), '..', 'coursera_validation_files', 'angular_rate_error_nadir_RN.txt'), 'w') as f:
    f.write(' '.join(str(x) for x in angular_rate_error_nadir_RN.flatten()))
with open(os.path.join(os.path.dirname(__file__), '..', 'coursera_validation_files', 'attitude_error_GMO_RN.txt'), 'w') as f:
    f.write(' '.join(str(x) for x in attitude_error_GMO_RN.flatten()))
with open(os.path.join(os.path.dirname(__file__), '..', 'coursera_validation_files', 'angular_rate_error_GMO_RN.txt'), 'w') as f:
    f.write(' '.join(str(x) for x in angular_rate_error_GMO_RN.flatten()))