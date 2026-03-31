import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from Project_Tools.attitude_error_eval import attitude_error_eval
from Project_Tools.nadir_frame_dcm import nadir_frame_dcm
from Project_Tools.GMO_pointing_frame_funcs import GMO_pointing_frame_dcm, GMO_pointing_frame_angular_velocity
from ASEN_5010.Final_Project.Project_Tools.sun_frame_funcs import sun_frame_dcm, sun_frame_angular_velocity
from Testing_Scripts.initial_conditions import sigma_0_LMO, omega_0_LMO

# Define time to test the function at
t = 0

# Get the DCM from inertial to each frame at time t
DCM_sun = sun_frame_dcm(t)
DCM_RN = nadir_frame_dcm(t)
DCM_GMO = GMO_pointing_frame_dcm(t)

# Get the angular velocity of each frame at time t
omega_sun = sun_frame_angular_velocity(t)
omega_GMO = GMO_pointing_frame_angular_velocity(t)


