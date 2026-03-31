import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from Project_Tools.attitude_error_eval import attitude_error_eval
from Project_Tools.nadir_frame_dcm import nadir_frame_dcm
from Project_Tools.GMO_pointing_frame_funcs import GMO_pointing_frame_dcm, GMO_pointing_frame_angular_velocity
from ASEN_5010.Final_Project.Project_Tools.hill_frame_funcs import hill_frame_dcm
from ASEN_5010.Final_Project.Project_Tools.sun_frame_funcs import sun_frame_dcm
from Testing_Scripts.initial_conditions import sigma_0_LMO, omega_0_LMO

# Define time to test the function at
t = 0

# Get the DCM from inertial to nadir frame at the specified time
DCM_RN = nadir_frame_dcm(t)

# 
