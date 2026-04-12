import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from Testing_Scripts.initial_conditions import mu, r_LMO, raan_LMO, inc_LMO, ta_0_LMO, ta_dot_LMO
from Project_Tools.get_orbit_state import get_orbit_state

def hill_frame_dcm(time):
    """
    This function calculates the DCM from the inertial frame to the LMO Hill frame at a given time.

    Parameters
    ----------
    time : float
        The time at which to calculate the DCM in seconds.

    Returns
    -------
    DCM : numpy array
        The direction cosine matrix from the inertial frame to the Hill frame.
    """

    # Get the state of the spacecraft in LMO at the given time
    pos, vel = get_orbit_state(r_LMO, raan_LMO, inc_LMO, ta_0_LMO + ta_dot_LMO * time)

    # Calculate the unit vectors for the Hill frame
    r_hat = pos / np.linalg.norm(pos)
    h_vec = np.cross(pos, vel)
    h_hat = h_vec / np.linalg.norm(h_vec)
    theta_hat = np.cross(h_hat, r_hat)

    # Construct the DCM from inertial to Hill frame
    DCM = np.array([r_hat, theta_hat, h_hat])

    return DCM