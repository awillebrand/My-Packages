import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from Project_Tools.get_orbit_state import get_orbit_state
from Testing_Scripts.initial_conditions import r_LMO, raan_LMO, inc_LMO, ta_0_LMO, ta_dot_LMO, r_GMO, raan_GMO, inc_GMO, ta_0_GMO, ta_dot_GMO
import numpy as np

def GMO_pointing_frame_dcm(time : float):
    """
    This function calculates the DCM from the inertial frame to the LMO to GMO pointing frame at a given time.

    Parameters
    ----------
    time : float
        The time at which to calculate the DCM in seconds.

    Returns
    -------
    DCM : numpy array
        The direction cosine matrix from the inertial frame to the GMO pointing frame.
    """

    # Get the state both spacecraft at the given time
    pos_LMO, vel_LMO = get_orbit_state(r_LMO, raan_LMO, inc_LMO, ta_0_LMO + ta_dot_LMO * time)
    pos_GMO, vel_GMO = get_orbit_state(r_GMO, raan_GMO, inc_GMO, ta_0_GMO + ta_dot_GMO * time)

    # Determine the vector from LMO to GMO
    delta_r = pos_GMO - pos_LMO

    # delta_r is alined with the negative r_1 axis of the pointing frame
    r_1 = -delta_r / np.linalg.norm(delta_r)

    # The r_2 axis is defined as the cross product of delta_r and and the inertial n_3 axis (which is [0, 0, 1])
    n_3 = np.array([0, 0, 1])
    r_2 = np.cross(delta_r, n_3)
    r_2 = r_2 / np.linalg.norm(r_2)

    # The r_3 axis is defined to complete the right-handed frame
    r_3 = np.cross(r_1, r_2)

    # Construct the DCM from inertial to pointing frame using the r_1, r_2, and r_3 axes as columns
    DCM = np.array([r_1, r_2, r_3])

    return DCM

def GMO_pointing_frame_angular_velocity(time : float, delta_t : float = 1e-8):
    """
    This function calculates the angular velocity of the LMO to GMO pointing frame relative to the inertial frame at a given time.
    This is done using numerical differencing to compute RN/delta_t, where RN is the DCM from inertial to pointing frame.

    Parameters
    ----------
    time : float
        The time at which to calculate the angular velocity in seconds.
    delta_t : float, optional
        The time step to use for numerical differencing in seconds. Default is 1e-9 seconds.
    Returns
    -------
    omega : numpy array
        The angular velocity of the pointing frame relative to the inertial frame in the pointing frame.
    """

    # Get the DCM from inertial to pointing frame at time and time + delta_t
    DCM_t = GMO_pointing_frame_dcm(time)
    DCM_t_plus_dt = GMO_pointing_frame_dcm(time + delta_t)

    # Compute the time derivative of the DCM using numerical differencing
    DCM_dot = (DCM_t_plus_dt - DCM_t) / delta_t

    # Compute the angular velocity using the relationship DCM_dot = -omega_tilde @ DCM, where omega_tilde is the skew-symmetric matrix of omega
    omega_tilde = -DCM_t.T @ DCM_dot
    breakpoint()
    # Extract the angular velocity vector from the skew-symmetric matrix
    omega = np.array([-omega_tilde[1, 2], omega_tilde[0, 2], -omega_tilde[0, 1]])

    return omega