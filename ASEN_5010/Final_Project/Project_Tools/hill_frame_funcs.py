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

def hill_frame_angular_velocity(time : float, delta_t : float = 1e-8):
    """
    This function calculates the angular velocity of the LMO Hill frame relative to the inertial frame at a given time.
    This is done using numerical differencing to compute RN/delta_t, where RN is the DCM from inertial to Hill frame.

    Parameters
    ----------
    time : float
        The time at which to calculate the angular velocity in seconds.
    delta_t : float, optional
        The time step to use for numerical differencing in seconds. Default is 1e-8 seconds.

    Returns
    -------
    omega : numpy array
        The angular velocity of the Hill frame relative to the inertial frame in the Hill frame.
    """

    omega_H = np.array([0, 0, np.deg2rad(ta_dot_LMO)])  # Assuming the nadir frame rotates about the z-axis of the Hill frame

    # Convert the angular velocity to the inertial frame using the DCM
    HN = hill_frame_dcm(time)  # DCM from inertial to Hill frame
    omega = HN.T @ omega_H  # Convert angular velocity to inertial frame

    return omega