import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from Project_Tools.hill_frame_funcs import hill_frame_dcm
import numpy as np

def nadir_frame_dcm(time):
    """
    This function calculates the DCM from the inertial frame to the LMO nadir frame at a given time.

    Parameters
    ----------
    time : float
        The time at which to calculate the DCM in seconds.

    Returns
    -------
    DCM : numpy array
        The direction cosine matrix from the inertial frame to the nadir frame.
    """

    # Get the DCM from inertial to Hill frame
    DCM_hill = hill_frame_dcm(time)

    # The nadir frame is defined such that the 1st axis points towards the center of the planet,
    # which is the opposite direction of the 1st axis of the Hill frame, the 2nd axis is the same
    # as the 2nd axis of the Hill frame, and the 3rd axis is defined to complete the right-handed frame.

    R_nadir_hill = np.array([[-1, 0, 0],
                             [0, 1, 0],
                             [0, 0, -1]])

    # Calculate the DCM from inertial to nadir frame
    DCM_nadir = R_nadir_hill @ DCM_hill

    return DCM_nadir