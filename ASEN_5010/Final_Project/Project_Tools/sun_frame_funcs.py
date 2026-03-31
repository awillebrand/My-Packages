import numpy as np

def sun_frame_dcm(time : float):
    """
    This function calculates the DCM from the inertial frame to the Sun frame at a given time.

    Parameters
    ----------
    time : float
        The time at which to calculate the DCM in seconds. Does not affect the result since the Sun frame is fixed in inertial space, but is included for consistency with the other frame DCM functions.

    Returns
    -------
    DCM : numpy array
        The direction cosine matrix from the inertial frame to the Sun frame.
    """

    # This frame is defined such that the 3rd axis points towards the Sun, which as defined by the problem points along the n2 inertial axis.
    # The 1st axis is defined to point in the -n1 inertial direction, and the 2nd axis is defined to complete the right-handed frame.
    # This is a simple rotation so directly construct the DCM from inertial to Sun frame

    DCM = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])

    return DCM

def sun_frame_angular_velocity(time : float):
    """
    This function calculates the angular velocity of the Sun frame relative to the inertial frame at a given time.
    Since the Sun frame is fixed in inertial space, the angular velocity is zero.

    Parameters
    ----------
    time : float
        The time at which to calculate the angular velocity in seconds. Does not affect the result since the Sun frame is fixed in inertial space, but is included for consistency with the other frame DCM functions.

    Returns
    -------
    omega : numpy array
        The angular velocity of the Sun frame relative to the inertial frame in the Sun frame (which is zero).
    """

    omega = np.array([0, 0, 0])  # Sun frame is fixed in inertial space, so angular velocity is zero

    return omega