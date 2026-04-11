import numpy as np

def mrp_to_dcm(sigma):
    """
    This function converts a Modified Rodrigues Parameter (MRP) attitude representation to a Direction Cosine Matrix (DCM).

    Parameters
    ----------
    sigma : numpy array
        The MRP attitude representation of the spacecraft relative to the inertial frame at the given time.

    Returns
    -------
    DCM : numpy array
        The direction cosine matrix from the inertial frame to the body frame.
    """

    sigma_tilde = np.array([[0, -sigma[2], sigma[1]], [sigma[2], 0, -sigma[0]], [-sigma[1], sigma[0], 0]])
    DCM = np.eye(3) + (8 * sigma_tilde @ sigma_tilde - 4 * (1 - np.dot(sigma, sigma)) * sigma_tilde) / (1 + np.dot(sigma, sigma))**2

    return DCM