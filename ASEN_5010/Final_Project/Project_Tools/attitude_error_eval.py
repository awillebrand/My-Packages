import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from Project_Tools.mrp_dcm import mrp_to_dcm

def attitude_error_eval(time : float, sigma_BN : np.ndarray, omega_BN : np.ndarray, DCM_RN : np.ndarray, omega_RN : np.ndarray):
    """
    This function evaluates the attitude error of the spacecraft at a given time by comparing the provided attitude (sigma_BN and omega_BN) to the expected attitude based on the provided DCM_RN and omega_RN.

    Parameters
    ----------
    time : float
        The time at which to evaluate the attitude error in seconds.
    sigma_BN : numpy array
        The MRP attitude of the spacecraft relative to the inertial frame at the given time.
    omega_BN : numpy array
        The angular velocity of the spacecraft relative to the inertial frame expressed in the body frame at the given time in rad per second.
    DCM_RN : numpy array
        The direction cosine matrix from the inertial frame to the reference frame (GMO pointing frame) at the given time.
    omega_RN : numpy array
        The angular velocity of the reference frame (GMO pointing frame) relative to the inertial frame expressed in inertial coordinates at the given time in rad per second.

    Returns
    -------
    sigma_BR : float
        The MRP attitude error of the spacecraft relative to the reference frame in degrees.
    omega_BR : float
        The angular velocity error of the spacecraft relative to the reference frame in rad per second.
    """

    # Convert the MRP attitude to a DCM
    DCM_BN = mrp_to_dcm(sigma_BN)

    # Compute the DCM from the body frame to the reference frame
    DCM_BR = DCM_BN @ DCM_RN.T

    zeta = np.sqrt(np.linalg.trace(DCM_BR) + 1)

    # Convert the MRPs
    vector_part = np.array([DCM_BR[1, 2] - DCM_BR[2, 1], DCM_BR[2, 0] - DCM_BR[0, 2], DCM_BR[0, 1] - DCM_BR[1, 0]])
    sigma_BR = vector_part / (zeta * (zeta + 2))
    
    if np.linalg.norm(sigma_BR) > 1:
        sigma_BR = -sigma_BR / np.dot(sigma_BR, sigma_BR)

    # Compute the angular velocity of the spacecraft relative to the reference frame in inertial coordinates
    omega_RN_body_rad = DCM_BN @ omega_RN
    omega_BR = omega_BN - omega_RN_body_rad

    # Check that the attitude error is less than 180 degrees (i.e. the norm of the MRP attitude error is less than 1)
    if np.linalg.norm(sigma_BR) > 1:
        # If so, convert to the shadow set of MRPs to get the equivalent attitude error that is less than 180 degrees
        sigma_BR = -sigma_BR / np.dot(sigma_BR, sigma_BR)

    return sigma_BR, omega_BR
    


