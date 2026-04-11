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

    # Compute the MRP attitude error from the DCM_BR by first computing the Euler Parameters using Shepard's method and then converting the PRPs to MRPs
    test_B0 = 0.25 * (1 + np.trace(DCM_BR))
    test_B1 = 0.25 * (1 + 2 * DCM_BR[0, 0] - np.trace(DCM_BR))
    test_B2 = 0.25 * (1 + 2 * DCM_BR[1, 1] - np.trace(DCM_BR))
    test_B3 = 0.25 * (1 + 2 * DCM_BR[2, 2] - np.trace(DCM_BR))

    B = np.array([test_B0, test_B1, test_B2, test_B3])
    max_index = np.argmax(B)

    if max_index == 0:
        B0 = np.sqrt(test_B0);
        B1 = 0.25 * (DCM_BR[1, 2] - DCM_BR[2, 1]) / B0;
        B2 = 0.25 * (DCM_BR[2, 0] - DCM_BR[0, 2]) / B0;
        B3 = 0.25 * (DCM_BR[0, 1] - DCM_BR[1, 0]) / B0;
    elif max_index == 1:
        B1 = np.sqrt(test_B1);
        B0 = 0.25 * (DCM_BR[1, 2] - DCM_BR[2, 1]) / B1;
        B2 = 0.25 * (DCM_BR[0, 1] + DCM_BR[1, 0]) / B1;
        B3 = 0.25 * (DCM_BR[0, 2] + DCM_BR[2, 0]) / B1;
    elif max_index == 2:
        B2 = np.sqrt(test_B2);
        B0 = 0.25 * (DCM_BR[2, 0] - DCM_BR[0, 2]) / B2;
        B1 = 0.25 * (DCM_BR[0, 1] + DCM_BR[1, 0]) / B2;
        B3 = 0.25 * (DCM_BR[1, 2] + DCM_BR[2, 1]) / B2;
    else:
        B3 = np.sqrt(test_B3);
        B0 = 0.25 * (DCM_BR[0, 1] - DCM_BR[1, 0]) / B3;
        B1 = 0.25 * (DCM_BR[0, 2] + DCM_BR[2, 0]) / B3;
        B2 = 0.25 * (DCM_BR[1, 2] + DCM_BR[2, 1]) / B3;

    # Convert the Euler Parameters to MRPs
    sigma_BR = np.array([B1, B2, B3]) / (1 + B0)

    # Compute the angular velocity of the spacecraft relative to the reference frame in inertial coordinates
    omega_RN_body_rad = DCM_BN @ omega_RN
    omega_BR = omega_BN - omega_RN_body_rad

    # Check that the attitude error is less than 180 degrees (i.e. the norm of the MRP attitude error is less than 1)
    if np.linalg.norm(sigma_BR) > 1:
        # If so, convert to the shadow set of MRPs to get the equivalent attitude error that is less than 180 degrees
        sigma_BR = -sigma_BR / np.dot(sigma_BR, sigma_BR)

    return sigma_BR, omega_BR
    


