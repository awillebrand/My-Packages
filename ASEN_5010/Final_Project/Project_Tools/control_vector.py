import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np

def control_vector(P, K, sigma_BR, omega_BR):
    """
    This function computes the control vector required for pointing control given input gains and current state error.
    Parameters
    ----------
    P : float
        The proportional gain for the control system.
    K : float
        The derivative gain for the control system.
    sigma_BR : numpy array
        The MRP attitude error of the spacecraft relative to the reference frame in degrees.
    omega_BR : numpy array
        The angular velocity error of the spacecraft relative to the reference frame in rad per second.
    """

    control_vector = -K * sigma_BR - P * omega_BR

    return control_vector

def compute_gains(I_mat : np.ndarray, decay_time_constant : float):
    """
    This function computes the P and K scalar gains for the control system based on the inertia matrix and desired decay time constant and damping ratio. Critical damping is only applied to the smallest MoI.

    Parameters
    ----------
    I_mat : numpy array
        The inertia matrix of the spacecraft.
    decay_time_constant : float
        The time constant for the exponential decay of the control vector in seconds.
    xi : float
        The damping ratio for the control system.
    """

    # Identify the smallest moment of inertia
    I_min = np.min(np.diag(I_mat))
    I_max = np.max(np.diag(I_mat))

    # Compute P gain for the i-th axis
    P = 2 * I_max / decay_time_constant
    
    # Compute K gain for the i-th axis
    K = P**2 / I_min

    return P, K