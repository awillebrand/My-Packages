import numpy as np
from Project_Tools.control_vector import control_vector
from Project_Tools.attitude_error_eval import attitude_error_eval
from Project_Tools.sun_frame_funcs import sun_frame_dcm, sun_frame_angular_velocity
from Project_Tools.nadir_frame_dcm import nadir_frame_dcm, nadir_frame_angular_velocity
from Project_Tools.GMO_pointing_frame_funcs import GMO_pointing_frame_dcm, GMO_pointing_frame_angular_velocity

"""
Runge-Kutta 4th order method for solving ordinary differential equations (ODEs).
"""
def rk4(y0, t, I, pointing_mode, P, K):
    """
    Perform the RK4 integration.

    Parameters:
    func : function
        The function that returns the derivative of y at time t.
    y0 : np.array
        Initial condition (value of y at t[0]).
    t : np.array
        Array of time points where the solution is computed.
    I : np.array
        Inertia matrix of the spacecraft (assumed to be constant for this implementation).
    pointing_mode : str
        The pointing mode for the control system (e.g., 'GMO', 'Nadir', etc.) which may affect the control input.
    P : np.array
        The P gain matrix for the control system.
    K : np.array
        The K gain matrix for the control system.

    Returns:
    y : np.array
        Array of solution values corresponding to each time point in t.
    """
    if pointing_mode.lower() not in ['gmo', 'nadir', 'sun']:
        raise ValueError("Invalid pointing mode. Must be 'GMO', 'Nadir', or 'Sun'.")

    n = len(t)
    y = [y0]
    
    for i in range(1, n):
        dt = t[i] - t[i-1]
        k1 = eom(y[-1], t[i-1], I, pointing_mode, P, K)
        k2 = eom(y[-1] + 0.5 * dt * k1, t[i-1] + 0.5 * dt, I, pointing_mode, P, K)
        k3 = eom(y[-1] + 0.5 * dt * k2, t[i-1] + 0.5 * dt, I, pointing_mode, P, K)
        k4 = eom(y[-1] + dt * k3, t[i-1] + dt, I, pointing_mode, P, K)
        
        y_next = y[-1] + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        sigma = y_next[0:3]

        if np.dot(sigma, sigma) > 1.0:
            y_next[0:3] = -sigma / np.dot(sigma, sigma)

        y.append(y_next)
    
    # Convert to numpy array for easier handling
    y = np.array(y)

    return y

def eom(state, t, I, pointing_mode, P, K):
    """
    Equations of motion for MRP attitude dynamics.
    Parameters:
    t : float
        Current time (not used in this case since the equations are time-invariant).
    state : np.array
        Current state vector [sigma1, sigma2, sigma3, omega1, omega2, omega3].
    u : np.array
        Control input.
    I : np.array
        Inertia matrix of the spacecraft.
    """

    # Unpack the state vector
    sigma = state[0:3]  # MRP vector
    omega = state[3:6]  # Angular velocity vector

    # Determine which reference frame to use based on the pointing mode. Assign appropriate handles to the DCM and angular velocity functions for the reference frame.
    if pointing_mode.lower() == 'gmo':
        DCM_ref_func = GMO_pointing_frame_dcm
        omega_ref_func = GMO_pointing_frame_angular_velocity
    elif pointing_mode.lower() == 'nadir':
        DCM_ref_func = nadir_frame_dcm
        omega_ref_func = nadir_frame_angular_velocity
    elif pointing_mode.lower() == 'sun':
        DCM_ref_func = sun_frame_dcm
        omega_ref_func = sun_frame_angular_velocity
    else:
        raise ValueError("Invalid pointing mode. Must be 'GMO', 'Nadir', or 'Sun'.")

    # Compute the DCM from the inertial frame to the reference frame at the current time
    DCM_ref = DCM_ref_func(t)

    # Compute the angular velocity of the reference frame relative to the inertial frame expressed in inertial coordinates at the current time
    omega_ref = omega_ref_func(t)

    # Compute attitude error and angular velocity error relative to the reference frame
    sigma_BR, omega_BR = attitude_error_eval(t, sigma, omega, DCM_ref, omega_ref)

    # Compute the control vector based on the attitude error and angular velocity error
    u = control_vector(P, K, sigma_BR, omega_BR)

    # Build identity matrix
    I_3_3 = np.eye(3)

    # Build the skew-symmetric matrix for the MRP vector
    sigma_tilde = np.zeros((3, 3))
    sigma_tilde[0, 1] = -sigma[2]
    sigma_tilde[0, 2] = sigma[1]
    sigma_tilde[1, 2] = -sigma[0]
    sigma_tilde -= sigma_tilde.T

    # Compute the derivative of the MRP vector
    sigma_dot = 0.25 * ( (1 - np.dot(sigma, sigma)) * I_3_3 + 2 * sigma_tilde + 2 * np.outer(sigma, sigma) ) @ omega

    # Compute the derivative of the angular velocity vector (assuming no external torques)
    omega_dot = np.linalg.inv(I) @ (u - np.cross(omega, I @ omega))

    # Combine derivatives into a single state derivative vector
    state_dot = np.concatenate((sigma_dot, omega_dot))

    return state_dot
