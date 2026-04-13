import numpy as np
from Project_Tools.control_vector import control_vector
from Project_Tools.attitude_error_eval import attitude_error_eval
from Project_Tools.sun_frame_funcs import sun_frame_dcm, sun_frame_angular_velocity
from Project_Tools.nadir_frame_dcm import nadir_frame_dcm, nadir_frame_angular_velocity
from Project_Tools.GMO_pointing_frame_funcs import GMO_pointing_frame_dcm, GMO_pointing_frame_angular_velocity
from Project_Tools.get_orbit_state import get_orbit_state
from Testing_Scripts.initial_conditions import r_mars, r_LMO, raan_LMO, inc_LMO, ta_0_LMO, ta_dot_LMO, r_GMO, raan_GMO, inc_GMO, ta_0_GMO, ta_dot_GMO
"""
Runge-Kutta 4th order method for solving ordinary differential equations (ODEs).
"""
def rk4(y0, t, I, P, K, pointing_mode=None):
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
    P : np.array
        The P gain matrix for the control system.
    K : np.array
        The K gain matrix for the control system.
    pointing_mode : str, optional
        The pointing mode override. If None, the function will determine the pointing mode based on the simulation scenario:
        1. Sun Pointing: Occurs when the spacecraft can see the Sun (LMO spacecraft has positive inertial position coordinates in the n2 axis).
        2. Nadir Pointing: Occurs when the spacecraft is on the shadow side of the planet (LMO spacecraft has positive inertial position coordinates in the n2 axis) and the GMO satellite is not visible.
        3. GMO Pointing: Occurs when the spacecraft is on the shadow side of the planet (LMO spacecraft has positive inertial position coordinates in the n2 axis) and the GMO satellite is visible.
         If a specific pointing mode is provided, it will override the automatic determination based on the simulation scenario.
    Returns:
    y : np.array
        Array of solution values corresponding to each time point in t.
    """
    if pointing_mode is not None:
        if pointing_mode.lower() not in ['gmo', 'nadir', 'sun']:
            raise ValueError("Invalid pointing mode. Must be 'GMO', 'Nadir', or 'Sun'.")
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

    n = len(t)
    y = [y0]
    
    for i in range(1, n):
        print(f"Integrating from t={t[i-1]} to t={t[i]}...")
        # Determine the current pointing mode based on mission scenario
        if pointing_mode is None:
            time = t[i-1]
            pos_LMO, _ = get_orbit_state(r_LMO, raan_LMO, inc_LMO, ta_0_LMO + ta_dot_LMO * time)

            if pos_LMO[1] > 0:
                DCM_ref_func = sun_frame_dcm
                omega_ref_func = sun_frame_angular_velocity
            else:
                pos_GMO, _ = get_orbit_state(r_GMO, raan_GMO, inc_GMO, ta_0_GMO + ta_dot_GMO * time)
                # Check if GMO is visible (this is a placeholder condition; the actual visibility check would depend on the spacecraft's position and the GMO's position)
                GMO_visible = evaluate_GMO_visibility(pos_LMO, pos_GMO)
                if GMO_visible:
                    DCM_ref_func = GMO_pointing_frame_dcm
                    omega_ref_func = GMO_pointing_frame_angular_velocity
                else:
                    DCM_ref_func = nadir_frame_dcm
                    omega_ref_func = nadir_frame_angular_velocity

        dt = t[i] - t[i-1]
        # Compute control vector for current state
        DCM_ref = DCM_ref_func(t[i-1])
        omega_ref = omega_ref_func(t[i-1])

        sigma = y[-1][0:3]
        omega = y[-1][3:6]

        # Compute attitude error and angular velocity error relative to the reference frame
        sigma_BR, omega_BR = attitude_error_eval(t, sigma, omega, DCM_ref, omega_ref)

        # Compute the control vector based on the attitude error and angular velocity error
        u = control_vector(P, K, sigma_BR, omega_BR)

        k1 = dt * eom(y[-1], t[i-1], I, u)
        k2 = dt * eom(y[-1] + 0.5 * k1, t[i-1] + 0.5 * dt, I, u)
        k3 = dt * eom(y[-1] + 0.5 * k2, t[i-1] + 0.5 * dt, I, u)
        k4 = dt * eom(y[-1] + k3, t[i-1], I, u)

        y_next = y[-1] + (1 / 6) * (k1 + 2*k2 + 2*k3 + k4)
        sigma_next = y_next[0:3]

        if np.dot(sigma_next, sigma_next) > 1.0:
            y_next[0:3] = -sigma_next / np.dot(sigma_next, sigma_next)
        y.append(y_next)
    
    # Convert to numpy array for easier handling
    y = np.array(y)

    return y

def eom(state, t, I, u):
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

def evaluate_GMO_visibility(pos_LMO, pos_GMO):
    """
    Evaluates the visibility condition of the GMO satellite relative to the LMO spacecraft at a given time.

    Parameters:
    time : float
        The current time at which to evaluate the visibility condition.
    Returns:
    bool
        True if the GMO satellite is visible from the LMO spacecraft, False otherwise.
    """
    # Define distance vetor from LMO to GMO
    vector_LMO_to_GMO = pos_GMO - pos_LMO

    # Using parametric equation theory, determine closest approach of the line of sight vector to the center of the planet (Mars)
    t_min = -np.dot(pos_LMO, vector_LMO_to_GMO) / np.dot(vector_LMO_to_GMO, vector_LMO_to_GMO)
    t_min = np.clip(t_min, 0, 1)  # Ensure t_min is within the segment defined by pos_LMO and pos_GMO

    closest_point = pos_LMO + t_min * vector_LMO_to_GMO

    if np.linalg.norm(closest_point) < r_mars:
        return False  # GMO is not visible (line of sight is blocked by Mars)
    else:
        return True