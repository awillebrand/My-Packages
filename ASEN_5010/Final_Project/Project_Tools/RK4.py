import numpy as np

"""
Runge-Kutta 4th order method for solving ordinary differential equations (ODEs).
"""
def rk4(y0, t, u, I):
    """
    Perform the RK4 integration.

    Parameters:
    func : function
        The function that returns the derivative of y at time t.
    y0 : np.array
        Initial condition (value of y at t[0]).
    t : np.array
        Array of time points where the solution is computed.
    u : np.array
        Control input (assumed to be constant for this implementation).
    I : np.array
        Inertia matrix of the spacecraft (assumed to be constant for this implementation).

    Returns:
    y : np.array
        Array of solution values corresponding to each time point in t.
    """
    n = len(t)
    y = [y0]
    
    for i in range(1, n):
        dt = t[i] - t[i-1]
        k1 = eom(y[-1], t[i-1], u, I)
        k2 = eom(y[-1] + 0.5 * dt * k1, t[i-1] + 0.5 * dt, u, I)
        k3 = eom(y[-1] + 0.5 * dt * k2, t[i-1] + 0.5 * dt, u, I)
        k4 = eom(y[-1] + dt * k3, t[i-1] + dt, u, I)
        
        y_next = y[-1] + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        y.append(y_next)
    
    # Convert to numpy array for easier handling
    y = np.array(y)

    return y

def eom(state, t, u, I):
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
