import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from constants import mu, R_e, J2, J3, C_d, spacecraft_mass, spacecraft_area
from Tools import Integrator, MeasurementMgr, CoordinateMgr, covariance_ellipse, UKF
from scenarios import FILE_PATH, time_vec, initial_state, initial_covariance, NUM_TRAJECTORIES

"""
This file runs CKF and UKF propagation on the initial covariance to analyze the validity
of the covariance propagation using CKF and UKF methods. Data from the Monte Carlo simulations
is used as a reference to evaluate the accuracy of the covariance propagation.
"""

def load_in_mc_data(file_path: str):
    """
    Loads Monte Carlo simulation data from a specified file path.

    Parameters
    ----------
    file_path : str
        The path to the file containing the Monte Carlo simulation data.

    Returns
    -------
    np.ndarray
        A numpy array containing the loaded Monte Carlo simulation trajectories.
    """
    return np.load(file_path)

def propagate_covariance_ckf(initial_covariance: np.ndarray, time_vec: np.ndarray):
    """
    Propagates the initial covariance using the Classical Kalman Filter (CKF) method.
    This just computes the STM time history and propagates the covariance using the linearized dynamics.

    Parameters
    ----------
    initial_covariance : np.ndarray
        The initial covariance matrix of the state vector.
    time_vec : np.ndarray
        The time vector for the propagation, specifying the time steps at which to compute the covariance.

    Returns
    -------
    ckf_state_estimate : np.ndarray
        A 2D numpy array containing the CKF state estimates at each time step.
    propagated_covariances : np.ndarray
        A 3D numpy array containing the propagated covariance matrices at each time step.
    """
    
    integrator = Integrator(mu=mu, R_e=R_e, J2=J2, Cd=C_d, spacecraft_mass=spacecraft_mass, spacecraft_area=spacecraft_area)
    final_time = time_vec[-1]

    augmented_state_history = integrator.integrate_stm(final_time, initial_state, teval=time_vec)

    # Separate the state transition matrix (STM) history from the augmented state history
    stm_history = augmented_state_history[:, 6:].reshape(-1, 6, 6)

    # Propagate the covariance using the STM history
    num_steps = stm_history.shape[0]
    
    ckf_state_estimate = np.zeros((6, num_steps))
    propagated_covariances = np.zeros((6, 6, num_steps))
    for i in range(num_steps):
        ckf_state_estimate[:, i] = stm_history[i] @ initial_state
        propagated_covariances[:,:, i] = stm_history[i] @ initial_covariance @ stm_history[i].T

    return ckf_state_estimate, propagated_covariances

def propagate_covariance_ukf(initial_covariance: np.ndarray, time_vec: np.ndarray, alpha : float = 1e-3, beta: float = 2.0, Q : np.ndarray= None):
    """
    Propagates the initial covariance using the Unscented Kalman Filter (UKF) method.
    This method uses the unscented transform to propagate the covariance through the nonlinear dynamics,
    which can be pulled directly from the time_update method of the UKF.

    Parameters
    ----------
    initial_covariance : np.ndarray
        The initial covariance matrix of the state vector.
    time_vec : np.ndarray
        The time vector for the propagation, specifying the time steps at which to compute the covariance.
    alpha : float
        UKF scaling parameter alpha.
    beta : float
        UKF scaling parameter beta.
    Q : np.ndarray
        Process noise covariance matrix. If None, process noise is assumed to be zero.
    Returns
    -------
    state_time_history : np.ndarray
        A 2D numpy array containing the UKF state estimates at each time step.
    covariance_time_history : np.ndarray
        A 3D numpy array containing the propagated covariance matrices at each time step.
    """
    # Initialize UKF. Since there are no measurements for this scenario, measurement variables are arbitrarily initialized.
    integrator = Integrator(mu=mu, R_e=R_e, J2=J2, Cd=C_d, spacecraft_mass=spacecraft_mass, spacecraft_area=spacecraft_area)
    ukf = UKF(integrator, [], 0.0)

    L = len(initial_state)
    Wm, Wc, gamma = ukf.compute_weights(alpha, beta, L)

    x_est = initial_state.copy()
    P_est = initial_covariance.copy()
    
    state_time_history = np.zeros((6, len(time_vec)))
    covariance_time_history = np.zeros((6, 6, len(time_vec)))

    for k, time in enumerate(time_vec):
        sigma_points = ukf.compute_sigma_points(x_est, P_est, gamma)
        x_est, P_est = ukf.time_update(sigma_points, Wm, Wc)
        if k == 0:
            dt = 0
            predicted_sigma_points = sigma_points
        else:
            dt=time - time_vec[k-1]
            predicted_sigma_points = ukf.propagate_sigma_points(sigma_points, dt=dt)

        # Time update to get predicted state mean and covariance
        x_est, P_est = ukf.time_update(predicted_sigma_points, Wm, Wc, Q, dt)

        state_time_history[:, k] = x_est
        covariance_time_history[:,:, k] = P_est

    return state_time_history, covariance_time_history

def percentage_of_mc_inside_cov(mc_trajectories : list, nominal_state_list : list,covariance_list : list):
    """
    Calculates the percentage of Monte Carlo simulation states that lie within the 2-sigma covariance ellipse.

    Parameters
    ----------
    mc_states : list
        A list of Monte Carlo simulation trajectories. Contains a list of numpy arrays, where each array represents the state vectors of a single trajectory at each time step.
    nominal_state_list : list
        A list of nominal state vectors at each time step. Covariance centered around this nominal state.
    covariance_list : np.ndarray
        The list of covariance matrices at the time steps corresponding to the Monte Carlo trajectories.

    Returns
    -------
    float
        The percentage of Monte Carlo states that lie within the 2-sigma covariance ellipse.
    """
    total_states = 0
    inside_count = 0

    num_trajectories = len(mc_trajectories)
    num_time_steps = mc_trajectories[0].shape[0]

    for t in range(num_time_steps):
        nominal_state = nominal_state_list[t]
        cov = covariance_list[t]

        sigma_bounds = 2.0 * np.sqrt(np.diag(cov))
        
        for traj in mc_trajectories:
            state = traj[t]
            total_states += 1
            if np.all(np.abs(state - nominal_state) <= sigma_bounds):
                inside_count += 1

    percentage_inside = (inside_count / total_states) * 100.0
    return percentage_inside




