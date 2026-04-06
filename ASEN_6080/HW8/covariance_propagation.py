import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from constants import mu, R_e, J2, J3, C_d, spacecraft_mass, spacecraft_area
from Tools import Integrator, MeasurementMgr, CoordinateMgr, covariance_ellipse, UKF
from scenarios import FILE_PATH, time_vec, initial_state, initial_covariance, NUM_TRAJECTORIES
np.set_printoptions(linewidth=200)
"""
This file runs CKF and UKF propagation on the initial covariance to analyze the validity
of the covariance propagation using CKF and UKF methods. Data from the Monte Carlo simulations
is used as a reference to evaluate the accuracy of the covariance propagation.
"""

def load_in_mc_data(file_path: str):
    """
    Loads Monte Carlo simulation data from a specified file path and only pulls states from 4 hour intervals.

    Parameters
    ----------
    file_path : str
        The path to the file containing the Monte Carlo simulation data.

    Returns
    -------
    np.ndarray
        A numpy array containing the loaded Monte Carlo simulation trajectories.
    """

    mc_data = np.load(file_path)
    four_hour_indices = [i for i, t in enumerate(time_vec) if t % (4 * 3600) == 0]
    return mc_data[:, :, four_hour_indices]

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

    _, augmented_state_history = integrator.integrate_stm(final_time, initial_state, teval=time_vec)

    # Separate the state transition matrix (STM) history from the augmented state history
    state_estimate = augmented_state_history[:6,:]  # Extract the state estimates (first 6 rows)
    stm_history = augmented_state_history[6:,:].reshape(-1, 6, 6)

    # Propagate the covariance using the STM history
    num_steps = stm_history.shape[0]
    
    propagated_covariances = np.zeros((6, 6, num_steps))
    for i in range(num_steps):
        propagated_covariances[:,:, i] = stm_history[i] @ initial_covariance @ stm_history[i].T
    return state_estimate, propagated_covariances

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
        print(f"UKF Time: {time}", flush=True)
        sigma_points = ukf.compute_sigma_points(x_est, P_est, gamma)
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
    percentage_per_time_step = np.zeros(num_time_steps)
    for t in range(num_time_steps):
        time_step_total = 0
        time_step_inside = 0
        nominal_state = nominal_state_list[t]
        cov = covariance_list[t]

        sigma_bounds = 2.0 * np.sqrt(np.diag(cov))
        
        for traj in mc_trajectories:
            print(f"Computing for trajectory {traj} at time step {t}", end="\r")
            state = traj[:,t]
            time_step_total += 1
            total_states += 1
            if np.all(np.abs(state - nominal_state) <= sigma_bounds):
                time_step_inside += 1
                inside_count += 1
        percentage_per_time_step[t] = (time_step_inside / time_step_total if time_step_total > 0 else 0) * 100.0


    percentage_inside = (inside_count / total_states) * 100.0
    return percentage_inside, percentage_per_time_step

if __name__ == "__main__":
    # Load Monte Carlo simulation data
    mc_trajectories = load_in_mc_data(f"{FILE_PATH}/monte_carlo_trajectories.npy")
    four_hour_indices = [i for i, t in enumerate(time_vec) if t % (4 * 3600) == 0]
    kf_time_vec = time_vec[four_hour_indices]

    # Propagate covariance using CKF
    ckf_state_estimate, ckf_covariances = propagate_covariance_ckf(initial_covariance, kf_time_vec)

    # Propagate covariance using UKF
    ukf_state_estimate, ukf_covariances = propagate_covariance_ukf(initial_covariance, kf_time_vec)

    # # Find covariances at the four hour intervals for both CKF and UKF
    # four_hour_indices = [i for i, t in enumerate(time_vec) if t % (4 * 3600) == 0]
    # ckf_covariances_at_intervals = ckf_covariances[:,:, four_hour_indices]
    # ukf_covariances_at_intervals = ukf_covariances[:,:, four_hour_indices]


    # Calculate percentage of Monte Carlo states inside the 2-sigma covariance ellipse for CKF and UKF
    print("Calculating percentage of Monte Carlo states inside the 2-sigma covariance ellipse for CKF...")
    ckf_percentage_inside, ckf_percentage_per_time_step = percentage_of_mc_inside_cov(mc_trajectories, ckf_state_estimate.T, ckf_covariances.transpose(2, 0, 1))
    print("Calculating percentage of Monte Carlo states inside the 2-sigma covariance ellipse for UKF...")
    ukf_percentage_inside, ukf_percentage_per_time_step = percentage_of_mc_inside_cov(mc_trajectories, ukf_state_estimate.T, ukf_covariances.transpose(2, 0, 1))

    print(f"Percentage of Monte Carlo states inside 2-sigma covariance ellipse (CKF): {ckf_percentage_inside:.2f}%")
    print(f"Percentage of Monte Carlo states inside 2-sigma covariance ellipse (UKF): {ukf_percentage_inside:.2f}%")
    print(f"Percentage of Monte Carlo states inside 2-sigma covariance ellipse at each time step (CKF): {ckf_percentage_per_time_step}")
    print(f"Percentage of Monte Carlo states inside 2-sigma covariance ellipse at each time step (UKF): {ukf_percentage_per_time_step}")

    # Write results to a text file
    with open(f"{FILE_PATH}/covariance_propagation_results.txt", "w") as f:
        f.write(f"Percentage of Monte Carlo states inside 2-sigma covariance ellipse (CKF): {ckf_percentage_inside:.2f}%\n")
        f.write(f"Percentage of Monte Carlo states inside 2-sigma covariance ellipse (UKF): {ukf_percentage_inside:.2f}%\n")
        f.write(f"Percentage of Monte Carlo states inside 2-sigma covariance ellipse per time step (CKF): {ckf_percentage_per_time_step}\n")
        f.write(f"Percentage of Monte Carlo states inside 2-sigma covariance ellipse per time step (UKF): {ukf_percentage_per_time_step}\n")