import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from constants import mu, R_e, J2, J3, C_d, spacecraft_mass, spacecraft_area
from Tools import Integrator, MeasurementMgr, CoordinateMgr, covariance_ellipse
from scenarios import FILE_PATH, time_vec, initial_state, initial_covariance, NUM_TRAJECTORIES
from multiprocessing import Pool, cpu_count
import signal
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
"""
This script generates Monte Carlo simulations trajectories in LEO (Low Earth Orbit) with mu, J2, and drag effects. The distribution is based on some inputted
a priori covariance given by the user. It generates plots and statistics of the resulting trajectories for later analysis. The __main__ function is the entry
point for the script, which sets up the initial conditions, runs the Monte Carlo simulations, and processes the results.
"""

def _integrate_single(args):
    sampled_state, final_time, time_vec = args
    integrator = Integrator(mu, R_e, J2=J2, Cd=C_d, spacecraft_mass=spacecraft_mass, spacecraft_area=spacecraft_area)
    _, state_history = integrator.integrate_eom(final_time, sampled_state, teval=time_vec)
    return state_history

def generate_monte_carlo_trajectories(initial_state : np.ndarray, covariance : np.ndarray, num_traj : int, time_vec : np.ndarray):
    """
    Generates Monte Carlo simulation trajectories for a spacecraft in LEO.

    Parameters
    ----------
    integrator : Integrator
        An instance of the Integrator class to perform numerical integration.
    initial_state : np.ndarray
        The initial state vector of the spacecraft (position and velocity).
    covariance : np.ndarray
        The covariance matrix representing the uncertainty in the initial state.
    num_traj : int
        The number of Monte Carlo trajectories to generate.
    time_vec : np.ndarray
        The time vector for the integration, specifying the time steps at which to compute the trajectories.
    Returns
    -------
    trajectories : list of np.ndarray
        A list of state vectors for each trajectory at each time step.
    """
    # trajectories = []
    # final_time = time_vec[-1]
    # for i in range(num_traj):
    #     print(f"Generating trajectory {i+1}/{num_traj}")
    #     # Sample initial state from the covariance
    #     sampled_state = np.random.multivariate_normal(initial_state, covariance)
    #     # Integrate the trajectory
    #     traj = integrator.integrate_eom(final_time, sampled_state, teval=time_vec)
    #     trajectories.append(traj)
    # return trajectories

    final_time = time_vec[-1]
    sampled_states = [
        np.random.multivariate_normal(initial_state, covariance)
        for _ in range(num_traj)
    ]
    args = [(s, final_time, time_vec) for s in sampled_states]

    trajectories = []
    # Workers ignore SIGINT — let the parent handle it
    num_workers = min(cpu_count() // 2, 6)  # Use half your cores
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_integrate_single, a) for a in args]
        try:
            for future in tqdm(as_completed(futures), total=num_traj, desc="Monte Carlo"):
                trajectories.append(future.result())
        except KeyboardInterrupt:
            print("\nCaught Ctrl+C — cancelling remaining work...")
            for f in futures:
                f.cancel()
            # Force kill all worker processes immediately
            executor.shutdown(wait=False, cancel_futures=True)
            raise
    print()
    return trajectories

def analyze_monte_carlo_results(nominal_trajectory : np.ndarray, time_vec : np.ndarray, trajectories: list, file_path: str):
    """
    Analyzes the results of Monte Carlo simulation trajectories. Performs several processes for every 4 hours of data:
    1. Plot all trajectories, the nominal trajectory, the covariance ellipses, and the mean trajectory.
    2. Lists the covariance for each state component.
    3. Lists the mean for each state component.
    4. Lists the nominal value for each state component.

    Parameters
    ----------
    nominal_trajectory : np.ndarray
        The nominal trajectory of the spacecraft for comparison.
    time_vec : np.ndarray
        The time vector corresponding to the trajectories and covariance history.
    trajectories : list of np.ndarray
        A list of state vectors for each trajectory at each time step.
    file_path : str
        The file path where the generated figures will be saved.

    Returns
    -------
    figures : list
        A list of figures generated from the analysis, including trajectory plots and covariance ellipses. Saves the generated figures to disk for later review.
    analysis_results : dict
        A dictionary containing statistical analysis of the trajectories, such as mean and covariance at each time step.
    """
    
    figures = []
    analysis_results = {}
    

    # Find indexes of every 4 hours in the time vector (assuming time_vec is in seconds)
    four_hour_intervals = np.where((time_vec % (4 * 3600)) == 0)[0]
    trajectory_data = np.zeros((len(trajectories), len(four_hour_intervals), 6))  # Assuming state vector has 6 components (x, y, z, vx, vy, vz)
    reduced_time_vec = time_vec[four_hour_intervals]
    for j, idx in enumerate(four_hour_intervals):
        # Extract the trajectories at the current time step

        for i, traj in enumerate(trajectories):
            traj_at_time = np.array(traj[:,idx])  # Extract the state vector at the current time step for trajectory i
            trajectory_data[i, j, :] = traj_at_time # Store the state vector for trajectory i at the current time step

        # Compute the mean and covariance at the current time step
        mean_at_time = np.zeros(6)
        for i in range(6):
            mean_at_time[i] = np.mean(trajectory_data[:, j, i])

        cov_at_time = np.cov(trajectory_data[:, j, :].T)
        print(f"Time: {time_vec[idx]}s, State Component {i}: Mean = {mean_at_time}")
        # mean_at_time = np.mean(traj_at_time, axis=0)
        # cov_at_time = np.cov(traj_at_time.T)
        analysis_results[time_vec[idx]] = {
            "mean": mean_at_time,
            "covariance": cov_at_time
        }
    # Generate the trajectory plot with covariance ellipses
    fig = go.Figure()
    # Plot all trajectories
    color_list = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta']
    for i, color in enumerate(color_list):
        fig.add_trace(go.Scatter3d(
            x=trajectory_data[:, i, 0],
            y=trajectory_data[:, i, 1],
            z=trajectory_data[:, i, 2],
            mode='markers',
            name=f'Sample Point {i+1}',
            line=dict(color=color, width=2),
            showlegend=True
        ))

    # Plot nominal trajectory
    fig.add_trace(go.Scatter3d(
        x=nominal_trajectory[1][0, :],
        y=nominal_trajectory[1][1, :],
        z=nominal_trajectory[1][2, :],
        mode='lines',
        name='Nominal Trajectory',
        line=dict(color='black', width=4)
    ))
    
    # Plot mean trajectory
    mean_trajectory = np.array([analysis_results[time]["mean"] for time in reduced_time_vec])
    fig.add_trace(go.Scatter3d(
        x=mean_trajectory[:, 0],
        y=mean_trajectory[:, 1],
        z=mean_trajectory[:, 2],
        mode='lines',
        name='Mean Trajectory',
        line=dict(color='blue', width=4)
    ))

    # Plot covariance ellipses at every 4-hour interval
    for idx in four_hour_intervals:
        cov_at_time = analysis_results[time_vec[idx]]["covariance"]
        mean_at_time = analysis_results[time_vec[idx]]["mean"]
        ellipse_points = covariance_ellipse(mean_at_time[:3], cov_at_time[:3,:3], num_points=40)

        fig.add_trace(go.Scatter3d(
            x=[ellipse_points[:, 0]],
            y=[ellipse_points[:, 1]],
            z=[ellipse_points[:, 2]],
            mode='markers',
            marker=dict(size=2, color='red'),
            name=f'Covariance Ellipse at t={time_vec[idx]}s'
        ))
    # set axes to be equal
    fig.update_layout(title='Monte Carlo Trajectories with Covariance Ellipses', scene=dict(
        xaxis_title='X (km)',
        yaxis_title='Y (km)',
        zaxis_title='Z (km)',
        aspectmode='data'
    ))

    figures.append(fig)
    fig.write_html(f"{file_path}/monte_carlo_trajectories.html")

    return figures, analysis_results

if __name__ == "__main__":
    """
    Entry point to the script. The user can set up the initial conditions, run the Monte Carlo simulations, and process the results.
    This section will define the initial state, covariance, number of trajectories, and time vector for the simulation. It will then
    call the functions to generate the trajectories and analyze the results, ultimately saving the generated figures to disk for later review.
    """
    integrator = Integrator(mu, R_e, J2=J2, Cd=C_d, spacecraft_mass=spacecraft_mass, spacecraft_area=spacecraft_area)
    nominal_trajectory = integrator.integrate_eom(time_vec[-1], initial_state, teval=time_vec)
    trajectories = generate_monte_carlo_trajectories(initial_state, initial_covariance, NUM_TRAJECTORIES, time_vec)
    figures, analysis_results = analyze_monte_carlo_results(nominal_trajectory, time_vec, trajectories, FILE_PATH)