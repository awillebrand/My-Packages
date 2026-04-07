import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from constants import mu, R_e, J2, J3, C_d, spacecraft_mass, spacecraft_area
from Tools import Integrator, MeasurementMgr, CoordinateMgr, covariance_ellipse, covariance_ellipse_2D
from scenarios import FILE_PATH, time_vec, initial_state, initial_covariance, NUM_TRAJECTORIES
from multiprocessing import Pool, cpu_count
import signal
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
np.set_printoptions(linewidth=200)

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

def generate_corner_plots(trajectory_data: np.ndarray, four_hour_intervals: np.ndarray, file_path: str):
    """
    Generates corner plots for the Monte Carlo simulation trajectories. Each plot shows the
    distribution of a component of the state vector at each time step.

    Parameters
    ----------
    trajectory_data : np.ndarray
        A 3D numpy array containing the state vectors for each trajectory at each 4-hour interval.
        The shape of the array is (num_trajectories, num_intervals, state_vector_length).
    four_hour_intervals : np.ndarray
        The indexes of every 4-hour interval in the time vector.
    file_path : str
        The file path where the generated figures will be saved.
    Returns
    -------
    figures : list of go.Figure
        The generated corner plot figures, one per 4-hour interval.
    """
    from plotly.subplots import make_subplots

    state_labels = ['X (km)', 'Y (km)', 'Z (km)', 'Vx (km/s)', 'Vy (km/s)', 'Vz (km/s)']
    n = 6
    figures = []

    for interval_idx in range(len(four_hour_intervals)):
        samples = trajectory_data[:, interval_idx, :]  # shape: (num_traj, 6)
        cov_full = np.cov(samples.T)  # shape: (6, 6)
        means = np.mean(samples, axis=0)  # shape: (6,)

        fig = make_subplots(
            rows=n, cols=n,
            horizontal_spacing=0.04,
            vertical_spacing=0.04,
        )

        for row in range(1, n + 1):
            for col in range(1, n + 1):
                ri, ci = row - 1, col - 1  # 0-indexed
                x_data = samples[:, ci]
                y_data = samples[:, ri]

                if col > row:
                    # Upper triangle — add invisible trace to keep grid consistent
                    fig.add_trace(
                        go.Scatter(x=[None], y=[None], showlegend=False),
                        row=row, col=col
                    )

                elif col == row:
                    # ── Diagonal: histogram + median/±1σ dashed lines ──
                    median_val = np.median(x_data)
                    sigma_plus  = np.percentile(x_data, 84.135) - median_val
                    sigma_minus = median_val - np.percentile(x_data, 15.865)

                    fig.add_trace(
                        go.Histogram(
                            x=x_data,
                            nbinsx=25,
                            marker=dict(color='steelblue', opacity=0.7),
                            showlegend=False,
                        ),
                        row=row, col=col
                    )

                    # Dashed vertical lines: median and ±1σ
                    for xval, dash in [
                        (median_val,              'dash'),
                        (median_val + sigma_plus,  'dot'),
                        (median_val - sigma_minus, 'dot'),
                    ]:
                        fig.add_vline(
                            x=xval,
                            line=dict(color='navy', dash=dash, width=1.5),
                            row=row, col=col
                        )

                    # Annotation: mean ± sigma above each diagonal plot
                    fig.add_annotation(
                        text=f"{state_labels[ci]} = {median_val:.3g}"
                             f"<sup>+{sigma_plus:.2g}</sup>"
                             f"<sub>−{sigma_minus:.2g}</sub>",
                        xref=f"x{(ri * n + ci) + 1}" if (ri * n + ci) > 0 else "x",
                        yref="paper",
                        x=median_val,
                        yanchor="bottom",
                        showarrow=False,
                        font=dict(size=13),  # Up from 10
                        y=(1.0 - (row - 1) / n) - 0.01,
                    )

                else:
                    # ── Lower triangle: scatter + KDE contours ──

                    # Scatter (low alpha dots like the reference image)
                    fig.add_trace(
                        go.Scatter(
                            x=x_data,
                            y=y_data,
                            mode='markers',
                            marker=dict(
                                color='rgba(100, 149, 237, 0.25)',
                                size=3,  # Up from 2
                            ),
                            showlegend=False,
                        ),
                        row=row, col=col
                    )

                    # Covariance ellipses at 1σ, 2σ, 3σ
                    center = means[[ci, ri]]
                    cov_2x2 = cov_full[np.ix_([ci, ri], [ci, ri])]
                    for sigma in [1, 2, 3]:
                        ellipse_pts = covariance_ellipse_2D(center, cov_2x2, num_points=120, sigma_level=sigma)
                        fig.add_trace(
                            go.Scatter(
                                x=ellipse_pts[:, 0],
                                y=ellipse_pts[:, 1],
                                mode='lines',
                                line=dict(color='navy', width=1.5),
                                showlegend=False,
                            ),
                            row=row, col=col
                        )

        # ── Axis labels: bottom row gets x-labels, leftmost col gets y-labels ──
        for i in range(n):
            axis_idx_bottom = (n - 1) * n + i + 1
            axis_idx_left   = i * n + 1

            bottom_key = f"xaxis{axis_idx_bottom}" if axis_idx_bottom > 1 else "xaxis"
            left_key   = f"yaxis{axis_idx_left}"   if axis_idx_left   > 1 else "yaxis"

            fig.update_layout(**{
                bottom_key: dict(title=dict(text=state_labels[i], font=dict(size=14))),
                left_key:   dict(title=dict(text=state_labels[i], font=dict(size=14))),
            })

        # Hide upper-triangle axes
        for row in range(1, n + 1):
            for col in range(row + 1, n + 1):
                axis_idx = (row - 1) * n + col
                xkey = f"xaxis{axis_idx}" if axis_idx > 1 else "xaxis"
                ykey = f"yaxis{axis_idx}" if axis_idx > 1 else "yaxis"
                fig.update_layout(**{
                    xkey: dict(visible=False),
                    ykey: dict(visible=False),
                })

        time_hours = four_hour_intervals[interval_idx] / 3600  # approximate label
        fig.update_layout(
            title=dict(
                text=f"Corner Plot — t = {time_hours:.2f} hrs",
                font=dict(size=30),
            ),
            height=200 * n,
            width=200 * n,
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=False,
        )

        fname = f"{file_path}/corner_plot_interval_{interval_idx:02d}.html"
        fig.write_html(fname)
        print(f"Saved: {fname}")
        figures.append(fig)

    return figures

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

    # Generate corner plots
    figures = generate_corner_plots(trajectory_data, reduced_time_vec, file_path)
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
        mode='markers',
        name='Mean Trajectory',
        line=dict(color='yellow', width=8)
    ))

    # Plot covariance ellipses at every 4-hour interval
    for i, idx in enumerate(four_hour_intervals):
        cov_at_time = analysis_results[time_vec[idx]]["covariance"]
        mean_at_time = analysis_results[time_vec[idx]]["mean"]
        n_pts = 40
        ellipse_points = covariance_ellipse(mean_at_time[:3], cov_at_time[:3,:3], num_points=n_pts)

        tri_i, tri_j, tri_k = [], [], []
        for row in range(n_pts - 1):
            for col in range(n_pts - 1):
                p0 = row * n_pts + col
                p1 = p0 + 1
                p2 = p0 + n_pts
                p3 = p2 + 1
                tri_i.extend([p0, p0])
                tri_j.extend([p1, p3])
                tri_k.extend([p2, p1])

        fig.add_trace(go.Mesh3d(
            x=ellipse_points[:, 0],
            y=ellipse_points[:, 1],
            z=ellipse_points[:, 2],
            i=tri_i,
            j=tri_j,
            k=tri_k,
            color=color_list[i],
            opacity=0.25,
            name=f'Covariance Ellipse at t={time_vec[idx]}s',
            showlegend=True
        ))

    # set axes to be equal
    fig.update_layout(title='Monte Carlo Trajectories with Covariance Ellipses', scene=dict(
        xaxis_title='X (km)',
        yaxis_title='Y (km)',
        zaxis_title='Z (km)',
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

    figures[0].write_image(f"{FILE_PATH}/monte_carlo_trajectories.png")

    # Save analysis results to a text file
    with open(f"{FILE_PATH}/analysis_results.txt", "w") as f:
        for time, results in analysis_results.items():
            f.write(f"Time: {time}s\n")
            f.write(f"Mean: {results['mean']}\n")
            diag_cov = np.diag(results['covariance'])
            sigma_vec = np.sqrt(diag_cov)
            f.write(f"Standard Deviations: {sigma_vec}\n")
            f.write(f"Nominal State: {nominal_trajectory[1][:, time_vec.tolist().index(time)]}\n\n")

    # Save the Monte Carlo trajectories to a file
    with open(f"{FILE_PATH}/monte_carlo_trajectories.npy", "wb") as f:
        np.save(f, trajectories)
