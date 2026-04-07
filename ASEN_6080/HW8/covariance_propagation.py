import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import scipy.stats
from constants import mu, R_e, J2, J3, C_d, spacecraft_mass, spacecraft_area
from Tools import Integrator, MeasurementMgr, CoordinateMgr, covariance_ellipse, covariance_ellipse_2D, UKF
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

def propagate_covariance_ckf(initial_covariance: np.ndarray, t_vec: np.ndarray):
    """
    Propagates the initial covariance using the Classical Kalman Filter (CKF) method.
    This just computes the STM time history and propagates the covariance using the linearized dynamics.

    Parameters
    ----------
    initial_covariance : np.ndarray
        The initial covariance matrix of the state vector.
    t_vec : np.ndarray
        The time vector for the propagation, specifying the time steps at which to compute the covariance.

    Returns
    -------
    ckf_state_estimate : np.ndarray
        A 2D numpy array containing the CKF state estimates at each time step.
    propagated_covariances : np.ndarray
        A 3D numpy array containing the propagated covariance matrices at each time step.
    """
    
    integrator = Integrator(mu=mu, R_e=R_e, J2=J2, Cd=C_d, spacecraft_mass=spacecraft_mass, spacecraft_area=spacecraft_area, mode=['J2', 'Drag'], parameter_indices=[6, 7])
    final_time = t_vec[-1]

    augmented_initial_state = np.hstack((initial_state, [J2, C_d]))
    augmented_initial_covariance = np.zeros((8, 8))
    augmented_initial_covariance[:6, :6] = initial_covariance

    _, augmented_state_history = integrator.integrate_stm(final_time, augmented_initial_state, teval=t_vec)

    # Separate the state transition matrix (STM) history from the augmented state history
    state_estimate = augmented_state_history[:6,:]  # Extract the state estimates (first 6 rows)
    
    num_steps = state_estimate.shape[1]
    propagated_covariances = np.zeros((8, 8, num_steps))
    for i in range(num_steps):
        stm_i = augmented_state_history[8:, i].reshape(8, 8)  # Extract the STM for the state variables
        propagated_covariances[:, :, i] = stm_i @ augmented_initial_covariance @ stm_i.T
    return state_estimate, propagated_covariances[:6, :6, :]

def propagate_covariance_ukf(initial_covariance: np.ndarray, time_vec: np.ndarray, alpha : float = 1e-2, beta: float = 2.0, Q : np.ndarray= None):
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
    
    state_time_history = np.zeros((6, len(time_vec)))
    covariance_time_history = np.zeros((6, 6, len(time_vec)))

    for k, time in enumerate(time_vec):
        x_est = initial_state.copy()
        P_est = initial_covariance.copy()
        print(f"UKF Time: {time}", flush=True)
        sigma_points = ukf.compute_sigma_points(x_est, P_est, gamma)
        if k == 0:
            dt = 0
            predicted_sigma_points = sigma_points
        else:
            dt=time - time_vec[0]
            predicted_sigma_points = ukf.propagate_sigma_points(sigma_points, dt=dt)

        # Time update to get predicted state mean and covariance
        x_est, P_est = ukf.time_update(predicted_sigma_points, Wm, Wc, Q, dt)

        state_time_history[:, k] = x_est
        covariance_time_history[:,:, k] = P_est

    return state_time_history, covariance_time_history

# def percentage_of_mc_inside_cov(mc_trajectories : list, nominal_state_list : list,covariance_list : list):
#     """
#     Calculates the percentage of Monte Carlo simulation states that lie within the 2-sigma covariance ellipse.

#     Parameters
#     ----------
#     mc_states : list
#         A list of Monte Carlo simulation trajectories. Contains a list of numpy arrays, where each array represents the state vectors of a single trajectory at each time step.
#     nominal_state_list : list
#         A list of nominal state vectors at each time step. Covariance centered around this nominal state.
#     covariance_list : np.ndarray
#         The list of covariance matrices at the time steps corresponding to the Monte Carlo trajectories.

#     Returns
#     -------
#     float
#         The percentage of Monte Carlo states that lie within the 2-sigma covariance ellipse.
#     """
#     total_states = 0
#     inside_count = 0

#     num_trajectories = len(mc_trajectories)
#     num_time_steps = mc_trajectories[0].shape[0]
#     percentage_per_time_step = np.zeros(num_time_steps)
#     for t in range(num_time_steps):
#         time_step_total = 0
#         time_step_inside = 0
#         nominal_state = nominal_state_list[t]
#         cov = covariance_list[t]

#         sigma_bounds = 2.0 * np.sqrt(np.diag(cov))
        
#         for traj in mc_trajectories:
#             print(f"Computing for trajectory {traj} at time step {t}", end="\r")
#             state = traj[:,t]
#             time_step_total += 1
#             total_states += 1
#             if np.all(np.abs(state - nominal_state) <= sigma_bounds):
#                 time_step_inside += 1
#                 inside_count += 1
#         percentage_per_time_step[t] = (time_step_inside / time_step_total if time_step_total > 0 else 0) * 100.0


#     percentage_inside = (inside_count / total_states) * 100.0
#     return percentage_inside, percentage_per_time_step

def percentage_of_mc_inside_cov(mc_trajectories, nominal_state_list, covariance_list):
    total_states = 0
    inside_count = 0
    num_time_steps = mc_trajectories[0].shape[1]
    percentage_per_time_step = np.zeros(num_time_steps)

    # 2-sigma threshold for a 6D chi-squared distribution
    chi2_threshold = scipy.stats.chi2.ppf(0.9545, df=6)  # ~6-state system

    for t in range(num_time_steps):
        time_step_inside = 0
        nominal_state = nominal_state_list[t]
        cov = covariance_list[t]
        cov_inv = np.linalg.inv(cov)

        for traj in mc_trajectories:
            state = traj[:, t]
            dx = state - nominal_state
            mahal_sq = dx @ cov_inv @ dx  # Mahalanobis distance squared
            if mahal_sq <= chi2_threshold:
                time_step_inside += 1
                inside_count += 1
            total_states += 1

        percentage_per_time_step[t] = (time_step_inside / len(mc_trajectories)) * 100.0

    return (inside_count / total_states) * 100.0, percentage_per_time_step

def plot_covariance_ellipses(ckf_state_estimate: np.ndarray, ckf_covariances: np.ndarray,
                              ukf_state_estimate: np.ndarray, ukf_covariances: np.ndarray,
                              kf_time_vec: np.ndarray, mc_trajectories: np.ndarray, file_path: str):
    """
    Plots 3D covariance ellipses for the CKF and UKF propagated covariances at each time step,
    overlaid with the Monte Carlo scatter points for comparison.

    Parameters
    ----------
    ckf_state_estimate : np.ndarray
        A 2D numpy array (6 x num_steps) of CKF state estimates at each time step.
    ckf_covariances : np.ndarray
        A 3D numpy array (6 x 6 x num_steps) of CKF propagated covariance matrices.
    ukf_state_estimate : np.ndarray
        A 2D numpy array (6 x num_steps) of UKF state estimates at each time step.
    ukf_covariances : np.ndarray
        A 3D numpy array (6 x 6 x num_steps) of UKF propagated covariance matrices.
    kf_time_vec : np.ndarray
        The time vector (in seconds) corresponding to the covariance time steps.
    mc_trajectories : np.ndarray
        Monte Carlo trajectory data, shape (num_trajectories, 6, num_time_steps).
    file_path : str
        The file path where the generated figure will be saved.

    Returns
    -------
    fig : go.Figure
        The Plotly figure containing the covariance ellipses and Monte Carlo scatter points.
    """
    n_pts = 40
    num_steps = len(kf_time_vec)
    ckf_colors = ['red', 'salmon', 'firebrick', 'indianred', 'darkred', 'orangered', 'tomato']
    ukf_colors = ['blue', 'cornflowerblue', 'darkblue', 'royalblue', 'navy', 'steelblue', 'dodgerblue']

    fig = go.Figure()

    # Plot Monte Carlo scatter points at each time step
    for i in range(num_steps):
        mc_positions = mc_trajectories[:, :3, i]  # shape: (num_traj, 3)
        fig.add_trace(go.Scatter3d(
            x=mc_positions[:, 0],
            y=mc_positions[:, 1],
            z=mc_positions[:, 2],
            mode='markers',
            marker=dict(size=1.5, color='gray', opacity=0.3),
            name=f'MC Samples t={kf_time_vec[i]/3600:.0f}h',
            showlegend=(i == 0)
        ))

    # Plot CKF state trajectory
    fig.add_trace(go.Scatter3d(
        x=ckf_state_estimate[0, :],
        y=ckf_state_estimate[1, :],
        z=ckf_state_estimate[2, :],
        mode='lines+markers',
        name='CKF State Estimate',
        line=dict(color='red', width=4),
        marker=dict(size=3, color='red')
    ))

    # Plot UKF state trajectory
    fig.add_trace(go.Scatter3d(
        x=ukf_state_estimate[0, :],
        y=ukf_state_estimate[1, :],
        z=ukf_state_estimate[2, :],
        mode='lines+markers',
        name='UKF State Estimate',
        line=dict(color='blue', width=4),
        marker=dict(size=3, color='blue')
    ))

    # Build triangle indices for the Mesh3d surface (same grid for all ellipses)
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

    # Plot CKF covariance ellipses
    for i in range(num_steps):
        center = ckf_state_estimate[:3, i]
        cov_3x3 = ckf_covariances[:3, :3, i]
  
        ellipse_points = covariance_ellipse(center, cov_3x3, num_points=n_pts)

        fig.add_trace(go.Mesh3d(
            x=ellipse_points[:, 0],
            y=ellipse_points[:, 1],
            z=ellipse_points[:, 2],
            i=tri_i,
            j=tri_j,
            k=tri_k,
            color=ckf_colors[i % len(ckf_colors)],
            opacity=0.2,
            name=f'CKF Cov Ellipse t={kf_time_vec[i]/3600:.0f}h',
            showlegend=True
        ))

    # Plot UKF covariance ellipses
    for i in range(num_steps):
        center = ukf_state_estimate[:3, i]
        cov_3x3 = ukf_covariances[:3, :3, i]
        ellipse_points = covariance_ellipse(center, cov_3x3, num_points=n_pts)

        fig.add_trace(go.Mesh3d(
            x=ellipse_points[:, 0],
            y=ellipse_points[:, 1],
            z=ellipse_points[:, 2],
            i=tri_i,
            j=tri_j,
            k=tri_k,
            color=ukf_colors[i % len(ukf_colors)],
            opacity=0.2,
            name=f'UKF Cov Ellipse t={kf_time_vec[i]/3600:.0f}h',
            showlegend=True
        ))

    fig.update_layout(
        title='CKF vs UKF Propagated Covariance Ellipses with Monte Carlo Samples',
        title_font=dict(size=24),
        width=1200,
        height=900,
        scene=dict(
            xaxis_title='X (km)',
            yaxis_title='Y (km)',
            zaxis_title='Z (km)',
        ),
        legend=dict(font=dict(size=12)),
    )

    fig.write_html(f"{file_path}/ckf_ukf_covariance_ellipses.html")
    print(f"Saved: {file_path}/ckf_ukf_covariance_ellipses.html")

    return fig

def generate_corner_plots(state_vecs : np.ndarray, cov_mat : np.ndarray, trajectory_data: np.ndarray, four_hour_intervals: np.ndarray, file_path: str, cov_type : str):
    """
    Generates corner plots for the Monte Carlo simulation trajectories. Each plot shows the
    distribution of a component of the state vector at each time step.

    Parameters
    ----------
    state_vecs : np.ndarray
        The nominal state vectors at each time step, used to center the covariance ellipses.
    cov_mat : np.ndarray
        The covariance matrix of the state vector, used to compute the covariance ellipses.
    trajectory_data : np.ndarray
        A 3D numpy array containing the state vectors for each trajectory at each 4-hour interval.
        The shape of the array is (num_trajectories, num_intervals, state_vector_length).
    four_hour_intervals : np.ndarray
        The indexes of every 4-hour interval in the time vector.
    file_path : str
        The file path where the generated figures will be saved.
    cov_type : str
        A string indicating the type of covariance being plotted (e.g., "CKF" or "UKF"), used for labeling the plots.
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
        samples = trajectory_data[:, :, interval_idx]
        cov_full = cov_mat[:,:,interval_idx]  # shape: (6, 6)
        means = state_vecs[:, interval_idx]  # shape: (6,)

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
                text=f"{cov_type} Corner Plot — t = {time_hours:.2f} hrs",
                font=dict(size=30),
            ),
            height=200 * n,
            width=200 * n,
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=False,
        )

        fname = f"{file_path}/{cov_type}_corner_plot_interval_{interval_idx:02d}.html"
        fig.write_html(fname)
        print(f"Saved: {fname}")
        figures.append(fig)

    return figures

if __name__ == "__main__":
    # Load Monte Carlo simulation data
    mc_trajectories = load_in_mc_data(f"{FILE_PATH}/monte_carlo_trajectories.npy")

    four_hour_indices = [i for i, t in enumerate(time_vec) if t % (4 * 3600) == 0]
    kf_time_vec = time_vec[four_hour_indices]

    # Propagate covariance using CKF
    ckf_state_estimate, ckf_covariances = propagate_covariance_ckf(initial_covariance, kf_time_vec)

    # Propagate covariance using UKF
    ukf_state_estimate, ukf_covariances = propagate_covariance_ukf(initial_covariance, kf_time_vec)

    # Make corner plots for CKF and UKF
    ckf_corner_figs = generate_corner_plots(ckf_state_estimate, ckf_covariances, mc_trajectories, four_hour_indices, FILE_PATH, cov_type="CKF")
    ukf_corner_figs = generate_corner_plots(ukf_state_estimate, ukf_covariances, mc_trajectories, four_hour_indices, FILE_PATH, cov_type="UKF")

    # Calculate percentage of Monte Carlo states inside the 2-sigma covariance ellipse for CKF and UKF
    print("Calculating percentage of Monte Carlo states inside the 2-sigma covariance ellipse for CKF...")
    ckf_percentage_inside, ckf_percentage_per_time_step = percentage_of_mc_inside_cov(mc_trajectories, ckf_state_estimate.T, ckf_covariances.transpose(2, 0, 1))
    print("Calculating percentage of Monte Carlo states inside the 2-sigma covariance ellipse for UKF...")
    ukf_percentage_inside, ukf_percentage_per_time_step = percentage_of_mc_inside_cov(mc_trajectories, ukf_state_estimate.T, ukf_covariances.transpose(2, 0, 1))

    # Plot covariance ellipses and Monte Carlo scatter points
    # Plot CKF and UKF covariance ellipses with Monte Carlo samples
    fig = plot_covariance_ellipses(ckf_state_estimate, ckf_covariances,
                                   ukf_state_estimate, ukf_covariances,
                                   kf_time_vec, mc_trajectories, FILE_PATH)
    fig.write_html(f"{FILE_PATH}/ckf_ukf_covariance_ellipses.html")
    print(f"Saved: {FILE_PATH}/ckf_ukf_covariance_ellipses.html")
    
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

    # Write CKF and UKF state estimates and covariances to text files
    with open(f"{FILE_PATH}/ckf_results.txt", "w") as f:
        f.write("CKF Standard Deviations at each time step:\n")
        for i, time in enumerate(kf_time_vec):
            std_devs = np.sqrt(np.diag(ckf_covariances[:, :, i]))
            f.write(f"Time: {time/3600:.2f} hrs, Std Dev: {std_devs}\n")
        f.write("CKF Nominal Value Estimates at each time step:\n")
        for i, time in enumerate(kf_time_vec):
            f.write(f"Time: {time/3600:.2f} hrs, Nominal Value: {ckf_state_estimate[:, i]}\n")
        f.write("CKF Mean Value Estimates at each time step:\n")
        for i, time in enumerate(kf_time_vec):
            f.write(f"Time: {time/3600:.2f} hrs, Mean: {ckf_state_estimate[:, i]}\n")

    with open(f"{FILE_PATH}/ukf_results.txt", "w") as f:
        f.write("UKF Standard Deviations at each time step:\n")
        for i, time in enumerate(kf_time_vec):
            std_devs = np.sqrt(np.diag(ukf_covariances[:, :, i]))
            f.write(f"Time: {time/3600:.2f} hrs, Std Dev: {std_devs}\n")
        f.write("UKF Nominal Value Estimates at each time step:\n")
        for i, time in enumerate(kf_time_vec):
            f.write(f"Time: {time/3600:.2f} hrs, Nominal Value: {ckf_state_estimate[:, i]}\n")
        f.write("UKF Mean Value Estimates at each time step:\n")
        for i, time in enumerate(kf_time_vec):
            f.write(f"Time: {time/3600:.2f} hrs, Mean: {ukf_state_estimate[:, i]}\n")


