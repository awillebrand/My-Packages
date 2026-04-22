import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from generic_functions import load_truth_data, load_measurement_data, convert_measurements_to_df, interpolate_truth_to_measurement_times, initialize_integrator
from B_plane_functions import perform_B_plane_analysis
from Tools.plotting_functions import plot_state_errors, plot_residuals
from Tools.measurement_manager import MeasurementMgr
from Tools.integrator import Integrator
from Tools.LKF import LKF
from Tools.batch_lls_estimator import BatchLLSEstimator
from Tools.SRIF import SRIF
from Tools.EKF import EKF
from constants import known_dynamics_measurement_file_path, truth_data_file_path, task_2_station_locations, testing_a_priori_state, testing_a_priori_covariance, observation_noise
from constants import mu_sun, mu_earth, R_e, solar_flux, SRP_area_to_mass, initial_epoch, initial_epoch_jd, earth_spin_rate, initial_spin_angle, testing_C_r, B_plane_target_coords

def run_task_2_analysis(period_of_data, filters_to_run, iterations_for_filters, tol_for_filters, ekf_start_mode, ekf_start_length):
     # Load truth data and measurement data
    truth_data = load_truth_data(truth_data_file_path)
    measurement_data = load_measurement_data(known_dynamics_measurement_file_path)
    measurement_df = convert_measurements_to_df(measurement_data, station_names=list(task_2_station_locations.keys()), period_of_data=period_of_data)

    meas_time_vector = measurement_df['time'].values
    truth_time_vector = truth_data['time_vector']

    station_mgrs = []
    for station_name, station_info in task_2_station_locations.items():
        mgr = MeasurementMgr(
            station_name,
            station_lat=station_info['lat'],
            station_lon=station_info['lon'],
            initial_earth_spin_angle=initial_spin_angle,
            earth_spin_rate=earth_spin_rate,
            R_e=station_info['radius']
        )
        
        station_mgrs.append(mgr)

    # Initialize Integrator with known dynamics (e.g., two-body problem)
    integrator = Integrator(
        mu=mu_earth,
        R_e=R_e,        
        dynamical_mode=['mu', 'SRP', 'Third Body'],  # Include 2-body and SRP effects for this test
        estimation_mode=['SRP'],
        parameter_indices=[6],
        Cr=testing_C_r,
        srp_area_to_mass=SRP_area_to_mass,  # Use the area-to-mass ratio from constants
        solar_flux=solar_flux,
        number_of_stations=0,
        mu_third_body=mu_sun,
        central_body='Earth',
        third_body='Sun',
        initial_epoch_jd=initial_epoch_jd,
        initial_epoch=initial_epoch,
        earth_spin_rate=earth_spin_rate
    )

    state_length = len(testing_a_priori_state)

    for filter_to_run, max_iterations, tol in zip(filters_to_run, iterations_for_filters, tol_for_filters):
        if filter_to_run == 'Batch':
            print("=" * 50)
            print("Running Batch LLS Estimator...")
            print("=" * 50, end='\n')
            filter = BatchLLSEstimator(integrator, station_mgrs, initial_earth_spin_angle=0, earth_rotation_rate=earth_spin_rate)
            x, P, residuals_df = filter.estimate_initial_state(testing_a_priori_state, measurement_df, observation_noise, a_priori_covariance=testing_a_priori_covariance, max_iterations=max_iterations, tol=tol)
            print("=" * 50)
            print("Batch LLS Estimation Complete...")
            print("=" * 50, end='\n')
            # Integrate the estimated initial state forward in time to compare to truth data
            _, augmented_x_hist = integrator.integrate_stm(meas_time_vector[-1], x[0:state_length], teval=meas_time_vector)
            x_hist = augmented_x_hist[:state_length, :]  # Extract the state history from the augmented state history
            STM_hist = augmented_x_hist[state_length:, :]  # Extract the STM history from the augmented state history
            P_hist = np.zeros((state_length,state_length, len(meas_time_vector)))  # Initialize an array to hold the covariance history
            for i in range(len(meas_time_vector)):
                STM = STM_hist[:, i].reshape((state_length, state_length))  # Reshape the STM from the augmented state history
                P_hist[:, :, i] = STM @ P @ STM.T  # Propagate the covariance using the STM

        elif filter_to_run == 'LKF':
            print("=" * 50)
            print("Running LKF...")
            print("=" * 50, end='\n')
            filter = LKF(integrator, station_mgrs, initial_earth_spin_angle=0, earth_rotation_rate=earth_spin_rate)
            x_hist, P_hist, residuals_df = filter.run(testing_a_priori_state, np.zeros(state_length), testing_a_priori_covariance, measurement_df, R=observation_noise, max_iterations=max_iterations, convergence_threshold=tol)
            print("=" * 50)
            print("LKF Run Complete...")
            print("=" * 50, end='\n')
        elif filter_to_run == 'EKF':
            filter = EKF(integrator, station_mgrs, initial_earth_spin_angle=0, earth_rotation_rate=earth_spin_rate)
            x_hist, P_hist, residuals_df = filter.run(testing_a_priori_state, np.zeros(state_length), testing_a_priori_covariance, measurement_df, R=observation_noise, start_mode=ekf_start_mode.lower(), start_length=ekf_start_length)
        elif filter_to_run == 'SRIF':
            print("=" * 50)
            print("Running SRIF...")
            print("=" * 50, end='\n')
            filter = SRIF(integrator, station_mgrs, initial_earth_spin_angle=0, earth_rotation_rate=earth_spin_rate)
            x_hist, P_hist, residuals_df = filter.run(testing_a_priori_state, np.zeros(state_length), testing_a_priori_covariance, measurement_df, R_noise=observation_noise, max_iterations=max_iterations, tolerance=tol)
            print("=" * 50)
            print("SRIF Run Complete...")
            print("=" * 50, end='\n')
        # Pull state estimates for first 50 days to compare to truth data
        day_50_idx = np.searchsorted(meas_time_vector, 50*24*3600)  # Find index corresponding to 50 days in seconds

        # Interpolate truth data to measurement time vector for first 50 days
        if filter_to_run == 'Batch':
            _, augmented_x_hist = integrator.integrate_stm(truth_time_vector[-1], x[0:state_length], teval=truth_time_vector)
            x_hist_50days = augmented_x_hist  # Extract the state history from the augmented state history
            STM_hist = augmented_x_hist[state_length:, :]  # Extract the STM history from the augmented state history
            P_hist_50days = np.zeros((state_length,state_length, len(truth_time_vector)))  # Initialize an array to hold the covariance history
            for i in range(len(truth_time_vector)):
                STM = STM_hist[:, i].reshape((state_length, state_length))  # Reshape the STM from the augmented state history
                P_hist_50days[:, :, i] = STM @ P @ STM.T  # Propagate the covariance using the STM
            interpolated_truth_state_vectors = truth_data['state_vectors']

            # Compute state estimation errors for first 50 days
            estimation_errors = x_hist_50days[0:6, :] - interpolated_truth_state_vectors[:, 0:6].T
            plot_state_errors(truth_time_vector, estimation_errors, P_hist_50days, filter_name=filter_to_run, file_directory='ASEN_6080/Project2/final_figures')

            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02, subplot_titles=['X Position', 'Y Position', 'Z Position'])
            for i in range(3):
                fig.add_trace(go.Scatter(x=truth_time_vector, y=truth_data['state_vectors'][:,i], mode='lines', name='Truth'), row=i+1, col=1)
                fig.add_trace(go.Scatter(x=truth_time_vector, y=x_hist_50days[i,:], mode='lines', name='Estimated'), row=i+1, col=1)
            fig.update_layout(title='Comparison of Estimated Trajectory to Truth Data for First 50 Days', xaxis_title='Time (s)', yaxis_title='Position (km)')
            fig.write_html(f"ASEN_6080/Project2/final_figures/trajectory_comparison_{filter_to_run}.html")
        else:
            x_hist_50days = x_hist[:, :day_50_idx]
            P_hist_50days = P_hist[:, :, :day_50_idx]
            interpolated_truth_state_vectors = interpolate_truth_to_measurement_times(truth_data, meas_time_vector)

            # Compute state estimation errors for first 50 days
            estimation_errors = x_hist_50days - interpolated_truth_state_vectors

            plot_state_errors(meas_time_vector[:day_50_idx], estimation_errors, P_hist_50days, filter_name=filter_to_run, file_directory='ASEN_6080/Project2/final_figures')

            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02, subplot_titles=['X Position', 'Y Position', 'Z Position'])
            for i in range(3):
                fig.add_trace(go.Scatter(x=truth_time_vector, y=truth_data['state_vectors'][:,i], mode='lines', name='Truth'), row=i+1, col=1)
                fig.add_trace(go.Scatter(x=meas_time_vector[:day_50_idx], y=x_hist_50days[i,:], mode='lines', name='Estimated'), row=i+1, col=1)
            fig.update_layout(title='Comparison of Estimated Trajectory to Truth Data for First 50 Days', xaxis_title='Time (s)', yaxis_title='Position (km)')
            fig.write_html(f"ASEN_6080/Project2/final_figures/trajectory_comparison_{filter_to_run}.html")

def run_task_2_B_plane_analysis(filters_to_run, iterations_for_filters=None, tol_for_filters=None, ekf_start_mode='warm', ekf_start_length=100, process_noise=None, Q=None):
    if iterations_for_filters is None:
        iterations_for_filters = [5] * len(filters_to_run)
    if tol_for_filters is None:
        tol_for_filters = [1e-6] * len(filters_to_run)

    # Setup Station Managers
    station_mgrs = []
    for station_name, station_info in task_2_station_locations.items():
        mgr = MeasurementMgr(
            station_name,
            station_lat=station_info['lat'],
            station_lon=station_info['lon'],
            initial_earth_spin_angle=initial_spin_angle,
            earth_spin_rate=earth_spin_rate,
            R_e=station_info['radius']
        )
        station_mgrs.append(mgr)

    state_length = len(testing_a_priori_state)

    times_in_consideration = [50, 100, 150, 200]
    colors = ['blue', 'green', 'red', 'purple']

    for filter_name, max_iterations, tol in zip(filters_to_run,
                                                 iterations_for_filters,
                                                 tol_for_filters):
        b_plane_fig = go.Figure()
        # Add the true B-plane target (B·T, B·R)
        b_plane_fig.add_trace(go.Scatter(
            x=[B_plane_target_coords[0]],
            y=[B_plane_target_coords[1]],
            mode='markers',
            name='True B-plane Target',
            marker=dict(color='orange', size=12, symbol='star')
        ))

        for i, days in enumerate(times_in_consideration):
            # Load and trim measurements to `days` days
            measurement_data = load_measurement_data(known_dynamics_measurement_file_path)
            measurement_df = convert_measurements_to_df(
                measurement_data,
                station_names=list(task_2_station_locations.keys()),
                period_of_data=[0, days]
            )
            meas_time_vector = measurement_df['time'].values


            # Initialize integrator
            integrator = initialize_integrator(
                starting_epoch=initial_epoch,
                estimation_mode=['SRP'],
                parameter_indices=[6],
                input_C_r=testing_C_r
            )

            if filter_name == 'Batch':
                print("=" * 50)
                print(f"Running Batch LLS Estimator ({days} days)...")
                print("=" * 50)
                filter = BatchLLSEstimator(integrator, station_mgrs,
                                           initial_earth_spin_angle=0,
                                           earth_rotation_rate=earth_spin_rate)
                x, P, residuals_df = filter.estimate_initial_state(
                    testing_a_priori_state,
                    measurement_df,
                    observation_noise,
                    a_priori_covariance=testing_a_priori_covariance,
                    max_iterations=max_iterations,
                    tol=tol
                )
                # Propagate state and covariance history
                _, augmented_x_hist = integrator.integrate_stm(
                    meas_time_vector[-1], x[:state_length], teval=meas_time_vector
                )
                x_hist = augmented_x_hist[:state_length, :]
                STM_hist = augmented_x_hist[state_length:, :]
                P_hist = np.zeros((state_length, state_length, len(meas_time_vector)))
                for j in range(len(meas_time_vector)):
                    STM = STM_hist[:, j].reshape((state_length, state_length))
                    P_hist[:, :, j] = STM @ P @ STM.T

            elif filter_name == 'LKF':
                print("=" * 50)
                print(f"Running LKF ({days} days)...")
                print("=" * 50)
                filter = LKF(integrator, station_mgrs,
                             initial_earth_spin_angle=0,
                             earth_rotation_rate=earth_spin_rate)
                x_hist, P_hist, residuals_df = filter.run(testing_a_priori_state,
                                                          np.zeros(state_length),
                                                          testing_a_priori_covariance,
                                                          measurement_df,
                                                          R=observation_noise,
                                                          max_iterations=max_iterations,
                                                          convergence_threshold=tol,
                                                          process_noise_approach=process_noise,
                                                          Q=Q
                                                          )

            elif filter_name == 'EKF':
                print("=" * 50)
                print(f"Running EKF ({days} days)...")
                print("=" * 50)
                filter = EKF(integrator, station_mgrs,
                             initial_earth_spin_angle=0,
                             earth_rotation_rate=earth_spin_rate)
                x_hist, P_hist, residuals_df = filter.run(testing_a_priori_state,
                                                          np.zeros(state_length),
                                                          testing_a_priori_covariance,
                                                          measurement_df,
                                                          R=observation_noise,
                                                          start_mode=ekf_start_mode.lower(),
                                                          start_length=ekf_start_length,
                                                          process_noise_approach=process_noise,
                                                          Q=Q
                                                        )

            elif filter_name == 'SRIF':
                print("=" * 50)
                print(f"Running SRIF ({days} days)...")
                print("=" * 50)
                filter = SRIF(integrator, station_mgrs,
                              initial_earth_spin_angle=0,
                              earth_rotation_rate=earth_spin_rate)
                x_hist, P_hist, residuals_df = filter.run(testing_a_priori_state,
                                                          np.zeros(state_length),
                                                          testing_a_priori_covariance,
                                                          measurement_df,
                                                          R_noise=observation_noise,
                                                          max_iterations=max_iterations,
                                                          tolerance=tol
                                                      )

            else:
                raise ValueError(f"Unknown filter: {filter_name}. Choose from Batch, LKF, EKF, SRIF.")

            # -------------------------------------------------------------------
            # Deliverable (h): B-plane ellipse for this time period
            # -------------------------------------------------------------------
            DCO_state = x_hist[:, -1]
            DCO_covariance = P_hist[:, :, -1]
            DCO_C_r = DCO_state[6]

            perform_B_plane_analysis(DCO_state, meas_time_vector[-1], DCO_covariance, DCO_C_r, b_plane_fig, colors[i], days, filter_name)

            if days == 200:
                # (f) 3-sigma covariance envelopes for all 7 states
                state_labels = ['X (km)', 'Y (km)', 'Z (km)',
                                 'Vx (km/s)', 'Vy (km/s)', 'Vz (km/s)', 'C_R']
                fig_cov = make_subplots(rows=7, cols=1,
                                        subplot_titles=state_labels,
                                        shared_xaxes=False)
                time_days = meas_time_vector / 86400
                for k in range(state_length):
                    row = k+1
                    col = 1
                    sigma = 3 * np.sqrt(np.abs(P_hist[k, k, :]))
                    fig_cov.add_trace(go.Scatter(
                        x=time_days,
                        y=sigma,
                        mode='lines', line=dict(color='red', dash='dash'),
                        name='3σ Bounds', showlegend=(k == 0)
                    ), row=row, col=col)
                    fig_cov.add_trace(go.Scatter(
                        x=time_days,
                        y=-sigma,
                        mode='lines', line=dict(color='red', dash='dash'),
                        name='3σ lower', showlegend=False
                    ), row=row, col=col)

                fig_cov.update_layout(
                    title=dict(text=f'3-σ Covariance Envelopes ({filter_name} with SNC)', font=dict(size=20)),
                    height=1200, width=1000,
                    legend=dict(font=dict(size=18), x=0.85, y=1.08)
                )
                fig_cov.update_annotations(font_size=20)
                for k in range(state_length):
                    row = k + 1
                    col = 1
                    fig_cov.update_xaxes(title=dict(text='Time (days)', font=dict(size=18)), row=row, col=col)
                    fig_cov.update_yaxes(title=dict(text=state_labels[k], font=dict(size=18)), showexponent="all", exponentformat="e", row=row, col=col)
                if process_noise is not None:
                    fig_cov.write_html(f'ASEN_6080/Project2/final_figures/task2_covariance_envelopes_{filter_name}_SNC.html')
                else:
                    fig_cov.write_html(f'ASEN_6080/Project2/final_figures/task2_covariance_envelopes_{filter_name}.html')

                # (g) Pre-fit and post-fit residuals
                plot_residuals(meas_time_vector, residuals_df, filter_name=f'{filter_name} with SNC', file_directory='ASEN_6080/Project2/final_figures/', auto_save=True, omit_outliers=False)

        # Save B-plane figure for this filter
        b_plane_fig.update_layout(title=f'B-plane Analysis for {filter_name} Filter',
                                  xaxis_title='B·T (km)',
                                  yaxis_title='B·R (km)',
                                  width=800,
                                  height=600,
                                  yaxis=dict(autorange='reversed'))
        if process_noise is not None:
            b_plane_fig.write_html(f'ASEN_6080/Project2/final_figures/task2_B_plane_{filter_name}_SNC.html')
            b_plane_fig.write_image(f'ASEN_6080/Project2/final_figures/task2_B_plane_{filter_name}_SNC.png')
        else:
            b_plane_fig.write_html(f'ASEN_6080/Project2/final_figures/task2_B_plane_{filter_name}.html')
            b_plane_fig.write_image(f'ASEN_6080/Project2/final_figures/task2_B_plane_{filter_name}.png')
        print(f"\nTask 2 figures saved for filter: {filter_name}")