import numpy as np
import pandas as pd
import ast
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from Tools.adaptive_snc import AdaptiveSNC
from Tools.plotting_functions import plot_residuals, plot_state_errors
from Tools.generic_functions import covariance_ellipse_2D
from Tools.EKF import EKF
from generic_functions import load_measurement_data, convert_measurements_to_df, initialize_integrator
from constants import unknown_dynamics_measurement_file_path, task_3_station_locations, initial_spin_angle, earth_spin_rate, observation_noise
# Generic imports
from constants import observation_noise, initial_epoch, initial_epoch_jd, a_priori_covariance

# Import analysis scripts
from generic_functions import load_measurement_data, load_truth_data, convert_measurements_to_df, initialize_integrator
from task_1_analysis import run_task_1_analysis
from task_2_analysis import run_task_2_analysis, run_task_2_B_plane_analysis
from task_3_analysis import run_task_3_analysis

np.set_printoptions(linewidth=200)

if __name__ == "__main__":
    # ----------------------------------------------------------------------------------------------------------------------------
    # TASK 1 ANALYSIS
    # ----------------------------------------------------------------------------------------------------------------------------
    # print("=" * 50)
    # print(f"Running Task 1 Dynamics Test...")
    # print("=" * 50 + "\n")
    # run_task_1_analysis()

    # ----------------------------------------------------------------------------------------------------------------------------
    # TASK 2 ANALYSIS
    # ----------------------------------------------------------------------------------------------------------------------------
    # print("\n" + "=" * 50)
    # print(f"Running Task 2 Known Dynamics Estimation...")
    # print("=" * 50 + "\n")
    # # run_task_2_analysis(period_of_data=[0, 51],
    # #                     filters_to_run=['Batch', 'LKF', 'EKF', 'SRIF'],
    # #                     iterations_for_filters=[5, 5, 1, 5],
    # #                     tol_for_filters=[1e-6, 1e-6, 1e-6, 1e-6],
    # #                     ekf_start_mode='warm',
    # #                     ekf_start_length=100)
    # run_task_2_B_plane_analysis(filters_to_run=['EKF'], process_noise='SNC', Q=np.diag([1e-22, 1e-22, 1e-22]))

    # ----------------------------------------------------------------------------------------------------------------------------
    # TASK 3 GENERAL ANALYSIS
    # ----------------------------------------------------------------------------------------------------------------------------

    print("=" * 50)
    print(f"Running Task 3 Final Analysis...")
    print("=" * 50)

    # # Start by showing approach when determining what is wrong with the model

    # # Measurement data time period to use (in days)
    # period_of_data = [0, 50]

    # # Estimation Parameters
    # estimation_mode = ['SRP']
    # parameter_indices = [6]

    # DSS_34_cov = [1e-16, 1e-16, 1e-16]
    # DSS_65_cov = [0.1, 0.1, 0.1]
    # DSS_13_cov = [1e-16, 1e-16, 1e-16]

    # # EKF start mode parameters
    # start_mode = 'warm'
    # start_length = 200

    # # Covariance reset parameters
    # mnvr_day = 217
    # mnvr_time = mnvr_day * 24 * 3600
    # mnvr_reset_covariance = np.diag([10, 10, 10, 0.01, 0.01, 0.01, 0.05, 1e-8, 1e-8, 1e-8, 1e-3, 1e-3, 1e-3, 1e-8, 1e-8, 1e-8])**2

    # # SNC parameters
    # process_noise_type = 'None'
    # Q = np.diag([5e-10, 5e-10, 5e-10])**2

    # # Adapative SNC parameters
    # alpha = 0.005
    # window = 10
    # Q_adaptive = 5e-7 # 1 mm/s adaptive process noise for velocity states

    # filter_name = f'LKF (Smoothed with Base Model)'

    # x_hist, P_hist, residuals_df = run_task_3_analysis(period_of_data,
    #                                                    estimation_mode,
    #                                                    parameter_indices,
    #                                                    DSS_34_cov,
    #                                                    DSS_65_cov,
    #                                                    DSS_13_cov,
    #                                                    start_mode,
    #                                                    start_length,
    #                                                    mnvr_day,
    #                                                    mnvr_reset_covariance,
    #                                                    process_noise_type,
    #                                                    Q,
    #                                                    alpha,
    #                                                    window,
    #                                                    Q_adaptive,
    #                                                    filter_name,
    #                                                    estimate_b_plane=False)
                                                       
    # # Now estimating DSS65 position with iterated and smoothed LKF
    # # Start by showing approach when determining what is wrong with the model

    # # Measurement data time period to use (in days)
    # period_of_data = [0, 50]

    # # Estimation Parameters
    # estimation_mode = ['SRP', 'Stations']
    # parameter_indices = [6, 7]

    # DSS_34_cov = [1e-16, 1e-16, 1e-16]
    # DSS_65_cov = [0.1, 0.1, 0.1]
    # DSS_13_cov = [1e-16, 1e-16, 1e-16]

    # # EKF start mode parameters
    # start_mode = 'warm'
    # start_length = 200

    # # Covariance reset parameters
    # mnvr_day = 217
    # mnvr_time = mnvr_day * 24 * 3600
    # mnvr_reset_covariance = np.diag([10, 10, 10, 0.01, 0.01, 0.01, 0.05, 1e-8, 1e-8, 1e-8, 1e-3, 1e-3, 1e-3, 1e-8, 1e-8, 1e-8])**2

    # # SNC parameters
    # process_noise_type = 'None'
    # Q = np.diag([5e-10, 5e-10, 5e-10])**2

    # # Adapative SNC parameters
    # alpha = 0.005
    # window = 10
    # Q_adaptive = 5e-7 # 1 mm/s adaptive process noise for velocity states

    # filter_name = f'LKF (50 Days Smoothed with DSS65 Estimated)'

    # x_hist, P_hist, residuals_df = run_task_3_analysis(period_of_data,
    #                                                    estimation_mode,
    #                                                    parameter_indices,
    #                                                    DSS_34_cov,
    #                                                    DSS_65_cov,
    #                                                    DSS_13_cov,
    #                                                    start_mode,
    #                                                    start_length,
    #                                                    mnvr_day,
    #                                                    mnvr_reset_covariance,
    #                                                    process_noise_type,
    #                                                    Q,
    #                                                    alpha,
    #                                                    window,
    #                                                    Q_adaptive,
    #                                                    filter_name,
    #                                                    estimate_b_plane=False)
    
    # Switching to 50 day estimation with the EKF to see if residuals improve

    # # Measurement data time period to use (in days)
    # period_of_data = [0, 100]

    # # Estimation Parameters
    # estimation_mode = ['SRP', 'Stations']
    # parameter_indices = [6, 7]

    # DSS_34_cov = [1e-16, 1e-16, 1e-16]
    # DSS_65_cov = [0.1, 0.1, 0.1]
    # DSS_13_cov = [1e-16, 1e-16, 1e-16]

    # # EKF start mode parameters
    # start_mode = 'warm'
    # start_length = 200

    # # Covariance reset parameters
    # mnvr_day = 217
    # mnvr_time = mnvr_day * 24 * 3600
    # mnvr_reset_covariance = np.diag([10, 10, 10, 0.01, 0.01, 0.01, 0.05, 1e-8, 1e-8, 1e-8, 1e-3, 1e-3, 1e-3, 1e-8, 1e-8, 1e-8])**2

    # # SNC parameters
    # process_noise_type = 'None'
    # Q = np.diag([5e-10, 5e-10, 5e-10])**2

    # # Adapative SNC parameters
    # alpha = 0.005
    # window = 10
    # Q_adaptive = 5e-7 # 1 mm/s adaptive process noise for velocity states

    # filter_name = f'LKF (100 Days Smoothed with DSS65 Estimated)'

    # x_hist, P_hist, residuals_df = run_task_3_analysis(period_of_data,
    #                                                    estimation_mode,
    #                                                    parameter_indices,
    #                                                    DSS_34_cov,
    #                                                    DSS_65_cov,
    #                                                    DSS_13_cov,
    #                                                    start_mode,
    #                                                    start_length,
    #                                                    mnvr_day,
    #                                                    mnvr_reset_covariance,
    #                                                    process_noise_type,
    #                                                    Q,
    #                                                    alpha,
    #                                                    window,
    #                                                    Q_adaptive,
    #                                                    filter_name,
    #                                                    estimate_b_plane=False)

    # # Adding SNC to EKF for 100 day estimation with DSS65 estimation to see if residuals improve and how covariance changes
    # Measurement data time period to use (in days)
    # period_of_data = [0, 100]

    # # Estimation Parameters
    # estimation_mode = ['SRP', 'Stations']
    # parameter_indices = [6, 7]

    # DSS_34_cov = [1e-16, 1e-16, 1e-16]
    # DSS_65_cov = [0.1, 0.1, 0.1]
    # DSS_13_cov = [1e-16, 1e-16, 1e-16]

    # # EKF start mode parameters
    # start_mode = 'warm'
    # start_length = 200

    # # Covariance reset parameters
    # mnvr_day = 217
    # mnvr_time = mnvr_day * 24 * 3600
    # mnvr_reset_covariance = np.diag([10, 10, 10, 0.01, 0.01, 0.01, 0.05, 1e-8, 1e-8, 1e-8, 1e-3, 1e-3, 1e-3, 1e-8, 1e-8, 1e-8])**2

    # # SNC parameters
    # process_noise_type = 'SNC'
    # Q = np.diag([5e-10, 5e-10, 5e-10])**2

    # # Adapative SNC parameters
    # alpha = 0.005
    # window = 10
    # Q_adaptive = 5e-7 # 1 mm/s adaptive process noise for velocity states

    # filter_name = f'EKF (100 Days with DSS65 Estimated and SNC)'

    # x_hist, P_hist, residuals_df, time_vector = run_task_3_analysis(period_of_data,
    #                                                    estimation_mode,
    #                                                    parameter_indices,
    #                                                    DSS_34_cov,
    #                                                    DSS_65_cov,
    #                                                    DSS_13_cov,
    #                                                    start_mode,
    #                                                    start_length,
    #                                                    mnvr_day,
    #                                                    mnvr_reset_covariance,
    #                                                    process_noise_type,
    #                                                    Q,
    #                                                    alpha,
    #                                                    window,
    #                                                    Q_adaptive,
    #                                                    filter_name,
    #                                                    estimate_b_plane=False)
    # df = pd.DataFrame({'time': time_vector})
    # df.to_pickle(f'ASEN_6080/Project2/data/time_vector_{filter_name}.pkl')
    # residuals_df.to_pickle(f'ASEN_6080/Project2/data/residuals_df_{filter_name}.pkl')

    # df_loaded = pd.read_pickle(f'ASEN_6080/Project2/data/time_vector_{filter_name}.pkl')
    # residuals_df_loaded = pd.read_pickle(f'ASEN_6080/Project2/data/residuals_df_{filter_name}.pkl')
    # plot_residuals(df_loaded['time'], residuals_df_loaded, filter_name=f"{filter_name}_no_outliers", file_directory=f'ASEN_6080/Project2/final_figures/', auto_save=True, omit_outliers=True)

    # Estimating for 247 days with DSS65 estimation and SNC to see how residuals evolve over longer time period and how covariance changes
    # Measurement data time period to use (in days)
    # period_of_data = [0, 250]

    # # Estimation Parameters
    # estimation_mode = ['SRP', 'Stations']
    # parameter_indices = [6, 7]

    # DSS_34_cov = [1e-16, 1e-16, 1e-16]
    # DSS_65_cov = [0.1, 0.1, 0.1]
    # DSS_13_cov = [1e-16, 1e-16, 1e-16]

    # # EKF start mode parameters
    # start_mode = 'warm'
    # start_length = 200

    # # Covariance reset parameters
    # mnvr_day = None
    # mnvr_time = None
    # mnvr_reset_covariance = np.diag([10, 10, 10, 0.01, 0.01, 0.01, 0.05, 1e-8, 1e-8, 1e-8, 1e-3, 1e-3, 1e-3, 1e-8, 1e-8, 1e-8])**2

    # # SNC parameters
    # process_noise_type = 'SNC'
    # Q = np.diag([5e-10, 5e-10, 5e-10])**2

    # # Adapative SNC parameters
    # alpha = 0.005
    # window = 10
    # Q_adaptive = 5e-7 # 1 mm/s adaptive process noise for velocity states

    # filter_name = f'EKF (247 Days with DSS65 Estimated and SNC)'

    # x_hist, P_hist, residuals_df, time_vector = run_task_3_analysis(period_of_data,
    #                                                                 estimation_mode,
    #                                                                 parameter_indices,
    #                                                                 DSS_34_cov,
    #                                                                 DSS_65_cov,
    #                                                                 DSS_13_cov,
    #                                                                 start_mode,
    #                                                                 start_length,
    #                                                                 mnvr_day,
    #                                                                 mnvr_reset_covariance,
    #                                                                 process_noise_type,
    #                                                                 Q,
    #                                                                 alpha,
    #                                                                 window,
    #                                                                 Q_adaptive,
    #                                                                 filter_name,
    #                                                                 estimate_b_plane=True)
    # # Also save time vector to use for plotting residuals later
    # df = pd.DataFrame({'time': time_vector})
    # df.to_pickle(f'ASEN_6080/Project2/data/time_vector_{filter_name}.pkl')
    # residuals_df.to_pickle(f'ASEN_6080/Project2/data/residuals_df_{filter_name}.pkl')

    # # Read back in df and residuals to verify they were saved correctly
    # # df_loaded = pd.read_pickle(f'ASEN_6080/Project2/data/time_vector_{filter_name}.pkl')
    # # residuals_df_loaded = pd.read_pickle(f'ASEN_6080/Project2/data/residuals_df_{filter_name}.pkl')

    # # ----------------------------------------------------------------------------------------------------------------------------
    # # TASK 3 FINAL ANALYSIS
    # # ----------------------------------------------------------------------------------------------------------------------------
    # print("=" * 50)
    # print(f"Running Task 3 Final Analysis...")
    # print("=" * 50)

    # # Measurement data time period to use (in days)
    # period_of_data = [0, 250]

    # # Estimation Parameters
    # estimation_mode = ['SRP', 'Stations']
    # parameter_indices = [6, 7]

    # DSS_34_cov = [1e-16, 1e-16, 1e-16]
    # DSS_65_cov = [0.1, 0.1, 0.1]
    # DSS_13_cov = [1e-16, 1e-16, 1e-16]

    # # EKF start mode parameters
    # start_mode = 'warm'
    # start_length = 200

    # # Covariance reset parameters
    # mnvr_day = 217
    # mnvr_time = mnvr_day * 24 * 3600
    # mnvr_reset_covariance = np.diag([10, 10, 10, 0.01, 0.01, 0.01, 0.05, 1e-8, 1e-8, 1e-8, 1e-3, 1e-3, 1e-3, 1e-8, 1e-8, 1e-8])**2

    # # SNC parameters
    # process_noise_type = 'SNC'
    # Q = np.diag([5e-10, 5e-10, 5e-10])**2

    # # Adapative SNC parameters
    # alpha = 0.005
    # window = 10
    # Q_adaptive = 5e-7 # 1 mm/s adaptive process noise for velocity states

    # filter_name = f'EKF (DSS65 Estimated, SNC, and Covariance Reset)'

    # x_hist, P_hist, residuals_df, time_vector = run_task_3_analysis(period_of_data,
    #                                                                 estimation_mode,
    #                                                                 parameter_indices,
    #                                                                 DSS_34_cov,
    #                                                                 DSS_65_cov,
    #                                                                 DSS_13_cov,
    #                                                                 start_mode,
    #                                                                 start_length,
    #                                                                 mnvr_day,
    #                                                                 mnvr_reset_covariance,
    #                                                                 process_noise_type,
    #                                                                 Q,
    #                                                                 alpha,
    #                                                                 window,
    #                                                                 Q_adaptive,
    #                                                                 filter_name,
    #                                                                 estimate_b_plane=True)
    # df = pd.DataFrame({'time': time_vector})
    # df.to_pickle(f'ASEN_6080/Project2/data/time_vector_{filter_name}.pkl')
    # residuals_df.to_pickle(f'ASEN_6080/Project2/data/residuals_df_{filter_name}.pkl')

    # # # ----------------------------------------------------------------------------------------------------------------------------
    # # # BACKWARD EKF CONSISTENCY CHECK
    # # # ----------------------------------------------------------------------------------------------------------------------------

    # print("=" * 50)
    # print("Running Backward EKF Consistency Check (25-day arc)...")
    # print("=" * 50)

    # BACKWARD_DAYS = 25
    # bwd_period = [period_of_data[1] - BACKWARD_DAYS, period_of_data[1]]  # e.g. [225, 250]

    # bwd_station_mgrs = []
    # for station_name, station_info in task_3_station_locations.items():
    #     from Tools.measurement_manager import MeasurementMgr
    #     mgr = MeasurementMgr(
    #         station_name,
    #         station_lat=station_info['lat'],
    #         station_lon=station_info['lon'],
    #         initial_earth_spin_angle=initial_spin_angle,
    #         earth_spin_rate=earth_spin_rate,
    #         R_e=station_info['radius']
    #     )
    #     bwd_station_mgrs.append(mgr)

    # bwd_integrator = initialize_integrator(
    #     starting_epoch=initial_epoch,
    #     estimation_mode=estimation_mode,
    #     parameter_indices=parameter_indices
    # )

    # # Build the dataframe for the backward EKF by loading the measurement data again and reversing it to run from t_f to t_start
    # measurement_data_bwd = load_measurement_data(unknown_dynamics_measurement_file_path)
    # bwd_measurement_df = convert_measurements_to_df(
    #     measurement_data_bwd,
    #     station_names=list(task_3_station_locations.keys()),
    #     period_of_data=bwd_period
    # )
    # bwd_measurement_df = bwd_measurement_df.iloc[::-1].reset_index(drop=True)

    # # Backward initial conditions
    # state_length = x_hist.shape[0]
    # X_bwd_init   = x_hist[:, -1].copy()
    # x_hat_bwd    = np.zeros(state_length)

    # # Extract the matching forward-filter window over the same 25-day arc.
    # fwd_times       = time_vector
    # bwd_t_min_s     = bwd_period[0] * 24 * 3600
    # fwd_overlap_idx = np.where(fwd_times >= bwd_t_min_s)[0]

    # x_fwd_overlap = x_hist[:, fwd_overlap_idx]
    # P_fwd_overlap = P_hist[:, :, fwd_overlap_idx]
    # overlap_days  = fwd_times[fwd_overlap_idx] / 86400

    # # Set the initial covariance for the backward filter to be large to reflect our lack of knowledge at the start of the backward arc, using the forward filter covariance before the overlap period as a reference
    # P_bwd_init = 1000 * P_hist[:, :, fwd_overlap_idx[0]].copy()

    # # Run EKF
    # bwd_ekf = EKF(bwd_integrator, bwd_station_mgrs,
    #               initial_earth_spin_angle=0,
    #               earth_rotation_rate=earth_spin_rate)

    # x_hist_bwd, P_hist_bwd, _ = bwd_ekf.run(
    #     X_bwd_init,
    #     x_hat_bwd,
    #     P_bwd_init,
    #     bwd_measurement_df,
    #     R=observation_noise,
    #     start_mode='cold',
    #     process_noise_approach=process_noise_type,
    #     Q=Q,
    #     adaptive_snc=None
    # )

    # print("=" * 50)
    # print("Backward EKF Complete.")
    # print("=" * 50)

    # # Flip backward arrays so both run from t_start to t_f
    # x_hist_bwd_aligned = x_hist_bwd[:, ::-1].copy()
    # P_hist_bwd_aligned = P_hist_bwd[:, :, ::-1].copy()

    # # Compute the difference between forward and backward state estimates over the overlapping time period, along with the combined 3-sigma bounds from both filters
    # state_diff = x_fwd_overlap - x_hist_bwd_aligned
    # combined_3_sigma = 3*np.sqrt(
    #     np.diagonal(P_fwd_overlap, axis1=0, axis2=1).T +
    #     np.diagonal(P_hist_bwd_aligned, axis1=0, axis2=1).T
    # )

    # state_labels = ['x (km)', 'y (km)', 'z (km)', 'vx (km/s)', 'vy (km/s)', 'vz (km/s)']

    # # Plot
    # fig_bwd = make_subplots(rows=6, cols=1, shared_xaxes=True, subplot_titles=state_labels)

    # for i in range(6):
    #     fig_bwd.add_trace(
    #         go.Scatter(x=overlap_days, y=state_diff[i, :],
    #                    name=f'fwd - bwd', line=dict(color='blue'),
    #                    showlegend=(i == 0)),
    #         row=i+1, col=1
    #     )
    #     fig_bwd.add_trace(
    #         go.Scatter(x=overlap_days, y=combined_3_sigma[i, :],
    #                    name='combined 3σ bounds', line=dict(color='red', dash='dash'),
    #                    showlegend=(i == 0)),
    #         row=i+1, col=1
    #     )
    #     fig_bwd.add_trace(
    #         go.Scatter(x=overlap_days, y=-combined_3_sigma[i, :],
    #                    name='-3σ combined', line=dict(color='red', dash='dash'),
    #                    showlegend=False),
    #         row=i+1, col=1
    #     )
    #     fig_bwd.update_yaxes(title_text=state_labels[i], row=i+1, col=1)

    # fig_bwd.update_xaxes(title_text='Time (days)', row=6, col=1)
    # fig_bwd.update_layout(
    #     title=f'Forward-Backward EKF Consistency Check — {filter_name} (last {BACKWARD_DAYS} days)',
    #     height=1200
    # )
    # fig_bwd.write_html(f'ASEN_6080/Project2/final_figures/BWD_CHECK_{filter_name}.html')
    # print(f"\nBackward consistency plot saved to: ASEN_6080/Project2/final_figures/BWD_CHECK_{filter_name}.html")

    # ------------------------------------------------------------------------------------
    # ADDITIONAL ANALYSIS: ADAPTIVE SNC
    # ------------------------------------------------------------------------------------

    # period_of_data = [0, 215]

    # # Estimation Parameters
    # estimation_mode = ['SRP', 'Stations']
    # parameter_indices = [6, 7]

    # DSS_34_cov = [1e-16, 1e-16, 1e-16]
    # DSS_65_cov = [0.1, 0.1, 0.1]
    # DSS_13_cov = [1e-16, 1e-16, 1e-16]

    # # EKF start mode parameters
    # start_mode = 'warm'
    # start_length = 200

    # # Covariance reset parameters
    # mnvr_day = 217
    # mnvr_time = mnvr_day * 24 * 3600
    # mnvr_reset_covariance = np.diag([10, 10, 10, 0.01, 0.01, 0.01, 0.05, 1e-8, 1e-8, 1e-8, 1e-3, 1e-3, 1e-3, 1e-8, 1e-8, 1e-8])**2

    # # SNC parameters
    # process_noise_type = 'Adaptive SNC'
    # Q = np.diag([5e-10, 5e-10, 5e-10])**2

    # # Adapative SNC parameters
    # alpha = 0.005
    # window = 10
    # Q_adaptive = 2e-7 # 1 mm/s adaptive process noise for velocity states
    # filter_name = f'EKF (215 Days Adaptive SNC)'

    # x_hist, P_hist, residuals_df, time_vector = run_task_3_analysis(period_of_data,
    #                                                    estimation_mode,
    #                                                    parameter_indices,
    #                                                    DSS_34_cov,
    #                                                    DSS_65_cov,
    #                                                    DSS_13_cov,
    #                                                    start_mode,
    #                                                    start_length,
    #                                                    mnvr_day,
    #                                                    mnvr_reset_covariance,
    #                                                    process_noise_type,
    #                                                    Q,
    #                                                    alpha,
    #                                                    window,
    #                                                    Q_adaptive,
    #                                                    filter_name,
    #                                                    estimate_b_plane=False)

    # df_x_hist = pd.DataFrame(x_hist.T, columns=['x', 'y', 'z', 'vx', 'vy', 'vz', 'Cr', 'DSS34_x', 'DSS34_y', 'DSS34_z', 'DSS65_x', 'DSS65_y', 'DSS65_z', 'DSS13_x', 'DSS13_y', 'DSS13_z'])
    # df_x_hist.to_pickle(f'ASEN_6080/Project2/data/x_hist_{filter_name}.pkl')
    # P_flattened = P_hist.transpose(2, 0, 1).reshape(P_hist.shape[2], -1)    
    # df_P_hist = pd.DataFrame(
    #     P_flattened, 
    #     columns=[f'P_{i}_{j}' for i in range(P_hist.shape[0]) for j in range(P_hist.shape[1])]
    # )
    # df_P_hist.to_pickle(f'ASEN_6080/Project2/data/P_hist_{filter_name}.pkl')
    # df = pd.DataFrame({'time': time_vector})
    # df.to_pickle(f'ASEN_6080/Project2/data/time_vector_{filter_name}.pkl')
    # residuals_df.to_pickle(f'ASEN_6080/Project2/data/residuals_df_{filter_name}.pkl')

    # period_of_data = [0, 250]

    # # Estimation Parameters
    # estimation_mode = ['SRP', 'Stations']
    # parameter_indices = [6, 7]

    # DSS_34_cov = [1e-16, 1e-16, 1e-16]
    # DSS_65_cov = [0.1, 0.1, 0.1]
    # DSS_13_cov = [1e-16, 1e-16, 1e-16]

    # # EKF start mode parameters
    # start_mode = 'warm'
    # start_length = 200

    # # Covariance reset parameters
    # mnvr_day = 217
    # mnvr_time = mnvr_day * 24 * 3600
    # mnvr_reset_covariance = np.diag([10, 10, 10, 0.01, 0.01, 0.01, 0.05, 1e-8, 1e-8, 1e-8, 1e-3, 1e-3, 1e-3, 1e-8, 1e-8, 1e-8])**2

    # # SNC parameters
    # process_noise_type = 'Adaptive SNC'
    # Q = np.diag([5e-10, 5e-10, 5e-10])**2

    # # Adapative SNC parameters
    # alpha = 0.005
    # window = 10
    # Q_adaptive = 2e-7 # 1 mm/s adaptive process noise for velocity states
    # filter_name = f'EKF (250 Days Adaptive SNC)'

    # x_hist, P_hist, residuals_df, time_vector = run_task_3_analysis(period_of_data,
    #                                                    estimation_mode,
    #                                                    parameter_indices,
    #                                                    DSS_34_cov,
    #                                                    DSS_65_cov,
    #                                                    DSS_13_cov,
    #                                                    start_mode,
    #                                                    start_length,
    #                                                    mnvr_day,
    #                                                    mnvr_reset_covariance,
    #                                                    process_noise_type,
    #                                                    Q,
    #                                                    alpha,
    #                                                    window,
    #                                                    Q_adaptive,
    #                                                    filter_name,
    #                                                    estimate_b_plane=True)

    # df_x_hist = pd.DataFrame(x_hist.T, columns=['x', 'y', 'z', 'vx', 'vy', 'vz', 'Cr', 'DSS34_x', 'DSS34_y', 'DSS34_z', 'DSS65_x', 'DSS65_y', 'DSS65_z', 'DSS13_x', 'DSS13_y', 'DSS13_z'])
    # df_x_hist.to_pickle(f'ASEN_6080/Project2/data/x_hist_{filter_name}.pkl')
    # P_flattened = P_hist.transpose(2, 0, 1).reshape(P_hist.shape[2], -1)    
    # df_P_hist = pd.DataFrame(
    #     P_flattened, 
    #     columns=[f'P_{i}_{j}' for i in range(P_hist.shape[0]) for j in range(P_hist.shape[1])]
    # )
    # df_P_hist.to_pickle(f'ASEN_6080/Project2/data/P_hist_{filter_name}.pkl')
    # df = pd.DataFrame({'time': time_vector})
    # df.to_pickle(f'ASEN_6080/Project2/data/time_vector_{filter_name}.pkl')
    # residuals_df.to_pickle(f'ASEN_6080/Project2/data/residuals_df_{filter_name}.pkl')

    # Overlay B-Plane estimation results for adaptive SNC EKF with previous EKF with SNC using the saved covariance information in txt files
    
    def parse_bplane_stats(filepath):
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Split on the labeled sections
        lines = content.strip().split('\n')
        
        # Find and parse the target estimate (single line list)
        target_line = lines[1].strip()
        target = np.array(ast.literal_eval(target_line))
        
        # Find and parse the covariance (multi-line, join and clean first)
        cov_raw = lines[4] + lines[5]
        cov_raw = cov_raw.replace(' ', '').replace('\n', '')
        cov = np.array(ast.literal_eval(cov_raw))
        
        return target, cov

    # Load both files
    target_adaptive, cov_adaptive = parse_bplane_stats('ASEN_6080/Project2/txt_results/b_plane_stats_Adaptive_SNC.txt')
    target_snc, cov_snc = parse_bplane_stats('ASEN_6080/Project2/txt_results/b_plane_stats_SNC.txt')

    b_plane_covariance_ellipse_1 = covariance_ellipse_2D(target_adaptive, cov_adaptive, n_std=3)  # Compute the covariance ellipse at 3-sigma
    b_plane_covariance_ellipse_2 = covariance_ellipse_2D(target_snc, cov_snc, n_std=3)  # Compute the covariance ellipse at 3-sigma

    fig_b_plane = go.Figure()
    fig_b_plane.add_trace(go.Scatter(x=b_plane_covariance_ellipse_1[:, 0], y=b_plane_covariance_ellipse_1[:, 1], mode='lines', name='Adaptive SNC EKF 3σ Ellipse', line=dict(color='blue')))
    fig_b_plane.add_trace(go.Scatter(x=b_plane_covariance_ellipse_2[:, 0], y=b_plane_covariance_ellipse_2[:, 1], mode='lines', name='SNC EKF 3σ Ellipse', line=dict(color='red')))
    fig_b_plane.add_trace(go.Scatter(x=[target_adaptive[0]], y=[target_adaptive[1]], mode='markers', name='Adaptive SNC EKF Target', marker=dict(color='blue', size=10, symbol='x')))
    fig_b_plane.add_trace(go.Scatter(x=[target_snc[0]], y=[target_snc[1]], mode='markers', name='SNC EKF Target', marker=dict(color='red', size=10, symbol='x')))
    fig_b_plane.update_layout(title='B-Plane Target Estimation with Adaptive SNC vs. SNC EKF',
                                    xaxis_title='B·T (km)',
                                    yaxis_title='B·R (km)',
                                    width=800,
                                    height=600,
                                    yaxis=dict(autorange='reversed'))
    fig_b_plane.show('ASEN_6080/Project2/final_figures/B_Plane_Comparison_Adaptive_SNC_vs_SNC.html')
