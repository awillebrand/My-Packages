import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import scipy.io
from Tools.measurement_manager import MeasurementMgr
from Tools.integrator import Integrator
from Tools.EKF import EKF
from Tools.adaptive_snc import AdaptiveSNC
from Tools.plotting_functions import plot_residuals, plot_state_errors

# Generic imports
from constants import observation_noise, initial_epoch, initial_epoch_jd
# from constants import unknown_dynamics_measurement_file_path, known_dynamics_measurement_file_path, truth_data_file_path, a_priori_state, a_priori_covariance, observation_noise, initial_spin_angle, earth_spin_rate, station_locations
# from constants import C_r, mu_sun, mu_earth, R_e, solar_flux, SRP_area_to_mass, initial_spin_angle, earth_spin_rate

# Import analysis scripts
from generic_functions import load_measurement_data, load_truth_data, convert_measurements_to_df, initialize_integrator
from task_1_analysis import run_task_1_analysis
from task_2_analysis import run_task_2_analysis

np.set_printoptions(linewidth=200)

# def run_task_3_analysis(period_of_data,
#                         estimation_mode,
#                         parameter_indices,
#                         DSS_34_cov,
#                         DSS_65_cov,
#                         DSS_13_cov,
#                         start_mode,
#                         start_length,
#                         mnvr_day,
#                         mnvr_reset_covariance,
#                         process_noise_type,
#                         Q,
#                         alpha,
#                         window,
#                         Q_adaptive):

#     mnvr_time = mnvr_day * 24 * 3600

#     # ---------------------------------------------------------------------------------------------------------------------------
#     # INITIALIZE MGRS, INTEGRATOR, AND FILTER
#     # ---------------------------------------------------------------------------------------------------------------------------

#     # Pull in measurement data and define time vector
#     measurement_data = load_measurement_data(unknown_dynamics_measurement_file_path)
#     measurement_df = convert_measurements_to_df(measurement_data, station_names=list(station_locations.keys()), period_of_data=period_of_data)
#     time_vector = measurement_data['time_vector']

#     # Station managers
#     station_mgrs = []
#     for station_name, station_info in station_locations.items():
#         mgr = MeasurementMgr(
#             station_name,
#             station_lat=station_info['lat'],
#             station_lon=station_info['lon'],
#             initial_earth_spin_angle=initial_spin_angle,
#             earth_spin_rate=earth_spin_rate,
#             R_e=station_info['radius']
#         )
        
#         station_mgrs.append(mgr)

#     # Integrator
#     integrator = initialize_integrator(starting_epoch=initial_epoch, estimation_mode=estimation_mode, parameter_indices=parameter_indices)

#     # Adaptive SNC
#     adaptive_snc_mat = np.diag([Q_adaptive**2, Q_adaptive**2, Q_adaptive**2])
#     adaptive_snc = AdaptiveSNC(alpha=alpha, window=window, Q_adaptive=adaptive_snc_mat)

#     # Add station postion states to the a priori state
#     for station in station_mgrs:
#         initial_state_estimate = np.concatenate((a_priori_state, station.station_state_ecef[0:3]))  # Add station position to the a priori state if it is being estimated
#     state_length = len()

#     # Initialize the intial covariance estimate
#     initial_covariance_estimate_flattened = np.diag(a_priori_covariance)
#     station_covariance_flattened = np.array(DSS_34_cov + DSS_65_cov + DSS_13_cov)
#     initial_covariance_estimate = np.diag(np.concatenate((initial_covariance_estimate_flattened, station_covariance_flattened)))

#     ekf = EKF(integrator, station_mgrs, initial_earth_spin_angle=0, earth_rotation_rate=earth_spin_rate)

#     # ---------------------------------------------------------------------------------------------------------------------------
#     # RUN FILTER
#     # ---------------------------------------------------------------------------------------------------------------------------

#     x_hist, P_hist, residuals_df = filter.run(a_priori_state,
#                                                   np.zeros(state_length),
#                                                   a_priori_covariance,
#                                                   measurement_df,
#                                                   R=observation_noise,
#                                                   start_mode=start_mode,
#                                                   start_length=start_length,
#                                                   process_noise_approach=process_noise_type,
#                                                   Q=Q,
#                                                   adaptive_snc=adaptive_snc,
#                                                   reset_time=mnvr_time,
#                                                   reset_covariance=mnvr_reset_covariance)
    
#     plot_residuals(time_vector, residuals_df, filter_name='EKF (DSS65 Estimated with SNC)', file_directory=f'ASEN_6080/Project2/final_figures/', auto_save=True, omit_outliers=False)

#     print(f"Initial State Estimate:")
#     print(f"Position: {(x_hist[0:3, 0])} km")
#     print(f"Velocity: {(x_hist[3:6, 0])} km/s")
#     print(f"Final State Estimate:")
#     print(f"Position: {(x_hist[0:3, -1])} km | Covariance: {np.diag(P_hist[0:3, 0:3, -1])}")
#     print(f"Velocity: {(x_hist[3:6, -1])} km/s | Covariance: {np.diag(P_hist[3:6, 3:6, -1])}")
#     if 'SRP' in estimation_mode:
#         print(f"SRP Coefficient Estimate: {x_hist[6, -1]} | Covariance: {P_hist[6, 6, -1]}")
#     if 'Stations' in estimation_mode:
#         station_index = parameter_indices[estimation_mode.index('Stations')]
#         for station in station_mgrs:
#             print(f"{station.station_name} Position Estimate: {x_hist[station_index:station_index+3, -1]} | Covariance: {np.diag(P_hist[station_index:station_index+3, station_index:station_index+3, -1])}")
#             # Also determine how much station position estimates have changed from the a priori state to the final estimate and print this information
#             station_position_change = x_hist[station_index:station_index+3, -1] - a_priori_state[station_index:station_index+3]
#             print(f"{station.station_name} Position Change from A Priori State: {station_position_change} km")
#             station_index += 3  # Move to the next station position in the state vector for the next iteration of the loop

#     # Save final state estimate and covariance to a text file
#     final_state_estimate = x_hist[:, -1]
#     final_covariance_estimate = np.diag(P_hist[:, :, -1])
#     with open(f'ASEN_6080/Project2/final_figures/final_state_estimate_and_covariance.txt', 'w') as f:
#         f.write(f"Final State Estimate:\n{final_state_estimate}\n\n")
#         f.write(f"Final Covariance Estimate (Diagonal):\n{final_covariance_estimate}\n")

if __name__ == "__main__":
    # ----------------------------------------------------------------------------------------------------------------------------
    # TASK 1 ANALYSIS
    # ----------------------------------------------------------------------------------------------------------------------------
    print("=" * 50)
    print(f"Running Task 1 Dynamics Test...")
    print("=" * 50 + "\n")
    run_task_1_analysis()

    # ----------------------------------------------------------------------------------------------------------------------------
    # TASK 2 ANALYSIS
    # ----------------------------------------------------------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print(f"Running Task 2 Unknown Dynamics Estimation...")
    print("=" * 50 + "\n")
    run_task_2_analysis(period_of_data=[0, 250], filters_to_run=['Batch'])

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
    # start_length = 100

    # # Covariance reset parameters
    # mnvr_day = 217
    # mnvr_time = mnvr_day * 24 * 3600
    # mnvr_reset_covariance = np.diag([10, 10, 10, 0.5, 0.5, 0.5, 0.05, 1e-8, 1e-8, 1e-8, 1e-3, 1e-3, 1e-3, 1e-8, 1e-8, 1e-8])**2

    # # SNC parameters
    # process_noise_type = 'SNC'
    # Q = np.diag([5e-10, 5e-10, 5e-10])**2

    # # Adapative SNC parameters
    # alpha = 0.005
    # window = 10
    # Q_adaptive = 5e-7 # 1 mm/s adaptive process noise for velocity states

    # run_task_3_analysis(period_of_data,
    #                     estimation_mode,
    #                     parameter_indices,
    #                     DSS_34_cov,
    #                     DSS_65_cov,
    #                     DSS_13_cov,
    #                     start_mode,
    #                     start_length,
    #                     mnvr_day,
    #                     mnvr_reset_covariance,
    #                     process_noise_type,
    #                     Q,
    #                     alpha,
    #                     window,
    #                     Q_adaptive)
    


