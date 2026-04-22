import numpy as np
from generic_functions import load_measurement_data, convert_measurements_to_df, initialize_integrator
from Tools.measurement_manager import MeasurementMgr
from Tools.adaptive_snc import AdaptiveSNC
from Tools.integrator import Integrator
from Tools.EKF import EKF
from Tools.plotting_functions import plot_residuals
from constants import task_3_station_locations, initial_epoch, initial_spin_angle, earth_spin_rate, unknown_dynamics_measurement_file_path, a_priori_state, a_priori_covariance, observation_noise

def run_task_3_analysis(period_of_data,
                        estimation_mode,
                        parameter_indices,
                        DSS_34_cov,
                        DSS_65_cov,
                        DSS_13_cov,
                        start_mode,
                        start_length,
                        mnvr_day,
                        mnvr_reset_covariance,
                        process_noise_type,
                        Q,
                        alpha,
                        window,
                        Q_adaptive,
                        filter_name):

    mnvr_time = mnvr_day * 24 * 3600

    # ---------------------------------------------------------------------------------------------------------------------------
    # INITIALIZE MGRS, INTEGRATOR, AND FILTER
    # ---------------------------------------------------------------------------------------------------------------------------

    # Pull in measurement data and define time vector
    measurement_data = load_measurement_data(unknown_dynamics_measurement_file_path)
    measurement_df = convert_measurements_to_df(measurement_data, station_names=list(task_3_station_locations.keys()), period_of_data=period_of_data)
    time_vector = measurement_data['time_vector']

    # Station managers
    station_mgrs = []
    for station_name, station_info in task_3_station_locations.items():
        mgr = MeasurementMgr(
            station_name,
            station_lat=station_info['lat'],
            station_lon=station_info['lon'],
            initial_earth_spin_angle=initial_spin_angle,
            earth_spin_rate=earth_spin_rate,
            R_e=station_info['radius']
        )
        
        station_mgrs.append(mgr)

    # Integrator
    integrator = initialize_integrator(starting_epoch=initial_epoch, estimation_mode=estimation_mode, parameter_indices=parameter_indices)

    # Adaptive SNC
    adaptive_snc_mat = np.diag([Q_adaptive**2, Q_adaptive**2, Q_adaptive**2])
    adaptive_snc = AdaptiveSNC(alpha=alpha, window=window, Q_adaptive=adaptive_snc_mat)

    # Add station postion states to the a priori state
    initial_state_estimate = a_priori_state.copy()
    for station in station_mgrs:
        initial_state_estimate = np.concatenate((initial_state_estimate, station.station_state_ecef[0:3]))  # Add station position to the a priori state if it is being estimated
    state_length = len(initial_state_estimate)

    # Initialize the intial covariance estimate
    initial_covariance_estimate_flattened = np.diag(a_priori_covariance)
    station_covariance_flattened = np.array(DSS_34_cov + DSS_65_cov + DSS_13_cov)
    initial_covariance_estimate = np.diag(np.concatenate((initial_covariance_estimate_flattened, station_covariance_flattened)))

    ekf = EKF(integrator, station_mgrs, initial_earth_spin_angle=0, earth_rotation_rate=earth_spin_rate)

    # ---------------------------------------------------------------------------------------------------------------------------
    # RUN FILTER
    # ---------------------------------------------------------------------------------------------------------------------------
    x_hist, P_hist, residuals_df = ekf.run(initial_state_estimate,
                                           np.zeros(state_length),
                                           initial_covariance_estimate,
                                           measurement_df,
                                           R=observation_noise,
                                           start_mode=start_mode,
                                           start_length=start_length,
                                           process_noise_approach=process_noise_type,
                                           Q=Q,
                                           adaptive_snc=adaptive_snc,
                                           reset_time=mnvr_time,
                                           reset_covariance=mnvr_reset_covariance)
    
    plot_residuals(time_vector, residuals_df, filter_name=filter_name, file_directory=f'ASEN_6080/Project2/final_figures/', auto_save=True, omit_outliers=False)

    print(f"Initial State Estimate:")
    print(f"Position: {(x_hist[0:3, 0])} km")
    print(f"Velocity: {(x_hist[3:6, 0])} km/s")
    print(f"Final State Estimate:")
    print(f"Position: {(x_hist[0:3, -1])} km | Covariance: {np.diag(P_hist[0:3, 0:3, -1])}")
    print(f"Velocity: {(x_hist[3:6, -1])} km/s | Covariance: {np.diag(P_hist[3:6, 3:6, -1])}")
    if 'SRP' in estimation_mode:
        print(f"SRP Coefficient Estimate: {x_hist[6, -1]} | Covariance: {P_hist[6, 6, -1]}")
    if 'Stations' in estimation_mode:
        station_index = parameter_indices[estimation_mode.index('Stations')]
        for station in station_mgrs:
            print(f"{station.station_name} Position Estimate: {x_hist[station_index:station_index+3, -1]} | Covariance: {np.diag(P_hist[station_index:station_index+3, station_index:station_index+3, -1])}")
            # Also determine how much station position estimates have changed from the a priori state to the final estimate and print this information
            station_position_change = x_hist[station_index:station_index+3, -1] - initial_state_estimate[station_index:station_index+3]
            print(f"{station.station_name} Position Change from A Priori State: {station_position_change} km")
            station_index += 3  # Move to the next station position in the state vector for the next iteration of the loop

    # Save final state estimate and covariance to a text file
    final_state_estimate = x_hist[:, -1]
    final_covariance_estimate = np.diag(P_hist[:, :, -1])
    with open(f'ASEN_6080/Project2/final_results/final_state_estimate_and_covariance.txt', 'w') as f:
        f.write(f"Final State Estimate:\n{final_state_estimate}\n\n")
        f.write(f"Final Covariance Estimate (Diagonal):\n{final_covariance_estimate}\n")