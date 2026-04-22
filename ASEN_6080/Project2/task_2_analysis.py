import numpy as np
from generic_functions import load_truth_data, load_measurement_data, convert_measurements_to_df
from Tools.measurement_manager import MeasurementMgr
from Tools.integrator import Integrator
from Tools.LKF import LKF
from Tools.batch_lls_estimator import BatchLLSEstimator
from Tools.SRIF import SRIF
from Tools.EKF import EKF
from constants import known_dynamics_measurement_file_path, truth_data_file_path, task_2_station_locations, testing_a_priori_state, testing_a_priori_covariance, observation_noise
from constants import mu_sun, mu_earth, R_e, solar_flux, SRP_area_to_mass, initial_epoch, initial_epoch_jd, earth_spin_rate, initial_spin_angle, testing_C_r

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
            filter_to_run = BatchLLSEstimator(integrator, station_mgrs, initial_earth_spin_angle=0, earth_rotation_rate=earth_spin_rate)
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

