import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import scipy.io
from Tools.measurement_manager import MeasurementMgr
from Tools.integrator import Integrator
from Tools.batch_lls_estimator import BatchLLSEstimator
from Tools.LKF import LKF
from Tools.EKF import EKF
from Tools.SRIF import SRIF
from Tools.adaptive_snc import AdaptiveSNC
from Tools.plotting_functions import plot_residuals, plot_state_errors

from constants import unknown_dynamics_measurement_file_path, a_priori_state, a_priori_covariance, observation_noise, initial_spin_angle, earth_spin_rate, station_locations
from constants import C_r, mu_sun, mu_earth, R_e, solar_flux, SRP_area_to_mass, initial_epoch, initial_epoch_jd
from constants import alpha, window, Q_adaptive, maneuver_reset_covariance
np.set_printoptions(linewidth=200)

def load_measurement_data(file_path):
    """
    Load the measurement data from the provided text file path. Converts measurements to a pa

    Parameters
    ----------
    file_path : str
        The path to the text file containing the measurement data.

    Returns
    -------
    dict
        A dictionary containing the time vector and measurement vectors from the measurement data.
    """
    
    with open(file_path, 'r') as f:
        data_string = f.read().replace(',', ' ')  # Replace commas with spaces to handle comma-separated values

    data = np.loadtxt(data_string.splitlines(), skiprows=1)  # Load the data into a numpy array

    time_vector = data[:, 0]  # Assuming first column is time
    measurements = data[:, 1:]  # Assuming remaining columns are measurements
    
    return {
        'time_vector': time_vector,
        'measurements': measurements
    }

def convert_measurements_to_df(measurements : dict, station_names : list, period_of_data : list):
    """
    Convert the measurement data into a pandas DataFrame to make it compatible with existing filtering code.

    Parameters
    ----------
    measurements : dict
        A dictionary containing the time vector and measurement vectors.
    station_names : list
        A list of station names corresponding to the measurements (e.g., ['DSS34', 'DSS65', 'DSS13']).
    dt : float
        The time step to use for the DataFrame index (default is 60 seconds).

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the time and measurements with appropriate column names.
    """
    #time_vector = np.arange(measurements['time_vector'][0], measurements['time_vector'][-1]+dt, dt)  # Create a time vector with consistent time steps
    time_vector = measurements['time_vector']  # Use the original time vector from the measurements to ensure consistency with measurement times

    measurement_vectors = measurements['measurements']

    # Find index where time exceeds the specified number of days and truncate the time vector and measurement vectors accordingly
    min_time = period_of_data[0] * 24 * 3600  # Convert days to seconds
    max_time = period_of_data[1]* 24 * 3600  # Convert days to seconds
    
    valid_indices = (np.where((time_vector >= min_time) & (time_vector <= max_time)))[0]  # Get indices where time is within the specified range

    time_vector = time_vector[valid_indices]
    measurement_vectors = measurement_vectors[valid_indices, :]

    # Create a DataFrame in format of 'Time', 'DSS34_measurements', 'DSS65_measurements', 'DSS13_measurements' for both range and range rate measurements
    # First need to separate out measurements from each station into separate 2xN numpy arrays with shape (2, N) where
    # first row is range and second row is range rate. Also need to make sure times between measurements are consistent
    # with time vector and all nans.

    measurement_matrix = np.full((len(time_vector), len(station_names)*2), np.nan)  # Initialize with nans
    # Loop through for each station
    for i, station_name in enumerate(station_names):
        station_measurements = measurement_vectors[:, [i, i+3]]  # Get the measurements for this station and transpose to shape (2, N)\

        # Determine what times these measurements occurred
        measurement_times = time_vector[~np.isnan(station_measurements[:, 0])]  # Get the times where range measurements are not nan
    
        # Find the index in the time_vector where these times occur
        measurement_indices = np.searchsorted(time_vector, measurement_times)

        # Initialize empty measurement array with shape (2, len(time_vector)) filled with nans
        full_measurements = np.full((len(time_vector), 2), np.nan)

        # Fill in the measurements at the correct indices
        full_measurements[measurement_indices, :] = station_measurements[~np.isnan(station_measurements[:, 0])]
        
        # Place the full measurements for this station into the correct columns of the measurement matrix
        measurement_matrix[:, i*2:(i+1)*2] = full_measurements
   
    measurement_data_frame = pd.DataFrame({
        'time': time_vector,
        f"{station_names[0]}_measurements": list(measurement_matrix[:, 0:2]),
        f"{station_names[1]}_measurements": list(measurement_matrix[:, 2:4]),
        f"{station_names[2]}_measurements": list(measurement_matrix[:, 4:6])
    })    

    return measurement_data_frame


def initialize_integrator(starting_epoch, estimation_mode, parameter_indices):
    """
    Initialize the Integrator object with the appropriate initial epoch and gravitational parameter.

    Parameters
    ----------
    starting_epoch : float
        The initial epoch in seconds to initialize the integrator.
    estimation_mode : list
        A list of strings indicating which parameters are being estimated (e.g., ['SRP']).
    parameter_indices : list
        A list of integers indicating the indices of the parameters being estimated in the state vector (e.g., [6] for C_r).

    Returns
    -------
    Integrator
        An instance of the Integrator class initialized with the appropriate initial epoch and gravitational parameter.
    """
    starting_epoch_jd = initial_epoch_jd + starting_epoch / 86400  # Convert starting epoch to Julian Date

    # Initialize Integrator with known dynamics (e.g., two-body problem)
    integrator = Integrator(
        mu=mu_earth,
        R_e=R_e,        
        dynamical_mode=['mu', 'SRP', 'Third Body'],  # Include 2-body and SRP effects for this test
        estimation_mode=estimation_mode,
        parameter_indices=parameter_indices,
        Cr=C_r,
        srp_area_to_mass=SRP_area_to_mass,  # Use the area-to-mass ratio from constants
        solar_flux=solar_flux,
        number_of_stations=3,
        mu_third_body=mu_sun,
        central_body='Earth',
        third_body='Sun',
        initial_epoch_jd=starting_epoch_jd,
        initial_epoch=starting_epoch,
        earth_spin_rate=earth_spin_rate
    )

    return integrator

if __name__ == "__main__":
    # Initialize measurement managers

    station_mgrs = []
    for station_name, station_info in station_locations.items():
        mgr = MeasurementMgr(
            station_name,
            station_lat=station_info['lat'],
            station_lon=station_info['lon'],
            initial_earth_spin_angle=initial_spin_angle,
            earth_spin_rate=earth_spin_rate,
            R_e=station_info['radius']
        )
        
        station_mgrs.append(mgr)

    # User specifies the filter to run in terminal. Options are 'Batch', 'LKF', 'EKF', 'SRIF', 'UKF'
    period_of_data = input("Enter the range of days of measurement data to use for the filters (e.g., 50, 100): ")
    period_of_data = [int(day.strip()) for day in period_of_data.split(',')]  # Split the input string into a list of floats representing the range of days to use for the filters

    inputted_estimation_mode = input("Enter the parameters to estimate (e.g., mu, Third Body). SRP is included by default: ")

    if len(inputted_estimation_mode) == 0:
        estimation_mode = ['SRP']  # If no parameters are entered, default to only estimating SRP
    elif ',' not in inputted_estimation_mode:
        estimation_mode = ['SRP', inputted_estimation_mode.strip()]  # Add the inputted estimation mode to the default estimation mode of SRP
    else:
        inputted_estimation_mode = [param.strip() for param in inputted_estimation_mode.split(',')]  # Split the input string into a list if more than one parameters was specified
        estimation_mode = ['SRP'].extend(inputted_estimation_mode)  # Add the inputted estimation mode to the default estimation mode of SRP

    parameter_indices = [6] # Initialize with index for C_r, which is always included in estimation
    flattened_cov = np.diag(a_priori_covariance)

    # Update the a priori state and covariance to include the parameters being estimated
    for param in estimation_mode:
        if param == 'SRP':
            continue  # Skip SRP since it is already included in the a priori state and covariance

        parameter_indices.append(parameter_indices[-1] + 1)  # Add the next index for the next parameter being estimated

        if param == 'mu':
            a_priori_state = np.concatenate((a_priori_state, [mu_earth]))  # Add mu_earth to the a priori state if it is being estimated
        elif param == 'Third Body':
            a_priori_state = np.concatenate((a_priori_state, [mu_sun]))  # Add mu_sun to the a priori state if it is being estimated
        elif param == 'Stations':
            # Need to pull a priori state position estimates from station location dictionaries
            for station in station_mgrs:
                a_priori_state = np.concatenate((a_priori_state, station.station_state_ecef[0:3]))  # Add station position to the a priori state if it is being estimated
        else:
            print(f"Parameter {param} not recognized. Please enter 'mu', 'Third Body', 'Stations', or leave blank if no additional parameters are being estimated.")
            exit()

        if param == 'Stations':
            DSS34_covariance = float(input(f"Enter the covariance estimates for DSS34 position (in km and km/s, e.g., 1e-3): "))
            DSS65_covariance = float(input(f"Enter the covariance estimates for DSS65 position (in km and km/s, e.g., 1e-3): "))
            DSS13_covariance = float(input(f"Enter the covariance estimates for DSS13 position (in km and km/s, e.g., 1e-3): "))
            # Use same value for 3D covariance of station
            DSS34_covariance = [DSS34_covariance, DSS34_covariance, DSS34_covariance]  # Create a list of the covariance values for the station position
            DSS65_covariance = [DSS65_covariance, DSS65_covariance, DSS65_covariance]  # Create a list of the covariance values for the station position
            DSS13_covariance = [DSS13_covariance, DSS13_covariance, DSS13_covariance]  # Create a list of the covariance values for the station position
            flattened_cov = np.concatenate((flattened_cov, DSS34_covariance, DSS65_covariance, DSS13_covariance))  # Add the covariance for the station positions to the flattened covariance array
            a_priori_covariance = np.diag(flattened_cov)  # Convert back to diagonal covariance matrix
        else:
            param_covariance = float(input(f"Enter the a priori covariance for {param}: "))
            # Add the covariance for this parameter to the a priori covariance matrix
            flattened_cov = np.concatenate((flattened_cov, [param_covariance**2]))  # Add the covariance for this parameter to the flattened covariance array
            a_priori_covariance = np.diag(flattened_cov)  # Convert back to diagonal covariance matrix

    filter_to_run = str(input("Enter the filter to run (Batch, LKF, EKF, SRIF, UKF): ")).lower()

    if filter_to_run not in ['batch', 'lkf', 'ekf', 'srif', 'ukf']:
        print("Invalid filter choice. Please enter one of the following: Batch, LKF, EKF, SRIF, UKF")
        exit()

    # Load measurement data
    # Load truth data and measurement data
    measurement_data = load_measurement_data(unknown_dynamics_measurement_file_path)

    # Convert measurement data to DataFrame format
    measurement_df = convert_measurements_to_df(measurement_data, station_names=list(station_locations.keys()), period_of_data=period_of_data)
    time_vector = measurement_data['time_vector']

    integrator = initialize_integrator(initial_epoch, estimation_mode, parameter_indices)

    state_length = len(a_priori_state)

    # Set up adaptive SNC
    
    adaptive_snc_mat = np.diag([Q_adaptive**2, Q_adaptive**2, Q_adaptive**2])

    adaptive_snc = AdaptiveSNC(alpha=alpha, window=window, Q_adaptive=adaptive_snc_mat)

    print(f"A Priori State: {a_priori_state}")
    print(f"A Priori Covariance: {a_priori_covariance}\n")
    if filter_to_run == 'batch':
        max_iterations = int(input("Enter the maximum number of iterations for the Batch LLS Estimator (default is 10): "))
        tol = float(input("Enter the convergence tolerance for the Batch LLS Estimator (default is 1e-6): "))
        print("Running Batch LLS Estimator...")
        print("=" * 50, end='\n')
        estimator = BatchLLSEstimator(integrator, station_mgrs, initial_earth_spin_angle=0, earth_rotation_rate=earth_spin_rate)
        x_est, P_est, residuals_df = estimator.estimate_initial_state(a_priori_state,
                                                        measurement_df,
                                                        observation_noise,
                                                        a_priori_covariance=a_priori_covariance,
                                                        max_iterations=max_iterations,
                                                        tol=tol)
        print("=" * 50)
        print("Batch LLS Estimation Complete...")
        print("=" * 50, end='\n')
        # Integrate the estimated initial state forward in time to compare to truth data
        _, augmented_x_hist = integrator.integrate_stm(time_vector[-1], x_est, teval=time_vector)
        x_hist = augmented_x_hist[:state_length, :]  # Extract the state history from the augmented state history
        STM_hist = augmented_x_hist[state_length:, :]  # Extract the STM history from the augmented state history
        P_hist = np.zeros((state_length,state_length, len(time_vector)))  # Initialize an array to hold the covariance history
        for j in range(len(time_vector)):
            STM = STM_hist[:, j].reshape((state_length, state_length))  # Reshape the STM from the augmented state history
            P_hist[:, :, j] = STM @ P_est @ STM.T  # Propagate the covariance using the STM

    elif filter_to_run == 'lkf':
        max_iterations = int(input("Enter the maximum number of iterations for the LKF (default is 10): "))
        tol = float(input("Enter the convergence tolerance for the LKF (default is 1e-6): "))
        process_noise_type = str(input("Enter the process noise approach for the LKF ('SNC', 'Adaptive SNC', or 'None'): "))
        apply_smoothing = input("Apply smoothing with the LKF? (True/False): ").lower() == 'true'
  
        if process_noise_type == "SNC":
            Q = input("Enter the process noise covariance matrix Q as a flattened list (e.g., for a 3x3 identity matrix, enter 1, 1, 1): ")
            Q = np.diag([float(q) for q in Q.split(',')])  # Convert the input string into a diagonal matrix
        else:
            Q = None
        print("Running LKF...")
        print("=" * 50, end='\n')
        filter = LKF(integrator, station_mgrs, initial_earth_spin_angle=0, earth_rotation_rate=earth_spin_rate)
        x_hist, P_hist, residuals_df = filter.run(a_priori_state,
                                                  np.zeros(state_length),
                                                  a_priori_covariance,
                                                  measurement_df,
                                                  R=observation_noise,
                                                  max_iterations=max_iterations,
                                                  convergence_threshold=tol,
                                                  process_noise_approach=process_noise_type,
                                                  Q=Q,
                                                  apply_smoothing=apply_smoothing,
                                                  adaptive_snc=adaptive_snc)
        print("=" * 50)
        print("LKF Run Complete...")
        print("=" * 50, end='\n')
    
    elif filter_to_run == 'ekf':
        start_mode = str(input("Enter Start Mode for EKF ('Warm' or 'Cold'): "))
        if start_mode.lower() == 'warm':
            start_length = int(input("Enter the number of measurements to use for the hot start (e.g., 10): "))
        else:
            start_length = 0
        process_noise_type = str(input("Enter the process noise approach for the EKF ('SNC', 'Adaptive SNC', or 'None'): "))
        if process_noise_type == "SNC":
            Q = input("Enter the process noise covariance matrix Q as a flattened list (e.g., for a 3x3 identity matrix, enter 1, 1, 1): ")
            Q = np.diag([float(q) for q in Q.split(',')])  # Convert the input string into a diagonal matrix
        else:
            Q = None
        reset_day_input = input("Enter a day at which to reset the covariance values due to an expected maneuver (or leave blank for no reset): ")
        if reset_day_input:
            reset_day = float(reset_day_input)
            reset_time = reset_day * 24 * 3600  # Convert day to seconds
            reset_covariance = maneuver_reset_covariance  # Reset to the initial covariance
        else:
            reset_time = None
        max_iterations = 0
        print("Running EKF...")
        print("=" * 50, end='\n')
        filter = EKF(integrator, station_mgrs, initial_earth_spin_angle=0, earth_rotation_rate=earth_spin_rate)
        x_hist, P_hist, residuals_df = filter.run(a_priori_state,
                                                  np.zeros(state_length),
                                                  a_priori_covariance,
                                                  measurement_df,
                                                  R=observation_noise,
                                                  start_mode=start_mode,
                                                  start_length=start_length,
                                                  process_noise_approach=process_noise_type,
                                                  Q=Q,
                                                  adaptive_snc=adaptive_snc,
                                                  reset_time=reset_time,
                                                  reset_covariance=reset_covariance)
        print("=" * 50)
        print("EKF Run Complete...")
        print("=" * 50, end='\n')

    fig_list = plot_residuals(time_vector, residuals_df, filter_name=filter_to_run, file_directory=f'ASEN_6080/Project2/part_3_figures/residuals/{filter_to_run}_{max_iterations}_iterations', auto_save=False, omit_outliers=True)

    parameters_estimated = ""
    flag = False
    for param, cov in zip(estimation_mode, flattened_cov[6:]):
        current_idx = estimation_mode.index(param)
        if param == 'Stations' and flag == False:
            # If estimating station positions, need to include group the 3 covariances for each station together in the parameters_estimated string
            parameters_estimated += f"{param}_{cov:.2e}_"
            # get next 6 covariances for the next two stations and add to the parameters_estimated string
            for i in range(2):
                idx = current_idx + (i+1)*3  # Get the index for the next station's covariance
                cov = flattened_cov[6 + idx]  # Get the covariance for the next station
                parameters_estimated += f"{cov:.2e}_"
            flag = True
        else:
            parameters_estimated += f"{param}_{cov:.2e}_"
    
    period_analyzed = f"{int(period_of_data[0])}-{int(period_of_data[1])}"

    fig_list[-1][0].write_html(f"ASEN_6080/Project2/part_3_figures/residuals/{filter_to_run}/final/PREFIT_{parameters_estimated}IT_{max_iterations}_PER_{period_analyzed}_{process_noise_type}.html")
    fig_list[-1][1].write_html(f"ASEN_6080/Project2/part_3_figures/residuals/{filter_to_run}/final/POSTFIT_{parameters_estimated}IT_{max_iterations}_PER_{period_analyzed}_{process_noise_type}.html")
    
    print(f"Initial State Estimate:")
    print(f"Position: {(x_hist[0:3, 0])} km")
    print(f"Velocity: {(x_hist[3:6, 0])} km/s")
    print(f"Final State Estimate:")
    print(f"Position: {(x_hist[0:3, -1])} km | Covariance: {np.diag(P_hist[0:3, 0:3, -1])}")
    print(f"Velocity: {(x_hist[3:6, -1])} km/s | Covariance: {np.diag(P_hist[3:6, 3:6, -1])}")
    if 'SRP' in estimation_mode:
        print(f"SRP Coefficient Estimate: {x_hist[6, -1]} | Covariance: {P_hist[6, 6, -1]}")
    if 'mu' in estimation_mode:
        # Find index of mu in the parameter_indices list to pull the correct estimate and covariance from the state history and covariance history
        mu_index = parameter_indices.index(parameter_indices[estimation_mode.index('mu')])  # Get the index of mu in the parameter_indices list
        print(f"Gravitational Parameter Estimate: {x_hist[mu_index, -1]} | Covariance: {P_hist[mu_index, mu_index, -1]}")
    if 'Third Body' in estimation_mode:
        # Find index of Third Body in the parameter_indices list to pull the correct estimate and covariance from the state history and covariance history
        third_body_index = parameter_indices.index(parameter_indices[estimation_mode.index('Third Body')])  # Get the index of Third Body in the parameter_indices list
        print(f"Third Body Gravitational Parameter Estimate: {x_hist[third_body_index, -1]} | Covariance: {P_hist[third_body_index, third_body_index, -1]}")
    if 'Stations' in estimation_mode:
        station_index = parameter_indices[estimation_mode.index('Stations')]
        for station in station_mgrs:
            print(f"{station.station_name} Position Estimate: {x_hist[station_index:station_index+3, -1]} | Covariance: {np.diag(P_hist[station_index:station_index+3, station_index:station_index+3, -1])}")
            # Also determine how much station position estimates have changed from the a priori state to the final estimate and print this information
            station_position_change = x_hist[station_index:station_index+3, -1] - a_priori_state[station_index:station_index+3]
            print(f"{station.station_name} Position Change from A Priori State: {station_position_change} km")
            station_index += 3  # Move to the next station position in the state vector for the next iteration of the loop

    # Save final state estimate and covariance to a text file
    np.savetxt(f'ASEN_6080/Project2/part_3_figures/residuals/{filter_to_run}/new/FINAL_ESTIMATE_{parameters_estimated}IT_{max_iterations}_PER_{period_analyzed}_{process_noise_type}.txt', x_hist[:, -1])
    np.savetxt(f'ASEN_6080/Project2/part_3_figures/residuals/{filter_to_run}/new/FINAL_COVARIANCE_{parameters_estimated}IT_{max_iterations}_PER_{period_analyzed}_{process_noise_type}.txt', P_hist[:, :, -1])