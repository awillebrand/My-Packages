import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import scipy.io
from constants import truth_data_file_path, known_dynamics_measurement_file_path, mu_sun, mu_earth, R_e, solar_flux, SRP_area_to_mass, C_r, initial_epoch, initial_epoch_jd, initial_spin_angle, earth_spin_rate, station_locations, part_2_station_locations, observation_noise, a_priori_state, a_priori_covariance, RSOI
from Tools.measurement_manager import MeasurementMgr
from Tools.integrator import Integrator
from Tools.batch_lls_estimator import BatchLLSEstimator
from Tools.LKF import LKF
from Tools.EKF import EKF
from Tools.SRIF import SRIF
from Tools.UKF import UKF
from Tools.plotting_functions import plot_residuals, plot_state_errors
np.set_printoptions(linewidth=200)
"""
This file performs filtering on the provided measurement data for part 2 of the project which uses the known dynamics model.
"""

def load_truth_state_data(file_path):
    """
    Load the truth data from the provided .mat file path.

    Parameters
    ----------
    file_path : str
        The path to the .mat file containing the truth data.

    Returns
    -------
    dict
        A dictionary containing the time vector and state vectors from the truth data.
    """
    data = scipy.io.loadmat(file_path)
    time_vector = data['Tt_50'].flatten()  # Flatten to convert from 2D array to 1D array
    state_vectors = data['Xt_50']  # This should already be in the correct shape (6, N)

    return {
        'time_vector': time_vector,
        'state_vectors': state_vectors
    }
    
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

def convert_measurements_to_df(measurements : dict, station_names : list, dt = 60):
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
    
    # Create a DataFrame in format of 'Time', 'DSS34_measurements', 'DSS65_measurements', 'DSS13_measurements' for both range and range rate measurements
    # First need to separate out measurements from each station into separate 2xN numpy arrays with shape (2, N) where
    # first row is range and second row is range rate. Also need to make sure times between measurements are consistent
    # with time vector and all nans.

    measurement_matrix = np.full((len(time_vector), len(station_names)*2), np.nan)  # Initialize with nans
    # Loop through for each station
    for i, station_name in enumerate(station_names):
        station_measurements = measurement_vectors[:, [i, i+3]]  # Get the measurements for this station and transpose to shape (2, N)\

        # Determine what times these measurements occurred
        measurement_times = measurements['time_vector'][~np.isnan(station_measurements[:, 0])]  # Get the times where range measurements are not nan
    
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

def interpolate_truth_to_measurement_times(truth_data, measurement_time_vector):
    """
    Interpolate the truth data state vectors to the measurement time vector for comparison.

    Parameters
    ----------
    truth_data : dict
        A dictionary containing the time vector and state vectors from the truth data.
    measurement_time_vector : np.ndarray
        The time vector corresponding to the measurements.
    Returns
    -------
    np.ndarray
        An array of interpolated state vectors corresponding to the measurement time vector.
    """
    day_50_idx = np.searchsorted(meas_time_vector, 50*24*3600)  # Find index corresponding to 50 days in seconds

    truth_time_vector = truth_data['time_vector']
    truth_state_vectors = truth_data['state_vectors']

    interpolated_state_vectors = np.zeros((7, len(measurement_time_vector[:day_50_idx])))  # Initialize array for interpolated state vectors

    for i in range(7):
        interpolated_state_vectors[i, :] = np.interp(measurement_time_vector[:day_50_idx], truth_time_vector, truth_state_vectors[:, i])

    return interpolated_state_vectors

if __name__ == "__main__":
    # User specifies the filter to run in terminal. Options are 'Batch', 'LKF', 'EKF', 'SRIF', 'UKF'
    filter_to_run = input("Enter the filter to run (Batch, LKF, EKF, SRIF, UKF): ")
    if filter_to_run not in ['Batch', 'LKF', 'EKF', 'SRIF', 'UKF']:
        print("Invalid filter choice. Please enter one of the following: Batch, LKF, EKF, SRIF, UKF")
        exit()
    

    # Load truth data and measurement data
    truth_data = load_truth_state_data(truth_data_file_path)
    measurement_data = load_measurement_data(known_dynamics_measurement_file_path)

    # Convert measurement data to DataFrame format
    measurement_df = convert_measurements_to_df(measurement_data, station_names=list(station_locations.keys()))

    meas_time_vector = measurement_df['time'].values
    truth_time_vector = truth_data['time_vector']

    station_mgrs = []
    for station_name, station_info in part_2_station_locations.items():
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
        Cr=C_r,
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

    # Perform filtering using the loaded measurement data and truth data
    if filter_to_run == 'Batch':
        max_iterations = int(input("Enter the maximum number of iterations for the Batch LLS Estimator (e.g., 10): "))
        tol = float(input("Enter the convergence tolerance for the Batch LLS Estimator (e.g., 1e-6): "))
        print("=" * 50)
        print("Running Batch LLS Estimator...")
        print("=" * 50, end='\n')
        filter = BatchLLSEstimator(integrator, station_mgrs, initial_earth_spin_angle=0, earth_rotation_rate=earth_spin_rate)
        x, P, residuals_df = filter.estimate_initial_state(a_priori_state, measurement_df, observation_noise, a_priori_covariance=a_priori_covariance, max_iterations=max_iterations, tol=tol)
        print("=" * 50)
        print("Batch LLS Estimation Complete...")
        print("=" * 50, end='\n')
        # Integrate the estimated initial state forward in time to compare to truth data
        _, augmented_x_hist = integrator.integrate_stm(meas_time_vector[-1], x[0:7], teval=meas_time_vector)
        x_hist = augmented_x_hist[:7, :]  # Extract the state history from the augmented state history
        STM_hist = augmented_x_hist[7:, :]  # Extract the STM history from the augmented state history
        P_hist = np.zeros((7,7, len(meas_time_vector)))  # Initialize an array to hold the covariance history
        for i in range(len(meas_time_vector)):
            STM = STM_hist[:, i].reshape((7, 7))  # Reshape the STM from the augmented state history
            P_hist[:, :, i] = STM @ P @ STM.T  # Propagate the covariance using the STM

    elif filter_to_run == 'LKF':
        max_iterations = int(input("Enter the maximum number of iterations for the LKF (e.g., 10): "))
        tol = float(input("Enter the convergence tolerance for the LKF (e.g., 1e-6): "))
        print("=" * 50)
        print("Running LKF...")
        print("=" * 50, end='\n')
        filter = LKF(integrator, station_mgrs, initial_earth_spin_angle=0, earth_rotation_rate=earth_spin_rate)
        x_hist, P_hist, residuals_df = filter.run(a_priori_state, np.zeros(7), a_priori_covariance, measurement_df, R=observation_noise, max_iterations=max_iterations, convergence_threshold=tol)
        print("=" * 50)
        print("LKF Run Complete...")
        print("=" * 50, end='\n')
    elif filter_to_run == 'EKF':
        start_mode = str(input("Enter Start Mode for EKF ('Warm' or 'Cold'): "))
        if start_mode.lower() == 'warm':
            start_length = int(input("Enter the number of measurements to use for the hot start (e.g., 10): "))
        else:
            start_length = 0
        filter = EKF(integrator, station_mgrs, initial_earth_spin_angle=0, earth_rotation_rate=earth_spin_rate)
        x_hist, P_hist, residuals_df = filter.run(a_priori_state, np.zeros(7), a_priori_covariance, measurement_df, R=observation_noise, start_mode=start_mode.lower(), start_length=start_length)
    elif filter_to_run == 'SRIF':
        max_iterations = int(input("Enter the maximum number of iterations for the SRIF (e.g., 10): "))
        print("=" * 50)
        print("Running SRIF...")
        print("=" * 50, end='\n')
        filter = SRIF(integrator, station_mgrs, initial_earth_spin_angle=0, earth_rotation_rate=earth_spin_rate)
        x_hist, P_hist, residuals_df = filter.run(a_priori_state, np.zeros(7), a_priori_covariance, measurement_df, R_noise=observation_noise, max_iterations=max_iterations)
        print("=" * 50)
        print("SRIF Run Complete...")
        print("=" * 50, end='\n')
    elif filter_to_run == 'UKF':
        alpha = float(input("Enter alpha parameter for UKF (e.g., 1e-3): "))
        beta = float(input("Enter beta parameter for UKF (e.g., 2): "))
        print("=" * 50)
        print("Running SRUKF...")
        print("=" * 50, end='\n')
        filter = UKF(integrator, station_mgrs, initial_earth_spin_angle=0, earth_rotation_rate=earth_spin_rate)
        x_hist, P_hist, residuals_df = filter.run(a_priori_state, a_priori_covariance, meas_time_vector, measurement_df, R=observation_noise, alpha=alpha, beta=beta)
        print("=" * 50)
        print("UKF Run Complete...")
        print("=" * 50, end='\n')

    # Pull state estimates for first 50 days to compare to truth data
    day_50_idx = np.searchsorted(meas_time_vector, 50*24*3600)  # Find index corresponding to 50 days in seconds
    
    
    # Interpolate truth data to measurement time vector for first 50 days
    if filter_to_run == 'Batch':
        _, augmented_x_hist = integrator.integrate_stm(truth_time_vector[-1], x[0:7], teval=truth_time_vector)
        x_hist_50days = augmented_x_hist  # Extract the state history from the augmented state history
        STM_hist = augmented_x_hist[7:, :]  # Extract the STM history from the augmented state history
        P_hist_50days = np.zeros((7,7, len(truth_time_vector)))  # Initialize an array to hold the covariance history
        for i in range(len(truth_time_vector)):
            STM = STM_hist[:, i].reshape((7, 7))  # Reshape the STM from the augmented state history
            P_hist_50days[:, :, i] = STM @ P @ STM.T  # Propagate the covariance using the STM
        interpolated_truth_state_vectors = truth_data['state_vectors']

        # Compute state estimation errors for first 50 days
        estimation_errors = x_hist_50days - interpolated_truth_state_vectors.T
        plot_state_errors(truth_time_vector, estimation_errors, P_hist_50days, filter_name=filter_to_run, file_directory='ASEN_6080/Project2/figures')
        plot_residuals(meas_time_vector, residuals_df, filter_name=filter_to_run, file_directory='ASEN_6080/Project2/figures/residuals')

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02, subplot_titles=['X Position', 'Y Position', 'Z Position'])
        for i in range(3):
            fig.add_trace(go.Scatter(x=truth_time_vector, y=truth_data['state_vectors'][:,i], mode='lines', name='Truth'), row=i+1, col=1)
            fig.add_trace(go.Scatter(x=truth_time_vector, y=x_hist_50days[i,:], mode='lines', name='Estimated'), row=i+1, col=1)
        fig.update_layout(title='Comparison of Estimated Trajectory to Truth Data for First 50 Days', xaxis_title='Time (s)', yaxis_title='Position (km)')
        fig.show()
    else:
        x_hist_50days = x_hist[:, :day_50_idx]
        P_hist_50days = P_hist[:, :, :day_50_idx]
        interpolated_truth_state_vectors = interpolate_truth_to_measurement_times(truth_data, meas_time_vector)

        # Compute state estimation errors for first 50 days
        estimation_errors = x_hist_50days - interpolated_truth_state_vectors

        plot_state_errors(meas_time_vector[:day_50_idx], estimation_errors, P_hist_50days, filter_name=filter_to_run, file_directory='ASEN_6080/Project2/figures')
        plot_residuals(meas_time_vector, residuals_df, filter_name=filter_to_run, file_directory='ASEN_6080/Project2/figures/residuals')

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02, subplot_titles=['X Position', 'Y Position', 'Z Position'])
        for i in range(3):
            fig.add_trace(go.Scatter(x=truth_time_vector, y=truth_data['state_vectors'][:,i], mode='lines', name='Truth'), row=i+1, col=1)
            fig.add_trace(go.Scatter(x=meas_time_vector[:day_50_idx], y=x_hist_50days[i,:], mode='lines', name='Estimated'), row=i+1, col=1)
        fig.update_layout(title='Comparison of Estimated Trajectory to Truth Data for First 50 Days', xaxis_title='Time (s)', yaxis_title='Position (km)')
        fig.show()

    
