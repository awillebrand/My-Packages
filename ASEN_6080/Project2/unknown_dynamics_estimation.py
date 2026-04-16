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
from Tools.UKF import UKF
from Tools.plotting_functions import plot_residuals, plot_state_errors

from constants import unknown_dynamics_measurement_file_path, a_priori_state, a_priori_covariance, observation_noise, initial_spin_angle, earth_spin_rate, station_locations
from constants import C_r, mu_sun, mu_earth, R_e, solar_flux, SRP_area_to_mass, initial_epoch, initial_epoch_jd
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

def convert_measurements_to_df(measurements : dict, station_names : list):
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


def initialize_integrator(starting_epoch):
    """
    Initialize the Integrator object with the appropriate initial epoch and gravitational parameter.

    Parameters
    ----------
    starting_epoch : float
        The initial epoch in seconds to initialize the integrator.

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
        estimation_mode=['SRP', 'mu'],
        parameter_indices=[6, 7],
        Cr=C_r,
        srp_area_to_mass=SRP_area_to_mass,  # Use the area-to-mass ratio from constants
        solar_flux=solar_flux,
        number_of_stations=0,
        mu_third_body=mu_sun,
        central_body='Earth',
        third_body='Sun',
        initial_epoch_jd=starting_epoch_jd,
        initial_epoch=starting_epoch,
        earth_spin_rate=earth_spin_rate
    )

    return integrator

if __name__ == "__main__":
    # User specifies the filter to run in terminal. Options are 'Batch', 'LKF', 'EKF', 'SRIF', 'UKF'
    filter_to_run = str(input("Enter the filter to run (Batch, LKF, EKF, SRIF, UKF): ")).lower()
    if filter_to_run not in ['batch', 'lkf', 'ekf', 'srif', 'ukf']:
        print("Invalid filter choice. Please enter one of the following: Batch, LKF, EKF, SRIF, UKF")
        exit()

    # Load measurement data
    # Load truth data and measurement data
    measurement_data = load_measurement_data(unknown_dynamics_measurement_file_path)

    # Convert measurement data to DataFrame format
    measurement_df = convert_measurements_to_df(measurement_data, station_names=list(station_locations.keys()))
    time_vector = measurement_data['time_vector']

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

    integrator = initialize_integrator(initial_epoch)

    state_length = len(a_priori_state)

    if filter_to_run == 'batch':
        max_iterations = int(input("Enter the maximum number of iterations for the Batch LLS Estimator (default is 10): "))
        tol = float(input("Enter the convergence tolerance for the Batch LLS Estimator (default is 1e-6): "))
        print("Running Batch LLS Estimator...")
        print("=" * 50, end='\n')
        estimator = BatchLLSEstimator(integrator, station_mgrs, initial_earth_spin_angle=0, earth_rotation_rate=earth_spin_rate)
        x_est, P_est = estimator.estimate(a_priori_state,
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
        process_noise_type = str(input("Enter the process noise approach for the LKF ('SNC' or 'None'): "))
        if process_noise_type != "None":
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
                                                  Q=Q)
        print("=" * 50)
        print("LKF Run Complete...")
        print("=" * 50, end='\n')
    
    elif filter_to_run == 'ekf':
        start_mode = str(input("Enter Start Mode for EKF ('Warm' or 'Cold'): "))
        if start_mode.lower() == 'warm':
            start_length = int(input("Enter the number of measurements to use for the hot start (e.g., 10): "))
        else:
            start_length = 0
        process_noise_type = str(input("Enter the process noise approach for the EKF ('SNC' or 'None'): "))
        if process_noise_type != "None":
            Q = input("Enter the process noise covariance matrix Q as a flattened list (e.g., for a 3x3 identity matrix, enter 1, 1, 1): ")
            Q = np.diag([float(q) for q in Q.split(',')])  # Convert the input string into a diagonal matrix
        else:
            Q = None

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
                                                  Q=Q)
        print("=" * 50)
        print("EKF Run Complete...")
        print("=" * 50, end='\n')

    parameter_estimated = str(input("What parameter was included in estimation: "))
    residual_fig = plot_residuals(time_vector, residuals_df, filter_name=filter_to_run, file_directory=f'ASEN_6080/Project2/part_3_figures/residuals/{filter_to_run}_{parameter_estimated}_{max_iterations}_iterations')
    residual_fig.show()