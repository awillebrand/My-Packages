import numpy as np
import pandas as pd
import scipy.io
from Tools.integrator import Integrator
from constants import mu_sun, mu_earth, R_e, solar_flux, SRP_area_to_mass, initial_epoch, initial_epoch_jd, earth_spin_rate, C_r
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

    time_vector = data[:, 0]
    measurements = data[:, 1:]
    
    return {
        'time_vector': time_vector,
        'measurements': measurements
    }

def load_truth_data(file_path):
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

def initialize_integrator(starting_epoch, estimation_mode, parameter_indices, input_C_r=C_r):
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
        Cr=input_C_r,
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
    day_50_idx = np.searchsorted(measurement_time_vector, 50*24*3600)  # Find index corresponding to 50 days in seconds

    truth_time_vector = truth_data['time_vector']
    truth_state_vectors = truth_data['state_vectors']

    interpolated_state_vectors = np.zeros((7, len(measurement_time_vector[:day_50_idx])))  # Initialize array for interpolated state vectors

    for i in range(7):
        interpolated_state_vectors[i, :] = np.interp(measurement_time_vector[:day_50_idx], truth_time_vector, truth_state_vectors[:, i])

    return interpolated_state_vectors
