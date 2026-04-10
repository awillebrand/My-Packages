import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import scipy.io
from constants import truth_data_file_path, known_dynamics_measurement_file_path, mu_sun, mu_earth, R_e, solar_flux, SRP_area_to_mass, AU, initial_epoch, initial_epoch_jd, initial_spin_angle, earth_spin_rate, station_locations, part_2_station_locations, observation_noise
from Tools.measurement_manager import MeasurementMgr
from Tools.integrator import Integrator
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
    time_vector = np.arange(measurements['time_vector'][0], measurements['time_vector'][-1]+dt, dt)  # Create a time vector with consistent time steps

    measurement_vectors = measurements['measurements']
    
    # Create a DataFrame in format of 'Time', 'DSS34', 'DSS65', 'DSS13' for both range and range rate measurements
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
        station_names[0]: list(measurement_matrix[:, 0:2]),
        station_names[1]: list(measurement_matrix[:, 2:4]),
        station_names[2]: list(measurement_matrix[:, 4:6])
    })    

    return measurement_data_frame

if __name__ == "__main__":
    # Load truth data and measurement data
    truth_data = load_truth_state_data(truth_data_file_path)
    measurement_data = load_measurement_data(known_dynamics_measurement_file_path)

    # Convert measurement data to DataFrame format
    measurement_df = convert_measurements_to_df(measurement_data, station_names=list(station_locations.keys()))

    # Initialize Measurement Manager with ground station parameters
    station_mgrs = []
    for station_name, station_info in station_locations.items():
        mgr = MeasurementMgr(
            station_name,
            station_lat=station_info['lat'],
            station_lon=station_info['lon'],
            initial_earth_spin_angle=initial_spin_angle
        )
        station_mgrs.append(mgr)

    # Initialize Integrator with known dynamics (e.g., two-body problem)
    integrator = Integrator(mu_sun, R_e, mode='TwoBody')

    # Perform filtering using the loaded measurement data and truth data
    # This is where you would implement your filtering algorithm (e.g., EKF, UKF) using the integrator and measurement managers