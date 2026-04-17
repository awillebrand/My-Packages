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
        Cr=1.5,
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
    # Initialize truth state
    a_priori_state = np.array([-2.74096757e+08, -9.28593692e+07, -4.01993995e+07, 32.67075862, -8.93745182, -3.87893414, 1.5])

    # Initialize measurement managers
    DSS34_state = np.array([-4456.14173967, 2679.4293976, -3694.97574763, 0, 0, 0])  # ECEF state of DSS34 station in km and km/s (velocity is zero since it's a ground station)
    DSS65_state = np.array([4842.49607654, -359.80453988, 4136.64572122, 0, 0, 0])  # ECEF state of DSS65 station in km and km/s (velocity is zero since it's a ground station)
    DSS13_state = np.array([-2348.53218558, -4650.31719266, 3681.46284672, 0, 0, 0])  # ECEF state of DSS13 station in km and km/s (velocity is zero since it's a ground station)

    state_list = [DSS34_state, DSS65_state, DSS13_state]
    station_mgrs = []
    for i, station_name in enumerate(station_locations.keys()):
        station_info = station_locations[station_name]
        mgr = MeasurementMgr(
            station_name,
            station_state_ecef = state_list[i],
            initial_earth_spin_angle=initial_spin_angle,
            earth_spin_rate=earth_spin_rate,
            R_e=station_info['radius']
        )
        
        station_mgrs.append(mgr)

    # Load measurement data
    # Load truth data and measurement data
    measurement_data = load_measurement_data(unknown_dynamics_measurement_file_path)

    # Convert measurement data to DataFrame format
    measurement_df = convert_measurements_to_df(measurement_data, station_names=list(station_locations.keys()), period_of_data=[0, 75])
    time_vector = measurement_df['time'].values

    integrator = initialize_integrator(initial_epoch, ['SRP'], [6])

    state_length = len(a_priori_state)

    # Integrate new a priori state forward in time to compare to truth data and see if we can visually identify maneuver times from the residuals
    _, augmented_x_hist = integrator.integrate_stm(time_vector[-1], a_priori_state, teval=time_vector)
    x_hist = augmented_x_hist[:7, :]  # Extract the state history from the augmented state history

    # Compute measurement errors for the new a priori state and plot these residuals to see if we can visually identify maneuver times from the residuals
    measurement_error_matrix = np.full((len(time_vector), len(station_mgrs)*2), np.nan)  # Initialize with nans
    for i, station in enumerate(station_mgrs):
        station_measurements = measurement_df[f"{station.station_name}_measurements"].dropna()
        simulated_measurements = station.simulate_measurements(x_hist, time_vector, 'ECI', noise=False, ignore_visibility=True)
        measurement_errors = np.vstack(station_measurements.values) - simulated_measurements.T
        measurement_error_matrix[:, i*2:(i+1)*2] = measurement_errors

    # Format measurement error matrix into a DataFrame for that is compatible with the plotting functions
    measurement_error_df = pd.DataFrame({
        'time': time_vector,
        f"{station_mgrs[0].station_name}_range_error": measurement_error_matrix[:, 0],
        f"{station_mgrs[0].station_name}_range_rate_error": measurement_error_matrix[:, 1],
        f"{station_mgrs[1].station_name}_range_error": measurement_error_matrix[:, 2],
        f"{station_mgrs[1].station_name}_range_rate_error": measurement_error_matrix[:, 3],
        f"{station_mgrs[2].station_name}_range_error": measurement_error_matrix[:, 4],
        f"{station_mgrs[2].station_name}_range_rate_error": measurement_error_matrix[:, 5]
    })

    fig = make_subplots(rows=3, cols=2, subplot_titles=(f"{station_mgrs[0].station_name} Range Error", f"{station_mgrs[0].station_name} Range Rate Error", f"{station_mgrs[1].station_name} Range Error", f"{station_mgrs[1].station_name} Range Rate Error", f"{station_mgrs[2].station_name} Range Error", f"{station_mgrs[2].station_name} Range Rate Error"))
    for i, station in enumerate(station_mgrs):
        fig.add_trace(go.Scatter(x=measurement_error_df['time'], y=measurement_error_df[f"{station.station_name}_range_error"], mode='markers', name=f"{station.station_name} Range Error"), row=i+1, col=1)
        fig.add_trace(go.Scatter(x=measurement_error_df['time'], y=measurement_error_df[f"{station.station_name}_range_rate_error"], mode='markers', name=f"{station.station_name} Range Rate Error"), row=i+1, col=2)
    fig.update_layout(height=900, width=1200, title_text="Measurement Residuals for A Priori State Estimate")
    fig.show()
    