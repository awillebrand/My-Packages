import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import scipy.io
from constants import truth_data_file_path, known_dynamics_measurement_file_path, mu_sun, mu_earth, R_e, solar_flux, SRP_area_to_mass, C_r, initial_epoch, initial_epoch_jd, initial_spin_angle, earth_spin_rate, station_locations, part_2_station_locations, observation_noise, a_priori_state, a_priori_covariance, RSOI, B_plane_target_coords
from Tools.measurement_manager import MeasurementMgr
from Tools.integrator import Integrator
from Tools.batch_lls_estimator import BatchLLSEstimator
from Tools.LKF import LKF
from Tools.EKF import EKF
from Tools.SRIF import SRIF
from Tools.UKF import UKF
from Tools.B_Plane_manager import BPlaneMgr
from Tools.generic_functions import covariance_ellipse_2D
np.set_printoptions(linewidth=200)
"""
This file performs filtering on the provided measurement data for part 2 of the project which uses the known dynamics model.
"""
    
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

def convert_measurements_to_df(measurements : dict, station_names : list, days_of_data : float):
    """
    Convert the measurement data into a pandas DataFrame to make it compatible with existing filtering code.

    Parameters
    ----------
    measurements : dict
        A dictionary containing the time vector and measurement vectors.
    station_names : list
        A list of station names corresponding to the measurements (e.g., ['DSS34', 'DSS65', 'DSS13']).
    days_of_data : float
        The number of days of data to include in the DataFrame. This will be used to filter the measurements to only include those within the specified time frame.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the time and measurements with appropriate column names.
    """

    time_vector = measurements['time_vector']  # Use the original time vector from the measurements to ensure consistency with measurement times
    measurement_vectors = measurements['measurements']

    # Find index where time exceeds the specified number of days and truncate the time vector and measurement vectors accordingly
    max_time = days_of_data * 24 * 3600  # Convert days to seconds
    
    valid_indices = time_vector <= max_time

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
        estimation_mode=['SRP'],
        parameter_indices=[6],
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

def event_3RSOI_crossing(t, y, DMC, beta_mat):
    """
    Event function to detect when the spacecraft crosses 3 times the radius of the sphere of influence (RSOI) of Earth.

    Parameters
    ----------
    t : float
        The current time in seconds.
    y : np.ndarray
        The current state vector of the spacecraft (position and velocity).
    RSOI : float
        The radius of the sphere of influence for Earth in km (default is 925,000 km).

    Returns
    -------
    float
        The value of the event function, which is the distance from Earth minus 3 times the RSOI. When this value crosses zero, it indicates a crossing event.
    """
    r_vec = y[:3]  # Extract position vector from state vector
    r_norm = np.linalg.norm(r_vec)  # Compute the norm of the position vector

    return r_norm - 3*RSOI  # Return the distance from Earth minus 3 times the RSOI

def integrate_to_3RSOI(DCO_state, DCO_epoch, t_final = 1e8):
    """
    Integrate the spacecraft's trajectory from the provided DCO state to the point where it crosses 3 times the radius of the sphere of influence (RSOI) of Earth.

    Parameters
    ----------
    DCO_state : np.ndarray
        The initial state vector of the spacecraft (position and velocity).
    t_final : float
        The maximum time to integrate to in seconds (default is 1e8 seconds). This should be sufficiently large to ensure that the crossing event occurs within this time frame.
    Returns
    -------
    tuple
        A tuple containing the time of RSOI crossing and the state vector at the time of crossing.
    """
    integrator = initialize_integrator(DCO_epoch)

    # Use solve_ivp with the event function to integrate until crossing 3*RSOI
    t_events, y_events = integrator.integrate_stm(t_final, DCO_state, events=event_3RSOI_crossing)

    if len(t_events[0]) > 0:  # Check if the event was triggered
        crossing_time = t_events[0][0]  # Get the time of crossing
        crossing_state = y_events[0][0]  # Get the state at the time of crossing
        return crossing_time, crossing_state[:len(DCO_state)]  # Return the state vector (position and velocity) at crossing
    else:
        raise RuntimeError("Integration did not reach the 3*RSOI crossing event within the specified time span.")
    
def get_LTOF_to_B_Plane(v_inf_state):
    """
    Compute the LTOF to the B-plane given the hyperbolic excess velocity state vector at RSOI.

    Parameters
    ----------
    v_inf_state : np.ndarray
        The state vector of the spacecraft at RSOI, which should include position and velocity components.
    mu : float
        The gravitational parameter of Earth in km^3/s^2 (default is 3.986004415E5 km^3/s^2).

    Returns
    -------
    float
        The LTOF to the B-plane in seconds.
    """
    b_plane_manager = BPlaneMgr(v_inf_state, mu_earth)
    LTOF = b_plane_manager.compute_LOTF()

    return LTOF

def integrate_to_B_plane_crossing(DCO_state : np.ndarray, DCO_epoch : float, B_plane_crossing_epoch : float):
    """
    Integrate the spacecraft's trajectory from the provided initial state to the point where it crosses the B-plane.

    Parameters
    ----------
    DCO_state : np.ndarray
        The DCO state vector of the spacecraft (position and velocity).
    DCO_epoch : float
        The DCO epoch corresponding to the DCO state in seconds.
    B_plane_crossing_epoch : float
        The epoch at which the spacecraft is expected to cross the B-plane in seconds. This can be computed as the DCO epoch plus the LTOF to the B-plane.

    Returns
    -------
    tuple
        A tuple containing the time of B-plane crossing, the state vector, and the STM at the time of crossing.
    """
    # Initialize integrator with the initial epoch
    integrator = initialize_integrator(DCO_epoch)

    # Integrate the trajectory to the B-plane crossing epoch
    sol = integrator.integrate_stm(B_plane_crossing_epoch, DCO_state, teval=np.array([B_plane_crossing_epoch]))

    crossing_time = sol[0][0]  # Get the time of crossing
    augmented_crossing_state = sol[1][:, 0]  # Get the state at the time of crossing

    state_length = len(DCO_state)
    crossing_state = augmented_crossing_state[:state_length]  # Extract the state vector (position and velocity) from the augmented state
    crossing_stm = augmented_crossing_state[state_length:].reshape((state_length, state_length))  # Extract the STM from the augmented state

    return crossing_time, crossing_state, crossing_stm

if __name__ == "__main__":
    filter_to_run = input("Enter the filter to run (Batch, LKF, EKF, SRIF, UKF): ")
    if filter_to_run not in ['Batch', 'LKF', 'EKF', 'SRIF', 'UKF']:
        print("Invalid filter choice. Please enter one of the following: Batch, LKF, EKF, SRIF, UKF")
        exit()

    # Plot the B-plane crossing point and covariance ellipse in the B-plane frame
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[B_plane_target_coords[0]], y=[B_plane_target_coords[1]], mode='markers', name='B-plane Target Point', marker=dict(color='orange', size=5)))

    times_in_consideration = [50, 100, 150, 200]
    colors = ['blue', 'green', 'red', 'purple']
    for i, time in enumerate(times_in_consideration):
        # Load the measurement data
        measurements = load_measurement_data(known_dynamics_measurement_file_path)

        # Convert the measurement data into a DataFrame
        station_names = ['DSS34', 'DSS65', 'DSS13']
        measurement_df = convert_measurements_to_df(measurements, station_names, time)

        meas_time_vector = measurement_df['time'].values

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

        # Initialize the integrator with the appropriate initial epoch and gravitational parameter
        integrator = initialize_integrator(initial_epoch)

        # Perform filtering using the loaded measurement data
        if filter_to_run == 'Batch':
            if i == 0:  # Only ask for these parameters once since they are the same for all runs
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
            _, augmented_x_hist = integrator.integrate_stm(meas_time_vector[-1], x, teval=meas_time_vector)
            x_hist = augmented_x_hist[:7, :]  # Extract the state history from the augmented state history
            STM_hist = augmented_x_hist[7:, :]  # Extract the STM history from the augmented state history
            P_hist = np.zeros((7,7, len(meas_time_vector)))  # Initialize an array to hold the covariance history
            for j in range(len(meas_time_vector)):
                STM = STM_hist[:, j].reshape((7, 7))  # Reshape the STM from the augmented state history
                P_hist[:, :, j] = STM @ a_priori_covariance @ STM.T  # Propagate the covariance using the STM
        elif filter_to_run == 'LKF':
            if i == 0:  # Only ask for these parameters once since they are the same for all runs
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
            if i == 0:  # Only ask for these parameters once since they are the same for all runs
                start_mode = str(input("Enter Start Mode for EKF ('Warm' or 'Cold'): "))
            if start_mode.lower() == 'warm':
                start_length = int(input("Enter the number of measurements to use for the hot start (e.g., 10): "))
            else:
                start_length = 0
            filter = EKF(integrator, station_mgrs, initial_earth_spin_angle=0, earth_rotation_rate=earth_spin_rate)
            x_hist, P_hist, residuals_df = filter.run(a_priori_state, np.zeros(7), a_priori_covariance, measurement_df, R=observation_noise, start_mode=start_mode.lower(), start_length=start_length)
        elif filter_to_run == 'SRIF':
            if i == 0:  # Only ask for these parameters once since they are the same for all runs
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
            if i == 0:  # Only ask for these parameters once since they are the same for all runs
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

        # Take final state estimate from the filter as DCO state and epoch
        DCO_state = x_hist[:, -1]
        DCO_epoch = meas_time_vector[-1]
        
        # Integrate to 3*RSOI crossing and get LTOF to B-plane
        RSOI_crossing_time, RSOI_crossing_state = integrate_to_3RSOI(DCO_state, DCO_epoch)
        LTOF_to_B_plane = get_LTOF_to_B_Plane(RSOI_crossing_state)

        # Integrate to B-plane crossing
        B_plane_crossing_epoch = RSOI_crossing_time + LTOF_to_B_plane
        B_plane_crossing_time, B_plane_crossing_state, B_plane_crossing_stm = integrate_to_B_plane_crossing(DCO_state, DCO_epoch, B_plane_crossing_epoch)

        # Map final filter covariance to B-plane crossing time using the STM
        final_covariance = P_hist[:, :, -1]  # Get the final covariance from the filter
        B_plane_crossing_covariance = B_plane_crossing_stm @ final_covariance @ B_plane_crossing_stm.T  # Propagate the covariance to the B-plane crossing time using the STM

        print("B-plane Crossing State:", B_plane_crossing_state)
        print("B-plane Crossing Covariance:", B_plane_crossing_covariance)

        # Rotate the B-plane crossing state and covariance into the B-plane frame using the DCM from the BPlaneMgr
        b_plane_manager = BPlaneMgr(RSOI_crossing_state, mu_earth)
        s_hat, t_hat, r_hat, B_vec = b_plane_manager.compute_b_plane_frame()

        # B·T and B·R are the projections of the B-vector onto the T and R axes
        B_dot_T = np.dot(B_vec, t_hat)
        B_dot_R = np.dot(B_vec, r_hat)
        center = -np.array([B_dot_T, B_dot_R])
        
        DCM_ECI_to_B_plane = b_plane_manager.compute_b_plane_DCM()
        # B_plane_crossing_pos_in_B_plane_frame = DCM_ECI_to_B_plane @ B_plane_crossing_state[0:3]
        B_plane_crossing_pos_covariance_in_B_plane_frame = DCM_ECI_to_B_plane @ B_plane_crossing_covariance[:3,:3] @ DCM_ECI_to_B_plane.T

        # Compute the covariance ellipse in the B-plane frame
        
        #center = B_plane_crossing_pos_in_B_plane_frame[1:3]  # The center of the ellipse is given by the y and z components of the state in the B-plane frame
        reduced_covariance = B_plane_crossing_pos_covariance_in_B_plane_frame[1:3, 1:3]  # The covariance for the ellipse is given by the y and z components of the covariance in the B-plane frame
        b_plane_covariance_ellipse = covariance_ellipse_2D(center, reduced_covariance, n_std=3)  # Compute the covariance ellipse at 3-sigma

        #fig.add_trace(go.Scatter(x=[B_plane_crossing_pos_in_B_plane_frame[1]], y=[B_plane_crossing_pos_in_B_plane_frame[2]], mode='markers', name=f'{time} days', marker=dict(color=colors[i], size=5)))
        fig.add_trace(go.Scatter(x=[center[0]], y=[center[1]], mode='markers', name=f'{time} days', marker=dict(color=colors[i], size=5)))
        fig.add_trace(go.Scatter(x=b_plane_covariance_ellipse[:, 0], y=b_plane_covariance_ellipse[:, 1], mode='lines', name=f'{time} days', marker=dict(color=colors[i]), showlegend=False))

    fig.update_layout(title='B-plane Crossing Point and Covariance Ellipses',
                      xaxis_title='B-plane T (km)',
                      yaxis_title='B-plane R (km)',
                      yaxis=dict(autorange='reversed'),
                      legend=dict(x=0.8, y=0.95))
    fig.show()

