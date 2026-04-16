import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import scipy.io
from constants import truth_data_file_path, known_dynamics_measurement_file_path, mu_sun, mu_earth, R_e, solar_flux, SRP_area_to_mass, AU, initial_epoch, initial_epoch_jd, initial_spin_angle, earth_spin_rate, station_locations, part_2_station_locations, observation_noise
from Tools.measurement_manager import MeasurementMgr
from Tools.integrator import Integrator
np.set_printoptions(linewidth=200)
"""
This file confirms that simulated measurements are being generated correctly by generating measurements about the truth trajectory and comparing them to the expected values. The tasks to test include:

"""

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

if __name__ == "__main__":
    # Load truth data
    truth_data = load_truth_data(truth_data_file_path)
    truth_time_vector = truth_data['time_vector']
    truth_state_vectors = truth_data['state_vectors'].T

    # Load measurement data
    truth_measurement_data = load_measurement_data(known_dynamics_measurement_file_path)
    measurement_time_vector = truth_measurement_data['time_vector']
    measurement_vectors = truth_measurement_data['measurements']

    # Integrate truth state to get values at measurement times
    initial_state = truth_state_vectors[:7, 0]  # Initial state from the truth data

    C_r = initial_state[6]
    # Initialize integrator with appropriate parameters for the test

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

    t_f = measurement_time_vector[-1]  # Final time from the truth dataq

    # Integrate the equations of motion using the integrator
    _, integrated_states = integrator.integrate_eom(t_f, initial_state, teval=measurement_time_vector)

    # # Plotting the results to compare the integrated trajectory to the truth data
    # fig = go.Figure()
    # fig.add_trace(go.Scatter3d(
    #     x=integrated_states[0, :], y=integrated_states[1, :], z=integrated_states[2, :],
    #     mode='markers',
    #     name='Integrated Trajectory',
    #     marker=dict(size=3, color='red')
    # ))
    # fig.update_layout(
    #     title='Comparison of Integrated Trajectory to Truth Data',
    #     scene=dict(
    #         xaxis_title='X (km)',
    #         yaxis_title='Y (km)',
    #         zaxis_title='Z (km)'
    #     )
    # )
    # fig.show()

    # Initialize measurement managers
    DSS34_measurement_mgr = MeasurementMgr(
        station_name='DSS34',
        station_lat=part_2_station_locations['DSS34']['lat'],
        station_lon=part_2_station_locations['DSS34']['lon'],
        initial_earth_spin_angle=initial_spin_angle,
        earth_spin_rate=earth_spin_rate,
        R_e=part_2_station_locations['DSS34']['radius']
    )

    DSS65_measurement_mgr = MeasurementMgr(
        station_name='DSS65',
        station_lat=part_2_station_locations['DSS65']['lat'],
        station_lon=part_2_station_locations['DSS65']['lon'],
        initial_earth_spin_angle=initial_spin_angle,
        earth_spin_rate=earth_spin_rate,
        R_e=part_2_station_locations['DSS65']['radius']
    )

    DSS13_measurement_mgr = MeasurementMgr(
        station_name='DSS13',
        station_lat=part_2_station_locations['DSS13']['lat'],
        station_lon=part_2_station_locations['DSS13']['lon'],
        initial_earth_spin_angle=initial_spin_angle,
        earth_spin_rate=earth_spin_rate,
        R_e=part_2_station_locations['DSS13']['radius']
    )

    measurement_mgr_list = [DSS34_measurement_mgr, DSS65_measurement_mgr, DSS13_measurement_mgr]
    # Using the integrated states, simulate measurements for each station at the measurement times and compare to the loaded measurement data.
    # This will confirm that the measurement simulation is working correctly and that the measurements are being generated about the truth
    # trajectory as expected.

    measurement_errors = np.zeros(measurement_vectors.shape)  # Initialize an array to store measurement errors
    truth_simulated_measurements_mat = np.zeros((truth_state_vectors.shape[1], measurement_vectors.shape[1]))  # Initialize an array to store simulated measurements for comparison
    for i, measurement_mgr in enumerate(measurement_mgr_list):
        truth_simulated_measurements = measurement_mgr.simulate_measurements(truth_state_vectors, truth_time_vector, coordinate_frame='ECI', noise=True, noise_sigma=np.diag(observation_noise), ignore_visibility=False).T
        truth_simulated_measurements_mat[:, 2*i:2*i+2] = truth_simulated_measurements  # Store the simulated measurements for this station
        simulated_measurements = measurement_mgr.simulate_measurements(integrated_states, measurement_time_vector, coordinate_frame='ECI', noise=True, noise_sigma=np.diag(observation_noise), ignore_visibility=False).T
        range_error = simulated_measurements[:, 0] - measurement_vectors[:, i]  # Assuming range measurements are in even columns
        range_rate_error = simulated_measurements[:, 1] - measurement_vectors[:, i+3]  # Assuming range rate measurements are in odd columns
        measurement_error = np.vstack((range_error, range_rate_error)).T  # Combine range and range rate errors into a single array
        measurement_errors[:, 2*i:2*i+2] = measurement_error  # Store the measurement error for this station

    # Plot the measurement errors over time for each station
    fig = make_subplots(rows=3, cols=2, subplot_titles=('DSS34 Range Error', 'DSS34 Range Rate Error', 'DSS65 Range Error', 'DSS65 Range Rate Error', 'DSS13 Range Error', 'DSS13 Range Rate Error'))
    for i in range(3):
        fig.add_trace(go.Scatter(x=measurement_time_vector, y=measurement_errors[:, 2*i], mode='markers', name=f'{measurement_mgr_list[i].station_name} Range Error'), row=i+1, col=1)
        fig.add_trace(go.Scatter(x=measurement_time_vector, y=measurement_errors[:, 2*i+1], mode='markers', name=f'{measurement_mgr_list[i].station_name} Range Rate Error'), row=i+1, col=2)
    fig.update_layout(title='Measurement Errors Over Time', showlegend=False)
    for i in range(1, 4):
        for j in range(1, 3):
            fig.update_xaxes(title_text='Time (s)', row=i, col=j)
            if j == 1:
                fig.update_yaxes(title_text='Range Error (km)', showexponent="all", exponentformat="e", row=i, col=j)
            else:
                fig.update_yaxes(title_text='Range Rate Error (km/s)', showexponent="all", exponentformat="e", row=i, col=j)
    fig.write_html('ASEN_6080/Project2/figures/measurement_errors.html')

    # Plot the simulated measurements about the truth trajectory with the truth measurements for comparison
    fig = make_subplots(rows=3, cols=2, subplot_titles=('DSS34 Range', 'DSS34 Range Rate', 'DSS65 Range', 'DSS65 Range Rate', 'DSS13 Range', 'DSS13 Range Rate'))
    for i in range(3):
        fig.add_trace(go.Scatter(x=truth_time_vector, y=truth_simulated_measurements_mat[:, 2*i], mode='markers', name=f'{measurement_mgr_list[i].station_name} Simulated Range'), row=i+1, col=1)
        fig.add_trace(go.Scatter(x=measurement_time_vector, y=measurement_vectors[:, i], mode='markers', name=f'{measurement_mgr_list[i].station_name} Truth Range'), row=i+1, col=1)
        fig.add_trace(go.Scatter(x=truth_time_vector, y=truth_simulated_measurements_mat[:, 2*i+1], mode='markers', name=f'{measurement_mgr_list[i].station_name} Simulated Range Rate'), row=i+1, col=2)
        fig.add_trace(go.Scatter(x=measurement_time_vector, y=measurement_vectors[:, i+3], mode='markers', name=f'{measurement_mgr_list[i].station_name} Truth Range Rate'), row=i+1, col=2)
    fig.update_layout(title='Simulated Measurements About Truth Trajectory vs Truth Measurements', showlegend=False)
    for i in range(1, 4):
        for j in range(1, 3):
            fig.update_xaxes(title_text='Time (s)', row=i, col=j)
            if j == 1:
                fig.update_yaxes(title_text='Range (km)', showexponent="all", exponentformat="e", row=i, col=j)
            else:
                fig.update_yaxes(title_text='Range Rate (km/s)', showexponent="all", exponentformat="e", row=i, col=j)
    fig.write_html('ASEN_6080/Project2/figures/simulated_measurements_vs_truth.html')