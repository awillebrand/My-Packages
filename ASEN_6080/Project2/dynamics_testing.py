import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import scipy.io
from constants import truth_data_file_path, mu_sun, mu_earth, R_e, solar_flux, SRP_area_to_mass, AU, initial_epoch, initial_epoch_jd
from Tools.integrator import Integrator

"""
This file tests the dynamical model needed for Project 2 by comparing it to the given data set. The tasks to test include:
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

if __name__ == "__main__":
    # Load truth data
    truth_data = load_truth_data(truth_data_file_path)
    time_vector = truth_data['time_vector']
    state_vectors = truth_data['state_vectors'].T

    initial_state = state_vectors[:7, 0]  # Initial state from the truth data

    C_r = initial_state[6]  # For test, 7th element is true C_r value. This will need to be estimated for the actual problem, but for testing the dynamics we can use the true value to ensure the SRP effects are being calculated correctly.
    # Initialize integrator with appropriate parameters for the test
    integrator = Integrator(
        mu=mu_earth,
        R_e=R_e,        
        dynamical_mode=['mu', 'SRP', 'Third Body'],  # Include 2-body and SRP effects for this test
        estimation_mode=['SRP'],
        parameter_indices=[6],
        srp_area_to_mass=SRP_area_to_mass,  # Use the area-to-mass ratio from constants
        number_of_stations=0,
        solar_flux=solar_flux,
        mu_third_body=mu_sun,
        central_body='Earth',
        third_body='Sun'
    )

    t_f = time_vector[-1]  # Final time from the truth data
    
    # Integrate the equations of motion using the integrator
    _, integrated_states = integrator.integrate_eom(t_f, initial_state, teval=time_vector)

    # Plotting the results to compare the integrated trajectory to the truth data
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=state_vectors[0, :], y=state_vectors[1, :], z=state_vectors[2, :],
        mode='markers',
        name='Truth Data',
        marker=dict(size=3, color='blue')
    ))
    fig.add_trace(go.Scatter3d(
        x=integrated_states[0, :], y=integrated_states[1, :], z=integrated_states[2, :],
        mode='markers',
        name='Integrated Trajectory',
        marker=dict(size=3, color='red')
    ))
    fig.update_layout(
        title='Comparison of Integrated Trajectory to Truth Data',
        scene=dict(
            xaxis_title='X (km)',
            yaxis_title='Y (km)',
            zaxis_title='Z (km)'
        )
    )
    fig.show()