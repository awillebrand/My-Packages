import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from generic_functions import load_truth_data
from Tools.integrator import Integrator
from constants import mu_sun, mu_earth, R_e, solar_flux, SRP_area_to_mass, initial_epoch, initial_epoch_jd, earth_spin_rate, truth_data_file_path

def run_task_1_analysis():
    # Load truth data
    truth_data = load_truth_data(truth_data_file_path)

    time_vector = truth_data['time_vector']
    state_vectors = truth_data['state_vectors'].T

    initial_state = state_vectors[:7, 0]  # Initial state from the truth data

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

    t_f = time_vector[-1]  # Final time from the truth data
    
    # Integrate the equations of motion using the integrator
    _, augmented_integrated_states = integrator.integrate_stm(t_f, initial_state, teval=time_vector)

    # Plotting the results to compare the integrated trajectory to the truth data
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=state_vectors[0, :], y=state_vectors[1, :], z=state_vectors[2, :],
        mode='markers',
        name='Truth Data',
        marker=dict(size=3, color='blue')
    ))
    fig.add_trace(go.Scatter3d(
        x=augmented_integrated_states[0, :], y=augmented_integrated_states[1, :], z=augmented_integrated_states[2, :],
        mode='markers',
        name='Integrated Trajectory',
        marker=dict(size=3, color='red')
    ))
    fig.update_layout(
        title=dict(
            text='Comparison of Integrated Trajectory to Truth Data', 
            font=dict(size=30)  # Increased from 20
        ),
        scene=dict(
            xaxis=dict(
                title=dict(text='X (km)', font=dict(size=18)),
                tickfont=dict(size=16)  # Larger axis ticks
            ),
            yaxis=dict(
                title=dict(text='Y (km)', font=dict(size=18)),
                tickfont=dict(size=16)
            ),
            zaxis=dict(
                title=dict(text='Z (km)', font=dict(size=18)),
                tickfont=dict(size=16)
            )
        ),
        legend=dict(
            font=dict(size=20),  # Larger legend text
            itemsizing='constant' # Keeps the legend symbols visible
        )
    )

    fig.write_html('ASEN_6080/Project2/final_figures/dynamics_test_trajectory_comparison.html')

    # Compute the state error between the integrated trajectory and the truth data
    state_error = augmented_integrated_states[0:7,:] - state_vectors[0:7,:]  # Compute error for the first 7 state components (position, velocity, and C_r)
    
    # Compute relative state error for each component
    relative_state_error = np.zeros(state_error.shape)
    for i in range(state_error.shape[0]):
        relative_state_error[i, :] = state_error[i, :] / np.linalg.norm(state_vectors[i, :])  # Relative error normalized by the norm of the truth state component

    # Plot the state error over time
    fig = make_subplots(rows=3, cols=2, subplot_titles=('X Error', 'Vx Error', 'Y Error', 'Vy Error', 'Z Error', 'Vz Error'))
    fig.add_trace(go.Scatter(x=time_vector, y=state_error[0, :], mode='lines', name='X Error'), row=1, col=1)
    fig.add_trace(go.Scatter(x=time_vector, y=state_error[1, :], mode='lines', name='Y Error'), row=2, col=1)
    fig.add_trace(go.Scatter(x=time_vector, y=state_error[2, :], mode='lines', name='Z Error'), row=3, col=1)
    fig.add_trace(go.Scatter(x=time_vector, y=state_error[3, :], mode='lines', name='Vx Error'), row=1, col=2)
    fig.add_trace(go.Scatter(x=time_vector, y=state_error[4, :], mode='lines', name='Vy Error'), row=2, col=2)
    fig.add_trace(go.Scatter(x=time_vector, y=state_error[5, :], mode='lines', name='Vz Error'), row=3, col=2)
    fig.update_layout(title='State Error Between Integrated Trajectory and Truth Data', showlegend=False, height = 800, width=1200)
    fig.update_layout(title_font_size=30)  # Increase main title font size
    fig.update_annotations(font_size=20)  # Increase subplot title font size


    for i in range(1, 4):
        for j in range(1, 3):
            fig.update_xaxes(title=dict(text='Time (s)', font=dict(size=18)), row=i, col=j)
            if j == 1:
                fig.update_yaxes(title=dict(text='Relative Position Error', font=dict(size=18)), showexponent="all", exponentformat="e", row=i, col=j)
            else:
                fig.update_yaxes(title=dict(text='Relative Velocity Error', font=dict(size=18)), showexponent="all", exponentformat="e", row=i, col=j)
    fig.write_html('ASEN_6080/Project2/final_figures/dynamics_test_state_error.html')
    fig.write_image('ASEN_6080/Project2/final_figures/pngs/dynamics_test_state_error.png')

    # Plot relative state error over time
    fig = make_subplots(rows=3, cols=2, subplot_titles=('Relative X Error', 'Relative Vx Error', 'Relative Y Error', 'Relative Vy Error', 'Relative Z Error', 'Relative Vz Error'))
    fig.add_trace(go.Scatter(x=time_vector, y=relative_state_error[0, :], mode='lines', name='Relative X Error'), row=1, col=1)
    fig.add_trace(go.Scatter(x=time_vector, y=relative_state_error[1, :], mode='lines', name='Relative Y Error'), row=2, col=1)
    fig.add_trace(go.Scatter(x=time_vector, y=relative_state_error[2, :], mode='lines', name='Relative Z Error'), row=3, col=1)
    fig.add_trace(go.Scatter(x=time_vector, y=relative_state_error[3, :], mode='lines', name='Relative Vx Error'), row=1, col=2)
    fig.add_trace(go.Scatter(x=time_vector, y=relative_state_error[4, :], mode='lines', name='Relative Vy Error'), row=2, col=2)
    fig.add_trace(go.Scatter(x=time_vector, y=relative_state_error[5, :], mode='lines', name='Relative Vz Error'), row=3, col=2)
    fig.update_layout(title='Relative State Error Between Integrated Trajectory and Truth Data', showlegend=False, height = 900, width=1200)
    fig.update_layout(title_font_size=30)  # Increase main title font size
    for i in range(1, 4):
        for j in range(1, 3):
            fig.update_xaxes(title=dict(text='Time (s)', font=dict(size=18)), row=i, col=j)
            if j == 1:
                fig.update_yaxes(title=dict(text='Relative Position Error', font=dict(size=18)), showexponent="all", exponentformat="e", row=i, col=j)
            else:
                fig.update_yaxes(title=dict(text='Relative Velocity Error', font=dict(size=18)), showexponent="all", exponentformat="e", row=i, col=j)
    fig.write_html('ASEN_6080/Project2/final_figures/dynamics_test_relative_state_error.html')
    fig.write_image('ASEN_6080/Project2/final_figures/pngs/dynamics_test_relative_state_error.png')
    
    # Compute difference between truth STM and integrated STM
    truth_stm = state_vectors[7:, :].reshape((7, 7, state_vectors.shape[1]), order='F')  # Reshape truth STM from the truth data
    integrated_stm = augmented_integrated_states[7:, :].reshape((7, 7, augmented_integrated_states.shape[1]))  # Reshape integrated STM from the integrator output
    stm_error = integrated_stm - truth_stm  # Compute STM error

    np.set_printoptions(linewidth=300, suppress=True, formatter={'float': lambda x: format(x, '1.3e')})  # Set print options for better readability
    print("Error in first STM:")
    print(stm_error[:, :, 0])
    print("Error in last STM:")
    print(stm_error[:, :, -1])
    # Print relative error in final STM with full precision
    relative_stm_error = np.zeros(stm_error.shape)
    print("Relative error in final STM:")
    for i in range(stm_error.shape[0]):
        for j in range(stm_error.shape[1]):
            relative_stm_error[i, j, -1] = stm_error[i, j, -1] / np.linalg.norm(truth_stm[i, j, -1]) # Relative error normalized by the norm of the truth STM element
            if np.isnan(relative_stm_error[i, j, -1]):  # Print only elements with relative error greater than a threshold
                relative_stm_error[i, j, -1] = 0.0  # Set relative error to zero if it is NaN (which can happen if the truth STM element is zero)
    print(relative_stm_error[:, :, -1])