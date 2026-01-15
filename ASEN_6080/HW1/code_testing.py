import numpy as np
import json
from generic_functions import perturbation_jacobian
from integrator import Integrator
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Question 1 Testing Code ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Read in test data from prob1c_solution.json
with open('prob1c_solution.json', 'r') as f:
    test_data = json.load(f)

# Pull out the necessary parameters
state = test_data['inputs']['state']
truth_jacobian = np.array(test_data['outputs']['A_matrix']['values'])
r = np.array(state['r'])
v = np.array(state['v'])
mu = state['mu']
J2 = state['J2']
J3 = state['J3']
R_e = 6378

# Call the perturbation_partials function

A = perturbation_jacobian(r, v, mu, J2, J3, R_e)

# Compare the computed Jacobian to the truth Jacobian
diff = A - truth_jacobian
print("Difference between computed and truth Jacobian:")
np.set_printoptions(linewidth=200)
print(diff)

# Question 2 Testing Code --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

a = 10000
e = 0.01
i = np.deg2rad(40)
LoN = np.deg2rad(80)
AoP = np.deg2rad(40)
f = np.deg2rad(0)
mu = 398600.4418
period = 2 * np.pi * np.sqrt(a**3 / mu)
mode = 'J2'
state_length = 7 # <--- Change this depending on mode

# Create an Integrator instance and convert state to Cartesian
integrator = Integrator(mu, R_e, mode)
r_vec, v_vec = integrator.keplerian_to_cartesian(a, e, i, LoN, AoP, f)

initial_state = np.hstack((r_vec, v_vec, J2)) # <--- Change this depending on mode

perturbation = np.array([1, 0, 0, 0, 0.010, 0, 0])
perturbed_state = initial_state + perturbation

# Integrate for 15 orbital periods
reference_time, reference_state_history = integrator.integrate_eom(15 * period, initial_state)
perturbed_time, perturbed_state_history = integrator.integrate_eom(15 * period, perturbed_state, reference_time)

# Compute deviation between reference and perturbed states
deviation_state = perturbed_state_history - reference_state_history

# Integrate with STM
stm_time, stm_history = integrator.integrate_stm(15 * period, initial_state, reference_time)

# Propagate the initial perturbation through the STM history

estimated_deviation = []
for column in stm_history.T:
    column_state = column[0:state_length]
    phi = column[state_length:].reshape((state_length, state_length))
    propagated_deviation = phi @ perturbation
    estimated_deviation.append(propagated_deviation)

estimated_deviation = np.array(estimated_deviation).T

# Testing STM against provided solution ----------------------------------------------------------------------------------------------------------------------------------------------------------------

# NEED TO KNOW HIS MU

# with open('prob2b_solution.json', 'r') as f:
#     test_data = json.load(f)

# initial_state = test_data['inputs']['X0']['values']

# # Integrate with STM
# stm_time, stm_history = integrator.integrate_stm(15 * period, initial_state)

# # Propagate the initial perturbation through the STM history

# estimated_deviation = []
# for column in stm_history.T:
#     column_state = column[0:state_length]
#     phi = column[state_length:].reshape((state_length, state_length))
#     propagated_deviation = phi @ perturbation
#     estimated_deviation.append(propagated_deviation)

# estimated_deviation = np.array(estimated_deviation).T


# Figure generation ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Plot the orbit
fig = go.Figure()
fig.add_trace(go.Scatter3d(x=reference_state_history[0, :],
                   y=reference_state_history[1, :],
                   z=reference_state_history[2, :],
                   mode='lines',
                   line=dict(width=2, color='blue'),
                   name='Reference Orbit'))
fig.add_trace(go.Scatter3d(
    x=perturbed_state_history[0, :],
    y=perturbed_state_history[1, :],
    z=perturbed_state_history[2, :],
    mode='lines',
    line=dict(width=2, color='red'),
    name='Perturbed Orbit'
))
fig.update_layout(
    title='Orbit Trajectories',
    scene=dict(
        xaxis_title='X (km)',
        yaxis_title='Y (km)',
        zaxis_title='Z (km)',
        aspectmode='data'
    )
)
fig.write_html('figures\orbit_trajectories.html')

# Plot the deviation in position coordinates over time in subplots
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=('X Deviation', 'Y Deviation', 'Z Deviation'))
fig.add_trace(go.Scatter(x=reference_time / period, y=deviation_state[0, :], mode='lines', name='Propagated Deviation', line=dict(color='blue')), row=1, col=1)
fig.add_trace(go.Scatter(x=stm_time / period, y=estimated_deviation[0, :], mode='lines', name='STM Estimated Deviation', line=dict(color='red')), row=1, col=1)
fig.add_trace(go.Scatter(x=reference_time / period, y=deviation_state[1, :], mode='lines', name='Propagated Deviation', line=dict(color='blue')), row=2, col=1)
fig.add_trace(go.Scatter(x=stm_time / period, y=estimated_deviation[1, :], mode='lines', name='STM Estimated Deviation', line=dict(color='red')), row=2, col=1)
fig.add_trace(go.Scatter(x=reference_time / period, y=deviation_state[2, :], mode='lines', name='Propagated Deviation', line=dict(color='blue')), row=3, col=1)
fig.add_trace(go.Scatter(x=stm_time / period, y=estimated_deviation[2, :], mode='lines', name='STM Estimated Deviation', line=dict(color='red')), row=3, col=1)
fig.update_xaxes(title_text='Time (Orbital Periods)', row=3, col=1)
fig.update_yaxes(title_text='Deviation (km)', row=1, col=1)
fig.update_yaxes(title_text='Deviation (km)', row=2, col=1)
fig.update_yaxes(title_text='Deviation (km)', row=3, col=1)
fig.update_layout(title='Position Deviations Over Time', height=900)
fig.write_html("figures/position_deviations.html")

# Plot the deviation in velocity coordinates over time in subplots
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=('U Deviation', 'V Deviation', 'W Deviation'))
fig.add_trace(go.Scatter(x=reference_time / period, y=deviation_state[3, :], mode='lines', name='Propagated Deviation', line=dict(color='blue')), row=1, col=1)
fig.add_trace(go.Scatter(x=stm_time / period, y=estimated_deviation[3, :], mode='lines', name='STM Estimated Deviation', line=dict(color='red')), row=1, col=1)
fig.add_trace(go.Scatter(x=reference_time / period, y=deviation_state[4, :], mode='lines', name='Propagated Deviation', line=dict(color='blue')), row=2, col=1)
fig.add_trace(go.Scatter(x=stm_time / period, y=estimated_deviation[4, :], mode='lines', name='STM Estimated Deviation', line=dict(color='red')), row=2, col=1)
fig.add_trace(go.Scatter(x=reference_time / period, y=deviation_state[5, :], mode='lines', name='Propagated Deviation', line=dict(color='blue')), row=3, col=1)
fig.add_trace(go.Scatter(x=stm_time / period, y=estimated_deviation[5, :], mode='lines', name='STM Estimated Deviation', line=dict(color='red')), row=3, col=1)
fig.update_xaxes(title_text='Time (Orbital Periods)', row=3, col=1)
fig.update_yaxes(title_text='Deviation (km/s)', row=1, col=1)
fig.update_yaxes(title_text='Deviation (km/s)', row=2, col=1)
fig.update_yaxes(title_text='Deviation (km/s)', row=3, col=1)
fig.update_layout(title='Velocity Deviations Over Time', height=900)
fig.write_html("figures/velocity_deviations.html")

# Subplots with difference in deviations ------------------------------------------------------------------------------------------------------------------------------------------------

fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=('X Deviation Difference', 'Y Deviation Difference', 'Z Deviation Difference'))
fig.add_trace(go.Scatter(x=reference_time / period, y=deviation_state[0, :] - estimated_deviation[0, :], mode='lines', name='Deviation Difference', line=dict(color='green')), row=1, col=1)
fig.add_trace(go.Scatter(x=reference_time / period, y=deviation_state[1, :] - estimated_deviation[1, :], mode='lines', name='Deviation Difference', line=dict(color='green')), row=2, col=1)
fig.add_trace(go.Scatter(x=reference_time / period, y=deviation_state[2, :] - estimated_deviation[2, :], mode='lines', name='Deviation Difference', line=dict(color='green')), row=3, col=1)
fig.update_xaxes(title_text='Time (Orbital Periods)', row=3, col=1)
fig.update_yaxes(title_text='Deviation Difference (km)', row=1, col=1)
fig.update_yaxes(title_text='Deviation Difference (km)', row=2, col=1)
fig.update_yaxes(title_text='Deviation Difference (km)', row=3, col=1)
fig.update_layout(title='Position Deviation Differences Over Time', height=900)
fig.write_html("figures/position_deviation_differences.html")

# Velocity deviation differences
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=('U Deviation Difference', 'V Deviation Difference', 'W Deviation Difference'))
fig.add_trace(go.Scatter(x=reference_time / period, y=deviation_state[3, :] - estimated_deviation[3, :], mode='lines', name='Deviation Difference', line=dict(color='green')), row=1, col=1)
fig.add_trace(go.Scatter(x=reference_time / period, y=deviation_state[4, :] - estimated_deviation[4, :], mode='lines', name='Deviation Difference', line=dict(color='green')), row=2, col=1)
fig.add_trace(go.Scatter(x=reference_time / period, y=deviation_state[5, :] - estimated_deviation[5, :], mode='lines', name='Deviation Difference', line=dict(color='green')), row=3, col=1)
fig.update_xaxes(title_text='Time (Orbital Periods)', row=3, col=1)
fig.update_yaxes(title_text='Deviation Difference (km/s)', row=1, col=1)
fig.update_yaxes(title_text='Deviation Difference (km/s)', row=2, col=1)
fig.update_yaxes(title_text='Deviation Difference (km/s)', row=3, col=1)
fig.update_layout(title='Velocity Deviation Differences Over Time', height=900)
fig.write_html("figures/velocity_deviation_differences.html")
