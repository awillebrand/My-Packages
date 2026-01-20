import numpy as np
import json
from generic_functions import state_jacobian, measurement_jacobian
from integrator import Integrator
from coordinate_manager import CoordinateMgr
from measurement_manager import MeasurementMgr
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

A = state_jacobian(r, v, mu, J2, J3, R_e)

# Compare the computed Jacobian to the truth Jacobian
diff = A - truth_jacobian
diff_percent = diff / truth_jacobian * 100
print("Percent difference between computed and truth Jacobian:")
np.set_printoptions(linewidth=200)
print(diff_percent)


# Question 2 Testing Code --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Testing orbit propagation
a = 10000
e = 0.001
i = np.deg2rad(40)
LoN = np.deg2rad(80)
AoP = np.deg2rad(40)
f = np.deg2rad(0)
mu = 3.986004415E5
period = 2 * np.pi * np.sqrt(a**3 / mu)
J2 = 0.0010826269
mode = 'J2'
state_length = 7 # <--- Change this depending on mode

# Create an Integrator instance and convert state to Cartesian
integrator = Integrator(mu, R_e, mode)
r_vec, v_vec = integrator.keplerian_to_cartesian(a, e, i, LoN, AoP, f)

initial_state = np.hstack((r_vec, v_vec, J2)) # <--- Change this depending on mode

perturbation = np.array([1, 0, 0, 0, 0.010, 0, 0])
perturbed_state = initial_state + perturbation

# Integrate for 15 orbital periods
time_list = np.arange(0, 15 * period, 10)
reference_time, reference_state_history = integrator.integrate_eom(15 * period, initial_state, time_list)
perturbed_time, perturbed_state_history = integrator.integrate_eom(15 * period, perturbed_state, time_list)

# Compute deviation between reference and perturbed states
deviation_state = perturbed_state_history - reference_state_history

# Integrate with STM
stm_time, stm_history = integrator.integrate_stm(15 * period, initial_state, teval=reference_time)

# Propagate the initial perturbation through the STM history

estimated_deviation = []
for column in stm_history.T:
    column_state = column[0:state_length]
    phi = column[state_length:].reshape((state_length, state_length))
    propagated_deviation = phi @ perturbation
    estimated_deviation.append(propagated_deviation)

estimated_deviation = np.array(estimated_deviation).T

# Pull given trajectory data for comparison
with open('HW1_truth.txt', 'r') as f:
    lines = f.readlines()
    truth_time = np.zeros(len(lines))
    truth_state_history = np.zeros((6, len(lines)))
    for i, line in enumerate(lines):
        data = line.split(' ')
        data[6] = data[6].strip()
        truth_time[i] = float(data[0])
        truth_state_history[:, i] = np.array([float(state) for state in data[1:]])

# Testing STM against provided solution

with open('prob2b_solution.json', 'r') as f:
    test_data = json.load(f)

initial_state = np.array(test_data['inputs']['X0']['values'])
initial_phi = np.array(test_data['inputs']['Phi0']['values']).reshape((state_length, state_length))

# Run through full_dynamics to get derivatives for test
derivative_state = integrator.full_dynamics(0, np.hstack((initial_state, initial_phi.flatten())))
state_dot = derivative_state[0:state_length]
phi_dot = derivative_state[state_length:].reshape((state_length, state_length))

output_state = np.array(test_data['outputs']['Xdot']['values'])
output_phi = np.array(test_data['outputs']['Phidot']['values']).reshape((state_length, state_length))

# Percent difference in state and STM derivatives
state_diff = state_dot - output_state
phi_diff = phi_dot - output_phi

percent_state_diff = state_diff / output_state * 100
percent_phi_diff = phi_diff / output_phi * 100
np.set_printoptions(linewidth=200)
print("Percent difference in state derivatives:")
print(percent_state_diff)
print("Percent difference in STM derivatives:")
print(percent_phi_diff)

# Question 3 Testing Code ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Load in test data from prob3_solution.json
with open('prob3b_solution.json', 'r') as f:
    test_data = json.load(f)

# Pull out necessary parameters
sc_pos = np.array(test_data['inputs']['spacecraft_state']['r'])
sc_vel = np.array(test_data['inputs']['spacecraft_state']['v'])

station_pos = np.array(test_data['inputs']['station_state']['Rs'])
station_vel = np.array(test_data['inputs']['station_state']['Vs'])

sc_state = np.hstack((sc_pos, sc_vel))
station_state = np.hstack((station_pos, station_vel))

truth_H_sc = np.array(test_data['outputs']['Htilde']['values'])

# Compute measurement Jacobian
H_sc, H_station = measurement_jacobian(sc_state, station_state)
print("Percent difference between computed and truth measurement Jacobian (spacecraft state):")
diff_H = H_sc - truth_H_sc
diff_H_percent = diff_H / truth_H_sc * 100
np.set_printoptions(linewidth=200)
print(diff_H_percent)

with open('prob3d_solution.json', 'r') as f:
    test_data = json.load(f)

truth_H_station = np.array(test_data['outputs']['Htilde']['values'])

print("Percent difference between computed and truth measurement Jacobian (station state):")
diff_H_station = H_station - truth_H_station
diff_H_station_percent = diff_H_station / truth_H_station * 100
print(diff_H_station_percent)
breakpoint()
# Question 4 Testing Code ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Initialize Measurement Manager with ground station parameters
station_1_mgr = MeasurementMgr(station_lat=-35.398333, station_lon=148.981944, initial_earth_spin_angle=np.deg2rad(122))
station_2_mgr = MeasurementMgr(station_lat=40.427222, station_lon=355.749444, initial_earth_spin_angle=np.deg2rad(122))
station_3_mgr = MeasurementMgr(station_lat=35.247163, station_lon=243.205, initial_earth_spin_angle=np.deg2rad(122))

# Simulate measurements from all three stations
station_1_measurements = station_1_mgr.simulate_measurements(reference_state_history, reference_time, 'ECI')
station_2_measurements = station_2_mgr.simulate_measurements(reference_state_history, reference_time, 'ECI')
station_3_measurements = station_3_mgr.simulate_measurements(reference_state_history, reference_time, 'ECI')

# Find first and last values that are not NaN for each station
station_1_not_nan = ~np.isnan(station_1_measurements[0, :])
station_2_not_nan = ~np.isnan(station_2_measurements[0, :])
station_3_not_nan = ~np.isnan(station_3_measurements[0, :])
station_1_first_index = np.where(station_1_not_nan)[0][0]
station_1_last_index = np.where(station_1_not_nan)[0][-1]
station_2_first_index = np.where(station_2_not_nan)[0][0]
station_2_last_index = np.where(station_2_not_nan)[0][-1]
station_3_first_index = np.where(station_3_not_nan)[0][0]
station_3_last_index = np.where(station_3_not_nan)[0][-1]

earliest_measurement = min(station_1_first_index, station_2_first_index, station_3_first_index)
latest_measurement = max(station_1_last_index, station_2_last_index, station_3_last_index)

print(f"Earliest measurement time: {reference_time[earliest_measurement]} sec")
print(f"Latest measurement time: {reference_time[latest_measurement]} sec")

# Find history of spacecraft elevation angles from each station
station_elevation_angles = np.zeros((3, reference_state_history.shape[1]))
for i, time in enumerate(reference_time):
    eci_to_ecef = station_1_mgr.coordinate_mgr.compute_DCM('ECI', 'ECEF', time=time)
    ecef_pos = eci_to_ecef @ reference_state_history[0:3,i]
    elevation_angle_1 = station_1_mgr.get_elevation_angle(ecef_pos)
    elevation_angle_2 = station_2_mgr.get_elevation_angle(ecef_pos)
    elevation_angle_3 = station_3_mgr.get_elevation_angle(ecef_pos)
    elevation_angle_vector = np.array([elevation_angle_1, elevation_angle_2, elevation_angle_3])
    station_elevation_angles[:, i] = elevation_angle_vector

# Convert measurements to DSN units
station_1_measurements_dsn = station_1_mgr.convert_to_DSN_units(station_1_measurements)
station_2_measurements_dsn = station_2_mgr.convert_to_DSN_units(station_2_measurements)
station_3_measurements_dsn = station_3_mgr.convert_to_DSN_units(station_3_measurements)

# Simulate noisy measurements
range_noise = 0.0
range_rate_noise = 0.5E-6
station_1_measurements_noisy = station_1_mgr.simulate_measurements(reference_state_history, reference_time, 'ECI', noise=True, noise_sigma=np.array([range_noise, range_rate_noise]))
station_2_measurements_noisy = station_2_mgr.simulate_measurements(reference_state_history, reference_time, 'ECI', noise=True, noise_sigma=np.array([range_noise, range_rate_noise]))
station_3_measurements_noisy = station_3_mgr.simulate_measurements(reference_state_history, reference_time, 'ECI', noise=True, noise_sigma=np.array([range_noise, range_rate_noise]))

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
    ),
    legend=dict(font=dict(size=18)),
    title_font=dict(size=28)
)
fig.write_html('figures\orbit_trajectories.html')

# Plot the deviation in position coordinates over time in subplots
fig = make_subplots(rows=3, cols=1, vertical_spacing=0.1, shared_xaxes=True, subplot_titles=('X Deviation', 'Y Deviation', 'Z Deviation'))
fig.add_trace(go.Scatter(x=reference_time / period, y=deviation_state[0, :], mode='lines', name='Propagated Deviation', line=dict(color='blue')), row=1, col=1)
fig.add_trace(go.Scatter(x=stm_time / period, y=estimated_deviation[0, :], mode='lines', name='STM Estimated Deviation', line=dict(color='red')), row=1, col=1)
fig.add_trace(go.Scatter(x=reference_time / period, y=deviation_state[1, :], mode='lines', name='Propagated Deviation', showlegend=False, line=dict(color='blue')), row=2, col=1)
fig.add_trace(go.Scatter(x=stm_time / period, y=estimated_deviation[1, :], mode='lines', name='STM Estimated Deviation', showlegend=False, line=dict(color='red')), row=2, col=1)
fig.add_trace(go.Scatter(x=reference_time / period, y=deviation_state[2, :], mode='lines', name='Propagated Deviation', showlegend=False, line=dict(color='blue')), row=3, col=1)
fig.add_trace(go.Scatter(x=stm_time / period, y=estimated_deviation[2, :], mode='lines', name='STM Estimated Deviation', showlegend=False, line=dict(color='red')), row=3, col=1)
fig.update_xaxes(title_text='Time (Orbital Periods)', row=3, col=1)
fig.update_yaxes(title_text='Deviation (km)', row=1, col=1)
fig.update_yaxes(title_text='Deviation (km)', row=2, col=1)
fig.update_yaxes(title_text='Deviation (km)', row=3, col=1)
fig.update_layout(title='Position Deviations Over Time',
                  title_font=dict(size=28),
                  width=1200,
                  height=800,
                  legend=dict(font=dict(size=18),
                              yanchor="top",
                              y=1.12,
                              xanchor="left",
                              x=0.72))
fig.update_annotations(font=dict(size=20))
fig.write_html("figures/position_deviations.html")
fig.write_image("figures/pngs/position_deviations.png")

# Plot the deviation in velocity coordinates over time in subplots
fig = make_subplots(rows=3, cols=1, vertical_spacing=0.1, shared_xaxes=True, subplot_titles=('U Deviation', 'V Deviation', 'W Deviation'))
fig.add_trace(go.Scatter(x=reference_time / period, y=deviation_state[3, :], mode='lines', name='Propagated Deviation', line=dict(color='blue')), row=1, col=1)
fig.add_trace(go.Scatter(x=stm_time / period, y=estimated_deviation[3, :], mode='lines', name='STM Estimated Deviation', line=dict(color='red')), row=1, col=1)
fig.add_trace(go.Scatter(x=reference_time / period, y=deviation_state[4, :], mode='lines', name='Propagated Deviation', showlegend=False, line=dict(color='blue')), row=2, col=1)
fig.add_trace(go.Scatter(x=stm_time / period, y=estimated_deviation[4, :], mode='lines', name='STM Estimated Deviation', showlegend=False, line=dict(color='red')), row=2, col=1)
fig.add_trace(go.Scatter(x=reference_time / period, y=deviation_state[5, :], mode='lines', name='Propagated Deviation', showlegend=False, line=dict(color='blue')), row=3, col=1)
fig.add_trace(go.Scatter(x=stm_time / period, y=estimated_deviation[5, :], mode='lines', name='STM Estimated Deviation', showlegend=False, line=dict(color='red')), row=3, col=1)
fig.update_xaxes(title_text='Time (Orbital Periods)', row=3, col=1)
fig.update_yaxes(title_text='Deviation (km/s)', row=1, col=1)
fig.update_yaxes(title_text='Deviation (km/s)', row=2, col=1)
fig.update_yaxes(title_text='Deviation (km/s)', row=3, col=1)
# Put legend to the right of the title
fig.update_layout(title='Velocity Deviations Over Time',
                  title_font=dict(size=28),
                  width=1200,
                  height=800,
                  legend=dict(font=dict(size=18),
                              yanchor="top",
                              y=1.12,
                              xanchor="left",
                              x=0.72))
fig.update_annotations(font=dict(size=20))
fig.write_html("figures/velocity_deviations.html")
fig.write_image("figures/pngs/velocity_deviations.png")

# Subplots with difference in deviations ------------------------------------------------------------------------------------------------------------------------------------------------

fig = make_subplots(rows=3, cols=1, vertical_spacing=0.1, shared_xaxes=True, subplot_titles=('X Deviation Difference', 'Y Deviation Difference', 'Z Deviation Difference'))
fig.add_trace(go.Scatter(x=reference_time / period, y=deviation_state[0, :] - estimated_deviation[0, :], mode='lines', name='Deviation Difference', line=dict(color='blue')), row=1, col=1)
fig.add_trace(go.Scatter(x=reference_time / period, y=deviation_state[1, :] - estimated_deviation[1, :], mode='lines', name='Deviation Difference', line=dict(color='blue')), row=2, col=1)
fig.add_trace(go.Scatter(x=reference_time / period, y=deviation_state[2, :] - estimated_deviation[2, :], mode='lines', name='Deviation Difference', line=dict(color='blue')), row=3, col=1)
fig.update_xaxes(title_text='Time (Orbital Periods)', row=3, col=1)
fig.update_yaxes(title_text='Deviation Difference (km)', row=1, col=1)
fig.update_yaxes(title_text='Deviation Difference (km)', row=2, col=1)
fig.update_yaxes(title_text='Deviation Difference (km)', row=3, col=1)
fig.update_layout(title='Position Deviation Differences Over Time',
                  title_font=dict(size=28),
                  width=1200,
                  height=800,
                  legend=dict(font=dict(size=18)),
                  showlegend=False)
fig.update_annotations(font=dict(size=20))
fig.write_html("figures/position_deviation_differences.html")
fig.write_image("figures/pngs/position_deviation_differences.png")

# Velocity deviation differences
fig = make_subplots(rows=3, cols=1, vertical_spacing=0.1, shared_xaxes=True, subplot_titles=('U Deviation Difference', 'V Deviation Difference', 'W Deviation Difference'))
fig.add_trace(go.Scatter(x=reference_time / period, y=deviation_state[3, :] - estimated_deviation[3, :], mode='lines', name='Deviation Difference', line=dict(color='blue')), row=1, col=1)
fig.add_trace(go.Scatter(x=reference_time / period, y=deviation_state[4, :] - estimated_deviation[4, :], mode='lines', name='Deviation Difference', line=dict(color='blue')), row=2, col=1)
fig.add_trace(go.Scatter(x=reference_time / period, y=deviation_state[5, :] - estimated_deviation[5, :], mode='lines', name='Deviation Difference', line=dict(color='blue')), row=3, col=1)
fig.update_xaxes(title_text='Time (Orbital Periods)', row=3, col=1)
fig.update_yaxes(title_text='Deviation Difference (km/s)', row=1, col=1)
fig.update_yaxes(title_text='Deviation Difference (km/s)', row=2, col=1)
fig.update_yaxes(title_text='Deviation Difference (km/s)', row=3, col=1)
fig.update_layout(title='Velocity Deviation Differences Over Time',
                  title_font=dict(size=28),
                  width=1200,
                  height=800,
                  legend=dict(font=dict(size=18)),
                  showlegend=False)
fig.update_annotations(font=dict(size=20))
#fig.write_html("figures/velocity_deviation_differences.html")
fig.write_image("figures/pngs/velocity_deviation_differences.png")

# Measurement Plots --------------------------------------------------------------------------------------------------------------------------------------------------------------------------

fig = make_subplots(rows=2, cols=1, vertical_spacing=0.1, shared_xaxes=True, subplot_titles=('Range Measurements', 'Range Rate Measurements'))
fig.add_trace(go.Scatter(x=reference_time, y=station_1_measurements[0, :], mode='lines', name='Station 1', line=dict(color='blue')), row=1, col=1)
fig.add_trace(go.Scatter(x=reference_time, y=station_2_measurements[0, :], mode='lines', name='Station 2', line=dict(color='red')), row=1, col=1)
fig.add_trace(go.Scatter(x=reference_time, y=station_3_measurements[0, :], mode='lines', name='Station 3', line=dict(color='green')), row=1, col=1)
fig.add_trace(go.Scatter(x=reference_time, y=station_1_measurements[1, :], mode='lines', name='Station 1', line=dict(color='blue'), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=reference_time, y=station_2_measurements[1, :], mode='lines', name='Station 2', line=dict(color='red'), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=reference_time, y=station_3_measurements[1, :], mode='lines', name='Station 3', line=dict(color='green'), showlegend=False), row=2, col=1)
fig.update_xaxes(title_text='Time (Seconds)', row=2, col=1)
fig.update_yaxes(title_text='Range (km)', row=1, col=1)
fig.update_yaxes(title_text='Range Rate (km/s)', row=2, col=1)
# Increase subplot title font size and overall figure title font size
fig.update_layout(title='Simulated Range and Range Rate Measurements from Ground Stations',
                  title_font=dict(size=28),
                  width=1200,
                  height=800,
                  legend=dict(font=dict(size=18),
                              yanchor="top",
                              y=1.12,
                              xanchor="left",
                              x=0.72))
fig.update_annotations(font=dict(size=20))
fig.write_html("figures/simulated_measurements.html")
fig.write_image("figures/pngs/simulated_measurements.png")

fig = make_subplots(rows=2, cols=1, vertical_spacing=0.1, shared_xaxes=True, subplot_titles=('Range Measurements', 'Doppler Measurements (DSN Units)'))
fig.add_trace(go.Scatter(x=reference_time, y=station_1_measurements_dsn[0, :], mode='lines', name='Station 1', line=dict(color='blue')), row=1, col=1)
fig.add_trace(go.Scatter(x=reference_time, y=station_2_measurements_dsn[0, :], mode='lines', name='Station 2', line=dict(color='red')), row=1, col=1)
fig.add_trace(go.Scatter(x=reference_time, y=station_3_measurements_dsn[0, :], mode='lines', name='Station 3', line=dict(color='green')), row=1, col=1)
fig.add_trace(go.Scatter(x=reference_time, y=station_1_measurements_dsn[1, :], mode='lines', name='Station 1', line=dict(color='blue'), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=reference_time, y=station_2_measurements_dsn[1, :], mode='lines', name='Station 2', line=dict(color='red'), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=reference_time, y=station_3_measurements_dsn[1, :], mode='lines', name='Station 3', line=dict(color='green'), showlegend=False), row=2, col=1)
fig.update_xaxes(title_text='Time (Seconds)', row=2, col=1)
fig.update_yaxes(title_text='Range (Range Units)', row=1, col=1)
fig.update_yaxes(title_text='Doppler Shift (Hz)', row=2, col=1)
fig.update_layout(title='Simulated Range and Doppler Measurements from Ground Stations (DSN Units)',
                  title_font=dict(size=28),
                  width=1200,
                  height=800,
                  legend=dict(font=dict(size=18),
                              yanchor="top",
                              y=1.12,
                              xanchor="left",
                              x=0.72))
fig.update_annotations(font=dict(size=20))
fig.write_html("figures/simulated_measurements_dsn.html")
fig.write_image("figures/pngs/simulated_measurements_dsn.png")

# Elevation Angle Plot --------------------------------------------------------------------------------------------------------------------------------------------------------------------------

fig = go.Figure()
fig.add_trace(go.Scatter(x=reference_time, y=station_elevation_angles[0, :], mode='lines', name='Station 1 Elevation Angle', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=reference_time, y=station_elevation_angles[1, :], mode='lines', name='Station 2 Elevation Angle', line=dict(color='red')))
fig.add_trace(go.Scatter(x=reference_time, y=station_elevation_angles[2, :], mode='lines', name='Station 3 Elevation Angle', line=dict(color='green')))
fig.add_hline(y=10.0, line_dash="dash", line_color="black", annotation_text="10° Elevation Mask", annotation_position="top left")
fig.update_xaxes(title_text='Time (Seconds)')
fig.update_yaxes(title_text='Elevation Angle (degrees)')
fig.update_layout(title='Spacecraft Elevation Angles from Ground Stations',
                  title_font=dict(size=28),
                  height=600,
                  legend=dict(font=dict(size=18)))
fig.write_html("figures/elevation_angles.html")

# Noisy Measurement Plots --------------------------------------------------------------------------------------------------------------------------------------------------------------------------

fig = go.Figure()
fig.add_trace(go.Scatter(x=reference_time, y=station_1_measurements_noisy[1, :], mode='lines', name='Station 1', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=reference_time, y=station_2_measurements_noisy[1, :], mode='lines', name='Station 2', line=dict(color='red')))
fig.add_trace(go.Scatter(x=reference_time, y=station_3_measurements_noisy[1, :], mode='lines', name='Station 3', line=dict(color='green')))
fig.update_xaxes(title_text='Time (Seconds)')
fig.update_yaxes(title_text='Range Rate (km/s)')
fig.update_layout(title='Simulated Noisy Range Rate Measurements from Ground Stations',
                  title_font=dict(size=28),
                  height=900,
                  legend=dict(font=dict(size=18)))
fig.write_html("figures/simulated_noisy_measurements.html")

fig = go.Figure()
fig.add_trace(go.Scatter(x=reference_time, y=(station_1_measurements_noisy[1, :] - station_1_measurements[1, :])*1E6, mode='markers', name='Station 1', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=reference_time, y=(station_2_measurements_noisy[1, :] - station_2_measurements[1, :])*1E6, mode='markers', name='Station 2', line=dict(color='red')))
fig.add_trace(go.Scatter(x=reference_time, y=(station_3_measurements_noisy[1, :] - station_3_measurements[1, :])*1E6, mode='markers', name='Station 3', line=dict(color='green')))
fig.update_xaxes(title_text='Time (Seconds)', range=[0, reference_time[-1]])
fig.update_yaxes(title_text='Range Rate Measurement Noise (mm/s)')
fig.update_layout(title='Range Rate Measurement Noise from Ground Stations',
                  title_font=dict(size=28),
                  height=900,
                  legend=dict(font=dict(size=18)))
fig.write_html("figures/range_rate_measurement_noise.html")

# Subplots with difference in my trajectory and truth ------------------------------------------------------------------------------------------------------------------------------------------------

fig = make_subplots(rows=3, cols=1, vertical_spacing=0.1, shared_xaxes=True, subplot_titles=('X Difference', 'Y Difference', 'Z Difference'))
fig.add_trace(go.Scatter(x=reference_time, y=reference_state_history[0, :] - truth_state_history[0, :], mode='lines', name='Deviation Difference', line=dict(color='blue')), row=1, col=1)
fig.add_trace(go.Scatter(x=reference_time, y=reference_state_history[1, :] - truth_state_history[1, :], mode='lines', name='Deviation Difference', line=dict(color='blue')), row=2, col=1)
fig.add_trace(go.Scatter(x=reference_time, y=reference_state_history[2, :] - truth_state_history[2, :], mode='lines', name='Deviation Difference', line=dict(color='blue')), row=3, col=1)
fig.update_xaxes(title_text='Time (Seconds)', row=3, col=1)
fig.update_yaxes(title_text='Difference (km)', row=1, col=1)
fig.update_yaxes(title_text='Difference (km)', row=2, col=1)
fig.update_yaxes(title_text='Difference (km)', row=3, col=1)
fig.update_layout(title='Position Differences Between My Trajectory and Truth Over Time',
                  title_font=dict(size=28),
                  width=1200,
                  height=800,
                  legend=dict(font=dict(size=18)),
                  showlegend=False)
fig.update_annotations(font=dict(size=20))
fig.write_html("figures/position_differences_truth.html")
fig.write_image("figures/pngs/position_differences_truth.png")

fig = make_subplots(rows=3, cols=1, vertical_spacing=0.1, shared_xaxes=True, subplot_titles=('U Difference', 'V Difference', 'W Difference'))
fig.add_trace(go.Scatter(x=reference_time, y=reference_state_history[3, :] - truth_state_history[3, :], mode='lines', name='Deviation Difference', line=dict(color='blue')), row=1, col=1)
fig.add_trace(go.Scatter(x=reference_time, y=reference_state_history[4, :] - truth_state_history[4, :], mode='lines', name='Deviation Difference', line=dict(color='blue')), row=2, col=1)
fig.add_trace(go.Scatter(x=reference_time, y=reference_state_history[5, :] - truth_state_history[5, :], mode='lines', name='Deviation Difference', line=dict(color='blue')), row=3, col=1)
fig.update_xaxes(title_text='Time (Seconds)', row=3, col=1)
fig.update_yaxes(title_text='Difference (km/s)', row=1, col=1)
fig.update_yaxes(title_text='Difference (km/s)', row=2, col=1)
fig.update_yaxes(title_text='Difference (km/s)', row=3, col=1)
fig.update_layout(title='Velocity Differences Between My Trajectory and Truth Over Time',
                  title_font=dict(size=28),
                  width=1200,
                  height=800,
                  legend=dict(font=dict(size=18)),
                  showlegend=False)
fig.update_annotations(font=dict(size=20))
fig.write_html("figures/velocity_differences_truth.html")
fig.write_image("figures/pngs/velocity_differences_truth.png")
