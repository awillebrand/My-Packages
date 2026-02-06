import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from ASEN_6080.Tools import Integrator, MeasurementMgr, CoordinateMgr, LKF
from plotly.subplots import make_subplots

measurements = pd.read_pickle(".\ASEN_6080\Project1\data\conditioned_measurements.pkl")

R = np.diag([1E-5**2, 1E-6**2])  # Noise covariance matrix for range and range rate. Corresponds to 1 cm range noise and 1 mm/s range rate noise.

sat_state = np.array([757700.0E-3, 5222607.0E-3, 4851500.0E-3, 2213.21E-3, 4678.34E-3, -5371.30E-3])  # Example satellite state in km and km/s
mu = 3.986004415E5  # Earth's gravitational parameter in km^3/s^2
J2 = 1.082626925638815E-3 # Earth's J2 coefficient
J3 = 0.0 # Earth's J3 coefficient
R_e = 6378.1363  # Earth's radius in km
C_d = 2.0 # Drag coefficient
spacecraft_mass = 970.0  # Spacecraft mass in kg
spacecraft_area = 3.0  # Spacecraft cross-sectional area in m^2Q
earth_spin_rate = 7.2921158553E-5  # Earth's rotation rate in rad/s

station_1_state = np.array([-5127510.0E-3, -3794160.0E-3,  0.0, 0.0, 0.0, 0.0])
station_2_state = np.array([3860910.0E-3, 3238490.0E-3,  3898094.0E-3, 0.0, 0.0, 0.0])
station_3_state = np.array([549505.0E-3, -1380872.0E-3,  6182197.0E-3, 0.0, 0.0, 0.0])

station_positions_ecef = np.array([station_1_state[0:3], station_2_state[0:3], station_3_state[0:3]])

initial_state_estimate = np.concatenate([sat_state[0:6], [mu, J2, C_d], station_1_state[0:3], station_2_state[0:3], station_3_state[0:3]]).flatten()

station_1_mgr = MeasurementMgr("station_101", station_state_ecef=station_1_state, initial_earth_spin_angle=0.0, earth_spin_rate=earth_spin_rate, R_e=R_e)
station_2_mgr = MeasurementMgr("station_337", station_state_ecef=station_2_state, initial_earth_spin_angle=0.0, earth_spin_rate=earth_spin_rate, R_e=R_e)
station_3_mgr = MeasurementMgr("station_394", station_state_ecef=station_3_state, initial_earth_spin_angle=0.0, earth_spin_rate=earth_spin_rate, R_e=R_e)

station_mgr_list = [station_1_mgr, station_2_mgr, station_3_mgr]

integrator = Integrator(mu, R_e, mode=['mu','J2','Drag','Stations'], parameter_indices=[6,7,8,9], spacecraft_area=spacecraft_area, spacecraft_mass=spacecraft_mass, number_of_stations=3)

a_priori_covariance = np.diag([1, 1, 1, 1, 1, 1, 1E2, 1E6, 1E6, 1E-16, 1E-16, 1E-16, 1, 1, 1, 1, 1, 1])  # Given
# a_priori_covariance = np.diag([1, 1, 1, 1, 1, 1, 1E2, 1E-8, 4.0, 1E-16, 1E-16, 1E-16, 1, 1, 1, 1, 1, 1])
lkf = LKF(integrator, station_mgr_list, initial_earth_spin_angle=0.0, earth_rotation_rate=earth_spin_rate)
state_history, covariance_history, post_fit_residuals = lkf.run(initial_state_estimate, np.zeros_like(initial_state_estimate), a_priori_covariance, measurements, R=R, max_iterations=5, convergence_threshold=1e-9)

# Simulate measurements and generate residuals for plotting
# Simulate measurements from estimated state for residuals
station_1_measurements = station_1_mgr.simulate_measurements(state_history, measurements['time'].values, 'ECI', ignore_visibility=True)
station_2_measurements = station_2_mgr.simulate_measurements(state_history, measurements['time'].values, 'ECI', ignore_visibility=True)
station_3_measurements = station_3_mgr.simulate_measurements(state_history, measurements['time'].values, 'ECI', ignore_visibility=True)
# Plot cd over time

cd_history = state_history[8, :]
cd_covariance = covariance_history[8,8,:]

fig = go.Figure()
fig.add_trace(go.Scatter(x=measurements['time'], y=cd_history, mode='lines+markers', name='Cd Estimate'))
fig.add_trace(go.Scatter(x=measurements['time'], y=cd_history+3*np.sqrt(abs(cd_covariance)), mode='lines', name='Cd + 2σ', line=dict(dash='dash')))
fig.add_trace(go.Scatter(x=measurements['time'], y=cd_history-3*np.sqrt(abs(cd_covariance)), mode='lines', name='Cd - 2σ', line=dict(dash='dash')))
fig.update_layout(title='Cd Estimate Over Time', xaxis_title='Time (s)', yaxis_title='Cd Estimate')
fig.show()
station_1_truth =np.vstack( measurements['station_101_measurements'].values).T
station_2_truth = np.vstack(measurements['station_337_measurements'].values).T
station_3_truth = np.vstack(measurements['station_394_measurements'].values).T

station_1_residuals = station_1_truth - station_1_measurements
station_2_residuals = station_2_truth - station_2_measurements
station_3_residuals = station_3_truth - station_3_measurements
residuals_matrix = np.empty((len(station_mgr_list), len(measurements['time'].values), 2))  # 2 for range and range rate
residuals_matrix[0,:,:] = station_1_residuals.T
residuals_matrix[1,:,:] = station_2_residuals.T
residuals_matrix[2,:,:] = station_3_residuals.T

station_residuals_list = [station_1_residuals, station_2_residuals, station_3_residuals]

# Compute standard deviation and mean of residuals
for i in range(3):
    residuals = station_residuals_list[i]
    non_nan_residuals = residuals[:, ~np.isnan(residuals).any(axis=0)]
    range_std = np.std(non_nan_residuals[0,:])
    range_mean = np.mean(non_nan_residuals[0,:])
    range_rate_std = np.std(non_nan_residuals[1,:])
    range_rate_mean = np.mean(non_nan_residuals[1,:])
    print(f"Station {i+1} Range Residuals Std Dev: {range_std*100000:.6f} cm, Mean: {range_mean*100000:.6f} cm")
    print(f"Station {i+1} Range Rate Residuals Std Dev: {range_rate_std*1e6:.6f} mm/s, Mean: {range_rate_mean*1e6:.6f} mm/s")

# Plot residuals over time for each station
time_vector = measurements['time'].values

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Range Residuals", "Range Rate Residuals"))
color_list = ['red', 'green', 'blue', 'red', 'green', 'blue']
for i, mgr in enumerate(station_mgr_list):
    station_name = mgr.station_name
    fig.add_trace(go.Scatter(x=time_vector, y=residuals_matrix[i,:,0]*1E5, mode='markers', name=f"{station_name} Residuals", marker=dict(color=color_list[i]), showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=time_vector, y=residuals_matrix[i,:,1]*1E6, mode='markers', name=f"{station_name} Residuals", marker=dict(color=color_list[i]), showlegend=False), row=2, col=1)
fig.update_xaxes(title_text="Time (s)", row=2, col=1)
fig.update_yaxes(title_text="Range Residuals (cm)", row=1, col=1)
fig.update_yaxes(title_text="Range Rate Residuals (mm/s)", row=2, col=1)
fig.update_layout(height=800, width=1200, title_text="Measurement Residuals After Batch LLS Estimation")
fig.write_html("ASEN_6080/Project1/figures/lkf_measurement_residuals.html")
fig.show()

# Plot post fit residuals over time for each station
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Post-Fit Range Residuals", "Post-Fit Range Rate Residuals"))
color_list = ['red', 'green', 'blue', 'red', 'green', 'blue']
for i, mgr in enumerate(station_mgr_list):
    station_name = mgr.station_name
    fig.add_trace(go.Scatter(x=time_vector, y=post_fit_residuals[0,:,i].squeeze()*1E5, mode='markers', name=f"{station_name} Post-Fit Residuals", marker=dict(color=color_list[i]), showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=time_vector, y=post_fit_residuals[1,:,i].squeeze()*1E6, mode='markers', name=f"{station_name} Post-Fit Residuals", marker=dict(color=color_list[i]), showlegend=False), row=2, col=1)
fig.update_xaxes(title_text="Time (s)", row=2, col=1)
fig.update_yaxes(title_text="Post-Fit Range Residuals (cm)", row=1, col=1)
fig.update_yaxes(title_text="Post-Fit Range Rate Residuals (mm/s)", row=2, col=1)
fig.update_layout(height=800, width=1200, title_text="Post-Fit Measurement Residuals After Batch LLS Estimation")
fig.write_html("ASEN_6080/Project1/figures/lkf_post_fit_residuals.html")
fig.show()