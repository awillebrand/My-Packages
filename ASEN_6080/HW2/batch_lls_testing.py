import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from ASEN_6080.Tools import Integrator, MeasurementMgr, CoordinateMgr, BatchLLSEstimator
from plotly.subplots import make_subplots

measurement_data = pd.read_pickle("ASEN_6080/HW2/measurement_data/simulated_measurements.pkl")
truth_data = pd.read_pickle("ASEN_6080/HW2/measurement_data/truth_data.pkl")

mu = 3.986004415E5
R_e = 6378
J2 = 0.0010826269
noise_var = np.array([1, 1e-6])**2 # [range noise = 1 km, range rate noise = 1 mm/s]

integrator = Integrator(mu, R_e, mode='J2')
station_1_mgr = MeasurementMgr("station_1", station_lat=-35.398333, station_lon=148.981944, initial_earth_spin_angle=np.deg2rad(122))
station_2_mgr = MeasurementMgr("station_2", station_lat=40.427222, station_lon=355.749444, initial_earth_spin_angle=np.deg2rad(122))
station_3_mgr = MeasurementMgr("station_3", station_lat=35.247163, station_lon=243.205, initial_earth_spin_angle=np.deg2rad(122))
station_mgr_list = [station_1_mgr, station_2_mgr, station_3_mgr]

initial_state_deviation = np.array([1.010e-02, -1.218e-01, -1.484e-01,  3.204e-05, -8.320e-05, 1.740e-04,  0.000e+00])
initial_state_guess = truth_data['initial_state'].values[0] + initial_state_deviation
P_0 = np.diag([1, 1, 1, 1e-3, 1e-3, 1e-3])**2
large_P_0 = np.diag([1000, 1000, 1000, 1, 1, 1])**2

batch_estimator = BatchLLSEstimator(integrator, station_mgr_list, np.deg2rad(122.0))

estimated_state, estimated_covariance = batch_estimator.estimate_initial_state(initial_state_guess, measurement_data, np.diag(noise_var), tol=1e-9, a_priori_covariance=large_P_0)

# Verify against truth data
augmented_truth_state = truth_data['augmented_state_history'].values

truth_state_history = np.zeros((7, augmented_truth_state.shape[0]))
covariance_history = np.zeros((6, 6, augmented_truth_state.shape[0]))

for i, state in enumerate(augmented_truth_state):
    truth_state = state[0:7]
    raw_stm = state[7:].reshape((7,7))
    stm = raw_stm[0:6,0:6]
    P = stm @ estimated_covariance @ stm.T
    truth_state_history[:, i] = truth_state
    covariance_history[:,:,i] = P

# Integrate estimated state to compare against truth
[_, estimated_state_history] = integrator.integrate_eom(measurement_data['time'].values[-1], estimated_state, teval=measurement_data['time'].values)

state_errors = estimated_state_history - truth_state_history

# Simulate measurements from estimated state for residuals
station_1_measurements = station_1_mgr.simulate_measurements(estimated_state_history, measurement_data['time'].values, 'ECI')
station_2_measurements = station_2_mgr.simulate_measurements(estimated_state_history, measurement_data['time'].values, 'ECI')
station_3_measurements = station_3_mgr.simulate_measurements(estimated_state_history, measurement_data['time'].values, 'ECI')

# Compute measurement residuals
station_1_truth = np.vstack(measurement_data["station_1_measurements"].values).T
station_2_truth = np.vstack(measurement_data["station_2_measurements"].values).T
station_3_truth = np.vstack(measurement_data["station_3_measurements"].values).T

station_1_residuals = station_1_truth - station_1_measurements
station_2_residuals = station_2_truth - station_2_measurements
station_3_residuals = station_3_truth - station_3_measurements

station_residuals_list = [station_1_residuals, station_2_residuals, station_3_residuals]
# Compute standard deviation and mean of residuals
for i in range(3):
    residuals = station_residuals_list[i]
    non_nan_residuals = residuals[:, ~np.isnan(residuals).any(axis=0)]
    range_std = np.std(non_nan_residuals[0,:])
    range_mean = np.mean(non_nan_residuals[0,:])
    range_rate_std = np.std(non_nan_residuals[1,:])
    range_rate_mean = np.mean(non_nan_residuals[1,:])
    print(f"Station {i+1} Range Residuals Std Dev: {range_std:.6f} km, Mean: {range_mean:.6f} km")
    print(f"Station {i+1} Range Rate Residuals Std Dev: {range_rate_std*1e6:.6f} mm/s, Mean: {range_rate_mean*1e6:.6f} mm/s")

# Compute RMS errors
rms_position_error = np.sqrt(np.mean(state_errors[0:3,:]**2, axis=1))
rms_position_error_3D = np.sqrt(np.mean(np.sum(state_errors[0:3,:]**2, axis=0)))
rms_velocity_error = np.sqrt(np.mean(state_errors[3:6,:]**2, axis=1))
rms_velocity_error_3D = np.sqrt(np.mean(np.sum(state_errors[3:6,:]**2, axis=0)))

print(f"RMS Position Errors (km): X: {rms_position_error[0]}, Y: {rms_position_error[1]}, Z: {rms_position_error[2]}")
print(f"RMS Velocity Errors (km/s): X: {rms_velocity_error[0]}, Y: {rms_velocity_error[1]}, Z: {rms_velocity_error[2]}")
print(f"3D RMS Position Error (km): {rms_position_error_3D}")
print(f"3D RMS Velocity Error (km/s): {rms_velocity_error_3D}")

# Plot results
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=("X Position Error", "Y Position Error", "Z Position Error"))
for i in range(3):
    fig.add_trace(go.Scatter(x=measurement_data['time'].values, y=state_errors[i,:], mode='lines', name='State Error', line=dict(color='blue'), showlegend=False if i>0 else True), row=i+1, col=1)
    fig.add_trace(go.Scatter(x=measurement_data['time'].values, y=3*np.sqrt(covariance_history[i,i,:]), mode='lines', name="3\u03C3 Bounds", line=dict(color='red', dash='dash'), showlegend=False if i>0 else True), row=i+1, col=1)
    fig.add_trace(go.Scatter(x=measurement_data['time'].values, y=-3*np.sqrt(covariance_history[i,i,:]), mode='lines', name="3\u03C3 Bounds", line=dict(color='red', dash='dash'), showlegend=False), row=i+1, col=1)
    fig.update_yaxes(title_text="Position Error (km)", showexponent="all", exponentformat="e", range=[-5e-4, 5e-4], row=i+1, col=1)
fig.update_xaxes(title_text="Time (s)", row=3, col=1)
fig.update_layout(title_text="Estimated State Position Errors Over Time",
                  title_font=dict(size=28),
                  width=1200,
                  height=800,
                  legend=dict(font=dict(size=18),
                              yanchor="top",
                              y=1.17,
                              xanchor="left",
                              x=0.87))
fig.update_annotations(font=dict(size=20))
fig.write_html('ASEN_6080/HW2/figures/batch_lls_results/estimated_state_position_errors.html')

fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=("X Velocity Error", "Y Velocity Error", "Z Velocity Error"))
for i in range(3):
    fig.add_trace(go.Scatter(x=measurement_data['time'].values, y=state_errors[i+3,:], mode='lines', name='State Error', line=dict(color='blue'), showlegend=False if i>0 else True), row=i+1, col=1)
    fig.add_trace(go.Scatter(x=measurement_data['time'].values, y=3*np.sqrt(covariance_history[i+3,i+3,:]), mode='lines', name="3\u03C3 Bounds", line=dict(color='red', dash='dash'), showlegend=False if i>0 else True), row=i+1, col=1)
    fig.add_trace(go.Scatter(x=measurement_data['time'].values, y=-3*np.sqrt(covariance_history[i+3,i+3,:]), mode='lines', name="3\u03C3 Bounds", line=dict(color='red', dash='dash'), showlegend=False), row=i+1, col=1)
    fig.update_yaxes(title_text="Velocity Error (km)", showexponent="all", exponentformat="e", range=[-4e-7, 4e-7], row=i+1, col=1)
fig.update_annotations(font=dict(size=20))
fig.update_xaxes(title_text="Time (s)", row=3, col=1)
fig.update_layout(title_text="Estimated State Velocity Errors Over Time",
                  title_font=dict(size=28),
                  width=1200,
                  height=800,
                  legend=dict(font=dict(size=18),
                              yanchor="top",
                              y=1.17,
                              xanchor="left",
                              x=0.87))
fig.write_html('ASEN_6080/HW2/figures/batch_lls_results/estimated_state_velocity_errors.html')

# Plot measurement residuals

fig = go.Figure()
fig.add_trace(go.Scatter(x=measurement_data['time'].values, y=station_1_residuals[0,1:], mode='markers', name='Station 1', line=dict(color='red')))
fig.add_trace(go.Scatter(x=measurement_data['time'].values, y=station_2_residuals[0,1:], mode='markers', name='Station 2', line=dict(color='green')))
fig.add_trace(go.Scatter(x=measurement_data['time'].values, y=station_3_residuals[0,1:], mode='markers', name='Station 3', line=dict(color='blue')))
fig.update_traces(marker=dict(size=4))
fig.update_xaxes(title_text="Time (s)")
fig.update_yaxes(title_text="Range Residuals (km)", showexponent="all", exponentformat="e", range=[-4,4])
fig.update_layout(title_text="Range Measurement Residuals Over Time",
                  title_font=dict(size=28),
                  width=1200,
                  height=800,
                  legend=dict(font=dict(size=18),
                              yanchor="top",
                              y=1.2,
                              xanchor="left",
                              x=0.87))
fig.write_html('ASEN_6080/HW2/figures/batch_lls_results/measurement_range_residuals.html')

fig = go.Figure()
fig.add_trace(go.Scatter(x=measurement_data['time'].values, y=station_1_residuals[1,1:]*10**6, mode='markers', name='Station 1', line=dict(color='red')))
fig.add_trace(go.Scatter(x=measurement_data['time'].values, y=station_2_residuals[1,1:]*10**6, mode='markers', name='Station 2', line=dict(color='green')))
fig.add_trace(go.Scatter(x=measurement_data['time'].values, y=station_3_residuals[1,1:]*10**6, mode='markers', name='Station 3', line=dict(color='blue')))
fig.update_traces(marker=dict(size=4))
fig.update_xaxes(title_text="Time (s)")
fig.update_yaxes(title_text="Range Rate Residuals (mm/s)", showexponent="all", exponentformat="e")
fig.update_layout(title_text="Range Rate Measurement Residuals Over Time",
                  title_font=dict(size=28),
                  width=1200,
                  height=800,
                  legend=dict(font=dict(size=18),
                              yanchor="top",
                              y=1.2,
                              xanchor="left",
                              x=0.87))
fig.write_html('ASEN_6080/HW2/figures/batch_lls_results/measurement_range_rate_residuals.html')

# Measurement histograms
fig = make_subplots(rows=2, cols=1, subplot_titles=("Range Residuals Histogram", "Range Rate Residuals Histogram"))
fig.add_trace(go.Histogram(x=station_1_residuals[0,1:], xbins=dict(size=1e-1), name='Station 1', marker_color='red', opacity=0.7), row=1, col=1)
fig.add_trace(go.Histogram(x=station_2_residuals[0,1:], xbins=dict(size=1e-1), name='Station 2', marker_color='green', opacity=0.7), row=1, col=1)
fig.add_trace(go.Histogram(x=station_3_residuals[0,1:], xbins=dict(size=1e-1), name='Station 3', marker_color='blue', opacity=0.7), row=1, col=1)
fig.add_trace(go.Histogram(x=station_1_residuals[1,1:], xbins=dict(size=1e-7), name='Station 1', marker_color='red', opacity=0.7, showlegend=False), row=2, col=1)
fig.add_trace(go.Histogram(x=station_2_residuals[1,1:], xbins=dict(size=1e-7), name='Station 2', marker_color='green', opacity=0.7, showlegend=False), row=2, col=1)
fig.add_trace(go.Histogram(x=station_3_residuals[1,1:], xbins=dict(size=1e-7), name='Station 3', marker_color='blue', opacity=0.7, showlegend=False), row=2, col=1)
fig.update_xaxes(title_text="Residuals", row=1, col=1)
fig.update_xaxes(title_text="Residuals", row=2, col=1)
fig.update_yaxes(title_text="Count", row=1, col=1)
fig.update_yaxes(title_text="Count", row=2, col=1)
fig.update_layout(title_text="Measurement Residuals Histograms",
                  title_font=dict(size=28),
                  width=1200,
                  height=800,
                  legend=dict(font=dict(size=18),
                              yanchor="top",
                              y=1.2,
                              xanchor="left",
                              x=0.87),
                  bargap=0.2)
fig.update_annotations(font=dict(size=20))
fig.write_html('ASEN_6080/HW2/figures/batch_lls_results/measurement_residuals_histograms.html')

# Plot difference between trajectories
[_, perturbed_trajectory] = integrator.integrate_eom(measurement_data['time'].values[-1], initial_state_guess, measurement_data['time'].values)
[_, true_trajectory] = integrator.integrate_eom(measurement_data['time'].values[-1], truth_data['initial_state'].values[0], measurement_data['time'].values)

trajectory_difference = perturbed_trajectory - true_trajectory
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=("X Position Difference", "Y Position Difference", "Z Position Difference"))
for i in range(3):
    fig.add_trace(go.Scatter(x=measurement_data['time'].values, y=trajectory_difference[i,:], mode='lines', name='Trajectory Difference', line=dict(color='blue'), showlegend=False if i>0 else True), row=i+1, col=1)
    fig.update_yaxes(title_text="Position Difference (km)", showexponent="all", exponentformat="e", row=i+1, col=1)
fig.update_xaxes(title_text="Time (s)", row=3, col=1)
fig.update_layout(title_text="Trajectory Position Differences Over Time",
                    title_font=dict(size=28),
                     width=1200,
                     height=800,    
                     legend=dict(font=dict(size=18),
                              yanchor="top",
                              y=1.2,
                              xanchor="left",
                              x=0.87))
fig.update_annotations(font=dict(size=20))
fig.write_html('ASEN_6080/HW2/figures/batch_lls_results/trajectory_position_differences.html')

fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=("X Velocity Difference", "Y Velocity Difference", "Z Velocity Difference"))
for i in range(3):
    fig.add_trace(go.Scatter(x=measurement_data['time'].values, y=trajectory_difference[i+3,:], mode='lines', name='Trajectory Difference', line=dict(color='blue'), showlegend=False if i>0 else True), row=i+1, col=1)
    fig.update_yaxes(title_text="Velocity Difference (km/s)", showexponent="all", exponentformat="e", row=i+1, col=1)
fig.update_xaxes(title_text="Time (s)", row=3, col=1)
fig.update_layout(title_text="Trajectory Velocity Differences Over Time",
                  title_font=dict(size=28),
                  width=1200,
                  height=800,
                  legend=dict(font=dict(size=18),
                              yanchor="top",
                              y=1.2,
                              xanchor="left",
                              x=0.87))
fig.update_annotations(font=dict(size=20))
fig.write_html('ASEN_6080/HW2/figures/batch_lls_results/trajectory_velocity_differences.html')