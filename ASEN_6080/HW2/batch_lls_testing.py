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
noise_std = np.array([1, 1e-6]) # [range noise = 1 km, range rate noise = 1 mm/s]

integrator = Integrator(mu, R_e, mode='J2')
station_1_mgr = MeasurementMgr("station_1", station_lat=-35.398333, station_lon=148.981944, initial_earth_spin_angle=np.deg2rad(122))
station_2_mgr = MeasurementMgr("station_2", station_lat=40.427222, station_lon=355.749444, initial_earth_spin_angle=np.deg2rad(122))
station_3_mgr = MeasurementMgr("station_3", station_lat=35.247163, station_lon=243.205, initial_earth_spin_angle=np.deg2rad(122))
station_mgr_list = [station_1_mgr, station_2_mgr, station_3_mgr]

initial_state_guess = truth_data['initial_state'].values[0]
initial_state_deviation = np.array([0.0505, -0.609, -0.742, 0.0001602, -0.000416, 0.000870])
initial_state_deviation = np.zeros(6)
P_0 = np.diag([1, 1, 1, 1e-3, 1e-3, 1e-3])**2

batch_estimator = BatchLLSEstimator(integrator, station_mgr_list, 122.0)

estimated_state, estimated_covariance = batch_estimator.estimate_initial_state(initial_state_guess, measurement_data, noise_std, tol=1e-6, a_priori_covariance=P_0, a_priori_state_correction=initial_state_deviation)

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

# Plot results
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=("X Position Error", "Y Position Error", "Z Position Error"))
for i in range(3):
    fig.add_trace(go.Scatter(x=measurement_data['time'].values, y=state_errors[i,:], mode='lines', name='State Error', line=dict(color='blue'), showlegend=False if i>0 else True), row=i+1, col=1)
    fig.add_trace(go.Scatter(x=measurement_data['time'].values, y=3*np.sqrt(covariance_history[i,i,:]), mode='lines', name="3\u03C3 Bounds", line=dict(color='red', dash='dash'), showlegend=False if i>0 else True), row=i+1, col=1)
    fig.add_trace(go.Scatter(x=measurement_data['time'].values, y=-3*np.sqrt(covariance_history[i,i,:]), mode='lines', name="3\u03C3 Bounds", line=dict(color='red', dash='dash'), showlegend=False), row=i+1, col=1)
    fig.update_yaxes(title_text="Position Error (km)", showexponent="all", exponentformat="e", row=i+1, col=1)

fig.update_xaxes(title_text="Time (s)", row=3, col=1)
fig.update_layout(title_text="Estimated State Position Errors Over Time",
                  title_font=dict(size=24),
                  width=1200,
                  height=800,
                  legend=dict(font=dict(size=14)))
fig.write_html("ASEN_6080/HW2/figures/estimated_state_position_errors.html")

fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=("X Velocity Error", "Y Velocity Error", "Z Velocity Error"))
for i in range(3):
    fig.add_trace(go.Scatter(x=measurement_data['time'].values, y=state_errors[i+3,:], mode='lines', name='State Error', line=dict(color='blue'), showlegend=False if i>0 else True), row=i+1, col=1)
    fig.add_trace(go.Scatter(x=measurement_data['time'].values, y=3*np.sqrt(covariance_history[i+3,i+3,:]), mode='lines', name="3\u03C3 Bounds", line=dict(color='red', dash='dash'), showlegend=False if i>0 else True), row=i+1, col=1)
    fig.add_trace(go.Scatter(x=measurement_data['time'].values, y=-3*np.sqrt(covariance_history[i+3,i+3,:]), mode='lines', name="3\u03C3 Bounds", line=dict(color='red', dash='dash'), showlegend=False), row=i+1, col=1)
    fig.update_yaxes(title_text="Velocity Error (km)", showexponent="all", exponentformat="e", row=i+1, col=1)

fig.update_xaxes(title_text="Time (s)", row=3, col=1)
fig.update_layout(title_text="Estimated State Velocity Errors Over Time",
                  title_font=dict(size=24),
                  width=1200,
                  height=800,
                  legend=dict(font=dict(size=14)))
fig.write_html("ASEN_6080/HW2/figures/estimated_state_velocity_errors.html")

# Plot measurement residuals

fig = go.Figure()
fig.add_trace(go.Scatter(x=measurement_data['time'].values, y=station_1_residuals[0,:], mode='markers', name='Station 1 Range Residuals', line=dict(color='red')))
fig.add_trace(go.Scatter(x=measurement_data['time'].values, y=station_2_residuals[0,:], mode='markers', name='Station 2 Range Residuals', line=dict(color='green')))
fig.add_trace(go.Scatter(x=measurement_data['time'].values, y=station_3_residuals[0,:], mode='markers', name='Station 3 Range Residuals', line=dict(color='blue')))
fig.update_traces(marker=dict(size=4))
fig.update_xaxes(title_text="Time (s)")
fig.update_yaxes(title_text="Range Residuals (km)", showexponent="all", exponentformat="e", range=[-4,4])
fig.update_layout(title_text="Measurement Residuals Over Time",
                  title_font=dict(size=24),
                  width=1200,
                  height=800,
                  legend=dict(font=dict(size=14)))
fig.write_html("ASEN_6080/HW2/figures/measurement_range_residuals.html")

fig = go.Figure()
fig.add_trace(go.Scatter(x=measurement_data['time'].values, y=station_1_residuals[1,:]*10**6, mode='markers', name='Station 1 Range Rate Residuals', line=dict(color='red')))
fig.add_trace(go.Scatter(x=measurement_data['time'].values, y=station_2_residuals[1,:]*10**6, mode='markers', name='Station 2 Range Rate Residuals', line=dict(color='green')))
fig.add_trace(go.Scatter(x=measurement_data['time'].values, y=station_3_residuals[1,:]*10**6, mode='markers', name='Station 3 Range Rate Residuals', line=dict(color='blue')))
fig.update_traces(marker=dict(size=4))
fig.update_xaxes(title_text="Time (s)")
fig.update_yaxes(title_text="Range Rate Residuals (mm/s)", showexponent="all", exponentformat="e", range=[-4,4])
fig.update_layout(title_text="Measurement Residuals Over Time",
                  title_font=dict(size=24),
                  width=1200,
                  height=800,
                  legend=dict(font=dict(size=14)))
fig.write_html("ASEN_6080/HW2/figures/measurement_range_rate_residuals.html")

# Measurement histograms
fig = make_subplots(rows=2, cols=1, subplot_titles=("Range Residuals Histogram", "Range Rate Residuals Histogram"))
fig.add_trace(go.Histogram(x=station_1_residuals[0,:], xbins=dict(size=1e-1), name='Station 1 Range Residuals', marker_color='red', opacity=0.7), row=1, col=1)
fig.add_trace(go.Histogram(x=station_2_residuals[0,:], xbins=dict(size=1e-1), name='Station 2 Range Residuals', marker_color='green', opacity=0.7), row=1, col=1)
fig.add_trace(go.Histogram(x=station_3_residuals[0,:], xbins=dict(size=1e-1), name='Station 3 Range Residuals', marker_color='blue', opacity=0.7), row=1, col=1)
fig.add_trace(go.Histogram(x=station_1_residuals[1,:], xbins=dict(size=1e-5), name='Station 1 Range Rate Residuals', marker_color='red', opacity=0.7), row=2, col=1)
fig.add_trace(go.Histogram(x=station_2_residuals[1,:], xbins=dict(size=1e-5), name='Station 2 Range Rate Residuals', marker_color='green', opacity=0.7), row=2, col=1)
fig.add_trace(go.Histogram(x=station_3_residuals[1,:], xbins=dict(size=1e-5), name='Station 3 Range Rate Residuals', marker_color='blue', opacity=0.7), row=2, col=1)
fig.update_xaxes(title_text="Residuals", row=1, col=1)
fig.update_xaxes(title_text="Residuals", row=2, col=1)
fig.update_yaxes(title_text="Count", row=1, col=1)
fig.update_yaxes(title_text="Count", row=2, col=1)
fig.update_layout(title_text="Measurement Residuals Histograms",
                  title_font=dict(size=24),
                  width=1200,
                  height=800,
                  legend=dict(font=dict(size=14)),
                  bargap=0.2)
fig.write_html("ASEN_6080/HW2/figures/measurement_residuals_histograms.html")