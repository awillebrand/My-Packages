import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from ASEN_6080.Tools import Integrator, MeasurementMgr, CoordinateMgr, LKF, BatchLLSEstimator, plot_state_errors
from plotly.subplots import make_subplots
import warnings
warnings.simplefilter('error', RuntimeWarning)
measurement_data = pd.read_pickle("ASEN_6080/HW2/measurement_data/simulated_measurements.pkl")
truth_data = pd.read_pickle("ASEN_6080/HW2/measurement_data/truth_data.pkl")

mu = 3.986004415E5
R_e = 6378
J2 = 0.0010826269

raw_state_length = 7
noise_var = np.array([1e-3, 1e-6])**2 # [range noise = 1 m, range rate noise = 1 mm/s]
#noise_var = np.zeros(2)  # No noise for testing
integrator = Integrator(mu, R_e, mode=['J2'], parameter_indices=[6])
station_1_mgr = MeasurementMgr("station_1", station_lat=-35.398333, station_lon=148.981944, initial_earth_spin_angle=np.deg2rad(122))
station_2_mgr = MeasurementMgr("station_2", station_lat=40.427222, station_lon=355.749444, initial_earth_spin_angle=np.deg2rad(122))
station_3_mgr = MeasurementMgr("station_3", station_lat=35.247163, station_lon=243.205, initial_earth_spin_angle=np.deg2rad(122))
station_mgr_list = [station_1_mgr, station_2_mgr, station_3_mgr]

initial_state_deviation = np.array([1.010e-02, -1.218e-01, -1.484e-01,  3.204e-05, -8.320e-05, 1.740e-04,  0.000e+00])
initial_state_guess = truth_data['initial_state'].values[0][0:7] + initial_state_deviation
P_0 = np.diag([1, 1, 1, 1e-3, 1e-3, 1e-3,1e-10])**2
large_P_0 = np.diag([1000, 1000, 1000, 1, 1, 1])**2

lkf = LKF(integrator, station_mgr_list, initial_earth_spin_angle=np.deg2rad(122))
batch_estimator = BatchLLSEstimator(integrator, station_mgr_list, np.deg2rad(122.0))

lkf_estimated_state_history, lkf_covariance_history, lkf_residuals_df = lkf.run(initial_state_guess, np.zeros(7), P_0, measurement_data, R=np.diag(noise_var), max_iterations=1, apply_smoothing=True)
batch_estimated_state, batch_estimated_covariance, batch_residuals_df = batch_estimator.estimate_initial_state(initial_state_guess, measurement_data, np.diag(noise_var), tol=2e-9, a_priori_covariance=P_0, max_iterations=1)

# Propagate batch estimate forward in time for comparison
[_, estimated_state_history] = integrator.integrate_stm(measurement_data['time'].values[-1], batch_estimated_state, teval=measurement_data['time'].values)

# Verify against truth data
augmented_truth_state = truth_data['augmented_state_history'].values

truth_state_history = np.zeros((7, augmented_truth_state.shape[0]))

for i, state in enumerate(augmented_truth_state):
    truth_state = state[0:7]
    truth_state_history[:, i] = truth_state

# Break integrated state into raw state and covariance
batch_estimated_state_history = np.zeros((7, estimated_state_history.shape[1]))
batch_covariance_history = np.zeros((7,7, estimated_state_history.shape[1]))
for k in range(estimated_state_history.shape[1]):
    batch_estimated_state_history[:,k] = estimated_state_history[:7,k]
    batch_covariance_history[:,:,k] = estimated_state_history[7:,k].reshape(7,7)

# Take difference between lkf and batch estimates
state_difference = lkf_estimated_state_history - batch_estimated_state_history

# Take difference between truth and filter estimates
lkf_truth_difference = lkf_estimated_state_history - truth_state_history
batch_truth_difference = batch_estimated_state_history - truth_state_history

# Plotting

fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=['X Position Difference', 'Y Position Difference', 'Z Position Difference'])
fig.add_trace(go.Scatter(x=measurement_data['time'].values, y=lkf_truth_difference[0,:]*1E3, mode='lines', name="LKF - Truth", line=dict(color='blue')), row=1, col=1)
fig.add_trace(go.Scatter(x=measurement_data['time'].values, y=batch_truth_difference[0,:]*1E3, mode='lines', name="Batch - Truth", line=dict(color='red')), row=1, col=1)
fig.add_trace(go.Scatter(x=measurement_data['time'].values, y=lkf_truth_difference[1,:]*1E3, mode='lines', name="LKF - Truth", line=dict(color='blue'), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=measurement_data['time'].values, y=batch_truth_difference[1,:]*1E3, mode='lines', name="Batch - Truth", line=dict(color='red'), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=measurement_data['time'].values, y=lkf_truth_difference[2,:]*1E3, mode='lines', name="LKF - Truth", line=dict(color='blue'), showlegend=False), row=3, col=1)
fig.add_trace(go.Scatter(x=measurement_data['time'].values, y=batch_truth_difference[2,:]*1E3, mode='lines', name="Batch - Truth", line=dict(color='red'), showlegend=False), row=3, col=1)
# Match layout of plots generated in plotting_functions.py
fig.update_layout(title="Position Error for LKF and Batch LLS",
                  xaxis_title="Time (s)",
                  yaxis_title="Position Difference (m)",
                  title_font=dict(size=30),
                    legend=dict(font=dict(size=22),
                                orientation="h",
                                yanchor="top",
                                y=1.1,
                                xanchor="left",
                                x=0.7,
                                itemsizing='constant'),
                  height=900, width=1500)
fig.update_xaxes(title_text="Time (s)", tickfont=dict(size=20), title_font=dict(size=22))
fig.update_yaxes(title_text="Position Error (m)", tickfont=dict(size=20), title_font=dict(size=22), showexponent="all", exponentformat="e")
fig.write_html("ASEN_6080/HW4/figures/position_filter_comparison.html")
fig.write_image("ASEN_6080/HW4/figures/pngs/position_filter_comparison.png")

fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=['X Velocity Difference', 'Y Velocity Difference', 'Z Velocity Difference'])
fig.add_trace(go.Scatter(x=measurement_data['time'].values, y=lkf_truth_difference[3,:]*1E6, mode='lines', name="LKF - Truth", line=dict(color='blue')), row=1, col=1)
fig.add_trace(go.Scatter(x=measurement_data['time'].values, y=batch_truth_difference[3,:]*1E6, mode='lines', name="Batch - Truth", line=dict(color='red')), row=1, col=1)
fig.add_trace(go.Scatter(x=measurement_data['time'].values, y=lkf_truth_difference[4,:]*1E6, mode='lines', name="LKF - Truth", line=dict(color='blue'), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=measurement_data['time'].values, y=batch_truth_difference[4,:]*1E6, mode='lines', name="Batch - Truth", line=dict(color='red'), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=measurement_data['time'].values, y=lkf_truth_difference[5,:]*1E6, mode='lines', name="LKF - Truth", line=dict(color='blue'), showlegend=False), row=3, col=1)
fig.add_trace(go.Scatter(x=measurement_data['time'].values, y=batch_truth_difference[5,:]*1E6, mode='lines', name="Batch - Truth", line=dict(color='red'), showlegend=False), row=3, col=1)
# Match layout of plots generated in plotting_functions.py
fig.update_layout(title="Velocity Error for LKF and Batch LLS",
                  title_font=dict(size=30),
                    xaxis_title="Time (s)",
                    yaxis_title="Velocity Difference (mm/s)",
                    legend=dict(font=dict(size=22),
                                orientation="h",
                                yanchor="top",
                                y=1.1,
                                xanchor="left",
                                x=0.7,
                                itemsizing='constant'),
                    height=900, width=1500)
fig.update_xaxes(title_text="Time (s)", tickfont=dict(size=20), title_font=dict(size=22))
fig.update_yaxes(title_text="Velocity Error (mm/s)", tickfont=dict(size=20), title_font=dict(size=22), showexponent="all", exponentformat="e")
fig.write_html("ASEN_6080/HW4/figures/velocity_filter_comparison.html")
fig.write_image("ASEN_6080/HW4/figures/pngs/velocity_filter_comparison.png")

lkf_position_fig, lkf_velocity_fig, lkf_error_stats = plot_state_errors(measurement_data['time'].values, lkf_truth_difference, lkf_covariance_history, "LKF", "ASEN_6080/HW4/figures", unit_multipliers=[1e3, 1e6], units=['m', 'mm/s'])
# batch_position_fig, batch_velocity_fig, batch_error_stats = plot_state_errors(measurement_data['time'].values, batch_truth_difference, batch_covariance_history, "Batch_LLS", "ASEN_6080/HW4/figures")

print("No Noise LKF Error Stats:")
for state_name, stats in lkf_error_stats.items():
    print(f"{state_name}: Mean Error = {stats['mean']:.3e}, Std Dev = {stats['std']:.3e}, RMS = {stats['rms']:.3e}")

# Test smoothing with SNC

measurement_data = pd.read_pickle("ASEN_6080/HW2/measurement_data/simulated_measurements_J3.pkl")
truth_data = pd.read_pickle("ASEN_6080/HW2/measurement_data/truth_data_J3.pkl")
initial_state_guess = truth_data['initial_state'].values[0][0:7]

# Verify against truth data
augmented_truth_state = truth_data['augmented_state_history'].values

truth_state_history = np.zeros((7, augmented_truth_state.shape[0]))

for i, state in enumerate(augmented_truth_state):
    truth_state = state[0:7]
    truth_state_history[:, i] = truth_state

optimal_sigma = 5e-8
# optimal_sigma = 1e-10
Q = np.diag([optimal_sigma, optimal_sigma, optimal_sigma])**2
lkf_estimated_state_history_snc_smoothed, lkf_covariance_history_snc_smoothed, lkf_residuals_df_snc_smoothed = lkf.run(initial_state_guess, np.zeros(7), P_0, measurement_data, R=np.diag(noise_var), max_iterations=1, apply_smoothing=True, Q=Q, process_noise_approach='SNC')
lkf_estimated_state_history_snc_not_smoothed, lkf_covariance_history_snc_not_smoothed, lkf_residuals_df_snc_not_smoothed = lkf.run(initial_state_guess, np.zeros(7), P_0, measurement_data, R=np.diag(noise_var), max_iterations=1, apply_smoothing=False, Q=Q, process_noise_approach='SNC')

smoothed_snc_lkf_truth_difference = lkf_estimated_state_history_snc_smoothed - truth_state_history
not_smoothed_snc_lkf_truth_difference = lkf_estimated_state_history_snc_not_smoothed - truth_state_history

snc_position_fig, snc_velocity_fig, snc_error_stats = plot_state_errors(measurement_data['time'].values, smoothed_snc_lkf_truth_difference, lkf_covariance_history_snc_smoothed, "LKF with SNC", "ASEN_6080/HW4/figures", unit_multipliers=[1e3, 1e6], units=['m', 'mm/s'])
not_smoothed_snc_position_fig, not_smoothed_snc_velocity_fig, not_smoothed_snc_error_stats = plot_state_errors(measurement_data['time'].values, not_smoothed_snc_lkf_truth_difference, lkf_covariance_history_snc_not_smoothed, "LKF with SNC (No Smoothing)", "ASEN_6080/HW4/figures", unit_multipliers=[1e3, 1e6], units=['m', 'mm/s'])
print("LKF with SNC Error Stats:")
for state_name, stats in snc_error_stats.items():
    print(f"{state_name}: Mean Error = {stats['mean']:.3e}, Std Dev = {stats['std']:.3e}, RMS = {stats['rms']:.3e}")
print("LKF with SNC (No Smoothing) Error Stats:")
for state_name, stats in not_smoothed_snc_error_stats.items():
    print(f"{state_name}: Mean Error = {stats['mean']:.3e}, Std Dev = {stats['std']:.3e}, RMS = {stats['rms']:.3e}")

