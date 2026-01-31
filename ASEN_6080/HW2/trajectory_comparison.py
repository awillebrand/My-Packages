import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from ASEN_6080.Tools import Integrator, MeasurementMgr, CoordinateMgr, LKF
from plotly.subplots import make_subplots

J2_measurement_data = pd.read_pickle("ASEN_6080/HW2/measurement_data/simulated_measurements.pkl")
J2_truth_data = pd.read_pickle("ASEN_6080/HW2/measurement_data/truth_data.pkl")
J3_measurement_data = pd.read_pickle("ASEN_6080/HW2/measurement_data/simulated_measurements_J3.pkl")
J3_truth_data = pd.read_pickle("ASEN_6080/HW2/measurement_data/truth_data_J3.pkl")

# Set time vector
times = J2_measurement_data['time'].values

# Pull truth states from truth data
J2_truth_augmented_states = np.array([state for state in J2_truth_data['augmented_state_history'].values]).T
J3_truth_augmented_states = np.array([state for state in J3_truth_data['augmented_state_history'].values]).T

J2_truth_states = J2_truth_augmented_states[0:6,:]
J3_truth_states = J3_truth_augmented_states[0:6,:]

state_diff = J3_truth_states - J2_truth_states

# Take measurement differences by station
station_1_diff = np.vstack(J3_measurement_data['station_1_measurements'].values - J2_measurement_data['station_1_measurements'].values)
station_2_diff = np.vstack(J3_measurement_data['station_2_measurements'].values - J2_measurement_data['station_2_measurements'].values)
station_3_diff = np.vstack(J3_measurement_data['station_3_measurements'].values - J2_measurement_data['station_3_measurements'].values)

# Plot comparison of trajectories
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=("X Position Difference", "Y Position Difference", "Z Position Difference"))
fig.add_trace(go.Scatter(x=times, y=state_diff[0,:], mode='lines', name='X Position Difference', line=dict(color='blue'), showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=times, y=state_diff[1,:], mode='lines', name='Y Position Difference', line=dict(color='blue'), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=times, y=state_diff[2,:], mode='lines', name='Z Position Difference', line=dict(color='blue'), showlegend=False), row=3, col=1)
fig.update_xaxes(title_text="Time (s)", row=3, col=1)
fig.update_yaxes(title_text="Position Difference (km)", row=1, col=1)
fig.update_yaxes(title_text="Position Difference (km)", row=2, col=1)
fig.update_yaxes(title_text="Position Difference (km)", row=3, col=1)
fig.update_layout(title_text="J3 - J2 Truth State Position Differences Over Time",
                  title_font=dict(size=28),
                  width=1200,
                  height=800)
fig.write_html("ASEN_6080/HW2/figures/j3_j2_position_differences.html")
fig.write_image("ASEN_6080/HW2/figures/pngs/j3_j2_position_differences.png")

fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=("X Velocity Difference", "Y Velocity Difference", "Z Velocity Difference"))
fig.add_trace(go.Scatter(x=times, y=state_diff[3,:]*1e6, mode='lines', name='X Velocity Difference', line=dict(color='blue'), showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=times, y=state_diff[4,:]*1e6, mode='lines', name='Y Velocity Difference', line=dict(color='blue'), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=times, y=state_diff[5,:]*1e6, mode='lines', name='Z Velocity Difference', line=dict(color='blue'), showlegend=False), row=3, col=1)
fig.update_xaxes(title_text="Time (s)", row=3, col=1)
fig.update_yaxes(title_text="Velocity Difference (mm/s)", row=1, col=1)
fig.update_yaxes(title_text="Velocity Difference (mm/s)", row=2, col=1)
fig.update_yaxes(title_text="Velocity Difference (mm/s)", row=3, col=1)
fig.update_layout(title_text="J3 - J2 Truth State Velocity Differences Over Time",
                  title_font=dict(size=28),
                  width=1200,
                  height=800)
fig.write_html("ASEN_6080/HW2/figures/j3_j2_velocity_differences.html")
fig.write_image("ASEN_6080/HW2/figures/pngs/j3_j2_velocity_differences.png")

# Plot comparison of measurements by station
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Range Difference", "Range Rate Difference"))
fig.add_trace(go.Scatter(x=times, y=station_1_diff[:,0], mode='markers', name='Station 1', line=dict(color='red')), row=1, col=1)
fig.add_trace(go.Scatter(x=times, y=station_2_diff[:,0], mode='markers', name='Station 2', line=dict(color='green')), row=1, col=1)
fig.add_trace(go.Scatter(x=times, y=station_3_diff[:,0], mode='markers', name='Station 3', line=dict(color='blue')), row=1, col=1)
fig.add_trace(go.Scatter(x=times, y=station_1_diff[:,1]*1e6, mode='markers', name='Station 1 Range Rate Difference', line=dict(color='red'), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=times, y=station_2_diff[:,1]*1e6, mode='markers', name='Station 2 Range Rate Difference', line=dict(color='green'), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=times, y=station_3_diff[:,1]*1e6, mode='markers', name='Station 3 Range Rate Difference', line=dict(color='blue'), showlegend=False), row=2, col=1)
fig.update_xaxes(title_text="Time (s)", row=2, col=1)
fig.update_yaxes(title_text="Range Difference (km)", row=1, col=1)
fig.update_yaxes(title_text="Range Rate Difference (mm/s)", row=2, col=1)
fig.update_layout(title_text="J3 - J2 Measurements Differences by Station Over Time",
                  title_font=dict(size=28),
                  width=1200,
                  height=800,
                  legend=dict(font=dict(size=18),
                              yanchor="top",
                              y=1.17,
                              xanchor="left",
                              x=0.87))
fig.update_traces(marker=dict(size=2))
fig.write_html("ASEN_6080/HW2/figures/j3_j2_measurement_differences.html")
fig.write_image("ASEN_6080/HW2/figures/pngs/j3_j2_measurement_differences.png")