import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from ASEN_6080.Tools import Integrator, MeasurementMgr, CoordinateMgr, LKF, EKF
from plotly.subplots import make_subplots
import warnings
warnings.simplefilter('error', RuntimeWarning)
measurement_data = pd.read_pickle("ASEN_6080/HW2/measurement_data/simulated_measurements_J3.pkl")
truth_data = pd.read_pickle("ASEN_6080/HW2/measurement_data/truth_data_J3.pkl")
time_vector = measurement_data['time'].values

mu = 3.986004415E5
R_e = 6378
J2 = 0.0010826269
J3 = -2.5324e-6

# Compute orbit period for time step calculation
a = 10000  # Semi-major axis in km
T = 2 * np.pi * np.sqrt(a**3 / mu)  # Orbital period in seconds
raw_state_length = 10
noise_var = np.array([1e-3, 1e-6])**2 # [range noise = 1 m, range rate noise = 1 mm/s]
#noise_var = np.zeros(2)  # No noise for testing
integrator = Integrator(mu, R_e, mode=['J2'], parameter_indices=[6])
station_1_mgr = MeasurementMgr("station_1", station_lat=-35.398333, station_lon=148.981944, initial_earth_spin_angle=np.deg2rad(122))
station_2_mgr = MeasurementMgr("station_2", station_lat=40.427222, station_lon=355.749444, initial_earth_spin_angle=np.deg2rad(122))
station_3_mgr = MeasurementMgr("station_3", station_lat=35.247163, station_lon=243.205, initial_earth_spin_angle=np.deg2rad(122))
station_mgr_list = [station_1_mgr, station_2_mgr, station_3_mgr]

initial_state_deviation = np.array([1.010e-02, -1.218e-01, -1.484e-01,  3.204e-05, -8.320e-05, 1.740e-04,  0.000e+00, 0, 0, 0])
initial_state = np.concatenate((truth_data['initial_state'].values[0][0:7], np.zeros(3)))  # Augment initial state with zeros for DMC states
initial_state_guess = initial_state + initial_state_deviation
P_0 = np.diag([1, 1, 1, 1e-3, 1e-3, 1e-3, 1e-8, 1e-6, 1e-6, 1e-6])**2

beta_mat = np.diag([30/T, 30/T, 30/T])  # Time constants for DMC in seconds

sigma = 5e-10

Q = np.diag([sigma, sigma, sigma])**2
lkf = LKF(integrator, station_mgr_list, initial_earth_spin_angle=np.deg2rad(122))
ekf = EKF(integrator, station_mgr_list, initial_earth_spin_angle=np.deg2rad(122))

lkf_state_history, lkf_covariance_history, lkf_residuals_df = lkf.run(initial_state_guess, np.zeros(10), P_0, measurement_data, R=np.diag(noise_var), max_iterations=1, process_noise_approach='DMC', Q=Q, beta_mat=beta_mat)
ekf_state_history, ekf_covariance_history, ekf_residuals_df = ekf.run(initial_state_guess, np.zeros(10), P_0, measurement_data, R=np.diag(noise_var), start_mode='warm', start_length=1000, process_noise_approach='DMC', Q=Q, beta_mat=beta_mat)

# Compute time history of J3 perturbation acceleration
J3_acceleration_history = np.zeros((3, (len(time_vector))))
for k in range(len(time_vector)):
    state = truth_data['augmented_state_history'].values[k]
    r_vec = state[0:3]
    x, y, z = r_vec
    r = np.linalg.norm(r_vec)
    J3_x = (5 / 2) * mu * J3 * R_e**3 * x * z / r**7 * (7 * z**2 / r**2 - 3)
    J3_y = (5 / 2) * mu * J3 * R_e**3 * y * z / r**7 * (7 * z**2 / r**2 - 3)
    J3_z = (5 / 2) * mu * J3 * R_e**3 / r**5 * (7 * z**4 / r**4 - 6 * z**2 / r**2 + 3 / 5)
    J3_acceleration_history[:,k] = np.array([J3_x, J3_y, J3_z])

# Plot the J3 acceleration history vs Estimated DMC acceleration history
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=("X-Component of Acceleration", "Y-Component of Acceleration", "Z-Component of Acceleration"))
fig.add_trace(go.Scatter(x=time_vector, y=J3_acceleration_history[0,:], mode='lines', name='True J3 Acceleration', line=dict(color='blue')), row=1, col=1)
fig.add_trace(go.Scatter(x=time_vector, y=lkf_state_history[7,:], mode='lines', name='Estimated Acceleration', line=dict(color='red')), row=1, col=1)
fig.add_trace(go.Scatter(x=time_vector, y=J3_acceleration_history[1,:], mode='lines', name='True J3 Acceleration', line=dict(color='blue'), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=time_vector, y=lkf_state_history[8,:], mode='lines', name='LKF Estimated DMC Acceleration', line=dict(color='red'), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=time_vector, y=J3_acceleration_history[2,:], mode='lines', name='True J3 Acceleration', line=dict(color='blue'), showlegend=False), row=3, col=1)
fig.add_trace(go.Scatter(x=time_vector, y=lkf_state_history[9,:], mode='lines', name='LKF Estimated DMC Acceleration', line=dict(color='red'), showlegend=False), row=3, col=1)
fig.update_xaxes(title_text="Time (s)", row=3, col=1)
fig.update_yaxes(title_text="Acceleration (km/s^2)", row=1, col=1, range = [-1.5*np.max(np.abs(J3_acceleration_history[0,:])) + np.mean(J3_acceleration_history[0,:]), 1.5*np.max(np.abs(J3_acceleration_history[0,:])) + np.mean(J3_acceleration_history[0,:])])
fig.update_yaxes(title_text="Acceleration (km/s^2)", row=2, col=1, range = [-1.5*np.max(np.abs(J3_acceleration_history[1,:])) + np.mean(J3_acceleration_history[1,:]), 1.5*np.max(np.abs(J3_acceleration_history[1,:])) + np.mean(J3_acceleration_history[1,:])])
fig.update_yaxes(title_text="Acceleration (km/s^2)", row=3, col=1, range = [-1.5*np.max(np.abs(J3_acceleration_history[2,:])) + np.mean(J3_acceleration_history[2,:]), 1.5*np.max(np.abs(J3_acceleration_history[2,:])) + np.mean(J3_acceleration_history[2,:])])
fig.update_layout(title_text="True J3 Acceleration vs LKF Estimated DMC Acceleration",
                  title_font=dict(size=28),
                  width=1200,
                  height=800,
                  legend=dict(font=dict(size=18)))
fig.write_html("ASEN_6080/HW3/figures/lkf_J3_acceleration_comparison.html")
fig.write_image("ASEN_6080/HW3/figures/pngs/lkf_J3_acceleration_comparison.png")

fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=("X-Component of Acceleration", "Y-Component of Acceleration", "Z-Component of Acceleration"))
fig.add_trace(go.Scatter(x=time_vector, y=J3_acceleration_history[0,:], mode='lines', name='True J3 Acceleration', line=dict(color='blue')), row=1, col=1)
fig.add_trace(go.Scatter(x=time_vector, y=ekf_state_history[7,:], mode='lines', name='Estimated Acceleration', line=dict(color='red')), row=1, col=1)
fig.add_trace(go.Scatter(x=time_vector, y=J3_acceleration_history[1,:], mode='lines', name='True J3 Acceleration', line=dict(color='blue'), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=time_vector, y=ekf_state_history[8,:], mode='lines', name='LKF Estimated DMC Acceleration', line=dict(color='red'), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=time_vector, y=J3_acceleration_history[2,:], mode='lines', name='True J3 Acceleration', line=dict(color='blue'), showlegend=False), row=3, col=1)
fig.add_trace(go.Scatter(x=time_vector, y=ekf_state_history[9,:], mode='lines', name='LKF Estimated DMC Acceleration', line=dict(color='red'), showlegend=False), row=3, col=1)
fig.update_xaxes(title_text="Time (s)", row=3, col=1)
fig.update_yaxes(title_text="Acceleration (km/s^2)", row=1, col=1, range = [-1.5*np.max(np.abs(J3_acceleration_history[0,:])) + np.mean(J3_acceleration_history[0,:]), 1.5*np.max(np.abs(J3_acceleration_history[0,:])) + np.mean(J3_acceleration_history[0,:])])
fig.update_yaxes(title_text="Acceleration (km/s^2)", row=2, col=1, range = [-1.5*np.max(np.abs(J3_acceleration_history[1,:])) + np.mean(J3_acceleration_history[1,:]), 1.5*np.max(np.abs(J3_acceleration_history[1,:])) + np.mean(J3_acceleration_history[1,:])])
fig.update_yaxes(title_text="Acceleration (km/s^2)", row=3, col=1, range = [-1.5*np.max(np.abs(J3_acceleration_history[2,:])) + np.mean(J3_acceleration_history[2,:]), 1.5*np.max(np.abs(J3_acceleration_history[2,:])) + np.mean(J3_acceleration_history[2,:])])
fig.update_layout(title_text="True J3 Acceleration vs EKF Estimated DMC Acceleration",
                  title_font=dict(size=28),
                  width=1200,
                  height=800,
                  legend=dict(font=dict(size=18)))
fig.write_html("ASEN_6080/HW3/figures/ekf_J3_acceleration_comparison.html")
fig.write_image("ASEN_6080/HW3/figures/pngs/ekf_J3_acceleration_comparison.png")

# Compute acceleration estimation error for both filters
lkf_acceleration_error = lkf_state_history[7:10,:] - J3_acceleration_history
ekf_acceleration_error = ekf_state_history[7:10,:] - J3_acceleration_history

# Plot the acceleration estimation error for both filters
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=("X-Component of Acceleration Error", "Y-Component of Acceleration Error", "Z-Component of Acceleration Error"))
fig.add_trace(go.Scatter(x=time_vector, y=lkf_acceleration_error[0,:], mode='lines', name='Acceleration Estimation Error', line=dict(color='blue')), row=1, col=1)
fig.add_trace(go.Scatter(x=time_vector, y=lkf_acceleration_error[1,:], mode='lines', name='LKF Acceleration Estimation Error', line=dict(color='blue'), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=time_vector, y=lkf_acceleration_error[2,:], mode='lines', name='LKF Acceleration Estimation Error', line=dict(color='blue'), showlegend=False), row=3, col=1)
# Add covariance bounds
fig.add_trace(go.Scatter(x=time_vector, y=3*np.sqrt(np.abs(lkf_covariance_history[7,7,:])), mode='lines', name='3-Sigma Bound', line=dict(color='red', dash='dash')), row=1, col=1)
fig.add_trace(go.Scatter(x=time_vector, y=-3*np.sqrt(np.abs(lkf_covariance_history[7,7,:])), mode='lines', name='-3-Sigma Bound', line=dict(color='red', dash='dash'), showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=time_vector, y=3*np.sqrt(np.abs(lkf_covariance_history[8,8,:])), mode='lines', name='3-Sigma Bound', line=dict(color='red', dash='dash'), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=time_vector, y=-3*np.sqrt(np.abs(lkf_covariance_history[8,8,:])), mode='lines', name='-3-Sigma Bound', line=dict(color='red', dash='dash'), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=time_vector, y=3*np.sqrt(np.abs(lkf_covariance_history[9,9,:])), mode='lines', name='3-Sigma Bound', line=dict(color='red', dash='dash'), showlegend=False), row=3, col=1)
fig.add_trace(go.Scatter(x=time_vector, y=-3*np.sqrt(np.abs(lkf_covariance_history[9,9,:])), mode='lines', name='-3-Sigma Bound', line=dict(color='red', dash='dash'), showlegend=False), row=3, col=1)
fig.update_xaxes(title_text="Time (s)", row=3, col=1)
fig.update_yaxes(title_text="Acceleration Error (km/s^2)", row=1, col=1, range = [-4.5*np.sqrt(np.abs(lkf_covariance_history[7,7,-1])), 4.5*np.sqrt(np.abs(lkf_covariance_history[7,7,-1]))])
fig.update_yaxes(title_text="Acceleration Error (km/s^2)", row=2, col=1, range = [-4.5*np.sqrt(np.abs(lkf_covariance_history[8,8,-1])), 4.5*np.sqrt(np.abs(lkf_covariance_history[8,8,-1]))])
fig.update_yaxes(title_text="Acceleration Error (km/s^2)", row=3, col=1, range = [-4.5*np.sqrt(np.abs(lkf_covariance_history[9,9,-1])), 4.5*np.sqrt(np.abs(lkf_covariance_history[9,9,-1]))])
fig.update_layout(title_text="LKF DMC Acceleration Estimation Error with 3-Sigma Bounds",
                  title_font=dict(size=28),
                  width=1400,
                  height=1000,
                  legend=dict(font=dict(size=18),
                  yanchor='top',
                  y=1.1,
                  xanchor='left',
                  x=0.8,
                  itemsizing='constant'))
fig.write_html("ASEN_6080/HW3/figures/lkf_J3_acceleration_error.html")
fig.write_image("ASEN_6080/HW3/figures/pngs/lkf_J3_acceleration_error.png")

fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=("X-Component of Acceleration Error", "Y-Component of Acceleration Error", "Z-Component of Acceleration Error"))
fig.add_trace(go.Scatter(x=time_vector, y=ekf_acceleration_error[0,:], mode='lines', name='Acceleration Estimation Error', line=dict(color='blue')), row=1, col=1)
fig.add_trace(go.Scatter(x=time_vector, y=ekf_acceleration_error[1,:], mode='lines', name='EKF Acceleration Estimation Error', line=dict(color='blue'), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=time_vector, y=ekf_acceleration_error[2,:], mode='lines', name='EKF Acceleration Estimation Error', line=dict(color='blue'), showlegend=False), row=3, col=1)
# Add covariance bounds
fig.add_trace(go.Scatter(x=time_vector, y=3*np.sqrt(np.abs(ekf_covariance_history[7,7,:])), mode='lines', name='3-Sigma Bound', line=dict(color='red', dash='dash')), row=1, col=1)
fig.add_trace(go.Scatter(x=time_vector, y=-3*np.sqrt(np.abs(ekf_covariance_history[7,7,:])), mode='lines', name='-3-Sigma Bound', line=dict(color='red', dash='dash'), showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=time_vector, y=3*np.sqrt(np.abs(ekf_covariance_history[8,8,:])), mode='lines', name='3-Sigma Bound', line=dict(color='red', dash='dash'), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=time_vector, y=-3*np.sqrt(np.abs(ekf_covariance_history[8,8,:])), mode='lines', name='-3-Sigma Bound', line=dict(color='red', dash='dash'), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=time_vector, y=3*np.sqrt(np.abs(ekf_covariance_history[9,9,:])), mode='lines', name='3-Sigma Bound', line=dict(color='red', dash='dash'), showlegend=False), row=3, col=1)
fig.add_trace(go.Scatter(x=time_vector, y=-3*np.sqrt(np.abs(ekf_covariance_history[9,9,:])), mode='lines', name='-3-Sigma Bound', line=dict(color='red', dash='dash'), showlegend=False), row=3, col=1)
fig.update_xaxes(title_text="Time (s)", row=3, col=1)
fig.update_yaxes(title_text="Acceleration Error (km/s^2)", row=1, col=1, range = [-4.5*np.sqrt(np.abs(ekf_covariance_history[7,7,-1])), 4.5*np.sqrt(np.abs(ekf_covariance_history[7,7,-1]))])
fig.update_yaxes(title_text="Acceleration Error (km/s^2)", row=2, col=1, range = [-4.5*np.sqrt(np.abs(ekf_covariance_history[8,8,-1])), 4.5*np.sqrt(np.abs(ekf_covariance_history[8,8,-1]))])
fig.update_yaxes(title_text="Acceleration Error (km/s^2)", row=3, col=1, range = [-4.5*np.sqrt(np.abs(ekf_covariance_history[9,9,-1])), 4.5*np.sqrt(np.abs(ekf_covariance_history[9,9,-1]))])
fig.update_layout(title_text="EKF DMC Acceleration Estimation Error with 3-Sigma Bounds",
                  title_font=dict(size=28),
                  width=1400,
                  height=1000,
                  legend=dict(font=dict(size=18),
                  yanchor='top',
                  y=1.1,
                  xanchor='left',
                  x=0.8,
                  itemsizing='constant'))
fig.write_html("ASEN_6080/HW3/figures/ekf_J3_acceleration_error.html")
fig.write_image("ASEN_6080/HW3/figures/pngs/ekf_J3_acceleration_error.png")
