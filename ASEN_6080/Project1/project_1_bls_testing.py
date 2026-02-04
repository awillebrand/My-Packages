import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from ASEN_6080.Tools import Integrator, MeasurementMgr, CoordinateMgr, BatchLLSEstimator
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
spacecraft_area = 3.0  # Spacecraft cross-sectional area in m^2
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

batch_estimator = BatchLLSEstimator(integrator, station_mgr_list, initial_earth_spin_angle=0.0, earth_rotation_rate=earth_spin_rate)

estimated_initial_state, estimated_covariance = batch_estimator.estimate_initial_state(
    a_priori_state=initial_state_estimate,
    a_priori_covariance=a_priori_covariance,
    measurement_data=measurements,
    R=R,
    max_iterations=10,
    tol=1E-6)

print("Estimated Initial State:")
np.set_printoptions(linewidth=200)
print(estimated_initial_state)

# Analyze validity of estimate by propagating estimated state and comparing to measurements
time_vector = measurements['time'].values
_, augmented_state_history = integrator.integrate_stm(time_vector[-1], estimated_initial_state, teval=time_vector)
residuals_matrix = np.empty((len(station_mgr_list), len(time_vector), 2))  # 2 for range and range rate
for i, mgr in enumerate(station_mgr_list):
    station_name = mgr.station_name

    truth_measurements = np.vstack(measurements[f"{station_name}_measurements"].values).T

    # Station position updated inside batch estimator
    simulated_measurements = mgr.simulate_measurements(augmented_state_history[0:6,:], time_vector, 'ECI', noise=False, ignore_visibility=True)

    # Compute measurement residuals
    residuals = truth_measurements - simulated_measurements
    residuals_matrix[i, :, :] = residuals.T

# Get residuals stats
for i, mgr in enumerate(station_mgr_list):
    station_name = mgr.station_name
    range_residuals = residuals_matrix[i,:,0]
    range_rate_residuals = residuals_matrix[i,:,1]
    print(f"{station_name} Residuals:")
    print(f"  Range Residuals: Mean = {np.nanmean(range_residuals)*1E5:.3f} cm, Std Dev = {np.nanstd(range_residuals)*1E5:.3f} cm")
    print(f"  Range Rate Residuals: Mean = {np.nanmean(range_rate_residuals)*1E6:.3f} mm/s, Std Dev = {np.nanstd(range_rate_residuals)*1E6:.3f} mm/s")

# Plot residuals
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
fig.write_html("ASEN_6080/Project1/figures/batch_lls_measurement_residuals.html")
