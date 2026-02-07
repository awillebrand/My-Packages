import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from ASEN_6080.Tools import Integrator, MeasurementMgr, CoordinateMgr, LKF, BatchLLSEstimator
from plotly.subplots import make_subplots

measurements = pd.read_pickle(".\ASEN_6080\Project1\data\conditioned_measurements.pkl")
time_vector = measurements['time'].values

# Initialize scenario

sat_state = np.array([757700.0E-3, 5222607.0E-3, 4851500.0E-3, 2213.21E-3, 4678.34E-3, -5371.30E-3])  # Satellite state in km and km/s
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

# Set up filter parameters

station_positions_ecef = np.array([station_1_state[0:3], station_2_state[0:3], station_3_state[0:3]])
initial_state_estimate = np.concatenate([sat_state[0:6], [mu, J2, C_d], station_1_state[0:3], station_2_state[0:3], station_3_state[0:3]]).flatten()

R = np.diag([1E-5**2, 1E-6**2])  # Noise covariance matrix for range and range rate. Corresponds to 1 cm range noise and 1 mm/s range rate noise.
a_priori_covariance = np.diag([1, 1, 1, 1, 1, 1, 1E2, 1E6, 1E6, 1E-16, 1E-16, 1E-16, 1, 1, 1, 1, 1, 1])  # Given a priori covariance

# Initialize integrator and measurement managers
integrator = Integrator(mu, R_e, mode=['mu','J2','Drag','Stations'], parameter_indices=[6,7,8,9], spacecraft_area=spacecraft_area, spacecraft_mass=spacecraft_mass, number_of_stations=3)

station_1_mgr = MeasurementMgr("station_101", station_state_ecef=station_1_state, initial_earth_spin_angle=0.0, earth_spin_rate=earth_spin_rate, R_e=R_e)
station_2_mgr = MeasurementMgr("station_337", station_state_ecef=station_2_state, initial_earth_spin_angle=0.0, earth_spin_rate=earth_spin_rate, R_e=R_e)
station_3_mgr = MeasurementMgr("station_394", station_state_ecef=station_3_state, initial_earth_spin_angle=0.0, earth_spin_rate=earth_spin_rate, R_e=R_e)

station_mgr_list = [station_1_mgr, station_2_mgr, station_3_mgr]

# Run filters
# lkf = LKF(integrator, station_mgr_list, initial_earth_spin_angle=0.0, earth_rotation_rate=earth_spin_rate)
# lkf_state_history, lkf_covariance_history, post_fit_residuals = lkf.run(initial_state_estimate, np.zeros_like(initial_state_estimate), a_priori_covariance, measurements, R=R, max_iterations=5, convergence_threshold=1e-9)

batch_estimator = BatchLLSEstimator(integrator, station_mgr_list, initial_earth_spin_angle=0.0, earth_rotation_rate=earth_spin_rate)
estimated_initial_state, estimated_covariance, residuals_df = batch_estimator.estimate_initial_state(
    a_priori_state=initial_state_estimate,
    a_priori_covariance=a_priori_covariance,
    measurement_data=measurements,
    R=R,
    max_iterations=3,
    tol=1E-6)

# Plot residual time history for each station
for iteration in range(residuals_df['iteration'].max()+1):
    fig = go.Figure()
    for station_name in residuals_df['station'].unique():
        mask = (residuals_df['iteration'] == iteration) & (residuals_df['station'] == station_name)

        post_fit_residuals = np.vstack(residuals_df[mask]['post-fit'])
        print(f"Residual at index 0: {post_fit_residuals[:,0:10]}")
        fig.add_trace(go.Scatter(x=time_vector, y=post_fit_residuals[0,:], mode='markers', name=f'{station_name} Range Residuals'))
        fig.add_trace(go.Scatter(x=time_vector, y=post_fit_residuals[1,:], mode='markers', name=f'{station_name} Range Rate Residuals'))
    fig.update_layout(title=f'Post-Fit Residuals at Iteration {iteration+1}', xaxis_title='Time (s)', yaxis_title='Residuals')
    fig.show()