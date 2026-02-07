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
batch_estimator = BatchLLSEstimator(integrator, station_mgr_list, initial_earth_spin_angle=0.0, earth_rotation_rate=earth_spin_rate)
estimated_initial_state, estimated_covariance, batch_residuals_df = batch_estimator.estimate_initial_state(
    a_priori_state=initial_state_estimate,
    a_priori_covariance=a_priori_covariance,
    measurement_data=measurements,
    R=R,
    max_iterations=3,
    tol=1E-6)

lkf = LKF(integrator, station_mgr_list, initial_earth_spin_angle=0.0, earth_rotation_rate=earth_spin_rate)
lkf_state_history, lkf_covariance_history, lkf_residuals_df = lkf.run(initial_state_estimate,
                                                                      np.zeros_like(initial_state_estimate),
                                                                      a_priori_covariance, measurements,
                                                                      R=R, max_iterations=1,
                                                                      convergence_threshold=1e-9)

breakpoint()
# Integrate batch estimated initial state forward for comparison
[_, batch_estimated_state_history] = integrator.integrate_stm(time_vector[-1], estimated_initial_state, time_vector)

# Plot residual time history for each station
colors_list = ['red', 'green', 'blue']
residual_df_list = [batch_residuals_df, lkf_residuals_df]
for residuals_df, filter_name in zip(residual_df_list, ['Batch LLS Filter', 'LKF']):
    for iteration in range(residuals_df['iteration'].max()+1):
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=('Range Residuals', 'Range Rate Residuals'))
        for i, station_name in enumerate(residuals_df['station'].unique()):
            mask = (residuals_df['iteration'] == iteration) & (residuals_df['station'] == station_name)

            pre_fit_residuals = np.vstack(residuals_df[mask]['pre-fit'])
            fig.add_trace(go.Scatter(x=time_vector, y=pre_fit_residuals[0,:]*100000, mode='markers', name=f'{station_name}', marker=dict(color=colors_list[i])), row=1, col=1)
            fig.add_trace(go.Scatter(x=time_vector, y=pre_fit_residuals[1,:]*1E6, mode='markers', name=f'{station_name}', marker=dict(color=colors_list[i]), showlegend=False), row=2, col=1)
        fig.update_traces(marker=dict(size=4))
        fig.update_xaxes(title_text="Time (s)")
        fig.update_yaxes(title_text="Range Residuals (cm)", showexponent="all", exponentformat="e", row=1, col=1)
        fig.update_yaxes(title_text="Range Rate Residuals (mm/s)", showexponent="all", exponentformat="e", row=2, col=1)
        fig.update_layout(title_text=f"{filter_name} Pre-Fit Residuals Time History at Iteration {iteration+1}",
                        title_font=dict(size=28),
                        width=1200,
                        height=800,
                        legend=dict(font=dict(size=18),
                                    yanchor="top",
                                    y=1.2,
                                    xanchor="left",
                                    x=0.87))
        fig.show()

    for iteration in range(residuals_df['iteration'].max()+1):
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=('Range Residuals', 'Range Rate Residuals'))
        for i, station_name in enumerate(residuals_df['station'].unique()):
            mask = (residuals_df['iteration'] == iteration) & (residuals_df['station'] == station_name)

            post_fit_residuals = np.vstack(residuals_df[mask]['post-fit'])
            fig.add_trace(go.Scatter(x=time_vector, y=post_fit_residuals[0,:]*100000, mode='markers', name=f'{station_name}', marker=dict(color=colors_list[i])), row=1, col=1)
            fig.add_trace(go.Scatter(x=time_vector, y=post_fit_residuals[1,:]*1E6, mode='markers', name=f'{station_name}', marker=dict(color=colors_list[i]), showlegend=False), row=2, col=1)
        fig.update_traces(marker=dict(size=4))
        fig.update_xaxes(title_text="Time (s)")
        fig.update_yaxes(title_text="Range Residuals (cm)", showexponent="all", exponentformat="e", row=1, col=1)
        fig.update_yaxes(title_text="Range Rate Residuals (mm/s)", showexponent="all", exponentformat="e", row=2, col=1)
        fig.update_layout(title_text=f"{filter_name} Post-Fit Residuals Time History at Iteration {iteration+1}",
                        title_font=dict(size=28),
                        width=1200,
                        height=800,
                        legend=dict(font=dict(size=18),
                                    yanchor="top",
                                    y=1.2,
                                    xanchor="left",
                                    x=0.87))
        fig.show()

# Plot state history for batch LLS and LKF
state_labels = ['x (km)', 'y (km)', 'z (km)', 'vx (km/s)', 'vy (km/s)', 'vz (km/s)', 'mu (km^3/s^2)', 'J2', 'C_d', 'Station 1 x (km)', 'Station 1 y (km)', 'Station 1 z (km)', 'Station 2 x (km)', 'Station 2 y (km)', 'Station 2 z (km)', 'Station 3 x (km)', 'Station 3 y (km)', 'Station 3 z (km)']
for i in range(17):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_vector, y=batch_estimated_state_history[i,:], mode='lines', name='Batch LLS Estimate'))
    fig.add_trace(go.Scatter(x=time_vector, y=lkf_state_history[i,:], mode='lines', name='LKF Estimate'))
    fig.update_layout(title=f"State Component {state_labels[i]} Over Time", xaxis_title='Time (s)', yaxis_title=state_labels[i])
    fig.write_html(f"ASEN_6080/Project1/figures/{state_labels[i]}_time_histories.html")