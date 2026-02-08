import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from ASEN_6080.Tools import Integrator, MeasurementMgr, CoordinateMgr, LKF, BatchLLSEstimator, covariance_ellipse
from plotly.subplots import make_subplots

measurements = pd.read_pickle("./ASEN_6080/Project1/data/conditioned_measurements.pkl")
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
    tol=1E-6,
    considered_measurements='All')

lkf = LKF(integrator, station_mgr_list, initial_earth_spin_angle=0.0, earth_rotation_rate=earth_spin_rate)
lkf_state_history, lkf_covariance_history, lkf_residuals_df = lkf.run(initial_state_estimate,
                                                                      np.zeros_like(initial_state_estimate),
                                                                      a_priori_covariance, measurements,
                                                                      R=R, max_iterations=1,
                                                                      convergence_threshold=1e-9,
                                                                      considered_measurements='All')

# Reset measurement managers positions
for i, mgr in enumerate(batch_estimator.measurement_mgrs):
    mgr.station_state_ecef[0:3] = station_positions_ecef[i]
    mgr.lat, mgr.lon = mgr.coordinate_mgr.ECEF_to_GCS(mgr.station_state_ecef)

for i, mgr in enumerate(lkf.measurement_mgrs):
    mgr.station_state_ecef[0:3] = station_positions_ecef[i]
    mgr.lat, mgr.lon = mgr.coordinate_mgr.ECEF_to_GCS(mgr.station_state_ecef)

# Run filters with reduced measurement sets for comparison
range_estimated_initial_state, range_estimated_initial_covariance, range_batch_residuals_df = batch_estimator.estimate_initial_state(
    a_priori_state=initial_state_estimate,
    a_priori_covariance=a_priori_covariance,
    measurement_data=measurements,
    R=R,
    max_iterations=3,
    tol=1E-6,
    considered_measurements='Range')

# Reset measurement managers positions
for i, mgr in enumerate(batch_estimator.measurement_mgrs):
    mgr.station_state_ecef[0:3] = station_positions_ecef[i]
    mgr.lat, mgr.lon = mgr.coordinate_mgr.ECEF_to_GCS(mgr.station_state_ecef)

range_rate_estimated_initial_state, range_rate_estimated_initial_covariance, range_rate_batch_residuals_df = batch_estimator.estimate_initial_state(
    a_priori_state=initial_state_estimate,
    a_priori_covariance=a_priori_covariance,
    measurement_data=measurements,
    R=R,
    max_iterations=3,
    tol=1E-6,
    considered_measurements='Range Rate')

# Reset measurement managers positions
for i, mgr in enumerate(batch_estimator.measurement_mgrs):
    mgr.station_state_ecef[0:3] = station_positions_ecef[i]
    mgr.lat, mgr.lon = mgr.coordinate_mgr.ECEF_to_GCS(mgr.station_state_ecef)

range_lkf_state_history, range_lkf_covariance_history, range_lkf_residuals_df = lkf.run(initial_state_estimate,
                                                                        np.zeros_like(initial_state_estimate),
                                                                        a_priori_covariance, measurements,
                                                                        R=R, max_iterations=1,
                                                                        convergence_threshold=1e-9,
                                                                        considered_measurements='Range')

for i, mgr in enumerate(lkf.measurement_mgrs):
    mgr.station_state_ecef[0:3] = station_positions_ecef[i]
    mgr.lat, mgr.lon = mgr.coordinate_mgr.ECEF_to_GCS(mgr.station_state_ecef)

range_rate_lkf_state_history, range_rate_lkf_covariance_history, range_rate_lkf_residuals_df = lkf.run(initial_state_estimate,
                                                                        np.zeros_like(initial_state_estimate),
                                                                        a_priori_covariance, measurements,
                                                                        R=R, max_iterations=1,
                                                                        convergence_threshold=1e-9,
                                                                        considered_measurements='Range Rate')

for i, mgr in enumerate(lkf.measurement_mgrs):
    mgr.station_state_ecef[0:3] = station_positions_ecef[i]
    mgr.lat, mgr.lon = mgr.coordinate_mgr.ECEF_to_GCS(mgr.station_state_ecef)

# Testing fixed station position scenarios

a_priori_covariance = np.diag([1, 1, 1, 1, 1, 1, 1E2, 1E6, 1E6, 1, 1, 1, 1, 1, 1, 1, 1, 1])  # No fixed station position covariance

non_fixed_batch_estimated_initial_state, non_fixed_estimated_covariance, non_fixed_batch_residuals_df = batch_estimator.estimate_initial_state(
    a_priori_state=initial_state_estimate,
    a_priori_covariance=a_priori_covariance,
    measurement_data=measurements,
    R=R,
    max_iterations=3,
    tol=1E-6,
    considered_measurements='All')

non_fixed_lkf_state_history, non_fixed_lkf_covariance_history, non_fixed_lkf_residuals_df = lkf.run(initial_state_estimate,
                                                                        np.zeros_like(initial_state_estimate),
                                                                        a_priori_covariance, measurements,
                                                                        R=R, max_iterations=1,
                                                                        convergence_threshold=1e-9,
                                                                        considered_measurements='All')

# Reset measurement managers positions
for i, mgr in enumerate(batch_estimator.measurement_mgrs):
    mgr.station_state_ecef[0:3] = station_positions_ecef[i]
    mgr.lat, mgr.lon = mgr.coordinate_mgr.ECEF_to_GCS(mgr.station_state_ecef)

for i, mgr in enumerate(lkf.measurement_mgrs):
    mgr.station_state_ecef[0:3] = station_positions_ecef[i]
    mgr.lat, mgr.lon = mgr.coordinate_mgr.ECEF_to_GCS(mgr.station_state_ecef)

a_prior_covariance = np.diag([1, 1, 1, 1, 1, 1, 1E2, 1E6, 1E6, 1, 1, 1, 1, 1, 1, 1E-16, 1E-16, 1E-16])  # Given a priori covariance

fixed_station_3_batch_estimated_initial_state, fixed_station_3_estimated_covariance, fixed_station_3_batch_residuals_df = batch_estimator.estimate_initial_state(
    a_priori_state=initial_state_estimate,
    a_priori_covariance=a_prior_covariance,
    measurement_data=measurements,
    R=R,
    max_iterations=3,
    tol=1E-6,
    considered_measurements='All')

fixed_station_3_lkf_state_history, fixed_station_3_lkf_covariance_history, fixed_station_3_lkf_residuals_df = lkf.run(initial_state_estimate,
                                                                        np.zeros_like(initial_state_estimate),
                                                                        a_prior_covariance, measurements,
                                                                        R=R, max_iterations=1,
                                                                        convergence_threshold=1e-9,
                                                                        considered_measurements='All')

# Reset measurement managers positions
for i, mgr in enumerate(batch_estimator.measurement_mgrs):
    mgr.station_state_ecef[0:3] = station_positions_ecef[i]
    mgr.lat, mgr.lon = mgr.coordinate_mgr.ECEF_to_GCS(mgr.station_state_ecef)

for i, mgr in enumerate(lkf.measurement_mgrs):
    mgr.station_state_ecef[0:3] = station_positions_ecef[i]
    mgr.lat, mgr.lon = mgr.coordinate_mgr.ECEF_to_GCS(mgr.station_state_ecef)

# Update measurement managers to new estimated initial state for covariance propagation
for i, mgr in enumerate(station_mgr_list):
    new_station_position = estimated_initial_state[9+3*i:12+3*i]
    mgr.station_state_ecef[0:3] = new_station_position
    mgr.lat, mgr.lon = mgr.coordinate_mgr.ECEF_to_GCS(new_station_position)

# Integrate batch estimated initial state forward for comparison
[_, batch_estimated_state_history] = integrator.integrate_stm(time_vector[-1], estimated_initial_state, teval=time_vector)
[_, range_batch_estimated_state_history] = integrator.integrate_stm(time_vector[-1], range_estimated_initial_state, teval=time_vector)
[_, range_rate_batch_estimated_state_history] = integrator.integrate_stm(time_vector[-1], range_rate_estimated_initial_state, teval=time_vector)
[_, non_fixed_batch_estimated_state_history] = integrator.integrate_stm(time_vector[-1], non_fixed_batch_estimated_initial_state, teval=time_vector)
[_, fixed_station_3_batch_estimated_state_history] = integrator.integrate_stm(time_vector[-1], fixed_station_3_batch_estimated_initial_state, teval=time_vector)

covariance_history = np.zeros((18, 18, len(time_vector)))
for i, time in enumerate(time_vector):
    stm = batch_estimated_state_history[18:,i].reshape((18,18))
    covariance_history[:,:,i] = stm @ estimated_covariance @ stm.T

# PLOTTING -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Plot residual time history for each station
colors_list = ['red', 'green', 'blue']
residual_df_list = [batch_residuals_df,
                    lkf_residuals_df,
                    range_batch_residuals_df,
                    range_lkf_residuals_df,
                    range_rate_batch_residuals_df,
                    range_rate_lkf_residuals_df,
                    non_fixed_batch_residuals_df,
                    non_fixed_lkf_residuals_df,
                    fixed_station_3_batch_residuals_df,
                    fixed_station_3_lkf_residuals_df]
filter_names = ['Batch Filter', 'LKF', 'Batch Filter (Range Only)', 'LKF (Range Only)', 'Batch Filter (Range Rate Only)', 'LKF (Range Rate Only)', 'Batch Filter (No-Fixed Stations)', 'LKF (No-Fixed Stations)', 'Batch Filter (Station 3 Fixed)', 'LKF (Station 3 Fixed)']
for residuals_df, filter_name in zip(residual_df_list, filter_names):
    for iteration in range(residuals_df['iteration'].max()+1):
        # Combine station residuals into a single vector for RMS calculation, this can be done by adding all the station residuals together for the given iteration (since none overlap in timing)
        relevant_residuals = residuals_df[residuals_df['iteration'] == iteration]['pre-fit'].values.copy()
        for i in range(len(residuals_df['station'].unique())):
            # Set any NaN values to zero for RMS calculation
            relevant_residuals[i][np.isnan(relevant_residuals[i])] = 0.0
        
        # Sum the residuals across stations to get a single residual vector for the iteration
        combined_residuals = np.sum(relevant_residuals, axis=0)

        # Reset zeros to NaN so they aren't included in RMS calculation
        combined_residuals[combined_residuals == 0.0] = np.nan
        
        # Compute RMS of combined residuals for the iteration
        rms_range_residual = np.sqrt(np.abs(np.nanmean(combined_residuals[0,:]*10000 **2))) # Convert from km to cm for RMS calculation
        rms_range_rate_residual = np.sqrt(np.abs(np.nanmean(combined_residuals[1,:]*1E6 **2))) # Convert from km/s to mm/s for RMS calculation

        # Reset zeros to NaN in individual station residuals as well for accurate RMS calculation
        for i in range(len(residuals_df['station'].unique())):
            # Set any NaN values to zero for RMS calculation
            relevant_residuals[i][relevant_residuals[i] == 0.0] = np.nan

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=(f'Range Residuals (RMS = {rms_range_residual:.4f} cm)', f'Range Rate Residuals (RMS = {rms_range_rate_residual:4f} mm/s)'))
        for i, station_name in enumerate(residuals_df['station'].unique()):
            mask = (residuals_df['iteration'] == iteration) & (residuals_df['station'] == station_name)
            pre_fit_residuals = np.vstack(residuals_df[mask]['pre-fit'])
            fig.add_trace(go.Scatter(x=time_vector, y=pre_fit_residuals[0,:]*100000, mode='markers', name=f'{station_name}', marker=dict(color=colors_list[i])), row=1, col=1)
            fig.add_trace(go.Scatter(x=time_vector, y=pre_fit_residuals[1,:]*1E6, mode='markers', name=f'{station_name}', marker=dict(color=colors_list[i]), showlegend=False), row=2, col=1)
        fig.update_traces(marker=dict(size=4))
        fig.update_xaxes(title_text="Time (s)")
        fig.update_yaxes(title_text="Range Residuals (cm)", showexponent="all", exponentformat="e", row=1, col=1)
        fig.update_yaxes(title_text="Range Rate Residuals (mm/s)", showexponent="all", exponentformat="e", row=2, col=1)
        fig.update_layout(title_text=f"{filter_name} Pre-Fit Residuals at Iteration {iteration+1}",
                        title_font=dict(size=28),
                        width=1200,
                        height=800,
                        legend=dict(font=dict(size=18),
                                    yanchor="top",
                                    y=1.2,
                                    xanchor="left",
                                    x=0.87))
        fig.write_html(f"ASEN_6080/Project1/figures/{filter_name.lower().replace(' ','_')}_pre_fit_residuals_iteration_{iteration+1}.html")

    for iteration in range(residuals_df['iteration'].max()+1):
        # Combine station residuals into a single vector for RMS calculation, this can be done by adding all the station residuals together for the given iteration (since none overlap in timing)
        relevant_residuals = residuals_df[residuals_df['iteration'] == iteration]['pre-fit'].values.copy()
        for i in range(len(residuals_df['station'].unique())):
            # Set any NaN values to zero for RMS calculation
            relevant_residuals[i][np.isnan(relevant_residuals[i])] = 0.0
        
        # Sum the residuals across stations to get a single residual vector for the iteration
        combined_residuals = np.sum(relevant_residuals, axis=0)

        # Reset zeros to NaN so they aren't included in RMS calculation
        combined_residuals[combined_residuals == 0.0] = np.nan
        
        # Compute RMS of combined residuals for the iteration
        rms_range_residual = np.sqrt(np.abs(np.nanmean(combined_residuals[0,:]*10000 **2))) # Convert from km to cm for RMS calculation
        rms_range_rate_residual = np.sqrt(np.abs(np.nanmean(combined_residuals[1,:]*1E6 **2))) # Convert from km/s to mm/s for RMS calculation

        # Reset zeros to NaN in individual station residuals as well for accurate RMS calculation
        for i in range(len(residuals_df['station'].unique())):
            # Set any NaN values to zero for RMS calculation
            relevant_residuals[i][relevant_residuals[i] == 0.0] = np.nan

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=(f'Range Residuals (RMS = {rms_range_residual:.4f} cm)', f'Range Rate Residuals (RMS = {rms_range_rate_residual:4f} mm/s)'))
        for i, station_name in enumerate(residuals_df['station'].unique()):
            mask = (residuals_df['iteration'] == iteration) & (residuals_df['station'] == station_name)

            post_fit_residuals = np.vstack(residuals_df[mask]['post-fit'])
            fig.add_trace(go.Scatter(x=time_vector, y=post_fit_residuals[0,:]*100000, mode='markers', name=f'{station_name}', marker=dict(color=colors_list[i])), row=1, col=1)
            fig.add_trace(go.Scatter(x=time_vector, y=post_fit_residuals[1,:]*1E6, mode='markers', name=f'{station_name}', marker=dict(color=colors_list[i]), showlegend=False), row=2, col=1)
        fig.update_traces(marker=dict(size=4))
        fig.update_xaxes(title_text="Time (s)")
        fig.update_yaxes(title_text="Range Residuals (cm)", showexponent="all", exponentformat="e", row=1, col=1)
        fig.update_yaxes(title_text="Range Rate Residuals (mm/s)", showexponent="all", exponentformat="e", row=2, col=1)
        fig.update_layout(title_text=f"{filter_name} Post-Fit Residuals at Iteration {iteration+1}",
                        title_font=dict(size=28),
                        width=1200,
                        height=800,
                        legend=dict(font=dict(size=18),
                                    yanchor="top",
                                    y=1.2,
                                    xanchor="left",
                                    x=0.87))
        fig.write_html(f"ASEN_6080/Project1/figures/{filter_name.lower().replace(' ','_')}_post_fit_residuals_iteration_{iteration+1}.html")

# Plot state history difference for batch LLS and LKF
state_labels = ['x (km)', 'y (km)', 'z (km)', 'vx (km/s)', 'vy (km/s)', 'vz (km/s)', 'mu (km^3/s^2)', 'J2', 'C_d', 'Station 1 x (km)', 'Station 1 y (km)', 'Station 1 z (km)', 'Station 2 x (km)', 'Station 2 y (km)', 'Station 2 z (km)', 'Station 3 x (km)', 'Station 3 y (km)', 'Station 3 z (km)']
file_labels = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'mu', 'J2', 'C_d', 'station_1_x', 'station_1_y', 'station_1_z', 'station_2_x', 'station_2_y', 'station_2_z', 'station_3_x', 'station_3_y', 'station_3_z']
time_history_list = [batch_estimated_state_history, lkf_state_history, range_batch_estimated_state_history, range_lkf_state_history, range_rate_batch_estimated_state_history, range_rate_lkf_state_history, non_fixed_batch_estimated_state_history, non_fixed_lkf_state_history, fixed_station_3_batch_estimated_state_history, fixed_station_3_lkf_state_history]
for state_history, filter_name in zip(time_history_list, filter_names):
    for i in range(6):
        fig = make_subplots(rows = 3, cols=1, shared_xaxes=True, subplot_titles=(f'{state_labels[3*i]} Difference', f'{state_labels[3*i+1]} Difference', f'{state_labels[3*i+2]} Difference'))
        for j in range(3):
            diff = state_history[3*i+j,:] - batch_estimated_state_history[3*i+j,:]
            fig.add_trace(go.Scatter(x=time_vector, y=diff, mode='lines', name=f'{state_labels[3*i+j]} Difference'), row=j+1, col=1)
            fig.update_yaxes(title_text=f'{state_labels[3*i+j]} Difference', showexponent="all", exponentformat="e", row=j+1, col=1)

        fig.update_layout(title=f"Difference in {state_labels[3*i]}, {state_labels[3*i+1]}, and {state_labels[3*i+2]} Between Base Batch and {filter_name}",
                        xaxis_title='Time (s)',
                        title_font=dict(size=28))
        fig.update_yaxes(showexponent="all", exponentformat="e")
        fig.write_html(f"ASEN_6080/Project1/figures/{filter_name}_states_{3*i}_{3*i+2}_time_histories.html")

# Plot trace of satellite state covariance using log scale for better visualization
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=('Position Covariance Trace', 'Velocity Covariance Trace'))
trace_pos = np.trace(lkf_covariance_history[:3,:3,:])
trace_vel = np.trace(lkf_covariance_history[3:6,3:6,:])
fig.add_trace(go.Scatter(x=time_vector, y=trace_pos, mode='lines', name='Satellite Position'), row=1, col=1)
fig.add_trace(go.Scatter(x=time_vector, y=trace_vel, mode='lines', name='Satellite Velocity'), row=2, col=1)
fig.update_yaxes(type="log", showexponent="all", exponentformat="e", title_text=f'Covariance Trace (km^2)', row=1, col=1)
fig.update_yaxes(type="log", showexponent="all", exponentformat="e", title_text=f'Covariance Trace (km^2/s^2)', row=2, col=1)
    
fig.update_layout(title=f"Covariance for Satellite States Over Time",
                    xaxis_title='Time (s)',
                    title_font=dict(size=28),
                    width=1200,
                    height=800,
                    legend=dict(font=dict(size=18),
                                yanchor="top",
                                y=1.2,
                                xanchor="left",
                                x=0.87))
fig.update_yaxes(showexponent="all", exponentformat="e")
fig.write_html(f"ASEN_6080/Project1/figures/covariance_traces.html")

# Plot covariance ellipse for satellite position at final time step
batch_center = batch_estimated_state_history[:6,-1]
lkf_center = lkf_state_history[:6,-1]
center_diff = batch_center - lkf_center
batch_pos_covariance_ellipse = covariance_ellipse(np.zeros(3), covariance_history[:3,:3,-1])
lkf_pos_covariance_ellipse = covariance_ellipse(center_diff[:3], lkf_covariance_history[:3,:3,-1])

# Plot 3D ellipses
fig = go.Figure()
fig.add_trace(go.Scatter3d(x=batch_pos_covariance_ellipse[:,0], y=batch_pos_covariance_ellipse[:,1], z=batch_pos_covariance_ellipse[:,2], mode='markers', name='Batch LLS Position Covariance Ellipse'))
fig.add_trace(go.Scatter3d(x=lkf_pos_covariance_ellipse[:,0], y=lkf_pos_covariance_ellipse[:,1], z=lkf_pos_covariance_ellipse[:,2], mode='markers', name='LKF Position Covariance Ellipse'))
fig.update_layout(title=f"Satellite Position Covariance Ellipse",
                    title_font=dict(size=28),
                    width=1200,
                    height=800,
                    legend=dict(font=dict(size=18),
                                yanchor="top",
                                y=1.2,
                                xanchor="left",
                                x=0.87),
                    scene=dict(xaxis_title='X Position (km)',
                               yaxis_title='Y Position (km)',
                               zaxis_title='Z Position (km)',
                               xaxis=dict(showexponent="all", exponentformat="e"),
                               yaxis=dict(showexponent="all", exponentformat="e"),
                               zaxis=dict(showexponent="all", exponentformat="e")))

fig.write_html(f"ASEN_6080/Project1/figures/position_covariance_ellipses.html")

# Plot covariance ellipse for satellite velocity at final time step
batch_vel_covariance_ellipse = covariance_ellipse(np.zeros(3), covariance_history[3:6,3:6,-1])
lkf_vel_covariance_ellipse = covariance_ellipse(center_diff[3:6], lkf_covariance_history[3:6,3:6,-1])

# Plot 3D ellipses
fig = go.Figure()
fig.add_trace(go.Scatter3d(x=batch_vel_covariance_ellipse[:,0], y=batch_vel_covariance_ellipse[:,1], z=batch_vel_covariance_ellipse[:,2], mode='markers', name='Batch LLS Velocity Covariance Ellipse'))
fig.add_trace(go.Scatter3d(x=lkf_vel_covariance_ellipse[:,0], y=lkf_vel_covariance_ellipse[:,1], z=lkf_vel_covariance_ellipse[:,2], mode='markers', name='LKF Velocity Covariance Ellipse'))
fig.update_layout(title=f"Satellite Velocity Covariance Ellipse",
                    title_font=dict(size=28),
                    width=1200,
                    height=800,
                    legend=dict(font=dict(size=18),
                                yanchor="top",
                                y=1.2,
                                xanchor="left",
                                x=0.5),
                    scene=dict(xaxis_title='X Velocity (km/s)',
                               yaxis_title='Y Velocity (km/s)',
                               zaxis_title='Z Velocity (km/s)',
                               xaxis=dict(showexponent="all", exponentformat="e"),
                               yaxis=dict(showexponent="all", exponentformat="e"),
                               zaxis=dict(showexponent="all", exponentformat="e")))

fig.write_html(f"ASEN_6080/Project1/figures/velocity_covariance_ellipses.html")

# Plot covariance ellipse to show difference between analyzing range and range rate
range_center = range_lkf_state_history[:6,-1]
range_rate_center = range_rate_lkf_state_history[:6,-1]
center_diff = range_center - range_rate_center
range_pos_covariance_ellipse = covariance_ellipse(np.zeros(3), range_lkf_covariance_history[:3,:3,-1])
range_rate_pos_covariance_ellipse = covariance_ellipse(center_diff[:3], range_rate_lkf_covariance_history[:3,:3,-1])

fig = go.Figure()
fig.add_trace(go.Scatter3d(x=range_pos_covariance_ellipse[:,0], y=range_pos_covariance_ellipse[:,1], z=range_pos_covariance_ellipse[:,2], mode='markers', name='Range Only'))
fig.add_trace(go.Scatter3d(x=range_rate_pos_covariance_ellipse[:,0], y=range_rate_pos_covariance_ellipse[:,1], z=range_rate_pos_covariance_ellipse[:,2], mode='markers', name='Range Rate Only'))
fig.update_layout(title=f"Position Covariance Ellipses from Analyzing Only Range or Range Rate Measurements",
                    title_font=dict(size=28),
                    width=1200,
                    height=800,
                    legend=dict(font=dict(size=18)),
                    scene=dict(xaxis_title='X Position (km)',
                               yaxis_title='Y Position (km)',
                               zaxis_title='Z Position (km)',
                               xaxis=dict(showexponent="all", exponentformat="e"),
                               yaxis=dict(showexponent="all", exponentformat="e"),
                               zaxis=dict(showexponent="all", exponentformat="e")))
fig.write_html(f"ASEN_6080/Project1/figures/range_vs_range_rate_position_covariance_ellipses.html")

range_vel_covariance_ellipse = covariance_ellipse(np.zeros(3), range_lkf_covariance_history[3:6,3:6,-1])
range_rate_vel_covariance_ellipse = covariance_ellipse(center_diff[3:6], range_rate_lkf_covariance_history[3:6,3:6,-1])

fig = go.Figure()
fig.add_trace(go.Scatter3d(x=range_vel_covariance_ellipse[:,0], y=range_vel_covariance_ellipse[:,1], z=range_vel_covariance_ellipse[:,2], mode='markers', name='Range Only'))
fig.add_trace(go.Scatter3d(x=range_rate_vel_covariance_ellipse[:,0], y=range_rate_vel_covariance_ellipse[:,1], z=range_rate_vel_covariance_ellipse[:,2], mode='markers', name='Range Rate Only'))
fig.update_layout(title=f"Velocity Covariance Ellipses from Analyzing Only Range or Range Rate Measurements",
                    title_font=dict(size=28),
                    width=1200,
                    height=800,
                    legend=dict(font=dict(size=18)),
                    scene=dict(xaxis_title='X Velocity (km/s)',
                               yaxis_title='Y Velocity (km/s)',
                               zaxis_title='Z Velocity (km/s)',
                               xaxis=dict(showexponent="all", exponentformat="e"),
                               yaxis=dict(showexponent="all", exponentformat="e"),
                               zaxis=dict(showexponent="all", exponentformat="e")))
fig.write_html(f"ASEN_6080/Project1/figures/range_vs_range_rate_velocity_covariance_ellipses.html")
