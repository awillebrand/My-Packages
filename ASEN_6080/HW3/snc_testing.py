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
P_0 = np.diag([1, 1, 1, 1e-3, 1e-3, 1e-3, 0])**2

sigma_values = [1e-18, 1e-16, 1e-14, 1e-12, 1e-10, 1e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5]

lkf = LKF(integrator, station_mgr_list, initial_earth_spin_angle=np.deg2rad(122))
ekf = EKF(integrator, station_mgr_list, initial_earth_spin_angle=np.deg2rad(122))


lkf_residual_rms_results = np.zeros((len(sigma_values), 2))  # Columns for range and range rate RMS
ekf_residual_rms_results = np.zeros((len(sigma_values), 2))  # Columns for range and range rate RMS

lkf_rms_position_error_3D_results = np.zeros(len(sigma_values))
ekf_rms_position_error_3D_results = np.zeros(len(sigma_values))
lkf_rms_velocity_error_3D_results = np.zeros(len(sigma_values))
ekf_rms_velocity_error_3D_results = np.zeros(len(sigma_values))

for sigma in sigma_values:
    print(f"Running LKF with SNC process noise approach and sigma = {sigma:.1e} km/s^2...")
    Q = np.diag([sigma, sigma, sigma])**2
    lkf_state_history, lkf_covariance_history, lkf_residuals_df = lkf.run(initial_state_guess, np.zeros(7), P_0, measurement_data, R=np.diag(noise_var), max_iterations=3, process_noise_approach='SNC', Q=Q)
    ekf_state_history, ekf_covariance_history, ekf_residuals_df = ekf.run(initial_state_guess, np.zeros(7), P_0, measurement_data, R=np.diag(noise_var), start_mode='warm', start_length=1000, process_noise_approach='SNC', Q=Q)

    residual_df_list = [lkf_residuals_df, ekf_residuals_df]
    filter_names = ['LKF with SNC', 'EKF with SNC']
    for residuals_df, filter_name in zip(residual_df_list, filter_names):
        if filter_name == 'LKF with SNC':
            iteration = 2  # LKF only runs for 3 iterations, so we can only analyze up to iteration 3
        else:
             iteration = 0
        
        relevant_residuals = residuals_df[residuals_df['iteration'] == iteration]['post-fit'].values.copy()
        for i in range(len(residuals_df['station'].unique())):
            # Set any NaN values to zero for RMS calculation
            relevant_residuals[i][np.isnan(relevant_residuals[i])] = 0.0

        # Sum the residuals across stations to get a single residual vector for the iteration
        combined_residuals = np.sum(relevant_residuals, axis=0)

        # Reset zeros to NaN so they aren't included in RMS calculation
        combined_residuals[combined_residuals == 0.0] = np.nan
        
        # Compute RMS of combined residuals for the iteration
        rms_range_residual = np.sqrt(np.abs(np.nanmean((combined_residuals[0,:]*1E5) **2))) # Convert from km to cm for RMS calculation
        rms_range_rate_residual = np.sqrt(np.abs(np.nanmean((combined_residuals[1,:]*1E6) **2))) # Convert from km/s to mm/s for RMS calculation

        if filter_name == 'LKF with SNC':
            lkf_residual_rms_results[sigma_values.index(sigma), 0] = rms_range_residual
            lkf_residual_rms_results[sigma_values.index(sigma), 1] = rms_range_rate_residual
        else:
            ekf_residual_rms_results[sigma_values.index(sigma), 0] = rms_range_residual
            ekf_residual_rms_results[sigma_values.index(sigma), 1] = rms_range_rate_residual
        
        # Compute 3D rms error of state estimate at final iteration compared to truth
        lkf_state_errors = np.zeros_like(lkf_state_history)  # Initialize state error array
        ekf_state_errors = np.zeros_like(lkf_state_history)  # Initialize state error array
        for k in range(lkf_state_history.shape[1]):
            lkf_state_errors[:,k] = lkf_state_history[:,k] - truth_data['augmented_state_history'].values[k][0:7]
            ekf_state_errors[:,k] = ekf_state_history[:,k] - truth_data['augmented_state_history'].values[k][0:7]

        lkf_rms_position_error_3D = np.sqrt(np.mean(np.sum(lkf_state_errors[0:3,:]**2, axis=0))) * 1000  # in meters
        ekf_rms_position_error_3D = np.sqrt(np.mean(np.sum(ekf_state_errors[0:3,:]**2, axis=0))) * 1000  # in meters
        lkf_rms_velocity_error_3D = np.sqrt(np.mean(np.sum(lkf_state_errors[3:6,:]**2, axis=0))) * 1e6  # in mm/s
        ekf_rms_velocity_error_3D = np.sqrt(np.mean(np.sum(ekf_state_errors[3:6,:]**2, axis=0))) * 1e6  # in mm/s

        lkf_rms_position_error_3D_results[sigma_values.index(sigma)] = lkf_rms_position_error_3D
        ekf_rms_position_error_3D_results[sigma_values.index(sigma)] = ekf_rms_position_error_3D
        lkf_rms_velocity_error_3D_results[sigma_values.index(sigma)] = lkf_rms_velocity_error_3D
        ekf_rms_velocity_error_3D_results[sigma_values.index(sigma)] = ekf_rms_velocity_error_3D

# Plot RMS of range and range rate residuals for LKF and EKF with SNC approach
fig = go.Figure()
fig.add_trace(go.Scatter(x=sigma_values, y=lkf_residual_rms_results[:,0], mode='markers+lines', name='LKF Range Residual RMS (SNC)'))
fig.add_trace(go.Scatter(x=sigma_values, y=lkf_residual_rms_results[:,1], mode='markers+lines', name='LKF Range Rate Residual RMS (SNC)'))
fig.add_trace(go.Scatter(x=sigma_values, y=ekf_residual_rms_results[:,0], mode='markers+lines', name='EKF Range Residual RMS (SNC)'))
fig.add_trace(go.Scatter(x=sigma_values, y=ekf_residual_rms_results[:,1], mode='markers+lines', name='EKF Range Rate Residual RMS (SNC)'))
fig.update_xaxes(type='log', title_text='Sigma Value (km/s^2)')
fig.update_yaxes(title_text='RMS of Post-Fit Residuals (cm for range, mm/s for range rate)', type='log')
fig.update_layout(title='RMS of Post-Fit Residuals vs Sigma for LKF and EKF with SNC Approach')
fig.write_html("ASEN_6080/HW3/figures/residual_rms_vs_sigma.html")
fig.show()

# Plot RMS of 3D position and velocity errors for LKF and EKF with SNC approach
fig = go.Figure()
fig.add_trace(go.Scatter(x=sigma_values, y=lkf_rms_position_error_3D_results, mode='markers+lines', name='LKF 3D Position Error RMS (SNC)'))
fig.add_trace(go.Scatter(x=sigma_values, y=lkf_rms_velocity_error_3D_results, mode='markers+lines', name='LKF 3D Velocity Error RMS (SNC)'))
fig.add_trace(go.Scatter(x=sigma_values, y=ekf_rms_position_error_3D_results, mode='markers+lines', name='EKF 3D Position Error RMS (SNC)'))
fig.add_trace(go.Scatter(x=sigma_values, y=ekf_rms_velocity_error_3D_results, mode='markers+lines', name='EKF 3D Velocity Error RMS (SNC)'))
fig.update_xaxes(type='log', title_text='Sigma Value (km/s^2)')
fig.update_yaxes(title_text='RMS of 3D State Estimation Error (m for position, mm/s for velocity)', type='log')
fig.update_layout(title='RMS of 3D State Estimation Error vs Sigma for LKF and EKF with SNC Approach')
fig.write_html("ASEN_6080/HW3/figures/state_error_rms_vs_sigma.html")
fig.show()