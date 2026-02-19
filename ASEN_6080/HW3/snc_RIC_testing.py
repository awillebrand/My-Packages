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
initial_state_guess = truth_data['initial_state'].values[0][0:7]
P_0 = np.diag([1, 1, 1, 1e-3, 1e-3, 1e-3, 0])**2

sigma_values = [1e-18, 1e-16, 1e-14, 1e-12, 1e-11, 5e-11, 1e-10, 5e-10, 1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5]

lkf = LKF(integrator, station_mgr_list, initial_earth_spin_angle=np.deg2rad(122))
ekf = EKF(integrator, station_mgr_list, initial_earth_spin_angle=np.deg2rad(122))

lkf_residual_rms_results = np.zeros((len(sigma_values), 2))  # Columns for range and range rate RMS
ekf_residual_rms_results = np.zeros((len(sigma_values), 2))  # Columns for range and range rate RMS

lkf_rms_position_error_3D_results = np.zeros(len(sigma_values))
ekf_rms_position_error_3D_results = np.zeros(len(sigma_values))
lkf_rms_velocity_error_3D_results = np.zeros(len(sigma_values))
ekf_rms_velocity_error_3D_results = np.zeros(len(sigma_values))

frame_list = ['ECI', 'ECI', 'RIC', 'RIC']
optimal_sigma = 5e-8
Q = np.diag([optimal_sigma, optimal_sigma, optimal_sigma*0.5])**2

lkf_state_history_eci, lkf_covariance_history_eci, lkf_residuals_df_eci = lkf.run(initial_state_guess, np.zeros(7), P_0, measurement_data, R=np.diag(noise_var), max_iterations=1, process_noise_approach='SNC', Q=Q, Q_frame='ECI')
ekf_state_history_eci, ekf_covariance_history_eci, ekf_residuals_df_eci = ekf.run(initial_state_guess, np.zeros(7), P_0, measurement_data, R=np.diag(noise_var), start_mode='warm', start_length=1000, process_noise_approach='SNC', Q=Q, Q_frame='ECI')
lkf_state_history_ric, lkf_covariance_history_ric, lkf_residuals_df_ric = lkf.run(initial_state_guess, np.zeros(7), P_0, measurement_data, R=np.diag(noise_var), max_iterations=1, process_noise_approach='SNC', Q=Q, Q_frame='RIC')
ekf_state_history_ric, ekf_covariance_history_ric, ekf_residuals_df_ric = ekf.run(initial_state_guess, np.zeros(7), P_0, measurement_data, R=np.diag(noise_var), start_mode='warm', start_length=1000, process_noise_approach='SNC', Q=Q, Q_frame='RIC')

residual_df_list = [lkf_residuals_df_eci, ekf_residuals_df_eci, lkf_residuals_df_ric, ekf_residuals_df_ric]
filter_names = ['LKF with SNC (ECI)', 'EKF with SNC (ECI)', 'LKF with SNC (RIC)', 'EKF with SNC (RIC)']
colors_list = ['red', 'green', 'blue']

for residuals_df, filter_name, frame in zip(residual_df_list, filter_names, frame_list):
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
        rms_range_residual = np.sqrt(np.abs(np.nanmean((combined_residuals[0,:]*1E5) **2))) # Convert from km to cm for RMS calculation
        rms_range_rate_residual = np.sqrt(np.abs(np.nanmean((combined_residuals[1,:]*1E6) **2))) # Convert from km/s to mm/s for RMS calculation

        # Find mean and standard deviation of range and range rate residuals for the iteration
        mean_range_residual = np.nanmean(combined_residuals[0,:]*1E3) # Convert from km to m for mean calculation
        std_range_residual = np.nanstd(combined_residuals[0,:]*1E3) # Convert from km to m for std calculation
        mean_range_rate_residual = np.nanmean(combined_residuals[1,:]*1E6) # Convert from km/s to mm/s for mean calculation
        std_range_rate_residual = np.nanstd(combined_residuals[1,:]*1E6) # Convert from km/s to mm/s for std calculation
        print(f"Pre-fit {filter_name} Iteration {iteration+1}:")
        print(f"Range Residuals: Mean = {mean_range_residual:.4f} m, Std Dev = {std_range_residual:.4f} m, RMS = {rms_range_residual:.4f} m")
        print(f"Range Rate Residuals: Mean = {mean_range_rate_residual:.4f} mm/s, Std Dev = {std_range_rate_residual:.4f} mm/s, RMS = {rms_range_rate_residual:.4f} mm/s")
        print("--------------------------------------------------")

        # Reset zeros to NaN in individual station residuals as well for accurate RMS calculation
        for i in range(len(residuals_df['station'].unique())):
            # Set any NaN values to zero for RMS calculation
            relevant_residuals[i][relevant_residuals[i] == 0.0] = np.nan

        
        fig = make_subplots(
            rows=2, cols=2, 
            shared_xaxes=False,
            column_widths=[0.85, 0.15],
            horizontal_spacing=0.06,
            subplot_titles=(f'Range Residuals (Mean = {mean_range_residual:.4f} m, Std Dev = {std_range_residual:.4f} m, RMS = {rms_range_residual:.4f} m)', 'Distribution',
                            f'Range Rate Residuals (Mean = {mean_range_rate_residual:.4f} mm/s, Std Dev = {std_range_rate_residual:.4f} mm/s, RMS = {rms_range_rate_residual:.4f} mm/s)', 'Distribution')
        )
        
        # Collect all residuals for histogram
        all_range_residuals = []
        all_range_rate_residuals = []
        
        for i, station_name in enumerate(residuals_df['station'].unique()):
            mask = (residuals_df['iteration'] == iteration) & (residuals_df['station'] == station_name)
            pre_fit_residuals = np.vstack(residuals_df[mask]['pre-fit'])
            
            # Add scatter plots (left column)
            fig.add_trace(go.Scatter(x=time_vector, y=pre_fit_residuals[0,:]*1E3, 
                                    mode='markers', name=f'{station_name}', 
                                    marker=dict(color=colors_list[i]), legendgroup=f'group{i}'), 
                         row=1, col=1)
            fig.add_trace(go.Scatter(x=time_vector, y=pre_fit_residuals[1,:]*1E6, 
                                    mode='markers', name=f'{station_name}', 
                                    marker=dict(color=colors_list[i]), 
                                    showlegend=False, legendgroup=f'group{i}'), 
                         row=2, col=1)
            
            # Collect valid (non-NaN) residuals for histograms
            valid_range = pre_fit_residuals[0,:][~np.isnan(pre_fit_residuals[0,:])] * 1E3
            valid_range_rate = pre_fit_residuals[1,:][~np.isnan(pre_fit_residuals[1,:])] * 1E6
            all_range_residuals.extend(valid_range)
            all_range_rate_residuals.extend(valid_range_rate)
        
        # Add histograms (right column) - rotated to be vertical
        fig.add_trace(go.Histogram(y=all_range_residuals, 
                                  marker=dict(color='lightblue'),
                                  showlegend=False,
                                  nbinsy=50), 
                     row=1, col=2)
        fig.add_trace(go.Histogram(y=all_range_rate_residuals, 
                                  marker=dict(color='lightcoral'),
                                  showlegend=False,
                                  nbinsy=50), 
                     row=2, col=2)
        
        fig.update_traces(marker=dict(size=4), selector=dict(mode='markers'))
        fig.update_xaxes(title_text="Time (s)", tickfont=dict(size=20), title_font=dict(size=22), row=1, col=1)
        fig.update_xaxes(title_text="Time (s)", tickfont=dict(size=20), title_font=dict(size=22), row=2, col=1)
        fig.update_xaxes(title_text="Count", tickfont=dict(size=20), title_font=dict(size=22), row=1, col=2)
        fig.update_xaxes(title_text="Count", tickfont=dict(size=20), title_font=dict(size=22), row=2, col=2)
        fig.update_yaxes(title_text="Range Residuals (m)", tickfont=dict(size=20), title_font=dict(size=22), showexponent="all", exponentformat="e", row=1, col=1)
        fig.update_yaxes(title_text="Range Rate Residuals (mm/s)", tickfont=dict(size=20), title_font=dict(size=22), showexponent="all", exponentformat="e", row=2, col=1)
        fig.update_yaxes(showexponent="all", exponentformat="e", tickfont=dict(size=20), title_font=dict(size=22), row=1, col=2)
        fig.update_yaxes(showexponent="all", exponentformat="e", tickfont=dict(size=20), title_font=dict(size=22), row=2, col=2)
        fig.update_annotations(font=dict(size=24))
        fig.update_layout(title_text=f"{filter_name} Pre-Fit Residuals at Iteration {iteration+1}",
                        title_font=dict(size=30),
                        width=1900,  # Increased width to accommodate histograms
                        height=800,
                        legend=dict(font=dict(size=22),
                                    yanchor="top",
                                    y=1.13,
                                    xanchor="left",
                                    x=0.7,
                                    itemsizing='constant'))
        fig.write_html(f"ASEN_6080/HW3/figures/{filter_name.lower().replace(' ','_')}_pre_fit_residuals_iteration_{iteration+1}_{frame}.html")
        fig.write_image(f"ASEN_6080/HW3/figures/pngs/{filter_name.lower().replace(' ','_')}_pre_fit_residuals_iteration_{iteration+1}_{frame}.png")

        # Combine station residuals into a single vector for RMS calculation, this can be done by adding all the station residuals together for the given iteration (since none overlap in timing)
        relevant_residuals = residuals_df[residuals_df['iteration'] == iteration]['post-fit'].values.copy()
        for i in range(len(residuals_df['station'].unique())):
            # Set any NaN values to zero for RMS calculation
            relevant_residuals[i][np.isnan(relevant_residuals[i])] = 0.0
        
        # Sum the residuals across stations to get a single residual vector for the iteration
        combined_residuals = np.sum(relevant_residuals, axis=0)

        # Reset zeros to NaN so they aren't included in RMS calculation
        combined_residuals[combined_residuals == 0.0] = np.nan
        
        # Compute RMS of combined residuals for the iteration
        rms_range_residual = np.sqrt(np.abs(np.nanmean((combined_residuals[0,:]*1E3) **2))) # Convert from km to m for RMS calculation
        rms_range_rate_residual = np.sqrt(np.abs(np.nanmean((combined_residuals[1,:]*1E6) **2))) # Convert from km/s to mm/s for RMS calculation

        mean_range_residual = np.nanmean(combined_residuals[0,:]*1E3) # Convert from km to m for mean calculation
        std_range_residual = np.nanstd(combined_residuals[0,:]*1E3) # Convert from km to m for std calculation
        mean_range_rate_residual = np.nanmean(combined_residuals[1,:]*1E6) # Convert from km/s to mm/s for mean calculation
        std_range_rate_residual = np.nanstd(combined_residuals[1,:]*1E6) # Convert from km/s to mm/s for std calculation
        print(f"Post-Fit {filter_name} Iteration {iteration+1}:")
        print(f"Range Residuals: Mean = {mean_range_residual:.4f} m, Std Dev = {std_range_residual:.4f} m, RMS = {rms_range_residual:.4f} m")
        print(f"Range Rate Residuals: Mean = {mean_range_rate_residual:.4f} mm/s, Std Dev = {std_range_rate_residual:.4f} mm/s, RMS = {rms_range_rate_residual:.4f} mm/s")
        print("--------------------------------------------------")

        # Reset zeros to NaN in individual station residuals as well for accurate RMS calculation
        for i in range(len(residuals_df['station'].unique())):
            # Set any NaN values to zero for RMS calculation
            relevant_residuals[i][relevant_residuals[i] == 0.0] = np.nan

        fig = make_subplots(
            rows=2, cols=2, 
            shared_xaxes=False,
            column_widths=[0.85, 0.15],
            horizontal_spacing=0.06,
            subplot_titles=(f'Range Residuals (Mean = {mean_range_residual:.4f} m, Std Dev = {std_range_residual:.4f} m, RMS = {rms_range_residual:.4f} m)', 'Distribution',
                            f'Range Rate Residuals (Mean = {mean_range_rate_residual:.4f} mm/s, Std Dev = {std_range_rate_residual:.4f} mm/s, RMS = {rms_range_rate_residual:.4f} mm/s)', 'Distribution')
        )
        
        # Collect all residuals for histogram
        all_range_residuals = []
        all_range_rate_residuals = []
        
        for i, station_name in enumerate(residuals_df['station'].unique()):
            mask = (residuals_df['iteration'] == iteration) & (residuals_df['station'] == station_name)
            post_fit_residuals = np.vstack(residuals_df[mask]['post-fit'])
            
            # Add scatter plots (left column)
            fig.add_trace(go.Scatter(x=time_vector, y=post_fit_residuals[0,:]*1E3, 
                                    mode='markers', name=f'{station_name}', 
                                    marker=dict(color=colors_list[i]), legendgroup=f'group{i}'), 
                         row=1, col=1)
            fig.add_trace(go.Scatter(x=time_vector, y=post_fit_residuals[1,:]*1E6, 
                                    mode='markers', name=f'{station_name}', 
                                    marker=dict(color=colors_list[i]), 
                                    showlegend=False, legendgroup=f'group{i}'), 
                         row=2, col=1)
            
            # Collect valid (non-NaN) residuals for histograms
            valid_range = post_fit_residuals[0,:][~np.isnan(post_fit_residuals[0,:])] * 1E3
            valid_range_rate = post_fit_residuals[1,:][~np.isnan(post_fit_residuals[1,:])] * 1E6
            all_range_residuals.extend(valid_range)
            all_range_rate_residuals.extend(valid_range_rate)
        
        # Add histograms (right column) - rotated to be vertical
        fig.add_trace(go.Histogram(y=all_range_residuals, 
                                  marker=dict(color='lightblue'),
                                  showlegend=False,
                                  nbinsy=50), 
                     row=1, col=2)
        fig.add_trace(go.Histogram(y=all_range_rate_residuals, 
                                  marker=dict(color='lightcoral'),
                                  showlegend=False,
                                  nbinsy=50), 
                     row=2, col=2)
        
        fig.update_traces(marker=dict(size=4), selector=dict(mode='markers'))
        fig.update_xaxes(title_text="Time (s)", tickfont=dict(size=20), title_font=dict(size=22), row=1, col=1)
        fig.update_xaxes(title_text="Time (s)", tickfont=dict(size=20), title_font=dict(size=22), row=2, col=1)
        fig.update_xaxes(title_text="Count", tickfont=dict(size=20), title_font=dict(size=22), row=1, col=2)
        fig.update_xaxes(title_text="Count", tickfont=dict(size=20), title_font=dict(size=22), row=2, col=2)
        fig.update_yaxes(title_text="Range Residuals (m)", tickfont=dict(size=20), title_font=dict(size=22), showexponent="all", exponentformat="e", row=1, col=1)
        fig.update_yaxes(title_text="Range Rate Residuals (mm/s)", tickfont=dict(size=20), title_font=dict(size=22), showexponent="all", exponentformat="e", row=2, col=1)
        fig.update_yaxes(showexponent="all", exponentformat="e", tickfont=dict(size=20), title_font=dict(size=22), row=1, col=2)
        fig.update_yaxes(showexponent="all", exponentformat="e", tickfont=dict(size=20), title_font=dict(size=22), row=2, col=2)
        fig.update_annotations(font=dict(size=24))
        fig.update_layout(title_text=f"{filter_name} Post-Fit Residuals at Iteration {iteration+1}",
                        title_font=dict(size=30),
                        width=1900,  # Increased width to accommodate histograms
                        height=800,
                        legend=dict(font=dict(size=22),
                                    orientation="h",
                                    yanchor="top",
                                    y=1.13,
                                    xanchor="left",
                                    x=0.7,
                                    itemsizing='constant'))
        fig.write_html(f"ASEN_6080/HW3/figures/{filter_name.lower().replace(' ','_')}_post_fit_residuals_iteration_{iteration+1}_{frame}.html")
        fig.write_image(f"ASEN_6080/HW3/figures/pngs/{filter_name.lower().replace(' ','_')}_post_fit_residuals_iteration_{iteration+1}_{frame}.png")
        
# Plotting difference in time history of state errors for LKF and EKF with rotated process noise
state_history_list = [lkf_state_history_eci, ekf_state_history_eci, lkf_state_history_ric, ekf_state_history_ric]
covariance_history_list = [lkf_covariance_history_eci, ekf_covariance_history_eci, lkf_covariance_history_ric, ekf_covariance_history_ric]
filter_types = ['LKF', 'EKF']
for i in range(2):
    eci_history = state_history_list[i]
    ric_history = state_history_list[i+2]
    eci_cov_history = covariance_history_list[i]
    ric_cov_history = covariance_history_list[i+2]

    eci_state_errors = np.zeros_like(eci_history)
    ric_state_errors = np.zeros_like(ric_history)

    for k in range(eci_history.shape[1]):
        eci_state_errors[:,k] = eci_history[:,k] - truth_data['augmented_state_history'].values[k][0:7]
        ric_state_errors[:,k] = ric_history[:,k] - truth_data['augmented_state_history'].values[k][0:7]

    error_diff = ric_state_errors - eci_state_errors
    cov_diff = np.sqrt(np.abs(ric_cov_history)) - np.sqrt(np.abs(eci_cov_history))

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=('X Position Error Difference', 'Y Position Error Difference', 'Z Position Error Difference'))
    fig.add_trace(go.Scatter(x=time_vector, y=error_diff[0,:]*1000, mode='lines', name='Position Error Difference', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=time_vector, y=error_diff[1,:]*1000, mode='lines', name='Position Error Difference (SNC)', line=dict(color='blue'), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=time_vector, y=error_diff[2,:]*1000, mode='lines', name='Position Error Difference (SNC)', line=dict(color='blue'), showlegend=False), row=3, col=1)

    fig.add_trace(go.Scatter(x=time_vector, y=3*(cov_diff[0,0,:])*1000, mode='lines', name='3-Sigma Bound Difference', line=dict(color='red', dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=time_vector, y=3*(cov_diff[1,1,:])*1000, mode='lines', line=dict(color='red', dash='dash'), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=time_vector, y=3*(cov_diff[2,2,:])*1000, mode='lines', line=dict(color='red', dash='dash'), showlegend=False), row=3, col=1)

    fig.update_xaxes(title_text="Time (s)", tickfont=dict(size=20), title_font=dict(size=22))
    fig.update_yaxes(title_text="Error Difference (m)", tickfont=dict(size=20), title_font=dict(size=22), showexponent="all", exponentformat="e")
    fig.update_annotations(font=dict(size=24))
    fig.update_layout(title_text=f"Difference in {filter_types[i]} Position Errors between SNC Q in RIC vs ECI",
                        title_font=dict(size=30),
                        width=1800,
                        height=900,
                        legend=dict(font=dict(size=27),
                                    yanchor="top",
                                    y=1.15,
                                    xanchor="left",
                                    x=0.8,
                                    itemsizing='constant'))
    fig.write_html(f"ASEN_6080/HW3/figures/{filter_types[i].lower()}_position_error_difference_snc.html")
    fig.write_image(f"ASEN_6080/HW3/figures/pngs/{filter_types[i].lower()}_position_error_difference_snc.png")

    print(f"Average Position Error Difference (RIC - ECI) for {filter_types[i]} with SNC: {np.nanmean(error_diff[0:3,:]*1000):.4f} m")

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=('X Velocity Error Difference', 'Y Velocity Error Difference', 'Z Velocity Error Difference'))
    fig.add_trace(go.Scatter(x=time_vector, y=error_diff[3,:]*1E6, mode='lines', name='Velocity Error Difference', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=time_vector, y=error_diff[4,:]*1E6, mode='lines', name='Velocity Error Difference (SNC)', line=dict(color='blue'), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=time_vector, y=error_diff[5,:]*1E6, mode='lines', name='Velocity Error Difference (SNC)', line=dict(color='blue'), showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=time_vector, y=3*(cov_diff[3,3,:])*1E6, mode='lines', name='3-Sigma Bound Difference', line=dict(color='red', dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=time_vector, y=3*(cov_diff[4,4,:])*1E6, mode='lines', line=dict(color='red', dash='dash'), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=time_vector, y=3*(cov_diff[5,5,:])*1E6, mode='lines', line=dict(color='red', dash='dash'), showlegend=False), row=3, col=1)
    fig.update_xaxes(title_text="Time (s)", tickfont=dict(size=20), title_font=dict(size=22))
    fig.update_yaxes(title_text="Error Difference (mm/s)", tickfont=dict(size=20), title_font=dict(size=22), showexponent="all", exponentformat="e")
    fig.update_annotations(font=dict(size=24))
    fig.update_layout(title_text=f"Difference in {filter_types[i]} Velocity Errors between SNC Q in RIC vs ECI",
                        title_font=dict(size=30),
                        width=1800,
                        height=900,
                        legend=dict(font=dict(size=27),
                                    yanchor="top",
                                    y=1.15,
                                    xanchor="left",
                                    x=0.8,
                                    itemsizing='constant'))
    fig.write_html(f"ASEN_6080/HW3/figures/{filter_types[i].lower()}_velocity_error_difference_snc.html")
    fig.write_image(f"ASEN_6080/HW3/figures/pngs/{filter_types[i].lower()}_velocity_error_difference_snc.png")

    print(f"Average Velocity Error Difference (RIC - ECI) for {filter_types[i]} with SNC: {np.nanmean(error_diff[3:6,:]*1E6):.4f} mm/s")