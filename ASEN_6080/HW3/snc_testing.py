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

sigma_values = [1e-18, 1e-16, 1e-14, 1e-12, 1e-11, 5e-11, 1e-10, 5e-10, 1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5]

lkf = LKF(integrator, station_mgr_list, initial_earth_spin_angle=np.deg2rad(122))
ekf = EKF(integrator, station_mgr_list, initial_earth_spin_angle=np.deg2rad(122))

lkf_residual_rms_results = np.zeros((len(sigma_values), 2))  # Columns for range and range rate RMS
ekf_residual_rms_results = np.zeros((len(sigma_values), 2))  # Columns for range and range rate RMS

lkf_rms_position_error_3D_results = np.zeros(len(sigma_values))
ekf_rms_position_error_3D_results = np.zeros(len(sigma_values))
lkf_rms_velocity_error_3D_results = np.zeros(len(sigma_values))
ekf_rms_velocity_error_3D_results = np.zeros(len(sigma_values))

optimal_sigma = 5e-10
Q = np.diag([optimal_sigma, optimal_sigma, optimal_sigma])**2

lkf_state_history, lkf_covariance_history, lkf_residuals_df = lkf.run(initial_state_guess, np.zeros(7), P_0, measurement_data, R=np.diag(noise_var), max_iterations=1, process_noise_approach='SNC', Q=Q)
ekf_state_history, ekf_covariance_history, ekf_residuals_df = ekf.run(initial_state_guess, np.zeros(7), P_0, measurement_data, R=np.diag(noise_var), start_mode='warm', start_length=1000, process_noise_approach='SNC', Q=Q)

residual_df_list = [lkf_residuals_df, ekf_residuals_df]
filter_names = ['LKF with SNC', 'EKF with SNC']
colors_list = ['red', 'green', 'blue']

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
                                    orientation="h",
                                    yanchor="top",
                                    y=1.13,
                                    xanchor="left",
                                    x=0.7,
                                    itemsizing='constant'))
        fig.write_html(f"ASEN_6080/HW3/figures/{filter_name.lower().replace(' ','_')}_pre_fit_residuals_iteration_{iteration+1}.html")
        fig.write_image(f"ASEN_6080/HW3/figures/pngs/{filter_name.lower().replace(' ','_')}_pre_fit_residuals_iteration_{iteration+1}.png")

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
        fig.write_html(f"ASEN_6080/HW3/figures/{filter_name.lower().replace(' ','_')}_post_fit_residuals_iteration_{iteration+1}.html")
        fig.write_image(f"ASEN_6080/HW3/figures/pngs/{filter_name.lower().replace(' ','_')}_post_fit_residuals_iteration_{iteration+1}.png")
        
# Plotting time history of state errors for LKF and EKF with SNC approach
lkf_state_errors = np.zeros_like(lkf_state_history)  # Initialize state error array
ekf_state_errors = np.zeros_like(lkf_state_history)  # Initialize state error array
for k in range(lkf_state_history.shape[1]):
    lkf_state_errors[:,k] = lkf_state_history[:,k] - truth_data['augmented_state_history'].values[k][0:7]
    ekf_state_errors[:,k] = ekf_state_history[:,k] - truth_data['augmented_state_history'].values[k][0:7]

fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=('X Position Error', 'Y Position Error', 'Z Position Error'))
fig.add_trace(go.Scatter(x=time_vector, y=lkf_state_errors[0,:]*1000, mode='lines', name='Position Error (SNC)', line=dict(color='blue')), row=1, col=1)
fig.add_trace(go.Scatter(x=time_vector, y=lkf_state_errors[1,:]*1000, mode='lines', name='Position Error (SNC)', line=dict(color='blue'), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=time_vector, y=lkf_state_errors[2,:]*1000, mode='lines', name='Position Error (SNC)', line=dict(color='blue'), showlegend=False), row=3, col=1)

fig.add_trace(go.Scatter(x=time_vector, y=3*np.sqrt(lkf_covariance_history[0,0,:])*1000, mode='lines', name='3-Sigma Bound (SNC)', line=dict(color='red', dash='dash')), row=1, col=1)
fig.add_trace(go.Scatter(x=time_vector, y=-3*np.sqrt(lkf_covariance_history[0,0,:])*1000, mode='lines', name='LKF X Position -3-Sigma Bound', line=dict(color='red', dash='dash'), showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=time_vector, y=3*np.sqrt(lkf_covariance_history[1,1,:])*1000, mode='lines', name='LKF Y Position 3-Sigma Bound (SNC)', line=dict(color='red', dash='dash'), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=time_vector, y=-3*np.sqrt(lkf_covariance_history[1,1,:])*1000, mode='lines', name='LKF Y Position -3-Sigma Bound (SNC)', line=dict(color='red', dash='dash'), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=time_vector, y=3*np.sqrt(lkf_covariance_history[2,2,:])*1000, mode='lines', name='LKF Z Position 3-Sigma Bound (SNC)', line=dict(color='red', dash='dash'), showlegend=False), row=3, col=1)
fig.add_trace(go.Scatter(x=time_vector, y=-3*np.sqrt(lkf_covariance_history[2,2,:])*1000, mode='lines', name='LKF Z Position -3-Sigma Bound (SNC)', line=dict(color='red', dash='dash'), showlegend=False), row=3, col=1)

fig.update_xaxes(title_text="Time (s)", tickfont=dict(size=20), title_font=dict(size=22))
fig.update_yaxes(title_text="Position Error (m)", tickfont=dict(size=20), title_font=dict(size=22), showexponent="all", exponentformat="e")
fig.update_annotations(font=dict(size=24))
fig.update_layout(title_text="LKF Position Errors with SNC Process Noise Approach",
                    title_font=dict(size=30),
                    width=1800,
                    height=900,
                    legend=dict(font=dict(size=27),
                                orientation="h",
                                yanchor="top",
                                y=1.1,
                                xanchor="left",
                                x=0.7,
                                itemsizing='constant'))
fig.write_html("ASEN_6080/HW3/figures/lkf_position_errors_snc.html")
fig.write_image("ASEN_6080/HW3/figures/pngs/lkf_position_errors_snc.png")

fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=('X Position Error', 'Y Position Error', 'Z Position Error'))
fig.add_trace(go.Scatter(x=time_vector, y=ekf_state_errors[0,:]*1000, mode='lines', name='Position Error (SNC)', line=dict(color='blue')), row=1, col=1)
fig.add_trace(go.Scatter(x=time_vector, y=ekf_state_errors[1,:]*1000, mode='lines', name='Position Error (SNC)', line=dict(color='blue'), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=time_vector, y=ekf_state_errors[2,:]*1000, mode='lines', name='Position Error (SNC)', line=dict(color='blue'), showlegend=False), row=3, col=1)

fig.add_trace(go.Scatter(x=time_vector, y=3*np.sqrt(ekf_covariance_history[0,0,:])*1000, mode='lines', name='3-Sigma Bound', line=dict(color='red', dash='dash'), showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=time_vector, y=-3*np.sqrt(ekf_covariance_history[0,0,:])*1000, mode='lines', line=dict(color='red', dash='dash'), showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=time_vector, y=3*np.sqrt(ekf_covariance_history[1,1,:])*1000, mode='lines', line=dict(color='red', dash='dash'), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=time_vector, y=-3*np.sqrt(ekf_covariance_history[1,1,:])*1000, mode='lines', line=dict(color='red', dash='dash'), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=time_vector, y=3*np.sqrt(ekf_covariance_history[2,2,:])*1000, mode='lines', line=dict(color='red', dash='dash'), showlegend=False), row=3, col=1)
fig.add_trace(go.Scatter(x=time_vector, y=-3*np.sqrt(ekf_covariance_history[2,2,:])*1000, mode='lines', line=dict(color='red', dash='dash'), showlegend=False), row=3, col=1)
fig.update_xaxes(title_text="Time (s)", tickfont=dict(size=20), title_font=dict(size=22))
fig.update_yaxes(title_text="Position Error (m)", tickfont=dict(size=20), title_font=dict(size=22), showexponent="all", exponentformat="e")
fig.update_annotations(font=dict(size=24))
fig.update_layout(title_text="EKF Position Errors with SNC Process Noise Approach",
                    title_font=dict(size=30),
                    width=1800,
                    height=900,
                    legend=dict(font=dict(size=27),
                                orientation="h",
                                yanchor="top",
                                y=1.1,
                                xanchor="left",
                                x=0.7,
                                itemsizing='constant'))
fig.write_html("ASEN_6080/HW3/figures/ekf_position_errors_snc.html")
fig.write_image("ASEN_6080/HW3/figures/pngs/ekf_position_errors_snc.png")

fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=('X Velocity Error', 'Y Velocity Error', 'Z Velocity Error'))
fig.add_trace(go.Scatter(x=time_vector, y=lkf_state_errors[3,:]*1E6, mode='lines', name='Velocity Error (SNC)', line=dict(color='blue')), row=1, col=1)
fig.add_trace(go.Scatter(x=time_vector, y=lkf_state_errors[4,:]*1E6, mode='lines', name='Velocity Error (SNC)', line=dict(color='blue'), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=time_vector, y=lkf_state_errors[5,:]*1E6, mode='lines', name='Velocity Error (SNC)', line=dict(color='blue'), showlegend=False), row=3, col=1)
fig.add_trace(go.Scatter(x=time_vector, y=3*np.sqrt(lkf_covariance_history[3,3,:])*1E6, mode='lines', name='3-Sigma Bound (SNC)', line=dict(color='red', dash='dash')), row=1, col=1)
fig.add_trace(go.Scatter(x=time_vector, y=-3*np.sqrt(lkf_covariance_history[3,3,:])*1E6, mode='lines', name='LKF X Velocity -3-Sigma Bound (SNC)', line=dict(color='red', dash='dash'), showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=time_vector, y=3*np.sqrt(lkf_covariance_history[4,4,:])*1E6, mode='lines', name='LKF Y Velocity 3-Sigma Bound (SNC)', line=dict(color='red', dash='dash'), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=time_vector, y=-3*np.sqrt(lkf_covariance_history[4,4,:])*1E6, mode='lines', name='LKF Y Velocity -3-Sigma Bound (SNC)', line=dict(color='red', dash='dash'), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=time_vector, y=3*np.sqrt(lkf_covariance_history[5,5,:])*1E6, mode='lines', name='LKF Z Velocity 3-Sigma Bound (SNC)', line=dict(color='red', dash='dash'), showlegend=False), row=3, col=1)
fig.add_trace(go.Scatter(x=time_vector, y=-3*np.sqrt(lkf_covariance_history[5,5,:])*1E6, mode='lines', name='LKF Z Velocity -3-Sigma Bound (SNC)', line=dict(color='red', dash='dash'), showlegend=False), row=3, col=1)
fig.update_xaxes(title_text="Time (s)", tickfont=dict(size=20), title_font=dict(size=22))
fig.update_yaxes(title_text="Velocity Error (mm/s)", tickfont=dict(size=20), title_font=dict(size=22), showexponent="all", exponentformat="e")
fig.update_annotations(font=dict(size=24))
fig.update_layout(title_text="LKF Velocity Errors with SNC Process Noise Approach",
                    title_font=dict(size=30),
                    width=1800,
                    height=900,
                    legend=dict(font=dict(size=27),
                                orientation="h",
                                yanchor="top",
                                y=1.1,
                                xanchor="left",
                                x=0.7,
                                itemsizing='constant'))
fig.write_html("ASEN_6080/HW3/figures/lkf_velocity_errors_snc.html")
fig.write_image("ASEN_6080/HW3/figures/pngs/lkf_velocity_errors_snc.png")

fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=('X Velocity Error', 'Y Velocity Error', 'Z Velocity Error'))
fig.add_trace(go.Scatter(x=time_vector, y=ekf_state_errors[3,:]*1E6, mode='lines', name='Velocity Error (SNC)', line=dict(color='blue')), row=1, col=1)
fig.add_trace(go.Scatter(x=time_vector, y=ekf_state_errors[4,:]*1E6, mode='lines', name='Velocity Error (SNC)', line=dict(color='blue'), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=time_vector, y=ekf_state_errors[5,:]*1E6, mode='lines', name='Velocity Error (SNC)', line=dict(color='blue'), showlegend=False), row=3, col=1)
fig.add_trace(go.Scatter(x=time_vector, y=3*np.sqrt(ekf_covariance_history[3,3,:])*1E6, mode='lines', name='3-Sigma Bound (SNC)', line=dict(color='red', dash='dash'), showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=time_vector, y=-3*np.sqrt(ekf_covariance_history[3,3,:])*1E6, mode='lines', line=dict(color='red', dash='dash'), showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=time_vector, y=3*np.sqrt(ekf_covariance_history[4,4,:])*1E6, mode='lines', line=dict(color='red', dash='dash'), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=time_vector, y=-3*np.sqrt(ekf_covariance_history[4,4,:])*1E6, mode='lines', line=dict(color='red', dash='dash'), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=time_vector, y=3*np.sqrt(ekf_covariance_history[5,5,:])*1E6, mode='lines', line=dict(color='red', dash='dash'), showlegend=False), row=3, col=1)
fig.add_trace(go.Scatter(x=time_vector, y=-3*np.sqrt(ekf_covariance_history[5,5,:])*1E6, mode='lines', line=dict(color='red', dash='dash'), showlegend=False), row=3, col=1)
fig.update_xaxes(title_text="Time (s)", tickfont=dict(size=20), title_font=dict(size=22))
fig.update_yaxes(title_text="Velocity Error (mm/s)", tickfont=dict(size=20), title_font=dict(size=22), showexponent="all", exponentformat="e")
fig.update_annotations(font=dict(size=24))
fig.update_layout(title_text="EKF Velocity Errors with SNC Process Noise Approach",
                    title_font=dict(size=30),
                    width=1800,
                    height=900,
                    legend=dict(font=dict(size=27),
                                orientation="h",
                                yanchor="top",
                                y=1.1,
                                xanchor="left",
                                x=0.7,
                                itemsizing='constant'))
fig.write_html("ASEN_6080/HW3/figures/ekf_velocity_errors_snc.html")
fig.write_image("ASEN_6080/HW3/figures/pngs/ekf_velocity_errors_snc.png")

# for sigma in sigma_values:
#     print(f"Running LKF with SNC process noise approach and sigma = {sigma:.1e} km/s^2...")
#     Q = np.diag([sigma, sigma, sigma])**2
#     lkf_state_history, lkf_covariance_history, lkf_residuals_df = lkf.run(initial_state_guess, np.zeros(7), P_0, measurement_data, R=np.diag(noise_var), max_iterations=1, process_noise_approach='SNC', Q=Q)
#     ekf_state_history, ekf_covariance_history, ekf_residuals_df = ekf.run(initial_state_guess, np.zeros(7), P_0, measurement_data, R=np.diag(noise_var), start_mode='warm', start_length=1000, process_noise_approach='SNC', Q=Q)

#     residual_df_list = [lkf_residuals_df, ekf_residuals_df]
#     filter_names = ['LKF with SNC', 'EKF with SNC']
#     for residuals_df, filter_name in zip(residual_df_list, filter_names):
#         iteration=0
#         relevant_residuals = residuals_df[residuals_df['iteration'] == iteration]['post-fit'].values.copy()

#         for i in range(len(residuals_df['station'].unique())):
#             # Set any NaN values to zero for RMS calculation
#             relevant_residuals[i][np.isnan(relevant_residuals[i])] = 0.0

#         # Sum the residuals across stations to get a single residual vector for the iteration
#         combined_residuals = np.sum(relevant_residuals, axis=0)

#         # Reset zeros to NaN so they aren't included in RMS calculation
#         combined_residuals[combined_residuals == 0.0] = np.nan
        
#         # Compute RMS of combined residuals for the iteration
#         rms_range_residual = np.sqrt(np.abs(np.nanmean((combined_residuals[0,:]*1E3) **2))) # Convert from km to m for RMS calculation
#         rms_range_rate_residual = np.sqrt(np.abs(np.nanmean((combined_residuals[1,:]*1E6) **2))) # Convert from km/s to mm/s for RMS calculation

#         if filter_name == 'LKF with SNC':
#             lkf_residual_rms_results[sigma_values.index(sigma), 0] = rms_range_residual
#             lkf_residual_rms_results[sigma_values.index(sigma), 1] = rms_range_rate_residual
#         else:
#             ekf_residual_rms_results[sigma_values.index(sigma), 0] = rms_range_residual
#             ekf_residual_rms_results[sigma_values.index(sigma), 1] = rms_range_rate_residual
        
#         # Compute 3D rms error of state estimate at final iteration compared to truth
#         lkf_state_errors = np.zeros_like(lkf_state_history)  # Initialize state error array
#         ekf_state_errors = np.zeros_like(lkf_state_history)  # Initialize state error array
#         for k in range(lkf_state_history.shape[1]):
#             lkf_state_errors[:,k] = lkf_state_history[:,k] - truth_data['augmented_state_history'].values[k][0:7]
#             ekf_state_errors[:,k] = ekf_state_history[:,k] - truth_data['augmented_state_history'].values[k][0:7]

#         lkf_rms_position_error_3D = np.sqrt(np.mean(np.sum(lkf_state_errors[0:3,:]**2, axis=0))) * 1000  # in meters
#         ekf_rms_position_error_3D = np.sqrt(np.mean(np.sum(ekf_state_errors[0:3,:]**2, axis=0))) * 1000  # in meters
#         lkf_rms_velocity_error_3D = np.sqrt(np.mean(np.sum(lkf_state_errors[3:6,:]**2, axis=0))) * 1e6  # in mm/s
#         ekf_rms_velocity_error_3D = np.sqrt(np.mean(np.sum(ekf_state_errors[3:6,:]**2, axis=0))) * 1e6  # in mm/s

#         lkf_rms_position_error_3D_results[sigma_values.index(sigma)] = lkf_rms_position_error_3D
#         ekf_rms_position_error_3D_results[sigma_values.index(sigma)] = ekf_rms_position_error_3D
#         lkf_rms_velocity_error_3D_results[sigma_values.index(sigma)] = lkf_rms_velocity_error_3D
#         ekf_rms_velocity_error_3D_results[sigma_values.index(sigma)] = ekf_rms_velocity_error_3D

# # Plot RMS of range and range rate residuals for LKF and EKF with SNC approach

# fig = make_subplots(rows=2, cols=1, subplot_titles=('Range Residual RMS vs Sigma', 'Range Rate Residual RMS vs Sigma'))
# fig.add_trace(go.Scatter(x=sigma_values, y=lkf_residual_rms_results[:,0], mode='markers+lines', name='LKF', line=dict(dash='solid', color='blue')), row=1, col=1)
# fig.add_trace(go.Scatter(x=sigma_values, y=ekf_residual_rms_results[:,0], mode='markers+lines', name='EKF', line=dict(dash='solid', color='red')), row=1, col=1)
# fig.add_trace(go.Scatter(x=sigma_values, y=lkf_residual_rms_results[:,1], mode='markers+lines', name='LKF Range Rate Residual RMS', line=dict(dash='solid', color='blue'), showlegend=False), row=2, col=1)
# fig.add_trace(go.Scatter(x=sigma_values, y=ekf_residual_rms_results[:,1], mode='markers+lines', name='EKF Range Rate Residual RMS', line=dict(dash='solid', color='red'), showlegend=False), row=2, col=1)
# fig.update_xaxes(type='log', title_text='Sigma Value (km/s^2)', tickfont=dict(size=20), title_font=dict(size=22), showexponent="all", exponentformat="e", row=1, col=1)
# fig.update_xaxes(type='log', title_text='Sigma Value (km/s^2)', tickfont=dict(size=20), title_font=dict(size=22), showexponent="all", exponentformat="e", row=2, col=1)
# fig.update_yaxes(title_text='RMS of Range Residuals (m)', type='log', tickfont=dict(size=20), title_font=dict(size=22), row=1, col=1)
# fig.update_yaxes(title_text='RMS of Range Rate Residuals (mm/s)', type='log', tickfont=dict(size=20), title_font=dict(size=22), row=2, col=1)
# fig.update_layout(title='RMS of Post-Fit Residuals vs Sigma for LKF and EKF with SNC', title_font=dict(size=30), legend=dict(font=dict(size=22)))
# fig.update_annotations(font=dict(size=24))
# fig.write_html("ASEN_6080/HW3/figures/residual_rms_vs_sigma_subplots.html")
# fig.write_image("ASEN_6080/HW3/figures/pngs/residual_rms_vs_sigma_subplots.png")
# fig.show()

# # Plot RMS of 3D position and velocity errors for LKF and EKF with SNC approach
# fig = make_subplots(rows=2, cols=1, subplot_titles=('3D Position Error RMS vs Sigma', '3D Velocity Error RMS vs Sigma'))
# fig.add_trace(go.Scatter(x=sigma_values, y=lkf_rms_position_error_3D_results, mode='markers+lines', name='LKF', line=dict(dash='solid', color='blue')), row=1, col=1)
# fig.add_trace(go.Scatter(x=sigma_values, y=ekf_rms_position_error_3D_results, mode='markers+lines', name='EKF', line=dict(dash='solid', color='red')), row=1, col=1)
# fig.add_trace(go.Scatter(x=sigma_values, y=lkf_rms_velocity_error_3D_results, mode='markers+lines', name='LKF 3D Velocity Error RMS', line=dict(dash='solid', color='blue'), showlegend=False), row=2, col=1)
# fig.add_trace(go.Scatter(x=sigma_values, y=ekf_rms_velocity_error_3D_results, mode='markers+lines', name='EKF 3D Velocity Error RMS', line=dict(dash='solid', color='red'), showlegend=False), row=2, col=1)
# fig.update_xaxes(type='log', title_text='Sigma Value (km/s^2)', tickfont=dict(size=20), title_font=dict(size=22), showexponent="all", exponentformat="e", row=1, col=1)
# fig.update_xaxes(type='log', title_text='Sigma Value (km/s^2)', tickfont=dict(size=20), title_font=dict(size=22), showexponent="all", exponentformat="e", row=2, col=1)
# fig.update_yaxes(title_text='RMS of 3D Position Error (m)', type='log', tickfont=dict(size=20), title_font=dict(size=22), row=1, col=1)
# fig.update_yaxes(title_text='RMS of 3D Velocity Error (mm/s)', type='log', tickfont=dict(size=20), title_font=dict(size=22), row=2, col=1)
# fig.update_layout(title='RMS of 3D State Estimation Error vs Sigma for LKF and EKF with SNC Approach', title_font=dict(size=30), legend=dict(font=dict(size=22)))
# fig.update_annotations(font=dict(size=24))
# fig.write_html("ASEN_6080/HW3/figures/state_error_rms_vs_sigma_subplots.html")
# fig.write_image("ASEN_6080/HW3/figures/pngs/state_error_rms_vs_sigma_subplots.png")
# fig.show()