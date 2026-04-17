import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots

def plot_residuals(time_vector : np.ndarray, residuals_df : pd.DataFrame,  filter_name: str, file_directory : str, auto_save = True, colors_list: list = ['red', 'green', 'blue']):
    """
    Plot pre-fit and post-fit residuals for each iteration of the filter, including scatter plots of residuals over time and histograms of residual distributions. Also compute and print mean, standard deviation, and RMS of residuals for each iteration.
    Parameters:
    time_vector (np.ndarray):
        Array of time values corresponding to the residuals.
    residuals_df (pd.DataFrame):residuals_df (pd.DataFrame):
        DataFrame containing residuals with columns ['iteration', 'station', 'pre-fit', 'post-fit'].
    filter_name (str):
        Name of the filter being analyzed (e.g., "LKF", "Batch LLS") for plot titles.
    file_directory (str):
        Directory where the plots will be saved.
    auto_save (bool), optional:
        If True, automatically save the plots to the specified directory. If False, display the plots without saving. Default is True.
    colors_list (list), optional:
        List of colors to use for different stations in the plots. Default is ['red', 'green', 'blue'].
    Returns:
        None, but saves plots to the specified directory.

    """
    fig_list = []
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
        mean_range_residual = np.nanmean(combined_residuals[0,:]*1E5) # Convert from km to cm for mean calculation
        std_range_residual = np.nanstd(combined_residuals[0,:]*1E5) # Convert from km to cm for std calculation
        mean_range_rate_residual = np.nanmean(combined_residuals[1,:]*1E6) # Convert from km/s to mm/s for mean calculation
        std_range_rate_residual = np.nanstd(combined_residuals[1,:]*1E6) # Convert from km/s to mm/s for std calculation

        # Reset zeros to NaN in individual station residuals as well for accurate RMS calculation
        for i in range(len(residuals_df['station'].unique())):
            # Set any NaN values to zero for RMS calculation
            relevant_residuals[i][relevant_residuals[i] == 0.0] = np.nan

        
        fig_1 = make_subplots(
            rows=2, cols=2, 
            shared_xaxes=False,
            column_widths=[0.85, 0.15],
            horizontal_spacing=0.06,
            subplot_titles=(f'Range Residuals (Mean = {mean_range_residual:.4f} cm, Std Dev = {std_range_residual:.4f} cm, RMS = {rms_range_residual:.4f} cm)', 'Distribution',
                            f'Range Rate Residuals (Mean = {mean_range_rate_residual:.4f} mm/s, Std Dev = {std_range_rate_residual:.4f} mm/s, RMS = {rms_range_rate_residual:.4f} mm/s)', 'Distribution')
        )
        
        # Collect all residuals for histogram
        all_range_residuals = []
        all_range_rate_residuals = []
        
        for i, station_name in enumerate(residuals_df['station'].unique()):
            mask = (residuals_df['iteration'] == iteration) & (residuals_df['station'] == station_name)
            pre_fit_residuals = np.vstack(residuals_df[mask]['pre-fit'])
            
            # Add scatter plots (left column)
            fig_1.add_trace(go.Scatter(x=time_vector, y=pre_fit_residuals[0,:]*1E5, 
                                    mode='markers', name=f'{station_name}', 
                                    marker=dict(color=colors_list[i]), legendgroup=f'group{i}'), 
                         row=1, col=1)
            fig_1.add_trace(go.Scatter(x=time_vector, y=pre_fit_residuals[1,:]*1E6, 
                                    mode='markers', name=f'{station_name}', 
                                    marker=dict(color=colors_list[i]), 
                                    showlegend=False, legendgroup=f'group{i}'), 
                         row=2, col=1)
            
            # Collect valid (non-NaN) residuals for histograms
            valid_range = pre_fit_residuals[0,:][~np.isnan(pre_fit_residuals[0,:])] * 1E5
            valid_range_rate = pre_fit_residuals[1,:][~np.isnan(pre_fit_residuals[1,:])] * 1E6
            all_range_residuals.extend(valid_range)
            all_range_rate_residuals.extend(valid_range_rate)
        
        # Add histograms (right column) - rotated to be vertical
        fig_1.add_trace(go.Histogram(y=all_range_residuals, 
                                  marker=dict(color='lightblue'),
                                  showlegend=False,
                                  nbinsy=50), 
                     row=1, col=2)
        fig_1.add_trace(go.Histogram(y=all_range_rate_residuals, 
                                  marker=dict(color='lightcoral'),
                                  showlegend=False,
                                  nbinsy=50), 
                     row=2, col=2)
        
        fig_1.update_traces(marker=dict(size=4), selector=dict(mode='markers'))
        fig_1.update_xaxes(title_text="Time (s)", tickfont=dict(size=20), title_font=dict(size=22), row=1, col=1)
        fig_1.update_xaxes(title_text="Time (s)", tickfont=dict(size=20), title_font=dict(size=22), row=2, col=1)
        fig_1.update_xaxes(title_text="Count", tickfont=dict(size=20), title_font=dict(size=22), row=1, col=2)
        fig_1.update_xaxes(title_text="Count", tickfont=dict(size=20), title_font=dict(size=22), row=2, col=2)
        fig_1.update_yaxes(title_text="Range Residuals (cm)", tickfont=dict(size=20), title_font=dict(size=22), showexponent="all", exponentformat="e", row=1, col=1)
        fig_1.update_yaxes(title_text="Range Rate Residuals (mm/s)", tickfont=dict(size=20), title_font=dict(size=22), showexponent="all", exponentformat="e", row=2, col=1)
        fig_1.update_yaxes(showexponent="all", exponentformat="e", tickfont=dict(size=20), title_font=dict(size=22), row=1, col=2)
        fig_1.update_yaxes(showexponent="all", exponentformat="e", tickfont=dict(size=20), title_font=dict(size=22), row=2, col=2)
        fig_1.update_annotations(font=dict(size=24))
        fig_1.update_layout(title_text=f"{filter_name} Pre-Fit Residuals at Iteration {iteration+1}",
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
        if auto_save:
            fig_1.write_html(f"{file_directory}/{filter_name}_pre_fit_residuals_iteration_{iteration+1}.html")
            # If a pngs folder is present in the file directory, also save as png
            try:
                fig_1.write_image(f"{file_directory}/pngs/{filter_name}_pre_fit_residuals_iteration_{iteration+1}.png")
            except Exception as e:
                pass

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
        rms_range_residual = np.sqrt(np.abs(np.nanmean((combined_residuals[0,:]*1E5) **2))) # Convert from km to cm for RMS calculation
        rms_range_rate_residual = np.sqrt(np.abs(np.nanmean((combined_residuals[1,:]*1E6) **2))) # Convert from km/s to mm/s for RMS calculation

        mean_range_residual = np.nanmean(combined_residuals[0,:]*1E5) # Convert from km to cm for mean calculation
        std_range_residual = np.nanstd(combined_residuals[0,:]*1E5) # Convert from km to cm for std calculation
        mean_range_rate_residual = np.nanmean(combined_residuals[1,:]*1E6) # Convert from km/s to mm/s for mean calculation
        std_range_rate_residual = np.nanstd(combined_residuals[1,:]*1E6) # Convert from km/s to mm/s for std calculation

        # Reset zeros to NaN in individual station residuals as well for accurate RMS calculation
        for i in range(len(residuals_df['station'].unique())):
            # Set any NaN values to zero for RMS calculation
            relevant_residuals[i][relevant_residuals[i] == 0.0] = np.nan

        fig_2 = make_subplots(
            rows=2, cols=2, 
            shared_xaxes=False,
            column_widths=[0.85, 0.15],
            horizontal_spacing=0.06,
            subplot_titles=(f'Range Residuals (Mean = {mean_range_residual:.4f} cm, Std Dev = {std_range_residual:.4f} cm, RMS = {rms_range_residual:.4f} cm)', 'Distribution',
                            f'Range Rate Residuals (Mean = {mean_range_rate_residual:.4f} mm/s, Std Dev = {std_range_rate_residual:.4f} mm/s, RMS = {rms_range_rate_residual:.4f} mm/s)', 'Distribution')
        )
        
        # Collect all residuals for histogram
        all_range_residuals = []
        all_range_rate_residuals = []
        
        for i, station_name in enumerate(residuals_df['station'].unique()):
            mask = (residuals_df['iteration'] == iteration) & (residuals_df['station'] == station_name)
            post_fit_residuals = np.vstack(residuals_df[mask]['post-fit'])
            
            # Add scatter plots (left column)
            fig_2.add_trace(go.Scatter(x=time_vector, y=post_fit_residuals[0,:]*1E5, 
                                    mode='markers', name=f'{station_name}', 
                                    marker=dict(color=colors_list[i]), legendgroup=f'group{i}'), 
                         row=1, col=1)
            fig_2.add_trace(go.Scatter(x=time_vector, y=post_fit_residuals[1,:]*1E6, 
                                    mode='markers', name=f'{station_name}', 
                                    marker=dict(color=colors_list[i]), 
                                    showlegend=False, legendgroup=f'group{i}'), 
                         row=2, col=1)
            
            # Collect valid (non-NaN) residuals for histograms
            valid_range = post_fit_residuals[0,:][~np.isnan(post_fit_residuals[0,:])] * 1E5
            valid_range_rate = post_fit_residuals[1,:][~np.isnan(post_fit_residuals[1,:])] * 1E6
            all_range_residuals.extend(valid_range)
            all_range_rate_residuals.extend(valid_range_rate)
        
        # Add histograms (right column) - rotated to be vertical
        fig_2.add_trace(go.Histogram(y=all_range_residuals, 
                                  marker=dict(color='lightblue'),
                                  showlegend=False,
                                  nbinsy=50), 
                     row=1, col=2)
        fig_2.add_trace(go.Histogram(y=all_range_rate_residuals, 
                                  marker=dict(color='lightcoral'),
                                  showlegend=False,
                                  nbinsy=50), 
                     row=2, col=2)
        
        fig_2.update_traces(marker=dict(size=4), selector=dict(mode='markers'))
        fig_2.update_xaxes(title_text="Time (s)", tickfont=dict(size=20), title_font=dict(size=22), row=1, col=1)
        fig_2.update_xaxes(title_text="Time (s)", tickfont=dict(size=20), title_font=dict(size=22), row=2, col=1)
        fig_2.update_xaxes(title_text="Count", tickfont=dict(size=20), title_font=dict(size=22), row=1, col=2)
        fig_2.update_xaxes(title_text="Count", tickfont=dict(size=20), title_font=dict(size=22), row=2, col=2)
        fig_2.update_yaxes(title_text="Range Residuals (cm)", tickfont=dict(size=20), title_font=dict(size=22), showexponent="all", exponentformat="e", row=1, col=1)
        fig_2.update_yaxes(title_text="Range Rate Residuals (mm/s)", tickfont=dict(size=20), title_font=dict(size=22), showexponent="all", exponentformat="e", row=2, col=1)
        fig_2.update_yaxes(showexponent="all", exponentformat="e", tickfont=dict(size=20), title_font=dict(size=22), row=1, col=2)
        fig_2.update_yaxes(showexponent="all", exponentformat="e", tickfont=dict(size=20), title_font=dict(size=22), row=2, col=2)
        fig_2.update_annotations(font=dict(size=24))
        fig_2.update_layout(title_text=f"{filter_name} Post-Fit Residuals at Iteration {iteration+1}",
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
        if auto_save:
            fig_2.write_html(f"{file_directory}/{filter_name}_post_fit_residuals_iteration_{iteration+1}.html")
            # If a pngs folder is present in the file directory, also save as png
            try:
                fig_2.write_image(f"{file_directory}/pngs/{filter_name}_post_fit_residuals_iteration_{iteration+1}.png")
            except Exception as e:
                pass
        fig_list.append((fig_1, fig_2))
    return fig_list

def plot_state_errors(time_vector: np.ndarray, state_errors: np.ndarray, covariance_history : np.array, filter_name: str, file_directory: str, unit_multipliers: list = [1, 1], units = ['km', 'km/s'], y_axis_limits: list[list] = None, sigma_num : int = 3):
    """
    Plot state estimation errors over time for each state component, including position and velocity errors. Also compute and print mean, standard deviation, and RMS of state errors for each component.
    Parameters:
    time_vector (np.ndarray):
        Array of time values corresponding to the state errors.
    state_errors (np.ndarray):
        Array of state estimation errors with shape (num_states, num_time_steps).
    covariance_history (np.ndarray):
        Array of covariance matrices over time with shape (num_states, num_states, num_time_steps).
    filter_name (str):
        Name of the filter being analyzed (e.g., "LKF", "Batch LLS") for plot titles.
    file_directory (str):
        Directory where the plots will be saved.
    unit_multipliers (list), optional:
        List of multipliers to convert state errors to desired units for plotting (e.g., [1E3, 1E6] to convert position errors from km to m and velocity errors from km/s to m/s). Default is [1, 1] (no conversion).
    units (list), optional:
        List of unit strings corresponding to the state errors for labeling the y-axis (e.g., ["m", "m/s"]). Default is ['km', 'km/s'].
    y_axis_limits (list[list]), optional:
        List of [min, max] limits for the y-axis of each subplot, in the same order as the state components. If None, limits will be determined automatically. Default is None.
    sigma_num (int), optional:
        Number of standard deviations to plot for the covariance bounds (e.g., 3 for 3-sigma bounds). Default is 3.
    Returns:
        pos_fig (go.Figure): Plotly figure object containing position error plots.
        vel_fig (go.Figure): Plotly figure object containing velocity error plots.
        error_stats (dict): Dictionary containing mean, standard deviation, and RMS of state errors for each component.
    """
    state_errors = state_errors.copy()
    
    state_names = ['X Position', 'Y Position', 'Z Position', 'X Velocity', 'Y Velocity', 'Z Velocity']
    error_stats = {}
    
    pos_fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=[f"{state_names[i]} Error" for i in range(3)])
    vel_fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=[f"{state_names[i+3]} Error" for i in range(3)])

    # Convert units of state errors for plotting
    state_errors[0:3,:] *= unit_multipliers[0]  # Convert position errors
    state_errors[3:6,:] *= unit_multipliers[1]  # Convert velocity errors

    for i in range(state_errors[0:6].shape[0]):
        state_error = state_errors[i,:]
        covariance_diagonal = np.abs(covariance_history[i,i,:])
        mean_error = np.nanmean(state_error)
        std_error = np.nanstd(state_error)
        rms_error = np.sqrt(np.nanmean(state_error**2))
        error_stats[state_names[i]] = {'mean': mean_error, 'std': std_error, 'rms': rms_error}

        if i < 3:
            pos_fig.add_trace(go.Scatter(x=time_vector, y=state_error, mode='lines', name=f'State Error', line=dict(color='blue'), showlegend=i==0), row=i+1, col=1)
            pos_fig.add_trace(go.Scatter(x=time_vector, y=sigma_num*np.sqrt(covariance_diagonal)*unit_multipliers[0], mode='lines', name=f'{sigma_num}-sigma bounds', line=dict(color='red', dash='dash'), showlegend=i==0), row=i+1, col=1)
            pos_fig.add_trace(go.Scatter(x=time_vector, y=-sigma_num*np.sqrt(covariance_diagonal)*unit_multipliers[0], mode='lines', name=f'{state_names[i]} -{sigma_num}-sigma bound', line=dict(color='red', dash='dash'), showlegend=False), row=i+1, col=1)
        else:   
            vel_fig.add_trace(go.Scatter(x=time_vector, y=state_error, mode='lines', name=f'State Error', line=dict(color='blue'), showlegend=i==3), row=i-2, col=1)
            vel_fig.add_trace(go.Scatter(x=time_vector, y=sigma_num*np.sqrt(covariance_diagonal)*unit_multipliers[1], mode='lines', name=f'{sigma_num}-sigma bounds', line=dict(color='red', dash='dash'), showlegend=i==3), row=i-2, col=1)
            vel_fig.add_trace(go.Scatter(x=time_vector, y=-sigma_num*np.sqrt(covariance_diagonal)*unit_multipliers[1], mode='lines', name=f'{state_names[i]} -{sigma_num}-sigma bound', line=dict(color='red', dash='dash'), showlegend=False), row=i-2, col=1)

    pos_fig.update_xaxes(title_text="Time (s)", tickfont=dict(size=20), title_font=dict(size=22))
    pos_fig.update_yaxes(title_text=f"Position Error ({units[0]})", tickfont=dict(size=20), title_font=dict(size=22), showexponent="all", exponentformat="e")
    if y_axis_limits is not None:
        for i in range(3):
            pos_fig.update_yaxes(range=y_axis_limits[0], row=i+1, col=1)
    pos_fig.update_annotations(font=dict(size=24))
    pos_fig.update_layout(title_text=f"{filter_name} Position Estimation Errors",
                        title_font=dict(size=30),
                        width=1500,
                        height=900,
                        legend=dict(font=dict(size=22),
                                    orientation="h",
                                    yanchor="top",
                                    y=1.1,
                                    xanchor="left",
                                    x=0.7,
                                    itemsizing='constant'))
    pos_fig.write_html(f"{file_directory}/{filter_name}_position_errors.html")
    try:
        pos_fig.write_image(f"{file_directory}/pngs/{filter_name}_position_errors.png")
    except Exception as e:
        pass

    vel_fig.update_xaxes(title_text="Time (s)", tickfont=dict(size=20), title_font=dict(size=22))
    vel_fig.update_yaxes(title_text=f"Velocity Error ({units[1]})", tickfont=dict(size=20), title_font=dict(size=22), showexponent="all", exponentformat="e")
    if y_axis_limits is not None:
        for i in range(3):
            vel_fig.update_yaxes(range=y_axis_limits[1], row=i+1, col=1)
    vel_fig.update_annotations(font=dict(size=24))
    vel_fig.update_layout(title_text=f"{filter_name} Velocity Estimation Errors",
                        title_font=dict(size=30),
                        width=1500,
                        height=900,
                        legend=dict(font=dict(size=22),
                                    orientation="h",
                                    yanchor="top",
                                    y=1.1,
                                    xanchor="left",
                                    x=0.7,
                                    itemsizing='constant'))
    vel_fig.write_html(f"{file_directory}/{filter_name}_velocity_errors.html")
    try:
        vel_fig.write_image(f"{file_directory}/pngs/{filter_name}_velocity_errors.png")
    except Exception as e:
        pass
    return pos_fig, vel_fig, error_stats