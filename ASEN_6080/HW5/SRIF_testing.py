import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from ASEN_6080.Tools import Integrator, MeasurementMgr, SRIF, LKF, plot_state_errors, plot_residuals
from plotly.subplots import make_subplots
import warnings
warnings.simplefilter('error', RuntimeWarning)
measurement_data = pd.read_pickle("ASEN_6080/HW2/measurement_data/simulated_measurements.pkl")
truth_data = pd.read_pickle("ASEN_6080/HW2/measurement_data/truth_data.pkl")
J3_measurement_data = pd.read_pickle("ASEN_6080/HW2/measurement_data/simulated_measurements_J3.pkl")
J3_truth_data = pd.read_pickle("ASEN_6080/HW2/measurement_data/truth_data_J3.pkl")
time_vector=measurement_data['time'].values
breakpoint()
mu = 3.986004415E5
R_e = 6378
J2 = 0.0010826269

raw_state_length = 7
noise_var = np.array([1e-3, 1e-6])**2 # [range noise = 1 m, range rate noise = 1 mm/s]
integrator = Integrator(mu, R_e, mode=['J2'], parameter_indices=[6])
station_1_mgr = MeasurementMgr("station_1", station_lat=-35.398333, station_lon=148.981944, initial_earth_spin_angle=np.deg2rad(122))
station_2_mgr = MeasurementMgr("station_2", station_lat=40.427222, station_lon=355.749444, initial_earth_spin_angle=np.deg2rad(122))
station_3_mgr = MeasurementMgr("station_3", station_lat=35.247163, station_lon=243.205, initial_earth_spin_angle=np.deg2rad(122))
station_mgr_list = [station_1_mgr, station_2_mgr, station_3_mgr]

initial_state_deviation = np.array([1.010e-02, -1.218e-01, -1.484e-01,  3.204e-05, -8.320e-05, 1.740e-04,  0.000e+00])
initial_state_guess = truth_data['initial_state'].values[0][0:7]+ initial_state_deviation
P_0 = np.diag([1, 1, 1, 1e-3, 1e-3, 1e-3,1e-10])**2
Q = optimal_sigma = 5e-8
Q = np.diag([optimal_sigma, optimal_sigma, optimal_sigma])**2

srif = SRIF(integrator, station_mgr_list, initial_earth_spin_angle=np.deg2rad(122))
lkf = LKF(integrator, station_mgr_list, initial_earth_spin_angle=np.deg2rad(122))
lkf_estimated_state_history, lkf_covariance_history, lkf_residuals_df = lkf.run(initial_state_guess, np.zeros(7), P_0, measurement_data, R=np.diag(noise_var), max_iterations=1)

triangular_srif_estimated_state_history, triangular_srif_estimated_cov_history, triangular_srif_residuals_history = srif.run(
    initial_state_guess.copy(),
    np.zeros(7),
    P_0.copy(),
    measurement_data,
    R_noise = np.diag(noise_var),
    triangularize_time_update=True
)

non_triangular_srif_estimated_state_history, non_triangular_srif_estimated_cov_history, non_triangular_srif_residuals_history = srif.run(
    initial_state_guess.copy(),
    np.zeros(7),
    P_0.copy(),
    measurement_data,
    R_noise = np.diag(noise_var),
    triangularize_time_update=False
)

noise_srif_estimated_state_history, noise_srif_estimated_cov_history, noise_srif_residuals_history = srif.run(
    initial_state_guess.copy(),
    np.zeros(7),
    P_0.copy(),
    J3_measurement_data,
    R_noise = np.diag(noise_var),
    Q_noise = Q,
    triangularize_time_update=True
)

augmented_truth_state = truth_data['augmented_state_history'].values
truth_state_history = np.zeros((7, augmented_truth_state.shape[0]))

J3_augmented_truth_state = J3_truth_data['augmented_state_history'].values
J3_truth_state_history = np.zeros((7, J3_augmented_truth_state.shape[0]))

for i, state in enumerate(augmented_truth_state):
    truth_state = state[0:7]
    J3_truth_state = J3_augmented_truth_state[i][0:7]
    truth_state_history[:, i] = truth_state
    J3_truth_state_history[:, i] = J3_truth_state

lkf_state_errors = lkf_estimated_state_history - truth_state_history
triangular_state_errors = triangular_srif_estimated_state_history - truth_state_history
non_triangular_state_errors = non_triangular_srif_estimated_state_history - truth_state_history
J3_state_errors = noise_srif_estimated_state_history - J3_truth_state_history

plot_state_errors(time_vector, triangular_state_errors, triangular_srif_estimated_cov_history, "SRIF with Eq. 5.10.44", file_directory="ASEN_6080/HW5/figures", unit_multipliers=[1e6, 1e6], units=["mm", "mm/s"])
plot_residuals(time_vector, triangular_srif_residuals_history, "SRIF Residuals with Eq. 5.10.44", file_directory="ASEN_6080/HW5/figures", colors_list=['red', 'green', 'blue'])

plot_state_errors(time_vector, non_triangular_state_errors, non_triangular_srif_estimated_cov_history, "SRIF without Eq. 5.10.44", file_directory="ASEN_6080/HW5/figures", unit_multipliers=[1e6, 1e6], units=["mm", "mm/s"])
plot_state_errors(time_vector, triangular_state_errors, triangular_srif_estimated_cov_history, "SRIF with Eq. 5.10.44 Zoomed", file_directory="ASEN_6080/HW5/figures", unit_multipliers=[1e6, 1e6], units=["mm", "mm/s"], y_axis_limits=[[-600, 600], [-0.5, 0.5]])
plot_residuals(time_vector, non_triangular_srif_residuals_history, "SRIF Residuals without Eq. 5.10.44", file_directory="ASEN_6080/HW5/figures", colors_list=['red', 'green', 'blue'])

plot_state_errors(time_vector, non_triangular_state_errors, non_triangular_srif_estimated_cov_history, "SRIF without Eq. 5.10.44 Zoomed", file_directory="ASEN_6080/HW5/figures", unit_multipliers=[1e6, 1e6], units=["mm", "mm/s"], y_axis_limits=[[-600, 600], [-0.5, 0.5]])
plot_state_errors(time_vector, J3_state_errors, noise_srif_estimated_cov_history, "SRIF with Process Noise", file_directory="ASEN_6080/HW5/figures", unit_multipliers=[1e3, 1e6], units=["m", "mm/s"])
plot_residuals(time_vector, noise_srif_residuals_history, "SRIF Residuals with Process Noise", file_directory="ASEN_6080/HW5/figures", colors_list=['red', 'green', 'blue'])

# Difference between triangular and non-triangular SRIF estimates
triangular_non_triangular_state_difference = triangular_srif_estimated_state_history - non_triangular_srif_estimated_state_history
triangular_non_triangular_covariance_difference = np.abs(triangular_srif_estimated_cov_history - non_triangular_srif_estimated_cov_history)
plot_state_errors(time_vector, triangular_non_triangular_state_difference, triangular_non_triangular_covariance_difference, "Difference from Applying Eq. 5.10.44", file_directory="ASEN_6080/HW5/figures", unit_multipliers=[1e6, 1e6], units=["mm", "mm/s"])

state_names = ['X Position', 'Y Position', 'Z Position', 'X Velocity', 'Y Velocity', 'Z Velocity']
error_stats = {}

pos_fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=[f"{state_names[i]} Error" for i in range(3)])
vel_fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=[f"{state_names[i+3]} Error" for i in range(3)])

# Convert units of state errors for plotting
unit_multipliers=[1e6, 1e6]
units=["mm", "mm/s"]
file_directory="ASEN_6080/HW5/figures"
y_axis_limits=[[-600, 600], [-0.5, 0.5]]

triangular_state_errors[0:3,:] *= unit_multipliers[0]  # Convert position errors
triangular_state_errors[3:6,:] *= unit_multipliers[1]  # Convert velocity errors
lkf_state_errors[0:3,:] *= unit_multipliers[0]  # Convert position errors
lkf_state_errors[3:6,:] *= unit_multipliers[1]  # Convert velocity errors

for i in range(lkf_state_errors[0:6].shape[0]):
    triangular_state_error = triangular_state_errors[i,:]
    lkf_state_error = lkf_state_errors[i,:]
    if i < 3:
        pos_fig.add_trace(go.Scatter(x=time_vector, y=triangular_state_error, mode='lines', name=f'SRIF State Error', line=dict(color='blue'), showlegend=i==0), row=i+1, col=1)
        pos_fig.add_trace(go.Scatter(x=time_vector, y=lkf_state_error, mode='lines', name=f'LKF State Error', line=dict(color='red', dash='dash'), showlegend=i==0), row=i+1, col=1)
    else:   
        vel_fig.add_trace(go.Scatter(x=time_vector, y=triangular_state_error, mode='lines', name=f'SRIF State Error', line=dict(color='blue'), showlegend=i==3), row=i-2, col=1)
        vel_fig.add_trace(go.Scatter(x=time_vector, y=lkf_state_error, mode='lines', name=f'LKF State Error', line=dict(color='red', dash='dash'), showlegend=i==3), row=i-2, col=1)
pos_fig.update_xaxes(title_text="Time (s)", tickfont=dict(size=20), title_font=dict(size=22))
pos_fig.update_yaxes(title_text=f"Position Error ({units[0]})", tickfont=dict(size=20), title_font=dict(size=22), showexponent="all", exponentformat="e")
if y_axis_limits is not None:
    for i in range(3):
        pos_fig.update_yaxes(range=y_axis_limits[0], row=i+1, col=1)
pos_fig.update_annotations(font=dict(size=24))
pos_fig.update_layout(title_text=f"SRIF and LKF Position Estimation Errors",
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
pos_fig.write_html(f"{file_directory}/SRIF_LKF_position_errors.html")
pos_fig.write_image(f"{file_directory}/pngs/SRIF_LKF_position_errors.png")

vel_fig.update_xaxes(title_text="Time (s)", tickfont=dict(size=20), title_font=dict(size=22))
vel_fig.update_yaxes(title_text=f"Velocity Error ({units[1]})", tickfont=dict(size=20), title_font=dict(size=22), showexponent="all", exponentformat="e")
if y_axis_limits is not None:
    for i in range(3):
        vel_fig.update_yaxes(range=y_axis_limits[1], row=i+1, col=1)
vel_fig.update_annotations(font=dict(size=24))
vel_fig.update_layout(title_text=f"SRIF and LKF Velocity Estimation Errors",
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
vel_fig.write_html(f"{file_directory}/SRIF_LKF_velocity_errors.html")
vel_fig.write_image(f"{file_directory}/pngs/SRIF_LKF_velocity_errors.png")