import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from ASEN_6080.Tools import Integrator, MeasurementMgr, CoordinateMgr, ConsiderCov, plot_state_errors, plot_residuals
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

raw_state_length = 6
noise_var = np.array([1e-3, 1e-6])**2 # [range noise = 1 m, range rate noise = 1 mm/s]
integrator = Integrator(mu, R_e, J2 = J2, mode=[])
station_1_mgr = MeasurementMgr("station_1", station_lat=-35.398333, station_lon=148.981944, initial_earth_spin_angle=np.deg2rad(122))
station_2_mgr = MeasurementMgr("station_2", station_lat=40.427222, station_lon=355.749444, initial_earth_spin_angle=np.deg2rad(122))
station_3_mgr = MeasurementMgr("station_3", station_lat=35.247163, station_lon=243.205, initial_earth_spin_angle=np.deg2rad(122))
station_mgr_list = [station_1_mgr, station_2_mgr, station_3_mgr]

initial_truth_state = truth_data['initial_state'].values[0][0:6]
P_0 = 1e-4 * np.eye(6)
R = np.diag(noise_var)
consider_params = ['J3']
c = np.array([J3])
P_cc = np.array([J3**2]).reshape(1, 1)

# Run consider covariance analysis
consider_cov = ConsiderCov(integrator, station_mgr_list, initial_earth_spin_angle=np.deg2rad(122))
state_estimates, state_covariance_estimates, S_estimates, total_covariance_estimates, psi_history = consider_cov.run(initial_truth_state, P_0, consider_params, initial_S=np.zeros((raw_state_length, len(consider_params))), c=c, P_cc=P_cc, R=R, time_vector=time_vector, measurement_data=measurement_data)

# Plot results
augmented_truth_state = truth_data['augmented_state_history'].values
truth_state_history = np.zeros((7, augmented_truth_state.shape[0]))

for i, state in enumerate(augmented_truth_state):
    truth_state = state[0:7]
    truth_state_history[:, i] = truth_state

state_errors = state_estimates - truth_state_history[0:6, :]
pos_fig, vel_fig, error_stats = plot_state_errors(time_vector, state_errors, total_covariance_estimates, "Sequential CCA", file_directory="ASEN_6080/HW7/figures", sigma_num=2, y_axis_limits=[[-0.5, 0.5], [-5e-4, 5e-4]])

state_names = ['X Position', 'Y Position', 'Z Position', 'X Velocity', 'Y Velocity', 'Z Velocity']
sigma_num = 2
for i in range(state_errors[0:6].shape[0]):
    state_error = state_errors[i,:]
    covariance_diagonal = np.abs(state_covariance_estimates[i,i,:])
    mean_error = np.nanmean(state_error)
    std_error = np.nanstd(state_error)
    rms_error = np.sqrt(np.nanmean(state_error**2))
    error_stats[state_names[i]] = {'mean': mean_error, 'std': std_error, 'rms': rms_error}

    if i < 3:
        pos_fig.add_trace(go.Scatter(x=time_vector, y=sigma_num*np.sqrt(covariance_diagonal), mode='lines', name=f'Original {sigma_num}-sigma bounds', line=dict(color='green', dash='dash'), showlegend=i==0), row=i+1, col=1)
        pos_fig.add_trace(go.Scatter(x=time_vector, y=-sigma_num*np.sqrt(covariance_diagonal), mode='lines', name=f'{state_names[i]} -{sigma_num}-sigma bound', line=dict(color='green', dash='dash'), showlegend=False), row=i+1, col=1)
    else:   
        vel_fig.add_trace(go.Scatter(x=time_vector, y=sigma_num*np.sqrt(covariance_diagonal), mode='lines', name=f'Original {sigma_num}-sigma bounds', line=dict(color='green', dash='dash'), showlegend=i==3), row=i-2, col=1)
        vel_fig.add_trace(go.Scatter(x=time_vector, y=-sigma_num*np.sqrt(covariance_diagonal), mode='lines', name=f'{state_names[i]} -{sigma_num}-sigma bound', line=dict(color='green', dash='dash'), showlegend=False), row=i-2, col=1)

pos_fig.update_layout(legend=dict(x=0.51, y=1.12),
                      width=1800,
                      height=900)
vel_fig.update_layout(legend=dict(x=0.51, y=1.12),
                      width=1800,
                      height=900)
pos_fig.write_html("ASEN_6080/HW7/figures/consider_cov_position_errors.html")
vel_fig.write_html("ASEN_6080/HW7/figures/consider_cov_velocity_errors.html")