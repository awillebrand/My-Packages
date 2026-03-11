import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from ASEN_6080.Tools import Integrator, MeasurementMgr, EKF, UKF, plot_state_errors, plot_residuals
from plotly.subplots import make_subplots

measurement_data = pd.read_pickle("ASEN_6080/HW2/measurement_data/simulated_measurements.pkl")
truth_data = pd.read_pickle("ASEN_6080/HW2/measurement_data/truth_data.pkl")
J3_measurement_data = pd.read_pickle("ASEN_6080/HW2/measurement_data/simulated_measurements_J3.pkl")
J3_truth_data = pd.read_pickle("ASEN_6080/HW2/measurement_data/truth_data_J3.pkl")
time_vector=measurement_data['time'].values

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
initial_state_guess_large = truth_data['initial_state'].values[0][0:7]+ initial_state_deviation*10

P_0 = np.diag([1, 1, 1, 1e-3, 1e-3, 1e-3,1e-10])**2
R = np.diag(noise_var)
optimal_sigma = 5e-8 #< 5e-8
Q = np.diag([optimal_sigma, optimal_sigma, optimal_sigma])**2

alpha = 1
beta = 2

ukf = UKF(integrator, station_mgr_list, initial_earth_spin_angle=np.deg2rad(122))
ekf = EKF(integrator, station_mgr_list, initial_earth_spin_angle=np.deg2rad(122))
ukf_estimated_states, ukf_estimated_covariances, ukf_residuals_df = ukf.run(initial_state_guess, P_0, time_vector, measurement_data, alpha=alpha, beta=beta, R=R)
ukf_estimated_states_large, ukf_estimated_covariances_large, ukf_residuals_df_large = ukf.run(initial_state_guess_large, P_0, time_vector, measurement_data, alpha=alpha, beta=beta, R=R)
ukf_estimated_states_Q, ukf_estimated_covariances_Q, ukf_residuals_df_Q = ukf.run(initial_state_guess, P_0, time_vector, measurement_data, alpha=alpha, beta=beta, R=R, Q=Q)
ukf_estimated_states_alpha, ukf_estimated_covariances_alpha, ukf_residuals_df_alpha = ukf.run(initial_state_guess, P_0, time_vector, measurement_data, alpha=1E-4, beta=beta, R=R, Q=Q)

ekf_estimated_states_large, ekf_estimated_covariances_large, ekf_residuals_df_large = ekf.run(initial_state_guess_large, np.zeros(7), P_0, measurement_data, R=R, start_mode='warm', start_length=1000)
ekf_estimated_states_Q, ekf_estimated_covariances_Q, ekf_residuals_df_Q = ekf.run(initial_state_guess, np.zeros(7), P_0, measurement_data, R=R, Q=Q, start_mode='warm', start_length=100, process_noise_approach = 'SNC')

augmented_truth_state = truth_data['augmented_state_history'].values
truth_state_history = np.zeros((7, augmented_truth_state.shape[0]))

for i, state in enumerate(augmented_truth_state):
    truth_state = state[0:7]
    truth_state_history[:, i] = truth_state

ukf_state_errors = ukf_estimated_states - truth_state_history
ukf_state_errors_Q = ukf_estimated_states_Q - truth_state_history
ukf_state_errors_alpha = ukf_estimated_states_alpha - truth_state_history
ukf_state_errors_large = ukf_estimated_states_large - truth_state_history

ekf_state_errors_large = ekf_estimated_states_large - truth_state_history
ekf_state_errors_Q = ekf_estimated_states_Q - truth_state_history

plot_state_errors(time_vector, ukf_state_errors, ukf_estimated_covariances, "UKF", file_directory="ASEN_6080/HW6/figures", unit_multipliers=[1e3, 1e6], units=["m", "mm/s"])
plot_state_errors(time_vector, ukf_state_errors, ukf_estimated_covariances, "UKF (Zoomed)", file_directory="ASEN_6080/HW6/figures", unit_multipliers=[1e3, 1e6], units=["m", "mm/s"], y_axis_limits=[[-1, 1], [-1, 1]])
plot_residuals(time_vector, ukf_residuals_df, "UKF", "ASEN_6080/HW6/figures")

_, _, ukf_error_stats = plot_state_errors(time_vector, ukf_state_errors_Q, ukf_estimated_covariances_Q, "UKF with Process Noise", file_directory="ASEN_6080/HW6/figures", unit_multipliers=[1e3, 1e6], units=["m", "mm/s"])
plot_state_errors(time_vector, ukf_state_errors_Q, ukf_estimated_covariances_Q, "UKF with Process Noise (Zoomed)", file_directory="ASEN_6080/HW6/figures", unit_multipliers=[1e3, 1e6], units=["m", "mm/s"], y_axis_limits=[[-50, 50], [-50, 50]])
plot_residuals(time_vector, ukf_residuals_df_Q, "UKF with Process Noise", "ASEN_6080/HW6/figures")

plot_state_errors(time_vector, ukf_state_errors_alpha, ukf_estimated_covariances_alpha, "UKF with alpha=1E-4", file_directory="ASEN_6080/HW6/figures", unit_multipliers=[1e3, 1e6], units=["m", "mm/s"])
plot_state_errors(time_vector, ukf_state_errors_alpha, ukf_estimated_covariances_alpha, "UKF with alpha=1E-4 (Zoomed)", file_directory="ASEN_6080/HW6/figures", unit_multipliers=[1e3, 1e6], units=["m", "mm/s"], y_axis_limits=[[-200, 200], [-200, 200]])
plot_residuals(time_vector, ukf_residuals_df_alpha, "UKF with alpha=1E-4", "ASEN_6080/HW6/figures")

plot_state_errors(time_vector, ukf_state_errors_large, ukf_estimated_covariances_large, "UKF (Large Initial Error)", file_directory="ASEN_6080/HW6/figures", unit_multipliers=[1e3, 1e6], units=["m", "mm/s"])
plot_state_errors(time_vector, ukf_state_errors_large, ukf_estimated_covariances_large, "UKF (Large Initial Error) (Zoomed)", file_directory="ASEN_6080/HW6/figures", unit_multipliers=[1e3, 1e6], units=["m", "mm/s"], y_axis_limits=[[-1, 1], [-1, 1]])
plot_residuals(time_vector, ukf_residuals_df_large, "UKF (Large Initial Error)", "ASEN_6080/HW6/figures")

plot_state_errors(time_vector, ekf_state_errors_large, ekf_estimated_covariances_large, "EKF (Large Initial Error)", file_directory="ASEN_6080/HW6/figures", unit_multipliers=[1e3, 1e6], units=["m", "mm/s"])
plot_state_errors(time_vector, ekf_state_errors_large, ekf_estimated_covariances_large, "EKF (Large Initial Error) (Zoomed)", file_directory="ASEN_6080/HW6/figures", unit_multipliers=[1e3, 1e6], units=["m", "mm/s"], y_axis_limits=[[-1, 1], [-1, 1]])
plot_residuals(time_vector, ekf_residuals_df_large, "EKF (Large Initial Error)", "ASEN_6080/HW6/figures")

_, _, ekf_error_stats = plot_state_errors(time_vector, ekf_state_errors_Q, ekf_estimated_covariances_Q, "EKF with Process Noise", file_directory="ASEN_6080/HW6/figures", unit_multipliers=[1e3, 1e6], units=["m", "mm/s"])
plot_state_errors(time_vector, ekf_state_errors_Q, ekf_estimated_covariances_Q, "EKF with Process Noise (Zoomed)", file_directory="ASEN_6080/HW6/figures", unit_multipliers=[1e3, 1e6], units=["m", "mm/s"], y_axis_limits=[[-50, 50], [-50, 50]])
plot_residuals(time_vector, ekf_residuals_df_Q, "EKF with Process Noise", "ASEN_6080/HW6/figures")

print("UKF with Process Noise Error Stats:")
for state_name, stats in ukf_error_stats.items():
    print(f"{state_name}: Mean Error = {stats['mean']:.3e}, Std Dev = {stats['std']:.3e}, RMS = {stats['rms']:.3e}")
print("EKF with Process Noise Error Stats:")
for state_name, stats in ekf_error_stats.items():
    print(f"{state_name}: Mean Error = {stats['mean']:.3e}, Std Dev = {stats['std']:.3e}, RMS = {stats['rms']:.3e}")

# J3 Testing

initial_state_deviation = np.array([1.010e-02, -1.218e-01, -1.484e-01,  3.204e-05, -8.320e-05, 1.740e-04,  0.000e+00, 0.000e+00])
initial_state_guess = J3_truth_data['initial_state'].values[0][0:8]+ initial_state_deviation

integrator = Integrator(mu, R_e, mode=['J2', 'J3'], parameter_indices=[6, 7])
P_0 = np.diag([1, 1, 1, 1e-3, 1e-3, 1e-3,1e-10, 1e-10])**2

ukf = UKF(integrator, station_mgr_list, initial_earth_spin_angle=np.deg2rad(122))
ukf_J3_estimated_states, ukf_J3_estimated_covariances, ukf_J3_residuals_df = ukf.run(initial_state_guess, P_0, time_vector, J3_measurement_data, alpha=alpha, beta=beta, R=R)
augmented_truth_state_J3 = J3_truth_data['augmented_state_history'].values
truth_state_history_J3 = np.zeros((8, augmented_truth_state_J3.shape[0]))

for i, state in enumerate(augmented_truth_state_J3):
    truth_state = state[0:8]
    truth_state_history_J3[:, i] = truth_state

ukf_J3_state_errors = ukf_J3_estimated_states - truth_state_history_J3

plot_state_errors(time_vector, ukf_J3_state_errors, ukf_J3_estimated_covariances, "UKF with J3", file_directory="ASEN_6080/HW6/figures", unit_multipliers=[1e3, 1e6, 1], units=["m", "mm/s", ""])
plot_state_errors(time_vector, ukf_J3_state_errors, ukf_J3_estimated_covariances, "UKF with J3 (Zoomed)", file_directory="ASEN_6080/HW6/figures", unit_multipliers=[1e3, 1e6, 1], units=["m", "mm/s", ""], y_axis_limits=[[-1, 1], [-1, 1], [-1e-6, 1e-6]])
plot_residuals(time_vector, ukf_J3_residuals_df, "UKF with J3", "ASEN_6080/HW6/figures")