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
integrator = Integrator(mu, R_e, J2 = J2, J3 = J3, mode=[])
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
plot_state_errors(time_vector, state_errors, state_covariance_estimates, "Base Covariance", file_directory="ASEN_6080/HW7/figures")
plot_state_errors(time_vector, state_errors, total_covariance_estimates, "Total Consider Covariance", file_directory="ASEN_6080/HW7/figures")