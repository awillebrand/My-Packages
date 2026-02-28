import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from ASEN_6080.Tools import Integrator, MeasurementMgr, CoordinateMgr, SRIF, BatchLLSEstimator, plot_state_errors
from plotly.subplots import make_subplots
import warnings
warnings.simplefilter('error', RuntimeWarning)
measurement_data = pd.read_pickle("ASEN_6080/HW2/measurement_data/simulated_measurements.pkl")
truth_data = pd.read_pickle("ASEN_6080/HW2/measurement_data/truth_data.pkl")
time_vector=measurement_data['time'].values

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
initial_state_guess = truth_data['initial_state'].values[0][0:7]+ initial_state_deviation
P_0 = np.diag([1, 1, 1, 1e-3, 1e-3, 1e-3,1e-10])**2

srif = SRIF(integrator, station_mgr_list, initial_earth_spin_angle=np.deg2rad(122))

srif_estimated_state_history, srif_estimated_cov_history, srif_residuals_history = srif.run(
    initial_state_guess,
    np.zeros(7),
    P_0,
    measurement_data,
    R_noise = np.diag(noise_var),
    triangularize_time_update=True
)

augmented_truth_state = truth_data['augmented_state_history'].values

truth_state_history = np.zeros((7, augmented_truth_state.shape[0]))

for i, state in enumerate(augmented_truth_state):
    truth_state = state[0:7]
    truth_state_history[:, i] = truth_state

state_errors = srif_estimated_state_history - truth_state_history

plot_state_errors(time_vector, state_errors, srif_estimated_cov_history, "SRIF", file_directory="ASEN_6080/HW5/figures")