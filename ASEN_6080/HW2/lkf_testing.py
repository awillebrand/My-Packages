import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from ASEN_6080.Tools import Integrator, MeasurementMgr, CoordinateMgr, LKF
from plotly.subplots import make_subplots
import warnings
warnings.simplefilter('error', RuntimeWarning)
measurement_data = pd.read_pickle("ASEN_6080/HW2/measurement_data/simulated_measurements.pkl")
truth_data = pd.read_pickle("ASEN_6080/HW2/measurement_data/truth_data.pkl")

mu = 3.986004415E5
R_e = 6378
J2 = 0.0010826269
noise_var = np.array([1, 1e-6])**2 # [range noise = 1 km, range rate noise = 1 mm/s]

integrator = Integrator(mu, R_e, mode='J2')
station_1_mgr = MeasurementMgr("station_1", station_lat=-35.398333, station_lon=148.981944, initial_earth_spin_angle=np.deg2rad(122))
station_2_mgr = MeasurementMgr("station_2", station_lat=40.427222, station_lon=355.749444, initial_earth_spin_angle=np.deg2rad(122))
station_3_mgr = MeasurementMgr("station_3", station_lat=35.247163, station_lon=243.205, initial_earth_spin_angle=np.deg2rad(122))
station_mgr_list = [station_1_mgr, station_2_mgr, station_3_mgr]

initial_state_deviation = np.array([0.0505, -0.609, -0.742, 0.0001602, -0.000416, 0.000870, 0])
initial_state_guess = truth_data['initial_state'].values[0] + initial_state_deviation
P_0 = np.diag([1, 1, 1, 1e-3, 1e-3, 1e-3])**2

lkf = LKF(integrator, station_mgr_list, initial_earth_spin_angle=np.deg2rad(122))

state_history, covariance_history = lkf.run(initial_state_guess, np.zeros(6), P_0, measurement_data, R=np.diag(noise_var))