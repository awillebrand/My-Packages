import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from ASEN_6080.Tools import Integrator, MeasurementMgr, CoordinateMgr, BatchLLSEstimator
from plotly.subplots import make_subplots

measurement_data = pd.read_pickle("ASEN_6080/HW2/measurement_data/simulated_measurements.pkl")

mu = 3.986004415E5
R_e = 6378
J2 = 0.0010826269

integrator = Integrator(mu, R_e, mode='J2')
station_1_mgr = MeasurementMgr(station_lat=-35.398333, station_lon=148.981944, initial_earth_spin_angle=np.deg2rad(122))
station_2_mgr = MeasurementMgr(station_lat=40.427222, station_lon=355.749444, initial_earth_spin_angle=np.deg2rad(122))
station_3_mgr = MeasurementMgr(station_lat=35.247163, station_lon=243.205, initial_earth_spin_angle=np.deg2rad(122))
station_mgr_list = [station_1_mgr, station_2_mgr, station_3_mgr]

batch_estimator = BatchLLSEstimator(integrator, station_mgr_list)
initial_state_guess = np.array([0,0,0,0,0,0])

output = batch_estimator.estimate_initial_state(initial_state_guess, measurement_data, max_iterations=10, tol=1e-6)