import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from ASEN_6080.Tools import Integrator, MeasurementMgr
from plotly.subplots import make_subplots

# Initialize orbital elements and parameters
mu = 3.986004415E5
R_e = 6378
J2 = 0.0010826269
J3 = -2.5324e-6

a = 10000
e = 0.001
i = np.deg2rad(40)
LoN = np.deg2rad(80)
AoP = np.deg2rad(40)
f = np.deg2rad(0)
period = 2 * np.pi * np.sqrt(a**3 / mu)

# Set integration and measurement settings
mode = 'Full'
noise_std = np.array([1e-3, 1e-6]) # [range noise = 1 m, range rate noise = 1 mm/s]
#noise_std = np.zeros(2)  # No noise for initial testing
# Integrate orbit trajectory
integrator = Integrator(mu, R_e, mode)
r_vec, v_vec = integrator.keplerian_to_cartesian(a, e, i, LoN, AoP, f)
initial_state = np.hstack((r_vec, v_vec, J2, J3))
time_list = np.arange(0, 15 * period, 10)
time_vector, state_history = integrator.integrate_eom(15*period, initial_state, teval=time_list)

# Initialize Measurement Manager with ground station parameters
station_1_mgr = MeasurementMgr("station_1", station_lat=-35.398333, station_lon=148.981944, initial_earth_spin_angle=np.deg2rad(122))
station_2_mgr = MeasurementMgr("station_2", station_lat=40.427222, station_lon=355.749444, initial_earth_spin_angle=np.deg2rad(122))
station_3_mgr = MeasurementMgr("station_3", station_lat=35.247163, station_lon=243.205, initial_earth_spin_angle=np.deg2rad(122))

# Simulate measurements from all three stations
station_1_measurements = station_1_mgr.simulate_measurements(state_history, time_vector, 'ECI', noise=True, noise_sigma=noise_std)
station_2_measurements = station_2_mgr.simulate_measurements(state_history, time_vector, 'ECI', noise=True, noise_sigma=noise_std)
station_3_measurements = station_3_mgr.simulate_measurements(state_history, time_vector, 'ECI', noise=True, noise_sigma=noise_std)

# Plot measurements to validate
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Range Measurements", "Range Rate Measurements"))
fig.add_trace(go.Scatter(x=time_vector/3600, y=station_1_measurements[0,:], mode='markers', name='Station 1', line=dict(color='red')), row=1, col=1)
fig.add_trace(go.Scatter(x=time_vector/3600, y=station_2_measurements[0,:], mode='markers', name='Station 2', line=dict(color='green')), row=1, col=1)
fig.add_trace(go.Scatter(x=time_vector/3600, y=station_3_measurements[0,:], mode='markers', name='Station 3', line=dict(color='blue')), row=1, col=1)
fig.add_trace(go.Scatter(x=time_vector/3600, y=station_1_measurements[1,:], mode='markers', name='Station 1', line=dict(color='red'), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=time_vector/3600, y=station_2_measurements[1,:], mode='markers', name='Station 2', line=dict(color='green'), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=time_vector/3600, y=station_3_measurements[1,:], mode='markers', name='Station 3', line=dict(color='blue'), showlegend=False), row=2, col=1)
fig.update_traces(marker=dict(size=2))
fig.update_xaxes(title_text="Time (hours)", row=2, col=1)
fig.update_yaxes(title_text="Range (km)", row=1, col=1)
fig.update_yaxes(title_text="Range Rate (km/s)", row=2, col=1)
fig.update_layout(title_text="Simulated Range and Range Rate Measurements from Ground Stations",
                  title_font=dict(size=28),
                  width=1200,
                  height=800,
                  legend=dict(font=dict(size=18)))
fig.write_html("ASEN_6080/HW2/figures/simulated_measurements.html")

# Save measurement data to pickle file
measurement_data_frame = pd.DataFrame({
    'time': time_vector,
    'station_1_measurements': list(station_1_measurements.T),
    'station_2_measurements': list(station_2_measurements.T),
    'station_3_measurements': list(station_3_measurements.T)
})

measurement_data_frame.to_pickle("ASEN_6080/HW2/measurement_data/simulated_measurements_J3.pkl")

# Integrate STM for future use
[_, augmented_state_history] = integrator.integrate_stm(time_vector[-1], initial_state, teval=time_vector)

# Add intial state to saved data as well
truth_data_frame = pd.DataFrame({
    'time': time_vector,
    'augmented_state_history': list(augmented_state_history.T),
    'initial_state': [initial_state for _ in time_vector]
})

truth_data_frame.to_pickle("ASEN_6080/HW2/measurement_data/truth_data_J3.pkl")