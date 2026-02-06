from ASEN_6080.Tools import Integrator, CoordinateMgr
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# DONT FORGET THAT DEFAULT RADIUS IS 6378 KM IN INTEGRATOR AND TOOLS

sat_state = np.array([757700.0E-3, 5222607.0E-3, 4851500.0E-3, 2213.21E-3, 4678.34E-3, -5371.30E-3])  # Example satellite state in km and km/s
mu = 3.986004415E5  # Earth's gravitational parameter in km^3/s^2
J2 = 1.082626925638815E-3 # Earth's J2 coefficient
J3 = 0.0 # Earth's J3 coefficient
R_e = 6378.1363  # Earth's radius in km
C_d = 2.0 # Drag coefficient
spacecraft_mass = 970.0  # Spacecraft mass in kg
spacecraft_area = 3.0  # Spacecraft cross-sectional area in m^2
station_positions_ecef = np.array([[-5127510.0E-3, -3794160.0E-3,  0.0],
                                    [3860910.0E-3, 3238490.0E-3,  3898094.0E-3],
                                    [549505.0E-3, -1380872.0E-3,  6182197.0E-3]])  # Example ground station positions in ECEF coordinates

integrator = Integrator(mu, R_e, mode=['mu','J2','Drag','Stations'], parameter_indices=[6,7,8,9], number_of_stations=3, spacecraft_area=spacecraft_area, spacecraft_mass=spacecraft_mass)

t_final = 3600.0 * 8 # 8 hour in seconds
augmented_state = np.hstack((sat_state, mu, J2, C_d, station_positions_ecef.flatten()))

time, state_history = integrator.integrate_eom(t_final, augmented_state)

# Plot trajectory to verify integration
fig = go.Figure()
fig.add_trace(go.Scatter3d(x=state_history[0, :], y=state_history[1, :], z=state_history[2, :],
                           mode='lines',
                           name='Satellite Trajectory'))
fig.update_layout(scene=dict(xaxis_title='X (km)',
                             yaxis_title='Y (km)',
                             zaxis_title='Z (km)'),
                    title='Satellite Trajectory over 8 Hour Integration')
fig.write_html('ASEN_6080/Project1/figures/variable_integration_trajectory.html')

# Plot Cd over time to verify drag effects
cd_history = state_history[8, :]
fig_cd = go.Figure()
fig_cd.add_trace(go.Scatter(x=time, y=cd_history, mode='lines+markers', name='Cd Estimate'))
fig_cd.update_layout(title='Cd Estimate Over Time', xaxis_title='Time (s)', yaxis_title='Cd Estimate')
fig_cd.write_html('ASEN_6080/Project1/figures/variable_integration_cd.html')