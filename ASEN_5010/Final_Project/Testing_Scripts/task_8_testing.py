import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import plotly.graph_objects as go
from Project_Tools.RK4 import rk4
from Project_Tools.control_vector import compute_gains
from Project_Tools.attitude_error_eval import attitude_error_eval
from Project_Tools.sun_frame_funcs import sun_frame_dcm, sun_frame_angular_velocity
from initial_conditions import sigma_0_LMO, omega_0_LMO, I_LMO

# Define time array for integration
t_0 = 0
t_f = 400
dt = 0.01
t = np.arange(t_0, t_f + dt, dt)

# Compute control gains
decay_time_constant = 120
P, K = compute_gains(I_LMO, decay_time_constant)

# Initial state vector (MRP and angular velocity)
y0 = np.hstack((sigma_0_LMO, np.deg2rad(omega_0_LMO)))

# Run RK4 integration for the given initial conditions and control gains
pointing_mode = 'Sun'

solution = rk4(y0, t, I_LMO, pointing_mode, P, K)

# Extract MRP and angular velocity from the solution
times_of_interest = [15, 100, 200, 400]

solutions_at_interest_times = np.array([solution[np.where(t == time)[0][0]] for time in times_of_interest])

# Evaluate the attitude error at the specified times

attitude_errors = np.zeros((len(t), 6))  # Initialize vector to store attitude errors

for i, time in enumerate(t):
    sigma_BR = solution[i, :3]
    omega_BR = solution[i, 3:]
    reference_frame_dcm = sun_frame_dcm(time)
    reference_frame_angular_velocity = sun_frame_angular_velocity(time)
    sigma_error, omega_error = attitude_error_eval(t, sigma_BR, omega_BR, reference_frame_dcm, reference_frame_angular_velocity)
    attitude_errors[i, :3] = sigma_error
    attitude_errors[i, 3:] = omega_error

# Plot the results
fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=attitude_errors[:, 0], mode='lines', name='sigma_1'))
fig.add_trace(go.Scatter(x=t, y=attitude_errors[:, 1], mode='lines', name='sigma_2'))
fig.add_trace(go.Scatter(x=t, y=attitude_errors[:, 2], mode='lines', name='sigma_3'))
fig.update_layout(title='MRP Attitude Error Over Time', xaxis_title='Time (s)', yaxis_title='MRP Components')
fig.show()

fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=attitude_errors[:, 3], mode='lines', name='omega_1'))
fig.add_trace(go.Scatter(x=t, y=attitude_errors[:, 4], mode='lines', name='omega_2'))
fig.add_trace(go.Scatter(x=t, y=attitude_errors[:, 5], mode='lines', name='omega_3'))
fig.update_layout(title='Angular Velocity Error Over Time', xaxis_title='Time (s)', yaxis_title='Angular Velocity Components (rad/s)')
fig.show()