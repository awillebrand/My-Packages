import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import plotly.graph_objects as go
from Project_Tools.RK4 import rk4
from Project_Tools.control_vector import compute_gains

from initial_conditions import sigma_0_LMO, omega_0_LMO, I_LMO

# Define time array for integration
t_0 = 0
t_f = 6500
dt = 1
t = np.arange(t_0, t_f + dt, dt)

# Compute control gains
decay_time_constant = 120
P, K = compute_gains(I_LMO, decay_time_constant)

# Initial state vector (MRP and angular velocity)
y0 = np.hstack((sigma_0_LMO, np.deg2rad(omega_0_LMO)))

solution = rk4(y0, t, I_LMO, P, K)

# Extract MRP from the solution at necessary times
times_of_interest = [300, 2100, 3400, 4400, 5600]

solutions_at_interest_times = np.array([solution[np.where(t == time)[0][0], 0:3] for time in times_of_interest])

fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=solution[:, 0], mode='lines', name='sigma_1'))
fig.add_trace(go.Scatter(x=t, y=solution[:, 1], mode='lines', name='sigma_2'))
fig.add_trace(go.Scatter(x=t, y=solution[:, 2], mode='lines', name='sigma_3'))
fig.update_layout(title='MRP Attitude Over Time', xaxis_title='Time (s)', yaxis_title='MRP Components')
fig.show()

fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=solution[:, 3], mode='lines', name='omega_1'))
fig.add_trace(go.Scatter(x=t, y=solution[:, 4], mode='lines', name='omega_2'))
fig.add_trace(go.Scatter(x=t, y=solution[:, 5], mode='lines', name='omega_3'))
fig.update_layout(title='Angular Velocity Over Time', xaxis_title='Time (s)', yaxis_title='Angular Velocity Components (rad/s)')
fig.show()

# Save solutions at interest times to files
for i, time in enumerate(times_of_interest):
    with open(os.path.join(os.path.dirname(__file__), '..', 'coursera_validation_files', f'task_11_solution_time_{time}.txt'), 'w') as f:
        f.write(' '.join(str(x) for x in solutions_at_interest_times[i,:3]))