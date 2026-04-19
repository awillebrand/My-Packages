import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import plotly.graph_objects as go
from Project_Tools.RK4 import rk4
from Project_Tools.mrp_dcm import mrp_to_dcm
from initial_conditions import sigma_0_LMO, omega_0_LMO, I_LMO
from plotly.subplots import make_subplots
# Assemble time vector
t0 = 0
tf = 500
dt = 0.01

t = np.arange(t0, tf + dt, dt)

y_0 = np.hstack((sigma_0_LMO, np.deg2rad(omega_0_LMO)))

# Define a simple control input (zero control)
u = np.zeros(3)

# Run the RK4 integration
y_hist = rk4(y_0, t, I_LMO, 0, 0)

# Extract final MRP and angular velocity
final_sigma = y_hist[-1, 0:3]
final_omega = y_hist[-1, 3:6]

H = I_LMO @ final_omega

kinetic_energy = 0.5 * final_omega.T @ I_LMO @ final_omega

print("Final angular momentum in body frame (kg*m^2/s):", H)
print("Final kinetic energy (J):", kinetic_energy)
print("Final MRP:", final_sigma)
print("Final angular velocity (rad/s):", final_omega)

# Rotate the final angular momentum to the inertial frame using the DCM from inertial to body frame at the final time step
final_DCM = mrp_to_dcm(final_sigma)
final_DCM_inertial = final_DCM.T  # DCM from body to inertial frame

H_inertial = final_DCM_inertial @ H

# Now test with applied control torque

t0 = 0
tf = 100
dt = 0.01

t = np.arange(t0, tf + dt, dt)

u = np.array([0.01, -0.01, 0.02])
y_hist_control = rk4(y_0, t, I_LMO, 0, 0, inputted_u=u)

final_sigma_control = y_hist_control[-1, 0:3]
print("Final MRP with control:", final_sigma_control)

# Write results to a file
with open(os.path.join(os.path.dirname(__file__), '..', 'coursera_validation_files', 'task_7_mrp.txt'), 'w') as f:
    f.write(' '.join(str(x) for x in final_sigma))
with open(os.path.join(os.path.dirname(__file__), '..', 'coursera_validation_files', 'task_7_mrp_control.txt'), 'w') as f:
    f.write(' '.join(str(x) for x in final_sigma_control))
with open(os.path.join(os.path.dirname(__file__), '..', 'coursera_validation_files', 'task_7_H_inertial.txt'), 'w') as f:
    f.write(' '.join(str(x) for x in H_inertial))
with open(os.path.join(os.path.dirname(__file__), '..', 'coursera_validation_files', 'task_7_H_body.txt'), 'w') as f:
    f.write(' '.join(str(x) for x in H))
with open(os.path.join(os.path.dirname(__file__), '..', 'coursera_validation_files', 'task_7_kinetic_energy.txt'), 'w') as f:
    f.write(str(kinetic_energy))

# Plot the results in a subplot
fig = make_subplots(rows=2, cols=1, subplot_titles=("MRP Evolution", "Angular Velocity Evolution"))
fig.add_trace(go.Scatter(x=t, y=y_hist[:, 0], mode='lines', name=r'$\sigma_1$'), row=1, col=1)
fig.add_trace(go.Scatter(x=t, y=y_hist[:, 1], mode='lines', name=r'$\sigma_2$'), row=1, col=1)
fig.add_trace(go.Scatter(x=t, y=y_hist[:, 2], mode='lines', name=r'$\sigma_3$'), row=1, col=1)
fig.add_trace(go.Scatter(x=t, y=np.rad2deg(y_hist[:, 3]), mode='lines', name=f"$\omega_1$ (deg/s)"), row=2, col=1)
fig.add_trace(go.Scatter(x=t, y=np.rad2deg(y_hist[:, 4]), mode='lines', name=f"$\omega_2$ (deg/s)"), row=2, col=1)
fig.add_trace(go.Scatter(x=t, y=np.rad2deg(y_hist[:, 5]), mode='lines', name=f"$\omega_3$ (deg/s)"), row=2, col=1)
fig.update_layout(title="Attitude and Angular Velocity Evolution", height=800, width=1300)
# Move legend up and to the left slightly
fig.update_layout(legend=dict(x=1.015, y=1.1))
# Increase all font sizes
fig.update_layout(title=dict(font=dict(size=24)),
                  legend=dict(font=dict(size=20)))
# Increase font size of subplot titles
fig.update_annotations(font_size=20)
fig.update_xaxes(title_text="Time (s)", title_font=dict(size=18), tickfont=dict(size=14), row=1, col=1)
fig.update_xaxes(title_text="Time (s)", title_font=dict(size=18), tickfont=dict(size=14), row=2, col=1)
fig.update_yaxes(title_text="MRP", title_font=dict(size=18), tickfont=dict(size=14), row=1, col=1)
fig.update_yaxes(title_text="Angular Velocity (deg/s)", title_font=dict(size=18), tickfont=dict(size=14), row=2, col=1)
fig.update_layout(margin=dict(r=100))
fig.write_html('ASEN_5010/Final_Project/figures/task_7_rk4_results.html', include_mathjax='cdn')