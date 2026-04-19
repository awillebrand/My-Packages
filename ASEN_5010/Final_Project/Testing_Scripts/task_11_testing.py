import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from Project_Tools.RK4 import rk4
from Project_Tools.control_vector import compute_gains
from Project_Tools.attitude_error_eval import attitude_error_eval
from Project_Tools.sun_frame_funcs import sun_frame_dcm, sun_frame_angular_velocity
from Project_Tools.nadir_frame_dcm import nadir_frame_dcm, nadir_frame_angular_velocity
from Project_Tools.GMO_pointing_frame_funcs import GMO_pointing_frame_dcm, GMO_pointing_frame_angular_velocity
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
y = solution[0]

# Extract MRP from the solution at necessary times
times_of_interest = [300, 2100, 3400, 4400, 5600]

solutions_at_interest_times = np.array([solution[0][np.where(t == time)[0][0], 0:3] for time in times_of_interest])

mode_switch_times = [0, 1918, 3057, 4067, 5469]
modes = ['Sun Pointing', 'Nadir Pointing', 'GMO Pointing', 'Nadir Pointing', 'Sun Pointing']

attitude_errors = np.zeros((len(t), 6))  # Initialize vector to store attitude errors

for i, time in enumerate(t):
    for j in range(len(mode_switch_times)-1):
        
        if mode_switch_times[j] <= time < mode_switch_times[j+1]:
            current_mode = modes[j]
            if current_mode.lower() == 'sun pointing':
                reference_frame_dcm_func = sun_frame_dcm
                reference_frame_angular_velocity_func = sun_frame_angular_velocity
            elif current_mode.lower() == 'nadir pointing':
                reference_frame_dcm_func = nadir_frame_dcm
                reference_frame_angular_velocity_func = nadir_frame_angular_velocity
            else:
                reference_frame_dcm_func = GMO_pointing_frame_dcm
                reference_frame_angular_velocity_func = GMO_pointing_frame_angular_velocity
            break
        else:
            current_mode = modes[-1]  # default to last mode
            if current_mode.lower() == 'sun pointing':
                reference_frame_dcm_func = sun_frame_dcm
                reference_frame_angular_velocity_func = sun_frame_angular_velocity
            elif current_mode.lower() == 'nadir pointing':
                reference_frame_dcm_func = nadir_frame_dcm
                reference_frame_angular_velocity_func = nadir_frame_angular_velocity
            else:
                reference_frame_dcm_func = GMO_pointing_frame_dcm
                reference_frame_angular_velocity_func = GMO_pointing_frame_angular_velocity

    sigma_BN = y[i, :3]
    omega_BN = y[i, 3:]
    reference_frame_dcm = reference_frame_dcm_func(time)
    reference_frame_angular_velocity = reference_frame_angular_velocity_func(time)
    sigma_error, omega_error = attitude_error_eval(time, sigma_BN, omega_BN, reference_frame_dcm, reference_frame_angular_velocity)
    attitude_errors[i, :3] = sigma_error
    attitude_errors[i, 3:] = omega_error

mode_colors = {
    'Sun Pointing':   'rgba(255, 220, 100, 0.15)',
    'Nadir Pointing': 'rgba(100, 180, 255, 0.15)',
    'GMO Pointing':   'rgba(100, 255, 150, 0.15)',
}

def add_mode_regions(fig, mode_switch_times, modes, mode_colors):
    for i, mode in enumerate(modes):
        t_start = mode_switch_times[i]
        t_end   = mode_switch_times[i + 1] if i + 1 < len(mode_switch_times) else t[-1]
        fig.add_vrect(
            x0=t_start, x1=t_end,
            fillcolor=mode_colors[mode],
            layer='below',       # draw under the traces
            line_width=0,        # no border
            annotation_text=mode,
            annotation_position='bottom left',
            annotation_font=dict(size=16),
        )
    return fig

plot_style = dict(
    font=dict(family='DejaVu Sans, Arial, sans-serif', size=24),
    xaxis=dict(
        showgrid=True, gridcolor='white', gridwidth=1,
        showline=False, zeroline=False,
        tickfont=dict(size=18),
    ),
    yaxis=dict(
        showgrid=True, gridcolor='white', gridwidth=1,
        showline=False, zeroline=False,
         tickfont=dict(size=18)
    ),
    legend=dict(
        x=1.02, y=0.95, xanchor='left', yanchor='middle',
        bgcolor='rgba(0,0,0,0)', borderwidth=0, font=dict(size=30)
    ),
    margin=dict(l=70, r=120, t=70, b=60),
    height = 400,
    width = 1200
)
title_font = dict(family='DejaVu Sans, Arial, sans-serif', size=30)

fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=y[:, 0], mode='lines', name=r'$\Large\sigma_1$', line=dict(width=4)))
fig.add_trace(go.Scatter(x=t, y=y[:, 1], mode='lines', name=r'$\Large\sigma_2$', line=dict(width=4)))
fig.add_trace(go.Scatter(x=t, y=y[:, 2], mode='lines', name=r'$\Large\sigma_3$', line=dict(width=4)))
fig.update_layout(**plot_style,
                  title=dict(text='MRP Attitude Time History', font=title_font),
                  xaxis_title='Time (s)',
                  yaxis_title='MRP')
add_mode_regions(fig, mode_switch_times, modes, mode_colors)
fig.write_html(f"ASEN_5010/Final_Project/figures/simulation_MRP_attitude.html", include_mathjax='cdn')
fig.write_image(f"ASEN_5010/Final_Project/figures/simulation_MRP_attitude.png")

fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=y[:, 3], mode='lines', name=r'$\Large\omega_1$', line=dict(width=4)))
fig.add_trace(go.Scatter(x=t, y=y[:, 4], mode='lines', name=r'$\Large\omega_2$', line=dict(width=4)))
fig.add_trace(go.Scatter(x=t, y=y[:, 5], mode='lines', name=r'$\Large\omega_3$', line=dict(width=4)))
fig.update_layout(**plot_style,
                  title=dict(text='Angular Velocity Time History', font=title_font),
                  xaxis_title='Time (s)',
                  yaxis_title='Angular Velocity (rad/s)')
add_mode_regions(fig, mode_switch_times, modes, mode_colors)
fig.write_html(f"ASEN_5010/Final_Project/figures/simulation_angular_velocity.html", include_mathjax='cdn')
fig.write_image(f"ASEN_5010/Final_Project/figures/simulation_angular_velocity.png")

fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=attitude_errors[:, 0], mode='lines', name=r"$\Large\sigma_1$ Error", line=dict(width=4)))
fig.add_trace(go.Scatter(x=t, y=attitude_errors[:, 1], mode='lines', name=r"$\Large\sigma_2$ Error", line=dict(width=4)))
fig.add_trace(go.Scatter(x=t, y=attitude_errors[:, 2], mode='lines', name=r"$\Large\sigma_3$ Error", line=dict(width=4)))
fig.update_layout(**plot_style,
                  title=dict(text='MRP Attitude Error Time History', font=title_font),
                  xaxis_title='Time (s)',
                  yaxis_title='MRP Error')
add_mode_regions(fig, mode_switch_times, modes, mode_colors)
fig.write_html(f"ASEN_5010/Final_Project/figures/simulation_MRP_attitude_error.html", include_mathjax='cdn')
fig.write_image(f"ASEN_5010/Final_Project/figures/simulation_MRP_attitude_error.png")

fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=attitude_errors[:, 3], mode='lines', name=r'$\Large\omega_1$ Error', line=dict(width=4)))
fig.add_trace(go.Scatter(x=t, y=attitude_errors[:, 4], mode='lines', name=r'$\Large\omega_2$ Error', line=dict(width=4)))
fig.add_trace(go.Scatter(x=t, y=attitude_errors[:, 5], mode='lines', name=r'$\Large\omega_3$ Error', line=dict(width=4)))
fig.update_layout(**plot_style,
                  title=dict(text='Angular Velocity Error Time History', font=title_font),
                  xaxis_title='Time (s)',
                  yaxis_title='Angular Velocity Error (rad/s)')
add_mode_regions(fig, mode_switch_times, modes, mode_colors)
fig.write_html(f"ASEN_5010/Final_Project/figures/simulation_angular_velocity_error.html", include_mathjax='cdn')
fig.write_image(f"ASEN_5010/Final_Project/figures/simulation_angular_velocity_error.png")

# Plot trajectories and color them based on the mode that was used at each time step
LMO_pos = solution[1]
GMO_pos = solution[2]
mode_list = solution[3]

# Make data frame from the lists
data = {
    'Time': t,
    'LMO_X': LMO_pos[:, 0],
    'LMO_Y': LMO_pos[:, 1],
    'LMO_Z': LMO_pos[:, 2],
    'GMO_X': GMO_pos[:, 0],
    'GMO_Y': GMO_pos[:, 1],
    'GMO_Z': GMO_pos[:, 2],
    'Mode': mode_list,
    'GMO Trajectory': ['GMO Trajectory'] * len(t)  # GMO trajectory will just be colored red
}
df = pd.DataFrame(data)

mode_colors = {
    'Sun Pointing':   'rgba(255, 220, 100, 1)',
    'Nadir Pointing': 'rgba(100, 180, 255, 1)',
    'GMO Pointing':   'rgba(100, 255, 150, 1)',
    'GMO Trajectory': 'rgba(255, 0, 0, 1)'
}

# Plot using plotly express
fig_lmo = px.scatter_3d(df, x='LMO_X', y='LMO_Y', z='LMO_Z', color='Mode',
                    title='LMO Spacecraft Trajectory Colored by Pointing Mode',
                    labels={'LMO_X': 'LMO X Position (km)', 'LMO_Y': 'LMO Y Position (km)', 'LMO_Z': 'LMO Z Position (km)'},
                    color_discrete_map=mode_colors)
# GMO color should just be red
fig_gmo = px.scatter_3d(df, x='GMO_X', y='GMO_Y', z='GMO_Z', color='GMO Trajectory',
                    title='GMO Spacecraft Trajectory',
                    labels={'GMO_X': 'GMO X Position (km)', 'GMO_Y': 'GMO Y Position (km)', 'GMO_Z': 'GMO Z Position (km)'},
                    color_discrete_map=mode_colors)
# Combine the two plots into one figure showing both trajectories
fig = go.Figure(data=fig_lmo.data + fig_gmo.data)
fig.update_layout(
    **plot_style,
    title=dict(text='Spacecraft Trajectory Colored by Pointing Mode', font=title_font),
    scene=dict(
        xaxis_title='Position X (km)',
        yaxis_title='Position Y (km)',
        zaxis_title='Position Z (km)',
        aspectmode='data'
    )
)
fig.update_layout(height=650, width=1600,
                  scene=dict(
                      xaxis=dict(showgrid=True, gridcolor='white', gridwidth=1, showline=False, zeroline=False, tickfont=dict(size=12)),
                      yaxis=dict(showgrid=True, gridcolor='white', gridwidth=1, showline=False, zeroline=False, tickfont=dict(size=12)),
                      zaxis=dict(showgrid=True, gridcolor='white', gridwidth=1, showline=False, zeroline=False, tickfont=dict(size=12))
                  ))
fig.write_html(f"ASEN_5010/Final_Project/figures/simulation_trajectory_colored_by_mode.html", include_mathjax='cdn')

# Save solutions at interest times to files
for i, time in enumerate(times_of_interest):
    with open(os.path.join(os.path.dirname(__file__), '..', 'coursera_validation_files', f'task_11_solution_time_{time}.txt'), 'w') as f:
        f.write(' '.join(str(x) for x in solutions_at_interest_times[i,:3]))