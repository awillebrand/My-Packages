# scenario.py
"""
Scenario configuration for monte_carlo_gen.py.
Edit this file to define your simulation parameters.
"""

import numpy as np

# ─────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────
FILE_PATH = "ASEN_6080/HW8/figures"

# ─────────────────────────────────────────────
# TIME VECTOR
# ─────────────────────────────────────────────
T_END = 24 * 3600.0   # seconds
DT    = 60.0          # seconds

time_vec = np.arange(0.0, T_END + DT, DT)

# ─────────────────────────────────────────────
# INITIAL STATE  [x, y, z (km), vx, vy, vz (km/s)]
# ─────────────────────────────────────────────
initial_state = np.array([
    6878.0, 0.0, 0.0,
    0.0,    7.612, 0.0
])

# ─────────────────────────────────────────────
# INITIAL COVARIANCE
# ─────────────────────────────────────────────
pos_sigma = 1.0    # km
vel_sigma = 0.01   # km/s

initial_covariance = np.diag([
    pos_sigma**2, pos_sigma**2, pos_sigma**2,
    vel_sigma**2, vel_sigma**2, vel_sigma**2
])

# ─────────────────────────────────────────────
# MONTE CARLO SIMULATION PARAMETERS
# ─────────────────────────────────────────────
NUM_TRAJECTORIES = 1000