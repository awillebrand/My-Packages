# scenario.py
"""
Scenario configuration for monte_carlo_gen.py.
Edit this file to define your simulation parameters.
"""

import numpy as np
from Tools.generic_functions import keplerian_to_cartesian
from constants import mu

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
initial_pos, initial_vel = keplerian_to_cartesian(
    mu=mu,             # Gravitational parameter (km^3/s^2)
    a=7000.0,          # Semi-major axis (km)
    e=0.001,           # Eccentricity
    i=30.0,            # Inclination (degrees)
    LoN=0.0,           # Right Ascension of Ascending Node (degrees)
    AoP=0.0,           # Argument of Perigee (degrees)
    f=0.0              # True Anomaly (degrees)
)

initial_state = np.hstack((initial_pos, initial_vel))

# ─────────────────────────────────────────────
# INITIAL COVARIANCE
# ─────────────────────────────────────────────
pos_sigma = 1.0    # km
vel_sigma = 0.001   # km/s

initial_covariance = np.diag([
    pos_sigma**2, pos_sigma**2, pos_sigma**2,
    vel_sigma**2, vel_sigma**2, vel_sigma**2
])

# ─────────────────────────────────────────────
# MONTE CARLO SIMULATION PARAMETERS
# ─────────────────────────────────────────────
NUM_TRAJECTORIES = 1000