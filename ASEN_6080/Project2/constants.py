import numpy as np

"""
This file contains constants used in the ASEN 6080 Project 2 code. These constants include physical
parameters,initial conditions, and any other fixed values that are relevant to the project.
"""
# Truth data file path
truth_data_file_path = 'ASEN_6080\Project2\data\Project2_Prob2_truth_traj_50days.mat'
known_dynamics_measurement_file_path = 'ASEN_6080\Project2\data\Project2a_Obs.txt'
unknown_dynamics_measurement_file_path = 'ASEN_6080\Project2\data\Project2b_Obs.txt'

# Physical Constants
mu_sun = 132712440017.987                   # Gravitational parameter of the Sun, km^3/s^2
mu_earth = 3.986004415E5                    # Gravitational parameter of the Earth, km^3/s^2

R_e = 6378.1363                             # Earth's radius in km
solar_flux = 1357                           # Solar radiation pressure flux at 1 AU in W/m^2
SRP_area_to_mass = 0.01                     # Area-to-mass ratio for solar radiation pressure in m^2/kg
AU = 149597870.7                            # Astronomical Unit in km

RSOI = 925000.0                               # Radius of Sphere of Influence for Earth in km

B_plane_target_coords = [9796.737, 14970.824]

# Initial Conditions
initial_epoch = 0                   # Initial epoch in seconds
initial_epoch_jd = 2456296.25       # Initial epoch in Julian Date (J2000)

# Measurement Constants
initial_spin_angle = 0.0                    # Initial Earth spin angle in radians
earth_spin_rate = 7.29211585275553E-5   # Earth's rotation rate in rad per second

station_locations = {
    'DSS34': {'lat': -35.398333, 'lon': 148.981944, 'radius': 0.691750 + R_e},  # Canberra, Australia
    'DSS65': {'lat': 40.427222, 'lon': 355.749444, 'radius': 0.834539 + R_e}, # Madrid, Spain
    'DSS13': {'lat': 35.247164, 'lon': 243.205, 'radius': 1.07114904 + R_e}   # Goldstone, CA
}

part_2_station_locations = {
    'DSS34': {'lat': -35.398333, 'lon': 148.981944, 'radius': 0.691750 + R_e},  # Canberra, Australia
    'DSS65': {'lat': 40.427222, 'lon': -355.749444, 'radius': 0.834539 + R_e}, # Madrid, Spain
    'DSS13': {'lat': 35.247164, 'lon': 243.205, 'radius': 1.07114904 + R_e}   # Goldstone, CA
}

observation_noise = np.diag([0.005, 5e-7])**2  # Range noise: 5 meters, Range rate noise: 0.5 mm/s

# Part 2 a priori state estimates

testing_x = -274096790.0
testing_y = -92859240.0
testing_z = -40199490.0
testing_vx = 32.67
testing_vy = -8.94
testing_vz = -3.88
testing_C_r = 1.2
testing_a_priori_state = np.array([testing_x, testing_y, testing_z, testing_vx, testing_vy, testing_vz, testing_C_r])

testing_a_priori_covariance = np.diag([100, 100, 100, 0.1, 0.1, 0.1, 0.1])**2

# Part 3 a priori state estimates

x = -274096770.76544
y = -92859266.4499061
z = -40199493.6677441
vx = 32.6704564599943
vy = -8.93838913761049
vz = -3.87881914050316
C_r = 1.0

mu_earth_covariance = 1e5
a_priori_state = np.array([x, y, z, vx, vy, vz, C_r, mu_earth])
a_priori_covariance = np.diag([100, 100, 100, 0.1, 0.1, 0.1, 0.1, mu_earth_covariance])**2

