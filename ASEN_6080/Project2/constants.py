import numpy as np

"""
This file contains constants used in the ASEN 6080 Project 2 code. These constants include physical
parameters,initial conditions, and any other fixed values that are relevant to the project.
"""
# Truth data file path
truth_data_file_path = 'ASEN_6080\Project2\data\Project2_Prob2_truth_traj_50days.mat'
known_dynamics_measurement_file_path = 'ASEN_6080\Project2\data\Project2a_Obs.txt'

# Physical Constants
mu_sun = 132712440017.987                   # Gravitational parameter of the Sun, km^3/s^2
mu_earth = 3.986004415E5                    # Gravitational parameter of the Earth, km^3/s^2
R_e = 6378.1363                             # Earth's radius in km
solar_flux = 1357                           # Solar radiation pressure flux at 1 AU in W/m^2
SRP_area_to_mass = 0.01                     # Area-to-mass ratio for solar radiation pressure in m^2/kg
AU = 149597870.7                            # Astronomical Unit in km

# Initial Conditions
initial_epoch = 0                   # Initial epoch in seconds
initial_epoch_jd = 2456296.25       # Initial epoch in Julian Date (J2000)

# Measurement Constants
initial_spin_angle = 0.0                    # Initial Earth spin angle in radians
earth_spin_rate = 7.29211585275553E-5   # Earth's rotation rate in rad per second

station_locations = {
    'DSS34': {'lat': -35.398333, 'lon': 148.981944, 'alt': 0.691750},  # Canberra, Australia
    'DSS65': {'lat': 40.427222, 'lon': 355.749444, 'alt': 0.834539}, # Madrid, Spain
    'DSS13': {'lat': 35.247164, 'lon': 243.205, 'alt': 1.07114904}   # Goldstone, CA
}

part_2_station_locations = {
    'DSS34': {'lat': -35.398333, 'lon': 148.981944, 'alt': 0.691750},  # Canberra, Australia
    'DSS65': {'lat': 40.427222, 'lon': -355.749444, 'alt': 0.834539}, # Madrid, Spain
    'DSS13': {'lat': 35.247164, 'lon': 243.205, 'alt': 1.07114904}   # Goldstone, CA
}

observation_noise = np.array([0.005, 0.0005])  # Range noise: 5 meters, Range rate noise: 0.5 mm/s

