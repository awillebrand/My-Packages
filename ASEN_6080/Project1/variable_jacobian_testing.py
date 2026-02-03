from ASEN_6080.Tools import state_jacobian
import numpy as np
# DONT FORGET THAT DEFAULT RADIUS IS 6378 KM IN INTEGRATOR AND TOOLS

sat_state = np.array([757700.0E-3, 5222607.0E-3, 4851500.0E-3, 2213.21E-3, 4678.34E-3, -5371.30E-3])  # Example satellite state in km and km/s
mu = 3.986004415E5  # Earth's gravitational parameter in km^3/s^2
J2 = 1.082626925638815E-3 # Earth's J2 coefficient
J3 = 0.0 # Earth's J3 coefficient
R_e = 6378.1363  # Earth's radius in km
C_d = 2.0 # Drag coefficient
station_positions_ecef = np.array([[-5127510.0E-3, -3794160.0E-3,  0.0],
                                    [3860910.0E-3, 3238490.0E-3,  3898094.0E-3],
                                    [549505.0E-3, -1380872.0E-3,  6182197.0E-3]])  # Example ground station positions in ECEF coordinates

jacobian = state_jacobian(sat_state[0:3], sat_state[3:6], mu=mu, J2=J2, J3=J3, R_e=R_e, C_d=C_d, station_positions_ecef=station_positions_ecef, mode=['mu','J2','Drag','Stations'])