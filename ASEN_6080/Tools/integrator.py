import numpy as np
from .generic_functions import state_jacobian, compute_density
from scipy.integrate import solve_ivp
class Integrator:
    def __init__(self, mu : float, R_e : float, mode : list = [], parameter_indices : list = [], spacecraft_area : float = None, spacecraft_mass : float = None, number_of_stations : int = 0, earth_spin_rate : float = 7.2921158553E-5):
        """
        Initializes the Integrator class for spacecraft orbit propagation.
        Parameters:
        mu : float
            Gravitational parameter of the central body (e.g., Earth) in km^3/s^2.
        R_e : float
            Radius of the central body (e.g., Earth) in km.
        mode : list, optional
            List of perturbation modes to include in the integration. Options are 'PointMass', 'J2', 'J3', 'Drag', and 'Stations'. Default is an empty list.
        parameter_indices : list, optional
            List of indices for parameters to be estimated during integration. The entry for station parameters is a list of the indices in the state vector. Default is an empty list.
        spacecraft_area : float, optional
            Cross-sectional area of the spacecraft in m^2, required if 'Drag' is included in mode. Default is None.
        spacecraft_mass : float, optional
            Mass of the spacecraft in kg, required if 'Drag' is included in mode. Default is None.
        number_of_stations : int, optional
            Number of ground stations being used, required if 'Stations' is included in mode. Default is 0.
        """
        self.mu = mu
        self.R_e = R_e
        self.mode = mode
        self.parameter_indices = parameter_indices
        self.spacecraft_area = spacecraft_area * 1e-6 if spacecraft_area is not None else 0  # Convert from m^2 to km^2 <---- DOUBLE CHECK THIS CONVERSION
        self.spacecraft_mass = spacecraft_mass if spacecraft_mass is not None else 1
        self.number_of_stations = number_of_stations
        self.earth_spin_rate = earth_spin_rate

        if set(mode).isdisjoint({'mu', 'J2', 'J3', 'Drag', 'Stations'}):
            raise ValueError("Invalid mode specified. Choose from 'mu', 'J2', 'J3', 'Drag', and/or 'Stations'.")
        if len(mode) != len(parameter_indices):
            raise ValueError("Length of mode and parameter_indices must be the same.")
        if 'Drag' in mode and (spacecraft_area is None or spacecraft_mass is None):
            raise ValueError("Spacecraft area and mass must be provided for drag calculations.")
        if 'Stations' in mode and number_of_stations <= 0:
            raise ValueError("Number of stations must be greater than zero when 'Stations' mode is selected.")

    def keplerian_to_cartesian(self, a, e, i, LoN, AoP, f):
        # Compute perifocal radius magnitude
        r_mag = a * (1 - e**2) / (1 + e * np.cos(f))

        # Compute perifocal radius vector
        r_perifocal = r_mag * np.array([np.cos(f), np.sin(f), 0])

        # Compute perifocal velocity vector
        h = np.sqrt(self.mu * a * (1 - e**2))
        v_perifocal = (self.mu / h) * np.array([-np.sin(f), e + np.cos(f), 0])
        
        # Rotation matrices
        DCM = np.array([[np.cos(LoN) * np.cos(AoP) - np.sin(LoN) * np.sin(AoP) * np.cos(i), -np.cos(LoN) * np.sin(AoP) - np.sin(LoN) * np.cos(AoP) * np.cos(i),  np.sin(LoN) * np.sin(i)],
                        [np.sin(LoN) * np.cos(AoP) + np.cos(LoN) * np.sin(AoP) * np.cos(i), -np.sin(LoN) * np.sin(AoP) + np.cos(LoN) * np.cos(AoP) * np.cos(i), -np.cos(LoN) * np.sin(i)],
                        [np.sin(AoP) * np.sin(i), np.cos(AoP) * np.sin(i), np.cos(i)]])
 
        # Transform to inertial frame
        r_inertial = DCM @ r_perifocal
        v_inertial = DCM @ v_perifocal

        return r_inertial, v_inertial
    
    def cartesian_to_keplerian(self, r_vec, v_vec):
        # Define unit vectors
        x = np.array([1, 0, 0])
        y = np.array([0, 1, 0])
        z = np.array([0, 0, 1])

        # Compute orbital elements
        h_vec = np.cross(r_vec, v_vec)
        h = np.linalg.norm(h_vec)
        h_norm = h_vec / h

        e_vec = (1/self.mu) * np.cross(v_vec, h_vec) - (r_vec / np.linalg.norm(r_vec))
        e = np.linalg.norm(e_vec)
        e_norm = e_vec / e
        e_vec_perp = np.cross(h_norm, e_norm)

        p = np.linalg.norm(h)**2 / self.mu
        a = p / (1 - e**2)
        i = np.arccos(np.dot(h_norm, z))

        node_vec = np.cross(z, h_norm) / np.linalg.norm(np.cross(z, h_norm))
        node_vec_perp = np.cross(h_norm, node_vec)

        LoN = np.arctan2(np.dot(y, node_vec), np.dot(x, node_vec))
        AoP = np.arctan2(np.dot(e_vec, node_vec_perp), np.dot(e_vec, node_vec))
    
        f = np.arctan2(np.dot(r_vec, e_vec_perp), np.dot(r_vec, e_vec))

        return a, e, i, LoN, AoP, f
    
    def equations_of_motion(self, t, state):
        mu = self.mu
        x, y, z = state[0:3]
        u, v, w = state[3:6]
        r = np.sqrt(x**2 + y**2 + z**2)
        J2 = 0
        J3 = 0
        Cd = 0
        spacecraft_area = 0
        spacecraft_mass = 1
        rho = compute_density(r)* 1e9 # Convert from kg/m^3 to kg/km^3 <---- DOUBLE CHECK THIS CONVERSION
        # Determine J2, J3, and Cd based on mode
        if 'mu' in self.mode:
            param_index = self.parameter_indices[self.mode.index('mu')]
            mu = state[param_index]
        if 'J2' in self.mode:
            param_index = self.parameter_indices[self.mode.index('J2')]
            J2 = state[param_index]
        if 'J3' in self.mode:
            param_index = self.parameter_indices[self.mode.index('J3')]
            J3 = state[param_index]
        if 'Drag' in self.mode:
            if self.spacecraft_area is None or self.spacecraft_mass is None:
                raise ValueError("Area and mass must be provided for drag calculation.")
            param_index = self.parameter_indices[self.mode.index('Drag')]
            Cd = state[param_index]
            spacecraft_area = self.spacecraft_area
            spacecraft_mass = self.spacecraft_mass
        if 'Stations' in self.mode:
            # Determine number of station variables, this is stored in the parameter_indices value for stations as a list
            num_station_vars = self.number_of_stations * 3
            
        x_dot = u
        y_dot = v
        z_dot = w
        u_dot = -mu * x / r**3 + (3 / 2) * (mu * J2 * self.R_e**2 * x / r**5) * (5 * (z**2 / r**2) - 1) + (5 / 2) * mu * J3 * self.R_e**3 * x * z / r**7 * (7 * z**2 / r**2 - 3)
        v_dot = -mu * y / r**3 + (3 / 2) * (mu * J2 * self.R_e**2 * y / r**5) * (5 * (z**2 / r**2) - 1) + (5 / 2) * mu * J3 * self.R_e**3 * y * z / r**7 * (7 * z**2 / r**2 - 3)
        w_dot = -mu * z / r**3 + (3 / 2) * (mu * J2 * self.R_e**2 * z / r**5) * (5 * (z**2 / r**2) - 3) + (5 / 2) * mu * J3 * self.R_e**3 / r**5 * (7 * z**4 / r**4 - 6 * z**2 / r**2 + 3 / 5)

        if 'Drag' in self.mode:
            V_rel = np.array([u + self.earth_spin_rate * y, v - self.earth_spin_rate * x, w])
            u_rel, v_rel, w_rel = V_rel
            V_rel_norm = np.linalg.norm(np.array([u, v, w]))
            u_dot_drag = -(rho * Cd * spacecraft_area * V_rel_norm * u_rel) / (2 * spacecraft_mass)
            v_dot_drag = -(rho * Cd * spacecraft_area * V_rel_norm * v_rel) / (2 * spacecraft_mass)
            w_dot_drag = -(rho * Cd * spacecraft_area * V_rel_norm * w_rel) / (2 * spacecraft_mass)

            u_dot += u_dot_drag
            v_dot += v_dot_drag
            w_dot += w_dot_drag

        output = np.array([x_dot, y_dot, z_dot, u_dot, v_dot, w_dot])
        if 'mu' in self.mode:
            output = np.append(output, 0)
        if 'J2' in self.mode:
            output = np.append(output, 0)
        if 'J3' in self.mode:
            output = np.append(output, 0)
        if 'Drag' in self.mode:
            output = np.append(output, 0)
        if 'Stations' in self.mode:
            for _ in range(num_station_vars):
                output = np.append(output, 0)
        return output
    
    def full_dynamics(self, t, augmented_state):
        # This function is passed through the integrator when the initial state is augmented by the STM

        # Determine state length based on mode and assign J2 and J3 according to mode
        mu = self.mu
        J2 = 0
        J3 = 0
        Cd = 0
        station_positions_ecef = np.array([])
        state_length = 6
        # Determine J2, J3, and Cd based on mode
        if 'mu' in self.mode:
            state_length += 1
            param_index = self.parameter_indices[self.mode.index('mu')]
            mu = augmented_state[param_index]
        if 'J2' in self.mode:
            state_length += 1
            param_index = self.parameter_indices[self.mode.index('J2')]
            J2 = augmented_state[param_index]
        if 'J3' in self.mode:
            state_length += 1
            param_index = self.parameter_indices[self.mode.index('J3')]
            J3 = augmented_state[param_index]
        if 'Drag' in self.mode:
            state_length += 1
            param_index = self.parameter_indices[self.mode.index('Drag')]
            Cd = augmented_state[param_index]
        if 'Stations' in self.mode:
            # Determine number of station variables, this is stored in the parameter_indices value for stations
            param_index = self.parameter_indices[self.mode.index('Stations')]
            num_station_vars = self.number_of_stations * 3
            station_positions_vector = augmented_state[param_index:param_index+num_station_vars]
            state_length += num_station_vars
            # For consistency sake, pull out station variables but they are not used in dynamics
            station_positions_ecef = np.zeros((self.number_of_stations, 3))
            for i in range(self.number_of_stations):
                station_positions_ecef[i, :] = station_positions_vector[3*i:3*i+3]
                
        state = augmented_state[0:state_length]
        phi_flat = augmented_state[state_length:]
        phi = phi_flat.reshape((state_length, state_length))

        # Compute state derivatives
        state_dot = self.equations_of_motion(t, state)

        # Compute STM derivative
        A = state_jacobian(state[0:3], state[3:6], mu, J2, J3, Cd, station_positions_ecef, self.R_e, mode=self.mode, spacecraft_area=self.spacecraft_area, spacecraft_mass=self.spacecraft_mass)
        phi_dot = A @ phi
        phi_dot_flat = phi_dot.flatten()

        return np.hstack((state_dot, phi_dot_flat))
        
    def integrate_eom(self, t_final, initial_state, teval = None):
        """Integrate the equations of motion for the spacecraft.
        Parameters:
        t_final : float
            Final time for integration in seconds.
        initial_state : np.array
            nx1 array of initial spacecraft state in ECI frame. First 6 elements are [x, y, z, u, v, w] in km and km/s.
        teval : np.array, optional
            1xN array of time points at which to store the computed solution. Default is None.
        Returns:
        time_vector : np.array
            1xN array of time points corresponding to the spacecraft states.
        state_history : np.array
            nxN array of spacecraft states over time in ECI frame."""
        
        t_span = (0, t_final)
        sol = solve_ivp(self.equations_of_motion, t_span, initial_state, method='RK45', rtol=1e-13, atol=1e-13, t_eval=teval)
        return sol.t, sol.y
    
    def integrate_stm(self, t_final, initial_state, phi_0 = None, teval = None, initial_time : float = 0):
        # Determine state length based on mode
        state_length = 6

        # Determine J2, J3, and Cd based on mode
        if 'mu' in self.mode:
            state_length += 1
        if 'J2' in self.mode:
            state_length += 1
        if 'J3' in self.mode:
            state_length += 1
        if 'Drag' in self.mode:
            state_length += 1
        if 'Stations' in self.mode:
            param_index = self.parameter_indices[self.mode.index('Stations')]
            num_station_vars = len(initial_state[param_index:])
            state_length += num_station_vars

        # Initialize STM as identity matrix
        if phi_0 is None:
            phi_0 = np.eye(state_length).flatten()

        augmented_initial_state = np.hstack((initial_state, phi_0))
        t_span = (initial_time, t_final)
        sol = solve_ivp(self.full_dynamics, t_span, augmented_initial_state, method='RK45', rtol=1e-13, atol=1e-13, t_eval=teval)
        return sol.t, sol.y