import numpy as np
from .generic_functions import state_jacobian, compute_density, compute_consider_parameter_partials
from .ephemeris_manager import EphemerisMgr
from scipy.integrate import solve_ivp
class IntegratorOld:
    def __init__(self, mu : float, R_e : float, J2 : float = 0, J3 : float = 0, Cd : float = 0, mode : list = [], parameter_indices : list = [], spacecraft_area : float = None, spacecraft_mass : float = None, number_of_stations : int = 0, earth_spin_rate : float = 7.2921158553E-5, solar_flux : float = 1357.0, initial_epoch : float = 0):
        """
        Initializes the Integrator class for spacecraft orbit propagation.
        Parameters:
        mu : float
            Gravitational parameter of the central body (e.g., Earth) in km^3/s^2.
        R_e : float
            Radius of the central body (e.g., Earth) in km.
        J2 : float, optional
            Second zonal harmonic coefficient for the central body's gravity field. Default is 0 (no J2 perturbation).
        J3 : float, optional
            Third zonal harmonic coefficient for the central body's gravity field. Default is 0 (no J3 perturbation).
        Cd : float, optional
            Drag coefficient for the spacecraft, used in atmospheric drag calculations. Default is 0 (no drag).
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
        earth_spin_rate : float, optional
            Angular velocity of the Earth's rotation in rad/s, used for drag calculations. Default is 7.2921158553E-5 rad/s.
        solar_flux : float, optional
            Solar flux at 1 AU in W/m^2, used for solar radiation pressure calculations. Default is 1357.0 W/m^2.
        initial_epoch : float, optional
            Initial epoch in Julian days for the integration, used for evaluating planetary positions if needed. Default is 0.
        Raises:
            ValueError: If an invalid mode is specified, if the length of mode and parameter_indices do not match, if spacecraft area and mass are not provided for drag calculations, or if the number of stations is not greater than zero when 'Stations' mode is selected.
        """

        self.mu = mu
        self.J2 = J2
        self.J3 = J3
        self.Cd = Cd
        self.R_e = R_e
        self.mode = mode
        self.parameter_indices = parameter_indices
        self.spacecraft_area = spacecraft_area * 1e-6 if spacecraft_area is not None else 0  # Convert from m^2 to km^2 <---- DOUBLE CHECK THIS CONVERSION
        self.spacecraft_mass = spacecraft_mass if spacecraft_mass is not None else 1
        self.number_of_stations = number_of_stations
        self.earth_spin_rate = earth_spin_rate
        self.solar_flux = solar_flux
        self.initial_epoch = initial_epoch

        # if set(mode).isdisjoint({'mu', 'J2', 'J3', 'Drag', 'Stations'}):
        #     raise ValueError("Invalid mode specified. Choose from 'mu', 'J2', 'J3', 'Drag', and/or 'Stations'.")
        if len(mode) != len(parameter_indices):
            raise ValueError("Length of mode and parameter_indices must be the same.")
        if 'Drag' in mode and (spacecraft_area is None or spacecraft_mass is None):
            raise ValueError("Spacecraft area and mass must be provided for drag calculations.")
        if 'SRP' in mode and (spacecraft_area is None or spacecraft_mass is None):
            raise ValueError("Spacecraft area and mass must be provided for SRP calculations.")
        if 'Stations' in mode and number_of_stations <= 0:
            raise ValueError("Number of stations must be greater than zero when 'Stations' mode is selected.")
    
    def equations_of_motion(self, t : float, state : np.ndarray, DMC : bool = False, beta_mat : np.ndarray = None, sigma_points : bool = False):
        """
        Computes the time derivative of the state vector for a spacecraft under various perturbations.
        Parameters:
        t : float
            Current time in seconds.
        state : np.ndarray
            State vector of the spacecraft. The first 6 elements must be [x, y, z, u, v, w] in km and km/s. Additional elements can include parameters for estimation based on the mode.
        DMC : bool, optional
            If True, include dynamic model compensation terms in the equations of motion. Default is False.
        beta_mat : np.ndarray, optional
            3x3 diagonal matrix of time constants for dynamic model compensation. Required if DMC is True. Default is None.
        sigma_points : bool, optional
            If True, the input state is of length L(2L+1) and represents the sigma points for the UKF. The function will return the time derivative of each sigma point. Default is False.
        Returns:
        state_dot : np.ndarray
            Time derivative of the state vector.
        """
        if sigma_points == False:
            mu = self.mu
            x, y, z = state[0:3]
            u, v, w = state[3:6]
            r = np.sqrt(x**2 + y**2 + z**2)
            J2 = self.J2
            J3 = self.J3
            Cd = self.Cd
            spacecraft_area = self.spacecraft_area
            spacecraft_mass = self.spacecraft_mass
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
                param_index = self.parameter_indices[self.mode.index('Drag')]
                Cd = state[param_index]
            if 'SRP' in self.mode:
                param_index = self.parameter_indices[self.mode.index('SRP')]
                Cr = state[param_index]
            if 'Stations' in self.mode:
                # Determine number of station variables, this is stored in the parameter_indices value for stations as a list
                num_station_vars = self.number_of_stations * 3
                
            x_dot = u
            y_dot = v
            z_dot = w
            u_dot = -mu * x / r**3 + (3 / 2) * (mu * J2 * self.R_e**2 * x / r**5) * (5 * (z**2 / r**2) - 1) + (5 / 2) * mu * J3 * self.R_e**3 * x * z / r**7 * (7 * z**2 / r**2 - 3)
            v_dot = -mu * y / r**3 + (3 / 2) * (mu * J2 * self.R_e**2 * y / r**5) * (5 * (z**2 / r**2) - 1) + (5 / 2) * mu * J3 * self.R_e**3 * y * z / r**7 * (7 * z**2 / r**2 - 3)
            w_dot = -mu * z / r**3 + (3 / 2) * (mu * J2 * self.R_e**2 * z / r**5) * (5 * (z**2 / r**2) - 3) + (5 / 2) * mu * J3 * self.R_e**3 / r**5 * (7 * z**4 / r**4 - 6 * z**2 / r**2 + 3 / 5)

            if Cd != None:
                
                V_rel = np.array([u + self.earth_spin_rate * y, v - self.earth_spin_rate * x, w])
                u_rel, v_rel, w_rel = V_rel
                V_rel_norm = np.linalg.norm(np.array([u, v, w]))
                u_dot_drag = -(rho * Cd * spacecraft_area * V_rel_norm * u_rel) / (2 * spacecraft_mass)
                v_dot_drag = -(rho * Cd * spacecraft_area * V_rel_norm * v_rel) / (2 * spacecraft_mass)
                w_dot_drag = -(rho * Cd * spacecraft_area * V_rel_norm * w_rel) / (2 * spacecraft_mass)

                u_dot += u_dot_drag
                v_dot += v_dot_drag
                w_dot += w_dot_drag

            if 'SRP' in self.mode:
                P_solar = self.solar_flux / 299792458.0  # Solar radiation pressure at 1 AU in N/m^2

                # For SRP calculation, we need the position of the Sun. We can use the EphemerisMgr to get this based on the initial epoch and current time.
                ephemeris_mgr = EphemerisMgr('Earth')
                earth_state = ephemeris_mgr.evaluate_state(self.initial_epoch + t / 86400)  # Convert time from seconds to days for ephemeris evaluation
                r_sun_earth = earth_state[0:3]  # Position of earth relative to sun in EME2000 frame (km)

                r_earth_sc = np.array([x, y, z])  # Spacecraft position relative to Earth in EME2000 frame (km)

                r_sun_sc = r_sun_earth + r_earth_sc  # Spacecraft position relative to Sun in EME2000 frame (km)
                r_sun_sc_norm = np.linalg.norm(r_sun_sc)

                # Compute SRP acceleration
                AU_KM = 149597870.700
                R_AU = r_sun_sc_norm / AU_KM  # dimensionless

                accel_srp_norm = P_solar * Cr * spacecraft_area / (R_AU**2 * spacecraft_mass) # Convert r_sun_sc_norm from km to m for SRP calculation
                accel_srp = accel_srp_norm / 1000 * (r_sun_sc / r_sun_sc_norm)  # Acceleration vector in km/s^2
                u_dot += accel_srp[0]
                v_dot += accel_srp[1]
                w_dot += accel_srp[2]

            if DMC:
                # Add DMC terms to the equations of motion, these are simple linear damping terms on the velocity components with time constants specified by beta_mat
                w_1, w_2, w_3 = state[-3:]
                u_dot += w_1
                v_dot += w_2
                w_dot += w_3

                w_1_dot = -beta_mat[0,0] * w_1
                w_2_dot = -beta_mat[1,1] * w_2
                w_3_dot = -beta_mat[2,2] * w_3

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

            if DMC:
                output = np.append(output, [w_1_dot, w_2_dot, w_3_dot]) 

        else:
            L = 0.25 * (-1 + np.sqrt(1 + 8 * len(state)))
            for i in range(2 * int(L) + 1):
                state_i = state[i*int(L):(i+1)*int(L)]
                state_dot_i = self.equations_of_motion(t, state_i, DMC=DMC, beta_mat=beta_mat, sigma_points=False)
                if i == 0:
                    output = state_dot_i
                else:
                    output = np.hstack((output, state_dot_i))
        return output
    
    def full_dynamics(self, t, augmented_state, DMC : bool = False , beta_mat : np.ndarray = None, consider_parameters : list = []):
        """
        Computes the time derivative of the augmented state vector, which includes both the spacecraft state and the state transition matrix (STM) for variational equations.
        Parameters:
        t : float
            Current time in seconds.
        augmented_state : np.ndarray
            Augmented state vector of the spacecraft and STM. The first n elements are the spacecraft state, and the remaining elements are the flattened STM (n x n).
        DMC : bool, optional
            If True, include dynamic model compensation terms in the equations of motion. Default is False.
        beta_mat : np.ndarray, optional
            3x3 diagonal matrix of time constants for dynamic model compensation. Required if DMC is True. Default is None.
        consider_parameters : list, optional
            List of parameters to compute sensitivity for. This is used to determine what partials need to be computed for theta propagation. Default is an empty list.
        Returns:
        augmented_state_dot : np.ndarray
            Time derivative of the augmented state vector, including both the spacecraft state derivatives and the STM derivatives.
        """
        # This function is passed through the integrator when the initial state is augmented by the STM

        # Determine state length based on mode and assign J2 and J3 according to mode
        mu = self.mu
        J2 = self.J2
        J3 = self.J3
        Cd = self.Cd
        Cr = 0
        spacecraft_area = self.spacecraft_area
        spacecraft_mass = self.spacecraft_mass
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
        if DMC:
            state_length += 3

        state = augmented_state[0:state_length]
        if len(consider_parameters) == 0:
            phi_flat = augmented_state[state_length:]
            phi = phi_flat.reshape((state_length, state_length))
            # Compute state derivatives
            state_dot = self.equations_of_motion(t, state, DMC=DMC, beta_mat=beta_mat)

            # Compute STM derivative
            if self.mode == []:
                A = state_jacobian(state[0:3],
                                   state[3:6],
                                   mu=mu,
                                   R_e=self.R_e,
                                   mode=['BaseMat'],
                                   spacecraft_area=self.spacecraft_area,
                                   spacecraft_mass=self.spacecraft_mass,
                                   DMC=DMC,
                                   beta_mat=beta_mat,
                                   earth_spin_rate=self.earth_spin_rate)
            else:
                A = state_jacobian(state[0:3],
                                   state[3:6],
                                   mu=mu,
                                   J2=J2,
                                   J3=J3,
                                   C_d=Cd,
                                   C_r=Cr,
                                   P_solar=0,
                                   sun_pos=0,
                                   mu_third_body=None,
                                   third_body_state=0,
                                   station_positions_ecef=station_positions_ecef,
                                   R_e=self.R_e,
                                   mode=self.mode,
                                   spacecraft_area=spacecraft_area,
                                   spacecraft_mass=spacecraft_mass,
                                   srp_area_to_mass=0,
                                   DMC=DMC,
                                   beta_mat=beta_mat,
                                   earth_spin_rate=self.earth_spin_rate)

            phi_dot = A @ phi
            phi_dot_flat = phi_dot.flatten()

            return np.hstack((state_dot, phi_dot_flat))
        else:
            # If consider parameters are included, we also need to compute the time derivative of the sensitivity matrix, theta_dot = A @ theta + B, where B contains the partial derivatives of the equations of motion with respect to the consider parameters.
            phi_flat = augmented_state[state_length:state_length+state_length**2]
            phi = phi_flat.reshape((state_length, state_length))
            theta_flat = augmented_state[state_length+state_length**2:]
            num_consider_parameters = len(consider_parameters)
            theta = theta_flat.reshape((6, num_consider_parameters))

            # Compute state derivatives
            state_dot = self.equations_of_motion(t, state, DMC=DMC, beta_mat=beta_mat)

            # Compute STM derivative
            A = state_jacobian(state[0:3], state[3:6], mu, J2, J3, Cd, station_positions_ecef, self.R_e, mode=self.mode, spacecraft_area=self.spacecraft_area, spacecraft_mass=self.spacecraft_mass, DMC=DMC, beta_mat=beta_mat)
            phi_dot = A @ phi

            # Compute sensitivity matrix derivative
            B = np.zeros((6, num_consider_parameters))

            for i in range(num_consider_parameters):
                consider_parameter = consider_parameters[i]
                parameter_partials = compute_consider_parameter_partials(consider_parameter, state[0:3], state[3:6], mu, J2, J3, Cd, station_positions_ecef, self.R_e, spacecraft_area=self.spacecraft_area, spacecraft_mass=self.spacecraft_mass)
                B[:, i] = parameter_partials

            theta_dot = A[:6,:6] @ theta + B

            phi_dot_flat = phi_dot.flatten()
            theta_dot_flat = theta_dot.flatten()

            return np.hstack((state_dot, phi_dot_flat, theta_dot_flat))
        
    def integrate_eom(self, t_final, initial_state, teval = None, sigma_points = False):
        """Integrate the equations of motion for the spacecraft.
        Parameters:
        t_final : float
            Final time for integration in seconds.
        initial_state : np.array
            nx1 array of initial spacecraft state in ECI frame. First 6 elements are [x, y, z, u, v, w] in km and km/s.
        teval : np.array, optional
            1xN array of time points at which to store the computed solution. Default is None.
        sigma_points : bool, optional
            If True, the initial_state is of length L(2L+1) and represents the sigma points for the UKF. The function will return the time derivative of each sigma point. Default is False.
        Returns:
        time_vector : np.array
            1xN array of time points corresponding to the spacecraft states.
        state_history : np.array
            nxN array of spacecraft states over time in ECI frame."""
        
        t_span = (0, t_final)
        sol = solve_ivp(self.equations_of_motion, t_span, initial_state, method='RK45', rtol=1e-13, atol=1e-13, t_eval=teval, args=(False, None, sigma_points))
        return sol.t, sol.y
    
    def integrate_stm(self, t_final, initial_state, phi_0 = None, teval = None, initial_time : float = 0, DMC : bool = False, beta_mat : np.ndarray = None):
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
        if DMC:
            state_length += 3
        
        # Initialize STM as identity matrix
        if phi_0 is None:
            phi_0 = np.eye(state_length).flatten()

        augmented_initial_state = np.hstack((initial_state, phi_0))
        t_span = (initial_time, t_final)
        sol = solve_ivp(self.full_dynamics, t_span, augmented_initial_state, method='RK45', rtol=1e-13, atol=1e-13, t_eval=teval, args=(DMC, beta_mat))
        return sol.t, sol.y
    
    def integrate_stm_and_theta(self, t_final, initial_state, phi_0 = None, theta_0 = None, teval = None, initial_time : float = 0, consider_parameters : list = []):
        # This function integrates the STM and the sensitivity matrix for the consider parameters, theta. The initial state is augmented by both the STM and theta, and the full_dynamics function is modified to compute the time derivative of both the STM and theta.

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
        
        if len(consider_parameters) > 0:
            theta_length = len(consider_parameters)
        else:
            theta_length = 0

        # Initialize STM as identity matrix
        if phi_0 is None:
            phi_0 = np.eye(state_length).flatten()
        
        # Initialize theta_dot as zeros
        if theta_0 is None:
            theta_0 = np.zeros((6, len(consider_parameters))).flatten()

        augmented_initial_state = np.hstack((initial_state, phi_0, theta_0))
        t_span = (initial_time, t_final)

        sol = solve_ivp(self.full_dynamics, t_span, augmented_initial_state, method='RK45', rtol=1e-13, atol=1e-13, t_eval=teval, args=(False, None, consider_parameters))
        return sol.t, sol.y
        