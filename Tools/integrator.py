import numpy as np
from .generic_functions import state_jacobian, compute_density, compute_consider_parameter_partials
from .ephemeris_manager import EphemerisMgr
from scipy.integrate import solve_ivp
class Integrator:
    def __init__(self, mu : float, R_e : float, J2 : float = None, J3 : float = None, Cd : float = None, Cr = None, dynamical_mode : list = [], estimation_mode : list = [], parameter_indices : list = [], spacecraft_area : float = None, spacecraft_mass : float = None, number_of_stations : int = 0, earth_spin_rate : float = 7.2921158553E-5, solar_flux : float = 1357.0, initial_epoch : float = 0):
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
        dynamical_mode : list, optional
            List of perturbation modes to include in the integration. Options are 'PointMass', 'J2', 'J3', 'Drag', and 'Stations'. Default is an empty list.
        estimation_mode : list, optional
            List of parameters to estimate during integration. Options are 'mu', 'J2', 'J3', 'Drag', and 'SRP'. Default is an empty list.
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
        self.Cr = Cr
        self.R_e = R_e
        self.dynamical_mode = dynamical_mode
        self.estimation_mode = estimation_mode
        self.parameter_indices = parameter_indices
        self.spacecraft_area = spacecraft_area * 1e-6 if spacecraft_area is not None else 0 
        self.spacecraft_mass = spacecraft_mass if spacecraft_mass is not None else 1
        self.number_of_stations = number_of_stations
        self.earth_spin_rate = earth_spin_rate
        self.solar_flux = solar_flux
        self.initial_epoch = initial_epoch

        if len(dynamical_mode) == 0:
            raise ValueError("At least one perturbation mode must be selected in dynamical_mode.")
        if not set(dynamical_mode).issubset({'mu', 'J2', 'J3', 'Drag', 'SRP'}):
            raise ValueError("Invalid mode specified. Choose from 'mu', 'J2', 'J3', 'Drag', and/or 'SRP'.")
        if not set(estimation_mode).issubset({'mu', 'J2', 'J3', 'Drag', 'Stations', 'SRP'}) and len(estimation_mode) > 0:
            raise ValueError("Invalid estimation mode specified. Choose from 'mu', 'J2', 'J3', 'Drag', 'Stations', and/or 'SRP'.")
        if len(estimation_mode) != len(parameter_indices):
            raise ValueError("Length of estimation_mode and parameter_indices must be the same.")
        if 'Drag' in dynamical_mode and (spacecraft_area is None or spacecraft_mass is None):
            raise ValueError("Spacecraft area and mass must be provided for drag calculations.")
        if 'SRP' in dynamical_mode and (spacecraft_area is None or spacecraft_mass is None):
            raise ValueError("Spacecraft area and mass must be provided for SRP calculations.")
        if 'Stations' in estimation_mode and number_of_stations <= 0:
            raise ValueError("Number of stations must be greater than zero when 'Stations' mode is selected.")

    def get_mu_effects(self, state, mu):
        x, y, z = state[0:3]
        u, v, w = state[3:6]
        r = np.sqrt(x**2 + y**2 + z**2)
        x_dot = u
        y_dot = v
        z_dot = w
        u_dot = -self.mu * x / r**3
        v_dot = -self.mu * y / r**3
        w_dot = -self.mu * z / r**3

        return np.array([x_dot, y_dot, z_dot, u_dot, v_dot, w_dot])
    
    def get_J2_effects(self, state, mu, J2):
        x, y, z = state[0:3]
        u, v, w = state[3:6]
        r = np.sqrt(x**2 + y**2 + z**2)
        x_dot = 0
        y_dot = 0
        z_dot = 0
        u_dot = (3 / 2) * (mu * J2 * self.R_e**2 * x / r**5) * (5 * (z**2 / r**2) - 1)
        v_dot = (3 / 2) * (mu * J2 * self.R_e**2 * y / r**5) * (5 * (z**2 / r**2) - 1)
        w_dot = (3 / 2) * (mu * J2 * self.R_e**2 * z / r**5) * (5 * (z**2 / r**2) - 3)

        return np.array([x_dot, y_dot, z_dot, u_dot, v_dot, w_dot])

    def get_J3_effects(self, state, mu, J3):
        x, y, z = state[0:3]
        u, v, w = state[3:6]
        r = np.sqrt(x**2 + y**2 + z**2)
        x_dot = 0
        y_dot = 0
        z_dot = 0
        u_dot = (5 / 2) * mu * J3 * self.R_e**3 * x * z / r**7 * (7 * z**2 / r**2 - 3)
        v_dot = (5 / 2) * mu * J3 * self.R_e**3 * y * z / r**7 * (7 * z**2 / r**2 - 3)
        w_dot = (5 / 2) * mu * J3 * self.R_e**3 / r**5 * (7 * z**4 / r**4 - 6 * z**2 / r**2 + 3 / 5)
        
        return np.array([x_dot, y_dot, z_dot, u_dot, v_dot, w_dot])
    
    def get_drag_effects(self, state, Cd, spacecraft_area, spacecraft_mass):
        x, y, z = state[0:3]
        u, v, w = state[3:6]
        r = np.sqrt(x**2 + y**2 + z**2)
        rho = compute_density(r)* 1e9
        V_rel = np.array([u + self.earth_spin_rate * y, v - self.earth_spin_rate * x, w])
        u_rel, v_rel, w_rel = V_rel
        V_rel_norm = np.linalg.norm(np.array([u, v, w]))
        u_dot_drag = -(rho * Cd * spacecraft_area * V_rel_norm * u_rel) / (2 * spacecraft_mass)
        v_dot_drag = -(rho * Cd * spacecraft_area * V_rel_norm * v_rel) / (2 * spacecraft_mass)
        w_dot_drag = -(rho * Cd * spacecraft_area * V_rel_norm * w_rel) / (2 * spacecraft_mass)
        return np.array([0, 0, 0, u_dot_drag, v_dot_drag, w_dot_drag])
    
    def get_SRP_effects(self, state, Cr, spacecraft_area, spacecraft_mass, t, AU_KM = 149597870.700):
        x, y, z = state[0:3]
        u, v, w = state[3:6]
        r = np.sqrt(x**2 + y**2 + z**2)
        P_solar = self.solar_flux / 299792458.0  # Solar radiation pressure at 1 AU in N/m^2

        # For SRP calculation, we need the position of the Sun. We can use the EphemerisMgr to get this based on the initial epoch and current time.
        ephemeris_mgr = EphemerisMgr('Earth')
        earth_state = ephemeris_mgr.evaluate_state(self.initial_epoch + t / 86400)  # Position of earth relative to sun in EME2000 frame (km)
        r_sun_earth = earth_state[0:3]

        r_earth_sc = np.array([x, y, z])  # Spacecraft position relative to Earth in EME2000 frame (km)

        r_sun_sc = r_sun_earth + r_earth_sc  # Spacecraft position relative to Sun in EME2000 frame (km)
        r_sun_sc_norm = np.linalg.norm(r_sun_sc)

        # Compute SRP acceleration
        R_AU = r_sun_sc_norm / AU_KM  # dimensionless

        accel_srp_norm = P_solar * Cr * spacecraft_area / (R_AU**2 * spacecraft_mass) # Convert r_sun_sc_norm from km to m for SRP calculation
        accel_srp = accel_srp_norm / 1000 * (r_sun_sc / r_sun_sc_norm)  # Acceleration vector in km/s^2

        return np.array([0, 0, 0, accel_srp[0], accel_srp[1], accel_srp[2]])
    
    def get_dmc_effects(self, state, beta_mat):
        w_1, w_2, w_3 = state[-3:]
        u_dot_dmc = w_1
        v_dot_dmc = w_2
        w_dot_dmc = w_3

        w_1_dot = -beta_mat[0,0] * w_1
        w_2_dot = -beta_mat[1,1] * w_2
        w_3_dot = -beta_mat[2,2] * w_3
        
        addition_term = np.array([0, 0, 0, u_dot_dmc, v_dot_dmc, w_dot_dmc])
        dmc_terms = np.array([w_1_dot, w_2_dot, w_3_dot])

        return addition_term, dmc_terms
    
    def equations_of_motion(self, t : float, state : np.ndarray, DMC : bool = False, beta_mat : np.ndarray = None):
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
        Returns:
        state_dot : np.ndarray
            Time derivative of the state vector.
        """

        x, y, z = state[0:3]
        u, v, w = state[3:6]
        r = np.sqrt(x**2 + y**2 + z**2)
        rho = compute_density(r)* 1e9

        # Pull out class dynamical parameters. If they are being estimated, their values will be updated in the state vector and pulled out based on the parameter_indices list and mode.
        # If they are not being estimated, use the nominal value from the class attributes.
        mu = self.mu
        J2 = self.J2
        J3 = self.J3
        Cd = self.Cd
        Cr = self.Cr
        spacecraft_area = self.spacecraft_area
        spacecraft_mass = self.spacecraft_mass
        
        # If parameters are being estimated, pull out their current values from the state vector based on the parameter_indices list and mode. If a parameter is not being estimated, use the nominal value from the class attributes.
        if 'mu' in self.estimation_mode:
            param_index = self.parameter_indices[self.estimation_mode.index('mu')]
            mu = state[param_index]
        if 'J2' in self.estimation_mode:
            param_index = self.parameter_indices[self.estimation_mode.index('J2')]
            J2 = state[param_index]
        if 'J3' in self.estimation_mode:
            param_index = self.parameter_indices[self.estimation_mode.index('J3')]
            J3 = state[param_index]
        if 'Drag' in self.estimation_mode:
            param_index = self.parameter_indices[self.estimation_mode.index('Drag')]
            Cd = state[param_index]
        if 'SRP' in self.estimation_mode:
            param_index = self.parameter_indices[self.estimation_mode.index('SRP')]
            Cr = state[param_index]
        if 'Stations' in self.estimation_mode:
            # Determine number of station variables, this is stored in the parameter_indices value for stations as a list
            num_station_vars = self.number_of_stations * 3
        
        # Initialize state derivatives to zero, then add contributions from each perturbation based on the mode. This allows for easy addition of new perturbations in the future by simply adding new functions to compute the effects
        # and including them in the mode list and equations of motion without having to rewrite the entire function
        state_dot = np.zeros(6)

        for param in self.dynamical_mode:
            if param == 'mu':
                state_dot += self.get_mu_effects(state, mu)
            elif param == 'J2':
                state_dot += self.get_J2_effects(state, mu, J2)
            elif param == 'J3':
                state_dot += self.get_J3_effects(state, mu, J3)
            elif param == 'Drag':
                state_dot += self.get_drag_effects(state, Cd, spacecraft_area, spacecraft_mass)
            elif param == 'SRP':
                state_dot += self.get_SRP_effects(state, Cr, spacecraft_area, spacecraft_mass, t)
            else:
                # Error handling for invalid mode entry, this should be caught in the initializer but just in case
                raise ValueError("Invalid found in equations of motion. Choose from 'mu', 'J2', 'J3', 'Drag', and/or 'SRP'.")

        if 'mu' in self.estimation_mode:
            state_dot = np.append(state_dot, 0)
        if 'J2' in self.estimation_mode:
            state_dot = np.append(state_dot, 0)
        if 'J3' in self.estimation_mode:
            state_dot = np.append(state_dot, 0)
        if 'Drag' in self.estimation_mode:
            state_dot = np.append(state_dot, 0)
        if 'Stations' in self.estimation_mode:
            for _ in range(num_station_vars):
                state_dot = np.append(state_dot, 0)

        if DMC:
            addition_term, dmc_terms = self.get_dmc_effects(state, beta_mat)
            state_dot += addition_term

            state_dot = np.append(state_dot, dmc_terms) 

        return state_dot
    
    def sigma_points_eom(self, t : float, state : np.ndarray):
        """
        This function is used when the input state is of length L(2L+1) and represents the sigma points for the UKF. The function will return the time derivative of each sigma point by calling the equations_of_motion function for each sigma point.
        Parameters:
        t : float
            Current time in seconds.
        state : np.ndarray
            State vector of length L(2L+1) representing the sigma points for the UKF. Each sigma point is of length L and must have the first 6 elements as [x, y, z, u, v, w] in km and km/s. Additional elements can include parameters for estimation based on the mode.
        Returns:
        state_dot : np.ndarray
            Time derivative of the state vector for each sigma point, concatenated into a single vector of length L(2L+1).
        """
        L = 0.25 * (-1 + np.sqrt(1 + 8 * len(state)))
        for i in range(2 * int(L) + 1):
            state_i = state[i*int(L):(i+1)*int(L)]
            state_dot_i = self.equations_of_motion(t, state_i, DMC=False, beta_mat=None)
            if i == 0:
                state_dot = state_dot_i
            else:
                state_dot = np.hstack((state_dot, state_dot_i))
        return state_dot
    
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
        # Pull out class dynamical parameters. If they are being estimated, their values will be updated in the state vector and pulled out based on the parameter_indices list and mode.
        # If they are not being estimated, use the nominal value from the class attributes.

        mu = self.mu
        J2 = self.J2
        J3 = self.J3
        Cd = self.Cd
        Cr = self.Cr
        spacecraft_area = self.spacecraft_area
        spacecraft_mass = self.spacecraft_mass
        
        
        station_positions_ecef = np.array([])
        state_length = 6

        # If parameters are being estimated, pull out their current values from the state vector based on the parameter_indices list and mode. If a parameter is not being estimated, use the nominal value from the class attributes.
        # Also determine the length of the state vector based on how many parameters are being estimated, this is needed to correctly pull out the STM from the augmented state vector later on.

        if 'mu' in self.estimation_mode:
            state_length += 1
            param_index = self.parameter_indices[self.estimation_mode.index('mu')]
            mu = augmented_state[param_index]
        if 'J2' in self.estimation_mode:
            state_length += 1
            param_index = self.parameter_indices[self.estimation_mode.index('J2')]
            J2 = augmented_state[param_index]
        if 'J3' in self.estimation_mode:
            state_length += 1
            param_index = self.parameter_indices[self.estimation_mode.index('J3')]
            J3 = augmented_state[param_index]
        if 'Drag' in self.estimation_mode:
            state_length += 1
            param_index = self.parameter_indices[self.estimation_mode.index('Drag')]
            Cd = augmented_state[param_index]
        if 'SRP' in self.estimation_mode:
            state_length += 1
            param_index = self.parameter_indices[self.estimation_mode.index('SRP')]
            Cr = augmented_state[param_index]
        if 'Stations' in self.estimation_mode:
            # Determine number of station variables, this is stored in the parameter_indices value for stations
            param_index = self.parameter_indices[self.estimation_mode.index('Stations')]
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

        # If consider parameters are not included, we only need to compute the time derivative of the STM, phi_dot = A @ phi, where A is the state Jacobian
        # matrix. If consider parameters are included, we also need to compute the time derivative of the sensitivity matrix, theta_dot = A @ theta + B,
        # where B contains the partial derivatives of the equations of motion with respect to the consider parameters.
        if len(consider_parameters) == 0:
            phi_flat = augmented_state[state_length:]
            phi = phi_flat.reshape((state_length, state_length))
            # Compute state derivatives
            state_dot = self.equations_of_motion(t, state, DMC=DMC, beta_mat=beta_mat)

            # Compute STM derivative
            if self.estimation_mode == []:
                A = state_jacobian(state[0:3], state[3:6], mu, 0, 0, 0, Cr, station_positions_ecef, self.R_e, mode=['BaseMat'], spacecraft_area=spacecraft_area, spacecraft_mass=spacecraft_mass, DMC=DMC, beta_mat=beta_mat)
            else:
                A = state_jacobian(state[0:3], state[3:6], mu, J2, J3, Cd, Cr, station_positions_ecef, self.R_e, mode=self.estimation_mode, spacecraft_area=spacecraft_area, spacecraft_mass=spacecraft_mass, DMC=DMC, beta_mat=beta_mat)

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
            A = state_jacobian(state[0:3], state[3:6], mu, J2, J3, Cd, station_positions_ecef, self.R_e, mode=self.estimation_mode, spacecraft_area=self.spacecraft_area, spacecraft_mass=self.spacecraft_mass, DMC=DMC, beta_mat=beta_mat)
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
        
    def integrate_eom(self, t_final, initial_state, teval = None):
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
        sol = solve_ivp(self.equations_of_motion, t_span, initial_state, method='RK45', rtol=1e-13, atol=1e-13, t_eval=teval, args=(False, None))
        return sol.t, sol.y
    
    def integrate_sigma_points(self, t_final, initial_state, teval = None):
        """Integrate the equations of motion for the sigma points.
        Parameters:
        t_final : float
            Final time for integration in seconds.
        initial_state : np.array
            L(2L+1)x1 array of initial sigma points, where each sigma point is of length L and has the first 6 elements as [x, y, z, u, v, w] in km and km/s.
        teval : np.array, optional
            1xN array of time points at which to store the computed solution. Default is None.
        Returns:
        time_vector : np.array
            1xN array of time points corresponding to the sigma point states.
        state_history : np.array
            L(2L+1)xN array of sigma point states over time in ECI frame."""
        
        t_span = (0, t_final)
        sol = solve_ivp(self.sigma_points_eom, t_span, initial_state, method='RK45', rtol=1e-13, atol=1e-13, t_eval=teval)
        return sol.t, sol.y
    
    def integrate_stm(self, t_final, initial_state, phi_0 = None, teval = None, initial_time : float = 0, DMC : bool = False, beta_mat : np.ndarray = None):
        # Determine state length based on mode
        state_length = 6

        # Determine J2, J3, and Cd based on mode
        if 'mu' in self.estimation_mode:
            state_length += 1
        if 'J2' in self.estimation_mode:
            state_length += 1
        if 'J3' in self.estimation_mode:
            state_length += 1
        if 'Drag' in self.estimation_mode:
            state_length += 1
        if 'Stations' in self.estimation_mode:
            param_index = self.parameter_indices[self.estimation_mode.index('Stations')]
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
        if 'mu' in self.estimation_mode:
            state_length += 1
        if 'J2' in self.estimation_mode:
            state_length += 1
        if 'J3' in self.estimation_mode:
            state_length += 1
        if 'Drag' in self.estimation_mode:
            state_length += 1
        if 'Stations' in self.estimation_mode:
            param_index = self.parameter_indices[self.estimation_mode.index('Stations')]
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
        