import numpy as np
import pandas as pd
from scipy.linalg import cholesky, solve_triangular, block_diag
from Tools import Integrator, MeasurementMgr, CoordinateMgr, measurement_jacobian

class SRIF:
    def __init__(self, integrator : Integrator, measurement_mgr_list : list, initial_earth_spin_angle : float, earth_rotation_rate : float = 2*np.pi/86164):
        """
        Initialize the Square Root Information Filter (SRIF).

        Parameters:
        integrator : Integrator
            An instance of the Integrator class for propagating the state.
        measurement_mgr_list : list
            A list of MeasurementMgr instances for handling measurements.
        initial_earth_spin_angle : float
            The initial spin angle of the Earth in radians.
        earth_rotation_rate : float, optional
            The rotation rate of the Earth in radians per second (default is 2*pi/86164 rad/s).
        """
        self.integrator = integrator
        self.measurement_mgrs = measurement_mgr_list.copy()
        self.coordinate_mgr = CoordinateMgr(initial_earth_spin_angle=initial_earth_spin_angle, earth_rotation_rate=earth_rotation_rate, R_e = integrator.R_e)

    def whiten_measurements(self, y : np.ndarray, H : np.ndarray, R : np.ndarray):
        """
        Whiten the measurements by applying the Cholesky decomposition of the measurement covariance.

        Parameters:
        y : np.ndarray
            The measurement vector.
        H : np.ndarray
            The measurement Jacobian matrix.
        R : np.ndarray
            The measurement covariance matrix.

        Returns:
        np.ndarray
            The whitened measurement vector.
        """
        if np.isnan(y).all():
            return y, H

        # Compute the Cholesky decomposition of R
        V = cholesky(R)
        
        # Solve for the whitened measurements
        y_whitened = solve_triangular(V, y)
        H_whitened = solve_triangular(V, H)

        return y_whitened, H_whitened
    
    def householder_transform(self, A : np.ndarray):
        """
        Perform a Householder transformation to upper triangularize the matrix A.

        Parameters:
        A : np.ndarray
            The matrix to be transformed.

        Returns:
        np.ndarray
            The upper triangularized matrix.
        """

        n = A.shape[1] - 1
        m = A.shape[0] - n

        for k in range(n):
            u = np.zeros(m + n)

            sigma = np.sign(A[k,k]) * np.linalg.norm(A[k:,k])
            if sigma == 0:
                continue

            u[k] = A[k,k] + sigma
            A[k,k] = -sigma
            beta = 1 / (sigma * u[k])

            for i in range(k+1, m+n):
                u[i] = A[i,k]
            
            for j in range(k+1, n+1):
                gamma = beta * np.dot(u[k:], A[k:,j])
                for i in range(k, m+n):
                    A[i,j] -= gamma * u[i]
            
            for i in range(k+1, m+n):
                A[i,k] = 0

        return A
    
    def time_update(self, x_hat : np.ndarray, R : np.ndarray, phi : np.ndarray, force_triangular : bool = True, Q_noise : np.ndarray = None, delta_t : float = None):
        """
        Perform the time update step of the SRIF.
        Parameters:
        x_hat : np.ndarray
            The current state estimate.
        R : np.ndarray
            The current information matrix.
        phi : np.ndarray
            The STM for the current time step.
        Q_noise : np.ndarray, optional
            The process noise covariance matrix (default is None, which assumes no process noise). 
        Returns:
        x_bar : np.ndarray
            The predicted state estimate.
        R_bar : np.ndarray
            The predicted information matrix.
        b_bar : np.ndarray
            The predicted information vector.
        force_triangular : bool, optional
            If True, forces the time update to perform a Householder transformation to maintain numerical stability (default is False).
        """
        if Q_noise is not None and delta_t is None:
            raise ValueError("delta_t must be provided if Q_noise is not None")

        # if Q_noise is None:
        #     x_bar = phi @ x_hat
        #     R_bar = R @ np.linalg.inv(phi)
        #     if force_triangular:
        #         R_bar = self.householder_transform(R_bar)
        #     b_bar = R_bar @ x_bar
        if Q_noise is None:
            x_bar = phi @ x_hat
            R_bar = R @ np.linalg.inv(phi)
            b_bar = R_bar @ x_bar
            if force_triangular:
                A = np.column_stack([R_bar, b_bar])
                A = self.householder_transform(A)
                R_bar = A[:, :-1]
                b_bar = A[:, -1]
                x_bar = solve_triangular(R_bar, b_bar)
        else:
            R_k_tilde = R @ np.linalg.inv(phi)
            R_u = cholesky(np.linalg.inv(Q_noise))
            b_hat = R @ x_hat
            b_bar_u = R_u @ np.zeros((3,1)) # Assuming zero process noise mean

            Gamma = delta_t * np.concatenate((0.5 * delta_t * np.eye(3), np.eye(3)), axis=0)

            # Pad zeros to gamma for non-spacecraft related states if necessary
            num_additional_states = x_hat.shape[0] - 6
            if num_additional_states > 0:
                Gamma = np.pad(Gamma, ((0, num_additional_states), (0, 0)), mode='constant')

            A = np.zeros((Gamma.shape[0] + R_u.shape[0], Gamma.shape[1] + R_k_tilde.shape[1] + 1))
            A[:R_u.shape[0], :R_u.shape[1]] = R_u
            A[:R_u.shape[0], -1] = b_bar_u.flatten()
            A[R_u.shape[0]:, :Gamma.shape[1]] = -R_k_tilde @ Gamma
            A[R_u.shape[0]:, Gamma.shape[1]:-1] = R_k_tilde
            A[R_u.shape[0]:, -1] = b_hat.flatten()

            # Householder transform A to get new R_bar and b_bar
            A = self.householder_transform(A)

            R_bar = A[R_u.shape[0]:, Gamma.shape[1]:-1]
            b_bar = A[R_u.shape[0]:, -1]
            x_bar = solve_triangular(R_bar, b_bar)

        return x_bar, R_bar, b_bar

    def measurement_update(self, R_bar : np.ndarray, b_bar : np.ndarray, H : np.ndarray, y : np.ndarray):
        """
        Perform the measurement update step of the SRIF.

        Parameters:
        R_bar : np.ndarray
            The predicted information matrix from the time update step.
        b_bar : np.ndarray
            The predicted information vector from the time update step.
        H : np.ndarray
            The measurement Jacobian matrix.
        y : np.ndarray
            The whitened measurement vector.

        Returns:
        R_new : np.ndarray
            The updated information matrix.
        b_new : np.ndarray
            The updated information vector.
        """

        # Format A matrix for Householder transformation with R_bar, b_bar, H, and y
        A = np.zeros((R_bar.shape[0] + H.shape[0], R_bar.shape[1] + 1))
        A[:R_bar.shape[0], :R_bar.shape[1]] = R_bar
        A[:R_bar.shape[0], -1] = b_bar
        A[R_bar.shape[0]:, :H.shape[1]] = H
        A[R_bar.shape[0]:, -1] = y.flatten()

        A = self.householder_transform(A)

        R_new = A[:R_bar.shape[0], :R_bar.shape[1]]
        b_new = A[:R_bar.shape[0], -1]

        # Since R_new is upper triangular, we can solve for x_hat using back substitution
        x_hat_new = solve_triangular(R_new, b_new)

        return x_hat_new, R_new, b_new
        
    def run(self,
            initial_state : np.ndarray,
            initial_x_correction : np.ndarray,
            initial_covariance : np.ndarray,
            measurement_data : pd.DataFrame,
            Q_noise : np.ndarray = None,
            R_noise : np.ndarray = 0,
            max_iterations : int = 1,
            triangularize_time_update : bool = True):
        """
        Run the SRIF for a given set of measurements.
        Parameters:
        initial_state : np.ndarray
            The initial state estimate.
        initial_x_correction : np.ndarray
            The initial correction to the state estimate.
        initial_covariance : np.ndarray
            The initial covariance of the state estimate.
        measurement_data : pd.DataFrame
            A DataFrame containing the measurement data, including time, measurement type, and measurement values.
        Q : np.ndarray, optional
            The process noise covariance matrix (default is None, which assumes no process noise).
        R : np.ndarray, optional
            The measurement noise covariance matrix (default is 0, which assumes no measurement noise).
        max_iterations : int, optional
            The maximum number of iterations for the filter (default is 1).
        triangularize_time_update : bool, optional
            If True, forces the time update to perform a Householder transformation to maintain numerical stability (default is True).
        Returns:
        x_estimates : list
            A list of state estimates at each measurement time step.
        covariances : list
            A list of covariance matrices corresponding to each state estimate.
        residuals_df : pd.DataFrame
            A DataFrame containing the measurement residuals for each measurement.
        """
        x_bar0 = np.zeros_like(initial_state)
        x_hat = x_bar0.copy()
        P = initial_covariance.copy() 
        raw_state_length = len(initial_state)
        x_0 = initial_state+x_bar0.flatten()
        time_vector = measurement_data['time'].values

        initial_station_positions = []
        for mgr in self.measurement_mgrs:
            initial_station_positions.append(mgr.station_state_ecef[0:3])

        meas_number = 2  # Assuming 2 measurements per station (range and range rate)

        residuals_df = pd.DataFrame(columns=['iteration', 'station', 'pre-fit', 'post-fit'])

        # Compute information matrix R from initial covariance
        R = cholesky(np.linalg.inv(initial_covariance))
        for iteration in range(max_iterations):
            [_, augmented_state_history] = self.integrator.integrate_stm(time_vector[-1], x_0, teval=time_vector)

            # Separate state and STM history
            reference_state_history = augmented_state_history[0:raw_state_length, :]
            stm_history = np.zeros((raw_state_length, raw_state_length, len(time_vector)))
            for i, raw_state in enumerate(augmented_state_history.T):
                stm = raw_state[raw_state_length:].reshape((raw_state_length, raw_state_length))
                stm_history[:,:,i] = stm
            # Compute measurement residuals and associated H matrices for each station and measurement time
            measurement_residuals_matrix = np.zeros((meas_number,1,len(self.measurement_mgrs),len(time_vector)))  # Assuming 2 measurements per station
            H_matrix = np.zeros((meas_number,raw_state_length,len(self.measurement_mgrs),len(time_vector)))

            for i, mgr in enumerate(self.measurement_mgrs):
                station_name = mgr.station_name
                truth_measurements = np.vstack(measurement_data[f"{station_name}_measurements"].values).T
                simulated_measurements = mgr.simulate_measurements(reference_state_history[0:6,:], time_vector, 'ECI', noise=False, ignore_visibility=True)
                
                residual_vector = np.zeros((2, len(time_vector)))
                for j, time in enumerate(time_vector):
                    # Compute measurement residual
                    residual = truth_measurements[:,j] - simulated_measurements[:,j]
                    # Add pre-fit residuals to DataFrame
                    
                    # Compute measurement Jacobian
                    station_state_eci = self.coordinate_mgr.ECEF_to_ECI(mgr.station_state_ecef, time)
                    [H_sc, H_station] = measurement_jacobian(reference_state_history[:6,j], station_state_eci)
                    H_total = np.concatenate((H_sc, np.zeros((2, raw_state_length - 6))), axis = 1)  # Pad H_sc to match full state size
                    if 'Stations' in self.integrator.estimation_mode:
                        ecef_to_eci = self.coordinate_mgr.compute_DCM('ECEF', 'ECI', time=time_vector[j])
                        H_station_ecef = H_station @ ecef_to_eci

                        num_stations = self.integrator.number_of_stations
                        first_station_partial_index = raw_state_length - 3 * num_stations # Assumes 3 position states per station and they are at the end of the state vector
                        station_partial_index = first_station_partial_index + i * 3
                        H_total[:, station_partial_index:station_partial_index+3] = H_station_ecef

                    # Whiten measurements and H matrix
                    residual_whitened, H_total_whitened = self.whiten_measurements(residual, H_total, R_noise)

                    residual_vector[:,j] = residual
                    measurement_residuals_matrix[:,:,i,j] = np.vstack(residual_whitened)
                    H_matrix[:,:,i,j] = H_total_whitened

                residuals_df = pd.concat([residuals_df, pd.DataFrame({'iteration': iteration, 'station': station_name, 'pre-fit': [residual_vector], 'post-fit': np.nan})], ignore_index=True)

            # Perform SRIF estimation process
            state_estimates = np.zeros((raw_state_length, len(time_vector)))
            prediction_covariance_estimates = np.zeros((raw_state_length, raw_state_length, len(time_vector)))
            covariance_estimates = np.zeros((raw_state_length, raw_state_length, len(time_vector)))

            for k, time in enumerate(time_vector):
                print(f"Processing time step {k+1} of {len(time_vector)}                       ", end='\r')
                current_measurement_residuals = measurement_residuals_matrix[:,:,:,k]

                if k == 0:
                    phi = stm_history[:,:,k]
                    delta_t = 0
                else:
                    phi = stm_history[:,:,k] @ np.linalg.inv(stm_history[:,:,k-1])
                    delta_t = time_vector[k] - time_vector[k-1]

                if np.isnan(current_measurement_residuals).all():
                    # No measurements available, propagate x_hat and R using time update only
                    if Q_noise is not None:
                        x_hat, R, b_bar = self.time_update(x_hat, R, phi, force_triangular=triangularize_time_update, Q_noise=Q_noise, delta_t=delta_t)
                    else:
                        x_hat, R, b_bar = self.time_update(x_hat, R, phi, force_triangular=triangularize_time_update)
                else:
                    # Determine which station has measurements at this time step
                    visible_station_indices = []
                    for i in range(len(self.measurement_mgrs)):
                        if ~np.isnan(current_measurement_residuals[:,:,i]).any():
                            visible_station_indices.append(i)

                    # Stack measurement residuals and H matrices for visible stations
                    visible_residuals = []
                    visible_H = []
                    visible_R_noise = []

                    for i in visible_station_indices:
                        visible_residuals.append(current_measurement_residuals[:,:,i])
                        visible_H.append(H_matrix[:,:,i,k])
                        visible_R_noise.append(R_noise)

                    stacked_residuals = np.vstack(visible_residuals)
                    stacked_H = np.vstack(visible_H)
                    stacked_R_noise = block_diag(*visible_R_noise)
                    
                    # Perform time and measurement updates
                    if Q_noise is not None:
                        x_bar, R_bar, b_bar = self.time_update(x_hat, R, phi, force_triangular=triangularize_time_update, Q_noise=Q_noise, delta_t=delta_t)
                    else:
                        x_bar, R_bar, b_bar = self.time_update(x_hat, R, phi, force_triangular=triangularize_time_update)
                    x_hat, R, b_bar = self.measurement_update(R_bar, b_bar, stacked_H, stacked_residuals)

                # Recompute covariance estimate from information matrix
                P = np.linalg.inv(R.T @ R)
                
                state_estimates[:,k] = x_hat.T + reference_state_history[:,k]
                covariance_estimates[:,:,k] = P

            # Add post-fit residuals to DataFrame
            for i, mgr in enumerate(self.measurement_mgrs):
                station_name = self.measurement_mgrs[i].station_name

                # Simulate measurements using updated state estimate
                simulated_measurements = mgr.simulate_measurements(state_estimates[0:6,:], time_vector, 'ECI', noise=False, ignore_visibility=True)
                truth_measurements = np.vstack(measurement_data[f"{station_name}_measurements"].values).T
                measurement_residuals = truth_measurements - simulated_measurements

                mask = (residuals_df['iteration'] == iteration) & (residuals_df['station'] == station_name)

                idx = residuals_df[mask].index[0]  # Get the index of the matching row
                residuals_df.at[idx, 'post-fit'] = measurement_residuals

            if 'Stations' in self.integrator.estimation_mode:
                num_stations = self.integrator.number_of_stations
                first_station_index = raw_state_length - 3 * num_stations
                for s in range(num_stations):
                    station_index = first_station_index + s * 3
                    new_station_position = x_0[station_index:station_index+3]
                    self.measurement_mgrs[s].station_state_ecef[0:3] = new_station_position
                    self.measurement_mgrs[s].lat, self.measurement_mgrs[s].lon = self.coordinate_mgr.ECEF_to_GCS(new_station_position)

        # Reset station positions to original values after LKF iterations
        for i, mgr in enumerate(self.measurement_mgrs):
            mgr.station_state_ecef[0:3] = initial_station_positions[i]
            mgr.lat, mgr.lon = self.coordinate_mgr.ECEF_to_GCS(initial_station_positions[i])

        return state_estimates, covariance_estimates, residuals_df