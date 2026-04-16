import numpy as np
import pandas as pd
from Tools import Integrator, MeasurementMgr, CoordinateMgr, measurement_jacobian
from scipy.linalg import block_diag
import warnings

class LKF:
    def __init__(self, integrator : Integrator, measurement_mgr_list : list, initial_earth_spin_angle : float, earth_rotation_rate : float = 2*np.pi/86164.0905):
        """
        Initialize the Linearized Kalman Filter.

        Parameters:
        integrator : Integrator
            An instance of the Integrator class for orbit propagation.
        measurement_mgr_list : list
            A list of MeasurementMgr instances for different ground stations.
        initial_earth_spin_angle : float
            Initial Earth spin angle in radians.
        earth_spin_rate : float, optional
            Earth's rotation rate in radians per second. Default is 2*pi/86164.0905 rad/s.
        """
        self.integrator = integrator
        self.measurement_mgrs = measurement_mgr_list.copy()
        self.coordinate_mgr = CoordinateMgr(initial_earth_spin_angle=initial_earth_spin_angle, earth_rotation_rate=earth_rotation_rate, R_e = integrator.R_e)
    def ensure_positive_definite(self, P : np.ndarray, min_eigenvalue: float = 1e-13):
        """
        Ensure covariance matrix is symmetric positive definite.
        
        Parameters:
        P : np.ndarray - Covariance matrix
        min_eigenvalue : float - Minimum allowed eigenvalue
        
        Returns:
        np.ndarray - Regularized positive definite covariance matrix
        """
        # Enforce symmetry
        P = 0.5 * (P + P.T)
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(P)
        
        # Clamp negative eigenvalues

        eigenvalues = np.maximum(eigenvalues, 1e-18)
        
        # Reconstruct
        P_fixed = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        
        return P_fixed

    def predict(self, x_hat : np.ndarray, P : np.ndarray, phi : np.ndarray, H : np.ndarray, R : np.ndarray):
        """
        Perform the prediction step of the Kalman Filter.

        Parameters:
        x_hat : np.ndarray
            The current state estimate.
        P : np.ndarray
            The current covariance estimate.
        phi : np.ndarray
            The state transition matrix.
        H : np.ndarray
            The measurement matrix.
        R : np.ndarray
            The measurement noise covariance matrix.

        Returns:
        tuple
            A tuple containing the predicted state and predicted covariance
        """
        # Predict state
        predicted_state = phi @ x_hat
        # Predict covariance
        predicted_covariance = phi @ P @ phi.T

        # # Ensure positive definiteness if predicted covariance is very large
        # if np.any(np.abs(np.diag(predicted_covariance)) > 1e3):
        #     predicted_covariance = self.ensure_positive_definite(predicted_covariance)

        return predicted_state, predicted_covariance
    
    def update(self, predicted_state : np.ndarray, predicted_covariance : np.ndarray, measurement_residual : np.ndarray, H : np.ndarray, R: np.ndarray):
        """
        Perform the update step of the Kalman Filter.

        Parameters:
        predicted_state : np.ndarray
            The predicted state estimate.
        predicted_covariance : np.ndarray
            The predicted covariance estimate.
        measurement_residual : np.ndarray
            The measurement residual (innovation).
        H : np.ndarray
            The measurement matrix.
        R : np.ndarray
            The measurement noise covariance matrix.
        Returns:
        tuple
            A tuple containing the updated state and updated covariance.
        """
        # Compute Kalman Gain
        kalman_gain = predicted_covariance @ H.T @ np.linalg.inv(H @ predicted_covariance @ H.T + R)

        # Update state estimate
        updated_state = np.vstack(predicted_state) + kalman_gain @ (measurement_residual - H @ np.vstack(predicted_state))

        # Update covariance estimate
        I = np.eye(predicted_covariance.shape[0])
        #updated_covariance = (I - kalman_gain @ H) @ predicted_covariance
        updated_covariance = (I - kalman_gain @ H) @ predicted_covariance @ (I - kalman_gain @ H).T + kalman_gain @ R @ kalman_gain.T
        
        return updated_state, updated_covariance
    
    def compute_DMC_covariance(self, beta_mat : np.ndarray, Q : np.ndarray, delta_t : float):
        """
        Compute the process noise covariance matrix for Dynamic Model Compensation (DMC).

        Parameters:
        beta_mat : np.ndarray
            A 3x3 diagonal matrix of time constants for DMC.
        Q : np.ndarray
            The base process noise covariance matrix.
        delta_t : float
            The time step size.

        Returns:
        Q_w : np.ndarray
            The process noise covariance matrix for DMC.
        """
        beta_list = np.diag(beta_mat)
        sigma_list = np.sqrt(np.diag(Q))

        Q_w = np.zeros((9, 9))  # Assuming 6 state variables and 3 DMC variables
        for i, (beta, sigma) in enumerate(zip(beta_list, sigma_list)):
            # Precompute exponential terms
            exp_beta = np.exp(-beta * delta_t)
            exp_2beta = np.exp(-2 * beta * delta_t)
            leading_term = (sigma**2) / (beta**2)

            # Compute covariance elements using closed-form solutions
            Q_ii = leading_term * (delta_t**3 / 3 - delta_t**2 / beta + delta_t / beta**2 * (1 - 2*exp_beta) + (1 - exp_2beta) / (2 * beta**3))
            Q_iv = leading_term * (0.5 * delta_t**2 - delta_t / beta * (1 - exp_beta) + (0.5 - exp_beta + 0.5*exp_2beta) / beta**2)
            Q_vv = leading_term * (delta_t - (1.5 + 0.5 * exp_2beta - 2*exp_beta) / beta)
            Q_iw = leading_term * ((1 - exp_2beta) / (2 * beta) - delta_t * exp_beta)
            Q_vw = leading_term * (0.5 * (1 + exp_2beta) - exp_beta)
            Q_ww = sigma **2 / (2 * beta) * (1 - exp_2beta)

            # Assign proper values to the covariance matrix
            Q_w[i, i] = Q_ii
            Q_w[i, i+3] = Q_iv
            Q_w[i, i+6] = Q_iw
            Q_w[i+3, i+3] = Q_vv
            Q_w[i+3, i+6] = Q_vw
            Q_w[i+6, i+6] = Q_ww

            # Assign symmetric elements by
        Q_w = Q_w + Q_w.T - np.diag(Q_w.diagonal())

        return Q_w

    def run(self, initial_state : np.ndarray,
            initial_x_correction : np.ndarray,
            initial_covariance : np.ndarray,
            measurement_data : pd.DataFrame,
            Q : np.ndarray = None, R : np.ndarray = 0,
            max_iterations : int = 1,
            convergence_threshold : float = 1e-5,
            considered_measurements : str = 'All',
            process_noise_approach : str = 'None',
            Q_frame : str = 'ECI',
            beta_mat : np.ndarray = None,
            apply_smoothing : bool = False):
        """
        Run the Linearized Kalman Filter over a series of measurements.
        Parameters:
        initial_state : np.ndarray
            The initial state estimate.
        initial_x_correction : np.ndarray
            The initial state correction.
        initial_covariance : np.ndarray
            The initial covariance estimate.
        measurement_data : pd.DataFrame
            DataFrame containing the measurement data.
        Q : np.ndarray
            The process noise covariance matrix.
        R : np.ndarray
            The measurement noise covariance matrix.
        max_iterations : int, optional
            The maximum number of iterations for the LKF. Default is 1.
        convergence_threshold : float, optional
            The convergence threshold for stopping criteria, linked to mean of residuals. Default is 1e-5 (1 cm).
        considered_measurements : str, optional
            A string indicating which measurements to consider in the LKF. Options are 'Range', 'Range Rate', or 'All'. Default is 'All'.
        process_noise_approach : str, optional
            A string indicating the approach for handling process noise. Options are 'None' for no process noise, 'SNC' for State Noise Compensation, or 'DMC' for Dynamic Model Compensation. Default is 'None'.
        Q_frame : str, optional
            The reference frame of the process noise covariance matrix Q ('ECI' or 'RIC'). Default is 'ECI'.
        beta_mat : np.ndarray, optional
            A 3x3 diagonal matrix of time constants for dynamic model compensation. Required if process_noise_approach is 'DMC'. Default is None.
        Returns:
        state_estimates : list
            A list of state estimates at each measurement time.
        covariance_estimates : list
            A list of covariance estimates at each measurement time.
        """
        if process_noise_approach not in ['None', 'SNC', 'DMC']:
            raise ValueError("Invalid process_noise_approach. Must be 'None', 'SNC', or 'DMC'.")
        if process_noise_approach == 'SNC' and Q is None:
            raise ValueError("Process noise covariance matrix Q must be provided for SNC approach.")
        if process_noise_approach == 'DMC' and (beta_mat is None or Q is None):
            raise ValueError("Beta matrix and process noise covariance matrix Q must be provided for DMC approach.")

        x_bar0 = np.zeros_like(initial_state)
        x_hat = x_bar0.copy()
        P = initial_covariance.copy() 
        raw_state_length = len(initial_state)
        x_0 = initial_state+x_bar0.flatten()
        time_vector = measurement_data['time'].values

        initial_station_positions = []
        for mgr in self.measurement_mgrs:
            initial_station_positions.append(mgr.station_state_ecef[0:3])
        if considered_measurements == 'Range':
            R = R[0::2, 0::2].reshape(1,1)  # Extract covariance for range measurements
            meas_number = 1
        elif considered_measurements == 'Range Rate':
            R = R[1::2, 1::2].reshape(1,1)  # Extract covariance for range rate measurements
            meas_number = 1
        elif considered_measurements == 'All':
            meas_number = 2
        else:
            raise ValueError("Invalid option for considered_measurements. Must be 'Range', 'Range Rate', or 'All'.")
        residuals_df = pd.DataFrame(columns=['iteration', 'station', 'pre-fit', 'post-fit'])
        # Begin iteration loop
        for iteration in range(max_iterations):
            print(f"Starting LKF iteration {iteration+1} of {max_iterations}                           ")
            if process_noise_approach == 'DMC':
                raw_state_length -= 3  # Remove DMC portion of state for integration and STM history, since DMC will be added in as process noise

            # Integrate over measurement times
            if process_noise_approach == 'DMC':
                # Ignore DMC portion of initial state for integration to get reference trajectory and STM history, since DMC will be added in as process noise
                [_, augmented_state_history] = self.integrator.integrate_stm(time_vector[-1], x_0[:-3], teval=time_vector)
            else:
                [_, augmented_state_history] = self.integrator.integrate_stm(time_vector[-1], x_0, teval=time_vector)

            # Separate state and STM history
            reference_state_history = augmented_state_history[0:raw_state_length, :]
            stm_history = np.zeros((raw_state_length, raw_state_length, len(time_vector)))
            for i, raw_state in enumerate(augmented_state_history.T):
                stm = raw_state[raw_state_length:].reshape((raw_state_length, raw_state_length))
                stm_history[:,:,i] = stm
                
            # Compute measurement residuals and associated H matrices for each station and measurement time
            measurement_residuals_matrix = np.zeros((meas_number,1,len(self.measurement_mgrs),len(time_vector)))  # Assuming 2 measurements per station
            if process_noise_approach == 'DMC':
                H_matrix = np.zeros((meas_number,raw_state_length+3,len(self.measurement_mgrs),len(time_vector)))
            else:
                H_matrix = np.zeros((meas_number,raw_state_length,len(self.measurement_mgrs),len(time_vector)))

            for i, mgr in enumerate(self.measurement_mgrs):
                station_name = mgr.station_name
                truth_measurements = np.vstack(measurement_data[f"{station_name}_measurements"].values).T
                simulated_measurements = mgr.simulate_measurements(reference_state_history[0:6,:], time_vector, 'ECI', noise=False, ignore_visibility=True)
                
                residual_vector = np.zeros((2, len(time_vector)))
                for j, time in enumerate(time_vector):
                    # Compute measurement residual
                    residual = truth_measurements[:,j] - simulated_measurements[:,j]
                    residual_vector[:,j] = residual
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

                    # Check considered measurements and adjust residuals and H
                    if considered_measurements == 'Range':
                        residual = residual[0].reshape(1,1)  # Only take range residual
                        H_total = H_total[0,:].reshape(1, -1)
                    elif considered_measurements == 'Range Rate':
                        residual = residual[1].reshape(1,1)  # Only take range residual
                        H_total = H_total[1,:].reshape(1, -1)
                    else:
                        pass
                    measurement_residuals_matrix[:,:,i,j] = np.vstack(residual)

                    # Pad H matrix to account for DMC portion of state
                    if process_noise_approach == 'DMC':
                        H_total = np.concatenate((H_total, np.zeros((H_total.shape[0], 3))), axis=1)
                    
                    H_matrix[:,:,i,j] = H_total

                residuals_df = pd.concat([residuals_df, pd.DataFrame({'iteration': iteration, 'station': station_name, 'pre-fit': [residual_vector], 'post-fit': np.nan})], ignore_index=True)
            # Perform LKF estimation process

            if process_noise_approach == 'DMC':
                # Re-add state length portion
                reference_state_history = np.concatenate((reference_state_history, np.zeros((3, reference_state_history.shape[1]))), axis=0)
                raw_state_length += 3

            state_estimates = np.zeros((raw_state_length, len(time_vector)))
            prediction_covariance_estimates = np.zeros((raw_state_length, raw_state_length, len(time_vector)))
            covariance_estimates = np.zeros((raw_state_length, raw_state_length, len(time_vector)))
            for k, time in enumerate(time_vector):
                print(f"Processing time step {k+1} of {len(time_vector)}                       ", end='\r')
                
                current_measurement_residuals = measurement_residuals_matrix[:,:,:,k]
                # Integrate directly between time steps to get phi for bad beta values, otherwise use STM history
                if process_noise_approach == 'DMC':
                    if k == 0:
                        phi = np.eye(raw_state_length)
                    else:
                        phi = stm_history[:,:,k] @ np.linalg.inv(stm_history[:,:,k-1])
                        phi = np.pad(phi, ((0,3),(0,3)))  # Pad phi to account for DMC portion of state
                        for i, beta in enumerate(np.diag(beta_mat)):
                            w_val = np.exp(-beta * (time - time_vector[k-1]))
                            v_val = (1 - w_val) / beta
                            r_val = (time - time_vector[k-1]) / beta - v_val / beta

                            phi[i-3,i-3] = w_val
                            phi[i+3,i-3] = v_val
                            phi[i,i-3] = r_val
                else:
                    if k == 0:
                        phi = stm_history[:,:,k]
                    else:
                        phi = stm_history[:,:,k] @ np.linalg.inv(stm_history[:,:,k-1])
                
                if np.isnan(current_measurement_residuals).all():
                    # No measurements available, propagate state and covariance
                    x_hat, P = self.predict(x_hat, P, phi, np.zeros((meas_number, raw_state_length)), R)

                    if process_noise_approach == 'SNC':
                        if Q_frame == 'RIC':
                            # Transform Q from RIC to ECI frame
                            dcm = self.coordinate_mgr.compute_DCM('ECI', 'RIC', time=time, orbit_state=reference_state_history[:,k])
                            Q_eci = dcm.T @ Q @ dcm
                        elif Q_frame == 'ECI':
                            Q_eci = Q
                        delta_t = time_vector[k] - time_vector[k-1] if k > 0 else 0
                        Gamma = delta_t * np.concatenate((0.5 * delta_t * np.eye(3), np.eye(3)), axis=0)
                        P[0:6, 0:6] = P[0:6, 0:6] + Gamma @ Q_eci @ Gamma.T
                    if process_noise_approach == 'DMC':
                        if Q_frame == 'RIC':
                            # Transform Q from RIC to ECI frame
                            dcm = self.coordinate_mgr.compute_DCM('ECI', 'RIC', time=time, orbit_state=reference_state_history[:,k])
                            Q_eci = dcm.T @ Q @ dcm
                        elif Q_frame == 'ECI':
                            Q_eci = Q
                        delta_t = time_vector[k] - time_vector[k-1] if k > 0 else 0
                        Q_w = self.compute_DMC_covariance(beta_mat, Q_eci, delta_t)
                        P[0:6, 0:6] = P[0:6, 0:6] + Q_w[0:6, 0:6]  # Add only the state covariance portion of Q_w
                        P[0:6, -3:] = P[0:6, -3:] + Q_w[0:6, 6:]  # Add state-DMC cross covariance
                        P[-3:, 0:6] = P[-3:, 0:6] + Q_w[6:, 0:6]  # Add DMC-state cross covariance
                        P[-3:, -3:] = P[-3:, -3:] + Q_w[6:, 6:]  # Add DMC covariance
                    prediction_covariance_estimates[:,:,k] = P
                else:
                    # Determine which stations are visible
                    visible_station_indices = []
                    for i in range(len(self.measurement_mgrs)):
                        if ~np.isnan(current_measurement_residuals[:,:,i]).any():
                            visible_station_indices.append(i)

                    # Stack measurement residuals and H matrices for visible stations
                    visible_residuals = []
                    visible_H = []
                    visible_R = []

                    for i in visible_station_indices:
                        visible_residuals.append(current_measurement_residuals[:,:,i])
                        visible_H.append(H_matrix[:,:,i,k])
                        visible_R.append(R)

                    stacked_residuals = np.vstack(visible_residuals)
                    stacked_H = np.vstack(visible_H)
                    stacked_R = block_diag(*visible_R)
                    
                    # Predict and update steps
                    x_bar, predict_P = self.predict(x_hat, P, phi, stacked_H, stacked_R)

                    # Add process noise
                    dt = time_vector[k] - time_vector[k-1] if k > 0 else 0
                    if process_noise_approach == 'SNC' and dt < 120:  # Only add SNC process noise for reasonable time steps to avoid numerical issues
                        if Q_frame == 'RIC':
                            # Transform Q from RIC to ECI frame
                            dcm = self.coordinate_mgr.compute_DCM('ECI', 'RIC', time=time, orbit_state=reference_state_history[:,k])
                            Q_eci = dcm.T @ Q @ dcm
                        elif Q_frame == 'ECI':
                            Q_eci = Q
                        delta_t = time_vector[k] - time_vector[k-1] if k > 0 else 0
                        Gamma = delta_t * np.concatenate((0.5 * delta_t * np.eye(3), np.eye(3)), axis=0)
                        predict_P[0:6, 0:6] = predict_P[0:6, 0:6] + Gamma @ Q_eci @ Gamma.T
                    if process_noise_approach == 'DMC':
                        if Q_frame == 'RIC':
                            # Transform Q from RIC to ECI frame
                            dcm = self.coordinate_mgr.compute_DCM('ECI', 'RIC', time=time, orbit_state=reference_state_history[:,k])
                            Q_eci = dcm.T @ Q @ dcm
                        elif Q_frame == 'ECI':
                            Q_eci = Q
                        delta_t = time_vector[k] - time_vector[k-1] if k > 0 else 0
                        Q_w = self.compute_DMC_covariance(beta_mat, Q_eci, delta_t)
                        predict_P[0:6, 0:6] = predict_P[0:6, 0:6] + Q_w[0:6, 0:6]  # Add only the state covariance portion of Q_w
                        predict_P[0:6, -3:] = predict_P[0:6, -3:] + Q_w[0:6, 6:]  # Add state-DMC cross covariance
                        predict_P[-3:, 0:6] = predict_P[-3:, 0:6] + Q_w[6:, 0:6]  # Add DMC-state cross covariance
                        predict_P[-3:, -3:] = predict_P[-3:, -3:] + Q_w[6:, 6:]  # Add DMC covariance
                    prediction_covariance_estimates[:,:,k] = predict_P

                    x_hat, P = self.update(x_bar, predict_P, stacked_residuals, stacked_H, stacked_R)
                # Store estimates
                state_estimates[:,k] = x_hat.T + reference_state_history[:,k]
                covariance_estimates[:,:,k] = P

            if apply_smoothing:
                x_hat_history = state_estimates - reference_state_history

                # Initialize smoothed estimates with filtered estimates
                smoothed_state_estimates = np.zeros_like(state_estimates)
                smoothed_covariance_estimates = np.zeros_like(covariance_estimates)

                # Set final smoothed estimates to final filtered estimates
                smoothed_state_estimates[:,-1] = x_hat_history[:,-1]
                smoothed_covariance_estimates[:,:,-1] = covariance_estimates[:,:,-1]

                # Loop through time steps in reverse order for smoothing
                for k in range(len(time_vector)-2, -1, -1):
                    print(f"Smoothing time step {k+1} of {len(time_vector)}                       ", end='\r')
                    x_k = x_hat_history[:,k]
                    P_k_plus_1 = prediction_covariance_estimates[:,:,k+1]
                    P_k = covariance_estimates[:,:,k]
                    phi_k_plus_1 = stm_history[:,:,k+1] @ np.linalg.inv(stm_history[:,:,k]) if k > 0 else stm_history[:,:,k+1]

                    s_k = np.linalg.solve(P_k_plus_1, (phi_k_plus_1 @ P_k)).T

                    # Update smoothed state estimate
                    smoothed_state_estimates[:,k] = x_k + s_k @ (smoothed_state_estimates[:,k+1] - phi_k_plus_1 @ x_k)

                    # Update smoothed covariance estimate
                    smoothed_covariance_estimates[:,:,k] = P_k + s_k @ (smoothed_covariance_estimates[:,:,k+1] - P_k_plus_1) @ s_k.T

                # Update state estimates with smoothed values
                state_estimates = smoothed_state_estimates + reference_state_history
                covariance_estimates = smoothed_covariance_estimates

            if 'DMC' in process_noise_approach:
                x_hat0 = np.linalg.solve(stm_history[:,:, -1], x_hat[:-3])
                # Append zeros for DMC portion of state
                x_hat0 = np.concatenate((x_hat0, np.zeros((3,1))), axis=0)
                x_0 += x_hat0.flatten()
            else:
                x_hat0, _, _, _ = np.linalg.lstsq(stm_history[:,:, -1], x_hat, rcond=None)
                x_0 += x_hat0.flatten()

            P = initial_covariance.copy()  # Reset covariance for next iteration
            
            x_bar0 = x_bar0 - x_hat0.flatten()  # Update x_bar0 for next iteration  
            x_hat = x_bar0.copy()

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
            
            # Determine if another iteration is needed based on residual behavior (detect if residuals are centered around zero)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mean_residual = np.nanmean(measurement_residuals_matrix, axis=(0,1,3))

            np.set_printoptions(linewidth=200)
            print(f"Mean measurement residuals after iteration {iteration+1}: {mean_residual.flatten()} meters")

            if np.all(np.abs(mean_residual) < convergence_threshold):
                print("Convergence achieved based on measurement residuals.")
                break
            
        # Reset station positions to original values after LKF iterations
        for i, mgr in enumerate(self.measurement_mgrs):
            mgr.station_state_ecef[0:3] = initial_station_positions[i]
            mgr.lat, mgr.lon = self.coordinate_mgr.ECEF_to_GCS(initial_station_positions[i])
        
        return state_estimates, covariance_estimates, residuals_df
