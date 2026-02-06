import numpy as np
import pandas as pd
from ASEN_6080.Tools import Integrator, MeasurementMgr, CoordinateMgr, measurement_jacobian
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
        self.measurement_mgrs = measurement_mgr_list
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
        eigenvalues = np.maximum(eigenvalues, min_eigenvalue)
        
        # Reconstruct
        P_fixed = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        
        return P_fixed

    def predict(self, x_hat : np.ndarray, P : np.ndarray, phi : np.ndarray, H : np.ndarray, Q : np.ndarray, R : np.ndarray):
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
        Q : np.ndarray
            The process noise covariance matrix.
        R : np.ndarray
            The measurement noise covariance matrix.

        Returns:
        tuple
            A tuple containing the predicted state, predicted covariance, and Kalman Gain.
        """
        # Predict state
        predicted_state = phi @ x_hat

        # Predict covariance
        predicted_covariance = phi @ P @ phi.T + Q

        # Ensure positive definiteness if predicted covariance is very large
        if np.any(np.abs(np.diag(predicted_covariance)) > 1e3):
            predicted_covariance = self.ensure_positive_definite(predicted_covariance)

        # Compute Kalman Gain
        kalman_gain = predicted_covariance @ H.T @ np.linalg.inv(H @ predicted_covariance @ H.T + R)

        return predicted_state, predicted_covariance, kalman_gain
    
    def update(self, predicted_state : np.ndarray, predicted_covariance : np.ndarray, kalman_gain : np.ndarray, measurement_residual : np.ndarray, H : np.ndarray, R: np.ndarray):
        """
        Perform the update step of the Kalman Filter.

        Parameters:
        predicted_state : np.ndarray
            The predicted state estimate.
        predicted_covariance : np.ndarray
            The predicted covariance estimate.
        kalman_gain : np.ndarray
            The Kalman Gain matrix
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
        # Update state estimate
        updated_state = np.vstack(predicted_state) + kalman_gain @ (measurement_residual - H @ np.vstack(predicted_state))

        # Update covariance estimate
        I = np.eye(predicted_covariance.shape[0])
        #updated_covariance = (I - kalman_gain @ H) @ predicted_covariance
        updated_covariance = (I - kalman_gain @ H) @ predicted_covariance @ (I - kalman_gain @ H).T + kalman_gain @ R @ kalman_gain.T
        
        return updated_state, updated_covariance
    
    def run(self, initial_state : np.ndarray, initial_x_correction : np.ndarray, initial_covariance : np.ndarray, measurement_data : pd.DataFrame, Q : np.ndarray = 0, R : np.ndarray = 0, max_iterations : int = 1, convergence_threshold : float = 1e-5):
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
        Returns:
        state_estimates : list
            A list of state estimates at each measurement time.
        covariance_estimates : list
            A list of covariance estimates at each measurement time.
        """
        x_bar0 = np.zeros_like(initial_state)
        x_hat = x_bar0.copy()
        P = initial_covariance.copy() 
        raw_state_length = len(initial_state)
        # x_0 = np.append(initial_state+x_hat.flatten(), initial_state[raw_state_length:])  # Augmented initial state with STM identity
        x_0 = initial_state+x_bar0.flatten()
        time_vector = measurement_data['time'].values
        # Begin iteration loop
        for iteration in range(max_iterations):
            print(f"Starting LKF iteration {iteration+1} of {max_iterations}                           ")
            # Integrate over measurement times
            [_, augmented_state_history] = self.integrator.integrate_stm(time_vector[-1], x_0, teval=time_vector)

            # Separate state and STM history
            reference_state_history = augmented_state_history[0:raw_state_length, :]
            stm_history = np.zeros((raw_state_length, raw_state_length, len(time_vector)))
            for i, raw_state in enumerate(augmented_state_history.T):
                stm = raw_state[raw_state_length:].reshape((raw_state_length, raw_state_length))
                stm_history[:,:,i] = stm

            # Compute measurement residuals and associated H matrices for each station and measurement time
            measurement_residuals_matrix = np.zeros((2,1,len(self.measurement_mgrs),len(time_vector)))  # Assuming 2 measurements per station
            H_matrix = np.zeros((2,raw_state_length,len(self.measurement_mgrs),len(time_vector)))

            for i, mgr in enumerate(self.measurement_mgrs):
                station_name = mgr.station_name
                truth_measurements = np.vstack(measurement_data[f"{station_name}_measurements"].values).T
                simulated_measurements = mgr.simulate_measurements(reference_state_history[0:6,:], time_vector, 'ECI', noise=False, ignore_visibility=True)

                for j, time in enumerate(time_vector):
                    # Compute measurement residual
                    residual = truth_measurements[:,j] - simulated_measurements[:,j]

                    # Compute measurement Jacobian
                    station_state_eci = self.coordinate_mgr.ECEF_to_ECI(mgr.station_state_ecef, time)
                    [H_sc, H_station] = measurement_jacobian(reference_state_history[:6,j], station_state_eci)
                    measurement_residuals_matrix[:,:,i,j] = np.vstack(residual)
                    H_total = np.concatenate((H_sc, np.zeros((2, raw_state_length - 6))), axis = 1)  # Pad H_sc to match full state size
                    if 'Stations' in self.integrator.mode:
                        ecef_to_eci = self.coordinate_mgr.compute_DCM('ECEF', 'ECI', time=time_vector[j])
                        H_station_ecef = H_station @ ecef_to_eci

                        num_stations = self.integrator.number_of_stations
                        first_station_partial_index = raw_state_length - 3 * num_stations # Assumes 3 position states per station and they are at the end of the state vector
                        station_partial_index = first_station_partial_index + i * 3
                        H_total[:, station_partial_index:station_partial_index+3] = H_station_ecef

                    H_matrix[:,:,i,j] = H_total
            # Perform LKF estimation process
            state_estimates = np.zeros((raw_state_length, len(time_vector)))
            covariance_estimates = np.zeros((raw_state_length, raw_state_length, len(time_vector)))

            for k, time in enumerate(time_vector):
                print(f"Processing time step {k+1} of {len(time_vector)}", end='\r')
                # Check if measurements are available at this time
                current_measurement_residuals = measurement_residuals_matrix[:,:,:,k]
                if k == 0:
                    phi = stm_history[:,:,k]
                else:
                    phi = stm_history[:,:,k] @ np.linalg.inv(stm_history[:,:,k-1])
                
                if np.isnan(current_measurement_residuals).all():
                    # No measurements available, propagate state and covariance
                    x_hat, P, _ = self.predict(x_hat, P, phi, np.zeros((2,raw_state_length)), Q, R)
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
                    x_bar, predict_P, K = self.predict(x_hat, P, phi, stacked_H, Q, stacked_R)
                    x_hat, P = self.update(x_bar, predict_P, K, stacked_residuals, stacked_H, stacked_R)
                # Store estimates
                state_estimates[:,k] = x_hat.T + reference_state_history[:,k]
                if np.any(np.diag(P) < 0):
                    raise ValueError("Covariance matrix has negative diagonal elements.")
                covariance_estimates[:,:,k] = P
            # Right before line 281

            x_hat0, _, _, _ = np.linalg.lstsq(stm_history[:,:, -1], x_hat, rcond=None)
            # After line 282
            x_0 += x_hat0.flatten()

            # improved_initial_covariance = np.linalg.inv(stm_history[:,:, -1]) @ P @ np.linalg.inv(stm_history[:,:, -1]).T
            # P = improved_initial_covariance*10
            P = initial_covariance.copy()  # Reset covariance for next iteration
            
            x_bar0 = x_bar0 - x_hat0.flatten()  # Update x_bar0 for next iteration  
            x_hat = x_bar0.copy()
            # x_hat = np.zeros_like(x_hat)  # Reset state correction for next iteration
            # Update station positions in measurement managers if estimating station position
            if 'Stations' in self.integrator.mode:
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

            np.nan_to_num(mean_residual, nan=0.0)
            np.set_printoptions(linewidth=200)
            # print(f"x_hat0 correction: {x_hat0.flatten()}")
            #print(f"Current mu estimate : {x_0[6]}")
            print(f"Mean measurement residuals after iteration {iteration+1}: {mean_residual.flatten()} meters")
            # print(f"x_hat0 correction Cd: {np.linalg.norm(x_hat0[8])}")
            # print(f"Current Cd estimate : {x_0[8]}")
            # print(f"Current Cd covariance : {covariance_estimates[8,8,-1]}")
            # print(f"Current J2 estimate : {x_0[7]}")
            # print(f"Current J2 covariance : {covariance_estimates[7,7,-1]}")

            # print(f"STM condition number: {np.linalg.cond(stm_history[0:6,0:6,-1])}")
            # print(f"Final covariance diagonal: {np.sqrt(np.diag(covariance_estimates[:,:,-1]))}")
            if np.all(np.abs(mean_residual) < convergence_threshold):
                print("Convergence achieved based on measurement residuals.")
                break
            
        return state_estimates, covariance_estimates
