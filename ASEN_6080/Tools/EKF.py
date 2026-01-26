import numpy as np
import pandas as pd
from ASEN_6080.Tools import Integrator, MeasurementMgr, CoordinateMgr, measurement_jacobian, LKF
from scipy.linalg import block_diag

class EKF:
    def __init__(self, integrator : Integrator, measurement_mgr_list : list, initial_earth_spin_angle : float):
        """
        Initialize the Extended Kalman Filter.

        Parameters:
        integrator : Integrator
            An instance of the Integrator class for orbit propagation.
        measurement_mgr_list : list
            A list of MeasurementMgr instances for different ground stations.
        initial_earth_spin_angle : float
            Initial Earth spin angle in radians.
        """

        self.integrator = integrator
        self.measurement_mgrs = measurement_mgr_list
        self.coordinate_mgr = CoordinateMgr(initial_earth_spin_angle=initial_earth_spin_angle)

    def predict(self, P : np.ndarray, phi : np.ndarray, Q : np.ndarray):
        """
        Perform the prediction step of the Extended Kalman Filter.

        Parameters:
        P : np.ndarray
            The current covariance estimate.
        phi : np.ndarray
            The state transition matrix.
        Q : np.ndarray
            The process noise covariance matrix.
        Returns:
        np.ndarray
            The predicted covariance estimate.
        """
        # Predict covariance
        predicted_covariance = phi @ P @ phi.T + Q

        return predicted_covariance
    
    def update(self, predicted_covariance : np.ndarray, measurement_residual : np.ndarray, H : np.ndarray, R : np.ndarray):
        """
        Perform the update step of the Extended Kalman Filter.

        Parameters:
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
        K = predicted_covariance @ H.T @ np.linalg.inv(H @ predicted_covariance @ H.T + R)

        # Update state estimate
        x_hat = K @ measurement_residual

        # Update covariance estimate
        identity_matrix = np.eye(predicted_covariance.shape[0])
        updated_covariance = (identity_matrix - K @ H) @ predicted_covariance @ (identity_matrix - K @ H).T + K @ R @ K.T
        #updated_covariance = (identity_matrix - K @ H) @ predicted_covariance

        return x_hat, updated_covariance
    
    def run(self, initial_state, initial_x_correction : np.ndarray, initial_covariance : np.ndarray, measurement_data : pd.DataFrame, Q : np.ndarray = 0, R : np.ndarray = 0, start_mode : str = 'cold', start_length : int = 100):
        """
        Run the Extended Kalman Filter over the provided measurement data.

        Parameters:
        initial_state : np.ndarray
            The initial state vector.
        initial_x_correction : np.ndarray
            The initial state correction vector.
        initial_covariance : np.ndarray
            The initial covariance matrix.
        measurement_data : pd.DataFrame
            A DataFrame containing the measurement data.
        Q : np.ndarray, optional
            The process noise covariance matrix. Default is 0.
        R : np.ndarray, optional
            The measurement noise covariance matrix. Default is 0.
        start_mode : str, optional
            The starting mode of the filter ('cold' or 'warm'). Default is 'cold'.
        start_length : int, optional
            The number of initial measurements to use for warm start. Default is 100.
        Returns:
        Tuple
            A tuple containing the state estimates and covariance estimates over time.
        """

        raw_state_length = len(initial_state)
        time_vector = measurement_data['time'].values
        state_estimates = np.zeros((6, len(time_vector)))
        covariance_estimates = np.zeros((6, 6, len(time_vector)))

        if start_mode == 'cold':
            print("Starting EKF in cold start mode.")
            x_hat = initial_x_correction
            P = initial_covariance
            X_k_0 = initial_state[0:6] + x_hat.T
            state_estimates[:,0] = X_k_0 + x_hat.T
            covariance_estimates[:,:,0] = P
            ekf_start = 0
        elif start_mode == 'warm':
            print("Starting EKF in warm start mode.")
            # Run LKF on initial measurements to get initial state correction
            lkf = LKF(self.integrator, self.measurement_mgrs, initial_earth_spin_angle=self.coordinate_mgr.initial_earth_spin_angle)
            [lkf_x_history, lkf_P_history] = lkf.run(initial_state, initial_x_correction, initial_covariance, measurement_data.iloc[0:start_length], Q=Q, R=R)
            P = lkf_P_history[:,:,-1]
            X_k_0 = lkf_x_history[-1,:]
            x_hat = np.zeros((6,1))  # No additional correction after warm start

            # Save initial LKF output as first state estimates
            state_estimates[:,0:start_length] = lkf_x_history
            covariance_estimates[:,:,0:start_length] = lkf_P_history
            ekf_start = start_length

            # Remove first start_length measurements from measurement data for EKF run
            measurement_data = measurement_data.iloc[start_length:].reset_index(drop=True)
            time_vector = measurement_data['time'].values
            breakpoint()
        else:
            raise ValueError("Invalid start_mode. Choose 'cold' or 'warm'.")

        # Reorganize measurement data into a 4D matrix: (measurement_type, measurement_dimension, station_index, time_index)
        measurement_matrix = np.zeros((2,1,len(self.measurement_mgrs),len(time_vector)))  # Assuming 2 measurements per station
        for i, mgr in enumerate(self.measurement_mgrs):
            station_name = mgr.station_name
            truth_measurements = np.vstack(measurement_data[f"{station_name}_measurements"].values).T
            for j, time in enumerate(time_vector):
                # Compute measurement residual
                measurement_matrix[:,:,i,j] = np.vstack(truth_measurements[:,j])

        # Perform EKF estimation process

        for k, time in enumerate(time_vector[1:], start=1):
            print(f"EKF Time Step {k}/{len(time_vector)-1}", end='\r')
            # Integrate from previous time to current time
            previous_time = time_vector[k-1]
            integration_state = np.append(X_k_0, initial_state[6])
            
            [_, augmented_state_history] = self.integrator.integrate_stm(time, integration_state, teval=[previous_time, time], initial_time=previous_time)

            # Separate state and STM
            raw_state = augmented_state_history[:,-1]
            X_k = raw_state[0:6]
            raw_stm = raw_state[raw_state_length:].reshape((raw_state_length, raw_state_length))
            phi = raw_stm[0:6,0:6]

            # Predict covariance
            predict_P = self.predict(P, phi, Q)

            # Check if measurements are available at this time
            current_measurements = measurement_matrix[:,:,:,k]
            if np.isnan(current_measurements).all():
                # No measurements available, propagate state and covariance
                x_hat = np.zeros((6,1))  # No correction
                P = predict_P

            # Perform measurement update steps
            else:
                # Determine which stations are visible
                visible_station_indices = []
                for i in range(len(self.measurement_mgrs)):
                    if not np.isnan(current_measurements[:,:,i]).all():
                        visible_station_indices.append(i)

                # Compute H matrices and measurement residuals for visible stations
                H_matrices = []
                measurement_residuals = []
                for i in visible_station_indices:
                    mgr = self.measurement_mgrs[i]
                    station_state_eci = self.coordinate_mgr.GCS_to_ECI(mgr.lat, mgr.lon, time)
                    measurement = mgr.simulate_measurements(np.vstack(X_k), np.array([time]), 'ECI', noise=False)
                        
                    residual = current_measurements[:,:,i] - measurement
                    [H, _] = measurement_jacobian(X_k, station_state_eci)
                    measurement_residuals.append(residual)
                    H_matrices.append(H)

                if np.isnan(np.vstack(measurement_residuals)).all():
                    # Treat as if no measurements are available, pure prediction
                    x_hat = np.zeros((6,1))  # No correction
                    P = predict_P
                else:
                    stacked_residuals = np.vstack(measurement_residuals)
                    stacked_H = np.vstack(H_matrices)

                    # Stack R matrices for visible stations
                    visible_R = [R for _ in visible_station_indices]
                    stacked_R = block_diag(*visible_R)

                    # Update step
                    x_hat, P = self.update(predict_P, stacked_residuals, stacked_H, stacked_R)
                    

            # Store estimates
            state_estimates[:,k+ekf_start] = X_k + x_hat.T
            covariance_estimates[:,:,k+ekf_start] = P
            if np.isnan(x_hat).any():
                print("NaN detected in state estimate!")
            X_k_0 = X_k + x_hat.T

        return state_estimates, covariance_estimates
            