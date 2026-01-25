import numpy as np
import pandas as pd
from ASEN_6080.Tools import Integrator, MeasurementMgr, CoordinateMgr, measurement_jacobian
from scipy.linalg import block_diag

class LKF:
    def __init__(self, integrator : Integrator, measurement_mgr_list : list, initial_earth_spin_angle : float):
        """
        Initialize the Linearized Kalman Filter.

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

        # Compute Kalman Gain
        kalman_gain = predicted_covariance @ H.T @ np.linalg.inv(H @ predicted_covariance @ H.T + R)

        return predicted_state, predicted_covariance, kalman_gain
    
    def update(self, predicted_state : np.ndarray, predicted_covariance : np.ndarray, kalman_gain : np.ndarray, measurement_residual : np.ndarray, H : np.ndarray):
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
        Returns:
        tuple
            A tuple containing the updated state and updated covariance.
        """
        # Update state estimate
        updated_state = np.vstack(predicted_state) + kalman_gain @ (measurement_residual - H @ np.vstack(predicted_state))

        # Update covariance estimate
        I = np.eye(predicted_covariance.shape[0])
        updated_covariance = (I - kalman_gain @ H) @ predicted_covariance

        return updated_state, updated_covariance
    
    def run(self, initial_state : np.ndarray, initial_x_correction : np.ndarray, initial_covariance : np.ndarray, measurement_data : pd.DataFrame, Q : np.ndarray = 0, R : np.ndarray = 0):
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
        Returns:
        state_estimates : list
            A list of state estimates at each measurement time.
        covariance_estimates : list
            A list of covariance estimates at each measurement time.
        """
        x_hat = initial_x_correction
        P = initial_covariance
        raw_state_length = len(initial_state)
        time_vector = measurement_data['time'].values
        # Integrate over measurement times
        [_, augmented_state_history] = self.integrator.integrate_stm(time_vector[-1], initial_state, teval=time_vector)

        # Separate state and STM history
        reference_state_history = augmented_state_history[0:6, :]
        stm_history = np.zeros((6, 6, len(time_vector)))
        for i, raw_state in enumerate(augmented_state_history.T):
            state = raw_state[0:6]
            reference_state_history[:,i] = state

            raw_stm = raw_state[raw_state_length:].reshape((raw_state_length, raw_state_length))
            stm = raw_stm[0:6,0:6]
            stm_history[:,:,i] = stm

        # Compute measurement residuals and associated H matrices for each station and measurement time
        measurement_residuals_matrix = np.zeros((2,1,len(self.measurement_mgrs),len(time_vector)))  # Assuming 2 measurements per station
        H_matrix = np.zeros((2,6,len(self.measurement_mgrs),len(time_vector)))

        for i, mgr in enumerate(self.measurement_mgrs):
            station_name = mgr.station_name
            truth_measurements = np.vstack(measurement_data[f"{station_name}_measurements"].values).T
            simulated_measurements = mgr.simulate_measurements(reference_state_history, time_vector, 'ECI', noise=False)

            for j, time in enumerate(time_vector):
                # Compute measurement residual
                residual = truth_measurements[:,j] - simulated_measurements[:,j]

                # Compute measurement Jacobian
                station_state_eci = self.coordinate_mgr.GCS_to_ECI(mgr.lat, mgr.lon, time)
                [H, _] = measurement_jacobian(reference_state_history[:,j], station_state_eci)
                measurement_residuals_matrix[:,:,i,j] = np.vstack(residual)
                H_matrix[:,:,i,j] = H
        # Perform LKF estimation process
        state_estimates = np.zeros((6, len(time_vector)))
        covariance_estimates = np.zeros((6, 6, len(time_vector)))

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
                x_hat, P, _ = self.predict(x_hat, P, phi, np.zeros((2,6)), Q, R)
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
                visible_Q = []

                for i in visible_station_indices:
                    visible_residuals.append(current_measurement_residuals[:,:,i])
                    visible_H.append(H_matrix[:,:,i,k])
                    visible_R.append(R)
                    visible_Q.append(Q)
            
                stacked_residuals = np.vstack(visible_residuals)
                stacked_H = np.vstack(visible_H)
                stacked_R = block_diag(*visible_R)
                stacked_Q = block_diag(*visible_Q)

                # Predict and update steps
                predict_x_hat, predict_P, K = self.predict(x_hat, P, phi, stacked_H, stacked_Q, stacked_R)
                x_hat, P = self.update(predict_x_hat, predict_P, K, stacked_residuals, stacked_H)

            # Store estimates

            state_estimates[:,k] = x_hat.T + reference_state_history[:,k]
            covariance_estimates[:,:,k] = P
        
        return state_estimates, covariance_estimates
