import numpy as np
import pandas as pd
from ASEN_6080.Tools import Integrator, MeasurementMgr, CoordinateMgr, measurement_jacobian
from scipy.linalg import block_diag

class ConsiderCov:
    def __init__(self, integrator : Integrator, measurement_mgr_list : list, initial_earth_spin_angle : float, earth_rotation_rate : float = 2*np.pi/86164.0905):
        """
        Class to compute the consider covariance matrix for a given set of measurements and an integrator.
        Parameters
        ----------
        integrator : Integrator
            An instance of the Integrator class that contains the dynamics and initial conditions for the problem.
        measurement_mgr_list : list
            A list of MeasurementMgr instances that contain the measurements and their associated covariance matrices.
        initial_earth_spin_angle : float
            The initial spin angle of the Earth in radians. This is used to compute the initial state of the Earth in the inertial frame.
        earth_rotation_rate : float, optional
            The rotation rate of the Earth in radians per second. The default value is 2*pi/86164.0905, which corresponds to one rotation per sidereal day.
        """

        self.integrator = integrator
        self.measurement_mgrs = measurement_mgr_list
        self.coordinate_mgr = CoordinateMgr(initial_earth_spin_angle=initial_earth_spin_angle, earth_rotation_rate=earth_rotation_rate, R_e = integrator.R_e)

    def time_update(self, x_hat : np.ndarray, P_hat : np.ndarray, S_hat : np.ndarray, theta : np.ndarray, phi : np.ndarray, c : np.ndarray, P_cc : np.ndarray):
        """
        Perform a time update for the consider covariance matrix.
        Parameters
        ----------
        x_hat : np.ndarray
            The current state estimate of the system.
        P_hat : np.ndarray
            The current covariance estimate of the system.
        S_hat : np.ndarray
            The sensitivity matrix of the system.
        theta : np.ndarray
            Consider covariance parameter partials.
        phi : np.ndarray
            The state transition matrix for the system.
        c : float
            Error in consider parameter.
        P_cc : np.ndarray
            The consider covariance matrix.
        Returns
        -------
        P_consider : np.ndarray
            The updated consider covariance matrix after the time update.
        """
        # Predict state
        x_bar = phi @ x_hat

        # Predicted covariance
        P_bar = phi @ P_hat @ phi.T

        # Sensitivity matrix
        S_bar = phi @ S_hat + theta

        # Parameter state estimate
        x_c_bar = x_bar + S_bar @ c

        # Consider covariance update
        P_c_bar = P_bar + S_bar @ P_cc @ S_bar.T

        # Consider cross covariance update
        P_xc_bar = S_bar @ P_cc

        return x_bar, P_bar, S_bar, x_c_bar, P_c_bar, P_xc_bar
    
    def measurement_update(self, predicted_state : np.ndarray, predicted_covariance : np.ndarray, S_bar : np.ndarray, P_cc : np.ndarray, c : float, H_x : np.ndarray, H_c : np.ndarray, R : np.ndarray, measurement_residual : np.ndarray):
        """
        Perform a measurement update for the consider covariance matrix.

        Parameters:
        predicted_state : np.ndarray
            The predicted state of the system.
        predicted_covariance : np.ndarray
            The predicted covariance of the system.
        S_bar : np.ndarray
            The sensitivity matrix of the system.
        P_cc : np.ndarray
            The consider covariance matrix.
        c : float
            Error in consider parameter.
        H_x : np.ndarray
            The measurement Jacobian with respect to the state.
        H_c : np.ndarray
            The measurement Jacobian with respect to the consider parameters.
        R : np.ndarray
            The measurement noise covariance matrix.
        measurement_residual : np.ndarray
            The residuals between the measurements and the predicted measurements.
        """
        kalman_gain = predicted_covariance @ H_x.T @ np.linalg.inv(H_x @ predicted_covariance @ H_x.T + R)

        x_hat = np.vstack(predicted_state) + kalman_gain @ (measurement_residual - H_x @ np.vstack(predicted_state))

        I = np.eye(predicted_covariance.shape[0])
        P_x_hat = (I - kalman_gain @ H_x) @ predicted_covariance @ (I - kalman_gain @ H_x).T + kalman_gain @ R @ kalman_gain.T

        # Sensitivity matrix update
        S_hat = (I - kalman_gain @ H_x) @ S_bar - kalman_gain @ H_c

        # Consider state update
        x_c_hat = x_hat + (S_hat @ c).reshape(-1, 1)

        # Consider covariance update
        P_c_hat = P_x_hat + S_hat @ P_cc @ S_hat.T

        # Consider cross covariance update
        P_xc_hat = S_hat @ P_cc

        return x_hat, P_x_hat, S_hat, x_c_hat, P_c_hat, P_xc_hat

    def propagate_traj(self, initial_state : np.ndarray, time_vector : np.ndarray, consider_parameters : list):
        state_length = initial_state.shape[0]
        num_consider_params = len(consider_parameters)
        final_time = time_vector[-1]

        [_, augmented_state_history] = self.integrator.integrate_stm_and_theta(final_time, initial_state, teval=time_vector, consider_parameters=consider_parameters)

        # Separate state, stm, and theta from augmented_state_history
        state_history = np.zeros((state_length, len(time_vector)))
        stm_history = np.zeros((state_length, state_length, len(time_vector)))

        theta_length = 6
        theta_history = np.zeros((theta_length, num_consider_params, len(time_vector)))

        for i, augmented_state in enumerate(augmented_state_history.T):
            state_history[:, i] = augmented_state[:state_length]
            flattened_stm = augmented_state[state_length:state_length + state_length**2]
            stm_history[:, :, i] = flattened_stm.reshape((state_length, state_length))
            flattened_theta = augmented_state[state_length + state_length**2:]
            theta_history[:, :, i] = flattened_theta.reshape((theta_length, num_consider_params))

        return state_history, stm_history, theta_history
    
    def compute_residuals_and_jacobians(self, reference_state_history : np.ndarray, measurement_data : pd.DataFrame, time_vector : np.ndarray, num_consider_params : int, meas_number : int = 2):
        raw_state_length = reference_state_history.shape[0]
        measurement_residuals_matrix = np.zeros((meas_number, 1, len(self.measurement_mgrs),len(time_vector)))  # Assuming 2 measurements per station
        H_x_matrix = np.zeros((meas_number, raw_state_length, len(self.measurement_mgrs), len(time_vector)))
        H_c_matrix = np.zeros((meas_number, num_consider_params, len(self.measurement_mgrs), len(time_vector)))

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
                    H_total = np.concatenate((H_sc, np.zeros((2, raw_state_length - 6))), axis = 1)  # Pad H_sc to match full state size

                    measurement_residuals_matrix[:,:,i,j] = np.vstack(residual)
                    H_x_matrix[:, :, i, j] = H_total
                    H_c_matrix[:, :, i, j] = np.zeros((2, num_consider_params))  # Assuming measurement model does not directly depend on consider parameters

        return measurement_residuals_matrix, H_x_matrix, H_c_matrix
    
    def run(self, initial_state : np.ndarray, initial_P : np.ndarray, consider_parameters : list, initial_S : np.ndarray, c : float, P_cc : np.ndarray, R : np.ndarray, time_vector : np.ndarray, measurement_data : pd.DataFrame):
        raw_state_length = initial_state.shape[0]

        state_history, stm_history, theta_history = self.propagate_traj(initial_state, time_vector, consider_parameters)
        measurement_residuals_matrix, H_x_matrix, H_c_matrix = self.compute_residuals_and_jacobians(state_history[:6,:], measurement_data, time_vector, len(consider_parameters))

        psi_history = np.zeros((raw_state_length + len(consider_parameters), raw_state_length + len(consider_parameters), len(time_vector)))
        
        # Build psi_history
        for i in range(stm_history.shape[2]):
            stm = stm_history[:,:,i]
            theta = theta_history[:,:,i]
            top_half = np.hstack((stm, theta))
            bottom_half = np.hstack((np.zeros((len(consider_parameters), raw_state_length)), np.eye(len(consider_parameters))))
            psi = np.vstack((top_half, bottom_half))
            psi_history[:,:,i] = psi
        
        state_correction_estimates = np.zeros((raw_state_length, len(time_vector)))
        state_covariance_estimates = np.zeros((raw_state_length, raw_state_length, len(time_vector)))
        total_covariance_estimates = np.zeros((raw_state_length + len(consider_parameters), raw_state_length + len(consider_parameters), len(time_vector)))
        S_estimates = np.zeros((raw_state_length, len(consider_parameters), len(time_vector)))
        for k, time in enumerate(time_vector):
            print(f"Processing time step {k+1} of {len(time_vector)}                       ", end='\r')
            current_measurement_residuals = measurement_residuals_matrix[:,:,:,k]

            if k == 0:
                phi = stm_history[:,:,k]
                theta = theta_history[:,:,k]
                if np.isnan(current_measurement_residuals).all():
                    # No measurements available assign initial state and covariance to first time step
                    state_correction_estimates[:,k] = np.zeros(raw_state_length)
                    state_covariance_estimates[:,:,k] = initial_P
                    S_estimates[:,:,k] = initial_S
                    total_covariance_estimates[:,:,k] = np.block([[initial_P, initial_S @ P_cc], [P_cc @ initial_S.T, P_cc]])             
                else:
                    # Measurements available, perform measurement update using initial conditions
                    # Determine which stations are visible
                    visible_station_indices = []
                    for i in range(len(self.measurement_mgrs)):
                        if ~np.isnan(current_measurement_residuals[:,:,i]).any():
                            visible_station_indices.append(i)

                    # Stack measurement residuals and H matrices for visible stations
                    visible_residuals = []
                    visible_H_x = []
                    visible_H_c = []
                    visible_R = []

                    for i in visible_station_indices:
                        visible_residuals.append(current_measurement_residuals[:,:,i])
                        visible_H_x.append(H_x_matrix[:,:,i,k])
                        visible_H_c.append(H_c_matrix[:,:,i,k])
                        visible_R.append(R)

                    stacked_residuals = np.vstack(visible_residuals)
                    stacked_H_x = np.vstack(visible_H_x)
                    stacked_H_c = np.vstack(visible_H_c)
                    stacked_R = block_diag(*visible_R)
                    x_hat, P_hat, S_hat, x_c_hat, P_c_hat, P_xc_hat = self.measurement_update(np.zeros(raw_state_length), initial_P, initial_S, P_cc, c, stacked_H_x, stacked_H_c, stacked_R, stacked_residuals)
                    state_correction_estimates[:,k] = x_hat.flatten()
                    state_covariance_estimates[:,:,k] = P_hat
                    S_estimates[:,:,k] = S_hat
                    total_covariance_estimates[:,:,k] = np.block([[P_c_hat, P_xc_hat], [P_xc_hat.T, P_cc]])
                continue  
            else:
                phi = stm_history[:,:,k] @ np.linalg.inv(stm_history[:,:,k-1])
                theta = theta_history[:,:,k] - phi @ theta_history[:,:,k-1]

            if np.isnan(current_measurement_residuals).all() and k > 0:
                # No measurements available, propagate state and covariance
                x_bar, P_bar, S_bar, x_c_bar, P_c_bar, P_xc_bar = self.time_update(state_correction_estimates[:,k-1], state_covariance_estimates[:,:,k-1], S_estimates[:,:,k-1], theta, phi, c, P_cc)
                state_correction_estimates[:,k] = x_bar.flatten()
                state_covariance_estimates[:,:,k] = P_bar
                S_estimates[:,:,k] = S_bar
                total_covariance_estimates[:,:,k] = np.block([[P_c_bar, P_xc_bar], [P_xc_bar.T, P_cc]])
            else:
                # Measurements available, perform time and measurement update
                # Determine which stations are visible
                visible_station_indices = []
                for i in range(len(self.measurement_mgrs)):
                    if ~np.isnan(current_measurement_residuals[:,:,i]).any():
                        visible_station_indices.append(i)

                # Stack measurement residuals and H matrices for visible stations
                visible_residuals = []
                visible_H_x = []
                visible_H_c = []
                visible_R = []

                for i in visible_station_indices:
                    visible_residuals.append(current_measurement_residuals[:,:,i])
                    visible_H_x.append(H_x_matrix[:,:,i,k])
                    visible_H_c.append(H_c_matrix[:,:,i,k])
                    visible_R.append(R)

                stacked_residuals = np.vstack(visible_residuals)
                stacked_H_x = np.vstack(visible_H_x)
                stacked_H_c = np.vstack(visible_H_c)
                stacked_R = block_diag(*visible_R)

                x_bar, P_bar, S_bar, x_c_bar, P_c_bar, P_xc_bar = self.time_update(state_correction_estimates[:,k-1], state_covariance_estimates[:,:,k-1], S_estimates[:,:,k-1], theta, phi, c, P_cc)
                x_hat, P_x_hat, S_hat, x_c_hat, P_c_hat, P_xc_hat = self.measurement_update(x_bar, P_bar, S_bar, P_cc, c, stacked_H_x, stacked_H_c, stacked_R, stacked_residuals)

                # Add to history list
                state_correction_estimates[:,k] = x_hat.flatten()
                state_covariance_estimates[:,:,k] = P_x_hat
                S_estimates[:,:,k] = S_hat
                total_covariance_estimates[:,:,k] = np.block([[P_c_hat, P_xc_hat], [P_xc_hat.T, P_cc]])

        state_estimates = state_history + state_correction_estimates
        return state_estimates, state_covariance_estimates, S_estimates, total_covariance_estimates, psi_history