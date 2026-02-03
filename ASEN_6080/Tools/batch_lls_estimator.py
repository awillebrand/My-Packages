import numpy as np
import pandas as pd
from ASEN_6080.Tools import Integrator, MeasurementMgr, CoordinateMgr, measurement_jacobian

class BatchLLSEstimator:
    def __init__(self, integrator : Integrator, measurement_mgr_list : list, initial_earth_spin_angle : float, earth_rotation_rate : float = 2*np.pi/86164.0905):
        """
        Initialize the Batch Least Squares Estimator.

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
        self.coordinate_mgr = CoordinateMgr(initial_earth_spin_angle=initial_earth_spin_angle, earth_rotation_rate=earth_rotation_rate, R_e=integrator.R_e)

    def estimate_initial_state(self, a_priori_state : np.ndarray, measurement_data : pd.DataFrame, R : np.array, a_priori_covariance : np.ndarray = None, a_priori_state_correction : np.ndarray = None, max_iterations : int = 20, tol : float = 1e-5):
        """
        Estimate the initial state using Batch Least Squares.

        Parameters:
        a_priori_state : np.ndarray
            Initial guess for the state vector.
        measurement_data : pd.DataFrame
            DataFrame containing the measurement data.
        R : np.array
            Measurement noise covariance matrix R.
        a_priori_covariance : np.ndarray
            Initial covariance matrix for the state estimate.
        a_priori_state_correction : np.ndarray, optional
            Initial correction to the a priori state. Default is None.
        max_iterations : int
            Maximum number of iterations for convergence.
        tol : float
            Tolerance for convergence.

        Returns:
        dict
            A dictionary containing the estimated state, covariance, and residuals.
        """

        estimated_state = a_priori_state.copy()
        if a_priori_state_correction is not None:
            x_correction = a_priori_state_correction.copy()
        else:
            x_correction = np.zeros_like(estimated_state)
        if a_priori_covariance is not None:
            P_0 = a_priori_covariance.copy()
        else:
            P_0 = None
        time_vector = measurement_data['time'].values
        raw_state_length = len(estimated_state)
        
        # Compute noise covariance matrix R
        R_inv = np.linalg.inv(R)

        for iteration in range(max_iterations):
            if P_0 is not None and x_correction is not None:
                Lambda = np.linalg.inv(P_0)
                N = Lambda @ x_correction
            else:
                Lambda = 0
                N = 0
                print("Insufficient a priori information provided. Setting initial Lambda and N to zero.")
            
            # Propagate state and STM
            [_, augmented_state_history] = self.integrator.integrate_stm(time_vector[-1], estimated_state, teval=time_vector)

            # Initialize measurement residuals and design matrix
            residuals_matrix = np.empty((len(self.measurement_mgrs), len(time_vector)), dtype=object)  # Assuming 2 measurements per station
            H_matrix = np.empty((len(self.measurement_mgrs), len(time_vector)), dtype=object)

            for i, mgr in enumerate(self.measurement_mgrs):
                station_name = mgr.station_name

                truth_measurements = np.vstack(measurement_data[f"{station_name}_measurements"].values).T
                simulated_measurements = mgr.simulate_measurements(augmented_state_history[0:6,:], time_vector, 'ECI', noise=False)
                if i == 2:
                    breakpoint()
                # Compute measurement residuals
                residuals = truth_measurements - simulated_measurements
                for j, residual in enumerate(residuals.T):
                    residuals_matrix[i, j] = residual

                # Compute H_tilde matrix
                for j, raw_state in enumerate(augmented_state_history.T):
                    sc_state = raw_state[0:6]
                    stm = raw_state[raw_state_length:].reshape((raw_state_length, raw_state_length))

                    station_state_eci = self.coordinate_mgr.GCS_to_ECI(mgr.lat, mgr.lon, time_vector[j]) # Double check conversion if things arent working

                    [H_sc_tilde, H_station_tilde] = measurement_jacobian(sc_state, station_state_eci)
                    H_tilde = np.concatenate((H_sc_tilde, np.zeros((2, raw_state_length - 6))), axis=1)  # Augment H_tilde to match full state size

                    # Add station position partials if estimating station positions
                    if 'Stations' in self.integrator.mode:
                        num_stations = self.integrator.number_of_stations
                        first_station_partial_index = raw_state_length - 3 * num_stations # Assumes 3 position states per station and they are at the end of the state vector
                        station_partial_index = first_station_partial_index + i * 3
                        H_tilde[:, station_partial_index:station_partial_index+3] = H_station_tilde

                    H = H_tilde @ stm
                    H_matrix[i, j] = H
            
            # Accumulate observations
            for i, time in enumerate(time_vector):
                residuals_i = residuals_matrix[:, i]
                H_i = H_matrix[:, i]
                
                # Only compute Lambda and N accumulation for available measurements
                for res, H in zip(residuals_i, H_i):
                    if ~np.isnan(res).any():
                        Lambda += H.T @ R_inv @ H
                        N += H.T @ R_inv @ res
                
            # Compute state correction

            x_hat = np.linalg.inv(Lambda) @ N
            estimated_state += x_hat

            if np.linalg.norm(x_hat) < tol:
                print(f"Converged in {iteration+1} iterations.")
                P_0 = np.linalg.inv(Lambda)
                return estimated_state, P_0
            else:
                print(f"Iteration {iteration+1}: State correction norm = {np.linalg.norm(x_correction)}")
                print(f"x_hat = {np.linalg.norm(x_hat)}")
                x_correction = x_correction - x_hat
        print("Maximum iterations reached without convergence.")
        P_0 = np.linalg.inv(Lambda)
        return estimated_state, P_0


