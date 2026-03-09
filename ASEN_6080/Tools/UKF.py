import numpy as np
import pandas as pd
from ASEN_6080.Tools import Integrator, MeasurementMgr, CoordinateMgr, measurement_jacobian

class UKF:
    def __init__(self, integrator : Integrator, measurement_mgr_list : list, initial_earth_spin_angle : float, earth_rotation_rate : float = 2*np.pi/86164.0905):
        """
        Initializes the Unscented Kalman Filter (UKF) for state estimation.
        Parameters:
        integrator : Integrator
            An instance of the Integrator class for state propagation.
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

    def compute_weights(self, alpha : float, beta : float, L : int):
        """
        Computes the weights for the sigma points based on the UKF parameters.
        Parameters:
        alpha : float
            Spread of the sigma points. Default is 1e-3.
        beta : float
            Incorporates prior knowledge of the distribution. Default is 2 for Gaussian distributions.
        L : int
            Dimensionality of the state vector.
        Returns:
        Wm : np.ndarray
            Weights for the mean.
        Wc : np.ndarray
            Weights for the covariance.
        gamma : float
            Scaling factor for the sigma points.
        """
        kappa = 3 - L
        lam = alpha**2 * (L + kappa) - L
        gamma = np.sqrt(L + lam)

        Wm_0 = lam / (L + lam)
        Wc_0 = Wm_0 + (1 - alpha**2 + beta)
        Wm = np.full(2 * L + 1, 1 / (2 * (L + lam)))
        Wc = np.full(2 * L + 1, 1 / (2 * (L + lam)))
        Wm[0] = Wm_0
        Wc[0] = Wc_0
        
        return Wm, Wc, gamma
    
    def compute_sigma_points(self, x : np.ndarray, P : np.ndarray, gamma : float):
        """
        Computes the sigma points for the UKF based on the current state and covariance.
        Parameters:
        x : np.ndarray
            Current state vector.
        P : np.ndarray
            Current state covariance matrix.
        gamma : float
            Scaling factor for the sigma points.
        Returns:
        sigma_points : np.ndarray
            Matrix of sigma points.
        """
        L = len(x)

        # Compute the square root of the covariance matrix using Cholesky decomposition
        try:
            sqrt_P = np.linalg.cholesky(P)
        except np.linalg.LinAlgError:
            eps = 1e-16
            P = P + eps * np.eye(L)
            try:
                sqrt_P = np.linalg.cholesky(P)
            except np.linalg.LinAlgError:
                raise ValueError("Covariance matrix is not positive definite.")

        # Initialize sigma points array
        sigma_points = np.zeros((L, 2 * L + 1))

        # First sigma point is the mean
        sigma_points[:,0] = x

        # Generate the remaining sigma points
        for i in range(L):
            sigma_points[:,i + 1] = x + gamma * sqrt_P[:, i]
            sigma_points[:,i + 1 + L] = x - gamma * sqrt_P[:, i]

        return sigma_points
    
    def propagate_sigma_points(self, sigma_points : np.ndarray, dt : float):
        """
        Propagates the sigma points through the process model to predict the next state.
        Parameters:
        sigma_points : np.ndarray
            Matrix of sigma points to be propagated.
        dt : float
            Time step for propagation.
        Returns:
        predicted_sigma_points : np.ndarray
            Matrix of predicted sigma points after propagation.
        """
        [_, predicted_sigma_points] = self.integrator.integrate_eom(dt, sigma_points.flatten(order='F'), teval=np.array([0, dt]), sigma_points=True)

        # Put the predicted sigma points back into the correct shape
        predicted_sigma_points = predicted_sigma_points[:, -1].reshape(sigma_points.shape, order='F')

        return predicted_sigma_points
    
    def time_update(self, sigma_points : np.ndarray, Wm : np.ndarray, Wc : np.ndarray, Q = None, delta_t = None):
        """
        Performs the time update (prediction step) of the UKF by computing the predicted mean and covariance.
        Parameters:
        sigma_points : np.ndarray
            Matrix of predicted sigma points after propagation.
        Wm : np.ndarray
            Weights for the mean.
        Wc : np.ndarray
            Weights for the covariance.
        Q : np.ndarray, optional
            Process noise covariance matrix. If provided, it will be added to the predicted covariance. Default is None.
        Returns:
        x_pred : np.ndarray
            Predicted state mean vector.
        P_pred : np.ndarray
            Predicted state covariance matrix.
        """
        # Compute the predicted state mean
        x_pred = np.dot(sigma_points, Wm)

        # Compute the predicted state covariance
        P_pred = np.zeros((len(x_pred), len(x_pred)))
        for i in range(sigma_points.shape[1]):
            diff = sigma_points[:, i] - x_pred
            P_pred += Wc[i] * np.outer(diff, diff)

        if Q is not None:
            Gamma = delta_t * np.concatenate((0.5 * delta_t * np.eye(3), np.eye(3)), axis=0)
            
            num_additional_states = x_pred.shape[0] - 6
            if num_additional_states > 0:
                Gamma = np.pad(Gamma, ((0, num_additional_states), (0, 0)), mode='constant')
            Q_augmented = Gamma @ Q @ Gamma.T

            P_pred += Q_augmented

        return x_pred, P_pred
    
    def compute_measurement_prediction(self, sigma_points : np.ndarray, measurement_mgr : MeasurementMgr, Wm : np.ndarray, time):
        """
        Computes the predicted measurement for each sigma point based on the measurement model.
        Parameters:
        sigma_points : np.ndarray
            Matrix of predicted sigma points after propagation.
        measurement_mgr : MeasurementMgr
            An instance of the MeasurementMgr class for the specific ground station.
        Wm : np.ndarray
            Weights for the mean, used to compute the predicted measurement mean.
        time : float
            Current time step for which to compute the predicted measurements.
        Returns:
        y_bar : np.ndarray
            Single predicted measurement vector computed as the weighted mean of the measurements from each sigma point.
        predicted_measurements : np.ndarray
            Matrix of predicted measurements for each sigma point.
        """
        predicted_measurements = np.zeros((2, sigma_points.shape[1]))

        for i in range(sigma_points.shape[1]):
            state_i = sigma_points[:, i]
            predicted_measurements[:, i] = measurement_mgr.simulate_measurements(state_i.reshape(-1, 1), np.array([time]), coordinate_frame='ECI', noise=False, ignore_visibility=True).flatten()

        # Compute the predicted measurement mean
        y_bar = np.dot(predicted_measurements, Wm)
        
        return y_bar, predicted_measurements
    
    def compute_cross_covariances(self, sigma_points : np.ndarray, predicted_measurements : np.ndarray, x_bar : np.ndarray, y_bar : np.ndarray, Wc : np.ndarray, R = None):
        """
        Computes the predicted measurement covariance matrix based on the predicted measurements and their mean.
        Parameters:
        sigma_points : np.ndarray
            Matrix of predicted sigma points after propagation.
        predicted_measurements : np.ndarray
            Matrix of predicted measurements for each sigma point.
        x_bar : np.ndarray
            Predicted state mean vector computed from the sigma points.
        y_bar : np.ndarray
            Single predicted measurement vector computed as the weighted mean of the measurements from each sigma point.
        Wc : np.ndarray
            Weights for the covariance, used to compute the measurement covariance.
        R : np.ndarray, optional
            Measurement noise covariance matrix. If provided, it will be added to the predicted measurement covariance. Default is None.
        Returns:
        P_yy : np.ndarray
            Predicted measurement covariance matrix.
        """
        P_yy = np.zeros((len(y_bar), len(y_bar)))
        P_xy = np.zeros((len(x_bar), len(y_bar)))
        for i in range(predicted_measurements.shape[1]):
            y_diff = predicted_measurements[:, i] - y_bar
            x_diff = sigma_points[:, i] - x_bar
            P_yy += Wc[i] * np.outer(y_diff, y_diff)
            P_xy += Wc[i] * np.outer(x_diff, y_diff)

        if R is not None:
            P_yy += R

        return P_yy, P_xy
    
    def measurement_update(self, x_bar : np.ndarray, P_bar : np.ndarray, y_bar : np.ndarray, P_yy : np.ndarray, P_xy : np.ndarray, y_meas : np.ndarray):
        """
        Performs the measurement update (correction step) of the UKF by computing the Kalman gain and updating the state estimate and covariance.
        Parameters:
        x_bar : np.ndarray
            Predicted state mean vector computed from the sigma points.
        P_bar : np.ndarray
            Predicted state covariance matrix computed from the sigma points.
        y_bar : np.ndarray
            Single predicted measurement vector computed as the weighted mean of the measurements from each sigma point.
        P_yy : np.ndarray
            Predicted measurement covariance matrix.
        P_xy : np.ndarray
            Cross-covariance matrix between the state and measurement.
        y_meas : np.ndarray
            Actual measurement vector obtained from the ground station.
        Returns:
        x_updated : np.ndarray
            Updated state mean vector after incorporating the measurement.
        P_updated : np.ndarray
            Updated state covariance matrix after incorporating the measurement.
        """
        # Compute Kalman gain
        K = P_xy @ np.linalg.inv(P_yy)

        # Update state estimate and covariance
        x_updated = x_bar + K @ (y_meas - y_bar)
        P_updated = P_bar - K @ P_yy @ K.T
        P_updated = 0.5 * (P_updated + P_updated.T)  # Force symmetry

        return x_updated, P_updated
    
    def compute_prefit_residuals(self, initial_state : np.ndarray, time_vector : np.ndarray, measurement_data : pd.DataFrame, residuals_df : pd.DataFrame):
        """
        Computes the pre-fit residuals for the UKF by simulating measurements based on the initial state and comparing them to the actual measurements.
        Parameters:
        initial_state : np.ndarray
            Initial state vector for the UKF.
        time_vector : np.ndarray
            1D array of time points at which to compute the pre-fit residuals.
        measurement_data : pd.DataFrame
            DataFrame containing the measurement data.
        residuals_df : pd.DataFrame
            DataFrame to which the computed pre-fit residuals will be appended. It should have columns for 'iteration', 'station', 'pre-fit', and 'post-fit'.
        Returns:
        residuals : np.ndarray
            Matrix of pre-fit residuals for each measurement at each time step.
        """

        [_, reference_state_history] = self.integrator.integrate_eom(time_vector[-1], initial_state, teval=time_vector)

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

            residuals_df = pd.concat([residuals_df, pd.DataFrame({'iteration': 0, 'station': station_name, 'pre-fit': [residual_vector], 'post-fit': np.nan})], ignore_index=True)
            
        return residuals_df
    
    def run(self, initial_state : np.ndarray, initial_covariance : np.ndarray, time_vector : np.ndarray, measurement_data : pd.DataFrame, alpha : float = 1e-3, beta : float = 2, Q = None, R = None):
        """
        Runs the Unscented Kalman Filter (UKF) for state estimation over a given time vector and measurement data.
        Parameters:
        initial_state : np.ndarray
            Initial state vector for the UKF.
        initial_covariance : np.ndarray
            Initial state covariance matrix for the UKF.
        time_vector : np.ndarray
            1D array of time points at which to perform the UKF estimation.
        measurement_data : pd.DataFrame
            DataFrame containing the measurement data.
        alpha : float, optional
            Spread of the sigma points. Default is 1e-3.
        beta : float, optional
            Incorporates prior knowledge of the distribution. Default is 2 for Gaussian distributions.
        Q : np.ndarray, optional
            Process noise covariance matrix. If provided, it will be added to the predicted covariance during the time update step. Default is None.
        R : np.ndarray, optional
            Measurement noise covariance matrix. If provided, it will be added to the predicted measurement covariance during the measurement update step. Default is None.
        """
        # Print inputted parameters for debugging
        np.set_printoptions(linewidth=200, precision=4, suppress=True)
        print("Beginning UKF run...")
        print("Inserted parameters:")
        print(f"Initial state: {initial_state}")
        print(f"Initial covariance: {initial_covariance}")
        print(f"Time vector: {time_vector}")
        print(f"Measurement data: {measurement_data}")
        print(f"Alpha: {alpha}")
        print(f"Beta: {beta}")

        # Initialize DataFrame to store residuals
        residuals_df = pd.DataFrame(columns=['iteration', 'station', 'pre-fit', 'post-fit'])

        # Compute UKF weights
        L = len(initial_state)
        Wm, Wc, gamma = self.compute_weights(alpha, beta, L)

        # Initialize state and covariance
        x_est = initial_state.copy()
        P_est = initial_covariance.copy()
        estimated_states = np.zeros((len(initial_state), len(time_vector)))
        estimated_covariances = np.zeros((len(initial_state), len(initial_state), len(time_vector)))


        # Begin UKF loop over time vector
        for k, t in enumerate(time_vector):
            print(f"Current Progress: Time = {t:.2f} of {time_vector[-1]} seconds", end='\r')
            # Compute sigma points
            sigma_points = self.compute_sigma_points(x_est, P_est, gamma)
            # Propagate sigma points through process model
            if t == time_vector[0]:
                dt = 0
                predicted_sigma_points = sigma_points
            else:
                dt=t - time_vector[k-1]
                predicted_sigma_points = self.propagate_sigma_points(sigma_points, dt=dt)

            # Time update to get predicted state mean and covariance
            x_bar, P_bar = self.time_update(predicted_sigma_points, Wm, Wc, Q, dt)

            # Pull out measurement data for current time step. If all measurements are NaN, skip measurement update
            current_measurements_df = measurement_data.iloc[k].values
            current_measurements = np.vstack((current_measurements_df[1], current_measurements_df[2], current_measurements_df[3]))

            if np.isnan(current_measurements).all():
                x_est = x_bar
                P_est = P_bar
            else:
                # Determine which measurement manager to use based on which measurements are available
                mgr_num = np.where(~np.isnan(current_measurements))[0][0]  # Get the index of the first non-NaN measurement <----- Assumes that only one measurement is available for any given time
                measurement_mgr = self.measurement_mgrs[mgr_num]

                # Compute predicted measurement and measurement covariance
                y_bar, predicted_measurements = self.compute_measurement_prediction(predicted_sigma_points, measurement_mgr, Wm, t)
                P_yy, P_xy = self.compute_cross_covariances(predicted_sigma_points, predicted_measurements, x_bar, y_bar, Wc, R)

                # Measurement update to get updated state mean and covariance
                x_est, P_est = self.measurement_update(x_bar, P_bar, y_bar, P_yy, P_xy, current_measurements[mgr_num, :])
            
            # Store estimated state and covariance
            estimated_states[:, k] = x_est
            estimated_covariances[:, :, k] = P_est

        print("\nUKF run complete.")

        # Add pre-fit residuals to DataFrame
        residuals_df = self.compute_prefit_residuals(initial_state, time_vector, measurement_data, residuals_df)

        # Add post-fit residuals to DataFrame
        for i, mgr in enumerate(self.measurement_mgrs):
            station_name = self.measurement_mgrs[i].station_name

            # Simulate measurements using updated state estimate
            simulated_measurements = mgr.simulate_measurements(estimated_states[0:6,:], time_vector, 'ECI', noise=False, ignore_visibility=True)
            truth_measurements = np.vstack(measurement_data[f"{station_name}_measurements"].values).T
            measurement_residuals = truth_measurements - simulated_measurements

            mask = (residuals_df['iteration'] == 0) & (residuals_df['station'] == station_name)

            idx = residuals_df[mask].index[0]  # Get the index of the matching row
            residuals_df.at[idx, 'post-fit'] = measurement_residuals

        return estimated_states, estimated_covariances, residuals_df

