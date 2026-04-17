import numpy as np
from scipy.stats import chi2

class AdaptiveSNC:
    def __init__(self, alpha : float, window : float, Q_adaptive : np.ndarray, measurement_dimensionality : int = 2):
        """
        This class implements an adaptive sequential noise covariance (SNC) algorithm for outlier detection in measurement data relying on chi squared distribution filter consistency.
        Parameters:
        alpha : float
            Significance level for outlier detection (e.g., 0.05 for 95% confidence).
        window : float
            Length of previous residuals to include in chi squared mean test.
        Q_adaptive : float
            Process noise covariance used to inflate covariance.
        measurement_dimensionality : int, optional
            Dimensionality of the measurement data (e.g., 2 for range and range rate). Default is 2 assuming measurements are range and range rate.
        """
        self.alpha = alpha
        self.window = window
        self.Q_adaptive = Q_adaptive
        self.measurement_dimensionality = measurement_dimensionality
        self.epsilon_window = []

    def add_Q_adaptive(self, residual : np.ndarray, H : np.ndarray, P_bar : np.ndarray, R : np.ndarray):
        """
        Checks if window of residuals pass chi squared consistency tests. Returns needed addition to 6D state covariance to pass chi squared for adaptive SNC.

        Parameters:
        residual : np.ndarray
            Measurement residual vector (e.g., range and range rate residuals).
        H : np.ndarray
            Measurement Jacobian matrix.
        P_bar : np.ndarray
            Prior state covariance matrix.
        R : np.ndarray
            Measurement noise covariance matrix.
        Returns:
        boolean
            True if the residuals in the window indicate an outlier and adaptive SNC should be applied, False otherwise.
        """
        # Compute the innovation covariance
        S = H @ P_bar @ H.T + R

        # Compute the normalized squared residual
        dof = self.measurement_dimensionality * self.window  # Degrees of freedom for the chi squared test
        epsilon = residual.T @ np.linalg.inv(S) @ residual

        # Append the normalized squared residual to the window
        self.epsilon_window.append(epsilon)

        # If the window is full, perform the chi squared test
        if len(self.epsilon_window) > self.window:
            self.epsilon_window.pop(0)  # Remove the oldest residual

        if len(self.epsilon_window) == self.window:
            mean_normalized_squared_residual = np.mean(self.epsilon_window)
            threshold = chi2.ppf(1 - self.alpha, df=dof) / self.window
            return mean_normalized_squared_residual > threshold  # Outlier detected, return adaptive process noise covariance
        else:
            return False 