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

    def time_update(self, x_hat : np.ndarray, P_hat : np.ndarray, S_hat : np.ndarray, phi : np.ndarray, H_x : np.ndarray, H_c : np.ndarray, R : np.ndarray):
        """
        Perform a time update for the consider covariance matrix.
        Parameters
        ----------
        x_hat : np.ndarray
            The current state estimate of the system.
        P_hat : np.ndarray
            The current covariance estimate of the system.
        S_hat : np.ndarray
            
        phi : np.ndarray
            The state transition matrix for the system.
        H_x : np.ndarray
            The measurement Jacobian with respect to the state.
        H_c : np.ndarray
            The measurement Jacobian with respect to the consider parameters.
        R : np.ndarray
            The measurement noise covariance matrix.
        Returns
        -------
        P_consider : np.ndarray
            The updated consider covariance matrix after the time update.
        """

        # Compute the Kalman gain for the consider parameters
        S = H_x @ P_hat @ H_x.T + R
        K_c = P_hat @ H_x.T @ np.linalg.inv(S) @ H_c

        # Update the consider covariance matrix
        P_consider = phi @ P_hat @ phi.T + K_c @ R @ K_c.T

        return P_consider