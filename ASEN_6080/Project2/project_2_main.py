import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import scipy.io
from Tools.measurement_manager import MeasurementMgr
from Tools.integrator import Integrator
from Tools.EKF import EKF
from Tools.adaptive_snc import AdaptiveSNC
from Tools.plotting_functions import plot_residuals, plot_state_errors

# Generic imports
from constants import observation_noise, initial_epoch, initial_epoch_jd
# from constants import unknown_dynamics_measurement_file_path, known_dynamics_measurement_file_path, truth_data_file_path, a_priori_state, a_priori_covariance, observation_noise, initial_spin_angle, earth_spin_rate, station_locations
# from constants import C_r, mu_sun, mu_earth, R_e, solar_flux, SRP_area_to_mass, initial_spin_angle, earth_spin_rate

# Import analysis scripts
from generic_functions import load_measurement_data, load_truth_data, convert_measurements_to_df, initialize_integrator
from task_1_analysis import run_task_1_analysis
from task_2_analysis import run_task_2_analysis, run_task_2_B_plane_analysis
from task_3_analysis import run_task_3_analysis

np.set_printoptions(linewidth=200)

if __name__ == "__main__":
    # ----------------------------------------------------------------------------------------------------------------------------
    # TASK 1 ANALYSIS
    # ----------------------------------------------------------------------------------------------------------------------------
    # print("=" * 50)
    # print(f"Running Task 1 Dynamics Test...")
    # print("=" * 50 + "\n")
    # run_task_1_analysis()

    # ----------------------------------------------------------------------------------------------------------------------------
    # TASK 2 ANALYSIS
    # ----------------------------------------------------------------------------------------------------------------------------
    # print("\n" + "=" * 50)
    # print(f"Running Task 2 Known Dynamics Estimation...")
    # print("=" * 50 + "\n")
    # # run_task_2_analysis(period_of_data=[0, 51],
    # #                     filters_to_run=['Batch', 'LKF', 'EKF', 'SRIF'],
    # #                     iterations_for_filters=[5, 5, 1, 5],
    # #                     tol_for_filters=[1e-6, 1e-6, 1e-6, 1e-6],
    # #                     ekf_start_mode='warm',
    # #                     ekf_start_length=100)
    # run_task_2_B_plane_analysis(filters_to_run=['EKF'], process_noise='SNC', Q=np.diag([1e-22, 1e-22, 1e-22]))

    # # ----------------------------------------------------------------------------------------------------------------------------
    # # TASK 3 FINAL ANALYSIS
    # # ----------------------------------------------------------------------------------------------------------------------------
    print("=" * 50)
    print(f"Running Task 3 Final Analysis...")
    print("=" * 50)

    # Measurement data time period to use (in days)
    period_of_data = [0, 250]

    # Estimation Parameters
    estimation_mode = ['SRP', 'Stations']
    parameter_indices = [6, 7]

    DSS_34_cov = [1e-16, 1e-16, 1e-16]
    DSS_65_cov = [0.1, 0.1, 0.1]
    DSS_13_cov = [1e-16, 1e-16, 1e-16]

    # EKF start mode parameters
    start_mode = 'warm'
    start_length = 100

    # Covariance reset parameters
    mnvr_day = 217
    mnvr_time = mnvr_day * 24 * 3600
    mnvr_reset_covariance = np.diag([10, 10, 10, 0.5, 0.5, 0.5, 0.05, 1e-8, 1e-8, 1e-8, 1e-3, 1e-3, 1e-3, 1e-8, 1e-8, 1e-8])**2

    # SNC parameters
    process_noise_type = 'SNC'
    Q = np.diag([5e-10, 5e-10, 5e-10])**2

    # Adapative SNC parameters
    alpha = 0.005
    window = 10
    Q_adaptive = 5e-7 # 1 mm/s adaptive process noise for velocity states

    filter_name = f'EKF (DSS65 Estimated with SNC)'

    run_task_3_analysis(period_of_data,
                        estimation_mode,
                        parameter_indices,
                        DSS_34_cov,
                        DSS_65_cov,
                        DSS_13_cov,
                        start_mode,
                        start_length,
                        mnvr_day,
                        mnvr_reset_covariance,
                        process_noise_type,
                        Q,
                        alpha,
                        window,
                        Q_adaptive,
                        filter_name)
    


