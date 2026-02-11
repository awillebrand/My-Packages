import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from ASEN_6080.Tools import Integrator, MeasurementMgr, CoordinateMgr, LKF, BatchLLSEstimator, covariance_ellipse
from plotly.subplots import make_subplots

measurements = pd.read_pickle("./ASEN_6080/Project1/data/conditioned_measurements.pkl")
time_vector = measurements['time'].values

# Initialize scenario

sat_state = np.array([757700.0E-3, 5222607.0E-3, 4851500.0E-3, 2213.21E-3, 4678.34E-3, -5371.30E-3])  # Satellite state in km and km/s
mu = 3.986004415E5  # Earth's gravitational parameter in km^3/s^2
J2 = 1.082626925638815E-3 # Earth's J2 coefficient
J3 = 0.0 # Earth's J3 coefficient
R_e = 6378.1363  # Earth's radius in km
C_d = 2.0 # Drag coefficient
spacecraft_mass = 970.0  # Spacecraft mass in kg
spacecraft_area = 3.0  # Spacecraft cross-sectional area in m^2Q
earth_spin_rate = 7.2921158553E-5  # Earth's rotation rate in rad/s

station_1_state = np.array([-5127510.0E-3, -3794160.0E-3,  0.0, 0.0, 0.0, 0.0])
station_2_state = np.array([3860910.0E-3, 3238490.0E-3,  3898094.0E-3, 0.0, 0.0, 0.0])
station_3_state = np.array([549505.0E-3, -1380872.0E-3,  6182197.0E-3, 0.0, 0.0, 0.0])

# Set up filter parameters

station_positions_ecef = np.array([station_1_state[0:3], station_2_state[0:3], station_3_state[0:3]])
initial_state_estimate = np.concatenate([sat_state[0:6], [mu, J2, C_d], station_1_state[0:3], station_2_state[0:3], station_3_state[0:3]]).flatten()

R = np.diag([1E-5**2, 1E-6**2])  # Noise covariance matrix for range and range rate. Corresponds to 1 cm range noise and 1 mm/s range rate noise.
a_priori_covariance = np.diag([1, 1, 1, 1, 1, 1, 1E2, 1E6, 1E6, 1E-16, 1E-16, 1E-16, 1, 1, 1, 1, 1, 1])  # Given a priori covariance

# Initialize integrator and measurement managers
integrator = Integrator(mu, R_e, mode=['mu','J2','Drag','Stations'], parameter_indices=[6,7,8,9], spacecraft_area=spacecraft_area, spacecraft_mass=spacecraft_mass, number_of_stations=3)

station_1_mgr = MeasurementMgr("station_101", station_state_ecef=station_1_state, initial_earth_spin_angle=0.0, earth_spin_rate=earth_spin_rate, R_e=R_e)
station_2_mgr = MeasurementMgr("station_337", station_state_ecef=station_2_state, initial_earth_spin_angle=0.0, earth_spin_rate=earth_spin_rate, R_e=R_e)
station_3_mgr = MeasurementMgr("station_394", station_state_ecef=station_3_state, initial_earth_spin_angle=0.0, earth_spin_rate=earth_spin_rate, R_e=R_e)

station_mgr_list = [station_1_mgr, station_2_mgr, station_3_mgr]

# Run filters
batch_estimator = BatchLLSEstimator(integrator, station_mgr_list, initial_earth_spin_angle=0.0, earth_rotation_rate=earth_spin_rate)
estimated_initial_state, estimated_covariance, batch_residuals_df = batch_estimator.estimate_initial_state(
    a_priori_state=initial_state_estimate,
    a_priori_covariance=a_priori_covariance,
    measurement_data=measurements,
    R=R,
    max_iterations=3,
    tol=1E-6,
    considered_measurements='All')

# Reset measurement managers positions
for i, mgr in enumerate(batch_estimator.measurement_mgrs):
    mgr.station_state_ecef[0:3] = station_positions_ecef[i]
    mgr.lat, mgr.lon = mgr.coordinate_mgr.ECEF_to_GCS(mgr.station_state_ecef)

lkf = LKF(integrator, station_mgr_list, initial_earth_spin_angle=0.0, earth_rotation_rate=earth_spin_rate)
lkf_state_history, lkf_covariance_history, lkf_residuals_df = lkf.run(initial_state_estimate,
                                                                      np.zeros_like(initial_state_estimate),
                                                                      a_priori_covariance, measurements,
                                                                      R=R, max_iterations=1,

                                                                      convergence_threshold=1e-9,
                                                                      considered_measurements='All')

for i, mgr in enumerate(lkf.measurement_mgrs):
    mgr.station_state_ecef[0:3] = station_positions_ecef[i]
    mgr.lat, mgr.lon = mgr.coordinate_mgr.ECEF_to_GCS(mgr.station_state_ecef)

# Run filters with reduced measurement sets for comparison
range_estimated_initial_state, range_estimated_initial_covariance, range_batch_residuals_df = batch_estimator.estimate_initial_state(
    a_priori_state=initial_state_estimate,
    a_priori_covariance=a_priori_covariance,
    measurement_data=measurements,
    R=R,
    max_iterations=3,
    tol=1E-6,
    considered_measurements='Range')

# Reset measurement managers positions
for i, mgr in enumerate(batch_estimator.measurement_mgrs):
    mgr.station_state_ecef[0:3] = station_positions_ecef[i]
    mgr.lat, mgr.lon = mgr.coordinate_mgr.ECEF_to_GCS(mgr.station_state_ecef)

range_rate_estimated_initial_state, range_rate_estimated_initial_covariance, range_rate_batch_residuals_df = batch_estimator.estimate_initial_state(
    a_priori_state=initial_state_estimate,
    a_priori_covariance=a_priori_covariance,
    measurement_data=measurements,
    R=R,
    max_iterations=3,
    tol=1E-6,
    considered_measurements='Range Rate')

# Reset measurement managers positions
for i, mgr in enumerate(batch_estimator.measurement_mgrs):
    mgr.station_state_ecef[0:3] = station_positions_ecef[i]
    mgr.lat, mgr.lon = mgr.coordinate_mgr.ECEF_to_GCS(mgr.station_state_ecef)

range_lkf_state_history, range_lkf_covariance_history, range_lkf_residuals_df = lkf.run(initial_state_estimate,
                                                                        np.zeros_like(initial_state_estimate),
                                                                        a_priori_covariance, measurements,
                                                                        R=R, max_iterations=1,
                                                                        convergence_threshold=1e-9,
                                                                        considered_measurements='Range')

for i, mgr in enumerate(lkf.measurement_mgrs):
    mgr.station_state_ecef[0:3] = station_positions_ecef[i]
    mgr.lat, mgr.lon = mgr.coordinate_mgr.ECEF_to_GCS(mgr.station_state_ecef)

range_rate_lkf_state_history, range_rate_lkf_covariance_history, range_rate_lkf_residuals_df = lkf.run(initial_state_estimate,
                                                                        np.zeros_like(initial_state_estimate),
                                                                        a_priori_covariance, measurements,
                                                                        R=R, max_iterations=1,
                                                                        convergence_threshold=1e-9,
                                                                        considered_measurements='Range Rate')

for i, mgr in enumerate(lkf.measurement_mgrs):
    mgr.station_state_ecef[0:3] = station_positions_ecef[i]
    mgr.lat, mgr.lon = mgr.coordinate_mgr.ECEF_to_GCS(mgr.station_state_ecef)

# Testing fixed station position scenarios

a_priori_covariance = np.diag([1, 1, 1, 1, 1, 1, 1E2, 1E6, 1E6, 1, 1, 1, 1, 1, 1, 1, 1, 1])  # No fixed station position covariance

non_fixed_batch_estimated_initial_state, non_fixed_estimated_covariance, non_fixed_batch_residuals_df = batch_estimator.estimate_initial_state(
    a_priori_state=initial_state_estimate,
    a_priori_covariance=a_priori_covariance,
    measurement_data=measurements,
    R=R,
    max_iterations=4,
    tol=1E-6,
    considered_measurements='All')

# Reset measurement managers positions
for i, mgr in enumerate(batch_estimator.measurement_mgrs):
    mgr.station_state_ecef[0:3] = station_positions_ecef[i]
    mgr.lat, mgr.lon = mgr.coordinate_mgr.ECEF_to_GCS(mgr.station_state_ecef)

# Fixing Station 337
a_prior_covariance = np.diag([1, 1, 1, 1, 1, 1, 1E2, 1E6, 1E6, 1, 1, 1, 1E-16, 1E-16, 1E-16, 1, 1, 1])  # Given a priori covariance

fixed_station_2_batch_estimated_initial_state, fixed_station_2_estimated_covariance, fixed_station_2_batch_residuals_df = batch_estimator.estimate_initial_state(
    a_priori_state=initial_state_estimate,
    a_priori_covariance=a_prior_covariance,
    measurement_data=measurements,
    R=R,
    max_iterations=4,
    tol=1E-6,
    considered_measurements='All')

# Reset measurement managers positions
for i, mgr in enumerate(batch_estimator.measurement_mgrs):
    mgr.station_state_ecef[0:3] = station_positions_ecef[i]
    mgr.lat, mgr.lon = mgr.coordinate_mgr.ECEF_to_GCS(mgr.station_state_ecef)

# Fixing Station 394
a_prior_covariance = np.diag([1, 1, 1, 1, 1, 1, 1E2, 1E6, 1E6, 1, 1, 1, 1, 1, 1, 1E-16, 1E-16, 1E-16])  # Given a priori covariance

fixed_station_3_batch_estimated_initial_state, fixed_station_3_estimated_covariance, fixed_station_3_batch_residuals_df = batch_estimator.estimate_initial_state(
    a_priori_state=initial_state_estimate,
    a_priori_covariance=a_prior_covariance,
    measurement_data=measurements,
    R=R,
    max_iterations=4,
    tol=1E-6,
    considered_measurements='All')

# Reset measurement managers positions
for i, mgr in enumerate(batch_estimator.measurement_mgrs):
    mgr.station_state_ecef[0:3] = station_positions_ecef[i]
    mgr.lat, mgr.lon = mgr.coordinate_mgr.ECEF_to_GCS(mgr.station_state_ecef)

# Changing a priori covariance to be more compatible with actual errors
a_priori_covariance = np.diag([1E-3, 1E-3, 1E-3, 1E-3, 1E-3, 1E-3, 1, 1E-4, 10, 1E-8, 1E-8, 1E-8, 1E-1, 1E-1, 1E-1, 1E-1, 1E-1, 1E-1])**2  # Altered a priori covariance

reasonable_batch_estimated_initial_state, reasonable_estimated_covariance, reasonable_batch_residuals_df = batch_estimator.estimate_initial_state(
    a_priori_state=initial_state_estimate,
    a_priori_covariance=a_priori_covariance,
    measurement_data=measurements,
    R=R,
    max_iterations=3,
    tol=1E-6,
    considered_measurements='All')

for i, mgr in enumerate(lkf.measurement_mgrs):
    mgr.station_state_ecef[0:3] = station_positions_ecef[i]
    mgr.lat, mgr.lon = mgr.coordinate_mgr.ECEF_to_GCS(mgr.station_state_ecef)

reasonable_lkf_state_history, reasonable_lkf_covariance_history, reasonable_lkf_residuals_df = lkf.run(initial_state_estimate,
                                                                        np.zeros_like(initial_state_estimate),
                                                                        a_priori_covariance, measurements,
                                                                        R=R, max_iterations=1,
                                                                        convergence_threshold=1e-9,
                                                                        considered_measurements='All')

for i, mgr in enumerate(lkf.measurement_mgrs):
    mgr.station_state_ecef[0:3] = station_positions_ecef[i]
    mgr.lat, mgr.lon = mgr.coordinate_mgr.ECEF_to_GCS(mgr.station_state_ecef)

# Change data noise covariance to be larger than truth
a_priori_covariance = np.diag([1, 1, 1, 1, 1, 1, 1E2, 1E6, 1E6, 1E-16, 1E-16, 1E-16, 1, 1, 1, 1, 1, 1])  # Given a priori covariance
R = np.diag([1E-4**2, 1E-5**2])  # Noise covariance matrix for range and range rate. Corresponds to 10 cm range noise and 10 mm/s range rate noise.
underconfident_batch_estimated_initial_state, underconfident_estimated_covariance, underconfident_batch_residuals_df = batch_estimator.estimate_initial_state(
    a_priori_state=initial_state_estimate,
    a_priori_covariance=a_priori_covariance,
    measurement_data=measurements,
    R=R,
    max_iterations=3,
    tol=1E-6,
    considered_measurements='All')

for i, mgr in enumerate(lkf.measurement_mgrs):
    mgr.station_state_ecef[0:3] = station_positions_ecef[i]
    mgr.lat, mgr.lon = mgr.coordinate_mgr.ECEF_to_GCS(mgr.station_state_ecef)

underconfident_lkf_state_history, underconfident_lkf_covariance_history, underconfident_lkf_residuals_df = lkf.run(initial_state_estimate,
                                                                        np.zeros_like(initial_state_estimate),
                                                                        a_priori_covariance, measurements,
                                                                        R=R, max_iterations=1,
                                                                        convergence_threshold=1e-9,
                                                                        considered_measurements='All')

for i, mgr in enumerate(lkf.measurement_mgrs):
    mgr.station_state_ecef[0:3] = station_positions_ecef[i]
    mgr.lat, mgr.lon = mgr.coordinate_mgr.ECEF_to_GCS(mgr.station_state_ecef)

# Change data noise covariance to be smaller than truth
R = np.diag([1E-6**2, 1E-7**2])  # Noise covariance matrix for range and range rate. Corresponds to 0.1 cm range noise and 0.1 mm/s range rate noise.
overconfident_batch_estimated_initial_state, overconfident_estimated_covariance, overconfident_batch_residuals_df = batch_estimator.estimate_initial_state(
    a_priori_state=initial_state_estimate,
    a_priori_covariance=a_priori_covariance,
    measurement_data=measurements,
    R=R,
    max_iterations=3,
    tol=1E-6,
    considered_measurements='All')

# Update measurement managers to new estimated initial state for covariance propagation
for i, mgr in enumerate(station_mgr_list):
    new_station_position = estimated_initial_state[9+3*i:12+3*i]
    mgr.station_state_ecef[0:3] = new_station_position
    mgr.lat, mgr.lon = mgr.coordinate_mgr.ECEF_to_GCS(new_station_position)

overconfident_lkf_state_history, overconfident_lkf_covariance_history, overconfident_lkf_residuals_df = lkf.run(initial_state_estimate,
                                                                        np.zeros_like(initial_state_estimate),
                                                                        a_priori_covariance, measurements,
                                                                        R=R, max_iterations=1,
                                                                        convergence_threshold=1e-9,
                                                                        considered_measurements='All')

for i, mgr in enumerate(lkf.measurement_mgrs):
    mgr.station_state_ecef[0:3] = station_positions_ecef[i]
    mgr.lat, mgr.lon = mgr.coordinate_mgr.ECEF_to_GCS(mgr.station_state_ecef)

# Integrate batch estimated initial state forward for comparison
[_, batch_estimated_state_history] = integrator.integrate_stm(time_vector[-1], estimated_initial_state, teval=time_vector)
[_, range_batch_estimated_state_history] = integrator.integrate_stm(time_vector[-1], range_estimated_initial_state, teval=time_vector)
[_, range_rate_batch_estimated_state_history] = integrator.integrate_stm(time_vector[-1], range_rate_estimated_initial_state, teval=time_vector)
[_, non_fixed_batch_estimated_state_history] = integrator.integrate_stm(time_vector[-1], non_fixed_batch_estimated_initial_state, teval=time_vector)
[_, fixed_station_2_batch_estimated_state_history] = integrator.integrate_stm(time_vector[-1], fixed_station_2_batch_estimated_initial_state, teval=time_vector)
[_, fixed_station_3_batch_estimated_state_history] = integrator.integrate_stm(time_vector[-1], fixed_station_3_batch_estimated_initial_state, teval=time_vector)
[_, reasonable_batch_estimated_state_history] = integrator.integrate_stm(time_vector[-1], reasonable_batch_estimated_initial_state, teval=time_vector)
[_, underconfident_batch_estimated_state_history] = integrator.integrate_stm(time_vector[-1], underconfident_batch_estimated_initial_state, teval=time_vector)
[_, overconfident_batch_estimated_state_history] = integrator.integrate_stm(time_vector[-1], overconfident_batch_estimated_initial_state, teval=time_vector)

covariance_history = np.zeros((18, 18, len(time_vector)))
range_covariance_history = np.zeros((18, 18, len(time_vector)))
range_rate_covariance_history = np.zeros((18, 18, len(time_vector)))
non_fixed_covariance_history = np.zeros((18, 18, len(time_vector)))
fixed_station_2_covariance_history = np.zeros((18, 18, len(time_vector)))
fixed_station_3_covariance_history = np.zeros((18, 18, len(time_vector)))
reasonable_covariance_history = np.zeros((18, 18, len(time_vector)))
underconfident_covariance_history = np.zeros((18, 18, len(time_vector)))
overconfident_covariance_history = np.zeros((18, 18, len(time_vector)))

for i, time in enumerate(time_vector):
    stm_batch = batch_estimated_state_history[18:,i].reshape((18,18))
    covariance_history[:,:,i] = stm_batch @ estimated_covariance @ stm_batch.T
    stm_range = range_batch_estimated_state_history[18:,i].reshape((18,18))
    range_covariance_history[:,:,i] = stm_range @ range_estimated_initial_covariance @ stm_range.T
    stm_range_rate = range_rate_batch_estimated_state_history[18:,i].reshape((18,18))
    range_rate_covariance_history[:,:,i] = stm_range_rate @ range_rate_estimated_initial_covariance @ stm_range_rate.T
    stm_non_fixed = non_fixed_batch_estimated_state_history[18:,i].reshape((18,18))
    non_fixed_covariance_history[:,:,i] = stm_non_fixed @ non_fixed_estimated_covariance @ stm_non_fixed.T
    stm_fixed_station_2 = fixed_station_2_batch_estimated_state_history[18:,i].reshape((18,18))
    fixed_station_2_covariance_history[:,:,i] = stm_fixed_station_2 @ fixed_station_2_estimated_covariance @ stm_fixed_station_2.T
    stm_fixed_station_3 = fixed_station_3_batch_estimated_state_history[18:,i].reshape((18,18))
    fixed_station_3_covariance_history[:,:,i] = stm_fixed_station_3 @ fixed_station_3_estimated_covariance @ stm_fixed_station_3.T
    stm_reasonable = reasonable_batch_estimated_state_history[18:,i].reshape((18,18))
    reasonable_covariance_history[:,:,i] = stm_reasonable @ reasonable_estimated_covariance @ stm_reasonable.T
    stm_underconfident = underconfident_batch_estimated_state_history[18:,i].reshape((18,18))
    underconfident_covariance_history[:,:,i] = stm_underconfident @ underconfident_estimated_covariance @ stm_underconfident.T
    stm_overconfident = overconfident_batch_estimated_state_history[18:,i].reshape((18,18))
    overconfident_covariance_history[:,:,i] = stm_overconfident @ overconfident_estimated_covariance @ stm_overconfident.T

print("----------------------------------------------------")
print("Batch LLS Estimated Final State:")
print(batch_estimated_state_history[:18,-1])
print("Diagonal of Batch LLS Estimated Final Covariance:")
print(np.diag(covariance_history[:,:,-1]))
print("---------------------------------------------------")
print("LKF Estimated Final State:")
print(lkf_state_history[:18,-1])
print("Diagonal of LKF Estimated Final Covariance:")
print(np.diag(lkf_covariance_history[:,:,-1]))
print("---------------------------------------------------")
print("Batch LLS (Range Only) Estimated Final State:")
print(range_batch_estimated_state_history[:18,-1])
print("Diagonal of Batch LLS (Range Only) Estimated Final Covariance:")
print(np.diag(range_covariance_history[:,:,-1]))
print("---------------------------------------------------")
print("Batch LLS (Range Rate Only) Estimated Final State:")
print(range_rate_batch_estimated_state_history[:18,-1])
print("Diagonal of Batch LLS (Range Rate Only) Estimated Final Covariance:")
print(np.diag(range_rate_covariance_history[:,:,-1]))
print("---------------------------------------------------")
print("Batch LLS (No-Fixed Stations) Estimated Final State:")   
print(non_fixed_batch_estimated_state_history[:18,-1])
print("Diagonal of Batch LLS (No-Fixed Stations) Estimated Final Covariance:")
print(np.diag(non_fixed_covariance_history[:,:,-1]))
print("---------------------------------------------------")
print("Batch LLS (Station 2 Fixed) Estimated Final State:")
print(fixed_station_2_batch_estimated_state_history[:18,-1])
print("Diagonal of Batch LLS (Station 2 Fixed) Estimated Final Covariance:")
print(np.diag(fixed_station_2_covariance_history[:,:,-1]))
print("---------------------------------------------------")
print("Batch LLS (Station 3 Fixed) Estimated Final State:")
print(fixed_station_3_batch_estimated_state_history[:18,-1])
print("Diagonal of Batch LLS (Station 3 Fixed) Estimated Final Covariance:")
print(np.diag(fixed_station_3_covariance_history[:,:,-1]))
print("---------------------------------------------------")
print("Batch LLS (Reasonable A Priori Covariance) Estimated Final State:")
print(reasonable_batch_estimated_state_history[:18,-1])
print("Diagonal of Batch LLS (Reasonable A Priori Covariance) Estimated Final Covariance:")
print(np.diag(reasonable_covariance_history[:,:,-1]))
print("---------------------------------------------------")
print("LKF (Reasonable A Priori Covariance) Estimated Final State:")
print(reasonable_lkf_state_history[:18,-1])
print("Diagonal of LKF (Reasonable A Priori Covariance) Estimated Final Covariance:")
print(np.diag(reasonable_lkf_covariance_history[:,:,-1]))
print("---------------------------------------------------")
print("Batch LLS (Underconfident Data Noise) Estimated Final State:")
print(underconfident_batch_estimated_state_history[:18,-1])
print("Diagonal of Batch LLS (Underconfident Data Noise) Estimated Final Covariance:")
print(np.diag(underconfident_covariance_history[:,:,-1]))
print("---------------------------------------------------")
print("LKF (Underconfident Data Noise) Estimated Final State:")
print(underconfident_lkf_state_history[:18,-1])
print("Diagonal of LKF (Underconfident Data Noise) Estimated Final Covariance:")
print(np.diag(underconfident_lkf_covariance_history[:,:,-1]))
print("---------------------------------------------------")
print("Batch LLS (Overconfident Data Noise) Estimated Final State:")
print(overconfident_batch_estimated_state_history[:18,-1])
print("Diagonal of Batch LLS (Overconfident Data Noise) Estimated Final Covariance:")
print(np.diag(overconfident_covariance_history[:,:,-1]))
print("---------------------------------------------------")
print("LKF (Overconfident Data Noise) Estimated Final State:")
print(overconfident_lkf_state_history[:18,-1])
print("Diagonal of LKF (Overconfident Data Noise) Estimated Final Covariance:")
print(np.diag(overconfident_lkf_covariance_history[:,:,-1]))
print("---------------------------------------------------")
# PLOTTING -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Plot residual time history for each station
colors_list = ['red', 'green', 'blue']
residual_df_list = [batch_residuals_df,
                    lkf_residuals_df,
                    range_batch_residuals_df,
                    range_rate_batch_residuals_df,
                    non_fixed_batch_residuals_df,
                    fixed_station_2_batch_residuals_df,
                    fixed_station_3_batch_residuals_df,
                    reasonable_batch_residuals_df,
                    reasonable_lkf_residuals_df,
                    underconfident_batch_residuals_df,
                    underconfident_lkf_residuals_df,
                    overconfident_batch_residuals_df,
                    overconfident_lkf_residuals_df]

filter_names = ['Batch Filter',
                'LKF',
                'Batch Filter (Range Only)',
                'Batch Filter (Range Rate Only)',
                'Batch Filter (No-Fixed Stations)',
                'Batch Filter (Station 2 Fixed)',
                'Batch Filter (Station 3 Fixed)',
                'Batch Filter (Reasonable A Priori Covariance)',
                'LKF (Reasonable A Priori Covariance)',
                'Batch Filter (Underconfident Data Noise)',
                'LKF (Underconfident Data Noise)',
                'Batch Filter (Overconfident Data Noise)',
                'LKF (Overconfident Data Noise)']
for residuals_df, filter_name in zip(residual_df_list, filter_names):
    for iteration in range(residuals_df['iteration'].max()+1):
        # Combine station residuals into a single vector for RMS calculation, this can be done by adding all the station residuals together for the given iteration (since none overlap in timing)
        relevant_residuals = residuals_df[residuals_df['iteration'] == iteration]['pre-fit'].values.copy()
        for i in range(len(residuals_df['station'].unique())):
            # Set any NaN values to zero for RMS calculation
            relevant_residuals[i][np.isnan(relevant_residuals[i])] = 0.0
        
        # Sum the residuals across stations to get a single residual vector for the iteration
        combined_residuals = np.sum(relevant_residuals, axis=0)

        # Reset zeros to NaN so they aren't included in RMS calculation
        combined_residuals[combined_residuals == 0.0] = np.nan
        
        # Compute RMS of combined residuals for the iteration
        rms_range_residual = np.sqrt(np.abs(np.nanmean((combined_residuals[0,:]*1E5) **2))) # Convert from km to cm for RMS calculation
        rms_range_rate_residual = np.sqrt(np.abs(np.nanmean((combined_residuals[1,:]*1E6) **2))) # Convert from km/s to mm/s for RMS calculation

        # Find mean and standard deviation of range and range rate residuals for the iteration
        mean_range_residual = np.nanmean(combined_residuals[0,:]*1E5) # Convert from km to cm for mean calculation
        std_range_residual = np.nanstd(combined_residuals[0,:]*1E5) # Convert from km to cm for std calculation
        mean_range_rate_residual = np.nanmean(combined_residuals[1,:]*1E6) # Convert from km/s to mm/s for mean calculation
        std_range_rate_residual = np.nanstd(combined_residuals[1,:]*1E6) # Convert from km/s to mm/s for std calculation
        print(f"Pre-fit {filter_name} Iteration {iteration+1}:")
        print(f"Range Residuals: Mean = {mean_range_residual:.4f} cm, Std Dev = {std_range_residual:.4f} cm, RMS = {rms_range_residual:.4f} cm")
        print(f"Range Rate Residuals: Mean = {mean_range_rate_residual:.4f} mm/s, Std Dev = {std_range_rate_residual:.4f} mm/s, RMS = {rms_range_rate_residual:.4f} mm/s")
        print("--------------------------------------------------")

        # Reset zeros to NaN in individual station residuals as well for accurate RMS calculation
        for i in range(len(residuals_df['station'].unique())):
            # Set any NaN values to zero for RMS calculation
            relevant_residuals[i][relevant_residuals[i] == 0.0] = np.nan

        
        fig = make_subplots(
            rows=2, cols=2, 
            shared_xaxes=False,
            column_widths=[0.85, 0.15],
            horizontal_spacing=0.06,
            subplot_titles=(f'Range Residuals (Mean = {mean_range_residual:.4f} cm, Std Dev = {std_range_residual:.4f} cm, RMS = {rms_range_residual:.4f} cm)', 'Distribution',
                            f'Range Rate Residuals (Mean = {mean_range_rate_residual:.4f} mm/s, Std Dev = {std_range_rate_residual:.4f} mm/s, RMS = {rms_range_rate_residual:.4f} mm/s)', 'Distribution')
        )
        
        # Collect all residuals for histogram
        all_range_residuals = []
        all_range_rate_residuals = []
        
        for i, station_name in enumerate(residuals_df['station'].unique()):
            mask = (residuals_df['iteration'] == iteration) & (residuals_df['station'] == station_name)
            pre_fit_residuals = np.vstack(residuals_df[mask]['pre-fit'])
            
            # Add scatter plots (left column)
            fig.add_trace(go.Scatter(x=time_vector, y=pre_fit_residuals[0,:]*1E5, 
                                    mode='markers', name=f'{station_name}', 
                                    marker=dict(color=colors_list[i]), legendgroup=f'group{i}'), 
                         row=1, col=1)
            fig.add_trace(go.Scatter(x=time_vector, y=pre_fit_residuals[1,:]*1E6, 
                                    mode='markers', name=f'{station_name}', 
                                    marker=dict(color=colors_list[i]), 
                                    showlegend=False, legendgroup=f'group{i}'), 
                         row=2, col=1)
            
            # Collect valid (non-NaN) residuals for histograms
            valid_range = pre_fit_residuals[0,:][~np.isnan(pre_fit_residuals[0,:])] * 1E5
            valid_range_rate = pre_fit_residuals[1,:][~np.isnan(pre_fit_residuals[1,:])] * 1E6
            all_range_residuals.extend(valid_range)
            all_range_rate_residuals.extend(valid_range_rate)
        
        # Add histograms (right column) - rotated to be vertical
        fig.add_trace(go.Histogram(y=all_range_residuals, 
                                  marker=dict(color='lightblue'),
                                  showlegend=False,
                                  nbinsy=50), 
                     row=1, col=2)
        fig.add_trace(go.Histogram(y=all_range_rate_residuals, 
                                  marker=dict(color='lightcoral'),
                                  showlegend=False,
                                  nbinsy=50), 
                     row=2, col=2)
        
        fig.update_traces(marker=dict(size=4), selector=dict(mode='markers'))
        fig.update_xaxes(title_text="Time (s)", tickfont=dict(size=20), title_font=dict(size=22), row=1, col=1)
        fig.update_xaxes(title_text="Time (s)", tickfont=dict(size=20), title_font=dict(size=22), row=2, col=1)
        fig.update_xaxes(title_text="Count", tickfont=dict(size=20), title_font=dict(size=22), row=1, col=2)
        fig.update_xaxes(title_text="Count", tickfont=dict(size=20), title_font=dict(size=22), row=2, col=2)
        fig.update_yaxes(title_text="Range Residuals (cm)", tickfont=dict(size=20), title_font=dict(size=22), showexponent="all", exponentformat="e", row=1, col=1)
        fig.update_yaxes(title_text="Range Rate Residuals (mm/s)", tickfont=dict(size=20), title_font=dict(size=22), showexponent="all", exponentformat="e", row=2, col=1)
        fig.update_yaxes(showexponent="all", exponentformat="e", tickfont=dict(size=20), title_font=dict(size=22), row=1, col=2)
        fig.update_yaxes(showexponent="all", exponentformat="e", tickfont=dict(size=20), title_font=dict(size=22), row=2, col=2)
        fig.update_annotations(font=dict(size=24))
        fig.update_layout(title_text=f"{filter_name} Pre-Fit Residuals at Iteration {iteration+1}",
                        title_font=dict(size=30),
                        width=1900,  # Increased width to accommodate histograms
                        height=800,
                        legend=dict(font=dict(size=22),
                                    orientation="h",
                                    yanchor="top",
                                    y=1.13,
                                    xanchor="left",
                                    x=0.7,
                                    itemsizing='constant'))
        fig.show()

        # Combine station residuals into a single vector for RMS calculation, this can be done by adding all the station residuals together for the given iteration (since none overlap in timing)
        relevant_residuals = residuals_df[residuals_df['iteration'] == iteration]['post-fit'].values.copy()
        for i in range(len(residuals_df['station'].unique())):
            # Set any NaN values to zero for RMS calculation
            relevant_residuals[i][np.isnan(relevant_residuals[i])] = 0.0
        
        # Sum the residuals across stations to get a single residual vector for the iteration
        combined_residuals = np.sum(relevant_residuals, axis=0)

        # Reset zeros to NaN so they aren't included in RMS calculation
        combined_residuals[combined_residuals == 0.0] = np.nan
        
        # Compute RMS of combined residuals for the iteration
        rms_range_residual = np.sqrt(np.abs(np.nanmean((combined_residuals[0,:]*1E5) **2))) # Convert from km to cm for RMS calculation
        rms_range_rate_residual = np.sqrt(np.abs(np.nanmean((combined_residuals[1,:]*1E6) **2))) # Convert from km/s to mm/s for RMS calculation

        mean_range_residual = np.nanmean(combined_residuals[0,:]*1E5) # Convert from km to cm for mean calculation
        std_range_residual = np.nanstd(combined_residuals[0,:]*1E5) # Convert from km to cm for std calculation
        mean_range_rate_residual = np.nanmean(combined_residuals[1,:]*1E6) # Convert from km/s to mm/s for mean calculation
        std_range_rate_residual = np.nanstd(combined_residuals[1,:]*1E6) # Convert from km/s to mm/s for std calculation
        print(f"Post-Fit {filter_name} Iteration {iteration+1}:")
        print(f"Range Residuals: Mean = {mean_range_residual:.4f} cm, Std Dev = {std_range_residual:.4f} cm, RMS = {rms_range_residual:.4f} cm")
        print(f"Range Rate Residuals: Mean = {mean_range_rate_residual:.4f} mm/s, Std Dev = {std_range_rate_residual:.4f} mm/s, RMS = {rms_range_rate_residual:.4f} mm/s")
        print("--------------------------------------------------")

        # Reset zeros to NaN in individual station residuals as well for accurate RMS calculation
        for i in range(len(residuals_df['station'].unique())):
            # Set any NaN values to zero for RMS calculation
            relevant_residuals[i][relevant_residuals[i] == 0.0] = np.nan

        fig = make_subplots(
            rows=2, cols=2, 
            shared_xaxes=False,
            column_widths=[0.85, 0.15],
            horizontal_spacing=0.06,
            subplot_titles=(f'Range Residuals (Mean = {mean_range_residual:.4f} cm, Std Dev = {std_range_residual:.4f} cm, RMS = {rms_range_residual:.4f} cm)', 'Distribution',
                            f'Range Rate Residuals (Mean = {mean_range_rate_residual:.4f} mm/s, Std Dev = {std_range_rate_residual:.4f} mm/s, RMS = {rms_range_rate_residual:.4f} mm/s)', 'Distribution')
        )
        
        # Collect all residuals for histogram
        all_range_residuals = []
        all_range_rate_residuals = []
        
        for i, station_name in enumerate(residuals_df['station'].unique()):
            mask = (residuals_df['iteration'] == iteration) & (residuals_df['station'] == station_name)
            post_fit_residuals = np.vstack(residuals_df[mask]['post-fit'])
            
            # Add scatter plots (left column)
            fig.add_trace(go.Scatter(x=time_vector, y=post_fit_residuals[0,:]*1E5, 
                                    mode='markers', name=f'{station_name}', 
                                    marker=dict(color=colors_list[i]), legendgroup=f'group{i}'), 
                         row=1, col=1)
            fig.add_trace(go.Scatter(x=time_vector, y=post_fit_residuals[1,:]*1E6, 
                                    mode='markers', name=f'{station_name}', 
                                    marker=dict(color=colors_list[i]), 
                                    showlegend=False, legendgroup=f'group{i}'), 
                         row=2, col=1)
            
            # Collect valid (non-NaN) residuals for histograms
            valid_range = post_fit_residuals[0,:][~np.isnan(post_fit_residuals[0,:])] * 1E5
            valid_range_rate = post_fit_residuals[1,:][~np.isnan(post_fit_residuals[1,:])] * 1E6
            all_range_residuals.extend(valid_range)
            all_range_rate_residuals.extend(valid_range_rate)
        
        # Add histograms (right column) - rotated to be vertical
        fig.add_trace(go.Histogram(y=all_range_residuals, 
                                  marker=dict(color='lightblue'),
                                  showlegend=False,
                                  nbinsy=50), 
                     row=1, col=2)
        fig.add_trace(go.Histogram(y=all_range_rate_residuals, 
                                  marker=dict(color='lightcoral'),
                                  showlegend=False,
                                  nbinsy=50), 
                     row=2, col=2)
        
        fig.update_traces(marker=dict(size=4), selector=dict(mode='markers'))
        fig.update_xaxes(title_text="Time (s)", tickfont=dict(size=20), title_font=dict(size=22), row=1, col=1)
        fig.update_xaxes(title_text="Time (s)", tickfont=dict(size=20), title_font=dict(size=22), row=2, col=1)
        fig.update_xaxes(title_text="Count", tickfont=dict(size=20), title_font=dict(size=22), row=1, col=2)
        fig.update_xaxes(title_text="Count", tickfont=dict(size=20), title_font=dict(size=22), row=2, col=2)
        fig.update_yaxes(title_text="Range Residuals (cm)", tickfont=dict(size=20), title_font=dict(size=22), showexponent="all", exponentformat="e", row=1, col=1)
        fig.update_yaxes(title_text="Range Rate Residuals (mm/s)", tickfont=dict(size=20), title_font=dict(size=22), showexponent="all", exponentformat="e", row=2, col=1)
        fig.update_yaxes(showexponent="all", exponentformat="e", tickfont=dict(size=20), title_font=dict(size=22), row=1, col=2)
        fig.update_yaxes(showexponent="all", exponentformat="e", tickfont=dict(size=20), title_font=dict(size=22), row=2, col=2)
        fig.update_annotations(font=dict(size=24))
        fig.update_layout(title_text=f"{filter_name} Post-Fit Residuals at Iteration {iteration+1}",
                        title_font=dict(size=30),
                        width=1900,  # Increased width to accommodate histograms
                        height=800,
                        legend=dict(font=dict(size=22),
                                    orientation="h",
                                    yanchor="top",
                                    y=1.13,
                                    xanchor="left",
                                    x=0.7,
                                    itemsizing='constant'))
        fig.show()

# Plot trace of satellite state covariance using log scale for better visualization
lkf_covariance_list = [lkf_covariance_history, reasonable_lkf_covariance_history]
lkf_file_labels = ['lkf_covariance', 'reasonable_lkf_covariance']
for covariance_history, filter_name, file_label in zip(lkf_covariance_list, ['LKF', 'LKF with Reasonable'], lkf_file_labels):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=('Position Covariance Trace', 'Velocity Covariance Trace'))
    trace_pos = np.trace(covariance_history[:3,:3,:])
    trace_vel = np.trace(covariance_history[3:6,3:6,:])
    fig.add_trace(go.Scatter(x=time_vector, y=trace_pos, mode='lines', name='Satellite Position'), row=1, col=1)
    fig.add_trace(go.Scatter(x=time_vector, y=trace_vel, mode='lines', name='Satellite Velocity'), row=2, col=1)
    fig.update_yaxes(type="log", showexponent="all", exponentformat="e", title_text=f'Covariance Trace (km^2)', row=1, col=1)
    fig.update_yaxes(type="log", showexponent="all", exponentformat="e", title_text=f'Covariance Trace (km^2/s^2)', row=2, col=1)
        
    fig.update_layout(title=f"{filter_name} Covariance Time History",
                        xaxis_title='Time (s)',
                        title_font=dict(size=28),
                        width=1200,
                        height=800,
                        legend=dict(font=dict(size=18),
                                    yanchor="top",
                                    y=1.2,
                                    xanchor="left",
                                    x=0.87))
    fig.update_yaxes(showexponent="all", exponentformat="e")
    fig.show()

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=('Position Covariance Trace', 'Velocity Covariance Trace'))
overconfident_trace_pos = np.trace(overconfident_lkf_covariance_history[:3,:3,:])
overconfident_trace_vel = np.trace(overconfident_lkf_covariance_history[3:6,3:6,:])
underconfident_trace_pos = np.trace(underconfident_lkf_covariance_history[:3,:3,:])
underconfident_trace_vel = np.trace(underconfident_lkf_covariance_history[3:6,3:6,:])
fig.add_trace(go.Scatter(x=time_vector, y=overconfident_trace_pos, mode='lines', name='Overconfident Noise Data', marker =dict(color='blue')), row=1, col=1)
fig.add_trace(go.Scatter(x=time_vector, y=overconfident_trace_vel, mode='lines', name='Overconfident Noise Data', marker = dict(color='blue'), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=time_vector, y=underconfident_trace_pos, mode='lines', name='Underconfident Position', marker =dict(color='red')), row=1, col=1)
fig.add_trace(go.Scatter(x=time_vector, y=underconfident_trace_vel, mode='lines', name='Underconfident Velocity', marker = dict(color='red'), showlegend=False), row=2, col=1)
fig.update_yaxes(type="log", showexponent="all", exponentformat="e", title_text=f'Covariance Trace (km^2)', row=1, col=1)
fig.update_yaxes(type="log", showexponent="all", exponentformat="e", title_text=f'Covariance Trace (km^2/s^2)', row=2, col=1)
fig.update_layout(title=f"Covariance Time Histories for Varying Data Noise",
                    xaxis_title='Time (s)',
                    title_font=dict(size=28),
                    width=1200,
                    height=800,
                    legend=dict(font=dict(size=18),
                                yanchor="top",
                                y=1.2,
                                xanchor="left",
                                x=0.87))
fig.update_yaxes(showexponent="all", exponentformat="e")
fig.show()

# Plot covariance ellipse for satellite position at final time step
batch_center = batch_estimated_state_history[:6,-1]
lkf_center = lkf_state_history[:6,-1]
center_diff = batch_center - lkf_center
batch_pos_covariance_ellipse = covariance_ellipse(np.zeros(3), covariance_history[:3,:3,-1])
lkf_pos_covariance_ellipse = covariance_ellipse(center_diff[:3], lkf_covariance_history[:3,:3,-1])

# Plot 3D ellipses
fig = go.Figure()
fig.add_trace(go.Scatter3d(x=batch_pos_covariance_ellipse[:,0], y=batch_pos_covariance_ellipse[:,1], z=batch_pos_covariance_ellipse[:,2], mode='markers', name='Batch LLS Position Covariance Ellipse'))
fig.add_trace(go.Scatter3d(x=lkf_pos_covariance_ellipse[:,0], y=lkf_pos_covariance_ellipse[:,1], z=lkf_pos_covariance_ellipse[:,2], mode='markers', name='LKF Position Covariance Ellipse'))
fig.update_layout(title=f"Satellite Position Covariance Ellipses",
                    title_font=dict(size=28),
                    width=1000,
                    height=800,
                    legend=dict(font=dict(size=18),
                                yanchor="top",
                                y=1.2,
                                xanchor="left",
                                x=0.6),
                    scene=dict(xaxis_title='X Position (km)',
                               yaxis_title='Y Position (km)',
                               zaxis_title='Z Position (km)',
                               xaxis=dict(showexponent="all", exponentformat="e"),
                               yaxis=dict(showexponent="all", exponentformat="e"),
                               zaxis=dict(showexponent="all", exponentformat="e")))
fig.show()

# Plot covariance ellipse for satellite velocity at final time step
batch_vel_covariance_ellipse = covariance_ellipse(np.zeros(3), covariance_history[3:6,3:6,-1])
lkf_vel_covariance_ellipse = covariance_ellipse(center_diff[3:6], lkf_covariance_history[3:6,3:6,-1])

# Plot 3D ellipses
fig = go.Figure()
fig.add_trace(go.Scatter3d(x=batch_vel_covariance_ellipse[:,0], y=batch_vel_covariance_ellipse[:,1], z=batch_vel_covariance_ellipse[:,2], mode='markers', name='Batch LLS Velocity Covariance Ellipse'))
fig.add_trace(go.Scatter3d(x=lkf_vel_covariance_ellipse[:,0], y=lkf_vel_covariance_ellipse[:,1], z=lkf_vel_covariance_ellipse[:,2], mode='markers', name='LKF Velocity Covariance Ellipse'))
fig.update_layout(title=f"Satellite Velocity Covariance Ellipses",
                    title_font=dict(size=28),
                    width=1000,
                    height=800,
                    legend=dict(font=dict(size=18),
                                yanchor="top",
                                y=1.2,
                                xanchor="left",
                                x=0.6),
                    scene=dict(xaxis_title='X Velocity (km/s)',
                               yaxis_title='Y Velocity (km/s)',
                               zaxis_title='Z Velocity (km/s)',
                               xaxis=dict(showexponent="all", exponentformat="e"),
                               yaxis=dict(showexponent="all", exponentformat="e"),
                               zaxis=dict(showexponent="all", exponentformat="e")))
fig.show()

# Plot covariance ellipse to show difference between analyzing range and range rate
range_center = range_lkf_state_history[:6,-1]
range_rate_center = range_rate_lkf_state_history[:6,-1]
center_diff = range_center - range_rate_center
range_pos_covariance_ellipse = covariance_ellipse(np.zeros(3), range_covariance_history[:3,:3,-1])
range_rate_pos_covariance_ellipse = covariance_ellipse(center_diff[:3], range_rate_covariance_history[:3,:3,-1])

fig = go.Figure()
fig.add_trace(go.Scatter3d(x=range_pos_covariance_ellipse[:,0], y=range_pos_covariance_ellipse[:,1], z=range_pos_covariance_ellipse[:,2], mode='markers', name='Range Only'))
fig.add_trace(go.Scatter3d(x=range_rate_pos_covariance_ellipse[:,0], y=range_rate_pos_covariance_ellipse[:,1], z=range_rate_pos_covariance_ellipse[:,2], mode='markers', name='Range Rate Only'))
fig.update_layout(title=f"Position Covariance Ellipses from Analyzing Only Range or Range Rate",
                    title_font=dict(size=28),
                    width=1200,
                    height=800,
                    legend=dict(font=dict(size=18)),
                    scene=dict(xaxis_title='X Position (km)',
                               yaxis_title='Y Position (km)',
                               zaxis_title='Z Position (km)',
                               xaxis=dict(showexponent="all", exponentformat="e"),
                               yaxis=dict(showexponent="all", exponentformat="e"),
                               zaxis=dict(showexponent="all", exponentformat="e")))
fig.show()

range_vel_covariance_ellipse = covariance_ellipse(np.zeros(3), range_covariance_history[3:6,3:6,-1])
range_rate_vel_covariance_ellipse = covariance_ellipse(center_diff[3:6], range_rate_covariance_history[3:6,3:6,-1])

fig = go.Figure()
fig.add_trace(go.Scatter3d(x=range_vel_covariance_ellipse[:,0], y=range_vel_covariance_ellipse[:,1], z=range_vel_covariance_ellipse[:,2], mode='markers', name='Range Only'))
fig.add_trace(go.Scatter3d(x=range_rate_vel_covariance_ellipse[:,0], y=range_rate_vel_covariance_ellipse[:,1], z=range_rate_vel_covariance_ellipse[:,2], mode='markers', name='Range Rate Only'))
fig.update_layout(title=f"Velocity Covariance Ellipses from Analyzing Only Range or Range Rate",
                    title_font=dict(size=28),
                    width=1200,
                    height=800,
                    legend=dict(font=dict(size=18)),
                    scene=dict(xaxis_title='X Velocity (km/s)',
                               yaxis_title='Y Velocity (km/s)',
                               zaxis_title='Z Velocity (km/s)',
                               xaxis=dict(showexponent="all", exponentformat="e"),
                               yaxis=dict(showexponent="all", exponentformat="e"),
                               zaxis=dict(showexponent="all", exponentformat="e")))
fig.show()

# Plot covariance ellipse to show difference between analyzing range and range rate
underconfident_center = underconfident_batch_estimated_state_history[:6,-1]
overconfident_center = overconfident_batch_estimated_state_history[:6,-1]
center_diff = underconfident_center - overconfident_center
underconfident_pos_covariance_ellipse = covariance_ellipse(np.zeros(3), underconfident_covariance_history[:3,:3,-1])
overconfident_pos_covariance_ellipse = covariance_ellipse(center_diff[:3], overconfident_covariance_history[:3,:3,-1])

fig = go.Figure()
fig.add_trace(go.Scatter3d(x=underconfident_pos_covariance_ellipse[:,0], y=underconfident_pos_covariance_ellipse[:,1], z=underconfident_pos_covariance_ellipse[:,2], mode='markers', name='Underconfident Data Noise', marker=dict(opacity=0.05)))
fig.add_trace(go.Scatter3d(x=overconfident_pos_covariance_ellipse[:,0], y=overconfident_pos_covariance_ellipse[:,1], z=overconfident_pos_covariance_ellipse[:,2], mode='markers', name='Overconfident Data Noise'))
fig.update_layout(title=f"Position Covariance Ellipses from Underconfident vs Overconfident Data Noise",
                  title_font=dict(size=28),
                  width=1200,
                  height=800,
                  legend=dict(font=dict(size=18)),
                  scene=dict(xaxis_title='X Position (km)',
                             yaxis_title='Y Position (km)',
                             zaxis_title='Z Position (km)',
                             xaxis=dict(showexponent="all", exponentformat="e"),
                             yaxis=dict(showexponent="all", exponentformat="e"),
                             zaxis=dict(showexponent="all", exponentformat="e")))
fig.show()

underconfident_vel_covariance_ellipse = covariance_ellipse(np.zeros(3), underconfident_covariance_history[3:6,3:6,-1])
overconfident_vel_covariance_ellipse = covariance_ellipse(center_diff[3:6], overconfident_covariance_history[3:6,3:6,-1])
fig = go.Figure()
fig.add_trace(go.Scatter3d(x=underconfident_vel_covariance_ellipse[:,0], y=underconfident_vel_covariance_ellipse[:,1], z=underconfident_vel_covariance_ellipse[:,2], mode='markers', name='Underconfident Data Noise', marker=dict(opacity=0.05)))
fig.add_trace(go.Scatter3d(x=overconfident_vel_covariance_ellipse[:,0], y=overconfident_vel_covariance_ellipse[:,1], z=overconfident_vel_covariance_ellipse[:,2], mode='markers', name='Overconfident Data Noise'))
fig.update_layout(title=f"Velocity Covariance Ellipses from Underconfident vs Overconfident Data Noise",
                  title_font=dict(size=28),
                  width=1200,
                  height=800,
                  legend=dict(font=dict(size=18)),
                  scene=dict(xaxis_title='X Velocity (km/s)',
                             yaxis_title='Y Velocity (km/s)',
                             zaxis_title='Z Velocity (km/s)',
                             xaxis=dict(showexponent="all", exponentformat="e"),
                             yaxis=dict(showexponent="all", exponentformat="e"),
                             zaxis=dict(showexponent="all", exponentformat="e")))
fig.show()