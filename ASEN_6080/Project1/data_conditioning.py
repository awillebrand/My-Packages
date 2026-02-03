import pandas as pd
import numpy as np

with open('ASEN_6080/Project1/data/project.txt', 'r') as file:
    measurement_data = pd.read_csv(file)

measurement_data.columns = ["data"] 
# Split single column up into respective columns

measurement_data[['time', 'station_id', 'range', 'range_rate']] = measurement_data["data"].str.split(expand=True).astype(float)
measurement_data = measurement_data.drop(columns=['data'])

final_time = measurement_data['time'].values[-1]
delta_t = measurement_data['time'].values[1] - measurement_data['time'].values[0]

time_vector = np.linspace(0, final_time, int(final_time/delta_t) + 1)

station_1_measurements = measurement_data[measurement_data['station_id'] == 101][['time', 'range', 'range_rate']].reset_index(drop=True)
station_2_measurements = measurement_data[measurement_data['station_id'] == 337][['time', 'range', 'range_rate']].reset_index(drop=True)
station_3_measurements = measurement_data[measurement_data['station_id'] == 394][['time', 'range', 'range_rate']].reset_index(drop=True)

station_list = [station_1_measurements, station_2_measurements, station_3_measurements]

nan_mat = np.full((len(time_vector), 2), np.nan)

for j, station_data in enumerate(station_list):
    full_station_measurements = nan_mat.copy()
    for i, t in enumerate(time_vector):
        if t in station_data['time'].values:
            idx = station_data[station_data['time'] == t].index[0]
            full_station_measurements[i, 0] = station_data.loc[idx, 'range'] / 1000  # Convert to km
            full_station_measurements[i, 1] = station_data.loc[idx, 'range_rate'] / 1000  # Convert to km/s
    # Check if time t exists in station 1 measurements
    station_list[j] = full_station_measurements
breakpoint()
measurement_data_frame = pd.DataFrame({
    'time': time_vector,
    'station_101_measurements': list(station_list[0]),
    'station_337_measurements': list(station_list[1]),
    'station_394_measurements': list(station_list[2])
})
breakpoint()
measurement_data_frame.to_pickle("ASEN_6080/Project1/data/conditioned_measurements.pkl")