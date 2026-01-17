import numpy as np
from coordinate_manager import CoordinateMgr

class MeasurementMgr:
    def __init__(self, station_lat : float, station_lon : float, initial_earth_spin_angle : float = 0.0):
        """This class manages measurement simulations for a station at the inputted GCS coordinates.
        Parameters:
        station_lat : float
            Latitude of the ground station in degrees.
        station_lon : float
            Longitude of the ground station in degrees.
        initial_earth_spin_angle : float, optional
            Initial Earth spin angle in radians. Default is 0.0.
        station_alt : float, optional
            Altitude of the ground station above Earth's surface in kilometers. Default is 0.0 km.
        """

        self.coordinate_mgr = CoordinateMgr(initial_earth_spin_angle)
        self.station_state_ecef = self.coordinate_mgr.GCS_to_ECEF(station_lat, station_lon)

    def get_elevation_angle(self, sc_position_ecef : np.array):
        """Compute the elevation angle of the spacecraft from the ground station.
        Parameters:
        sc_position_ecef : np.array
            3x1 array of spacecraft position in ECEF coordinates.
        Returns:
        elevation_angle : float
            Elevation angle in degrees.
        """
        station_pos = self.station_state_ecef[0:3]
        range_vec = sc_position_ecef - station_pos
        range_mag = np.linalg.norm(range_vec)
        station_mag = np.linalg.norm(station_pos)

        elevation_angle = np.arcsin(np.dot(range_vec, station_pos) / (range_mag * station_mag))

        return np.rad2deg(elevation_angle)

    def is_visible(self, sc_position_ecef : np.array, visibility_elevation_angle : float = 10.0):
        """Determine if the spacecraft is visible from the ground station.
        Parameters:
        sc_position_ecef : np.array
            3x1 array of spacecraft position in ECEF coordinates.
        visibility_elevation_angle : float, optional
            Minimum elevation angle (in degrees) for visibility. Default is 10 degrees.
        Returns:
        visible : bool
            True if the spacecraft is visible from the ground station, False otherwise.
        """
        elevation_angle = self.get_elevation_angle(sc_position_ecef)

        return elevation_angle > visibility_elevation_angle

    def simulate_measurements(self, inputted_state_history : np.array, time_vector : np.array, coordinate_frame : str):
        """Simulate range measurements from the ground station to the spacecraft over time.
        Parameters:
        sc_state_history : np.array
            nxN array of spacecraft states in over time.
        time_vector : np.array
            1xN array of time points corresponding to the spacecraft states.
        coordinate_frame : str
            Coordinate frame of the inputted spacecraft states ('ECI' or 'ECEF').
        Returns:
        measurement_history : np.array
            2xN array of range and range rate measurements in kilometers.
        """
        # Check of coordinate conversion is needed
        if coordinate_frame == 'ECI':
            # Convert spacecraft states to ECEF
            sc_state_history = np.zeros(inputted_state_history.shape)
            for i, time in enumerate(time_vector):
                eci_to_ecef = self.coordinate_mgr.compute_DCM('ECI', 'ECEF', time=time)
                ecef_pos = eci_to_ecef @ inputted_state_history[0:3,i]
                ecef_vel = eci_to_ecef @ inputted_state_history[3:6,i] - np.cross(np.array([0, 0, self.coordinate_mgr.earth_rotation_rate]), ecef_pos)
                full_ecef_state = np.hstack((ecef_pos, ecef_vel)).T
                sc_state_history[0:6, i] = full_ecef_state
        else:
            sc_state_history = inputted_state_history
        
        # Simulate measurements
        measurement_history = np.zeros((2, sc_state_history.shape[1]))
        for i in range(sc_state_history.shape[1]):
            sc_pos = sc_state_history[0:3, i]
            # Check visibility
            if self.is_visible(sc_pos) == True:
                sc_vel = sc_state_history[3:6, i]
                
                range_vec = sc_pos - self.station_state_ecef[0:3]
                range_mag = np.linalg.norm(range_vec)
                range_rate = np.dot(range_vec, (sc_vel - self.station_state_ecef[3:6])) / range_mag

                measurement_history[0, i] = range_mag
                measurement_history[1, i] = range_rate
            else:
                measurement_history[0, i] = np.nan
                measurement_history[1, i] = np.nan

        return measurement_history
