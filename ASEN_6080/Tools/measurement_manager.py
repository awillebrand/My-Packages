import numpy as np
from .coordinate_manager import CoordinateMgr

class MeasurementMgr:
    def __init__(self, station_name : str, station_lat : float = None, station_lon : float = None, station_state_ecef : np.ndarray = None, initial_earth_spin_angle : float = 0.0, earth_spin_rate : float = 2*np.pi/86164.0905, R_e : float = 6378):
        """This class manages measurement simulations for a station at the inputted GCS coordinates.
        Parameters:
        station_name : str
            Name of the ground station.
        station_lat : float, optional
            Latitude of the ground station in degrees. If None, it will be computed from station_state_ecef.
        station_lon : float, optional
            Longitude of the ground station in degrees. If None, it will be computed from station_state_ecef.
        station_state_ecef : np.ndarray, optional
            6x1 array of ground station state in ECEF coordinates. If None, it will be computed from lat/lon.
        initial_earth_spin_angle : float, optional
            Initial Earth spin angle in radians. Default is 0.0.
        earth_spin_rate : float, optional
            Earth's rotation rate in radians per second. Default is 2*pi/86164.0905 rad/s.
        R_e : float, optional
            Earth's radius in kilometers. Default is 6378 km.
        """
        self.station_name = station_name
        self.coordinate_mgr = CoordinateMgr(initial_earth_spin_angle, earth_rotation_rate=earth_spin_rate, R_e=R_e)
        if station_lat != None and station_lon != None:
            self.lat = station_lat
            self.lon = station_lon
            self.station_state_ecef = self.coordinate_mgr.GCS_to_ECEF(station_lat, station_lon)
        elif station_state_ecef is not None:
            self.station_state_ecef = station_state_ecef
            self.lat, self.lon = self.coordinate_mgr.ECEF_to_GCS(station_state_ecef)
        else:
            raise ValueError("Either station_lat and station_lon or station_state_ecef must be provided.")
        
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

    def is_visible(self, sc_position_ecef : np.array, visibility_elevation_angle : float = 0.0):
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

    def simulate_measurements(self, inputted_state_history : np.array, time_vector : np.array, coordinate_frame : str, noise : bool = False, noise_sigma : np.array = np.array([0.0, 0.0])):
        """Simulate range measurements from the ground station to the spacecraft over time.
        Parameters:
        sc_state_history : np.array
            nxN array of spacecraft states in over time.
        time_vector : np.array
            1xN array of time points corresponding to the spacecraft states.
        coordinate_frame : str
            Coordinate frame of the inputted spacecraft states ('ECI' or 'ECEF').
        noise : bool, optional
            Whether to add noise to the measurements. Default is False.
        noise_sigma : np.array, optional
            2x1 array of standard deviations for range and range rate noise. Default is [0.0, 0.0].
        Returns:
        measurement_history : np.array
            2xN array of range and range rate measurements in kilometers.
        """
        # Check of coordinate conversion is needed
        if coordinate_frame == 'ECI':
            eci_sc_state_history = inputted_state_history
            # Convert spacecraft states to ECEF
            ecef_sc_state_history = np.zeros(inputted_state_history.shape)
            for i, time in enumerate(time_vector):
                eci_to_ecef = self.coordinate_mgr.compute_DCM('ECI', 'ECEF', time=time)
                ecef_pos = eci_to_ecef @ inputted_state_history[0:3,i]
                ecef_vel = eci_to_ecef @ inputted_state_history[3:6,i] - np.cross(np.array([0, 0, self.coordinate_mgr.earth_rotation_rate]), ecef_pos)
                full_ecef_state = np.hstack((ecef_pos, ecef_vel)).T
                ecef_sc_state_history[0:6, i] = full_ecef_state
        elif coordinate_frame == 'ECEF':
            ecef_sc_state_history = inputted_state_history
            # Convert spacecraft states to ECI
            eci_sc_state_history = np.zeros(inputted_state_history.shape)
            for i, time in enumerate(time_vector):
                ecef_to_eci = self.coordinate_mgr.compute_DCM('ECEF', 'ECI', time=time)
                eci_pos = ecef_to_eci @ inputted_state_history[0:3,i]
                eci_vel = ecef_to_eci @ inputted_state_history[3:6,i] + np.cross(np.array([0, 0, self.coordinate_mgr.earth_rotation_rate]), eci_pos)
                full_eci_state = np.hstack((eci_pos, eci_vel)).T
                eci_sc_state_history[0:6, i] = full_eci_state
        else:
            raise ValueError("Invalid coordinate frame. Must be 'ECI' or 'ECEF'.")
        
        # Simulate measurements
        measurement_history = np.zeros((2, eci_sc_state_history.shape[1]))
        for i in range(eci_sc_state_history.shape[1]):
            eci_sc_pos = eci_sc_state_history[0:3, i]
            ecef_sc_pos = ecef_sc_state_history[0:3, i]
            # Check visibility
            if self.is_visible(ecef_sc_pos) == True:
                # Convert station to ECI at the current time
                time = time_vector[i]
                station_state_eci = self.coordinate_mgr.ECEF_to_ECI(self.station_state_ecef, time)

                eci_sc_vel = eci_sc_state_history[3:6, i]
                
                range_vec = eci_sc_pos - station_state_eci[0:3]
                range_mag = np.linalg.norm(range_vec)
                range_rate = np.dot(range_vec, (eci_sc_vel - station_state_eci[3:6])) / range_mag

                # Add noise if specified
                if noise:
                    range_mag += np.random.normal(0.0, noise_sigma[0])
                    range_rate += np.random.normal(0.0, noise_sigma[1])

                measurement_history[0, i] = range_mag
                measurement_history[1, i] = range_rate
            else:
                measurement_history[0, i] = np.nan
                measurement_history[1, i] = np.nan

        return measurement_history

    def convert_to_DSN_units(self, measurement_history : np.array, reference_frequency : float = 8.44e9):
        """Convert measurements to DSN units.
        Parameters:
        measurement_history : np.array
            2xN array of range (km) and range rate (km/s) measurements.
        reference_frequency : float, optional
            Reference frequency for Doppler shift calculation in Hz. Default is 8.44 GHz.
        Returns:
        dsn_measurement_history : np.array
            2xN array of range (Range Units) and Doppler shift (Hz) measurements.
        """
        # Speed of light in km/s
        c = 299792.458

        dsn_measurement_history = np.zeros(measurement_history.shape)

        # Convert range to Range Units
        dsn_measurement_history[0, :] = (221 / 749) * measurement_history[0, :] * reference_frequency / c
        # Convert range rate to Doppler shift
        dsn_measurement_history[1, :] = - 2 * measurement_history[1, :] * reference_frequency / c

        return dsn_measurement_history