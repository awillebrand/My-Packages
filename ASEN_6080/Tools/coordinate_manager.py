import numpy as np
from .integrator import Integrator

class CoordinateMgr:
    def __init__(self, initial_earth_spin_angle : float = 0.0):
        """This class manages coordinate frame transformations."""
        self.earth_rotation_rate = 2*np.pi/86164.0905  # rad/s
        self.initial_earth_spin_angle = initial_earth_spin_angle

    def compute_DCM(self, coordinate_frame_1 : str, coordinate_frame_2 : str, time : float = None, orbit_state : np.array = None):
        """Compute direction cosine matrix between two coordinate frames. Outputted DCM converts from frame 1 to frame 2.
        Parameters:
        coordinate_frame_1 : str
            Name of the first coordinate frame (e.g., 'ECI', 'ECEF', 'Perifocal').
        coordinate_frame_2 : str
            Name of the second coordinate frame (e.g., 'ECI', 'ECEF', 'Perifocal).
        time : float, optional
            Time in seconds since epoch (required for ECI/ECEF transformations).
        orbit_state : np.array, optional
            Orbital state vector in Cartesian coordinates or orbital orientation [i, LoN, AoP] (required for Perifocal transformations)."""
        
        # Raise errors if necessary inputs are not provided
        if (coordinate_frame_1 == 'ECI' or coordinate_frame_2 == 'ECI') and time == None:
            raise ValueError("Time must be provided when converting to/from ECI frame.")
        if (coordinate_frame_1 == 'Perifocal' or coordinate_frame_2 == 'Perifocal') and orbit_state is None:
            raise ValueError("Orbit state must be provided when converting to/from Perifocal frame.")
        
        # Compute ECI to ECEF DCM
        if coordinate_frame_1 == 'ECI' and coordinate_frame_2 == 'ECEF':
            theta = self.initial_earth_spin_angle + self.earth_rotation_rate * time
            DCM = np.array([[ np.cos(theta),  np.sin(theta), 0],
                            [-np.sin(theta),  np.cos(theta), 0],
                            [           0,             0, 1]])
            return DCM
        
        # Compute ECEF to ECI DCM
        elif coordinate_frame_1 == 'ECEF' and coordinate_frame_2 == 'ECI':
            theta = self.initial_earth_spin_angle + self.earth_rotation_rate * time
            DCM = np.array([[ np.cos(theta), -np.sin(theta), 0],
                            [ np.sin(theta),  np.cos(theta), 0],
                            [           0,             0, 1]])
            return DCM
        
        # Compute ECI to Perifocal DCM
        elif coordinate_frame_1 == 'ECI' and coordinate_frame_2 == 'Perifocal':
            if len(orbit_state) == 6:
                r_vec = orbit_state[0:3]
                v_vec = orbit_state[3:6]
                integrator = Integrator(mu=398600.4418, R_e=6378)
                i, LoN, AoP, _, _, _ = integrator.cartesian_to_keplerian(r_vec, v_vec)
            else:
                i, LoN, AoP = orbit_state
            
            DCM = np.array([[np.cos(LoN) * np.cos(AoP) - np.sin(LoN) * np.sin(AoP) * np.cos(i), -np.cos(LoN) * np.sin(AoP) - np.sin(LoN) * np.cos(AoP) * np.cos(i),  np.sin(LoN) * np.sin(i)],
                            [np.sin(LoN) * np.cos(AoP) + np.cos(LoN) * np.sin(AoP) * np.cos(i), -np.sin(LoN) * np.sin(AoP) + np.cos(LoN) * np.cos(AoP) * np.cos(i), -np.cos(LoN) * np.sin(i)],
                            [np.sin(AoP) * np.sin(i), np.cos(AoP) * np.sin(i), np.cos(i)]]).T
            return DCM
    
        # Compute Perifocal to ECI DCM
        elif coordinate_frame_1 == 'Perifocal' and coordinate_frame_2 == 'ECI':
            if len(orbit_state) == 6:
                r_vec = orbit_state[0:3]
                v_vec = orbit_state[3:6]
                integrator = Integrator(mu=398600.4418, R_e=6378)
                i, LoN, AoP, _, _, _ = integrator.cartesian_to_keplerian(r_vec, v_vec)
            else:
                i, LoN, AoP = orbit_state
            
            DCM = np.array([[np.cos(LoN) * np.cos(AoP) - np.sin(LoN) * np.sin(AoP) * np.cos(i), -np.cos(LoN) * np.sin(AoP) - np.sin(LoN) * np.cos(AoP) * np.cos(i),  np.sin(LoN) * np.sin(i)],
                            [np.sin(LoN) * np.cos(AoP) + np.cos(LoN) * np.sin(AoP) * np.cos(i), -np.sin(LoN) * np.sin(AoP) + np.cos(LoN) * np.cos(AoP) * np.cos(i), -np.cos(LoN) * np.sin(i)],
                            [np.sin(AoP) * np.sin(i), np.cos(AoP) * np.sin(i), np.cos(i)]])
            return DCM
        
        # Compute ECEF to Perifocal DCM
        elif coordinate_frame_1 == 'ECEF' and coordinate_frame_2 == 'Perifocal':
            ECEF_to_ECI = self.compute_DCM('ECEF', 'ECI', time=time)
            ECI_to_Perifocal = self.compute_DCM('ECI', 'Perifocal', orbit_state=orbit_state)
            DCM = ECI_to_Perifocal @ ECEF_to_ECI
            return DCM
        
        # Compute Perifocal to ECEF DCM
        elif coordinate_frame_1 == 'Perifocal' and coordinate_frame_2 == 'ECEF':
            Perifocal_to_ECI = self.compute_DCM('Perifocal', 'ECI', orbit_state=orbit_state)
            ECI_to_ECEF = self.compute_DCM('ECI', 'ECEF', time=time)
            DCM = ECI_to_ECEF @ Perifocal_to_ECI
            return DCM
        
        elif coordinate_frame_1 == coordinate_frame_2:
            return np.eye(3)
        else:
            raise ValueError("Invalid coordinate frame transformation requested.")
        
    def GCS_to_ECI(self, lat : float, lon : float, time : float, R_e : float = 6378):
        """"""
        """
        Convert Geocentric Spherical Coordinates (latitude, longitude, altitude) to ECI state.
        Parameters:
        lat : float
            Latitude in degrees.
        lon : float
            Longitude in degrees.
        time : float
            Time in seconds since initial epoch.
        R_e : float, optional
            Earth's radius in kilometers. Default is 6378 km.
        Returns:
        state : np.array
            ECI state vector [x, y, z, u, v, w] in kilometers.
        """
        lat_rad = np.deg2rad(lat)
        lon_rad = np.deg2rad(lon)

        r_mag = R_e
        x = r_mag * np.cos(lat_rad) * np.cos(lon_rad)
        y = r_mag * np.cos(lat_rad) * np.sin(lon_rad)
        z = r_mag * np.sin(lat_rad)
        r_vec = np.array([x, y, z])

        # Convert to ECI frame
        ecef_to_eci_dcm = self.compute_DCM('ECEF', 'ECI', time=time)
        r_vec = ecef_to_eci_dcm @ r_vec

        # Velocity components (assuming stationary on Earth's surface)
        v_vec = np.cross(np.array([0, 0, self.earth_rotation_rate]), r_vec)

        return np.hstack((r_vec, v_vec))
    
    def GCS_to_ECEF(self, lat : float, lon : float, R_e : float = 6378):
        """"""
        """
        Convert Geocentric Spherical Coordinates (latitude, longitude, altitude) to ECEF state.
        Parameters:
        lat : float
            Latitude in degrees.
        lon : float
            Longitude in degrees.
        R_e : float, optional
            Earth's radius in kilometers. Default is 6378 km.
        Returns:
        state : np.array
            ECEF state vector [x, y, z, u, v, w] in kilometers.
        """
        lat_rad = np.deg2rad(lat)
        lon_rad = np.deg2rad(lon)

        r_mag = R_e
        x = r_mag * np.cos(lat_rad) * np.cos(lon_rad)
        y = r_mag * np.cos(lat_rad) * np.sin(lon_rad)
        z = r_mag * np.sin(lat_rad)
        r_vec = np.array([x, y, z])

        # Velocity components (zero in ECEF frame)
        v_vec = np.zeros(3)

        return np.hstack((r_vec, v_vec))
    
    def ECEF_to_GCS(self, state_ecef : np.array):
        """
        Convert ECEF state to Geocentric Spherical Coordinates (latitude, longitude, altitude).
        Parameters:
        state_ecef : np.array
            ECEF state vector [x, y, z, u, v, w] in kilometers.
        Returns:
        lat : float
            Latitude in degrees.
        lon : float
            Longitude in degrees.
        """
        x = state_ecef[0]
        y = state_ecef[1]
        z = state_ecef[2]

        r_mag = np.sqrt(x**2 + y**2 + z**2)
        lat_rad = np.arcsin(z / r_mag)
        lon_rad = np.arctan2(y, x)

        lat = np.rad2deg(lat_rad)
        lon = np.rad2deg(lon_rad)

        return lat, lon