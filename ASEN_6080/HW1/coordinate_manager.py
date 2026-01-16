import numpy as np
from integrator import Integrator

class CoordinateMgr:
    def __init__(self, initial_earth_spin_angle : float = 0.0):
        """This class manages coordinate frame transformations."""
        self.earth_rotation_rate = 2*np.pi / (24 * 3600)  # rad/s
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
                    [np.sin(AoP) * np.sin(i), np.cos(AoP) * np.sin(i), np.cos(i)]])
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
                    [np.sin(AoP) * np.sin(i), np.cos(AoP) * np.sin(i), np.cos(i)]]).T
            return DCM
        
        # Compute ECEF to Perifocal DCM
        elif coordinate_frame_1 == 'ECEF' and coordinate_frame_2 == 'Perifocal':
            DCM_ECI_ECEF = self.compute_DCM('ECEF', 'ECI', time=time)
            DCM_ECI_Perifocal = self.compute_DCM('ECI', 'Perifocal', orbit_state=orbit_state)
            DCM = DCM_ECI_Perifocal @ DCM_ECI_ECEF
            return DCM
        
        # Compute Perifocal to ECEF DCM
        elif coordinate_frame_1 == 'Perifocal' and coordinate_frame_2 == 'ECEF':
            DCM_ECI_Perifocal = self.compute_DCM('Perifocal', 'ECI', orbit_state=orbit_state)
            DCM_ECI_ECEF = self.compute_DCM('ECI', 'ECEF', time=time)
            DCM = DCM_ECI_ECEF @ DCM_ECI_Perifocal
            return DCM