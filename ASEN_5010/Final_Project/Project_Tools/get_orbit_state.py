import numpy as np

def get_orbit_state(orbit_radius : float, raan : float, inc : float, ta : float, mu : float = 42828.3):
    """
    This function takes in the orbit radius and 3-1-3 Euler set describing a circular orbit and returns the spacecraft's state (pos and vel).

    Parameters
    ----------
    orbit_radius : float
        The radius of the circular orbit in kilometers.
    raan : float
        The right ascension of the ascending node in degrees.
    inc : float
        The inclination of the orbit in degrees.
    ta : float
        The true anomaly in degrees.
    mu : float, optional
        The standard gravitational parameter of the central body (default is 42828.3 km^3/s^2 for Mars).
    Returns
    -------
    pos : numpy array
        The position vector of the spacecraft in kilometers.
    vel : numpy array
        The velocity vector of the spacecraft in kilometers per second.
    """
    # Define radius vector in RTN frame (circular orbit, so radius is constant)
    r_rtn = np.array([orbit_radius, 0, 0])

    # Calculate velocity for circular orbit in RTN frame
    v_mag = np.sqrt(mu / orbit_radius)
    v_rtn = np.array([0, v_mag, 0])

    # Convert angles from degrees to radians
    theta_1 = np.radians(raan)
    theta_2 = np.radians(inc)
    theta_3 = np.radians(ta)

    # Build 3-1-3 rotation matrix using individual rotation matrices
    R_1 = np.array([[np.cos(theta_1), -np.sin(theta_1), 0],
                    [np.sin(theta_1), np.cos(theta_1), 0],
                    [0, 0, 1]])
    
    R_2 = np.array([[1, 0, 0],
                    [0, np.cos(theta_2), -np.sin(theta_2)],
                    [0, np.sin(theta_2), np.cos(theta_2)]])
    
    R_3 = np.array([[np.cos(theta_3), -np.sin(theta_3), 0],
                    [np.sin(theta_3), np.cos(theta_3), 0],
                    [0, 0, 1]])

    DCM = R_3 @ R_2 @ R_1

    # Transform position and velocity from RTN to inertial frame
    pos = DCM @ r_rtn
    vel = DCM @ v_rtn

    return pos, vel

