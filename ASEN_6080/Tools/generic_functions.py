import numpy as np

def compute_density(r_norm : float, rho_0 : float = 3.614e-13, r_0 : float = 700000.0 + 6678.0, H : float = 88667.0):
    """Compute atmospheric density at the satellite's position using an exponential model.
    Inputs:
    r_norm : float
        Magnitude of the satellite's position vector in km.
    rho_0 : float, optional
        Reference atmospheric density at reference altitude in kg/m^3. Default is 3.614e-13 kg/m^3 (approx. 700 km).
    r_0 : float, optional
        Reference radius from Earth's center in km. Default is 700 km altitude + Earth's radius (6678 km).
    H : float, optional
        Scale height in km. Default is 88667 m (88.667 km).
    Returns:
    density : float
        Atmospheric density at the satellite's position in kg/m^3.
    """
    r = r_norm*1000  # Convert km to m

    rho = rho_0 * np.exp(-(r-r_0)/ H)

    return rho

def state_jacobian(r : np.array, V : np.array, mu : float, J2 : float, J3 : float, C_d : float, station_positions_ecef : np.array, R_e, mode : list = ['BaseMat'], area : float = 3.0, spacecraft_mass : float = 970.0):
    """
    This function computes the partial derivatives of the acceleration associated with the J2 and J3 perturbations in a gravitational field and outputs the associated Jacobian.

    Parameters:
    r : np.Array
        Position vector in Cartesian coordinates (x, y, z).
    v : np.Array
        Velocity vector in Cartesian coordinates (vx, vy, vz).
    mu : float
        Gravitational parameter.
    J2 : float
        J2 coefficient.
    J3 : float
        J3 coefficient.
    C_d : float
        Drag coefficient.
    station_positions_ecef : np.array
        Nx3 array of ground station positions in ECEF coordinates, where N is the number of stations.
    R_e : float
        Earth's radius.
    mode : list
        List of strings specifying which partials to include in the Jacobian. Options are 'BaseMat', 'J2', 'J3', and/or 'Drag'.
    area : float
        Cross-sectional area of the spacecraft in m^2. Default is 3.0 m^2.
    spacecraft_mass : float
        Mass of the spacecraft in kg. Default is 970.0 kg.
    Returns:
    A : np.Array
        State Jacobian matrix.
    """

    if set(mode).isdisjoint({'BaseMat', 'J2', 'J3', 'Drag', 'Stations'}):
        raise ValueError("Invalid mode specified. Choose from 'Basemat', 'mu', 'J2', 'J3', 'Drag', and/or 'Stations'.")
    if mu == 0:
        raise ValueError("mu must be non-zero!")
    if 'J2' in mode and J2 == 0:
        raise ValueError("J2 must be non-zero to include J2 partials.")
    if 'J3' in mode and J3 == 0:
        raise ValueError("J3 must be non-zero to include J3 partials.")
    if 'Drag' in mode and C_d == 0:
        raise ValueError("C_d must be non-zero to include drag partials.")
    if 'Stations' in mode and len(station_positions_ecef) == 0:
        raise ValueError("Station positions must be provided to include station partials.")

    x, y, z = r
    r_norm = np.linalg.norm(r)
    u, v, w = V
    V_norm = np.linalg.norm(V)

    # Compute position partials
    # Point mass partials

    a_xx_pm = mu / r_norm**5 * (3 * x**2 - r_norm**2)
    a_yy_pm = mu / r_norm**5 * (3 * y**2 - r_norm**2)
    a_zz_pm = mu / r_norm**5 * (3 * z**2 - r_norm**2)
    a_xy_pm = 3 * mu * x * y / r_norm**5
    a_xz_pm = 3 * mu * x * z / r_norm**5
    a_yz_pm = 3 * mu * y * z / r_norm**5

    # J2 partials
    a_xx_J2 = 1.5 * mu * J2 * R_e**2 * (5 * z**2 * (r_norm**2 - 7 * x**2) / r_norm**9 - (r_norm**2 - 5 * x**2) / r_norm**7)
    a_yy_J2 = 1.5 * mu * J2 * R_e**2 * (5 * z**2 * (r_norm**2 - 7 * y**2) / r_norm**9 - (r_norm**2 - 5 * y**2) / r_norm**7)
    a_zz_J2 = 1.5 * mu * J2 * R_e**2 * (5 * z**2 * (3 * r_norm**2 - 7 * z**2) / r_norm**9 - 3 * (r_norm**2 - 5 * z**2) / r_norm**7)
    a_xy_J2 = (3 / 2) * mu * J2 * R_e**2 * x * (-35 * z**2 * y / r_norm**9 + 5 * y / r_norm**7)
    a_xz_J2 = (3 / 2) * mu * J2 * R_e**2 * x * ((15 * z * r_norm**2 - 35 * z**3) / r_norm**9)
    a_yz_J2 = (3 / 2) * mu * J2 * R_e**2 * y * ((15 * z * r_norm**2 - 35 * z**3) / r_norm**9)

    # J3 partials
    a_xx_J3 = (5 / 2) * mu * J3 * R_e**3 * z / r_norm**9 * (7 * z**2 * (r_norm **2 - 9 * x**2) / r_norm**2 - 3 * (r_norm**2 - 7 * x**2))
    a_yy_J3 = (5 / 2) * mu * J3 * R_e**3 * z / r_norm**9 * (7 * z**2 * (r_norm **2 - 9 * y**2) / r_norm**2 - 3 * (r_norm**2 - 7 * y**2))
    a_zz_J3 = (5 / 2) * mu * J3 * R_e**3 * z / r_norm**7 * (70 * z**2 / r_norm**2 - 63 * z**4 / r_norm**4 - 15)
    a_xy_J3 = (5 / 2) * mu * J3 * R_e**3 * x * y * z / r_norm**9 * (21 - 63 * z**2 / r_norm**2)
    a_xz_J3 = (5 / 2) * mu * J3 * R_e**3 * x / r_norm**9 * (42 * z**2 - 63 * z**4 / r_norm**2 - 3 * r_norm**2)
    a_yz_J3 = (5 / 2) * mu * J3 * R_e**3 * y / r_norm**9 * (42 * z**2 - 63 * z**4 / r_norm**2 - 3 * r_norm**2)

    # Drag partials
    rho = compute_density(r_norm)
    a_xx_drag = -(rho * C_d * area) / (2*spacecraft_mass) * (u**2 / V_norm + V_norm)
    a_yy_drag = -(rho * C_d * area) / (2*spacecraft_mass) * (v**2 / V_norm + V_norm)
    a_zz_drag = -(rho * C_d * area) / (2*spacecraft_mass) * (w**2 / V_norm + V_norm)
    a_xy_drag = -(rho * C_d * area) / (2*spacecraft_mass) * (u * v / V_norm)
    a_xz_drag = -(rho * C_d * area) / (2*spacecraft_mass) * (u * w / V_norm)
    a_yz_drag = -(rho * C_d * area) / (2*spacecraft_mass) * (v * w / V_norm)

    # Combine all partials
    a_xx = a_xx_pm + a_xx_J2 + a_xx_J3 + a_xx_drag
    a_yy = a_yy_pm + a_yy_J2 + a_yy_J3 + a_yy_drag
    a_zz = a_zz_pm + a_zz_J2 + a_zz_J3 + a_zz_drag
    a_xy = a_xy_pm + a_xy_J2 + a_xy_J3 + a_xy_drag
    a_xz = a_xz_pm + a_xz_J2 + a_xz_J3 + a_xz_drag
    a_yz = a_yz_pm + a_yz_J2 + a_yz_J3 + a_yz_drag
    a_yx = a_xy
    a_zx = a_xz
    a_zy = a_yz

    # Compute velocity partials (all zeros)
    vel_partials = np.zeros((3, 3))

    # Compute gravity parameter partials
    a_xmu = -x / r_norm**3 + (3 / 2) * J2 * R_e**2 * x / r_norm ** 5 * (5 * z**2 / r_norm**2 - 1) + (5 / 2) * J3 * R_e**3 * x * z / r_norm**7 * (7 * z**2 / r_norm**2 - 3)
    a_ymu = -y / r_norm**3 + (3 / 2) * J2 * R_e**2 * y / r_norm ** 5 * (5 * z**2 / r_norm**2 - 1) + (5 / 2) * J3 * R_e**3 * y * z / r_norm**7 * (7 * z**2 / r_norm**2 - 3)
    a_zmu = -z / r_norm**3 + (3 / 2) * J2 * R_e**2 * z / r_norm ** 5 * (5 * z**2 / r_norm**2 - 3) + (5 / 2) * J3 * R_e**3 / r_norm**5 * (7 * z**4 / r_norm**4 - 6 * z**2 / r_norm**2 + 3 / 5)

    a_xJ2 = (3 / 2) * mu * R_e**2 * x / r_norm**5 * (5 * z**2 / r_norm**2 - 1)
    a_yJ2 = (3 / 2) * mu * R_e**2 * y / r_norm**5 * (5 * z**2 / r_norm**2 - 1)
    a_zJ2 = (3 / 2) * mu * R_e**2 * z / r_norm**5 * (5 * z**2 / r_norm**2 - 3)

    a_xJ3 = (5 / 2) * mu * R_e**3 * x * z / r_norm**7 * (7 * z**2 / r_norm**2 - 3)
    a_yJ3 = (5 / 2) * mu * R_e**3 * y * z / r_norm**7 * (7 * z**2 / r_norm**2 - 3)
    a_zJ3 = (5 / 2) * mu * R_e**3 / r_norm**5 * (7 * z**4 / r_norm**4 - 6 * z**2 / r_norm**2 + 3 / 5)

    # Assemble the Jacobian matrix
    A = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0],
                  [a_xx, a_xy, a_xz, 0, 0, 0, a_xmu, a_xJ2, a_xJ3],
                  [a_yx, a_yy, a_yz, 0, 0, 0, a_ymu, a_yJ2, a_yJ3],
                  [a_zx, a_zy, a_zz, 0, 0, 0, a_zmu, a_zJ2, a_zJ3],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0]])
    
    if 'BaseMat' in mode:
        return A

    temp_A = A[0:6, 0:6]
    for value in mode:
        if 'mu' in value:
            # Append mu partials to A
            temp_A = np.pad(temp_A, ((0,1),(0,1)), 'constant')
            needed_column = A[0:temp_A.shape[0], 6].reshape((temp_A.shape[0],1))
            temp_A[:, -1] = needed_column.flatten()
            # A = A[np.ix_([0,1,2,3,4,5,6],[0,1,2,3,4,5,6])]
        if 'J2' in value:
            # Append J2 partials to A. Needs to be done this way to maintain correct order
            temp_A = np.pad(temp_A, ((0,1),(0,1)), 'constant')
            needed_column = A[0:temp_A.shape[0], 7].reshape((temp_A.shape[0],1))
            temp_A[:, -1] = needed_column.flatten()
            #A = A[np.ix_([0,1,2,3,4,5,7],[0,1,2,3,4,5,7])]
        if 'J3' in value:
            # Append J3 partials to A
            temp_A = np.pad(temp_A, ((0,1),(0,1)), 'constant')
            needed_column = A[0:temp_A.shape[0], 8].reshape((temp_A.shape[0],1))
            temp_A[:, -1] = needed_column.flatten()
            # A = A[np.ix_([0,1,2,3,4,5,8],[0,1,2,3,4,5,8])]
        if 'Drag' in value:
            # Compute needed drag partials and append to A
            temp_A = np.pad(temp_A, ((0,1),(0,1)), 'constant')
            a_xCd = -(rho * area * V_norm * u) / (2*spacecraft_mass)
            a_yCd = -(rho * area * V_norm * v) / (2*spacecraft_mass)
            a_zCd = -(rho * area * V_norm * w) / (2*spacecraft_mass)
            # Set appropriate rows in last column
            temp_A[3, -1] = a_xCd
            temp_A[4, -1] = a_yCd
            temp_A[5, -1] = a_zCd
        if 'Stations' in value:
            # Append station partials to A, just adding 3 zero rows and columns per station
            for _ in range(station_positions_ecef.shape[0]):
                temp_A = np.pad(temp_A, ((0,3),(0,3)), 'constant')
            
    return temp_A

def compute_DCM(i, LoN, AoP):
    # Compute direction cosine matrix from perifocal to inertial frame
    DCM = np.array([[np.cos(LoN) * np.cos(AoP) - np.sin(LoN) * np.sin(AoP) * np.cos(i), -np.cos(LoN) * np.sin(AoP) - np.sin(LoN) * np.cos(AoP) * np.cos(i),  np.sin(LoN) * np.sin(i)],
                    [np.sin(LoN) * np.cos(AoP) + np.cos(LoN) * np.sin(AoP) * np.cos(i), -np.sin(LoN) * np.sin(AoP) + np.cos(LoN) * np.cos(AoP) * np.cos(i), -np.cos(LoN) * np.sin(i)],
                    [np.sin(AoP) * np.sin(i), np.cos(AoP) * np.sin(i), np.cos(i)]])
    
    return DCM

def measurement_jacobian(sat_state : np.array, station_state : np.array, earth_rotation_rate : float =2*np.pi/86164.0905):
    """
    This function computes measurement Jacobian associated with range and range rate measurements between a satellite and a ground station.
    Parameters:
    sat_state : np.Array
        Satellite state vector in Cartesian coordinates (x, y, z, u, v, w).
    station_state : np.Array
        Ground station state vector in Cartesian coordinates (x_s, y_s, z_s, u_s, v_s, w_s).
    earth_rotation_rate : float
        Earth's rotation rate in radians per second. Default is 2*pi/86164.0905 rad/s.
    Returns:
    H_sc : np.Array
        Measurement Jacobian with respect to the satellite state.
    H_station : np.Array
        Measurement Jacobian with respect to the ground station state.
    """

    x, y, z = sat_state[0:3]
    u, v, w = sat_state[3:6]
    x_s, y_s, z_s = station_state[0:3]
    u_s, v_s, w_s = station_state[3:6]

    rho = np.linalg.norm(sat_state[0:3] - station_state[0:3])
    rho_dot = np.dot((sat_state[0:3] - station_state[0:3]), (sat_state[3:6] - station_state[3:6])) / rho

    # Spacecraft range partials
    rho_x_sc = (x - x_s) / rho
    rho_y_sc = (y - y_s) / rho
    rho_z_sc = (z - z_s) / rho
    rho_u_sc = 0
    rho_v_sc = 0
    rho_w_sc = 0

    # Spacecraft range rate partials
    rho_dot_x_sc = (1 / rho) * ((u - u_s) - rho_dot * (x - x_s) / rho)
    rho_dot_y_sc = (1 / rho) * ((v - v_s) - rho_dot * (y - y_s) / rho)
    rho_dot_z_sc = (1 / rho) * ((w - w_s) - rho_dot * (z - z_s) / rho)
    rho_dot_u_sc = (x - x_s) / rho
    rho_dot_v_sc = (y - y_s) / rho
    rho_dot_w_sc = (z - z_s) / rho

    # Station range rate partials
    rho_dot_x_station = -(1 / rho) * ((u + earth_rotation_rate * y_s) + earth_rotation_rate * (y - y_s) - rho_dot * (x - x_s) / rho)
    rho_dot_y_station = -(1 / rho) * ((v - earth_rotation_rate * x_s) - earth_rotation_rate * (x - x_s) - rho_dot * (y - y_s) / rho)
    rho_dot_z_station = -rho_dot_z_sc

    # Construct measurement Jacobian
    H_sc = np.array([[rho_x_sc, rho_y_sc, rho_z_sc, rho_u_sc, rho_v_sc, rho_w_sc],
                  [rho_dot_x_sc, rho_dot_y_sc, rho_dot_z_sc, rho_dot_u_sc, rho_dot_v_sc, rho_dot_w_sc]])
    
    H_station = np.array([[-rho_x_sc, -rho_y_sc, -rho_z_sc],
                          [rho_dot_x_station, rho_dot_y_station, rho_dot_z_station]])
    
    return H_sc, H_station