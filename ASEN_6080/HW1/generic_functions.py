import numpy as np

def perturbation_jacobian(r : np.array, v : np.array, mu : float, J2 : float, J3 : float, R_e : float, mode : str = 'BaseMat'):
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
    """
    if mode != 'PointMass' and mode != 'J2' and mode != 'J3' and mode != 'Full' and mode != 'BaseMat':
        raise ValueError("Invalid mode. Choose from 'PointMass', 'J2', 'J3', 'Full', or 'BaseMat'.")

    x, y, z = r
    r_norm = np.linalg.norm(r)

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

    # Combine all partials
    a_xx = a_xx_pm + a_xx_J2 + a_xx_J3
    a_yy = a_yy_pm + a_yy_J2 + a_yy_J3
    a_zz = a_zz_pm + a_zz_J2 + a_zz_J3
    a_xy = a_xy_pm + a_xy_J2 + a_xy_J3
    a_xz = a_xz_pm + a_xz_J2 + a_xz_J3
    a_yz = a_yz_pm + a_yz_J2 + a_yz_J3
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
    
    if mode == 'PointMass':
        A = A[0:6, 0:6]
    elif mode == 'J2':
        # Keep initial 6x6 and J2 partials in the 8th column and row
        A = A[np.ix_([0,1,2,3,4,5,7],[0,1,2,3,4,5,7])]
    elif mode == 'J3':
        # Keep initial 6x6 and J3 partials in the 9th column and row
        A = A[np.ix_([0,1,2,3,4,5,8],[0,1,2,3,4,5,8])]
    elif mode == 'Full':
        A = A[np.ix_([0,1,2,3,4,5,7,8],[0,1,2,3,4,5,7,8])]
    else:
        pass
    return A

def compute_DCM(i, LoN, AoP):
    # Compute direction cosine matrix from perifocal to inertial frame
    DCM = np.array([[np.cos(LoN) * np.cos(AoP) - np.sin(LoN) * np.sin(AoP) * np.cos(i), -np.cos(LoN) * np.sin(AoP) - np.sin(LoN) * np.cos(AoP) * np.cos(i),  np.sin(LoN) * np.sin(i)],
                    [np.sin(LoN) * np.cos(AoP) + np.cos(LoN) * np.sin(AoP) * np.cos(i), -np.sin(LoN) * np.sin(AoP) + np.cos(LoN) * np.cos(AoP) * np.cos(i), -np.cos(LoN) * np.sin(i)],
                    [np.sin(AoP) * np.sin(i), np.cos(AoP) * np.sin(i), np.cos(i)]])
    
    return DCM

def measurement_jacobian(sat_state : np.array, station_state : np.array):
    """
    This function computes measurement Jacobian associated with range and range rate measurements between a satellite and a ground station.
    Parameters:
    sat_state : np.Array
        atellite state vector in Cartesian coordinates (x, y, z, u, v, w).
    station_state : np.Array
        Ground station state vector in Cartesian coordinates (x_s, y_s, z_s, u_s, v_s, w_s).
    """

    x, y, z = sat_state[0:3]
    u, v, w = sat_state[3:6]
    x_s, y_s, z_s = station_state[0:3]
    u_s, v_s, w_s = station_state[3:6]

    rho = np.linalg.norm(sat_state[0:3] - station_state[0:3])
    rho_dot = np.dot((sat_state[0:3] - station_state[0:3]), (sat_state[3:6] - station_state[3:6])) / rho

    # Range partials
    rho_x= (x - x_s) / rho
    rho_y = (y - y_s) / rho
    rho_z = (z - z_s) / rho
    rho_u = 0
    rho_v = 0
    rho_w = 0

    # Range rate partials
    rho_dot_x = (1 / rho) * ((u - u_s) - rho_dot * (x - x_s) / rho)
    rho_dot_y = (1 / rho) * ((v - v_s) - rho_dot * (y - y_s) / rho)
    rho_dot_z = (1 / rho) * ((w - w_s) - rho_dot * (z - z_s) / rho)
    rho_dot_u = (x - x_s) / rho
    rho_dot_v = (y - y_s) / rho
    rho_dot_w = (z - z_s) / rho

    # Construct measurement Jacobian
    H = np.array([[rho_x, rho_y, rho_z, rho_u, rho_v, rho_w],
                  [rho_dot_x, rho_dot_y, rho_dot_z, rho_dot_u, rho_dot_v, rho_dot_w]])
    
    return H