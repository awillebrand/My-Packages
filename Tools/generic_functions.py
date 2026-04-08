import numpy as np

def compute_density(r_norm : float, rho_0 : float = 3.614e-13, r_0 : float = 700000.0 + 6378136.3, H : float = 88667.0):
    """Compute atmospheric density at the satellite's position using an exponential model.
    Inputs:
    r_norm : float
        Magnitude of the satellite's position vector in km.
    rho_0 : float, optional
        Reference atmospheric density at reference altitude in kg/m^3. Default is 3.614e-13 kg/m^3 (approx. 700 km).
    r_0 : float, optional
        Reference radius from Earth's center in m. Default is 700 km altitude + Earth's radius (6378136.3 m).
    H : float, optional
        Scale height in km. Default is 88667 m (88.667 km).
    Returns:
    density : float
        Atmospheric density at the satellite's position in kg/m^3.
    """
    r = r_norm*1000  # Convert km to m

    rho = rho_0 * np.exp(-(r-r_0)/ H)

    return rho

def state_jacobian(r : np.array,
                   V : np.array,
                   mu : float = None,
                   J2 : float = None,
                   J3 : float = None,
                   C_d : float = None,
                   C_r  : float = None,
                   P_solar : float = None,
                   sun_pos : np.array = None,
                   mu_third_body : float = None,
                   third_body_state : float = None,
                   station_positions_ecef : np.array = [],
                   R_e : float = None,
                   mode : list = ['BaseMat'],
                   spacecraft_area : float = None,
                   spacecraft_mass : float = None,
                   srp_area_to_mass : float = None,
                   earth_spin_rate : float = None,
                   DMC : bool = False,
                   beta_mat : np.ndarray = None):
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
    C_r : float
        Solar radiation pressure coefficient.
    epoch_jd : float
        Epoch in Julian Date, used for computing third body positions and any time-dependent effects.
    mu_third_body : float
        Gravitational parameter of the third body for third body perturbation partials.
    third_body : str
        Name of the third body (e.g., 'Sun', 'Moon') for which to compute third body perturbation partials.
    station_positions_ecef : np.array
        Nx3 array of ground station positions in ECEF coordinates, where N is the number of stations.
    R_e : float
        Earth's radius.
    mode : list
        List of strings specifying which partials to include in the Jacobian. Options are 'BaseMat', 'J2', 'J3', and/or 'Drag'.
    spacecraft_area : float
        Cross-sectional area of the spacecraft in m^2. Default is 3.0 m^2.
    spacecraft_mass : float
        Mass of the spacecraft in kg. Default is 970.0 kg.
    srp_area_to_mass : float
        Area-to-mass ratio for solar radiation pressure calculations in m^2/kg.
    DMC : bool
        If True, include dynamic model compensation terms in the Jacobian. Default is False.
    beta_mat : np.ndarray
        3x3 diagnoal matrix of time constants for dynamic model compensation. Required if DMC is True.
    Returns:
    A : np.Array
        State Jacobian matrix.
    """

    # if set(mode).isdisjoint({'BaseMat', 'J2', 'J3', 'Drag', 'Stations'}):
    #     raise ValueError("Invalid mode specified. Choose from 'BaseMat', 'mu', 'J2', 'J3', 'Drag', and/or 'Stations'.")
    if mu == None:
        if 'mu' in mode:
            raise Warning("Gravitational parameter partials requested but mu not provided. Defaulting to zero.")
        mu = 0
    if J2 == None:
        if 'J2' in mode:
            raise Warning("J2 partials requested but J2 not provided. Defaulting to zero.")
        J2 = 0
    if J3 == None:
        if 'J3' in mode:
            raise Warning("J3 partials requested but J3 not provided. Defaulting to zero.")
        J3 = 0
    if C_d == None:
        if 'Drag' in mode:
            raise Warning("Drag partials requested but C_d not provided. Defaulting to zero.")
        C_d = 0
    if C_r == None:
        if 'SRP' in mode:
            raise Warning("SRP partials requested but C_r not provided. Defaulting to zero.")
        if srp_area_to_mass == None:
            raise Warning("SRP partials requested but srp_area_to_mass not provided. Defaulting to zero.")
        C_r = 0
    if mu_third_body == None:
        if 'Third Body' in mode:
            raise Warning("Third body partials requested but mu_third_body not provided. Defaulting to zero.")
        mu_third_body = 0
    if DMC and beta_mat is None:
        raise ValueError("Beta must be provided for dynamic model compensation.")
    
    x, y, z = r
    r_norm = np.linalg.norm(r)
    u, v, w = V
    V_norm = np.linalg.norm(V)

    # Initialize position partials to zero
    a_xx = 0
    a_xy = 0
    a_xz = 0
    a_yx = 0
    a_yy = 0
    a_yz = 0
    a_zx = 0
    a_zy = 0
    a_zz = 0

    # Initialize velocity partials to zero
    a_xu = 0
    a_xv = 0
    a_xw = 0
    a_yu = 0
    a_yv = 0
    a_yw = 0
    a_zu = 0
    a_zv = 0
    a_zw = 0

    # Compute position partials
    # Point mass partials
    if 'mu' in mode:
        a_xx_pm = mu / r_norm**5 * (3 * x**2 - r_norm**2)
        a_yy_pm = mu / r_norm**5 * (3 * y**2 - r_norm**2)
        a_zz_pm = mu / r_norm**5 * (3 * z**2 - r_norm**2)
        a_xy_pm = 3 * mu * x * y / r_norm**5
        a_xz_pm = 3 * mu * x * z / r_norm**5
        a_yz_pm = 3 * mu * y * z / r_norm**5

        a_xx += a_xx_pm
        a_xy += a_xy_pm
        a_xz += a_xz_pm
        a_yx += a_xy_pm
        a_yy += a_yy_pm
        a_yz += a_yz_pm
        a_zx += a_xz_pm
        a_zy += a_yz_pm
        a_zz += a_zz_pm

    # J2 partials
    if 'J2' in mode:
        a_xx_J2 = 1.5 * mu * J2 * R_e**2 * (5 * z**2 * (r_norm**2 - 7 * x**2) / r_norm**9 - (r_norm**2 - 5 * x**2) / r_norm**7)
        a_yy_J2 = 1.5 * mu * J2 * R_e**2 * (5 * z**2 * (r_norm**2 - 7 * y**2) / r_norm**9 - (r_norm**2 - 5 * y**2) / r_norm**7)
        a_zz_J2 = 1.5 * mu * J2 * R_e**2 * (5 * z**2 * (3 * r_norm**2 - 7 * z**2) / r_norm**9 - 3 * (r_norm**2 - 5 * z**2) / r_norm**7)
        a_xy_J2 = (3 / 2) * mu * J2 * R_e**2 * x * (-35 * z**2 * y / r_norm**9 + 5 * y / r_norm**7)
        a_xz_J2 = (3 / 2) * mu * J2 * R_e**2 * x * ((15 * z * r_norm**2 - 35 * z**3) / r_norm**9)
        a_yz_J2 = (3 / 2) * mu * J2 * R_e**2 * y * ((15 * z * r_norm**2 - 35 * z**3) / r_norm**9)

        a_xx += a_xx_J2
        a_xy += a_xy_J2
        a_xz += a_xz_J2
        a_yx += a_xy_J2
        a_yy += a_yy_J2
        a_yz += a_yz_J2
        a_zx += a_xz_J2
        a_zy += a_yz_J2
        a_zz += a_zz_J2

    # J3 partials
    if 'J3' in mode:
        a_xx_J3 = (5 / 2) * mu * J3 * R_e**3 * z / r_norm**9 * (7 * z**2 * (r_norm **2 - 9 * x**2) / r_norm**2 - 3 * (r_norm**2 - 7 * x**2))
        a_yy_J3 = (5 / 2) * mu * J3 * R_e**3 * z / r_norm**9 * (7 * z**2 * (r_norm **2 - 9 * y**2) / r_norm**2 - 3 * (r_norm**2 - 7 * y**2))
        a_zz_J3 = (5 / 2) * mu * J3 * R_e**3 * z / r_norm**7 * (70 * z**2 / r_norm**2 - 63 * z**4 / r_norm**4 - 15)
        a_xy_J3 = (5 / 2) * mu * J3 * R_e**3 * x * y * z / r_norm**9 * (21 - 63 * z**2 / r_norm**2)
        a_xz_J3 = (5 / 2) * mu * J3 * R_e**3 * x / r_norm**9 * (42 * z**2 - 63 * z**4 / r_norm**2 - 3 * r_norm**2)
        a_yz_J3 = (5 / 2) * mu * J3 * R_e**3 * y / r_norm**9 * (42 * z**2 - 63 * z**4 / r_norm**2 - 3 * r_norm**2)

        a_xx += a_xx_J3
        a_xy += a_xy_J3
        a_xz += a_xz_J3
        a_yx += a_xy_J3
        a_yy += a_yy_J3
        a_yz += a_yz_J3
        a_zx += a_xz_J3
        a_zy += a_yz_J3
        a_zz += a_zz_J3

    # Drag partials
    if 'Drag' in mode:
        # Convert velocity to relative velocity in ECEF frame by subtracting Earth's rotation
        V_rel = np.array([u + earth_spin_rate * y, v - earth_spin_rate * x, w])
        u_rel, v_rel, w_rel = V_rel
        V_rel_norm = np.linalg.norm(V_rel)
        rho = compute_density(r_norm) * 1e9 # Convert from kg/m^3 to kg/km^3 <---- DOUBLE CHECK THIS CONVERSION
        H = 88.6670 # Scale height in km

        a_xx_drag = u_rel * (x * rho * V_rel_norm * C_d * spacecraft_area) / (2 * spacecraft_mass * H * r_norm)
        a_xy_drag = u_rel * (y * rho * V_rel_norm * C_d * spacecraft_area) / (2 * spacecraft_mass * H * r_norm)
        a_xz_drag = u_rel * (z * rho * V_rel_norm * C_d * spacecraft_area) / (2 * spacecraft_mass * H * r_norm)

        a_yx_drag = v_rel * (x * rho * V_rel_norm * C_d * spacecraft_area) / (2 * spacecraft_mass * H * r_norm)
        a_yy_drag = v_rel * (y * rho * V_rel_norm * C_d * spacecraft_area) / (2 * spacecraft_mass * H * r_norm)
        a_yz_drag = v_rel * (z * rho * V_rel_norm * C_d * spacecraft_area) / (2 * spacecraft_mass * H * r_norm)

        a_zx_drag = w_rel * (x * rho * V_rel_norm * C_d * spacecraft_area) / (2 * spacecraft_mass * H * r_norm)
        a_zy_drag = w_rel * (y * rho * V_rel_norm * C_d * spacecraft_area) / (2 * spacecraft_mass * H * r_norm)
        a_zz_drag = w_rel * (z * rho * V_rel_norm * C_d * spacecraft_area) / (2 * spacecraft_mass * H * r_norm)
        

        a_xu_drag = -(rho * C_d * spacecraft_area) / (2*spacecraft_mass) * (u_rel**2 / V_rel_norm + V_rel_norm)
        a_yv_drag = -(rho * C_d * spacecraft_area) / (2*spacecraft_mass) * (v_rel**2 / V_rel_norm + V_rel_norm)
        a_zw_drag = -(rho * C_d * spacecraft_area) / (2*spacecraft_mass) * (w_rel**2 / V_rel_norm + V_rel_norm)
        a_xv_drag = -(rho * C_d * spacecraft_area) / (2*spacecraft_mass) * (u_rel * v_rel / V_rel_norm)
        a_xw_drag = -(rho * C_d * spacecraft_area) / (2*spacecraft_mass) * (u_rel * w_rel / V_rel_norm)
        a_yw_drag = -(rho * C_d * spacecraft_area) / (2*spacecraft_mass) * (v_rel * w_rel / V_rel_norm)
        a_yu_drag = a_xv_drag
        a_zu_drag = a_xw_drag
        a_zv_drag = a_yw_drag

        # Add drag partials to position partials
        a_xx += a_xx_drag
        a_xy += a_xy_drag
        a_xz += a_xz_drag
        a_yx += a_yx_drag
        a_yy += a_yy_drag
        a_yz += a_yz_drag
        a_zx += a_zx_drag
        a_zy += a_zy_drag
        a_zz += a_zz_drag

        # Add drag partials to velocity partials
        a_xu += a_xu_drag
        a_xv += a_xv_drag
        a_xw += a_xw_drag
        a_yu += a_yu_drag
        a_yv += a_yv_drag
        a_yw += a_yw_drag
        a_zu += a_zu_drag
        a_zv += a_zv_drag
        a_zw += a_zw_drag

    # Compute SRP partials
    if 'SRP' in mode:
        # Compute vector from sun to spacecraft and its magnitude
        r_sun_sc = r - sun_pos
        R = np.linalg.norm(r_sun_sc)

        # Break vector pointing from third body to spacecraft into components
        delta_x_srp = r_sun_sc[0]
        delta_y_srp = r_sun_sc[1]
        delta_z_srp = r_sun_sc[2]

        # Compute partial derivatives of SRP acceleration with respect to position components
        AU_KM = 149597870.700
        srp_scale = AU_KM**2 / 1000.0  # to match the acceleration formulation

        a_xx_SRP = C_r * P_solar * srp_scale * srp_area_to_mass * (R**2 - 3 * delta_x_srp**2) / R**5
        a_yy_SRP = C_r * P_solar * srp_scale * srp_area_to_mass * (R**2 - 3 * delta_y_srp**2) / R**5
        a_zz_SRP = C_r * P_solar * srp_scale * srp_area_to_mass * (R**2 - 3 * delta_z_srp**2) / R**5
        a_xy_SRP = -3 * C_r * P_solar * srp_scale * srp_area_to_mass * delta_x_srp * delta_y_srp / R**5
        a_xz_SRP = -3 * C_r * P_solar * srp_scale * srp_area_to_mass * delta_x_srp * delta_z_srp / R**5
        a_yz_SRP = -3 * C_r * P_solar * srp_scale * srp_area_to_mass * delta_y_srp * delta_z_srp / R**5
        a_yx_SRP = a_xy_SRP
        a_zx_SRP = a_xz_SRP
        a_zy_SRP = a_yz_SRP

        # Add SRP partials to position partials
        a_xx += a_xx_SRP
        a_xy += a_xy_SRP
        a_xz += a_xz_SRP
        a_yx += a_yx_SRP
        a_yy += a_yy_SRP
        a_yz += a_yz_SRP
        a_zx += a_zx_SRP
        a_zy += a_zy_SRP
        a_zz += a_zz_SRP

    # Compute third body partials
    if 'Third Body' in mode:
        # Compute vector from spacecraft to third body and its magnitude
        r_sc_third_body = sun_pos - r
        r_third_body = np.linalg.norm(r_sc_third_body)
        delta_x_third_body = r_sc_third_body[0]
        delta_y_third_body = r_sc_third_body[1]
        delta_z_third_body = r_sc_third_body[2]

        a_xx_third_body = -mu_third_body * (r_third_body**2 - 3 * delta_x_third_body**2) / r_third_body**5
        a_yy_third_body = -mu_third_body * (r_third_body**2 - 3 * delta_y_third_body**2) / r_third_body**5
        a_zz_third_body = -mu_third_body * (r_third_body**2 - 3 * delta_z_third_body**2) / r_third_body**5
        a_xy_third_body = 3 * mu_third_body * delta_x_third_body * delta_y_third_body / r_third_body**5
        a_xz_third_body = 3 * mu_third_body * delta_x_third_body * delta_z_third_body / r_third_body**5
        a_yz_third_body = 3 * mu_third_body * delta_y_third_body * delta_z_third_body / r_third_body**5
        a_yx_third_body = a_xy_third_body
        a_zx_third_body = a_xz_third_body
        a_zy_third_body = a_yz_third_body

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
                  [a_xx, a_xy, a_xz, a_xu, a_xv, a_xw, a_xmu, a_xJ2, a_xJ3],
                  [a_yx, a_yy, a_yz, a_yu, a_yv, a_yw, a_ymu, a_yJ2, a_yJ3],
                  [a_zx, a_zy, a_zz, a_zu, a_zv, a_zw, a_zmu, a_zJ2, a_zJ3],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0]])
    
    if 'BaseMat' in mode:
        return A[0:6, 0:6]

    temp_A = A[0:6, 0:6]
    for value in mode:
        if 'mu' in value:
            #  mu partials to A
            temp_A = np.pad(temp_A, ((0,1),(0,1)), 'constant')
            needed_column = A[0:temp_A.shape[0], 6].reshape((temp_A.shape[0],1))
            temp_A[:, -1] = needed_column.flatten()
            # A = A[np.ix_([0,1,2,3,4,5,6],[0,1,2,3,4,5,6])]
        if 'J2' in value:
            #  J2 partials to A. Needs to be done this way to maintain correct order
            temp_A = np.pad(temp_A, ((0,1),(0,1)), 'constant')
            needed_column = A[0:temp_A.shape[0], 7].reshape((temp_A.shape[0],1))
            temp_A[:, -1] = needed_column.flatten()
            #A = A[np.ix_([0,1,2,3,4,5,7],[0,1,2,3,4,5,7])]
        if 'J3' in value:
            #  J3 partials to A
            temp_A = np.pad(temp_A, ((0,1),(0,1)), 'constant')
            needed_column = A[0:temp_A.shape[0], 8].reshape((temp_A.shape[0],1))
            temp_A[:, -1] = needed_column.flatten()
            # A = A[np.ix_([0,1,2,3,4,5,8],[0,1,2,3,4,5,8])]
        if 'Drag' in value:
            # Compute needed drag partials and  to A
            temp_A = np.pad(temp_A, ((0,1),(0,1)), 'constant')
            a_xCd = -(rho * spacecraft_area * V_rel_norm * u_rel) / (2*spacecraft_mass)
            a_yCd = -(rho * spacecraft_area * V_rel_norm * v_rel) / (2*spacecraft_mass)
            a_zCd = -(rho * spacecraft_area * V_rel_norm * w_rel) / (2*spacecraft_mass)
            # Set appropriate rows in last column
            temp_A[3, -1] = a_xCd
            temp_A[4, -1] = a_yCd
            temp_A[5, -1] = a_zCd
        if 'SRP' in value:
            # Compute needed SRP partials and add to A
            temp_A = np.pad(temp_A, ((0,1),(0,1)), 'constant')
            a_xCr = P_solar * srp_scale * srp_area_to_mass * delta_x_srp / R**3
            a_yCr = P_solar * srp_scale * srp_area_to_mass * delta_y_srp / R**3
            a_zCr = P_solar * srp_scale * srp_area_to_mass * delta_z_srp / R**3
            # Set appropriate rows in last column
            temp_A[3, -1] = a_xCr
            temp_A[4, -1] = a_yCr
            temp_A[5, -1] = a_zCr
        if 'Third Body' in value:
            # Compute needed third body partials and add to A
            temp_A = np.pad(temp_A, ((0,1),(0,1)), 'constant')
            r_central_body_to_third_body = third_body_state[0:3]
            partial_vec = r_sc_third_body / r_third_body**3 - r_central_body_to_third_body / np.linalg.norm(r_central_body_to_third_body)**3
            a_x_third_body = partial_vec[0]
            a_y_third_body = partial_vec[1]
            a_z_third_body = partial_vec[2]
            # Set appropriate rows in last column
            temp_A[3, -1] = a_x_third_body
            temp_A[4, -1] = a_y_third_body
            temp_A[5, -1] = a_z_third_body
        if 'Stations' in value:
            #  station partials to A, just adding 3 zero rows and columns per station
            for _ in range(station_positions_ecef.shape[0]):
                temp_A = np.pad(temp_A, ((0,3),(0,3)), 'constant')
                
    if DMC:
        # Add DMC partials to A
        D = np.concatenate((np.zeros((3,3)), np.eye(3)), axis=0)

        temp_A = np.pad(temp_A, ((0,3),(0,3)), 'constant')
        temp_A[0:6, -3:] = D
        temp_A[-3:, -3:] = -beta_mat

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

def covariance_ellipse(center, cov_matrix, num_points=120):
    """
    This function computes the covariance ellipse for a N dimensional Gaussian distribution given its mean and covariance matrix.
    Parameters:
    center : np.Array
        Array representing the mean (x, y, z) of the distribution.
    cov_matrix : np.Array
        3x3 covariance matrix for the x, y, and z dimensions.
    num_points : int
        Number of points to generate along the ellipse. Default is 120.
    Returns:
    ellipse_points : np.Array
        Array of shape (num_points, N) containing the coordinates of the covariance ellipse.
    """

    # Eigenvalue decomposition of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Sort the eigenvalues and corresponding eigenvectors
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    # Generate points on a unit sphere
    phi = np.linspace(0, np.pi, num_points)
    theta = np.linspace(0, 2 * np.pi, num_points)
    phi, theta = np.meshgrid(phi, theta)
    x_sphere = np.sin(phi) * np.cos(theta)
    y_sphere = np.sin(phi) * np.sin(theta)
    z_sphere = np.cos(phi)

    # Scale the unit sphere by the eigenvalues (which represent the lengths of the ellipse axes)
    ellipse_points = eigenvectors @ np.diag(3*np.sqrt(eigenvalues)) @ np.array([x_sphere.flatten(), y_sphere.flatten(), z_sphere.flatten()])

    return ellipse_points.T + center

def covariance_ellipse_2D(center, cov_matrix, num_points=120, sigma_level=3):
    """
    This function computes the covariance ellipse for a 2 dimensional Gaussian distribution given its mean and covariance matrix.
    Parameters:
    center : np.Array
        Array representing the mean (x, y) of the distribution.
    cov_matrix : np.Array
        2x2 covariance matrix for the x and y dimensions.
    num_points : int, optional
        Number of points to generate along the ellipse. Default is 120.
    sigma_level : float, optional
        The sigma level to scale the ellipse by. Default is 3, which corresponds to a 99.7% confidence interval for a 2D Gaussian distribution.
    Returns:
    ellipse_points : np.Array
        Array of shape (num_points, 2) containing the coordinates of the covariance ellipse.
    """

    # Eigenvalue decomposition of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Sort the eigenvalues and corresponding eigenvectors
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    # Generate points on a unit sphere
    phi = np.linspace(0, np.pi, num_points)
    theta = np.linspace(0, 2 * np.pi, num_points)
    phi, theta = np.meshgrid(phi, theta)
    x_circle = np.cos(theta)
    y_circle = np.sin(theta)

    # Scale the unit circle by the eigenvalues (which represent the lengths of the ellipse axes)
    ellipse_points = eigenvectors @ np.diag(sigma_level*np.sqrt(eigenvalues)) @ np.array([x_circle.flatten(), y_circle.flatten()])

    return ellipse_points.T + center

def keplerian_to_cartesian(mu, a, e, i, LoN, AoP, f):
    # Compute perifocal radius magnitude
    r_mag = a * (1 - e**2) / (1 + e * np.cos(f))

    # Compute perifocal radius vector
    r_perifocal = r_mag * np.array([np.cos(f), np.sin(f), 0])

    # Compute perifocal velocity vector
    h = np.sqrt(mu * a * (1 - e**2))
    v_perifocal = (mu / h) * np.array([-np.sin(f), e + np.cos(f), 0])
    
    # Rotation matrices
    DCM = np.array([[np.cos(LoN) * np.cos(AoP) - np.sin(LoN) * np.sin(AoP) * np.cos(i), -np.cos(LoN) * np.sin(AoP) - np.sin(LoN) * np.cos(AoP) * np.cos(i),  np.sin(LoN) * np.sin(i)],
                    [np.sin(LoN) * np.cos(AoP) + np.cos(LoN) * np.sin(AoP) * np.cos(i), -np.sin(LoN) * np.sin(AoP) + np.cos(LoN) * np.cos(AoP) * np.cos(i), -np.cos(LoN) * np.sin(i)],
                    [np.sin(AoP) * np.sin(i), np.cos(AoP) * np.sin(i), np.cos(i)]])

    # Transform to inertial frame
    r_inertial = DCM @ r_perifocal
    v_inertial = DCM @ v_perifocal

    return r_inertial, v_inertial

def cartesian_to_keplerian(mu, r_vec, v_vec):
    # Define unit vectors
    x = np.array([1, 0, 0])
    y = np.array([0, 1, 0])
    z = np.array([0, 0, 1])

    # Compute orbital elements
    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)
    h_norm = h_vec / h

    e_vec = (1/mu) * np.cross(v_vec, h_vec) - (r_vec / np.linalg.norm(r_vec))
    e = np.linalg.norm(e_vec)
    e_norm = e_vec / e
    e_vec_perp = np.cross(h_norm, e_norm)

    p = np.linalg.norm(h)**2 / mu
    a = p / (1 - e**2)
    i = np.arccos(np.dot(h_norm, z))

    node_vec = np.cross(z, h_norm) / np.linalg.norm(np.cross(z, h_norm))
    node_vec_perp = np.cross(h_norm, node_vec)

    LoN = np.arctan2(np.dot(y, node_vec), np.dot(x, node_vec))
    AoP = np.arctan2(np.dot(e_vec, node_vec_perp), np.dot(e_vec, node_vec))

    f = np.arctan2(np.dot(r_vec, e_vec_perp), np.dot(r_vec, e_vec))

    return a, e, i, LoN, AoP, f

def compute_consider_parameter_partials(consider_parameter : str, r : np.array, V : np.array, mu : float, J2 : float, J3 : float, C_d : float, station_positions_ecef : np.array, R_e, spacecraft_area : float = 0, spacecraft_mass : float = 1, earth_spin_rate : float = 7.2921158553E-5):
    """
    Computes vector partial derivatives of inputted consider parameters based on dynamics. Matches those compute for the A matrix in state_jacobian.

    Parameters:
    consider_parameter : str
        String specifying which consider parameter to compute partials for. Options are 'mu', 'J2', 'J3', and/or 'Drag'.
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
    Returns:
        parameter_partials : np.Array
            Vector of partial derivatives of the consider parameter with respect to the state vector.
    """
    x, y, z = r
    r_norm = np.linalg.norm(r)
    u, v, w = V
    V_norm = np.linalg.norm(V)

    # Drag partials
    # Convert velocity to relative velocity in ECEF frame by subtracting Earth's rotation
    V_rel = np.array([u + earth_spin_rate * y, v - earth_spin_rate * x, w])
    u_rel, v_rel, w_rel = V_rel
    V_rel_norm = np.linalg.norm(V_rel)
    rho = compute_density(r_norm) * 1e9 # Convert from kg/m^3 to kg/km^3 <---- DOUBLE CHECK THIS CONVERSION

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

    a_xCd = -(rho * spacecraft_area * V_rel_norm * u_rel) / (2*spacecraft_mass)
    a_yCd = -(rho * spacecraft_area * V_rel_norm * v_rel) / (2*spacecraft_mass)
    a_zCd = -(rho * spacecraft_area * V_rel_norm * w_rel) / (2*spacecraft_mass)

    if consider_parameter == 'mu':
        parameter_partials = np.array([0, 0, 0, a_xmu, a_ymu, a_zmu])
    elif consider_parameter == 'J2':
        parameter_partials = np.array([0, 0, 0, a_xJ2, a_yJ2, a_zJ2])
    elif consider_parameter == 'J3':
        parameter_partials = np.array([0, 0, 0, a_xJ3, a_yJ3, a_zJ3])
    elif consider_parameter == 'Drag':
        parameter_partials = np.array([0, 0, 0, a_xCd, a_yCd, a_zCd])
    else:
        raise ValueError("Invalid consider parameter. Must be 'mu', 'J2', 'J3', or 'Drag'.")
    
    return parameter_partials