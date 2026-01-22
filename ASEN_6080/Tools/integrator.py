import numpy as np
from .generic_functions import state_jacobian
from scipy.integrate import solve_ivp
class Integrator:
    def __init__(self, mu : float, R_e : float, mode : str = 'PointMass'):
        self.mu = mu
        self.R_e = R_e
        self.mode = mode

        if mode != 'PointMass' and mode != 'J2' and mode != 'J3' and mode != 'Full':
            raise ValueError("Invalid mode. Choose from 'PointMass', 'J2', 'J3', or 'Full'.")

    def keplerian_to_cartesian(self, a, e, i, LoN, AoP, f):
        # Compute perifocal radius magnitude
        r_mag = a * (1 - e**2) / (1 + e * np.cos(f))

        # Compute perifocal radius vector
        r_perifocal = r_mag * np.array([np.cos(f), np.sin(f), 0])

        # Compute perifocal velocity vector
        h = np.sqrt(self.mu * a * (1 - e**2))
        v_perifocal = (self.mu / h) * np.array([-np.sin(f), e + np.cos(f), 0])
        
        # Rotation matrices
        DCM = np.array([[np.cos(LoN) * np.cos(AoP) - np.sin(LoN) * np.sin(AoP) * np.cos(i), -np.cos(LoN) * np.sin(AoP) - np.sin(LoN) * np.cos(AoP) * np.cos(i),  np.sin(LoN) * np.sin(i)],
                        [np.sin(LoN) * np.cos(AoP) + np.cos(LoN) * np.sin(AoP) * np.cos(i), -np.sin(LoN) * np.sin(AoP) + np.cos(LoN) * np.cos(AoP) * np.cos(i), -np.cos(LoN) * np.sin(i)],
                        [np.sin(AoP) * np.sin(i), np.cos(AoP) * np.sin(i), np.cos(i)]])
 
        # Transform to inertial frame
        r_inertial = DCM @ r_perifocal
        v_inertial = DCM @ v_perifocal

        return r_inertial, v_inertial
    
    def cartesian_to_keplerian(self, r_vec, v_vec):
        # Define unit vectors
        x = np.array([1, 0, 0])
        y = np.array([0, 1, 0])
        z = np.array([0, 0, 1])

        # Compute orbital elements
        h_vec = np.cross(r_vec, v_vec)
        h = np.linalg.norm(h_vec)
        h_norm = h_vec / h

        e_vec = (1/self.mu) * np.cross(v_vec, h_vec) - (r_vec / np.linalg.norm(r_vec))
        e = np.linalg.norm(e_vec)
        e_norm = e_vec / e
        e_vec_perp = np.cross(h_norm, e_norm)

        p = np.linalg.norm(h)**2 / self.mu
        a = p / (1 - e**2)
        i = np.arccos(np.dot(h_norm, z))

        node_vec = np.cross(z, h_norm) / np.linalg.norm(np.cross(z, h_norm))
        node_vec_perp = np.cross(h_norm, node_vec)

        LoN = np.arctan2(np.dot(y, node_vec), np.dot(x, node_vec))
        AoP = np.arctan2(np.dot(e_vec, node_vec_perp), np.dot(e_vec, node_vec))
    
        f = np.arctan2(np.dot(r_vec, e_vec_perp), np.dot(r_vec, e_vec))

        return a, e, i, LoN, AoP, f
    
    def equations_of_motion(self, t, state):
        x, y, z = state[0:3]
        u, v, w = state[3:6]
        r = np.sqrt(x**2 + y**2 + z**2)

        # Determine J2 and J3 based on mode
        if self.mode == 'PointMass':
            J2 = 0
            J3 = 0
        elif self.mode == 'J2':
            J2 = state[6]
            J3 = 0
        elif self.mode == 'J3':
            J2 = 0
            J3 = state[6]
        else:
            J2 = state[6]
            J3 = state[7]

        x_dot = u
        y_dot = v
        z_dot = w
        u_dot = -self.mu * x / r**3 + (3 / 2) * (self.mu * J2 * self.R_e**2 * x / r**5) * (5 * (z**2 / r**2) - 1) + (5 / 2) * self.mu * J3 * self.R_e**3 * x * z / r**7 * (7 * z**2 / r**2 - 3)
        v_dot = -self.mu * y / r**3 + (3 / 2) * (self.mu * J2 * self.R_e**2 * y / r**5) * (5 * (z**2 / r**2) - 1) + (5 / 2) * self.mu * J3 * self.R_e**3 * y * z / r**7 * (7 * z**2 / r**2 - 3)
        w_dot = -self.mu * z / r**3 + (3 / 2) * (self.mu * J2 * self.R_e**2 * z / r**5) * (5 * (z**2 / r**2) - 3) + (5 / 2) * self.mu * J3 * self.R_e**3 / r**5 * (7 * z**4 / r**4 - 6 * z**2 / r**2 + 3 / 5)

        if self.mode == 'PointMass':
            return np.array([x_dot, y_dot, z_dot, u_dot, v_dot, w_dot])
        elif self.mode == 'J2' or self.mode == 'J3':
            return np.array([x_dot, y_dot, z_dot, u_dot, v_dot, w_dot, 0])
        else:
            return np.array([x_dot, y_dot, z_dot, u_dot, v_dot, w_dot, 0, 0])
    
    def full_dynamics(self, t, augmented_state):
        # This function is passed through the integrator when the initial state is augmented by the STM

        # Determine state length based on mode and assign J2 and J3 according to mode
        if self.mode == 'PointMass':
            state_length = 6
            J2 = 0
            J3 = 0
        elif self.mode == 'J2' or self.mode == 'J3':
            state_length = 7
            J2 = augmented_state[6] if self.mode == 'J2' else 0
            J3 = augmented_state[6] if self.mode == 'J3' else 0
        else:
            state_length = 8
            J2 = augmented_state[6]
            J3 = augmented_state[7]

        state = augmented_state[0:state_length]
        phi_flat = augmented_state[state_length:]
        phi = phi_flat.reshape((state_length, state_length))

        # Compute state derivatives
        state_dot = self.equations_of_motion(t, state)

        # Compute STM derivative
        A = state_jacobian(state[0:3], state[3:6], self.mu, J2, J3, self.R_e, mode=self.mode)
        phi_dot = A @ phi
        phi_dot_flat = phi_dot.flatten()

        return np.hstack((state_dot, phi_dot_flat))
        
    def integrate_eom(self, t_final, initial_state, teval = None):
        """Integrate the equations of motion for the spacecraft.
        Parameters:
        t_final : float
            Final time for integration in seconds.
        initial_state : np.array
            nx1 array of initial spacecraft state in ECI frame. First 6 elements are [x, y, z, u, v, w] in km and km/s.
        teval : np.array, optional
            1xN array of time points at which to store the computed solution. Default is None.
        Returns:
        time_vector : np.array
            1xN array of time points corresponding to the spacecraft states.
        state_history : np.array
            nxN array of spacecraft states over time in ECI frame."""
        
        t_span = (0, t_final)
        sol = solve_ivp(self.equations_of_motion, t_span, initial_state, method='RK45', rtol=1e-13, atol=1e-13, t_eval=teval)
        return sol.t, sol.y
    
    def integrate_stm(self, t_final, initial_state, phi_0 = None, teval = None):
        # Determine state length based on mode
        if self.mode == 'PointMass':
            state_length = 6
        elif self.mode == 'J2' or self.mode == 'J3':
            state_length = 7
        else:
            state_length = 8

        # Initialize STM as identity matrix
        if phi_0 is None:
            phi_0 = np.eye(state_length).flatten()

        augmented_initial_state = np.hstack((initial_state, phi_0))
        t_span = (0, t_final)
        sol = solve_ivp(self.full_dynamics, t_span, augmented_initial_state, method='RK45', rtol=1e-13, atol=1e-13, t_eval=teval)
        return sol.t, sol.y