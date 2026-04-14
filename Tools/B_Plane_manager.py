import numpy as np

class BPlaneMgr:
    """
    This class handles the computation of the B-plane, LTOF, and related parameters for a spacecraft trajectory. It uses the state vector of the spacecraft relative to the Earth to compute these parameters.
    """
    def __init__(self, state_at_RSOI : np.ndarray, mu : float):
        self.mu = mu
        self.state_vector = state_at_RSOI

    def get_ecc_and_sma(self):
        r_vec = self.state_vector[:3]
        v_vec = self.state_vector[3:]
        r_norm = np.linalg.norm(r_vec)
        v_norm = np.linalg.norm(v_vec)

        h_vec = np.cross(r_vec, v_vec)

        # Compute eccentricity vector
        e_vec = (np.cross(v_vec, h_vec) / self.mu) - (r_vec / np.linalg.norm(r_vec))
        e_mag = np.linalg.norm(e_vec)

        # Compute semi-major axis using Vis-viva equation
        a = -self.mu / (v_norm**2 - 2*self.mu/r_norm)

        return e_vec, e_mag, a

    def compute_perifocal_frame_vectors(self):
        r_vec = self.state_vector[:3]
        v_vec = self.state_vector[3:]

        # Compute specific angular momentum vector
        h_vec = np.cross(r_vec, v_vec)
        h_mag = np.linalg.norm(h_vec)

        # Compute eccentricity vector
        e_vec, e_mag, _ = self.get_ecc_and_sma()

        # Compute perifocal frame unit vectors
        p_hat = e_vec / e_mag
        w_hat = h_vec / h_mag
        q_hat = np.cross(w_hat, p_hat)
        
        return p_hat, q_hat, w_hat
    
    def compute_b_plane_frame(self):
        r_vec = self.state_vector[:3]
        v_vec = self.state_vector[3:]
        r_norm = np.linalg.norm(r_vec)
        v_norm = np.linalg.norm(v_vec)

        # Get the perifocal frame unit vectors and eccentricity magnitude
        _, _, w_hat = self.compute_perifocal_frame_vectors()

        # Use Vis-viva to compute semi-major axis
        _, e_mag, a = self.get_ecc_and_sma()

        # Compute the semi-minor axis
        b = a * np.sqrt(1 - e_mag**2)

        # Compute the B-plane frame unit vectors
        s_hat = v_vec / v_norm

        n_hat = np.array([0, 0, -1]).T

        t_hat = np.cross(s_hat, n_hat) / np.linalg.norm(np.cross(s_hat, n_hat))

        r_hat = np.cross(s_hat, t_hat)

        # Compute B-vector
        B = b * np.cross(s_hat, w_hat)

        return s_hat, t_hat, r_hat, B
    
    def compute_b_plane_DCM(self):
        """
        This function computes the DCM associated with converting from the ECI frame (assumed frame of inputted state) to B-Plane frame.
        """
        s_hat, t_hat, r_hat, _ = self.compute_b_plane_frame()
        DCM = np.array([s_hat, t_hat, r_hat])

        return DCM
    
    def compute_LOTF(self):
        v_inf = np.linalg.norm(self.state_vector[3:])

        _, e_mag, a = self.get_ecc_and_sma()
        p_hat, _, _ = self.compute_perifocal_frame_vectors()

        r_hat = self.state_vector[:3] / np.linalg.norm(self.state_vector[:3])

        cosv = np.dot(r_hat, p_hat)

        f = np.acosh(1 + (v_inf**2 / self.mu) * (a * (1 - e_mag**2)) / (1 + e_mag * cosv))

        LTOF = (self.mu / v_inf**3) * (np.sinh(f) - f)

        return LTOF

