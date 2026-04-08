import numpy as np
from .generic_functions import keplerian_to_cartesian

class EphemerisMgr:
    """
    This class handles the evaluation of a specific planets states using the methodology employed by classical ephemeris models. All coefficients are with respect to EME2000.
    Attributes:
        ephemeris_data (dict): A dictionary containing the ephemeris coefficients for the planetary body.
    Methods:
        load_ephemeris_coeffs(planetary_body): Loads the ephemeris coefficients for the specified planetary body. These coefficients are self contained within the class and are based on classical ephemeris models.
        evaluate_state(body, time): Evaluates the state of the specified planetary body at a given time using the loaded ephemeris data.
    """

    def __init__(self, planetary_body : str):
        """
        Constructor of Ephemeris Manager class.

        Parameters:
            planetary_body (str): Name of the planetary body to load ephemeris data for. Must be one of:
                'mercury', 'venus', 'earth', 'mars', 'jupiter',
                'saturn', 'uranus', 'neptune', 'pluto'
        """
        self.planetary_body = planetary_body
        if planetary_body.lower() != 'sun':  # Sun is at the center of the EME2000 frame, so we can skip loading coefficients for it
            self.ephemeris_coeffs = self.load_ephemeris_coeffs(planetary_body)

    def load_ephemeris_coeffs(self, planet : str) -> dict:
        """
        Returns orbital element coefficients for a given planet.

        Parameters:
            planet (str): Planet name, one of:
                'mercury', 'venus', 'earth', 'mars', 'jupiter',
                'saturn', 'uranus', 'neptune', 'pluto'

        Returns:
            dict with keys: L, a, e, i, W, P, mu_p
        """
        planet = planet.strip().lower()

        if planet == 'mercury':
            L = [252.250906,    149472.6746358, -0.00000535,     0.000000002]
            a = [0.387098310,   0.0,             0.0,            0.0]
            e = [0.20563175,    0.000020406,    -0.0000000284,  -0.00000000017]
            i = [7.004986,     -0.0059516,       0.00000081,     0.000000041]
            W = [48.330893,    -0.1254229,      -0.00008833,    -0.000000196]
            P = [77.456119,     0.1588643,      -0.00001343,     0.000000039]
            mu_p = 2.20320804864179e4

        elif planet == 'venus':
            L = [181.979801,    58517.8156760,   0.00000165,    -0.000000002]
            a = [0.72332982,    0.0,             0.0,            0.0]
            e = [0.00677188,   -0.000047766,     0.0000000975,   0.00000000044]
            i = [3.394662,     -0.0008568,      -0.00003244,     0.000000010]
            W = [76.679920,    -0.2780080,      -0.00014256,    -0.000000198]
            P = [131.563707,    0.0048646,      -0.00138232,    -0.000005332]
            mu_p = 3.2485859882646e5

        elif planet == 'earth':
            L = [100.466449,    35999.3728519,  -0.00000568,     0.0]
            a = [1.000001018,   0.0,             0.0,            0.0]
            e = [0.01670862,   -0.000042037,    -0.0000001236,   0.00000000004]
            i = [0.0,           0.0130546,      -0.00000931,    -0.000000034]
            W = [174.873174,   -0.2410908,       0.00004067,    -0.000001327]
            P = [102.937348,    0.3225557,       0.00015026,     0.000000478]
            mu_p = 3.98600432896939e5

        elif planet == 'mars':
            L = [355.433275,    19140.2993313,   0.00000261,    -0.000000003]
            a = [1.523679342,   0.0,             0.0,            0.0]
            e = [0.09340062,    0.000090483,    -0.0000000806,  -0.00000000035]
            i = [1.849726,     -0.0081479,      -0.00002255,    -0.000000027]
            W = [49.558093,    -0.2949846,      -0.00063993,    -0.000002143]
            P = [336.060234,    0.4438898,      -0.00017321,     0.000000300]
            mu_p = 4.28283142580671e4

        elif planet == 'jupiter':
            L = [34.351484,     3034.9056746,   -0.00008501,     0.000000004]
            a = [5.202603191,   0.0000001913,    0.0,            0.0]
            e = [0.04849485,    0.000163244,    -0.0000004719,  -0.00000000197]
            i = [1.303270,     -0.0019872,       0.00003318,     0.000000092]
            W = [100.464441,    0.1766828,       0.00090387,    -0.000007032]
            P = [14.331309,     0.2155525,       0.00072252,    -0.000004590]
            mu_p = 1.26712767857796e8

        elif planet == 'saturn':
            L = [50.077471,     1222.1137943,    0.00021004,    -0.000000019]
            a = [9.554909596,  -0.0000021389,    0.0,            0.0]
            e = [0.05550862,   -0.000346818,    -0.0000006456,   0.00000000338]
            i = [2.488878,      0.0025515,      -0.00004903,     0.000000018]
            W = [113.665524,   -0.2566649,      -0.00018345,     0.000000357]
            P = [93.056787,     0.5665496,       0.00052809,     0.000004882]
            mu_p = 3.79406260611373e7

        elif planet == 'uranus':
            L = [314.055005,    428.4669983,    -0.00000486,     0.000000006]
            a = [19.218446062, -0.0000000372,    0.00000000098,  0.0]
            e = [0.04629590,   -0.000027337,     0.0000000790,   0.00000000025]
            i = [0.773196,     -0.0016869,       0.00000349,     0.000000016]
            W = [74.005947,     0.0741461,       0.00040540,     0.000000104]
            P = [173.005159,    0.0893206,      -0.00009470,     0.000000413]
            mu_p = 5.79454900707188e6

        elif planet == 'neptune':
            L = [304.348665,    218.4862002,     0.00000059,    -0.000000002]
            a = [30.110386869, -0.0000001663,    0.00000000069,  0.0]
            e = [0.00898809,    0.000006408,    -0.0000000008,  -0.00000000005]
            i = [1.769952,      0.0002257,       0.00000023,     0.0]
            W = [131.784057,   -0.0061651,      -0.00000219,    -0.000000078]
            P = [48.123691,     0.0291587,       0.00007051,    -0.000000023]
            mu_p = 6.83653406387926e6

        elif planet == 'pluto':
            L = [238.92903833,  145.20780515,    0.0,            0.0]
            a = [39.48211675,  -0.00031596,      0.0,            0.0]
            e = [0.24882730,    0.00005170,      0.0,            0.0]
            i = [17.14001206,   0.00004818,      0.0,            0.0]
            W = [110.30393684, -0.01183482,      0.0,            0.0]
            P = [224.06891629, -0.04062942,      0.0,            0.0]
            mu_p = 9.81600887707005e2

        else:
            valid = ['mercury', 'venus', 'earth', 'mars', 'jupiter',
                    'saturn', 'uranus', 'neptune', 'pluto']
            raise ValueError(f"Unknown planet '{planet}'. Must be one of: {valid}")

        return {"L": L, "a": a, "e": e, "i": i, "W": W, "P": P, "mu_p": mu_p}

    def evaluate_state(self, epoch : float) -> np.ndarray:
        """
        Evaluates the state of the planetary body at a given time using the loaded ephemeris data.

        Parameters:
            epoch (float): Epoch in Julian days. Is converted to centuries internally for use in the ephemeris equations.

        Returns:
            np.ndarray: State vector [x, y, z, vx, vy, vz] in EME2000 frame
        """

        if self.planetary_body.lower() == 'sun':
            return np.zeros(6)  # Sun is at the center of the EME2000 frame, so its state vector is always zero
        
        # Convert epoch from Julian days to centuries and define gravitational parameter of the Sun
        time = (epoch - 2451545.0) / 36525.0
        mu_s = 132712440017.987

        # Extract coefficients
        L = self.ephemeris_coeffs["L"]
        a = self.ephemeris_coeffs["a"]
        e = self.ephemeris_coeffs["e"]
        i = self.ephemeris_coeffs["i"]
        W = self.ephemeris_coeffs["W"]
        P = self.ephemeris_coeffs["P"]

        # Compute orbital elements at given time
        L_t = L[0] + L[1]*time + L[2]*time**2 + L[3]*time**3
        a_t = a[0] + a[1]*time + a[2]*time**2 + a[3]*time**3
        e_t = e[0] + e[1]*time + e[2]*time**2 + e[3]*time**3
        i_t = i[0] + i[1]*time + i[2]*time**2 + i[3]*time**3
        W_t = W[0] + W[1]*time + W[2]*time**2 + W[3]*time**3
        P_t = P[0] + P[1]*time + P[2]*time**2 + P[3]*time**3

        # Convert angles from degrees to radians
        L_rad = np.radians(L_t)
        i_rad = np.radians(i_t)
        W_rad = np.radians(W_t)
        P_rad = np.radians(P_t)

        # Compute argument of periapsis
        AoP_rad = P_rad - W_rad

        # Compute mean anomaly M
        M_rad = L_rad - P_rad

        # Solve for C_cen using equation given in J. Meeus, Astronomical Algorithms, 1991
        C_cen = (2*e_t - 0.25*e_t**3 + 5*e_t**5/96) * np.sin(M_rad) + (1.25*e_t**2 - 11*e_t**4/24) * np.sin(2*M_rad) + (13/12*e_t**3 - 43*e_t**5/64) * np.sin(3*M_rad) + (103/96*e_t**4) * np.sin(4*M_rad) + (1097/960*e_t**5) * np.sin(5*M_rad)

        # Compute true anomaly
        nu_rad = M_rad + C_cen

        # Convert distances from AU to km
        a_km = a_t * 149597870.7

        # Compute state vector in EME2000 frame
        R, V = keplerian_to_cartesian(mu_s, a_km, e_t, i_rad, W_rad, AoP_rad, nu_rad)

        return np.hstack((R, V))
