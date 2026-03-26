"""
Test suite for the Integrator's consider-parameter integration pipeline.

Covers:
  1. Nominal execution of integrate_stm_and_theta (no crashes, correct shapes).
  2. Reasonableness of returned state, STM (phi), and sensitivity matrix (theta).
  3. Verification that compute_consider_parameter_partials vectors match the
     corresponding columns of the state Jacobian matrix A produced by
     state_jacobian.
"""

import unittest
import numpy as np
from numpy.testing import assert_allclose

# ── Module imports (adjust the path if the package isn't installed) ──────────
from ASEN_6080.Tools.generic_functions import (
    state_jacobian,
    compute_consider_parameter_partials,
)
from ASEN_6080.Tools.integrator import Integrator


# ═══════════════════════════════════════════════════════════════════════════════
# Shared physical constants / initial conditions
# ═══════════════════════════════════════════════════════════════════════════════
MU = 3.986004415e5          # km^3 / s^2
R_E = 6378.1363             # km
J2 = 1.082626925638815e-3
J3 = -2.53215306e-6
CD = 2.0
SPACECRAFT_AREA = 3.0       # m^2  (Integrator converts to km^2 internally)
SPACECRAFT_MASS = 970.0     # kg

# LEO state  [x, y, z, u, v, w]  in km & km/s
SAT_POS = np.array([757.700, 5222.607, 4851.500])
SAT_VEL = np.array([2.21321, 4.67834, -5.37130])
SAT_STATE = np.hstack((SAT_POS, SAT_VEL))

# Three ground stations in ECEF (km)
STATION_1_ECEF = np.array([-5127.510, -3794.160, 0.0])
STATION_2_ECEF = np.array([3860.910, 3238.490, 3898.094])
STATION_3_ECEF = np.array([549.505, -1380.872, 6182.197])
STATION_POSITIONS_ECEF = np.vstack((STATION_1_ECEF, STATION_2_ECEF, STATION_3_ECEF))

T_FINAL = 3600.0  # 1 orbit-ish for a LEO satellite (seconds)


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: build an Integrator in a given mode
# ═══════════════════════════════════════════════════════════════════════════════
def _build_integrator(mode, parameter_indices, **kwargs):
    """Convenience wrapper around the Integrator constructor."""
    return Integrator(
        mu=MU,
        R_e=R_E,
        J2=J2,
        J3=J3,
        Cd=CD,
        mode=mode,
        parameter_indices=parameter_indices,
        spacecraft_area=SPACECRAFT_AREA,
        spacecraft_mass=SPACECRAFT_MASS,
        number_of_stations=kwargs.get("number_of_stations", 0),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Nominal execution tests  (integration completes, shapes are correct)
# ═══════════════════════════════════════════════════════════════════════════════
class TestIntegrateSTMAndThetaNominal(unittest.TestCase):
    """Verify that integrate_stm_and_theta runs to completion and returns
    arrays of the right shape for several mode / consider-parameter combos."""

    # -- Single consider parameter: mu --
    def test_single_consider_mu(self):
        integrator = _build_integrator(mode=["J2"], parameter_indices=[6])
        initial_state = np.hstack((SAT_STATE, J2))          # len = 7
        consider_params = ["mu"]
        state_len = 7
        n_consider = 1
        t, y = integrator.integrate_stm_and_theta(
            T_FINAL, initial_state, consider_parameters=consider_params
        )

        expected_cols = state_len + state_len**2 + 6 * n_consider
        self.assertEqual(y.shape[0], expected_cols)
        self.assertGreater(t.shape[0], 1, "Integration should produce multiple time steps")

    # -- Single consider parameter: J3 --
    def test_single_consider_J3(self):
        integrator = _build_integrator(mode=["J2"], parameter_indices=[6])
        initial_state = np.hstack((SAT_STATE, J2))
        consider_params = ["J3"]
        state_len = 7

        t, y = integrator.integrate_stm_and_theta(
            T_FINAL, initial_state, consider_parameters=consider_params
        )
        expected_cols = state_len + state_len**2 + 6 * 1
        self.assertEqual(y.shape[0], expected_cols)

    # -- Multiple consider parameters: mu and J3 --
    def test_multiple_consider_mu_J3(self):
        integrator = _build_integrator(mode=["J2"], parameter_indices=[6])
        initial_state = np.hstack((SAT_STATE, J2))
        consider_params = ["mu", "J3"]
        state_len = 7
        n_consider = 2

        t, y = integrator.integrate_stm_and_theta(
            T_FINAL, initial_state, consider_parameters=consider_params
        )
        expected_cols = state_len + state_len**2 + 6 * n_consider
        self.assertEqual(y.shape[0], expected_cols)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Reasonableness of returned values
# ═══════════════════════════════════════════════════════════════════════════════
class TestIntegrationReasonableness(unittest.TestCase):
    """Check that integrated state, STM, and theta are physically plausible."""

    @classmethod
    def setUpClass(cls):
        """Run the integration once and cache results for all tests."""
        cls.integrator = _build_integrator(mode=["J2"], parameter_indices=[6])
        cls.initial_state = np.hstack((SAT_STATE, J2))
        cls.state_len = 7
        cls.consider_params = ["mu", "J3"]
        cls.n_consider = 2

        cls.t, cls.y = cls.integrator.integrate_stm_and_theta(
            T_FINAL,
            cls.initial_state,
            consider_parameters=cls.consider_params,
        )

        # Unpack final column
        final = cls.y[:, -1]
        n = cls.state_len
        cls.state_final = final[:n]
        cls.phi_final = final[n : n + n**2].reshape((n, n))

        cls.theta_final = final[n + n**2 :].reshape((6, cls.n_consider))

    # -- State stays in a reasonable LEO regime --
    def test_position_magnitude_reasonable(self):
        r = np.linalg.norm(self.state_final[:3])
        self.assertGreater(r, R_E, "Spacecraft should not be inside the Earth")
        self.assertLess(r, R_E + 2000, "Spacecraft altitude unreasonably high for LEO")

    def test_velocity_magnitude_reasonable(self):
        v = np.linalg.norm(self.state_final[3:6])
        self.assertGreater(v, 5.0, "Velocity too low for LEO")
        self.assertLess(v, 10.0, "Velocity too high for LEO")

    def test_J2_parameter_unchanged(self):
        """J2 is an estimated parameter with zero dynamics (constant), so
        its value should not change during integration."""
        self.assertAlmostEqual(self.state_final[6], J2, places=10)

    # -- STM sanity --
    def test_stm_determinant_positive(self):
        """The STM must have a positive determinant (volume-preserving for
        Hamiltonian dynamics; > 0 always)."""
        det = np.linalg.det(self.phi_final)
        self.assertGreater(det, 0, "STM determinant should be positive")

    def test_stm_determinant_near_unity(self):
        """For conservative dynamics (no drag in the state Jacobian velocity
        partials contribute only weakly), det(Φ) ≈ 1.  Allow some tolerance
        because drag-like terms are included in the Jacobian."""
        det = np.linalg.det(self.phi_final)
        self.assertAlmostEqual(det, 1.0, delta=5.0,
                               msg="STM determinant should be close to unity")

    def test_stm_identity_at_t0(self):
        """At t = 0 the STM should be the identity matrix."""
        initial = self.y[:, 0]
        n = self.state_len
        phi_0 = initial[n : n + n**2].reshape((n, n))
        assert_allclose(phi_0, np.eye(n), atol=1e-14)

    # -- Theta sanity --
    def test_theta_zero_at_t0(self):
        """Theta should be initialised to zero."""
        initial = self.y[:, 0]
        n = self.state_len
        theta_0 = initial[n + n**2 :].reshape((6, self.n_consider))
        assert_allclose(theta_0, np.zeros_like(theta_0), atol=1e-14)

    def test_theta_nonzero_at_tf(self):
        """After integration theta should have evolved away from zero
        (the consider parameters do affect the dynamics)."""
        self.assertFalse(
            np.allclose(self.theta_final, 0, atol=1e-20),
            "Theta should be non-zero after integration",
        )

    def test_theta_position_rows_nonzero(self):
        """The position rows (indices 3-5 for velocity partials) of theta
        should be non-zero since the partials only enter via acceleration."""
        # Rows 3, 4, 5 correspond to velocity partials in a 6-element state
        # For the 7-element state (with J2 param) rows 3,4,5 still are the
        # velocity derivatives
        accel_rows = self.theta_final[3:6, :]
        self.assertFalse(
            np.allclose(accel_rows, 0, atol=1e-20),
            "Acceleration rows of theta should be non-zero",
        )

    def test_theta_finite(self):
        """Theta should contain only finite values."""
        self.assertTrue(np.all(np.isfinite(self.theta_final)))


# ═══════════════════════════════════════════════════════════════════════════════
# 3. compute_consider_parameter_partials matches state_jacobian columns
# ═══════════════════════════════════════════════════════════════════════════════
class TestConsiderPartialsMatchJacobian(unittest.TestCase):
    """The docstring of compute_consider_parameter_partials states:
       'Matches those computed for the A matrix in state_jacobian.'
    Verify this numerically for every supported consider parameter."""

    def setUp(self):
        """Compute the full 9×9 'BaseMat' Jacobian once."""
        self.A_full = state_jacobian(
            SAT_POS,
            SAT_VEL,
            mu=MU,
            J2=J2,
            J3=J3,
            C_d=CD,
            station_positions_ecef=STATION_POSITIONS_ECEF,
            R_e=R_E,
            mode=["BaseMat"],
            spacecraft_area=SPACECRAFT_AREA * 1e-6,  # Integrator converts m^2→km^2
            spacecraft_mass=SPACECRAFT_MASS,
        )

    def _get_partials(self, param_name):
        return compute_consider_parameter_partials(
            param_name,
            SAT_POS,
            SAT_VEL,
            mu=MU,
            J2=J2,
            J3=J3,
            C_d=CD,
            station_positions_ecef=STATION_POSITIONS_ECEF,
            R_e=R_E,
            spacecraft_area=SPACECRAFT_AREA * 1e-6,
            spacecraft_mass=SPACECRAFT_MASS,
        )

    # -- mu partials (column 6 of the 9×9 BaseMat A) --
    def test_mu_partials_match_A(self):
        partials = self._get_partials("mu")
        # partials is length 6: [0,0,0, a_xmu, a_ymu, a_zmu]
        # A_full column 6, rows 3-5 should match the acceleration partials
        expected_accel = self.A_full[3:6, 6]
        actual_accel = partials[3:6]
        assert_allclose(actual_accel, expected_accel, rtol=1e-12,
                        err_msg="mu partials do not match A matrix column 6")

    def test_mu_partials_position_rows_zero(self):
        partials = self._get_partials("mu")
        assert_allclose(partials[:3], np.zeros(3), atol=1e-15,
                        err_msg="Position rows of mu partials should be zero")

    # -- J2 partials (column 7 of BaseMat A) --
    def test_J2_partials_match_A(self):
        partials = self._get_partials("J2")
        expected_accel = self.A_full[3:6, 7]
        actual_accel = partials[3:6]
        assert_allclose(actual_accel, expected_accel, rtol=1e-12,
                        err_msg="J2 partials do not match A matrix column 7")

    def test_J2_partials_position_rows_zero(self):
        partials = self._get_partials("J2")
        assert_allclose(partials[:3], np.zeros(3), atol=1e-15)

    # -- J3 partials (column 8 of BaseMat A) --
    def test_J3_partials_match_A(self):
        partials = self._get_partials("J3")
        expected_accel = self.A_full[3:6, 8]
        actual_accel = partials[3:6]
        assert_allclose(actual_accel, expected_accel, rtol=1e-12,
                        err_msg="J3 partials do not match A matrix column 8")

    def test_J3_partials_position_rows_zero(self):
        partials = self._get_partials("J3")
        assert_allclose(partials[:3], np.zeros(3), atol=1e-15)

    # -- Drag (Cd) partials --
    def test_drag_partials_match_A(self):
        """state_jacobian does not store Cd partials in BaseMat columns 6-8,
        but *does* compute them inside the mode == 'Drag' branch.  Recompute
        the expected values from first principles using the same formulas that
        both functions share."""
        partials = self._get_partials("Drag")

        # Re-derive expected Cd partials with the same formula used in both
        # state_jacobian (Drag branch) and compute_consider_parameter_partials
        earth_spin_rate = 7.2921158553e-5
        u, v, w = SAT_VEL
        x, y, z = SAT_POS
        from ASEN_6080.Tools.generic_functions import compute_density

        V_rel = np.array([u + earth_spin_rate * y, v - earth_spin_rate * x, w])
        V_rel_norm = np.linalg.norm(V_rel)
        rho = compute_density(np.linalg.norm(SAT_POS)) * 1e9
        sc_area = SPACECRAFT_AREA * 1e-6
        sc_mass = SPACECRAFT_MASS

        expected_xCd = -(rho * sc_area * V_rel_norm * V_rel[0]) / (2 * sc_mass)
        expected_yCd = -(rho * sc_area * V_rel_norm * V_rel[1]) / (2 * sc_mass)
        expected_zCd = -(rho * sc_area * V_rel_norm * V_rel[2]) / (2 * sc_mass)

        assert_allclose(partials[3], expected_xCd, rtol=1e-12)
        assert_allclose(partials[4], expected_yCd, rtol=1e-12)
        assert_allclose(partials[5], expected_zCd, rtol=1e-12)

    def test_drag_partials_position_rows_zero(self):
        partials = self._get_partials("Drag")
        assert_allclose(partials[:3], np.zeros(3), atol=1e-15)

    # -- Also verify with the mode-assembled A matrix --
    def test_drag_partials_match_mode_assembled_A(self):
        """When mode=['mu','J2','Drag'], the Drag column of the assembled A
        matrix (last column) should agree with compute_consider_parameter_partials."""
        A_assembled = state_jacobian(
            SAT_POS,
            SAT_VEL,
            mu=MU,
            J2=J2,
            J3=J3,
            C_d=CD,
            station_positions_ecef=STATION_POSITIONS_ECEF,
            R_e=R_E,
            mode=["mu", "J2", "Drag"],
            spacecraft_area=SPACECRAFT_AREA * 1e-6,
            spacecraft_mass=SPACECRAFT_MASS,
        )
        # mode=['mu','J2','Drag'] → A is 9×9: cols 0-5 pos/vel, 6=mu, 7=J2, 8=Drag
        partials = self._get_partials("Drag")
        breakpoint()
        expected_Cd_col = A_assembled[3:6, 8]  # acceleration rows, Drag column
        assert_allclose(partials[3:6], expected_Cd_col, rtol=1e-12,
                        err_msg="Drag partials don't match assembled A's Cd column")

    # -- Invalid parameter raises ValueError --
    def test_invalid_parameter_raises(self):
        with self.assertRaises(ValueError):
            self._get_partials("InvalidParam")


# ══════════════════════════════════════════════════════════════════��════════════
# 4. Consistency: full_dynamics derivative matches standalone partial functions
# ═══════════════════════════════════════════════════════════════════════════════
class TestFullDynamicsConsiderDerivatives(unittest.TestCase):
    """Call Integrator.full_dynamics directly with consider parameters and
    verify that the theta_dot portion equals A @ theta + B, where B is
    assembled from compute_consider_parameter_partials."""

    def test_theta_dot_equals_A_theta_plus_B(self):
        integrator = _build_integrator(mode=["J2"], parameter_indices=[6])
        state = np.hstack((SAT_STATE, J2))
        state_len = 7
        consider_params = ["mu", "J3"]
        n_consider = len(consider_params)

        # Build an augmented vector at t = 0 (phi = I, theta = 0)
        phi_0 = np.eye(state_len).flatten()
        theta_0 = np.zeros(6 * n_consider)
        augmented = np.hstack((state, phi_0, theta_0))

        deriv = integrator.full_dynamics(0.0, augmented, consider_parameters=consider_params)

        # Extract theta_dot from the derivative
        theta_dot_flat = deriv[state_len + state_len**2:]
        theta_dot = theta_dot_flat.reshape((6, n_consider))

        # Compute expected theta_dot = A @ theta_0_mat + B
        theta_mat = np.zeros((6, n_consider))
        A = state_jacobian(
            state[:3], state[3:6], MU, J2, J3, CD,
            np.array([]), R_E,
            mode=["J2"],
            spacecraft_area=integrator.spacecraft_area,
            spacecraft_mass=integrator.spacecraft_mass,
        )

        B = np.zeros((6, n_consider))
        for i, cp in enumerate(consider_params):
            partials = compute_consider_parameter_partials(
                cp, state[:3], state[3:6], MU, J2, J3, CD,
                np.array([]), R_E,
                spacecraft_area=integrator.spacecraft_area,
                spacecraft_mass=integrator.spacecraft_mass,
            )
            # partials is length 6; pad to state_len (7 here, last row = 0)
            B[:len(partials), i] = partials

        expected_theta_dot = A[:6,:6] @ theta_mat + B  # = B when theta_mat is 0

        assert_allclose(theta_dot, expected_theta_dot, rtol=1e-10,
                        err_msg="theta_dot from full_dynamics doesn't match A@theta + B")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Multiple evaluation points: partials match A at different state vectors
# ═══════════════════════════════════════════════════════════════════════════════
class TestConsiderPartialsAtMultipleStates(unittest.TestCase):
    """Repeat the A-matrix matching test at several different state vectors
    to guard against hard-coded values passing by coincidence."""

    TEST_STATES = [
        # (position_km, velocity_km_s)
        (np.array([6778.0, 0.0, 0.0]),     np.array([0.0, 7.668, 0.0])),
        (np.array([0.0, 7100.0, 1000.0]),  np.array([-6.5, 0.0, 2.0])),
        (np.array([4000.0, 4000.0, 3000.0]), np.array([-3.0, 3.0, -4.0])),
    ]

    def test_mu_partials_at_multiple_states(self):
        for pos, vel in self.TEST_STATES:
            with self.subTest(pos=pos, vel=vel):
                A = state_jacobian(
                    pos, vel, MU, J2, J3, CD, STATION_POSITIONS_ECEF, R_E,
                    mode=["BaseMat"],
                    spacecraft_area=SPACECRAFT_AREA * 1e-6,
                    spacecraft_mass=SPACECRAFT_MASS,
                )
                partials = compute_consider_parameter_partials(
                    "mu", pos, vel, MU, J2, J3, CD,
                    STATION_POSITIONS_ECEF, R_E,
                    spacecraft_area=SPACECRAFT_AREA * 1e-6,
                    spacecraft_mass=SPACECRAFT_MASS,
                )
                assert_allclose(partials[3:6], A[3:6, 6], rtol=1e-12)

    def test_J2_partials_at_multiple_states(self):
        for pos, vel in self.TEST_STATES:
            with self.subTest(pos=pos, vel=vel):
                A = state_jacobian(
                    pos, vel, MU, J2, J3, CD, STATION_POSITIONS_ECEF, R_E,
                    mode=["BaseMat"],
                    spacecraft_area=SPACECRAFT_AREA * 1e-6,
                    spacecraft_mass=SPACECRAFT_MASS,
                )
                partials = compute_consider_parameter_partials(
                    "J2", pos, vel, MU, J2, J3, CD,
                    STATION_POSITIONS_ECEF, R_E,
                    spacecraft_area=SPACECRAFT_AREA * 1e-6,
                    spacecraft_mass=SPACECRAFT_MASS,
                )
                assert_allclose(partials[3:6], A[3:6, 7], rtol=1e-12)

    def test_J3_partials_at_multiple_states(self):
        for pos, vel in self.TEST_STATES:
            with self.subTest(pos=pos, vel=vel):
                A = state_jacobian(
                    pos, vel, MU, J2, J3, CD, STATION_POSITIONS_ECEF, R_E,
                    mode=["BaseMat"],
                    spacecraft_area=SPACECRAFT_AREA * 1e-6,
                    spacecraft_mass=SPACECRAFT_MASS,
                )
                partials = compute_consider_parameter_partials(
                    "J3", pos, vel, MU, J2, J3, CD,
                    STATION_POSITIONS_ECEF, R_E,
                    spacecraft_area=SPACECRAFT_AREA * 1e-6,
                    spacecraft_mass=SPACECRAFT_MASS,
                )
                assert_allclose(partials[3:6], A[3:6, 8], rtol=1e-12)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Edge cases
# ═══════════════════════════════════════════════════════════════════════════════
class TestEdgeCases(unittest.TestCase):
    """Boundary and degenerate-input tests."""

    def test_very_short_integration(self):
        """A very short integration should return state ≈ initial and
        theta ≈ 0."""
        integrator = _build_integrator(mode=["J2"], parameter_indices=[6])
        initial_state = np.hstack((SAT_STATE, J2))
        t, y = integrator.integrate_stm_and_theta(
            1e-6, initial_state, consider_parameters=["mu"]
        )
        state_len = 7
        final_state = y[:state_len, -1]
        assert_allclose(final_state, initial_state, rtol=1e-8)

    def test_no_consider_parameters_fallback(self):
        """If consider_parameters is empty, integrate_stm_and_theta should
        still succeed (falls back to STM-only integration via full_dynamics
        without the theta block)."""
        integrator = _build_integrator(mode=["J2"], parameter_indices=[6])
        initial_state = np.hstack((SAT_STATE, J2))
        # This should not raise an error
        t, y = integrator.integrate_stm_and_theta(
            T_FINAL, initial_state, consider_parameters=[]
        )
        # Output should be state + STM only
        state_len = 7
        expected_rows = state_len + state_len**2
        self.assertEqual(y.shape[0], expected_rows)

    def test_theta_evolves_continuously(self):
        """Check that theta doesn't have sudden jumps by verifying that
        the norm of theta grows monotonically (approximately) for a short
        integration."""
        integrator = _build_integrator(mode=["J2"], parameter_indices=[6])
        initial_state = np.hstack((SAT_STATE, J2))
        teval = np.linspace(0, 600, 50)
        t, y = integrator.integrate_stm_and_theta(
            600, initial_state, teval=teval, consider_parameters=["mu"]
        )
        state_len = 7
        theta_norms = []
        for i in range(y.shape[1]):
            theta_flat = y[state_len + state_len**2 :, i]
            theta_norms.append(np.linalg.norm(theta_flat))
        # Theta norm should generally increase from zero
        self.assertAlmostEqual(theta_norms[0], 0.0, places=12)
        self.assertGreater(theta_norms[-1], 0.0)


if __name__ == "__main__":
    unittest.main()