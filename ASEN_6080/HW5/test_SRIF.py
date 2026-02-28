"""
Test suite for the Square Root Information Filter (SRIF) class and its associated methods.

Tests cover:
    1. SRIF.__init__         - Constructor / attribute assignment
    2. SRIF.whiten_measurements - Measurement whitening via Cholesky decomposition
    3. SRIF.householder_transform - Householder QR triangularisation
    4. SRIF.time_update      - SRIF prediction / time-update step
    5. SRIF.measurement_update - SRIF measurement-update step
    6. SRIF.run              - Full filter execution on synthetic data

Dependencies are mocked where appropriate so that the mathematical core of
the SRIF can be verified in isolation without needing real orbit propagation.
"""

import unittest
from unittest.mock import MagicMock, patch, PropertyMock
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from scipy.linalg import cholesky, solve_triangular


# ---------------------------------------------------------------------------
# Import the class under test
# ---------------------------------------------------------------------------
from ASEN_6080.Tools.SRIF import SRIF
from ASEN_6080.Tools import Integrator, MeasurementMgr, CoordinateMgr


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _make_mock_integrator(mode=None, R_e=6378.0, number_of_stations=0):
    """Return a lightweight mock Integrator with the attributes SRIF needs."""
    integrator = MagicMock(spec=Integrator)
    integrator.R_e = R_e
    integrator.mode = mode if mode is not None else ['mu', 'J2']
    integrator.number_of_stations = number_of_stations
    return integrator


def _make_mock_measurement_mgr(station_name="station_1"):
    """Return a lightweight mock MeasurementMgr."""
    mgr = MagicMock(spec=MeasurementMgr)
    mgr.station_name = station_name
    mgr.station_state_ecef = np.array([6378.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    mgr.lat = 0.0
    mgr.lon = 0.0
    return mgr


def _build_srif(mode=None, num_stations=0, station_names=None):
    """Construct an SRIF instance with mocked dependencies."""
    integrator = _make_mock_integrator(mode=mode, number_of_stations=num_stations)

    if station_names is None:
        station_names = ["station_1"]
    mgr_list = [_make_mock_measurement_mgr(name) for name in station_names]

    srif = SRIF(
        integrator=integrator,
        measurement_mgr_list=mgr_list,
        initial_earth_spin_angle=0.0,
        earth_rotation_rate=2 * np.pi / 86164,
    )
    return srif


# ===================================================================
# 1.  __init__  Tests
# ===================================================================
class TestSRIFInit(unittest.TestCase):
    """Verify that the constructor stores all attributes correctly."""

    def test_attributes_stored(self):
        integrator = _make_mock_integrator()
        mgr1 = _make_mock_measurement_mgr("s1")
        mgr2 = _make_mock_measurement_mgr("s2")

        srif = SRIF(integrator, [mgr1, mgr2], initial_earth_spin_angle=0.5)

        self.assertIs(srif.integrator, integrator)
        self.assertEqual(len(srif.measurement_mgrs), 2)
        self.assertIsInstance(srif.coordinate_mgr, CoordinateMgr)

    def test_measurement_mgr_list_is_copied(self):
        """Mutating the original list must not affect the SRIF's copy."""
        mgr = _make_mock_measurement_mgr()
        original_list = [mgr]
        srif = SRIF(_make_mock_integrator(), original_list, 0.0)
        original_list.append(_make_mock_measurement_mgr("extra"))
        self.assertEqual(len(srif.measurement_mgrs), 1)

    def test_coordinate_mgr_uses_integrator_R_e(self):
        integrator = _make_mock_integrator(R_e=1234.0)
        srif = SRIF(integrator, [_make_mock_measurement_mgr()], 0.0)
        self.assertEqual(srif.coordinate_mgr.earth_radius, 1234.0)


# ===================================================================
# 2.  whiten_measurements  Tests
# ===================================================================
class TestWhitenMeasurements(unittest.TestCase):
    """Verify the Cholesky-based whitening of measurements and Jacobian."""

    def setUp(self):
        self.srif = _build_srif()

    def test_identity_covariance_is_noop(self):
        """With R = I the whitened values should be unchanged."""
        n = 4
        y = np.random.randn(n)
        H = np.random.randn(n, 6)
        R_cov = np.eye(n)

        y_w, H_w = self.srif.whiten_measurements(y, H, R_cov)

        assert_allclose(y_w, y, atol=1e-12)
        # Note: the code transposes after solve_triangular on H
        # so H_whitened = solve_triangular(V, H).T
        # With V=I this gives H.T  (the code has a transpose)
        assert_allclose(H_w, H, atol=1e-12)

    def test_diagonal_covariance(self):
        """With R = diag(σ²), whitened y_i = y_i / σ_i."""
        sigmas = np.array([2.0, 3.0])
        R_cov = np.diag(sigmas ** 2)
        y = np.array([4.0, 9.0])
        H = np.eye(2, 6)

        y_w, H_w = self.srif.whiten_measurements(y, H, R_cov)

        # V = cholesky(R_cov) = diag(sigmas)
        # y_w = V^{-1} y => y / sigmas
        assert_allclose(y_w, y / sigmas, atol=1e-12)

    def test_output_shapes(self):
        """Output dimensions must be consistent with inputs."""
        m, n = 4, 8
        y = np.random.randn(m)
        H = np.random.randn(m, n)
        R_cov = np.eye(m) * 5.0

        y_w, H_w = self.srif.whiten_measurements(y, H, R_cov)
        self.assertEqual(y_w.shape, (m,))
        # H_whitened is transposed: shape (n, m)
        self.assertEqual(H_w.shape, (m, n))
        

    def test_non_diagonal_covariance(self):
        """Whitening with a non-trivial SPD covariance matrix."""
        R_cov = np.array([[4.0, 1.0],
                          [1.0, 3.0]])
        y = np.array([1.0, 2.0])
        H = np.eye(2, 6)

        V = cholesky(R_cov)
        expected_y = solve_triangular(V, y)

        y_w, _ = self.srif.whiten_measurements(y, H, R_cov)
        assert_allclose(y_w, expected_y, atol=1e-12)


# ===================================================================
# 3.  householder_transform  Tests
# ===================================================================
class TestHouseholderTransform(unittest.TestCase):
    """Verify the Householder QR triangularisation."""

    def setUp(self):
        self.srif = _build_srif()

    def _build_A(self, R_bar, b_bar, H, y):
        """Manually assemble the augmented [R_bar b_bar; H y] matrix used by the SRIF."""
        n = R_bar.shape[0]
        m = H.shape[0]
        A = np.zeros((n + m, n + 1))
        A[:n, :n] = R_bar
        A[:n, -1] = b_bar
        A[n:, :n] = H
        A[n:, -1] = y
        return A

    def test_upper_triangular_result(self):
        """After transformation the top n×n block must be upper-triangular."""
        np.random.seed(42)
        n = 4
        R_bar = np.eye(n)
        b_bar = np.random.randn(n)
        H = np.random.randn(3, n)
        y = np.random.randn(3)

        A = self._build_A(R_bar, b_bar, H, y)
        T = self.srif.householder_transform(A.copy())

        # The top n rows, first n cols should be upper triangular
        R_top = T[:n, :n]
        for i in range(1, n):
            for j in range(i):
                self.assertAlmostEqual(R_top[i, j], 0.0, places=12,
                                       msg=f"Element ({i},{j}) should be zero")

    def test_sub_diagonal_zeroed(self):
        """Elements below the diagonal in columns 0..n-1 must be zero."""
        np.random.seed(7)
        n = 3
        m = 2
        A = np.random.randn(n + m, n + 1)
        T = self.srif.householder_transform(A.copy())

        for k in range(n):
            for i in range(k + 1, n + m):
                self.assertAlmostEqual(T[i, k], 0.0, places=12)

    def test_preserves_column_norms(self):
        """Householder reflections are orthogonal → column norms are preserved
        for the first n columns (up to sign/rotation)."""
        np.random.seed(99)
        n, m = 4, 3
        A = np.random.randn(n + m, n + 1)
        A_orig = A.copy()
        T = self.srif.householder_transform(A)

        for k in range(n):
            orig_norm = np.linalg.norm(A_orig[:, k])
            trans_norm = np.linalg.norm(T[:, k])
            self.assertAlmostEqual(orig_norm, trans_norm, places=10,
                                   msg=f"Column {k} norm changed")

    def test_square_matrix(self):
        """Edge case: m = 0 (square A, only n rows)."""
        np.random.seed(123)
        n = 3
        A = np.random.randn(n, n + 1)
        T = self.srif.householder_transform(A.copy())
        # Should still produce an upper-triangular first n×n block
        for i in range(n):
            for j in range(i):
                self.assertAlmostEqual(T[i, j], 0.0, places=12)


# ===================================================================
# 4.  time_update  Tests
# ===================================================================
class TestTimeUpdate(unittest.TestCase):
    """Verify the SRIF time-update (prediction) step."""

    def setUp(self):
        self.srif = _build_srif()

    def test_identity_stm(self):
        """With phi = I the state and info-matrix should be unchanged."""
        n = 4
        x_hat = np.random.randn(n)
        R = np.eye(n) * 3.0
        phi = np.eye(n)

        x_bar, R_bar, b_bar = self.srif.time_update(x_hat, R, phi)

        assert_allclose(x_bar, x_hat, atol=1e-14)
        assert_allclose(R_bar, R, atol=1e-14)
        assert_allclose(b_bar, R @ x_hat, atol=1e-14)

    def test_scaling_stm(self):
        """phi = 2*I  →  x_bar = 2*x_hat, R_bar = R/2."""
        n = 3
        x_hat = np.array([1.0, 2.0, 3.0])
        R = np.eye(n) * 4.0
        phi = 2.0 * np.eye(n)

        x_bar, R_bar, b_bar = self.srif.time_update(x_hat, R, phi)

        assert_allclose(x_bar, 2.0 * x_hat, atol=1e-14)
        assert_allclose(R_bar, R @ np.linalg.inv(phi), atol=1e-14)
        assert_allclose(b_bar, R_bar @ x_bar, atol=1e-14)

    def test_general_stm(self):
        """Verify against manual formula: x_bar=phi@x, R_bar=R@inv(phi), b_bar=R_bar@x_bar."""
        np.random.seed(0)
        n = 5
        phi = np.random.randn(n, n)
        phi = phi + 3 * np.eye(n)  # make invertible
        x_hat = np.random.randn(n)
        R = np.random.randn(n, n)
        R = R.T @ R + np.eye(n)  # SPD

        x_bar, R_bar, b_bar = self.srif.time_update(x_hat, R, phi)

        assert_allclose(x_bar, phi @ x_hat, atol=1e-10)
        assert_allclose(R_bar, R @ np.linalg.inv(phi), atol=1e-10)
        assert_allclose(b_bar, R_bar @ x_bar, atol=1e-10)

    def test_output_shapes(self):
        n = 6
        x_hat = np.zeros(n)
        R = np.eye(n)
        phi = np.eye(n)

        x_bar, R_bar, b_bar = self.srif.time_update(x_hat, R, phi)

        self.assertEqual(x_bar.shape, (n,))
        self.assertEqual(R_bar.shape, (n, n))
        self.assertEqual(b_bar.shape, (n,))


# ===================================================================
# 5.  measurement_update  Tests
# ===================================================================
class TestMeasurementUpdate(unittest.TestCase):
    """Verify the SRIF measurement-update step."""

    def setUp(self):
        self.srif = _build_srif()

    def test_output_shapes(self):
        n = 4
        m = 2
        R_bar = np.eye(n) * 2.0
        b_bar = np.random.randn(n)
        H = np.random.randn(m, n)
        y = np.random.randn(m)

        x_hat_new, R_new, b_new = self.srif.measurement_update(R_bar, b_bar, H, y)

        self.assertEqual(x_hat_new.shape, (n,))
        self.assertEqual(R_new.shape, (n, n))
        self.assertEqual(b_new.shape, (n,))

    def test_updated_R_is_upper_triangular(self):
        """R_new (the information matrix) should be upper triangular."""
        np.random.seed(10)
        n = 4
        m = 2
        R_bar = np.triu(np.random.randn(n, n))
        np.fill_diagonal(R_bar, np.abs(np.diag(R_bar)) + 1)
        b_bar = np.random.randn(n)
        H = np.random.randn(m, n)
        y = np.random.randn(m)

        _, R_new, _ = self.srif.measurement_update(R_bar, b_bar, H, y)

        for i in range(n):
            for j in range(i):
                self.assertAlmostEqual(R_new[i, j], 0.0, places=10,
                                       msg=f"R_new[{i},{j}] should be zero")

    def test_back_substitution_consistency(self):
        """x_hat_new should satisfy R_new @ x_hat_new = b_new."""
        np.random.seed(42)
        n = 3
        m = 2
        R_bar = np.triu(np.random.randn(n, n))
        np.fill_diagonal(R_bar, np.abs(np.diag(R_bar)) + 1)
        b_bar = np.random.randn(n)
        H = np.random.randn(m, n)
        y = np.random.randn(m)

        x_hat_new, R_new, b_new = self.srif.measurement_update(R_bar, b_bar, H, y)

        assert_allclose(R_new @ x_hat_new, b_new, atol=1e-10)

    def test_zero_measurement_jacobian(self):
        """With H = 0 the update should essentially return the prediction."""
        n = 3
        R_bar = np.eye(n) * 5.0
        b_bar = np.array([1.0, 2.0, 3.0])
        H = np.zeros((2, n))
        y = np.zeros(2)

        x_hat_new, R_new, b_new = self.srif.measurement_update(R_bar, b_bar, H, y)

        # With zero H, the Householder should not alter the top block
        expected_x = solve_triangular(R_bar, b_bar)
        assert_allclose(x_hat_new, expected_x, atol=1e-10)

    def test_single_measurement(self):
        """Measurement update with a single scalar measurement (m=1)."""
        np.random.seed(55)
        n = 3
        R_bar = np.triu(np.random.randn(n, n))
        np.fill_diagonal(R_bar, np.abs(np.diag(R_bar)) + 2)
        b_bar = np.random.randn(n)
        H = np.random.randn(1, n)
        y = np.random.randn(1)

        x_hat_new, R_new, b_new = self.srif.measurement_update(R_bar, b_bar, H, y)

        self.assertEqual(x_hat_new.shape, (n,))
        assert_allclose(R_new @ x_hat_new, b_new, atol=1e-10)


# ===================================================================
# 6.  Integrated  time_update + measurement_update  Tests
# ===================================================================
class TestTimeAndMeasurementUpdateCombined(unittest.TestCase):
    """Test the time update followed by a measurement update to ensure
    the information-form algebra is self-consistent with the covariance-form."""

    def setUp(self):
        self.srif = _build_srif()

    def test_covariance_recovery(self):
        """P = inv(R.T @ R) after an update should be SPD and well-conditioned."""
        np.random.seed(21)
        n = 4

        # Initial information matrix (upper-triangular, positive diagonal)
        R_info = np.triu(np.random.randn(n, n))
        np.fill_diagonal(R_info, np.abs(np.diag(R_info)) + 2)

        x_hat = np.random.randn(n)
        phi = np.eye(n) + 0.01 * np.random.randn(n, n)

        # Time update
        x_bar, R_bar, b_bar = self.srif.time_update(x_hat, R_info, phi)

        # Measurement update
        m = 2
        H = np.random.randn(m, n)
        y = np.random.randn(m)
        x_new, R_new, b_new = self.srif.measurement_update(R_bar, b_bar, H, y)

        P = np.linalg.inv(R_new.T @ R_new)

        # P should be symmetric positive definite
        assert_allclose(P, P.T, atol=1e-10)
        eigvals = np.linalg.eigvalsh(P)
        self.assertTrue(np.all(eigvals > 0), "Recovered covariance is not SPD")

    def test_information_increases_with_measurements(self):
        """After a measurement update the trace of R.T@R should increase
        (more information → larger information matrix)."""
        np.random.seed(77)
        n = 4

        R_info = np.eye(n) * 2.0
        x_hat = np.zeros(n)
        phi = np.eye(n)

        _, R_bar, b_bar = self.srif.time_update(x_hat, R_info, phi)
        trace_before = np.trace(R_bar.T @ R_bar)

        H = np.random.randn(2, n)
        y = np.random.randn(2)
        _, R_new, _ = self.srif.measurement_update(R_bar, b_bar, H, y)
        trace_after = np.trace(R_new.T @ R_new)

        self.assertGreater(trace_after, trace_before)


# ===================================================================
# 7.  Householder Numerical Properties
# ===================================================================
class TestHouseholderNumerical(unittest.TestCase):
    """Additional numerical checks for the Householder transformation."""

    def setUp(self):
        self.srif = _build_srif()

    def test_deterministic(self):
        """Running twice on identical input should give identical output."""
        np.random.seed(3)
        A = np.random.randn(6, 4)
        T1 = self.srif.householder_transform(A.copy())
        T2 = self.srif.householder_transform(A.copy())
        assert_allclose(T1, T2, atol=1e-15)

    def test_large_matrix(self):
        """Smoke test with a larger matrix to check no index-out-of-range."""
        np.random.seed(8)
        n, m = 10, 5
        A = np.random.randn(n + m, n + 1)
        T = self.srif.householder_transform(A.copy())
        # All sub-diagonal elements in first n columns must be zero
        for k in range(n):
            for i in range(k + 1, n + m):
                self.assertAlmostEqual(T[i, k], 0.0, places=10)

    def test_zero_column_handling(self):
        """A column of all zeros should not cause a crash (sigma==0 guard)."""
        A = np.zeros((5, 4))
        A[0, 3] = 1.0  # non-trivial last column so it isn't entirely empty
        T = self.srif.householder_transform(A.copy())
        # Should run without error
        self.assertEqual(T.shape, A.shape)


# ===================================================================
# 8.  Edge Cases
# ===================================================================
class TestEdgeCases(unittest.TestCase):
    """Boundary / edge-case tests."""

    def setUp(self):
        self.srif = _build_srif()

    def test_time_update_zero_state(self):
        """Zero state should propagate to zero regardless of phi."""
        n = 4
        x_hat = np.zeros(n)
        R = np.eye(n)
        phi = np.random.randn(n, n) + 3 * np.eye(n)

        x_bar, R_bar, b_bar = self.srif.time_update(x_hat, R, phi)

        assert_allclose(x_bar, np.zeros(n), atol=1e-15)
        assert_allclose(b_bar, np.zeros(n), atol=1e-14)

    def test_measurement_update_many_measurements(self):
        """More measurements than states (overdetermined) should still work."""
        np.random.seed(50)
        n = 3
        m = 10
        R_bar = np.eye(n) * 5.0
        b_bar = np.random.randn(n)
        H = np.random.randn(m, n)
        y = np.random.randn(m)

        x_hat, R_new, b_new = self.srif.measurement_update(R_bar, b_bar, H, y)

        self.assertEqual(x_hat.shape, (n,))
        assert_allclose(R_new @ x_hat, b_new, atol=1e-9)

    def test_whiten_measurements_scalar_like(self):
        """Single measurement dimension (m=1)."""
        y = np.array([5.0])
        H = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        R_cov = np.array([[4.0]])

        y_w, H_w = self.srif.whiten_measurements(y, H, R_cov)

        assert_allclose(y_w, np.array([2.5]), atol=1e-12)


# ===================================================================
# 9. Full SRIF.run Integration Smoke Test (mocked propagation)
# ===================================================================
class TestSRIFRun(unittest.TestCase):
    """
    Smoke test for the full SRIF.run() method using mocked integrator
    and measurement managers.  We verify the returned shapes and types
    rather than numerical accuracy, since real orbit dynamics are mocked.
    """

    def _setup_run(self):
        """Build all mocked objects needed for a minimal SRIF.run() call."""
        n = 8  # state size: 6 SC + mu + J2
        n_time = 5
        time_vector = np.linspace(0, 1000, n_time)

        # Build mock integrator
        integrator = _make_mock_integrator(mode=['mu', 'J2'])

        # integrate_stm returns (time, augmented_state)
        # augmented state = [state (n); flattened STM (n*n)] for each time
        state_hist = np.random.randn(n, n_time)
        stm_flat = np.tile(np.eye(n).flatten(), (n_time, 1)).T
        augmented = np.vstack([state_hist, stm_flat])
        integrator.integrate_stm.return_value = (time_vector, augmented)

        # Build mock measurement manager
        mgr = _make_mock_measurement_mgr("station_1")
        # simulate_measurements returns (2, n_time) array
        sim_meas = np.random.randn(2, n_time)
        mgr.simulate_measurements.return_value = sim_meas

        # Build measurement DataFrame
        truth_meas = [sim_meas[:, i] + np.random.randn(2) * 0.001 for i in range(n_time)]
        meas_df = pd.DataFrame({
            'time': time_vector,
            'station_1_measurements': truth_meas,
        })

        # R_noise must be a 2x2 SPD matrix
        R_noise = np.eye(2) * 0.01

        initial_state = np.zeros(n)
        initial_state[0] = 7000.0  # position x
        initial_state[3] = 7.0     # velocity u
        initial_x_correction = np.zeros(n)
        initial_covariance = np.eye(n) * 100.0

        # Patch CoordinateMgr methods used inside run()
        srif = SRIF(integrator, [mgr], initial_earth_spin_angle=0.0)
        srif.coordinate_mgr.ECEF_to_ECI = MagicMock(return_value=np.array([6378.0, 0, 0, 0, 0, 0]))
        srif.coordinate_mgr.ECEF_to_GCS = MagicMock(return_value=(0.0, 0.0))
        srif.coordinate_mgr.compute_DCM = MagicMock(return_value=np.eye(3))

        return srif, initial_state, initial_x_correction, initial_covariance, meas_df, R_noise

    @patch('ASEN_6080.Tools.SRIF.measurement_jacobian')
    def test_run_returns_correct_shapes(self, mock_meas_jac):
        """SRIF.run() should return arrays with the expected dimensions."""
        srif, x0, dx0, P0, meas_df, R_noise = self._setup_run()
        n = len(x0)
        n_time = len(meas_df)

        # Mock measurement_jacobian to return correctly shaped arrays
        H_sc = np.random.randn(2, 6)
        H_station = np.random.randn(2, 3)
        mock_meas_jac.return_value = [H_sc, H_station]

        state_est, cov_est, residuals_df = srif.run(
            initial_state=x0,
            initial_x_correction=dx0,
            initial_covariance=P0,
            measurement_data=meas_df,
            R_noise=R_noise,
            max_iterations=1,
        )

        self.assertEqual(state_est.shape, (n, n_time))
        self.assertEqual(cov_est.shape, (n, n, n_time))
        self.assertIsInstance(residuals_df, pd.DataFrame)

    @patch('ASEN_6080.Tools.SRIF.measurement_jacobian')
    def test_run_residuals_dataframe_columns(self, mock_meas_jac):
        """The residuals DataFrame should contain the expected columns."""
        srif, x0, dx0, P0, meas_df, R_noise = self._setup_run()

        H_sc = np.random.randn(2, 6)
        H_station = np.random.randn(2, 3)
        mock_meas_jac.return_value = [H_sc, H_station]

        _, _, residuals_df = srif.run(
            initial_state=x0,
            initial_x_correction=dx0,
            initial_covariance=P0,
            measurement_data=meas_df,
            R_noise=R_noise,
            max_iterations=1,
        )

        for col in ['iteration', 'station', 'pre-fit', 'post-fit']:
            self.assertIn(col, residuals_df.columns)

    @patch('ASEN_6080.Tools.SRIF.measurement_jacobian')
    def test_run_multiple_iterations(self, mock_meas_jac):
        """Running with max_iterations > 1 should produce residuals rows
        for each iteration."""
        srif, x0, dx0, P0, meas_df, R_noise = self._setup_run()

        H_sc = np.random.randn(2, 6)
        H_station = np.random.randn(2, 3)
        mock_meas_jac.return_value = [H_sc, H_station]

        _, _, residuals_df = srif.run(
            initial_state=x0,
            initial_x_correction=dx0,
            initial_covariance=P0,
            measurement_data=meas_df,
            R_noise=R_noise,
            max_iterations=3,
        )

        # Each iteration should add one row per station
        n_stations = len(srif.measurement_mgrs)
        expected_rows = 3 * n_stations
        self.assertEqual(len(residuals_df), expected_rows)

    @patch('ASEN_6080.Tools.SRIF.measurement_jacobian')
    def test_run_covariances_are_spd(self, mock_meas_jac):
        """All returned covariance matrices should be symmetric positive definite."""
        srif, x0, dx0, P0, meas_df, R_noise = self._setup_run()

        H_sc = np.random.randn(2, 6)
        H_station = np.random.randn(2, 3)
        mock_meas_jac.return_value = [H_sc, H_station]

        _, cov_est, _ = srif.run(
            initial_state=x0,
            initial_x_correction=dx0,
            initial_covariance=P0,
            measurement_data=meas_df,
            R_noise=R_noise,
            max_iterations=1,
        )

        for k in range(cov_est.shape[2]):
            P = cov_est[:, :, k]
            # Symmetric
            assert_allclose(P, P.T, atol=1e-8,
                            err_msg=f"Covariance at step {k} is not symmetric")
            # Positive definite
            eigvals = np.linalg.eigvalsh(P)
            self.assertTrue(np.all(eigvals > 0),
                            f"Covariance at step {k} is not positive definite: eigvals={eigvals}")


# ===================================================================
# Run
# ===================================================================
if __name__ == "__main__":
    unittest.main()