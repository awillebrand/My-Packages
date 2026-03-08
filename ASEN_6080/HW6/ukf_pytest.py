"""
Test script for the Unscented Kalman Filter (UKF) implementation.
Tests each UKF method individually and runs a full filter integration test
using a simple two-body orbit scenario with a single ground station.
"""
import numpy as np
import pandas as pd
import pytest
from ASEN_6080.Tools import Integrator, MeasurementMgr, CoordinateMgr
from ASEN_6080.Tools.UKF import UKF


# ── Constants ────────────────────────────────────────────────────────────────
MU = 398600.4418          # km^3/s^2  (Earth)
R_E = 6378.0              # km        (Earth radius)
OMEGA_E = 2 * np.pi / 86164.0905   # rad/s  (sidereal rotation)

# Spacecraft in a ~7078 km circular orbit (700 km altitude), equatorial
R0 = np.array([7078.0, 0.0, 0.0])            # km
V0 = np.array([0.0, np.sqrt(MU / 7078.0), 0.0])  # km/s  (circular speed)
STATE_6 = np.concatenate([R0, V0])

# Simple J2 setup  — state length = 8  (pos, vel, mu, J2)
MODE  = ['mu', 'J2']
PARAM_IDX = [6, 7]
J2_VAL = 1.08263e-3
STATE_8 = np.concatenate([STATE_6, [MU, J2_VAL]])

# Station at lat=0°, lon=0° on the equator
STATION_LAT = 0.0
STATION_LON = 0.0
INITIAL_SPIN_ANGLE = 0.0


# ── Fixtures ─────────────────────────────────────────────────────────────────
@pytest.fixture
def integrator():
    """Integrator with mu + J2 mode (state length 8)."""
    return Integrator(mu=MU, R_e=R_E, mode=MODE, parameter_indices=PARAM_IDX)


@pytest.fixture
def measurement_mgr():
    """Ground-station measurement manager at (0°, 0°)."""
    return MeasurementMgr(
        station_name="Equator-0",
        station_lat=STATION_LAT,
        station_lon=STATION_LON,
        initial_earth_spin_angle=INITIAL_SPIN_ANGLE,
        earth_spin_rate=OMEGA_E,
        R_e=R_E,
    )


@pytest.fixture
def ukf(integrator, measurement_mgr):
    """UKF instance wired to a single measurement manager."""
    return UKF(
        integrator=integrator,
        measurement_mgr_list=[measurement_mgr],
        initial_earth_spin_angle=INITIAL_SPIN_ANGLE,
        earth_rotation_rate=OMEGA_E,
    )


@pytest.fixture
def weights(ukf):
    """Pre-computed weights for the 8-dim state."""
    L = len(STATE_8)
    return ukf.compute_weights(alpha=1e-3, beta=2, L=L)


@pytest.fixture
def sigma_pts(ukf, weights):
    """Sigma points generated from a diagonal covariance."""
    _, _, gamma = weights
    P0 = np.diag([1.0, 1.0, 1.0, 1e-6, 1e-6, 1e-6, 1.0, 1e-6])
    return ukf.compute_sigma_points(STATE_8, P0, gamma)


# ── 1. Weight computation ────────────────────────────────────────────────────
class TestComputeWeights:
    def test_weight_shapes(self, ukf):
        L = 8
        Wm, Wc, gamma = ukf.compute_weights(alpha=1e-3, beta=2, L=L)
        assert Wm.shape == (2 * L + 1,)
        assert Wc.shape == (2 * L + 1,)
        assert isinstance(gamma, float)

    def test_mean_weights_sum_to_one(self, ukf):
        """The mean weights must sum to 1 for an unbiased estimate."""
        L = 8
        Wm, _, _ = ukf.compute_weights(alpha=1e-3, beta=2, L=L)
        assert np.isclose(np.sum(Wm), 1.0, atol=1e-12)

    def test_gamma_positive(self, ukf):
        L = 8
        _, _, gamma = ukf.compute_weights(alpha=1e-3, beta=2, L=L)
        assert gamma > 0

    def test_different_alpha_changes_output(self, ukf):
        L = 8
        Wm_a, _, gamma_a = ukf.compute_weights(alpha=1e-3, beta=2, L=L)
        Wm_b, _, gamma_b = ukf.compute_weights(alpha=0.5, beta=2, L=L)
        assert not np.allclose(Wm_a, Wm_b)
        assert gamma_a != gamma_b

    def test_covariance_weight_0_differs(self, ukf):
        """Wc[0] includes the (1 - alpha^2 + beta) correction term."""
        L = 8
        alpha, beta = 1e-3, 2
        Wm, Wc, _ = ukf.compute_weights(alpha, beta, L)
        expected_diff = 1 - alpha**2 + beta  # ≈ 3.0
        assert np.isclose(Wc[0] - Wm[0], expected_diff, atol=1e-10)


# ── 2. Sigma-point generation ────────────────────────────────────────────────
class TestComputeSigmaPoints:
    def test_sigma_points_shape(self, sigma_pts):
        L = len(STATE_8)
        assert sigma_pts.shape == (L, 2 * L + 1)

    def test_first_sigma_point_is_mean(self, sigma_pts):
        np.testing.assert_array_almost_equal(sigma_pts[:, 0], STATE_8)

    def test_symmetric_pairs(self, sigma_pts):
        """sigma_i and sigma_{i+L} should be symmetric about the mean."""
        L = len(STATE_8)
        for i in range(L):
            diff_plus  = sigma_pts[:, i + 1] - STATE_8
            diff_minus = sigma_pts[:, i + 1 + L] - STATE_8
            np.testing.assert_array_almost_equal(diff_plus, -diff_minus)

    def test_non_positive_definite_raises(self, ukf, weights):
        _, _, gamma = weights
        P_bad = -np.eye(len(STATE_8))
        with pytest.raises(ValueError, match="not positive definite"):
            ukf.compute_sigma_points(STATE_8, P_bad, gamma)


# ── 3. Sigma-point propagation ──────────────────────────────────────────────
class TestPropagateSigmaPoints:
    def test_output_shape_preserved(self, ukf, sigma_pts):
        """Propagated sigma points must keep the same (L, 2L+1) shape."""
        dt = 10.0  # seconds
        pred = ukf.propagate_sigma_points(sigma_pts, dt)
        assert pred.shape == sigma_pts.shape

    def test_propagation_changes_state(self, ukf, sigma_pts):
        """After a non-zero dt the sigma points should differ from the originals."""
        dt = 60.0
        pred = ukf.propagate_sigma_points(sigma_pts, dt)
        assert not np.allclose(pred, sigma_pts)

    def test_zero_dt_returns_original(self, ukf, sigma_pts):
        """Propagating with dt~0 should return nearly the same sigma points."""
        dt = 1e-10
        pred = ukf.propagate_sigma_points(sigma_pts, dt)
        np.testing.assert_array_almost_equal(pred, sigma_pts, decimal=4)


# ── 4. Time update ──────────────────────────────────────────────────────────
class TestTimeUpdate:
    def test_output_shapes(self, ukf, weights, sigma_pts):
        Wm, Wc, _ = weights
        x_pred, P_pred = ukf.time_update(sigma_pts, Wm, Wc)
        L = len(STATE_8)
        assert x_pred.shape == (L,)
        assert P_pred.shape == (L, L)

    def test_covariance_symmetric(self, ukf, weights, sigma_pts):
        Wm, Wc, _ = weights
        _, P_pred = ukf.time_update(sigma_pts, Wm, Wc)
        np.testing.assert_array_almost_equal(P_pred, P_pred.T)

    def test_process_noise_addition(self, ukf, weights, sigma_pts):
        Wm, Wc, _ = weights
        _, P_no_Q = ukf.time_update(sigma_pts, Wm, Wc, Q=None)
        Q = 1e-8 * np.eye(len(STATE_8))
        _, P_with_Q = ukf.time_update(sigma_pts, Wm, Wc, Q=Q)
        # P_with_Q should be element-wise >= P_no_Q (within numerical noise)
        diff = P_with_Q - P_no_Q
        assert np.all(np.diag(diff) >= -1e-15)

    def test_mean_close_to_input_for_small_cov(self, ukf):
        """With a tiny covariance, the predicted mean ≈ the original state."""
        L = len(STATE_8)
        Wm, Wc, gamma = ukf.compute_weights(1e-3, 2, L)
        P_tiny = 1e-20 * np.eye(L)
        sp = ukf.compute_sigma_points(STATE_8, P_tiny, gamma)
        x_pred, _ = ukf.time_update(sp, Wm, Wc)
        np.testing.assert_allclose(x_pred, STATE_8, rtol=1e-6, atol=1e-12)


# ── 5. Measurement prediction ───────────────────────────────────────────────
class TestMeasurementPrediction:
    def test_output_shapes(self, ukf, weights, sigma_pts, measurement_mgr):
        Wm, _, _ = weights
        y_bar, y_sigma = ukf.compute_measurement_prediction(sigma_pts, measurement_mgr, Wm)
        L = len(STATE_8)
        assert y_bar.shape == (2,)           # same dim as state (range, rr padded)
        assert y_sigma.shape == sigma_pts[0:2,:].shape

    def test_predicted_measurement_nonzero(self, ukf, weights, sigma_pts, measurement_mgr):
        Wm, _, _ = weights
        y_bar, _ = ukf.compute_measurement_prediction(sigma_pts, measurement_mgr, Wm)
        # The first two elements (range, range-rate) should be non-trivial
        assert not np.isclose(y_bar[0], 0.0)


# ── 6. Cross-covariances ────────────────────────────────────────────────────
class TestCrossCovariances:
    def _get_covariances(self, ukf, weights, sigma_pts, measurement_mgr, R=None):
        Wm, Wc, _ = weights
        x_bar, _ = ukf.time_update(sigma_pts, Wm, Wc)
        y_bar, y_sigma = ukf.compute_measurement_prediction(sigma_pts, measurement_mgr, Wm)
        return ukf.compute_cross_covariances(sigma_pts, y_sigma, x_bar, y_bar, Wc, R=R)

    def test_Pyy_shape(self, ukf, weights, sigma_pts, measurement_mgr):
        P_yy, P_xy = self._get_covariances(ukf, weights, sigma_pts, measurement_mgr)
        L = len(STATE_8)
        assert P_yy.shape == (2, 2)
        assert P_xy.shape == (L, 2)

    def test_Pyy_symmetric(self, ukf, weights, sigma_pts, measurement_mgr):
        P_yy, _ = self._get_covariances(ukf, weights, sigma_pts, measurement_mgr)
        np.testing.assert_array_almost_equal(P_yy, P_yy.T)

    def test_measurement_noise_addition(self, ukf, weights, sigma_pts, measurement_mgr):
        P_yy_no_R, _ = self._get_covariances(ukf, weights, sigma_pts, measurement_mgr, R=None)
        R = 1e-4 * np.eye(2)
        P_yy_with_R, _ = self._get_covariances(ukf, weights, sigma_pts, measurement_mgr, R=R)
        diff = P_yy_with_R - P_yy_no_R
        assert np.all(np.diag(diff) >= -1e-15)


# ── 7. Measurement update ───────────────────────────────────────────────────
class TestMeasurementUpdate:
    def test_output_shapes(self, ukf):
        L = len(STATE_8)
        x_bar = STATE_8.copy()
        P_bar = np.eye(L)
        y_bar = np.zeros(L)
        P_yy  = np.eye(L)
        P_xy  = 0.5 * np.eye(L)
        y_meas = np.ones(L)
        x_upd, P_upd = ukf.measurement_update(x_bar, P_bar, y_bar, P_yy, P_xy, y_meas)
        assert x_upd.shape == (L,)
        assert P_upd.shape == (L, L)

    def test_zero_innovation_no_change(self, ukf):
        """If y_meas == y_bar the state should not change."""
        L = len(STATE_8)
        x_bar = STATE_8.copy()
        P_bar = np.eye(L)
        y_bar = np.ones(L)
        P_yy  = np.eye(L)
        P_xy  = 0.5 * np.eye(L)
        y_meas = np.ones(L)  # same as y_bar
        x_upd, _ = ukf.measurement_update(x_bar, P_bar, y_bar, P_yy, P_xy, y_meas)
        np.testing.assert_array_almost_equal(x_upd, x_bar)

    def test_covariance_decreases(self, ukf):
        """After a measurement update, the covariance should shrink (trace-wise)."""
        L = len(STATE_8)
        P_bar = 10.0 * np.eye(L)
        P_yy  = np.eye(L)
        P_xy  = 0.5 * np.eye(L)
        x_bar = STATE_8.copy()
        y_bar = np.zeros(L)
        y_meas = np.ones(L)
        _, P_upd = ukf.measurement_update(x_bar, P_bar, y_bar, P_yy, P_xy, y_meas)
        assert np.trace(P_upd) < np.trace(P_bar)

    def test_kalman_gain_correctness(self, ukf):
        """Verify K = P_xy @ inv(P_yy) produces the expected state shift."""
        L = len(STATE_8)
        x_bar = np.zeros(L)
        P_bar = np.eye(L)
        P_yy  = 2.0 * np.eye(L)
        P_xy  = np.eye(L)
        y_bar = np.zeros(L)
        y_meas = np.ones(L)

        K_expected = P_xy @ np.linalg.inv(P_yy)  # 0.5 * I
        x_expected = x_bar + K_expected @ (y_meas - y_bar)

        x_upd, _ = ukf.measurement_update(x_bar, P_bar, y_bar, P_yy, P_xy, y_meas)
        np.testing.assert_array_almost_equal(x_upd, x_expected)


# ── 8. Full UKF run (integration test) ──────────────────────────────────────
class TestUKFRun:
    @staticmethod
    def _build_measurement_df(integrator, measurement_mgr, time_vector, true_state):
        """Propagate the true state and produce a measurement DataFrame."""
        # Propagate truth
        _, truth = integrator.integrate_eom(
            time_vector[-1], true_state, teval=time_vector
        )

        # Simulate measurements at each time (range, range-rate) from the station
        meas = measurement_mgr.simulate_measurements(
            truth, time_vector, coordinate_frame="ECI", noise=False, ignore_visibility=True
        )

        # Build DataFrame expected by UKF.run()
        # Columns: time, station_1, station_2, station_3   (only station_1 has data)
        n = len(time_vector)
        data = {
            "time": time_vector,
            "station_1": [meas[:, i] for i in range(n)],
            "station_2": [np.array([np.nan, np.nan])] * n,
            "station_3": [np.array([np.nan, np.nan])] * n,
        }
        return pd.DataFrame(data), truth

    def test_run_output_shapes(self, ukf, integrator, measurement_mgr):
        """Verify estimated_states and covariances have correct shapes."""
        L = len(STATE_8)
        time_vector = np.linspace(0, 300, 4)  # short run, few steps
        df, _ = self._build_measurement_df(
            integrator, measurement_mgr, time_vector, STATE_8
        )
        P0 = np.diag([1.0, 1.0, 1.0, 1e-6, 1e-6, 1e-6, 1.0, 1e-6])

        est_states, est_covs = ukf.run(
            initial_state=STATE_8,
            initial_covariance=P0,
            time_vector=time_vector,
            measurement_data=df,
            alpha=1e-3,
            beta=2,
        )
        assert est_states.shape == (L, len(time_vector))
        assert est_covs.shape == (L, L, len(time_vector))

    def test_initial_epoch_stored(self, ukf, integrator, measurement_mgr):
        """At t=0 the filter should store the initial state unchanged."""
        time_vector = np.linspace(0, 300, 4)
        df, _ = self._build_measurement_df(
            integrator, measurement_mgr, time_vector, STATE_8
        )
        P0 = np.diag([1.0, 1.0, 1.0, 1e-6, 1e-6, 1e-6, 1.0, 1e-6])
        est_states, est_covs = ukf.run(
            STATE_8, P0, time_vector, df, alpha=1e-3, beta=2
        )
        np.testing.assert_array_almost_equal(est_states[:, 0], STATE_8)
        np.testing.assert_array_almost_equal(est_covs[:, :, 0], P0)

    def test_no_nan_in_estimates(self, ukf, integrator, measurement_mgr):
        """Ensure the filter does not produce NaN values."""
        time_vector = np.linspace(0, 300, 4)
        df, _ = self._build_measurement_df(
            integrator, measurement_mgr, time_vector, STATE_8
        )
        P0 = np.diag([1.0, 1.0, 1.0, 1e-6, 1e-6, 1e-6, 1.0, 1e-6])
        est_states, est_covs = ukf.run(
            STATE_8, P0, time_vector, df, alpha=1e-3, beta=2
        )
        assert not np.any(np.isnan(est_states))
        assert not np.any(np.isnan(est_covs))

    def test_nan_measurements_skip_update(self, ukf, integrator, measurement_mgr):
        """When all measurements are NaN the filter should just propagate."""
        time_vector = np.array([0.0, 60.0, 120.0])
        n = len(time_vector)
        # All NaN measurements  → pure prediction
        data = {
            "time": time_vector,
            "station_1": [np.array([np.nan, np.nan])] * n,
            "station_2": [np.array([np.nan, np.nan])] * n,
            "station_3": [np.array([np.nan, np.nan])] * n,
        }
        df = pd.DataFrame(data)
        P0 = np.diag([1.0, 1.0, 1.0, 1e-6, 1e-6, 1e-6, 1.0, 1e-6])

        est_states, est_covs = ukf.run(
            STATE_8, P0, time_vector, df, alpha=1e-3, beta=2
        )
        # Should still produce valid output (no crash, no NaN)
        assert not np.any(np.isnan(est_states))

    def test_process_noise_increases_uncertainty(self, ukf, integrator, measurement_mgr):
        """Adding Q should make the final covariance larger than without it."""
        time_vector = np.array([0.0, 60.0, 120.0])
        n = len(time_vector)
        # All NaN  → no measurement updates, purely checks Q propagation
        data = {
            "time": time_vector,
            "station_1": [np.array([np.nan, np.nan])] * n,
            "station_2": [np.array([np.nan, np.nan])] * n,
            "station_3": [np.array([np.nan, np.nan])] * n,
        }
        df = pd.DataFrame(data)
        L = len(STATE_8)
        P0 = np.diag([1.0, 1.0, 1.0, 1e-6, 1e-6, 1e-6, 1.0, 1e-6])
        Q  = 1e-6 * np.eye(L)

        _, cov_no_Q = ukf.run(STATE_8, P0, time_vector, df, alpha=1e-3, beta=2, Q=None)
        _, cov_with_Q = ukf.run(STATE_8, P0, time_vector, df, alpha=1e-3, beta=2, Q=Q)

        assert np.trace(cov_with_Q[:, :, -1]) > np.trace(cov_no_Q[:, :, -1])


# ── 9. Constructor / wiring ─────────────────────────────────────────────────
class TestUKFConstructor:
    def test_integrator_stored(self, ukf, integrator):
        assert ukf.integrator is integrator

    def test_measurement_mgrs_stored(self, ukf, measurement_mgr):
        assert len(ukf.measurement_mgrs) == 1
        assert ukf.measurement_mgrs[0] is measurement_mgr

    def test_coordinate_mgr_created(self, ukf):
        assert isinstance(ukf.coordinate_mgr, CoordinateMgr)

    def test_multiple_stations(self, integrator):
        mgrs = [
            MeasurementMgr("A", station_lat=0, station_lon=0,
                           initial_earth_spin_angle=0, R_e=R_E),
            MeasurementMgr("B", station_lat=30, station_lon=90,
                           initial_earth_spin_angle=0, R_e=R_E),
        ]
        ukf_multi = UKF(integrator, mgrs, initial_earth_spin_angle=0.0)
        assert len(ukf_multi.measurement_mgrs) == 2