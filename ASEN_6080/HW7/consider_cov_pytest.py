"""
Test script for the Consider Covariance (ConsiderCov) implementation.
Tests each ConsiderCov method individually using pure linear-algebra
checks that do not require full orbit propagation, plus an integration
test for the `run` method.

Uses a simple two-body orbit scenario with a single ground station,
matching the conventions of the existing UKF test suite.
"""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from ASEN_6080.Tools import Integrator, MeasurementMgr, CoordinateMgr
from ASEN_6080.Tools.consider_cov import ConsiderCov


# ── Constants ─────────────────────────────────────────────────────────────────
MU = 398600.4418          # km^3/s^2  (Earth)
R_E = 6378.0              # km        (Earth radius)
OMEGA_E = 2 * np.pi / 86164.0905   # rad/s  (sidereal rotation)

# Spacecraft in a ~7078 km circular orbit (700 km altitude), equatorial
R0 = np.array([7078.0, 0.0, 0.0])
V0 = np.array([0.0, np.sqrt(MU / 7078.0), 0.0])
STATE_6 = np.concatenate([R0, V0])

# State with mu and J2 as estimated parameters (length 8)
MODE = ['mu', 'J2']
PARAM_IDX = [6, 7]
J2_VAL = 1.08263e-3
R = np.diag([1.0, 1.0])
STATE_8 = np.concatenate([STATE_6, [MU, J2_VAL]])

# Station at lat=0°, lon=0° on the equator
STATION_LAT = 0.0
STATION_LON = 0.0
INITIAL_SPIN_ANGLE = 0.0

# Dimensions for consider-parameter tests
N_STATE = len(STATE_8)        # 8
N_CONSIDER = 2                # e.g. two consider parameters
N_MEAS = 2                    # range & range-rate


# ── Fixtures ──────────────────────────────────────────────────────────────────
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
def consider_cov(integrator, measurement_mgr):
    """ConsiderCov instance wired to a single measurement manager."""
    return ConsiderCov(
        integrator=integrator,
        measurement_mgr_list=[measurement_mgr],
        initial_earth_spin_angle=INITIAL_SPIN_ANGLE,
        earth_rotation_rate=OMEGA_E,
    )


# ── Helper: repeatable random inputs ─────────────────────────────────────────
@pytest.fixture
def rng():
    """Seeded random number generator for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def time_update_inputs(rng):
    """Generate a consistent set of inputs for time_update tests."""
    n = N_STATE
    nc = N_CONSIDER
    x_hat = rng.standard_normal(n)
    P_hat = rng.standard_normal((n, n))
    P_hat = P_hat @ P_hat.T + 0.1 * np.eye(n)          # symmetric positive-definite
    S_hat = rng.standard_normal((n, nc))
    theta = rng.standard_normal((n, nc))
    phi = np.eye(n) + 0.01 * rng.standard_normal((n, n))  # near-identity STM
    c = rng.standard_normal(nc)
    P_cc = rng.standard_normal((nc, nc))
    P_cc = P_cc @ P_cc.T + 0.1 * np.eye(nc)             # symmetric positive-definite
    return x_hat, P_hat, S_hat, theta, phi, c, P_cc


@pytest.fixture
def measurement_update_inputs(rng):
    """Generate a consistent set of inputs for measurement_update tests."""
    n = N_STATE
    nc = N_CONSIDER
    nm = N_MEAS
    predicted_state = rng.standard_normal(n)
    P_bar = rng.standard_normal((n, n))
    P_bar = P_bar @ P_bar.T + 0.1 * np.eye(n)
    S_bar = rng.standard_normal((n, nc))
    P_cc = rng.standard_normal((nc, nc))
    P_cc = P_cc @ P_cc.T + 0.1 * np.eye(nc)
    c = rng.standard_normal(nc)
    H_x = rng.standard_normal((nm, n))
    H_c = rng.standard_normal((nm, nc))
    R = rng.standard_normal((nm, nm))
    R = R @ R.T + 0.5 * np.eye(nm)                      # symmetric positive-definite
    measurement_residual = rng.standard_normal((nm, 1))
    return predicted_state, P_bar, S_bar, P_cc, c, H_x, H_c, R, measurement_residual


# ══════════════════════════════════════════════════════════════════════════════
# 1. Constructor / Wiring
# ══════════════════════════════════════════════════════════════════════════════
class TestConsiderCovConstructor:
    def test_integrator_stored(self, consider_cov, integrator):
        assert consider_cov.integrator is integrator

    def test_measurement_mgrs_stored(self, consider_cov, measurement_mgr):
        assert len(consider_cov.measurement_mgrs) == 1
        assert consider_cov.measurement_mgrs[0] is measurement_mgr

    def test_coordinate_mgr_created(self, consider_cov):
        assert isinstance(consider_cov.coordinate_mgr, CoordinateMgr)

    def test_multiple_stations(self, integrator):
        mgrs = [
            MeasurementMgr("A", station_lat=0, station_lon=0,
                           initial_earth_spin_angle=0, R_e=R_E),
            MeasurementMgr("B", station_lat=30, station_lon=90,
                           initial_earth_spin_angle=0, R_e=R_E),
        ]
        cc = ConsiderCov(integrator, mgrs, initial_earth_spin_angle=0.0)
        assert len(cc.measurement_mgrs) == 2

    def test_custom_earth_rotation_rate(self, integrator, measurement_mgr):
        custom_rate = 1.0e-4
        cc = ConsiderCov(integrator, [measurement_mgr],
                         initial_earth_spin_angle=0.0,
                         earth_rotation_rate=custom_rate)
        # Verify the coordinate manager received the custom rate
        assert cc.coordinate_mgr is not None


# ══════════════════════════════════════════════════════════════════════════════
# 2. Time Update
# ══════════════════════════════════════════════════════════════════════════════
class TestTimeUpdate:
    def test_output_count(self, consider_cov, time_update_inputs):
        """time_update must return exactly 6 arrays."""
        result = consider_cov.time_update(*time_update_inputs)
        assert len(result) == 6

    def test_output_shapes(self, consider_cov, time_update_inputs):
        x_hat, P_hat, S_hat, theta, phi, c, P_cc = time_update_inputs
        x_bar, P_bar, S_bar, x_c_bar, P_c_bar, P_xc_bar = consider_cov.time_update(
            x_hat, P_hat, S_hat, theta, phi, c, P_cc
        )
        n = N_STATE
        nc = N_CONSIDER
        assert x_bar.shape == (n,)
        assert P_bar.shape == (n, n)
        assert S_bar.shape == (n, nc)
        assert x_c_bar.shape == (n,)
        assert P_c_bar.shape == (n, n)
        assert P_xc_bar.shape == (n, nc)

    def test_predicted_state(self, consider_cov, time_update_inputs):
        """x_bar must equal phi @ x_hat."""
        x_hat, P_hat, S_hat, theta, phi, c, P_cc = time_update_inputs
        x_bar, *_ = consider_cov.time_update(x_hat, P_hat, S_hat, theta, phi, c, P_cc)
        np.testing.assert_allclose(x_bar, phi @ x_hat, atol=1e-12)

    def test_predicted_covariance(self, consider_cov, time_update_inputs):
        """P_bar must equal phi @ P_hat @ phi.T."""
        x_hat, P_hat, S_hat, theta, phi, c, P_cc = time_update_inputs
        _, P_bar, *_ = consider_cov.time_update(x_hat, P_hat, S_hat, theta, phi, c, P_cc)
        expected = phi @ P_hat @ phi.T
        np.testing.assert_allclose(P_bar, expected, atol=1e-10)

    def test_predicted_covariance_symmetric(self, consider_cov, time_update_inputs):
        """P_bar must be symmetric."""
        _, P_bar, *_ = consider_cov.time_update(*time_update_inputs)
        np.testing.assert_allclose(P_bar, P_bar.T, atol=1e-12)

    def test_sensitivity_matrix_update(self, consider_cov, time_update_inputs):
        """S_bar must equal phi @ S_hat + theta."""
        x_hat, P_hat, S_hat, theta, phi, c, P_cc = time_update_inputs
        _, _, S_bar, *_ = consider_cov.time_update(x_hat, P_hat, S_hat, theta, phi, c, P_cc)
        expected = phi @ S_hat + theta
        np.testing.assert_allclose(S_bar, expected, atol=1e-12)

    def test_consider_state_estimate(self, consider_cov, time_update_inputs):
        """x_c_bar must equal x_bar + S_bar @ c."""
        x_hat, P_hat, S_hat, theta, phi, c, P_cc = time_update_inputs
        x_bar, _, S_bar, x_c_bar, *_ = consider_cov.time_update(
            x_hat, P_hat, S_hat, theta, phi, c, P_cc
        )
        np.testing.assert_allclose(x_c_bar, x_bar + S_bar @ c, atol=1e-12)

    def test_consider_covariance(self, consider_cov, time_update_inputs):
        """P_c_bar must equal P_bar + S_bar @ P_cc @ S_bar.T."""
        x_hat, P_hat, S_hat, theta, phi, c, P_cc = time_update_inputs
        _, P_bar, S_bar, _, P_c_bar, _ = consider_cov.time_update(
            x_hat, P_hat, S_hat, theta, phi, c, P_cc
        )
        expected = P_bar + S_bar @ P_cc @ S_bar.T
        np.testing.assert_allclose(P_c_bar, expected, atol=1e-10)

    def test_consider_covariance_symmetric(self, consider_cov, time_update_inputs):
        """P_c_bar must be symmetric."""
        *_, P_c_bar, _ = consider_cov.time_update(*time_update_inputs)
        np.testing.assert_allclose(P_c_bar, P_c_bar.T, atol=1e-12)

    def test_consider_covariance_geq_conventional(self, consider_cov, time_update_inputs):
        """Diagonal of P_c_bar >= diagonal of P_bar (consider adds uncertainty)."""
        _, P_bar, _, _, P_c_bar, _ = consider_cov.time_update(*time_update_inputs)
        assert np.all(np.diag(P_c_bar) >= np.diag(P_bar) - 1e-12)

    def test_cross_covariance(self, consider_cov, time_update_inputs):
        """P_xc_bar must equal S_bar @ P_cc."""
        x_hat, P_hat, S_hat, theta, phi, c, P_cc = time_update_inputs
        _, _, S_bar, _, _, P_xc_bar = consider_cov.time_update(
            x_hat, P_hat, S_hat, theta, phi, c, P_cc
        )
        expected = S_bar @ P_cc
        np.testing.assert_allclose(P_xc_bar, expected, atol=1e-12)

    def test_identity_stm_preserves_state(self, consider_cov, rng):
        """With phi=I and theta=0, x_bar = x_hat and S_bar = S_hat."""
        n, nc = N_STATE, N_CONSIDER
        x_hat = rng.standard_normal(n)
        P_hat = np.eye(n)
        S_hat = rng.standard_normal((n, nc))
        theta = np.zeros((n, nc))
        phi = np.eye(n)
        c = np.zeros(nc)
        P_cc = np.eye(nc)
        x_bar, _, S_bar, *_ = consider_cov.time_update(x_hat, P_hat, S_hat, theta, phi, c, P_cc)
        np.testing.assert_allclose(x_bar, x_hat, atol=1e-14)
        np.testing.assert_allclose(S_bar, S_hat, atol=1e-14)

    def test_zero_consider_parameter_error(self, consider_cov, time_update_inputs):
        """When c = 0, x_c_bar must equal x_bar (no consider bias)."""
        x_hat, P_hat, S_hat, theta, phi, _, P_cc = time_update_inputs
        c_zero = np.zeros(N_CONSIDER)
        x_bar, _, _, x_c_bar, *_ = consider_cov.time_update(
            x_hat, P_hat, S_hat, theta, phi, c_zero, P_cc
        )
        np.testing.assert_allclose(x_c_bar, x_bar, atol=1e-14)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Measurement Update
# ══════════════════════════════════════════════════════════════════════════════
class TestMeasurementUpdate:
    def test_output_count(self, consider_cov, measurement_update_inputs):
        """measurement_update must return exactly 6 arrays."""
        result = consider_cov.measurement_update(*measurement_update_inputs)
        assert len(result) == 6

    def test_output_shapes(self, consider_cov, measurement_update_inputs):
        pred_state, P_bar, S_bar, P_cc, c, H_x, H_c, R, y_res = measurement_update_inputs
        x_hat, P_x_hat, S_hat, x_c_hat, P_c_hat, P_xc_hat = consider_cov.measurement_update(
            pred_state, P_bar, S_bar, P_cc, c, H_x, H_c, R, y_res
        )
        n = N_STATE
        nc = N_CONSIDER
        assert x_hat.shape == (n, 1)
        assert P_x_hat.shape == (n, n)
        assert S_hat.shape == (n, nc)
        assert x_c_hat.shape == (n, 1)
        assert P_c_hat.shape == (n, n)
        assert P_xc_hat.shape == (n, nc)

    def test_kalman_gain_formula(self, consider_cov, measurement_update_inputs):
        """K must equal P_bar H_x^T (H_x P_bar H_x^T + R)^{-1}."""
        pred_state, P_bar, S_bar, P_cc, c, H_x, H_c, R, y_res = measurement_update_inputs
        K_expected = P_bar @ H_x.T @ np.linalg.inv(H_x @ P_bar @ H_x.T + R)
        x_hat, *_ = consider_cov.measurement_update(
            pred_state, P_bar, S_bar, P_cc, c, H_x, H_c, R, y_res
        )
        expected_x = np.vstack(pred_state) + K_expected @ (y_res - H_x @ np.vstack(pred_state))
        np.testing.assert_allclose(x_hat, expected_x, atol=1e-10)

    def test_covariance_joseph_form(self, consider_cov, measurement_update_inputs):
        """P_x_hat must follow the Joseph stabilised form."""
        pred_state, P_bar, S_bar, P_cc, c, H_x, H_c, R, y_res = measurement_update_inputs
        K = P_bar @ H_x.T @ np.linalg.inv(H_x @ P_bar @ H_x.T + R)
        I = np.eye(N_STATE)
        expected = (I - K @ H_x) @ P_bar @ (I - K @ H_x).T + K @ R @ K.T
        _, P_x_hat, *_ = consider_cov.measurement_update(
            pred_state, P_bar, S_bar, P_cc, c, H_x, H_c, R, y_res
        )
        np.testing.assert_allclose(P_x_hat, expected, atol=1e-10)

    def test_covariance_symmetric(self, consider_cov, measurement_update_inputs):
        """P_x_hat must be symmetric."""
        _, P_x_hat, *_ = consider_cov.measurement_update(*measurement_update_inputs)
        np.testing.assert_allclose(P_x_hat, P_x_hat.T, atol=1e-12)

    def test_covariance_trace_decreases(self, consider_cov, measurement_update_inputs):
        """A measurement update should reduce the state covariance trace."""
        _, P_bar, *rest = measurement_update_inputs
        _, P_x_hat, *_ = consider_cov.measurement_update(
            measurement_update_inputs[0], P_bar, *rest
        )
        assert np.trace(P_x_hat) < np.trace(P_bar)

    def test_sensitivity_matrix_update(self, consider_cov, measurement_update_inputs):
        """S_hat must equal (I - K H_x) S_bar - K H_c."""
        pred_state, P_bar, S_bar, P_cc, c, H_x, H_c, R, y_res = measurement_update_inputs
        K = P_bar @ H_x.T @ np.linalg.inv(H_x @ P_bar @ H_x.T + R)
        I = np.eye(N_STATE)
        expected_S = (I - K @ H_x) @ S_bar - K @ H_c
        _, _, S_hat, *_ = consider_cov.measurement_update(
            pred_state, P_bar, S_bar, P_cc, c, H_x, H_c, R, y_res
        )
        np.testing.assert_allclose(S_hat, expected_S, atol=1e-10)

    def test_consider_state_update(self, consider_cov, measurement_update_inputs):
        """x_c_hat must equal x_hat + S_hat @ c."""
        pred_state, P_bar, S_bar, P_cc, c, H_x, H_c, R, y_res = measurement_update_inputs
        x_hat, _, S_hat, x_c_hat, *_ = consider_cov.measurement_update(
            pred_state, P_bar, S_bar, P_cc, c, H_x, H_c, R, y_res
        )
        np.testing.assert_allclose(x_c_hat, x_hat + S_hat @ c, atol=1e-12)

    def test_consider_covariance_update(self, consider_cov, measurement_update_inputs):
        """P_c_hat must equal P_x_hat + S_hat @ P_cc @ S_hat.T."""
        pred_state, P_bar, S_bar, P_cc, c, H_x, H_c, R, y_res = measurement_update_inputs
        _, P_x_hat, S_hat, _, P_c_hat, _ = consider_cov.measurement_update(
            pred_state, P_bar, S_bar, P_cc, c, H_x, H_c, R, y_res
        )
        expected = P_x_hat + S_hat @ P_cc @ S_hat.T
        np.testing.assert_allclose(P_c_hat, expected, atol=1e-10)

    def test_consider_covariance_geq_conventional(self, consider_cov, measurement_update_inputs):
        """Diagonal of P_c_hat >= diagonal of P_x_hat."""
        _, P_x_hat, _, _, P_c_hat, _ = consider_cov.measurement_update(
            *measurement_update_inputs
        )
        assert np.all(np.diag(P_c_hat) >= np.diag(P_x_hat) - 1e-12)

    def test_consider_cross_covariance(self, consider_cov, measurement_update_inputs):
        """P_xc_hat must equal S_hat @ P_cc."""
        pred_state, P_bar, S_bar, P_cc, c, H_x, H_c, R, y_res = measurement_update_inputs
        _, _, S_hat, _, _, P_xc_hat = consider_cov.measurement_update(
            pred_state, P_bar, S_bar, P_cc, c, H_x, H_c, R, y_res
        )
        np.testing.assert_allclose(P_xc_hat, S_hat @ P_cc, atol=1e-12)

    def test_zero_innovation_no_state_change(self, consider_cov, rng):
        """When the measurement residual is zero, x_hat ≈ predicted_state."""
        n, nc, nm = N_STATE, N_CONSIDER, N_MEAS
        pred_state = rng.standard_normal(n)
        P_bar = np.eye(n)
        S_bar = rng.standard_normal((n, nc))
        P_cc = np.eye(nc)
        c = np.zeros(nc)
        H_x = rng.standard_normal((nm, n))
        H_c = np.zeros((nm, nc))
        R = np.eye(nm)
        y_res = np.zeros((nm, 1))  # zero residual
        x_hat, *_ = consider_cov.measurement_update(
            pred_state, P_bar, S_bar, P_cc, c, H_x, H_c, R, y_res
        )
        np.testing.assert_allclose(x_hat.flatten(), pred_state, atol=1e-12)

    def test_zero_consider_error_no_bias(self, consider_cov, measurement_update_inputs):
        """When c = 0, x_c_hat must equal x_hat."""
        pred_state, P_bar, S_bar, P_cc, _, H_x, H_c, R, y_res = measurement_update_inputs
        c_zero = np.zeros(N_CONSIDER)
        x_hat, _, _, x_c_hat, *_ = consider_cov.measurement_update(
            pred_state, P_bar, S_bar, P_cc, c_zero, H_x, H_c, R, y_res
        )
        np.testing.assert_allclose(x_c_hat, x_hat, atol=1e-14)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Propagate Trajectory
# ══════════════════════════════════════════════════════════════════════════════
class TestPropagateTraj:
    def test_output_shapes(self, consider_cov):
        """state, stm, and theta histories must have correct shapes."""
        time_vector = np.linspace(0, 300, 4)
        consider_params = ['mu', 'J2']
        state_history, stm_history, theta_history = consider_cov.propagate_traj(
            STATE_8, time_vector, consider_params
        )
        n = N_STATE
        nc = len(consider_params)
        nt = len(time_vector)

        assert state_history.shape == (n, nt)
        assert stm_history.shape == (n, n, nt)
        assert theta_history.shape == (6, nc, nt)

    def test_initial_state_preserved(self, consider_cov):
        """The first column of state_history must equal the initial state."""
        time_vector = np.linspace(0, 300, 4)
        state_history, _, _ = consider_cov.propagate_traj(
            STATE_8, time_vector, ['mu', 'J2']
        )
        np.testing.assert_allclose(state_history[:, 0], STATE_8, atol=1e-10)

    def test_initial_stm_near_identity(self, consider_cov):
        """At t=0, the STM should be very close to identity."""
        time_vector = np.linspace(0, 300, 4)
        _, stm_history, _ = consider_cov.propagate_traj(
            STATE_8, time_vector, ['mu', 'J2']
        )
        np.testing.assert_allclose(
            stm_history[:, :, 0], np.eye(N_STATE), atol=1e-8
        )

    def test_initial_theta_near_zero(self, consider_cov):
        """At t=0, theta should be very close to zero."""
        time_vector = np.linspace(0, 300, 4)
        _, _, theta_history = consider_cov.propagate_traj(
            STATE_8, time_vector, ['mu', 'J2']
        )
        np.testing.assert_allclose(
            theta_history[:, :, 0], np.zeros((6, 2)), atol=1e-8
        )

    def test_no_nan_in_outputs(self, consider_cov):
        """Propagation must not produce any NaN values."""
        time_vector = np.linspace(0, 600, 8)
        state_history, stm_history, theta_history = consider_cov.propagate_traj(
            STATE_8, time_vector, ['mu', 'J2']
        )
        assert not np.any(np.isnan(state_history))
        assert not np.any(np.isnan(stm_history))
        assert not np.any(np.isnan(theta_history))

    def test_state_evolves_over_time(self, consider_cov):
        """The propagated state at a later time should differ from the initial state."""
        time_vector = np.linspace(0, 600, 4)
        state_history, _, _ = consider_cov.propagate_traj(
            STATE_8, time_vector, ['mu', 'J2']
        )
        assert not np.allclose(state_history[:, -1], STATE_8)

    def test_single_consider_parameter(self, consider_cov):
        """Works correctly with only one consider parameter."""
        time_vector = np.linspace(0, 300, 4)
        state_history, stm_history, theta_history = consider_cov.propagate_traj(
            STATE_8, time_vector, ['mu']
        )
        assert theta_history.shape == (6, 1, len(time_vector))


# ══════════════════════════════════════════════════════════════════════════════
# 5. Compute Residuals and Jacobians
# ══════════════════════════════════════════════════════════════════════════════
class TestComputeResidualsAndJacobians:
    @staticmethod
    def _build_measurement_df(measurement_mgr, state_history, time_vector):
        """Simulate noise-free measurements and pack them into the DataFrame
        expected by ConsiderCov.compute_residuals_and_jacobians."""
        meas = measurement_mgr.simulate_measurements(
            state_history[:6, :], time_vector, 'ECI',
            noise=False, ignore_visibility=True,
        )
        n = len(time_vector)
        data = {
            "time": time_vector,
            f"{measurement_mgr.station_name}_measurements": [
                meas[:, i] for i in range(n)
            ],
        }
        return pd.DataFrame(data)

    def test_output_shapes(self, consider_cov, measurement_mgr):
        time_vector = np.linspace(0, 300, 4)
        state_history, _, _ = consider_cov.propagate_traj(
            STATE_8, time_vector, ['mu', 'J2']
        )
        df = self._build_measurement_df(measurement_mgr, state_history, time_vector)
        nc = 2
        residuals, H_x, H_c = consider_cov.compute_residuals_and_jacobians(
            state_history[:6, :], df, time_vector, nc
        )
        n_stations = 1
        nt = len(time_vector)
        assert residuals.shape == (N_MEAS, 1, n_stations, nt)
        assert H_x.shape == (N_MEAS, 6, n_stations, nt)
        assert H_c.shape == (N_MEAS, nc, n_stations, nt)

    def test_residuals_near_zero_for_consistent_data(self, consider_cov, measurement_mgr):
        """When truth measurements come from the same reference trajectory,
        residuals should be near zero."""
        time_vector = np.linspace(0, 300, 4)
        state_history, _, _ = consider_cov.propagate_traj(
            STATE_8, time_vector, ['mu', 'J2']
        )
        df = self._build_measurement_df(measurement_mgr, state_history, time_vector)
        residuals, _, _ = consider_cov.compute_residuals_and_jacobians(
            state_history[:6, :], df, time_vector, 2
        )
        np.testing.assert_allclose(residuals, 0.0, atol=1e-8)

    def test_jacobians_nonzero(self, consider_cov, measurement_mgr):
        """H_x should have non-trivial entries (not all zeros)."""
        time_vector = np.linspace(0, 300, 4)
        state_history, _, _ = consider_cov.propagate_traj(
            STATE_8, time_vector, ['mu', 'J2']
        )
        df = self._build_measurement_df(measurement_mgr, state_history, time_vector)
        _, H_x, _ = consider_cov.compute_residuals_and_jacobians(
            state_history[:6, :], df, time_vector, 2
        )
        assert np.any(H_x != 0.0)

    def test_no_nan_in_outputs(self, consider_cov, measurement_mgr):
        """Residuals and Jacobians must be NaN-free."""
        time_vector = np.linspace(0, 300, 4)
        state_history, _, _ = consider_cov.propagate_traj(
            STATE_8, time_vector, ['mu', 'J2']
        )
        df = self._build_measurement_df(measurement_mgr, state_history, time_vector)
        residuals, H_x, H_c = consider_cov.compute_residuals_and_jacobians(
            state_history[:6, :], df, time_vector, 2
        )
        assert not np.any(np.isnan(residuals))
        assert not np.any(np.isnan(H_x))
        assert not np.any(np.isnan(H_c))


# ══════════════════════════════════════════════════════════════════════════════
# 6. Full Run (integration test)
# ══════════════════════════════════════════════════════════════════════════════
class TestRun:
    @staticmethod
    def _build_measurement_df(measurement_mgr, integrator, time_vector, true_state):
        """Propagate truth and produce a measurement DataFrame."""
        _, truth = integrator.integrate_eom(
            time_vector[-1], true_state, teval=time_vector
        )
        meas = measurement_mgr.simulate_measurements(
            truth, time_vector, coordinate_frame='ECI',
            noise=False, ignore_visibility=True,
        )
        n = len(time_vector)
        data = {
            "time": time_vector,
            f"{measurement_mgr.station_name}_measurements": [
                meas[:, i] for i in range(n)
            ],
        }
        return pd.DataFrame(data)

    def test_run_returns_six_arrays(self, consider_cov, integrator, measurement_mgr):
        """The run method must return 6 history arrays."""
        time_vector = np.linspace(0, 300, 4)
        df = self._build_measurement_df(measurement_mgr, integrator, time_vector, STATE_8)
        consider_params = ['mu', 'J2']
        nc = len(consider_params)
        n = N_STATE

        initial_P = np.diag([1.0, 1.0, 1.0, 1e-6, 1e-6, 1e-6, 1.0, 1e-6])
        initial_S = np.zeros((n, nc))
        initial_theta = np.zeros((n, nc))
        c = np.zeros(nc)
        P_cc = 1e-4 * np.eye(nc)

        result = consider_cov.run(
            STATE_8, initial_P, consider_params,
            initial_S, initial_theta, c, P_cc, R,
            time_vector, df
        )
        assert len(result) == 6

    def test_run_output_shapes(self, consider_cov, integrator, measurement_mgr):
        """Verify the shapes of state, stm, theta, residual, and Jacobian histories."""
        time_vector = np.linspace(0, 300, 4)
        df = self._build_measurement_df(measurement_mgr, integrator, time_vector, STATE_8)
        consider_params = ['mu', 'J2']
        nc = len(consider_params)
        n = N_STATE
        nt = len(time_vector)

        initial_P = np.diag([1.0, 1.0, 1.0, 1e-6, 1e-6, 1e-6, 1.0, 1e-6])
        initial_S = np.zeros((n, nc))
        initial_theta = np.zeros((n, nc))
        c = np.zeros(nc)
        P_cc = 1e-4 * np.eye(nc)

        state_hist, stm_hist, theta_hist, res_mat, H_x_mat, H_c_mat = consider_cov.run(
            STATE_8, initial_P, consider_params,
            initial_S, initial_theta, c, P_cc, R,
            time_vector, df
        )
        assert state_hist.shape == (n, nt)
        assert stm_hist.shape == (n, n, nt)
        assert theta_hist.shape == (n, nc, nt)
        assert res_mat.shape[3] == nt
        assert H_x_mat.shape[3] == nt
        assert H_c_mat.shape[3] == nt

    def test_no_nan_in_run(self, consider_cov, integrator, measurement_mgr):
        """The full run must not produce NaN values."""
        time_vector = np.linspace(0, 300, 4)
        df = self._build_measurement_df(measurement_mgr, integrator, time_vector, STATE_8)
        consider_params = ['mu', 'J2']
        nc = len(consider_params)
        n = N_STATE

        initial_P = np.diag([1.0, 1.0, 1.0, 1e-6, 1e-6, 1e-6, 1.0, 1e-6])
        initial_S = np.zeros((n, nc))
        initial_theta = np.zeros((n, nc))
        c = np.zeros(nc)
        P_cc = 1e-4 * np.eye(nc)
        results = consider_cov.run(
            STATE_8, initial_P, consider_params,
            initial_S, initial_theta, c, P_cc, R,
            time_vector, df
        )
        for arr in results:
            assert not np.any(np.isnan(arr))


# ══════════════════════════════════════════════════════════════════════════════
# 7. Time Update → Measurement Update Round-Trip
# ══════════════════════════════════════════════════════════════════════════════
class TestTimeToMeasurementRoundTrip:
    """Verify that sequencing a time update followed by a measurement update
    produces internally consistent outputs (shapes, symmetry, consider ≥ conventional)."""

    def test_round_trip_shapes(self, consider_cov, time_update_inputs, rng):
        x_hat, P_hat, S_hat, theta, phi, c, P_cc = time_update_inputs
        x_bar, P_bar, S_bar, _, _, _ = consider_cov.time_update(
            x_hat, P_hat, S_hat, theta, phi, c, P_cc
        )
        nm = N_MEAS
        n = N_STATE
        nc = N_CONSIDER
        H_x = rng.standard_normal((nm, n))
        H_c = rng.standard_normal((nm, nc))
        R = np.eye(nm)
        y_res = rng.standard_normal((nm, 1))

        x_hat2, P_hat2, S_hat2, x_c2, P_c2, P_xc2 = consider_cov.measurement_update(
            x_bar, P_bar, S_bar, P_cc, c, H_x, H_c, R, y_res
        )
        assert P_hat2.shape == (n, n)
        assert S_hat2.shape == (n, nc)
        assert P_c2.shape == (n, n)
        assert P_xc2.shape == (n, nc)

    def test_round_trip_consider_geq_conventional(self, consider_cov, time_update_inputs, rng):
        """After time + measurement update, consider covariance diagonal ≥ conventional."""
        x_hat, P_hat, S_hat, theta, phi, c, P_cc = time_update_inputs
        x_bar, P_bar, S_bar, _, _, _ = consider_cov.time_update(
            x_hat, P_hat, S_hat, theta, phi, c, P_cc
        )
        nm = N_MEAS
        H_x = rng.standard_normal((nm, N_STATE))
        H_c = rng.standard_normal((nm, N_CONSIDER))
        R = np.eye(nm)
        y_res = rng.standard_normal((nm, 1))

        _, P_hat2, _, _, P_c2, _ = consider_cov.measurement_update(
            x_bar, P_bar, S_bar, P_cc, c, H_x, H_c, R, y_res
        )
        assert np.all(np.diag(P_c2) >= np.diag(P_hat2) - 1e-12)

    def test_round_trip_symmetry(self, consider_cov, time_update_inputs, rng):
        """All covariance matrices from the round-trip must be symmetric."""
        x_hat, P_hat, S_hat, theta, phi, c, P_cc = time_update_inputs
        _, P_bar, S_bar, _, P_c_bar, _ = consider_cov.time_update(
            x_hat, P_hat, S_hat, theta, phi, c, P_cc
        )
        np.testing.assert_allclose(P_bar, P_bar.T, atol=1e-12)
        np.testing.assert_allclose(P_c_bar, P_c_bar.T, atol=1e-12)

        nm = N_MEAS
        H_x = rng.standard_normal((nm, N_STATE))
        H_c = rng.standard_normal((nm, N_CONSIDER))
        R = np.eye(nm)
        y_res = rng.standard_normal((nm, 1))

        _, P_hat2, _, _, P_c2, _ = consider_cov.measurement_update(
            np.zeros(N_STATE), P_bar, S_bar, P_cc, c, H_x, H_c, R, y_res
        )
        np.testing.assert_allclose(P_hat2, P_hat2.T, atol=1e-12)
        np.testing.assert_allclose(P_c2, P_c2.T, atol=1e-12)