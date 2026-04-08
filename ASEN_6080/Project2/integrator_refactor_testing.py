"""
Pytest suite to verify that the refactored integrator.py produces the same
numerical results as integrator_old.py for every shared code-path.

Run with:
    pytest tests/test_integrator_equivalence.py -v

NOTE: Both modules live inside the `Tools` package, which uses relative
imports.  We therefore import the package-level names and alias the old
class via a small importlib trick so both classes can coexist.
"""

import importlib
import sys
import types
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Bootstrap: make the old integrator importable alongside the new one
# ---------------------------------------------------------------------------

# Import the real (new) Integrator through the package
from Tools.integrator import Integrator as NewIntegrator
from Tools.generic_functions import compute_density

# Import the old integrator under a different name.  Because the file uses
# relative imports (from .generic_functions …) we need to load it as a
# sub-module of the Tools package.
_spec = importlib.util.spec_from_file_location(
    "Tools.integrator_old", "Tools/integrator_old.py",
    submodule_search_locations=[]
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["Tools.integrator_old"] = _mod
_spec.loader.exec_module(_mod)
OldIntegrator = _mod.Integrator


# ---------------------------------------------------------------------------
# Shared physical constants / test state
# ---------------------------------------------------------------------------
MU = 398600.4418            # km^3/s^2
R_E = 6378.137              # km
J2 = 1.08263e-3
J3 = -2.5327e-6
CD = 2.0
CR = 1.5
AREA = 3.0                  # m^2
MASS = 970.0                # kg
EPOCH_JD = 2451545.0        # J2000

# A typical LEO state [x, y, z, u, v, w] in km & km/s
STATE_6 = np.array([
    -2436.45, -2436.45, 6891.037,
     5.088611, -5.088611, 5.0
])

T_FINAL = 600.0             # 10-minute propagation (short, keeps CI fast)
TEVAL = np.linspace(0, T_FINAL, 50)

# Tolerance – both integrators use rtol/atol = 1e-13, so results should
# agree to essentially machine precision over a short arc.
ATOL = 1e-10
RTOL = 0


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _make_old(mode, parameter_indices, Cd=0, J2_val=0, J3_val=0,
              area=None, mass=None, n_stations=0, epoch=0):
    """Instantiate the OLD integrator."""
    return OldIntegrator(
        mu=MU, R_e=R_E, J2=J2_val, J3=J3_val, Cd=Cd,
        mode=mode, parameter_indices=parameter_indices,
        spacecraft_area=area, spacecraft_mass=mass,
        number_of_stations=n_stations, initial_epoch=epoch,
    )


def _make_new(dynamical_mode, estimation_mode, parameter_indices,
              Cd=None, Cr=None, J2_val=None, J3_val=None,
              area=None, mass=None, n_stations=0, epoch=0):
    """Instantiate the NEW integrator."""
    return NewIntegrator(
        mu=MU, R_e=R_E, J2=J2_val, J3=J3_val, Cd=Cd, Cr=Cr,
        dynamical_mode=dynamical_mode,
        estimation_mode=estimation_mode,
        parameter_indices=parameter_indices,
        spacecraft_area=area, spacecraft_mass=mass,
        number_of_stations=n_stations, initial_epoch=epoch,
    )


# ===================================================================
# 1. equations_of_motion – point-mass + J2 + J3 (no estimation)
# ===================================================================

class TestEquationsOfMotion:
    """Compare single-step state derivatives between old and new."""

    def _old_eom_no_estimation(self, state, t=0.0):
        """Old integrator with J2/J3 in dynamics but nothing estimated."""
        old = _make_old(mode=[], parameter_indices=[],
                        J2_val=J2, J3_val=J3)
        return old.equations_of_motion(t, state)

    def _new_eom_no_estimation(self, state, t=0.0):
        """New integrator with mu+J2+J3 in dynamics but nothing estimated."""
        new = _make_new(
            dynamical_mode=['mu', 'J2', 'J3'],
            estimation_mode=[], parameter_indices=[],
            J2_val=J2, J3_val=J3,
        )
        return new.equations_of_motion(t, state)

    def test_point_mass_j2_j3_derivative(self):
        """State derivative must match for a basic mu+J2+J3 force model."""
        old_dot = self._old_eom_no_estimation(STATE_6)
        new_dot = self._new_eom_no_estimation(STATE_6)
        np.testing.assert_allclose(new_dot, old_dot, atol=ATOL, rtol=RTOL,
                                   err_msg="EOM mismatch (mu+J2+J3, no estimation)")

    def test_with_drag(self):
        """State derivative with drag must match."""
        old = _make_old(mode=[], parameter_indices=[],
                        Cd=CD, J2_val=J2, J3_val=J3,
                        area=AREA, mass=MASS)
        new = _make_new(dynamical_mode=['mu', 'J2', 'J3', 'Drag'],
                        estimation_mode=[], parameter_indices=[],
                        Cd=CD, J2_val=J2, J3_val=J3,
                        area=AREA, mass=MASS)
        old_dot = old.equations_of_motion(0, STATE_6)
        new_dot = new.equations_of_motion(0, STATE_6)
        np.testing.assert_allclose(new_dot, old_dot, atol=ATOL, rtol=RTOL,
                                   err_msg="EOM mismatch with drag")


# ===================================================================
# 2. equations_of_motion – with parameters in state (estimation)
# ===================================================================

class TestEOMWithEstimation:
    """When parameters are estimated they live in the state vector."""

    def test_estimate_mu(self):
        """Estimating mu: the mu value comes from state[6]."""
        state = np.append(STATE_6, MU)  # mu at index 6
        old = _make_old(mode=['mu'], parameter_indices=[6],
                        J2_val=J2, J3_val=J3)
        new = _make_new(dynamical_mode=['mu', 'J2', 'J3'],
                        estimation_mode=['mu'], parameter_indices=[6],
                        J2_val=J2, J3_val=J3)
        old_dot = old.equations_of_motion(0, state)
        new_dot = new.equations_of_motion(0, state)
        np.testing.assert_allclose(new_dot, old_dot, atol=ATOL, rtol=RTOL,
                                   err_msg="EOM mismatch (estimate mu)")

    def test_estimate_j2(self):
        """Estimating J2: the J2 value comes from state[6]."""
        state = np.append(STATE_6, J2)
        old = _make_old(mode=['J2'], parameter_indices=[6],
                        J2_val=J2, J3_val=J3)
        new = _make_new(dynamical_mode=['mu', 'J2', 'J3'],
                        estimation_mode=['J2'], parameter_indices=[6],
                        J2_val=J2, J3_val=J3)
        old_dot = old.equations_of_motion(0, state)
        new_dot = new.equations_of_motion(0, state)
        np.testing.assert_allclose(new_dot, old_dot, atol=ATOL, rtol=RTOL,
                                   err_msg="EOM mismatch (estimate J2)")

    def test_estimate_drag(self):
        """Estimating Cd: the Cd value comes from state[6]."""
        state = np.append(STATE_6, CD)
        old = _make_old(mode=['Drag'], parameter_indices=[6],
                        Cd=CD, J2_val=J2, J3_val=J3,
                        area=AREA, mass=MASS)
        new = _make_new(dynamical_mode=['mu', 'J2', 'J3', 'Drag'],
                        estimation_mode=['Drag'], parameter_indices=[6],
                        Cd=CD, J2_val=J2, J3_val=J3,
                        area=AREA, mass=MASS)
        old_dot = old.equations_of_motion(0, state)
        new_dot = new.equations_of_motion(0, state)
        np.testing.assert_allclose(new_dot, old_dot, atol=ATOL, rtol=RTOL,
                                   err_msg="EOM mismatch (estimate Cd)")

    def test_estimate_mu_j2_j3(self):
        """Estimating mu, J2, J3 together."""
        state = np.concatenate([STATE_6, [MU, J2, J3]])
        old = _make_old(mode=['mu', 'J2', 'J3'],
                        parameter_indices=[6, 7, 8],
                        J2_val=J2, J3_val=J3)
        new = _make_new(dynamical_mode=['mu', 'J2', 'J3'],
                        estimation_mode=['mu', 'J2', 'J3'],
                        parameter_indices=[6, 7, 8],
                        J2_val=J2, J3_val=J3)
        old_dot = old.equations_of_motion(0, state)
        new_dot = new.equations_of_motion(0, state)
        np.testing.assert_allclose(new_dot, old_dot, atol=ATOL, rtol=RTOL,
                                   err_msg="EOM mismatch (estimate mu+J2+J3)")


# ===================================================================
# 3. equations_of_motion – DMC (dynamic model compensation)
# ===================================================================

class TestDMC:
    """DMC adds 3 extra state variables and damping terms."""

    def test_dmc_terms(self):
        beta = np.diag([1e-4, 1e-4, 1e-4])
        w_init = np.array([1e-9, -1e-9, 5e-10])
        state_dmc = np.concatenate([STATE_6, w_init])
        old = _make_old(mode=[], parameter_indices=[],
                        J2_val=J2, J3_val=J3)
        new = _make_new(dynamical_mode=['mu', 'J2', 'J3'],
                        estimation_mode=[], parameter_indices=[],
                        J2_val=J2, J3_val=J3)
        old_dot = old.equations_of_motion(0, state_dmc, DMC=True, beta_mat=beta)
        new_dot = new.equations_of_motion(0, state_dmc, DMC=True, beta_mat=beta)
        np.testing.assert_allclose(new_dot, old_dot, atol=ATOL, rtol=RTOL,
                                   err_msg="EOM mismatch with DMC")


# ===================================================================
# 4. integrate_eom – full RK45 propagation
# ===================================================================

class TestIntegrateEOM:
    """End-to-end propagation comparison."""

    def test_propagation_no_estimation(self):
        """10-min propagation, mu+J2+J3, no estimation."""
        old = _make_old(mode=[], parameter_indices=[],
                        J2_val=J2, J3_val=J3)
        new = _make_new(dynamical_mode=['mu', 'J2', 'J3'],
                        estimation_mode=[], parameter_indices=[],
                        J2_val=J2, J3_val=J3)
        t_old, y_old = old.integrate_eom(T_FINAL, STATE_6.copy(), teval=TEVAL)
        t_new, y_new = new.integrate_eom(T_FINAL, STATE_6.copy(), teval=TEVAL)
        np.testing.assert_allclose(t_new, t_old, atol=1e-12,
                                   err_msg="Time vectors differ")
        np.testing.assert_allclose(y_new, y_old, atol=ATOL, rtol=RTOL,
                                   err_msg="State histories differ (propagation)")

    def test_propagation_with_drag(self):
        """10-min propagation with drag."""
        old = _make_old(mode=[], parameter_indices=[],
                        Cd=CD, J2_val=J2, J3_val=J3,
                        area=AREA, mass=MASS)
        new = _make_new(dynamical_mode=['mu', 'J2', 'J3', 'Drag'],
                        estimation_mode=[], parameter_indices=[],
                        Cd=CD, J2_val=J2, J3_val=J3,
                        area=AREA, mass=MASS)
        t_old, y_old = old.integrate_eom(T_FINAL, STATE_6.copy(), teval=TEVAL)
        t_new, y_new = new.integrate_eom(T_FINAL, STATE_6.copy(), teval=TEVAL)
        np.testing.assert_allclose(y_new, y_old, atol=ATOL, rtol=RTOL,
                                   err_msg="Propagation mismatch with drag")


# ===================================================================
# 5. full_dynamics (STM propagation)
# ===================================================================

class TestFullDynamics:
    """Compare the augmented state derivative (state + STM)."""

    def test_stm_derivative_no_estimation(self):
        """STM derivative must match (no estimation parameters)."""
        old = _make_old(mode=[], parameter_indices=[],
                        J2_val=J2, J3_val=J3)
        new = _make_new(dynamical_mode=['mu', 'J2', 'J3'],
                        estimation_mode=[], parameter_indices=[],
                        J2_val=J2, J3_val=J3)
        n = 6
        phi0 = np.eye(n).flatten()
        aug = np.concatenate([STATE_6, phi0])
        old_dot = old.full_dynamics(0, aug)
        new_dot = new.full_dynamics(0, aug)
        np.testing.assert_allclose(new_dot, old_dot, atol=ATOL, rtol=RTOL,
                                   err_msg="full_dynamics mismatch (no estimation)")

    def test_stm_derivative_estimate_mu_j2(self):
        """STM derivative with mu and J2 estimated."""
        state_est = np.concatenate([STATE_6, [MU, J2]])  # indices 6, 7
        old = _make_old(mode=['mu', 'J2'], parameter_indices=[6, 7],
                        J2_val=J2, J3_val=J3)
        new = _make_new(dynamical_mode=['mu', 'J2', 'J3'],
                        estimation_mode=['mu', 'J2'],
                        parameter_indices=[6, 7],
                        J2_val=J2, J3_val=J3)
        
        n = 8  # 6 + mu + J2
        phi0 = np.eye(n).flatten()
        aug = np.concatenate([state_est, phi0])
        old_dot = old.full_dynamics(0, aug)
        new_dot = new.full_dynamics(0, aug)
        np.testing.assert_allclose(new_dot, old_dot, atol=ATOL, rtol=RTOL,
                                   err_msg="full_dynamics mismatch (estimate mu+J2)")

    def test_stm_derivative_with_dmc(self):
        """STM derivative with DMC enabled."""
        beta = np.diag([1e-4, 1e-4, 1e-4])
        w_init = np.array([1e-9, -1e-9, 5e-10])
        state_dmc = np.concatenate([STATE_6, w_init])
        old = _make_old(mode=[], parameter_indices=[],
                        J2_val=J2, J3_val=J3)
        new = _make_new(dynamical_mode=['mu', 'J2', 'J3'],
                        estimation_mode=[], parameter_indices=[],
                        J2_val=J2, J3_val=J3)
        n = 9  # 6 + 3 DMC
        phi0 = np.eye(n).flatten()
        aug = np.concatenate([state_dmc, phi0])
        old_dot = old.full_dynamics(0, aug, DMC=True, beta_mat=beta)
        new_dot = new.full_dynamics(0, aug, DMC=True, beta_mat=beta) 
        
        np.testing.assert_allclose(new_dot, old_dot, atol=ATOL, rtol=RTOL,
                                   err_msg="full_dynamics mismatch with DMC")


# ===================================================================
# 6. integrate_stm – full RK45 STM propagation
# ===================================================================

class TestIntegrateSTM:
    """Full numerical integration of state + STM."""

    def test_stm_integration_no_estimation(self):
        old = _make_old(mode=[], parameter_indices=[],
                        J2_val=J2, J3_val=J3)
        new = _make_new(dynamical_mode=['mu', 'J2', 'J3'],
                        estimation_mode=[], parameter_indices=[],
                        J2_val=J2, J3_val=J3)
        t_old, y_old = old.integrate_stm(T_FINAL, STATE_6.copy(), teval=TEVAL)
        t_new, y_new = new.integrate_stm(T_FINAL, STATE_6.copy(), teval=TEVAL)
        np.testing.assert_allclose(t_new, t_old, atol=1e-12)
        np.testing.assert_allclose(y_new, y_old, atol=ATOL, rtol=RTOL,
                                   err_msg="integrate_stm mismatch (no estimation)")

    def test_stm_integration_estimate_mu(self):
        state_est = np.concatenate([STATE_6, [MU]])
        old = _make_old(mode=['mu'], parameter_indices=[6],
                        J2_val=J2, J3_val=J3)
        new = _make_new(dynamical_mode=['mu', 'J2', 'J3'],
                        estimation_mode=['mu'], parameter_indices=[6],
                        J2_val=J2, J3_val=J3)
        t_old, y_old = old.integrate_stm(T_FINAL, state_est.copy(), teval=TEVAL)
        t_new, y_new = new.integrate_stm(T_FINAL, state_est.copy(), teval=TEVAL)
        np.testing.assert_allclose(y_new, y_old, atol=ATOL, rtol=RTOL,
                                   err_msg="integrate_stm mismatch (estimate mu)")


# ===================================================================
# 7. Sigma-point propagation (UKF)
# ===================================================================

class TestSigmaPoints:
    """The new code splits sigma-point handling into its own method."""

    def test_sigma_eom(self):
        """Build a small set of sigma points and compare derivatives."""
        L = 6
        num_sigma = 2 * L + 1
        # Create sigma points as small perturbations of the nominal state
        rng = np.random.default_rng(42)
        sigma_states = np.tile(STATE_6, num_sigma)
        sigma_states += rng.normal(0, 1e-6, sigma_states.shape)

        old = _make_old(mode=[], parameter_indices=[],
                        J2_val=J2, J3_val=J3)
        new = _make_new(dynamical_mode=['mu', 'J2', 'J3'],
                        estimation_mode=[], parameter_indices=[],
                        J2_val=J2, J3_val=J3)
        # Old path: equations_of_motion with sigma_points=True
        old_dot = old.equations_of_motion(0, sigma_states, sigma_points=True)
        # New path: dedicated sigma_points_eom method
        new_dot = new.sigma_points_eom(0, sigma_states)
        np.testing.assert_allclose(new_dot, old_dot, atol=ATOL, rtol=RTOL,
                                   err_msg="Sigma-point EOM mismatch")


# ===================================================================
# 8. Individual effect methods (new code only – sanity checks)
# ===================================================================

class TestIndividualEffects:
    """Verify that the sum of individual effects in the new code
    reproduces the monolithic old EOM result."""

    def test_sum_of_effects_equals_old(self):
        """get_mu + get_J2 + get_J3 should equal old EOM output."""
        new = _make_new(dynamical_mode=['mu', 'J2', 'J3'],
                        estimation_mode=[], parameter_indices=[],
                        J2_val=J2, J3_val=J3)
        mu_eff = new.get_mu_effects(STATE_6, MU)
        j2_eff = new.get_J2_effects(STATE_6, MU, J2)
        j3_eff = new.get_J3_effects(STATE_6, MU, J3)
        combined = mu_eff + j2_eff + j3_eff

        old = _make_old(mode=[], parameter_indices=[],
                        J2_val=J2, J3_val=J3)
        old_dot = old.equations_of_motion(0, STATE_6)
        np.testing.assert_allclose(combined, old_dot, atol=ATOL, rtol=RTOL,
                                   err_msg="Sum of individual effects != old EOM")

    def test_drag_effect_matches(self):
        """get_drag_effects should match the drag portion of the old EOM."""
        old_no_drag = _make_old(mode=[], parameter_indices=[],
                                J2_val=J2, J3_val=J3)
        old_with_drag = _make_old(mode=[], parameter_indices=[],
                                  Cd=CD, J2_val=J2, J3_val=J3,
                                  area=AREA, mass=MASS)
        drag_old = (old_with_drag.equations_of_motion(0, STATE_6)
                    - old_no_drag.equations_of_motion(0, STATE_6))

        new = _make_new(dynamical_mode=['mu', 'J2', 'J3', 'Drag'],
                        estimation_mode=[], parameter_indices=[],
                        Cd=CD, J2_val=J2, J3_val=J3,
                        area=AREA, mass=MASS)
        drag_new = new.get_drag_effects(STATE_6, CD,
                                        new.spacecraft_area,
                                        new.spacecraft_mass)
        np.testing.assert_allclose(drag_new, drag_old, atol=ATOL, rtol=RTOL,
                                   err_msg="Drag effect mismatch")


# ===================================================================
# 9. Edge cases and constructor validation
# ===================================================================

class TestValidation:
    """Verify that both integrators raise on invalid inputs."""

    def test_old_mismatched_mode_indices(self):
        with pytest.raises(ValueError, match="Length"):
            _make_old(mode=['mu'], parameter_indices=[])

    def test_new_mismatched_mode_indices(self):
        with pytest.raises(ValueError, match="Length"):
            _make_new(dynamical_mode=['mu'],
                      estimation_mode=['mu', 'J2'],
                      parameter_indices=[6])

    def test_old_drag_without_area(self):
        with pytest.raises(ValueError):
            _make_old(mode=['Drag'], parameter_indices=[6],
                      Cd=CD, area=None, mass=MASS)

    def test_new_drag_without_area(self):
        with pytest.raises(ValueError):
            _make_new(dynamical_mode=['mu', 'Drag'],
                      estimation_mode=['Drag'], parameter_indices=[6],
                      Cd=CD, area=None, mass=MASS)

    def test_new_empty_dynamical_mode(self):
        with pytest.raises(ValueError, match="At least one"):
            _make_new(dynamical_mode=[],
                      estimation_mode=[], parameter_indices=[])

    def test_old_stations_zero(self):
        with pytest.raises(ValueError, match="Number of stations"):
            _make_old(mode=['Stations'], parameter_indices=[6],
                      n_stations=0)

    def test_new_stations_zero(self):
        with pytest.raises(ValueError, match="Number of stations"):
            _make_new(dynamical_mode=['mu'],
                      estimation_mode=['Stations'], parameter_indices=[6],
                      n_stations=0)