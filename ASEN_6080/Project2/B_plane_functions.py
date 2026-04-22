import numpy as np
from Tools.B_Plane_manager import BPlaneMgr
from generic_functions import initialize_integrator
from Tools.generic_functions import covariance_ellipse_2D
import plotly.graph_objects as go
from constants import mu_earth, RSOI

def event_3RSOI_crossing(t, y, DMC, beta_mat):
    """
    Event function to detect when the spacecraft crosses 3 times the radius of the sphere of influence (RSOI) of Earth.

    Parameters
    ----------
    t : float
        The current time in seconds.
    y : np.ndarray
        The current state vector of the spacecraft (position and velocity).
    RSOI : float
        The radius of the sphere of influence for Earth in km (default is 925,000 km).

    Returns
    -------
    float
        The value of the event function, which is the distance from Earth minus 3 times the RSOI. When this value crosses zero, it indicates a crossing event.
    """
    r_vec = y[:3]  # Extract position vector from state vector
    r_norm = np.linalg.norm(r_vec)  # Compute the norm of the position vector

    return r_norm - 3*RSOI  # Return the distance from Earth minus 3 times the RSOI

def integrate_to_3RSOI(DCO_state, DCO_epoch, C_r, t_final = 1e8):
    """
    Integrate the spacecraft's trajectory from the provided DCO state to the point where it crosses 3 times the radius of the sphere of influence (RSOI) of Earth.

    Parameters
    ----------
    DCO_state : np.ndarray
        The initial state vector of the spacecraft (position and velocity).
    t_final : float
        The maximum time to integrate to in seconds (default is 1e8 seconds). This should be sufficiently large to ensure that the crossing event occurs within this time frame.
    Returns
    -------
    tuple
        A tuple containing the time of RSOI crossing and the state vector at the time of crossing.
    """
    integrator = initialize_integrator(DCO_epoch, input_C_r=C_r, estimation_mode=['SRP'], parameter_indices=[6])

    # Use solve_ivp with the event function to integrate until crossing 3*RSOI
    t_events, y_events = integrator.integrate_stm(t_final, DCO_state, events=event_3RSOI_crossing)

    if len(t_events[0]) > 0:  # Check if the event was triggered
        crossing_time = t_events[0][0]  # Get the time of crossing
        crossing_state = y_events[0][0]  # Get the state at the time of crossing
        return crossing_time, crossing_state[:len(DCO_state)]  # Return the state vector (position and velocity) at crossing
    else:
        raise RuntimeError("Integration did not reach the 3*RSOI crossing event within the specified time span.")
    
def get_LTOF_to_B_Plane(v_inf_state):
    """
    Compute the LTOF to the B-plane given the hyperbolic excess velocity state vector at RSOI.

    Parameters
    ----------
    v_inf_state : np.ndarray
        The state vector of the spacecraft at RSOI, which should include position and velocity components.
    mu : float
        The gravitational parameter of Earth in km^3/s^2 (default is 3.986004415E5 km^3/s^2).

    Returns
    -------
    float
        The LTOF to the B-plane in seconds.
    """
    b_plane_manager = BPlaneMgr(v_inf_state, mu_earth)
    LTOF = b_plane_manager.compute_LOTF()

    return LTOF

def integrate_to_B_plane_crossing(DCO_state : np.ndarray, DCO_epoch : float, B_plane_crossing_epoch : float):
    """
    Integrate the spacecraft's trajectory from the provided initial state to the point where it crosses the B-plane.

    Parameters
    ----------
    DCO_state : np.ndarray
        The DCO state vector of the spacecraft (position and velocity).
    DCO_epoch : float
        The DCO epoch corresponding to the DCO state in seconds.
    B_plane_crossing_epoch : float
        The epoch at which the spacecraft is expected to cross the B-plane in seconds. This can be computed as the DCO epoch plus the LTOF to the B-plane.

    Returns
    -------
    tuple
        A tuple containing the time of B-plane crossing, the state vector, and the STM at the time of crossing.
    """
    # Initialize integrator with the initial epoch
    integrator = initialize_integrator(DCO_epoch, estimation_mode=['SRP'], parameter_indices=[6])

    # Integrate the trajectory to the B-plane crossing epoch
    sol = integrator.integrate_stm(B_plane_crossing_epoch, DCO_state, teval=np.array([B_plane_crossing_epoch]))

    crossing_time = sol[0][0]  # Get the time of crossing
    augmented_crossing_state = sol[1][:, 0]  # Get the state at the time of crossing

    state_length = len(DCO_state)
    crossing_state = augmented_crossing_state[:state_length]  # Extract the state vector (position and velocity) from the augmented state
    crossing_stm = augmented_crossing_state[state_length:].reshape((state_length, state_length))  # Extract the STM from the augmented state

    return crossing_time, crossing_state, crossing_stm

def perform_B_plane_analysis(DCO_state, DCO_epoch, DCO_covariance, C_r, fig, color, time, filter_name):
    """
    Perform the full B-plane analysis by integrating to 3*RSOI crossing, computing the LTOF to the B-plane, and integrating to the B-plane crossing.

    Parameters
    ----------
    DCO_state : np.ndarray
        The initial state vector of the spacecraft (position and velocity).
    DCO_epoch : float
        The initial epoch corresponding to the DCO state in seconds.
    C_r : float
        The coefficient of reflectivity for solar radiation pressure.

    Returns
    -------
    dict
        A dictionary containing the results of the B-plane analysis, including the time and state at 3*RSOI crossing, LTOF to the B-plane, and time, state, and STM at B-plane crossing.
    """    
    # Integrate to 3*RSOI crossing and get LTOF to B-plane
    RSOI_crossing_time, RSOI_crossing_state = integrate_to_3RSOI(DCO_state, DCO_epoch, C_r)

    LTOF_to_B_plane = get_LTOF_to_B_Plane(RSOI_crossing_state)
    
    # Integrate to B-plane crossing
    B_plane_crossing_epoch = RSOI_crossing_time + LTOF_to_B_plane
    B_plane_crossing_time, B_plane_crossing_state, B_plane_crossing_stm = integrate_to_B_plane_crossing(DCO_state, DCO_epoch, B_plane_crossing_epoch)

    # Map final filter covariance to B-plane crossing time using the STM
    B_plane_crossing_covariance = B_plane_crossing_stm @ DCO_covariance @ B_plane_crossing_stm.T  # Propagate the covariance to the B-plane crossing time using the STM

    # Project the RSOI crossing state into 
    print("B-plane Crossing State:", B_plane_crossing_state)
    print("B-plane Crossing Covariance:", B_plane_crossing_covariance)

    # Rotate the B-plane crossing covariance into the B-plane frame
    b_plane_manager = BPlaneMgr(RSOI_crossing_state[:6], mu_earth)
    DCM_ECI_to_B_plane = b_plane_manager.compute_b_plane_DCM()
    B_plane_crossing_pos_covariance_in_B_plane_frame = DCM_ECI_to_B_plane @ B_plane_crossing_covariance[:3,:3] @ DCM_ECI_to_B_plane.T

    # B-Plane target is defined by removing the s_hat (v_hat) component of the position vector at the RSOI crossing point (Odd but how professor wants us to do it)
    B_plane_RSOI_crossing_pos = DCM_ECI_to_B_plane @ RSOI_crossing_state[0:3]

    center = B_plane_RSOI_crossing_pos[1:3]
    reduced_covariance = B_plane_crossing_pos_covariance_in_B_plane_frame[1:3, 1:3]  # The covariance for the ellipse is given by the y and z components of the covariance in the B-plane frame
    b_plane_covariance_ellipse = covariance_ellipse_2D(center, reduced_covariance, n_std=3)  # Compute the covariance ellipse at 3-sigma

    fig.add_trace(go.Scatter(x=[center[0]], y=[center[1]], mode='markers', name=f'{time} days', marker=dict(color=color, size=10)))
    fig.add_trace(go.Scatter(x=b_plane_covariance_ellipse[:, 0], y=b_plane_covariance_ellipse[:, 1], mode='lines', name=f'{time} days', marker=dict(color=color), showlegend=False))