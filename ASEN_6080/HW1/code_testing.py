import numpy as np
import json
from generic_functions import perturbation_jacobian
from integrator import Integrator

# Question 1 Testing Code ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Read in test data from prob1c_solution.json
with open('prob1c_solution.json', 'r') as f:
    test_data = json.load(f)

# Pull out the necessary parameters
state = test_data['inputs']['state']
truth_jacobian = np.array(test_data['outputs']['A_matrix']['values'])
r = np.array(state['r'])
v = np.array(state['v'])
mu = state['mu']
J2 = state['J2']
J3 = state['J3']

R_e = 6378

# Call the perturbation_partials function

A = perturbation_jacobian(r, v, mu, J2, J3, R_e)

# Compare the computed Jacobian to the truth Jacobian
diff = A - truth_jacobian
print("Difference between computed and truth Jacobian:")
np.set_printoptions(linewidth=200)
print(diff)

# Question 2 Testing Code --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

a = 10000
e = 0.01
i = np.deg2rad(40)
LoN = np.deg2rad(80)
AoP = np.deg2rad(40)
f = np.deg2rad(0)
mu = 398600.4418
period = 2 * np.pi * np.sqrt(a**3 / mu)
# Create an Integrator instance and convert state to Cartesian
integrator = Integrator(mu, R_e, 'J2')
r_vec, v_vec = integrator.keplerian_to_cartesian(a, e, i, LoN, AoP, f)

initial_state = np.hstack((r_vec, v_vec, 0))

# Integrate for 15 orbital periods
reference_time, reference_state_history = integrator.integrate(15 * period, initial_state)

breakpoint()
