import numpy as np
import json
from perturbation_jacobian import perturbation_jacobian

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