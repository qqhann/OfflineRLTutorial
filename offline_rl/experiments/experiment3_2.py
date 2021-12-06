import numpy as np

from offline_rl.nn import FCNetwork
from offline_rl.utils import compute_policy_deterministic, compute_visitation
from offline_rl.algorithm.fitted_q_iter import fitted_q_iteration
from offline_rl.algorithm.conservative_q_iter import conservative_q_iteration
from offline_rl.plot import plot_sa_values, plot_s_values

from offline_rl.experiments.parameters import weighting_only
from offline_rl.experiments.prepare import (
    env,
    optimal_qvalues,
    sa_visitations,
    training_dataset,
)

# Weighting distribution
weights = sa_visitations


# @title Run conservative Q-iteration (or CQL) with finite data

# Use a tabular or feedforward NN approximator
network = FCNetwork(env, layers=[20, 20])
# network = TabularNetwork(env)

cql_alpha_val = 0.1  # @param {type:"slider", min:0.0, max:10.0, step:0.01}

print(weighting_only)
# Run Q-iteration
q_values = conservative_q_iteration(
    env,
    network,
    num_itrs=100,
    discount=0.95,
    cql_alpha=cql_alpha_val,
    weights=weights,
    render=True,
    sampled=not (weighting_only),
    training_dataset=training_dataset,
)

# Compute and plot the value function
v_values = np.max(q_values, axis=1)
plot_s_values(env, v_values, title="Values")


# @title Plot Q-functions, overestimation error

print("Total Error:", np.sum(np.abs(q_values - optimal_qvalues)))

# Compute over-estimation in the training distribution
total_overestimation = np.sum((q_values - optimal_qvalues) * weights)
print(
    "Total Weighted Overestimation under the training distribution: ",
    total_overestimation,
)

# Compute over-estimation under the resulting policy
policy = compute_policy_deterministic(q_values, eps_greedy=0.1)
policy_sa_visitations = compute_visitation(env, policy)
weights_policy = policy_sa_visitations / np.sum(policy_sa_visitations)
total_policy_overestimation = np.sum((q_values - optimal_qvalues) * weights_policy)
print("Total Overestimation under the learned policy: ", total_policy_overestimation)

# Compute unweighted overestimation
total_overestimation_unweighted = np.mean((q_values - optimal_qvalues))
print("Total Overestimation: ", total_overestimation_unweighted)

plot_sa_values(env, (q_values - optimal_qvalues), title="Q-function Error (Q - Q*)")


# @title Compute visitations of the learned policy
policy = compute_policy_deterministic(q_values, eps_greedy=0.1)
policy_sa_visitations = compute_visitation(env, policy)
plot_sa_values(env, policy_sa_visitations, title="Q-hat_CQL Visitation")
