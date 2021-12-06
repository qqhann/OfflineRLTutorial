import numpy as np

from offline_rl.experiments.parameters import (
    env_type,
    dataset_composition,
    weighting_only,
    dataset_size,
)
from offline_rl.env import GridSpec, spec_from_string, GridEnv
from offline_rl.plot import plot_sa_values
from offline_rl.utils import compute_policy_deterministic, compute_visitation
from offline_rl.algorithm.tabular_q_iter import q_iteration


# @title Define environment and compute optimal Q-values
# SOOOOOO#\
# O##O###O\
# OO#OO##O\
# O#RO#OO#\
maze: GridSpec = spec_from_string(
    "SOOOOOO#\\" + "O##O###O\\" + "OO#OO##O\\" + "O#RO#OO#\\"
)

env: GridEnv = GridEnv(maze, observation_type=env_type, dim_obs=8)
optimal_qvalues = q_iteration(env, num_itrs=100, discount=0.95, render=False)

plot_sa_values(env, optimal_qvalues, title="Q*-values")

policy = compute_policy_deterministic(optimal_qvalues, eps_greedy=0.1)
sa_visitations = compute_visitation(env, policy)
plot_sa_values(env, sa_visitations, title="Optimal policy state-action visitation")


# @title Compute weights
if dataset_composition == "optimal":
    """Distribution of the optimal policy (+ some noise)"""
    weights = sa_visitations
    weights = weights / np.sum(weights)
elif dataset_composition == "random":
    """A random distribution over states and actions"""
    weights = np.random.uniform(size=env.num_states * env.num_actions)
    weights = np.reshape(weights, (env.num_states, env.num_actions))
    weights = weights / np.sum(weights)
elif dataset_composition == "random+optimal":
    """Mixture of random and optimal policies"""
    weights = sa_visitations / np.sum(sa_visitations)
    weights_rand = np.random.uniform(size=env.num_states * env.num_actions)
    weights_rand = np.reshape(weights_rand, (env.num_states, env.num_actions)) / np.sum(
        weights_rand
    )
    weights = (weights_rand + weights) / 2.0
elif dataset_composition == "mixed":
    """Mixture of policies corresponding to random Q-values"""
    num_policies_mix = 4
    weights = np.zeros_like(sa_visitations)
    for idx in range(num_policies_mix):
        rand_q_vals_idx = np.random.uniform(
            low=0.0, high=10.0, size=(env.num_states, env.num_actions)
        )
        policy_idx = compute_policy_deterministic(rand_q_vals_idx, eps_greedy=0.1)
        sa_visitations_idx = compute_visitation(env, policy_idx)
        weights = weights + sa_visitations_idx
    weights = weights / np.sum(weights)


# @title Generate dataset
if not weighting_only:
    weights_flatten = np.reshape(weights, -1)
    weights_flatten = weights_flatten / np.sum(weights_flatten)
    dataset = np.random.choice(
        np.arange(env.num_states * env.num_actions),
        size=dataset_size,
        replace=True,
        p=weights_flatten,
    )
    training_sa_pairs = [
        (int(val // env.num_actions), val % env.num_actions) for val in dataset
    ]

    # Now sample (s', r) values for training as well
    training_dataset = []
    training_data_dist = np.zeros((env.num_states, env.num_actions))
    for idx in range(len(training_sa_pairs)):
        s, a = training_sa_pairs[idx]
        assert env._transition_matrix is not None
        prob_s_prime = env._transition_matrix[s, a]
        s_prime = np.random.choice(np.arange(env.num_states), p=prob_s_prime)
        r = env.reward(s, a, s_prime)
        training_dataset.append((s, a, r, s_prime))
        training_data_dist[s, a] += 1.0
else:
    # Using only weighting style dataset
    training_dataset = None
    training_data_dist = None


# @title Visualize dataset or weights
if not weighting_only:
    plot_sa_values(env, training_data_dist, title="Dataset composition")
else:
    plot_sa_values(env, weights, title="Weighting Distribution")
