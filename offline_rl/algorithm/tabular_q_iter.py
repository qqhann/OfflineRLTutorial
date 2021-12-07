import numpy as np

from offline_rl.plot import plot_sa_values
from offline_rl.env import GridEnv

# @title Tabular Q-iteration


def q_backup_sparse(env: GridEnv, q_values: np.ndarray, discount=0.99) -> np.ndarray:
    dS = env.num_states
    dA = env.num_actions

    new_q_values = np.zeros_like(q_values)
    value = np.max(q_values, axis=1)
    for s in range(dS):
        for a in range(dA):
            new_q_value = 0
            for ns, prob in env.transitions(s, a).items():
                new_q_value += prob * (env.reward(s, a, ns) + discount * value[ns])
            new_q_values[s, a] = new_q_value
    return new_q_values


def q_backup_sparse_sampled(env, q_values: np.ndarray, s, a, ns, r, discount=0.99):
    q_values_ns = q_values[ns, :]
    values = np.max(q_values_ns, axis=-1)
    target_value = r + discount * values
    return target_value


def q_iteration(env: GridEnv, num_itrs=100, render=False, **kwargs):
    """
    Run tabular Q-iteration

    Args:
      env: A GridEnv object
      num_itrs (int): Number of FQI iterations to run
      render (bool): If True, will plot q-values after each iteration
    """
    # dS x dA
    q_values = np.zeros((env.num_states, env.num_actions))
    for _ in range(num_itrs):
        q_values = q_backup_sparse(env, q_values, **kwargs)
        if render:
            plot_sa_values(env, q_values, update=True, title="Q-values")
    return q_values  # dS x dA
