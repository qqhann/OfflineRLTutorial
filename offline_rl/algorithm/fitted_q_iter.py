# @title Fitted Q-iteration
import numpy as np
import torch

from offline_rl.utils import get_tensors
from offline_rl.plot import plot_sa_values
from offline_rl.algorithm.tabular_q_iter import q_backup_sparse, q_backup_sparse_sampled
from offline_rl.env import GridEnv


def project_qvalues(
    q_values: np.ndarray,
    network: torch.nn.Module,
    optimizer,
    num_steps=50,
    weights=None,
):
    # regress onto q_values (aka projection)
    q_values_tensor = torch.tensor(q_values, dtype=torch.float32)
    for _ in range(num_steps):
        # Eval the network at each state
        pred_qvalues = network(torch.arange(q_values.shape[0]))
        if weights is None:
            loss = torch.mean((pred_qvalues - q_values_tensor) ** 2)
        else:
            loss = torch.mean(weights * (pred_qvalues - q_values_tensor) ** 2)
        network.zero_grad()
        loss.backward()
        optimizer.step()
    return pred_qvalues.detach().numpy()


def project_qvalues_sampled(
    env: GridEnv, s, a, target_values, network, optimizer, num_steps=50, weights=None
):
    # train with a sampled dataset
    target_qvalues = torch.tensor(target_values, dtype=torch.float32)
    s = torch.tensor(s, dtype=torch.int64)
    a = torch.tensor(a, dtype=torch.int64)
    pred_qvalues = network(s)
    pred_qvalues = pred_qvalues.gather(1, a.reshape(-1, 1)).squeeze()
    loss = torch.mean((pred_qvalues - target_qvalues) ** 2)
    network.zero_grad()
    loss.backward()
    optimizer.step()

    pred_qvalues = network(torch.arange(env.num_states))
    return pred_qvalues.detach().numpy()


def fitted_q_iteration(
    env: GridEnv,
    network,
    num_itrs=100,
    project_steps=50,
    render=False,
    weights=None,
    sampled=False,
    training_dataset=None,
    **kwargs
):
    """
    Runs Fitted Q-iteration.

    Args:
      env: A GridEnv object.
      num_itrs (int): Number of FQI iterations to run.
      project_steps (int): Number of gradient steps used for projection.
      render (bool): If True, will plot q-values after each iteration.
      sampled (bool): Whether to use sampled datasets for training or not.
      training_dataset (list): list of (s, a, r, ns) pairs
    """
    dS = env.num_states
    dA = env.num_actions

    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    weights_tensor = None
    if weights is not None:
        weights_tensor = torch.tensor(weights, dtype=torch.float32)

    q_values = np.zeros((dS, dA))
    for i in range(num_itrs):
        if sampled:
            for j in range(project_steps):
                training_idx = np.random.choice(
                    np.arange(len(training_dataset)), size=128
                )
                s, a, ns, r = get_tensors(training_dataset, training_idx)
                target_values = q_backup_sparse_sampled(
                    env, q_values, s, a, ns, r, **kwargs
                )
                intermed_values = project_qvalues_sampled(
                    env,
                    s,
                    a,
                    target_values,
                    network,
                    optimizer,
                    weights=None,
                )
                if j == project_steps - 1:
                    q_values = intermed_values
        else:
            target_values = q_backup_sparse(env, q_values, **kwargs)
            q_values = project_qvalues(
                target_values,
                network,
                optimizer,
                weights=weights_tensor,
                num_steps=project_steps,
            )
        if render:
            plot_sa_values(
                env, q_values, update=True, title="Q-values Iteration %d" % i
            )
    return q_values
