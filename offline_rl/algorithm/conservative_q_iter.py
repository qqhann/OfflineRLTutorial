# @title Conservative Q-Learning
import numpy as np
import torch

from offline_rl.utils import get_tensors
from offline_rl.plot import plot_sa_values
from offline_rl.algorithm.tabular_q_iter import q_backup_sparse, q_backup_sparse_sampled


def project_qvalues_cql(
    q_values, network, optimizer, num_steps=50, cql_alpha=0.1, weights=None
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

        # Add cql_loss
        # You can have two variants of this loss, one where data q-values
        # also maximized (CQL-v2), and one where only the large Q-values
        # are pushed down (CQL-v1) as covered in the tutorial
        cql_loss = torch.logsumexp(pred_qvalues, dim=-1, keepdim=True)  # - pred_qvalues
        loss = loss + cql_alpha * torch.mean(weights * cql_loss)
        network.zero_grad()
        loss.backward()
        optimizer.step()
    return pred_qvalues.detach().numpy()


def project_qvalues_cql_sampled(
    env,
    s,
    a,
    target_values,
    network,
    optimizer,
    cql_alpha=0.1,
    num_steps=50,
    weights=None,
):
    # train with a sampled dataset
    target_qvalues = torch.tensor(target_values, dtype=torch.float32)
    s = torch.tensor(s, dtype=torch.int64)
    a = torch.tensor(a, dtype=torch.int64)
    pred_qvalues = network(s)
    logsumexp_qvalues = torch.logsumexp(pred_qvalues, dim=-1)

    pred_qvalues = pred_qvalues.gather(1, a.reshape(-1, 1)).squeeze()
    cql_loss = logsumexp_qvalues - pred_qvalues

    loss = torch.mean((pred_qvalues - target_qvalues) ** 2)
    loss = loss + cql_alpha * torch.mean(cql_loss)

    network.zero_grad()
    loss.backward()
    optimizer.step()

    pred_qvalues = network(torch.arange(env.num_states))
    return pred_qvalues.detach().numpy()


def conservative_q_iteration(
    env,
    network,
    num_itrs=100,
    project_steps=50,
    cql_alpha=0.1,
    render=False,
    weights=None,
    sampled=False,
    training_dataset=None,
    **kwargs
):
    """
    Runs Conservative Q-iteration.

    Args:
      env: A GridEnv object.
      num_itrs (int): Number of FQI iterations to run.
      project_steps (int): Number of gradient steps used for projection.
      cql_alpha (float): Value of weight on the CQL coefficient.
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
                intermed_values = project_qvalues_cql_sampled(
                    env,
                    s,
                    a,
                    target_values,
                    network,
                    optimizer,
                    cql_alpha=cql_alpha,
                    weights=None,
                )
                if j == project_steps - 1:
                    q_values = intermed_values
        else:
            target_values = q_backup_sparse(env, q_values, **kwargs)
            q_values = project_qvalues_cql(
                target_values,
                network,
                optimizer,
                weights=weights_tensor,
                cql_alpha=cql_alpha,
                num_steps=project_steps,
            )
        if render:
            plot_sa_values(
                env, q_values, update=True, title="Q-values Iteration %d" % i
            )
    return q_values
