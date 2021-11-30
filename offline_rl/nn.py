import numpy as np
import torch

from offline_rl.env import GridEnv

# @title Neural Network Code


def stack_observations(env: GridEnv):
    obs = []
    for s in range(env.num_states):
        obs.append(env.observation(s))
    return np.stack(obs)


class FCNetwork(torch.nn.Module):
    def __init__(self, env, layers=[20, 20]):
        super(FCNetwork, self).__init__()
        self.all_observations = torch.tensor(
            stack_observations(env), dtype=torch.float32
        )
        dim_input = env.dim_obs
        dim_output = env.num_actions
        net_layers = []

        dim = dim_input
        for i, layer_size in enumerate(layers):
            net_layers.append(torch.nn.Linear(dim, layer_size))
            net_layers.append(torch.nn.ReLU())
            dim = layer_size
        net_layers.append(torch.nn.Linear(dim, dim_output))
        self.layers = net_layers
        self.network = torch.nn.Sequential(*net_layers)

    def forward(self, states):
        observations = torch.index_select(self.all_observations, 0, states)
        return self.network(observations)


def one_hot(y, n_dims=None):
    """Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims."""
    y_tensor = y.view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return y_one_hot


class TabularNetwork(torch.nn.Module):
    def __init__(self, env):
        super(TabularNetwork, self).__init__()
        self.num_states = env.num_states
        self.network = torch.nn.Sequential(
            torch.nn.Linear(self.num_states, env.num_actions)
        )

    def forward(self, states):
        onehot = one_hot(states, self.num_states)
        return self.network(onehot)
