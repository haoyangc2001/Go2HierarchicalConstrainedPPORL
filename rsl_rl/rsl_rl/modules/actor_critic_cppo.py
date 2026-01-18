# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCriticCPPO(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=None,
        critic_hidden_dims=None,
        cost_critic_hidden_dims=None,
        activation="elu",
        init_noise_std=1.0,
        action_squash=None,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticCPPO.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        if actor_hidden_dims is None:
            actor_hidden_dims = [256, 256, 256]
        if critic_hidden_dims is None:
            critic_hidden_dims = [256, 256, 256]
        if cost_critic_hidden_dims is None:
            cost_critic_hidden_dims = list(critic_hidden_dims)

        activation = get_activation(activation)

        # Policy network
        actor_layers = [nn.Linear(num_actor_obs, actor_hidden_dims[0]), activation]
        for idx, dim in enumerate(actor_hidden_dims):
            if idx == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(dim, num_actions))
            else:
                actor_layers.append(nn.Linear(dim, actor_hidden_dims[idx + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Reward value function
        critic_layers = [nn.Linear(num_critic_obs, critic_hidden_dims[0]), activation]
        for idx, dim in enumerate(critic_hidden_dims):
            if idx == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(dim, 1))
            else:
                critic_layers.append(nn.Linear(dim, critic_hidden_dims[idx + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        # Cost value function
        cost_layers = [nn.Linear(num_critic_obs, cost_critic_hidden_dims[0]), activation]
        for idx, dim in enumerate(cost_critic_hidden_dims):
            if idx == len(cost_critic_hidden_dims) - 1:
                cost_layers.append(nn.Linear(dim, 1))
            else:
                cost_layers.append(nn.Linear(dim, cost_critic_hidden_dims[idx + 1]))
                cost_layers.append(activation)
        self.cost_critic = nn.Sequential(*cost_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Reward Critic MLP: {self.critic}")
        print(f"Cost Critic MLP: {self.cost_critic}")

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        self.action_squash = action_squash
        self._squash_eps = 1e-6
        Normal.set_default_validate_args = False

    def reset(self, dones=None):
        pass

    @property
    def action_mean(self):
        mean = self.distribution.mean
        if self.action_squash == "tanh":
            return torch.tanh(mean)
        return mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        actions = self.distribution.sample()
        if self.action_squash == "tanh":
            return torch.tanh(actions)
        return actions

    def get_actions_log_prob(self, actions):
        if self.action_squash == "tanh":
            clipped_actions = torch.clamp(actions, -1.0 + self._squash_eps, 1.0 - self._squash_eps)
            raw_actions = 0.5 * (torch.log1p(clipped_actions) - torch.log1p(-clipped_actions))
            log_prob = self.distribution.log_prob(raw_actions)
            log_prob = log_prob - torch.log(1.0 - clipped_actions.pow(2) + self._squash_eps)
            return log_prob.sum(dim=-1)
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        if self.action_squash == "tanh":
            return torch.tanh(actions_mean)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        return self.critic(critic_observations)

    def evaluate_cost(self, critic_observations, **kwargs):
        return self.cost_critic(critic_observations)


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    if act_name == "selu":
        return nn.SELU()
    if act_name == "relu":
        return nn.ReLU()
    if act_name == "crelu":
        return nn.ReLU()
    if act_name == "lrelu":
        return nn.LeakyReLU()
    if act_name == "tanh":
        return nn.Tanh()
    if act_name == "sigmoid":
        return nn.Sigmoid()
    print("invalid activation function!")
    return None
