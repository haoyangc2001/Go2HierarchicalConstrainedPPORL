# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import os
import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCriticCPPO
from rsl_rl.storage import ConstrainedRolloutStorage


class CPPO:
    actor_critic: ActorCriticCPPO

    def __init__(
        self,
        actor_critic,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        cost_value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        value_clip_param=None,
        min_lr=1e-5,
        max_lr=1e-2,
        cost_limit=0.0,
        lambda_init=0.0,
        lambda_lr=0.05,
        lambda_max=100.0,
        normalize_advantage=True,
        cost_gamma=None,
        cost_lam=None,
        device="cpu",
    ):

        self.device = device
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.min_lr = min_lr
        self.max_lr = max_lr

        self.value_clip_param = clip_param if value_clip_param is None else value_clip_param

        # CPPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = ConstrainedRolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.cost_value_loss_coef = cost_value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.cost_gamma = cost_gamma if cost_gamma is not None else gamma
        self.cost_lam = cost_lam if cost_lam is not None else lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.normalize_advantage = normalize_advantage

        # Lagrange multiplier parameters
        self.cost_limit = float(cost_limit)
        self.lagrange_multiplier = float(lambda_init)
        self.lambda_lr = float(lambda_lr)
        self.lambda_max = float(lambda_max)

        self.last_stats = {}
        self.debug_dump_dir = None
        self.debug_iter = 0
        self.debug_raise_on_nan = False
        self._ratio_check_dumped = False

    def set_debug_dump_dir(self, dump_dir) -> None:
        self.debug_dump_dir = dump_dir

    def set_debug_iter(self, iteration: int) -> None:
        self.debug_iter = int(iteration)

    def set_debug_raise_on_nan(self, enabled: bool) -> None:
        self.debug_raise_on_nan = bool(enabled)

    def _tensor_stats(self, tensor: torch.Tensor) -> dict:
        finite = torch.isfinite(tensor)
        finite_ratio = float(finite.float().mean().item())
        if finite.any():
            finite_vals = tensor[finite]
            return {
                "finite_ratio": finite_ratio,
                "min": float(finite_vals.min().item()),
                "max": float(finite_vals.max().item()),
                "mean": float(finite_vals.mean().item()),
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
            }
        return {
            "finite_ratio": finite_ratio,
            "min": None,
            "max": None,
            "mean": None,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        }

    def _dump_nonfinite(self, tag: str, tensors: dict) -> None:
        if not self.debug_dump_dir:
            return
        os.makedirs(self.debug_dump_dir, exist_ok=True)
        dump = {
            "tag": tag,
            "iter": self.debug_iter,
            "learning_rate": float(self.learning_rate),
            "lagrange_multiplier": float(self.lagrange_multiplier),
            "tensors": {},
            "stats": {},
        }
        for name, value in tensors.items():
            if value is None:
                continue
            if torch.is_tensor(value):
                dump["stats"][name] = self._tensor_stats(value.detach())
                dump["tensors"][name] = value.detach().cpu()
            else:
                dump["stats"][name] = value
        # Dump tensors once to help trace the first non-finite batch.
        filename = f"nan_dump_iter{self.debug_iter:05d}_{tag}.pt"
        torch.save(dump, os.path.join(self.debug_dump_dir, filename))

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = ConstrainedRolloutStorage(
            num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device
        )

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute actions and value predictions
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.cost_values = self.actor_critic.evaluate_cost(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions

    def process_env_step(self, rewards, costs, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.costs = costs.clone()
        self.transition.dones = dones
        # Bootstrapping on timeouts
        if "time_outs" in infos:
            time_outs = infos["time_outs"].to(self.device)
            self.transition.rewards += self.gamma * (
                self.transition.values.squeeze(1) * time_outs
            )
            self.transition.costs += self.cost_gamma * (
                self.transition.cost_values.squeeze(1) * time_outs
            )

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        last_cost_values = self.actor_critic.evaluate_cost(last_critic_obs).detach()
        self.storage.compute_returns(
            last_values,
            last_cost_values,
            self.gamma,
            self.lam,
            cost_gamma=self.cost_gamma,
            cost_lam=self.cost_lam,
        )

    def _mean_episode_cost(self):
        costs = self.storage.costs.squeeze(-1)
        dones = self.storage.dones.squeeze(-1).bool()
        running_cost = torch.zeros(costs.shape[1], device=costs.device)
        episode_cost_sum = 0.0
        episode_count = 0
        for step in range(costs.shape[0]):
            running_cost += costs[step]
            done_mask = dones[step]
            if done_mask.any():
                episode_cost_sum += running_cost[done_mask].sum().item()
                episode_count += int(done_mask.sum().item())
                running_cost[done_mask] = 0.0
        if episode_count == 0:
            return 0.0, 0
        return episode_cost_sum / float(episode_count), episode_count

    def _update_lagrange_multiplier(self):
        mean_episode_cost, episode_count = self._mean_episode_cost()
        if episode_count == 0:
            return mean_episode_cost, episode_count
        lagrange = self.lagrange_multiplier + self.lambda_lr * (mean_episode_cost - self.cost_limit)
        lagrange = max(0.0, min(self.lambda_max, lagrange))
        self.lagrange_multiplier = float(lagrange)
        return mean_episode_cost, episode_count

    def update(self):
        mean_value_loss = 0.0
        mean_cost_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_kl = 0.0
        mean_clip_frac = 0.0
        mean_value_clip_frac = 0.0
        mean_grad_norm = 0.0
        mean_adv_finite = 0.0
        mean_logp_finite = 0.0
        mean_old_logp_finite = 0.0
        mean_ratio_finite = 0.0
        adv_std_sum = 0.0
        cost_adv_std_sum = 0.0
        lag_adv_std_sum = 0.0
        adv_std_finite_batches = 0
        cost_adv_std_finite_batches = 0
        lag_adv_std_finite_batches = 0
        stat_batches = 0
        performed_updates = 0
        nonfinite_loss_batches = 0
        nonfinite_grad_batches = 0

        if self.normalize_advantage:
            advantages = self.storage.advantages
            cost_advantages = self.storage.cost_advantages
            self.storage.advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            self.storage.cost_advantages = (
                cost_advantages - cost_advantages.mean()
            ) / (cost_advantages.std() + 1e-8)

        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )

        early_stop = False
        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            target_cost_values_batch,
            advantages_batch,
            cost_advantages_batch,
            returns_batch,
            cost_returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:

            self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
            )
            cost_value_batch = self.actor_critic.evaluate_cost(
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
            )
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # KL for adaptive schedule
            kl = torch.sum(
                torch.log(sigma_batch / (old_sigma_batch + 1e-5))
                + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                / (2.0 * torch.square(sigma_batch))
                - 0.5,
                dim=-1,
            )
            kl_mean = torch.mean(kl)

            if self.desired_kl is not None:
                if kl_mean > self.desired_kl * 2.0:
                    self.learning_rate = max(self.min_lr, self.learning_rate / 1.5)
                    early_stop = True
                elif self.schedule == "adaptive" and kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                    self.learning_rate = min(self.max_lr, self.learning_rate * 1.5)
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.learning_rate
                if early_stop:
                    break

            lagrange_advantage = advantages_batch - self.lagrange_multiplier * cost_advantages_batch

            # Surrogate loss
            logp_diff = actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
            logp_diff = torch.clamp(logp_diff, -20.0, 20.0)
            ratio = torch.exp(logp_diff)
            if self.debug_dump_dir and not self._ratio_check_dumped:
                with torch.no_grad():
                    ratio_check = torch.exp(logp_diff)
                    ratio_diff = ratio - ratio_check
                    ratio_rel_diff = ratio_diff / (ratio_check + 1e-12)
                    ratio_max = ratio.max()
                    logp_diff_max = logp_diff.max()
                    if (not torch.isfinite(ratio_max)) or (ratio_max > 1e6) or (logp_diff_max > 20):
                        self._dump_nonfinite(
                            "ratio_check",
                            {
                                "logp_diff": logp_diff,
                                "ratio": ratio,
                                "ratio_check": ratio_check,
                                "ratio_check_diff": ratio_diff,
                                "ratio_check_rel_diff": ratio_rel_diff,
                                "ratio_check_stats": {
                                    "ratio_max": float(ratio_max)
                                    if torch.isfinite(ratio_max)
                                    else float("inf"),
                                    "logp_diff_max": float(logp_diff_max)
                                    if torch.isfinite(logp_diff_max)
                                    else float("inf"),
                                },
                            },
                        )
                        self._ratio_check_dumped = True
            surrogate = -torch.squeeze(lagrange_advantage) * ratio
            surrogate_clipped = -torch.squeeze(lagrange_advantage) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Lightweight diagnostics for numerical stability.
            mean_adv_finite += torch.isfinite(advantages_batch).float().mean().item()
            mean_logp_finite += torch.isfinite(actions_log_prob_batch).float().mean().item()
            mean_old_logp_finite += torch.isfinite(old_actions_log_prob_batch).float().mean().item()
            mean_ratio_finite += torch.isfinite(ratio).float().mean().item()
            adv_std = advantages_batch.std()
            if torch.isfinite(adv_std):
                adv_std_sum += adv_std.item()
                adv_std_finite_batches += 1
            cost_adv_std = cost_advantages_batch.std()
            if torch.isfinite(cost_adv_std):
                cost_adv_std_sum += cost_adv_std.item()
                cost_adv_std_finite_batches += 1
            lag_adv_std = lagrange_advantage.std()
            if torch.isfinite(lag_adv_std):
                lag_adv_std_sum += lag_adv_std.item()
                lag_adv_std_finite_batches += 1
            stat_batches += 1

            # Value losses
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.value_clip_param, self.value_clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
                value_clip_frac = (
                    (value_batch - target_values_batch).abs() > self.value_clip_param
                ).float().mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()
                value_clip_frac = torch.zeros(1, device=value_batch.device).mean()

            if self.use_clipped_value_loss:
                cost_value_clipped = target_cost_values_batch + (
                    cost_value_batch - target_cost_values_batch
                ).clamp(-self.value_clip_param, self.value_clip_param)
                cost_value_losses = (cost_value_batch - cost_returns_batch).pow(2)
                cost_value_losses_clipped = (cost_value_clipped - cost_returns_batch).pow(2)
                cost_value_loss = torch.max(cost_value_losses, cost_value_losses_clipped).mean()
            else:
                cost_value_loss = (cost_returns_batch - cost_value_batch).pow(2).mean()

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                + self.cost_value_loss_coef * cost_value_loss
                - self.entropy_coef * entropy_batch.mean()
            )

            # Gradient step
            self.optimizer.zero_grad()
            if not torch.isfinite(loss):
                nonfinite_loss_batches += 1
                self._dump_nonfinite(
                    "loss",
                    {
                        "loss": loss,
                        "surrogate_loss": surrogate_loss,
                        "value_loss": value_loss,
                        "cost_value_loss": cost_value_loss,
                        "entropy": entropy_batch,
                        "ratio": ratio,
                        "lagrange_advantage": lagrange_advantage,
                        "advantages": advantages_batch,
                        "cost_advantages": cost_advantages_batch,
                        "returns": returns_batch,
                        "cost_returns": cost_returns_batch,
                        "target_values": target_values_batch,
                        "target_cost_values": target_cost_values_batch,
                        "value_batch": value_batch,
                        "cost_value_batch": cost_value_batch,
                        "obs_batch": obs_batch,
                        "critic_obs_batch": critic_obs_batch,
                        "actions_batch": actions_batch,
                        "actions_log_prob": actions_log_prob_batch,
                        "old_actions_log_prob": old_actions_log_prob_batch,
                        "mu_batch": mu_batch,
                        "sigma_batch": sigma_batch,
                        "old_mu_batch": old_mu_batch,
                        "old_sigma_batch": old_sigma_batch,
                    },
                )
                if self.debug_raise_on_nan:
                    raise RuntimeError(
                        f"Non-finite loss detected at iter {self.debug_iter}."
                    )
                continue
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            if not torch.isfinite(grad_norm):
                self.optimizer.zero_grad()
                nonfinite_grad_batches += 1
                self._dump_nonfinite(
                    "grad",
                    {
                        "grad_norm": float(grad_norm),
                        "loss": loss,
                        "surrogate_loss": surrogate_loss,
                        "value_loss": value_loss,
                        "cost_value_loss": cost_value_loss,
                        "entropy": entropy_batch,
                        "ratio": ratio,
                        "lagrange_advantage": lagrange_advantage,
                        "advantages": advantages_batch,
                        "cost_advantages": cost_advantages_batch,
                        "returns": returns_batch,
                        "cost_returns": cost_returns_batch,
                        "target_values": target_values_batch,
                        "target_cost_values": target_cost_values_batch,
                        "value_batch": value_batch,
                        "cost_value_batch": cost_value_batch,
                        "obs_batch": obs_batch,
                        "critic_obs_batch": critic_obs_batch,
                        "actions_batch": actions_batch,
                        "actions_log_prob": actions_log_prob_batch,
                        "old_actions_log_prob": old_actions_log_prob_batch,
                        "mu_batch": mu_batch,
                        "sigma_batch": sigma_batch,
                        "old_mu_batch": old_mu_batch,
                        "old_sigma_batch": old_sigma_batch,
                    },
                )
                if self.debug_raise_on_nan:
                    raise RuntimeError(
                        f"Non-finite grad_norm detected at iter {self.debug_iter}."
                    )
                continue
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_cost_value_loss += cost_value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_kl += kl_mean.item()
            mean_clip_frac += ((ratio - 1.0).abs() > self.clip_param).float().mean().item()
            mean_value_clip_frac += value_clip_frac.item()
            mean_grad_norm += float(grad_norm)
            performed_updates += 1

        if performed_updates > 0:
            mean_value_loss /= performed_updates
            mean_cost_value_loss /= performed_updates
            mean_surrogate_loss /= performed_updates
            mean_entropy /= performed_updates
            mean_kl /= performed_updates
            mean_clip_frac /= performed_updates
            mean_value_clip_frac /= performed_updates
            mean_grad_norm /= performed_updates
        if stat_batches > 0:
            mean_adv_finite /= stat_batches
            mean_logp_finite /= stat_batches
            mean_old_logp_finite /= stat_batches
            mean_ratio_finite /= stat_batches

        episode_cost, _ = self._update_lagrange_multiplier()

        self.last_stats = {
            "entropy": mean_entropy,
            "lr": self.learning_rate,
            "grad_norm": mean_grad_norm,
            "value_clip_frac": mean_value_clip_frac,
            "cost_value_loss": mean_cost_value_loss,
            "lagrange_multiplier": self.lagrange_multiplier,
            "episode_cost": episode_cost,
            "nonfinite_loss_batches": nonfinite_loss_batches,
            "nonfinite_grad_batches": nonfinite_grad_batches,
            "performed_updates": performed_updates,
            "adv_std": adv_std_sum / max(1, adv_std_finite_batches),
            "cost_adv_std": cost_adv_std_sum / max(1, cost_adv_std_finite_batches),
            "lagrange_adv_std": lag_adv_std_sum / max(1, lag_adv_std_finite_batches),
            "adv_std_finite": (adv_std_finite_batches / stat_batches) if stat_batches > 0 else 0.0,
            "cost_adv_std_finite": (cost_adv_std_finite_batches / stat_batches) if stat_batches > 0 else 0.0,
            "lagrange_adv_std_finite": (lag_adv_std_finite_batches / stat_batches) if stat_batches > 0 else 0.0,
            "adv_finite": mean_adv_finite,
            "logp_finite": mean_logp_finite,
            "old_logp_finite": mean_old_logp_finite,
            "ratio_finite": mean_ratio_finite,
        }

        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_kl, mean_clip_frac
