# Repository Guidelines

## Project Overview
MCRA_RL 是 Unitree Go2 分层导航系统。低层运动控制策略固定且预训练；高层策略采用 CMDP + CPPO（PPO-Lagrangian）训练，目标是在到达目标点的同时尽量避免碰撞障碍物/边界。

## Key Paths
- Environments: `legged_gym_go2/legged_gym/envs/go2/`
- Training scripts: `legged_gym_go2/legged_gym/scripts/`
- RL algorithms: `rsl_rl/rsl_rl/algorithms/`
- Deployment: `legged_gym_go2/deploy/`
- Logs/checkpoints: `/home/caohy/repositories/MCRA_RL/logs/`

## Hierarchical RL Structure
- Low-level (locomotion): `legged_gym_go2/legged_gym/envs/go2/go2_env.py`
- High-level (navigation wrapper): `legged_gym_go2/legged_gym/envs/go2/high_level_navigation_env.py`
- Hierarchical wrapper: `legged_gym_go2/legged_gym/envs/go2/hierarchical_go2_env.py`
- High-level actions are repeated at low level via `GO2HighLevelCfg.env.high_level_action_repeat`.

## Environment Overview
### Low-Level Environment (Locomotion)
- 实现 `GO2Robot`，继承自 `LeggedRobot`。
- `step()` 返回 `(obs, privileged_obs, reward, done, info)`。
- `reset()` 调用 `reset_idx` 后执行一次零动作 `step` 以初始化观测。
- `_compute_safety_metrics()` 计算：
  - `reach_metric`：目标点 XY 距离
  - `min_hazard_distance`：最近障碍/边界表面距离
  - `obstacle_surface_distance`、`boundary_distance`
- `check_termination()` 在 `terminate_on_reach_avoid` 时启用到达/碰撞终止；碰撞优先 `min_hazard_distance < collision_dist`。

### High-Level Navigation Wrapper
- 高层观测由低层状态构造：
  - 基础 8 维：`cos(heading)`、`sin(heading)`、`body_vx`、`body_vy`、`yaw_rate`、`reach_metric`（缩放）、`target_dir_body_x`、`target_dir_body_y`
  - 可选 target lidar bins（平滑角度分箱 + 距离衰减）
  - 可选 obstacle/boundary lidar bins（每个 bin 取最大强度，边界采用射线交点）
- 提供距离解析：
  - 目标距离：由 target lidar 强度反推
  - 最近 hazard 距离：由 obstacle/boundary lidar 强度反推
- 高层动作映射到底层速度指令见下文。

### Hierarchical Wrapper
- 固定低层策略，通过高层动作输出速度指令。
- 每个高层动作重复 `high_level_action_repeat` 次低层步，done 聚合。
- 高层 `step()` 返回 `(obs, reward, cost, done, info)`。
- Cost 在高层时间尺度下累计（每个 low-level repeat 的成本求和）。
- Info 字段：
  - `time_outs`, `reached`, `success`, `collision`, `terminated`, `truncated`
  - `target_distance`, `min_hazard_distance`, `boundary_distance`, `obstacle_surface_distance`
  - `base_lin_vel`, `desired_commands`
  - `progress`, `command_speed`, `body_speed`, `command_delta`, `reward_clip_frac`
  - `cost`, `cost_near`, `cost_collision`

### Vectorized Adapter
- `HierarchicalVecEnv` 提供向量化接口，返回 `(obs, reward, cost, done, info)`。
- `num_privileged_obs = None`。

## High-Level Action Mapping (Current)
In `update_velocity_commands`:
- Clip high-level actions to `[-1, 1]`.
- Multiply by `HighLevelNavigationConfig.action_scale`.
- Map to base commands:
  - `vx = action[0] * 0.6`
  - `vy = action[1] * 0.2`
  - `vyaw = action[2] * 0.8`
With default `action_scale = [1, 1, 1]`, the effective command ranges are:
`vx in [-0.6, 0.6]`, `vy in [-0.2, 0.2]`, `vyaw in [-0.8, 0.8]`.

## High-Level Observations
- Base features (8):
  1) `cos(heading)`
  2) `sin(heading)`
  3) `body_vx` (scaled, clipped)
  4) `body_vy` (scaled, clipped)
  5) `yaw_rate` (scaled, clipped)
  6) `reach_metric` (true XY distance to target center, scaled by `GO2HighLevelCfg.reach_metric_scale`)
  7) `target_dir_body_x`
  8) `target_dir_body_y`
- Optional target lidar bins: `target_lidar_num_bins`
- Optional obstacle/boundary lidar bins: `lidar_num_bins`
- Total dim: `8 + target_lidar_num_bins + lidar_num_bins` when manual lidar is enabled.

## Reward & Cost Design (High Level)
- 实现位置：`HierarchicalGO2Env._compute_reward`
- 奖励（reward）：
  - 进展奖励：`progress_scale * (prev_target_distance - target_distance)`
  - 动作平滑惩罚：`- action_smooth_scale * ||cmd_t - cmd_{t-1}||`
  - 到达奖励：`success_reward`（成功 done 时加）
  - 可选 reward 裁剪：`reward_clip`
- 成本（cost）：
  - 近障成本：`cost_near = max((cost_safe_dist - hazard_distance)/cost_safe_dist, 0)`
  - 碰撞成本：`cost_collision = 1`（碰撞时）
  - 合成：`cost = cost_collision_weight * cost_collision + cost_near_weight * cost_near`
- 终止与 done：
  - done 基于 low-level reset 聚合，避免不同步。
  - `collision` 使用 `hazard_distance <= collision_dist`。
  - `time_outs` 来自低层 `info`。

## CPPO Training (High Level)
- 训练脚本：`legged_gym_go2/legged_gym/scripts/train_cppo.py`
- 算法实现：`rsl_rl/rsl_rl/algorithms/cppo.py`
- 关键机制：
  - 双 critic：`V_r(o)` + `V_c(o)`
  - Lagrange 对偶更新：基于 episode total cost
  - `time_outs` 对 reward 与 cost 都进行 bootstrap

## Logging and Outputs
- 日志与模型：`/home/caohy/repositories/MCRA_RL/logs/<experiment_name>/<timestamp>/`
- 日志字段（精简后）：
  - `success`, `reach`, `collision`, `boundary_collision_rate`, `obstacle_collision_rate`, `timeout`
  - `cost`, `cost_limit`, `lambda`, `cost_step`, `cost_near`, `cost_collision`
  - `success_steps`, `avg_reward`, `progress`, `goal_dist`, `min_hazard`
  - `cmd_speed`, `body_speed`, `cmd_delta`, `action_std`
  - `policy_loss`, `value_loss`, `cost_value_loss`
  - `approx_kl`, `clip_frac`, `entropy`, `lr`, `grad_norm`, `value_clip_frac`
  - `reward_clip`, `hazard_p10`, `hazard_p50`, `hazard_p90`, `boundary_violation`
  - `ep_len_mean`, `init_goal_dist`, `elapsed`

### Training Log Field Meanings
- `iter`: 迭代编号（一次 rollout + 一次更新），从 1 开始。
- `success`: 成功率（到达目标且无碰撞）。
- `reach`: 到达率（done 且 `target_distance <= goal_reached_dist`）。
- `collision`: 碰撞率（done 且 `hazard_distance <= collision_dist`）。
- `timeout`: 超时率（`time_outs` 且非到达/碰撞）。
- `cost`: episode total cost 平均值。
- `cost_step`: 每步成本均值（高层时间尺度）。
- `cost_near`: 近障成本均值。
- `cost_collision`: 碰撞成本均值。
- `avg_reward`: 每步奖励均值。
- `progress`: 每步目标距离减少量均值。
- `goal_dist`: 目标距离均值（米）。
- `min_hazard`: 最近 hazard 距离均值（米）。
- `cmd_speed`: 命令平面速度均值。
- `body_speed`: 机体平面速度均值。
- `cmd_delta`: 高层指令变化幅度均值。
- `action_std`: 策略探索噪声均值。
- `policy_loss`, `value_loss`, `cost_value_loss`: PPO/CPPO 损失。
- `approx_kl`, `clip_frac`: PPO 诊断指标。
- `entropy`: 策略熵。
- `lr`: 学习率。
- `grad_norm`: 梯度范数。
- `value_clip_frac`: value clip 触发比例。
- `reward_clip`: 奖励裁剪比例。
- `hazard_p10/p50/p90`: `min_hazard_distance` 分位数。
- `boundary_violation`: `boundary_distance < 0` 的比例。
- `ep_len_mean`: 回合长度均值。
- `init_goal_dist`: 初始目标距离均值。
- `elapsed`: 日志间隔耗时（秒）。

## Configuration Entry Points
- Reward/Cost 参数：`legged_gym_go2/legged_gym/envs/go2/go2_config.py`（`GO2HighLevelCfg.reward_shaping`）
- CPPO 超参数：`legged_gym_go2/legged_gym/envs/go2/go2_config.py`（`GO2HighLevelCfgPPO`）
- 观测维度：`legged_gym_go2/legged_gym/envs/go2/go2_config.py` 文件末尾计算
- 低层模型路径：`GO2HighLevelCfgPPO.runner.low_level_model_path`

## Common Commands
Before running any script:
```bash
conda activate unitree-rl
```

Train CPPO:
```bash
python legged_gym_go2/legged_gym/scripts/train_cppo.py --headless=true --num_envs=32
```

Plot arena layout:
```bash
python legged_gym_go2/legged_gym/scripts/plot_env_layout.py
```

Plot training logs:
```bash
python legged_gym_go2/legged_gym/scripts/plot_training_results.py /home/caohy/repositories/MCRA_RL/logs/<experiment>/<timestamp>/training.log
```

Deploy in Mujoco (example):
```bash
python legged_gym_go2/deploy/deploy_mujoco/deploy.py --checkpoint=model.pt --cfg=configs/go2.yaml
```

## Development Notes
- `train_cppo.py` 在 `__main__` 中覆盖 headless 与 device 相关参数；需要时在脚本内改。
- 低层策略固定，高层训练不应修改低层参数。
- 若修改 lidar bin 数量/范围，请同步更新 `GO2HighLevelCfg` 并检查观测维度。
