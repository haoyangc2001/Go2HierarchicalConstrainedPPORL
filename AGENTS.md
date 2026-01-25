# Repository Guidelines

## Project Overview
MCRA_RL 是 Unitree Go2 分层导航系统。低层运动控制策略固定且预训练；高层策略采用 **CMDP + CPPO（PPO-Lagrangian，on-policy、model-free）** 训练，目标是在到达目标点的同时尽量避免碰撞障碍物/边界。

## Runtime / Dependencies
- 物理仿真：Isaac Gym（PhysX）。
- 训练框架：`legged_gym_go2` + `rsl_rl`。
- 典型环境：`conda activate unitree-rl`。

## Key Paths
- Environments: `legged_gym_go2/legged_gym/envs/go2/`
- Training scripts: `legged_gym_go2/legged_gym/scripts/`
- RL algorithms: `rsl_rl/rsl_rl/algorithms/`
- Deployment: `legged_gym_go2/deploy/`
- Logs/checkpoints: `/home/caohy/repositories/MCRA_RL/logs/`

## Architecture & Data Flow
### Stack
- Low-level locomotion: `GO2Robot` (`legged_gym_go2/legged_gym/envs/go2/go2_env.py`)
- High-level navigation wrapper: `HighLevelNavigationEnv` (`legged_gym_go2/legged_gym/envs/go2/high_level_navigation_env.py`)
- Hierarchical wrapper: `HierarchicalGO2Env` (`legged_gym_go2/legged_gym/envs/go2/hierarchical_go2_env.py`)
- Vectorized adapter: `HierarchicalVecEnv` (`legged_gym_go2/legged_gym/utils/hierarchical_env_utils.py`)

### High-level Step Flow (宏步)
1) 高层动作 `a_t` 经过 clip/scale 映射为速度指令，写入 `base_env.commands`。
2) 每个高层动作重复 `high_level_action_repeat` 次低层步：
   - 低层策略（固定）根据低层观测输出关节动作。
   - 累计低层 cost 相关量（近障、碰撞指示）。
   - 聚合 done 与 timeouts，避免层间不同步。
3) 计算新的高层观测与高层奖励/成本（宏步尺度）。
4) 返回 `(obs, reward, cost, done, info)`。

### Low-level Policy Loading
- 通过 `task_registry.get_cfgs(name="go2")` 获取低层训练配置。
- 根据 `policy_class_name` 加载 `ActorCritic` 或 `ActorCriticRecurrent`。
- 低层模型路径由 `GO2HighLevelCfgPPO.runner.low_level_model_path` 指定。

## Low-Level Environment (GO2Robot)
- `reset()`：调用 `reset_idx()` 后执行一次零动作 `step()`，确保观测初始化。
- `_reset_root_states()`：
  - 在 terrain 的全域范围内采样初始 XY（默认 `terrain_length=12`, `terrain_width=12`）。
  - 通过 `spawn_clearance` + `spawn_max_tries` 避免与障碍/边界过近。
  - 随机 yaw 增强多样性。
- `_compute_safety_metrics()`：
  - 支持多障碍 `unsafe_spheres_pos` 与单障碍 `unsafe_sphere_pos`。
  - `obstacle_surface_distance = dist_to_unsafe - unsafe_radius_reward_scale * unsafe_sphere_radius`。
  - `boundary_distance` 由矩形边界（terrain half-extents）计算；负值表示越界。
  - `min_hazard_distance = min(obstacle_surface_distance, boundary_distance)`。
  - `reach_metric` 为目标点 XY 平面距离。
- `check_termination()`：若 `terminate_on_reach_avoid` 开启，使用：
  - 碰撞判定：`min_hazard_distance < collision_dist`
  - 到达判定：`reach_metric <= goal_reached_dist`

## High-Level Observations
- Base features (8):
  1) `cos(heading)`
  2) `sin(heading)`
  3) `body_vx`（`lin_vel_scale=2.0`，clip 到 [-1,1]）
  4) `body_vy`（`lin_vel_scale=2.0`，clip 到 [-1,1]）
  5) `yaw_rate`（`ang_vel_scale=0.25`，clip 到 [-1,1]）
  6) `reach_metric * reach_metric_scale`
  7) `target_dir_body_x`
  8) `target_dir_body_y`
- Target lidar bins（平滑角度分箱 + 距离衰减）：
  - 强度 `1 - surface_dist / target_lidar_max_range`，在相邻 2 个 bin 线性插值。
- Obstacle/boundary lidar bins：
  - 障碍物按 bin 取最大强度；边界通过射线与矩形边界交点计算强度。
- 观测包含 `nan_to_num` 保护，避免几何异常导致 NaN/Inf。
- 距离解析：
  - `extract_target_distance()`：从 target bins 反推 surface dist，再加 `target_radius`。
  - `extract_hazard_distance()`：从 lidar bins 取最大强度，映射到最近 hazard 距离。

## High-Level Action Mapping (Current)
In `update_velocity_commands`:
- Clip high-level actions to `[-1, 1]`.
- Multiply by `GO2HighLevelCfg.action_scale`.
- Map to base commands:
  - `vx = action[0] * 0.6`
  - `vy = action[1] * 0.2`
  - `vyaw = action[2] * 0.8`
- Heading alignment：`commands[:, 3] = heading + 2.0 * vyaw`。
With `GO2HighLevelCfg.action_scale = [1.0, 1.0, 1.2]`, effective ranges:
`vx in [-0.6, 0.6]`, `vy in [-0.2, 0.2]`, `vyaw in [-0.96, 0.96]`.

## Hierarchical Wrapper (HierarchicalGO2Env)
- 创建 base env：`task_registry.make_env(name="go2")`。
- 将高层 reward_shaping 的 `goal_reached_dist` / `collision_dist` 写回 `base_env.cfg.rewards_ext`。
- `terminate_on_reach_avoid` 由 `terminate_on_safety_violation` 或 `terminate_on_success` 控制。
- 每个高层动作重复 `high_level_action_repeat` 次低层步：
  - 近障成本与碰撞成本在低层步内累积。
  - done/time_outs 在低层步内聚合，确保宏步一致。
- Reward 使用真实 `reach_metric`（而非 lidar 估计）；info 同时提供 true/estimated 距离。

## Reward & Cost Design (High Level)
- 实现位置：`HierarchicalGO2Env._compute_reward`。
- Reward：
  - 进展奖励：`progress_scale * (prev_target_distance - target_distance)`
  - 动作平滑惩罚：`- action_smooth_scale * ||cmd_t - cmd_{t-1}||`
  - 到达奖励：`success_reward`（`reached & done & ~collision`）
  - 可选碰撞惩罚：`- collision_penalty * collision`
  - 可选近障惩罚：`- reward_near_penalty_scale * cost_near`
  - 可选 reward 缩放/裁剪：`reward_scale`, `reward_clip`
- Cost：
  - `cost_near = max((cost_safe_dist - hazard_distance)/cost_safe_dist, 0)`
  - `cost_collision = 1`（当 `hazard_distance <= collision_dist`）
  - `cost_collision_terminal`（碰撞附加终止成本）
  - `cost = cost_collision_weight * cost_collision + cost_near_weight * cost_near + cost_collision_terminal`
- Done：
  - done 对齐 base env reset 逻辑；`time_outs` 表示截断。
  - collision 判定使用 `hazard_distance <= collision_dist`（与低层终止 `min_hazard_distance < collision_dist` 略有差异）。

### Reward/Cost Defaults (GO2HighLevelCfg.reward_shaping)
- `goal_reached_dist = 0.3`
- `collision_dist = 0.35`
- `progress_scale = 12.0`
- `action_smooth_scale = 0.03`
- `success_reward = 100.0`
- `reward_near_penalty_scale = 3.0`
- `reward_scale = 1.0`
- `reward_clip = 200.0`
- `terminate_on_safety_violation = True`
- `terminate_on_success = True`
- `cost_safe_dist = 1.2`
- `cost_collision_weight = 20.0`
- `cost_collision_terminal = 75.0`
- `cost_near_weight = 0.3`
- `collision_penalty = 180.0`

## CPPO Training (High Level)
- 训练脚本：`legged_gym_go2/legged_gym/scripts/train_cppo.py`
- 算法实现：`rsl_rl/rsl_rl/algorithms/cppo.py`
- 关键机制：
  - `ActorCriticCPPO`：双 critic（reward/cost）。
  - `ConstrainedRolloutStorage`：存储 reward/cost returns & advantages。
  - Lagrange 对偶更新：基于 episode total cost，投影到 `[0, lambda_max]`。
  - Reward/Cost advantage 可标准化。
  - `time_outs` 对 reward 与 cost 同时 bootstrap。
  - KL 自适应学习率调度。

### CPPO Defaults (GO2HighLevelCfgPPO.algorithm)
- `entropy_coef = 0.008`
- `learning_rate = 2e-4`
- `clip_param = 0.15`
- `value_clip_param = 0.15`
- `schedule = 'adaptive'`, `desired_kl = 0.04`
- `min_lr = 5e-6`, `max_lr = 3e-4`
- `num_learning_epochs = 3`, `num_mini_batches = 12`
- `num_steps_per_env = 200`
- `max_grad_norm = 0.5`
- `cost_limit = 90.0`
- `lambda_init = 0.0`, `lambda_lr = 0.01`, `lambda_max = 100.0`
- `normalize_advantage = True`

## Training Script Notes (`train_cppo.py`)
- 强制 actor/critic 隐层为 `[512, 512, 512, 512]`。
- 若 `experiment_name == "high_level_go2"`，会改为 `high_level_go2_CPPO`。
- 日志路径：`/home/caohy/repositories/MCRA_RL/logs/high_level_go2_CPPO/<timestamp>/training.log`。
- `__main__` 中覆盖参数：
  - `headless=True`, `compute_device_id=3`, `sim_device_id=3`, `rl_device="cuda:3"`, `sim_device="cuda:3"`。
  - 如需改设备/渲染模式，请直接改脚本或传参后覆盖。

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

## Configuration Entry Points (Current Defaults)
- 高层配置：`legged_gym_go2/legged_gym/envs/go2/go2_config.py`
  - `GO2HighLevelCfg.seed = 1`
  - `reach_metric_scale = 0.2`
  - `action_scale = [1.0, 1.0, 1.2]`
  - `enable_manual_lidar = True`
  - `lidar_max_range = 8.0`, `target_lidar_max_range = 8.0`
  - `lidar_num_bins = 16`, `target_lidar_num_bins = 16`
  - `env.high_level_action_repeat = 5`
  - `env.episode_length_s = 40`
- PPO/CPPO 超参数：`GO2HighLevelCfgPPO`（见上文 defaults）。
- 观测维度：`GO2HighLevelCfg.env.num_observations` 在配置末尾自动计算。
- 低层模型路径：`GO2HighLevelCfgPPO.runner.low_level_model_path`。
- Resume 路径：`GO2HighLevelCfgPPO.runner.resume_path`。

### Base Env Geometry (GO2RoughCfg.rewards_ext / terrain)
- `unsafe_spheres_pos`: 多障碍位置列表（已按 6m 边界缩放）。
- `unsafe_sphere_radius = 0.25`, `boundary_margin = 0.25`。
- `target_sphere_pos = [0.0, 0.0, 0.4]`, `target_sphere_radius = 0.3`。
- `unsafe_radius_h_eval_scale = 2.0`, `unsafe_radius_reward_scale = 1.0`。
- `terrain_length = 12.0`, `terrain_width = 12.0`（边界 half-extents = 6m）。

## Debug Workflow (Training Logs)
目标：对给定训练日志做可复现的异常诊断 + 收敛判断，并将结论与历史 debug 记录及代码变更对齐，最终输出可执行的下一步建议。

### 1) 日志分析与异常检测（必须先做）
从日志中抽取并分段统计（前/中/后段或滑动均值）：
- 任务结果：`success`, `reach`, `collision`, `timeout`, `ep_len_mean`
- 约束与成本：`cost`, `cost_limit`, `lambda`, `cost_near`, `cost_collision`, `cost_collision_terminal`
- 学习稳定性：`approx_kl`, `clip_frac`, `value_loss`, `cost_value_loss`, `grad_norm`, `lr`
- 行为强度：`cmd_speed`, `body_speed`, `cmd_delta`, `action_std`, `entropy`
- 安全几何：`goal_dist`, `min_hazard`, `boundary_violation`, `hazard_p10/p50/p90`

必须判断并记录：
- 成功率趋势：是否提升、是否达到平台、是否后期回落。
- 收敛判断：success 的滑动均值是否在连续区间稳定波动（振幅小且无系统性下降）。
- 异常模式：
  - 碰撞主导：`collision` 高且 `timeout` 低，`ep_len_mean` 下降。
  - 约束饱和：`cost >> cost_limit` 且 `lambda` 长期接近 `lambda_max`。
  - 约束失效：`cost << cost_limit` 且 `lambda≈0`，CPPO 退化为 PPO。
  - 数值不稳：`grad_norm` 爆炸/`inf`、`nan_loss/nan_grad` 非零、`value_loss` 激增。
  - 策略漂移：`approx_kl`/`clip_frac` 长期偏高，`lr` 过早降到 `min_lr`。
  - 进展停滞：`goal_dist` 不降、`progress` 低，成功率停滞。

输出要求：形成结构化分析报告，至少包含：
- 关键区间统计（起始/峰值/末尾，或分段均值）
- 结论（是否收敛、主要异常、伴随证据）
- 可能的机制解释（例如“碰撞省成本”“lambda 长期 0 导致无约束”等）

### 2) 回顾历史 Debug 记录（DEBUG_SUMMARY.md）并关联代码
针对发现的异常：
- 先检索 `/home/caohy/repositories/MCRA_RL/DEBUG_SUMMARY.md` 中是否已有同类或相关的异常记录。
- 若有：核对当时的改动项与目标指标，判断是否对当前异常生效。
- 若无：标记为新异常模式，后续建议需明确验证策略。

同时回查相关代码与配置，确认当前实验是否包含这些改动或被覆盖，并检查是否存在项目逻辑错误：
- 约束/成本：`legged_gym_go2/legged_gym/envs/go2/go2_config.py`
- 高层 reward/cost 逻辑：`legged_gym_go2/legged_gym/envs/go2/hierarchical_go2_env.py`
- 动作/观测缩放与 lidar：`legged_gym_go2/legged_gym/envs/go2/high_level_navigation_env.py`
- 训练超参数与设备覆盖：`legged_gym_go2/legged_gym/scripts/train_cppo.py`

### 3) 评估“之前 debug 是否起作用”
必须回答以下问题：
- 相关改动是否已生效？（对比 log 中的 `cost_limit/lambda/clip_param/...` 与配置）
- 目标指标是否改善？（例如碰撞下降、lambda 不再饱和、success 上升）
- 若无效：分析导致无效的原因。

### 4) 下一步建议（必须给出且可执行）
- 基于异常类型给出优先级明确的调整建议（参数/代码/训练策略）。
- 每条建议需说明：目标指标 + 预期变化方向 + 验证窗口（如“重新训练 200–400 iter 后观察 success、collision、lambda”）。
- 若怀疑数值问题，建议加入最小化的日志/保护以定位来源（例如 NaN 检测）。

### 推荐报告结构（输出模板）
1) 日志结论：是否收敛 + 主要异常 + 证据。
2) 历史关联：DEBUG_SUMMARY 中的相关条目 + 是否生效。
3) 原因分析：为何异常持续/复现（机制解释）。
4) 下一步建议：按优先级列出 2–4 条可执行动作。

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
- 低层策略固定，高层训练不应修改低层参数。
- 修改 lidar bin 数量/范围时，请同步更新 `GO2HighLevelCfg`，并确认观测维度自动计算正确。
- 修改 terrain 尺寸会影响边界检测与 boundary lidar；需确保 `terrain_length/width` 与 boundary 逻辑一致。
- `train_cppo.py` 在 `__main__` 中覆盖 headless 与 device 参数；需要时直接改脚本。
