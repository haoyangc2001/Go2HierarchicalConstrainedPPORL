# Go2 分层 CPPO 导航

## 项目概述
本仓库实现了 Unitree Go2 分层强化学习导航。低层为固定的预训练运动控制器，高层采用 CMDP + CPPO（PPO-Lagrangian）进行训练，使机器人到达目标点并尽量避免与障碍物/边界碰撞。

## 关键路径
- 环境：`legged_gym_go2/legged_gym/envs/go2/`
- 训练脚本：`legged_gym_go2/legged_gym/scripts/`
- 强化学习算法：`rsl_rl/rsl_rl/algorithms/`
- 部署：`legged_gym_go2/deploy/`
- 日志/模型：`/home/caohy/repositories/MCRA_RL/logs/`

## 分层结构
- 低层（运动控制）：`legged_gym_go2/legged_gym/envs/go2/go2_env.py`
- 高层（导航封装）：`legged_gym_go2/legged_gym/envs/go2/high_level_navigation_env.py`
- 分层封装：`legged_gym_go2/legged_gym/envs/go2/hierarchical_go2_env.py`

## CPPO 训练要点
- 目标：最大化奖励且满足回合总成本约束（CMDP）。
- 高层奖励：以目标进展为主，带动作平滑惩罚与到达奖励。
- 高层成本：近障稠密成本 + 碰撞硬成本（支持加权）。
- 算法：`rsl_rl/rsl_rl/algorithms/cppo.py`
- 方案细节见：`CPPO方案指导.md`

## 训练
先激活环境：
```bash
conda activate unitree-rl
```

启动 CPPO 训练：
```bash
python legged_gym_go2/legged_gym/scripts/train_cppo.py --headless=true --num_envs=32
```

## 配置入口
- 高层奖励/成本参数：`legged_gym_go2/legged_gym/envs/go2/go2_config.py`（`GO2HighLevelCfg.reward_shaping`）
- CPPO 超参数：`legged_gym_go2/legged_gym/envs/go2/go2_config.py`（`GO2HighLevelCfgPPO`）

## 日志与模型
训练日志与模型保存到：
```
/home/caohy/repositories/MCRA_RL/logs/<experiment_name>/<timestamp>/
```

## 说明
- 低层策略固定，不参与训练。
- 高层动作会映射为速度命令并按 `high_level_action_repeat` 重复执行。
