# Go2 Hierarchical Constrained PPO RL

## 项目概述

本仓库实现了一个面向 Unitree Go2 的分层导航强化学习系统。底层策略负责稳定行走与速度跟踪，高层策略负责导航、避障和到达目标。高层训练采用约束马尔可夫决策过程（CMDP）上的 Constrained PPO，也就是 PPO-Lagrangian 形式的 CPPO。

与普通导航 PPO 相比，这个项目的重点不只是“更快到达目标”，还要求把碰撞和近障风险显式建模为成本，并通过拉格朗日乘子在线调节奖励与安全之间的权衡。

完整算法说明、公式推导和参数映射见 [ALGORITHM_DESIGN.md](ALGORITHM_DESIGN.md)。

## 核心特性

- 分层控制架构：高层输出速度指令，低层跟踪执行关节动作。
- CMDP 安全建模：将碰撞和近障风险统一纳入 cost 信号。
- CPPO 训练流程：同时学习 reward critic 和 cost critic，并在线更新拉格朗日乘子。
- 宏步决策封装：每个高层动作可重复执行多个低层控制步，适配“低频导航 + 高频控制”。
- 多障碍物与边界感知：高层观测包含目标方向编码和手工 lidar 风格障碍编码。
- 介绍文档中的路径统一使用仓库相对路径，不暴露本机敏感目录。

## 仓库结构

| 路径 | 作用 |
| --- | --- |
| `legged_gym_go2/legged_gym/envs/go2/go2_env.py` | Go2 底层运动环境 |
| `legged_gym_go2/legged_gym/envs/go2/high_level_navigation_env.py` | 高层导航观测与动作封装 |
| `legged_gym_go2/legged_gym/envs/go2/hierarchical_go2_env.py` | 分层环境桥接、reward/cost 计算 |
| `legged_gym_go2/legged_gym/envs/go2/go2_config.py` | 环境配置与 CPPO 超参数默认值 |
| `legged_gym_go2/legged_gym/scripts/train_cppo.py` | 高层 CPPO 训练入口 |
| `rsl_rl/rsl_rl/algorithms/cppo.py` | CPPO 算法实现 |
| `rsl_rl/rsl_rl/storage/constrained_rollout_storage.py` | 约束 rollout 存储 |
| `legged_gym_go2/deploy/` | Mujoco/实机部署相关配置与脚本 |
| `logs/` | 默认日志、checkpoint 与导出策略目录 |

## 分层控制架构

### 低层策略

- 低层策略是预训练好的 locomotion policy，不在本仓库当前训练流程中更新。
- 输入为机器人本体状态与速度命令。
- 输出为关节控制动作。
- 默认 checkpoint 路径配置在 `legged_gym_go2/legged_gym/envs/go2/go2_config.py` 的 `GO2HighLevelCfgPPO.runner.low_level_model_path`。

### 高层策略

- 高层策略观察目标、障碍物、边界和机体速度信息。
- 动作维度为 3：`[v_x, v_y, \omega]`。
- 真实发送给底层环境的命令会在 `high_level_navigation_env.py` 中进一步缩放：
  - `v_x` 乘以 `0.6`
  - `v_y` 乘以 `0.2`
  - `v_yaw` 乘以 `0.8`

### 分层交互

- 高层动作在 `HierarchicalGO2Env` 中通过 `high_level_action_repeat` 重复执行多个低层步。
- 当前默认值为 `5`，定义在 `GO2HighLevelCfg.env.high_level_action_repeat`。
- 高层 reward 和 cost 都在宏步尺度上统计，而不是直接复用底层 reward。

## 训练算法概览

高层任务被建模为 CMDP：

- reward 负责进展、成功到达和动作平滑。
- cost 负责近障风险、碰撞硬成本和终止附加成本。
- `rsl_rl/rsl_rl/algorithms/cppo.py` 中使用两个 critic：
  - `V_r(s)`：reward value
  - `V_c(s)`：cost value
- actor 更新使用拉格朗日优势：
  - `A_lag = A_r - lambda * A_c`

当前默认超参数位于 `GO2HighLevelCfgPPO.algorithm`：

| 参数 | 默认值 |
| --- | --- |
| `learning_rate` | `1e-4` |
| `clip_param` | `0.15` |
| `value_clip_param` | `0.15` |
| `num_learning_epochs` | `2` |
| `num_mini_batches` | `12` |
| `num_steps_per_env` | `200` |
| `cost_limit` | `230.0` |
| `lambda_init` | `0.0` |
| `lambda_lr` | `0.01` |
| `lambda_max` | `100.0` |

`train_cppo.py` 会把 actor 和 critic 的隐藏层都覆盖为四层 `512` 单元：

```python
train_cfg.policy.actor_hidden_dims = [512, 512, 512, 512]
train_cfg.policy.critic_hidden_dims = [512, 512, 512, 512]
```

## 高层观测、奖励与成本

### 观测

高层观测来自 `HighLevelNavigationEnv`，默认维度为 `40`：

- 基础状态 8 维：航向余弦、航向正弦、机体速度、偏航角速度、reach metric、目标方向。
- 目标方向编码 16 维：将目标投影到 body frame 的角度桶中。
- 障碍物/边界 lidar 编码 16 维：记录最近障碍或边界的强度。

### 奖励

高层 reward 在 `HierarchicalGO2Env._compute_reward()` 中构造，默认包括：

- 进展奖励：`progress_scale = 20.0`
- 动作平滑惩罚：`action_smooth_scale = 0.03`
- 到达奖励：`success_reward = 150.0`
- 近障 reward 惩罚：`reward_near_penalty_scale = 0.8`
- 碰撞惩罚：`collision_penalty = 150.0`
- 最终 reward clip：`reward_clip = 200.0`

### 成本

cost 由以下三部分组成：

- 近障成本：基于 `cost_safe_dist = 1.2`
- 碰撞成本：`cost_collision_weight = 25.0`
- 碰撞终止附加成本：`cost_collision_terminal = 150.0`

具体公式与含义见 [ALGORITHM_DESIGN.md](ALGORITHM_DESIGN.md)。

## 安装与依赖

### 系统建议

- Ubuntu 18.04 及以上
- NVIDIA GPU
- CUDA 环境与 PyTorch 版本相匹配

### 创建环境

```bash
conda create -n unitree-rl python=3.8
conda activate unitree-rl
```

### 安装 PyTorch

请按你的 CUDA 版本选择合适的 PyTorch。示例：

```bash
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

### 安装 Isaac Gym

1. 下载 Isaac Gym。
2. 进入 `isaacgym/python/`。
3. 执行：

```bash
pip install -e .
```

### 安装仓库内 Python 包

在仓库根目录执行：

```bash
pip install -e legged_gym_go2
pip install -e rsl_rl
```

## 训练

### 启动高层 CPPO 训练

```bash
python legged_gym_go2/legged_gym/scripts/train_cppo.py --headless --num_envs 32
```

### 训练前需要确认的默认路径

- 低层策略 checkpoint：`logs/rough_go2/Sep08_11-57-26_/model_18500.pt`
- 高层 resume checkpoint：`logs/high_level_go2_CPPO/20260120-171158/model_200.pt`

如果你的日志目录结构不同，请修改 `legged_gym_go2/legged_gym/envs/go2/go2_config.py` 中对应字段。

### 当前脚本入口的设备说明

`train_cppo.py` 的 `__main__` 末尾目前会直接覆盖：

- `args.headless = True`
- `args.compute_device_id = 3`
- `args.sim_device_id = 3`
- `args.rl_device = "cuda:3"`
- `args.sim_device = "cuda:3"`

如果你使用的不是 `cuda:3`，请先修改该脚本末尾的这几行。

## 日志与输出

### 默认输出目录

高层训练日志默认保存到：

```text
logs/<experiment_name>/<timestamp>/
```

其中通常 `experiment_name` 会被自动设置为 `high_level_go2_CPPO`。

目录下主要文件包括：

- `training.log`
- `model_<iteration>.pt`
- `model_final.pt`
- 数值异常时的 `nan_dump_iter*.pt`

### 日志指标

`training.log` 中会记录：

- 成功率、到达率、碰撞率、超时率
- 平均 step cost、平均 episode cost、成功 episode cost
- 拉格朗日乘子 `lambda`
- 平均目标距离、最近危险距离、命令速度、机体速度
- PPO/CPPO 更新指标，如 `policy_loss`、`value_loss`、`cost_value_loss`、`approx_kl`

## 关键配置入口

### 环境与 reward/cost

- `legged_gym_go2/legged_gym/envs/go2/go2_config.py`
  - `GO2HighLevelCfg.reward_shaping`
  - `GO2HighLevelCfg.env`
  - `GO2HighLevelCfg.rewards_ext`

### 算法超参数

- `legged_gym_go2/legged_gym/envs/go2/go2_config.py`
  - `GO2HighLevelCfgPPO.algorithm`
  - `GO2HighLevelCfgPPO.runner`

### 分层封装与 reward/cost 实现

- `legged_gym_go2/legged_gym/envs/go2/hierarchical_go2_env.py`

### 高层观测与动作映射

- `legged_gym_go2/legged_gym/envs/go2/high_level_navigation_env.py`

### CPPO 优化逻辑

- `rsl_rl/rsl_rl/algorithms/cppo.py`

## 备注

- `AGENTS.md` 与 `DEBUG_SUMMARY.md` 已加入忽略规则，并从 Git 跟踪中移除，保留为本地文档使用。
- 介绍文档中的路径均已改为项目相对路径。
