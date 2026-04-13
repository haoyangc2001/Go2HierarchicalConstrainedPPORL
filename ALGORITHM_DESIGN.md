# Go2 Hierarchical Constrained PPO 设计说明

本文档说明本项目高层导航策略所采用的算法原理、分层交互方式、reward/cost 设计，以及这些设计如何映射到仓库中的具体配置与代码实现。本文档中的公式统一使用 GitHub Markdown 数学语法，推送到远程仓库后可直接渲染。

## 1. 问题定义

本项目将高层导航任务建模为约束马尔可夫决策过程（CMDP）：

- 状态 / 观测：机器人航向、机体速度、目标相对方向、障碍物与边界感知等高层观测。
- 动作：高层策略输出三维速度命令 $a_t = [v_x, v_y, \omega]$。
- 奖励：鼓励尽快接近目标、平滑控制并成功到达。
- 成本：显式惩罚碰撞与近障风险。
- 约束：控制每个 episode 的累计成本不超过预算。

对应的优化目标是：

$$
\max_{\theta} \ \mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^{T} r_t \right]
\quad
\text{s.t.}
\quad
\mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^{T} c_t \right] \le d
$$

其中：

- $\pi_\theta$ 是高层导航策略；
- $r_t$ 是宏步奖励；
- $c_t$ 是宏步成本；
- $d$ 是允许的平均 episode 总成本预算。

本项目采用的是 on-policy、model-free 的 PPO-Lagrangian 形式，也就是常说的 CPPO。

## 2. 分层控制架构

### 2.1 总体结构

系统由两层策略组成：

1. 低层 locomotion policy
   - 已预训练并固定。
   - 负责把速度命令转换为关节动作。
2. 高层 navigation policy
   - 需要训练。
   - 负责根据环境感知生成速度命令。

代码映射：

- 低层环境：`legged_gym_go2/legged_gym/envs/go2/go2_env.py`
- 高层观测/动作包装：`legged_gym_go2/legged_gym/envs/go2/high_level_navigation_env.py`
- 分层桥接：`legged_gym_go2/legged_gym/envs/go2/hierarchical_go2_env.py`

### 2.2 宏步执行

高层一步并不只对应一个物理仿真步，而是会重复执行多个低层控制步。设重复次数为 $K$，则：

$$
a_t^{high} \rightarrow
\underbrace{a_{t,1}^{low}, a_{t,2}^{low}, \dots, a_{t,K}^{low}}_{\text{由固定低层策略跟踪执行}}
$$

本项目中该参数是：

- `GO2HighLevelCfg.env.high_level_action_repeat = 5`

实现位置：

- `HierarchicalGO2Env.step()`

宏步设计的意义：

- 高层策略以较低频率思考导航；
- 低层策略以较高频率维持稳定行走；
- reward/cost 在高层时间尺度汇总，更符合导航任务的决策粒度。

## 3. 高层观测与动作

## 3.1 动作空间

高层动作维度为 3：

$$
a_t = [v_x, v_y, \omega]
$$

动作先被裁剪到 $[-1, 1]$，再经过两级缩放：

1. 配置缩放：`GO2HighLevelCfg.action_scale = [1.0, 1.0, 1.2]`
2. 环境映射缩放：

$$
\begin{aligned}
v_x^{cmd} &= 0.6 \cdot v_x \\
v_y^{cmd} &= 0.2 \cdot v_y \\
\omega^{cmd} &= 0.8 \cdot \omega
\end{aligned}
$$

实现位置：

- `HighLevelNavigationEnv.update_velocity_commands()`

### 3.2 观测空间

默认高层观测总维度为 40：

$$
8\ (\text{基础状态}) + 16\ (\text{目标编码}) + 16\ (\text{障碍/边界编码}) = 40
$$

其中：

- 基础 8 维
  - $\cos(\text{heading})$
  - $\sin(\text{heading})$
  - body frame 下的 $v_x, v_y$
  - yaw rate
  - reach metric
  - 目标方向单位向量的两个分量
- 目标编码 16 维
  - 把目标在 body frame 下投影到角度桶中
- 障碍/边界编码 16 维
  - 把障碍物表面距离和边界距离编码为手工 lidar 风格强度

对应实现：

- `HighLevelNavigationEnv._compute_high_level_observations()`

## 4. 奖励设计

高层 reward 在 `HierarchicalGO2Env._compute_reward()` 中定义。核心目标是“向目标推进 + 到达成功 + 平滑控制”，并在必要时额外惩罚碰撞和近障行为。

### 4.1 进展奖励

设当前目标距离为 $d_t$，上一宏步目标距离为 $d_{t-1}$，则进展为：

$$
\text{progress}_t = d_{t-1} - d_t
$$

奖励项为：

$$
r_t^{prog} = w_{prog} \cdot \text{progress}_t
$$

默认参数：

- `progress_scale = 20.0`

### 4.2 动作平滑惩罚

设当前和上一步高层指令分别为 $a_t, a_{t-1}$，则：

$$
\Delta a_t = \|a_t - a_{t-1}\|_2
$$

平滑惩罚为：

$$
r_t^{smooth} = - w_{smooth} \cdot \Delta a_t
$$

默认参数：

- `action_smooth_scale = 0.03`

### 4.3 成功奖励

若目标距离小于成功阈值 $d_{goal}$，并且该宏步发生 episode 结束且无碰撞，则视为成功：

$$
\text{success}_t = \mathbb{I}[d_t \le d_{goal}] \cdot \mathbb{I}[\text{done}] \cdot \mathbb{I}[\neg \text{collision}]
$$

对应奖励：

$$
r_t^{success} = R_{success} \cdot \text{success}_t
$$

默认参数：

- `goal_reached_dist = 0.3`
- `success_reward = 150.0`

### 4.4 碰撞惩罚与近障 reward shaping

如果发生碰撞，还会额外施加 reward 惩罚：

$$
r_t^{coll} = - \beta_{coll} \cdot \mathbb{I}[\text{collision}]
$$

近障 risk 也可直接参与 reward：

$$
r_t^{near} = - \beta_{near} \cdot c_t^{near}
$$

默认参数：

- `collision_penalty = 150.0`
- `reward_near_penalty_scale = 0.8`

### 4.5 总奖励

因此高层单步 reward 为：

$$
r_t = w_{prog}(d_{t-1} - d_t)
- w_{smooth}\|a_t - a_{t-1}\|_2
+ R_{success}\,\text{success}_t
- \beta_{coll}\,\mathbb{I}[\text{collision}]
- \beta_{near}\,c_t^{near}
$$

在实现中，reward 还会经过全局缩放与裁剪：

$$
r_t \leftarrow \text{clip}(s_r \cdot r_t,\ -r_{clip},\ r_{clip})
$$

当前默认值：

- `reward_scale = 1.0`
- `reward_clip = 200.0`

## 5. 成本设计

本项目的 cost 是 CMDP 约束的核心。成本在宏步尺度上聚合，由近障风险、碰撞和碰撞终止附加成本组成。

### 5.1 近障成本

设最近危险距离为 $d_t^{hazard}$，安全距离阈值为 $d_{safe}$，则近障成本定义为：

$$
c_t^{near} =
\max\left(
0,\ \frac{d_{safe} - d_t^{hazard}}{d_{safe}}
\right)
$$

性质：

- 当 $d_t^{hazard} \ge d_{safe}$ 时，成本为 0；
- 离障碍越近，成本越大；
- 在宏步内会对多个低层步累积。

默认参数：

- `cost_safe_dist = 1.2`

### 5.2 碰撞成本

碰撞判定采用最近危险距离与碰撞阈值比较：

$$
c_t^{coll} = \mathbb{I}[d_t^{hazard} \le d_{coll}]
$$

默认参数：

- `collision_dist = 0.35`

### 5.3 终止碰撞附加成本

如果发生终止碰撞，可以再施加一次大额成本：

$$
c_t^{terminal} = \alpha_{terminal} \cdot \mathbb{I}[\text{collision}]
$$

默认参数：

- `cost_collision_terminal = 150.0`

### 5.4 总成本

因此单步总成本为：

$$
c_t = w_{coll} c_t^{coll} + w_{near} c_t^{near} + c_t^{terminal}
$$

默认参数：

- `cost_collision_weight = 25.0`
- `cost_near_weight = 0.3`

## 6. CPPO 目标函数

PPO-Lagrangian 的思想是把原始约束问题转成拉格朗日松弛问题：

$$
\mathcal{L}(\theta, \lambda)
=
\mathbb{E}\left[\sum_{t=0}^{T} \left(r_t - \lambda c_t\right)\right]
+
\lambda d
$$

其中 $\lambda \ge 0$ 是拉格朗日乘子。

直观理解：

- 当平均 episode cost 超过 `cost_limit` 时，$\lambda$ 增大，策略更保守；
- 当平均 episode cost 低于限制时，$\lambda$ 减小，策略更重视到达效率。

## 7. Reward Critic 与 Cost Critic

本项目不是只学习一个价值函数，而是同时学习：

- reward critic：$V_r(s)$
- cost critic：$V_c(s)$

对应实现：

- `rsl_rl/rsl_rl/modules/actor_critic_cppo.py`
- `rsl_rl/rsl_rl/algorithms/cppo.py`

这样做的原因是 reward 和 cost 来自不同目标，需要分别做 bootstrap、GAE 和 value regression。

## 8. GAE 与优势函数

### 8.1 Reward TD 残差

$$
\delta_t^r = r_t + \gamma V_r(s_{t+1}) - V_r(s_t)
$$

### 8.2 Cost TD 残差

$$
\delta_t^c = c_t + \gamma_c V_c(s_{t+1}) - V_c(s_t)
$$

### 8.3 两套 GAE

$$
A_t^r = \text{GAE}(\delta_t^r; \gamma, \lambda)
$$

$$
A_t^c = \text{GAE}(\delta_t^c; \gamma_c, \lambda_c)
$$

当前实现中：

- 若未显式指定，则 `cost_gamma = gamma`
- 若未显式指定，则 `cost_lam = lam`

### 8.4 拉格朗日优势

actor 更新用的不是普通 reward advantage，而是：

$$
A_t^{lag} = A_t^r - \lambda A_t^c
$$

这就是安全约束真正进入策略更新的地方。

## 9. Actor 与 Critic 更新

### 9.1 PPO actor 目标

设概率比为：

$$
\rho_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}
$$

则 actor 的 clipped objective 为：

$$
L_{actor}(\theta)
=
\mathbb{E}\left[
\min\left(
\rho_t A_t^{lag},
\text{clip}(\rho_t, 1-\epsilon, 1+\epsilon)A_t^{lag}
\right)
\right]
$$

实现位置：

- `CPPO.update()`

### 9.2 Reward value loss

$$
L_{V_r} = \mathbb{E}\left[(V_r(s_t) - \hat{R}_t)^2\right]
$$

### 9.3 Cost value loss

$$
L_{V_c} = \mathbb{E}\left[(V_c(s_t) - \hat{C}_t)^2\right]
$$

### 9.4 总损失

训练时实际优化的是：

$$
L =
L_{actor}
+ \alpha_r L_{V_r}
+ \alpha_c L_{V_c}
- \beta_H \mathcal{H}[\pi_\theta]
$$

默认权重：

- `value_loss_coef = 0.5`
- `cost_value_loss_coef = 0.5`
- `entropy_coef = 0.008`

## 10. 拉格朗日乘子更新

每轮 rollout 结束后，算法会统计 batch 中已结束 episode 的平均总成本：

$$
\bar{C} = \frac{1}{N}\sum_{i=1}^{N} \sum_{t=0}^{T_i} c_t^{(i)}
$$

然后按照对偶上升更新：

$$
\lambda \leftarrow
\Pi_{[0, \lambda_{max}]}
\left(
\lambda + \eta_\lambda (\bar{C} - d)
\right)
$$

其中：

- $d$ 对应 `cost_limit`
- $\eta_\lambda$ 对应 `lambda_lr`
- $\lambda_{max}$ 对应 `lambda_max`

默认参数：

- `cost_limit = 230.0`
- `lambda_init = 0.0`
- `lambda_lr = 0.01`
- `lambda_max = 100.0`

## 11. Time-out Bootstrapping

在 `CPPO.process_env_step()` 中，若 episode 因 time-out 截断而不是自然终止，则 reward 和 cost 都会做 bootstrap：

$$
r_t \leftarrow r_t + \gamma V_r(s_{t+1})
$$

$$
c_t \leftarrow c_t + \gamma_c V_c(s_{t+1})
$$

这样可以减少固定时长截断带来的偏差。

## 12. 成功、碰撞与终止判定

在 `HierarchicalGO2Env._compute_reward()` 中：

- 到达判定：

$$
\text{reached}_t = \mathbb{I}[d_t \le d_{goal}]
$$

- 碰撞判定：

$$
\text{collision}_t = \mathbb{I}[d_t^{hazard} \le d_{coll}] \cdot \mathbb{I}[\text{done}]
$$

- 成功判定：

$$
\text{success}_t = \text{reached}_t \cdot \mathbb{I}[\text{done}] \cdot \mathbb{I}[\neg \text{collision}]
$$

这里特别注意：

- done 标志以底层环境真实 reset 为准，避免高低层不同步；
- 成功并不是“进入目标半径就算成功”，而是“到达并在该宏步结束且没有碰撞”。

## 13. 配置与代码映射

### 13.1 环境相关

`legged_gym_go2/legged_gym/envs/go2/go2_config.py`

- `GO2HighLevelCfg.env`
  - `num_actions = 3`
  - `high_level_action_repeat = 5`
  - `episode_length_s = 40`
- `GO2HighLevelCfg.reward_shaping`
  - reward/cost 相关参数
- `GO2HighLevelCfg.rewards_ext`
  - 障碍物位置、边界、目标球半径等

### 13.2 算法相关

`GO2HighLevelCfgPPO.algorithm`

- `learning_rate = 1e-4`
- `clip_param = 0.15`
- `value_clip_param = 0.15`
- `num_learning_epochs = 2`
- `num_mini_batches = 12`
- `num_steps_per_env = 200`
- `max_grad_norm = 0.5`
- `cost_limit = 230.0`

### 13.3 训练脚本相关

`legged_gym_go2/legged_gym/scripts/train_cppo.py`

- 将 actor / critic 隐藏层统一设为 `[512, 512, 512, 512]`
- 创建日志目录：`logs/<experiment_name>/<timestamp>/`
- 记录训练统计到 `training.log`
- 周期性保存 `model_<iteration>.pt`

## 14. 调参建议

### 14.1 如果机器人太激进、碰撞率过高

优先检查：

- `cost_collision_weight`
- `cost_collision_terminal`
- `cost_limit`
- `high_level_action_repeat`
- `action_scale`

典型现象：

- `lambda` 快速升高
- `collision_rate` 居高不下
- `min_hazard` 长期偏低

### 14.2 如果机器人过于保守、不愿意接近目标

可以考虑：

- 提高 `cost_limit`
- 降低 `cost_near_weight`
- 降低 `reward_near_penalty_scale`
- 适度提高 `progress_scale`

### 14.3 如果策略抖动明显

优先检查：

- `action_smooth_scale`
- `clip_param`
- `desired_kl`
- `learning_rate`

## 15. 默认关键参数一览

| 类别 | 参数 | 默认值 |
| --- | --- | --- |
| 宏步 | `high_level_action_repeat` | `5` |
| 成功阈值 | `goal_reached_dist` | `0.3` |
| 碰撞阈值 | `collision_dist` | `0.35` |
| 进展奖励 | `progress_scale` | `20.0` |
| 平滑惩罚 | `action_smooth_scale` | `0.03` |
| 成功奖励 | `success_reward` | `150.0` |
| 近障 reward 惩罚 | `reward_near_penalty_scale` | `0.8` |
| 近障安全距离 | `cost_safe_dist` | `1.2` |
| 碰撞 cost 权重 | `cost_collision_weight` | `25.0` |
| 终止碰撞 cost | `cost_collision_terminal` | `150.0` |
| 近障 cost 权重 | `cost_near_weight` | `0.3` |
| 碰撞 reward 惩罚 | `collision_penalty` | `150.0` |
| 学习率 | `learning_rate` | `1e-4` |
| PPO clip | `clip_param` | `0.15` |
| rollout 长度 | `num_steps_per_env` | `200` |
| 成本预算 | `cost_limit` | `230.0` |
| 拉格朗日学习率 | `lambda_lr` | `0.01` |

## 16. 小结

本项目的核心不是“在 PPO 上简单加一个碰撞惩罚”，而是完整地把高层导航问题写成 CMDP，再通过 CPPO 同时处理：

- 到达目标的效率；
- 近障与碰撞的安全约束；
- reward critic 与 cost critic 的双价值估计；
- 拉格朗日乘子的在线自适应调节。

如果你要继续优化训练表现，最值得联动观察的量是：

- `success`
- `collision`
- `cost`
- `lambda`
- `goal_dist`
- `min_hazard`

这些指标会直接反映 reward、cost 和 `cost_limit` 是否处在合理平衡区间。
