# CPPO DEBUG Summary

日期：2026-01-18
范围：CPPO 训练日志分析 + 代码改动 + 下一步建议

## 日志分析结论
- 日志文件：`logs/high_level_go2_CPPO/20260118-192753/training.log`
- 现象：成功率前期上升，中期达到峰值，后期快速下降；同时碰撞率飙升、回合长度显著缩短。
- 证据（关键迭代）：
  - iter 00111：`success=0.115`，`collision=0.506`，`timeout=0.380`，`ep_len_mean=62.6`
  - iter 00157：`success=0.006`，`collision=0.865`，`timeout=0.129`，`ep_len_mean=40.7`
  - iter 00172：`success=0.001`，`collision=0.937`，`timeout=0.061`，`ep_len_mean=34.3`
- 主要原因：
  - `cost_limit=10` 远小于实际每回合总成本（约 120–180），导致拉格朗日乘子很早触顶并长期饱和。
  - 存在稠密近障成本时，缩短回合会显著降低总成本，碰撞反而变成满足约束的“捷径”。
  - 训练后期稳定性变差（`approx_kl` 高、`grad_norm` 大、学习率降到最小），加速成功率崩塌。

## 代码改动记录
- 新增 CPPO 算法（PPO-Lagrangian，包含成本 critic、成本 GAE、拉格朗日更新）：
  - `rsl_rl/rsl_rl/algorithms/cppo.py`
- 更新 PPO 支持公共超参数：
  - `rsl_rl/rsl_rl/algorithms/ppo.py`（value clip 参数、自适应学习率上下限）
- 使分层环境的成本与 CMDP 设计一致：
  - `legged_gym_go2/legged_gym/envs/go2/hierarchical_go2_env.py`（按 low-level repeat 累计 cost）
- 约束预算与碰撞成本重标定：
  - `legged_gym_go2/legged_gym/envs/go2/go2_config.py`
    - `GO2HighLevelCfgPPO.algorithm.cost_limit = 150.0`
    - `GO2HighLevelCfg.reward_shaping.cost_collision_weight = 5.0`
- 精简日志输出，只保留有参考价值的指标：
  - `legged_gym_go2/legged_gym/scripts/train_cppo.py`
  - `legged_gym_go2/legged_gym/scripts/train_reward_shaping.py`

## 下一步建议
1) 重新训练 CPPO，观察 `cost` 是否能稳定收敛到 `cost_limit` 附近，同时成功率不再崩塌。
2) 若碰撞仍占主导，继续提高 `cost_collision_weight`（例如 10–20），或在碰撞终止时增加显式的 terminal cost。
3) 若 `approx_kl` 和 `grad_norm` 仍偏高，降低学习率或增大 batch size 以稳定更新。



## 2026-01-18 CPPO 成功率异常收敛分析
- 日志：`/home/caohy/repositories/MCRA_RL/logs/high_level_go2_CPPO/20260118-224710/training.log`
- 结论：
  - 成本尺度与约束不匹配：高层成本按步累加，`cost_limit=150` 偏低，拉格朗日乘子早期快速抬升并长期偏高，导致策略偏向“尽快结束回合”。
  - 碰撞提前终止导致“省成本捷径”：碰撞缩短回合、降低累计成本，成功率在中期见顶后快速下滑，碰撞率显著上升。
  - 更新不稳定放大崩塌：`approx_kl`、`clip_frac`、`grad_norm` 较高，策略在约束压力下出现明显漂移。
- 代码变更：
  - `legged_gym_go2/legged_gym/envs/go2/go2_config.py`
    - `cost_collision_weight: 5.0 -> 10.0`
    - 新增 `cost_collision_terminal = 20.0`
    - `cost_limit: 150.0 -> 300.0`
    - 稳定性调整：`clip_param/value_clip_param: 0.2 -> 0.15`，`desired_kl: 0.03 -> 0.02`，`max_grad_norm: 1.0 -> 0.5`
  - `legged_gym_go2/legged_gym/envs/go2/hierarchical_go2_env.py`
    - 添加 terminal 碰撞成本项 `cost_collision_terminal`，并在 `infos` 中记录。
- 下一步建议：
  - 重新训练并观察 `cost`、`lambda`、`collision`、`ep_len_mean` 是否不再“碰撞省成本”。
  - 若碰撞仍主导：进一步提高 `cost_collision_terminal`（例如 30–40）或 `cost_collision_weight`（例如 15）。
- 若学习过慢或仍不稳定：适度回调 `clip_param` 或 `learning_rate`，同时监控 `approx_kl` 与 `grad_norm`。

## 2026-01-19 CPPO 成功率异常收敛分析（基于 20260119-094846）
- 日志：`/home/caohy/repositories/MCRA_RL/logs/high_level_go2_CPPO/20260119-094846/training.log`
- 结论：
  - 约束未生效：`cost_limit=300` 明显高于当前 episode cost（~90–150），导致 `lambda` 长期为 0，CPPO 退化为 PPO，碰撞成本无法约束策略。
  - 奖励结构倾向“速度优先”：reward 只鼓励进展、无显式碰撞惩罚；高层动作放大 + 重复执行使策略更易撞障，成功率后期崩塌。
  - 更新不稳定：`approx_kl`、`clip_frac`、`grad_norm` 在后期显著升高，极端时出现数值异常，放大了策略漂移与碰撞率上升。
- 代码变更：
  - `legged_gym_go2/legged_gym/envs/go2/go2_config.py`
    - `cost_limit: 300.0 -> 120.0`（让 `lambda` 有机会进入非零区间）
  - `legged_gym_go2/legged_gym/envs/go2/high_level_navigation_env.py`
    - 高层观测添加 `nan_to_num` 保护，避免 NaN/Inf 触发 `Normal` 构造报错
- 下一步建议：
  - 重新训练，重点观察 `lambda` 是否从 0 抬升、`cost` 是否围绕 `cost_limit` 稳定。
  - 若仍出现 NaN/Inf，可增加轻量日志统计（观测/动作数值是否有限）定位触发条件。
  - 若碰撞率仍偏高，考虑下调 `action_scale[0]` 或减小 `high_level_action_repeat` 以提升避障反应。

## 2026-01-20 CPPO 成功率异常收敛分析（基于 20260120-145909）
- 日志：`/home/caohy/repositories/MCRA_RL/logs/high_level_go2_CPPO/20260120-145909/training.log`
- 结论：
  - 约束过紧导致 `lambda` 快速饱和：episode cost 长期在 ~190–220，而 `cost_limit=120`，`lambda` 在 20 多次迭代内触顶 100，策略几乎只优化成本，成功率长期低位（最高 ~0.037）。
  - 近障成本主导，仍存在“碰撞省成本”动机：`cost_near` 明显高于 `cost_collision`，缩短回合可以快速降低累计成本，导致碰撞率上升、`ep_len_mean` 下降。
  - 更新稳定性仍偏弱：`grad_norm` 多次大幅升高，`approx_kl` 持续偏高，进一步放大策略漂移和碰撞率上升。
- 代码变更：
  - `legged_gym_go2/legged_gym/envs/go2/go2_config.py`
    - `cost_near_weight: 1.0 -> 0.5`（降低稠密近障成本规模，缓解 `lambda` 过快饱和）
- 下一步建议：
  - 重新训练，观察 `cost_near`、`cost` 是否下降，`lambda` 是否不再长期顶满。
  - 若 `lambda` 仍快速饱和，可进一步下调 `cost_near_weight`（例如 0.3）或减小 `cost_safe_dist`。
  - 若碰撞率仍上升，考虑提高 `cost_collision_terminal` 或单独增加碰撞终止惩罚以打破“碰撞省成本”路径。

## 2026-01-21 CPPO 成功率异常收敛分析（基于 20260120-171158）
- 日志：`/home/caohy/repositories/MCRA_RL/logs/high_level_go2_CPPO/20260120-171158/training.log`
- 结论：
  - 约束未生效：`lambda` 全程为 0，`cost_limit=120` 高于实际 episode cost（后期约 45–55），CPPO 退化为 PPO，成功率中期见顶后下滑。
  - 速度驱动导致碰撞上升：`cmd_speed/body_speed` 持续升高、`ep_len_mean` 下降，碰撞率后期攀升，成功率异常收敛。
  - 数值不稳定导致崩塌：后期 `approx_kl/clip_frac` 走高，`grad_norm` 爆炸到 `inf`，actor 输出 NaN，触发 `Normal` 构造报错。
- 代码变更：
  - `legged_gym_go2/legged_gym/envs/go2/go2_config.py`
    - `cost_limit: 120.0 -> 100.0`
    - `cost_collision_weight: 10.0 -> 15.0`
    - `cost_collision_terminal: 20.0 -> 40.0`
    - `cost_near_weight: 0.5 -> 0.4`
    - 新增 `collision_penalty = 100.0`（碰撞终止奖励惩罚）
  - `legged_gym_go2/legged_gym/envs/go2/hierarchical_go2_env.py`
    - 碰撞终止时加入 `collision_penalty` 负奖励，直接压制碰撞策略。
  - `rsl_rl/rsl_rl/modules/actor_critic_cppo.py`
    - 对 actor/critic 输入与 actor 输出做 `nan_to_num`，防止 `Normal` 接收 NaN。
  - `rsl_rl/rsl_rl/algorithms/cppo.py`
    - 遇到非有限 `loss` 或 `grad_norm` 时跳过 batch 更新，避免 NaN 污染模型。
    - 统计非有限 batch 次数与实际更新次数。
  - `legged_gym_go2/legged_gym/scripts/train_cppo.py`
    - 训练日志增加 `nan_loss` / `nan_grad` / `updates` 字段，便于定位 NaN/Inf 触发频率。
- 下一步建议：
  - 重新训练 200–400 iter，重点观察 `lambda` 是否抬升、`nan_loss/nan_grad` 是否为 0、成功率是否不再后期崩塌。
  - 若 `lambda` 仍为 0：继续下调 `cost_limit`（例如 80–90）或适度增大 `cost_collision_weight`。
  - 若 NaN 仍出现：进一步排查观测来源（例如 lidar/边界计算）是否在极端状态下产生异常值。

## 2026-01-22 CPPO 成功率与稳定性分析（基于 20260121-223754）
- 日志：`/home/caohy/repositories/MCRA_RL/logs/high_level_go2_CPPO/20260121-223754/training.log`
- 结论：
  - 成功率未稳定收敛：峰值约 `0.31`（iter ~369），后期回落并稳定在 `~0.24–0.26`（最后 30 iter 均值 ~0.25）。
  - 约束介入过晚：`cost_limit=105` 高于前期 episode cost（前 100 iter 平均 ~92），`lambda` 直到 iter ~406 才开始非零，前半程基本退化为 PPO。
  - 训练稳定性一般：`approx_kl` 长期偏高（最高 ~0.123，末尾 ~0.043），`clip_frac` 偏高；自适应学习率在 iter ~87 下降到 `min_lr=5e-6`，后期更新幅度偏小。
  - 安全与到达均未改善：`collision` 长期 ~0.27–0.31，`timeout` ~0.45；`goal_dist` 基本维持在 ~4.2，`progress` 很低，成功率难继续提升。
- 代码变更：
  - `legged_gym_go2/legged_gym/envs/go2/go2_config.py`
    - `GO2HighLevelCfg.env.high_level_action_repeat: 10 -> 6`（提高高层反应频率）
    - `cost_collision_weight: 15.0 -> 20.0`
    - `cost_collision_terminal: 40.0 -> 60.0`
    - `cost_near_weight: 0.4 -> 0.3`
    - `cost_limit: 105.0 -> 95.0`（让 `lambda` 更早进入非零区间）
    - `learning_rate: 3e-4 -> 2e-4`
    - `max_lr: 5e-4 -> 3e-4`
    - `desired_kl: 0.02 -> 0.04`（减轻 KL 约束压力，避免过早降到 min_lr）
- 下一步建议：
  - 重新训练 200–400 iter，重点看 `lambda` 是否在前 50–100 iter 进入非零区间、`cost` 是否围绕 `cost_limit` 波动、成功率是否稳定上升到 >0.3。
  - 若 `collision` 仍高于 0.25：继续提高 `cost_collision_terminal`（例如 80）或 `cost_collision_weight`（例如 25），并适度下调 `cost_near_weight`（例如 0.25）。
  - 若 `approx_kl/clip_frac` 仍偏高：考虑降低 `num_learning_epochs` 或增加 `num_mini_batches`，进一步平滑更新。

## 2026-01-22 CPPO 成功率与稳定性分析（基于 20260122-101523）
- 日志：`/home/caohy/repositories/MCRA_RL/logs/high_level_go2_CPPO/20260122-101523/training.log`
- 结论：
  - 成功率未收敛且后期回落：峰值约 `0.385`（iter 228），后期稳定在 `~0.24–0.28`（最后 100 iter 均值 ~0.28），趋势轻微下降。
  - 碰撞率偏高并主导终止：最后 100 iter `collision` 均值 ~0.46（范围 ~0.36–0.55），障碍碰撞占比高于边界碰撞。
  - 约束未生效：episode `cost` 约 72–85，明显低于 `cost_limit=95`，`lambda` 全程为 0，CPPO 退化为 PPO。
  - 更新不稳定/早衰：`approx_kl`、`clip_frac` 偏高，`value_loss` 后期显著增大；学习率在 iter ~337 降到 `min_lr=5e-6` 后更新幅度不足。
- 代码变更：
  - `legged_gym_go2/legged_gym/envs/go2/go2_config.py`
    - `cost_limit: 95.0 -> 80.0`（使 `lambda` 更早进入非零区间）
    - `collision_penalty: 100.0 -> 180.0`（强化碰撞终止负奖励）
    - `cost_collision_terminal: 60.0 -> 90.0`（碰撞在 cost 上显著“超标”）
    - 新增 `reward_near_penalty_scale = 20.0`（稠密近障惩罚项）
    - `high_level_action_repeat: 6 -> 5`（提高高层反应频率）
    - `num_mini_batches: 4 -> 8`（平滑更新）
  - `legged_gym_go2/legged_gym/envs/go2/hierarchical_go2_env.py`
    - reward 增加稠密近障惩罚项：`- reward_near_penalty_scale * cost_near`
- 下一步建议：
  - 重新训练 200–400 iter，重点观察 `lambda` 是否在前 50–100 iter 进入非零区间，`cost` 是否围绕 `cost_limit` 波动，`collision` 是否下降到 <0.35。
  - 若 `lambda` 仍为 0：继续下调 `cost_limit`（例如 70–75）或适度上调 `cost_near_weight`。
  - 若 `approx_kl/clip_frac` 仍偏高：考虑降低 `num_learning_epochs` 或进一步增大 `num_mini_batches`，稳定更新步长。



# 2026-01-22 训练日志分析与改动记录（20260122-161921）

## 一、日志结论（是否收敛/异常）
- 成功率低且未改善：最近 100 次迭代 success 均值≈0.131，范围 0.084–0.188。
- 碰撞极高且占主导：collision≈0.869，timeout=0，回合几乎全部以碰撞结束。
- 约束过紧导致饱和：cost 长期 120–145，高于 cost_limit=80，lambda 快速升至 100 并持续饱和。
- 数值不稳定明显：value_loss 达 1e5 量级、grad_norm 1e4–1e5，更新被大量 clip。
- 探索坍塌：action_std 与 entropy 下降，策略趋于小动作但仍高碰撞。

## 二、本次代码变更（回调惩罚强度）
1) 降低碰撞终止成本，保留 cost 侧近障惩罚
   - 文件：`legged_gym_go2/legged_gym/envs/go2/go2_config.py`
   - 修改：
     - `cost_collision_terminal` 90.0 → 75.0
     - 移除 `reward_near_penalty_scale`

2) 取消 reward 侧稠密近障惩罚项
   - 文件：`legged_gym_go2/legged_gym/envs/go2/hierarchical_go2_env.py`
   - 修改：删除 `_compute_reward` 中 `- reward_near_penalty_scale * cost_near`

## 三、下一步建议（按顺序）
1) 重新训练 200–400 iter，重点观察 lambda 是否不再长期饱和、collision 是否下降到 <0.35。
2) 若 lambda 仍饱和：继续下调 `cost_collision_terminal`（如 60–70）或适度上调 `cost_limit`（如 90–100）。
3) 若仍数值不稳：考虑降低 `collision_penalty` 或整体 reward scale，缓解 value_loss/grad_norm 爆炸。



# 2026-01-22 训练日志分析与改动记录（20260122-210018）

## 一、日志结论（是否收敛/异常）
- 成功率有提升但未稳定收敛：最佳 50-iter success 均值≈0.421（iter 1150–1199），末尾 100 iter 均值≈0.358，呈平台后轻微回落。
- 碰撞仍占主导：末尾 100 iter collision≈0.641（范围 0.584–0.700），timeout=0，回合主要由碰撞/到达结束。
- 约束持续饱和：cost 均值≈87.8（范围 78.7–98.6）高于 cost_limit=80，lambda 在 iter≈86 后长期=100。
- 更新稳定性一般：approx_kl≈0.075、clip_frac≈0.49，lr 早期降到 min_lr=5e-6，grad_norm 波动较大但无 NaN。
- 探索收缩但未解决碰撞：action_std≈0.231、entropy≈-0.31，策略趋于保守但碰撞率仍高。

## 二、本次代码变更（防“卡死”+稳定更新+日志细化）
1) 放宽成本预算，避免拉格朗日乘子长期饱和
   - 文件：`legged_gym_go2/legged_gym/envs/go2/go2_config.py`
   - 修改：`cost_limit` 80.0 → 90.0

2) 降低前向速度上限并提升转向响应
   - 文件：`legged_gym_go2/legged_gym/envs/go2/go2_config.py`
   - 修改：`action_scale` [1.3, 1.0, 1.0] → [1.0, 1.0, 1.2]

3) 提高 mini-batch 数量以平滑更新
   - 文件：`legged_gym_go2/legged_gym/envs/go2/go2_config.py`
   - 修改：`num_mini_batches` 8 → 12

4) 细分 episode 成本统计（成功/碰撞）
   - 文件：`legged_gym_go2/legged_gym/scripts/train_cppo.py`
   - 修改：新增 `cost_success`、`cost_collision_ep` 日志字段

## 三、下一步建议（按顺序）
1) 重新训练 200–400 iter，重点观察 `lambda` 是否不再长期顶满、`cost_success` 是否低于 `cost_limit`、`collision` 是否下降到 <0.5。
2) 若 `lambda` 仍饱和：继续上调 `cost_limit`（如 95–110）或适度下调 `cost_collision_terminal`（如 60–70）。
3) 若碰撞仍高：继续降低前向上限（或增大 `vyaw` 比例），并检查 `cost_collision_terminal` 与 `collision_penalty` 的平衡是否过强抑制探索。
4) 若 `approx_kl/clip_frac` 仍偏高：考虑将 `num_learning_epochs` 从 3 降到 2，或进一步增大 `num_mini_batches`。



# 2026-01-23 训练日志分析与改动记录（20260123-133702）

## 一、日志结论（是否收敛/异常）
- 未稳定收敛：success 在中期达到峰值后回落，最佳 50-iter 均值≈0.396（iter 295–344），末尾 100 iter 均值≈0.299。
- 碰撞仍占主导：末尾 100 iter collision≈0.701，timeout=0，回合多数以碰撞结束。
- 约束仍偏紧、乘子高位：cost 末尾≈92.4 高于 cost_limit=90，lambda 末尾≈90 且在 iter≈120 后多次触顶（≈29% 迭代 lambda=100）。
- 训练稳定性一般：approx_kl≈0.040、clip_frac≈0.306；grad_norm 末尾均值≈105，最大到 766，学习率早降到 min_lr（8e-6）。
- 探索收缩：action_std≈0.20、entropy≈-0.63，策略趋于保守但碰撞率未下降。

## 二、本次代码变更（缓解饱和 + 提升探索 + 直接避障梯度）
1) 降低拉格朗日更新步长
   - 文件：`legged_gym_go2/legged_gym/envs/go2/go2_config.py`
   - 修改：`lambda_lr` 0.02 → 0.01

2) 恢复小尺度近障奖励惩罚
   - 文件：`legged_gym_go2/legged_gym/envs/go2/go2_config.py`
   - 修改：新增 `reward_near_penalty_scale = 3.0`
   - 文件：`legged_gym_go2/legged_gym/envs/go2/hierarchical_go2_env.py`
   - 修改：`_compute_reward` 中加入 `- reward_near_penalty_scale * cost_near`

3) 提升探索强度
   - 文件：`legged_gym_go2/legged_gym/envs/go2/go2_config.py`
   - 修改：`entropy_coef` 0.003 → 0.008

## 三、下一步建议（按顺序）
1) 重新训练 200–400 iter，重点观察 `lambda` 是否不再长期高位、`cost` 是否围绕 90 波动、`collision` 是否下降到 <0.6。
2) 若 `lambda` 仍接近饱和：在保持 cost_limit=90 前提下，考虑继续下调 `cost_near_weight`（如 0.2）或降低 `lambda_lr` 到 0.005。
3) 若碰撞仍高：降低前向上限或进一步降低 `high_level_action_repeat`（如 4），提升避障反应速度。



# 2026-01-24 训练日志分析与改动记录（20260123-202452）

## 一、日志结论（是否收敛/异常）
- 成功率先上升后期崩塌：best 50-iter success 均值≈0.343（iter 358 结束），末尾 100 iter success 均值≈0.0615。
- 碰撞主导终止：末尾 100 iter collision≈0.938，timeout=0；障碍碰撞占比高于边界（obstacle_collision_rate≈0.582 > boundary_collision_rate≈0.356）。
- 约束长期饱和：cost≈133.3 显著高于 cost_limit=90，lambda=100 长期饱和（≈68.5% 迭代 λ≥99）。
- 价值函数不稳定：value_clip_frac≈0.975、grad_norm≈7.9e3（峰值 2.17e4）；虽无 NaN（nan_loss/nan_grad=0），但 critic 失真严重。
- 探索塌缩与行为收缩：action_std≈0.156、entropy≈-1.60、cmd_speed≈0.061，仍高碰撞；goal_dist 末期≈4.37、progress≈0.00469，进展停滞。

## 二、本次代码变更（降低 reward 侧成本尺度）
- 文件：`legged_gym_go2/legged_gym/envs/go2/go2_config.py`
- 修改：
  - `reward_near_penalty_scale: 3.0 -> 1.5`

## 三、下一步建议（按顺序）
1) 重新训练 200–400 iter，重点观察 `lambda` 是否不再长期饱和、`success` 是否回升并稳定、`collision` 是否下降到 <0.7。
2) 若 `value_clip_frac` 仍接近 1 或 `grad_norm` 仍高：继续下调 `reward_near_penalty_scale`（如 1.0），或降低 `reward_scale` 以缩小 value 目标幅度。
3) 若 `lambda` 仍长期=100：考虑引入 `cost_limit` 课程（先 110–120，再回落至 90），避免过早饱和。
4) 若碰撞仍占主导：提高 `cost_collision_terminal` 或 `cost_collision_weight`，并观察 `cost_collision_ep` 是否显著下降。



# 2026-01-24 训练日志分析与改动记录（20260124-105256）

## 一、日志结论（是否收敛/异常）
- 未收敛：success 中期上升但后期明显回落，峰值 iter 414 达 0.349，末尾 100 iter 均值≈0.111；最后 200 个滑动窗口均值仅≈0.084–0.114。
- 碰撞主导终止：末尾 100 iter collision≈0.889，timeout=0；障碍碰撞占比高于边界（obstacle≈0.591 > boundary≈0.298）。
- 约束长期饱和：末尾 cost≈125.4 显著高于 cost_limit=90，lambda 在 iter≥200 中约 95% 时间 ≥95，几乎常态顶满。
- 更新稳定性偏弱：approx_kl≈0.036、clip_frac≈0.338；grad_norm 随训练升高（末尾均值≈1.16e3），学习率早降到 min_lr≈1.2e-5。
- 探索/速度收缩但未改善安全：action_std≈0.20、entropy≈-1.27、cmd_speed≈0.083；goal_dist≈4.27、progress≈0.0054，进展停滞。

## 二、原因分析
- 约束过紧导致策略长期受限：episode cost 持续高于 cost_limit=90，拉格朗日乘子饱和后策略主要压成本，成功率无法稳定提升。
- 成本结构对碰撞区分度不足：cost_collision 量级极小（≈0.001/step），cost_near 长期占主导，碰撞终止未形成强惩罚信号。
- 更新与探索衰减：学习率较早降到最小、grad_norm 持续走高，熵与动作方差下降，策略趋于保守但仍高碰撞。

## 三、本次代码变更（放宽约束 + 加强碰撞成本 + 稳定更新 + 日志补充）
1) 放宽 cost 预算，避免拉格朗日乘子长期饱和
   - 文件：`legged_gym_go2/legged_gym/envs/go2/go2_config.py`
   - 修改：`cost_limit` 90.0 → 130.0

2) 强化碰撞成本权重
   - 文件：`legged_gym_go2/legged_gym/envs/go2/go2_config.py`
   - 修改：`cost_collision_weight` 20.0 → 25.0

3) 降低更新次数以稳定训练
   - 文件：`legged_gym_go2/legged_gym/envs/go2/go2_config.py`
   - 修改：`num_learning_epochs` 3 → 2

4) 补充碰撞终止成本的均值日志
   - 文件：`legged_gym_go2/legged_gym/scripts/train_cppo.py`
   - 修改：新增 `cost_collision_terminal` 的 per-step 均值输出

## 四、下一步建议（按顺序）
1) 重新训练 200–400 iter，重点观察 `lambda` 是否脱离长期顶满、`cost` 是否围绕 130 波动、`success` 是否稳定回升。
2) 结合新增日志确认 `cost_collision_terminal` 量级是否有效（不再接近 0）；若仍偏低且 collision 仍高，继续上调 `cost_collision_terminal`（如 100–120）或 `collision_penalty`。
3) 若 `lambda` 仍接近饱和：适度下调 `cost_near_weight`（如 0.2）或再上调 `cost_limit`（如 140）。
4) 若 `approx_kl/clip_frac` 仍偏高：增大 `num_mini_batches`（如 16）或固定较低 lr（如 1e-4）以减小单次更新步长。
