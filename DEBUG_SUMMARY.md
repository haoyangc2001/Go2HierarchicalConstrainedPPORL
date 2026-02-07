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



# 2026-01-25 训练日志分析与改动记录（20260124-204829）

## 一、日志结论（是否收敛/异常）
- 未收敛：success 中期上升到峰值后回落，峰值 iter 453 达 0.664，末尾 100 iter 均值≈0.219。
- 碰撞主导终止：末尾 100 iter collision≈0.781，timeout=0；障碍碰撞占比高于边界（obstacle≈0.538 > boundary≈0.243）。
- 约束失效：cost 均值多在 70–101，低于 cost_limit=130，lambda 在中后期长期为 0，CPPO 退化为 PPO。
- 数值不稳定并触发冻结：iter 500–873 出现高 approx_kl/clip_frac 与极端 grad_norm；iter 873/874 开始 nan_loss/nan_grad，updates 归零，训练进入冻结态。
- reward 结构偏惩罚：reward 正向项占比持续 <30%，末尾仅 ≈11%，负向项（碰撞/近障）占比 ≈89%。

## 二、原因分析
- 约束上限过高导致拉格朗日乘子长期为 0，成本约束无法抑制碰撞策略，训练退化为 PPO。
- 更新不稳定（高 KL/clip + 梯度爆炸）导致非有限值出现，更新被跳过，模型冻结在次优解，success 回落且无法恢复。
- reward 侧惩罚主导：进展/到达的正向驱动不足，碰撞/近障惩罚占据主要比例，容易形成“减负捷径”的局部最优。

## 三、本次代码变更（奖励结构调整 + 稳定性诊断）
1) 增强正向驱动、降低惩罚尺度
   - 文件：`legged_gym_go2/legged_gym/envs/go2/go2_config.py`
   - 修改：
     - `progress_scale` 12.0 → 20.0
     - `success_reward` 100.0 → 150.0
     - `reward_near_penalty_scale` 1.5 → 0.8
     - `collision_penalty` 180.0 → 150.0

2) 强化碰撞终止成本（cost 侧）
   - 文件：`legged_gym_go2/legged_gym/envs/go2/go2_config.py`
   - 修改：`cost_collision_terminal` 75.0 → 150.0

3) 轻量诊断钩子（定位 NaN 来源）
   - 文件：`rsl_rl/rsl_rl/algorithms/cppo.py`
   - 新增统计：`adv_std`、`adv_finite`、`logp_finite`、`ratio_finite`
   - 文件：`legged_gym_go2/legged_gym/scripts/train_cppo.py`
   - 训练日志输出上述指标

## 四、下一步建议（按顺序）
1) 重新训练 200–300 iter，观察新增诊断字段：
   - `adv_finite/logp_finite/ratio_finite` 是否接近 1.0；`adv_std` 是否稳定非零；
   - 若 `logp_finite/ratio_finite` 降低 → 优先排查观测/动作 NaN 进入网络；
   - 若 `adv_finite` 降低但 logp 正常 → 优先排查 advantage/return 归一化或奖励尺度异常。
2) 监控 reward 正/负占比是否回升到 40–50% 区间；若仍 <30%，继续提升 `progress_scale`（如 24）或 `success_reward`（如 180），并适度下调 `reward_near_penalty_scale`（如 0.6）。
3) 若碰撞率仍 >0.6 且 lambda 仍接近 0：考虑将 `cost_limit` 下调至 100–110，让约束重新介入；同时保留 `cost_collision_terminal=150`。
4) 若 `approx_kl/clip_frac` 仍偏高：固定较低 lr（如 1e-4）或增加 `num_mini_batches` 以降低单次更新幅度。



# 2026-01-26 训练日志分析与改动记录（20260125-104530）

## 一、日志结论（是否收敛/异常）
- 未收敛：best100 success≈0.304（iter 317–416），best50≈0.319（iter 343–392），末尾 100 iter success 均值≈0.0217，滑动均值后期几乎无波动。
- 碰撞主导终止：末尾 100 iter collision≈0.976，timeout=0，障碍碰撞占比高于边界（obstacle≈0.586 > boundary≈0.377）。
- 约束长期饱和：cost≈229.9 显著高于 cost_limit=130，lambda≥99 的迭代比例≈96.2%，中后期几乎长期顶满。
- 更新不稳定但未 NaN：grad_norm 末尾均值≈5.45e3（峰值≈1.89e4），clip_frac≈0.30，value_clip_frac≈0.96；nan_loss/nan_grad=0。
- 探索/行为塌缩：cmd_speed≈0.01、action_std≈0.098、entropy≈-7.33，goal_dist≈4.36、progress≈0.005，进展停滞。

## 二、原因分析
- 约束过紧导致长期饱和：cost_limit=130 小于碰撞终止成本 cost_collision_terminal=150，且 collision≈0.97，使得成本几乎不可能满足，lambda 很快顶满。
- 成本结构对碰撞区分度不足：cost_collision per-step 量级极小（≈0.001），cost_near 累计主导，策略更倾向“尽快结束回合”以降低累计成本。
- 更新与探索衰减：高 grad_norm + clip_frac 偏高导致策略保守（cmd_speed≈0.01），但碰撞率未改善。

## 三、本次代码变更（放宽约束 + 固定学习率）
1) 放宽 cost 预算，缓解 lambda 长期饱和
   - 文件：`legged_gym_go2/legged_gym/envs/go2/go2_config.py`
   - 修改：`cost_limit` 130.0 → 160.0

2) 固定学习率，降低更新不稳定
   - 文件：`legged_gym_go2/legged_gym/envs/go2/go2_config.py`
   - 修改：`learning_rate` 2e-4 → 1e-4
   - 修改：`schedule` adaptive → fixed（保证 lr 固定）

## 四、下一步建议（按顺序）
1) 重新训练 200–400 iter，观察 `lambda` 是否脱离长期顶满、`cost` 是否围绕 160 波动、`success` 是否稳定回升、`collision` 是否下降到 <0.7。
2) 若 `lambda` 仍长期顶满：继续上调 `cost_limit`（如 180–200）或下调 `cost_near_weight`（如 0.2），观察 `cost_near` 与 `collision` 变化。
3) 若 `grad_norm/clip_frac` 仍偏高：增大 `num_mini_batches`（如 16）或进一步降低 `learning_rate`（如 5e-5），验证 200 iter 内 `approx_kl < 0.03`、`grad_norm < 500`。
4) 若碰撞仍占主导：降低前向上限或将 `high_level_action_repeat` 调为 4，观察 `collision` 是否下降、`goal_dist` 是否开始下降。



# 2026-01-27 训练日志分析与改动记录（20260126-102812）

## 一、日志结论（是否收敛/异常）
- 未收敛：200-iter 滑动均值峰值约 `0.190`（iter ~863），后期回落至 `~0.019`（最后 300 iter 均值 `0.019±0.008`）。
- 碰撞主导终止：后期 `collision` 均值 `~0.981`（最后 300 iter），`timeout=0`，几乎所有回合以碰撞终止。
- 约束长期饱和：`cost` 后段均值 ~229，显著高于 `cost_limit=160`；`lambda` 在 iter ~178 达到 100 并长期顶满。
- 更新与探索退化：`approx_kl` 后段均值 ~0.044（峰值 ~0.128），`grad_norm` 后段均值 ~4.1e3（峰值 ~1.5e4）；`action_std` 在 iter ~1114 < 0.05，`entropy` 转负并持续下降，策略塌缩。

## 二、原因分析
- 约束尺度失配：成本长期高于预算，`lambda` 过早饱和且无法推动 `cost` 回落，CPPO 长期处于强约束压力下的无效收敛状态。
- 探索衰减过快：`action_std/entropy` 持续下降导致策略输出趋于低速/低变化，但碰撞率仍高，成功率持续走低。
- 更新不稳：`approx_kl/clip_frac` 偏高 + `grad_norm` 放大，价值更新被频繁裁剪（`value_clip_frac` 高），进一步放大策略漂移与不稳定。

## 三、本次代码变更（放宽预算 + 恢复自适应 KL）
1) 放宽成本预算，避免 `lambda` 长期顶满
   - 文件：`legged_gym_go2/legged_gym/envs/go2/go2_config.py`
   - 修改：`cost_limit` 160.0 → 230.0

2) 恢复自适应学习率
   - 文件：`legged_gym_go2/legged_gym/envs/go2/go2_config.py`
   - 修改：`schedule` fixed → adaptive
   - 修改：`desired_kl` 0.04 → 0.03

## 四、下一步建议（按顺序）
1) 重新训练 200–400 iter，观察 `cost` 是否围绕 230 波动、`lambda` 是否回落到 10–60 区间、`success` 是否不再持续下降。
2) 若 `approx_kl/clip_frac` 仍偏高：进一步降低 `learning_rate`（如 1e-4 → 5e-5）或增加 `num_mini_batches`，验证 200 iter 内 `approx_kl < 0.03`、`grad_norm` 明显下降。
3) 若探索继续塌缩（`action_std < 0.05`）：适度提高 `entropy_coef` 或 `init_noise_std`，保持非零探索。



# 2026-01-28 训练日志分析与改动记录（20260127-143011）

## 一、日志结论（是否收敛/异常）
- 未收敛且后期平台化：`success` 峰值 `0.532`（iter 166），末尾 200 iter 均值 `~0.277`，rolling mean 稳定但来源于更新停摆。
- 碰撞主导终止：末尾 `collision≈0.723`、`timeout=0`、`ep_len_mean≈59.5`，边界违规为 0，主要是障碍碰撞。
- 约束未生效：`cost_limit=230` 明显高于实际 `cost`（后期 ~145），`lambda` 基本全程为 0，CPPO 退化为 PPO。
- 数值不稳导致“停摆”：`nan_grad` 从 iter 734 开始，`nan_loss` 从 iter 742 开始；iter 744 起 `updates=0`，随后 `approx_kl/clip_frac/value_loss/grad_norm/entropy` 变为 0。

## 二、原因分析
- 约束预算偏高：`cost` 长期低于 `cost_limit`，`lambda≈0`，安全约束无法介入，碰撞率难下降。
- 数值问题触发无更新：loss 或梯度出现 NaN/Inf 后被保护逻辑跳过更新，训练继续 rollout 但参数冻结。
- 行为偏快：后期 `cmd_speed≈0.486`、`body_speed≈0.432`，在障碍密集场景中碰撞概率更高。

## 三、本次代码变更（NaN 自动 dump + 终止）
1) 在非有限 loss/grad 时 dump batch 并直接抛异常
   - 文件：`rsl_rl/rsl_rl/algorithms/cppo.py`
   - 新增：`set_debug_dump_dir` / `set_debug_iter` / `set_debug_raise_on_nan`
   - 新增：`nan_dump_iterXXXXX_{loss|grad}.pt` 保存（含关键张量与统计）

2) 训练脚本写入当前 log 目录并启用自动 raise
   - 文件：`legged_gym_go2/legged_gym/scripts/train_cppo.py`
   - 新增：`alg.set_debug_dump_dir(log_dir)`、`alg.set_debug_raise_on_nan(True)`、`alg.set_debug_iter(iter)`

## 四、下一步建议（按顺序）
1) 重新训练 200–400 iter，捕获首个 NaN 触发的 dump 文件，定位是观测、returns、advantages 还是分布参数导致。
2) 下调 `cost_limit`（建议 140–160 区间起步），验证 `lambda` 在前 50–100 iter 进入非零区间、`collision` 是否下降。
3) 若碰撞仍高：适度降低 `action_scale[0]` 或将 `high_level_action_repeat` 调为 4，观察 `cmd_speed` 与 `collision` 变化。



# 2026-01-29 训练日志分析与改动记录（20260128-210846）

## 一、日志结论（是否收敛/异常）
- 平台化未完全收敛：`success` 峰值 `0.532`（iter 166），末尾 100 iter 均值 `0.444`，波动区间 `0.381–0.48`。
- 碰撞仍主导终止：末尾 `collision≈0.562`、`timeout=0`、`ep_len_mean≈65.7`；边界碰撞率上升（early 0.044 → late 0.099），障碍碰撞仍占主。
- 约束未生效：`cost` 均值约 `120`，`cost_limit=230` 仅在 iter 3 超限，`lambda` 最大 `0.015` 后长期为 0，CPPO 退化为 PPO。
- 数值不稳后期恶化：`clip_frac` 从 iter 508 起 >0.5，`approx_kl` 从 iter 603 起 >0.1，iter 733 出现 `grad_norm=8.5e7`、`policy_loss=5.22e5`。

## 二、原因分析
- 约束预算偏高：`cost_limit` 远高于实际 `cost`，`lambda≈0` 导致安全约束无法介入，碰撞率难下降。
- 更新偏离累积：`clip_frac/approx_kl` 走高，`ratio` 极端放大导致 surrogate 爆炸，梯度范数溢出到 `inf`。
- 行为偏快：`cmd_speed` 逐段上升（0.226 → 0.281 → 0.346），在障碍场景下提高碰撞概率并缩短回合。

## 三、本次代码变更（ratio 一致性检查）
1) 新增一次性 ratio/logp_diff dump，用于确认 `ratio == exp(logp_diff)` 是否成立，并定位 logp_diff 极端样本
   - 文件：`rsl_rl/rsl_rl/algorithms/cppo.py`
   - 新增：`self._ratio_check_dumped` 标记
   - 新增：`logp_diff / ratio / ratio_check / ratio_check_diff / ratio_check_rel_diff` 的单次 dump
   - 触发条件：`ratio_max > 1e6` 或 `logp_diff_max > 20` 或非有限值
   - 输出：`nan_dump_iterXXXXX_ratio_check.pt`

## 四、下一步建议（按顺序）
1) 让约束介入：将 `cost_limit` 下调至 `110–120`，观察前 50–100 iter `lambda > 0`，并在 200–400 iter 内验证 `collision` 是否降到 <0.5。
2) 稳定更新：`num_mini_batches` 12 → 16 或 `num_learning_epochs` 2 → 1，目标 200 iter 内 `approx_kl < 0.05`、`clip_frac < 0.4`、`grad_norm` 明显下降。
3) 降低速度：适度下调 `action_scale[0]` 或将 `high_level_action_repeat` 调为 4，验证 `cmd_speed` 回落与碰撞率变化。
4) 触发 ratio_check dump 后比对 `logp_diff` 与 `ratio`，若不一致，检查 `old_actions_log_prob` 存取与 shape/broadcast 对齐。



# 2026-01-30 训练日志分析与改动记录（20260129-103921）

## 一、日志结论（是否收敛/异常）
- 平台化未完全收敛：`success` 峰值 `0.532`（iter 166），末尾 50 iter 均值 `~0.445`；`collision` 末尾均值 `~0.555`，`timeout=0`。
- 约束未生效：`cost` 均值约 `116`，`cost_limit=230` 明显偏高，`lambda` 仅 iter 3 有 `0.015`，其余全程 `0`，CPPO 退化为 PPO。
- 更新稳定性恶化：`clip_frac` 后期 ~0.5，`approx_kl` 在 iter 603/733 出现 >0.1；iter 733 `policy_loss` 和 `grad_norm` 极端增大。
- 数值 dump 定位：`nan_dump_iter00733_ratio_check.pt` 显示 `logp_diff_max≈21.66`、`ratio_max≈2.56e9`；`nan_dump_iter00734_grad.pt` 显示 `ratio_max≈6.87e21`、`surrogate_loss≈3.22e17`、`grad_norm=inf`。

## 二、原因分析
- 约束尺度失配：`cost_limit` 远高于实际 `cost`，导致 `lambda≈0`，安全约束无法介入，碰撞率难下降。
- 极端 logp_diff 的根因（tanh-squash + 动作饱和放大 log_prob 差异）：
  - 高层动作在训练后期接近饱和（dump 中 `actions_batch` max=1.0, min=-0.996），tanh-squash 需要 `atanh(actions)` 计算 raw 动作，`|a|→1` 时 `atanh(a)` 会快速变大。
  - 高斯 log_prob 对 `(u-μ)^2/σ^2` 敏感，`u=atanh(a)` 变大后，即便 `μ` 轻微漂移也会造成 `logp_diff` 快速增大（>20），从而将 `ratio=exp(logp_diff)` 推到 `1e9–1e21`。
  - `ratio_check_diff/rel_diff=0` 且 `finite_ratio=1` 说明公式计算完全一致且无 NaN/Inf，问题在于样本自身的 `logp_diff` 极端。
- 更新偏离累积：`approx_kl/clip_frac` 走高，`ratio` 极端放大导致 surrogate 爆炸，`grad_norm` 溢出，训练后期不稳定。

## 三、本次代码变更（动作分布对齐）
1) 去掉 tanh-squash，改为与 Go2HierarchicalReachAvoidRL 一致的纯高斯分布
   - 文件：`legged_gym_go2/legged_gym/scripts/train_cppo.py`
   - 修改：移除 `ActorCriticCPPO(..., action_squash="tanh")`

## 四、下一步建议（按顺序）
1) 重新训练 200–400 iter，对比 `logp_diff_max/ratio_max/grad_norm/approx_kl` 是否明显下降，验证动作分布改动对数值稳定性的效果。
2) 下调 `cost_limit`（建议 110–130 区间），确保前 50–100 iter `lambda > 0`，并观察 `collision` 是否下降。
3) 若仍出现极端 ratio：增加保护（例如 `logp_diff` clamp 或 `ratio_max` 触发跳过 batch），并输出 top-k `logp_diff` 样本定位源头。
4) 若碰撞仍高：适度降低 `action_scale[0]` 或将 `high_level_action_repeat` 调为 4，验证 `cmd_speed` 回落与碰撞变化。
