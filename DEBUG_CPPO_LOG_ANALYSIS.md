# CPPO 调试记录

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
