# IsaacGym 四足导航（分层RL）上层策略 CPPO（CMDP + PPO-Lagrangian）实现方案

> 目标：使用以下CPPO（CMDP + PPO-Lagrangian）实现方案训练**上层导航策略**，在布满障碍物与目标点的环境中输出三维速度指令，使四足机器人到达目标点且全程不碰撞障碍物。  
> 分层结构：  
> - **上层策略（需训练）**：感知环境，输出动作 `a_t = [v_x, v_y, ω]`（三维速度执行：x轴速度、y轴速度、角速度），负责导航避障。  
> - **底层策略（已训练、确定性、固定）**：接收上层指令，输出各关节动作，实现跟踪控制。  
> 训练算法：**CMDP + CPPO（Constrained PPO / PPO-Lagrangian）**（**on-policy、model-free**）  
> 约束口径：**cost 为每回合总成本**（episode total cost）。

---

## 1. 将上层任务建模为 CMDP（Constrained MDP）

上层策略 \(\pi_\theta(a|o)\)：

- 观测：`o_t`（**保持现有观测设计不变**）
- 动作：`a_t = [v_x, v_y, ω]`（**保持现有动作设计不变**）
- 奖励：`r_t`（用于效率与到达目标）
- 成本：`c_t`（用于安全约束，包含硬成本与稠密成本）
- 约束形式（episode total cost）：
  \[
  \mathbb{E}_{\pi}\left[C\right] = \mathbb{E}_{\pi}\left[\sum_{t=0}^{T} c_t\right] \le d
  \]
  其中 \(d\) 是每回合总成本预算（训练可从宽松到严格）。

> 说明：本实现不包含环境动力学模型或 imagined rollout，数据完全来自 on-policy 的真实交互轨迹。

---

## 2. 上层与底层交互：宏步长（macro-step）环境封装

由于底层控制器固定且确定性，上层采用**低频决策**：上层每一步动作在底层执行 \(K\) 个物理仿真步。

### 高层 step 流程（语义级描述）

输入：上层动作 \(a_t=[v_x, v_y, \omega]\)。

1. 将 \(a_t\) 映射为期望速度指令并发送给底层控制器。
2. 底层控制器连续运行 \(K\) 个低层步，执行关节控制与物理仿真。
3. 在 \(K\) 个低层步内，累计得到**高层成本**（基于最近风险与碰撞指示，见第 3 节）。
4. 高层奖励不直接累加低层奖励，而由高层状态量（如目标距离变化与指令平滑）在宏步尺度上计算（见第 4 节）。
5. 终止信号由底层环境的终止/截断标记聚合得到，以避免层间去同步。

> 说明：rollout 在**高层时间尺度**上进行，符合“低频导航 + 高频控制”的分层结构；同时保持终止逻辑与底层一致，避免误配。

---

## 3. 成本 Cost 设计（两类：硬成本 + 稠密近障风险）

> 成本在宏步尺度上统计，并作为 CMDP 约束的 episode total cost。成本采用“碰撞指示 + 近障稠密风险”的组合形式。

### 3.1 硬成本：碰撞（collision cost）

- 若发生碰撞：在终止步上记录碰撞指示 \(c^{coll}=1\)，并在成本组合中乘以权重。
  终止由底层环境的终止标记决定，碰撞判定由最近危险距离与碰撞阈值比较得到；当启用安全终止时，碰撞会触发终止。

### 3.2 稠密成本：近障风险（near-obstacle cost，强烈建议）

用最近障碍距离 `d_min` 构造近障成本：

**(A) 安全阈值线性形式（推荐易调参）**
- 设安全距离阈值 `d_safe`：
  \[
  c^{near} = \max\left(0,\ \frac{d_{safe} - d_{min}}{d_{safe}}\right)
  \]
- 当 `d_min >= d_safe`，成本为 0；当越接近障碍，成本越大。


### 3.3 成本组合（每个高层 step 的 cost）

\[
c_t = w_{coll}\,c^{coll} + w_{near}\,c^{near} + c^{coll\_terminal}
\]

其中 \(c^{coll\_terminal}\) 是可选的终止碰撞附加成本项，用于强化“碰撞即高代价”的约束信号。

> 说明：宏步成本由 \(K\) 个低层步的风险信号累积得到，随后参与 episode total cost 统计。

---

## 4. 奖励 Reward 设计（不包含翻倒/卡住惩罚）

> 奖励函数不包含翻倒/卡住惩罚，当前实现围绕“到达目标 + 高效导航 + 平滑控制”构造，并允许奖励裁剪与碰撞惩罚项。

1. **进展奖励（progress）**：鼓励向目标前进  
   令 `dist_t = ||p_t - g||`：
   \[
   r_{prog} = dist_{t-1} - dist_t
   \]
   即距离减少为正奖励。

2. **到达奖励（goal bonus）**：到达阈值内给大正奖励  
   若 `dist_t < r_goal`：
   - `r_goal_bonus = R_goal`（较大正数）  
   终止与否由底层环境的终止/截断标记决定；在启用“到达终止”时，到达会触发终止。

3. **动作平滑（可选）**：惩罚速度指令变化，防抖  
   \[
   r_{smooth} = -\alpha \cdot \|a_t - a_{t-1}\|
   \]

最终每个高层 step 的奖励：
\[
r_t = r_{prog} + r_{goal\_bonus} + r_{smooth} - \beta\,\mathbb{I}[\text{collision}]
\]
并可再乘以全局缩放系数，随后进行对称裁剪以稳定训练。

> 终止与成功：成功通常定义为“到达目标且本步终止且无碰撞”，以避免目标已到但仍未终止的歧义。

---

## 5. CPPO / PPO-Lagrangian：核心训练目标

将约束 CMDP 转为拉格朗日形式：

### 5.1 目标（episode total cost 约束）

\[
\max_\theta\ \mathbb{E}\left[\sum r_t\right]\quad s.t.\quad \mathbb{E}[C] \le d,\ \ C=\sum c_t
\]

拉格朗日松弛：
\[
\max_\theta\ \mathbb{E}\left[\sum (r_t - \lambda c_t)\right] + \lambda d
\]
其中 \(\lambda \ge 0\) 通过对偶上升在线更新。

---

## 6. 需要的网络与估计量

建议使用：
- **Policy**：\(\pi_\theta(a|o)\)（输出速度指令分布）
- **Reward Value Critic**：\(V_r(o)\)
- **Cost Value Critic**：\(V_c(o)\)

（网络完全独立，稳定性优先可用独立。）

---

## 7. GAE 与优势函数（reward 与 cost 各一套）

对 rollout 序列计算：

### 7.1 TD 残差
- Reward TD：
  \[
  \delta^r_t = r_t + \gamma V_r(o_{t+1}) - V_r(o_t)
  \]
- Cost TD：
  \[
  \delta^c_t = c_t + \gamma V_c(o_{t+1}) - V_c(o_t)
  \]

### 7.2 GAE
- \(A^r_t = \text{GAE}(\delta^r)\)
- \(A^c_t = \text{GAE}(\delta^c)\)

> 说明：cost 的折扣因子与 GAE 系数可独立设置（\(\gamma_c,\lambda_c\)），默认与 reward 的 \(\gamma,\lambda\) 相同。

### 7.3 拉格朗日优势（用于 actor 更新）
\[
A^{lag}_t = A^r_t - \lambda A^c_t
\]

---

## 8. PPO 更新：actor/critic 损失

### 8.1 Actor：PPO clipped objective（优势用 \(A^{lag}\)）
\[
L_{actor}(\theta)=\mathbb{E}\left[\min(\rho_t A^{lag}_t,\ \text{clip}(\rho_t,1-\epsilon,1+\epsilon)A^{lag}_t)\right]
\]
其中：
\[
\rho_t=\frac{\pi_\theta(a_t|o_t)}{\pi_{\theta_{old}}(a_t|o_t)}
\]

（可加 entropy bonus 促进探索；本实现对 reward/cost advantage 都进行标准化，并支持 value clipping。）

### 8.2 Critic：两套 value 回归
- Reward critic loss：
  \[
  L_{V_r}=\mathbb{E}\left[(V_r(o_t)-\hat{R}_t)^2\right]
  \]
- Cost critic loss：
  \[
  L_{V_c}=\mathbb{E}\left[(V_c(o_t)-\hat{C}_t)^2\right]
  \]

其中 \(\hat{R}_t\)、\(\hat{C}_t\) 为 reward/cost 的 bootstrap return，遇到 time-out 会对 reward 与 cost 同时进行 bootstrapping 以降低截断偏差。

---

## 9. 拉格朗日乘子 \(\lambda\) 更新（对偶上升，使用 episode total cost）

每个 iteration 收集完 rollouts 后：

1. 统计本批次 episode 总成本：
   \[
   C^{(i)} = \sum_{t=0}^{T_i} c_t^{(i)}
   \]
2. 计算 batch 平均：
   \[
   \bar{C} = \frac{1}{N}\sum_i C^{(i)}
   \]
3. 更新拉格朗日乘子（带投影）：
   \[
   \lambda \leftarrow \Pi_{[0,\lambda_{\max}]}\left(\lambda + \alpha_\lambda(\bar{C}-d)\right)
   \]

解释：
- 若平均总成本 \(\bar{C} > d\)：\(\lambda\) 增大，策略更重视安全规避
- 若 \(\bar{C} < d\)：\(\lambda\) 减小，策略更重视效率与到达

---
