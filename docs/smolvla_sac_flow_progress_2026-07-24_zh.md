# SmolVLA SAC-Flow 后训练最新进度

更新时间：2026-07-24

分支：`codex/v2`

服务器目录：`/root/autodl-tmp/lerobot/src`

## 1. 当前结论

项目已完成上一轮静态根因审计中两个 P0 问题的代码修复，并在 AutoDL
无卡模式下通过完整 SAC-Flow 专项测试。

当前只能确认训练语义和软件链路已修正，尚不能确认 RL 后训练能够提升
LIBERO 成功率。GPU smoke、critic 重新训练和 actor milestone 实验仍需在
服务器恢复 GPU 后执行。

## 2. 修复前的实验结论

预训练 SmolVLA 在 `libero_object/task 0`、4 episodes 上的基线为 `3/4`
（75%）。

旧实现中：

- 0 次 actor update：75%。
- actor LR `1e-5` 时，2 次更新降至 50%，4 次降至 25%，59 次降至 0%。
- actor LR `3e-6` 时，14 次更新仍为 75%，16 次降至 50%。

这些结果不能再解释为单纯的学习率问题。根因是 actor 在利用一个动作语义
错误、entropy 尺度错误且数据支撑不足的 critic。

## 3. 已修复问题

### 3.1 replay 与 critic 动作语义统一

训练 loop 每次重新规划，只执行 chunk 的第一个环境动作。因此现在：

```text
环境执行动作：a0
replay 保存动作：a0
critic 输入动作：a0
next_obs / reward：由 a0 产生
```

不再把从未执行的 `a1...a49` 展平后交给 critic。以 LIBERO 的 7 维环境
动作为例，critic action dimension 从 `50 x 7 = 350` 降为 `1 x 7 = 7`。

训练入口同时限制 `max_chunk_steps=1`，避免再次创建动作与 transition
不一致的配置。

完整 action chunk 仍由 SmolVLA 生成并用于 rollout；trajectory KL 仍约束
完整 flow 轨迹，防止共享 action expert 随 Q 梯度快速偏离 phase 起点。

### 3.2 去噪路径 log-prob 不再充当环境动作 entropy

SmolVLA 当前计算的是完整 flow 去噪路径 likelihood，包含去噪步、50 个
action positions 和 padded action dimensions。它不是环境动作 `a0` 的
log-prob，不能直接代入标准 SAC entropy 公式。

默认配置现为：

```text
entropy_regularization = false
backup_entropy = false
actor_agg_q = min
```

因此：

- critic target 不再包含 `-alpha * path_log_pi`。
- actor loss 不再包含 `alpha * path_log_pi`。
- alpha 默认不更新。
- `log_pi` 和 `entropy` 只保留为诊断指标。
- actor 默认优化保守的 `-min(Q) + 0.05 * trajectory_KL`。

CLI 仍保留显式开关，只有实现了与实际环境动作同量纲的正确 action density
后，才应重新开启 entropy regularization。

### 3.3 replay 和训练脚本调整

当前脚本关键值：

| 配置 | short-run | baseline-run |
|---|---:|---:|
| `max_chunk_steps` | 1 | 1 |
| `num_envs` | 4 | 4 |
| `replay_capacity` | 2048 | 2048 |
| `min_buffer_size` | 256 | 256 |
| `batch_size` | 8 | 16 |
| `actor_warmup_updates` | 2000 | 2000 |
| `actor_lr` | `3e-6` | `3e-6` |
| `actor_agg_q` | `min` | `min` |
| entropy backup | off | off |

replay 容量已覆盖超过一个 4-env、280-step episode 的 transition 数量，不再
只有约 64 个 vector-env 时间步的窗口。batch size 暂不激进增大，避免在未做
GPU 显存验证前引入新的 OOM 风险。

## 4. 验证结果

AutoDL 无卡模式命令：

```bash
cd /root/autodl-tmp/lerobot/src
source /root/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
python -m pytest tests/test_rlinf_smolvla_sac_flow_*.py -q
```

结果：

```text
140 passed, 12 subtests passed
```

覆盖内容包括：

- critic 只接收实际执行动作。
- replay 只保存已执行 action prefix。
- 默认 actor loss 排除 path entropy。
- 默认 alpha 保持不变。
- 显式 entropy 模式仍保留旧 SAC 数学路径。
- trajectory KL 梯度与 reference freeze。
- checkpoint、resume、并行环境、CLI、WandB 和 shell 脚本。

## 5. Checkpoint 兼容性

旧 checkpoint 的 critic action dimension 是 350，新 critic action dimension
是 7，网络权重形状不同。因此：

```text
不要恢复 2026-07-24 修复前的 critic / target critic / optimizer / replay。
```

下一轮实验必须：

1. 从原始预训练 SmolVLA policy 启动。
2. 新建 critic、target critic、replay 和 optimizer。
3. 重新完成 critic warm-up。

旧 checkpoint 仅可用于历史对照和策略退化分析，不能作为新链路的训练起点。

## 6. GPU 恢复后的执行顺序

### 阶段 A：GPU smoke

```bash
cd /root/autodl-tmp/lerobot/src
bash scripts/rlinf_smolvla_libero/gpu_smoke.sh
```

确认 7 维 critic、one-step replay、完整 trajectory KL 的 tensor shape、显存
和 checkpoint 保存均正常。

### 阶段 B：重新训练 critic

从预训练 policy 开始运行 short-run。先确认：

- replay action shape 为 `[batch, 7]`。
- entropy backup 为关闭状态。
- alpha 保持 `0.01`。
- critic target 与稀疏 reward 同量级。
- critic loss、Q mean 无 NaN/Inf。

short-run 在当前预算下主要用于重新积累 replay 和验证 critic，不用于证明
actor 有效。

### 阶段 C：受控 actor milestone

critic warm-up 完成后，只进行小规模 actor 更新，并保存：

```text
1, 2, 4, 8, 12, 16 actor updates
```

每个 milestone 都用与预训练基线相同的 LIBERO 评估设置测试。至少在 16 次
更新前保持 75%，才说明 P0 修复消除了原先的快速退化；随后才能判断是否有
统计上可信的提升。

### 阶段 D：baseline

只有 GPU smoke、critic 诊断和 milestone 评估全部通过后，才运行
`baseline_run_wandb.sh`。当前无卡阶段不启动 baseline。

## 7. 仍未解决的风险

- 真实 GPU 下 trajectory KL 的显存和吞吐尚未复验。
- critic observation representation 仍冻结，是否足够表达稀疏奖励需要实验。
- actor 更新 action expert 的共享参数，虽然 Q 只看 `a0`，仍可能影响后续
  chunk positions；完整 trajectory KL 是当前保护措施。
- 4 episodes 的评估方差较大，确认提升时需要增加 seeds/episodes。
- 当前 path log-prob 仍不能作为标准 SAC action entropy；重新启用 entropy
  前需要单独设计并验证正确的环境动作密度。

## 8. 当前状态

```text
P0 action-transition 语义修复：完成
P0 entropy/log-prob 隔离：完成
保守 Q 聚合与脚本参数调整：完成
AutoDL 无卡专项测试：完成
GitHub / 服务器代码同步：见本次提交
GPU smoke：待有卡
新 critic warm-up：待有卡
actor milestone 效果验证：待有卡
baseline：暂停
```
