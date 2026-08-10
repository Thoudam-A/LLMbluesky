# 小组综合指标评估集成说明

## 1. 当前实现范围

本分支在管制指令模仿精度平台基础上接入五项指标。平台负责指标发现、任务调度、
输入校验、运行状态、结果JSON和SHA256证据归档；各指标目录只实现自己的输入契约和评分器。

| 指标 | 主要输入 | 当前状态 |
|---|---|---|
| 管制指令模仿精度 | 历史指令、冻结态势、决策系统输出 | 已实现，待配置正式输入 |
| 动态间隔调整成功率 | H-PPO事件日志、回合诊断CSV | 已实现，初版事件级口径 |
| 管制指令执行接受度 | H-PPO事件日志 | 已实现，仅表示BlueSky接口接受 |
| 自主生成响应时间 | H-PPO回合诊断CSV | 已实现，按样本数加权 |
| 管制意图理解准确率 | 指令、历史意图、分类预测 | 已实现，按event_id结构化匹配 |

“已实现”只表示代码和合成测试可运行，不代表已经得到正式指标结果。

## 2. 本地检查

```powershell
python -m evaluation_platform.doctor --implementation-only
python -m unittest discover -s tests -v
```

正式运行需从示例创建本地目录清单：

```powershell
Copy-Item evaluation_platform/config/catalog.example.json evaluation_platform/config/catalog.json
python -m evaluation_platform.doctor
```

`catalog.json`、数据、模型、checkpoint和运行输出均不上传Git。

## 3. H-PPO日志生成

H-PPO评估默认查找本地文件：

```text
artifacts/hppo/checkpoints/hppo_candidate_selector_v2_curriculum.pt
```

仓库不包含该权重。使用者需要通过`--checkpoint`提供兼容文件，或者先完成本地训练。
当前H-PPO迁移只覆盖`bluesky_project/routes/hppo/`中的有限场景，不得直接宣称已适配
成都—重庆动态扇区，也不能与原动态扇区自动求解器同时控制航空器。

## 4. 结果边界

- 模仿精度是历史态势冻结回放下的指令一致性，不是真实同场景复现；
- 动态间隔成功表示变更事件所在回合安全结束，尚未证明恢复窗口连续稳定；
- 指令接受度是BlueSky接口接受率，不是管制员、飞行员或机组人因接受度；
- 响应时间从冲突状态进入决策步骤后开始，不包含冲突检测和航空器执行时间；
- 意图理解指标不包含语音识别误差，也不评价指令执行安全性。

## 5. 下一步初步跑通顺序

1. 先运行合成测试，确认五个评分器及任务编排；
2. 配置意图指令、参考标签和模型预测，跑通意图理解指标；
3. 使用已有系统输出重新评分模仿精度，暂不启动完整冻结回放；
4. 准备一份H-PPO`events.jsonl`和诊断CSV，依次跑通三项日志指标；
5. 最后配置兼容checkpoint，生成新的H-PPO运行日志并复算三项指标；
6. 每次运行保留`run_config.json`、`input_manifest.json`、`metric_result.json`和`hashes.json`。
