# 上海进近管制指令数据交接说明

## 1. 数据包范围

本数据包为项目在原始上海进近语音、CAT062雷达航迹、MH4029飞行计划和AIXM空域数据基础上形成的加工数据、模型及离线评估结果。

本包不包含约6.31 GB的原始数据。需要重新执行语音解析、雷达报文解码或完整飞行计划关联时，应另行获取：

`D:\ATC_data\seu_data\seu_data`

## 2. 目录说明

### datasets/seu_controller_imitation_v06

历史语音指令的规则解析、航班号识别、角色推断、航迹关联和参考事件数据。

- `instruction_events.jsonl`：从转写文本抽取的原子指令事件。
- `linked_main_events_all_roles.jsonl`：所有角色候选的指令—航迹关联结果。
- `reference_events.jsonl`：去重和质量筛选后的历史参考事件。
- `reference_events_2025-07-01.jsonl`：训练日参考事件。
- `reference_events_2025-07-02.jsonl`：测试日参考事件。
- `reference_sequences.jsonl`：按航空器组织的历史指令序列。
- `state_snapshots.jsonl`：指令附近的航空器状态快照。

上述参考事件属于规则解析和启发式关联形成的弱监督参考数据，不是管制员人工确认的正式金标准。

### datasets/seu_training_tiers_v6_1

完成因果时间对齐和质量门控后的训练分层数据。

- `train_core_v6.jsonl`：主训练样本。
- `train_weak_v6.jsonl`：弱监督辅助样本。
- `audit_only_v6.jsonl`：仅用于审计、不建议训练的样本。
- `label_tier_ledger_v6.parquet`：样本质量分层台账。

### datasets/seu_compact_model_view_v7

从完整关联数据中整理出的紧凑模型输入视图，包括主机状态、飞行计划/航段特征、交通关系和目标动作。

该版本以正指令样本为主，缺少由管制员确认的完整NOOP/HOLD负样本，不应直接解释为完整指令触发金标准。

### datasets/seu_program_aware_policy_v12_expanded_20260802_173713

上海进近程序感知决策模型使用的状态样本、候选动作、弱HOLD样本、训练/测试划分和校准数据。

- 2025-07-01主要作为训练日期。
- 2025-07-02主要作为离线测试日期。
- HOLD为基于一定时间内未抽取到指令构造的弱负样本。

### models

包含当前保留的上海进近程序感知模型、重校准模型和混合目标模型。推荐优先使用：

- `shanghai_program_aware_policy_v12_expanded_recalibrated_20260802_173713.joblib`
- `shanghai_hybrid_target_policy_v1_1_20260810.joblib`

### evaluation

2026-08-10离线指标评估的输入、系统输出、匹配明细和意图识别结果。

- `imitation_decision_events.jsonl`：模仿精度参考事件。
- `imitation_system_outputs.jsonl`：决策系统生成的指令。
- `imitation_metric_result.json`：模仿精度匹配明细和汇总。
- `intent_predictions_merged_500.jsonl`：500条意图识别预测。
- `intent_metric_result.json`：意图识别评估结果。

这些结果是历史数据离线回放/构造场景评估结果，不代表真实运行环境下的正式业务准确率。

### intent_reference

- `remote_gold_500_validated.jsonl`：500条历史ATC话语的结构化参考标注。

该文件记录的预标注器为GPT-5.5，属于大模型辅助形成的参考标注，不是独立人工双盲金标准。

## 3. 配套代码

训练、回放、指标计算和评估平台代码位于：

<https://github.com/caijinjun/LLMbluesky.git>

主要代码目录：

- `evaluation_scripts/`
- `evaluation_platform/`
- `tests/`

## 4. 使用注意事项

1. 保持本数据包目录结构不变，避免脚本路径失效。
2. 使用前核对 `SHA256SUMS.csv`，确认文件在传输过程中未损坏。
3. 数据含历史航空通信文本、航班号和运行状态，公开传播前需完成数据授权及脱敏审查。
4. 训练和评估时必须记录使用的数据版本、日期划分、模型文件和指标配置。
5. 不得将弱标签、规则匹配结果或离线回放指标表述为管制员人工确认的正式准确率。

## 5. 版本信息

- 交接包日期：2026-08-12
- 加工数据根目录：`C:\Users\Administrator\Documents\research`
- 配套代码仓库：`caijinjun/LLMbluesky`

