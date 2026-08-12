# 上海进近数据与模仿精度流水线交接

## 1. 交接范围

本仓库保存从历史语音、CAT062雷达航迹和MH4029飞行计划生成训练数据、训练决策模型、执行离线回放并计算模仿精度所需的代码。

大体量数据和模型通过独立的 `ATC_processed_data_handoff_20260812.zip` 交接，不提交Git。建议解压到仓库外部，例如：

```text
<workspace>/
├─ LLMbluesky/                         # 本仓库
└─ ATC_processed_data_handoff_20260812/
   ├─ datasets/
   ├─ models/
   ├─ evaluation/
   └─ intent_reference/
```

原始SEU数据不在交接包内。只有在重新解析语音、CAT062或MH4029时才需要另外申请原始数据。

## 2. 数据性质与结论边界

- 历史指令参考事件由规则解析、角色启发式判断和时间/航班号关联得到，属于弱监督参考数据。
- `HOLD`由一定时间内没有抽取到历史指令推断得到，是弱负样本。
- 当前结果属于历史数据离线回放/构造场景评估，不是真实同场景运行准确率。
- GPT-5.5辅助整理的500条意图参考标注不是独立人工双盲金标准。
- 不得把关联成功率、规则筛选通过率或恒等自检结果写成正式模仿精度。

## 3. 代码目录

| 目录 | 内容 |
|---|---|
| `scripts/` | 数据审计、指令抽取、航迹/计划关联、质量门控、训练、回放和评分 |
| `configs/` | 日期划分、候选动作、时间窗口、质量阈值和模型参数 |
| `schemas/` | 意图识别结构化输出格式 |
| `evaluation_scripts/` | 指标平台当前使用的训练、回放、转换和评分入口 |
| `tests/` | 数据处理、程序感知决策和指标计算测试 |

关键构建顺序如下：

```text
VAD/ASR + rl_teacher_dataset.parquet
  -> build_seu_imitation_dataset.py
  -> build_shanghai_lossless_training_data_v2.py
  -> build_shanghai_training_data_v3.py
  -> build_shanghai_causal_alignment_v4.py
  -> build_shanghai_causal_alignment_v5.py
  -> build_shanghai_quality_gated_v6.py
  -> build_shanghai_label_tiers_v6.py
  -> build_shanghai_compact_v7.py
  -> train_shanghai_hybrid_target_policy.py
```

程序感知候选动作模型是另一条并行链路：

```text
reference_events.jsonl + rl_teacher_dataset.parquet
  -> build_shanghai_program_policy_dataset.py
  -> train_shanghai_program_policy.py
  -> run_shanghai_program_policy_replay.py
  -> score_seu_imitation.py
```

## 4. 环境准备

建议使用Python 3.10或3.11。在仓库根目录运行：

```powershell
python -m pip install -r .\scripts\requirements-shanghai-data-pipeline.txt
```

本地Qwen推理另按`evaluation_platform/requirements-qwen-reranker.txt`安装PyTorch和Transformers。vLLM入口还需要额外安装与本机CUDA匹配的`vllm`；普通数据构建和scikit-learn训练不需要这些GPU依赖。

## 5. 从交接包继续训练

以下示例不会修改交接包，输出写入本地忽略目录。把变量改成实际解压位置：

```powershell
$ATC_HANDOFF = 'D:\shared\ATC_processed_data_handoff_20260812'
New-Item -ItemType Directory -Force .\models, .\experiments | Out-Null

python .\scripts\train_shanghai_hybrid_target_policy.py `
  --compact-data "$ATC_HANDOFF\datasets\seu_compact_model_view_v7\model_samples_v7.parquet" `
  --config .\configs\shanghai_program_aware_policy_v12_expanded.json `
  --output-model .\models\shanghai_hybrid_target_policy_local.joblib `
  --output-report .\experiments\shanghai_hybrid_target_policy_local.json
```

程序感知候选模型可从交接包中的候选动作数据重新训练：

```powershell
python .\evaluation_scripts\train_shanghai_program_policy.py `
  --candidate-data "$ATC_HANDOFF\datasets\seu_program_aware_policy_v12_expanded_20260802_173713\candidate_rows.parquet" `
  --continuous-calibration-data "$ATC_HANDOFF\datasets\seu_program_aware_policy_v12_expanded_20260802_173713\continuous_calibration_candidate_rows.parquet" `
  --config .\configs\shanghai_program_aware_policy_v12_expanded.json `
  --output-model .\models\shanghai_program_policy_local.joblib `
  --output-metrics .\experiments\shanghai_program_policy_local.json
```

以脚本的`--help`输出为准；如模型训练入口的参数发生调整，应同时更新本文档和测试。

## 6. 使用既有输出重新计算模仿精度

```powershell
$ATC_HANDOFF = 'D:\shared\ATC_processed_data_handoff_20260812'

python .\evaluation_scripts\score_seu_imitation.py `
  --references "$ATC_HANDOFF\evaluation\imitation_decision_events.jsonl" `
  --system-outputs "$ATC_HANDOFF\evaluation\imitation_system_outputs.jsonl" `
  --families altitude,speed `
  --system-name shanghai_program_policy `
  --output .\experiments\imitation_metric_recomputed.json
```

评分时必须固定：测试日期、扇区/高度范围、指令族、时间容差、目标值容差和系统版本。不同配置产生的数字不可直接横向比较。

## 7. 从原始数据全量重建

以下命令仅给出主要输入关系。原始数据目录由数据接收方自行设置，不要把绝对路径写入代码或提交Git。

```powershell
$SEU_RAW = 'D:\ATC_data\seu_data\seu_data'
New-Item -ItemType Directory -Force .\datasets | Out-Null

python .\scripts\build_seu_imitation_dataset.py `
  --vad-json "$SEU_RAW\ZSSSAP01_2507_vad.json" `
  --trajectory-parquet "$SEU_RAW\seu_rl_data\rl_teacher_dataset.parquet" `
  --output-dir .\datasets\seu_controller_imitation_v06

python .\scripts\build_shanghai_lossless_training_data_v2.py `
  --trajectory-parquet "$SEU_RAW\seu_rl_data\rl_teacher_dataset.parquet" `
  --references .\datasets\seu_controller_imitation_v06\reference_events.jsonl `
  --config .\configs\shanghai_lossless_training_data_v2_expanded.json `
  --output-dir .\datasets\seu_lossless_training_v2_expanded

python .\scripts\validate_shanghai_lossless_training_data_v2.py `
  --dataset-dir .\datasets\seu_lossless_training_v2_expanded
```

后续v3至v7入口均要求显式传入上一步目录。v5还需要：

- `$SEU_RAW\seu_raw_replay\raw_replay.sqlite`
- `$SEU_RAW\seu_raw_replay\extract_sector_4d.py`作为`--base-decoder`
- `$SEU_RAW\aixm_sector.xml`

先使用`--limit`完成小样本检查，再执行全量重建。每次重建都应保留manifest、质量台账、SHA256和验证报告。

## 8. 测试

不需要原始数据的核心测试：

```powershell
python -m unittest `
  tests.test_seu_imitation_pipeline `
  tests.test_shanghai_program_policy `
  tests.test_shanghai_lossless_training_data_v2 `
  tests.test_shanghai_causal_alignment_v4 `
  tests.test_shanghai_training_data_v3 `
  tests.test_build_shanghai_compact_v7
```

语法检查：

```powershell
python -m compileall -q .\scripts .\evaluation_scripts .\tests
```

## 9. 修改和提交要求

1. 不提交原始语音、航迹、飞行计划、模型、Parquet、JSONL评估明细或数据压缩包。
2. 新增/修改字段时，同时更新Schema、manifest生成逻辑和验证测试。
3. 时间关联必须保持因果约束，禁止使用指令时刻之后的状态或计划更新作为当前输入。
4. 训练集与测试集按日期隔离；当前约定2025-07-01用于开发，2025-07-02用于留出评估。
5. 汇报结果时附数据版本、模型版本、评分参数、支持数和失败样例，不能只给单一百分比。
