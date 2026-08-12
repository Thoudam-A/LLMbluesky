# 指标评估工作交接（2026-08-11）

## 1. 当前可接续状态

当前协作分支为 `codex/group-metrics-integration`。仓库已经包含：

- 综合指标评估平台及五项指标注册、任务编排和评分器；
- 上海进近冻结回放、系统指令转换和模仿精度评分脚本；
- 管制意图评估、预测合并和本周汇总脚本；
- 本周汇报结果、图表、平台截图和可复现汇报视图；
- 默认关闭的Qwen Top-K候选重排实验扩展；
- 单元测试、平台实现检查和外部工件校验脚本。

当前建议作为汇报主口径的结果：

| 指标 | 数值 | 口径 |
|---|---:|---|
| 管制员指令模仿精度 | 48.72% | 匹配历史指令数 / 系统输出指令数 |
| 管制意图动作识别准确率 | 87.14% | 参考意图中的动作槽位命中率 |

辅助结果和适用边界见 `docs/weekly_metrics_20260810/WEEKLY_METRICS_REPORT.md`。

## 2. 克隆后的最短验证路径

```powershell
git switch codex/group-metrics-integration
python -m pip install -r .\evaluation_platform\requirements-full-replay.txt
python -m evaluation_platform.doctor --implementation-only
python -m pytest -p no:cacheprovider tests
.\start_evaluation_platform.cmd
```

汇报页面：

```text
http://127.0.0.1:8765/?metric=controller_imitation&weekly=20260810#presentation
http://127.0.0.1:8765/?metric=controller_intent_understanding_accuracy&weekly=20260810#presentation
```

## 3. 正式数据的团队交接方式

GitHub不包含原始语音、5.35 GiB原始SQLite、296 MB冻结回放、模型权重或逐航班日志。
原因是仓库体积、数据治理和语音隐私，而不是文件缺失。

从批准的团队存储取得外部工件后，按以下结构放置：

```text
ATC_SHARED_ROOT/
  data/
    reference_events_2025-07-02.jsonl
    shadow_replay_test_2025-07-02.jsonl
    rl_teacher_dataset.parquet
    intent/
      controller_instructions.jsonl
      intent_reference_events.jsonl
  models/
    shanghai_program_aware_policy_v12_expanded_20260802_173713.joblib
    shanghai_program_aware_policy_v12_expanded.json
  outputs/
    test_replay_recalibrated_outputs.jsonl
    imitation_metric_result.json
    intent_predictions_merged_500.jsonl
```

核验模仿精度核心工件：

```powershell
$env:ATC_SHARED_ROOT='D:\ATC_metric_handoff'
python .\evaluation_scripts\verify_external_artifacts.py --root $env:ATC_SHARED_ROOT
Copy-Item .\evaluation_platform\config\catalog.team.example.json `
  .\evaluation_platform\config\catalog.json
python -m evaluation_platform.doctor
```

哈希清单位于 `docs/handoff/external_artifact_manifest_20260811.json`。意图识别三份JSONL尚未进入
该哈希清单，接收者需要从团队存储取得同一冻结版本，并在第一次正式重算前补充SHA256。

## 4. 关键代码入口

| 任务 | 入口 |
|---|---|
| 冻结回放决策 | `evaluation_scripts/run_shanghai_program_policy_replay.py` |
| 指令转换 | `evaluation_scripts/convert_decision_log_for_imitation.py` |
| 模仿精度评分 | `evaluation_scripts/score_seu_imitation.py` |
| 意图预测合并 | `evaluation_scripts/merge_intent_predictions.py` |
| 周报结果构建 | `evaluation_scripts/build_weekly_metrics_bundle.py` |
| 平台任务编排 | `evaluation_platform/run_manager.py` |
| Qwen候选约束 | `evaluation_scripts/qwen_candidate_reranker.py` |

## 5. 推荐优化顺序

1. 修复速度候选触发缺失：当前784条系统输出全部为高度指令，速度召回为0%。
2. 对齐许可高度、计划高度、下一航路点、SID/STAR和进离场阶段，减少高度目标偏差。
3. 对意图识别的目标值、单位、频率、跑道和进近类型槽位做专项改进；动作准确率不能替代完整框架准确率。
4. 使用按航班或时间段隔离的测试划分，避免相邻态势泄漏。
5. Qwen只作为候选重排实验，保持默认关闭；只有严格参数召回提升且系统指令精度不下降时才考虑启用。

## 6. 提交新结果必须保留的证据

- 固化数据版本、分割方式和外部工件SHA256；
- 推理阶段参考指令隐藏证明；
- 参考数、系统输出数、匹配数和按指令族分项；
- 参数容差、时间窗口和一对一匹配配置；
- 逐次运行配置、失败记录和结果边界；
- 新旧模型在同一冻结测试集上的并排比较。

禁止只替换界面数值而不保留可复现结果文件。
