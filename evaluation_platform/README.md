# 智能空管综合指标评估平台

本模块随 `LLMbluesky` 仓库分发。Web页面负责配置、进度、结果和汇报展示；本地Python服务负责运行冻结回放、指令转换、评分和证据归档；BlueSky/PyQt通过异步事件桥上传结构化仿真事件。

仓库不包含历史管制数据、模型权重或正式指标结果。新用户可以先打开无结果的汇报界面，再配置自己的本地数据运行评估。

## 1. 克隆后启动

仅查看平台及配置状态：

```powershell
.\start_evaluation_platform.cmd
```

同时启动平台和BlueSky：

```powershell
.\start_atc_simulation.cmd
```

默认地址：`http://127.0.0.1:8765/`。

先检查仓库中的五项指标实现是否完整：

```powershell
python -m evaluation_platform.doctor --implementation-only
python -m unittest discover -s tests -v
```

启动脚本依次使用：

1. `ATC_PYTHON`环境变量；
2. 仓库内 `.venv\Scripts\python.exe`；
3. 本机Anaconda默认路径；
4. 系统PATH中的 `python`。

## 2. 配置本地数据

复制示例配置：

```powershell
Copy-Item .\evaluation_platform\config\catalog.example.json `
  .\evaluation_platform\config\catalog.json
```

然后设置数据根目录：

```powershell
$env:ATC_DATA_ROOT='D:\your_atc_data'
$env:ATC_MODEL_ROOT='D:\your_atc_models'
```

也可以直接在 `catalog.json` 中填写绝对路径。该文件已被 `.gitignore` 排除，不会把个人数据路径上传到GitHub。

需要配置的文件：

| 字段 | 内容 |
|---|---|
| `references` | 历史参考指令JSONL |
| `controller_instructions` | 待分类的管制指令JSONL，只包含`event_id`与`instruction_text` |
| `intent_references` | 与管制指令同`event_id`的历史结构化意图JSONL |
| `replay` | 冻结态势回放JSONL |
| `trajectory_parquet` | 飞行计划、程序和轨迹状态Parquet |
| `model` | 决策模型joblib |
| `config` | 模型特征及运行配置JSON |
| `system_outputs` | 已有独立系统输出JSONL |
| `intent_predictions` | 意图分类模型结构化输出JSONL |
| `archived_metric` | 可选的已固化指标JSON |

## 3. 安装完整回放依赖

Web服务本身只使用Python标准库。完整回放需要：

```powershell
python -m pip install -r .\evaluation_platform\requirements-full-replay.txt
```

然后运行环境检查：

```powershell
python -m evaluation_platform.doctor
```

全部显示 `PASS` 后再运行完整冻结回放。

## 4. 运行方式

- `archived_result`：读取已固化结果；
- `rescore_existing`：使用已有独立系统输出重新评分；
- `full_replay`：执行冻结回放、指令转换和评分。

开发人员可以在API请求顶层设置 `max_ticks` 进行小规模链路测试。正式结果必须省略该字段并完整运行冻结测试集。

## 5. 结果目录

```text
output/evaluation_runs/<run_id>/
  run_config.json
  run_state.json
  input_manifest.json
  decision_events.jsonl
  system_outputs.jsonl
  metric_result.json
  hashes.json
  *.log
```

页面只展示已完成运行生成的 `metric_result.json`。默认页面不展示历史数值。

## 6. PyQt事件桥

`bluesky_project/bluesky/ui/qtgl/evaluation_bridge.py` 使用后台线程上传事件。服务不可用时不会阻塞Qt主线程，也不会影响原有本地JSONL日志。

接口：

```text
POST /api/simulation/sessions
POST /api/simulation/sessions/<session_id>/events
```

## 7. 其他成员接入新指标

复制：

```text
evaluation_platform/metrics/_template/
```

每个指标提供唯一manifest、输入契约、runner和scorer。未完成注册的指标不会出现在界面中。平台当前注册五项指标：

- 管制指令模仿精度；
- 动态间隔调整成功率；
- 管制指令执行接受度；
- 管制指令自主生成响应时间；
- 管制意图智能理解准确率。

自主生成响应时间读取 H-PPO 回合诊断日志，并使用响应样本数加权汇总。

管制意图智能理解准确率使用三份隔离输入：`controller_instructions.jsonl`提供分类模型输入，
`intent_reference_events.jsonl`保存评分阶段才读取的真实意图，`intent_predictions.jsonl`保存分类结果。
三者使用相同`event_id`逐条关联；航班对象、意图族、意图类型、动作和已记录目标参数全部正确时，
该条指令才计为完整意图命中。当前仓库不包含正式配对数据和分类模型结果。

三项H-PPO指标读取`output/H_PPO/<run>/`中的`events.jsonl`以及诊断CSV。
Git仓库不保存checkpoint、训练输出或正式运行日志。生成新H-PPO日志前，必须通过
`--checkpoint`提供兼容权重，或者先在本机训练；仅重新评分已有日志时不需要加载权重。

界面中的“已实现”表示评分器和任务编排已通过合成测试，不表示当前机器已经配置正式输入，
也不表示指标已经得到正式数值。

## 8. 数据安全与结果边界

- 不提交数据集、模型权重、运行日志或本地 `catalog.json`；
- 服务默认只绑定 `127.0.0.1`；
- 推理阶段不得读取历史参考指令；
- 模仿精度表示冻结历史态势下的指令一致性，不是人工金标准准确率或运行安全证明。
