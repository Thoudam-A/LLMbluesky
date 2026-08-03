# 智能空管综合评估验证平台设计说明

## 0. 平台化改造结论

原单指标页面升级为“平台外壳 + 指标插件”的结构。平台负责统一导航、数据资产、任务运行、结果固化、证据链、报告导出和汇报展示；每个指标模块只负责自己的输入要求、计算器、结果卡片和诊断视图。

当前只注册和展示“管制指令模仿精度”。其他成员后续可以在不修改主界面骨架的情况下注册自己的指标；未完成接入的指标不在界面中提前展示或命名。

当前版本只保留“指标评估”一级入口。尚未实现独立页面的总览、仿真任务、报告中心和数据资产不显示为可点击导航。冻结回放仍是模仿精度评估内部的一种运行方式，不作为独立功能模块宣传。

指标评估页采用三栏布局：左侧指标库、中部结果与诊断、右侧运行配置和证据链。汇报模式隐藏左右操作栏，只保留指标结论、计算口径和关键证据，适合1600×900投屏。

## 1. 页面定位

该页面是现有“智能空管冲突解脱验证平台”的第二个主工作区，负责上海进近历史态势冻结回放、决策系统独立推理、历史指令匹配、指标计算、误差分析和证据导出。

页面必须与现有实时仿真页分离：实时页验证冲突检测、候选生成、安全验证和指令执行；评估页验证系统输出与历史管制指令的一致性。

## 2. 用户流程

1. 选择运行方式：加载历史结果、重新评分或完整冻结回放；
2. 选择冻结测试集、决策系统和输入对齐版本；
3. 冻结指令范围、时间窗口和参数容差；
4. 校验数据清单、模型与配置SHA256以及测试标签隔离状态；
5. 在历史标签隐藏状态下逐时刻运行决策系统；
6. 标准化系统输出；
7. 解封历史标签，执行按航班、按时间顺序的一对一匹配；
8. 计算总体、分类、逐航空器和严格参数指标；
9. 查看匹配、漏发、误发、参数偏差、时间超窗和安全过滤原因；
10. 导出配置、清单、输出、匹配明细、指标、报告和哈希。

## 3. 指标口径

### 3.1 宽松匹配

当前兼容口径要求：

- 标准化航班号相同；
- 指令族相同；
- 时间绝对差不超过冻结窗口，默认60秒；
- 在每个航班内部采用保持时间顺序的一对一匹配；
- 优先最大化匹配数，再最小化总时间误差。

### 3.2 管制员模仿宏平均召回率

对第 i 架航空器：

`R_i = matched_i / reference_i`

对 N 架具有历史参考指令的航空器：

`R_macro = (1/N) * Σ R_i`

### 3.3 微平均召回率

`R_micro = total_matches / total_references`

### 3.4 系统指令精确率

`P_system = total_matches / total_system_outputs`

### 3.5 输出膨胀比

`I_command = total_system_outputs / total_references`

### 3.6 严格参数宏平均召回率

严格匹配在宽松匹配基础上增加：

- 高度爬升/下降动作方向一致；
- 数值目标与单位可比较；
- 高度误差不超过100米、300英尺或FL3；
- 速度误差不超过10节或5米/秒；
- 航向误差不超过10度。

对每架航空器计算严格匹配数/严格参考数，再进行宏平均。

### 3.7 参数匹配准确率

`A_parameter = parameter_hits / comparable_broad_matches`

该指标只统计数值目标和单位可比较的宽松匹配，不得替代总体召回率。

## 4. 页面状态机

`IDLE → VALIDATING → READY → REPLAYING → NORMALIZING → MATCHING → REPORTING → COMPLETE`

失败状态：`VALIDATION_FAILED`、`REPLAY_FAILED`、`SCORING_FAILED`、`CANCELLED`。

页面必须保存最后一个稳定产物；任务失败时不得显示上一轮指标为本轮结果。

## 5. PyQt组件映射

- 主导航：`QButtonGroup + QPushButton`；
- 页面切换：`QStackedWidget`；
- 配置区：`QComboBox / QCheckBox / QLabel`；
- 过程条：四个状态组件或`QProgressBar`；
- 指标卡：自定义`QFrame`；
- 明细区域：`QTabWidget + QTableView`；
- 大数据表：`QAbstractTableModel`，禁止逐单元格同步填充；
- 后台任务：`QProcess`调用独立评估脚本；
- 进度通信：子进程逐行输出JSON事件；
- 报告导出：运行目录中的JSON、JSONL、Parquet、Markdown和SHA256清单。

## 6. 建议代码模块

```text
bluesky_project/bluesky/ui/qtgl/
  evaluation_bridge.py

evaluation_platform/
  server.py
  run_manager.py
  registry.py
  static/index.html

evaluation_scripts/
  run_shanghai_program_policy_replay.py
  convert_decision_log_for_imitation.py
  score_seu_imitation.py

output/evaluation_runs/<run_id>/
  run_config.json
  input_manifest.json
  system_outputs.jsonl
  replay_summary.json
  metric.json
  matching_details.jsonl
  report.md
  hashes.json
```

## 7. 页面数据接口

评估子进程向GUI输出：

```json
{"event":"progress","stage":"replay","processed":12000,"total":21600}
{"event":"stage_complete","stage":"replay","outputs":784}
{"event":"metric_ready","path":".../metric.json"}
{"event":"complete","run_dir":"..."}
```

GUI只展示已落盘并完成哈希记录的结果，不从控制台文本临时拼接最终指标。

## 8. 验收测试

1. GUI显示数值与命令行评分JSON误差不超过1e-6；
2. 身份映射测试输出100%；
3. 空系统输出时召回为0，精确率为空；
4. 重复系统输出不能重复匹配同一历史指令；
5. 时间窗口边界30秒、60秒测试通过；
6. 高度、速度、航向容差边界测试通过；
7. 推理阶段不读取历史指令标签；
8. 任务取消后保留明确的CANCELLED状态；
9. 结果目录包含模型、配置、输入输出和报告SHA256；
10. 旧输入版本和v4输入版本不得在页面中混报。

## 9. 展示数据说明

原型中的53.68%、48.72%、43.45%和0.981来自现有2025-07-02冻结回放历史结果，仅用于展示页面效果。原型中的逐条匹配表部分为界面示例，不应作为新的实验结果引用。

## 10. 新指标接入规范

每个指标以独立目录注册，不允许把指标专用逻辑写入平台主窗口：

```text
metrics/<metric_id>/
  manifest.json          # 名称、版本、负责人、状态和展示配置
  input_schema.json      # 所需数据字段及校验规则
  runner.py              # 可选：模型推理或仿真任务
  scorer.py              # 指标计算，读取冻结输入并生成标准结果
  result_schema.json     # 指标输出结构
  report_template.md     # 报告文字模板
```

推荐的 `manifest.json`：

```json
{
  "metric_id": "controller_imitation",
  "name": "管制指令模仿精度",
  "group": "decision_consistency",
  "version": "1.0.0",
  "status": "ready",
  "primary_metric": "macro_recall",
  "runner": "runner.py",
  "scorer": "scorer.py",
  "required_inputs": ["frozen_state", "historical_instruction", "flight_plan", "procedure"],
  "views": ["summary", "details", "breakdown", "error_analysis"]
}
```

所有评分器输出统一的外层数据结构：

```json
{
  "metric_id": "controller_imitation",
  "run_id": "SH-APP-0702-V12",
  "status": "complete",
  "primary": {"name": "macro_recall", "value": 0.5368, "unit": "%"},
  "secondary": [],
  "breakdowns": [],
  "evidence_path": "matching_details.jsonl",
  "claim_boundary": "历史态势冻结回放下的一致性评估"
}
```

这样新增指标只需完成清单、输入校验、运行器、评分器和视图配置，平台自动完成指标列表展示、任务状态、运行编号、日志、报告和证据归档。

## 11. 汇报展示规则

- 主指标必须在首屏显示，并明确分母、比较基线和运行编号；
- “历史结果”“本次新运行”“界面示例”使用不同状态文字，禁止混报；
- 首屏最多展示四个结果卡，次要指标放入分类统计；
- 正常模式保留配置和证据链，汇报模式隐藏操作区；
- 每个指标必须显示一句结论和一句适用边界；
- 未完成指标显示“待接入”，不得放置虚构结果；
- 导出的截图、PDF和Markdown报告必须带运行编号、数据版本、模型版本、评分器版本和生成时间。

## 12. 平台级实现拆分

```text
evaluation_platform/
  app_shell.py                 # 一级导航和页面容器
  metric_registry.py           # 扫描并校验指标manifest
  run_manager.py               # 后台任务、取消、恢复和状态机
  artifact_store.py            # 运行目录、哈希和证据索引
  report_center.py             # 汇报/验收报告生成
  widgets/
    metric_catalog.py
    metric_cards.py
    evidence_panel.py
    presentation_mode.py
  metrics/
    controller_imitation/
    conflict_resolution/
    separation_margin/
```

平台与指标模块之间只通过冻结配置、标准进度事件和标准结果JSON通信。某个指标失败时，只标记该运行失败，不影响其他指标页面和已固化结果。

## 13. 当前实现状态（v0.1）

已经落地的模块：

- `evaluation_platform/server.py`：本地HTTP服务、静态页面和仿真事件入口；
- `evaluation_platform/run_manager.py`：后台任务、取消、失败状态和证据归档；
- `evaluation_platform/registry.py`：指标manifest自动发现与字段校验；
- `evaluation_platform/catalog.py`：数据集、模型和外部轨迹路径白名单；
- `evaluation_platform/metrics/controller_imitation/manifest.json`：模仿精度插件；
- `bluesky_project/bluesky/ui/qtgl/evaluation_bridge.py`：不阻塞Qt线程的事件桥；
- `start_evaluation_platform.cmd`：单独启动评估平台；
- `atc-hmi-bluesky-visual-collab/start_atc_simulation.cmd`：同时启动评估服务和BlueSky。

当前支持 `archived_result`、`rescore_existing` 和 `full_replay` 三种模式。Web页面通过真实API加载指标JSON，不再依赖写死的指标卡数值。其他指标目前只有界面卡位，必须实现manifest、运行器和评分器后才能标记为“可运行”。
