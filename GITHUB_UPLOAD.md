# GitHub发布检查清单

当前仓库远程地址：`https://github.com/caijinjun/LLMbluesky.git`。

## 发布前验证

```powershell
python -m unittest discover -s tests -v
python -m evaluation_platform.doctor --implementation-only
git diff --check
git status --short
```

正式运行前再执行`python -m evaluation_platform.doctor`；该检查需要本机已经配置
`evaluation_platform/config/catalog.json`，本地配置不会上传。

## 禁止上传

- 历史管制数据和飞行轨迹；
- 模型权重；
- H-PPO checkpoint、训练输出和运行日志；
- `evaluation_platform/config/catalog.json`；
- `output/`运行产物；
- 日志、缓存和个人环境文件。

## 本功能应包含的文件

```text
evaluation_platform/
evaluation_scripts/
tests/test_evaluation_platform.py
tests/test_hppo_evaluation_metrics.py
tests/test_intent_understanding_metric.py
docs/IMITATION_EVALUATION_PLATFORM.md
docs/MULTI_METRIC_INTEGRATION.md
bluesky_project/bluesky/ui/qtgl/evaluation_bridge.py
bluesky_project/bluesky/ui/qtgl/aiassist.py
bluesky_project/hppo_runtime/
bluesky_project/plugins/case_hppo_bridge.py
bluesky_project/config/settings_hppo.cfg
bluesky_project/routes/hppo/
hppo_tools/
start_evaluation_platform.cmd
start_atc_simulation.cmd
README.md
CONTRIBUTING.md
.gitignore
```

## 安全提交

仓库可能同时存在其他人的修改，不要直接执行 `git add .`。逐项检查并暂存本功能文件：

```powershell
git diff -- bluesky_project/bluesky/ui/qtgl/aiassist.py
git add evaluation_platform evaluation_scripts tests/test_evaluation_platform.py
git add docs/IMITATION_EVALUATION_PLATFORM.md
git add bluesky_project/bluesky/ui/qtgl/evaluation_bridge.py
git add bluesky_project/bluesky/ui/qtgl/aiassist.py
git add start_evaluation_platform.cmd start_atc_simulation.cmd
git add README.md CONTRIBUTING.md .gitignore
git diff --cached --check
git diff --cached --stat
```

确认暂存内容后再提交和推送：

```powershell
git switch -c codex/group-metrics-integration
git commit -m "feat: integrate group evaluation metrics and H-PPO bridge"
git push -u origin codex/group-metrics-integration
```

在GitHub上从`codex/group-metrics-integration`创建Pull Request，不要直接覆盖主分支。
