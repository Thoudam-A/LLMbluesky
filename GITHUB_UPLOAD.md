# GitHub发布检查清单

当前仓库远程地址：`https://github.com/caijinjun/LLMbluesky.git`。

## 发布前验证

```powershell
python -m unittest tests.test_evaluation_platform -v
python -m evaluation_platform.doctor
git diff --check
git status --short
```

`doctor`需要本机已经配置 `evaluation_platform/config/catalog.json`。该本地文件不会上传。

## 禁止上传

- 历史管制数据和飞行轨迹；
- 模型权重；
- `evaluation_platform/config/catalog.json`；
- `output/`运行产物；
- 日志、缓存和个人环境文件。

## 本功能应包含的文件

```text
evaluation_platform/
evaluation_scripts/
tests/test_evaluation_platform.py
docs/IMITATION_EVALUATION_PLATFORM.md
bluesky_project/bluesky/ui/qtgl/evaluation_bridge.py
bluesky_project/bluesky/ui/qtgl/aiassist.py
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
git switch -c codex/controller-imitation-platform
git commit -m "feat: add extensible controller imitation evaluation platform"
git push -u origin codex/controller-imitation-platform
```

在GitHub上从 `codex/controller-imitation-platform` 向 `main` 创建Pull Request，不要直接推送主分支。
