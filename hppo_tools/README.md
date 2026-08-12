# Migrated RL+LLM H-PPO

`run_hppo_target.py` runs the RL+LLM H-PPO runtime inside
`bluesky_project` without modifying that project's default `settings.cfg`.
It uses `bluesky_project/config/settings_hppo.cfg`, which enables the flat
legacy plugin `CASE_HPPO_BRIDGE` and sets `simdt = 0.05`.

## Evaluation

```powershell
python .\hppo_tools\run_hppo_target.py eval `
  --scenarios 01,02,03,04 `
  --episodes 1 `
  --device cuda `
  --speed 0 `
  --output .\output\H_PPO\evaluation_candidate_v2
```

Model checkpoints are intentionally not stored in Git. By default the command
looks for:

```text
artifacts/hppo/checkpoints/hppo_candidate_selector_v2_curriculum.pt
```

Place a compatible local checkpoint there or pass `--checkpoint <path>`.
Without one, evaluation exits before BlueSky starts. The checkpoint must match
the 60-dimensional local observation, 26-dimensional global slots and
six-candidate pool used by this runtime.

## Training

```powershell
python .\hppo_tools\run_hppo_target.py train `
  --scenarios all `
  --episodes 100 `
  --selection cycle `
  --device cuda `
  --speed 0 `
  --save-every 10 `
  --output .\output\H_PPO\train_candidate_v2
```

`--speed 0` uses BlueSky's fast-forward mode. Training uses the frozen local
candidate dataset and does not call an LLM API. Disable candidate selection
with `--no-candidate-selector` to run the original five-macro-action H-PPO path.

## QtGL Training View

Use `--gui qtgl` to run the same H-PPO configuration through the target
project's QtGL interface:

```powershell
python .\hppo_tools\run_hppo_target.py train `
  --scenarios all --episodes 100 --selection cycle `
  --device cuda --speed 0 --save-every 10 `
  --gui qtgl `
  --output .\output\H_PPO\train_candidate_v2_gui
```

This opens the target UI and starts an H-PPO simulation node automatically.
The UI displays plugin events; do not start its separate dynamic-sector
auto-traffic solver during this finite-scenario H-PPO run.

## UI Boundary

The QtGL panel can receive `HPPO_EVENT` records from the bridge and display
H-PPO activity. The current migrated model is validated only on the finite
01-09 route scenarios under `bluesky_project/routes/hppo`. It is **not** yet
validated for the UI's dynamic Chengdu-Chongqing sector: that sector has
different traffic generation, route semantics and altitude ranges, so it
requires a dedicated adapter plus retraining before controller commands may
be enabled there.

## Evaluation-platform inputs

The three H-PPO metric plug-ins read immutable files under
`output/H_PPO/<run>/`:

- `events.jsonl` for separation-change and command-acceptance scoring;
- `validation_diagnostics.csv` or `training_diagnostics.csv` for response-time
  and separation-outcome scoring.

These files are local run artifacts and remain excluded from Git.
