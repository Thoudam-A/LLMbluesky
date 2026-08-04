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

The bundled checkpoint is used by default. It is compatible with a 60-dimensional
local observation, 26-dimensional global slots and a six-candidate pool.

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
