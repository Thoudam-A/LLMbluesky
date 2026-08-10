"""Start the migrated RL+LLM H-PPO runtime inside LLMbluesky-main.

This wrapper deliberately uses the target project's dedicated legacy-BlueSky
configuration. It does not alter the UI project's default settings.cfg.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT / "bluesky_project"
DEFAULT_SETTINGS = PROJECT_ROOT / "config" / "settings_hppo.cfg"
DEFAULT_CANDIDATES = PROJECT_ROOT / "hppo_runtime" / "assets" / "candidates" / "coarse_candidates_flash_v2_rescreened.jsonl"
DEFAULT_CHECKPOINT = REPO_ROOT / "artifacts" / "hppo" / "checkpoints" / "hppo_candidate_selector_v2_curriculum.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run migrated RL+LLM H-PPO in the legacy BlueSky project.")
    parser.add_argument("mode", choices=("train", "eval"))
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--scenarios", default="all", help="Comma-separated 01..09 identifiers, or all.")
    parser.add_argument(
        "--testset-root",
        type=Path,
        help="Directory of generated .npz scenario samples. Overrides --scenarios and runs samples in sorted order.",
    )
    parser.add_argument("--selection", choices=("cycle", "random", "fixed"), default="cycle")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", action="store_true", help="Resume a training checkpoint.")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--allow-existing-output",
        action="store_true",
        help="Append to an existing output directory. Disabled by default to protect evaluation results.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--speed", type=float, default=0.0, help="0 uses BlueSky fast-forward.")
    parser.add_argument("--gui", choices=("none", "qtgl"), default="none")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-episode-time", type=float, default=900.0)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--candidate-selector", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--candidate-dataset", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--candidate-min-safe", type=int, default=3)
    parser.add_argument("--candidate-prior-weight", type=float, default=1.0)
    parser.add_argument("--group-planner-shadow", action="store_true", help="Log conflict-group plan predictions without changing H-PPO actions.")
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not PROJECT_ROOT.is_dir():
        raise FileNotFoundError(f"Target BlueSky project is missing: {PROJECT_ROOT}")
    if not args.settings.is_file():
        raise FileNotFoundError(f"H-PPO settings file is missing: {args.settings}")
    if args.candidate_selector and not args.candidate_dataset.is_file():
        raise FileNotFoundError(f"Candidate dataset is missing: {args.candidate_dataset}")
    if args.testset_root is not None:
        testset_root = args.testset_root.expanduser().resolve()
        if not testset_root.is_dir():
            raise FileNotFoundError(f"Test-set directory is missing: {testset_root}")
        if not any(testset_root.glob("*.npz")):
            raise FileNotFoundError(f"Test-set directory contains no .npz samples: {testset_root}")

    checkpoint = args.checkpoint
    if args.mode == "eval" and checkpoint is None:
        checkpoint = DEFAULT_CHECKPOINT
    if args.mode == "eval" and not checkpoint.is_file():
        raise FileNotFoundError(
            "Evaluation requires a compatible checkpoint. Expected the bundled artifact at "
            f"{checkpoint}; pass --checkpoint after training a replacement."
        )

    output = args.output or (REPO_ROOT / "output" / "H_PPO" / ("training" if args.mode == "train" else "evaluation"))
    output = output.expanduser().resolve()
    output_is_populated = output.is_dir() and any(output.iterdir())
    if output_is_populated and not args.allow_existing_output and not (args.mode == "train" and args.resume):
        raise FileExistsError(
            f"Output directory already contains results: {output}. "
            "Choose a new --output directory, or pass --allow-existing-output to append intentionally."
        )
    env = os.environ.copy()
    env.update({
        "HPPO_MODE": args.mode,
        "HPPO_EPISODES": str(args.episodes),
        "HPPO_SCENARIOS": args.scenarios,
        "HPPO_TESTSET_ROOT": str(args.testset_root.expanduser().resolve()) if args.testset_root else "",
        "HPPO_SCENARIO_SELECTION": args.selection,
        "HPPO_OUTPUT": str(output),
        "HPPO_DEVICE": args.device,
        "HPPO_SPEED": str(args.speed),
        "HPPO_SEED": str(args.seed),
        "HPPO_MAX_EPISODE_TIME": str(args.max_episode_time),
        "HPPO_SAVE_EVERY": str(args.save_every),
        "HPPO_RESUME": "1" if args.resume else "0",
        "HPPO_GUI": args.gui,
        "HPPO_CANDIDATE_SELECTOR": "1" if args.candidate_selector else "0",
        "HPPO_CANDIDATE_DATASET": str(args.candidate_dataset.resolve()) if args.candidate_selector else "",
        "HPPO_CANDIDATE_MIN_SAFE": str(args.candidate_min_safe),
        "HPPO_CANDIDATE_PRIOR_WEIGHT": str(args.candidate_prior_weight),
        "HPPO_LLM_GUIDANCE": "0",
        "HPPO_GROUP_PLANNER_SHADOW": "1" if args.group_planner_shadow else "0",
    })
    if checkpoint is not None:
        env["HPPO_CHECKPOINT"] = str(checkpoint.resolve())

    if args.gui == "qtgl":
        # The UI server starts its own simulation node. Tell the legacy server
        # to forward the H-PPO config to that node.
        env["BLUESKY_SIM_CONFIG_FILE"] = str(args.settings.resolve())
        command = [sys.executable, "BlueSky.py", "--config-file", str(args.settings.resolve())]
    else:
        command = [sys.executable, "BlueSky.py", "--detached", "--config-file", str(args.settings.resolve())]
    print("Launching:", " ".join(command))
    print("Output:", output)
    return subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
