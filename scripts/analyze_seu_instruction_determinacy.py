#!/usr/bin/env python
"""Probe whether state + flight plan + SID/STAR uniquely determine ATC commands.

This is a diagnostic analysis, not a formal model benchmark.  It uses only
historical positive instruction events and therefore cannot measure whether a
system knows when to issue HOLD/no-command.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SHANGHAI_AIRPORTS = {"ZSSS", "ZSPD"}
PLAN_COLUMNS = [
    "trajectory_id",
    "event_time_epoch",
    "flight_plan_found",
    "plan_available_at_point",
    "plan_adep",
    "plan_ades",
    "sid",
    "star",
    "departure_runway",
    "arrival_runway",
    "cleared_flight_level",
    "requested_flight_level",
    "planned_speed",
    "planned_route",
    "planned_route_point_count",
    "plan_rtepts_json",
    "longitude",
    "latitude",
    "altitude_m",
    "ground_speed_mps",
    "track_heading_deg",
    "vertical_rate_mps",
    "inside_sector",
    "time_from_sector_entry_sec",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "MISSING"
    text = str(value).strip()
    return text if text else "MISSING"


def parse_level(value: Any) -> float:
    text = clean_text(value)
    if text == "MISSING":
        return np.nan
    digits = "".join(ch for ch in text if ch.isdigit() or ch == ".")
    if not digits:
        return np.nan
    return float(digits)


def derive_phase(adep: Any, ades: Any) -> str:
    dep = clean_text(adep)
    arr = clean_text(ades)
    if dep in SHANGHAI_AIRPORTS:
        return "departure"
    if arr in SHANGHAI_AIRPORTS:
        return "arrival"
    return "overflight"


def action_label(row: pd.Series) -> str:
    family = str(row["intent_family"]).upper()
    value = row["target_value"]
    if pd.isna(value):
        return f"{family}_MISSING"
    return f"{family}_{int(round(float(value)))}"


def load_pre_instruction_rows(refs: pd.DataFrame, parquet_path: Path) -> pd.DataFrame:
    states = pd.read_parquet(parquet_path, columns=PLAN_COLUMNS)
    states = states[states["trajectory_id"].isin(refs["trajectory_id"].unique())].copy()
    states.sort_values(["trajectory_id", "event_time_epoch"], inplace=True)

    grouped: dict[str, tuple[np.ndarray, pd.DataFrame]] = {}
    for trajectory_id, group in states.groupby("trajectory_id", sort=False):
        grouped[str(trajectory_id)] = (
            group["event_time_epoch"].to_numpy(dtype=float),
            group.reset_index(drop=True),
        )

    linked: list[dict[str, Any]] = []
    for _, ref in refs.iterrows():
        trajectory_id = str(ref["trajectory_id"])
        cutoff = float(ref["event_time_epoch"]) - 4.0
        match = grouped.get(trajectory_id)
        if match is None:
            linked.append({})
            continue
        times, group = match
        index = int(np.searchsorted(times, cutoff, side="right") - 1)
        if index < 0:
            linked.append({})
            continue
        item = group.iloc[index].to_dict()
        item["pre_state_age_sec"] = cutoff - float(item["event_time_epoch"])
        linked.append(item)

    plan_state = pd.DataFrame(linked).add_prefix("pre_")
    return pd.concat([refs.reset_index(drop=True), plan_state], axis=1)


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["phase"] = [
        derive_phase(a, b) for a, b in zip(frame["pre_plan_adep"], frame["pre_plan_ades"])
    ]
    frame["speed_kt"] = frame["pre_ground_speed_mps"] * 1.9438444924406
    frame["cleared_level_num"] = frame["pre_cleared_flight_level"].map(parse_level)
    frame["requested_level_num"] = frame["pre_requested_flight_level"].map(parse_level)
    frame["planned_speed_num"] = frame["pre_planned_speed"].map(parse_level)
    for column in [
        "pre_plan_adep",
        "pre_plan_ades",
        "pre_sid",
        "pre_star",
        "pre_departure_runway",
        "pre_arrival_runway",
    ]:
        frame[column] = frame[column].map(clean_text)
    frame["action_label"] = frame.apply(action_label, axis=1)
    frame["vertical_direction"] = np.select(
        [
            frame["pre_vertical_rate_mps"] > 0.5,
            frame["pre_vertical_rate_mps"] < -0.5,
        ],
        ["climbing", "descending"],
        default="level",
    )
    target_delta = frame["target_value"] - frame["pre_altitude_m"]
    frame["historical_altitude_direction"] = np.select(
        [target_delta > 150.0, target_delta < -150.0],
        ["climb", "descend"],
        default="near_current",
    )
    return frame


def scope(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[
        frame["intent_family"].isin(["altitude", "speed"])
        & frame["pre_inside_sector"].fillna(False)
        & frame["pre_altitude_m"].between(1400.0, 6200.0)
        & frame["target_value"].notna()
    ].copy()


def repeated_group_ambiguity(frame: pd.DataFrame, keys: list[str]) -> dict[str, Any]:
    grouped = frame.groupby(keys, dropna=False)["action_label"].agg(list)
    repeated = grouped[grouped.map(len) >= 2]
    ambiguous = repeated[repeated.map(lambda labels: len(set(labels)) >= 2)]
    repeated_events = int(sum(map(len, repeated)))
    ambiguous_events = int(sum(map(len, ambiguous)))
    return {
        "keys": keys,
        "total_groups": int(len(grouped)),
        "singleton_groups": int(sum(grouped.map(len) == 1)),
        "repeated_groups": int(len(repeated)),
        "ambiguous_repeated_groups": int(len(ambiguous)),
        "repeated_events": repeated_events,
        "ambiguous_events": ambiguous_events,
        "ambiguous_group_rate": (
            float(len(ambiguous) / len(repeated)) if len(repeated) else None
        ),
        "ambiguous_event_rate": (
            float(ambiguous_events / repeated_events) if repeated_events else None
        ),
    }


def build_model(
    numeric_features: list[str], categorical_features: list[str]
) -> Pipeline:
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
        ]
    )
    preprocess = ColumnTransformer(
        [
            ("numeric", numeric, numeric_features),
            ("categorical", categorical, categorical_features),
        ]
    )
    classifier = RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=20260731,
        n_jobs=-1,
    )
    return Pipeline([("preprocess", preprocess), ("classifier", classifier)])


def probe_task(
    train: pd.DataFrame,
    test: pd.DataFrame,
    label: str,
    task_name: str,
    feature_sets: dict[str, tuple[list[str], list[str]]],
) -> list[dict[str, Any]]:
    y_train = train[label].astype(str)
    y_test = test[label].astype(str)
    majority = y_train.mode().iloc[0]
    results = [
        {
            "task": task_name,
            "feature_set": "majority",
            "train_n": int(len(train)),
            "test_n": int(len(test)),
            "accuracy": float((y_test == majority).mean()),
            "balanced_accuracy": float(
                balanced_accuracy_score(y_test, np.repeat(majority, len(y_test)))
            ),
            "majority_label": majority,
        }
    ]
    for name, (numeric, categorical) in feature_sets.items():
        model = build_model(numeric, categorical)
        model.fit(train[numeric + categorical], y_train)
        prediction = model.predict(test[numeric + categorical])
        results.append(
            {
                "task": task_name,
                "feature_set": name,
                "train_n": int(len(train)),
                "test_n": int(len(test)),
                "accuracy": float(accuracy_score(y_test, prediction)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, prediction)),
                "majority_label": None,
            }
        )
    return results


def ambiguity_examples(
    frame: pd.DataFrame, keys: list[str], limit: int = 5
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for values, group in frame.groupby(keys, dropna=False):
        labels = sorted(set(group["action_label"]))
        if len(group) < 2 or len(labels) < 2:
            continue
        if not isinstance(values, tuple):
            values = (values,)
        signature = {
            key: (None if pd.isna(value) else value)
            for key, value in zip(keys, values)
        }
        examples.append(
            {
                "count": int(len(group)),
                "distinct_action_count": int(len(labels)),
                "signature": signature,
                "actions": dict(Counter(group["action_label"]).most_common()),
            }
        )
    examples.sort(
        key=lambda item: (item["count"], item["distinct_action_count"]), reverse=True
    )
    return examples[:limit]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("datasets/seu_controller_imitation_v0/reference_events.jsonl"),
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        required=True,
        help="Path to the supplied rl_teacher_dataset.parquet file.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("reports/seu_instruction_determinacy_analysis.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("reports/seu_instruction_determinacy_analysis.md"),
    )
    args = parser.parse_args()

    raw_refs = read_jsonl(args.reference)
    refs = pd.json_normalize(raw_refs)
    refs["date"] = refs["event_time_local"].str.slice(0, 10)
    refs["trajectory_id"] = refs["target_state.trajectory_id"]
    joined = add_features(load_pre_instruction_rows(refs, args.parquet))
    scoped = scope(joined)
    train = scoped[scoped["date"] == "2025-07-01"].copy()
    test = scoped[scoped["date"] == "2025-07-02"].copy()

    numeric_state = [
        "pre_longitude",
        "pre_latitude",
        "pre_altitude_m",
        "speed_kt",
        "pre_track_heading_deg",
        "pre_vertical_rate_mps",
        "pre_time_from_sector_entry_sec",
    ]
    categorical_state = ["vertical_direction"]
    numeric_plan = [
        "cleared_level_num",
        "requested_level_num",
        "planned_speed_num",
    ]
    categorical_plan = ["phase", "pre_plan_adep", "pre_plan_ades"]
    categorical_procedure = [
        "pre_sid",
        "pre_star",
        "pre_departure_runway",
        "pre_arrival_runway",
    ]
    feature_sets = {
        "state": (numeric_state, categorical_state),
        "state_plus_plan": (
            numeric_state + numeric_plan,
            categorical_state + categorical_plan,
        ),
        "state_plus_plan_plus_procedure": (
            numeric_state + numeric_plan,
            categorical_state + categorical_plan + categorical_procedure,
        ),
    }

    probes: list[dict[str, Any]] = []
    probes += probe_task(train, test, "intent_family", "instruction_family", feature_sets)
    probes += probe_task(train, test, "action_label", "exact_action_all", feature_sets)
    altitude_train = train[train["intent_family"] == "altitude"].copy()
    altitude_test = test[test["intent_family"] == "altitude"].copy()
    speed_train = train[train["intent_family"] == "speed"].copy()
    speed_test = test[test["intent_family"] == "speed"].copy()
    probes += probe_task(
        altitude_train, altitude_test, "action_label", "exact_altitude_target", feature_sets
    )
    probes += probe_task(
        speed_train, speed_test, "action_label", "exact_speed_target", feature_sets
    )

    plan_signature = [
        "phase",
        "pre_sid",
        "pre_star",
        "pre_departure_runway",
        "pre_arrival_runway",
    ]
    coarse = scoped.copy()
    coarse["altitude_bin_600m"] = (
        np.floor(coarse["pre_altitude_m"] / 600.0) * 600.0
    )
    coarse["speed_bin_40kt"] = np.floor(coarse["speed_kt"] / 40.0) * 40.0
    coarse["heading_bin_90deg"] = (
        np.floor(coarse["pre_track_heading_deg"] / 90.0) * 90.0
    )
    coarse["lat_bin_02deg"] = np.floor(coarse["pre_latitude"] / 0.2) * 0.2
    coarse["lon_bin_02deg"] = np.floor(coarse["pre_longitude"] / 0.2) * 0.2
    coarse_state_signature = [
        "altitude_bin_600m",
        "speed_bin_40kt",
        "heading_bin_90deg",
        "vertical_direction",
        "lat_bin_02deg",
        "lon_bin_02deg",
    ]

    coverage = {
        "scoped_total": int(len(scoped)),
        "train_2025_07_01": int(len(train)),
        "test_2025_07_02": int(len(test)),
        "pre_state_age_median_sec": float(scoped["pre_pre_state_age_sec"].median()),
        "plan_found": int(scoped["pre_flight_plan_found"].fillna(False).sum()),
        "plan_available": int(scoped["pre_plan_available_at_point"].fillna(False).sum()),
        "sid_present": int((scoped["pre_sid"] != "MISSING").sum()),
        "star_present": int((scoped["pre_star"] != "MISSING").sum()),
        "route_present": int(
            scoped["pre_planned_route"].map(clean_text).ne("MISSING").sum()
        ),
        "route_points_present": int(
            scoped["pre_plan_rtepts_json"].map(clean_text).ne("MISSING").sum()
        ),
    }
    target_counts = {
        "action_label": dict(Counter(scoped["action_label"]).most_common()),
        "phase": dict(Counter(scoped["phase"]).most_common()),
    }
    ambiguity = {
        "plan_procedure_only": repeated_group_ambiguity(coarse, plan_signature),
        "coarse_state_only": repeated_group_ambiguity(coarse, coarse_state_signature),
        "coarse_state_plus_plan_procedure": repeated_group_ambiguity(
            coarse, coarse_state_signature + plan_signature
        ),
    }
    altitude_direction = pd.crosstab(
        scoped.loc[scoped["intent_family"] == "altitude", "phase"],
        scoped.loc[
            scoped["intent_family"] == "altitude", "historical_altitude_direction"
        ],
    )
    altitude_direction_by_phase = {
        str(index): {str(key): int(value) for key, value in row.items()}
        for index, row in altitude_direction.iterrows()
    }
    test_label_coverage = {}
    for task_name, train_part, test_part in [
        ("exact_action_all", train, test),
        ("exact_altitude_target", altitude_train, altitude_test),
        ("exact_speed_target", speed_train, speed_test),
    ]:
        seen = set(train_part["action_label"])
        test_label_coverage[task_name] = {
            "test_events_with_label_seen_in_train": int(
                test_part["action_label"].isin(seen).sum()
            ),
            "test_events": int(len(test_part)),
            "event_coverage": float(test_part["action_label"].isin(seen).mean()),
            "unseen_test_labels": sorted(set(test_part["action_label"]) - seen),
        }

    result = {
        "analysis_type": "diagnostic_not_formal_benchmark",
        "scope": (
            "altitude/speed positive instructions, pre-instruction state at T-4s, "
            "inside ZSSSAP01, altitude 1400-6200m"
        ),
        "coverage": coverage,
        "target_counts": target_counts,
        "altitude_direction_by_phase": altitude_direction_by_phase,
        "cross_day_label_coverage": test_label_coverage,
        "ambiguity": ambiguity,
        "ambiguity_examples": {
            "coarse_state_plus_plan_procedure": ambiguity_examples(
                coarse, coarse_state_signature + plan_signature
            )
        },
        "cross_day_random_forest_probes": probes,
        "limitations": [
            "Only positive instruction events are included; HOLD/no-command timing is not evaluated.",
            "Reference instructions are rule-extracted candidates, not human-verified gold labels.",
            "The T-4s state reduces immediate post-command leakage but does not reconstruct the controller's full mental state.",
            "Route points contain ptid/planned flight level/ETO/ispass, but the active leg is not yet derived and these fields are not authoritative published procedure constraints.",
            "The probe is a fixed diagnostic model on two days, not a tuned production policy.",
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    probe_df = pd.DataFrame(probes)
    lines = [
        "# 上海进近：状态、飞行计划与进离场程序对指令可判定性的诊断",
        "",
        "## 结论",
        "",
        "这些输入可以显著缩小候选范围并提高历史指令预测能力，但不能唯一决定“此刻是否下指令、下哪一条精确指令”。",
        "本分析只检验已有正指令发生时的类别和目标值，不能检验 HOLD/不下指令。",
        "",
        "## 数据口径",
        "",
        f"- 样本：{coverage['scoped_total']} 条；7月1日 {coverage['train_2025_07_01']} 条，7月2日 {coverage['test_2025_07_02']} 条。",
        "- 指令前态势：使用历史指令时刻前 4 秒的最近轨迹点，避免直接使用指令后响应状态。",
        "- 范围：扇区内、1,400–6,200 米、高度或速度指令。",
        "- 标签仍为规则抽取候选，不是人工金标准。",
        "",
        "## 字段覆盖",
        "",
        "| 字段 | 覆盖数 | 覆盖率 |",
        "|---|---:|---:|",
    ]
    for label, key in [
        ("飞行计划已找到", "plan_found"),
        ("该时刻计划可用", "plan_available"),
        ("SID", "sid_present"),
        ("STAR", "star_present"),
        ("计划航路", "route_present"),
        ("计划航路点", "route_points_present"),
    ]:
        count = coverage[key]
        lines.append(f"| {label} | {count} | {count / len(scoped):.2%} |")
    lines += [
        "",
        "## 跨日期预测探针",
        "",
        "训练固定为7月1日，测试固定为7月2日；模型只是用于判断信息量的随机森林探针。",
        "",
        "| 任务 | 输入 | 测试数 | 准确率 | 平衡准确率 |",
        "|---|---|---:|---:|---:|",
    ]
    for _, row in probe_df.iterrows():
        lines.append(
            f"| {row['task']} | {row['feature_set']} | {int(row['test_n'])} | "
            f"{row['accuracy']:.2%} | {row['balanced_accuracy']:.2%} |"
        )
    lines += [
        "",
        "精确动作标签在训练日的测试事件覆盖率：",
        "",
        "| 任务 | 测试事件 | 标签在训练日出现 | 覆盖率 |",
        "|---|---:|---:|---:|",
    ]
    for task_name, item in test_label_coverage.items():
        lines.append(
            f"| {task_name} | {item['test_events']} | "
            f"{item['test_events_with_label_seen_in_train']} | "
            f"{item['event_coverage']:.2%} |"
        )
    lines += [
        "",
        "## 条件歧义",
        "",
        "| 条件签名 | 可重复组 | 其中含多个不同指令的组 | 歧义组占比 | 单例组 |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, key in [
        ("飞行阶段+SID/STAR+跑道", "plan_procedure_only"),
        ("粗粒度飞机状态", "coarse_state_only"),
        ("粗粒度状态+计划程序", "coarse_state_plus_plan_procedure"),
    ]:
        item = ambiguity[key]
        rate = item["ambiguous_group_rate"]
        rate_text = "N/A" if rate is None else f"{rate:.2%}"
        lines.append(
        f"| {label} | {item['repeated_groups']} | "
            f"{item['ambiguous_repeated_groups']} | {rate_text} | "
            f"{item['singleton_groups']} |"
        )
    lines += [
        "",
        "## 历史高度方向与阶段",
        "",
        "| 阶段 | 上升目标 | 下降目标 | 接近当前高度 |",
        "|---|---:|---:|---:|",
    ]
    for phase, values in altitude_direction_by_phase.items():
        lines.append(
            f"| {phase} | {values.get('climb', 0)} | "
            f"{values.get('descend', 0)} | {values.get('near_current', 0)} |"
        )
    lines += [
        "",
        "## 解释边界",
        "",
        "1. 能较可靠推断的是进/离场阶段、升降方向以及常用候选高度/速度。",
        "2. 不能由这些字段唯一恢复的是指令触发时机、冲突/排序目标、是否暂缓、精确目标值和多机联合决策。",
        "3. 航路点中已有 ptid、计划高度、ETO、ispass，可用于推导活动航段；但 SID/STAR 名称和计划剖面不等同于权威程序限制，仍需补充下一航路点及该航段高度/速度限制。",
        "4. 还需交通排序、相邻机状态、跑道运行构型、天气、流控、移交和近期指令记忆，才能逼近真实决策条件。",
    ]
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
