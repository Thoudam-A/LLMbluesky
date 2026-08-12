"""Audit raw SEU sources and causal Shanghai instruction-alignment gaps.

This is a read-only audit.  It does not rebuild labels or overwrite a dataset.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import math
import sqlite3
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable


KEY_PLAN_FIELDS = (
    "TITLE", "FILTIM", "IFPLID", "ARCID", "ADEP", "ADES", "ARCTYP",
    "CEQPT", "SEQPT", "CFL", "XFL", "RFL", "SPEED", "SID", "STAR",
    "DRWY", "ARWY", "ROUTE", "SECTOR", "SECDEST", "FPCTST", "ISCOUPLE",
    "EOBD", "EOBT", "ATD", "ETA", "NBARC", "PSSRCODE", "SSRCODE",
    "WKTRC", "VIP", "MISAPP", "PKC", "TASK", "UACID", "TEAM18",
    "COMMAND", "COM", "DAT", "NAV", "OPR", "PER", "RALT", "STS",
    "PBN", "REG", "ARCADDR",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--v4-dir", type=Path, required=True)
    parser.add_argument("--v2-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--cat-latent-audit", type=Path)
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def q(values: list[float]) -> dict[str, float | None]:
    clean = sorted(v for v in values if math.isfinite(v))
    if not clean:
        return {"min": None, "p50": None, "p90": None, "max": None}

    def at(fraction: float) -> float:
        return clean[min(len(clean) - 1, int(round((len(clean) - 1) * fraction)))]

    return {
        "min": round(clean[0], 6),
        "p50": round(at(0.5), 6),
        "p90": round(at(0.9), 6),
        "max": round(clean[-1], 6),
    }


def pct(numerator: int, denominator: int) -> float:
    return round(100.0 * numerator / denominator, 3) if denominator else 0.0


def parse_plan_message(raw: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for index, line in enumerate(raw.replace("\r", "").splitlines()):
        stripped = line.strip().lstrip("\ufeff")
        if index == 0 and stripped.startswith("ZCZC"):
            title = stripped[4:].strip()
            if title:
                fields["TITLE"] = title
        if stripped.startswith("-"):
            name, _, value = stripped[1:].partition(" ")
            fields[name.upper()] = value.strip()
    return fields


def sqlite_summary(path: Path) -> dict[str, Any]:
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    output: dict[str, Any] = {}
    try:
        for table in ("cat062_raw", "mh4029_raw"):
            row = connection.execute(
                f"SELECT COUNT(*), MIN(receive_time_ms), MAX(receive_time_ms), "
                f"COUNT(DISTINCT partition_dt), COUNT(DISTINCT source_file) FROM {table}"
            ).fetchone()
            output[table] = {
                "rows": row[0], "receive_time_ms_min": row[1],
                "receive_time_ms_max": row[2], "partition_days": row[3],
                "source_files": row[4],
            }

        present = collections.Counter()
        nonempty = collections.Counter()
        title = collections.Counter()
        versions = collections.Counter()
        blank_clears = collections.Counter()
        total = 0
        for (outer_json,) in connection.execute("SELECT outer_json FROM mh4029_raw"):
            total += 1
            try:
                outer = json.loads(outer_json)
                raw = outer.get("raw_value")
                if not isinstance(raw, str):
                    continue
                fields = parse_plan_message(raw)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            title[fields.get("TITLE", "MISSING")] += 1
            if fields.get("IFPLID"):
                versions[fields["IFPLID"]] += 1
            for name in KEY_PLAN_FIELDS:
                if name in fields:
                    present[name] += 1
                    if fields[name]:
                        nonempty[name] += 1
                    else:
                        blank_clears[name] += 1
        output["mh4029_fields"] = {
            "parsed_messages": total,
            "title_counts": dict(title.most_common()),
            "unique_ifplid": len(versions),
            "versions_per_ifplid": q([float(v) for v in versions.values()]),
            "fields": {
                name: {
                    "present": present[name], "nonempty": nonempty[name],
                    "blank": blank_clears[name],
                    "nonempty_rate_pct": pct(nonempty[name], total),
                }
                for name in KEY_PLAN_FIELDS
            },
        }
    finally:
        connection.close()
    return output


def parquet_summary(path: Path) -> dict[str, Any]:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    null_counts: dict[str, int] = collections.Counter()
    for group_index in range(parquet.metadata.num_row_groups):
        group = parquet.metadata.row_group(group_index)
        for column_index in range(group.num_columns):
            column = group.column(column_index)
            if column.statistics is not None and column.statistics.null_count is not None:
                null_counts[column.path_in_schema] += column.statistics.null_count
    rows = parquet.metadata.num_rows
    selected = ["trajectory_id", "callsign", "event_time_utc", "inside_sector"]
    trajectories: set[str] = set()
    callsigns: set[str] = set()
    days = collections.Counter()
    inside = 0
    for batch in parquet.iter_batches(columns=selected, batch_size=65536):
        values = batch.to_pydict()
        trajectories.update(str(x) for x in values["trajectory_id"] if x)
        callsigns.update(str(x) for x in values["callsign"] if x)
        days.update(str(x)[:10] for x in values["event_time_utc"] if x)
        inside += sum(bool(x) for x in values["inside_sector"] if x is not None)
    coverage = {
        field.name: {
            "non_null": rows - null_counts.get(field.name, 0),
            "coverage_pct": pct(rows - null_counts.get(field.name, 0), rows),
        }
        for field in parquet.schema_arrow
    }
    return {
        "rows": rows, "row_groups": parquet.metadata.num_row_groups,
        "trajectories": len(trajectories), "callsigns": len(callsigns),
        "days": dict(days), "inside_sector_rows": inside,
        "column_coverage": coverage,
        "schema_metadata": {
            key.decode("utf-8", errors="replace"): value.decode("utf-8", errors="replace")
            for key, value in (parquet.schema_arrow.metadata or {}).items()
        },
    }


def vad_summary(path: Path) -> dict[str, Any]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    classes = collections.Counter(str(row.get("class")) for row in rows)
    days = collections.Counter(str(row.get("utt") or "")[7:15] for row in rows)
    nonempty = sum(bool(str(row.get("transcript") or "").strip()) for row in rows)
    paths_with_offsets = sum("_spk" in str(row.get("path") or "") for row in rows)
    return {
        "segments": len(rows), "nonempty_transcripts": nonempty,
        "class_counts": dict(classes), "day_counts": dict(days),
        "paths_with_segment_offsets": paths_with_offsets,
    }


def aixm_summary(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    sectors = [node for node in root.iter() if node.tag.rsplit("}", 1)[-1] == "sector"]
    volumes = [node for node in root.iter() if node.tag.rsplit("}", 1)[-1] == "volume"]
    polygons = [node for node in root.iter() if node.tag.rsplit("}", 1)[-1] == "polygon"]
    points = [node for node in root.iter() if node.tag.rsplit("}", 1)[-1] == "point"]
    shanghai = [node for node in sectors if str(node.get("code") or "").startswith("ZSSS")]
    return {
        "effective_date": root.get("effective_date"), "version": root.get("version"),
        "sectors": len(sectors), "volumes": len(volumes), "polygons": len(polygons),
        "points": len(points), "shanghai_sectors": len(shanghai),
        "shanghai_sector_codes": [node.get("code") for node in shanghai],
        "contains_only_airspace_geometry": True,
    }


def v2_timing(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    by_reference: dict[str, dict[str, Any]] = {}
    durations: list[float] = []
    pre_before_start: list[float] = []
    unsafe = 0
    for row in read_jsonl(path):
        instruction = row["instruction"]
        reference = instruction["reference_event_id"]
        start = dt.datetime.fromisoformat(instruction["segment_start_local"]).timestamp()
        midpoint = float(instruction["event_time_epoch"])
        prestate = row.get("state_link", {}).get("pre_state_event_epoch")
        duration = float(instruction.get("segment_duration_sec") or 0.0)
        durations.append(duration)
        margin = None if prestate is None else start - float(prestate)
        if margin is not None:
            pre_before_start.append(margin)
            unsafe += int(margin < 0)
        by_reference[reference] = {
            "segment_start_epoch": start, "segment_midpoint_epoch": midpoint,
            "segment_duration_sec": duration, "prestate_epoch": prestate,
            "prestate_margin_before_segment_start_sec": margin,
        }
    return by_reference, {
        "records": len(by_reference), "segment_duration_sec": q(durations),
        "prestate_margin_before_segment_start_sec": q(pre_before_start),
        "prestate_not_before_segment_start": unsafe,
    }


def compare_level(value: Any, reference: Any, tolerance: float = 100.0) -> bool:
    return value is not None and reference is not None and abs(float(value) - float(reference)) <= tolerance


def alignment_summary(path: Path, timing: dict[str, dict[str, Any]]) -> dict[str, Any]:
    records = list(read_jsonl(path))
    actions = collections.Counter()
    families = collections.Counter()
    phases = collections.Counter()
    route_methods = collections.Counter()
    route_confidence = collections.Counter()
    sector_status = collections.Counter()
    effective_level_sources = collections.Counter()
    plan_ages: list[float] = []
    cfl_update_ages: list[float] = []
    altitude_total = 0
    prior_available = 0
    next_vs_prior = collections.Counter()
    action_by_prior_relation = collections.Counter()
    action_relation_by_phase = collections.Counter()
    action_direction_consistent = 0
    equals = collections.Counter()
    near_update_equal_current_label = collections.Counter()
    history_counts: list[int] = []
    traffic_counts: list[int] = []
    procedure_observed = 0
    declared_procedure = 0
    geometry_route = 0
    speed_prior = 0
    unsafe_plan_after_segment_start = 0
    unsafe_cfl_after_segment_start = 0
    inconsistent_examples: list[dict[str, Any]] = []

    for record in records:
        model = record["model_input"]
        plan = model["flight_plan_context"]
        route = model["route_context"]
        procedure = model["procedure_context"]
        vertical = model["vertical_context"]
        speed = model["speed_context"]
        state = model["target_aircraft_state"]
        phases[procedure.get("flight_phase")] += 1
        route_methods[route.get("active_leg_method")] += 1
        route_confidence[route.get("route_context_confidence")] += 1
        sector_status[model["sector_context"].get("sector_alignment_status")] += 1
        effective_level_sources[vertical.get("effective_prior_cleared_level_source")] += 1
        history_counts.append(len(model.get("instruction_history") or []))
        traffic_counts.append(len(model.get("traffic_context") or []))
        declared_procedure += int(bool(procedure.get("declared_procedure_name")))
        procedure_observed += int(bool(procedure.get("procedure_activity_observed")))
        geometry_route += int(bool(route.get("geometry_cross_check_available")))
        speed_prior += int(speed.get("prior_controller_speed_constraint") is not None)
        if plan.get("plan_information_age_at_cutoff_sec") is not None:
            plan_ages.append(float(plan["plan_information_age_at_cutoff_sec"]))

        refs = record.get("member_reference_event_ids") or []
        first_timing = timing.get(refs[0]) if refs else None
        segment_start = None if first_timing is None else first_timing["segment_start_epoch"]
        selected_receive = plan.get("selected_plan_receive_time_utc")
        if segment_start is not None and selected_receive:
            receive_epoch = dt.datetime.fromisoformat(selected_receive.replace("Z", "+00:00")).timestamp()
            unsafe_plan_after_segment_start += int(receive_epoch >= segment_start)
        cfl_update = vertical.get("causal_plan_cleared_level_update_utc")
        cutoff = record["provenance"]["causal_alignment"]["input_cutoff_epoch"]
        cfl_age = None
        if cfl_update:
            cfl_epoch = dt.datetime.fromisoformat(cfl_update.replace("Z", "+00:00")).timestamp()
            cfl_age = float(cutoff) - cfl_epoch
            cfl_update_ages.append(cfl_age)
            if segment_start is not None:
                unsafe_cfl_after_segment_start += int(cfl_epoch >= segment_start)

        for action in record["label"]["actions"]:
            actions[action.get("action")] += 1
            families[action.get("intent_family")] += 1
            if action.get("intent_family") != "altitude":
                continue
            altitude_total += 1
            target = action.get("target_value")
            prior = vertical.get("effective_prior_cleared_level_m")
            observed = state.get("altitude_m")
            if prior is not None:
                prior_available += 1
                delta = float(target) - float(prior) if target is not None else 0.0
                relation = "higher" if delta > 100 else "lower" if delta < -100 else "same"
                next_vs_prior[relation] += 1
                action_by_prior_relation[f"{action.get('action')}->{relation}"] += 1
                action_relation_by_phase[
                    f"{procedure.get('flight_phase')}:{action.get('action')}->{relation}"
                ] += 1
                expected = {
                    "climb": "higher", "descend": "lower",
                    "maintain_altitude": "same",
                }.get(action.get("action"))
                consistent = expected is None or expected == relation
                action_direction_consistent += int(consistent)
                if not consistent and len(inconsistent_examples) < 30:
                    inconsistent_examples.append({
                        "reference_event_id": refs[0] if refs else None,
                        "callsign": record.get("provenance", {}).get("callsign"),
                        "action": action.get("action"), "target_m": target,
                        "effective_prior_m": prior, "relation": relation,
                        "effective_prior_source": vertical.get("effective_prior_cleared_level_source"),
                        "plan_cfl_m": vertical.get("causal_plan_cleared_level", {}).get("value_m"),
                        "cfl_update_age_sec": cfl_age,
                        "observed_altitude_m": observed,
                        "vertical_rate_mps": state.get("vertical_rate_mps"),
                    })
            equals["prior_clearance"] += int(compare_level(target, prior))
            equals["observed_altitude"] += int(compare_level(target, observed))
            equals["next_route_profile"] += int(compare_level(target, route.get("next_route_point_planned_level_m")))
            equals["requested_level"] += int(compare_level(target, plan.get("requested_level_m")))
            equals["plan_cfl"] += int(compare_level(target, vertical.get("causal_plan_cleared_level", {}).get("value_m")))
            equals["plan_xfl"] += int(compare_level(target, vertical.get("causal_plan_exit_level", {}).get("value_m")))
            if compare_level(target, vertical.get("causal_plan_cleared_level", {}).get("value_m")) and cfl_age is not None:
                for threshold in (5, 10, 30, 60):
                    near_update_equal_current_label[str(threshold)] += int(0 <= cfl_age <= threshold)

    return {
        "records": len(records), "actions": sum(actions.values()),
        "action_counts": dict(actions), "family_counts": dict(families),
        "phase_counts": dict(phases), "route_method_counts": dict(route_methods),
        "route_confidence_counts": dict(route_confidence),
        "route_geometry_cross_check": geometry_route,
        "sector_status_counts": dict(sector_status),
        "declared_procedure": declared_procedure,
        "procedure_activity_observed": procedure_observed,
        "effective_prior_level_source_counts": dict(effective_level_sources),
        "prior_speed_constraint": speed_prior,
        "history_items_per_record": q([float(x) for x in history_counts]),
        "traffic_items_per_record": q([float(x) for x in traffic_counts]),
        "plan_information_age_sec": q(plan_ages),
        "cfl_update_age_sec": q(cfl_update_ages),
        "altitude_clearance_impact": {
            "altitude_actions": altitude_total,
            "prior_clearance_available": prior_available,
            "next_target_vs_prior_clearance": dict(next_vs_prior),
            "action_by_prior_relation": dict(action_by_prior_relation),
            "action_relation_by_phase": dict(action_relation_by_phase),
            "action_direction_consistent": action_direction_consistent,
            "target_equal_within_100m": dict(equals),
            "cfl_recent_and_equal_current_label": dict(near_update_equal_current_label),
            "inconsistent_examples": inconsistent_examples,
        },
        "segment_start_causality": {
            "selected_plan_receive_at_or_after_segment_start": unsafe_plan_after_segment_start,
            "cfl_update_at_or_after_segment_start": unsafe_cfl_after_segment_start,
        },
    }


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    sql = result["raw_sources"]["sqlite"]
    teacher = result["raw_sources"]["teacher_parquet"]
    vad = result["raw_sources"]["vad"]
    aixm = result["raw_sources"]["aixm"]
    latent = result["raw_sources"].get("cat062_latent_sample")
    current = result["current_alignment"]
    altitude = current["altitude_clearance_impact"]
    timing = result["current_alignment"]["v2_timing"]
    lines = [
        "# 上海进近航迹—飞行计划—指令关联再审计",
        "",
        f"生成时间：{dt.datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "## 1. 原始数据实况",
        "",
        f"- CAT062 原始回放：{sql['cat062_raw']['rows']:,} 条外层记录；MH4029：{sql['mh4029_raw']['rows']:,} 条计划报文。",
        f"- 教师轨迹 Parquet：{teacher['rows']:,} 个 4D 点，{teacher['trajectories']:,} 条航迹，{teacher['callsigns']:,} 个呼号。",
        f"- VAD：{vad['segments']:,} 个话音片段，{vad['nonempty_transcripts']:,} 条非空转写。",
        f"- AIXM：{aixm['sectors']} 个静态扇区、{aixm['volumes']} 个高度体、{aixm['points']:,} 个边界点；不含 SID/STAR 航图或动态扇区开合。",
        *([] if not latent else [
            f"- CAT062 分层抽样：{latent['sampling']['decoded_records']:,} 条解码记录中，"
            f"{latent['selected_altitude']['count']:,} 条带 selected altitude，抽样覆盖率 "
            f"{latent['selected_altitude']['coverage_pct']}%；该字段目前被教师表丢弃。"
        ]),
        "",
        "## 2. 当前训练集实况",
        "",
        f"- 话轮：{current['records']:,}；动作：{current['actions']:,}。",
        f"- 指令族：{json.dumps(current['family_counts'], ensure_ascii=False)}。",
        f"- 航段方法：{json.dumps(current['route_method_counts'], ensure_ascii=False)}；有几何交叉校验 {current['route_geometry_cross_check']} 条。",
        f"- 声明了 SID/STAR：{current['declared_procedure']} 条；直接观测到程序正在执行：{current['procedure_activity_observed']} 条。",
        f"- 历史许可高度来源：{json.dumps(current['effective_prior_level_source_counts'], ensure_ascii=False)}。",
        "",
        "## 3. 许可高度对下一指令的影响",
        "",
        f"- 高度动作 {altitude['altitude_actions']} 条，其中 {altitude['prior_clearance_available']} 条存在有效既往许可高度。",
        f"- 下一目标相对既往许可：{json.dumps(altitude['next_target_vs_prior_clearance'], ensure_ascii=False)}。",
        f"- 动作方向与许可变化一致：{altitude['action_direction_consistent']}/{altitude['altitude_actions']}。",
        f"- 目标值与各候选信息在 ±100m 内相同：{json.dumps(altitude['target_equal_within_100m'], ensure_ascii=False)}。",
        "",
        "许可高度应作为状态机变量，而不是普通静态计划字段：它决定航空器当前被允许达到的上/下界，结合实际高度和垂直率可判断许可执行阶段；下一条高度指令本质上经常是从当前许可向新的许可迁移。CFL 只能作为一个来源，优先级应低于已确认的历史话音许可和同一时刻的系统选定高度，并保留来源、更新时间和冲突状态。",
        "",
        "## 4. 当前关联的关键风险",
        "",
        f"- VAD 事件采用片段中点；片段时长统计：{json.dumps(timing['segment_duration_sec'], ensure_ascii=False)}。",
        f"- 当前前态势未早于话音片段开始的记录：{timing['prestate_not_before_segment_start']}。",
        f"- 按更严格的片段开始边界检查，计划接收时间落在话音开始之后：{current['segment_start_causality']['selected_plan_receive_at_or_after_segment_start']} 条；CFL 更新时间落在话音开始之后：{current['segment_start_causality']['cfl_update_at_or_after_segment_start']} 条。",
        "- 当前路线大多靠 ETO 括号推断，几何校验覆盖很低；ETO 是预测值，不是过点观测。",
        "- 当前 SID/STAR 只是计划声明，AIXM 文件不含程序航图，无法验证飞机是否沿程序飞行或下一航段约束。",
        "- 历史指令状态机只规范化高度和速度，航向、直飞、偏置、等待、进近许可、跑道和移交状态未完整进入活动许可。",
        "- 教师 Parquet 丢弃了 CAT062 已经能解析的 selected altitude，并未解码/保留 IAS、Mach、TAS、最终选定高度、轨迹意图和 I062/390 CFL/控制席位等字段。",
        "",
        "## 5. 建议的新关联流程",
        "",
        "1. 以 VAD 片段开始时刻而非中点定义决策时刻；保留开始、中点、结束三个时间和音频时钟误差。",
        "2. 航迹使用严格 `state_time < segment_start - guard` 的最后一点，并以地址+航迹号+呼号三键关联，记录每个键的一致性。",
        "3. 飞行计划按 IFPLID 逐字段事件溯源重放；所有计划接收、FILTIM 和字段更新时间必须早于安全截断点。",
        "4. 建立多通道许可状态机：既往管制话音是许可依据，飞行员复诵是确认，I062/390/MH4029 CFL 是系统记录，selected/final selected altitude 是机载执行意图；四者并行保存并形成共识/冲突，不能用简单优先级静默覆盖。",
        "5. 路线进度以几何投影为主、ETO 为辅、ISPASS 只作审计；缺少可靠坐标的航路点不得给高置信度。",
        "6. 引入权威 SID/STAR/进近程序航图，形成程序点序列、航段类型、高度/速度限制及跑道构型；当前 AIXM 不能替代。",
        "7. 建立完整控制状态：高度、速度、航向、直飞、偏置、等待、进近许可、跑道、移交/联系和取消指令。",
        "8. 交通上下文加入排序、同程序前后机、跑道序列、尾流类别、冲突对和扇区移交边界，而不只保留最近距离关系。",
        "9. 每个字段输出 value/source/source_time/age/confidence/conflict/mask，训练输入与审计字段物理分离。",
        "10. 按航班/IFPLID/日期切分，测试集不得共享同一航班计划版本；同时加入无指令时刻作为 HOLD/NOOP 负样本。",
        "",
        "## 6. 推荐新增但当前未充分进入训练输入的信息",
        "",
        "- CAT062：selected altitude、final-state selected altitude、IAS/Mach/TAS、磁航向、双来源垂直率、加速度、转弯率、轨迹意图、气象、Mode-S MB、I062/390 CFL/控制席位。",
        "- MH4029：控制席位/扇区目的地、耦合状态、飞行计划状态、SSR 码、VIP/特情、复飞、任务、数据链和逐字段更新事件。",
        "- 语音：片段起止时刻、说话人簇连续性、角色置信度、ASR 置信度、复诵闭环和同一话轮多动作边界。",
        "- 外部资料：程序航图、扇区动态开合、跑道运行方向、天气/QNH、流控措施、移交协调和机场容量。",
        "",
        "本报告是数据与关联逻辑审计，不是新数据集完成证明，也不产生新的模仿精度。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    sqlite_path = args.raw_root / "seu_raw_replay/raw_replay.sqlite"
    parquet_path = args.raw_root / "seu_rl_data/rl_teacher_dataset.parquet"
    vad_path = args.raw_root / "ZSSSAP01_2507_vad.json"
    aixm_path = args.raw_root / "aixm_sector.xml"
    timing_by_reference, timing_summary = v2_timing(args.v2_dir / "event_state_records.jsonl")
    current = alignment_summary(
        args.v4_dir / "instruction_bundle_training_records_v4.jsonl",
        timing_by_reference,
    )
    current["v2_timing"] = timing_summary
    result = {
        "audit_version": "shanghai_alignment_gap_audit_v1.0",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "raw_sources": {
            "sqlite": sqlite_summary(sqlite_path),
            "teacher_parquet": parquet_summary(parquet_path),
            "vad": vad_summary(vad_path),
            "aixm": aixm_summary(aixm_path),
        },
        "current_alignment": current,
    }
    if args.cat_latent_audit:
        result["raw_sources"]["cat062_latent_sample"] = json.loads(
            args.cat_latent_audit.read_text(encoding="utf-8")
        )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown(args.output_md, result)
    print(json.dumps({
        "status": "PASS", "output_json": str(args.output_json),
        "output_md": str(args.output_md), "records": current["records"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
