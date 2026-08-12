"""Edition-1.20 CAT062 enrichment for the local SEU raw replay source.

The local extractor already provides a safe CAT062 record framing parser, but
it discards several I062/380 and I062/390 subfields.  This module loads that
parser and replaces only the relevant subfield handlers.  It deliberately
keeps complex Mode-S BDS and ACAS payloads losslessly as hexadecimal rather
than inventing a BDS interpretation.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any


FT_TO_M = 0.3048


def _ascii(raw: bytes) -> str:
    return raw.decode("ascii", errors="replace").strip()


def _install_shapely_placeholders() -> None:
    try:
        import shapely.geometry  # type: ignore  # noqa: F401
    except ImportError:
        shapely = types.ModuleType("shapely")
        geometry = types.ModuleType("shapely.geometry")
        ops = types.ModuleType("shapely.ops")
        prepared = types.ModuleType("shapely.prepared")
        for name in ("LineString", "MultiPoint", "Point", "Polygon"):
            setattr(geometry, name, type(name, (), {}))
        ops.unary_union = lambda value: value
        prepared.prep = lambda value: value
        shapely.geometry, shapely.ops, shapely.prepared = geometry, ops, prepared
        sys.modules.update({
            "shapely": shapely, "shapely.geometry": geometry,
            "shapely.ops": ops, "shapely.prepared": prepared,
        })


def load_enriched_decoder(path: Path) -> Any:
    """Import the supplied local decoder and install Edition-1.20 handlers."""

    _install_shapely_placeholders()
    spec = importlib.util.spec_from_file_location("seu_cat062_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load CAT062 decoder: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    install_enriched_handlers(module)
    return module


def install_enriched_handlers(decoder: Any) -> None:
    """Patch the original decoder in place; no raw source file is modified."""

    def i380_magnetic_heading(reader: Any, out: dict[str, Any]) -> None:
        out["magnetic_heading_deg"] = reader.uint(2, "I062/380 MHG") * 360.0 / 65536.0

    def i380_ias_mach(reader: Any, out: dict[str, Any]) -> None:
        raw = reader.uint(2, "I062/380 IAS/Mach")
        if raw & 0x8000:
            out["legacy_mach"] = (raw & 0x7FFF) * 0.001
        else:
            out["legacy_indicated_airspeed_kt"] = (raw & 0x7FFF) * 3600.0 / 16384.0

    def i380_true_airspeed(reader: Any, out: dict[str, Any]) -> None:
        raw = reader.uint(2, "I062/380 TAS")
        out["true_airspeed_kt"] = raw
        out["true_airspeed_range_valid"] = raw <= 2046

    def i380_selected_altitude(reader: Any, out: dict[str, Any]) -> None:
        raw = reader.uint(2, "I062/380 SAL")
        source_code = (raw >> 13) & 0x03
        out.update({
            "selected_altitude_source_available": bool(raw & 0x8000),
            "selected_altitude_source": {
                0: "unknown", 1: "aircraft_altitude", 2: "fcu_mcp_selected", 3: "fms_selected",
            }[source_code],
            "selected_altitude_ft": decoder.sign_extend(raw & 0x1FFF, 13) * 25.0,
        })

    def i380_final_selected_altitude(reader: Any, out: dict[str, Any]) -> None:
        raw = reader.uint(2, "I062/380 FSS")
        out.update({
            "final_selected_altitude_ft": decoder.sign_extend(raw & 0x1FFF, 13) * 25.0,
            "managed_vertical_mode_active": bool(raw & 0x8000),
            "altitude_hold_active": bool(raw & 0x4000),
            "approach_mode_active": bool(raw & 0x2000),
        })

    def i380_trajectory_intent_status(reader: Any, out: dict[str, Any]) -> None:
        raw = reader.u8("I062/380 TIS")
        out.update({
            "trajectory_intent_available": not bool(raw & 0x80),
            "trajectory_intent_valid": not bool(raw & 0x40),
            "trajectory_intent_status_raw": raw,
        })

    def i380_trajectory_intent(reader: Any, out: dict[str, Any]) -> None:
        rep = reader.u8("I062/380 TID.REP")
        points: list[dict[str, Any]] = []
        for index in range(rep):
            raw = reader.read(15, f"I062/380 TID[{index}]")
            header = raw[0]
            flags = raw[9]
            points.append({
                "tcp_number": header & 0x3F,
                "tcp_number_available": not bool(header & 0x80),
                "tcp_compliance": not bool(header & 0x40),
                "altitude_ft": decoder.sign_extend(int.from_bytes(raw[1:3], "big"), 16) * 10.0,
                "latitude": decoder.sign_extend(int.from_bytes(raw[3:6], "big"), 24) * 180.0 / (2 ** 23),
                "longitude": decoder.sign_extend(int.from_bytes(raw[6:9], "big"), 24) * 180.0 / (2 ** 23),
                "point_type_code": (flags >> 4) & 0x0F,
                "turn_direction_code": (flags >> 2) & 0x03,
                "turn_radius_available": bool(flags & 0x02),
                "time_over_point_available": not bool(flags & 0x01),
                "time_over_point_sec": int.from_bytes(raw[10:13], "big"),
                "turn_radius_nm": int.from_bytes(raw[13:15], "big") * 0.01,
            })
        out["trajectory_intent_points"] = points

    def i380_com_acas(reader: Any, out: dict[str, Any]) -> None:
        raw = reader.uint(2, "I062/380 COM")
        out.update({
            "mode_s_communications_capability": (raw >> 13) & 0x07,
            "mode_s_flight_status": (raw >> 10) & 0x07,
            "mode_s_specific_service_capable": bool(raw & 0x80),
            "mode_s_25ft_altitude_capable": bool(raw & 0x40),
            "mode_s_aircraft_id_capable": bool(raw & 0x20),
        })

    def i380_adsb_status(reader: Any, out: dict[str, Any]) -> None:
        raw = reader.uint(2, "I062/380 SAB")
        out.update({
            "adsb_acas_status_code": (raw >> 14) & 0x03,
            "adsb_navigation_mode_status_code": (raw >> 12) & 0x03,
            "adsb_differential_correction_status_code": (raw >> 10) & 0x03,
            "adsb_ground_bit_set": bool(raw & 0x0200),
            "adsb_emergency_status_code": raw & 0x07,
        })

    def i380_acas_ra(reader: Any, out: dict[str, Any]) -> None:
        out["acas_resolution_advisory_bds30_hex"] = reader.read(7, "I062/380 ACS").hex().upper()

    def i380_baro_vertical_rate(reader: Any, out: dict[str, Any]) -> None:
        out["barometric_vertical_rate_fpm"] = reader.sint(2, "I062/380 BVR") * 6.25

    def i380_geo_vertical_rate(reader: Any, out: dict[str, Any]) -> None:
        out["geometric_vertical_rate_fpm"] = reader.sint(2, "I062/380 GVR") * 6.25

    def i380_roll(reader: Any, out: dict[str, Any]) -> None:
        out["roll_angle_deg"] = reader.sint(2, "I062/380 RAN") * 0.01

    def i380_track_turn(reader: Any, out: dict[str, Any]) -> None:
        raw = reader.uint(2, "I062/380 TAR")
        out.update({
            "turn_indicator_code": (raw >> 14) & 0x03,
            "track_angle_rate_degps": decoder.sign_extend((raw >> 1) & 0x7F, 7) * 0.25,
        })

    def i380_velocity_uncertainty(reader: Any, out: dict[str, Any]) -> None:
        out["velocity_uncertainty_category"] = reader.u8("I062/380 VUN")

    def i380_met(reader: Any, out: dict[str, Any]) -> None:
        flags = reader.u8("I062/380 MET flags")
        out["met_data_flags"] = flags
        if flags & 0x80:
            out["met_wind_speed_kt"] = reader.uint(2, "I062/380 MET wind speed")
        if flags & 0x40:
            out["met_wind_direction_deg"] = reader.uint(2, "I062/380 MET wind direction")
        if flags & 0x20:
            out["met_temperature_c"] = reader.sint(2, "I062/380 MET temperature") * 0.25
        if flags & 0x10:
            out["met_turbulence"] = reader.u8("I062/380 MET turbulence")

    def i380_emitter(reader: Any, out: dict[str, Any]) -> None:
        out["emitter_category_code"] = reader.u8("I062/380 EMC")

    def i380_position_uncertainty(reader: Any, out: dict[str, Any]) -> None:
        out["position_uncertainty_code"] = reader.u8("I062/380 PUN") & 0x0F

    def i380_bds(reader: Any, out: dict[str, Any]) -> None:
        rep = reader.u8("I062/380 MB.REP")
        values = []
        for index in range(rep):
            raw = reader.read(8, f"I062/380 MB[{index}]")
            values.append({
                "bds_data_hex": raw[:7].hex().upper(),
                "bds1": (raw[7] >> 4) & 0x0F, "bds2": raw[7] & 0x0F,
            })
        out["mode_s_bds_reports"] = values

    def i380_ias(reader: Any, out: dict[str, Any]) -> None:
        out["indicated_airspeed_kt"] = reader.uint(2, "I062/380 IAR")

    def i380_mach(reader: Any, out: dict[str, Any]) -> None:
        out["mach_number"] = reader.uint(2, "I062/380 MAC") * 0.008

    def i380_bps(reader: Any, out: dict[str, Any]) -> None:
        out["barometric_pressure_setting_hpa"] = 800.0 + (reader.uint(2, "I062/380 BPS") & 0x0FFF) * 0.1

    i380 = decoder.I380_HANDLERS
    i380[2:28] = [
        i380_magnetic_heading, i380_ias_mach, i380_true_airspeed,
        i380_selected_altitude, i380_final_selected_altitude,
        i380_trajectory_intent_status, i380_trajectory_intent, i380_com_acas,
        i380_adsb_status, i380_acas_ra, i380_baro_vertical_rate,
        i380_geo_vertical_rate, i380_roll, i380_track_turn,
        decoder.parse_i380_sf17, decoder.parse_i380_sf18,
        i380_velocity_uncertainty, i380_met, i380_emitter,
        decoder.parse_i380_sf22, decoder.parse_i380_sf23,
        i380_position_uncertainty, i380_bds, i380_ias, i380_mach, i380_bps,
    ]

    def i390_tag(reader: Any, out: dict[str, Any]) -> None:
        out["fpps_sac"], out["fpps_sic"] = reader.u8("I062/390 TAG SAC"), reader.u8("I062/390 TAG SIC")

    def i390_ifps(reader: Any, out: dict[str, Any]) -> None:
        raw = reader.uint(4, "I062/390 IFI")
        out["i390_ifps_flight_id_type"] = (raw >> 30) & 0x03
        out["i390_ifps_flight_id_number"] = raw & 0x07FFFFFF

    def i390_category(reader: Any, out: dict[str, Any]) -> None:
        raw = reader.u8("I062/390 FCT")
        out.update({
            "i390_gat_oat_code": (raw >> 6) & 0x03,
            "i390_flight_rules_code": (raw >> 4) & 0x03,
            "i390_rvsm_code": (raw >> 2) & 0x03,
            "i390_high_priority": bool(raw & 0x02),
        })

    def i390_aircraft_type(reader: Any, out: dict[str, Any]) -> None:
        out["i390_aircraft_type"] = _ascii(reader.read(4, "I062/390 TAC"))

    def i390_wtc(reader: Any, out: dict[str, Any]) -> None:
        out["i390_wake_turbulence_category"] = _ascii(reader.read(1, "I062/390 WTC"))

    def i390_airport(name: str):
        return lambda reader, out: out.__setitem__(name, _ascii(reader.read(4, f"I062/390 {name}")))

    def i390_runway(reader: Any, out: dict[str, Any]) -> None:
        out["i390_runway_designation"] = _ascii(reader.read(3, "I062/390 RDS"))

    def i390_cfl(reader: Any, out: dict[str, Any]) -> None:
        out["i390_current_cleared_flight_level"] = reader.uint(2, "I062/390 CFL") * 0.25
        out["i390_current_cleared_altitude_m"] = out["i390_current_cleared_flight_level"] * 100.0 * FT_TO_M

    def i390_control_position(reader: Any, out: dict[str, Any]) -> None:
        out["i390_control_centre_code"], out["i390_control_position_code"] = reader.u8("I062/390 CTL centre"), reader.u8("I062/390 CTL position")

    def i390_tod(reader: Any, out: dict[str, Any]) -> None:
        rep = reader.u8("I062/390 TOD.REP")
        out["i390_time_departure_arrival_raw"] = [reader.read(4, f"I062/390 TOD[{i}]").hex().upper() for i in range(rep)]

    def i390_stand(reader: Any, out: dict[str, Any]) -> None:
        out["i390_aircraft_stand"] = _ascii(reader.read(6, "I062/390 AST"))

    def i390_stand_status(reader: Any, out: dict[str, Any]) -> None:
        raw = reader.u8("I062/390 STS")
        out["i390_stand_occupancy_code"], out["i390_stand_availability_code"] = (raw >> 6) & 0x03, (raw >> 4) & 0x03

    def i390_ascii(name: str, length: int):
        return lambda reader, out: out.__setitem__(name, _ascii(reader.read(length, f"I062/390 {name}")))

    def i390_pem(reader: Any, out: dict[str, Any]) -> None:
        raw = reader.uint(2, "I062/390 PEM")
        out["i390_pre_emergency_mode3a_valid"] = bool(raw & 0x1000)
        out["i390_pre_emergency_mode3a_octal"] = f"{raw & 0x0FFF:04o}"

    i390 = decoder.I390_HANDLERS
    i390[:] = [
        i390_tag, decoder.parse_i390_callsign, i390_ifps, i390_category,
        i390_aircraft_type, i390_wtc, i390_airport("i390_departure_airport"),
        i390_airport("i390_destination_airport"), i390_runway, i390_cfl,
        i390_control_position, i390_tod, i390_stand, i390_stand_status,
        i390_ascii("i390_sid", 7), i390_ascii("i390_star", 7), i390_pem,
        i390_ascii("i390_pre_emergency_callsign", 7),
    ]

    # I062/290: each sensor-update age is one octet, with an LSB of 0.25 s.
    # These are reliability features, not aircraft-performance features.
    def age_handler(group: str, name: str):
        def parse(reader: Any, out: dict[str, Any]) -> None:
            values = out.setdefault(group, {})
            values[name] = reader.u8(f"{group}.{name}") * 0.25
        return parse

    decoder.I290_HANDLERS[:] = [
        age_handler("system_track_update_ages_sec", name) for name in (
            "track", "psr", "ssr", "mode_s", "ads_c", "adsb_extended_squitter",
            "adsb_vdl_mode4", "adsb_uat", "magnetic_loop", "multilateration",
        )
    ]

    # I062/295 holds freshness of individual derived aircraft parameters.
    track_age_names = (
        "measured_flight_level", "mode1", "mode2", "mode3a", "flight_identification",
        "aircraft_identification", "magnetic_heading", "reserved_sf8", "true_airspeed",
        "selected_altitude", "final_state_selected_altitude", "trajectory_intent",
        "communications_acas", "adsb_status", "acas_resolution_advisory",
        "barometric_vertical_rate", "geometric_vertical_rate", "roll_angle",
        "track_angle_rate", "track_angle", "ground_speed", "velocity_uncertainty",
        "meteorological_data", "emitter_category", "position", "geometric_altitude",
        "position_uncertainty", "mode_s_mb", "indicated_airspeed", "mach",
        "barometric_pressure_setting",
    )
    decoder.I295_HANDLERS[:] = [age_handler("track_data_ages_sec", name) for name in track_age_names]

    def i500_cartesian_position_accuracy(reader: Any, out: dict[str, Any]) -> None:
        out["estimated_position_accuracy_cartesian_m"] = {
            "x": reader.uint(2, "I062/500 APC X") * 0.5,
            "y": reader.uint(2, "I062/500 APC Y") * 0.5,
        }

    def i500_covariance(reader: Any, out: dict[str, Any]) -> None:
        out["estimated_position_covariance_xy_raw"] = reader.sint(2, "I062/500 COV")

    def i500_wgs84_accuracy(reader: Any, out: dict[str, Any]) -> None:
        out["estimated_position_accuracy_wgs84_raw_hex"] = reader.read(4, "I062/500 APW").hex().upper()

    def i500_geometric_altitude_accuracy(reader: Any, out: dict[str, Any]) -> None:
        out["estimated_geometric_altitude_accuracy_m"] = reader.u8("I062/500 AGA") * 6.25 * FT_TO_M

    def i500_barometric_altitude_accuracy(reader: Any, out: dict[str, Any]) -> None:
        out["estimated_barometric_altitude_accuracy_m"] = reader.u8("I062/500 ABA") * 25.0 * FT_TO_M

    def i500_velocity_accuracy(reader: Any, out: dict[str, Any]) -> None:
        out["estimated_velocity_accuracy_mps"] = {
            "vx": reader.u8("I062/500 ATV Vx") * 0.25,
            "vy": reader.u8("I062/500 ATV Vy") * 0.25,
        }

    def i500_acceleration_accuracy(reader: Any, out: dict[str, Any]) -> None:
        out["estimated_acceleration_accuracy_mps2"] = {
            "ax": reader.u8("I062/500 AA Ax") * 0.25,
            "ay": reader.u8("I062/500 AA Ay") * 0.25,
        }

    def i500_rate_of_climb_accuracy(reader: Any, out: dict[str, Any]) -> None:
        out["estimated_rate_of_climb_accuracy_fpm"] = reader.u8("I062/500 ARC") * 6.25

    decoder.I500_HANDLERS[:] = [
        i500_cartesian_position_accuracy, i500_covariance, i500_wgs84_accuracy,
        i500_geometric_altitude_accuracy, i500_barometric_altitude_accuracy,
        i500_velocity_accuracy, i500_acceleration_accuracy, i500_rate_of_climb_accuracy,
    ]

    def i340_sensor(reader: Any, out: dict[str, Any]) -> None:
        out["last_measurement_sensor"] = {"sac": reader.u8("I062/340 SID SAC"), "sic": reader.u8("I062/340 SID SIC")}

    def i340_position(reader: Any, out: dict[str, Any]) -> None:
        out["last_measured_polar_position"] = {"rho_nm": reader.uint(2, "I062/340 POS rho") / 256.0, "theta_deg": reader.uint(2, "I062/340 POS theta") * 360.0 / 65536.0}

    def i340_height(reader: Any, out: dict[str, Any]) -> None:
        out["last_measured_3d_height_raw"] = reader.read(2, "I062/340 HEI").hex().upper()

    def i340_mode_c(reader: Any, out: dict[str, Any]) -> None:
        raw = reader.uint(2, "I062/340 MDC")
        out["last_measured_mode_c"] = {"raw": raw, "flight_level": decoder.sign_extend(raw & 0x3FFF, 14) * 0.25, "validated": not bool(raw & 0x8000), "garbled": bool(raw & 0x4000)}

    def i340_mode3a(reader: Any, out: dict[str, Any]) -> None:
        raw = reader.uint(2, "I062/340 MDA")
        out["last_measured_mode3a"] = {"raw": raw, "mode3a_octal": f"{raw & 0x0FFF:04o}", "validated": not bool(raw & 0x8000), "garbled": bool(raw & 0x4000)}

    def i340_report_type(reader: Any, out: dict[str, Any]) -> None:
        out["last_measurement_report_type_raw"] = reader.u8("I062/340 TYP")

    decoder.I340_HANDLERS[:] = [i340_sensor, i340_position, i340_height, i340_mode_c, i340_mode3a, i340_report_type]

    original_parse_block = decoder.parse_cat062_data_block

    def postprocess_quality(record: dict[str, Any]) -> None:
        status = bytes.fromhex(str(record.get("track_status_hex") or ""))
        if len(status) >= 3:
            record["track_status_coasting"] = bool(status[2] & 0x80)
        if len(status) >= 4:
            record["track_status_flight_plan_coupled"] = bool(status[3] & 0x10)
        movement = record.get("mode_of_movement_raw")
        if movement is not None:
            raw = int(movement)
            record["mode_of_movement"] = {
                "turn_code": (raw >> 6) & 0x03,
                "longitudinal_code": (raw >> 4) & 0x03,
                "vertical_code": (raw >> 2) & 0x03,
                "altitude_discrepancy": bool(raw & 0x02),
            }
        sensor_ages = record.get("system_track_update_ages_sec") or {}
        data_ages = record.get("track_data_ages_sec") or {}
        if sensor_ages:
            record["maximum_system_track_update_age_sec"] = max(sensor_ages.values())
        critical = {key: value for key, value in data_ages.items() if key in {
            "measured_flight_level", "barometric_vertical_rate", "geometric_vertical_rate",
            "track_angle", "ground_speed", "position", "geometric_altitude",
        }}
        if critical:
            record["maximum_critical_track_data_age_sec"] = max(critical.values())

    def parse_with_quality(block: bytes) -> list[dict[str, Any]]:
        records = original_parse_block(block)
        for record in records:
            postprocess_quality(record)
        return records

    decoder.parse_cat062_data_block = parse_with_quality
