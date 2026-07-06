#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inspect an MS-ready HDF5 file produced by test_with_antenna_uvw.py.

This script does not modify the HDF5 file. It checks whether the HDF5 file is
internally consistent and ready for the next step: HDF5 -> MeasurementSet.

Usage:
    python3 inspect_ms_ready_hdf5.py input.h5
    python3 inspect_ms_ready_hdf5.py input.h5 --strict
    python3 inspect_ms_ready_hdf5.py input.h5 --show-examples 5
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

# Compatibility for old h5py + new numpy combinations.
if not hasattr(np, "typeDict"):
    np.typeDict = np.sctypeDict

try:
    import h5py
except Exception as error:  # pragma: no cover
    h5py = None
    H5PY_IMPORT_ERROR = error
else:
    H5PY_IMPORT_ERROR = None


REQUIRED_GROUPS = [
    "baseline",
    "signal",
    "time",
    "frequency",
    "antenna",
    "field",
    "polarization",
    "ms_rows",
    "uvw",
    "ms_defaults",
]

REQUIRED_PATHS = [
    "vis",
    "baseline_pairs",
    "baseline/signal_pairs",
    "baseline/antenna_pairs",
    "baseline/polarization_pairs",
    "signal/present",
    "signal/input_signal_no",
    "signal/antenna_id",
    "signal/polarization_id",
    "signal/file",
    "time/start_mjd",
    "time/center_mjd",
    "time/end_mjd",
    "time/interval_sec",
    "time/exposure_sec",
    "frequency/chan_freq_hz",
    "frequency/chan_width_hz",
    "frequency/ref_frequency_hz",
    "frequency/nchan",
    "antenna/id",
    "antenna/name",
    "antenna/station",
    "antenna/latitude_deg",
    "antenna/longitude_deg",
    "antenna/altitude_m",
    "antenna/position_itrf_m",
    "antenna/dish_diameter_m",
    "antenna/present_in_antenna_txt",
    "antenna/used_in_input",
    "antenna/position_is_placeholder_by_antenna",
    "antenna/position_is_placeholder",
    "field/source_name",
    "field/phase_center_ra_rad",
    "field/phase_center_dec_rad",
    "field/phase_center_ra_deg",
    "field/phase_center_dec_deg",
    "field/phase_center_ra_hms",
    "field/phase_center_dec_dms",
    "field/frame",
    "field/is_placeholder",
    "polarization/input_pol_id",
    "polarization/input_pol_name",
    "polarization/ms_export_mode",
    "polarization/all_corr_names",
    "polarization/all_corr_pol_i",
    "polarization/all_corr_pol_j",
    "ms_rows/time_index",
    "ms_rows/signal_baseline_index",
    "ms_rows/signal_i",
    "ms_rows/signal_j",
    "ms_rows/antenna1",
    "ms_rows/antenna2",
    "ms_rows/pol_i",
    "ms_rows/pol_j",
    "ms_rows/corr_name",
    "ms_rows/data_desc_id",
    "ms_rows/field_id",
    "ms_rows/scan_number",
    "ms_rows/row_has_missing_signal",
    "ms_rows/row_is_auto_signal",
    "ms_rows/row_is_cross_signal",
    "ms_rows/row_is_same_antenna",
    "ms_rows/row_is_cross_antenna",
    "uvw/uvw_m",
    "uvw/is_placeholder",
    "ms_defaults/flag_default",
    "ms_defaults/weight_default",
    "ms_defaults/sigma_default",
    "ms_defaults/missing_signal_should_flag",
]

ROW_LEVEL_PATHS = [
    "ms_rows/time_index",
    "ms_rows/signal_baseline_index",
    "ms_rows/signal_i",
    "ms_rows/signal_j",
    "ms_rows/antenna1",
    "ms_rows/antenna2",
    "ms_rows/pol_i",
    "ms_rows/pol_j",
    "ms_rows/corr_name",
    "ms_rows/data_desc_id",
    "ms_rows/field_id",
    "ms_rows/scan_number",
    "ms_rows/row_has_missing_signal",
    "ms_rows/row_is_auto_signal",
    "ms_rows/row_is_cross_signal",
    "ms_rows/row_is_same_antenna",
    "ms_rows/row_is_cross_antenna",
]

ROOT_LEGACY_DATASETS = [
    "ANTENNA1",
    "ANTENNA2",
    "CHAN_FREQ",
    "CHAN_WIDTH",
    "TIME",
]


def as_text(value: Any) -> str:
    """Decode HDF5 scalar/bytes/object values into normal text."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return as_text(value[()])
        return str(value)
    return str(value)


def read_scalar(h5: h5py.File, path: str) -> Any:
    return h5[path][()]


def read_text_scalar(h5: h5py.File, path: str) -> str:
    return as_text(read_scalar(h5, path))


def decode_string_array(array: np.ndarray) -> list[str]:
    out: list[str] = []
    for item in array:
        out.append(as_text(item))
    return out


def fmt_bool(value: Any) -> str:
    return "yes" if bool(value) else "no"


@dataclass
class Report:
    strict: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def note(self, message: str) -> None:
        self.notes.append(message)

    def ok(self) -> bool:
        if self.errors:
            return False
        if self.strict and self.warnings:
            return False
        return True


def check_exists(h5: h5py.File, report: Report) -> None:
    for group in REQUIRED_GROUPS:
        if group not in h5:
            report.error(f"missing required group: /{group}")

    for path in REQUIRED_PATHS:
        if path not in h5:
            report.error(f"missing required path: /{path}")

    for name in ROOT_LEGACY_DATASETS:
        if name in h5:
            report.warn(f"legacy root dataset exists and should not be used: /{name}")


def check_shapes_and_counts(h5: h5py.File, report: Report) -> dict[str, int]:
    info: dict[str, int] = {}

    if "vis" not in h5 or "time/center_mjd" not in h5 or "baseline_pairs" not in h5:
        return info

    vis_shape = h5["vis"].shape
    if len(vis_shape) != 3:
        report.error(f"/vis should be 3-D, got shape {vis_shape}")
        return info

    n_corr_time, n_baseline, nchan = map(int, vis_shape)
    info.update(n_corr_time=n_corr_time, n_baseline=n_baseline, nchan=nchan)

    if h5["time/center_mjd"].shape != (n_corr_time,):
        report.error(
            f"/time/center_mjd length mismatch: "
            f"{h5['time/center_mjd'].shape} != {(n_corr_time,)}"
        )

    if h5["baseline_pairs"].shape != (n_baseline, 2):
        report.error(
            f"/baseline_pairs shape mismatch: "
            f"{h5['baseline_pairs'].shape} != {(n_baseline, 2)}"
        )

    if h5["baseline/signal_pairs"].shape != (n_baseline, 2):
        report.error(
            f"/baseline/signal_pairs shape mismatch: "
            f"{h5['baseline/signal_pairs'].shape} != {(n_baseline, 2)}"
        )

    if h5["baseline/antenna_pairs"].shape != (n_baseline, 2):
        report.error(
            f"/baseline/antenna_pairs shape mismatch: "
            f"{h5['baseline/antenna_pairs'].shape} != {(n_baseline, 2)}"
        )

    if h5["baseline/polarization_pairs"].shape != (n_baseline, 2):
        report.error(
            f"/baseline/polarization_pairs shape mismatch: "
            f"{h5['baseline/polarization_pairs'].shape} != {(n_baseline, 2)}"
        )

    freq_nchan = int(read_scalar(h5, "frequency/nchan"))
    if freq_nchan != nchan:
        report.error(f"/frequency/nchan mismatch: {freq_nchan} != /vis nchan {nchan}")

    for path in ["frequency/chan_freq_hz", "frequency/chan_width_hz"]:
        if h5[path].shape != (nchan,):
            report.error(f"/{path} shape mismatch: {h5[path].shape} != {(nchan,)}")

    signal_present = h5["signal/present"][()].astype(bool)
    n_signal = signal_present.size
    n_present_signal = int(signal_present.sum())
    expected_all_pairs = n_signal * (n_signal + 1) // 2
    expected_selected_pairs = n_present_signal * (n_present_signal + 1) // 2

    info.update(
        n_signal=n_signal,
        n_present_signal=n_present_signal,
        expected_all_pairs=expected_all_pairs,
        expected_selected_pairs=expected_selected_pairs,
    )

    if n_baseline != expected_all_pairs:
        report.warn(
            f"/vis baseline count is {n_baseline}, but signal count {n_signal} "
            f"implies {expected_all_pairs} unique signal pairs"
        )

    selected_baseline_count = int(h5["ms_rows"].attrs.get("selected_baseline_count", -1))
    n_ms_rows = int(h5["ms_rows"].attrs.get("n_ms_rows", -1))
    expected_n_ms_rows = n_corr_time * selected_baseline_count

    info.update(selected_baseline_count=selected_baseline_count, n_ms_rows=n_ms_rows)

    if selected_baseline_count != expected_selected_pairs:
        report.error(
            f"selected_baseline_count mismatch: {selected_baseline_count} != "
            f"n_present_signal*(n_present_signal+1)//2 = {expected_selected_pairs}"
        )

    if n_ms_rows != expected_n_ms_rows:
        report.error(f"n_ms_rows mismatch: {n_ms_rows} != {expected_n_ms_rows}")

    for path in ROW_LEVEL_PATHS:
        if path in h5 and h5[path].shape != (n_ms_rows,):
            report.error(f"/{path} length mismatch: {h5[path].shape} != {(n_ms_rows,)}")

    if "uvw/uvw_m" in h5 and h5["uvw/uvw_m"].shape != (n_ms_rows, 3):
        report.error(f"/uvw/uvw_m shape mismatch: {h5['uvw/uvw_m'].shape} != {(n_ms_rows, 3)}")

    return info


def check_field(h5: h5py.File, report: Report) -> None:
    if "field/is_placeholder" not in h5:
        return

    is_placeholder = int(read_scalar(h5, "field/is_placeholder"))
    if is_placeholder != 0:
        report.error("/field/is_placeholder is not 0; phase center is still placeholder")

    ra_rad = float(read_scalar(h5, "field/phase_center_ra_rad"))
    dec_rad = float(read_scalar(h5, "field/phase_center_dec_rad"))
    ra_deg = float(read_scalar(h5, "field/phase_center_ra_deg"))
    dec_deg = float(read_scalar(h5, "field/phase_center_dec_deg"))

    if not np.isfinite(ra_rad) or not (0.0 <= ra_rad < 2.0 * np.pi):
        report.error(f"bad RA radian value: {ra_rad}")
    if not np.isfinite(dec_rad) or not (-0.5 * np.pi <= dec_rad <= 0.5 * np.pi):
        report.error(f"bad Dec radian value: {dec_rad}")

    if abs(np.rad2deg(ra_rad) - ra_deg) > 1e-8:
        report.error("field RA rad/deg values are inconsistent")
    if abs(np.rad2deg(dec_rad) - dec_deg) > 1e-8:
        report.error("field Dec rad/deg values are inconsistent")


def check_antenna(h5: h5py.File, report: Report) -> None:
    used = h5["antenna/used_in_input"][()].astype(bool)
    placeholder_by_ant = h5["antenna/position_is_placeholder_by_antenna"][()].astype(bool)
    present_in_txt = h5["antenna/present_in_antenna_txt"][()].astype(bool)
    position = h5["antenna/position_itrf_m"][()]

    if position.ndim != 2 or position.shape[1] != 3:
        report.error(f"/antenna/position_itrf_m should have shape (nant,3), got {position.shape}")
        return

    if used.shape != placeholder_by_ant.shape:
        report.error("antenna used_in_input and placeholder arrays have different shapes")
        return

    bad_used = np.where(used & placeholder_by_ant)[0]
    if bad_used.size > 0:
        report.error(
            "used antennas still have placeholder positions: "
            + ", ".join([f"index {int(i)}" for i in bad_used])
        )

    missing_txt_used = np.where(used & ~present_in_txt)[0]
    if missing_txt_used.size > 0:
        report.error(
            "used antennas missing from antenna txt: "
            + ", ".join([f"index {int(i)}" for i in missing_txt_used])
        )

    used_positions = position[used]
    if used_positions.size > 0 and not np.all(np.isfinite(used_positions)):
        report.error("some used antenna ITRF positions are not finite")

    global_placeholder = int(read_scalar(h5, "antenna/position_is_placeholder"))
    if global_placeholder != 0:
        report.warn(
            "/antenna/position_is_placeholder is 1. This can be OK if only unused antennas "
            "are missing from antenna txt, but all used antennas must be real."
        )


def check_frequency(h5: h5py.File, report: Report) -> None:
    chan_freq = h5["frequency/chan_freq_hz"][()]
    chan_width = h5["frequency/chan_width_hz"][()]

    if not np.all(np.isfinite(chan_freq)):
        report.error("/frequency/chan_freq_hz contains non-finite values")
    if not np.all(np.isfinite(chan_width)):
        report.error("/frequency/chan_width_hz contains non-finite values")
    if np.any(chan_width <= 0.0):
        report.error("/frequency/chan_width_hz contains non-positive widths")

    if chan_freq.size >= 2:
        diffs = np.diff(chan_freq)
        positive = np.all(diffs > 0.0)
        negative = np.all(diffs < 0.0)
        if not (positive or negative):
            report.warn("frequency channels are not strictly monotonic")

        channel_order = h5["frequency"].attrs.get("channel_order", "")
        channel_order = as_text(channel_order)
        if channel_order == "ascending" and not positive:
            report.warn("frequency/channel_order says ascending, but channel frequencies are not ascending")
        if channel_order == "descending" and not negative:
            report.warn("frequency/channel_order says descending, but channel frequencies are not descending")


def check_time(h5: h5py.File, report: Report) -> None:
    start = h5["time/start_mjd"][()]
    center = h5["time/center_mjd"][()]
    end = h5["time/end_mjd"][()]
    interval_sec = float(read_scalar(h5, "time/interval_sec"))
    exposure_sec = float(read_scalar(h5, "time/exposure_sec"))

    if not (start.shape == center.shape == end.shape):
        report.error("time start/center/end arrays have different shapes")
        return

    if start.size == 0:
        report.error("time arrays are empty")
        return

    if not np.all(start < center) or not np.all(center < end):
        report.error("time ordering should satisfy start_mjd < center_mjd < end_mjd")

    if start.size >= 2:
        step_sec = np.median(np.diff(center) * 86400.0)
        if abs(step_sec - interval_sec) > max(1e-9, interval_sec * 1e-6):
            report.warn(f"time center step {step_sec} sec differs from interval_sec {interval_sec}")

    if interval_sec <= 0.0 or exposure_sec <= 0.0:
        report.error("time interval/exposure must be positive")

    if exposure_sec > interval_sec * 1.000001:
        report.warn("exposure_sec is larger than interval_sec")


def check_ms_rows(h5: h5py.File, report: Report) -> None:
    signal_present = h5["signal/present"][()].astype(bool)
    signal_pairs = h5["baseline/signal_pairs"][()]
    antenna_pairs = h5["baseline/antenna_pairs"][()]
    pol_pairs = h5["baseline/polarization_pairs"][()]

    row_bl = h5["ms_rows/signal_baseline_index"][()].astype(np.int64)
    row_si = h5["ms_rows/signal_i"][()].astype(np.int64)
    row_sj = h5["ms_rows/signal_j"][()].astype(np.int64)
    row_a1 = h5["ms_rows/antenna1"][()].astype(np.int64)
    row_a2 = h5["ms_rows/antenna2"][()].astype(np.int64)
    row_pi = h5["ms_rows/pol_i"][()].astype(np.int64)
    row_pj = h5["ms_rows/pol_j"][()].astype(np.int64)
    row_missing = h5["ms_rows/row_has_missing_signal"][()].astype(bool)
    row_auto = h5["ms_rows/row_is_auto_signal"][()].astype(bool)
    row_same_ant = h5["ms_rows/row_is_same_antenna"][()].astype(bool)

    if row_bl.size == 0:
        report.error("/ms_rows has zero rows")
        return

    if np.any(row_bl < 0) or np.any(row_bl >= signal_pairs.shape[0]):
        report.error("some ms_rows/signal_baseline_index values are out of range")
        return

    expected_si_sj = signal_pairs[row_bl]
    expected_a1_a2 = antenna_pairs[row_bl]
    expected_pi_pj = pol_pairs[row_bl]

    if not np.array_equal(row_si, expected_si_sj[:, 0]) or not np.array_equal(row_sj, expected_si_sj[:, 1]):
        report.error("ms_rows signal_i/signal_j do not match baseline/signal_pairs")

    if not np.array_equal(row_a1, expected_a1_a2[:, 0]) or not np.array_equal(row_a2, expected_a1_a2[:, 1]):
        report.error("ms_rows antenna1/antenna2 do not match baseline/antenna_pairs")

    if not np.array_equal(row_pi, expected_pi_pj[:, 0]) or not np.array_equal(row_pj, expected_pi_pj[:, 1]):
        report.error("ms_rows pol_i/pol_j do not match baseline/polarization_pairs")

    if np.any(~signal_present[row_si]) or np.any(~signal_present[row_sj]):
        report.error("ms_rows includes signal pairs that are not present in the input .fil files")

    if np.any(row_missing):
        report.error("row_has_missing_signal should be all False for PRESENT_SIGNALS_ONLY mode")

    if not np.array_equal(row_auto, row_si == row_sj):
        report.error("row_is_auto_signal does not match signal_i == signal_j")

    if not np.array_equal(row_same_ant, row_a1 == row_a2):
        report.error("row_is_same_antenna does not match antenna1 == antenna2")

    # Check selected baseline pattern repeats for every time index.
    selected_baseline_count = int(h5["ms_rows"].attrs.get("selected_baseline_count", -1))
    n_ms_rows = int(h5["ms_rows"].attrs.get("n_ms_rows", -1))
    if selected_baseline_count > 0 and n_ms_rows % selected_baseline_count == 0:
        first_block = row_bl[:selected_baseline_count]
        n_time = n_ms_rows // selected_baseline_count
        for t in range(1, n_time):
            block = row_bl[t * selected_baseline_count:(t + 1) * selected_baseline_count]
            if not np.array_equal(block, first_block):
                report.error("signal_baseline_index pattern does not repeat for each time step")
                break


def check_polarization(h5: h5py.File, report: Report) -> None:
    all_corr_names = decode_string_array(h5["polarization/all_corr_names"][()])
    corr_name = decode_string_array(h5["ms_rows/corr_name"][()])
    allowed = set(all_corr_names)
    actual = set(corr_name)

    unknown = sorted(actual - allowed)
    if unknown:
        report.error("ms_rows/corr_name contains values not in polarization/all_corr_names: " + ", ".join(unknown))

    if not allowed.issuperset({"XX", "XY", "YX", "YY"}):
        report.warn("polarization/all_corr_names does not contain the full XX, XY, YX, YY set")

    ms_export_mode = read_text_scalar(h5, "polarization/ms_export_mode")
    if ms_export_mode != "ALL_SIGNAL_PAIRS":
        report.warn(f"polarization/ms_export_mode is {ms_export_mode!r}, expected 'ALL_SIGNAL_PAIRS'")


def check_uvw(h5: h5py.File, report: Report, same_ant_tol: float, uvw_equal_tol: float) -> None:
    uvw = h5["uvw/uvw_m"][()]
    ant1 = h5["ms_rows/antenna1"][()].astype(np.int64)
    ant2 = h5["ms_rows/antenna2"][()].astype(np.int64)
    time_index = h5["ms_rows/time_index"][()].astype(np.int64)
    corr_name = decode_string_array(h5["ms_rows/corr_name"][()])

    if int(read_scalar(h5, "uvw/is_placeholder")) != 0:
        report.error("/uvw/is_placeholder is not 0")

    method = as_text(h5["uvw"].attrs.get("method", ""))
    if method != "katpoint.Target.uvw":
        report.warn(f"UVW method is {method!r}; expected 'katpoint.Target.uvw'")

    if not np.all(np.isfinite(uvw)):
        report.error("/uvw/uvw_m contains non-finite values")
        return

    same_ant = ant1 == ant2
    if np.any(same_ant):
        max_same = float(np.max(np.abs(uvw[same_ant])))
        if max_same > same_ant_tol:
            report.error(f"same-antenna UVW should be zero; max abs value is {max_same:g}")

    cross_ant = ant1 != ant2
    if np.any(cross_ant):
        max_cross = float(np.max(np.abs(uvw[cross_ant])))
        if max_cross <= 0.0:
            report.error("cross-antenna UVW is all zero")
    else:
        report.warn("there are no cross-antenna rows; UV coverage cannot be inspected")

    # Same physical baseline and same time should have identical UVW for all polarization products.
    # This is important because XX/XY/YX/YY are different DATA products, not different geometry.
    key_to_first: dict[tuple[int, int, int], int] = {}
    bad_equal_count = 0
    for row, key in enumerate(zip(time_index, ant1, ant2)):
        key_i = (int(key[0]), int(key[1]), int(key[2]))
        if key_i not in key_to_first:
            key_to_first[key_i] = row
            continue
        ref_row = key_to_first[key_i]
        diff = float(np.max(np.abs(uvw[row] - uvw[ref_row])))
        if diff > uvw_equal_tol:
            bad_equal_count += 1
            if bad_equal_count <= 5:
                report.error(
                    "UVW differs for same time/antenna baseline across polarization products: "
                    f"row {row} ({corr_name[row]}) vs row {ref_row} ({corr_name[ref_row]}), diff={diff:g}"
                )
    if bad_equal_count > 5:
        report.error(f"UVW equality failed for {bad_equal_count} same-baseline rows total")


def check_vis_access(h5: h5py.File, report: Report, show_examples: int) -> None:
    vis = h5["vis"]
    row_bl = h5["ms_rows/signal_baseline_index"][()].astype(np.int64)
    n_baseline = vis.shape[1]

    if np.any(row_bl < 0) or np.any(row_bl >= n_baseline):
        report.error("some ms_rows/signal_baseline_index values cannot index /vis")
        return

    # Read a tiny sample only. The real data may be very large.
    sample_rows = min(show_examples, row_bl.size)
    for row in range(sample_rows):
        bl = int(row_bl[row])
        _ = vis[0, bl, 0]


def print_summary(h5: h5py.File, info: dict[str, int], report: Report, show_examples: int) -> None:
    print("\n========== MS-READY HDF5 INSPECTION ==========")
    print("file:", h5.filename)
    print("file size:", human_bytes(os.path.getsize(h5.filename)))

    if info:
        print("\n--- basic shape ---")
        print("vis shape              :", h5["vis"].shape)
        print("n_corr_time            :", info.get("n_corr_time"))
        print("n_baseline slots       :", info.get("n_baseline"))
        print("nchan                  :", info.get("nchan"))
        print("n_input_signal         :", info.get("n_signal"))
        print("n_present_signal       :", info.get("n_present_signal"))
        print("selected_baseline_count:", info.get("selected_baseline_count"))
        print("n_ms_rows              :", info.get("n_ms_rows"))

    print("\n--- field ---")
    print("source_name            :", read_text_scalar(h5, "field/source_name"))
    print("phase RA HMS           :", read_text_scalar(h5, "field/phase_center_ra_hms"))
    print("phase Dec DMS          :", read_text_scalar(h5, "field/phase_center_dec_dms"))
    print("phase RA deg           :", float(read_scalar(h5, "field/phase_center_ra_deg")))
    print("phase Dec deg          :", float(read_scalar(h5, "field/phase_center_dec_deg")))
    print("field placeholder      :", int(read_scalar(h5, "field/is_placeholder")))

    print("\n--- antenna ---")
    used = h5["antenna/used_in_input"][()].astype(bool)
    present_txt = h5["antenna/present_in_antenna_txt"][()].astype(bool)
    placeholder = h5["antenna/position_is_placeholder_by_antenna"][()].astype(bool)
    print("used antenna indices   :", list(np.where(used)[0].astype(int)))
    print("present in txt indices :", list(np.where(present_txt)[0].astype(int)))
    print("placeholder indices    :", list(np.where(placeholder)[0].astype(int)))
    print("global placeholder     :", int(read_scalar(h5, "antenna/position_is_placeholder")))

    print("\n--- frequency ---")
    chan_freq = h5["frequency/chan_freq_hz"]
    print("frequency nchan        :", int(read_scalar(h5, "frequency/nchan")))
    print("first chan Hz          :", float(chan_freq[0]))
    print("last chan Hz           :", float(chan_freq[-1]))
    print("channel order attr     :", as_text(h5["frequency"].attrs.get("channel_order", "")))

    print("\n--- rows / polarization ---")
    corr_names = decode_string_array(h5["ms_rows/corr_name"][()])
    unique_corr, counts = np.unique(np.array(corr_names, dtype=object), return_counts=True)
    print("corr_name counts       :", dict(zip(unique_corr.tolist(), counts.astype(int).tolist())))
    print("export mode            :", read_text_scalar(h5, "polarization/ms_export_mode"))
    print("row missing any        :", bool(np.any(h5["ms_rows/row_has_missing_signal"][()])))

    print("\n--- uvw ---")
    uvw = h5["uvw/uvw_m"]
    ant1 = h5["ms_rows/antenna1"][()].astype(np.int64)
    ant2 = h5["ms_rows/antenna2"][()].astype(np.int64)
    cross = ant1 != ant2
    print("uvw shape              :", uvw.shape)
    print("uvw placeholder        :", int(read_scalar(h5, "uvw/is_placeholder")))
    print("uvw method             :", as_text(h5["uvw"].attrs.get("method", "")))
    if np.any(cross):
        uvw_cross = uvw[()][cross]
        print("max |uvw| cross m      :", float(np.max(np.abs(uvw_cross))))
    else:
        print("max |uvw| cross m      : no cross antenna rows")

    if show_examples > 0:
        print("\n--- example ms_rows ---")
        n = min(show_examples, int(h5["ms_rows"].attrs.get("n_ms_rows", 0)))
        row_bl = h5["ms_rows/signal_baseline_index"][()].astype(np.int64)
        si = h5["ms_rows/signal_i"][()].astype(np.int64)
        sj = h5["ms_rows/signal_j"][()].astype(np.int64)
        pi = h5["ms_rows/pol_i"][()].astype(np.int64)
        pj = h5["ms_rows/pol_j"][()].astype(np.int64)
        ti = h5["ms_rows/time_index"][()].astype(np.int64)
        uvw_arr = h5["uvw/uvw_m"][()]
        for row in range(n):
            print(
                f"row {row:5d}: time={ti[row]} bl_slot={row_bl[row]} "
                f"sig=({si[row]},{sj[row]}) ant=({ant1[row]},{ant2[row]}) "
                f"pol=({pi[row]},{pj[row]}) corr={corr_names[row]} "
                f"uvw=({uvw_arr[row,0]:.6f}, {uvw_arr[row,1]:.6f}, {uvw_arr[row,2]:.6f})"
            )

    print("\n--- check result ---")
    if report.errors:
        print(f"ERRORS ({len(report.errors)}):")
        for msg in report.errors:
            print("  [ERROR]", msg)
    else:
        print("ERRORS: none")

    if report.warnings:
        print(f"WARNINGS ({len(report.warnings)}):")
        for msg in report.warnings:
            print("  [WARN]", msg)
    else:
        print("WARNINGS: none")

    if report.ok():
        print("\nRESULT: PASS")
        print("Next step: this HDF5 is structurally ready for an HDF5 -> MS converter.")
    else:
        print("\nRESULT: FAIL")
        if report.strict and report.warnings and not report.errors:
            print("Reason: --strict treats warnings as failures.")
        print("Fix the issues above before converting to MeasurementSet.")

    print("=============================================\n")


def human_bytes(n_bytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(n_bytes)
    unit = units[0]
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            break
        value /= 1024.0
    return f"{value:.3f} {unit}"


def inspect_file(args: argparse.Namespace) -> int:
    if h5py is None:
        print(f"[ERROR] h5py is unavailable: {H5PY_IMPORT_ERROR}", file=sys.stderr)
        return 2

    if not os.path.isfile(args.hdf5_file):
        print(f"[ERROR] HDF5 file not found: {args.hdf5_file}", file=sys.stderr)
        return 2

    report = Report(strict=args.strict)

    with h5py.File(args.hdf5_file, "r") as h5:
        check_exists(h5, report)

        # Stop early if required core paths are missing, otherwise later checks become noisy.
        if report.errors:
            print_summary_minimal(h5, report)
            return 1

        info = check_shapes_and_counts(h5, report)
        check_field(h5, report)
        check_antenna(h5, report)
        check_frequency(h5, report)
        check_time(h5, report)
        check_ms_rows(h5, report)
        check_polarization(h5, report)
        check_uvw(
            h5,
            report,
            same_ant_tol=args.same_antenna_uvw_tol,
            uvw_equal_tol=args.uvw_equal_tol,
        )
        check_vis_access(h5, report, show_examples=args.show_examples)
        print_summary(h5, info, report, show_examples=args.show_examples)

    return 0 if report.ok() else 1


def print_summary_minimal(h5: h5py.File, report: Report) -> None:
    print("\n========== MS-READY HDF5 INSPECTION ==========")
    print("file:", h5.filename)
    print("\n--- check result ---")
    if report.errors:
        print(f"ERRORS ({len(report.errors)}):")
        for msg in report.errors:
            print("  [ERROR]", msg)
    if report.warnings:
        print(f"WARNINGS ({len(report.warnings)}):")
        for msg in report.warnings:
            print("  [WARN]", msg)
    print("\nRESULT: FAIL")
    print("=============================================\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect an MS-ready HDF5 file before converting it to CASA "
            "MeasurementSet. The script only reads the file and performs "
            "structure, geometry, UVW, row-mapping and metadata checks."
        )
    )
    parser.add_argument(
        "hdf5_file",
        help="input HDF5 file produced by test_with_antenna_uvw.py",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="treat warnings as failures",
    )
    parser.add_argument(
        "--show-examples",
        type=int,
        default=5,
        help="number of example ms_rows to print, default: 5",
    )
    parser.add_argument(
        "--same-antenna-uvw-tol",
        type=float,
        default=1e-6,
        help="tolerance in metres for same-antenna UVW zero check, default: 1e-6",
    )
    parser.add_argument(
        "--uvw-equal-tol",
        type=float,
        default=1e-9,
        help=(
            "tolerance in metres for checking that the same physical baseline "
            "has identical UVW across polarization products, default: 1e-9"
        ),
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    code = inspect_file(args)
    sys.exit(code)


if __name__ == "__main__":
    main()
