#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot a time-frequency waterfall directly from an MS-ready CARRY HDF5 file.

Examples:
    python plot_h5_phase_waterfall_timefreq_bjt.py 0X 1Y xxx_cal.h5
    python plot_h5_phase_waterfall_timefreq_bjt.py 0X 1Y xxx_tar.h5 out.png
    python plot_h5_phase_waterfall_timefreq_bjt.py 0X 1Y xxx_cal.h5 --mode amp
    python plot_h5_phase_waterfall_timefreq_bjt.py --list-signals xxx_cal.h5
    python plot_h5_phase_waterfall_timefreq_bjt.py --list-baselines xxx_cal.h5

The selected signal pair is interpreted literally:
    0X 1Y means ant0-X x conj(ant1-Y)

Dependencies:
    h5py, numpy, matplotlib

This script does not read MeasurementSet files and does not import casacore or
CASA tools.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime, timedelta, timezone

import numpy as np

try:
    import h5py
except Exception as error:  # pragma: no cover - exercised only without h5py
    h5py = None
    H5PY_IMPORT_ERROR = error
else:
    H5PY_IMPORT_ERROR = None

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


plt.rcParams["axes.unicode_minus"] = False


MJD_UNIX_EPOCH = 40587.0
BEIJING_TZ = timezone(timedelta(hours=8), name="BJT")

POL_NAME_TO_ID = {
    "X": 0,
    "Y": 1,
}
POL_ID_TO_NAME = {
    0: "X",
    1: "Y",
}

REQUIRED_PATHS = [
    "vis",
    "baseline/signal_pairs",
    "signal/present",
    "signal/antenna_id",
    "signal/polarization_id",
    "signal/file",
    "time/center_mjd",
    "time/exposure_sec",
    "frequency/chan_freq_hz",
    "frequency/chan_width_hz",
    "field/source_name",
    "field/phase_center_ra_hms",
    "field/phase_center_dec_dms",
    "field/frame",
]


def require_h5py() -> None:
    if h5py is None:
        raise RuntimeError("h5py is required to read HDF5 files") from H5PY_IMPORT_ERROR


def as_text(value) -> str:
    """Convert common h5py scalar/bytes values to plain text."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8", errors="replace")

    if isinstance(value, np.generic):
        return as_text(value.item())

    if isinstance(value, np.ndarray):
        if value.shape == ():
            return as_text(value[()])
        if value.size == 1:
            return as_text(value.reshape(-1)[0])
        return str([as_text(item) for item in value.reshape(-1)])

    return str(value)


def read_text_dataset(h5, path: str, default: str = "unknown") -> str:
    if path not in h5:
        return default
    return as_text(h5[path][()])


def read_attr_text(h5, name: str, default: str = "unknown") -> str:
    if name not in h5.attrs:
        return default
    return as_text(h5.attrs[name])


def parse_signal_text(text: str) -> tuple[int, str]:
    """
    Parse signal strings such as 0X, 1Y, ant0X, ANT03Y, 0:x, or 1-y.

    Returns:
        antenna_id, pol_name
    """
    if text is None:
        raise ValueError("signal text is None")

    clean = str(text).strip()
    if clean == "":
        raise ValueError("signal text is empty")

    clean = clean.replace(" ", "")
    clean = clean.replace("_", "")
    clean = clean.replace("-", "")
    clean = clean.replace(":", "")
    clean = clean.replace("&", "")
    clean = re.sub(r"(?i)^ant", "", clean)

    match = re.match(r"^([0-9]+)([A-Za-z])$", clean)
    if match is None:
        raise ValueError(
            "Bad signal format: {0}. Expected examples: 0X, 1Y, ant0X".format(
                text
            )
        )

    antenna_id = int(match.group(1))
    pol_name = match.group(2).upper()

    if pol_name not in POL_NAME_TO_ID:
        raise ValueError(
            "Unsupported polarization {0}. Current script supports X/Y only.".format(
                pol_name
            )
        )

    return antenna_id, pol_name


def pol_name_to_id(pol_name: str) -> int:
    key = str(pol_name).strip().upper()
    if key not in POL_NAME_TO_ID:
        raise ValueError(
            "Unsupported polarization {0}. Current script supports X/Y only.".format(
                pol_name
            )
        )
    return POL_NAME_TO_ID[key]


def pol_id_to_name(pol_id: int) -> str:
    return POL_ID_TO_NAME.get(int(pol_id), "pol{0}".format(int(pol_id)))


def default_output_path(sig1: str, sig2: str, h5_file: str, mode: str) -> str:
    h5_path = os.path.abspath(h5_file)
    parent = os.path.dirname(h5_path)
    stem = os.path.splitext(os.path.basename(h5_path))[0]

    if mode == "amp":
        prefix = "amp_waterfall"
    elif mode == "real":
        prefix = "real_waterfall"
    elif mode == "imag":
        prefix = "imag_waterfall"
    else:
        prefix = "phase_waterfall"

    return os.path.join(parent, "{0}_{1}_{2}_{3}.png".format(prefix, sig1, sig2, stem))


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description=(
            "Plot a phase/amplitude waterfall directly from CARRY HDF5 /vis "
            "for a selected signal pair."
        )
    )
    parser.add_argument(
        "items",
        nargs="*",
        help=(
            "plot mode: signal1 signal2 h5_file [out_png]; "
            "list mode: h5_file"
        ),
    )
    parser.add_argument(
        "--mode",
        default="phase",
        choices=["phase", "amp", "real", "imag"],
        help="plot mode. Default: phase",
    )
    parser.add_argument(
        "--save-npy",
        action="store_true",
        help="also save complex visibility, phase, amplitude, frequency and time arrays",
    )
    parser.add_argument(
        "--save-txt",
        action="store_true",
        help="also save a text summary next to the PNG",
    )
    parser.add_argument(
        "--zero-as-nan",
        action="store_true",
        help="mask samples with amplitude <= --amp-threshold as NaN",
    )
    parser.add_argument(
        "--amp-threshold",
        type=float,
        default=0.0,
        help="amplitude threshold used by --zero-as-nan. Default: 0.0",
    )
    parser.add_argument(
        "--list-signals",
        action="store_true",
        help="list HDF5 signal metadata and exit",
    )
    parser.add_argument(
        "--list-baselines",
        action="store_true",
        help="list plottable HDF5 baseline signal pairs and exit",
    )
    parser.add_argument(
        "--max-list",
        type=int,
        default=50,
        help="maximum rows to print in --list-baselines. Default: 50",
    )

    args = parser.parse_args(argv)

    if args.max_list < 1:
        parser.error("--max-list must be >= 1")

    if not np.isfinite(args.amp_threshold):
        parser.error("--amp-threshold must be finite")

    if args.list_signals or args.list_baselines:
        if len(args.items) != 1:
            parser.error("--list-signals/--list-baselines expects exactly one h5_file")
        args.signal1 = None
        args.signal2 = None
        args.h5_file = os.path.abspath(args.items[0].rstrip("/\\"))
        args.out_png = None
        return args

    if len(args.items) not in (3, 4):
        parser.error(
            "plot mode expects: signal1 signal2 h5_file [out_png]. "
            "Example: 0X 1Y xxx_cal.h5"
        )

    args.signal1 = str(args.items[0]).strip()
    args.signal2 = str(args.items[1]).strip()
    args.h5_file = os.path.abspath(args.items[2].rstrip("/\\"))

    if len(args.items) == 4:
        args.out_png = os.path.abspath(args.items[3])
    else:
        args.out_png = default_output_path(
            args.signal1,
            args.signal2,
            args.h5_file,
            args.mode,
        )

    return args


def validate_h5_for_plot(h5) -> None:
    for path in REQUIRED_PATHS:
        if path not in h5:
            raise ValueError("missing required HDF5 path: /{0}".format(path))

    vis = h5["vis"]
    if len(vis.shape) != 3:
        raise ValueError("/vis must be 3D, got shape={0}".format(vis.shape))

    signal_pairs = np.asarray(h5["baseline/signal_pairs"][()], dtype=np.int64)
    if signal_pairs.ndim != 2 or signal_pairs.shape[1] != 2:
        raise ValueError(
            "/baseline/signal_pairs must have shape (Nbaseline, 2), got {0}".format(
                signal_pairs.shape
            )
        )

    center_mjd = np.asarray(h5["time/center_mjd"][()], dtype=np.float64)
    if center_mjd.ndim != 1:
        raise ValueError("/time/center_mjd must be 1D, got shape={0}".format(center_mjd.shape))
    if center_mjd.size == 0:
        raise ValueError("/time/center_mjd is empty")
    if np.any(~np.isfinite(center_mjd)):
        raise ValueError("/time/center_mjd contains non-finite values")

    exposure_sec = float(np.asarray(h5["time/exposure_sec"][()]))
    if not np.isfinite(exposure_sec) or exposure_sec <= 0.0:
        raise ValueError("/time/exposure_sec must be finite and > 0")

    freq_hz = np.asarray(h5["frequency/chan_freq_hz"][()], dtype=np.float64)
    if freq_hz.ndim != 1:
        raise ValueError(
            "/frequency/chan_freq_hz must be 1D, got shape={0}".format(
                freq_hz.shape
            )
        )
    if freq_hz.size == 0:
        raise ValueError("/frequency/chan_freq_hz is empty")
    if np.any(~np.isfinite(freq_hz)):
        raise ValueError("/frequency/chan_freq_hz contains non-finite values")

    chan_width_hz = np.asarray(h5["frequency/chan_width_hz"][()], dtype=np.float64)
    if chan_width_hz.ndim != 1 or chan_width_hz.size != freq_hz.size:
        raise ValueError(
            "/frequency/chan_width_hz shape mismatch: {0} != ({1},)".format(
                chan_width_hz.shape,
                freq_hz.size,
            )
        )
    if np.any(~np.isfinite(chan_width_hz)):
        raise ValueError("/frequency/chan_width_hz contains non-finite values")

    if vis.shape[0] != center_mjd.size:
        raise ValueError(
            "/vis time axis mismatch: {0} != {1}".format(
                vis.shape[0],
                center_mjd.size,
            )
        )
    if vis.shape[1] != signal_pairs.shape[0]:
        raise ValueError(
            "/vis baseline axis mismatch: {0} != {1}".format(
                vis.shape[1],
                signal_pairs.shape[0],
            )
        )
    if vis.shape[2] != freq_hz.size:
        raise ValueError(
            "/vis frequency axis mismatch: {0} != {1}".format(
                vis.shape[2],
                freq_hz.size,
            )
        )

    present = np.asarray(h5["signal/present"][()]).reshape(-1)
    antenna_ids = np.asarray(h5["signal/antenna_id"][()]).reshape(-1)
    polarization_ids = np.asarray(h5["signal/polarization_id"][()]).reshape(-1)
    signal_files = np.asarray(h5["signal/file"][()]).reshape(-1)

    if antenna_ids.size != present.size:
        raise ValueError("/signal/antenna_id length mismatch")
    if polarization_ids.size != present.size:
        raise ValueError("/signal/polarization_id length mismatch")
    if signal_files.size != present.size:
        raise ValueError("/signal/file length mismatch")

    if signal_pairs.size > 0:
        min_signal = int(np.min(signal_pairs))
        max_signal = int(np.max(signal_pairs))
        if min_signal < 0 or max_signal >= present.size:
            raise ValueError(
                "/baseline/signal_pairs references signal index outside [0, {0}]".format(
                    present.size - 1
                )
            )

    if "field/is_placeholder" in h5:
        try:
            if int(np.asarray(h5["field/is_placeholder"][()])) != 0:
                print("[WARN] /field/is_placeholder is non-zero")
        except Exception as error:
            print("[WARN] cannot read /field/is_placeholder:", error)

    if "uvw/is_placeholder" in h5:
        try:
            if int(np.asarray(h5["uvw/is_placeholder"][()])) != 0:
                print("[WARN] /uvw/is_placeholder is non-zero; UVW is not used here")
        except Exception as error:
            print("[WARN] cannot read /uvw/is_placeholder:", error)


def read_h5_metadata(h5) -> dict:
    freq_hz = np.asarray(h5["frequency/chan_freq_hz"][()], dtype=np.float64)
    chan_width_hz = np.asarray(h5["frequency/chan_width_hz"][()], dtype=np.float64)
    center_mjd = np.asarray(h5["time/center_mjd"][()], dtype=np.float64)
    exposure_sec = float(np.asarray(h5["time/exposure_sec"][()]))

    return {
        "role_code": read_attr_text(h5, "observation_role_code"),
        "role_name": read_attr_text(h5, "observation_role"),
        "ms_obs_mode": read_attr_text(h5, "ms_obs_mode"),
        "corr_output_mode": read_attr_text(h5, "corr_output_mode"),
        "source_name": read_text_dataset(h5, "field/source_name"),
        "ra_hms": read_text_dataset(h5, "field/phase_center_ra_hms"),
        "dec_dms": read_text_dataset(h5, "field/phase_center_dec_dms"),
        "frame": read_text_dataset(h5, "field/frame"),
        "freq_hz": freq_hz,
        "chan_width_hz": chan_width_hz,
        "center_mjd": center_mjd,
        "exposure_sec": exposure_sec,
    }


def read_signal_arrays(h5) -> dict:
    present = np.asarray(h5["signal/present"][()], dtype=bool).reshape(-1)
    antenna_ids = np.asarray(h5["signal/antenna_id"][()], dtype=np.int64).reshape(-1)
    polarization_ids = np.asarray(
        h5["signal/polarization_id"][()],
        dtype=np.int64,
    ).reshape(-1)
    files = np.asarray(h5["signal/file"][()]).reshape(-1)

    if "signal/input_signal_no" in h5:
        input_signal_no = np.asarray(
            h5["signal/input_signal_no"][()],
            dtype=np.int64,
        ).reshape(-1)
        if input_signal_no.size != present.size:
            input_signal_no = np.arange(1, present.size + 1, dtype=np.int64)
    else:
        input_signal_no = np.arange(1, present.size + 1, dtype=np.int64)

    return {
        "present": present,
        "antenna_id": antenna_ids,
        "polarization_id": polarization_ids,
        "file": files,
        "input_signal_no": input_signal_no,
    }


def signal_index_to_label(h5, signal_index: int) -> str:
    arrays = read_signal_arrays(h5)
    index = int(signal_index)
    if index < 0 or index >= arrays["present"].size:
        return "signal{0}".format(index)

    ant = int(arrays["antenna_id"][index])
    pol = pol_id_to_name(int(arrays["polarization_id"][index]))
    return "{0}{1}".format(ant, pol)


def available_signal_lines(h5, present_only: bool = True) -> list[str]:
    arrays = read_signal_arrays(h5)
    lines = []

    for index in range(arrays["present"].size):
        present = bool(arrays["present"][index])
        if present_only and not present:
            continue

        ant = int(arrays["antenna_id"][index])
        pol_id = int(arrays["polarization_id"][index])
        pol_name = pol_id_to_name(pol_id)
        file_text = as_text(arrays["file"][index])
        input_no = int(arrays["input_signal_no"][index])
        present_text = "yes" if present else "no"
        lines.append(
            "signal {0:02d}: input_signal_no={1} ant{2} {3} "
            "present={4} file={5}".format(
                index,
                input_no,
                ant,
                pol_name,
                present_text,
                file_text,
            )
        )

    return lines


def list_available_signals(h5) -> None:
    arrays = read_signal_arrays(h5)

    print("========== HDF5 SIGNALS ==========")
    print(
        "{0:<13} {1:<16} {2:<8} {3:<4} {4:<7} {5}".format(
            "signal_index",
            "input_signal_no",
            "antenna",
            "pol",
            "present",
            "file",
        )
    )

    for index in range(arrays["present"].size):
        print(
            "{0:<13d} {1:<16d} {2:<8d} {3:<4} {4:<7} {5}".format(
                index,
                int(arrays["input_signal_no"][index]),
                int(arrays["antenna_id"][index]),
                pol_id_to_name(int(arrays["polarization_id"][index])),
                "yes" if bool(arrays["present"][index]) else "no",
                as_text(arrays["file"][index]),
            )
        )

    print("==================================")


def list_available_baselines(h5, max_list: int = 50) -> None:
    arrays = read_signal_arrays(h5)
    signal_pairs = np.asarray(h5["baseline/signal_pairs"][()], dtype=np.int64)
    rows = []

    for baseline_index, pair in enumerate(signal_pairs):
        signal_i = int(pair[0])
        signal_j = int(pair[1])

        if not arrays["present"][signal_i] or not arrays["present"][signal_j]:
            continue

        label_i = signal_index_to_label(h5, signal_i)
        label_j = signal_index_to_label(h5, signal_j)
        rows.append(
            (
                int(baseline_index),
                signal_i,
                signal_j,
                "{0} x conj({1})".format(label_i, label_j),
            )
        )

    print("========== HDF5 BASELINES ==========")
    print(
        "{0:<15} {1:<8} {2:<8} {3}".format(
            "baseline_index",
            "signal_i",
            "signal_j",
            "meaning",
        )
    )

    for row in rows[:max_list]:
        print("{0:<15d} {1:<8d} {2:<8d} {3}".format(row[0], row[1], row[2], row[3]))

    print(
        "printed {0} of {1} plottable baselines".format(
            min(len(rows), max_list),
            len(rows),
        )
    )
    if len(rows) > max_list:
        print("Use --max-list {0} to print more.".format(len(rows)))
    print("====================================")


def find_signal_index(h5, antenna_id: int, pol_name: str) -> int:
    arrays = read_signal_arrays(h5)
    pol_id = pol_name_to_id(pol_name)

    mask = (
        arrays["present"]
        & (arrays["antenna_id"] == int(antenna_id))
        & (arrays["polarization_id"] == int(pol_id))
    )
    indices = np.nonzero(mask)[0]

    if indices.size == 1:
        return int(indices[0])

    if indices.size == 0:
        lines = available_signal_lines(h5, present_only=True)
        available = "\n".join(lines) if lines else "(no present signals)"
        raise ValueError(
            "requested signal not found: ant{0}{1}\nAvailable present signals:\n{2}".format(
                antenna_id,
                pol_name,
                available,
            )
        )

    raise ValueError(
        "HDF5 signal metadata is ambiguous for ant{0}{1}: indices={2}".format(
            antenna_id,
            pol_name,
            indices.tolist(),
        )
    )


def build_baseline_pair_lookup(h5) -> dict[tuple[int, int], int]:
    signal_pairs = np.asarray(h5["baseline/signal_pairs"][()], dtype=np.int64)
    lookup = {}

    for baseline_index, pair in enumerate(signal_pairs):
        if np.asarray(pair).shape[0] != 2:
            raise ValueError(
                "bad signal pair at baseline_index={0}: {1}".format(
                    baseline_index,
                    pair,
                )
            )

        key = (int(pair[0]), int(pair[1]))
        if key in lookup:
            raise ValueError("duplicate baseline signal pair: {0}".format(key))

        lookup[key] = int(baseline_index)

    return lookup


def find_baseline_for_signal_pair(h5, signal_a: int, signal_b: int) -> tuple[int, bool]:
    lookup = build_baseline_pair_lookup(h5)

    forward_key = (int(signal_a), int(signal_b))
    if forward_key in lookup:
        return lookup[forward_key], False

    reverse_key = (int(signal_b), int(signal_a))
    if reverse_key in lookup:
        return lookup[reverse_key], True

    label_a = signal_index_to_label(h5, signal_a)
    label_b = signal_index_to_label(h5, signal_b)
    raise ValueError(
        "baseline not found for {0} x conj({1}) or reverse pair".format(
            label_a,
            label_b,
        )
    )


def extract_visibility_from_h5(h5, ant_a: int, pol_a: str, ant_b: int, pol_b: str):
    signal_a = find_signal_index(h5, ant_a, pol_a)
    signal_b = find_signal_index(h5, ant_b, pol_b)
    baseline_index, need_conjugate = find_baseline_for_signal_pair(
        h5,
        signal_a,
        signal_b,
    )

    vis_matrix = np.asarray(h5["vis"][:, baseline_index, :])

    if need_conjugate:
        vis_matrix = np.conjugate(vis_matrix)

    detail = {
        "baseline_index": int(baseline_index),
        "signal_a": int(signal_a),
        "signal_b": int(signal_b),
        "need_conjugate": bool(need_conjugate),
    }

    return vis_matrix.astype(np.complex64, copy=False), detail


def mjd_to_unix_ms(mjd: float) -> int:
    unix_seconds = (float(mjd) - MJD_UNIX_EPOCH) * 86400.0
    return int(np.rint(unix_seconds * 1000.0))


def unix_ms_to_beijing_text(unix_ms: int) -> str:
    seconds, milliseconds = divmod(int(unix_ms), 1000)
    dt_utc = datetime.fromtimestamp(seconds, tz=timezone.utc)
    dt_bjt = dt_utc.astimezone(BEIJING_TZ)
    return dt_bjt.strftime("%Y-%m-%d %H:%M:%S") + ".{0:03d}".format(milliseconds)


def get_h5_time_edges_mjd(center_mjd, exposure_sec: float) -> tuple[float, float]:
    center_mjd = np.asarray(center_mjd, dtype=np.float64)
    exposure_day = float(exposure_sec) / 86400.0
    start_edge_mjd = float(center_mjd[0]) - 0.5 * exposure_day
    end_edge_mjd = float(center_mjd[-1]) + 0.5 * exposure_day
    return start_edge_mjd, end_edge_mjd


def get_h5_time_range_text(center_mjd, exposure_sec: float) -> tuple[str, str, str]:
    start_edge_mjd, end_edge_mjd = get_h5_time_edges_mjd(center_mjd, exposure_sec)
    start_text = unix_ms_to_beijing_text(mjd_to_unix_ms(start_edge_mjd))
    end_text = unix_ms_to_beijing_text(mjd_to_unix_ms(end_edge_mjd))
    range_text = "{0} - {1} BJT".format(start_text, end_text)
    return start_text, end_text, range_text


def get_time_extent_sec_from_h5(center_mjd, exposure_sec: float) -> list[float]:
    start_edge_mjd, end_edge_mjd = get_h5_time_edges_mjd(center_mjd, exposure_sec)
    duration_sec = (end_edge_mjd - start_edge_mjd) * 86400.0
    if duration_sec < 0.0:
        duration_sec = 0.0
    return [0.0, float(duration_sec)]


def get_time_sec_from_start(center_mjd, exposure_sec: float) -> np.ndarray:
    center_mjd = np.asarray(center_mjd, dtype=np.float64)
    start_edge_mjd, _end_edge_mjd = get_h5_time_edges_mjd(center_mjd, exposure_sec)
    return (center_mjd - start_edge_mjd) * 86400.0


def get_frequency_extent_mhz(freq_hz, chan_width_hz) -> list[float]:
    freq_mhz = np.asarray(freq_hz, dtype=float) / 1e6

    if freq_mhz.size == 1:
        width_mhz = 1.0
        if chan_width_hz is not None and len(chan_width_hz) > 0:
            if float(chan_width_hz[0]) != 0.0:
                width_mhz = abs(float(chan_width_hz[0])) / 1e6

        return [
            float(freq_mhz[0] - 0.5 * width_mhz),
            float(freq_mhz[0] + 0.5 * width_mhz),
        ]

    if chan_width_hz is not None and len(chan_width_hz) == freq_mhz.size:
        first_width_mhz = abs(float(chan_width_hz[0])) / 1e6
        last_width_mhz = abs(float(chan_width_hz[-1])) / 1e6
    else:
        first_width_mhz = abs(float(freq_mhz[1] - freq_mhz[0]))
        last_width_mhz = abs(float(freq_mhz[-1] - freq_mhz[-2]))

    if freq_mhz[-1] >= freq_mhz[0]:
        x0 = freq_mhz[0] - 0.5 * first_width_mhz
        x1 = freq_mhz[-1] + 0.5 * last_width_mhz
    else:
        x0 = freq_mhz[0] + 0.5 * first_width_mhz
        x1 = freq_mhz[-1] - 0.5 * last_width_mhz

    return [float(x0), float(x1)]


def prepare_frequency_axis_for_plot(freq_hz, chan_width_hz, matrix_time_freq):
    """
    Sort frequency ascending and reorder matrix columns to match.

    Input matrix convention:
        matrix_time_freq.shape = (Ntime, Nfreq)
    """
    freq = np.asarray(freq_hz, dtype=float).reshape(-1)

    if matrix_time_freq.ndim != 2:
        raise RuntimeError(
            "plot matrix must be 2D, got shape={0}".format(matrix_time_freq.shape)
        )

    if matrix_time_freq.shape[1] != freq.size:
        raise RuntimeError(
            "frequency axis mismatch: matrix shape={0}, freq size={1}".format(
                matrix_time_freq.shape,
                freq.size,
            )
        )

    if chan_width_hz is None:
        chan_width = None
    else:
        chan_width = np.asarray(chan_width_hz, dtype=float).reshape(-1)
        if chan_width.size != freq.size:
            chan_width = None

    order = np.argsort(freq)

    if np.array_equal(order, np.arange(freq.size)):
        return freq, chan_width, matrix_time_freq

    matrix_sorted = matrix_time_freq[:, order]
    if chan_width is None:
        chan_width_sorted = None
    else:
        chan_width_sorted = chan_width[order]

    return freq[order], chan_width_sorted, matrix_sorted


def finite_min_max(values) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    if not np.any(valid):
        return float("nan"), float("nan")
    return float(np.min(values[valid])), float(np.max(values[valid]))


def make_plot_matrix(mode: str, vis_matrix, zero_as_nan: bool, amp_threshold: float):
    amp_matrix = np.abs(vis_matrix).astype(float)

    if mode == "phase":
        plot_matrix = np.angle(vis_matrix).astype(float)
        colorbar_label = "Phase (rad)"
        title_prefix = "Phase waterfall"
        vmin = -np.pi
        vmax = np.pi
    elif mode == "amp":
        plot_matrix = amp_matrix.copy()
        colorbar_label = "Amplitude"
        title_prefix = "Amplitude waterfall"
        vmin = None
        vmax = None
    elif mode == "real":
        plot_matrix = np.real(vis_matrix).astype(float)
        colorbar_label = "Real"
        title_prefix = "Real waterfall"
        vmin = None
        vmax = None
    elif mode == "imag":
        plot_matrix = np.imag(vis_matrix).astype(float)
        colorbar_label = "Imaginary"
        title_prefix = "Imaginary waterfall"
        vmin = None
        vmax = None
    else:
        raise ValueError("bad mode: {0}".format(mode))

    if zero_as_nan:
        plot_matrix = plot_matrix.copy()
        plot_matrix[amp_matrix <= float(amp_threshold)] = np.nan

    return plot_matrix, colorbar_label, title_prefix, vmin, vmax


def print_input_summary(args, metadata, ant_a, pol_a, ant_b, pol_b, detail, vis_matrix):
    freq_hz = metadata["freq_hz"]
    center_mjd = metadata["center_mjd"]
    exposure_sec = metadata["exposure_sec"]
    phase_min, phase_max = finite_min_max(np.angle(vis_matrix))
    amp_min, amp_max = finite_min_max(np.abs(vis_matrix))

    print("========== HDF5 WATERFALL INPUT ==========")
    print("HDF5 file              :", args.h5_file)
    print("role code              :", metadata["role_code"])
    print("observation role       :", metadata["role_name"])
    print("MS OBS_MODE            :", metadata["ms_obs_mode"])
    print("corr output mode       :", metadata["corr_output_mode"])
    print("source name            :", metadata["source_name"])
    print("phase center RA        :", metadata["ra_hms"])
    print("phase center Dec       :", metadata["dec_dms"])
    print("phase center frame     :", metadata["frame"])
    print("frequency first Hz     :", float(freq_hz[0]))
    print("frequency last Hz      :", float(freq_hz[-1]))
    print("frequency min Hz       :", float(np.min(freq_hz)))
    print("frequency max Hz       :", float(np.max(freq_hz)))
    print("Ntime                  :", int(center_mjd.size))
    print("Nfreq                  :", int(freq_hz.size))
    print("exposure sec           :", float(exposure_sec))
    print("requested signal1      :", args.signal1)
    print("requested signal2      :", args.signal2)
    print(
        "meaning                : ant{0}{1} x conj(ant{2}{3})".format(
            ant_a,
            pol_a,
            ant_b,
            pol_b,
        )
    )
    print("signal index a         :", detail["signal_a"])
    print("signal index b         :", detail["signal_b"])
    print("baseline index         :", detail["baseline_index"])
    print("conjugate applied      :", detail["need_conjugate"])
    print("vis matrix shape       :", vis_matrix.shape)
    print("phase min/max rad      :", phase_min, phase_max)
    print("amplitude min/max      :", amp_min, amp_max)
    print("output png             :", args.out_png)
    print("==========================================")


def save_outputs(
    args,
    metadata,
    ant_a: int,
    pol_a: str,
    ant_b: int,
    pol_b: str,
    vis_matrix,
    detail: dict,
) -> None:
    out_png = os.path.abspath(args.out_png)
    out_dir = os.path.dirname(out_png)

    if out_dir != "" and not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    mode = str(args.mode).lower()
    freq_hz = metadata["freq_hz"]
    chan_width_hz = metadata["chan_width_hz"]
    center_mjd = metadata["center_mjd"]
    exposure_sec = metadata["exposure_sec"]

    plot_matrix, colorbar_label, title_prefix, vmin, vmax = make_plot_matrix(
        mode,
        vis_matrix,
        args.zero_as_nan,
        args.amp_threshold,
    )

    freq_hz_plot, chan_width_hz_plot, plot_matrix_plot = prepare_frequency_axis_for_plot(
        freq_hz,
        chan_width_hz,
        plot_matrix,
    )
    time_extent = get_time_extent_sec_from_h5(center_mjd, exposure_sec)
    freq_extent = get_frequency_extent_mhz(freq_hz_plot, chan_width_hz_plot)
    extent = [time_extent[0], time_extent[1], freq_extent[0], freq_extent[1]]
    image_matrix = plot_matrix_plot.T

    _start_bjt, _end_bjt, bjt_range_text = get_h5_time_range_text(
        center_mjd,
        exposure_sec,
    )
    h5_base = os.path.basename(args.h5_file)
    signal_meaning = "ant{0}{1} x conj(ant{2}{3})".format(
        ant_a,
        pol_a,
        ant_b,
        pol_b,
    )

    plt.figure(figsize=(12, 6))
    plt.imshow(
        image_matrix,
        aspect="auto",
        origin="lower",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    plt.colorbar(label=colorbar_label)
    plt.xlabel("Time from HDF5 start (s)")
    plt.ylabel("Frequency (MHz)")
    plt.title(
        "{0} | {1} | role={2} source={3}\n{4} {5}: {6} | BJT: {7}".format(
            title_prefix,
            h5_base,
            metadata["role_code"],
            metadata["source_name"],
            args.signal1,
            args.signal2,
            signal_meaning,
            bjt_range_text,
        )
    )
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    print("[OK] PNG saved:", out_png)

    base, _ext = os.path.splitext(out_png)
    phase_matrix = np.angle(vis_matrix).astype(float)
    amp_matrix = np.abs(vis_matrix).astype(float)

    if args.zero_as_nan:
        zero_mask = amp_matrix <= float(args.amp_threshold)
        phase_matrix = phase_matrix.copy()
        phase_matrix[zero_mask] = np.nan

    if args.save_npy:
        complex_path = base + "_complex_visibility.npy"
        phase_path = base + "_phase_rad.npy"
        amp_path = base + "_amplitude.npy"
        freq_path = base + "_freq_hz.npy"
        time_center_path = base + "_time_center_mjd.npy"
        time_sec_path = base + "_time_sec_from_start.npy"

        np.save(complex_path, vis_matrix)
        np.save(phase_path, phase_matrix)
        np.save(amp_path, amp_matrix)
        np.save(freq_path, freq_hz)
        np.save(time_center_path, center_mjd)
        np.save(time_sec_path, get_time_sec_from_start(center_mjd, exposure_sec))

        print("[OK] complex visibility saved:", complex_path)
        print("[OK] phase rad saved:", phase_path)
        print("[OK] amplitude saved:", amp_path)
        print("[OK] frequency axis saved:", freq_path)
        print("[OK] time center MJD saved:", time_center_path)
        print("[OK] time axis saved:", time_sec_path)

    if args.save_txt:
        txt_path = base + "_summary.txt"
        start_bjt, end_bjt, bjt_range_text = get_h5_time_range_text(
            center_mjd,
            exposure_sec,
        )
        phase_min, phase_max = finite_min_max(phase_matrix)
        amp_min, amp_max = finite_min_max(amp_matrix)

        with open(txt_path, "w", encoding="utf-8") as handle:
            handle.write("HDF5 waterfall extraction summary\n")
            handle.write("=================================\n\n")
            handle.write("HDF5 file: {0}\n".format(args.h5_file))
            handle.write("Role code: {0}\n".format(metadata["role_code"]))
            handle.write("Observation role: {0}\n".format(metadata["role_name"]))
            handle.write("MS OBS_MODE: {0}\n".format(metadata["ms_obs_mode"]))
            handle.write("Corr output mode: {0}\n".format(metadata["corr_output_mode"]))
            handle.write("Source name: {0}\n".format(metadata["source_name"]))
            handle.write(
                "Phase center: RA={0} Dec={1} frame={2}\n".format(
                    metadata["ra_hms"],
                    metadata["dec_dms"],
                    metadata["frame"],
                )
            )
            handle.write("Signal 1: {0}\n".format(args.signal1))
            handle.write("Signal 2: {0}\n".format(args.signal2))
            handle.write("Meaning: {0}\n".format(signal_meaning))
            handle.write("Signal index a: {0}\n".format(detail["signal_a"]))
            handle.write("Signal index b: {0}\n".format(detail["signal_b"]))
            handle.write("Baseline index: {0}\n".format(detail["baseline_index"]))
            handle.write("Conjugate applied: {0}\n".format(detail["need_conjugate"]))
            handle.write("Mode: {0}\n".format(args.mode))
            handle.write("Zero as NaN: {0}\n".format(args.zero_as_nan))
            handle.write("Amplitude threshold: {0}\n".format(args.amp_threshold))
            handle.write("Matrix shape: {0}\n".format(vis_matrix.shape))
            handle.write("Matrix convention: rows=time, columns=frequency\n")
            handle.write("Plot orientation: x-axis=time, y-axis=frequency\n")
            handle.write("Frequency first Hz: {0}\n".format(float(freq_hz[0])))
            handle.write("Frequency last Hz: {0}\n".format(float(freq_hz[-1])))
            handle.write("Frequency min Hz: {0}\n".format(float(np.min(freq_hz))))
            handle.write("Frequency max Hz: {0}\n".format(float(np.max(freq_hz))))
            handle.write("BJT start: {0}\n".format(start_bjt))
            handle.write("BJT end: {0}\n".format(end_bjt))
            handle.write("BJT time range: {0}\n".format(bjt_range_text))
            handle.write("Exposure sec: {0}\n".format(float(exposure_sec)))
            handle.write("Phase min rad: {0}\n".format(phase_min))
            handle.write("Phase max rad: {0}\n".format(phase_max))
            handle.write("Amplitude min: {0}\n".format(amp_min))
            handle.write("Amplitude max: {0}\n".format(amp_max))
            handle.write("Output PNG: {0}\n".format(out_png))

        print("[OK] summary saved:", txt_path)


def main(argv=None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    args = parse_args(argv)
    require_h5py()

    if not os.path.isfile(args.h5_file):
        raise RuntimeError("HDF5 file not found: {0}".format(args.h5_file))

    with h5py.File(args.h5_file, "r") as h5:
        validate_h5_for_plot(h5)

        if args.list_signals:
            list_available_signals(h5)

        if args.list_baselines:
            list_available_baselines(h5, args.max_list)

        if args.list_signals or args.list_baselines:
            return 0

        ant_a, pol_a = parse_signal_text(args.signal1)
        ant_b, pol_b = parse_signal_text(args.signal2)
        metadata = read_h5_metadata(h5)
        vis_matrix, detail = extract_visibility_from_h5(
            h5,
            ant_a,
            pol_a,
            ant_b,
            pol_b,
        )

        print_input_summary(
            args,
            metadata,
            ant_a,
            pol_a,
            ant_b,
            pol_b,
            detail,
            vis_matrix,
        )
        save_outputs(
            args,
            metadata,
            ant_a,
            pol_a,
            ant_b,
            pol_b,
            vis_matrix,
            detail,
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print("[ERROR] {0}".format(error), file=sys.stderr)
        raise SystemExit(1)
