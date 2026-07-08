#!/usr/bin/env python3
"""
Convert an MS-ready HDF5 file into a CASA MeasurementSet.
"""

from __future__ import annotations

import argparse
import inspect
import os
import platform
import shutil
import sys
import traceback
from typing import Any

import numpy as np

if not hasattr(np, "typeDict"):
    np.typeDict = np.sctypeDict
if not hasattr(np, "asscalar"):
    np.asscalar = lambda value: np.asarray(value).item()

try:
    import h5py
except Exception as error:  # pragma: no cover
    h5py = None
    H5PY_IMPORT_ERROR = error
else:
    H5PY_IMPORT_ERROR = None

try:
    import pyuvdata as pyuvdata_module
    from pyuvdata import UVData
except Exception as error:  # pragma: no cover
    pyuvdata_module = None
    UVData = None
    PYUVDATA_IMPORT_ERROR = error
else:
    PYUVDATA_IMPORT_ERROR = None

try:
    from pyuvdata import Telescope
except Exception:  # pragma: no cover
    Telescope = None

try:
    import casacore.tables as casacore_tables
except Exception as error:  # pragma: no cover
    casacore_tables = None
    CASACORE_IMPORT_ERROR = error
else:
    CASACORE_IMPORT_ERROR = None

try:
    import astropy
    from astropy.coordinates import EarthLocation
    import astropy.units as u
except Exception as error:  # pragma: no cover
    astropy = None
    EarthLocation = None
    u = None
    ASTROPY_IMPORT_ERROR = error
else:
    ASTROPY_IMPORT_ERROR = None

# =========================
# CASA telescope / array names
# =========================
#
# CASA_TELESCOPE_NAME is written into:
#
#     MS/OBSERVATION/TELESCOPE_NAME
#
# CASA 5.x listobs checks this name against CASA's known observatory table.
# The user has registered CARRY_1 in CASA, so use CARRY_1 here.
#
# ARRAY_NAME and INSTRUMENT_NAME record the local array identity.
CASA_TELESCOPE_NAME = "CARRY_1"
ARRAY_NAME = "CARRY_1"
INSTRUMENT_NAME = "CARRY_1"

# Keep the in-memory DATA polarization axis in the same order that
# CASA MeasurementSet POLARIZATION/CORR_TYPE will advertise.
#
# CASA/Stokes enum:
#   XX = 9
#   XY = 10
#   YX = 11
#   YY = 12
#
# pyuvdata/AIPS polarization numbers:
#   XX = -5
#   XY = -7
#   YX = -8
#   YY = -6
POL_ORDER = ["XX", "XY", "YX", "YY"]
POL_NUMS = np.array([-5, -7, -8, -6], dtype=np.int64)
MS_CORR_TYPES = np.array([9, 10, 11, 12], dtype=np.int64)
REQUIRED_PATHS = [
    "vis",
    "baseline/signal_pairs",
    "signal/present",
    "signal/antenna_id",
    "signal/polarization_id",
    "time/center_mjd",
    "time/exposure_sec",
    "frequency/chan_freq_hz",
    "frequency/chan_width_hz",
    "antenna/id",
    "antenna/name",
    "antenna/station",
    "antenna/position_itrf_m",
    "antenna/dish_diameter_m",
    "antenna/used_in_input",
    "antenna/position_is_placeholder_by_antenna",
    "field/source_name",
    "field/phase_center_ra_rad",
    "field/phase_center_dec_rad",
    "field/phase_center_ra_hms",
    "field/phase_center_dec_dms",
    "field/frame",
    "field/is_placeholder",
    "ms_rows/time_index",
    "ms_rows/signal_baseline_index",
    "ms_rows/antenna1",
    "ms_rows/antenna2",
    "ms_rows/pol_i",
    "ms_rows/pol_j",
    "ms_rows/corr_name",
    "ms_rows/row_has_missing_signal",
    "ms_rows/row_is_same_antenna",
    "ms_rows/row_is_cross_antenna",
    "uvw/uvw_m",
    "uvw/is_placeholder",
]


def require_h5py() -> None:
    if h5py is None:
        raise RuntimeError("h5py is required") from H5PY_IMPORT_ERROR


def require_pyuvdata() -> None:
    if UVData is None:
        raise RuntimeError("pyuvdata is required") from PYUVDATA_IMPORT_ERROR


def require_casacore() -> None:
    if casacore_tables is None:
        raise RuntimeError(
            "python-casacore is required for writing MeasurementSet"
        ) from CASACORE_IMPORT_ERROR


def require_astropy() -> None:
    if EarthLocation is None or u is None:
        raise RuntimeError(
            "astropy is required for constructing telescope locations"
        ) from ASTROPY_IMPORT_ERROR


def warn(message: str) -> None:
    print(f"[WARN] {message}", file=sys.stderr)


def as_text(value: Any) -> str:
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
    return [as_text(item) for item in array]


def format_bytes(n_bytes: int) -> str:
    gb = n_bytes / 1_000_000_000
    gib = n_bytes / (1024 ** 3)
    return f"{gb:.3f} GB ({gib:.3f} GiB)"


def print_runtime_versions() -> None:
    print("\n========== RUNTIME ==========")
    print("python executable :", sys.executable)
    print("python version    :", sys.version.replace("\n", " "))
    print("platform          :", platform.platform())
    print("numpy             :", np.__version__)
    if h5py is not None:
        print("h5py              :", h5py.__version__)
    else:
        print("h5py              : unavailable")
    if astropy is not None:
        print("astropy           :", astropy.__version__)
    else:
        print("astropy           : unavailable")
    if pyuvdata_module is not None:
        print("pyuvdata          :", pyuvdata_module.__version__)
    else:
        print("pyuvdata          : unavailable")
    if casacore_tables is not None:
        print("casacore.tables   : available")
    else:
        print("casacore.tables   : unavailable")
    if UVData is not None:
        try:
            print("UVData.new sig    :", inspect.signature(UVData.new))
        except Exception as error:
            print("UVData.new sig    : unavailable", error)
    print("=============================")


def build_signal_lookup(h5: h5py.File) -> dict[tuple[int, int], int]:
    signal_present = np.asarray(h5["signal/present"][()], dtype=bool)
    signal_ant_ids = np.asarray(h5["signal/antenna_id"][()], dtype=np.int64)
    signal_pols = np.asarray(h5["signal/polarization_id"][()], dtype=np.int64)

    if signal_ant_ids.shape != signal_present.shape:
        raise ValueError("signal/antenna_id shape mismatch")
    if signal_pols.shape != signal_present.shape:
        raise ValueError("signal/polarization_id shape mismatch")

    lookup: dict[tuple[int, int], int] = {}
    for signal_index in range(signal_present.size):
        if not signal_present[signal_index]:
            continue

        key = (int(signal_ant_ids[signal_index]), int(signal_pols[signal_index]))
        if key in lookup:
            raise ValueError(
                f"duplicate signal mapping for antenna={key[0]}, pol={key[1]}"
            )
        lookup[key] = int(signal_index)

    return lookup


def corr_to_ant_pol_pair(
    ant1: int,
    ant2: int,
    corr: str,
) -> tuple[tuple[int, int], tuple[int, int]]:
    if len(corr) != 2:
        raise ValueError(f"bad corr name: {corr}")

    pol_map = {"X": 0, "Y": 1}
    pol_a = pol_map.get(corr[0])
    pol_b = pol_map.get(corr[1])

    if pol_a is None or pol_b is None:
        raise ValueError(f"unsupported corr name: {corr}")

    return (int(ant1), pol_a), (int(ant2), pol_b)


def ensure_required_paths(h5: h5py.File) -> None:
    for path in REQUIRED_PATHS:
        if path not in h5:
            raise ValueError(f"missing required HDF5 path: /{path}")


def validate_hdf5_input(h5: h5py.File) -> None:
    ensure_required_paths(h5)

    if int(read_scalar(h5, "field/is_placeholder")) != 0:
        raise ValueError("/field is still placeholder")

    if int(read_scalar(h5, "uvw/is_placeholder")) != 0:
        raise ValueError("/uvw is still placeholder")

    ra_rad = float(read_scalar(h5, "field/phase_center_ra_rad"))
    dec_rad = float(read_scalar(h5, "field/phase_center_dec_rad"))

    if not np.isfinite(ra_rad):
        raise ValueError("bad field RA: not finite")
    if not np.isfinite(dec_rad):
        raise ValueError("bad field Dec: not finite")
    if dec_rad < -0.5 * np.pi or dec_rad > 0.5 * np.pi:
        raise ValueError(f"bad field Dec out of range: {dec_rad}")

    freq = np.asarray(h5["frequency/chan_freq_hz"][()], dtype=np.float64)
    chan_width = np.asarray(h5["frequency/chan_width_hz"][()], dtype=np.float64)

    if np.any(~np.isfinite(freq)):
        raise ValueError("frequency/chan_freq_hz contains non-finite values")
    if np.any(~np.isfinite(chan_width)):
        raise ValueError("frequency/chan_width_hz contains non-finite values")
    if np.any(chan_width <= 0.0):
        raise ValueError("frequency/chan_width_hz must be positive")

    signal_present = np.asarray(h5["signal/present"][()], dtype=bool)
    signal_ant = np.asarray(h5["signal/antenna_id"][()], dtype=np.int64)
    signal_pol = np.asarray(h5["signal/polarization_id"][()], dtype=np.int64)

    if signal_ant.shape != signal_present.shape:
        raise ValueError("signal/antenna_id shape mismatch")
    if signal_pol.shape != signal_present.shape:
        raise ValueError("signal/polarization_id shape mismatch")

    if np.any(h5["ms_rows/row_has_missing_signal"][()]):
        raise ValueError("ms_rows contains missing signals")

    n_times = h5["time/center_mjd"].shape[0]
    n_baselines = h5["baseline/signal_pairs"].shape[0]
    n_freqs = h5["frequency/chan_freq_hz"].shape[0]
    vis_shape = h5["vis"].shape

    if vis_shape != (n_times, n_baselines, n_freqs):
        raise ValueError(
            f"/vis shape mismatch: {vis_shape} != {(n_times, n_baselines, n_freqs)}"
        )

    n_ms_rows = int(h5["ms_rows"].attrs["n_ms_rows"])
    uvw_shape = h5["uvw/uvw_m"].shape

    if uvw_shape != (n_ms_rows, 3):
        raise ValueError(f"/uvw/uvw_m shape mismatch: {uvw_shape} != {(n_ms_rows, 3)}")

    for path in [
        "ms_rows/time_index",
        "ms_rows/signal_baseline_index",
        "ms_rows/antenna1",
        "ms_rows/antenna2",
        "ms_rows/pol_i",
        "ms_rows/pol_j",
        "ms_rows/corr_name",
        "ms_rows/row_has_missing_signal",
        "ms_rows/row_is_same_antenna",
        "ms_rows/row_is_cross_antenna",
    ]:
        if h5[path].shape != (n_ms_rows,):
            raise ValueError(f"/{path} length mismatch")

    uvw = np.asarray(h5["uvw/uvw_m"][()], dtype=np.float64)
    ant1 = np.asarray(h5["ms_rows/antenna1"][()], dtype=np.int64)
    ant2 = np.asarray(h5["ms_rows/antenna2"][()], dtype=np.int64)

    same = ant1 == ant2
    if np.any(same):
        max_same = float(np.max(np.abs(uvw[same])))
        if max_same > 1e-6:
            raise ValueError(f"same-antenna UVW should be zero, max={max_same}")

    cross = ant1 != ant2
    if np.any(cross):
        max_cross = float(np.max(np.abs(uvw[cross])))
        if max_cross <= 0.0:
            raise ValueError("cross-antenna UVW is all zero")


def build_baseline_pair_lookup(signal_pairs: np.ndarray) -> dict[tuple[int, int], int]:
    lookup: dict[tuple[int, int], int] = {}

    for baseline_index, pair in enumerate(np.asarray(signal_pairs, dtype=np.int64)):
        if pair.shape[0] != 2:
            raise ValueError(
                f"bad signal pair at baseline_index={baseline_index}: {pair}"
            )

        key = (int(pair[0]), int(pair[1]))
        if key in lookup:
            raise ValueError(f"duplicate baseline signal pair: {key}")
        lookup[key] = int(baseline_index)

    return lookup


def get_visibility_for_signal_pair(
    vis: h5py.Dataset,
    time_index: int,
    baseline_lookup: dict[tuple[int, int], int],
    signal_a: int,
    signal_b: int,
) -> tuple[np.ndarray | None, bool]:
    exact_key = (int(signal_a), int(signal_b))
    baseline_index = baseline_lookup.get(exact_key)
    if baseline_index is not None:
        return np.asarray(vis[time_index, baseline_index, :]), False

    reverse_key = (int(signal_b), int(signal_a))
    baseline_index = baseline_lookup.get(reverse_key)
    if baseline_index is not None:
        return np.conj(np.asarray(vis[time_index, baseline_index, :])), True

    return None, False


def build_used_antenna_metadata(h5: h5py.File) -> dict[str, Any]:
    antenna_ids = np.asarray(h5["antenna/id"][()], dtype=np.int64)
    antenna_names = decode_string_array(h5["antenna/name"][()])
    station_names = decode_string_array(h5["antenna/station"][()])
    antenna_positions_abs = np.asarray(
        h5["antenna/position_itrf_m"][()],
        dtype=np.float64,
    )
    dish_diameter_m = np.asarray(
        h5["antenna/dish_diameter_m"][()],
        dtype=np.float64,
    )
    used_mask_from_antenna = np.asarray(h5["antenna/used_in_input"][()], dtype=bool)
    placeholder_mask = np.asarray(
        h5["antenna/position_is_placeholder_by_antenna"][()],
        dtype=bool,
    )
    signal_present = np.asarray(h5["signal/present"][()], dtype=bool)
    signal_ant_ids = np.asarray(h5["signal/antenna_id"][()], dtype=np.int64)
    used_from_signal = set(int(item) for item in signal_ant_ids[signal_present])
    used_from_antenna = set(
        int(item) for item in antenna_ids[used_mask_from_antenna]
    )

    if used_from_signal != used_from_antenna:
        warn(
            "antenna/used_in_input does not match signal/present; "
            "using the union of both."
        )

    used_id_union = used_from_signal | used_from_antenna
    used_mask = np.array(
        [int(ant_id) in used_id_union for ant_id in antenna_ids],
        dtype=bool,
    )

    if np.any(used_mask & placeholder_mask):
        bad = antenna_ids[used_mask & placeholder_mask]
        raise ValueError(
            "used antennas still have placeholder positions: "
            + ", ".join([str(int(item)) for item in bad])
        )

    used_ids = antenna_ids[used_mask]
    used_names = [antenna_names[idx] for idx, flag in enumerate(used_mask) if flag]
    used_stations = [station_names[idx] for idx, flag in enumerate(used_mask) if flag]
    used_positions_abs = antenna_positions_abs[used_mask]
    used_dish_m = dish_diameter_m[used_mask]

    if used_ids.size == 0:
        raise ValueError("no used antennas found in HDF5")

    array_center = np.mean(used_positions_abs, axis=0)
    used_positions_rel = used_positions_abs - array_center

    return {
        "antenna_ids": used_ids.astype(np.int64),
        "antenna_names": np.array(used_names, dtype=object),
        "station_names": np.array(used_stations, dtype=object),
        "antenna_positions_abs_m": used_positions_abs.astype(np.float64),
        "antenna_positions_rel_m": used_positions_rel.astype(np.float64),
        "array_center_m": array_center.astype(np.float64),
        "dish_diameter_m": used_dish_m.astype(np.float64),
    }


def build_physical_blt_axis(h5: h5py.File, include_autos: bool = False) -> dict[str, Any]:
    used_meta = build_used_antenna_metadata(h5)
    used_ants = [int(item) for item in used_meta["antenna_ids"]]
    antpairs: list[tuple[int, int]] = []

    for ant1 in used_ants:
        for ant2 in used_ants:
            if include_autos:
                if ant1 <= ant2:
                    antpairs.append((ant1, ant2))
            else:
                if ant1 < ant2:
                    antpairs.append((ant1, ant2))

    center_mjd = np.asarray(h5["time/center_mjd"][()], dtype=np.float64)
    time_jd = center_mjd + 2400000.5
    ant_1_array = []
    ant_2_array = []
    time_array = []
    time_index_array = []

    for ti, jd in enumerate(time_jd):
        for ant1, ant2 in antpairs:
            ant_1_array.append(ant1)
            ant_2_array.append(ant2)
            time_array.append(jd)
            time_index_array.append(ti)

    return {
        "used_ants": np.array(used_ants, dtype=np.int64),
        "antpairs": np.array(antpairs, dtype=np.int64),
        "ant_1_array": np.array(ant_1_array, dtype=np.int64),
        "ant_2_array": np.array(ant_2_array, dtype=np.int64),
        "time_array_jd": np.array(time_array, dtype=np.float64),
        "time_index_array": np.array(time_index_array, dtype=np.int64),
        "n_times": center_mjd.size,
        "used_meta": used_meta,
    }


def estimate_packed_array_memory(h5: h5py.File, include_autos: bool) -> dict[str, Any]:
    used_meta = build_used_antenna_metadata(h5)
    n_used_ant = int(used_meta["antenna_ids"].size)
    n_times = int(h5["time/center_mjd"].shape[0])
    n_freqs = int(h5["frequency/chan_freq_hz"].shape[0])
    n_pols = len(POL_ORDER)

    if include_autos:
        n_phys_bl = n_used_ant * (n_used_ant + 1) // 2
    else:
        n_phys_bl = n_used_ant * (n_used_ant - 1) // 2

    nblts = n_times * n_phys_bl
    data_bytes = nblts * n_freqs * n_pols * np.dtype(np.complex64).itemsize
    flag_bytes = nblts * n_freqs * n_pols * np.dtype(np.bool_).itemsize
    nsample_bytes = nblts * n_freqs * n_pols * np.dtype(np.float32).itemsize
    uvw_bytes = nblts * 3 * np.dtype(np.float64).itemsize
    total_bytes = data_bytes + flag_bytes + nsample_bytes + uvw_bytes

    return {
        "n_used_ant": n_used_ant,
        "n_times": n_times,
        "n_freqs": n_freqs,
        "n_pols": n_pols,
        "n_phys_bl": n_phys_bl,
        "nblts": nblts,
        "data_bytes": data_bytes,
        "flag_bytes": flag_bytes,
        "nsample_bytes": nsample_bytes,
        "uvw_bytes": uvw_bytes,
        "total_bytes": total_bytes,
    }


def estimate_and_check_memory(
    h5: h5py.File,
    include_autos: bool,
    max_memory_gb: float,
) -> None:
    estimate = estimate_packed_array_memory(h5, include_autos=include_autos)
    max_bytes = int(max_memory_gb * 1024 ** 3)

    print("\n========== MEMORY ESTIMATE ==========")
    print("used antennas       :", estimate["n_used_ant"])
    print("n_times             :", estimate["n_times"])
    print("n_freqs             :", estimate["n_freqs"])
    print("n_pols              :", estimate["n_pols"])
    print("physical baselines  :", estimate["n_phys_bl"])
    print("Nblts               :", estimate["nblts"])
    print("data_array bytes    :", format_bytes(estimate["data_bytes"]))
    print("flag_array bytes    :", format_bytes(estimate["flag_bytes"]))
    print("nsample_array bytes :", format_bytes(estimate["nsample_bytes"]))
    print("uvw_array bytes     :", format_bytes(estimate["uvw_bytes"]))
    print("total approx bytes  :", format_bytes(estimate["total_bytes"]))
    print("max memory allowed  :", f"{max_memory_gb:.3f} GiB")
    print("=====================================")

    if estimate["n_used_ant"] < 2 and not include_autos:
        raise ValueError(
            "no cross-antenna baselines to write: "
            "need at least 2 used antennas, or enable auto-correlations"
        )

    if estimate["n_phys_bl"] <= 0:
        raise ValueError("no physical baselines to write")

    if estimate["total_bytes"] > max_bytes:
        raise MemoryError(
            "estimated packed arrays exceed --max-memory-gb: "
            f"{format_bytes(estimate['total_bytes'])} > {max_memory_gb:.3f} GiB"
        )


def build_uvw_lookup(
    h5: h5py.File,
    allow_uvw_warnings: bool = False,
    tolerance: float = 1e-6,
) -> dict[tuple[int, int, int], np.ndarray]:
    row_time = np.asarray(h5["ms_rows/time_index"][()], dtype=np.int64)
    row_ant1 = np.asarray(h5["ms_rows/antenna1"][()], dtype=np.int64)
    row_ant2 = np.asarray(h5["ms_rows/antenna2"][()], dtype=np.int64)
    uvw_rows = np.asarray(h5["uvw/uvw_m"][()], dtype=np.float64)

    grouped: dict[tuple[int, int, int], list[np.ndarray]] = {}

    for row in range(row_time.size):
        key = (int(row_time[row]), int(row_ant1[row]), int(row_ant2[row]))
        grouped.setdefault(key, []).append(uvw_rows[row])

    uvw_lookup: dict[tuple[int, int, int], np.ndarray] = {}

    for key, candidates in grouped.items():
        ref = np.asarray(candidates[0], dtype=np.float64)
        for other in candidates[1:]:
            diff = float(np.max(np.abs(np.asarray(other, dtype=np.float64) - ref)))
            if diff > tolerance:
                message = (
                    f"UVW mismatch for same physical baseline {key}: "
                    f"max diff = {diff}"
                )
                if allow_uvw_warnings:
                    warn(message)
                else:
                    raise ValueError(message)
        uvw_lookup[key] = ref

    return uvw_lookup


def get_scalar_seconds(dataset: h5py.Dataset) -> float:
    value = dataset[()]
    array = np.asarray(value, dtype=np.float64)
    return float(array.reshape(-1)[0])


def normalize_frequency_axis(
    freq_array_hz: np.ndarray,
    channel_width_hz: np.ndarray,
    data_array: np.ndarray,
    flag_array: np.ndarray,
    nsample_array: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool, str]:
    """
    Normalize frequency-dependent arrays to ascending frequency order.

    Parameters
    ----------
    freq_array_hz
        Frequency array in Hz, shape (Nfreqs,).
    channel_width_hz
        Channel width array in Hz, shape (Nfreqs,). Must be positive.
    data_array
        Visibility data array, shape (Nblts, Nfreqs, Npols).
    flag_array
        Flag array, shape (Nblts, Nfreqs, Npols).
    nsample_array
        Nsample array, shape (Nblts, Nfreqs, Npols).

    Returns
    -------
    freq_out
        Frequency array in ascending order.
    chan_width_out
        Channel width array reordered to match freq_out.
    data_out
        Data array reordered along frequency axis.
    flag_out
        Flag array reordered along frequency axis.
    nsample_out
        Nsample array reordered along frequency axis.
    reordered
        True if the input was descending and arrays were reordered.
    input_order
        "ascending" or "descending".
    """
    freq = np.asarray(freq_array_hz, dtype=np.float64)
    chan_width = np.asarray(channel_width_hz, dtype=np.float64)

    if freq.ndim != 1:
        raise ValueError(f"freq_array_hz must be 1D, got shape={freq.shape}")

    if chan_width.ndim != 1:
        raise ValueError(
            f"channel_width_hz must be 1D, got shape={chan_width.shape}"
        )

    if freq.shape != chan_width.shape:
        raise ValueError(
            f"freq/channel_width shape mismatch: {freq.shape} != {chan_width.shape}"
        )

    if data_array.ndim != 3:
        raise ValueError(f"data_array must be 3D, got shape={data_array.shape}")

    nfreqs = freq.size
    if data_array.shape[1] != nfreqs:
        raise ValueError(
            f"data_array frequency axis mismatch: data shape={data_array.shape}, "
            f"Nfreqs={nfreqs}"
        )
    if flag_array.shape != data_array.shape:
        raise ValueError(
            f"flag_array shape mismatch: {flag_array.shape} != {data_array.shape}"
        )
    if nsample_array.shape != data_array.shape:
        raise ValueError(
            f"nsample_array shape mismatch: {nsample_array.shape} != {data_array.shape}"
        )

    if nfreqs == 0:
        raise ValueError("frequency axis is empty")

    if np.any(~np.isfinite(freq)):
        raise ValueError("freq_array_hz contains non-finite values")

    if np.any(~np.isfinite(chan_width)):
        raise ValueError("channel_width_hz contains non-finite values")

    if np.any(chan_width <= 0.0):
        raise ValueError("channel_width_hz must be positive")

    if nfreqs == 1:
        return (
            freq,
            chan_width,
            data_array,
            flag_array,
            nsample_array,
            False,
            "ascending",
        )

    diff = np.diff(freq)

    if np.all(diff > 0.0):
        return (
            freq,
            chan_width,
            data_array,
            flag_array,
            nsample_array,
            False,
            "ascending",
        )

    if np.all(diff < 0.0):
        order = np.argsort(freq)
        return (
            freq[order],
            chan_width[order],
            data_array[:, order, :],
            flag_array[:, order, :],
            nsample_array[:, order, :],
            True,
            "descending",
        )

    raise ValueError(
        "frequency axis is not strictly monotonic. "
        f"freq first={float(freq[0])}, last={float(freq[-1])}, "
        f"min={float(np.min(freq))}, max={float(np.max(freq))}"
    )


def build_data_flag_nsample(
    h5: h5py.File,
    ant_1_array: np.ndarray,
    ant_2_array: np.ndarray,
    time_index_array: np.ndarray,
    include_autos: bool = False,
    allow_partial_pols: bool = False,
) -> dict[str, Any]:
    signal_pairs = np.asarray(h5["baseline/signal_pairs"][()], dtype=np.int64)
    signal_lookup = build_signal_lookup(h5)
    baseline_lookup = build_baseline_pair_lookup(signal_pairs)
    vis = h5["vis"]
    nblts = ant_1_array.size
    nfreqs = int(h5["frequency/chan_freq_hz"].shape[0])
    npols = len(POL_ORDER)

    data_array = np.zeros((nblts, nfreqs, npols), dtype=np.complex64)
    flag_array = np.ones((nblts, nfreqs, npols), dtype=np.bool_)
    nsample_array = np.zeros((nblts, nfreqs, npols), dtype=np.float32)
    keep_mask = np.zeros(nblts, dtype=bool)
    unflagged_by_pol = {name: 0 for name in POL_ORDER}
    flagged_by_pol = {name: 0 for name in POL_ORDER}

    rows_with_any_data = 0
    rows_with_partial_pols = 0
    rows_dropped_all_missing = 0
    rows_auto_candidate = 0
    rows_auto_written = 0
    rows_auto_dropped_all_missing = 0
    rows_cross_written = 0

    # Backward-compatible old key.
    # Autos are no longer deliberately flagged by this function.
    rows_autos_flagged = 0

    for blt in range(nblts):
        ti = int(time_index_array[blt])
        ant1 = int(ant_1_array[blt])
        ant2 = int(ant_2_array[blt])
        is_auto = ant1 == ant2

        if is_auto:
            rows_auto_candidate += 1

        found_count = 0

        for pol_index, corr in enumerate(POL_ORDER):
            signal_key_a, signal_key_b = corr_to_ant_pol_pair(
                ant1,
                ant2,
                corr,
            )

            if signal_key_a not in signal_lookup:
                continue
            if signal_key_b not in signal_lookup:
                continue

            signal_a = signal_lookup[signal_key_a]
            signal_b = signal_lookup[signal_key_b]
            spectrum, _used_conjugate = get_visibility_for_signal_pair(
                vis,
                ti,
                baseline_lookup,
                signal_a,
                signal_b,
            )
            if spectrum is None:
                continue

            data_array[blt, :, pol_index] = spectrum.astype(
                np.complex64,
                copy=False,
            )
            flag_array[blt, :, pol_index] = False
            nsample_array[blt, :, pol_index] = 1.0
            found_count += 1
            unflagged_by_pol[corr] += nfreqs

        if found_count == 0:
            rows_dropped_all_missing += 1
            if is_auto:
                rows_auto_dropped_all_missing += 1
            continue

        keep_mask[blt] = True
        rows_with_any_data += 1

        if is_auto:
            rows_auto_written += 1
        else:
            rows_cross_written += 1

        if found_count != len(POL_ORDER):
            rows_with_partial_pols += 1
            if not allow_partial_pols:
                raise ValueError(
                    f"physical baseline row has only {found_count}/4 pol products "
                    f"at time_index={ti}, antenna1={ant1}, antenna2={ant2}. "
                    "Allow partial polarizations to write missing products as "
                    "flagged."
                )

    for pol_index, corr in enumerate(POL_ORDER):
        flagged_by_pol[corr] = int(np.sum(flag_array[keep_mask, :, pol_index]))

    if np.any(~np.isfinite(data_array[keep_mask])):
        bad_count = int(np.sum(~np.isfinite(data_array[keep_mask])))
        raise ValueError(
            f"data_array contains non-finite values, count={bad_count}"
        )

    if np.any(nsample_array < 0):
        raise ValueError("nsample_array contains negative values")

    if np.any((nsample_array[keep_mask] == 0) & (~flag_array[keep_mask])):
        raise ValueError("unflagged data has zero nsample")

    return {
        "data_array": data_array[keep_mask],
        "flag_array": flag_array[keep_mask],
        "nsample_array": nsample_array[keep_mask],
        "keep_mask": keep_mask,
        "rows_with_any_data": rows_with_any_data,
        "rows_with_partial_pols": rows_with_partial_pols,
        "rows_dropped_all_missing": rows_dropped_all_missing,
        "rows_auto_candidate": rows_auto_candidate,
        "rows_auto_written": rows_auto_written,
        "rows_auto_dropped_all_missing": rows_auto_dropped_all_missing,
        "rows_cross_written": rows_cross_written,
        "rows_autos_flagged": rows_autos_flagged,
        "unflagged_by_pol": unflagged_by_pol,
        "flagged_by_pol": flagged_by_pol,
    }


def build_uvw_array(
    h5: h5py.File,
    ant_1_array: np.ndarray,
    ant_2_array: np.ndarray,
    time_index_array: np.ndarray,
    allow_uvw_warnings: bool = False,
    tolerance: float = 1e-6,
) -> np.ndarray:
    uvw_lookup = build_uvw_lookup(
        h5,
        allow_uvw_warnings=allow_uvw_warnings,
        tolerance=tolerance,
    )
    uvw_array = np.zeros((ant_1_array.size, 3), dtype=np.float64)

    for blt in range(ant_1_array.size):
        ti = int(time_index_array[blt])
        ant1 = int(ant_1_array[blt])
        ant2 = int(ant_2_array[blt])

        if ant1 == ant2:
            uvw_array[blt, :] = 0.0
            continue

        key = (ti, ant1, ant2)
        uvw = uvw_lookup.get(key)

        if uvw is None:
            reverse_key = (ti, ant2, ant1)
            reverse_uvw = uvw_lookup.get(reverse_key)
            if reverse_uvw is None:
                raise ValueError(
                    f"missing UVW for time_index={ti}, antenna1={ant1}, antenna2={ant2}"
                )
            uvw = -np.asarray(reverse_uvw, dtype=np.float64)

        uvw_array[blt, :] = uvw

    cross = ant_1_array != ant_2_array
    if np.any(cross):
        max_cross = float(np.max(np.abs(uvw_array[cross])))
        if max_cross <= 0.0:
            raise ValueError("packed cross-antenna UVW is all zero")

    return uvw_array


def build_phase_center_catalog(h5: h5py.File) -> dict[int, dict[str, Any]]:
    source_name = read_text_scalar(h5, "field/source_name").strip()
    if source_name == "":
        source_name = "PhaseCenter"

    ra_rad = float(read_scalar(h5, "field/phase_center_ra_rad"))
    dec_rad = float(read_scalar(h5, "field/phase_center_dec_rad"))
    frame_text = read_text_scalar(h5, "field/frame").strip().upper()

    if frame_text in ["J2000", "FK5"]:
        return {
            0: {
                "cat_name": source_name,
                "cat_type": "sidereal",
                "cat_lon": ra_rad,
                "cat_lat": dec_rad,
                "cat_frame": "fk5",
                "cat_epoch": 2000.0,
            }
        }

    if frame_text == "ICRS":
        return {
            0: {
                "cat_name": source_name,
                "cat_type": "sidereal",
                "cat_lon": ra_rad,
                "cat_lat": dec_rad,
                "cat_frame": "icrs",
            }
        }

    raise ValueError(f"unsupported field/frame for MS export: {frame_text}")


def build_telescope_location(used_meta: dict[str, Any]) -> Any:
    require_astropy()
    return EarthLocation.from_geocentric(
        used_meta["array_center_m"][0] * u.m,
        used_meta["array_center_m"][1] * u.m,
        used_meta["array_center_m"][2] * u.m,
    )


def filter_uvdata_new_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """
    Keep only keyword arguments supported by the installed pyuvdata UVData.new().

    Important:
    - Some pyuvdata versions expose UVData.new(**kwargs), in which case we should
      pass all non-None kwargs.
    - Some pyuvdata versions expose explicit keyword-only parameters, in which case
      unsupported optional kwargs must be removed.
    """
    signature = inspect.signature(UVData.new)
    parameters = signature.parameters

    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )

    clean_kwargs = {
        key: value
        for key, value in kwargs.items()
        if value is not None
    }

    if accepts_var_kwargs:
        return clean_kwargs

    allowed = set(parameters.keys())
    return {
        key: value
        for key, value in clean_kwargs.items()
        if key in allowed
    }


def validate_payload_before_uvdata(payload: dict[str, Any]) -> None:
    nblts = payload["Nblts"]
    nfreqs = payload["Nfreqs"]
    npols = payload["Npols"]
    expected_data_shape = (nblts, nfreqs, npols)

    if payload["data_array"].shape != expected_data_shape:
        raise ValueError(
            "payload data_array shape mismatch: "
            f"{payload['data_array'].shape} != {expected_data_shape}"
        )
    if payload["flag_array"].shape != expected_data_shape:
        raise ValueError("payload flag_array shape mismatch")
    if payload["nsample_array"].shape != expected_data_shape:
        raise ValueError("payload nsample_array shape mismatch")
    if payload["uvw_array_m"].shape != (nblts, 3):
        raise ValueError(
            "payload uvw_array_m shape mismatch: "
            f"{payload['uvw_array_m'].shape} != {(nblts, 3)}"
        )
    if payload["ant_1_array"].shape != (nblts,):
        raise ValueError("payload ant_1_array length mismatch")
    if payload["ant_2_array"].shape != (nblts,):
        raise ValueError("payload ant_2_array length mismatch")
    if payload["time_array_jd"].shape != (nblts,):
        raise ValueError("payload time_array_jd length mismatch")
    if payload["integration_time_array_s"].shape != (nblts,):
        raise ValueError("payload integration_time_array_s length mismatch")
    if payload["freq_array_hz"].shape != (nfreqs,):
        raise ValueError("payload freq_array_hz length mismatch")
    if payload["channel_width_hz"].shape != (nfreqs,):
        raise ValueError("payload channel_width_hz length mismatch")

    if not np.all(np.diff(payload["time_array_jd"]) >= 0):
        warn("time_array_jd is not monotonically increasing")

    if payload["freq_array_hz"].size > 1:
        freq_diff = np.diff(payload["freq_array_hz"])
        if not np.all(freq_diff > 0.0):
            raise ValueError(
                "payload freq_array_hz must be strictly ascending after normalization"
            )

    if np.any(payload["channel_width_hz"] <= 0):
        raise ValueError("payload channel_width_hz must be positive")

    if payload.get("frequency_axis_output_order") != "ascending":
        raise ValueError(
            "payload frequency_axis_output_order must be 'ascending'"
        )

    if np.any(~np.isfinite(payload["uvw_array_m"])):
        raise ValueError("payload UVW contains non-finite values")

    if np.any(~np.isfinite(payload["data_array"])):
        raise ValueError("payload data_array contains non-finite values")

    if np.any(payload["nsample_array"] < 0):
        raise ValueError("payload nsample_array contains negative values")

    if np.any((payload["nsample_array"] == 0) & (~payload["flag_array"])):
        raise ValueError("payload has unflagged data with zero nsample")


def baseline_to_array(
    uvd: Any,
    ant_1_array: np.ndarray,
    ant_2_array: np.ndarray,
) -> np.ndarray:
    if hasattr(uvd, "antnums_to_baseline"):
        return np.asarray(
            uvd.antnums_to_baseline(ant_1_array, ant_2_array),
            dtype=np.int64,
        )

    raise RuntimeError("cannot derive baseline_array: UVData.antnums_to_baseline missing")


def build_telescope_object(
    used_meta: dict[str, Any],
    x_orientation: str = "east",
) -> Any:
    if Telescope is None:
        raise RuntimeError("pyuvdata Telescope class is unavailable")

    tel = Telescope()
    tel.name = CASA_TELESCOPE_NAME
    tel.instrument = INSTRUMENT_NAME
    tel.location = build_telescope_location(used_meta)
    tel.antenna_positions = used_meta["antenna_positions_rel_m"]
    tel.antenna_names = used_meta["antenna_names"]
    tel.antenna_numbers = used_meta["antenna_ids"]

    if hasattr(tel, "x_orientation") and x_orientation != "none":
        tel.x_orientation = x_orientation

    return tel


def apply_uvdata_payload_arrays(
    uvd: Any,
    ms_payload: dict[str, Any],
    phase_center_catalog: dict[int, dict[str, Any]],
    phase_center_id_array: np.ndarray,
) -> None:
    uvd.data_array = ms_payload["data_array"]
    uvd.flag_array = ms_payload["flag_array"]
    uvd.nsample_array = ms_payload["nsample_array"]
    uvd.integration_time = ms_payload["integration_time_array_s"]
    uvd.channel_width = ms_payload["channel_width_hz"]
    uvd.uvw_array = ms_payload["uvw_array_m"]
    uvd.vis_units = "uncalib"
    uvd.history = ms_payload["history"]
    uvd.phase_center_catalog = phase_center_catalog
    uvd.phase_center_id_array = phase_center_id_array


def normalize_uvdata_writer_metadata(uvd: Any, ms_payload: dict[str, Any]) -> None:
    """
    Normalize legacy UVData metadata fields that pyuvdata 2.4.x MS writer reads
    directly.

    In pyuvdata 2.4.x, UVData.new() can leave telescope_location as a Python list,
    but write_ms() later assumes it is a numpy array and calls reshape() on it.
    """
    used_meta = ms_payload["used_meta"]

    uvd.telescope_location = np.asarray(
        getattr(uvd, "telescope_location", used_meta["array_center_m"]),
        dtype=np.float64,
    )
    uvd.antenna_positions = np.asarray(
        used_meta["antenna_positions_rel_m"],
        dtype=np.float64,
    )
    uvd.antenna_numbers = np.asarray(
        used_meta["antenna_ids"],
        dtype=np.int64,
    )
    uvd.antenna_names = np.asarray(
        used_meta["antenna_names"],
        dtype=object,
    )

    if "dish_diameter_m" in used_meta:
        uvd.antenna_diameters = np.asarray(
            used_meta["dish_diameter_m"],
            dtype=np.float64,
        )

    uvd.telescope_name = CASA_TELESCOPE_NAME
    uvd.instrument = INSTRUMENT_NAME


def build_uvdata_with_explicit_new(ms_payload: dict[str, Any]) -> Any:
    require_astropy()

    if not hasattr(UVData, "new"):
        raise RuntimeError("this pyuvdata version does not provide UVData.new()")

    used_meta = ms_payload["used_meta"]
    phase_center_catalog = build_phase_center_catalog(ms_payload["h5"])
    phase_center_id_array = np.zeros(ms_payload["Nblts"], dtype=np.int64)
    antpairs = np.column_stack(
        [ms_payload["ant_1_array"], ms_payload["ant_2_array"]]
    ).astype(np.int64)
    x_orientation = (
        None if ms_payload["x_orientation"] == "none" else ms_payload["x_orientation"]
    )

    # IMPORTANT:
    # In pyuvdata 2.4.2, UVData.new() must be called with keyword arguments.
    # Do not pass these as positional arguments.
    core_kwargs: dict[str, Any] = {
        "freq_array": ms_payload["freq_array_hz"].astype(np.float64),
        "polarization_array": POL_NUMS.astype(np.int64),
        "antenna_positions": used_meta["antenna_positions_rel_m"].astype(np.float64),
        "telescope_location": build_telescope_location(used_meta),
        "telescope_name": CASA_TELESCOPE_NAME,
        "times": ms_payload["time_array_jd"].astype(np.float64),
    }

    full_kwargs: dict[str, Any] = {
        **core_kwargs,
        "antpairs": antpairs,
        "do_blt_outer": False,
        "data_array": ms_payload["data_array"],
        "flag_array": ms_payload["flag_array"],
        "nsample_array": ms_payload["nsample_array"],
        "integration_time": ms_payload["integration_time_array_s"].astype(np.float64),
        "channel_width": ms_payload["channel_width_hz"].astype(np.float64),
        "antenna_names": used_meta["antenna_names"].tolist(),
        "antenna_numbers": used_meta["antenna_ids"].astype(np.int64).tolist(),
        "instrument": INSTRUMENT_NAME,
        "vis_units": "uncalib",
        "history": ms_payload["history"],
        "phase_center_catalog": phase_center_catalog,
        "phase_center_id_array": phase_center_id_array,
        "x_orientation": x_orientation,
        "uvw_array": ms_payload["uvw_array_m"].astype(np.float64),
    }

    try:
        uvd = UVData.new(**filter_uvdata_new_kwargs(full_kwargs))
        normalize_uvdata_writer_metadata(uvd, ms_payload)
        return uvd
    except TypeError as error:
        warn(f"Full UVData.new call failed, retrying minimal call: {error}")

    minimal_kwargs: dict[str, Any] = {
        **core_kwargs,
        "antpairs": antpairs,
        "do_blt_outer": False,
        "integration_time": ms_payload["integration_time_array_s"].astype(np.float64),
        "channel_width": ms_payload["channel_width_hz"].astype(np.float64),
        "antenna_names": used_meta["antenna_names"].tolist(),
        "antenna_numbers": used_meta["antenna_ids"].astype(np.int64).tolist(),
        "instrument": INSTRUMENT_NAME,
        "phase_center_catalog": phase_center_catalog,
        "phase_center_id_array": phase_center_id_array,
        "x_orientation": x_orientation,
    }

    uvd = UVData.new(**filter_uvdata_new_kwargs(minimal_kwargs))
    apply_uvdata_payload_arrays(
        uvd,
        ms_payload,
        phase_center_catalog=phase_center_catalog,
        phase_center_id_array=phase_center_id_array,
    )
    normalize_uvdata_writer_metadata(uvd, ms_payload)
    return uvd


def build_uvdata_manual(ms_payload: dict[str, Any]) -> Any:
    phase_center_catalog = build_phase_center_catalog(ms_payload["h5"])
    phase_center_id_array = np.zeros(ms_payload["Nblts"], dtype=np.int64)

    uvd = UVData()
    uvd.telescope = build_telescope_object(
        ms_payload["used_meta"],
        x_orientation=ms_payload["x_orientation"],
    )
    uvd.time_array = ms_payload["time_array_jd"]
    uvd.ant_1_array = ms_payload["ant_1_array"]
    uvd.ant_2_array = ms_payload["ant_2_array"]
    uvd.freq_array = ms_payload["freq_array_hz"]
    uvd.polarization_array = POL_NUMS
    apply_uvdata_payload_arrays(
        uvd,
        ms_payload,
        phase_center_catalog=phase_center_catalog,
        phase_center_id_array=phase_center_id_array,
    )
    normalize_uvdata_writer_metadata(uvd, ms_payload)
    uvd.scan_number_array = np.ones(ms_payload["Nblts"], dtype=np.int64)
    uvd.spw_array = np.array([0], dtype=np.int64)
    uvd.flex_spw_id_array = np.zeros(ms_payload["Nfreqs"], dtype=np.int64)
    uvd.baseline_array = baseline_to_array(
        uvd,
        ms_payload["ant_1_array"],
        ms_payload["ant_2_array"],
    )

    uvd.Nblts = ms_payload["Nblts"]
    uvd.Nbls = ms_payload["Nbls"]
    uvd.Ntimes = ms_payload["Ntimes"]
    uvd.Nfreqs = ms_payload["Nfreqs"]
    uvd.Npols = ms_payload["Npols"]
    uvd.Nspws = 1
    uvd.Nants_data = int(
        np.unique(
            np.concatenate([ms_payload["ant_1_array"], ms_payload["ant_2_array"]])
        ).size
    )
    uvd.Nants_telescope = int(ms_payload["used_meta"]["antenna_ids"].size)

    if hasattr(uvd, "set_lsts_from_time_array"):
        uvd.set_lsts_from_time_array()

    if hasattr(uvd, "_set_app_coords_helper"):
        try:
            uvd._set_app_coords_helper()
        except Exception as error:  # pragma: no cover
            warn(f"_set_app_coords_helper failed: {error}")

    return uvd


def build_uvdata_object(
    ms_payload: dict[str, Any],
    constructor: str = "new",
) -> Any:
    require_pyuvdata()

    if constructor == "new":
        uvd = build_uvdata_with_explicit_new(ms_payload)
    elif constructor == "manual":
        warn(
            "Using manual UVData construction. "
            "This is intended for debugging only; production export should use "
            "the default UVData.new path."
        )
        uvd = build_uvdata_manual(ms_payload)
    else:
        raise ValueError(f"unknown UVData constructor: {constructor}")

    uvd.check(check_extra=True, run_check_acceptability=True)
    verify_uvdata_payload_consistency(uvd, ms_payload)
    return uvd


def verify_uvdata_payload_consistency(uvd: Any, payload: dict[str, Any]) -> None:
    if not hasattr(uvd, "data_array"):
        raise ValueError("UVData object has no data_array")

    if uvd.data_array.shape != payload["data_array"].shape:
        raise ValueError(
            f"UVData data_array shape mismatch: "
            f"{uvd.data_array.shape} != {payload['data_array'].shape}"
        )

    if uvd.flag_array.shape != payload["flag_array"].shape:
        raise ValueError("UVData flag_array shape mismatch")

    if uvd.nsample_array.shape != payload["nsample_array"].shape:
        raise ValueError("UVData nsample_array shape mismatch")

    if uvd.uvw_array.shape != payload["uvw_array_m"].shape:
        raise ValueError(
            f"UVData uvw_array shape mismatch: "
            f"{uvd.uvw_array.shape} != {payload['uvw_array_m'].shape}"
        )

    max_uvw_diff = float(
        np.max(np.abs(np.asarray(uvd.uvw_array) - payload["uvw_array_m"]))
    )
    if max_uvw_diff > 1e-6:
        raise ValueError(
            f"UVData uvw_array differs from HDF5 UVW: max diff = {max_uvw_diff}"
        )

    if not np.array_equal(np.asarray(uvd.ant_1_array), payload["ant_1_array"]):
        raise ValueError("UVData ant_1_array does not match payload")

    if not np.array_equal(np.asarray(uvd.ant_2_array), payload["ant_2_array"]):
        raise ValueError("UVData ant_2_array does not match payload")

    if not np.allclose(
        np.asarray(uvd.time_array, dtype=np.float64),
        payload["time_array_jd"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("UVData time_array does not match payload")

    if not np.array_equal(
        np.asarray(uvd.polarization_array, dtype=np.int64),
        POL_NUMS,
    ):
        raise ValueError("UVData polarization_array does not match expected POL_NUMS")

    telescope_location = np.asarray(getattr(uvd, "telescope_location"), dtype=np.float64)
    if telescope_location.shape != (3,):
        raise ValueError(
            f"UVData telescope_location shape mismatch: {telescope_location.shape}"
        )

    antenna_positions = np.asarray(getattr(uvd, "antenna_positions"), dtype=np.float64)
    expected_ant_shape = payload["used_meta"]["antenna_positions_rel_m"].shape
    if antenna_positions.shape != expected_ant_shape:
        raise ValueError(
            "UVData antenna_positions shape mismatch: "
            f"{antenna_positions.shape} != {expected_ant_shape}"
        )


def prepare_ms_payload(
    h5: h5py.File,
    include_autos: bool = True,
    allow_partial_pols: bool = True,
    allow_uvw_warnings: bool = False,
    x_orientation: str = "east",
) -> dict[str, Any]:
    layout = build_physical_blt_axis(h5, include_autos=include_autos)
    packed = build_data_flag_nsample(
        h5,
        layout["ant_1_array"],
        layout["ant_2_array"],
        layout["time_index_array"],
        include_autos=include_autos,
        allow_partial_pols=allow_partial_pols,
    )

    keep_mask = packed["keep_mask"]
    ant_1_array = layout["ant_1_array"][keep_mask]
    ant_2_array = layout["ant_2_array"][keep_mask]
    time_array_jd = layout["time_array_jd"][keep_mask]
    time_index_array = layout["time_index_array"][keep_mask]
    if ant_1_array.size <= 0:
        raise ValueError(
            "no rows to write to MeasurementSet. "
            "Check used antennas, input polarizations, and the current "
            "auto-correlation policy."
        )

    uvw_array_m = build_uvw_array(
        h5,
        ant_1_array,
        ant_2_array,
        time_index_array,
        allow_uvw_warnings=allow_uvw_warnings,
    )

    freq_array_hz = np.asarray(h5["frequency/chan_freq_hz"][()], dtype=np.float64)
    channel_width_hz = np.asarray(
        h5["frequency/chan_width_hz"][()],
        dtype=np.float64,
    )
    (
        freq_array_hz,
        channel_width_hz,
        data_array,
        flag_array,
        nsample_array,
        frequency_axis_reordered,
        frequency_axis_input_order,
    ) = normalize_frequency_axis(
        freq_array_hz=freq_array_hz,
        channel_width_hz=channel_width_hz,
        data_array=packed["data_array"],
        flag_array=packed["flag_array"],
        nsample_array=packed["nsample_array"],
    )
    exposure_sec = get_scalar_seconds(h5["time/exposure_sec"])
    integration_time_array_s = np.full(
        ant_1_array.size,
        exposure_sec,
        dtype=np.float64,
    )

    return {
        "h5": h5,
        "used_meta": layout["used_meta"],
        "antpairs": layout["antpairs"],
        "ant_1_array": ant_1_array,
        "ant_2_array": ant_2_array,
        "time_array_jd": time_array_jd,
        "time_index_array": time_index_array,
        "uvw_array_m": uvw_array_m,
        "data_array": data_array,
        "flag_array": flag_array,
        "nsample_array": nsample_array,
        "freq_array_hz": freq_array_hz,
        "channel_width_hz": channel_width_hz,
        "frequency_axis_reordered": frequency_axis_reordered,
        "frequency_axis_input_order": frequency_axis_input_order,
        "frequency_axis_output_order": "ascending",
        "integration_time_array_s": integration_time_array_s,
        "rows_with_any_data": packed["rows_with_any_data"],
        "rows_with_partial_pols": packed["rows_with_partial_pols"],
        "rows_dropped_all_missing": packed["rows_dropped_all_missing"],
        "rows_auto_candidate": packed["rows_auto_candidate"],
        "rows_auto_written": packed["rows_auto_written"],
        "rows_auto_dropped_all_missing": packed["rows_auto_dropped_all_missing"],
        "rows_cross_written": packed["rows_cross_written"],
        "rows_autos_flagged": packed["rows_autos_flagged"],
        "include_autos": include_autos,
        "allow_partial_pols": allow_partial_pols,
        "allow_uvw_warnings": allow_uvw_warnings,
        "x_orientation": x_orientation,
        "unflagged_by_pol": packed["unflagged_by_pol"],
        "flagged_by_pol": packed["flagged_by_pol"],
        "Nblts": int(ant_1_array.size),
        "Nbls": int(np.unique(np.column_stack([ant_1_array, ant_2_array]), axis=0).shape[0])
        if ant_1_array.size > 0
        else 0,
        "Ntimes": int(np.unique(time_array_jd).size),
        "Nfreqs": int(freq_array_hz.size),
        "Npols": int(len(POL_ORDER)),
        "history": (
            "Created from MS-ready HDF5 by hdf5_to_ms.py via pyuvdata. "
            f"source_hdf5={h5.filename}; "
            f"array_name={ARRAY_NAME}; "
            f"instrument_name={INSTRUMENT_NAME}; "
            f"casa_telescope_name={CASA_TELESCOPE_NAME}; "
            f"corr_output_mode={as_text(h5.attrs.get('corr_output_mode', 'unknown'))}; "
            f"polarization_order={','.join(POL_ORDER)}; "
            f"include_autos={include_autos}; "
            f"auto_data_policy={'preserve_hdf5_data_even_if_zero' if include_autos else 'not_written'}; "
            f"allow_partial_pols={allow_partial_pols}; "
            f"x_orientation={x_orientation}; "
            "uvw_source=/uvw/uvw_m; "
            f"uvw_method={as_text(h5['uvw'].attrs.get('method', 'unknown'))}; "
            f"frequency_axis_input_order={frequency_axis_input_order}; "
            "frequency_axis_output_order=ascending; "
            f"frequency_axis_reordered={frequency_axis_reordered}; "
            "vis_units=uncalib."
        ),
    }


def print_payload_summary(h5: h5py.File, payload: dict[str, Any], include_autos: bool) -> None:
    print("\n========== HDF5 -> MS SUMMARY ==========")
    print("input hdf5               :", h5.filename)
    print("source name              :", read_text_scalar(h5, "field/source_name"))
    print("CASA telescope name     :", CASA_TELESCOPE_NAME)
    print("array name              :", ARRAY_NAME)
    print("instrument name         :", INSTRUMENT_NAME)
    print("phase center RA HMS      :", read_text_scalar(h5, "field/phase_center_ra_hms"))
    print("phase center Dec DMS     :", read_text_scalar(h5, "field/phase_center_dec_dms"))
    print("corr output mode         :", h5.attrs.get("corr_output_mode", "unknown"))
    print("include autos            :", include_autos)
    print("allow partial pols       :", payload["allow_partial_pols"])
    print("allow uvw warnings       :", payload["allow_uvw_warnings"])
    print("overwrite output         :", True)
    print("uvdata constructor       :", "new")
    print("uvw sign policy          :", "use HDF5 /uvw/uvw_m as-is")
    print("x orientation            :", payload["x_orientation"])
    print("polarization order       :", ", ".join(POL_ORDER))
    print(
        "used antennas            :",
        ", ".join([str(int(item)) for item in payload["used_meta"]["antenna_ids"]]),
    )
    print("candidate physical bls   :", int(payload["antpairs"].shape[0]))
    print("written Nblts            :", payload["Nblts"])
    print("written Nbls             :", payload["Nbls"])
    print("written Ntimes           :", payload["Ntimes"])
    print("Nfreqs                   :", payload["Nfreqs"])
    print("Npols                    :", payload["Npols"])
    print("payload data shape       :", payload["data_array"].shape)
    print("payload data convention  :", "(Nblts, Nfreqs, Npols)")
    print(
        "MS DATA cell shape note  :",
        "(Ncorr, Nchan) or (Nchan, Ncorr) depending on writer",
    )
    print("rows with any data       :", payload["rows_with_any_data"])
    print("rows with partial pols   :", payload["rows_with_partial_pols"])
    print("rows dropped all missing :", payload["rows_dropped_all_missing"])
    print("rows auto candidate      :", payload.get("rows_auto_candidate", 0))
    print("rows auto written        :", payload.get("rows_auto_written", 0))
    print("rows auto dropped missing:", payload.get("rows_auto_dropped_all_missing", 0))
    print("rows cross written       :", payload.get("rows_cross_written", 0))
    print("rows autos flagged       :", payload.get("rows_autos_flagged", 0))
    print("unflagged by pol         :", payload["unflagged_by_pol"])
    print("flagged by pol           :", payload["flagged_by_pol"])
    data_abs_max = float(np.max(np.abs(payload["data_array"])))
    data_nonzero_count = int(np.count_nonzero(payload["data_array"]))
    print("payload global max |DATA|:", data_abs_max)
    print("payload nonzero samples  :", data_nonzero_count)
    if data_abs_max <= 0.0:
        warn(
            "payload DATA are all zero. "
            "The MeasurementSet can still be written for format and plotting "
            "tests, but all visibility amplitudes are zero and phase plots have "
            "no scientific meaning."
        )
    print(
        "data array bytes         :",
        format_bytes(int(payload["data_array"].nbytes)),
    )
    print(
        "flag array bytes         :",
        format_bytes(int(payload["flag_array"].nbytes)),
    )
    print(
        "nsample array bytes      :",
        format_bytes(int(payload["nsample_array"].nbytes)),
    )
    uvw = payload["uvw_array_m"]
    cross = payload["ant_1_array"] != payload["ant_2_array"]
    if np.any(cross):
        uvw_cross = uvw[cross]
        print("cross UVW min m          :", np.min(uvw_cross, axis=0))
        print("cross UVW max m          :", np.max(uvw_cross, axis=0))
        print("cross max |UVW| m        :", float(np.max(np.abs(uvw_cross))))
    freq = payload["freq_array_hz"]
    print(
        "frequency input order   :",
        payload.get("frequency_axis_input_order", "unknown"),
    )
    print(
        "frequency output order  :",
        payload.get("frequency_axis_output_order", "unknown"),
    )
    print(
        "frequency axis reordered:",
        payload.get("frequency_axis_reordered", "unknown"),
    )
    print("freq first Hz            :", float(freq[0]))
    print("freq last Hz             :", float(freq[-1]))
    print("freq min Hz              :", float(np.min(freq)))
    print("freq max Hz              :", float(np.max(freq)))
    time = payload["time_array_jd"]
    print("time JD first            :", float(time[0]))
    print("time JD last             :", float(time[-1]))
    print("========================================")


def remove_existing_output(output_ms: str, overwrite: bool) -> None:
    if not os.path.exists(output_ms):
        return

    if not overwrite:
        raise FileExistsError(
            f"output MeasurementSet already exists: {output_ms}. "
            "Set overwrite=True to replace it."
        )

    if os.path.isdir(output_ms):
        shutil.rmtree(output_ms)
    else:
        os.remove(output_ms)


def verify_ms_directory_layout(output_ms: str) -> None:
    required_paths = [
        output_ms,
        os.path.join(output_ms, "table.dat"),
        os.path.join(output_ms, "ANTENNA"),
        os.path.join(output_ms, "FIELD"),
        os.path.join(output_ms, "SPECTRAL_WINDOW"),
        os.path.join(output_ms, "POLARIZATION"),
        os.path.join(output_ms, "DATA_DESCRIPTION"),
    ]

    for path in required_paths:
        if not os.path.exists(path):
            raise ValueError(f"missing expected MeasurementSet path: {path}")


def verify_ms_with_casacore(output_ms: str, payload: dict[str, Any]) -> None:
    require_casacore()

    tb = casacore_tables.table(output_ms, readonly=True)
    try:
        print("\n========== MS MAIN CHECK ==========")
        print("main rows:", tb.nrows())
        colnames = tb.colnames()
        print("main columns:", ", ".join(colnames))

        required_cols = [
            "DATA",
            "FLAG",
            "UVW",
            "TIME",
            "ANTENNA1",
            "ANTENNA2",
            "DATA_DESC_ID",
            "FIELD_ID",
            "INTERVAL",
            "EXPOSURE",
            "WEIGHT",
            "SIGMA",
            "FLAG_ROW",
        ]
        for name in required_cols:
            if name not in colnames:
                raise ValueError(f"missing MAIN column: {name}")

        if tb.nrows() != payload["Nblts"]:
            raise ValueError(f"MAIN row count mismatch: {tb.nrows()} != {payload['Nblts']}")

        ant1 = np.asarray(tb.getcol("ANTENNA1"), dtype=np.int64)
        ant2 = np.asarray(tb.getcol("ANTENNA2"), dtype=np.int64)
        if not np.array_equal(ant1, payload["ant_1_array"]):
            raise ValueError("MAIN ANTENNA1 does not match payload")
        if not np.array_equal(ant2, payload["ant_2_array"]):
            raise ValueError("MAIN ANTENNA2 does not match payload")

        uvw = np.asarray(tb.getcol("UVW"), dtype=np.float64)
        if uvw.shape != payload["uvw_array_m"].shape:
            raise ValueError(f"MAIN UVW shape mismatch: {uvw.shape}")

        max_uvw_diff = float(np.max(np.abs(uvw - payload["uvw_array_m"])))
        if max_uvw_diff > 1e-6:
            raise ValueError(f"MAIN UVW differs from payload: max diff={max_uvw_diff}")

        ddid = np.asarray(tb.getcol("DATA_DESC_ID"), dtype=np.int64)
        if not np.all(ddid == 0):
            raise ValueError("DATA_DESC_ID should be 0 for this single-spw/single-pol setup")

        flag_row = np.asarray(tb.getcol("FLAG_ROW"), dtype=bool)
        if flag_row.shape != (payload["Nblts"],):
            raise ValueError("FLAG_ROW shape mismatch")

        auto_mask = ant1 == ant2
        cross_mask = ant1 != ant2
        print("MAIN auto rows:", int(np.sum(auto_mask)))
        print("MAIN cross rows:", int(np.sum(cross_mask)))

        if payload.get("include_autos", False):
            if not np.any(auto_mask):
                raise ValueError(
                    "auto-correlations are required, but MAIN has no auto rows"
                )

            auto_uvw = uvw[auto_mask]
            max_auto_uvw = float(np.max(np.abs(auto_uvw)))
            if max_auto_uvw > 1e-6:
                raise ValueError(f"auto rows should have zero UVW, max={max_auto_uvw}")

            auto_row_indices = np.where(auto_mask)[0]
            payload_auto_mask = payload["ant_1_array"] == payload["ant_2_array"]
            payload_auto_data = np.asarray(payload["data_array"])[payload_auto_mask]
            payload_auto_flag = np.asarray(payload["flag_array"])[payload_auto_mask]

            if payload_auto_data.size == 0:
                raise ValueError(
                    "auto-correlations are required, but payload has no auto DATA "
                    "rows"
                )

            if payload_auto_flag.shape != payload_auto_data.shape:
                raise ValueError("payload auto FLAG/DATA shape mismatch")

            payload_auto_row_max = np.max(np.abs(payload_auto_data), axis=(1, 2))
            payload_auto_max = float(np.max(payload_auto_row_max))
            payload_auto_nonzero_rows = int(np.sum(payload_auto_row_max > 0.0))

            print("payload auto max |DATA|:", payload_auto_max)
            print(
                "payload auto nonzero rows:",
                payload_auto_nonzero_rows,
                "/",
                int(payload_auto_row_max.size),
            )

            if payload_auto_max > 0.0:
                # When payload autos contain real nonzero data, validate against
                # a row that actually carries signal instead of blindly sampling
                # the first timestamp.
                local_auto_index = int(np.argmax(payload_auto_row_max))
                sample_auto_row = int(auto_row_indices[local_auto_index])
            else:
                local_auto_index = 0
                sample_auto_row = int(auto_row_indices[0])
                warn(
                    "payload auto DATA are all zero before MS writing. "
                    "This is allowed for pipeline/format/plotting tests. "
                    "It usually means the input .fil data or HDF5 /vis are all "
                    "zero. The generated MS can still contain 0&0 and 1&1 auto "
                    "rows, but phase/amplitude plots will not have scientific "
                    "meaning."
                )

            auto_data = np.asarray(tb.getcell("DATA", sample_auto_row))
            auto_flag = np.asarray(tb.getcell("FLAG", sample_auto_row))

            if auto_data.shape not in [(4, payload["Nfreqs"]), (payload["Nfreqs"], 4)]:
                raise ValueError(f"unexpected auto DATA shape: {auto_data.shape}")

            if auto_flag.shape != auto_data.shape:
                raise ValueError("auto FLAG shape mismatch")

            if np.all(auto_flag):
                raise ValueError(
                    "sample auto row is fully flagged; autos should be unflagged "
                    "when source HDF5 signal pairs exist, even if the DATA values "
                    "are zero"
                )

            auto_data_max = float(np.max(np.abs(auto_data)))
            print("sample auto row:", sample_auto_row)
            print("sample auto local index:", local_auto_index)
            print("MS sample auto max |DATA|:", auto_data_max)

            if payload_auto_max > 0.0 and auto_data_max <= 0.0:
                raise ValueError(
                    "payload has nonzero auto DATA, but the corresponding MS auto "
                    "row is all zero"
                )
        elif np.any(auto_mask):
            raise ValueError("include_autos=False, but MAIN contains auto rows")

        if tb.nrows() > 0:
            data0 = np.asarray(tb.getcell("DATA", 0))
            flag0 = np.asarray(tb.getcell("FLAG", 0))
            uvw0 = np.asarray(tb.getcell("UVW", 0))

            if data0.shape not in [(4, payload["Nfreqs"]), (payload["Nfreqs"], 4)]:
                raise ValueError(
                    f"unexpected DATA shape in row 0: {data0.shape}"
                )
            if flag0.shape != data0.shape:
                raise ValueError(
                    f"FLAG shape mismatch in row 0: {flag0.shape} != {data0.shape}"
                )
            if uvw0.shape != (3,):
                raise ValueError(f"unexpected UVW shape in row 0: {uvw0.shape}")

            print("row0 DATA shape:", data0.shape)
            print("row0 UVW:", uvw0)
        print("===================================")
    finally:
        tb.close()


def verify_ms_subtables(output_ms: str, payload: dict[str, Any]) -> None:
    require_casacore()

    nfreqs = payload["Nfreqs"]
    expected_freq = payload["freq_array_hz"]
    expected_chan_width = payload["channel_width_hz"]
    expected_ra = float(read_scalar(payload["h5"], "field/phase_center_ra_rad"))
    expected_dec = float(read_scalar(payload["h5"], "field/phase_center_dec_rad"))

    spw = casacore_tables.table(
        os.path.join(output_ms, "SPECTRAL_WINDOW"),
        readonly=True,
    )
    try:
        if spw.nrows() < 1:
            raise ValueError("SPECTRAL_WINDOW has no rows")

        num_chan = int(np.asarray(spw.getcell("NUM_CHAN", 0)).reshape(-1)[0])
        if num_chan != nfreqs:
            raise ValueError(f"NUM_CHAN mismatch: {num_chan} != {nfreqs}")

        chan_freq = np.asarray(spw.getcell("CHAN_FREQ", 0), dtype=np.float64).reshape(-1)
        chan_width = np.asarray(spw.getcell("CHAN_WIDTH", 0), dtype=np.float64).reshape(-1)

        if chan_freq.shape != (nfreqs,):
            raise ValueError(f"CHAN_FREQ shape mismatch: {chan_freq.shape}")
        if chan_width.shape != (nfreqs,):
            raise ValueError(f"CHAN_WIDTH shape mismatch: {chan_width.shape}")
        if chan_freq.size > 1 and not np.all(np.diff(chan_freq) > 0.0):
            raise ValueError("MS SPECTRAL_WINDOW CHAN_FREQ is not strictly ascending")

        if not np.allclose(chan_freq, expected_freq, rtol=0.0, atol=1e-6):
            raise ValueError("CHAN_FREQ does not match HDF5 frequency axis")
        if not np.allclose(np.abs(chan_width), expected_chan_width, rtol=0.0, atol=1e-6):
            raise ValueError("CHAN_WIDTH does not match HDF5 channel width")
    finally:
        spw.close()

    pol = casacore_tables.table(
        os.path.join(output_ms, "POLARIZATION"),
        readonly=True,
    )
    try:
        if pol.nrows() < 1:
            raise ValueError("POLARIZATION has no rows")

        num_corr = int(np.asarray(pol.getcell("NUM_CORR", 0)).reshape(-1)[0])
        corr_type = np.asarray(pol.getcell("CORR_TYPE", 0), dtype=np.int64).reshape(-1)
        print("MS CORR_TYPE:", corr_type)
        print("expected CORR_TYPE:", MS_CORR_TYPES)

        if num_corr != len(POL_ORDER):
            raise ValueError(f"NUM_CORR mismatch: {num_corr} != {len(POL_ORDER)}")
        if corr_type.shape[0] != len(MS_CORR_TYPES):
            raise ValueError(f"CORR_TYPE length mismatch: {corr_type.shape[0]}")
        if not np.array_equal(corr_type, MS_CORR_TYPES):
            raise ValueError(
                "CORR_TYPE mismatch. "
                f"MS has {corr_type.tolist()}, "
                f"expected {MS_CORR_TYPES.tolist()} for POL_ORDER={POL_ORDER}. "
                "If these do not match, plot_ms_phase_waterfall.py may select the "
                "wrong correlation product."
            )
    finally:
        pol.close()

    dd = casacore_tables.table(
        os.path.join(output_ms, "DATA_DESCRIPTION"),
        readonly=True,
    )
    try:
        if dd.nrows() < 1:
            raise ValueError("DATA_DESCRIPTION has no rows")

        spw_id = int(np.asarray(dd.getcell("SPECTRAL_WINDOW_ID", 0)).reshape(-1)[0])
        pol_id = int(np.asarray(dd.getcell("POLARIZATION_ID", 0)).reshape(-1)[0])

        if spw_id != 0:
            raise ValueError(f"DATA_DESCRIPTION SPECTRAL_WINDOW_ID mismatch: {spw_id}")
        if pol_id != 0:
            raise ValueError(f"DATA_DESCRIPTION POLARIZATION_ID mismatch: {pol_id}")
    finally:
        dd.close()

    ant = casacore_tables.table(
        os.path.join(output_ms, "ANTENNA"),
        readonly=True,
    )
    try:
        if ant.nrows() < payload["used_meta"]["antenna_ids"].size:
            raise ValueError("ANTENNA subtable has fewer rows than used antennas")

        names = [as_text(item) for item in ant.getcol("NAME")]
        pos = np.asarray(ant.getcol("POSITION"), dtype=np.float64)

        if pos.shape[1] != 3:
            raise ValueError(f"ANTENNA POSITION shape mismatch: {pos.shape}")
        if np.max(np.abs(pos)) <= 0.0:
            raise ValueError("ANTENNA POSITION is all zero")

        for name in payload["used_meta"]["antenna_names"]:
            if as_text(name) not in names:
                warn(f"used antenna name not found in ANTENNA subtable: {name}")
    finally:
        ant.close()

    field = casacore_tables.table(
        os.path.join(output_ms, "FIELD"),
        readonly=True,
    )
    try:
        if field.nrows() < 1:
            raise ValueError("FIELD has no rows")

        phase_dir = np.asarray(field.getcell("PHASE_DIR", 0), dtype=np.float64)
        flat = phase_dir.reshape(-1)
        if flat.size < 2:
            raise ValueError(f"FIELD PHASE_DIR bad shape: {phase_dir.shape}")

        ra = float(flat[0])
        dec = float(flat[1])

        if not np.isfinite(ra) or not np.isfinite(dec):
            raise ValueError("FIELD PHASE_DIR has non-finite RA/Dec")
        if abs(ra - expected_ra) > 1e-8:
            raise ValueError(f"FIELD RA mismatch: {ra} != {expected_ra}")
        if abs(dec - expected_dec) > 1e-8:
            raise ValueError(f"FIELD Dec mismatch: {dec} != {expected_dec}")
    finally:
        field.close()

    print("\n========== MS SUBTABLE CHECK ==========")
    print("SPECTRAL_WINDOW : OK")
    print("POLARIZATION    : OK")
    print("DATA_DESCRIPTION: OK")
    print("ANTENNA         : OK")
    print("FIELD           : OK")
    print("=======================================")


def write_measurement_set(
    output_ms: str,
    payload: dict[str, Any],
    overwrite: bool = False,
    uvdata_constructor: str = "new",
) -> None:
    require_pyuvdata()
    require_casacore()
    if os.path.exists(output_ms) and not overwrite:
        raise FileExistsError(
            f"output MeasurementSet already exists: {output_ms}. "
            "Set overwrite=True to replace it."
        )

    uvd = build_uvdata_object(payload, constructor=uvdata_constructor)
    remove_existing_output(output_ms, overwrite=overwrite)

    try:
        uvd.write_ms(output_ms, clobber=overwrite)
        verify_ms_directory_layout(output_ms)
        verify_ms_with_casacore(output_ms, payload)
        verify_ms_subtables(output_ms, payload)
    except Exception:
        if os.path.exists(output_ms):
            warn(
                "MeasurementSet validation failed; removing incomplete output: "
                f"{output_ms}"
            )
            try:
                remove_existing_output(output_ms, overwrite=True)
            except Exception as cleanup_error:  # pragma: no cover
                warn(f"Failed to clean up invalid output {output_ms}: {cleanup_error}")
        raise


def parse_command_line(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert an MS-ready HDF5 file into a CASA MeasurementSet. "
            "Default behavior: overwrite output, include auto-correlations, "
            "allow partial polarizations, and use UVData.new."
        )
    )
    parser.add_argument(
        "input_h5",
        help="input HDF5 produced by test_with_antenna_uvw.py",
    )
    parser.add_argument(
        "output_ms",
        help="output MeasurementSet directory",
    )
    parser.add_argument(
        "--allow-uvw-warnings",
        action="store_true",
        help=(
            "downgrade UVW mismatch between polarization rows to warnings. "
            "Missing UVW is always fatal."
        ),
    )
    parser.add_argument(
        "--x-orientation",
        choices=["east", "north", "none"],
        default="east",
        help=(
            "assumed X-feed orientation for pyuvdata metadata. "
            "Use 'none' if unknown."
        ),
    )
    parser.add_argument(
        "--max-memory-gb",
        type=float,
        default=8.0,
        help="maximum estimated RAM allowed for packed arrays before writing MS",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="only inspect and pack the HDF5 layout, do not write a MeasurementSet",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "validate HDF5 schema and payload consistency only. "
            "This exits before writing a MeasurementSet."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_command_line(sys.argv[1:] if argv is None else argv)
    print_runtime_versions()
    require_h5py()

    if not os.path.isfile(args.input_h5):
        raise FileNotFoundError(args.input_h5)

    # Production defaults:
    #   - Always overwrite old MS output.
    #   - Always include auto-correlations, so 0&0 and 1&1 are written.
    #   - Allow missing polarization products as flagged data.
    #   - Use UVData.new to construct the UVData object.
    include_autos = True
    allow_partial_pols = True
    overwrite = True
    uvdata_constructor = "new"

    with h5py.File(args.input_h5, "r") as h5:
        validate_hdf5_input(h5)
        estimate_and_check_memory(
            h5,
            include_autos=include_autos,
            max_memory_gb=args.max_memory_gb,
        )
        payload = prepare_ms_payload(
            h5,
            include_autos=include_autos,
            allow_partial_pols=allow_partial_pols,
            allow_uvw_warnings=args.allow_uvw_warnings,
            x_orientation=args.x_orientation,
        )
        validate_payload_before_uvdata(payload)
        print_payload_summary(h5, payload, include_autos=include_autos)

        if args.validate_only:
            print("\n[VALIDATE-ONLY] HDF5 schema and payload validation passed.")
            return 0

        if args.dry_run:
            print("\n[DRY-RUN] Payload validation passed. Skipping MS write step.")
            return 0

        write_measurement_set(
            args.output_ms,
            payload,
            overwrite=overwrite,
            uvdata_constructor=uvdata_constructor,
        )

    print("\n[OK] MeasurementSet written:", args.output_ms)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print("\n[ERROR]")
        print(error)
        print("\n========== TRACEBACK ==========")
        traceback.print_exc()
        print("================================")
        raise SystemExit(1)
