import sys
import argparse

import numpy as np

if not hasattr(np, "typeDict"):
    np.typeDict = np.sctypeDict

import h5py


REQUIRED_PATHS = [
    "vis",
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
    "antenna/position_itrf_m",
    "antenna/dish_diameter_m",
    "antenna/position_is_placeholder",
    "field/source_name",
    "field/phase_center_ra_rad",
    "field/phase_center_dec_rad",
    "field/frame",
    "field/is_placeholder",
    "polarization/input_pol_id",
    "polarization/input_pol_name",
    "polarization/ms_export_mode",
    "polarization/corr_type",
    "polarization/corr_pol_i",
    "polarization/corr_pol_j",
    "ms_rows/time_index",
    "ms_rows/signal_baseline_index",
    "ms_rows/antenna1",
    "ms_rows/antenna2",
    "ms_rows/data_desc_id",
    "ms_rows/field_id",
    "ms_rows/scan_number",
    "ms_rows/row_has_missing_signal",
    "uvw/uvw_m",
    "uvw/is_placeholder",
    "ms_defaults/flag_default",
    "ms_defaults/weight_default",
    "ms_defaults/sigma_default",
    "ms_defaults/missing_signal_should_flag",
]

FORBIDDEN_ROOT_DATASETS = [
    "ANTENNA1",
    "ANTENNA2",
    "CHAN_FREQ",
    "CHAN_WIDTH",
    "TIME",
]

TIME_STEP_ATOL_DAY = 1e-12
TIME_STEP_ULP_FACTOR = 2.0
ANGLE_RANGE_ATOL_RAD = 1e-12
ALLOWED_FIELD_FRAMES = {"J2000", "ICRS", "UNKNOWN"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate whether an HDF5 file is ready for HDF5 -> CASA MS row-based conversion."
    )
    parser.add_argument("h5_file", help="Input HDF5 file to validate")
    parser.add_argument(
        "--show-rows",
        type=int,
        default=0,
        help="Print the first N ms_rows mappings",
    )
    parser.add_argument(
        "--check-data",
        action="store_true",
        help="Read a few /vis rows to confirm DATA access works",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Print strong warnings when placeholder metadata is still present",
    )
    return parser.parse_args()


def require_path(h5, path):
    normalized = path[1:] if path.startswith("/") else path
    if normalized not in h5:
        raise ValueError(f"missing required path: /{normalized}")
    return h5[normalized]


def read_scalar(dataset):
    value = dataset[()]

    if isinstance(value, np.ndarray):
        if value.shape != ():
            raise ValueError(
                f"expected scalar dataset at {dataset.name}, got shape {value.shape}"
            )
        value = value.item()

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, (bytes, np.bytes_)):
        value = value.decode("utf-8", errors="replace")

    return value


def decode_string(value):
    if isinstance(value, np.ndarray):
        if value.shape != ():
            raise ValueError(f"expected scalar string-like value, got shape {value.shape}")
        value = value.item()

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="replace")

    if isinstance(value, str):
        return value

    return str(value)


def decode_string_array(dataset):
    values = dataset[()]

    if values.ndim != 1:
        raise ValueError(
            f"{dataset.name} must be 1D string array, got shape {values.shape}"
        )

    decoded_values = []
    for index, value in enumerate(np.asarray(values, dtype=object)):
        try:
            decoded_values.append(decode_string(value))
        except Exception as exc:
            raise ValueError(
                f"{dataset.name}[{index}] could not be decoded as string: {value!r}"
            ) from exc

    return decoded_values


def coerce_bool_scalar(path, value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)

    raise ValueError(
        f"{path} must be bool-like scalar 0/1 or True/False, got "
        f"{value!r} ({type(value).__name__})"
    )


def print_header(title):
    print()
    print("=" * 16, title, "=" * 16)


def check_required_structure(h5):
    print_header("Required Structure")

    for path in REQUIRED_PATHS:
        require_path(h5, path)

    for dataset_name in FORBIDDEN_ROOT_DATASETS:
        if dataset_name in h5:
            raise ValueError(f"unexpected legacy root dataset: /{dataset_name}")

    print("[OK] required structure exists")


def check_vis_shape(h5):
    print_header("VIS")

    vis = require_path(h5, "vis")
    vis_shape = vis.shape
    vis_dtype = vis.dtype

    if len(vis_shape) != 3:
        raise ValueError(f"/vis must be 3D, got shape {vis_shape}")

    if vis_dtype not in (np.dtype(np.complex64), np.dtype(np.complex128)):
        raise ValueError(
            f"/vis dtype must be complex64 or complex128, got {vis_dtype}"
        )

    n_time, n_signal_baseline, nchan = vis_shape

    if n_time <= 0 or n_signal_baseline <= 0 or nchan <= 0:
        raise ValueError(f"/vis has non-positive shape: {vis_shape}")

    print("/vis shape          :", vis_shape)
    print("/vis dtype          :", vis_dtype)
    print("n_time              :", int(n_time))
    print("n_signal_baseline   :", int(n_signal_baseline))
    print("nchan               :", int(nchan))
    print("[OK] /vis shape valid")


def check_time_axis(h5):
    print_header("Time Axis")

    n_time = require_path(h5, "vis").shape[0]
    start_mjd = require_path(h5, "time/start_mjd")[()]
    center_mjd = require_path(h5, "time/center_mjd")[()]
    end_mjd = require_path(h5, "time/end_mjd")[()]
    interval_sec = float(read_scalar(require_path(h5, "time/interval_sec")))
    exposure_sec = float(read_scalar(require_path(h5, "time/exposure_sec")))

    expected_shape = (n_time,)
    if start_mjd.shape != expected_shape:
        raise ValueError(
            f"/time/start_mjd shape mismatch: {start_mjd.shape} != {expected_shape}"
        )
    if center_mjd.shape != expected_shape:
        raise ValueError(
            f"/time/center_mjd shape mismatch: {center_mjd.shape} != {expected_shape}"
        )
    if end_mjd.shape != expected_shape:
        raise ValueError(
            f"/time/end_mjd shape mismatch: {end_mjd.shape} != {expected_shape}"
        )

    if not np.all(np.isfinite(start_mjd)):
        bad_index = int(np.where(~np.isfinite(start_mjd))[0][0])
        raise ValueError(
            f"/time/start_mjd contains non-finite value at index {bad_index}"
        )
    if not np.all(np.isfinite(center_mjd)):
        bad_index = int(np.where(~np.isfinite(center_mjd))[0][0])
        raise ValueError(
            f"/time/center_mjd contains non-finite value at index {bad_index}"
        )
    if not np.all(np.isfinite(end_mjd)):
        bad_index = int(np.where(~np.isfinite(end_mjd))[0][0])
        raise ValueError(
            f"/time/end_mjd contains non-finite value at index {bad_index}"
        )

    bad_order = np.where(~((start_mjd < center_mjd) & (center_mjd < end_mjd)))[0]
    if bad_order.size:
        k = int(bad_order[0])
        raise ValueError(
            "time ordering invalid at index "
            f"{k}: start={start_mjd[k]}, center={center_mjd[k]}, end={end_mjd[k]}"
        )

    if not np.isfinite(interval_sec):
        raise ValueError(f"/time/interval_sec is not finite: {interval_sec}")
    if not np.isfinite(exposure_sec):
        raise ValueError(f"/time/exposure_sec is not finite: {exposure_sec}")
    if interval_sec <= 0.0:
        raise ValueError(f"/time/interval_sec must be > 0, got {interval_sec}")
    if exposure_sec <= 0.0:
        raise ValueError(f"/time/exposure_sec must be > 0, got {exposure_sec}")

    expected_interval_day = interval_sec / 86400.0
    time_spacing_day = float(
        max(
            np.max(np.spacing(start_mjd)),
            np.max(np.spacing(center_mjd)),
            np.max(np.spacing(end_mjd)),
        )
    )
    effective_step_atol_day = max(
        TIME_STEP_ATOL_DAY,
        TIME_STEP_ULP_FACTOR * time_spacing_day,
    )

    expected_center_mjd = 0.5 * (start_mjd + end_mjd)
    bad_center = np.where(
        np.abs(center_mjd - expected_center_mjd) > effective_step_atol_day
    )[0]
    if bad_center.size:
        k = int(bad_center[0])
        raise ValueError(
            f"/time/center_mjd midpoint mismatch at index {k}: "
            f"got {center_mjd[k]}, expected {(expected_center_mjd[k])}, "
            f"atol={effective_step_atol_day}"
        )

    actual_interval_day = end_mjd - start_mjd
    bad_interval = np.where(
        np.abs(actual_interval_day - expected_interval_day) > effective_step_atol_day
    )[0]
    if bad_interval.size:
        k = int(bad_interval[0])
        raise ValueError(
            f"/time interval mismatch at index {k}: "
            f"end_mjd-start_mjd={actual_interval_day[k]}, "
            f"expected {expected_interval_day}, atol={effective_step_atol_day}"
        )

    if n_time > 1:
        actual_step_day = np.diff(center_mjd)
        bad_step = np.where(
            np.abs(actual_step_day - expected_interval_day) > effective_step_atol_day
        )[0]
        if bad_step.size:
            k = int(bad_step[0])
            raise ValueError(
                "time step mismatch between /time/center_mjd entries "
                f"{k} and {k + 1}: got {actual_step_day[k]} days, "
                f"expected {expected_interval_day} days, "
                f"atol={effective_step_atol_day} days"
            )

        boundary_gap_day = start_mjd[1:] - end_mjd[:-1]
        bad_boundary = np.where(np.abs(boundary_gap_day) > effective_step_atol_day)[0]
        if bad_boundary.size:
            k = int(bad_boundary[0])
            raise ValueError(
                f"/time boundary continuity mismatch between index {k} and {k + 1}: "
                f"end_mjd[{k}]={end_mjd[k]}, start_mjd[{k + 1}]={start_mjd[k + 1]}, "
                f"difference={boundary_gap_day[k]}, atol={effective_step_atol_day}"
            )

    print("time start MJD      :", float(start_mjd[0]))
    print("time first center   :", float(center_mjd[0]))
    print("time last center    :", float(center_mjd[-1]))
    print("interval_sec        :", interval_sec)
    print("exposure_sec        :", exposure_sec)
    print("time interval day   :", expected_interval_day)
    print("time step atol day  :", effective_step_atol_day)
    print("center consistency checked      : yes")
    if n_time > 1:
        print("adjacent boundary continuity checked : yes")
    else:
        print("adjacent boundary continuity checked : single sample")
    print("[OK] time axis valid")


def check_frequency_axis(h5):
    print_header("Frequency Axis")

    nchan = require_path(h5, "vis").shape[2]
    group = require_path(h5, "frequency")
    chan_freq_hz = require_path(h5, "frequency/chan_freq_hz")[()]
    chan_width_hz = require_path(h5, "frequency/chan_width_hz")[()]
    ref_frequency_hz = float(
        read_scalar(require_path(h5, "frequency/ref_frequency_hz"))
    )
    nchan_scalar = int(read_scalar(require_path(h5, "frequency/nchan")))

    expected_shape = (nchan,)
    if chan_freq_hz.shape != expected_shape:
        raise ValueError(
            f"/frequency/chan_freq_hz shape mismatch: {chan_freq_hz.shape} != {expected_shape}"
        )
    if chan_width_hz.shape != expected_shape:
        raise ValueError(
            f"/frequency/chan_width_hz shape mismatch: {chan_width_hz.shape} != {expected_shape}"
        )
    if nchan_scalar != nchan:
        raise ValueError(f"/frequency/nchan mismatch: {nchan_scalar} != {nchan}")

    if not np.all(np.isfinite(chan_freq_hz)):
        bad_index = int(np.where(~np.isfinite(chan_freq_hz))[0][0])
        raise ValueError(
            f"/frequency/chan_freq_hz contains non-finite value at index {bad_index}"
        )
    if not np.all(np.isfinite(chan_width_hz)):
        bad_index = int(np.where(~np.isfinite(chan_width_hz))[0][0])
        raise ValueError(
            f"/frequency/chan_width_hz contains non-finite value at index {bad_index}"
        )
    if not np.all(chan_width_hz > 0):
        bad_index = int(np.where(~(chan_width_hz > 0))[0][0])
        raise ValueError(
            f"/frequency/chan_width_hz must be > 0 at index {bad_index}, got {chan_width_hz[bad_index]}"
        )
    if not np.isfinite(ref_frequency_hz):
        raise ValueError(
            f"/frequency/ref_frequency_hz is not finite: {ref_frequency_hz}"
        )

    for attr_name in ("input_unit", "fch1_original", "foff_original", "channel_order"):
        if attr_name not in group.attrs:
            raise ValueError(
                f"missing required attribute: /frequency.attrs['{attr_name}']"
            )

    channel_order = group.attrs["channel_order"]
    if isinstance(channel_order, (bytes, np.bytes_)):
        channel_order = channel_order.decode("utf-8", errors="replace")

    if not np.allclose(chan_width_hz, chan_width_hz[0], rtol=1e-12, atol=0.0):
        bad_index = int(
            np.where(
                ~np.isclose(chan_width_hz, chan_width_hz[0], rtol=1e-12, atol=0.0)
            )[0][0]
        )
        raise ValueError(
            f"/frequency/chan_width_hz is not uniform at index {bad_index}: "
            f"got {chan_width_hz[bad_index]}, expected {chan_width_hz[0]}"
        )

    if nchan >= 2:
        channel_diff_hz = np.diff(chan_freq_hz)

        if channel_order not in ("ascending", "descending"):
            raise ValueError(
                f"/frequency channel_order has unsupported value: "
                f"got '{channel_order}', expected 'ascending' or 'descending'"
            )

        if channel_order == "ascending":
            bad_diff = np.where(channel_diff_hz <= 0)[0]
            expected_order = "ascending"
        else:
            bad_diff = np.where(channel_diff_hz >= 0)[0]
            expected_order = "descending"

        if bad_diff.size:
            k = int(bad_diff[0])
            raise ValueError(
                f"/frequency/chan_freq_hz direction mismatch at diff index {k}: "
                f"channel_order='{channel_order}', diff={channel_diff_hz[k]}, "
                f"expected strictly {expected_order}"
            )

    print("first channel Hz    :", float(chan_freq_hz[0]))
    print("last channel Hz     :", float(chan_freq_hz[-1]))
    print("channel width Hz    :", float(chan_width_hz[0]))
    print("channel_order       :", channel_order)
    print("[OK] frequency axis valid")


def check_antenna_table(h5):
    print_header("Antenna Table")

    antenna_id = require_path(h5, "antenna/id")[()]
    name_ds = require_path(h5, "antenna/name")
    station_ds = require_path(h5, "antenna/station")
    position_itrf_m = require_path(h5, "antenna/position_itrf_m")[()]
    dish_diameter_m = require_path(h5, "antenna/dish_diameter_m")[()]
    position_is_placeholder = int(
        read_scalar(require_path(h5, "antenna/position_is_placeholder"))
    )

    if antenna_id.ndim != 1:
        raise ValueError(
            f"/antenna/id shape mismatch: got {antenna_id.shape}, expected 1D"
        )

    n_ant = int(antenna_id.size)
    if n_ant <= 0:
        raise ValueError(f"/antenna/id must contain at least one antenna, got {n_ant}")

    expected_vector_shape = (n_ant,)
    expected_position_shape = (n_ant, 3)

    if name_ds.shape != expected_vector_shape:
        raise ValueError(
            f"/antenna/name shape mismatch: got {name_ds.shape}, expected {expected_vector_shape}"
        )
    if station_ds.shape != expected_vector_shape:
        raise ValueError(
            f"/antenna/station shape mismatch: got {station_ds.shape}, expected {expected_vector_shape}"
        )
    if position_itrf_m.shape != expected_position_shape:
        raise ValueError(
            f"/antenna/position_itrf_m shape mismatch: got {position_itrf_m.shape}, expected {expected_position_shape}"
        )
    if dish_diameter_m.shape != expected_vector_shape:
        raise ValueError(
            f"/antenna/dish_diameter_m shape mismatch: got {dish_diameter_m.shape}, expected {expected_vector_shape}"
        )

    if position_is_placeholder not in (0, 1):
        raise ValueError(
            f"/antenna/position_is_placeholder value mismatch: got {position_is_placeholder}, expected 0 or 1"
        )

    if not np.issubdtype(antenna_id.dtype, np.integer):
        raise ValueError(
            f"/antenna/id dtype mismatch: got {antenna_id.dtype}, expected integer type"
        )

    unique_ids, unique_counts = np.unique(antenna_id, return_counts=True)
    duplicate_ids = unique_ids[unique_counts > 1]
    if duplicate_ids.size:
        duplicate_id = int(duplicate_ids[0])
        duplicate_count = int(unique_counts[unique_ids == duplicate_id][0])
        raise ValueError(
            f"/antenna/id contains duplicate value {duplicate_id}, count={duplicate_count}, expected unique antenna ids"
        )

    if not np.all(np.isfinite(position_itrf_m)):
        bad_row, bad_col = np.argwhere(~np.isfinite(position_itrf_m))[0]
        raise ValueError(
            f"/antenna/position_itrf_m contains non-finite value at index ({int(bad_row)}, {int(bad_col)})"
        )

    if not np.all(np.isfinite(dish_diameter_m)):
        bad_index = int(np.where(~np.isfinite(dish_diameter_m))[0][0])
        raise ValueError(
            f"/antenna/dish_diameter_m contains non-finite value at index {bad_index}: {dish_diameter_m[bad_index]}"
        )

    if not np.all(dish_diameter_m > 0):
        bad_index = int(np.where(~(dish_diameter_m > 0))[0][0])
        raise ValueError(
            f"/antenna/dish_diameter_m must be > 0 at index {bad_index}: got {dish_diameter_m[bad_index]}"
        )

    decode_string_array(name_ds)
    decode_string_array(station_ds)

    if position_is_placeholder == 1:
        print(
            "WARNING: Antenna positions are placeholder. MS can be format-tested, "
            "but physical imaging is not reliable."
        )

    print("antenna count       :", n_ant)
    print("antenna id range    :", f"{int(np.min(antenna_id))} -> {int(np.max(antenna_id))}")
    print("position placeholder:", position_is_placeholder)
    print("dish diameter first value:", float(dish_diameter_m[0]))
    print("[OK] antenna table valid")


def check_field_table(h5):
    print_header("Field Table")

    field_group = require_path(h5, "field")
    source_name = decode_string(read_scalar(require_path(h5, "field/source_name")))
    frame = decode_string(read_scalar(require_path(h5, "field/frame")))
    phase_center_ra_rad = float(read_scalar(require_path(h5, "field/phase_center_ra_rad")))
    phase_center_dec_rad = float(
        read_scalar(require_path(h5, "field/phase_center_dec_rad"))
    )
    is_placeholder = int(read_scalar(require_path(h5, "field/is_placeholder")))

    if not np.isfinite(phase_center_ra_rad):
        raise ValueError(
            f"/field/phase_center_ra_rad must be finite, got {phase_center_ra_rad}"
        )
    if not np.isfinite(phase_center_dec_rad):
        raise ValueError(
            f"/field/phase_center_dec_rad must be finite, got {phase_center_dec_rad}"
        )
    if is_placeholder not in (0, 1):
        raise ValueError(
            f"/field/is_placeholder value mismatch: got {is_placeholder}, expected 0 or 1"
        )

    if frame not in ALLOWED_FIELD_FRAMES:
        print(
            f"WARNING: /field/frame='{frame}' is not one of "
            f"{sorted(ALLOWED_FIELD_FRAMES)}."
        )

    if is_placeholder == 0:
        if phase_center_ra_rad < -ANGLE_RANGE_ATOL_RAD or phase_center_ra_rad > (
            2.0 * np.pi + ANGLE_RANGE_ATOL_RAD
        ):
            raise ValueError(
                f"/field/phase_center_ra_rad out of range: got {phase_center_ra_rad}, "
                f"expected within [0, 2*pi] with atol {ANGLE_RANGE_ATOL_RAD}"
            )

        dec_min = -0.5 * np.pi - ANGLE_RANGE_ATOL_RAD
        dec_max = 0.5 * np.pi + ANGLE_RANGE_ATOL_RAD
        if phase_center_dec_rad < dec_min or phase_center_dec_rad > dec_max:
            raise ValueError(
                f"/field/phase_center_dec_rad out of range: got {phase_center_dec_rad}, "
                f"expected within [-pi/2, pi/2] with atol {ANGLE_RANGE_ATOL_RAD}"
            )
    else:
        print(
            "WARNING: Field phase center is placeholder. MS can be format-tested, "
            "but imaging is not physically meaningful."
        )

    if "src_raj_header" in field_group.attrs:
        print(
            "field attr src_raj_header:",
            decode_string(field_group.attrs["src_raj_header"]),
            "(header raw value, not used directly as radians)",
        )
    if "src_dej_header" in field_group.attrs:
        print(
            "field attr src_dej_header:",
            decode_string(field_group.attrs["src_dej_header"]),
            "(header raw value, not used directly as radians)",
        )

    print("source_name         :", source_name)
    print("frame               :", frame)
    print("phase_center_ra_rad :", phase_center_ra_rad)
    print("phase_center_dec_rad:", phase_center_dec_rad)
    print("field placeholder   :", is_placeholder)
    print("[OK] field table valid")


def check_polarization_table(h5):
    print_header("Polarization Table")

    input_pol_id = require_path(h5, "polarization/input_pol_id")[()]
    input_pol_name_ds = require_path(h5, "polarization/input_pol_name")
    ms_export_mode = decode_string(
        read_scalar(require_path(h5, "polarization/ms_export_mode"))
    )
    corr_type_ds = require_path(h5, "polarization/corr_type")
    corr_pol_i = require_path(h5, "polarization/corr_pol_i")[()]
    corr_pol_j = require_path(h5, "polarization/corr_pol_j")[()]
    signal_baseline_index = require_path(h5, "ms_rows/signal_baseline_index")[()]
    polarization_pairs = require_path(h5, "baseline/polarization_pairs")[()]
    ms_rows_group = require_path(h5, "ms_rows")

    if input_pol_id.ndim != 1:
        raise ValueError(
            f"/polarization/input_pol_id shape mismatch: got {input_pol_id.shape}, expected 1D"
        )
    if input_pol_name_ds.shape != input_pol_id.shape:
        raise ValueError(
            f"/polarization/input_pol_name shape mismatch: got {input_pol_name_ds.shape}, expected {input_pol_id.shape}"
        )

    corr_type = decode_string_array(corr_type_ds)
    input_pol_name = decode_string_array(input_pol_name_ds)

    if corr_pol_i.ndim != 1:
        raise ValueError(
            f"/polarization/corr_pol_i shape mismatch: got {corr_pol_i.shape}, expected 1D"
        )
    if corr_pol_j.ndim != 1:
        raise ValueError(
            f"/polarization/corr_pol_j shape mismatch: got {corr_pol_j.shape}, expected 1D"
        )
    if corr_pol_i.shape != corr_pol_j.shape:
        raise ValueError(
            f"/polarization/corr_pol_i and /polarization/corr_pol_j shape mismatch: "
            f"got {corr_pol_i.shape} and {corr_pol_j.shape}, expected equal shapes"
        )
    if corr_pol_i.shape[0] != len(corr_type):
        raise ValueError(
            f"/polarization corr length mismatch: got len(corr_type)={len(corr_type)}, "
            f"len(corr_pol_i)={corr_pol_i.shape[0]}, expected equal lengths"
        )

    if not np.all(np.isin(corr_pol_i, input_pol_id)):
        bad_index = int(np.where(~np.isin(corr_pol_i, input_pol_id))[0][0])
        raise ValueError(
            f"/polarization/corr_pol_i value at index {bad_index} not found in /polarization/input_pol_id: "
            f"got {corr_pol_i[bad_index]}, valid={input_pol_id.tolist()}"
        )
    if not np.all(np.isin(corr_pol_j, input_pol_id)):
        bad_index = int(np.where(~np.isin(corr_pol_j, input_pol_id))[0][0])
        raise ValueError(
            f"/polarization/corr_pol_j value at index {bad_index} not found in /polarization/input_pol_id: "
            f"got {corr_pol_j[bad_index]}, valid={input_pol_id.tolist()}"
        )

    if "export_pol_mode" not in ms_rows_group.attrs:
        raise ValueError("missing required attribute: /ms_rows.attrs['export_pol_mode']")

    ms_rows_export_pol_mode = decode_string(ms_rows_group.attrs["export_pol_mode"])
    if ms_rows_export_pol_mode != ms_export_mode:
        raise ValueError(
            f"/ms_rows.attrs['export_pol_mode'] mismatch: got '{ms_rows_export_pol_mode}', "
            f"expected '{ms_export_mode}' from /polarization/ms_export_mode"
        )

    if ms_export_mode != "XX_ONLY":
        raise ValueError(
            f"/polarization/ms_export_mode unsupported: got '{ms_export_mode}', "
            "expected 'XX_ONLY' because full-pol strict checking is not implemented yet"
        )

    if len(corr_type) != 1:
        raise ValueError(
            f"/polarization/corr_type length mismatch for XX_ONLY: got {len(corr_type)}, expected 1"
        )
    if int(corr_pol_i[0]) != 0:
        raise ValueError(
            f"/polarization/corr_pol_i[0] mismatch for XX_ONLY: got {int(corr_pol_i[0])}, expected 0"
        )
    if int(corr_pol_j[0]) != 0:
        raise ValueError(
            f"/polarization/corr_pol_j[0] mismatch for XX_ONLY: got {int(corr_pol_j[0])}, expected 0"
        )

    if 0 not in input_pol_id:
        raise ValueError(
            f"/polarization/input_pol_id must contain 0 for XX_ONLY, got {input_pol_id.tolist()}"
        )

    pol0_name = input_pol_name[int(np.where(input_pol_id == 0)[0][0])]
    expected_corr_type = pol0_name + pol0_name
    if corr_type[0] not in ("XX", expected_corr_type):
        raise ValueError(
            f"/polarization/corr_type[0] mismatch for XX_ONLY: got '{corr_type[0]}', "
            f"expected 'XX' or '{expected_corr_type}'"
        )

    selected_pol_pairs = polarization_pairs[signal_baseline_index]
    bad_selected_pol = np.where(
        (selected_pol_pairs[:, 0] != 0) | (selected_pol_pairs[:, 1] != 0)
    )[0]
    if bad_selected_pol.size:
        row = int(bad_selected_pol[0])
        raise ValueError(
            f"/baseline/polarization_pairs selected by /ms_rows contains non-(0,0) pair at row {row}: "
            f"got {tuple(int(x) for x in selected_pol_pairs[row])}, expected (0, 0)"
        )

    input_pols_summary = [
        f"{int(pol_id)}:{pol_name}"
        for pol_id, pol_name in zip(input_pol_id.tolist(), input_pol_name)
    ]

    print("input polarizations :", input_pols_summary)
    print("ms_export_mode      :", ms_export_mode)
    print("corr_type           :", corr_type)
    print("[OK] polarization table valid")


def check_signal_axis(h5):
    print_header("Signal Axis")

    present = require_path(h5, "signal/present")[()]
    input_signal_no = require_path(h5, "signal/input_signal_no")[()]
    antenna_id = require_path(h5, "signal/antenna_id")[()]
    polarization_id = require_path(h5, "signal/polarization_id")[()]
    file_ds = require_path(h5, "signal/file")
    antenna_catalog = require_path(h5, "antenna/id")[()]
    input_pol_catalog = require_path(h5, "polarization/input_pol_id")[()]

    if antenna_catalog.ndim != 1:
        raise ValueError(f"/antenna/id must be 1D, got shape {antenna_catalog.shape}")
    if input_pol_catalog.ndim != 1:
        raise ValueError(
            f"/polarization/input_pol_id must be 1D, got shape {input_pol_catalog.shape}"
        )

    expected_signal_count = int(antenna_catalog.size * input_pol_catalog.size)
    expected_shape = (expected_signal_count,)

    if present.shape != expected_shape:
        raise ValueError(
            f"/signal/present shape mismatch: {present.shape} != {expected_shape}"
        )
    if input_signal_no.shape != expected_shape:
        raise ValueError(
            f"/signal/input_signal_no shape mismatch: {input_signal_no.shape} != {expected_shape}"
        )
    if antenna_id.shape != expected_shape:
        raise ValueError(
            f"/signal/antenna_id shape mismatch: {antenna_id.shape} != {expected_shape}"
        )
    if polarization_id.shape != expected_shape:
        raise ValueError(
            f"/signal/polarization_id shape mismatch: {polarization_id.shape} != {expected_shape}"
        )
    if file_ds.shape != expected_shape:
        raise ValueError(
            f"/signal/file shape mismatch: {file_ds.shape} != {expected_shape}"
        )

    if not np.all(np.isin(present, [0, 1, False, True])):
        bad_index = int(np.where(~np.isin(present, [0, 1, False, True]))[0][0])
        raise ValueError(
            f"/signal/present must contain only 0 or 1, bad value at index {bad_index}: {present[bad_index]}"
        )

    expected_signal_no = np.arange(
        1, expected_signal_count + 1, dtype=input_signal_no.dtype
    )
    if not np.array_equal(input_signal_no, expected_signal_no):
        bad_index = int(np.where(input_signal_no != expected_signal_no)[0][0])
        raise ValueError(
            f"/signal/input_signal_no mismatch at index {bad_index}: "
            f"got {input_signal_no[bad_index]}, expected {expected_signal_no[bad_index]}"
        )

    if not np.all(np.isin(polarization_id, [0, 1])):
        bad_index = int(np.where(~np.isin(polarization_id, [0, 1]))[0][0])
        raise ValueError(
            f"/signal/polarization_id must contain only 0 or 1, bad value at index {bad_index}: {polarization_id[bad_index]}"
        )

    if not np.all(np.isin(polarization_id, input_pol_catalog)):
        bad_index = int(np.where(~np.isin(polarization_id, input_pol_catalog))[0][0])
        raise ValueError(
            f"/signal/polarization_id value at index {bad_index} is not in /polarization/input_pol_id: {polarization_id[bad_index]}"
        )

    antenna_min = int(np.min(antenna_catalog))
    antenna_max = int(np.max(antenna_catalog))
    if not np.all(np.isin(antenna_id, antenna_catalog)):
        bad_index = int(np.where(~np.isin(antenna_id, antenna_catalog))[0][0])
        raise ValueError(
            f"/signal/antenna_id value at index {bad_index} is not in /antenna/id: {antenna_id[bad_index]}"
        )
    if np.any(antenna_id < antenna_min) or np.any(antenna_id > antenna_max):
        bad_index = int(
            np.where((antenna_id < antenna_min) | (antenna_id > antenna_max))[0][0]
        )
        raise ValueError(
            f"/signal/antenna_id out of range at index {bad_index}: "
            f"{antenna_id[bad_index]} not in [{antenna_min}, {antenna_max}]"
        )

    present_count = int(np.count_nonzero(present))
    missing_count = int(present.size - present_count)

    print("total signal count   :", int(present.size))
    print("present signal count :", present_count)
    print("missing signal count :", missing_count)
    print("signal mapping:")
    for index in range(present.size):
        print(
            f"  signal {index + 1:02d}: "
            f"present={int(bool(present[index]))} "
            f"antenna_id={int(antenna_id[index])} "
            f"polarization_id={int(polarization_id[index])}"
        )
    print("[OK] signal axis valid")


def check_baseline_axis(h5):
    print_header("Baseline Axis")

    n_signal_baseline = require_path(h5, "vis").shape[1]
    signal_pairs = require_path(h5, "baseline/signal_pairs")[()]
    antenna_pairs = require_path(h5, "baseline/antenna_pairs")[()]
    polarization_pairs = require_path(h5, "baseline/polarization_pairs")[()]
    signal_antenna_id = require_path(h5, "signal/antenna_id")[()]
    signal_polarization_id = require_path(h5, "signal/polarization_id")[()]
    antenna_catalog = require_path(h5, "antenna/id")[()]

    expected_shape = (n_signal_baseline, 2)
    if signal_pairs.shape != expected_shape:
        raise ValueError(
            f"/baseline/signal_pairs shape mismatch: {signal_pairs.shape} != {expected_shape}"
        )
    if antenna_pairs.shape != expected_shape:
        raise ValueError(
            f"/baseline/antenna_pairs shape mismatch: {antenna_pairs.shape} != {expected_shape}"
        )
    if polarization_pairs.shape != expected_shape:
        raise ValueError(
            f"/baseline/polarization_pairs shape mismatch: {polarization_pairs.shape} != {expected_shape}"
        )

    n_signal = int(signal_antenna_id.size)
    signal_i = signal_pairs[:, 0]
    signal_j = signal_pairs[:, 1]

    bad_signal_i = np.where((signal_i < 0) | (signal_i >= n_signal))[0]
    if bad_signal_i.size:
        b = int(bad_signal_i[0])
        raise ValueError(
            f"/baseline/signal_pairs[{b}, 0] out of range: {signal_i[b]} not in [0, {n_signal})"
        )

    bad_signal_j = np.where((signal_j < 0) | (signal_j >= n_signal))[0]
    if bad_signal_j.size:
        b = int(bad_signal_j[0])
        raise ValueError(
            f"/baseline/signal_pairs[{b}, 1] out of range: {signal_j[b]} not in [0, {n_signal})"
        )

    bad_order = np.where(signal_i > signal_j)[0]
    if bad_order.size:
        b = int(bad_order[0])
        raise ValueError(
            f"/baseline/signal_pairs row {b} violates signal_i <= signal_j: "
            f"({signal_i[b]}, {signal_j[b]})"
        )

    expected_ant_pairs = np.column_stack(
        (signal_antenna_id[signal_i], signal_antenna_id[signal_j])
    )
    expected_pol_pairs = np.column_stack(
        (signal_polarization_id[signal_i], signal_polarization_id[signal_j])
    )

    bad_ant = np.where(np.any(antenna_pairs != expected_ant_pairs, axis=1))[0]
    if bad_ant.size:
        b = int(bad_ant[0])
        raise ValueError(
            f"/baseline/antenna_pairs row {b} mismatch: "
            f"got {tuple(int(x) for x in antenna_pairs[b])}, "
            f"expected {tuple(int(x) for x in expected_ant_pairs[b])}"
        )

    bad_pol = np.where(np.any(polarization_pairs != expected_pol_pairs, axis=1))[0]
    if bad_pol.size:
        b = int(bad_pol[0])
        raise ValueError(
            f"/baseline/polarization_pairs row {b} mismatch: "
            f"got {tuple(int(x) for x in polarization_pairs[b])}, "
            f"expected {tuple(int(x) for x in expected_pol_pairs[b])}"
        )

    pol00_mask = np.all(polarization_pairs == np.array([0, 0]), axis=1)
    unique_ant_pairs_pol00 = np.unique(antenna_pairs[pol00_mask], axis=0)
    n_unique_ant_pairs_pol00 = int(unique_ant_pairs_pol00.shape[0])
    expected_pol00_count = int(
        antenna_catalog.size * (antenna_catalog.size + 1) // 2
    )
    if n_unique_ant_pairs_pol00 != expected_pol00_count:
        raise ValueError(
            "unexpected unique physical baseline count for pol0-pol0: "
            f"{n_unique_ant_pairs_pol00} != {expected_pol00_count}"
        )

    print("baseline count       :", int(n_signal_baseline))
    print(
        "first baseline       :",
        f"signal={tuple(int(x) for x in signal_pairs[0])} "
        f"antenna={tuple(int(x) for x in antenna_pairs[0])} "
        f"pol={tuple(int(x) for x in polarization_pairs[0])}",
    )
    print(
        "last baseline        :",
        f"signal={tuple(int(x) for x in signal_pairs[-1])} "
        f"antenna={tuple(int(x) for x in antenna_pairs[-1])} "
        f"pol={tuple(int(x) for x in polarization_pairs[-1])}",
    )
    print("unique pol0-pol0 baselines :", n_unique_ant_pairs_pol00)
    print("[OK] baseline mapping valid")


def check_ms_rows(h5):
    print_header("MS Rows")

    n_time = require_path(h5, "vis").shape[0]
    n_signal_baseline = require_path(h5, "vis").shape[1]
    time_index = require_path(h5, "ms_rows/time_index")[()]
    signal_baseline_index = require_path(h5, "ms_rows/signal_baseline_index")[()]
    antenna1 = require_path(h5, "ms_rows/antenna1")[()]
    antenna2 = require_path(h5, "ms_rows/antenna2")[()]
    data_desc_id = require_path(h5, "ms_rows/data_desc_id")[()]
    field_id = require_path(h5, "ms_rows/field_id")[()]
    scan_number = require_path(h5, "ms_rows/scan_number")[()]
    row_has_missing_signal = require_path(h5, "ms_rows/row_has_missing_signal")[()]
    antenna_catalog = require_path(h5, "antenna/id")[()]
    ms_export_mode_dataset = read_scalar(
        require_path(h5, "polarization/ms_export_mode")
    )

    group = require_path(h5, "ms_rows")
    for attr_name in ("selected_baseline_count", "n_ms_rows", "export_pol_mode"):
        if attr_name not in group.attrs:
            raise ValueError(
                f"missing required attribute: /ms_rows.attrs['{attr_name}']"
            )

    selected_baseline_count = int(group.attrs["selected_baseline_count"])
    n_ms_rows_attr = int(group.attrs["n_ms_rows"])
    export_pol_mode = group.attrs["export_pol_mode"]
    if isinstance(export_pol_mode, (bytes, np.bytes_)):
        export_pol_mode = export_pol_mode.decode("utf-8", errors="replace")

    n_ms_rows = int(time_index.size)
    expected_shape = (n_ms_rows,)
    for path, array in (
        ("/ms_rows/time_index", time_index),
        ("/ms_rows/signal_baseline_index", signal_baseline_index),
        ("/ms_rows/antenna1", antenna1),
        ("/ms_rows/antenna2", antenna2),
        ("/ms_rows/data_desc_id", data_desc_id),
        ("/ms_rows/field_id", field_id),
        ("/ms_rows/scan_number", scan_number),
        ("/ms_rows/row_has_missing_signal", row_has_missing_signal),
    ):
        if array.shape != expected_shape:
            raise ValueError(
                f"{path} shape mismatch: {array.shape} != {expected_shape}"
            )

    if n_ms_rows_attr != n_ms_rows:
        raise ValueError(
            f"/ms_rows attr n_ms_rows mismatch: {n_ms_rows_attr} != {n_ms_rows}"
        )

    if export_pol_mode != ms_export_mode_dataset:
        raise ValueError(
            "export_pol_mode mismatch between /ms_rows attrs and "
            f"/polarization/ms_export_mode: '{export_pol_mode}' != '{ms_export_mode_dataset}'"
        )

    bad_time = np.where((time_index < 0) | (time_index >= n_time))[0]
    if bad_time.size:
        row = int(bad_time[0])
        raise ValueError(
            f"/ms_rows/time_index row {row} out of range: {time_index[row]} not in [0, {n_time})"
        )

    bad_baseline = np.where(
        (signal_baseline_index < 0) | (signal_baseline_index >= n_signal_baseline)
    )[0]
    if bad_baseline.size:
        row = int(bad_baseline[0])
        raise ValueError(
            f"/ms_rows/signal_baseline_index row {row} out of range: "
            f"{signal_baseline_index[row]} not in [0, {n_signal_baseline})"
        )

    if not np.all(np.isin(antenna1, antenna_catalog)):
        row = int(np.where(~np.isin(antenna1, antenna_catalog))[0][0])
        raise ValueError(
            f"/ms_rows/antenna1 row {row} is not in /antenna/id: {antenna1[row]}"
        )
    if not np.all(np.isin(antenna2, antenna_catalog)):
        row = int(np.where(~np.isin(antenna2, antenna_catalog))[0][0])
        raise ValueError(
            f"/ms_rows/antenna2 row {row} is not in /antenna/id: {antenna2[row]}"
        )

    bad_ant_order = np.where(antenna1 > antenna2)[0]
    if bad_ant_order.size:
        row = int(bad_ant_order[0])
        raise ValueError(
            f"/ms_rows antenna order invalid at row {row}: antenna1={antenna1[row]}, antenna2={antenna2[row]}"
        )

    if not np.all(data_desc_id == 0):
        row = int(np.where(data_desc_id != 0)[0][0])
        raise ValueError(
            f"/ms_rows/data_desc_id must be all 0, bad row {row}: {data_desc_id[row]}"
        )
    if not np.all(field_id == 0):
        row = int(np.where(field_id != 0)[0][0])
        raise ValueError(
            f"/ms_rows/field_id must be all 0, bad row {row}: {field_id[row]}"
        )
    if not np.all(scan_number == 1):
        row = int(np.where(scan_number != 1)[0][0])
        raise ValueError(
            f"/ms_rows/scan_number must be all 1, bad row {row}: {scan_number[row]}"
        )

    unique_selected_baselines = int(np.unique(signal_baseline_index).size)
    if selected_baseline_count != unique_selected_baselines:
        raise ValueError(
            f"/ms_rows attr selected_baseline_count mismatch: "
            f"{selected_baseline_count} != {unique_selected_baselines}"
        )

    if export_pol_mode == "XX_ONLY":
        expected_baseline_count = int(
            antenna_catalog.size * (antenna_catalog.size + 1) // 2
        )
        expected_n_ms_rows = int(n_time * expected_baseline_count)
        if selected_baseline_count != expected_baseline_count:
            raise ValueError(
                f"selected_baseline_count mismatch for XX_ONLY: "
                f"{selected_baseline_count} != {expected_baseline_count}"
            )
        if n_ms_rows != expected_n_ms_rows:
            raise ValueError(
                f"n_ms_rows mismatch for XX_ONLY: {n_ms_rows} != {expected_n_ms_rows}"
            )

    print("n_ms_rows           :", n_ms_rows)
    print("selected_baseline_count :", selected_baseline_count)
    print("export_pol_mode     :", export_pol_mode)
    print("[OK] ms_rows valid")


def check_ms_rows_time_baseline_completeness(h5):
    print_header("MS Rows Completeness")

    n_time = int(require_path(h5, "vis").shape[0])
    n_ms_rows = int(require_path(h5, "ms_rows/time_index").shape[0])
    time_index = require_path(h5, "ms_rows/time_index")[()]
    signal_baseline_index = require_path(h5, "ms_rows/signal_baseline_index")[()]
    antenna_id = require_path(h5, "antenna/id")[()]
    ms_rows_group = require_path(h5, "ms_rows")
    selected_baseline_count = int(ms_rows_group.attrs["selected_baseline_count"])
    export_pol_mode = decode_string(ms_rows_group.attrs["export_pol_mode"])

    expected_n_ms_rows = int(n_time * selected_baseline_count)
    if n_ms_rows != expected_n_ms_rows:
        raise ValueError(
            f"/ms_rows row count mismatch: got {n_ms_rows}, expected "
            f"n_time * selected_baseline_count = {n_time} * {selected_baseline_count} = {expected_n_ms_rows}"
        )

    counts_per_time = np.bincount(time_index.astype(np.int64), minlength=n_time)
    bad_count = np.where(counts_per_time != selected_baseline_count)[0]
    if bad_count.size:
        t = int(bad_count[0])
        raise ValueError(
            f"/ms_rows/time_index completeness mismatch at time index {t}: "
            f"got {int(counts_per_time[t])} rows, expected {selected_baseline_count}"
        )

    reference_rows = np.where(time_index == 0)[0]
    if reference_rows.size != selected_baseline_count:
        raise ValueError(
            f"/ms_rows reference time block mismatch at time index 0: "
            f"got {reference_rows.size} rows, expected {selected_baseline_count}"
        )

    reference_baselines = np.sort(signal_baseline_index[reference_rows])
    for t in range(n_time):
        current_rows = np.where(time_index == t)[0]
        if current_rows.size != selected_baseline_count:
            raise ValueError(
                f"/ms_rows/time_index completeness mismatch at time index {t}: "
                f"got {current_rows.size} rows, expected {selected_baseline_count}"
            )

        current_baselines = np.sort(signal_baseline_index[current_rows])
        if not np.array_equal(current_baselines, reference_baselines):
            mismatch_index = int(
                np.where(current_baselines != reference_baselines)[0][0]
            )
            raise ValueError(
                f"/ms_rows/signal_baseline_index baseline set mismatch at time index {t}: "
                f"got baseline {int(current_baselines[mismatch_index])}, "
                f"expected {int(reference_baselines[mismatch_index])} at sorted position {mismatch_index}"
            )

    if export_pol_mode == "XX_ONLY":
        n_ant = int(antenna_id.size)
        expected_selected_baseline_count = int(n_ant * (n_ant + 1) // 2)
        if selected_baseline_count != expected_selected_baseline_count:
            raise ValueError(
                f"/ms_rows.attrs['selected_baseline_count'] mismatch for XX_ONLY: "
                f"got {selected_baseline_count}, expected {expected_selected_baseline_count}"
            )

    expected_time_index = np.repeat(
        np.arange(n_time, dtype=time_index.dtype),
        selected_baseline_count,
    )
    if not np.array_equal(time_index, expected_time_index):
        bad_row = int(np.where(time_index != expected_time_index)[0][0])
        raise ValueError(
            f"/ms_rows/time_index ordering mismatch at row {bad_row}: "
            f"got {int(time_index[bad_row])}, expected {int(expected_time_index[bad_row])}"
        )

    selected_baselines = signal_baseline_index[:selected_baseline_count]
    expected_signal_baseline_index = np.tile(selected_baselines, n_time)
    if not np.array_equal(signal_baseline_index, expected_signal_baseline_index):
        bad_row = int(
            np.where(signal_baseline_index != expected_signal_baseline_index)[0][0]
        )
        raise ValueError(
            f"/ms_rows/signal_baseline_index ordering mismatch at row {bad_row}: "
            f"got {int(signal_baseline_index[bad_row])}, "
            f"expected {int(expected_signal_baseline_index[bad_row])}"
        )

    print("n_time              :", n_time)
    print("selected_baseline_count :", selected_baseline_count)
    print("n_ms_rows           :", n_ms_rows)
    print(
        "first selected baselines:",
        selected_baselines[: min(10, selected_baselines.size)].tolist(),
    )
    print("[OK] each time has complete selected baseline set")


def check_row_mapping_consistency(h5):
    print_header("Row Mapping")

    signal_baseline_index = require_path(h5, "ms_rows/signal_baseline_index")[()]
    antenna1 = require_path(h5, "ms_rows/antenna1")[()]
    antenna2 = require_path(h5, "ms_rows/antenna2")[()]
    antenna_pairs = require_path(h5, "baseline/antenna_pairs")[()]
    polarization_pairs = require_path(h5, "baseline/polarization_pairs")[()]
    export_pol_mode = require_path(h5, "ms_rows").attrs["export_pol_mode"]

    if isinstance(export_pol_mode, (bytes, np.bytes_)):
        export_pol_mode = export_pol_mode.decode("utf-8", errors="replace")

    expected_ant_pairs = antenna_pairs[signal_baseline_index]
    bad_ant = np.where(
        (antenna1 != expected_ant_pairs[:, 0]) | (antenna2 != expected_ant_pairs[:, 1])
    )[0]
    if bad_ant.size:
        row = int(bad_ant[0])
        b = int(signal_baseline_index[row])
        raise ValueError(
            f"ms row mapping antenna mismatch at row {row}, baseline index {b}: "
            f"ms_rows=({antenna1[row]}, {antenna2[row]}), "
            f"baseline/antenna_pairs=({expected_ant_pairs[row, 0]}, {expected_ant_pairs[row, 1]})"
        )

    if export_pol_mode == "XX_ONLY":
        expected_pol_pairs = polarization_pairs[signal_baseline_index]
        bad_pol = np.where(
            (expected_pol_pairs[:, 0] != 0) | (expected_pol_pairs[:, 1] != 0)
        )[0]
        if bad_pol.size:
            row = int(bad_pol[0])
            b = int(signal_baseline_index[row])
            raise ValueError(
                f"XX_ONLY row mapping selects non-(0,0) polarization at row {row}, "
                f"baseline index {b}: {tuple(int(x) for x in expected_pol_pairs[row])}"
            )

    print("[OK] row mapping consistent")


def check_missing_signal_flags(h5):
    print_header("Missing Signal Flags")

    signal_present = require_path(h5, "signal/present")[()].astype(bool)
    signal_pairs = require_path(h5, "baseline/signal_pairs")[()]
    signal_baseline_index = require_path(h5, "ms_rows/signal_baseline_index")[()]
    row_has_missing_signal = require_path(
        h5, "ms_rows/row_has_missing_signal"
    )[()].astype(bool)

    row_signal_pairs = signal_pairs[signal_baseline_index]
    expected_missing = np.logical_or(
        ~signal_present[row_signal_pairs[:, 0]],
        ~signal_present[row_signal_pairs[:, 1]],
    )

    bad_rows = np.where(row_has_missing_signal != expected_missing)[0]
    if bad_rows.size:
        row = int(bad_rows[0])
        b = int(signal_baseline_index[row])
        signal_i = int(row_signal_pairs[row, 0])
        signal_j = int(row_signal_pairs[row, 1])
        raise ValueError(
            f"/ms_rows/row_has_missing_signal mismatch at row {row}, baseline index {b}: "
            f"signal_pair=({signal_i}, {signal_j}), "
            f"got {bool(row_has_missing_signal[row])}, expected {bool(expected_missing[row])}"
        )

    rows_with_missing = int(np.count_nonzero(row_has_missing_signal))
    rows_without_missing = int(row_has_missing_signal.size - rows_with_missing)

    print("rows with missing signal    :", rows_with_missing)
    print("rows without missing signal :", rows_without_missing)
    print("[OK] row_has_missing_signal valid")


def check_uvw(h5):
    print_header("UVW")

    n_ms_rows = require_path(h5, "ms_rows/time_index").shape[0]
    uvw_m = require_path(h5, "uvw/uvw_m")[()]
    is_placeholder = int(read_scalar(require_path(h5, "uvw/is_placeholder")))

    expected_shape = (n_ms_rows, 3)
    if uvw_m.shape != expected_shape:
        raise ValueError(
            f"/uvw/uvw_m shape mismatch: {uvw_m.shape} != {expected_shape}"
        )

    if is_placeholder not in (0, 1):
        raise ValueError(f"/uvw/is_placeholder must be 0 or 1, got {is_placeholder}")

    if is_placeholder == 1:
        print(
            "WARNING: UVW is placeholder. MS can be created for format testing, "
            "but imaging is not physically meaningful."
        )
    else:
        if not np.all(np.isfinite(uvw_m)):
            bad_row, bad_col = np.argwhere(~np.isfinite(uvw_m))[0]
            raise ValueError(
                f"/uvw/uvw_m contains non-finite value at row {int(bad_row)}, column {int(bad_col)}"
            )

    print("[OK] uvw structure valid")


def check_ms_defaults(h5):
    print_header("MS Defaults")

    flag_default = coerce_bool_scalar(
        "/ms_defaults/flag_default",
        read_scalar(require_path(h5, "ms_defaults/flag_default")),
    )
    weight_default = float(read_scalar(require_path(h5, "ms_defaults/weight_default")))
    sigma_default = float(read_scalar(require_path(h5, "ms_defaults/sigma_default")))
    missing_signal_should_flag = coerce_bool_scalar(
        "/ms_defaults/missing_signal_should_flag",
        read_scalar(require_path(h5, "ms_defaults/missing_signal_should_flag")),
    )

    if not np.isfinite(weight_default):
        raise ValueError(
            f"/ms_defaults/weight_default must be finite, got {weight_default}"
        )
    if not np.isfinite(sigma_default):
        raise ValueError(
            f"/ms_defaults/sigma_default must be finite, got {sigma_default}"
        )
    if weight_default <= 0.0:
        raise ValueError(
            f"/ms_defaults/weight_default must be > 0, got {weight_default}"
        )
    if sigma_default <= 0.0:
        raise ValueError(
            f"/ms_defaults/sigma_default must be > 0, got {sigma_default}"
        )

    print("flag_default             :", flag_default)
    print("weight_default           :", weight_default)
    print("sigma_default            :", sigma_default)
    print("missing_signal_should_flag:", missing_signal_should_flag)
    print("[OK] MS defaults valid")


def check_placeholders(h5, strict=False):
    print_header("Placeholders")

    antenna_placeholder = int(
        read_scalar(require_path(h5, "antenna/position_is_placeholder"))
    )
    field_placeholder = int(read_scalar(require_path(h5, "field/is_placeholder")))
    uvw_placeholder = int(read_scalar(require_path(h5, "uvw/is_placeholder")))

    for path, value in (
        ("/antenna/position_is_placeholder", antenna_placeholder),
        ("/field/is_placeholder", field_placeholder),
        ("/uvw/is_placeholder", uvw_placeholder),
    ):
        if value not in (0, 1):
            raise ValueError(f"{path} must be 0 or 1, got {value}")

    print("antenna position placeholder :", antenna_placeholder)
    print("field placeholder            :", field_placeholder)
    print("uvw placeholder              :", uvw_placeholder)

    if antenna_placeholder or field_placeholder or uvw_placeholder:
        if strict:
            print(
                "STRONG WARNING: placeholder metadata is still present. "
                "This HDF5 is MS-format-ready, but not physically imaging-ready."
            )
        else:
            print(
                "WARNING: placeholder metadata is still present. "
                "This HDF5 is MS-format-ready, but not physically imaging-ready."
            )


def show_example_rows(h5, n_rows):
    if n_rows <= 0:
        return

    print_header("Example Rows")

    n_ms_rows = require_path(h5, "ms_rows/time_index").shape[0]
    if n_ms_rows == 0:
        raise ValueError("/ms_rows contains zero rows")

    n_rows = min(int(n_rows), int(n_ms_rows))
    time_index = require_path(h5, "ms_rows/time_index")[:n_rows]
    signal_baseline_index = require_path(h5, "ms_rows/signal_baseline_index")[:n_rows]
    antenna1 = require_path(h5, "ms_rows/antenna1")[:n_rows]
    antenna2 = require_path(h5, "ms_rows/antenna2")[:n_rows]
    row_has_missing_signal = require_path(h5, "ms_rows/row_has_missing_signal")[:n_rows]
    center_mjd = require_path(h5, "time/center_mjd")[()]
    signal_pairs = require_path(h5, "baseline/signal_pairs")[()]
    antenna_pairs = require_path(h5, "baseline/antenna_pairs")[()]
    polarization_pairs = require_path(h5, "baseline/polarization_pairs")[()]

    for row in range(n_rows):
        t = int(time_index[row])
        b = int(signal_baseline_index[row])
        signal_pair = signal_pairs[b]
        ant_pair = antenna_pairs[b]
        pol_pair = polarization_pairs[b]

        print(f"row {row}:")
        print(f"    time_index = {t}")
        print(f"    center_mjd = {float(center_mjd[t])}")
        print(f"    signal_baseline_index = {b}")
        print(f"    signal_pair = ({int(signal_pair[0])}, {int(signal_pair[1])})")
        print(f"    antenna_pair = ({int(ant_pair[0])}, {int(ant_pair[1])})")
        print(f"    polarization_pair = ({int(pol_pair[0])}, {int(pol_pair[1])})")
        print(f"    ms antenna1 = {int(antenna1[row])}")
        print(f"    ms antenna2 = {int(antenna2[row])}")
        print(f"    data source = /vis[{t}, {b}, :]")
        print(f"    row_has_missing_signal = {bool(row_has_missing_signal[row])}")


def check_example_data_access(h5):
    print_header("Example DATA Access")

    vis = require_path(h5, "vis")
    nchan = vis.shape[2]
    n_ms_rows = require_path(h5, "ms_rows/time_index").shape[0]
    time_index = require_path(h5, "ms_rows/time_index")[()]
    signal_baseline_index = require_path(h5, "ms_rows/signal_baseline_index")[()]
    row_has_missing_signal = require_path(
        h5, "ms_rows/row_has_missing_signal"
    )[()].astype(bool)

    candidate_rows = [0, n_ms_rows // 2, n_ms_rows - 1]
    rows_to_check = []
    for row in candidate_rows:
        if row not in rows_to_check:
            rows_to_check.append(row)

    for row in rows_to_check:
        t = int(time_index[row])
        b = int(signal_baseline_index[row])
        data = vis[t, b, :]

        if data.shape != (nchan,):
            raise ValueError(
                f"/vis[{t}, {b}, :] shape mismatch: {data.shape} != ({nchan},)"
            )
        if not np.iscomplexobj(data):
            raise ValueError(
                f"/vis[{t}, {b}, :] is not complex data, dtype={data.dtype}"
            )

        finite_mask = np.isfinite(data)
        finite_ratio = float(np.count_nonzero(finite_mask) / data.size)
        if finite_ratio < 1.0:
            raise ValueError(
                f"/vis[{t}, {b}, :] contains non-finite values, finite_ratio={finite_ratio}"
            )

        abs_data = np.abs(data)
        print(f"row index          : {row}")
        print(f"vis index          : ({t}, {b})")
        print(f"data shape         : {data.shape}")
        print(f"abs(data).min      : {float(abs_data.min())}")
        print(f"abs(data).max      : {float(abs_data.max())}")
        print(f"finite ratio       : {finite_ratio}")
        if row_has_missing_signal[row]:
            print(
                "note               : row_has_missing_signal=True; zero-filled or flaggable data is acceptable."
            )
        print()

    print("[OK] sample /vis access valid")


def print_summary(h5):
    print_header("Summary")

    vis_shape = require_path(h5, "vis").shape
    n_time = int(vis_shape[0])
    n_signal_baseline = int(vis_shape[1])
    nchan = int(vis_shape[2])
    start_mjd = require_path(h5, "time/start_mjd")[()]
    end_mjd = require_path(h5, "time/end_mjd")[()]
    chan_freq_hz = require_path(h5, "frequency/chan_freq_hz")[()]
    present = require_path(h5, "signal/present")[()]
    ms_rows_group = require_path(h5, "ms_rows")
    export_pol_mode = ms_rows_group.attrs["export_pol_mode"]
    if isinstance(export_pol_mode, (bytes, np.bytes_)):
        export_pol_mode = export_pol_mode.decode("utf-8", errors="replace")

    selected_baseline_count = int(ms_rows_group.attrs["selected_baseline_count"])
    n_ms_rows = int(ms_rows_group.attrs["n_ms_rows"])
    antenna_placeholder = int(
        read_scalar(require_path(h5, "antenna/position_is_placeholder"))
    )
    field_placeholder = int(read_scalar(require_path(h5, "field/is_placeholder")))
    uvw_placeholder = int(read_scalar(require_path(h5, "uvw/is_placeholder")))
    present_count = int(np.count_nonzero(present))
    missing_count = int(present.size - present_count)

    if export_pol_mode == "XX_ONLY":
        print("[OK] HDF5 is MS-ready for minimal XX_ONLY conversion.")
    else:
        print(f"[OK] HDF5 is MS-ready for minimal {export_pol_mode} conversion.")

    print("HDF5 file           :", h5.filename)
    print("vis shape           :", vis_shape)
    print("n_time              :", n_time)
    print("n_signal_baseline   :", n_signal_baseline)
    print("nchan               :", nchan)
    print("selected_baseline_count :", selected_baseline_count)
    print("n_ms_rows           :", n_ms_rows)
    print(
        "frequency range     :",
        f"{float(chan_freq_hz[0])} -> {float(chan_freq_hz[-1])} Hz",
    )
    print(
        "time range          :",
        f"{float(start_mjd[0])} -> {float(end_mjd[-1])} MJD",
    )
    print("signal count        :", int(present.size))
    print("present signal count:", present_count)
    print("missing signal count:", missing_count)
    print("antenna placeholder :", antenna_placeholder)
    print("field placeholder   :", field_placeholder)
    print("uvw placeholder     :", uvw_placeholder)
    print()
    print("Next step:")
    print("    Implement hdf5_to_ms.py using /ms_rows mapping.")
    print("    For each MS row:")
    print("        DATA = /vis[time_index[row], signal_baseline_index[row], :]")
    print("        TIME = /time/center_mjd[time_index[row]]")
    print("        ANTENNA1 = /ms_rows/antenna1[row]")
    print("        ANTENNA2 = /ms_rows/antenna2[row]")
    print("        UVW = /uvw/uvw_m[row]")
    print("        FLAG = row_has_missing_signal or flag_default")


def main():
    args = parse_args()

    if args.show_rows < 0:
        raise ValueError(f"--show-rows must be >= 0, got {args.show_rows}")

    try:
        with h5py.File(args.h5_file, "r") as h5:
            check_required_structure(h5)
            check_vis_shape(h5)
            check_time_axis(h5)
            check_frequency_axis(h5)
            check_antenna_table(h5)
            check_field_table(h5)
            check_polarization_table(h5)
            check_signal_axis(h5)
            check_baseline_axis(h5)
            check_ms_rows(h5)
            check_ms_rows_time_baseline_completeness(h5)
            check_row_mapping_consistency(h5)
            check_missing_signal_flags(h5)
            check_uvw(h5)
            check_ms_defaults(h5)
            check_placeholders(h5, strict=args.strict)

            if args.show_rows:
                show_example_rows(h5, args.show_rows)

            if args.check_data:
                check_example_data_access(h5)

            print_summary(h5)
    except OSError as exc:
        raise ValueError(
            f"failed to open HDF5 file '{args.h5_file}': {exc}"
        ) from exc


if __name__ == "__main__":
    main()
