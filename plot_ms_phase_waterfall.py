#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_ms_phase_waterfall_cli.py

命令行直接运行版：

    python plot_ms_phase_waterfall_cli.py 0X 1Y xxxx.ms

例子：

    python plot_ms_phase_waterfall_cli.py 0X 1Y /home/carrylab/Downloads/conda/0705test.ms

这会画：

    ant0-X  x  conj(ant1-Y)

输出默认在 MS 同级目录：

    phase_waterfall_0X_1Y.png

图像含义：

    横轴：频率 MHz
    纵轴：这个 MS 文件对应 baseline 的完整持续时间 s
    颜色：相位 rad，范围 [-pi, pi]

也可以指定输出图片：

    python plot_ms_phase_waterfall_cli.py 0X 1Y xxxx.ms out.png

可选参数：

    --mode phase     画相位瀑布图，默认
    --mode amp       画幅度瀑布图
    --save-npy       额外保存 complex visibility / phase / freq / time 的 npy 文件
    --save-txt       额外保存 txt 摘要

依赖：
    推荐在安装了 python-casacore 的环境里运行。
    如果没有 python-casacore，但在 CASA 6 Python 环境里，也会尝试使用 casatools.table。
"""

from __future__ import print_function

import argparse
import os
import re
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# Table backend
# =============================================================================

class CasacoreTableBackend(object):
    """
    Thin wrapper around python-casacore table.
    """

    def __init__(self):
        from casacore.tables import table
        self._table_func = table
        self._tb = None

    def open(self, path):
        self.close()
        self._tb = self._table_func(path, readonly=True, ack=False)

    def close(self):
        if self._tb is not None:
            self._tb.close()
            self._tb = None

    def getcol(self, name):
        return self._tb.getcol(name)

    def getcell(self, name, row):
        return self._tb.getcell(name, int(row))

    def nrows(self):
        return self._tb.nrows()


class CasaToolsTableBackend(object):
    """
    Thin wrapper around CASA 6 casatools.table.
    """

    def __init__(self):
        from casatools import table
        self._tb = table()

    def open(self, path):
        self.close()
        self._tb.open(path, nomodify=True)

    def close(self):
        try:
            self._tb.close()
        except Exception:
            pass

    def getcol(self, name):
        return self._tb.getcol(name)

    def getcell(self, name, row):
        return self._tb.getcell(name, int(row))

    def nrows(self):
        return self._tb.nrows()


def get_table_backend():
    """
    Prefer python-casacore for normal command-line Python.

    Fallback to casatools.table if running under CASA 6 Python.
    """
    try:
        backend = CasacoreTableBackend()
        print("[OK] table backend: python-casacore")
        return backend
    except Exception as error1:
        try:
            backend = CasaToolsTableBackend()
            print("[OK] table backend: casatools")
            return backend
        except Exception as error2:
            raise RuntimeError(
                "Cannot open MeasurementSet tables. Please install python-casacore "
                "or run in CASA 6 Python environment.\n"
                "python-casacore error: {0}\n"
                "casatools error: {1}".format(error1, error2)
            )


# =============================================================================
# Parse input
# =============================================================================

def parse_signal_text(text):
    """
    Parse signal string.

    Supported:
        0X
        1Y
        ant0X
        ANT03Y
        0:x
        1-y

    Returns:
        antenna_id, pol_name

    pol_name is X or Y.
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
            "Bad signal format: {0}. Expected examples: 0X, 1Y, ant0X".format(text)
        )

    antenna_id = int(match.group(1))
    pol_name = match.group(2).upper()

    if pol_name not in ["X", "Y"]:
        raise ValueError(
            "Unsupported polarization {0}. Current script supports X/Y only.".format(pol_name)
        )

    return antenna_id, pol_name


def default_output_path(sig1, sig2, ms_file, mode):
    ms_path = os.path.abspath(ms_file.rstrip("/"))
    parent = os.path.dirname(ms_path)

    if mode == "amp":
        prefix = "amp_waterfall"
    else:
        prefix = "phase_waterfall"

    return os.path.join(parent, "{0}_{1}_{2}.png".format(prefix, sig1, sig2))


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description=(
            "Plot phase waterfall from a CASA MeasurementSet for a selected "
            "antenna/polarization pair. Example: python plot_ms_phase_waterfall_cli.py 0X 1Y test.ms"
        )
    )

    parser.add_argument(
        "signal1",
        help="first signal, for example 0X",
    )
    parser.add_argument(
        "signal2",
        help="second signal, for example 1Y",
    )
    parser.add_argument(
        "ms_file",
        help="input MeasurementSet path, for example /path/to/test.ms",
    )
    parser.add_argument(
        "out_png",
        nargs="?",
        default=None,
        help="optional output png path. If omitted, save next to the MS.",
    )
    parser.add_argument(
        "--mode",
        default="phase",
        choices=["phase", "amp"],
        help="plot mode: phase in radians [-pi, pi], or amplitude. Default: phase",
    )
    parser.add_argument(
        "--save-npy",
        action="store_true",
        help="also save complex visibility, phase, frequency and time axes as .npy",
    )
    parser.add_argument(
        "--save-txt",
        action="store_true",
        help="also save a text summary",
    )

    args = parser.parse_args(argv)

    args.signal1 = str(args.signal1).strip()
    args.signal2 = str(args.signal2).strip()
    args.ms_file = os.path.abspath(args.ms_file.rstrip("/"))

    if args.out_png is None:
        args.out_png = default_output_path(
            args.signal1,
            args.signal2,
            args.ms_file,
            args.mode,
        )
    else:
        args.out_png = os.path.abspath(args.out_png)

    return args


# =============================================================================
# Read MS subtables
# =============================================================================

def table_path(ms_file, subtable):
    return os.path.join(ms_file.rstrip("/"), subtable)


def read_antenna_table(ms_file, tb):
    tb.open(table_path(ms_file, "ANTENNA"))
    try:
        names = tb.getcol("NAME")
        stations = tb.getcol("STATION")
    finally:
        tb.close()

    return names, stations


def read_frequency_axis(ms_file, tb):
    tb.open(table_path(ms_file, "SPECTRAL_WINDOW"))
    try:
        freq_hz = np.asarray(tb.getcell("CHAN_FREQ", 0), dtype=float).reshape(-1)

        try:
            chan_width_hz = np.asarray(tb.getcell("CHAN_WIDTH", 0), dtype=float).reshape(-1)
        except Exception:
            chan_width_hz = np.zeros(freq_hz.shape, dtype=float)
    finally:
        tb.close()

    if freq_hz.size < 1:
        raise RuntimeError("No frequency channels found in SPECTRAL_WINDOW")

    return freq_hz, chan_width_hz


def read_correlation_names(ms_file, tb):
    tb.open(table_path(ms_file, "POLARIZATION"))
    try:
        corr_type = np.asarray(tb.getcell("CORR_TYPE", 0), dtype=int).reshape(-1)
    finally:
        tb.close()

    # CASA Stokes enum for linear correlations.
    corr_type_to_name = {
        9: "XX",
        10: "XY",
        11: "YX",
        12: "YY",
    }

    corr_names = []
    for value in corr_type:
        value_int = int(value)
        corr_names.append(
            corr_type_to_name.get(value_int, "UNKNOWN_{0}".format(value_int))
        )

    return corr_type, corr_names


def read_main_basic_columns(ms_file, tb):
    tb.open(ms_file)
    try:
        ant1 = np.asarray(tb.getcol("ANTENNA1"), dtype=int)
        ant2 = np.asarray(tb.getcol("ANTENNA2"), dtype=int)
        time = np.asarray(tb.getcol("TIME"), dtype=float)
    finally:
        tb.close()

    return ant1, ant2, time


# =============================================================================
# Extract selected correlation
# =============================================================================

def select_rows_for_baseline(ant1, ant2, time, ant_a, ant_b):
    forward = (ant1 == int(ant_a)) & (ant2 == int(ant_b))
    reverse = (ant1 == int(ant_b)) & (ant2 == int(ant_a))
    rows = np.where(forward | reverse)[0]

    if rows.size < 1:
        raise RuntimeError(
            "No rows found for physical baseline {0}&{1}".format(ant_a, ant_b)
        )

    rows = rows[np.argsort(time[rows])]

    return rows


def get_spectrum_from_data_cell(data, flag, nchan, corr_index):
    data = np.asarray(data)
    flag = np.asarray(flag)

    if data.ndim != 2:
        raise RuntimeError("DATA cell is not 2D, shape={0}".format(data.shape))

    if flag.shape != data.shape:
        raise RuntimeError(
            "FLAG shape mismatch: FLAG={0}, DATA={1}".format(flag.shape, data.shape)
        )

    # pyuvdata/casacore may expose DATA cell as either (nchan, ncorr) or (ncorr, nchan).
    if data.shape[0] == nchan:
        spec = data[:, corr_index]
        flg = flag[:, corr_index]
    elif data.shape[1] == nchan:
        spec = data[corr_index, :]
        flg = flag[corr_index, :]
    else:
        raise RuntimeError(
            "Cannot identify frequency axis in DATA cell. DATA shape={0}, nchan={1}".format(
                data.shape,
                nchan,
            )
        )

    return np.asarray(spec), np.asarray(flg, dtype=bool)


def read_interval_for_rows(tb, rows):
    values = []

    for row in rows:
        try:
            value = float(tb.getcell("INTERVAL", int(row)))
            if np.isfinite(value) and value > 0.0:
                values.append(value)
        except Exception:
            pass

    if len(values) == 0:
        return None

    return float(np.median(np.asarray(values, dtype=float)))


def extract_requested_visibility(
    ms_file,
    tb,
    rows,
    ant_a,
    pol_a,
    ant_b,
    pol_b,
    corr_names,
    nchan,
):
    """
    Extract:
        ant_a-pol_a x conj(ant_b-pol_b)

    Forward row:
        ANTENNA1=ant_a, ANTENNA2=ant_b
        use correlation pol_a + pol_b.

    Reverse row:
        ANTENNA1=ant_b, ANTENNA2=ant_a
        use correlation pol_b + pol_a, then conjugate.
    """
    forward_corr = pol_a + pol_b
    reverse_corr = pol_b + pol_a

    if forward_corr not in corr_names:
        raise RuntimeError(
            "Forward correlation {0} not found. Available: {1}".format(
                forward_corr,
                ", ".join(corr_names),
            )
        )

    if reverse_corr not in corr_names:
        raise RuntimeError(
            "Reverse correlation {0} not found. Available: {1}".format(
                reverse_corr,
                ", ".join(corr_names),
            )
        )

    forward_corr_index = corr_names.index(forward_corr)
    reverse_corr_index = corr_names.index(reverse_corr)

    vis_rows = []
    flag_rows = []
    time_rows = []
    row_direction = []

    tb.open(ms_file)
    try:
        interval_sec = read_interval_for_rows(tb, rows)

        for row in rows:
            row = int(row)

            a1 = int(tb.getcell("ANTENNA1", row))
            a2 = int(tb.getcell("ANTENNA2", row))
            t = float(tb.getcell("TIME", row))

            data = tb.getcell("DATA", row)
            flag = tb.getcell("FLAG", row)

            if a1 == ant_a and a2 == ant_b:
                spec, flg = get_spectrum_from_data_cell(
                    data,
                    flag,
                    nchan,
                    forward_corr_index,
                )
                direction = "forward"
            elif a1 == ant_b and a2 == ant_a:
                spec, flg = get_spectrum_from_data_cell(
                    data,
                    flag,
                    nchan,
                    reverse_corr_index,
                )
                spec = np.conj(spec)
                direction = "reverse_conjugated"
            else:
                raise RuntimeError(
                    "Internal selection error at row={0}, ANTENNA1={1}, ANTENNA2={2}".format(
                        row,
                        a1,
                        a2,
                    )
                )

            vis_rows.append(np.asarray(spec, dtype=np.complex64))
            flag_rows.append(np.asarray(flg, dtype=bool))
            time_rows.append(t)
            row_direction.append(direction)
    finally:
        tb.close()

    vis_matrix = np.vstack(vis_rows)
    flag_matrix = np.vstack(flag_rows)
    time_array = np.asarray(time_rows, dtype=float)

    # Safety sort by time.
    order = np.argsort(time_array)
    vis_matrix = vis_matrix[order, :]
    flag_matrix = flag_matrix[order, :]
    time_array = time_array[order]
    row_direction = [row_direction[i] for i in order]

    return vis_matrix, flag_matrix, time_array, interval_sec, row_direction


# =============================================================================
# Plot extents
# =============================================================================

def get_frequency_extent_mhz(freq_hz, chan_width_hz):
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


def get_time_extent_sec(time_array, interval_sec):
    """
    Use row center time and INTERVAL to show full duration.

    Example:
        500 rows, 1 ms interval:
            center span = 0.499 s
            displayed duration = 0.500 s
    """
    time_array = np.asarray(time_array, dtype=float)

    if time_array.size == 0:
        return [0.0, 0.0]

    if interval_sec is None:
        if time_array.size > 1:
            dt = np.median(np.diff(time_array))
            if np.isfinite(dt) and dt > 0.0:
                interval_sec = float(dt)
            else:
                interval_sec = 0.0
        else:
            interval_sec = 0.0

    t_start_edge = float(time_array[0]) - 0.5 * float(interval_sec)
    t_end_edge = float(time_array[-1]) + 0.5 * float(interval_sec)
    duration = t_end_edge - t_start_edge

    if duration < 0.0:
        duration = 0.0

    return [0.0, float(duration)]


# =============================================================================
# Save outputs
# =============================================================================

def save_outputs(
    args,
    ant_a,
    pol_a,
    ant_b,
    pol_b,
    freq_hz,
    chan_width_hz,
    vis_matrix,
    flag_matrix,
    time_array,
    interval_sec,
    row_direction,
):
    out_png = os.path.abspath(args.out_png)
    out_dir = os.path.dirname(out_png)

    if out_dir != "" and not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    mode = str(args.mode).lower()

    if mode == "phase":
        plot_matrix = np.angle(vis_matrix).astype(float)
        plot_matrix[flag_matrix] = np.nan
        colorbar_label = "Phase (rad)"
        title_prefix = "Phase waterfall"
        vmin = -np.pi
        vmax = np.pi
    elif mode == "amp":
        plot_matrix = np.abs(vis_matrix).astype(float)
        plot_matrix[flag_matrix] = np.nan
        colorbar_label = "Amplitude"
        title_prefix = "Amplitude waterfall"
        vmin = None
        vmax = None
    else:
        raise ValueError("bad mode: {0}".format(mode))

    x_extent = get_frequency_extent_mhz(freq_hz, chan_width_hz)
    y_extent = get_time_extent_sec(time_array, interval_sec)
    extent = [x_extent[0], x_extent[1], y_extent[0], y_extent[1]]

    plt.figure(figsize=(12, 6))
    plt.imshow(
        plot_matrix,
        aspect="auto",
        origin="lower",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    plt.colorbar(label=colorbar_label)
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Time from MS start (s)")
    plt.title(
        "{0}: {1} x conj({2})".format(
            title_prefix,
            args.signal1,
            args.signal2,
        )
    )
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    print("[OK] PNG saved:", out_png)

    base, ext = os.path.splitext(out_png)

    if args.save_npy:
        complex_path = base + "_complex_visibility.npy"
        phase_path = base + "_phase_rad.npy"
        freq_path = base + "_freq_hz.npy"
        time_path = base + "_time_sec_from_start.npy"

        np.save(complex_path, vis_matrix)
        np.save(phase_path, np.angle(vis_matrix))
        np.save(freq_path, freq_hz)

        interval_for_axis = interval_sec
        if interval_for_axis is None:
            interval_for_axis = 0.0

        time_from_start = time_array - (
            time_array[0] - 0.5 * float(interval_for_axis)
        )
        np.save(time_path, time_from_start)

        print("[OK] complex visibility saved:", complex_path)
        print("[OK] phase rad saved:", phase_path)
        print("[OK] frequency axis saved:", freq_path)
        print("[OK] time axis saved:", time_path)

    if args.save_txt:
        txt_path = base + "_summary.txt"
        phase_all = np.angle(vis_matrix)
        flagged_fraction = float(np.sum(flag_matrix)) / float(flag_matrix.size)

        with open(txt_path, "w") as f:
            f.write("MS waterfall extraction summary\n")
            f.write("===============================\n\n")
            f.write("MS file: {0}\n".format(args.ms_file))
            f.write("Signal 1: {0}\n".format(args.signal1))
            f.write("Signal 2: {0}\n".format(args.signal2))
            f.write("Meaning: antenna {0} pol {1} x conj(antenna {2} pol {3})\n".format(
                ant_a,
                pol_a,
                ant_b,
                pol_b,
            ))
            f.write("Mode: {0}\n".format(args.mode))
            f.write("Selected rows: {0}\n".format(vis_matrix.shape[0]))
            f.write("Frequency channels: {0}\n".format(vis_matrix.shape[1]))
            f.write("Matrix shape: {0}\n".format(vis_matrix.shape))
            f.write("Forward rows: {0}\n".format(row_direction.count("forward")))
            f.write("Reverse rows conjugated: {0}\n".format(
                row_direction.count("reverse_conjugated")
            ))
            f.write("Frequency first Hz: {0}\n".format(float(freq_hz[0])))
            f.write("Frequency last Hz: {0}\n".format(float(freq_hz[-1])))
            f.write("Frequency min Hz: {0}\n".format(float(np.min(freq_hz))))
            f.write("Frequency max Hz: {0}\n".format(float(np.max(freq_hz))))
            f.write("Interval sec: {0}\n".format(interval_sec))
            f.write("TIME center span sec: {0}\n".format(
                float(time_array[-1] - time_array[0])
            ))
            f.write("Displayed duration sec: {0}\n".format(
                float(get_time_extent_sec(time_array, interval_sec)[1])
            ))
            f.write("Phase min rad: {0}\n".format(float(np.nanmin(phase_all))))
            f.write("Phase max rad: {0}\n".format(float(np.nanmax(phase_all))))
            f.write("Flagged fraction: {0}\n".format(flagged_fraction))
            f.write("Output PNG: {0}\n".format(out_png))

        print("[OK] summary saved:", txt_path)


# =============================================================================
# Main
# =============================================================================

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    args = parse_args(argv)

    if not os.path.isdir(args.ms_file):
        raise RuntimeError("MeasurementSet directory not found: {0}".format(args.ms_file))

    ant_a, pol_a = parse_signal_text(args.signal1)
    ant_b, pol_b = parse_signal_text(args.signal2)

    tb = get_table_backend()

    print("\n========== INPUT ==========")
    print("MS file:", args.ms_file)
    print("signal1:", args.signal1, "=> antenna", ant_a, "pol", pol_a)
    print("signal2:", args.signal2, "=> antenna", ant_b, "pol", pol_b)
    print("requested product:", pol_a + pol_b)
    print("mode:", args.mode)
    print("output:", args.out_png)

    names, stations = read_antenna_table(args.ms_file, tb)

    print("\n========== ANTENNA ==========")
    print("antenna rows:", len(names))
    print("NAME:", names)
    print("STATION:", stations)

    if ant_a < 0 or ant_a >= len(names):
        raise RuntimeError(
            "signal1 antenna id out of range: {0}; ANTENNA rows={1}".format(
                ant_a,
                len(names),
            )
        )

    if ant_b < 0 or ant_b >= len(names):
        raise RuntimeError(
            "signal2 antenna id out of range: {0}; ANTENNA rows={1}".format(
                ant_b,
                len(names),
            )
        )

    freq_hz, chan_width_hz = read_frequency_axis(args.ms_file, tb)
    nchan = int(freq_hz.size)

    print("\n========== FREQUENCY ==========")
    print("nchan:", nchan)
    print("first freq Hz:", float(freq_hz[0]))
    print("last freq Hz:", float(freq_hz[-1]))
    print("min freq Hz:", float(np.min(freq_hz)))
    print("max freq Hz:", float(np.max(freq_hz)))

    corr_type, corr_names = read_correlation_names(args.ms_file, tb)

    print("\n========== POLARIZATION ==========")
    print("CORR_TYPE:", corr_type)
    print("CORR_NAMES:", corr_names)

    ant1, ant2, time = read_main_basic_columns(args.ms_file, tb)
    rows = select_rows_for_baseline(ant1, ant2, time, ant_a, ant_b)

    print("\n========== ROW SELECTION ==========")
    print("physical baseline:", "{0}&{1}".format(ant_a, ant_b))
    print("selected rows:", rows.size)

    vis_matrix, flag_matrix, time_array, interval_sec, row_direction = (
        extract_requested_visibility(
            args.ms_file,
            tb,
            rows,
            ant_a,
            pol_a,
            ant_b,
            pol_b,
            corr_names,
            nchan,
        )
    )

    time_extent = get_time_extent_sec(time_array, interval_sec)

    print("\n========== MATRIX ==========")
    print("complex visibility matrix shape:", vis_matrix.shape)
    print("flag matrix shape:", flag_matrix.shape)
    print("interval sec:", interval_sec)
    print("TIME center span sec:", float(time_array[-1] - time_array[0]))
    print("displayed duration sec:", float(time_extent[1] - time_extent[0]))
    print("phase min/max rad:", float(np.nanmin(np.angle(vis_matrix))), float(np.nanmax(np.angle(vis_matrix))))
    print("flagged fraction:", float(np.sum(flag_matrix)) / float(flag_matrix.size))
    print("forward rows:", row_direction.count("forward"))
    print("reverse rows conjugated:", row_direction.count("reverse_conjugated"))

    save_outputs(
        args,
        ant_a,
        pol_a,
        ant_b,
        pol_b,
        freq_hz,
        chan_width_hz,
        vis_matrix,
        flag_matrix,
        time_array,
        interval_sec,
        row_direction,
    )

    print("\n[OK] Finished.")


if __name__ == "__main__":
    main()
