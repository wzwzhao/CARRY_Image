#!/usr/bin/env python3
"""
Inspect a CASA MeasurementSet written by hdf5_to_ms.py.
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np

try:
    import casacore.tables as casacore_tables
except Exception as error:  # pragma: no cover
    casacore_tables = None
    CASACORE_IMPORT_ERROR = error
else:
    CASACORE_IMPORT_ERROR = None


def require_casacore() -> None:
    if casacore_tables is None:
        raise RuntimeError("python-casacore is required") from CASACORE_IMPORT_ERROR


def as_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray) and value.shape == ():
        return as_text(value[()])
    return str(value)


def parse_command_line(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a CASA MeasurementSet and print key table information."
    )
    parser.add_argument("input_ms", help="MeasurementSet directory")
    parser.add_argument(
        "--show-rows",
        type=int,
        default=5,
        help="number of MAIN rows to print for TIME/ANTENNA/UVW summary",
    )
    return parser.parse_args(argv)


def inspect_main_table(input_ms: str, show_rows: int) -> None:
    tb = casacore_tables.table(input_ms, readonly=True)
    try:
        print("\n========== MAIN ==========")
        print("rows:", tb.nrows())
        print("columns:", ", ".join(tb.colnames()))

        required_cols = [
            "DATA",
            "FLAG",
            "UVW",
            "TIME",
            "ANTENNA1",
            "ANTENNA2",
            "DATA_DESC_ID",
            "FIELD_ID",
        ]
        for col in required_cols:
            print(f"has {col}:", col in tb.colnames())

        if tb.nrows() > 0:
            data0 = np.asarray(tb.getcell("DATA", 0))
            flag0 = np.asarray(tb.getcell("FLAG", 0))
            uvw0 = np.asarray(tb.getcell("UVW", 0), dtype=np.float64)
            print("row0 DATA shape:", data0.shape)
            print("row0 FLAG shape:", flag0.shape)
            print("row0 UVW:", uvw0)

            count = min(show_rows, tb.nrows())
            times = np.asarray(tb.getcol("TIME", 0, count), dtype=np.float64)
            ant1 = np.asarray(tb.getcol("ANTENNA1", 0, count), dtype=np.int64)
            ant2 = np.asarray(tb.getcol("ANTENNA2", 0, count), dtype=np.int64)
            uvw = np.asarray(tb.getcol("UVW", 0, count), dtype=np.float64)
            print("\nfirst rows:")
            for row in range(count):
                print(
                    f"row {row}: TIME={times[row]:.12f} "
                    f"ANTENNA1={int(ant1[row])} ANTENNA2={int(ant2[row])} "
                    f"UVW={uvw[row].tolist()}"
                )
        print("==========================")
    finally:
        tb.close()


def inspect_subtable(input_ms: str, subtable: str) -> None:
    path = os.path.join(input_ms, subtable)
    tb = casacore_tables.table(path, readonly=True)
    try:
        print(f"\n========== {subtable} ==========")
        print("rows:", tb.nrows())
        print("columns:", ", ".join(tb.colnames()))

        if tb.nrows() > 0:
            if subtable == "SPECTRAL_WINDOW":
                chan_freq = np.asarray(tb.getcell("CHAN_FREQ", 0), dtype=np.float64).reshape(-1)
                chan_width = np.asarray(tb.getcell("CHAN_WIDTH", 0), dtype=np.float64).reshape(-1)
                print("NUM_CHAN:", int(np.asarray(tb.getcell("NUM_CHAN", 0)).reshape(-1)[0]))
                print("CHAN_FREQ shape:", chan_freq.shape)
                print("CHAN_WIDTH shape:", chan_width.shape)
                if chan_freq.size > 0:
                    print("CHAN_FREQ first/last:", float(chan_freq[0]), float(chan_freq[-1]))
            elif subtable == "POLARIZATION":
                corr_type = np.asarray(tb.getcell("CORR_TYPE", 0), dtype=np.int64).reshape(-1)
                print("NUM_CORR:", int(np.asarray(tb.getcell("NUM_CORR", 0)).reshape(-1)[0]))
                print("CORR_TYPE:", corr_type.tolist())
            elif subtable == "DATA_DESCRIPTION":
                print(
                    "SPW/POL IDs:",
                    int(np.asarray(tb.getcell("SPECTRAL_WINDOW_ID", 0)).reshape(-1)[0]),
                    int(np.asarray(tb.getcell("POLARIZATION_ID", 0)).reshape(-1)[0]),
                )
            elif subtable == "FIELD":
                name = as_text(tb.getcell("NAME", 0)) if "NAME" in tb.colnames() else "N/A"
                phase_dir = np.asarray(tb.getcell("PHASE_DIR", 0), dtype=np.float64)
                print("NAME:", name)
                print("PHASE_DIR shape:", phase_dir.shape)
                print("PHASE_DIR flat:", phase_dir.reshape(-1).tolist())
            elif subtable == "ANTENNA":
                names = [as_text(item) for item in tb.getcol("NAME")]
                pos = np.asarray(tb.getcol("POSITION"), dtype=np.float64)
                print("antenna names:", names[: min(10, len(names))])
                print("POSITION shape:", pos.shape)
            elif subtable == "OBSERVATION":
                if "TELESCOPE_NAME" in tb.colnames():
                    print("TELESCOPE_NAME:", as_text(tb.getcell("TELESCOPE_NAME", 0)))
        print("=" * (27 + len(subtable)))
    finally:
        tb.close()


def main(argv: list[str] | None = None) -> int:
    args = parse_command_line(argv)
    require_casacore()

    if not os.path.isdir(args.input_ms):
        raise FileNotFoundError(args.input_ms)

    inspect_main_table(args.input_ms, show_rows=max(0, args.show_rows))
    for subtable in [
        "ANTENNA",
        "SPECTRAL_WINDOW",
        "POLARIZATION",
        "DATA_DESCRIPTION",
        "FIELD",
        "OBSERVATION",
    ]:
        inspect_subtable(args.input_ms, subtable)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
