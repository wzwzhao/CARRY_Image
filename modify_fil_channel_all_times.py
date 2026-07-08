#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
modify_fil_channel_all_times.py

Set one frequency channel in a .fil file to a nonzero complex int8 value
for EVERY FFT/time sample in the data section by default.

Default behavior:
    channel number : 1024, 1-based
    channel index  : 1023, 0-based
    value          : 50 + 0j
    time/FFT range : all FFT samples in the file
    output         : copy file, do not overwrite original

Data format assumed:
    after HEADER_END:
        FFT0: ch0(real int8, imag int8), ch1(...), ..., ch2047(...)
        FFT1: ch0(real int8, imag int8), ch1(...), ..., ch2047(...)
        ...

Each complex point is 2 bytes:
    real int8 + imag int8
"""

import argparse
import mmap
import os
import shutil
import struct
import sys


HEADER_VALUE_TYPES = {
    "source_name": "string",
    "rawdatafile": "string",
    "az_start": "double",
    "za_start": "double",
    "src_raj": "double",
    "src_dej": "double",
    "tstart": "double",
    "tsamp": "double",
    "period": "double",
    "fch1": "double",
    "foff": "double",
    "refdm": "double",
    "nchans": "int",
    "telescope_id": "int",
    "machine_id": "int",
    "data_type": "int",
    "ibeam": "int",
    "nbeams": "int",
    "nbits": "int",
    "barycentric": "int",
    "pulsarcentric": "int",
    "nifs": "int",
    "nbins": "int",
    "nsamples": "int",
}

BYTES_PER_COMPLEX_POINT = 2


def read_int(f):
    data = f.read(4)
    if len(data) != 4:
        raise EOFError("failed to read int32")
    return struct.unpack("<i", data)[0]


def read_double(f):
    data = f.read(8)
    if len(data) != 8:
        raise EOFError("failed to read float64")
    return struct.unpack("<d", data)[0]


def read_string(f):
    length = read_int(f)
    if length < 0:
        raise ValueError("bad string length: {0}".format(length))
    data = f.read(length)
    if len(data) != length:
        raise EOFError("failed to read string")
    return data.decode("utf-8", errors="replace")


def parse_header(file_path):
    header = {}

    with open(file_path, "rb") as f:
        start = read_string(f)
        if start != "HEADER_START":
            raise ValueError("missing HEADER_START in {0}".format(file_path))

        header["HEADER_START"] = start

        while True:
            key = read_string(f)

            if key == "HEADER_END":
                header["HEADER_END"] = key
                data_offset = f.tell()
                break

            value_type = HEADER_VALUE_TYPES.get(key)
            if value_type is None:
                raise ValueError(
                    "unknown header key {0!r} in {1}".format(key, file_path)
                )

            if value_type == "string":
                value = read_string(f)
            elif value_type == "int":
                value = read_int(f)
            elif value_type == "double":
                value = read_double(f)
            else:
                raise ValueError("bad header value type: {0}".format(value_type))

            header[key] = value

    return header, data_offset


def int8_to_byte(value):
    value = int(value)
    if value < -128 or value > 127:
        raise ValueError("int8 value out of range [-128, 127]: {0}".format(value))
    return value & 0xFF


def make_output_path(input_path, out_dir, suffix):
    directory = os.path.dirname(os.path.abspath(input_path))
    base = os.path.basename(input_path)
    stem, ext = os.path.splitext(base)

    if ext == "":
        ext = ".fil"

    if out_dir is None:
        out_dir = directory

    return os.path.abspath(os.path.join(out_dir, stem + suffix + ext))


def check_file_layout(file_path, channel_index):
    header, data_offset = parse_header(file_path)

    nchan = int(header.get("nchans", 0))
    if nchan <= 0:
        raise ValueError("bad or missing header nchans in {0}".format(file_path))

    if channel_index < 0 or channel_index >= nchan:
        raise ValueError(
            "channel index out of range: index={0}, nchans={1}".format(
                channel_index,
                nchan,
            )
        )

    bytes_per_fft = nchan * BYTES_PER_COMPLEX_POINT
    file_size = os.path.getsize(file_path)
    data_bytes = file_size - data_offset

    if data_bytes <= 0:
        raise ValueError("empty data section in {0}".format(file_path))

    if data_bytes % bytes_per_fft != 0:
        raise ValueError(
            "data section is not aligned with FFT size in {0}: "
            "data_bytes={1}, bytes_per_fft={2}".format(
                file_path,
                data_bytes,
                bytes_per_fft,
            )
        )

    n_fft = data_bytes // bytes_per_fft

    return {
        "header": header,
        "data_offset": data_offset,
        "nchan": nchan,
        "bytes_per_fft": bytes_per_fft,
        "file_size": file_size,
        "data_bytes": data_bytes,
        "n_fft": n_fft,
    }


def complex_point_offset(data_offset, bytes_per_fft, fft_index, channel_index):
    return data_offset + int(fft_index) * bytes_per_fft + int(channel_index) * 2


def read_complex_point(file_path, data_offset, bytes_per_fft, fft_index, channel_index):
    pos = complex_point_offset(data_offset, bytes_per_fft, fft_index, channel_index)

    with open(file_path, "rb") as f:
        f.seek(pos)
        data = f.read(2)

    if len(data) != 2:
        raise EOFError("failed to read complex point at byte offset {0}".format(pos))

    real = struct.unpack("b", data[0:1])[0]
    imag = struct.unpack("b", data[1:2])[0]

    return real, imag


def modify_one_file(
    input_path,
    output_path,
    in_place,
    channel_index,
    real_value,
    imag_value,
):
    if not os.path.isfile(input_path):
        raise FileNotFoundError(input_path)

    layout = check_file_layout(input_path, channel_index)
    n_fft = int(layout["n_fft"])

    if in_place:
        target_path = os.path.abspath(input_path)
    else:
        target_path = os.path.abspath(output_path)
        target_dir = os.path.dirname(target_path)
        if target_dir != "" and not os.path.isdir(target_dir):
            os.makedirs(target_dir)
        shutil.copy2(input_path, target_path)

    verify_fft_indices = sorted(set([0, n_fft // 2, n_fft - 1]))
    before_values = []
    for fft_index in verify_fft_indices:
        before_values.append(
            read_complex_point(
                target_path,
                layout["data_offset"],
                layout["bytes_per_fft"],
                fft_index,
                channel_index,
            )
        )

    real_byte = int8_to_byte(real_value)
    imag_byte = int8_to_byte(imag_value)

    with open(target_path, "r+b") as f:
        mm = mmap.mmap(f.fileno(), 0)
        try:
            for fft_index in range(n_fft):
                pos = complex_point_offset(
                    layout["data_offset"],
                    layout["bytes_per_fft"],
                    fft_index,
                    channel_index,
                )
                mm[pos] = real_byte
                mm[pos + 1] = imag_byte
            mm.flush()
        finally:
            mm.close()

    after_values = []
    for fft_index in verify_fft_indices:
        after_values.append(
            read_complex_point(
                target_path,
                layout["data_offset"],
                layout["bytes_per_fft"],
                fft_index,
                channel_index,
            )
        )

    print("\n========== MODIFY FIL ALL TIMES ==========")
    print("input file             :", os.path.abspath(input_path))
    print("output file            :", target_path)
    print("in place               :", in_place)
    print("nchans                 :", layout["nchan"])
    print("channel index          :", channel_index, "(0-based)")
    print("channel number         :", channel_index + 1, "(1-based)")
    print("data offset bytes      :", layout["data_offset"])
    print("bytes per FFT          :", layout["bytes_per_fft"])
    print("n FFT / time samples   :", layout["n_fft"])
    print("modified FFT range     :", "0 -> {0}".format(n_fft - 1))
    print("modified FFT count     :", n_fft)
    print("target value           :", "{0} + {1}j".format(real_value, imag_value))
    print("verification FFT index :", verify_fft_indices)
    print("before values          :", before_values)
    print("after values           :", after_values)
    print("==========================================")

    expected = (int(real_value), int(imag_value))
    for value in after_values:
        if value != expected:
            raise RuntimeError(
                "verification failed: expected {0}, got {1}".format(expected, value)
            )

    return target_path


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description=(
            "Set one frequency channel in filterbank .fil payload to a nonzero "
            "int8 complex value for EVERY FFT/time sample."
        )
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="input .fil files to modify",
    )
    parser.add_argument(
        "--channel-number",
        type=int,
        default=1024,
        help=(
            "frequency channel number to modify, 1-based. "
            "Default: 1024 means Python index 1023."
        ),
    )
    parser.add_argument(
        "--channel-index",
        type=int,
        default=None,
        help=(
            "frequency channel index to modify, 0-based. "
            "If supplied, it overrides --channel-number."
        ),
    )
    parser.add_argument(
        "--real",
        type=int,
        default=50,
        help="real int8 value to write, range [-128, 127]. Default: 50",
    )
    parser.add_argument(
        "--imag",
        type=int,
        default=0,
        help="imag int8 value to write, range [-128, 127]. Default: 0",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="output directory for copied modified files. Default: same as input",
    )
    parser.add_argument(
        "--suffix",
        default="_ch1024_alltimes_nonzero",
        help=(
            "suffix used for copied output files when not using --in-place. "
            "Default: _ch1024_alltimes_nonzero"
        ),
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="modify original files directly. Use with caution.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    args = parse_args(argv)

    if args.channel_index is not None:
        channel_index = int(args.channel_index)
    else:
        channel_index = int(args.channel_number) - 1

    if channel_index < 0:
        raise ValueError("channel index must be >= 0")

    output_paths = []

    for input_path in args.files:
        if args.in_place:
            output_path = input_path
        else:
            output_path = make_output_path(input_path, args.out_dir, args.suffix)

        output_paths.append(
            modify_one_file(
                input_path=input_path,
                output_path=output_path,
                in_place=args.in_place,
                channel_index=channel_index,
                real_value=args.real,
                imag_value=args.imag,
            )
        )

    print("\n========== DONE ==========")
    print("modified files:")
    for path in output_paths:
        print(" ", path)
    print("==========================")


if __name__ == "__main__":
    main()
