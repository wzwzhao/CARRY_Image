import argparse
import os
import struct
import sys
from contextlib import nullcontext
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP

import numpy as np

# 兼容旧版 h5py + 新版 numpy
# 有些旧版 h5py 会访问 np.typeDict，
# 但新版 numpy 已经删除了这个名字。
if not hasattr(np, "typeDict"):
    np.typeDict = np.sctypeDict

try:
    import h5py
    H5PY_IMPORT_ERROR = None
except Exception as error:
    h5py = None
    H5PY_IMPORT_ERROR = error

try:
    import katpoint
    KATPOINT_IMPORT_ERROR = None
except Exception as error:
    katpoint = None
    KATPOINT_IMPORT_ERROR = error

# =========================
# 用户配置参数
# =========================

MAX_FILES = 20

# 当前允许的 header nbits（你可以手动改 8 / 16）
# 注意：
#   这里检查的是 fil header 里面的 nbits
#   你的复数频点总大小由 BITS_PER_FREQ_POINT 控制
REQUIRED_NBITS = 8

# 天线编号范围
# 文件名里 xx 从 00 到 09，表示10面天线
MIN_ANTENNA_ID = 0
MAX_ANTENNA_ID = 9

# 极化编号范围
# P 只能是 0 或 1
VALID_POLARIZATIONS = [0, 1]

# 输出HDF5文件
# None 表示根据第一个输入文件 header 里的 tstart 自动生成：
# YYYYMMDDHHMMSSmmm_YYYYMMDDHHMMSSmmm.h5
OUTPUT_HDF5_FILE = None

# 是否把相关结果写入HDF5文件
# True ：正常落盘保存
# False：只跑读取和相关计算流程，不写HDF5
ENABLE_HDF5_OUTPUT = True

# 输出文件已存在时是否覆盖
OVERWRITE_OUTPUT = True

# HDF5压缩
# None 表示不压缩
# 也可以改成 "gzip"
HDF5_COMPRESSION = None

# frequency metadata input unit in filterbank header
FREQUENCY_INPUT_UNIT = "MHz"

# physical antenna metadata for future MS export
N_PHYSICAL_ANTENNAS = 10
ANTENNA_NAMES = [f"ANT{ant_id:02d}" for ant_id in range(N_PHYSICAL_ANTENNAS)]
ANTENNA_STATIONS = [
    f"STATION{ant_id:02d}" for ant_id in range(N_PHYSICAL_ANTENNAS)
]
ANTENNA_DISH_DIAMETER_M = 1.0
ANTENNA_POSITION_ITRF_M = None

# antenna txt file for real antenna positions.
# None means use placeholder positions unless --antenna-txt is given.
# Format:
#   # name lat lon [alt_m] [diam_m]
#   ant0 29.784402 109.779625 1581 7.5
ANTENNA_INFO_TXT = None
ANTENNA_TXT_NAME_PREFIX = "ant"
ANTENNA_POSITION_FRAME = "ITRF/WGS84_ECEF"
_ANTENNA_INFO_CACHE = None
_ANTENNA_INFO_CACHE_PATH = None

# =========================
# Field / phase center metadata
# =========================

# Phase center, J2000.
# RA format:  "HH:MM:SS.s"
# Dec format: "[+/-]DD:MM:SS.s"
FIELD_RA_HMS = "19:35:00.00"
FIELD_DEC_DMS = "21:54:00.00"
FIELD_FRAME = "J2000"

# These will be computed from FIELD_RA_HMS / FIELD_DEC_DMS.
# Keep them as None at config level; use helper functions to resolve.
FIELD_RA_RAD = None
FIELD_DEC_RAD = None

# =========================
# Polarization / row mode
# =========================

# Old mode was XX_ONLY.
# New mode keeps all available signal-pair correlations.
MS_EXPORT_POL_MODE = "ALL_SIGNAL_PAIRS"
POL0_NAME = "X"
POL1_NAME = "Y"

# Row selection mode:
# PRESENT_SIGNALS_ONLY means:
#   only create rows for signal pairs where both input .fil files exist.
MS_ROW_SELECTION_MODE = "PRESENT_SIGNALS_ONLY"

# Correlation output mode:
#   "sum"  : save sum over FFT_PER_CORR, current behavior
#   "mean" : save average over FFT_PER_CORR
CORR_OUTPUT_MODE = "sum"


# =========================
# 第二部分：相关计算参数
# =========================

# 20路输入信号
N_INPUT_SIGNALS = 20

# 1个积分周期包含多少个FFT
FFT_PER_INTEGRATION = 10

# 1个fil文件包含多少个积分周期
INTEGRATION_PER_FILE = 12500

# 1个fil文件理论包含多少个FFT
FFT_PER_FILE = FFT_PER_INTEGRATION * INTEGRATION_PER_FILE

# 1个FFT包含多少个频点
NCHAN = 2048

# 一个复数频点总大小：
# real int8 + imag int8 = 16 bit = 2 byte
BITS_PER_FREQ_POINT = 16
BYTES_PER_FREQ_POINT = BITS_PER_FREQ_POINT // 8

# 每个FFT的时间分辨率，单位：微秒
FFT_TIME_US = 4

# 相关积分后的时间分辨率，单位：微秒
CORR_TIME_US = 1000

# 一次读多少个FFT
# 必须能被 FFT_PER_CORR 整除
READ_FFT_PER_BLOCK = 36000

# 输出格式
OUTPUT_COMPLEX_BYTES = 8
OUTPUT_DTYPE = np.complex64
OUTPUT_DTYPE_NAME = "complex64"


# =========================
# header类型定义
# =========================

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


# =========================
# 第一部分：基础IO
# =========================

def open_binary_file(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(file_path)
    return open(file_path, "rb")


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
        raise ValueError(f"bad string length: {length}")

    data = f.read(length)

    if len(data) != length:
        raise EOFError("failed to read string")

    return data.decode("utf-8", errors="replace")


# =========================
# 第一部分：header解析
# =========================

def parse_header(file_path):
    """
    解析fil包头。

    返回：
        header
        data_offset

    data_offset 是 HEADER_END 后面的位置。
    后面读取FFT数据要从这里开始。
    """
    header = {}

    with open_binary_file(file_path) as f:

        start = read_string(f)
        if start != "HEADER_START":
            raise ValueError("missing HEADER_START")

        header["HEADER_START"] = start

        while True:
            key = read_string(f)

            if key == "HEADER_END":
                header["HEADER_END"] = key
                data_offset = f.tell()
                break

            value_type = HEADER_VALUE_TYPES.get(key)

            if value_type is None:
                raise ValueError(f"unknown key: {key}")

            if value_type == "string":
                value = read_string(f)
            elif value_type == "int":
                value = read_int(f)
            elif value_type == "double":
                value = read_double(f)
            else:
                raise ValueError("bad type")

            header[key] = value

    return header, data_offset


# =========================
# 第一部分：文件名解析
# =========================

def calc_input_signal_no(antenna_id, polarization):
    """
    根据天线编号和极化编号计算输入信号路数。

    P = 0:
        输入信号路数 = (xx + 1) * 2 - 1

    P = 1:
        输入信号路数 = (xx + 1) * 2

    返回：
        从1开始的输入信号路数。
    """
    if polarization == 0:
        return (antenna_id + 1) * 2 - 1

    if polarization == 1:
        return (antenna_id + 1) * 2

    raise ValueError(f"bad polarization: {polarization}")


def parse_filename(file_path):
    """
    新文件名格式：
        YYYYMMDD_HHMMSS_xx_P.fil

    例子：
        20260624_113210_01_0.fil
    """
    base = os.path.basename(file_path)
    name = os.path.splitext(base)[0]
    parts = name.split("_")

    if len(parts) != 4:
        raise ValueError(f"bad filename: {base}")

    date = parts[0]
    time = parts[1]
    antenna_id = int(parts[2])
    polarization = int(parts[3])

    if len(date) != 8:
        raise ValueError(f"bad date in filename: {base}")

    if len(time) != 6:
        raise ValueError(f"bad time in filename: {base}")

    if antenna_id < MIN_ANTENNA_ID or antenna_id > MAX_ANTENNA_ID:
        raise ValueError(f"bad antenna id in filename: {base}")

    if polarization not in VALID_POLARIZATIONS:
        raise ValueError(f"bad polarization in filename: {base}")

    input_signal_no = calc_input_signal_no(antenna_id, polarization)

    return {
        "date": date,
        "time": time,
        "time_tag": f"{date}_{time}",
        "antenna_id": antenna_id,
        "polarization": polarization,
        "input_signal_no": input_signal_no,
        "input_signal_index": input_signal_no - 1,
    }


# =========================
# 第一部分：输出
# =========================

def print_file(file_path, fname_info, header):
    print("\n================ FILE ================")
    print("file               :", file_path)
    print("time               :", fname_info["time_tag"])
    print("antenna_id         :", fname_info["antenna_id"])
    print("polarization       :", fname_info["polarization"])
    print("input_signal_no    :", fname_info["input_signal_no"])
    print("input_signal_index :", fname_info["input_signal_index"])
    print("source             :", header.get("source_name"))
    print("tstart             :", header.get("tstart"))
    print("nbits              :", header.get("nbits"))
    print("nchans             :", header.get("nchans"))


def print_header(header):
    print("\n------ HEADER ------")
    for k, v in header.items():
        print(k, ":", v)


def print_input_signal_summary(infos):
    print("\n========== INPUT SIGNAL SUMMARY ==========")

    for i in infos:
        fname = i["fname"]

        print(
            "antenna_id:",
            fname["antenna_id"],
            " polarization:",
            fname["polarization"],
            " input_signal_no:",
            fname["input_signal_no"],
            " file:",
            i["file"]
        )

    print("==========================================\n")


# =========================
# 第一部分：校验函数
# =========================

def check_file_count(files):
    if len(files) == 0:
        raise ValueError("no input files")

    if len(files) > MAX_FILES:
        raise ValueError("too many files (>20)")


def check_time_consistency(infos):
    ref = infos[0]["fname"]["time_tag"]

    for i in infos:
        if i["fname"]["time_tag"] != ref:
            raise ValueError(
                f"time mismatch:\nref={ref}\nbad={i['fname']['time_tag']}"
            )


def check_duplicate_input_signal(infos):
    """
    检查是否重复输入了同一路信号。
    """
    used = {}

    for i in infos:
        signal_no = i["fname"]["input_signal_no"]

        if signal_no in used:
            raise ValueError(
                f"duplicate input signal:\n"
                f"signal_no={signal_no}\n"
                f"file1={used[signal_no]}\n"
                f"file2={i['file']}"
            )

        used[signal_no] = i["file"]


def check_header_consistency(infos):
    """
    检查多个文件是否属于同一次观测。
    """
    ref = infos[0]["header"]

    for i in infos:
        h = i["header"]

        if h["source_name"] != ref["source_name"]:
            raise ValueError("source_name mismatch")

        if h["tstart"] != ref["tstart"]:
            raise ValueError("tstart mismatch")

        if h["nbits"] != REQUIRED_NBITS:
            raise ValueError(
                f"nbits mismatch: file={i['file']} "
                f"header={h['nbits']} required={REQUIRED_NBITS}"
            )


def check_nchans(infos):
    """
    检查所有输入文件的 nchans 是否等于代码设置的 NCHAN。
    """
    for i in infos:
        h = i["header"]

        if h.get("nchans") != NCHAN:
            raise ValueError(
                f"nchans mismatch:\n"
                f"file={i['file']}\n"
                f"header nchans={h.get('nchans')}\n"
                f"expected nchans={NCHAN}"
            )


def check_corr_config():
    """
    检查相关计算参数是否合理。
    """
    if BITS_PER_FREQ_POINT != 16:
        raise ValueError(
            "current reader only supports 16-bit complex point: int8 real + int8 imag"
        )

    if BYTES_PER_FREQ_POINT != 2:
        raise ValueError("BYTES_PER_FREQ_POINT must be 2 for int8 real + int8 imag")

    if CORR_TIME_US % FFT_TIME_US != 0:
        raise ValueError(
            f"CORR_TIME_US must be divisible by FFT_TIME_US: "
            f"{CORR_TIME_US} / {FFT_TIME_US}"
        )

    fft_per_corr = CORR_TIME_US // FFT_TIME_US

    if FFT_PER_FILE % fft_per_corr != 0:
        raise ValueError(
            f"FFT_PER_FILE must be divisible by FFT_PER_CORR: "
            f"{FFT_PER_FILE} / {fft_per_corr}"
        )

    if READ_FFT_PER_BLOCK % fft_per_corr != 0:
        raise ValueError(
            f"READ_FFT_PER_BLOCK must be divisible by FFT_PER_CORR: "
            f"{READ_FFT_PER_BLOCK} / {fft_per_corr}"
        )

    if CORR_OUTPUT_MODE not in ["sum", "mean"]:
        raise ValueError("CORR_OUTPUT_MODE must be 'sum' or 'mean'")


# =========================
# 第二部分：准备层
# =========================

def get_bytes_per_fft():
    return NCHAN * BYTES_PER_FREQ_POINT


def get_fft_per_corr():
    return CORR_TIME_US // FFT_TIME_US


def get_n_corr_time():
    return FFT_PER_FILE // get_fft_per_corr()


def get_baseline_pairs():
    """
    只保留唯一相关对：
        自相关：i == j
        互相关：i < j

    20路一共：
        20个自相关 + 190个互相关 = 210个baseline
    """
    pairs = []

    for i in range(N_INPUT_SIGNALS):
        for j in range(i, N_INPUT_SIGNALS):
            pairs.append((i, j))

    return pairs


def count_existing_signals(signal_map):
    count = 0

    for item in signal_map:
        if item is not None:
            count += 1

    return count


def build_signal_file_map(infos):
    """
    建立20路输入信号映射。

    signal_map[index] = file_info 或 None

    如果某一路没有输入文件，就保持 None。
    后面相关计算时逻辑补零，物理上不读零数组。
    """
    signal_map = [None] * N_INPUT_SIGNALS

    for i in infos:
        index = i["fname"]["input_signal_index"]

        if index < 0 or index >= N_INPUT_SIGNALS:
            raise ValueError(
                f"input signal index out of range: {index}"
            )

        signal_map[index] = i

    return signal_map


def add_data_info(infos):
    """
    给每个输入文件补充数据区信息。
    """
    bytes_per_fft = get_bytes_per_fft()

    for i in infos:
        file_size = os.path.getsize(i["file"])
        data_offset = i["data_offset"]
        data_bytes = file_size - data_offset

        if data_bytes <= 0:
            raise ValueError(f"empty data section: {i['file']}")

        if data_bytes % bytes_per_fft != 0:
            raise ValueError(
                f"data size is not aligned with FFT size:\n"
                f"file={i['file']}\n"
                f"data_bytes={data_bytes}\n"
                f"bytes_per_fft={bytes_per_fft}"
            )

        n_fft = data_bytes // bytes_per_fft

        if n_fft != FFT_PER_FILE:
            raise ValueError(
                f"n_fft mismatch:\n"
                f"file={i['file']}\n"
                f"n_fft={n_fft}\n"
                f"expected={FFT_PER_FILE}"
            )

        if n_fft % FFT_PER_INTEGRATION != 0:
            raise ValueError(
                f"n_fft is not divisible by FFT_PER_INTEGRATION:\n"
                f"file={i['file']}\n"
                f"n_fft={n_fft}\n"
                f"FFT_PER_INTEGRATION={FFT_PER_INTEGRATION}"
            )

        i["file_size"] = file_size
        i["data_bytes"] = data_bytes
        i["n_fft"] = n_fft
        i["n_integration"] = n_fft // FFT_PER_INTEGRATION


def check_n_fft_consistency(infos):
    """
    检查所有实际输入文件的FFT数量是否一致。
    """
    ref_n_fft = infos[0]["n_fft"]

    for i in infos:
        if i["n_fft"] != ref_n_fft:
            raise ValueError(
                f"n_fft mismatch:\n"
                f"ref={ref_n_fft}\n"
                f"file={i['file']}\n"
                f"n_fft={i['n_fft']}"
            )


def estimate_output_size_bytes():
    """
    估算HDF5中vis数据集大小。

    vis shape:
        [n_corr_time, n_baseline, nchan]
    """
    n_corr_time = get_n_corr_time()
    n_baseline = len(get_baseline_pairs())

    return n_corr_time * n_baseline * NCHAN * OUTPUT_COMPLEX_BYTES


def estimate_block_cache_bytes(signal_map):
    """
    估算一次读取block时，真实存在信号占用的complex64缓存大小。
    """
    n_exist = count_existing_signals(signal_map)
    return n_exist * READ_FFT_PER_BLOCK * NCHAN * OUTPUT_COMPLEX_BYTES


def format_bytes(n_bytes):
    gb = n_bytes / 1_000_000_000
    gib = n_bytes / (1024 ** 3)
    return f"{gb:.3f} GB ({gib:.3f} GiB)"


def print_corr_config():
    print("\n========== CORRELATION CONFIG ==========")
    print("N_INPUT_SIGNALS       :", N_INPUT_SIGNALS)
    print("FFT_PER_INTEGRATION   :", FFT_PER_INTEGRATION)
    print("INTEGRATION_PER_FILE  :", INTEGRATION_PER_FILE)
    print("FFT_PER_FILE          :", FFT_PER_FILE)
    print("NCHAN                 :", NCHAN)
    print("BITS_PER_FREQ_POINT   :", BITS_PER_FREQ_POINT)
    print("BYTES_PER_FREQ_POINT  :", BYTES_PER_FREQ_POINT)
    print("BYTES_PER_FFT         :", get_bytes_per_fft())
    print("FFT_TIME_US           :", FFT_TIME_US)
    print("CORR_TIME_US          :", CORR_TIME_US)
    print("FFT_PER_CORR          :", get_fft_per_corr())
    print("READ_FFT_PER_BLOCK    :", READ_FFT_PER_BLOCK)
    print("CORR_OUTPUT_MODE      :", CORR_OUTPUT_MODE)
    print("OUTPUT_DTYPE          :", OUTPUT_DTYPE_NAME)
    print("========================================")


def print_signal_map(signal_map):
    print("\n========== SIGNAL MAP ==========")

    for index, item in enumerate(signal_map):
        signal_no = index + 1

        if item is None:
            print(
                f"signal {signal_no:02d}: missing -> logical zero fill"
            )
            continue

        fname = item["fname"]

        print(
            f"signal {signal_no:02d}: "
            f"antenna {fname['antenna_id']:02d} "
            f"pol {fname['polarization']} "
            f"-> {item['file']}"
        )

    print("================================")


def print_data_info(infos):
    print("\n========== DATA INFO ==========")

    for i in infos:
        print("\nfile              :", i["file"])
        print("input_signal_no   :", i["fname"]["input_signal_no"])
        print("input_signal_index:", i["fname"]["input_signal_index"])
        print("data_offset       :", i["data_offset"])
        print("file_size         :", i["file_size"])
        print("data_bytes        :", i["data_bytes"])
        print("n_fft             :", i["n_fft"])
        print("n_integration     :", i["n_integration"])

    print("================================")


def print_baseline_info():
    pairs = get_baseline_pairs()

    print("\n========== BASELINE INFO ==========")
    print("auto correlation count :", N_INPUT_SIGNALS)
    print("cross correlation count:", N_INPUT_SIGNALS * (N_INPUT_SIGNALS - 1) // 2)
    print("total baseline count   :", len(pairs))
    print("baseline rule          : keep i <= j only")
    print("example baseline[0]    :", pairs[0])
    print("example baseline[-1]   :", pairs[-1])
    print("===================================")


def print_selected_row_info(signal_map, baseline_pairs):
    selected = build_selected_signal_baseline_indices(
        signal_map,
        baseline_pairs
    )
    n_present_signal = count_existing_signals(signal_map)

    print("\n========== SELECTED ROW INFO ==========")
    print("MS_EXPORT_POL_MODE     :", MS_EXPORT_POL_MODE)
    print("MS_ROW_SELECTION_MODE  :", MS_ROW_SELECTION_MODE)
    print("present signal count   :", n_present_signal)
    print("selected pair count    :", len(selected))
    print(
        "expected selected pairs:",
        n_present_signal * (n_present_signal + 1) // 2
    )
    print("auto signal count      :", n_present_signal)
    print(
        "cross signal count     :",
        n_present_signal * (n_present_signal - 1) // 2
    )
    print("=======================================")


def print_output_estimate(signal_map):
    n_corr_time = get_n_corr_time()
    n_baseline = len(get_baseline_pairs())
    output_bytes = estimate_output_size_bytes()
    block_cache_bytes = estimate_block_cache_bytes(signal_map)

    print("\n========== OUTPUT ESTIMATE ==========")
    print("vis shape logical :", f"({n_corr_time}, {n_baseline}, {NCHAN})")
    print("vis dtype         :", OUTPUT_DTYPE_NAME)
    print("estimated vis size:", format_bytes(output_bytes))
    print("block cache size  :", format_bytes(block_cache_bytes))
    print("zero fill rule    : missing signal results are saved as 0+0j")
    print("corr output mode  :", CORR_OUTPUT_MODE)
    print("=====================================")


def run_prepare_layer(infos):
    """
    第二部分准备层。

    当前阶段输出映射和数据规模。
    """
    check_corr_config()
    check_nchans(infos)

    add_data_info(infos)
    check_n_fft_consistency(infos)

    signal_map = build_signal_file_map(infos)
    check_antenna_info_for_inputs(signal_map)

    baseline_pairs = get_baseline_pairs()

    print_corr_config()
    print_signal_map(signal_map)
    print_data_info(infos)
    print_baseline_info()
    print_selected_row_info(signal_map, baseline_pairs)
    print_output_estimate(signal_map)

    return signal_map, baseline_pairs


# =========================
# 第三部分：数据读取与相关计算
# =========================

def read_signal_block(file_info, block_start_fft, block_fft_count):
    """
    读取某一路信号的一个FFT block。

    返回：
        x.shape = (block_fft_count, NCHAN)
        dtype = complex64

    数据格式：
        每个频点 2 byte：
            real int8
            imag int8
    """
    bytes_per_fft = get_bytes_per_fft()
    offset = file_info["data_offset"] + block_start_fft * bytes_per_fft
    read_bytes = block_fft_count * bytes_per_fft

    with open_binary_file(file_info["file"]) as f:
        f.seek(offset)
        data = f.read(read_bytes)

    if len(data) != read_bytes:
        raise EOFError(
            f"failed to read data block:\n"
            f"file={file_info['file']}\n"
            f"want={read_bytes}\n"
            f"got={len(data)}"
        )

    raw = np.frombuffer(data, dtype=np.int8)

    expected_count = block_fft_count * NCHAN * BYTES_PER_FREQ_POINT

    if raw.size != expected_count:
        raise ValueError(
            f"bad raw data size:\n"
            f"file={file_info['file']}\n"
            f"raw.size={raw.size}\n"
            f"expected={expected_count}"
        )

    raw = raw.reshape(block_fft_count, NCHAN, BYTES_PER_FREQ_POINT)

    real = raw[:, :, 0].astype(np.float32)
    imag = raw[:, :, 1].astype(np.float32)

    x = real + 1j * imag

    return x.astype(np.complex64, copy=False)


def load_existing_signal_blocks(signal_map, block_start_fft, block_fft_count):
    """
    读取当前block中实际存在的信号。

    返回：
        data_cache

    data_cache[signal_index] = complex array

    缺失信号不读取、不补零。
    """
    data_cache = {}

    for signal_index, file_info in enumerate(signal_map):
        if file_info is None:
            continue

        data_cache[signal_index] = read_signal_block(
            file_info,
            block_start_fft,
            block_fft_count
        )

    return data_cache


def correlate_one_pair(x_i, x_j, fft_per_corr):
    """
    计算一对信号的相关。

    输入：
        x_i.shape = (block_fft_count, NCHAN)
        x_j.shape = (block_fft_count, NCHAN)

    输出：
        v.shape = (n_corr_in_block, NCHAN)

    计算：
        每个FFT先做 Xi * conj(Xj)
        然后每 fft_per_corr 个FFT相加一次。

    注意：
        当前只保留累加值，不做平均。
    """
    block_fft_count = x_i.shape[0]

    if block_fft_count != x_j.shape[0]:
        raise ValueError("x_i and x_j block length mismatch")

    if block_fft_count % fft_per_corr != 0:
        raise ValueError("block_fft_count is not divisible by fft_per_corr")

    n_corr_in_block = block_fft_count // fft_per_corr

    prod = x_i * np.conj(x_j)
    prod = prod.reshape(n_corr_in_block, fft_per_corr, NCHAN)

    if CORR_OUTPUT_MODE == "sum":
        v = prod.sum(axis=1, dtype=np.complex64)
    elif CORR_OUTPUT_MODE == "mean":
        v = prod.mean(axis=1, dtype=np.complex64)
    else:
        raise ValueError(f"bad CORR_OUTPUT_MODE: {CORR_OUTPUT_MODE}")

    return v.astype(np.complex64, copy=False)


def compute_correlation_block(data_cache, baseline_pairs, block_fft_count):
    """
    计算一个block内所有实际存在信号之间的相关。

    返回：
        vis_dict

    vis_dict[baseline_index] = v

    说明：
        如果baseline里有一路缺失，就不计算。
        后面HDF5写入时，这些baseline保持 0+0j。
    """
    fft_per_corr = get_fft_per_corr()

    if block_fft_count % fft_per_corr != 0:
        raise ValueError("block_fft_count is not divisible by fft_per_corr")

    vis_dict = {}

    for baseline_index, pair in enumerate(baseline_pairs):
        i, j = pair

        if i not in data_cache:
            continue

        if j not in data_cache:
            continue

        v = correlate_one_pair(
            data_cache[i],
            data_cache[j],
            fft_per_corr
        )

        vis_dict[baseline_index] = v

    return vis_dict


# =========================
# 第四部分：HDF5保存
# =========================


def mjd_to_unix_ns(mjd_value):
    """
    把 MJD 时间转换成 Unix 纳秒时间。

    假设：
        tstart 是 MJD
        MJD 40587.0 = Unix epoch 1970-01-01 00:00:00 UTC

    返回：
        unix_ns，单位 ns，整数
    """
    mjd = Decimal(str(mjd_value))
    unix_seconds = (mjd - Decimal("40587")) * Decimal("86400")
    unix_ns = unix_seconds * Decimal("1000000000")

    return int(unix_ns.to_integral_value(rounding=ROUND_HALF_UP))


def unix_ns_to_mjd(unix_ns):
    """
    把 Unix 纳秒时间转换回 MJD。
    """
    unix_ns_dec = Decimal(int(unix_ns))
    mjd = Decimal("40587") + unix_ns_dec / Decimal("86400") / Decimal("1000000000")

    return mjd


def unix_ns_to_filename_time(unix_ns):
    """
    把 Unix 纳秒时间转换成文件名用的时间字符串。

    文件名只保留到毫秒：
        YYYYMMDDHHMMSSmmm

    其中：
        mmm 是毫秒

    微秒和纳秒不写进文件名，
    但会写入 HDF5 属性。
    """
    sec = unix_ns // 1_000_000_000
    ns_remain = unix_ns % 1_000_000_000

    dt = datetime.fromtimestamp(sec, tz=timezone.utc)
    ms = ns_remain // 1_000_000

    return dt.strftime("%Y%m%d%H%M%S") + f"{ms:03d}"


def unix_ns_to_iso_text(unix_ns):
    """
    把 Unix 纳秒时间转换成可读 UTC 字符串。

    格式：
        YYYY-MM-DDTHH:MM:SS.nnnnnnnnnZ

    保留到纳秒。
    """
    sec = unix_ns // 1_000_000_000
    ns_remain = unix_ns % 1_000_000_000

    dt = datetime.fromtimestamp(sec, tz=timezone.utc)

    return dt.strftime("%Y-%m-%dT%H:%M:%S") + f".{ns_remain:09d}Z"


def get_hdf5_time_info(infos):
    """
    根据第一个输入文件的 tstart 计算 HDF5 数据开始和结束时间。

    开始时间：
        header["tstart"]

    结束时间：
        start_time + n_corr_time * CORR_TIME_US

    说明：
        这里的结束时间是数据结束边界，也就是 exclusive end。
        例如有 500 个 1ms 积分点，结束时间 = 开始时间 + 500 ms。
    """
    ref_header = infos[0]["header"]

    if "tstart" not in ref_header:
        raise ValueError("missing tstart in header")

    start_mjd = ref_header["tstart"]
    n_corr_time = get_n_corr_time()

    duration_us = n_corr_time * CORR_TIME_US
    duration_ns = duration_us * 1000

    start_unix_ns = mjd_to_unix_ns(start_mjd)
    end_unix_ns = start_unix_ns + duration_ns

    end_mjd = unix_ns_to_mjd(end_unix_ns)

    start_name = unix_ns_to_filename_time(start_unix_ns)
    end_name = unix_ns_to_filename_time(end_unix_ns)

    return {
        "start_mjd": start_mjd,
        "end_mjd": end_mjd,
        "start_unix_ns": start_unix_ns,
        "end_unix_ns": end_unix_ns,
        "start_utc": unix_ns_to_iso_text(start_unix_ns),
        "end_utc": unix_ns_to_iso_text(end_unix_ns),
        "start_name": start_name,
        "end_name": end_name,
        "duration_us": duration_us,
        "duration_ns": duration_ns,
        "n_corr_time": n_corr_time,
    }


def get_output_hdf5_file(infos):
    """
    获取 HDF5 输出文件名。

    如果 OUTPUT_HDF5_FILE 不为 None：
        使用用户手动指定的文件名。

    如果 OUTPUT_HDF5_FILE 为 None：
        根据第一个输入文件 header["tstart"] 自动生成时间段文件名：

        YYYYMMDDHHMMSSmmm_YYYYMMDDHHMMSSmmm.h5
    """
    if OUTPUT_HDF5_FILE is not None:
        return OUTPUT_HDF5_FILE

    time_info = get_hdf5_time_info(infos)

    return f"{time_info['start_name']}_{time_info['end_name']}.h5"


def check_output_file(output_file):
    if os.path.exists(output_file):
        if OVERWRITE_OUTPUT:
            os.remove(output_file)
        else:
            raise FileExistsError(f"output file already exists: {output_file}")




def write_correlation_block(h5, corr_start, corr_end, vis_dict):
    """
    把一个block的相关结果写入HDF5。

    vis_dict:
        key   = baseline_index
        value = v.shape = (n_corr_in_block, NCHAN)

    没有出现在 vis_dict 里的baseline不写，
    HDF5中保持 fillvalue 0+0j。
    """
    vis = h5["vis"]

    for baseline_index, v in vis_dict.items():
        vis[corr_start:corr_end, baseline_index, :] = v



# Clean MS-ready HDF5 writer definitions.
def get_string_dtype():
    return h5py.string_dtype(encoding="utf-8")


def _split_angle_text(text):
    if text is None:
        raise ValueError("angle text is None")

    clean = str(text).strip()

    if clean == "":
        raise ValueError("angle text is empty")

    clean = clean.lower()
    clean = clean.replace("h", " ")
    clean = clean.replace("m", " ")
    clean = clean.replace("s", " ")
    clean = clean.replace(":", " ")
    parts = [part for part in clean.split() if part != ""]

    if len(parts) != 3:
        raise ValueError(f"bad angle text: {text}")

    return parts


def parse_hms_to_rad(text):
    """
    Convert RA string 'HH:MM:SS.s' to radians.

    Example:
        '19:35:00' -> 293.75 deg -> radians
    """
    parts = _split_angle_text(text)

    try:
        hour = int(parts[0])
        minute = int(parts[1])
        second = float(parts[2])
    except ValueError as error:
        raise ValueError(f"bad RA text: {text}") from error

    if hour < 0 or hour >= 24:
        raise ValueError(f"RA hour out of range: {hour}")
    if minute < 0 or minute >= 60:
        raise ValueError(f"RA minute out of range: {minute}")
    if second < 0.0 or second >= 60.0:
        raise ValueError(f"RA second out of range: {second}")

    hours = hour + minute / 60.0 + second / 3600.0
    degree = hours * 15.0

    return np.deg2rad(degree)


def parse_dms_to_rad(text):
    """
    Convert Dec string '[+/-]DD:MM:SS.s' to radians.

    Example:
        '21:54:00' -> +21.9 deg -> radians
    """
    parts = _split_angle_text(text)
    degree_text = parts[0]
    sign = -1.0 if degree_text.startswith("-") else 1.0

    try:
        degree = abs(int(degree_text))
        minute = int(parts[1])
        second = float(parts[2])
    except ValueError as error:
        raise ValueError(f"bad Dec text: {text}") from error

    if minute < 0 or minute >= 60:
        raise ValueError(f"Dec minute out of range: {minute}")
    if second < 0.0 or second >= 60.0:
        raise ValueError(f"Dec second out of range: {second}")

    abs_degree = degree + minute / 60.0 + second / 3600.0
    signed_degree = sign * abs_degree

    if signed_degree < -90.0 or signed_degree > 90.0:
        raise ValueError(f"Dec degree out of range: {signed_degree}")

    return np.deg2rad(signed_degree)


def get_field_phase_center():
    """
    Return phase center metadata:
        ra_rad
        dec_rad
        ra_deg
        dec_deg
        ra_hms
        dec_dms
        frame
    """
    if FIELD_RA_RAD is not None and FIELD_DEC_RAD is not None:
        ra_rad = float(FIELD_RA_RAD)
        dec_rad = float(FIELD_DEC_RAD)
    else:
        ra_rad = float(parse_hms_to_rad(FIELD_RA_HMS))
        dec_rad = float(parse_dms_to_rad(FIELD_DEC_DMS))

    return {
        "ra_rad": ra_rad,
        "dec_rad": dec_rad,
        "ra_deg": np.rad2deg(ra_rad),
        "dec_deg": np.rad2deg(dec_rad),
        "ra_hms": FIELD_RA_HMS,
        "dec_dms": FIELD_DEC_DMS,
        "frame": FIELD_FRAME,
    }


def parse_antenna_name_to_id(name):
    """
    把天线名转换成天线编号。

    例子：
        ant0  -> 0
        ant9  -> 9
        ANT03 -> 3
    """
    text = str(name).strip()
    prefix = ANTENNA_TXT_NAME_PREFIX

    if not text.lower().startswith(prefix.lower()):
        raise ValueError(
            f"bad antenna name '{name}', expected format like {prefix}0"
        )

    number_text = text[len(prefix):]

    if number_text == "" or not number_text.isdigit():
        raise ValueError(
            f"bad antenna name '{name}', expected format like {prefix}0"
        )

    antenna_id = int(number_text)

    if antenna_id < MIN_ANTENNA_ID or antenna_id > MAX_ANTENNA_ID:
        raise ValueError(
            f"antenna id out of range in antenna txt: {name} -> {antenna_id}"
        )

    return antenna_id


def geodetic_to_itrf_m(lat_deg, lon_deg, alt_m):
    """
    WGS84 经纬度高程转 ITRF/ECEF XYZ，单位：米。

    输入：
        lat_deg : 纬度，单位 degree
        lon_deg : 经度，单位 degree
        alt_m   : 海拔，单位 m

    输出：
        np.array([x, y, z], dtype=float64)，单位 m
    """
    lat_rad = np.deg2rad(float(lat_deg))
    lon_rad = np.deg2rad(float(lon_deg))
    alt_m = float(alt_m)

    # WGS84 ellipsoid
    semi_major_axis_m = 6378137.0
    flattening = 1.0 / 298.257223563
    eccentricity2 = flattening * (2.0 - flattening)

    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    cos_lon = np.cos(lon_rad)
    sin_lon = np.sin(lon_rad)

    normal_radius = semi_major_axis_m / np.sqrt(
        1.0 - eccentricity2 * sin_lat * sin_lat
    )

    x = (normal_radius + alt_m) * cos_lat * cos_lon
    y = (normal_radius + alt_m) * cos_lat * sin_lon
    z = (normal_radius * (1.0 - eccentricity2) + alt_m) * sin_lat

    return np.array([x, y, z], dtype=np.float64)


def deg_to_dms_str_for_katpoint(value_deg):
    """
    Convert decimal degrees to katpoint DMS string.

    Example:
        29.784402 -> "29:47:03.8472"
        -29.784402 -> "-29:47:03.8472"

    This follows the logic from CARRY_antenna_uv.py.
    """
    sign = "-" if float(value_deg) < 0.0 else ""
    value = abs(float(value_deg))

    degree = int(value)
    minute_float = (value - degree) * 60.0
    minute = int(minute_float)
    second = (minute_float - minute) * 60.0

    second = round(second, 4)

    if second >= 60.0:
        second -= 60.0
        minute += 1

    if minute >= 60:
        minute -= 60
        degree += 1

    return f"{sign}{degree}:{minute:02d}:{second:07.4f}"


def read_antenna_info_txt(file_path):
    """
    读取天线位置 txt 文件。

    支持格式：
        # name lat lon [alt_m] [diam_m]
        ant0 29.784402 109.779625 1581 7.5

    说明：
        name 的数字部分就是天线编号。
        ant0 是第 1 面天线，对应 antenna_id=0。
        ant9 是第 10 面天线，对应 antenna_id=9。
    """
    if file_path is None:
        return {}

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"antenna txt not found: {file_path}")

    catalog = {}
    used_names = {}

    with open(file_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()

            if text == "" or text.startswith("#"):
                continue

            parts = text.split()

            if len(parts) < 3:
                raise ValueError(
                    f"bad antenna txt line {line_no}: need at least name lat lon"
                )

            name = parts[0]
            antenna_id = parse_antenna_name_to_id(name)

            if antenna_id in catalog:
                raise ValueError(
                    f"duplicate antenna id in antenna txt: {name} -> {antenna_id}"
                )

            name_key = name.lower()
            if name_key in used_names:
                raise ValueError(
                    f"duplicate antenna name in antenna txt: {name}"
                )
            used_names[name_key] = line_no

            try:
                lat_deg = float(parts[1])
                lon_deg = float(parts[2])
                alt_m = float(parts[3]) if len(parts) >= 4 else 0.0
                diam_m = (
                    float(parts[4])
                    if len(parts) >= 5
                    else float(ANTENNA_DISH_DIAMETER_M)
                )
            except ValueError as error:
                raise ValueError(
                    f"bad numeric value in antenna txt line {line_no}: {text}"
                ) from error

            if lat_deg < -90.0 or lat_deg > 90.0:
                raise ValueError(
                    f"bad latitude in antenna txt line {line_no}: {lat_deg}"
                )

            if lon_deg < -180.0 or lon_deg > 360.0:
                raise ValueError(
                    f"bad longitude in antenna txt line {line_no}: {lon_deg}"
                )

            if diam_m <= 0.0:
                raise ValueError(
                    f"bad dish diameter in antenna txt line {line_no}: {diam_m}"
                )

            catalog[antenna_id] = {
                "id": antenna_id,
                "name": name,
                "station": name,
                "lat_deg": lat_deg,
                "lon_deg": lon_deg,
                "alt_m": alt_m,
                "diam_m": diam_m,
                "position_itrf_m": geodetic_to_itrf_m(
                    lat_deg,
                    lon_deg,
                    alt_m
                ),
                "line_no": line_no,
            }

    if len(catalog) == 0:
        raise ValueError(f"antenna txt has no antenna rows: {file_path}")

    return catalog


def get_antenna_catalog():
    """
    读取并缓存天线 txt。
    """
    global _ANTENNA_INFO_CACHE
    global _ANTENNA_INFO_CACHE_PATH

    if ANTENNA_INFO_TXT is None:
        return {}

    if (
        _ANTENNA_INFO_CACHE is not None
        and _ANTENNA_INFO_CACHE_PATH == ANTENNA_INFO_TXT
    ):
        return _ANTENNA_INFO_CACHE

    _ANTENNA_INFO_CACHE = read_antenna_info_txt(ANTENNA_INFO_TXT)
    _ANTENNA_INFO_CACHE_PATH = ANTENNA_INFO_TXT

    return _ANTENNA_INFO_CACHE


def get_used_antenna_ids(signal_map):
    """
    根据实际输入的 filterbank 文件，找出哪些物理天线真的参与了本次计算。
    """
    used = set()

    for item in signal_map:
        if item is None:
            continue

        used.add(int(item["fname"]["antenna_id"]))

    return sorted(used)


def check_antenna_info_for_inputs(signal_map):
    """
    检查天线 txt 能否覆盖本次实际输入的天线。

    注意：
        filterbank 输入可以只包含部分天线。
        但是只要某个天线出现在输入文件名里，天线 txt 就必须有对应 antX。
    """
    used_ids = get_used_antenna_ids(signal_map)

    if ANTENNA_INFO_TXT is None:
        print("[WARN] no antenna txt given, antenna positions will be placeholders")
        return

    catalog = get_antenna_catalog()
    missing_ids = [antenna_id for antenna_id in used_ids if antenna_id not in catalog]

    if len(missing_ids) > 0:
        missing_names = [f"ant{antenna_id}" for antenna_id in missing_ids]
        raise ValueError(
            "antenna txt is missing antennas used by input files: "
            + ", ".join(missing_names)
        )

    print("[OK] antenna txt loaded:", ANTENNA_INFO_TXT)
    print("[OK] antenna txt rows:", len(catalog))
    print(
        "[OK] used antennas covered by antenna txt:",
        ", ".join([f"ant{antenna_id}" for antenna_id in used_ids])
    )


def build_antenna_axis_metadata(signal_map=None):
    """
    生成 HDF5 /antenna 分组要保存的元数据。

    对于天线 txt 中存在的天线：
        使用 txt 中的 name / lat / lon / alt / diam，
        并把经纬度海拔转换成 ITRF/ECEF XYZ。

    对于 txt 中不存在的天线：
        保留占位坐标，避免只输入部分天线时程序崩溃。
        后续转 MS 时，这类天线如果没有数据行，应保持 flag。
    """
    catalog = get_antenna_catalog()
    antenna_ids = np.arange(
        MIN_ANTENNA_ID,
        MIN_ANTENNA_ID + N_PHYSICAL_ANTENNAS,
        dtype=np.int16
    )

    names = []
    stations = []
    latitude_deg = np.full(N_PHYSICAL_ANTENNAS, np.nan, dtype=np.float64)
    longitude_deg = np.full(N_PHYSICAL_ANTENNAS, np.nan, dtype=np.float64)
    altitude_m = np.full(N_PHYSICAL_ANTENNAS, np.nan, dtype=np.float64)
    dish_diameter_m = np.full(
        N_PHYSICAL_ANTENNAS,
        float(ANTENNA_DISH_DIAMETER_M),
        dtype=np.float64
    )
    position_itrf_m = np.zeros((N_PHYSICAL_ANTENNAS, 3), dtype=np.float64)
    present_in_txt = np.zeros(N_PHYSICAL_ANTENNAS, dtype=np.int8)
    position_is_placeholder_by_antenna = np.ones(
        N_PHYSICAL_ANTENNAS,
        dtype=np.int8
    )

    if signal_map is None:
        used_in_input = np.zeros(N_PHYSICAL_ANTENNAS, dtype=np.int8)
    else:
        used_in_input = np.zeros(N_PHYSICAL_ANTENNAS, dtype=np.int8)
        for antenna_id in get_used_antenna_ids(signal_map):
            used_in_input[antenna_id - MIN_ANTENNA_ID] = 1

    for local_index, antenna_id in enumerate(antenna_ids):
        antenna_id_int = int(antenna_id)
        default_name = ANTENNA_NAMES[local_index]
        default_station = ANTENNA_STATIONS[local_index]

        item = catalog.get(antenna_id_int)

        if item is None:
            names.append(default_name)
            stations.append(default_station)
            position_itrf_m[local_index, 0] = float(antenna_id_int) * 10.0
            continue

        names.append(item["name"])
        stations.append(item["station"])
        latitude_deg[local_index] = item["lat_deg"]
        longitude_deg[local_index] = item["lon_deg"]
        altitude_m[local_index] = item["alt_m"]
        dish_diameter_m[local_index] = item["diam_m"]
        position_itrf_m[local_index, :] = item["position_itrf_m"]
        present_in_txt[local_index] = 1
        position_is_placeholder_by_antenna[local_index] = 0

    if ANTENNA_POSITION_ITRF_M is not None:
        manual_positions = np.asarray(ANTENNA_POSITION_ITRF_M, dtype=np.float64)

        if manual_positions.shape != (N_PHYSICAL_ANTENNAS, 3):
            raise ValueError(
                "ANTENNA_POSITION_ITRF_M must have shape "
                f"({N_PHYSICAL_ANTENNAS}, 3)"
            )

        position_itrf_m = manual_positions
        position_is_placeholder_by_antenna[:] = 0

    position_is_placeholder = np.int8(
        np.any(position_is_placeholder_by_antenna != 0)
    )

    return {
        "id": antenna_ids,
        "name": np.array(names, dtype=object),
        "station": np.array(stations, dtype=object),
        "latitude_deg": latitude_deg,
        "longitude_deg": longitude_deg,
        "altitude_m": altitude_m,
        "position_itrf_m": position_itrf_m,
        "dish_diameter_m": dish_diameter_m,
        "present_in_antenna_txt": present_in_txt,
        "used_in_input": used_in_input,
        "position_is_placeholder_by_antenna": position_is_placeholder_by_antenna,
        "position_is_placeholder": position_is_placeholder,
    }


def build_katpoint_antennas(signal_map):
    """
    Build katpoint.Antenna objects for all physical antennas.

    Returns:
        kat_antennas: list length N_PHYSICAL_ANTENNAS
            kat_antennas[antenna_id] is a katpoint.Antenna object
            or None if this antenna is not in antenna txt.

    Rules:
        1. For antennas used by input .fil files, antenna txt must contain real lat/lon/alt.
        2. For unused antennas, None is allowed.
        3. Do not use placeholder positions for real UVW.
    """
    if katpoint is None:
        raise RuntimeError(
            f"katpoint is unavailable, cannot compute real UVW: "
            f"{KATPOINT_IMPORT_ERROR}"
        )

    catalog = get_antenna_catalog()
    used_ids = set(get_used_antenna_ids(signal_map))
    kat_antennas = [None] * N_PHYSICAL_ANTENNAS

    for antenna_id in range(MIN_ANTENNA_ID, MAX_ANTENNA_ID + 1):
        local_index = antenna_id - MIN_ANTENNA_ID
        item = catalog.get(antenna_id)

        if item is None:
            if antenna_id in used_ids:
                raise ValueError(
                    f"cannot compute UVW: used antenna ant{antenna_id} "
                    "is missing from antenna txt"
                )
            continue

        name = item["name"]
        lat_dms = deg_to_dms_str_for_katpoint(item["lat_deg"])
        lon_dms = deg_to_dms_str_for_katpoint(item["lon_deg"])
        alt_m = float(item["alt_m"])
        diam_m = float(item["diam_m"])

        desc = f"{name}, {lat_dms}, {lon_dms}, {alt_m}, {diam_m}"
        kat_antennas[local_index] = katpoint.Antenna(desc)

    return kat_antennas


def build_katpoint_target():
    """
    Build katpoint.Target from configured phase center.

    Uses:
        FIELD_RA_HMS
        FIELD_DEC_DMS
    """
    if katpoint is None:
        raise RuntimeError(
            f"katpoint is unavailable, cannot compute real UVW: "
            f"{KATPOINT_IMPORT_ERROR}"
        )

    return katpoint.Target(
        f"PhaseCenter, radec, {FIELD_RA_HMS}, {FIELD_DEC_DMS}"
    )


def validate_ms_ready_config(signal_map=None):
    if N_PHYSICAL_ANTENNAS != MAX_ANTENNA_ID - MIN_ANTENNA_ID + 1:
        raise ValueError(
            "N_PHYSICAL_ANTENNAS must match filename antenna id range"
        )

    if N_INPUT_SIGNALS != N_PHYSICAL_ANTENNAS * len(VALID_POLARIZATIONS):
        raise ValueError(
            "N_INPUT_SIGNALS must equal N_PHYSICAL_ANTENNAS * number of polarizations"
        )

    if len(ANTENNA_NAMES) != N_PHYSICAL_ANTENNAS:
        raise ValueError("ANTENNA_NAMES length mismatch")

    if len(ANTENNA_STATIONS) != N_PHYSICAL_ANTENNAS:
        raise ValueError("ANTENNA_STATIONS length mismatch")

    if FREQUENCY_INPUT_UNIT not in ["MHz", "Hz"]:
        raise ValueError("FREQUENCY_INPUT_UNIT must be 'MHz' or 'Hz'")

    if MS_EXPORT_POL_MODE not in ["ALL_SIGNAL_PAIRS"]:
        raise ValueError(
            "current version only supports "
            "MS_EXPORT_POL_MODE='ALL_SIGNAL_PAIRS'"
        )

    if MS_ROW_SELECTION_MODE != "PRESENT_SIGNALS_ONLY":
        raise ValueError(
            "current version only supports "
            "MS_ROW_SELECTION_MODE='PRESENT_SIGNALS_ONLY'"
        )

    if CORR_OUTPUT_MODE not in ["sum", "mean"]:
        raise ValueError("CORR_OUTPUT_MODE must be 'sum' or 'mean'")

    phase_center = get_field_phase_center()

    if not np.isfinite(phase_center["ra_rad"]):
        raise ValueError("bad phase center RA")

    if not np.isfinite(phase_center["dec_rad"]):
        raise ValueError("bad phase center Dec")

    if katpoint is None:
        raise RuntimeError(
            f"katpoint is unavailable, cannot compute real UVW: "
            f"{KATPOINT_IMPORT_ERROR}"
        )

    if ANTENNA_POSITION_ITRF_M is not None:
        positions = np.asarray(ANTENNA_POSITION_ITRF_M, dtype=np.float64)

        if positions.shape != (N_PHYSICAL_ANTENNAS, 3):
            raise ValueError(
                "ANTENNA_POSITION_ITRF_M must have shape "
                f"({N_PHYSICAL_ANTENNAS}, 3)"
            )

    if signal_map is not None:
        antenna_meta = build_antenna_axis_metadata(signal_map)
        used = antenna_meta["used_in_input"].astype(bool)
        placeholder = antenna_meta[
            "position_is_placeholder_by_antenna"
        ].astype(bool)

        if np.any(used & placeholder):
            raise ValueError(
                "cannot compute real UVW: some used antennas have placeholder positions"
            )


def signal_index_to_antenna_pol(signal_index):
    if signal_index < 0 or signal_index >= N_INPUT_SIGNALS:
        raise ValueError(f"signal index out of range: {signal_index}")

    n_pol = len(VALID_POLARIZATIONS)
    antenna_id = MIN_ANTENNA_ID + signal_index // n_pol
    polarization_id = signal_index % n_pol

    return antenna_id, polarization_id


def get_corr_name(pol_i, pol_j):
    names = {
        0: POL0_NAME,
        1: POL1_NAME,
    }
    return names[pol_i] + names[pol_j]


def build_selected_signal_baseline_indices(signal_map, baseline_pairs):
    """
    Return baseline indices where both signal_i and signal_j exist in input files.
    """
    present = np.array([item is not None for item in signal_map], dtype=bool)
    selected = []

    for baseline_index, (signal_i, signal_j) in enumerate(baseline_pairs):
        if present[signal_i] and present[signal_j]:
            selected.append(baseline_index)

    return np.array(selected, dtype=np.int32)


def build_signal_axis_metadata(signal_map):
    signal_present = np.zeros(N_INPUT_SIGNALS, dtype=np.int8)
    input_signal_no = np.arange(1, N_INPUT_SIGNALS + 1, dtype=np.int16)
    antenna_ids = np.zeros(N_INPUT_SIGNALS, dtype=np.int16)
    polarization_ids = np.zeros(N_INPUT_SIGNALS, dtype=np.int16)
    file_names = []

    for index in range(N_INPUT_SIGNALS):
        antenna_id, polarization_id = signal_index_to_antenna_pol(index)
        antenna_ids[index] = antenna_id
        polarization_ids[index] = polarization_id

        item = signal_map[index]

        if item is None:
            file_names.append("")
            continue

        fname = item["fname"]

        if fname["antenna_id"] != antenna_id:
            raise ValueError(
                f"signal index {index} antenna mismatch: "
                f"{fname['antenna_id']} != {antenna_id}"
            )

        if fname["polarization"] != polarization_id:
            raise ValueError(
                f"signal index {index} polarization mismatch: "
                f"{fname['polarization']} != {polarization_id}"
            )

        signal_present[index] = 1
        file_names.append(item["file"])

    return {
        "present": signal_present,
        "input_signal_no": input_signal_no,
        "antenna_id": antenna_ids,
        "polarization_id": polarization_ids,
        "file": np.array(file_names, dtype=object),
    }


def build_baseline_metadata(baseline_pairs):
    signal_pairs = np.array(baseline_pairs, dtype=np.int16)
    antenna_pairs = np.zeros((len(baseline_pairs), 2), dtype=np.int16)
    polarization_pairs = np.zeros((len(baseline_pairs), 2), dtype=np.int16)

    for baseline_index, (signal_i, signal_j) in enumerate(baseline_pairs):
        antenna_i, pol_i = signal_index_to_antenna_pol(signal_i)
        antenna_j, pol_j = signal_index_to_antenna_pol(signal_j)

        antenna_pairs[baseline_index, 0] = antenna_i
        antenna_pairs[baseline_index, 1] = antenna_j
        polarization_pairs[baseline_index, 0] = pol_i
        polarization_pairs[baseline_index, 1] = pol_j

    return {
        "signal_pairs": signal_pairs,
        "antenna_pairs": antenna_pairs,
        "polarization_pairs": polarization_pairs,
    }


def build_ms_row_mapping(signal_map, baseline_pairs):
    selected_signal_baseline_index = build_selected_signal_baseline_indices(
        signal_map,
        baseline_pairs
    )

    selected_signal_i = []
    selected_signal_j = []
    selected_antenna1 = []
    selected_antenna2 = []
    selected_pol_i = []
    selected_pol_j = []
    selected_corr_name = []

    for baseline_index in selected_signal_baseline_index:
        signal_i, signal_j = baseline_pairs[baseline_index]
        antenna_i, pol_i = signal_index_to_antenna_pol(signal_i)
        antenna_j, pol_j = signal_index_to_antenna_pol(signal_j)

        selected_signal_i.append(signal_i)
        selected_signal_j.append(signal_j)
        selected_antenna1.append(antenna_i)
        selected_antenna2.append(antenna_j)
        selected_pol_i.append(pol_i)
        selected_pol_j.append(pol_j)
        selected_corr_name.append(get_corr_name(pol_i, pol_j))

    selected_signal_i = np.array(selected_signal_i, dtype=np.int16)
    selected_signal_j = np.array(selected_signal_j, dtype=np.int16)
    selected_antenna1 = np.array(selected_antenna1, dtype=np.int16)
    selected_antenna2 = np.array(selected_antenna2, dtype=np.int16)
    selected_pol_i = np.array(selected_pol_i, dtype=np.int16)
    selected_pol_j = np.array(selected_pol_j, dtype=np.int16)
    selected_corr_name = np.array(selected_corr_name, dtype=object)

    n_corr_time = get_n_corr_time()
    selected_baseline_count = len(selected_signal_baseline_index)

    time_index = np.repeat(
        np.arange(n_corr_time, dtype=np.int32),
        selected_baseline_count
    )
    signal_baseline_index = np.tile(
        selected_signal_baseline_index,
        n_corr_time
    )
    signal_i = np.tile(selected_signal_i, n_corr_time)
    signal_j = np.tile(selected_signal_j, n_corr_time)
    antenna1 = np.tile(selected_antenna1, n_corr_time)
    antenna2 = np.tile(selected_antenna2, n_corr_time)
    pol_i = np.tile(selected_pol_i, n_corr_time)
    pol_j = np.tile(selected_pol_j, n_corr_time)
    corr_name = np.tile(selected_corr_name, n_corr_time)
    n_ms_rows = time_index.size

    row_is_auto_signal = signal_i == signal_j
    row_is_cross_signal = signal_i != signal_j
    row_is_same_antenna = antenna1 == antenna2
    row_is_cross_antenna = antenna1 != antenna2

    return {
        "time_index": time_index,
        "signal_baseline_index": signal_baseline_index,
        "signal_i": signal_i,
        "signal_j": signal_j,
        "antenna1": antenna1,
        "antenna2": antenna2,
        "pol_i": pol_i,
        "pol_j": pol_j,
        "corr_name": corr_name,
        "data_desc_id": np.zeros(n_ms_rows, dtype=np.int32),
        "field_id": np.zeros(n_ms_rows, dtype=np.int32),
        "scan_number": np.ones(n_ms_rows, dtype=np.int32),
        "row_is_auto_signal": row_is_auto_signal,
        "row_is_cross_signal": row_is_cross_signal,
        "row_is_same_antenna": row_is_same_antenna,
        "row_is_cross_antenna": row_is_cross_antenna,
        "selected_baseline_count": selected_baseline_count,
        "n_ms_rows": n_ms_rows,
    }


def build_row_missing_signal_flags(signal_meta, baseline_meta, ms_row_map):
    present = signal_meta["present"].astype(bool)
    row_signal_i = ms_row_map["signal_i"]
    row_signal_j = ms_row_map["signal_j"]

    missing_signal_i = ~present[row_signal_i]
    missing_signal_j = ~present[row_signal_j]

    return np.logical_or(missing_signal_i, missing_signal_j)


def write_signal_metadata(h5, signal_map):
    signal_meta = build_signal_axis_metadata(signal_map)
    string_dtype = get_string_dtype()

    h5.create_dataset("signal_present", data=signal_meta["present"])
    h5.create_dataset("signal_antenna_id", data=signal_meta["antenna_id"])
    h5.create_dataset(
        "signal_polarization",
        data=signal_meta["polarization_id"]
    )
    h5.create_dataset("input_signal_no", data=signal_meta["input_signal_no"])
    h5.create_dataset(
        "signal_files",
        data=signal_meta["file"],
        dtype=string_dtype
    )


def write_global_metadata(h5, infos, n_corr_time, n_baseline):
    ref_header = infos[0]["header"]
    ref_fname = infos[0]["fname"]
    time_info = get_hdf5_time_info(infos)

    h5.attrs["source_name"] = str(ref_header.get("source_name", ""))
    h5.attrs["time_tag"] = str(ref_fname.get("time_tag", ""))
    h5.attrs["tstart"] = ref_header.get("tstart", 0.0)
    h5.attrs["data_start_mjd"] = float(time_info["start_mjd"])
    h5.attrs["data_end_mjd"] = float(time_info["end_mjd"])
    h5.attrs["data_start_unix_ns"] = np.int64(time_info["start_unix_ns"])
    h5.attrs["data_end_unix_ns"] = np.int64(time_info["end_unix_ns"])
    h5.attrs["data_start_utc"] = time_info["start_utc"]
    h5.attrs["data_end_utc"] = time_info["end_utc"]
    h5.attrs["data_start_name_ms"] = time_info["start_name"]
    h5.attrs["data_end_name_ms"] = time_info["end_name"]
    h5.attrs["data_duration_us"] = np.int64(time_info["duration_us"])
    h5.attrs["data_duration_ns"] = np.int64(time_info["duration_ns"])
    h5.attrs["data_end_is_exclusive"] = np.int8(1)
    h5.attrs["fch1"] = ref_header.get("fch1", 0.0)
    h5.attrs["foff"] = ref_header.get("foff", 0.0)
    h5.attrs["nchans"] = NCHAN
    h5.attrs["required_nbits"] = REQUIRED_NBITS
    h5.attrs["bits_per_freq_point"] = BITS_PER_FREQ_POINT
    h5.attrs["bytes_per_freq_point"] = BYTES_PER_FREQ_POINT
    h5.attrs["fft_time_us"] = FFT_TIME_US
    h5.attrs["corr_time_us"] = CORR_TIME_US
    h5.attrs["fft_per_corr"] = get_fft_per_corr()
    h5.attrs["fft_per_file"] = FFT_PER_FILE
    h5.attrs["read_fft_per_block"] = READ_FFT_PER_BLOCK
    h5.attrs["n_corr_time"] = n_corr_time
    h5.attrs["n_baseline"] = n_baseline
    h5.attrs["output_dtype"] = OUTPUT_DTYPE_NAME
    h5.attrs["corr_output_mode"] = CORR_OUTPUT_MODE
    h5.attrs["corr_save_mode"] = CORR_OUTPUT_MODE
    h5.attrs["corr_normalization_factor"] = (
        np.float32(1.0)
        if CORR_OUTPUT_MODE == "sum"
        else np.float32(get_fft_per_corr())
    )


def write_signal_group(h5, signal_map):
    signal_meta = build_signal_axis_metadata(signal_map)
    string_dtype = get_string_dtype()
    group = h5.create_group("signal")

    group.create_dataset("present", data=signal_meta["present"])
    group.create_dataset("input_signal_no", data=signal_meta["input_signal_no"])
    group.create_dataset("antenna_id", data=signal_meta["antenna_id"])
    group.create_dataset(
        "polarization_id",
        data=signal_meta["polarization_id"]
    )
    group.create_dataset(
        "file",
        data=signal_meta["file"],
        dtype=string_dtype
    )


def write_baseline_group(h5, baseline_pairs):
    baseline_meta = build_baseline_metadata(baseline_pairs)
    group = h5.create_group("baseline")

    group.create_dataset("signal_pairs", data=baseline_meta["signal_pairs"])
    group.create_dataset("antenna_pairs", data=baseline_meta["antenna_pairs"])
    group.create_dataset(
        "polarization_pairs",
        data=baseline_meta["polarization_pairs"]
    )


def write_time_group(h5, infos):
    group = h5.create_group("time")

    n_corr_time = get_n_corr_time()
    interval_sec = CORR_TIME_US * 1e-6
    interval_day = interval_sec / 86400.0
    start_mjd0 = float(infos[0]["header"]["tstart"])
    index = np.arange(n_corr_time, dtype=np.float64)

    start_mjd = start_mjd0 + index * interval_day
    center_mjd = start_mjd0 + (index + 0.5) * interval_day
    end_mjd = start_mjd0 + (index + 1.0) * interval_day

    group.create_dataset("start_mjd", data=start_mjd.astype(np.float64))
    group.create_dataset("center_mjd", data=center_mjd.astype(np.float64))
    group.create_dataset("end_mjd", data=end_mjd.astype(np.float64))
    group.create_dataset("interval_sec", data=np.float64(interval_sec))
    group.create_dataset("exposure_sec", data=np.float64(interval_sec))


def write_frequency_group(h5, infos):
    ref_header = infos[0]["header"]
    fch1 = float(ref_header.get("fch1", 0.0))
    foff = float(ref_header.get("foff", 0.0))
    scale = 1e6 if FREQUENCY_INPUT_UNIT == "MHz" else 1.0
    channel_order = "descending" if foff < 0 else "ascending"

    chan_freq_hz = (fch1 + np.arange(NCHAN, dtype=np.float64) * foff) * scale
    chan_width_hz = np.full(NCHAN, abs(foff) * scale, dtype=np.float64)

    group = h5.create_group("frequency")
    group.create_dataset("chan_freq_hz", data=chan_freq_hz)
    group.create_dataset("chan_width_hz", data=chan_width_hz)
    group.create_dataset("ref_frequency_hz", data=np.float64(fch1 * scale))
    group.create_dataset("nchan", data=np.int32(NCHAN))
    group.attrs["input_unit"] = FREQUENCY_INPUT_UNIT
    group.attrs["fch1_original"] = fch1
    group.attrs["foff_original"] = foff
    group.attrs["channel_order"] = channel_order


def write_antenna_group(h5, signal_map):
    string_dtype = get_string_dtype()
    group = h5.create_group("antenna")
    antenna_meta = build_antenna_axis_metadata(signal_map)

    group.create_dataset("id", data=antenna_meta["id"])
    group.create_dataset(
        "name",
        data=antenna_meta["name"],
        dtype=string_dtype
    )
    group.create_dataset(
        "station",
        data=antenna_meta["station"],
        dtype=string_dtype
    )
    group.create_dataset("latitude_deg", data=antenna_meta["latitude_deg"])
    group.create_dataset("longitude_deg", data=antenna_meta["longitude_deg"])
    group.create_dataset("altitude_m", data=antenna_meta["altitude_m"])
    group.create_dataset("position_itrf_m", data=antenna_meta["position_itrf_m"])
    group.create_dataset("dish_diameter_m", data=antenna_meta["dish_diameter_m"])
    group.create_dataset(
        "present_in_antenna_txt",
        data=antenna_meta["present_in_antenna_txt"]
    )
    group.create_dataset(
        "used_in_input",
        data=antenna_meta["used_in_input"]
    )
    group.create_dataset(
        "position_is_placeholder_by_antenna",
        data=antenna_meta["position_is_placeholder_by_antenna"]
    )
    group.create_dataset(
        "position_is_placeholder",
        data=antenna_meta["position_is_placeholder"]
    )

    group.attrs["antenna_info_txt"] = (
        "" if ANTENNA_INFO_TXT is None else str(ANTENNA_INFO_TXT)
    )
    group.attrs["position_frame"] = ANTENNA_POSITION_FRAME
    group.attrs["lat_lon_unit"] = "degree"
    group.attrs["altitude_unit"] = "m"


def write_field_group(h5, infos):
    string_dtype = get_string_dtype()
    ref_header = infos[0]["header"]
    phase_center = get_field_phase_center()
    group = h5.create_group("field")

    group.create_dataset(
        "source_name",
        data=str(ref_header.get("source_name", "")),
        dtype=string_dtype
    )
    group.create_dataset(
        "phase_center_ra_rad",
        data=np.float64(phase_center["ra_rad"])
    )
    group.create_dataset(
        "phase_center_dec_rad",
        data=np.float64(phase_center["dec_rad"])
    )
    group.create_dataset(
        "phase_center_ra_deg",
        data=np.float64(phase_center["ra_deg"])
    )
    group.create_dataset(
        "phase_center_dec_deg",
        data=np.float64(phase_center["dec_deg"])
    )
    group.create_dataset(
        "phase_center_ra_hms",
        data=str(phase_center["ra_hms"]),
        dtype=string_dtype
    )
    group.create_dataset(
        "phase_center_dec_dms",
        data=str(phase_center["dec_dms"]),
        dtype=string_dtype
    )
    group.create_dataset(
        "frame",
        data=str(phase_center["frame"]),
        dtype=string_dtype
    )
    group.create_dataset("is_placeholder", data=np.int8(0))
    group.attrs["src_raj_header"] = ref_header.get("src_raj", "")
    group.attrs["src_dej_header"] = ref_header.get("src_dej", "")
    group.attrs["phase_center_source"] = "code_config"


def write_polarization_group(h5):
    string_dtype = get_string_dtype()
    group = h5.create_group("polarization")
    all_corr_names = np.array(
        [
            get_corr_name(0, 0),
            get_corr_name(0, 1),
            get_corr_name(1, 0),
            get_corr_name(1, 1),
        ],
        dtype=object
    )
    all_corr_pol_i = np.array([0, 0, 1, 1], dtype=np.int16)
    all_corr_pol_j = np.array([0, 1, 0, 1], dtype=np.int16)

    group.create_dataset("input_pol_id", data=np.array([0, 1], dtype=np.int16))
    group.create_dataset(
        "input_pol_name",
        data=np.array([POL0_NAME, POL1_NAME], dtype=object),
        dtype=string_dtype
    )
    group.create_dataset(
        "ms_export_mode",
        data=MS_EXPORT_POL_MODE,
        dtype=string_dtype
    )
    group.create_dataset(
        "all_corr_names",
        data=all_corr_names,
        dtype=string_dtype
    )
    group.create_dataset("all_corr_pol_i", data=all_corr_pol_i)
    group.create_dataset("all_corr_pol_j", data=all_corr_pol_j)
    group.create_dataset("corr_type", data=all_corr_names, dtype=string_dtype)
    group.create_dataset("corr_pol_i", data=all_corr_pol_i)
    group.create_dataset("corr_pol_j", data=all_corr_pol_j)


def write_ms_rows_group(h5, ms_row_map, row_has_missing_signal):
    string_dtype = get_string_dtype()
    group = h5.create_group("ms_rows")

    group.create_dataset("time_index", data=ms_row_map["time_index"])
    group.create_dataset(
        "signal_baseline_index",
        data=ms_row_map["signal_baseline_index"]
    )
    group.create_dataset("signal_i", data=ms_row_map["signal_i"])
    group.create_dataset("signal_j", data=ms_row_map["signal_j"])
    group.create_dataset("antenna1", data=ms_row_map["antenna1"])
    group.create_dataset("antenna2", data=ms_row_map["antenna2"])
    group.create_dataset("pol_i", data=ms_row_map["pol_i"])
    group.create_dataset("pol_j", data=ms_row_map["pol_j"])
    group.create_dataset(
        "corr_name",
        data=ms_row_map["corr_name"],
        dtype=string_dtype
    )
    group.create_dataset("data_desc_id", data=ms_row_map["data_desc_id"])
    group.create_dataset("field_id", data=ms_row_map["field_id"])
    group.create_dataset("scan_number", data=ms_row_map["scan_number"])
    group.create_dataset(
        "row_has_missing_signal",
        data=np.asarray(row_has_missing_signal, dtype=np.bool_)
    )
    group.create_dataset(
        "row_is_auto_signal",
        data=np.asarray(ms_row_map["row_is_auto_signal"], dtype=np.bool_)
    )
    group.create_dataset(
        "row_is_cross_signal",
        data=np.asarray(ms_row_map["row_is_cross_signal"], dtype=np.bool_)
    )
    group.create_dataset(
        "row_is_same_antenna",
        data=np.asarray(ms_row_map["row_is_same_antenna"], dtype=np.bool_)
    )
    group.create_dataset(
        "row_is_cross_antenna",
        data=np.asarray(ms_row_map["row_is_cross_antenna"], dtype=np.bool_)
    )
    group.attrs["selected_baseline_count"] = int(
        ms_row_map["selected_baseline_count"]
    )
    group.attrs["n_ms_rows"] = int(ms_row_map["n_ms_rows"])
    group.attrs["export_pol_mode"] = MS_EXPORT_POL_MODE


def build_center_time_seconds_for_katpoint(infos):
    """
    Build center timestamps for every correlation integration.

    Returns:
        time_seconds.shape = (n_corr_time,)
        dtype float64

    Unit:
        Unix seconds, compatible with katpoint.Target.uvw().
    """
    n_corr_time = get_n_corr_time()
    interval_sec = CORR_TIME_US * 1e-6
    interval_day = interval_sec / 86400.0
    start_mjd0 = float(infos[0]["header"]["tstart"])
    index = np.arange(n_corr_time, dtype=np.float64)
    center_mjd = start_mjd0 + (index + 0.5) * interval_day
    time_seconds = (center_mjd - 40587.0) * 86400.0

    return time_seconds.astype(np.float64)


def compute_uvw_for_ms_rows_with_katpoint(infos, signal_map, ms_row_map):
    """
    Compute UVW for every selected HDF5/MS row using katpoint.

    Returns:
        uvw_m: np.ndarray, shape (n_ms_rows, 3), dtype float64
    """
    kat_antennas = build_katpoint_antennas(signal_map)
    target = build_katpoint_target()
    time_seconds_axis = build_center_time_seconds_for_katpoint(infos)

    n_ms_rows = int(ms_row_map["n_ms_rows"])
    uvw_m = np.zeros((n_ms_rows, 3), dtype=np.float64)
    selected_baseline_count = int(ms_row_map["selected_baseline_count"])

    for baseline_slot in range(selected_baseline_count):
        row0 = baseline_slot
        ant1_id = int(ms_row_map["antenna1"][row0])
        ant2_id = int(ms_row_map["antenna2"][row0])
        rows = np.arange(
            baseline_slot,
            n_ms_rows,
            selected_baseline_count,
            dtype=np.int64
        )

        if ant1_id == ant2_id:
            uvw_m[rows, :] = 0.0
            continue

        local1 = ant1_id - MIN_ANTENNA_ID
        local2 = ant2_id - MIN_ANTENNA_ID
        ant1 = kat_antennas[local1]
        ant2 = kat_antennas[local2]

        if ant1 is None or ant2 is None:
            raise ValueError(
                f"cannot compute UVW: missing katpoint antenna "
                f"ant{ant1_id} or ant{ant2_id}"
            )

        u, v, w = target.uvw(ant1, time_seconds_axis, ant2)

        # katpoint.Target.uvw(ant1, time, ant2) returns the opposite sign
        # relative to the pyuvdata/MS UVW convention for ANTENNA1=ant1,
        # ANTENNA2=ant2. Flip the sign here so /uvw/uvw_m is MS-ready.
        uvw_m[rows, 0] = -np.asarray(u, dtype=np.float64)
        uvw_m[rows, 1] = -np.asarray(v, dtype=np.float64)
        uvw_m[rows, 2] = -np.asarray(w, dtype=np.float64)

    return uvw_m


def write_uvw_group(h5, infos, signal_map, ms_row_map):
    group = h5.create_group("uvw")
    uvw_m = compute_uvw_for_ms_rows_with_katpoint(
        infos,
        signal_map,
        ms_row_map
    )

    group.create_dataset("uvw_m", data=uvw_m)
    group.create_dataset("is_placeholder", data=np.int8(0))
    group.attrs["unit"] = "m"
    group.attrs["method"] = "katpoint.Target.uvw with sign flip for MS convention"
    group.attrs["target"] = f"radec {FIELD_RA_HMS} {FIELD_DEC_DMS}"
    group.attrs["time_input"] = "center_mjd converted to unix seconds"
    group.attrs["antenna_input"] = (
        "katpoint.Antenna from antenna txt lat lon alt diam"
    )
    group.attrs["katpoint_call"] = "target.uvw(antenna1, time, antenna2)"
    group.attrs["sign_flip_applied"] = np.int8(1)
    group.attrs["baseline_order"] = (
        "ANTENNA1=ms_rows/antenna1, ANTENNA2=ms_rows/antenna2; "
        "uvw_m = -target.uvw(antenna1, time, antenna2), "
        "chosen to match pyuvdata/MS ANTENNA1-ANTENNA2 UVW convention"
    )
    group.attrs["uvw_convention"] = "pyuvdata/MS ANTENNA1-ANTENNA2"


def write_ms_defaults_group(h5):
    group = h5.create_group("ms_defaults")

    group.create_dataset("flag_default", data=np.bool_(False))
    group.create_dataset("weight_default", data=np.float32(1.0))
    group.create_dataset("sigma_default", data=np.float32(1.0))
    group.create_dataset("missing_signal_should_flag", data=np.bool_(True))


def validate_uvw_result(h5):
    n_ms_rows = int(h5["ms_rows"].attrs["n_ms_rows"])

    if "uvw/uvw_m" not in h5:
        raise ValueError("missing /uvw/uvw_m")

    if h5["uvw/uvw_m"].shape != (n_ms_rows, 3):
        raise ValueError(
            f"/uvw/uvw_m shape mismatch: "
            f"{h5['uvw/uvw_m'].shape} != {(n_ms_rows, 3)}"
        )

    if int(h5["uvw/is_placeholder"][()]) != 0:
        raise ValueError("UVW is still placeholder")

    uvw = h5["uvw/uvw_m"][()]
    ant1 = h5["ms_rows/antenna1"][()]
    ant2 = h5["ms_rows/antenna2"][()]

    same_ant = ant1 == ant2
    if np.any(same_ant):
        max_same_ant_uvw = np.max(np.abs(uvw[same_ant]))
        if max_same_ant_uvw > 1e-6:
            raise ValueError(
                f"same-antenna UVW should be zero, max={max_same_ant_uvw}"
            )

    cross_ant = ant1 != ant2
    if np.any(cross_ant):
        max_cross_ant_uvw = np.max(np.abs(uvw[cross_ant]))
        if max_cross_ant_uvw <= 0.0:
            raise ValueError("cross-antenna UVW is all zero")


def validate_ms_ready_output(h5):
    required_groups = [
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
    required_paths = [
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
    forbidden_root_datasets = [
        "ANTENNA1",
        "ANTENNA2",
        "CHAN_FREQ",
        "CHAN_WIDTH",
        "TIME",
    ]

    for group_name in required_groups:
        if group_name not in h5:
            raise ValueError(f"missing required group: /{group_name}")

    for path in required_paths:
        if path not in h5:
            raise ValueError(f"missing required dataset: /{path}")

    for dataset_name in forbidden_root_datasets:
        if dataset_name in h5:
            raise ValueError(f"unexpected legacy root dataset: /{dataset_name}")

    n_corr_time = get_n_corr_time()
    n_baseline = len(h5["baseline_pairs"])
    expected_vis_shape = (n_corr_time, n_baseline, NCHAN)

    if h5["vis"].shape != expected_vis_shape:
        raise ValueError(
            f"/vis shape mismatch: {h5['vis'].shape} != {expected_vis_shape}"
        )

    if h5["time/center_mjd"].shape != (n_corr_time,):
        raise ValueError("time/center_mjd length mismatch")

    if h5["baseline/signal_pairs"].shape[0] != n_baseline:
        raise ValueError("baseline/signal_pairs first dimension mismatch")

    if h5["frequency/chan_freq_hz"].shape != (NCHAN,):
        raise ValueError("frequency/chan_freq_hz length mismatch")

    n_present_signal = int(np.sum(h5["signal/present"][()].astype(bool)))
    expected_selected_baseline_count = (
        n_present_signal * (n_present_signal + 1) // 2
    )
    selected_baseline_count = int(h5["ms_rows"].attrs["selected_baseline_count"])
    n_ms_rows = int(h5["ms_rows"].attrs["n_ms_rows"])
    expected_n_ms_rows = n_corr_time * selected_baseline_count

    if selected_baseline_count != expected_selected_baseline_count:
        raise ValueError(
            "selected_baseline_count mismatch: "
            f"{selected_baseline_count} != {expected_selected_baseline_count}"
        )

    if n_ms_rows != expected_n_ms_rows:
        raise ValueError(
            f"n_ms_rows mismatch: {n_ms_rows} != {expected_n_ms_rows}"
        )

    if h5["ms_rows/time_index"].shape != (n_ms_rows,):
        raise ValueError("ms_rows/time_index length mismatch")

    row_level_paths = [
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

    for path in row_level_paths:
        if h5[path].shape != (n_ms_rows,):
            raise ValueError(f"{path} length mismatch")

    if int(h5["field/is_placeholder"][()]) != 0:
        raise ValueError("field is still placeholder")

    if "ms_rows/corr_name" not in h5:
        raise ValueError("missing ms_rows/corr_name")

    if np.any(h5["ms_rows/row_has_missing_signal"][()]):
        raise ValueError("row_has_missing_signal should be all False")

    validate_uvw_result(h5)

    print("MS-ready HDF5 groups written:", ", ".join(required_groups))
    print(
        "selected_baseline_count:",
        selected_baseline_count
    )
    print("n_ms_rows:", n_ms_rows)
    print(
        "antenna position placeholder:",
        int(h5["antenna/position_is_placeholder"][()])
    )
    print("field placeholder:", int(h5["field/is_placeholder"][()]))
    print("uvw placeholder:", int(h5["uvw/is_placeholder"][()]))
    print("UVW method:", h5["uvw"].attrs["method"])
    print("UVW placeholder:", int(h5["uvw/is_placeholder"][()]))
    print("UVW shape:", h5["uvw/uvw_m"].shape)
    if "sign_flip_applied" in h5["uvw"].attrs:
        print("UVW sign flip applied:", int(h5["uvw"].attrs["sign_flip_applied"]))
    if "uvw_convention" in h5["uvw"].attrs:
        print("UVW convention:", h5["uvw"].attrs["uvw_convention"])

    uvw = h5["uvw/uvw_m"][()]
    ant1 = h5["ms_rows/antenna1"][()]
    ant2 = h5["ms_rows/antenna2"][()]
    cross = ant1 != ant2

    if np.any(cross):
        print(
            "max |uvw| for cross antennas:",
            float(np.max(np.abs(uvw[cross])))
        )


def write_ms_ready_metadata(h5, infos, signal_map, baseline_pairs):
    validate_ms_ready_config(signal_map)
    signal_meta = build_signal_axis_metadata(signal_map)
    baseline_meta = build_baseline_metadata(baseline_pairs)
    ms_row_map = build_ms_row_mapping(signal_map, baseline_pairs)
    row_has_missing_signal = build_row_missing_signal_flags(
        signal_meta,
        baseline_meta,
        ms_row_map
    )

    if np.any(row_has_missing_signal):
        raise ValueError("internal error: selected rows contain missing signals")

    write_baseline_group(h5, baseline_pairs)
    write_signal_group(h5, signal_map)
    write_time_group(h5, infos)
    write_frequency_group(h5, infos)
    write_antenna_group(h5, signal_map)
    write_field_group(h5, infos)
    write_polarization_group(h5)
    write_ms_rows_group(h5, ms_row_map, row_has_missing_signal)
    write_uvw_group(h5, infos, signal_map, ms_row_map)
    write_ms_defaults_group(h5)


def create_hdf5_file(output_file, infos, signal_map, baseline_pairs):
    if h5py is None:
        raise RuntimeError(
            f"h5py is unavailable, cannot save HDF5 output: {H5PY_IMPORT_ERROR}"
        )

    check_output_file(output_file)

    n_corr_time = get_n_corr_time()
    n_baseline = len(baseline_pairs)

    fft_per_corr = get_fft_per_corr()
    chunk_time = READ_FFT_PER_BLOCK // fft_per_corr

    if chunk_time < 1:
        chunk_time = 1

    if chunk_time > n_corr_time:
        chunk_time = n_corr_time

    create_kwargs = {
        "shape": (n_corr_time, n_baseline, NCHAN),
        "dtype": OUTPUT_DTYPE,
        "chunks": (chunk_time, 1, NCHAN),
        "fillvalue": np.complex64(0.0 + 0.0j),
    }

    if HDF5_COMPRESSION is not None:
        create_kwargs["compression"] = HDF5_COMPRESSION
        create_kwargs["shuffle"] = True

    h5 = h5py.File(output_file, "w")

    h5.create_dataset("vis", **create_kwargs)
    h5.create_dataset(
        "baseline_pairs",
        data=np.array(baseline_pairs, dtype=np.int16)
    )

    write_signal_metadata(h5, signal_map)
    write_global_metadata(h5, infos, n_corr_time, n_baseline)
    write_ms_ready_metadata(h5, infos, signal_map, baseline_pairs)
    validate_ms_ready_output(h5)

    return h5


# =========================
# 说明：无需修改其它部分
# =========================
def run_correlation_and_save(infos, signal_map, baseline_pairs):
    """
    相关计算 + 可选HDF5保存的总调度函数。

    ENABLE_HDF5_OUTPUT = True:
        计算相关并写入HDF5

    ENABLE_HDF5_OUTPUT = False:
        只读取数据、计算相关、不写入HDF5
        这种模式适合存储空间不足时测试流程
    """
    output_file = get_output_hdf5_file(infos)

    print("\n========== START CORRELATION ==========")
    print("HDF5 output enabled:", ENABLE_HDF5_OUTPUT)
    print("corr output mode   :", CORR_OUTPUT_MODE)

    if ENABLE_HDF5_OUTPUT:
        print("output file:", output_file)
        h5_context = create_hdf5_file(
            output_file,
            infos,
            signal_map,
            baseline_pairs
        )
    else:
        print("output file: disabled")
        print("generated output name:", output_file)
        h5_context = nullcontext(None)

    n_corr_done = 0
    fft_per_corr = get_fft_per_corr()

    with h5_context as h5:

        for block_start_fft in range(0, FFT_PER_FILE, READ_FFT_PER_BLOCK):
            block_fft_count = min(
                READ_FFT_PER_BLOCK,
                FFT_PER_FILE - block_start_fft
            )

            if block_fft_count % fft_per_corr != 0:
                raise ValueError(
                    f"block_fft_count must be divisible by fft_per_corr: "
                    f"{block_fft_count} / {fft_per_corr}"
                )

            n_corr_in_block = block_fft_count // fft_per_corr
            corr_start = n_corr_done
            corr_end = n_corr_done + n_corr_in_block

            print(
                "\n[BLOCK]",
                "fft:",
                block_start_fft,
                "->",
                block_start_fft + block_fft_count - 1,
                " corr_index:",
                corr_start,
                "->",
                corr_end - 1
            )

            # 第三部分：读取实际存在信号的数据
            data_cache = load_existing_signal_blocks(
                signal_map,
                block_start_fft,
                block_fft_count
            )

            print("loaded signals:", sorted([k + 1 for k in data_cache.keys()]))

            # 第三部分：计算当前block的相关结果
            vis_dict = compute_correlation_block(
                data_cache,
                baseline_pairs,
                block_fft_count
            )

            print("computed baselines:", len(vis_dict))

            # 第四部分：可选写入HDF5
            if ENABLE_HDF5_OUTPUT:
                write_correlation_block(
                    h5,
                    corr_start,
                    corr_end,
                    vis_dict
                )
                print("[OK] HDF5 block written")
            else:
                print("[SKIP] HDF5 output disabled, block result not saved")

            n_corr_done = corr_end

            del data_cache
            del vis_dict

            print("[OK] block finished")

    print("\n========== CORRELATION FINISHED ==========")
    print("HDF5 output enabled:", ENABLE_HDF5_OUTPUT)
    print("corr output mode   :", CORR_OUTPUT_MODE)

    if ENABLE_HDF5_OUTPUT:
        print("output file:", output_file)
    else:
        print("output file: disabled")
        print("generated output name:", output_file)

    print("total corr time:", n_corr_done)
    print("==========================================")

    if ENABLE_HDF5_OUTPUT:
        return output_file

    return None


# =========================
# 主流程
# =========================

def parse_command_line(argv):
    parser = argparse.ArgumentParser(
        description="Read filterbank files, correlate signals, and save HDF5."
    )
    parser.add_argument(
        "-ant",
        default=ANTENNA_INFO_TXT,
        help=(
            "antenna position txt file. Format: "
            "name lat lon [alt_m] [diam_m], for example: "
            "ant0 29.784402 109.779625 1581 7.5"
        ),
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="input filterbank .fil files"
    )

    return parser.parse_args(argv)


def main():

    global ANTENNA_INFO_TXT
    args = parse_command_line(sys.argv[1:])
    files = args.files
    ANTENNA_INFO_TXT = args.antenna_txt

    try:
        check_file_count(files)

        infos = []

        # 1. 先解析文件名（不读header）
        for f in files:
            infos.append({
                "file": f,
                "fname": parse_filename(f),
                "header": None,
                "data_offset": None,
            })

        # 2. 文件名时间一致性检查（最先做）
        check_time_consistency(infos)
        print("[OK] filename time consistent")

        # 3. 检查是否重复输入同一路信号
        check_duplicate_input_signal(infos)
        print("[OK] no duplicate input signal")

        # 4. 输出当前输入了哪些天线和极化
        print_input_signal_summary(infos)

        # 5. 再读header
        for i in infos:
            h, data_offset = parse_header(i["file"])
            i["header"] = h
            i["data_offset"] = data_offset

            print_file(i["file"], i["fname"], h)
            print_header(h)

        # 6. header一致性检查
        check_header_consistency(infos)

        print("\n[OK] ALL FILES VALID FOR CORRELATION")

        # 7. 第二部分准备层
        signal_map, baseline_pairs = run_prepare_layer(infos)

        print("\n[OK] PREPARE LAYER FINISHED")

        # 8. 第三部分相关计算 + 第四部分HDF5保存
        run_correlation_and_save(infos, signal_map, baseline_pairs)

    except Exception as e:
        print("\n[ERROR]")
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
