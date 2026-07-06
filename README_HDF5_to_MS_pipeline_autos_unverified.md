# HDF5 相关结果转 CASA MeasurementSet 项目 README

本文档对应当前这一组脚本：

```text
test_with_antenna_uvw.py        # 从多路 filterbank .fil 做相关计算，输出 MS-ready HDF5
inspect_ms_ready_hdf5.py        # 检查 HDF5 是否满足后续转 MS 的结构和几何要求
hdf5_to_ms.py                   # 把 MS-ready HDF5 转成 CASA MeasurementSet
plot_ms_phase_waterfall.py      # 从 MS 中提取指定天线/极化对，画相位或幅度瀑布图
```

本 README 的目标是让你可以完整理解：

```text
原始 .fil 文件
    -> 相关计算
    -> MS-ready HDF5
    -> CASA MeasurementSet
    -> CASA/listobs/plotms/tclean/自定义瀑布图
```

整条流程里每一步做了什么、输入输出是什么、关键参数怎么影响结果、如何检查是否正确、常见错误如何排查。

---

## 1. 项目目标

这个项目的核心目标是：

```text
把多路天线/极化的 filterbank 频域复数数据做相关计算，
保存为结构清晰的 HDF5，
再转换成 CASA 能读的 MeasurementSet。
```

最终希望生成的 MS 可以被下面这些工具正常读取：

```text
CASA listobs
CASA plotms
CASA tclean
python-casacore
pyuvdata
自定义 Python 脚本
```

尤其要保证：

```text
1. MS 文件结构不是“看起来像 MS”，而是真的能被 CASA 读。
2. DATA、FLAG、UVW、TIME、ANTENNA、FIELD、SPECTRAL_WINDOW、POLARIZATION 等表一致。
3. 频率轴顺序正确。
4. 极化产品正确。
5. UVW 符号和 MS 约定一致。
6. 自相关和互相关的写入逻辑清楚。
```

---

## 2. 四个脚本的关系

### 2.1 `test_with_antenna_uvw.py`

功能：

```text
输入：多个 .fil 文件
输出：一个 MS-ready HDF5 文件
```

它负责：

```text
1. 解析 .fil 文件名，得到天线编号和极化编号。
2. 解析 filterbank header。
3. 读取频域复数数据。
4. 对实际存在的信号两两做相关计算。
5. 保存相关结果到 HDF5 的 /vis。
6. 写入时间、频率、天线、相位中心、极化、MS 行映射等元数据。
7. 使用 katpoint 根据真实天线位置和相位中心计算 UVW。
```

它是整个流程的第一步，也是最重要的数据生成步骤。

---

### 2.2 `inspect_ms_ready_hdf5.py`

功能：

```text
输入：MS-ready HDF5
输出：终端检查报告
```

它不会修改 HDF5，只负责检查：

```text
1. 必需 group 是否存在。
2. 必需 dataset 是否存在。
3. /vis shape 是否合理。
4. /time、/frequency、/baseline、/ms_rows 的长度是否匹配。
5. 天线位置是否是真实坐标，实际使用天线是否还在用 placeholder。
6. FIELD 相位中心是否合理。
7. UVW 是否存在、是否非 placeholder。
8. 同天线 UVW 是否为 0。
9. 跨天线 UVW 是否不是全 0。
10. 同一物理 baseline 的不同极化产品 UVW 是否一致。
```

它是 HDF5 转 MS 前的强烈推荐检查步骤。

---

### 2.3 `hdf5_to_ms.py`

功能：

```text
输入：MS-ready HDF5
输出：CASA MeasurementSet
```

它负责：

```text
1. 读取 HDF5 schema。
2. 检查 HDF5 是否完整。
3. 将 signal-level baseline 重新打包成 physical baseline + polarization products。
4. 生成 pyuvdata.UVData 对象。
5. 调用 pyuvdata 写出 MS。
6. 用 python-casacore 验证 MS MAIN 表和子表。
```

它还会处理：

```text
1. 频率轴升序化。
2. UVW 读取和可选符号翻转。
3. CARRY_1 telescope name。
4. 自相关 autos 是否写入。
5. partial polarization 是否允许。
6. 内存估算。
```

---

### 2.4 `plot_ms_phase_waterfall.py`

功能：

```text
输入：MeasurementSet + 两个信号名，例如 0X 1Y
输出：相位或幅度瀑布图 PNG
```

它负责：

```text
1. 用 python-casacore 或 casatools 读取 MS。
2. 根据 0X、1Y 这种格式解析天线编号和极化。
3. 在 MAIN 表中找到对应物理 baseline。
4. 从 DATA 中取出对应极化相关产品。
5. 如果 MS 中是反向 baseline，则自动取共轭，统一方向。
6. 画 phase 或 amplitude waterfall。
```

当前上传版本的绘图方向是：

```text
横轴：频率 MHz
纵轴：时间 s
颜色：相位 rad 或幅度
```

如果你希望横轴为时间、纵轴为频率，需要修改 `save_outputs()` 中 `imshow()` 前的矩阵方向和 `extent` 设置。

---

## 3. 推荐目录结构

建议把代码、输入数据、输出数据分开：

```text
project/
├── code/
│   ├── test_with_antenna_uvw.py
│   ├── inspect_ms_ready_hdf5.py
│   ├── hdf5_to_ms.py
│   └── plot_ms_phase_waterfall.py
│
├── antenna/
│   └── antenna_positions.txt
│
├── fil/
│   ├── 20000101_030853_00_0.fil
│   ├── 20000101_030853_00_1.fil
│   ├── 20000101_030853_01_0.fil
│   └── 20000101_030853_01_1.fil
│
├── h5/
│   └── 20000101030853153_20000101030853653.h5
│
├── ms/
│   ├── 0705test.ms
│   └── 0705test_with_autos.ms
│
└── plots/
    ├── phase_waterfall_0X_1Y.png
    └── amp_waterfall_0X_1Y.png
```

---

## 4. Python 环境

### 4.1 推荐环境

建议使用 conda 环境，例如：

```bash
conda create -n hdf2ms2 \
  python=3.8 \
  numpy=1.24.4 \
  h5py=3.11.0 \
  astropy=5.2.2 \
  pyuvdata=2.4.2 \
  python-casacore \
  matplotlib \
  -y

conda activate hdf2ms2
```

如果要运行 `test_with_antenna_uvw.py` 计算 UVW，还需要：

```bash
pip install katpoint
```

检查环境：

```bash
python3 - <<'PY'
import sys
import numpy
import h5py
import astropy
import pyuvdata
import casacore.tables as ct
import matplotlib

print("python:", sys.executable)
print("numpy:", numpy.__version__)
print("h5py:", h5py.__version__)
print("astropy:", astropy.__version__)
print("pyuvdata:", pyuvdata.__version__)
print("casacore:", ct.__file__)
print("matplotlib:", matplotlib.__version__)
PY
```

### 4.2 不要用 Python 2

如果出现：

```text
SyntaxError: future feature annotations is not defined
```

或者：

```text
SyntaxError: invalid syntax
```

要先检查是不是错误地用了 Python 2：

```bash
which python
python --version

which python3
python3 --version
```

推荐始终使用：

```bash
python3 xxx.py
```

---

## 5. 输入 `.fil` 文件要求

### 5.1 文件名格式

`test_with_antenna_uvw.py` 要求输入文件名格式为：

```text
YYYYMMDD_HHMMSS_xx_P.fil
```

例如：

```text
20000101_030853_00_0.fil
20000101_030853_00_1.fil
20000101_030853_01_0.fil
20000101_030853_01_1.fil
```

含义：

```text
YYYYMMDD    日期
HHMMSS      时间
xx          天线编号，00 到 09
P           极化编号，0 或 1
```

当前代码中：

```text
P = 0 -> X 极化
P = 1 -> Y 极化
```

---

### 5.2 输入信号编号规则

代码中共有：

```text
N_PHYSICAL_ANTENNAS = 10
VALID_POLARIZATIONS = [0, 1]
N_INPUT_SIGNALS = 20
```

也就是：

```text
10 根天线 × 2 个极化 = 20 路输入信号
```

输入信号编号规则：

```text
ant0 pol0 -> signal 1 -> index 0
ant0 pol1 -> signal 2 -> index 1
ant1 pol0 -> signal 3 -> index 2
ant1 pol1 -> signal 4 -> index 3
...
ant9 pol1 -> signal 20 -> index 19
```

公式：

```text
如果 polarization = 0:
    input_signal_no = (antenna_id + 1) * 2 - 1

如果 polarization = 1:
    input_signal_no = (antenna_id + 1) * 2
```

---

### 5.3 filterbank header 要求

脚本会检查：

```text
source_name 一致
tstart 一致
nbits 等于 REQUIRED_NBITS
nchans 等于 NCHAN
```

当前关键配置：

```text
REQUIRED_NBITS = 8
NCHAN = 2048
```

注意：

```text
header 中的 nbits 是 filterbank header 记录的采样位数。
实际每个频点复数占用由 BITS_PER_FREQ_POINT 控制。
```

当前代码假设每个频点是：

```text
real int8 + imag int8 = 2 bytes = 16 bits
```

也就是说数据区每个频点布局为：

```text
byte 0: real int8
byte 1: imag int8
```

如果你的真实数据不是这个布局，必须先修改 `read_signal_block()`。

---

## 6. 天线位置文件 `antenna_positions.txt`

### 6.1 为什么必须提供

要生成真实 UVW，必须知道每根参与观测天线的位置。

当前代码支持天线位置 txt，格式：

```text
# name lat lon [alt_m] [diam_m]
ant0 29.784402 109.779625 1581 7.5
ant1 29.784410 109.779640 1581 7.5
```

每列含义：

```text
name      天线名，当前默认必须类似 ant0、ant1
lat       纬度，单位 degree
lon       经度，单位 degree
alt_m     海拔，单位 m，可选，默认 0
diam_m    口径，单位 m，可选，默认 ANTENNA_DISH_DIAMETER_M
```

### 6.2 天线名规则

当前 `test_with_antenna_uvw.py` 默认：

```text
ANTENNA_TXT_NAME_PREFIX = "ant"
```

因此允许：

```text
ant0
ant1
ANT03
```

不建议直接写：

```text
CARRY_1_ANT00
```

因为当前解析函数是从 `ant` 后面的数字解析天线编号。

如果你想让 MS 里的阵列名是 `CARRY_1`，应该在 `hdf5_to_ms.py` 中设置：

```python
CASA_TELESCOPE_NAME = "CARRY_1"
ARRAY_NAME = "CARRY_1"
INSTRUMENT_NAME = "CARRY_1"
```

而不是把 antenna txt 第一列改成复杂名字。

---

## 7. 第一步：生成 MS-ready HDF5

### 7.1 命令示例

假设你有 4 个输入文件：

```text
20000101_030853_00_0.fil
20000101_030853_00_1.fil
20000101_030853_01_0.fil
20000101_030853_01_1.fil
```

建议运行：

```bash
python3 test_with_antenna_uvw.py \
  -ant antenna_positions.txt \
  20000101_030853_00_0.fil \
  20000101_030853_00_1.fil \
  20000101_030853_01_0.fil \
  20000101_030853_01_1.fil
```

### 7.2 当前上传版本需要注意的命令行小问题

当前上传的 `test_with_antenna_uvw.py` 中，参数定义使用了：

```python
parser.add_argument("-ant", default=ANTENNA_INFO_TXT, ...)
```

但是 `main()` 中访问的是：

```python
args.antenna_txt
```

严格来说，这里应该统一。建议把 parser 改成：

```python
parser.add_argument(
    "-ant",
    "--antenna-txt",
    dest="antenna_txt",
    default=ANTENNA_INFO_TXT,
    help=(
        "antenna position txt file. Format: "
        "name lat lon [alt_m] [diam_m], for example: "
        "ant0 29.784402 109.779625 1581 7.5"
    ),
)
```

改完后命令就是：

```bash
python3 test_with_antenna_uvw.py \
  --antenna-txt antenna_positions.txt \
  *.fil
```

或者：

```bash
python3 test_with_antenna_uvw.py \
  -ant antenna_positions.txt \
  *.fil
```

### 7.3 输出 HDF5 文件名

如果 `OUTPUT_HDF5_FILE = None`，代码会根据第一个输入文件 header 中的 `tstart` 自动生成：

```text
YYYYMMDDHHMMSSmmm_YYYYMMDDHHMMSSmmm.h5
```

例如：

```text
20000101030853153_20000101030853653.h5
```

前半段是数据开始时间，后半段是数据结束时间。

结束时间是 exclusive end，也就是：

```text
end = start + n_corr_time × CORR_TIME_US
```

---

## 8. 相关计算参数

当前主要参数：

```text
N_INPUT_SIGNALS       = 20
NCHAN                 = 2048
FFT_PER_INTEGRATION   = 10
INTEGRATION_PER_FILE  = 12500
FFT_PER_FILE          = 125000
FFT_TIME_US           = 4
CORR_TIME_US          = 1000
READ_FFT_PER_BLOCK    = 36000
CORR_OUTPUT_MODE      = "sum"
OUTPUT_DTYPE          = complex64
```

### 8.1 每个相关积分包含多少 FFT

公式：

```text
FFT_PER_CORR = CORR_TIME_US / FFT_TIME_US
```

当前：

```text
FFT_PER_CORR = 1000 / 4 = 250
```

所以每 250 个 FFT 做一次相关积分。

### 8.2 HDF5 时间点数量

公式：

```text
n_corr_time = FFT_PER_FILE / FFT_PER_CORR
```

当前：

```text
n_corr_time = 125000 / 250 = 500
```

所以一个 HDF5 文件有 500 个时间积分点。

总时长：

```text
500 × 1 ms = 0.5 s
```

### 8.3 相关公式

对任意两路信号：

```text
Xi(t, f)
Xj(t, f)
```

相关计算为：

```text
Vij(f, T) = sum over k in integration [ Xi(k, f) × conj(Xj(k, f)) ]
```

代码中对应：

```python
prod = x_i * np.conj(x_j)
v = prod.sum(axis=1)
```

如果 `CORR_OUTPUT_MODE = "mean"`，则改为平均：

```python
v = prod.mean(axis=1)
```

注意：

```text
当前默认是 sum，不是 mean。
```

所以幅度大小和积分内 FFT 数有关。

---

## 9. HDF5 输出结构

生成的 HDF5 是 MS-ready schema，核心结构如下：

```text
/
├── vis
├── baseline_pairs
├── baseline/
│   ├── signal_pairs
│   ├── antenna_pairs
│   └── polarization_pairs
├── signal/
│   ├── present
│   ├── input_signal_no
│   ├── antenna_id
│   ├── polarization_id
│   └── file
├── time/
│   ├── start_mjd
│   ├── center_mjd
│   ├── end_mjd
│   ├── interval_sec
│   └── exposure_sec
├── frequency/
│   ├── chan_freq_hz
│   ├── chan_width_hz
│   ├── ref_frequency_hz
│   └── nchan
├── antenna/
│   ├── id
│   ├── name
│   ├── station
│   ├── position_itrf_m
│   ├── dish_diameter_m
│   ├── used_in_input
│   └── position_is_placeholder_by_antenna
├── field/
│   ├── source_name
│   ├── phase_center_ra_rad
│   ├── phase_center_dec_rad
│   ├── phase_center_ra_hms
│   ├── phase_center_dec_dms
│   └── frame
├── polarization/
│   ├── input_pol_id
│   ├── input_pol_name
│   ├── all_corr_names
│   ├── all_corr_pol_i
│   └── all_corr_pol_j
├── ms_rows/
│   ├── time_index
│   ├── signal_baseline_index
│   ├── antenna1
│   ├── antenna2
│   ├── pol_i
│   ├── pol_j
│   ├── corr_name
│   ├── row_is_same_antenna
│   └── row_is_cross_antenna
├── uvw/
│   ├── uvw_m
│   └── is_placeholder
└── ms_defaults/
    ├── flag_default
    ├── weight_default
    └── sigma_default
```

### 9.1 `/vis`

形状：

```text
(n_corr_time, n_baseline, nchan)
```

当前典型为：

```text
(500, 210, 2048)
```

含义：

```text
axis 0: 时间积分点
axis 1: signal-level baseline index
axis 2: 频率通道
```

注意：

```text
这里的 baseline 是 signal-level baseline，不是最终 MS 中的物理 baseline。
```

### 9.2 signal-level baseline 数量

20 路信号的唯一相关对数量：

```text
20 × 21 / 2 = 210
```

其中：

```text
20 个自相关
190 个互相关
```

---

## 10. UVW 计算逻辑

UVW 由 `test_with_antenna_uvw.py` 使用 `katpoint` 计算。

核心逻辑：

```text
1. 根据天线 txt 构造 katpoint.Antenna。
2. 根据 FIELD_RA_HMS / FIELD_DEC_DMS 构造 katpoint.Target。
3. 对每个积分中心时刻计算 UVW。
4. 对同一天线行，UVW = [0, 0, 0]。
5. 对跨天线行，调用 target.uvw()。
6. 代码中已经做了符号翻转，用于匹配 pyuvdata/MS 的 ANTENNA1-ANTENNA2 约定。
```

重要提醒：

```text
新生成的 HDF5 已经修正了 UVW 符号。
转换 MS 时不要再加 --flip-uvw-sign。
```

只有旧 HDF5 才考虑：

```bash
--flip-uvw-sign
```

---

## 11. 第二步：检查 HDF5

生成 HDF5 后，建议立刻运行：

```bash
python3 inspect_ms_ready_hdf5.py 20000101030853153_20000101030853653.h5
```

严格模式：

```bash
python3 inspect_ms_ready_hdf5.py \
  20000101030853153_20000101030853653.h5 \
  --strict
```

显示更多行例子：

```bash
python3 inspect_ms_ready_hdf5.py \
  20000101030853153_20000101030853653.h5 \
  --show-examples 10
```

如果检查通过，会看到类似：

```text
RESULT: PASS
Next step: this HDF5 is structurally ready for an HDF5 -> MS converter.
```

如果检查失败，先不要转 MS。应先修复 HDF5 问题。

---

## 12. 第三步：HDF5 转 MS

### 12.1 不包含自相关的 MS

默认只写跨天线互相关：

```bash
python3 hdf5_to_ms.py \
  20000101030853153_20000101030853653.h5 \
  /home/carrylab/Downloads/conda/0705test.ms \
  --overwrite \
  --uvdata-constructor new
```

如果有 2 根天线，默认物理 baseline 只有：

```text
0&1
```

如果有 10 根天线，默认物理 baseline 数量：

```text
10 × 9 / 2 = 45
```

### 12.2 包含自相关的 MS

如果需要保留自相关，比如要画：

```text
0X 0X
1X 1X
0X 0Y
```

必须加：

```bash
--include-autos
```

示例：

```bash
python3 hdf5_to_ms.py \
  20000101030853153_20000101030853653.h5 \
  /home/carrylab/Downloads/conda/0705test_with_autos.ms \
  --overwrite \
  --include-autos \
  --uvdata-constructor new
```

当前版本的 `hdf5_to_ms.py` 已经把 autos 作为真实数据写入，而不是简单写成全 flag。也就是说：

```text
auto DATA 来自 HDF5 /vis
auto UVW = [0, 0, 0]
auto FLAG 不应该全 True
auto NSAMPLE 应为 1
```

### 12.3 2 根天线时的行数

如果有 2 根天线、500 个时间点：

不加 autos：

```text
physical baseline = 1
Nblts = 500 × 1 = 500
```

加 autos：

```text
physical baselines = 0&0, 0&1, 1&1
Nblts = 500 × 3 = 1500
```

所以如果你想画 `0X 0X`，但 MS 只有 500 行，通常说明你没有包含 autos。

---

## 13. `hdf5_to_ms.py` 常用参数

### 13.1 `--overwrite`

如果输出 MS 已经存在，覆盖它：

```bash
--overwrite
```

不加时，如果目标目录已经存在，会报错。

### 13.2 `--include-autos`

写入自相关行：

```bash
--include-autos
```

不加时只写互相关。

### 13.3 `--allow-partial-pols`

默认要求每个物理 baseline 都有完整 4 个极化产品：

```text
XX
XY
YX
YY
```

如果缺少某些产品，默认报错。

如果你希望缺失产品写成 0 并 flag，可以加：

```bash
--allow-partial-pols
```

### 13.4 `--allow-uvw-warnings`

如果同一物理 baseline 的不同极化行 UVW 有微小差异，默认报错。  
如果只是希望继续写出，可以降级为 warning：

```bash
--allow-uvw-warnings
```

注意：

```text
缺失 UVW 仍然是 fatal error。
```

### 13.5 `--flip-uvw-sign`

只用于旧 HDF5：

```bash
--flip-uvw-sign
```

新 HDF5 不要加。

### 13.6 `--x-orientation`

设置 X feed orientation：

```bash
--x-orientation east
--x-orientation north
--x-orientation none
```

默认：

```text
east
```

如果不确定，可以用：

```bash
--x-orientation none
```

### 13.7 `--max-memory-gb`

转换前估算内存，如果超过限制则停止：

```bash
--max-memory-gb 8
```

大阵列、长时间、全频率、全极化时内存会快速增加。

### 13.8 `--dry-run`

只检查和打包，不写 MS：

```bash
--dry-run
```

推荐正式写 MS 前先跑：

```bash
python3 hdf5_to_ms.py input.h5 output.ms \
  --overwrite \
  --include-autos \
  --uvdata-constructor new \
  --dry-run
```

### 13.9 `--validate-only`

只做 HDF5 schema 和 payload 检查，不构造 UVData，不写 MS：

```bash
--validate-only
```

比 `--dry-run` 更靠前退出。

### 13.10 `--uvdata-constructor`

可选：

```text
new
manual
```

推荐：

```bash
--uvdata-constructor new
```

`manual` 只建议调试用。

---

## 14. 频率轴处理

HDF5 中的频率轴可能是降序，例如：

```text
1500 MHz -> 1000 MHz
```

`hdf5_to_ms.py` 会把它变成升序：

```text
1000 MHz -> 1500 MHz
```

并同步重排：

```text
freq_array_hz
channel_width_hz
data_array
flag_array
nsample_array
```

因此 MS 中的 `SPECTRAL_WINDOW/CHAN_FREQ` 应该严格升序。

检查命令：

```bash
python3 - <<'PY'
import casacore.tables as ct
import numpy as np

ms = "/home/carrylab/Downloads/conda/0705test.ms"

tb = ct.table(ms + "/SPECTRAL_WINDOW", readonly=True, ack=False)
freq = np.asarray(tb.getcell("CHAN_FREQ", 0), dtype=float).reshape(-1)
tb.close()

print("first Hz:", freq[0])
print("last Hz :", freq[-1])
print("ascending:", bool(np.all(np.diff(freq) > 0)))
PY
```

---

## 15. MS 验证

### 15.1 检查 MS 目录结构

`hdf5_to_ms.py` 写完后会自动检查：

```text
MS 主目录
table.dat
ANTENNA
FIELD
SPECTRAL_WINDOW
POLARIZATION
DATA_DESCRIPTION
```

### 15.2 检查 MAIN 表

它会检查：

```text
DATA
FLAG
UVW
TIME
ANTENNA1
ANTENNA2
DATA_DESC_ID
FIELD_ID
INTERVAL
EXPOSURE
WEIGHT
SIGMA
FLAG_ROW
```

还会检查：

```text
MAIN 行数是否等于 payload Nblts
ANTENNA1/ANTENNA2 是否匹配 payload
UVW 是否匹配 payload
DATA cell shape 是否合理
```

### 15.3 检查子表

它会检查：

```text
SPECTRAL_WINDOW
POLARIZATION
DATA_DESCRIPTION
ANTENNA
FIELD
```

如果这些检查通过，说明这个 MS 至少在结构上比较可靠。

---

## 16. CASA 中检查 MS

### 16.1 listobs

进入 CASA：

```python
vis = "/home/carrylab/Downloads/conda/0705test.ms"
listobs(vis=vis, listfile="0705test.listobs.txt", verbose=True)
```

如果 MS 正常，`listobs` 应能输出：

```text
nfields
numrecords
scan
spw
field direction
```

### 16.2 读取 MAIN 表

```python
tb.open(vis)

print("nrows =", tb.nrows())
print("columns =", tb.colnames())

data0 = tb.getcell("DATA", 0)
flag0 = tb.getcell("FLAG", 0)
uvw0 = tb.getcell("UVW", 0)
time0 = tb.getcell("TIME", 0)
ant10 = tb.getcell("ANTENNA1", 0)
ant20 = tb.getcell("ANTENNA2", 0)

print("DATA shape =", data0.shape)
print("FLAG shape =", flag0.shape)
print("UVW =", uvw0)
print("TIME =", time0)
print("ANTENNA1 =", ant10)
print("ANTENNA2 =", ant20)

tb.close()
```

### 16.3 检查有哪些 baseline

```bash
python3 - <<'PY'
import casacore.tables as ct

ms = "/home/carrylab/Downloads/conda/0705test.ms"

tb = ct.table(ms, readonly=True, ack=False)
ant1 = tb.getcol("ANTENNA1")
ant2 = tb.getcol("ANTENNA2")
tb.close()

pairs = sorted(set(zip(ant1.tolist(), ant2.tolist())))

print("baselines in MAIN:")
for pair in pairs:
    print(pair)
PY
```

如果不含 autos，2 根天线应该看到：

```text
(0, 1)
```

如果含 autos，2 根天线应该看到：

```text
(0, 0)
(0, 1)
(1, 1)
```

---

## 17. 画瀑布图

### 17.1 基本用法

```bash
python3 plot_ms_phase_waterfall.py \
  0X 1Y \
  /home/carrylab/Downloads/conda/0705test.ms
```

含义：

```text
0X 1Y
= antenna 0 的 X 极化 × conj(antenna 1 的 Y 极化)
```

对应相关产品：

```text
XY
```

### 17.2 支持的信号格式

支持：

```text
0X
1Y
ant0X
ANT03Y
0:x
1-y
```

当前只支持线极化：

```text
X
Y
```

### 17.3 常见相关产品

```text
0X 1X -> XX
0X 1Y -> XY
0Y 1X -> YX
0Y 1Y -> YY
```

### 17.4 自相关

如果要画：

```bash
python3 plot_ms_phase_waterfall.py 0X 0X /path/to/test.ms
```

MS 中必须有：

```text
ANTENNA1 = 0
ANTENNA2 = 0
```

也就是转 MS 时必须加：

```bash
--include-autos
```

否则会报：

```text
No rows found for physical baseline 0&0
```

### 17.5 画幅度而不是相位

```bash
python3 plot_ms_phase_waterfall.py \
  0X 1Y \
  /home/carrylab/Downloads/conda/0705test.ms \
  --mode amp
```

### 17.6 保存 numpy 数据

```bash
python3 plot_ms_phase_waterfall.py \
  0X 1Y \
  /home/carrylab/Downloads/conda/0705test.ms \
  --save-npy \
  --save-txt
```

会额外保存：

```text
_complex_visibility.npy
_phase_rad.npy
_freq_hz.npy
_time_sec_from_start.npy
_summary.txt
```

---

## 18. 相位瀑布图如何理解

### 18.1 相位范围

脚本使用：

```python
np.angle(spec)
```

所以相位单位是 rad，范围：

```text
-π 到 +π
```

颜色条通常固定为：

```text
vmin = -np.pi
vmax = +np.pi
```

### 18.2 互相关相位

例如：

```text
0X 1Y
```

表示：

```text
ant0-X × conj(ant1-Y)
```

相位包含：

```text
几何延迟相位
仪器相位
线缆相位
本振/时钟相位
源结构相位
噪声
未校准项
```

如果没有校准，图上相位不一定平滑。

### 18.3 自相关相位

例如：

```text
0X 0X
```

表示：

```text
ant0-X × conj(ant0-X)
```

理论上是功率谱，应该接近实数非负，相位通常接近 0。

但由于数值误差、噪声、flag、数据格式问题，也可能有异常。

### 18.4 同天线交叉极化

例如：

```text
0X 0Y
```

表示：

```text
ant0-X × conj(ant0-Y)
```

这可以用于查看同一天线 X/Y 两路之间的相位关系。

---

## 19. UV 覆盖注意事项

如果只有 2 根天线、0.5 秒数据：

```text
physical baseline = 1
time = 0.5 s
```

那么 uv 覆盖几乎就是一个点，或者带共轭镜像的两个点。

这不是错误。

想要更丰富 uv 覆盖，需要：

```text
更多天线
更长观测时间
更多 baseline
```

例如：

```text
2 根天线 0.5 秒：几乎一个 uv 点
2 根天线数小时：一条 uv 轨迹
10 根天线 0.5 秒：45 条 baseline，但每条很短
10 根天线数小时：较丰富 uv 覆盖
```

---

## 20. 常见错误和排查

### 20.1 `No rows found for physical baseline 0&0`

原因：

```text
MS 里没有自相关行。
```

解决：

```bash
python3 hdf5_to_ms.py input.h5 output_with_autos.ms \
  --overwrite \
  --include-autos \
  --uvdata-constructor new
```

然后再画：

```bash
python3 plot_ms_phase_waterfall.py 0X 0X output_with_autos.ms
```

### 20.2 CASA `listobs` 崩溃或不认识 telescope

如果 CASA 报类似：

```text
Telescope HDF5_ARRAY is not recognized by CASA
```

要确认 `hdf5_to_ms.py` 中：

```python
CASA_TELESCOPE_NAME = "CARRY_1"
ARRAY_NAME = "CARRY_1"
INSTRUMENT_NAME = "CARRY_1"
```

并且 CASA 本地 observatory table 已注册 `CARRY_1`。

### 20.3 HDF5 检查提示 used antennas still have placeholder positions

原因：

```text
参与输入数据的天线没有真实天线坐标。
```

解决：

```text
检查 antenna_positions.txt 是否包含所有实际参与输入的 antX。
```

### 20.4 UVW 全零

如果跨天线 UVW 全零，说明 UVW 没有正确计算或天线位置不真实。

检查：

```bash
python3 inspect_ms_ready_hdf5.py input.h5 --strict
```

### 20.5 新 HDF5 不要加 `--flip-uvw-sign`

新版本 HDF5 的 UVW 已经在生成阶段做了符号修正。

错误使用：

```bash
python3 hdf5_to_ms.py input_new.h5 output.ms --flip-uvw-sign
```

可能导致 UVW 符号再次翻转。

### 20.6 DATA shape 是 `(4, 2048)` 还是 `(2048, 4)`

不同 writer 或读取方式可能显示不同：

```text
(Ncorr, Nchan)
```

或者：

```text
(Nchan, Ncorr)
```

脚本中已经兼容这两种情况。

---

## 21. 推荐完整流程

### 21.1 生成 HDF5

```bash
python3 test_with_antenna_uvw.py \
  --antenna-txt antenna_positions.txt \
  20000101_030853_00_0.fil \
  20000101_030853_00_1.fil \
  20000101_030853_01_0.fil \
  20000101_030853_01_1.fil
```

如果当前脚本还没修 `--antenna-txt` 参数名，请先按第 7.2 节修正。

### 21.2 检查 HDF5

```bash
python3 inspect_ms_ready_hdf5.py \
  20000101030853153_20000101030853653.h5 \
  --strict \
  --show-examples 10
```

### 21.3 dry-run 转 MS

```bash
python3 hdf5_to_ms.py \
  20000101030853153_20000101030853653.h5 \
  /home/carrylab/Downloads/conda/0705test_with_autos.ms \
  --overwrite \
  --include-autos \
  --uvdata-constructor new \
  --dry-run
```

### 21.4 正式转 MS

```bash
python3 hdf5_to_ms.py \
  20000101030853153_20000101030853653.h5 \
  /home/carrylab/Downloads/conda/0705test_with_autos.ms \
  --overwrite \
  --include-autos \
  --uvdata-constructor new
```

### 21.5 CASA 检查

```python
vis = "/home/carrylab/Downloads/conda/0705test_with_autos.ms"
listobs(vis=vis, listfile="0705test_with_autos.listobs.txt", verbose=True)
```

### 21.6 画互相关瀑布图

```bash
python3 plot_ms_phase_waterfall.py \
  0X 1Y \
  /home/carrylab/Downloads/conda/0705test_with_autos.ms \
  --save-npy \
  --save-txt
```

### 21.7 画自相关瀑布图

```bash
python3 plot_ms_phase_waterfall.py \
  0X 0X \
  /home/carrylab/Downloads/conda/0705test_with_autos.ms \
  --save-npy \
  --save-txt
```

---

## 22. 开发者注意事项

### 22.1 不要随意改 UVW 符号

UVW 是成像正确性的关键。当前流程中：

```text
test_with_antenna_uvw.py 生成 HDF5 时已经做了 MS 约定的符号修正。
hdf5_to_ms.py 默认直接使用 HDF5 中的 UVW。
```

只有旧 HDF5 才用：

```bash
--flip-uvw-sign
```

### 22.2 不要把 autos 强制 flag

如果用户加了：

```bash
--include-autos
```

应保留真实自相关 DATA。

正确逻辑：

```text
auto DATA 从 HDF5 /vis 读取
auto FLAG = False
auto NSAMPLE = 1
auto UVW = [0,0,0]
```

### 22.3 频率轴必须和 DATA 同步重排

如果频率轴从降序改成升序，必须同步重排：

```text
DATA
FLAG
NSAMPLE
CHAN_FREQ
CHAN_WIDTH
```

否则瀑布图和频率坐标会错位。

### 22.4 HDF5 中 signal-level baseline 和 MS 中 physical baseline 不是同一层概念

HDF5 `/vis`：

```text
time × signal_pair × freq
```

MS `DATA`：

```text
physical_baseline_time × freq × pol
```

转换时要把：

```text
signal pair
```

重新打包为：

```text
physical baseline + correlation product
```

例如：

```text
0X-1X -> physical baseline 0&1, corr XX
0X-1Y -> physical baseline 0&1, corr XY
0Y-1X -> physical baseline 0&1, corr YX
0Y-1Y -> physical baseline 0&1, corr YY
```

---

## 23. 最小验收标准

一个成功的 MS 至少应该满足：

```text
1. CASA listobs 可以正常运行。
2. MAIN 表行数符合预期。
3. SPECTRAL_WINDOW/CHAN_FREQ 严格升序。
4. POLARIZATION/CORR_TYPE 是 9,10,11,12。
5. FIELD/PHASE_DIR 和 HDF5 相位中心一致。
6. ANTENNA 表包含实际使用天线。
7. 跨天线 UVW 不是全 0。
8. 自相关 UVW 是 0。
9. DATA shape 是 4 × nchan 或 nchan × 4。
10. 如果使用 --include-autos，MAIN 中有 0&0、1&1 等自相关行。
```

---

## 24. 术语表

### signal-level baseline

指输入信号之间的相关对。例如：

```text
0X-0X
0X-0Y
0X-1X
0X-1Y
```

它存在于 HDF5 `/vis` 的 baseline 轴。

### physical baseline

指物理天线之间的 baseline。例如：

```text
0&1
1&2
```

它存在于 MS MAIN 表的 `ANTENNA1` 和 `ANTENNA2`。

### auto correlation

自相关，`ANTENNA1 == ANTENNA2`。

例如：

```text
0X 0X
0Y 0Y
0X 0Y
```

### cross correlation

互相关，`ANTENNA1 != ANTENNA2`。

例如：

```text
0X 1X
0X 1Y
```

### Nblts

pyuvdata 中常用的维度，表示：

```text
baseline-time 数量
```

也就是：

```text
Nblts = Ntimes × Nphysical_baselines
```

### Nfreqs

频率通道数，当前典型为：

```text
2048
```

### Npols

极化产品数量，当前为：

```text
4
```

即：

```text
XX, XY, YX, YY
```

---

## 25. 一句话总结

这个项目的核心思想是：

```text
test_with_antenna_uvw.py 负责把原始多路 .fil 数据相关成完整的 MS-ready HDF5；
inspect_ms_ready_hdf5.py 负责在转 MS 前发现 HDF5 结构和几何错误；
hdf5_to_ms.py 负责把 HDF5 中的 signal-level 相关结果重组为 CASA MeasurementSet；
plot_ms_phase_waterfall.py 负责从 MS 中按 0X 1Y 这种方式取出指定相关产品并画相位/幅度瀑布图。
```

最推荐的工作流是：

```text
生成 HDF5
    -> inspect 检查
    -> hdf5_to_ms dry-run
    -> hdf5_to_ms 正式写 MS
    -> CASA listobs
    -> plot_ms_phase_waterfall 画图
```
