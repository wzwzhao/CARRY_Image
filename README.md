# CARRY Visibility: 从 `.fil` 到 CASA MeasurementSet 的完整流程

本文档说明当前 `Visibility` 目录下这组脚本的用途、输入输出、推荐运行顺序，以及在原始数据全 0 时如何做人造测试。

当前主流程是：

```text
原始 .fil 文件
    -> test_with_antenna_uvw.py
    -> MS-ready HDF5
    -> hdf5_to_ms.py
    -> CASA MeasurementSet (.ms)
    -> plot_ms_phase_waterfall_timefreq_bjt.py
    -> 相位/幅度瀑布图 PNG
```

如果原始 `.fil` 数据全 0，可以先用：

```text
modify_fil_channel_all_times.py
```

在指定频点上写入非零复数值，再重新走完整链路做测试。

---

## 1. 当前主要脚本

### `test_with_antenna_uvw.py`

功能：

```text
输入：多个 .fil 文件
输出：一个 MS-ready HDF5 文件
```

它负责：

1. 解析 `.fil` 文件名，识别天线编号和极化编号。
2. 解析 filterbank header。
3. 读取频域复数数据。
4. 对实际存在的信号两两做相关计算。
5. 把相关结果写入 HDF5 的 `/vis`。
6. 写入时间、频率、天线、极化、相位中心、MS 行映射等元数据。
7. 根据天线位置和相位中心计算 UVW。

典型用法：

```bash
python3 test_with_antenna_uvw.py \
  --antenna-txt antenna_positions.txt \
  raw_fil/20000101_030853_00_0.fil \
  raw_fil/20000101_030853_00_1.fil \
  raw_fil/20000101_030853_01_0.fil \
  raw_fil/20000101_030853_01_1.fil
```

---

### `hdf5_to_ms.py`

功能：

```text
输入：MS-ready HDF5
输出：CASA MeasurementSet
```

它负责：

1. 检查 HDF5 结构是否完整。
2. 把 HDF5 中的 signal-level 相关结果重组为 MS 所需的 physical baseline + polarization DATA。
3. 默认保留自相关，所以会写入 `0&0`、`1&1` 等 auto baseline。
4. 默认覆盖旧输出。
5. 默认允许 partial polarization。
6. 默认使用 `UVData.new` 构造 pyuvdata 对象。
7. 直接使用 HDF5 中的 `/uvw/uvw_m`，不再做下游 UVW 符号翻转。
8. 检查 MS MAIN 表和子表。

当前版本的日常用法已经简化为：

```bash
python3 hdf5_to_ms.py input.h5 output.ms
```

例如：

```bash
python3 hdf5_to_ms.py \
  h5/20000101030853153_20000101030853653.h5 \
  ms/0708test.ms
```

当前默认行为固定为：

```text
overwrite=True
include_autos=True
allow_partial_pols=True
uvdata_constructor="new"
```

保留的调试参数主要有：

```text
--allow-uvw-warnings
--x-orientation {east,north,none}
--max-memory-gb
--dry-run
--validate-only
```

推荐正式写 MS 前先做 dry-run：

```bash
python3 hdf5_to_ms.py \
  h5/20000101030853153_20000101030853653.h5 \
  ms/0708test.ms \
  --dry-run
```

---

### `plot_ms_phase_waterfall_timefreq_bjt.py`

功能：

```text
输入：MeasurementSet + 两个信号名，例如 0X 1Y
输出：相位或幅度瀑布图 PNG
```

它支持的信号形式例如：

```text
0X 0X
0X 0Y
0X 1X
0X 1Y
1X 1X
1X 1Y
```

当前绘图方向为：

```text
横轴：时间，单位 s
纵轴：频率，单位 MHz
颜色：相位 rad 或幅度
```

图标题会包含：

```text
Pol pair
BJT 起止时间
```

画相位图：

```bash
python3 plot_ms_phase_waterfall_timefreq_bjt.py \
  0X 0Y \
  ms/0708test.ms
```

画幅度图：

```bash
python3 plot_ms_phase_waterfall_timefreq_bjt.py \
  0X 0Y \
  ms/0708test.ms \
  --mode amp
```

---

### `modify_fil_channel_all_times.py`

功能：

```text
把一个或多个指定频点在所有 FFT / 所有时间样本上都改成非零复数值
```

当前默认会同时修改：

```text
第 1 个频点
第 10 个频点
第 1024 个频点
```

默认写入值：

```text
50 + 0j
```

示例：

```bash
python3 modify_fil_channel_all_times.py \
  raw_fil/20000101_030853_00_0.fil \
  raw_fil/20000101_030853_00_1.fil \
  raw_fil/20000101_030853_01_0.fil \
  raw_fil/20000101_030853_01_1.fil
```

如果你要测试 `0X 0Y` 非零，必须同时修改 `0X` 和 `0Y` 两个输入文件。只改其中一路，相关结果仍然可能是 0。

---

## 2. 输入 `.fil` 文件要求

### 文件名格式

默认要求：

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
xx = 天线编号
P  = 极化编号
```

默认极化映射：

```text
P = 0 -> X
P = 1 -> Y
```

所以：

```text
00_0.fil -> 0X
00_1.fil -> 0Y
01_0.fil -> 1X
01_1.fil -> 1Y
```

### 数据区格式

当前假设 `.fil` 数据区中每个频点是：

```text
real int8 + imag int8
```

也就是每个复数频点占 2 字节。

---

## 3. 天线位置文件 `antenna_positions.txt`

推荐格式：

```text
# name lat lon alt_m diam_m
ant0 29.784402 109.779625 1581 7.5
ant1 29.784410 109.779640 1581 7.5
```

字段含义：

```text
name   天线名，建议 ant0、ant1、ant2 ...
lat    纬度，单位 degree
lon    经度，单位 degree
alt_m  海拔，单位 m
diam_m 口径，单位 m
```

只要输入 `.fil` 中实际出现了 `ant0` 或 `ant1`，这里就必须提供对应的天线位置。

---

## 4. 推荐完整流程

### 正常数据流程

1. 从 `.fil` 生成 HDF5：

```bash
python3 test_with_antenna_uvw.py \
  --antenna-txt antenna_positions.txt \
  raw_fil/20000101_030853_00_0.fil \
  raw_fil/20000101_030853_00_1.fil \
  raw_fil/20000101_030853_01_0.fil \
  raw_fil/20000101_030853_01_1.fil
```

2. 整理输出目录：

```bash
mkdir -p h5 ms plots
mv 20000101030853153_20000101030853653.h5 h5/
```

3. HDF5 转 MS：

```bash
python3 hdf5_to_ms.py \
  h5/20000101030853153_20000101030853653.h5 \
  ms/0708test.ms
```

4. 画图：

```bash
python3 plot_ms_phase_waterfall_timefreq_bjt.py \
  0X 0X \
  ms/0708test.ms \
  plots/amp_0X_0X.png \
  --mode amp
```

```bash
python3 plot_ms_phase_waterfall_timefreq_bjt.py \
  0X 0Y \
  ms/0708test.ms \
  plots/phase_0X_0Y.png
```

### 原始 `.fil` 全 0 时的测试流程

1. 先对输入 `.fil` 写入测试频点：

```bash
python3 modify_fil_channel_all_times.py \
  raw_fil/20000101_030853_00_0.fil \
  raw_fil/20000101_030853_00_1.fil \
  raw_fil/20000101_030853_01_0.fil \
  raw_fil/20000101_030853_01_1.fil
```

2. 用修改后的 `.fil` 重新生成 HDF5：

```bash
python3 test_with_antenna_uvw.py \
  --antenna-txt antenna_positions.txt \
  raw_fil/20000101_030853_00_0_ch1_ch10_ch1024_alltimes_nonzero.fil \
  raw_fil/20000101_030853_00_1_ch1_ch10_ch1024_alltimes_nonzero.fil \
  raw_fil/20000101_030853_01_0_ch1_ch10_ch1024_alltimes_nonzero.fil \
  raw_fil/20000101_030853_01_1_ch1_ch10_ch1024_alltimes_nonzero.fil
```

3. 再做 HDF5 -> MS -> 画图。

---

## 5. 常见检查命令

### 检查 HDF5 `/vis` 是否全 0

```bash
python3 - <<'PY'
import h5py
import numpy as np

h5file = "h5/20000101030853153_20000101030853653.h5"

with h5py.File(h5file, "r") as f:
    vis = f["vis"]
    print("vis shape:", vis.shape)
    print("vis dtype:", vis.dtype)
    print("global max abs:", float(np.max(np.abs(vis[()]))))
    print("global nonzero:", int(np.count_nonzero(vis[()])))
PY
```

如果输出：

```text
global max abs: 0.0
global nonzero: 0
```

说明 HDF5 的可见度数据全 0。此时 MS 全 0 是保真结果，不代表转换器写坏了。

### 检查 MS 里有哪些 baseline

```bash
python3 - <<'PY'
import casacore.tables as ct
import numpy as np

ms = "ms/0708test.ms"

tb = ct.table(ms, readonly=True, ack=False)
ant1 = tb.getcol("ANTENNA1")
ant2 = tb.getcol("ANTENNA2")
tb.close()

pairs = sorted(set(zip(ant1.tolist(), ant2.tolist())))

print("baselines:")
for p in pairs:
    print(p)

print("nrows:", len(ant1))
print("auto rows:", int(np.sum(ant1 == ant2)))
print("cross rows:", int(np.sum(ant1 != ant2)))
PY
```

对于 2 根天线、500 个时间点，当前默认包含 autos 时应看到：

```text
(0, 0)
(0, 1)
(1, 1)
nrows: 1500
auto rows: 1000
cross rows: 500
```

### 检查 MS 极化顺序

```bash
python3 - <<'PY'
import casacore.tables as ct
import numpy as np

ms = "ms/0708test.ms"

tb = ct.table(ms + "/POLARIZATION", readonly=True, ack=False)
corr_type = np.asarray(tb.getcell("CORR_TYPE", 0), dtype=int).reshape(-1)
tb.close()

print("CORR_TYPE:", corr_type)
PY
```

当前期望：

```text
[9, 10, 11, 12]
```

对应：

```text
XX, XY, YX, YY
```

---

## 6. 重要说明

### 自相关默认保留

当前版本的 `hdf5_to_ms.py` 默认会写入自相关，所以可以直接画：

```text
0X 0X
0X 0Y
1X 1X
1X 1Y
```

前提是这些数据在 HDF5 中实际存在。

### HDF5 全 0 时允许写 MS

如果输入 HDF5 `/vis` 本身全 0：

```text
payload DATA 全 0 -> 只 warning，不报错
MS 仍然会写出
phase / amp 图可以用于验证流程是否跑通
但没有科学意义
```

### 不再提供下游 UVW 翻转

当前 `hdf5_to_ms.py` 直接使用 HDF5 中的：

```text
/uvw/uvw_m
```

不再提供 `--flip-uvw-sign`。也就是说，下游默认相信 HDF5 中的 UVW 已经是 MS-ready。

---

## 7. 常见问题

### `No rows found for physical baseline 0&0`

这说明你当前使用的 MS 里没有 `0&0` 自相关行。请确认：

1. 你画的是刚生成的新 MS，而不是旧文件。
2. MS 的 baseline 检查结果里确实有 `(0, 0)`。

### 图全黑或全 0

优先检查 HDF5 `/vis` 是否全 0。如果全 0，说明输入数据本身就没有信号，需要先从 `.fil` 端写入测试频点。

### 自相关相位看起来接近 0

这是正常的。像：

```text
0X 0X = 0X x conj(0X)
```

本质上更适合看幅度，不适合验证复杂相位结构。

### 想看非零相位

优先测试：

```text
0X 0Y
0X 1X
0X 1Y
```

而且相关的两个输入 `.fil` 都必须是非零。

---

## 8. 推荐环境

建议使用 conda，例如：

```bash
conda create -n hdf2ms2 python=3.8 -y
conda activate hdf2ms2
conda install -c conda-forge numpy h5py astropy pyuvdata python-casacore matplotlib -y
pip install katpoint
```

检查：

```bash
python3 - <<'PY'
import numpy
import h5py
import astropy
import pyuvdata
import casacore.tables
import matplotlib
import katpoint

print("numpy", numpy.__version__)
print("h5py", h5py.__version__)
print("astropy", astropy.__version__)
print("pyuvdata", pyuvdata.__version__)
print("matplotlib", matplotlib.__version__)
print("katpoint OK")
print("casacore OK")
PY
```

---

## 9. 一句话总结

```text
正常链路：
    .fil -> test_with_antenna_uvw.py -> HDF5 -> hdf5_to_ms.py -> MS -> plot_ms_phase_waterfall_timefreq_bjt.py

原始 .fil 全 0 时的测试链路：
    .fil -> modify_fil_channel_all_times.py -> test_with_antenna_uvw.py -> HDF5 -> hdf5_to_ms.py -> MS -> plot_ms_phase_waterfall_timefreq_bjt.py
```
