# CARRY Visibility：从 `.fil` 相关计算到 CASA MeasurementSet 成图的完整流程

本文档是本项目的总说明。目标是让任何第一次打开项目的人，都能知道：

- 这个项目是干什么的；
- 每个脚本分别负责什么；
- 输入文件需要满足什么格式；
- 怎样从 `.fil` 生成 HDF5；
- 怎样从 HDF5 生成 CASA MeasurementSet；
- 怎样在 MS 里画指定天线/极化对的相位或幅度瀑布图；
- 如果原始 `.fil` 全 0，怎样人工写入一个非零频点来测试整条链路；
- 常见报错是什么意思，应该怎么检查。

---

## 1. 项目一句话说明

本项目用于把多路天线/极化的 filterbank `.fil` 文件读取出来，按信号对做相关计算，保存成 MS-ready HDF5，再转换成 CASA 可读取的 MeasurementSet，最后画出指定天线/极化组合的相位或幅度瀑布图。

完整数据链路是：

```text
原始 .fil 文件
    -> test_with_antenna_uvw.py
    -> MS-ready HDF5
    -> hdf5_to_ms.py
    -> CASA MeasurementSet .ms
    -> plot_ms_phase_waterfall_timefreq_bjt.py
    -> 相位/幅度瀑布图 PNG
```

如果原始 `.fil` 文件全 0，也可以先用：

```text
modify_fil_channel_all_times.py
```

在指定频点上给所有 FFT / 所有时间采样写入非零复数值，用来测试后续流程。

---

## 2. 当前项目脚本一览

当前主要脚本如下：

```text
test_with_antenna_uvw.py
hdf5_to_ms.py
plot_ms_phase_waterfall_timefreq_bjt.py
modify_fil_channel_all_times.py
```

建议最终目录结构如下：

```text
Visibility/
├── README.md
├── test_with_antenna_uvw.py
├── hdf5_to_ms.py
├── plot_ms_phase_waterfall_timefreq_bjt.py
├── modify_fil_channel_all_times.py
├── antenna_positions.txt
├── raw_fil/
│   ├── 20000101_030853_00_0.fil
│   ├── 20000101_030853_00_1.fil
│   ├── 20000101_030853_01_0.fil
│   └── 20000101_030853_01_1.fil
├── h5/
│   └── 20000101030853153_20000101030853653.h5
├── ms/
│   └── 0708test.ms
└── plots/
    ├── phase_waterfall_0X_0X.png
    ├── phase_waterfall_0X_0Y.png
    └── amp_waterfall_0X_0X.png
```

---

## 3. 每个脚本的作用

### 3.1 `test_with_antenna_uvw.py`

这是第一步主脚本，用于读取多个 `.fil` 文件，做相关计算，并输出 MS-ready HDF5。

它负责：

1. 解析 `.fil` 文件名，识别天线编号和极化编号；
2. 解析 filterbank header；
3. 检查输入文件数量、时间一致性、重复信号、header 一致性、通道数；
4. 读取 `.fil` 数据区，数据格式为 `real int8 + imag int8`；
5. 按信号对计算相关；
6. 保存相关结果到 HDF5 的 `/vis`；
7. 写入天线、频率、时间、极化、相位中心、UVW、MS 行映射等信息；
8. 使用 `katpoint` 根据天线位置和相位中心计算 UVW；
9. 输出一个后续可被 `hdf5_to_ms.py` 转成 MS 的 HDF5 文件。

它的命令行参数主要是：

```text
--antenna-txt    天线位置文件
files            一个或多个输入 .fil 文件
```

示例：

```bash
python3 test_with_antenna_uvw.py \
  --antenna-txt antenna_positions.txt \
  raw_fil/20000101_030853_00_0.fil \
  raw_fil/20000101_030853_00_1.fil \
  raw_fil/20000101_030853_01_0.fil \
  raw_fil/20000101_030853_01_1.fil
```

如果你的脚本同时支持短参数，也可以写：

```bash
python3 test_with_antenna_uvw.py \
  -ant antenna_positions.txt \
  raw_fil/20000101_030853_00_0.fil \
  raw_fil/20000101_030853_00_1.fil \
  raw_fil/20000101_030853_01_0.fil \
  raw_fil/20000101_030853_01_1.fil
```

脚本会自动根据 header 里的 `tstart` 生成一个 HDF5 文件名，例如：

```text
20000101030853153_20000101030853653.h5
```

这个文件名表示数据起止时间。

---

### 3.2 `hdf5_to_ms.py`

这是第二步主脚本，用于把 MS-ready HDF5 转换成 CASA MeasurementSet。

它负责：

1. 检查 HDF5 是否包含必要路径，例如 `/vis`、`/time`、`/frequency`、`/antenna`、`/uvw` 等；
2. 把 HDF5 里的 signal-level 相关结果重组为 MS 需要的 physical baseline + polarization DATA；
3. 默认保留自相关，所以会写入 `0&0`、`1&1` 等 auto baseline；
4. 默认覆盖旧的输出 MS；
5. 默认允许 partial polarization，缺失极化会写成 flag；
6. 默认使用 `UVData.new` 构造 pyuvdata 对象；
7. 默认使用 HDF5 里的 `/uvw/uvw_m`，不再做 UVW 符号翻转；
8. 检查 MS 主表和子表；
9. 检查 `SPECTRAL_WINDOW` 频率轴是否升序；
10. 检查 `POLARIZATION/CORR_TYPE` 是否是 CASA 标准顺序 `XX, XY, YX, YY`；
11. 输出 CASA 可读的 `.ms` 目录。

日常运行只需要：

```bash
python3 hdf5_to_ms.py input.h5 output.ms
```

例如：

```bash
python3 hdf5_to_ms.py \
  h5/20000101030853153_20000101030853653.h5 \
  ms/0708test.ms
```

现在这个脚本的默认行为是：

```text
overwrite=True
include_autos=True
allow_partial_pols=True
uvdata_constructor="new"
```

也就是说不需要再手动输入：

```text
--overwrite
--include-autos
--allow-partial-pols
--uvdata-constructor new
```

保留的可选参数主要是：

```text
--allow-uvw-warnings
--x-orientation {east,north,none}
--max-memory-gb
--dry-run
--validate-only
```

常用检查命令：

```bash
python3 hdf5_to_ms.py \
  h5/20000101030853153_20000101030853653.h5 \
  ms/0708test.ms \
  --dry-run
```

只验证，不写 MS：

```bash
python3 hdf5_to_ms.py \
  h5/20000101030853153_20000101030853653.h5 \
  ms/0708test.ms \
  --validate-only
```

---

### 3.3 `plot_ms_phase_waterfall_timefreq_bjt.py`

这是第三步画图脚本，用于从 MS 里提取指定天线/极化对，然后画瀑布图。

它支持输入：

```text
0X 0X
0X 0Y
0X 1X
0X 1Y
1X 1X
1X 1Y
```

含义示例：

```text
0X 0X = antenna 0 的 X 极化 x conj(antenna 0 的 X 极化)
0X 0Y = antenna 0 的 X 极化 x conj(antenna 0 的 Y 极化)
0X 1Y = antenna 0 的 X 极化 x conj(antenna 1 的 Y 极化)
```

输出图像方向是：

```text
横轴：时间，单位 s
纵轴：频率，单位 MHz
颜色：相位 rad 或幅度
```

标题里会写：

```text
Pol pair: 0X 0X (XX)
BJT: 起始北京时间 - 结束北京时间
```

时间范围来自 MS MAIN 表里的 `TIME` 和 `INTERVAL`，会用行中心时间加减半个积分时间得到完整时间范围，并转换成北京时间，精确到毫秒。

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

保存 `.npy` 和 `.txt` 摘要：

```bash
python3 plot_ms_phase_waterfall_timefreq_bjt.py \
  0X 0Y \
  ms/0708test.ms \
  --save-npy \
  --save-txt
```

指定输出图片路径：

```bash
python3 plot_ms_phase_waterfall_timefreq_bjt.py \
  0X 0Y \
  ms/0708test.ms \
  plots/phase_0X_0Y.png
```

---

### 3.4 `modify_fil_channel_all_times.py`

这是测试辅助脚本。当原始 `.fil` 数据区全 0 时，可以用它把指定频点的所有 FFT / 所有时间采样都改成非零复数值。

默认行为：

```text
channel number = 1024，1-based
channel index  = 1023，0-based
value          = 50 + 0j
time/FFT range = 文件内全部 FFT
output         = 复制新文件，不覆盖原文件
```

数据区假设：

```text
FFT0:
    ch0 real int8, ch0 imag int8
    ch1 real int8, ch1 imag int8
    ...
    ch2047 real int8, ch2047 imag int8

FFT1:
    同样结构
```

每个复数频点占 2 字节：

```text
real int8 + imag int8
```

改一个文件：

```bash
python3 modify_fil_channel_all_times.py \
  raw_fil/20000101_030853_00_0.fil
```

会生成：

```text
raw_fil/20000101_030853_00_0_ch1024_alltimes_nonzero.fil
```

改多个文件：

```bash
python3 modify_fil_channel_all_times.py \
  raw_fil/20000101_030853_00_0.fil \
  raw_fil/20000101_030853_00_1.fil \
  raw_fil/20000101_030853_01_0.fil \
  raw_fil/20000101_030853_01_1.fil
```

修改复数值：

```bash
python3 modify_fil_channel_all_times.py \
  raw_fil/20000101_030853_00_0.fil \
  --real 50 \
  --imag 20
```

直接覆盖原文件，不推荐：

```bash
python3 modify_fil_channel_all_times.py \
  raw_fil/20000101_030853_00_0.fil \
  --in-place
```

如果你想让 `0X 0Y` 非零，必须同时修改 `0X` 和 `0Y` 两个文件。只改其中一个，另一个仍然全 0，相关结果还是 0。

---

## 4. 输入 `.fil` 文件格式要求

### 4.1 文件名格式

`test_with_antenna_uvw.py` 默认要求文件名格式为：

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

字段含义：

```text
YYYYMMDD = 日期
HHMMSS   = 时间
xx       = 天线编号，例如 00、01
P        = 极化编号，0 或 1
```

极化编号默认含义：

```text
P=0 -> X 极化
P=1 -> Y 极化
```

所以：

```text
00_0.fil -> 0X
00_1.fil -> 0Y
01_0.fil -> 1X
01_1.fil -> 1Y
```

### 4.2 数据区格式

`.fil` header 结束后，数据区按 FFT 组织：

```text
FFT0: 2048 个复数频点
FFT1: 2048 个复数频点
...
```

每个频点是：

```text
real int8 + imag int8
```

所以每个频点 2 字节，每个 FFT 的大小是：

```text
2048 * 2 = 4096 bytes
```

---

## 5. 天线位置文件 `antenna_positions.txt`

`test_with_antenna_uvw.py` 需要天线位置来计算 UVW。

推荐格式：

```text
# name lat lon alt_m diam_m
ant0 29.784402 109.779625 1581 7.5
ant1 29.784410 109.779640 1581 7.5
```

字段说明：

```text
name    天线名，建议 ant0、ant1、ant2 ...
lat     纬度，单位 degree
lon     经度，单位 degree
alt_m   海拔，单位 m
diam_m  口径，单位 m
```

注意：

```text
只要输入文件里出现 ant0 或 ant1，天线位置文件里就必须有 ant0 或 ant1。
```

运行时用：

```bash
python3 test_with_antenna_uvw.py \
  --antenna-txt antenna_positions.txt \
  raw_fil/*.fil
```

---

## 6. 正常数据完整流程

### 第一步：从 `.fil` 生成 HDF5

```bash
python3 test_with_antenna_uvw.py \
  --antenna-txt antenna_positions.txt \
  raw_fil/20000101_030853_00_0.fil \
  raw_fil/20000101_030853_00_1.fil \
  raw_fil/20000101_030853_01_0.fil \
  raw_fil/20000101_030853_01_1.fil
```

成功后会生成一个 `.h5` 文件，例如：

```text
20000101030853153_20000101030853653.h5
```

建议移动到 `h5/`：

```bash
mkdir -p h5
mv 20000101030853153_20000101030853653.h5 h5/
```

### 第二步：HDF5 转 MS

```bash
mkdir -p ms

python3 hdf5_to_ms.py \
  h5/20000101030853153_20000101030853653.h5 \
  ms/0708test.ms
```

成功后会看到：

```text
[OK] MeasurementSet written: ms/0708test.ms
```

### 第三步：画瀑布图

画 `0X 0X` 幅度：

```bash
mkdir -p plots

python3 plot_ms_phase_waterfall_timefreq_bjt.py \
  0X 0X \
  ms/0708test.ms \
  plots/amp_0X_0X.png \
  --mode amp
```

画 `0X 0Y` 相位：

```bash
python3 plot_ms_phase_waterfall_timefreq_bjt.py \
  0X 0Y \
  ms/0708test.ms \
  plots/phase_0X_0Y.png
```

画 `0X 1Y` 相位：

```bash
python3 plot_ms_phase_waterfall_timefreq_bjt.py \
  0X 1Y \
  ms/0708test.ms \
  plots/phase_0X_1Y.png
```

---

## 7. 原始 `.fil` 全 0 时的测试流程

如果检查发现 HDF5 `/vis` 全 0，或者画图全黑、幅度全 0，可以先人工写入一个非零频点来验证流程。

### 第一步：修改 `.fil` 指定频点

例如修改 4 个输入文件的第 1024 个频点，所有 FFT 都改成 `50 + 0j`：

```bash
python3 modify_fil_channel_all_times.py \
  raw_fil/20000101_030853_00_0.fil \
  raw_fil/20000101_030853_00_1.fil \
  raw_fil/20000101_030853_01_0.fil \
  raw_fil/20000101_030853_01_1.fil
```

会生成：

```text
raw_fil/20000101_030853_00_0_ch1024_alltimes_nonzero.fil
raw_fil/20000101_030853_00_1_ch1024_alltimes_nonzero.fil
raw_fil/20000101_030853_01_0_ch1024_alltimes_nonzero.fil
raw_fil/20000101_030853_01_1_ch1024_alltimes_nonzero.fil
```

### 第二步：用修改后的 `.fil` 生成 HDF5

```bash
python3 test_with_antenna_uvw.py \
  --antenna-txt antenna_positions.txt \
  raw_fil/20000101_030853_00_0_ch1024_alltimes_nonzero.fil \
  raw_fil/20000101_030853_00_1_ch1024_alltimes_nonzero.fil \
  raw_fil/20000101_030853_01_0_ch1024_alltimes_nonzero.fil \
  raw_fil/20000101_030853_01_1_ch1024_alltimes_nonzero.fil
```

### 第三步：转 MS

```bash
python3 hdf5_to_ms.py \
  h5/生成的新文件.h5 \
  ms/0708test_nonzero.ms
```

### 第四步：画图

```bash
python3 plot_ms_phase_waterfall_timefreq_bjt.py \
  0X 0X \
  ms/0708test_nonzero.ms \
  plots/amp_0X_0X_nonzero.png \
  --mode amp
```

由于第 1024 个频点在所有时间上都是非零，瀑布图中应出现一条沿时间方向延伸的水平亮线。

---

## 8. 相位和幅度应该怎么测试

### 8.1 自相关适合看幅度

例如：

```text
0X 0X = 0X x conj(0X)
```

数学上：

```text
x * conj(x) = |x|^2
```

所以自相关结果通常是实数非负，相位接近 0。

因此：

```bash
python3 plot_ms_phase_waterfall_timefreq_bjt.py \
  0X 0X \
  ms/0708test.ms \
  plots/amp_0X_0X.png \
  --mode amp
```

最适合检查自相关是否有信号。

### 8.2 交叉极化或跨天线相关适合看相位

例如：

```text
0X 0Y
0X 1X
0X 1Y
```

这些相关的相位更适合用来测试：

```bash
python3 plot_ms_phase_waterfall_timefreq_bjt.py \
  0X 0Y \
  ms/0708test.ms \
  plots/phase_0X_0Y.png
```

如果两个输入信号写入不同复数相位，例如一个写 `50+0j`，另一个写 `0+50j`，则交叉相关会出现接近 `-pi/2` 的相位。

---

## 9. 检查 HDF5 是否全 0

可以运行：

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

说明 HDF5 的可见度数据全 0。此时可以用 `modify_fil_channel_all_times.py` 先人工写入非零频点测试流程。

---

## 10. 检查 MS 里有哪些 baseline

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

2 根天线、500 个时间点时，正常应有：

```text
(0, 0)
(0, 1)
(1, 1)
nrows = 1500
auto rows = 1000
cross rows = 500
```

如果没有 `(0, 0)` 或 `(1, 1)`，说明 MS 没有包含自相关。当前 `hdf5_to_ms.py` 默认应该包含自相关。

---

## 11. 检查 MS 频率轴

```bash
python3 - <<'PY'
import casacore.tables as ct
import numpy as np

ms = "ms/0708test.ms"

tb = ct.table(ms + "/SPECTRAL_WINDOW", readonly=True, ack=False)
freq = np.asarray(tb.getcell("CHAN_FREQ", 0), dtype=float).reshape(-1)
tb.close()

print("nchan:", freq.size)
print("first Hz:", freq[0])
print("last Hz:", freq[-1])
print("min Hz:", freq.min())
print("max Hz:", freq.max())
print("ascending:", bool(np.all(np.diff(freq) > 0)))
PY
```

`hdf5_to_ms.py` 会把频率轴整理成升序写入 MS，所以应看到：

```text
ascending: True
```

---

## 12. 检查 MS 极化顺序

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

线极化全极化的 CASA 标准顺序应为：

```text
[9, 10, 11, 12]
```

含义：

```text
9  = XX
10 = XY
11 = YX
12 = YY
```

---

## 13. 在 CASA 中检查 MS

进入 CASA 后：

```python
vis = "/path/to/ms/0708test.ms"
listobs(vis=vis, listfile="0708test.listobs.txt", verbose=True)
```

如果 CASA 能正常读取，说明 MS 结构基本通过。

还可以用 CASA 的 `tb` 工具看主表：

```python
tb.open(vis)
print("nrows =", tb.nrows())
print("columns =", tb.colnames())
data0 = tb.getcell("DATA", 0)
flag0 = tb.getcell("FLAG", 0)
uvw0 = tb.getcell("UVW", 0)
print("DATA shape =", data0.shape)
print("FLAG shape =", flag0.shape)
print("UVW =", uvw0)
tb.close()
```

---

## 14. 常见问题

### 14.1 `No rows found for physical baseline 0&0`

说明你画 `0X 0X` 或 `0X 0Y` 时，MS 中没有 `ANTENNA1=0, ANTENNA2=0` 自相关行。

当前新版 `hdf5_to_ms.py` 默认包含自相关。如果仍报这个错，通常是：

1. 你画的是旧 MS；
2. 新 MS 没有生成成功；
3. `plot_ms_phase_waterfall_timefreq_bjt.py` 的路径指错了；
4. 你用了另一个旧版 `hdf5_to_ms.py`。

检查：

```bash
python3 - <<'PY'
import casacore.tables as ct
ms = "ms/0708test.ms"
tb = ct.table(ms, readonly=True, ack=False)
pairs = sorted(set(zip(tb.getcol("ANTENNA1").tolist(), tb.getcol("ANTENNA2").tolist())))
tb.close()
print(pairs)
PY
```

---

### 14.2 图是全 0 或全黑

先检查 HDF5：

```bash
python3 - <<'PY'
import h5py
import numpy as np
h5file = "h5/your_file.h5"
with h5py.File(h5file, "r") as f:
    vis = f["vis"]
    print(float(np.max(np.abs(vis[()]))))
PY
```

如果是 0，说明输入数据本身全 0。用 `modify_fil_channel_all_times.py` 修改 `.fil` 后重新生成 HDF5 和 MS。

---

### 14.3 只有一条很细的亮线，看不清

如果只改一个频点，瀑布图里确实只有一条很窄的水平线。可以：

1. 增大 `--real` 或 `--imag`，但不要超过 int8 范围；
2. 修改多个相邻频点；
3. 写一个多频段测试脚本，或者多次运行单频点修改脚本修改不同 channel。

---

### 14.4 相位图还是 0

如果画的是自相关，例如：

```text
0X 0X
```

相位接近 0 是正常的。

想看非零相位，应画：

```text
0X 0Y
0X 1X
0X 1Y
```

而且两个输入 `.fil` 必须写入不同复数相位。

---

### 14.5 matplotlib 中文字体警告

项目里的画图脚本已经使用英文标题，比如：

```text
Pol pair
BJT
```

这样可以避免 Linux 上缺少中文字体导致的 glyph warning。

---

## 15. 推荐完整测试命令

假设你有四个原始 `.fil`：

```text
raw_fil/20000101_030853_00_0.fil
raw_fil/20000101_030853_00_1.fil
raw_fil/20000101_030853_01_0.fil
raw_fil/20000101_030853_01_1.fil
```

完整流程如下：

```bash
# 1. 可选：如果原始 fil 全 0，先写入测试频点
python3 modify_fil_channel_all_times.py \
  raw_fil/20000101_030853_00_0.fil \
  raw_fil/20000101_030853_00_1.fil \
  raw_fil/20000101_030853_01_0.fil \
  raw_fil/20000101_030853_01_1.fil

# 2. 用修改后的 fil 生成 HDF5
python3 test_with_antenna_uvw.py \
  --antenna-txt antenna_positions.txt \
  raw_fil/20000101_030853_00_0_ch1024_alltimes_nonzero.fil \
  raw_fil/20000101_030853_00_1_ch1024_alltimes_nonzero.fil \
  raw_fil/20000101_030853_01_0_ch1024_alltimes_nonzero.fil \
  raw_fil/20000101_030853_01_1_ch1024_alltimes_nonzero.fil

# 3. 整理输出目录
mkdir -p h5 ms plots
mv 20000101030853153_20000101030853653.h5 h5/

# 4. HDF5 -> MS
python3 hdf5_to_ms.py \
  h5/20000101030853153_20000101030853653.h5 \
  ms/0708test.ms

# 5. 画自相关幅度
python3 plot_ms_phase_waterfall_timefreq_bjt.py \
  0X 0X \
  ms/0708test.ms \
  plots/amp_0X_0X.png \
  --mode amp

# 6. 画同天线交叉极化相位
python3 plot_ms_phase_waterfall_timefreq_bjt.py \
  0X 0Y \
  ms/0708test.ms \
  plots/phase_0X_0Y.png
```

---

## 16. 推荐环境

建议使用 conda 环境，例如：

```bash
conda create -n hdf2ms2 python=3.8 -y
conda activate hdf2ms2
```

安装依赖：

```bash
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

## 17. 数据科学解释

### 17.1 自相关

```text
0X 0X = 0X x conj(0X)
```

它本质上是功率谱：

```text
|0X|^2
```

所以主要看幅度，不适合测试非零相位。

### 17.2 同天线交叉极化

```text
0X 0Y = 0X x conj(0Y)
```

它可以反映同一天线 X/Y 两路之间的相位差，适合测试 phase waterfall。

### 17.3 跨天线互相关

```text
0X 1Y = 0X x conj(1Y)
```

它包含几何相位、仪器相位、源结构、线缆、时钟等影响。未校准数据的相位不一定平滑。

---

## 18. 最重要的注意事项

1. 不要把全 0 输入误认为 MS 写错。先检查 HDF5 `/vis`。
2. 要画 `0X 0X`、`0X 0Y`，MS 中必须有 `0&0` baseline。
3. 当前新版 `hdf5_to_ms.py` 默认包含自相关。
4. `0X 0Y` 如果想非零，必须 `0X` 和 `0Y` 两个 `.fil` 都非零。
5. 画相位时，自相关相位通常是 0；交叉相关更适合看相位。
6. 频率轴最终在 MS 里应为升序。
7. 瀑布图现在是横轴时间、纵轴频率。
8. 标题里的时间是北京时间，精确到毫秒。
9. 不要随便修改 UVW 符号；当前 HDF5 中 `/uvw/uvw_m` 已经是 MS-ready。
10. `.fil` 数据修改脚本默认复制新文件，不覆盖原始文件，更安全。

---

## 19. 一句话完整流程

```text
如果有真实非零 .fil：
    test_with_antenna_uvw.py -> hdf5_to_ms.py -> plot_ms_phase_waterfall_timefreq_bjt.py

如果 .fil 全 0：
    modify_fil_channel_all_times.py -> test_with_antenna_uvw.py -> hdf5_to_ms.py -> plot_ms_phase_waterfall_timefreq_bjt.py
```
