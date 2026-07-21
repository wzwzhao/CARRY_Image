# CARRY_PHASE1 Radio Interferometer Processing Pipeline

## 1. Project Overview

This project implements a complete radio interferometer data processing
pipeline:

    Raw FPGA/filterbank data (.fil)
            |
            v
    fil_to_hdf5.py
            |
            v
    MS-ready HDF5 visibility archive
            |
            v
    hdf5_to_ms.py
            |
            v
    CASA MeasurementSet (.ms)
            |
            v
    CASA calibration / imaging / analysis

The purpose is to convert custom backend correlation products into a
standard CASA-compatible format while preserving all scientific
metadata:

-   antenna positions
-   array reference information
-   frequency axis
-   time axis
-   polarization products
-   visibility data
-   UVW coordinates
-   phase center information

The generated MeasurementSet can be directly used by CASA tasks such as:

-   listobs
-   plotms
-   tclean
-   calibration workflows

------------------------------------------------------------------------

# 2. Current Array Configuration

Current array:

    Array name:
        CARRY_PHASE1

    Real antennas:
        ant0
        ant1
        ant2
        ant3

    Polarizations:
        X
        Y

    Input signals:
        4 antennas × 2 polarization = 8 signals

    System capability:
        10 antennas × 2 polarization = 20 signals

The code keeps the full 20-input architecture, but the current
observation only uses the four real antennas.

The antenna geometry is defined by the `--ant` antenna file.

The HDF5 array metadata records:

    /array/name = CARRY_PHASE1
    /array/config_name = CARRY_PHASE1
    /array/center_source = mean_of_phase1_antennas

The array reference center is the geometric center of ant0-ant3.

The fil-to-HDF5 code stores this array metadata together with the
visibility data. fileciteturn38file17

------------------------------------------------------------------------

# 3. fil_to_hdf5.py

## Purpose

`fil_to_hdf5.py` converts raw filterbank files into an MS-ready HDF5
format.

Main functions:

1.  Read filterbank header
2.  Parse antenna/polarization from filename
3.  Map input signals
4.  Read complex frequency samples
5.  Calculate correlations
6.  Generate visibility data
7.  Calculate UVW
8.  Save all metadata into HDF5

The script reads the filterbank header through `parse_header()` and
obtains parameters such as:

-   source_name
-   tstart
-   nchans
-   nbits
-   frequency information

fileciteturn38file0

------------------------------------------------------------------------

# 4. Input filename format

Required format:

    YYYYMMDD_HHMMSS_xx_P.fil

Example:

    20260717_101258_00_0.fil
    20260717_101258_00_1.fil

Meaning:

    xx:
        antenna ID

    P:
        polarization

    0:
        X polarization

    1:
        Y polarization

The function `calc_input_signal_no()` maps antenna and polarization into
the 20-input system.

fileciteturn38file0

------------------------------------------------------------------------

# 5. Correlation calculation

The system contains:

    20 input signals

Current backend parameters:

    FFT time resolution:
        4 us

    Channels:
        2048

    Complex format:
        int8(real)+int8(imag)

    Integration:
        1000 us

The visibility calculation is:

    Vij = Xi * conj(Xj)

The output HDF5 visibility array:

    vis

    shape:

    (time,
     baseline,
     frequency)

The HDF5 writer stores visibility and all MS-related metadata.

fileciteturn38file12

------------------------------------------------------------------------

# 6. HDF5 Structure

The generated HDF5 contains:

    /
    ├── vis
    ├── signal
    ├── baseline
    ├── antenna
    ├── array
    ├── field
    ├── frequency
    ├── time
    ├── ms_rows
    ├── uvw
    └── polarization

Important groups:

## antenna

Stores:

-   antenna ID
-   antenna name
-   station
-   latitude
-   longitude
-   altitude
-   ITRF position
-   dish diameter

## array

Stores:

-   array name
-   array center
-   reference frame

## field

Stores:

-   source name
-   phase center RA/Dec

## uvw

Stores:

-   UVW coordinates used by MS export

------------------------------------------------------------------------

# 7. Calibrator and Target Observation

The pipeline supports two observation roles:

    cal

    tar

The generated HDF5 contains machine-readable metadata:

Calibration:

    CALIBRATE_PHASE#ON_SOURCE

Target:

    OBSERVE_TARGET#ON_SOURCE

This information is later converted into MS STATE and SCAN tables.

fileciteturn38file10

------------------------------------------------------------------------

# 8. hdf5_to_ms.py

## Purpose

Convert MS-ready HDF5 into CASA MeasurementSet.

Pipeline:

    HDF5
     |
     v
    pyuvdata UVData
     |
     v
    MeasurementSet

The converter reads:

-   visibility
-   antenna metadata
-   array center
-   field information
-   polarization
-   UVW

and writes CASA standard tables.

The script validates:

-   SPECTRAL_WINDOW
-   POLARIZATION
-   DATA_DESCRIPTION
-   ANTENNA
-   FIELD

after writing.

fileciteturn38file6

------------------------------------------------------------------------

# 9. MeasurementSet content

Generated MS contains:

    MAIN

    ANTENNA

    FIELD

    STATE

    OBSERVATION

    SPECTRAL_WINDOW

    POLARIZATION

    DATA_DESCRIPTION

Important mappings:

HDF5:

    /array/name

becomes:

    MS OBSERVATION/TELESCOPE_NAME

HDF5:

    /antenna/position_itrf_m

becomes:

    MS ANTENNA positions

------------------------------------------------------------------------

# 10. Multiple HDF5 files

The converter supports combining:

    cal.h5
    +
    tar.h5

into:

    one MS

The final MS contains:

    FIELD table

    STATE table

    SCAN table

allowing CASA to distinguish:

-   calibrator scans
-   target scans

------------------------------------------------------------------------

# 11. CASA workflow

## Check MS

    listobs("example.ms")

Check:

-   antenna number
-   fields
-   scans
-   polarization
-   frequency

## UV coverage

    plotms(
        vis="example.ms",
        xaxis="u",
        yaxis="v"
    )

## Dirty image

Use:

    tclean(
        vis="example.ms",
        niter=0
    )

`niter=0` means:

-   no CLEAN
-   only produce dirty image

## CLEAN imaging

Increase:

    niter > 0

for deconvolution.

------------------------------------------------------------------------

# 12. Current scientific pipeline

Final workflow:

    FPGA backend

          |

    Filterbank files

          |

    fil_to_hdf5.py

          |
          |
          +-- header parsing
          +-- signal mapping
          +-- correlation
          +-- antenna metadata
          +-- UVW calculation

          |

    MS-ready HDF5

          |

    hdf5_to_ms.py

          |
          +-- pyuvdata conversion
          +-- CASA MS creation
          +-- metadata validation

          |

    CASA

          |
          +-- calibration
          +-- imaging
          +-- source localization
          +-- FRB analysis

------------------------------------------------------------------------

# 13. Important Notes

## Antenna metadata

Only real antennas should be included.

Current:

    ant0-ant3

Do not add artificial positions for future antennas.

------------------------------------------------------------------------

## Array center

Current:

    CARRY_PHASE1

reference center:

    geometric mean of ant0-ant3

------------------------------------------------------------------------

## UVW

The HDF5 UVW:

    /uvw/uvw_m

is the authoritative UVW used during MS export.

------------------------------------------------------------------------


------------------------------------------------------------------------

# 14. HDF5 Data Format Detailed Description

HDF5 is the intermediate scientific archive format between the backend
correlation system and CASA MeasurementSet.

The design goal is that a researcher can understand the observation only
from the HDF5 file without reading the conversion code.

The HDF5 structure is:

```
/
├── vis
├── signal
├── baseline
├── antenna
├── array
├── field
├── frequency
├── time
├── ms_rows
├── uvw
└── polarization
```

---

## 14.1 Visibility Dataset

Path:

```
/vis
```

Meaning:

Correlation visibility products.

Shape:

```
(Ntime, Nbaseline, Nfrequency)
```

Example:

```
(1440, 210, 2048)
```

where:

- Ntime: integrated time samples
- Nbaseline: correlation pairs
- Nfrequency: frequency channels

Data type:

```
complex64
```

The correlation equation is:

```
Vij = Xi × conj(Xj)
```

The baseline dimension contains:

```
20 auto correlations
+
190 cross correlations

= 210 total signal pairs
```

---

## 14.2 Signal Group

Path:

```
/signal
```

Purpose:

Describe the mapping between input files and the 20-input hardware model.

Important datasets:

```
signal/present
signal/antenna_id
signal/polarization_id
signal/file
```

Signal mapping:

```
signal 0  -> ant0 X
signal 1  -> ant0 Y

signal 2  -> ant1 X
signal 3  -> ant1 Y

...

signal 19 -> ant9 Y
```

For the current CARRY_PHASE1 observation:

```
signal 0-7   present=True
signal 8-19  present=False
```

The system capability remains 20 inputs, while the current physical array uses
8 signals.

---

## 14.3 Antenna Group

Path:

```
/antenna
```

Stores physical antenna information.

Datasets:

```
id
name
station
latitude_deg
longitude_deg
altitude_m
position_itrf_m
dish_diameter_m
```

Coordinate system:

```
ITRF / WGS84 ECEF
```

Unit:

```
meter
```

`position_itrf_m` contains absolute antenna positions.

---

## 14.4 Array Group

Path:

```
/array
```

Stores observatory-level information.

Current configuration:

```
array/name = CARRY_PHASE1
```

Array center:

```
center_source = mean_of_phase1_antennas
```

The center is calculated from:

```
ant0
ant1
ant2
ant3
```

using their geometric mean position.

Stored values:

```
center_itrf_m
center_longitude_deg
center_latitude_deg
center_altitude_m
```

This information is used by:

- pyuvdata telescope location
- CASA Observatory registration
- coordinate consistency checking

---

## 14.5 Field Group

Path:

```
/field
```

Stores the observed sky direction.

Datasets:

```
source_name

phase_center_ra_rad

phase_center_dec_rad

frame
```

Example:

```
frame = J2000
```

The field information is used for:

- UVW calculation
- CASA FIELD table
- imaging phase center

---

## 14.6 Frequency Group

Path:

```
/frequency
```

Stores:

```
chan_freq_hz
chan_width_hz
```

The HDF5 format preserves the original frequency ordering.

During HDF5 → MS conversion:

```
frequency axis is normalized to ascending order
```

to satisfy CASA requirements.

---

## 14.7 Time Group

Path:

```
/time
```

Stores:

```
start_mjd
center_mjd
end_mjd
interval_sec
```

Time reference:

```
UTC MJD
```

The center time of each visibility integration is used for UVW calculation.

---

## 14.8 UVW Group

Path:

```
/uvw
```

Stores:

```
uvw_m
```

Shape:

```
(NMS_rows,3)
```

Unit:

```
meter
```

The UVW stored here is directly used during HDF5 → MS conversion.

---

# 15. Running Guide

## 15.1 Environment Requirements

Python:

```
Python >= 3.8
```

Required packages:

```
numpy
h5py
katpoint
astropy
pyuvdata
python-casacore
```

CASA:

```
CASA 5.8+
```

---

## 15.2 fil → HDF5

Input:

```
8 filterbank files
```

Example:

```
20260717_101258_00_0.fil
20260717_101258_00_1.fil
...
20260717_101258_03_1.fil
```

Calibrator:

```bash
python fil_to_hdf5.py \
    -ant CARRY_PHASE1_antennas.txt \
    -o cal \
    *.fil
```

Output:

```
*_cal.h5
```

Target:

```bash
python fil_to_hdf5.py \
    -ant CARRY_PHASE1_antennas.txt \
    -o tar \
    *.fil
```

Output:

```
*_tar.h5
```

---

## 15.3 HDF5 Inspection

Before converting to MS, check:

```
array:

CARRY_PHASE1


antenna:

ant0-ant3


field:

RA/DEC


uvw:

is_placeholder = False
```

---

## 15.4 HDF5 → MS

Input:

```
cal.h5
tar.h5
```

Output:

```
CARRY_PHASE1.ms
```

Example:

```bash
python hdf5_to_ms.py \
    cal.h5 \
    tar.h5 \
    CARRY_PHASE1.ms
```

---

# 16. Data Quality Verification

## UV Coverage

CASA:

```python
plotms(
    vis="CARRY_PHASE1.ms",
    xaxis="u",
    yaxis="v"
)
```

Check:

- baseline existence
- UV distribution
- missing baselines


---

## Dirty Image

```python
tclean(
    vis="CARRY_PHASE1.ms",
    imagename="dirty",
    niter=0
)
```

`niter=0`:

- no CLEAN
- produce dirty image only


---

## Calibration Source

Check:

- phase stability
- amplitude stability
- visibility phase behavior


---

## Target Source

Check:

- imaging
- source position
- localization accuracy


---

# 17. Common Problems and Solutions

## Problem: Telescope not recognized

Example:

```
Telescope CARRY_PHASE1 is not recognized by CASA
```

Cause:

CASA Observatory table does not contain CARRY_PHASE1.

Solution:

Register CARRY_PHASE1 using the array center stored in:

```
/array/center_itrf_m
```

---

## Problem: UVW mismatch warning

Example:

```
uvw_array does not match expected values
```

Possible cause:

pyuvdata checks UVW before complete phase-center metadata is applied.

Solution:

HDF5 → MS conversion order:

```
create UVData

apply phase center metadata

attach HDF5 UVW

check UVData
```

---

## Problem: CASA viewer failure

Example:

```
DBus daemon has died
```

Cause:

GUI environment problem.

Solution:

Export FITS:

```
exportfits()
```

and view using:

- DS9
- CARTA
- other FITS viewers

---

# 18. Software Design Principles

## Single Source of Array Geometry

The authoritative geometry source:

```
CARRY_PHASE1_antennas.txt
```

Flow:

```
antenna file
      |
      v
HDF5 metadata
      |
      v
MeasurementSet
      |
      v
CASA
```

CASA is not the source of antenna geometry.

---

## Scientific Metadata Preservation

Every visibility dataset preserves:

```
time

frequency

antenna

array

field

polarization

UVW
```

---

## Reproducibility

The same:

```
fil data

+

antenna configuration

+

processing parameters
```

should always produce identical:

```
HDF5

MeasurementSet
```


------------------------------------------------------------------------

# 19. Future extension

When more antennas are built:

Example:

    CARRY_FULL

can be introduced with:

    ant0-ant9

The same HDF5-MS architecture can continue to be used.

Future pipeline:

    beamforming

    candidate detection

    FRB localization

    interferometric imaging verification

will be built on this data format.
