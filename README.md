# Arctic Sea Ice Monitoring

This repository contains the code base of our first research-oriented Engineering Project at TUM. It was developed as an academic prototype and learning project. The workflow, structure, and documentation aim to be technically clear and reproducible, but the repository should not be read as a production-ready operational system.

## Overview

This repository implements a satellite-based data fusion workflow for Arctic sea-ice state monitoring using Sentinel-1 SAR, Sentinel-3 SLSTR thermal infrared data, and Canadian Ice Service ice charts in SIGRID-3 format.

The main objective is to construct an analysis-ready Arctic datacube for sea-ice monitoring and machine learning. The workflow combines structural radar information with thermal surface-state information and extends the fused dataset with time-series features such as measurement age and short-term trends. The final output is a NetCDF datacube that can be used for unsupervised clustering, feature validation, and supervised machine-learning experiments.

## Scientific motivation

Single-sensor Arctic monitoring remains limited. SAR is robust in darkness and under cloud cover and captures surface roughness and structural contrast well. Thermal infrared data provide physically meaningful information on surface temperature and freezing state. However, both sensor types become ambiguous when used alone. SAR can confuse rough open water with ice and smooth ice with water. Thermal infrared data are weather dependent and too coarse to resolve many fine leads and local edge structures.

The repository implements a fused, time-aware representation intended to reduce these limitations and provide a stronger basis for downstream ice-state classification.

## What the repository does

The repository is organized around five main tasks:

1. Acquire satellite data from CDSE via openEO
2. Preprocess and harmonize SAR and thermal imagery on a common Arctic grid
3. Build a fused datacube with temporal alignment, trend features, and recency features
4. Integrate validation data from NSIDC/CIS SIGRID-3 charts
5. Evaluate the fused feature space with clustering and machine-learning scripts

## Pipeline summary

### 1. Data acquisition

The acquisition layer is implemented in `src/data_fetcher.py`.

It connects to `https://openeo.dataspace.copernicus.eu`, authenticates through `authenticate_oidc()`, and queries:

* `SENTINEL1_GRD` with bands `VV`, `VH` or `HH`, `HV`
* `SENTINEL3_SLSTR` with bands `S8`, `S9`

It also applies:

* `sar_backscatter()` with `COPERNICUS_30` and `sigma0-ellipsoid`
* reprojection to `EPSG:3995`
* resampling to a 40 m grid

Monthly GeoTIFF batches are exported to:

* `data/GeoTIFF/SAR/<year>/`
* `data/GeoTIFF/Thermal/<year>/`

### 2. Local datacube construction

The local datacube logic is implemented in `src/datacube.py`.

Main processing steps are:

* lazy loading of GeoTIFFs with `rioxarray`
* chronological stacking into `xarray` time series
* SAR denoising via a 3x3 rolling mean
* thermal fusion by averaging `S8` and `S9`
* reprojection-aware land masking with `BedMachineArctic.nc`
* temporal synchronization of SAR and thermal data to a daily `12:00 UTC` grid

### 3. Time-series feature engineering

The workflow is not limited to static fusion.

It computes two main feature families:

* Measurement age

  * `Time_To_Nearest_SAR`
  * `Time_To_Nearest_Thermal`
  * quantifies how recent the nearest valid observation is for each pixel and day

* Trend features over multiple windows

  * `SAR_Trend_3d`, `SAR_Trend_7d`, `SAR_Trend_14d`, `SAR_Trend_30d`
  * `Thermal_Trend_3d`, `Thermal_Trend_7d`, `Thermal_Trend_14d`, `Thermal_Trend_30d`
  * capture short-term structural and thermal change

These features are central to the project because they transform the dataset from a static fused snapshot into a time-aware Arctic datacube.

### 4. Validation integration

Validation is implemented in `src/data_fetcher_validation.py`.

The pipeline downloads Canadian Ice Service Arctic regional sea-ice charts from the NSIDC archive and rasterizes them to the project grid.

The main SIGRID-3 Egg Code variables used in the project are:

* `CT` for total concentration
* `SA` for the stage of development of the thickest ice
* `SB` for the stage of development of the second thickest ice

These are added to the datacube as:

* `val_CT`
* `val_SA`
* `val_SB`

The validation layer is used as an operational proxy, not as perfect ground truth.

### 5. Datacube export

The final fused datasets are saved year by year and then merged into one full-period master cube:

* yearly: `data/NetCDF/arctic_datacube_<year>.nc`
* merged: `data/NetCDF/arctic_datacube_full_period.nc`

## Repository structure

```text
.
├── config.py
├── main.py
├── ML.py
├── kmean.py
├── STvsGT.py
├── check.py
├── check_datacube.py
├── src/
│   ├── data_fetcher.py
│   ├── data_fetcher_validation.py
│   ├── datacube.py
│   └── time_series_extraction.py
├── plots/
└── data/
    ├── GeoTIFF/
    ├── Validation/
    ├── Masks/
    └── NetCDF/
```

## Key files

### Core pipeline

* `config.py`
  Defines AOI, time period, and target CRS.

* `main.py`
  Main orchestration script. It can run monthly downloads, yearly fusion, validation integration, and final full-period merge.

* `src/data_fetcher.py`
  openEO/CDSE acquisition of SAR and thermal GeoTIFFs.

* `src/datacube.py`
  SAR and thermal loading, denoising, land masking, temporal alignment, trend generation, and fused dataset creation.

* `src/data_fetcher_validation.py`
  Download, rasterize, and align NSIDC/CIS validation charts.

### Analysis and visualization scripts

* `ML.py`
  Ablation-style machine-learning comparison between SAR only, SAR plus thermal, and the full 4D cube with trend and recency features.

* `kmean.py`
  Unsupervised K-means clustering to test whether the fused SAR and thermal feature space already separates physically plausible ice and water zones.

* `STvsGT.py`
  Generates visual comparisons between SAR, thermal, and chart-derived validation data across selected dates.

* `check_datacube.py`
  Visual inspection of datacube slices.

* `check.py`
  Inspection of raw GeoTIFF tiles.

* `src/time_series_extraction.py`
  Pixel-level temporal drill-down of SAR and thermal signals.

## Configuration

The main configuration is defined in `config.py`.

Example:

```python
AOI_arctic = "POLYGON((15.0 81.0, 16.0 81.0, 16.0 81.2, 15.0 81.2, 15.0 81.0))"
TIME_PERIOD = ("2020-01-01", "2023-12-31")
TARGET_CRS = "EPSG:3995"
```

You can modify:

* the AOI in WKT format
* the time period
* the target projection

## Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd ArcticSeaIceMonitoring
```

### 2. Create an environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

A minimal package set inferred from the code base is:

```bash
pip install openeo xarray rioxarray rasterio geopandas pandas numpy matplotlib scikit-learn requests netCDF4
```

Depending on your platform, you may also need:

```bash
pip install shapely pyproj fiona
```

## Data requirements

The pipeline expects the following local resources or generated directories:

* `data/GeoTIFF/SAR/`
* `data/GeoTIFF/Thermal/`
* `data/Validation/G02171/`
* `data/Masks/BedMachineArctic.nc`
* `data/NetCDF/`

`BedMachineArctic.nc` is required for ocean-only masking.

## Authentication

The code uses:

```python
self.connection.authenticate_oidc()
```

This means you need a working openEO/CDSE login flow configured on the machine running the acquisition. Depending on the setup, authentication may open a browser-based sign-in or require an already configured environment.

## How to run

### Option A: Full pipeline

In `main.py`, the monthly acquisition and yearly processing block is currently commented out, while the full-period merge remains active.

For a full run:

1. Uncomment the acquisition and yearly processing block in `main.py`
2. Ensure openEO/CDSE access works
3. Ensure `BedMachineArctic.nc` is available
4. Run:

```bash
python main.py
```

### Option B: Merge existing yearly cubes only

If yearly NetCDF cubes already exist in `data/NetCDF/`, the current `main.py` can merge them into:

```text
data/NetCDF/arctic_datacube_full_period.nc
```

Run:

```bash
python main.py
```

### Option C: Analysis scripts

Run individual scripts after the full-period datacube exists:

```bash
python kmean.py
python ML.py
python STvsGT.py
python check_datacube.py
python src/time_series_extraction.py
```

## Expected outputs

### Intermediate outputs

* monthly SAR GeoTIFFs
* monthly thermal GeoTIFFs
* rasterized validation NetCDFs
* yearly datacubes

### Final outputs

* `arctic_datacube_full_period.nc`
* clustering figures
* machine-learning comparison charts
* SAR, thermal, and ground-truth comparison figures
* datacube inspection plots

## Machine-learning setup in this repository

Two main ML-style workflows are included.

### 1. Unsupervised validation with K-means

`kmean.py`:

* finds the best day with overlapping SAR and thermal coverage
* standardizes SAR and thermal features
* performs `KMeans(n_clusters=4)`
* visualizes whether the fused input space separates physically plausible surface zones without labels

### 2. Supervised feature ablation

`ML.py`:

* samples valid pixels from the datacube
* uses `val_CT` to define a simplified binary target between water and consolidated ice
* compares three feature sets:

  * SAR only
  * SAR + thermal
  * full time-aware feature space
* trains a `HistGradientBoostingClassifier`
* exports an ablation chart showing how much thermal and temporal features improve separability

## Important caveats

* The validation data are proxy labels, not perfect ground truth.
* Validation resolution is coarser than many fine leads and local structures visible in SAR.
* The code base contains traces of multiple project stages, including earlier report wording that referenced different polarizations or downstream models. The current acquisition code uses `VV/VH`.
* In `main.py`, the full acquisition and fusion loop is commented out. The repository currently behaves more like a reproducible project code base than a one-command production pipeline.
* openEO/CDSE behavior may change over time; Arctic SAR processing issues encountered during development may not reproduce identically.

## Current project status

The repository demonstrates:

* Arctic multi-sensor data acquisition through openEO
* SAR and thermal fusion on a common Arctic grid
* time-aware feature engineering
* SIGRID-3 validation integration
* creation of an ML-ready NetCDF datacube
* first unsupervised and supervised feature-space checks

The repository should be read as an academic research prototype and Engineering Project code base, not as a production-grade operational sea-ice routing system.

## Suggested future extensions

* add a proper `requirements.txt` or `environment.yml`
* externalize openEO credentials and configuration
* formalize experiment configuration
* document exact AOI and test-region branches
* add Random Forest and POLARIS-based routing experiments
* package the pipeline as callable modules or CLI commands

## Citation

If you use or adapt this repository, cite the associated engineering project report and the external data sources used in the workflow, especially:

* Sentinel-1 and Sentinel-3 documentation
* NSIDC Canadian Ice Service SIGRID-3 archive
* Copernicus Data Space Ecosystem / openEO

## License

No formal license has been assigned yet. As mentioned above, this project was developed as our first introduction to scientific research at TUM and should be understood primarily as an academic learning and prototype repository. If you are interested in using, adapting, or building on this work, please contact us first:
ferdinand.lieben-seutter@tum.de
lennart.gottwald@tum.de

