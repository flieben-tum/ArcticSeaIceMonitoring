# import libraries and mute warnings
import os
os.environ['CPL_LOG'] = '/dev/null'
import xarray as xr
import rioxarray
import gc
import pandas as pd
import numpy as np
from pathlib import Path
from config import TARGET_CRS

# mute generic warnings
import warnings
warnings.filterwarnings("ignore")

class ArcticDataCube:
    def __init__(self, base_dir = "data/GeoTIFF/"):
        """Initialize the ArcticDataCube with the base directory for GeoTIFF files"""

        self.sar_dir = Path(base_dir) / "SAR"
        self.thermal_dir = Path(base_dir) / "Thermal"
        self.target_crs = TARGET_CRS

    def get_date_from_filename(self, filepath):
        """Extract date from the filename assuming format contains 'openEO_YYYY_MM_DDZ.tif'"""
        filename = filepath.name
        try:
            date_str = filename.split('_')[1].split('Z')[0]
            return pd.to_datetime(date_str)
        except IndexError:
            print(f"Could not extract date from filename: {filename} and will be skipped.")
            return None
        
    def load_sar_data(self, year = None):
        """Load all SAR GeoTIFF for a specific year and stack them into an timeline"""

        # find all SAR files and sort them
        if year is not None:
            sar_dir = self.sar_dir / f"{year}"
            sar_files = sorted(list(sar_dir.glob("*.tif")))
        else:
            sar_files = sorted(list(self.sar_dir.rglob("*.tif")))
        
        if not sar_files:
            raise FileNotFoundError(f"No SAR GeoTIFF files found in {self.sar_dir}")

        sar_list = []
        timestamps = []

        # loop over all files
        for f in sar_files:
            dt = self.get_date_from_filename(f)
            if dt is None:
                continue

            ds = rioxarray.open_rasterio(f, chunks={"x": 512, "y": 512})

            ds = ds.rio.write_nodata(-9999)

            if ds.rio.count == 2:
                ds.coords['band'] = ['HH', 'HV']

            sar_list.append(ds)
            timestamps.append(dt)

        # check if we have any valid data for a given year
        if not sar_list:
            print(f"Warning: No valid SAR data found for year {year}")
            return None

        # build the stack
        try:
            sar_cube = xr.concat(sar_list, dim = pd.Index(timestamps, name="time"))
            sar_cube = sar_cube.sortby("time")
            sar_cube = sar_cube.groupby("time").mean(keep_attrs=True)
            sar_cube.name = "SAR"
            return sar_cube
        
        except Exception as e:
            print(f"Error concatenating SAR data: {e}")
            return None
        
    def load_thermal_data(self, target_sar_cube, year = None):
        """Load all Thermal data, averages S8/S9 bands, and resamples to match the SAR grid"""

        if year is not None:
            thermal_dir = self.thermal_dir / f"{year}"
            thermal_files = sorted(list(thermal_dir.glob("*.tif")))
        else:
            thermal_files = sorted(list(self.thermal_dir.rglob("*.tif")))

        if not thermal_files:
            print(f"Warning: No Thermal files found in {self.thermal_dir}")
            return None
        
        # template only of the grid and CRS, data will be filled in the loop
        template = target_sar_cube.isel(time=0).copy(deep = False)

        thermal_list = []
        timestamps = []

        for f in thermal_files:
            dt = self.get_date_from_filename(f)
            if dt is None:
                continue

            # load the data in small chunks
            ds = rioxarray.open_rasterio(f, chunks={"x": 512, "y": 512})
            # build mean of S8 and S9 if there is a band dimension
            if 'band' in ds.dims:
                ds = ds.mean(dim='band', keep_attrs=True)

            ds.name = "Thermal"

            thermal_list.append(ds)
            timestamps.append(dt)

            #explicit garbage collection
            del ds

        if not thermal_list:
            return None
        
        # concatenate and mask out invalid values
        try:
            thermal_cube = xr.concat(thermal_list, dim = pd.Index(timestamps, name="time"))
            thermal_cube = thermal_cube.sortby("time")
            thermal_cube = thermal_cube.groupby("time").mean(keep_attrs=True)
            thermal_cube = thermal_cube.where(thermal_cube  > 0)
            return thermal_cube
        
        except Exception as e:
            print(f"Error concatenating Thermal data: {e}")
            return None
        
    def denoise_sar(self, sar_cube, window_size = 3):
        """Apply a simple Box-Filter to cut speckle noise in SAR data"""

        # apply rolling mean
        sar_denoised = sar_cube.rolling(x=window_size, y=window_size, center=True, min_periods=1).mean()

        return sar_denoised
    
    def apply_land_mask(self, dataset):
        """Load BedMashine and project to Grid, then apply as a mask to the dataset (0 = Ocean, 1 = Land, 2 = Grounded Ice)"""

        mask_path = Path("data/Masks/BedMachineArctic.nc")

        # load mask 
        bm_ds = rioxarray.open_rasterio(mask_path, chunks="auto")

        # if bm_ds is a dataset we take it flat
        if isinstance(bm_ds, xr.Dataset):
            bm_ds = bm_ds.to_array().isel(variable=0, drop=True)

        # reproject to target CRS
        if bm_ds.rio.crs is None: 
            bm_ds = bm_ds.rio.write_crs("EPSG:3413")

        # reproject
        bm_reproject = bm_ds.rio.reproject_match(dataset["SAR"], resampling = 0, nodata = 127)

        # cleanup the dimensions
        mask_layer = bm_reproject.squeeze(drop=True)

        # name
        mask_layer.name = "land_class"

        # check mask values
        """unique_vals = list(mask_layer.values.flatten())
        print(f"Values in reprojected mask: {unique_vals[:10]}")"""

        # add to dataset
        dataset["land_mask"] = mask_layer

        # filter (keep only ocean pixels)
        ds_masked = dataset.where(dataset["land_mask"] == 0)

        return ds_masked


    def create_fused_dataset(self, sar_cube, thermal_cube):
        """Create a fused dataset containing both SAR and Thermal data on a daily grid"""

        # find start and end date
        t_min = min(sar_cube.time.min().values, thermal_cube.time.min().values)
        t_max = max(sar_cube.time.max().values, thermal_cube.time.max().values)

        # create daily time scale at noon
        daily_time = pd.date_range(start = t_min, end = t_max, freq = 'D').normalize() + pd.Timedelta('12h')
        target_coords = pd.Index(daily_time, name = "time")

        # match coordinates
        thermal_cube = thermal_cube.assign_coords(x = sar_cube.x, y = sar_cube.y)

        # filter for >0 values
        sar_cube = sar_cube.where(sar_cube > 0)
        thermal_cube = thermal_cube.where(thermal_cube > 0)

        # interpolate SAR and Thermal filled
        sar_aligned = self._align_to_grid(sar_cube, target_coords)
        thermal_aligned = self._align_to_grid(thermal_cube, target_coords)

        # merge into one dataset
        ds_fused = xr.Dataset({
            "SAR": sar_aligned,
            "Thermal": thermal_aligned
        })

        # calculate measurement age and trends
        ds_fused = self.calculate_measurement_age(ds_fused)
        ds_fused = self.calculate_trends(ds_fused, variable="Thermal", days_list=[3, 7, 14, 30])
        ds_fused = self.calculate_trends(ds_fused, variable="SAR", days_list=[3, 7, 14, 30])
                    
        # Setup names and attributes
        ds_fused["SAR"].attrs = sar_cube.attrs
        ds_fused["Thermal"].attrs = thermal_cube.attrs
        ds_fused.attrs["crs"] = self.target_crs
        ds_fused.attrs["description"] = "Fused Arctic Data Cube with SAR and Thermal data"

        # apply land mask
        ds_fused = self.apply_land_mask(ds_fused)

        return ds_fused
    
    def _align_to_grid(self, source_cube, target_time_index):
        """Snaps Satellite data to a common daily grid using nearest neighbor interpolation within a 12h tolerance"""

        result = source_cube.reindex(time = target_time_index, method = "nearest", tolerance = pd.Timedelta('12h'))

        return result
    
    def calculate_trends(self, dataset, variable = "Thermal", days_list = [3,7,14,30], tolerance_days = 3):
        """Calculate short-term trends a new variable over given day windows"""

        for days in days_list:
            target_dates = dataset.time - pd.Timedelta(days=days)

            # search for past value by shifting the time dimension
            past_values = dataset[variable].reindex(
                time = target_dates,
                method = "nearest",
                tolerance = pd.Timedelta(days=tolerance_days)
            )
            past_values['time'] = dataset.time

            # calculate delta tangent
            delta = dataset[variable] - past_values

            tangent_name = f"{variable}_Trend_{days}d"

            # calculate rate of change as delta divided by days
            dataset[tangent_name] = delta / days

        return dataset
    
    def calculate_measurement_age(self, dataset):
        """Calculate the age of each measurement in days since the last valid observation"""

        # get timescale from dataset
        time_da = dataset.time

        # loop for both SAR and Thermal variables
        for sensor in ["SAR", "Thermal"]:
            if sensor not in dataset:
                continue

            if "band" in dataset[sensor].dims:
                valid = dataset[sensor].notnull().any(dim="band")
            else:
                valid = dataset[sensor].notnull()

            measured_times = time_da.where(valid)

            past = (time_da - measured_times.ffill(dim="time")) / np.timedelta64(1, 'D')
            future = (measured_times.bfill(dim="time") - time_da) / np.timedelta64(1, 'D')

            age = xr.where(past < future, past, future)
            age = age.fillna(past).fillna(future)

            dataset[f"Time_To_Nearest_{sensor}"] = age

        return dataset