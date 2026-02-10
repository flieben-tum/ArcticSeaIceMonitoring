# import libraries and mute warnings
import os
os.environ['CPL_LOG'] = '/dev/null'
import xarray as xr
import rioxarray
import pandas as pd
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
        
    def load_sar_data(self):
        """Load all SAR GeoTIFF and stack them into an timeline"""

        # find all SAR files and sort them
        sar_files = sorted(list(self.sar_dir.glob("*.tif")))
        
        if not sar_files:
            raise FileNotFoundError(f"No SAR GeoTIFF files found in {self.sar_dir}")

        sar_list = []
        timestamps = []
        # loop over all files
        for f in sar_files:
            dt = self.get_date_from_filename(f)
            if dt is None:
                continue

            ds = rioxarray.open_rasterio(f, chunks={"x": 1024, "y": 1024})

            try:
                ds = ds.rio.reproject(self.target_crs, resolution = 10, resampling = 0)
                ds = ds.rio.write_nodata(-9999, inplace=True)
            except Exception as e:
                print(f"Error reprojecting {f.name}: {e}")
                continue

            if ds.rio.count == 2:
                ds.coords['band'] = ['HH', 'HV']

            sar_list.append(ds)
            timestamps.append(dt)

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
        
    def load_thermal_data(self, target_sar_cube):
        """Load all Thermal data, averages S8/S9 bands, and resamples to match the SAR grid"""

        thermal_files = sorted(list(self.thermal_dir.glob("*.tif")))

        if not thermal_files:
            print(f"Warning: No Thermal files found in {self.thermal_dir}")
            return None

        thermal_list = []
        timestamps = []

        for f in thermal_files:
            dt = self.get_date_from_filename(f)
            if dt is None:
                continue

            ds = rioxarray.open_rasterio(f, chunks={"x": 1024, "y": 1024})

            if 'band' in ds.dims:
                ds = ds.mean(dim='band', keep_attrs=True)

            # Nodata cleaning
            ds = ds.where(ds > 0)

            template = target_sar_cube.isel(time=0)

            try:               
                ds_resampled = ds.rio.reproject_match(template, from_disk = True)
                ds_resampled.name = "Thermal"
                thermal_list.append(ds_resampled)
                timestamps.append(dt)
            except Exception as e:
                print(f"Error reprojecting {f.name}: {e}")
                continue

        if not thermal_list:
            return None
        
        try:
            thermal_cube = xr.concat(thermal_list, dim = pd.Index(timestamps, name="time"))
            thermal_cube = thermal_cube.sortby("time")
            thermal_cube = thermal_cube.groupby("time").mean(keep_attrs=True)
            return thermal_cube
        
        except Exception as e:
            print(f"Error concatenating Thermal data: {e}")
            return None
        
    def denoise_sar(self, sar_cube, window_size = 3):
        """Apply a simple Box-Filter to cut speckle noise in SAR data"""

        sar_denoised = sar_cube.rolling(x=window_size, y=window_size, center=True).mean()

        sar_denoised = sar_denoised.fillna(sar_cube)

        return sar_denoised

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

        # load cubes into memory for processing
        sar_cube = sar_cube.load()
        thermal_cube = thermal_cube.load()

        # interpolate SAR and Thermal filled
        sar_filled = self._interpolate_to_grid(sar_cube, target_coords)
        thermal_filled = self._interpolate_to_grid(thermal_cube, target_coords)

        # create raw data layers for validation
        sar_raw = sar_cube.reindex(time = target_coords, method = "nearest", tolerance = pd.Timedelta('12h'))
        thermal_raw = thermal_cube.reindex(time = target_coords, method = "nearest", tolerance = pd.Timedelta('12h'))

        # merge into one dataset
        ds_fused = xr.Dataset({
            "SAR": sar_filled,
            "Thermal": thermal_filled,
            "SAR_Raw": sar_raw,
            "Thermal_Raw": thermal_raw
        })
                    
        # Setup names and attributes
        ds_fused["SAR"].attrs = sar_cube.attrs
        ds_fused["Thermal"].attrs = thermal_cube.attrs
        ds_fused.attrs["crs"] = self.target_crs
        ds_fused.attrs["description"] = "Fused Arctic Data Cube with SAR and Thermal data"

        return ds_fused
    
    def _interpolate_to_grid(self, source_cube, target_time_index):
        """Helper function to perform pixel-wise time interpolation to a target time index"""

        # combine original times with target times
        original_times = pd.Index(source_cube.time.values)
        combined_times = original_times.union(target_time_index).sort_values().unique()

        # reindex to the combined timeline
        ds_combined = source_cube.reindex(time = combined_times)

        # interpolate pixel-wise
        ds_interpolated = ds_combined.interpolate_na(dim = "time", method = "nearest")

        # select only target times
        result = ds_interpolated.sel(time = target_time_index)

        # fallback: forward fill then backward fill for any remaining NaNs
        result = result.ffill(dim = "time").bfill(dim = "time")

        return result
    
    def calculate_trends(self, dataset, variable = "Thermal", days_list = [3,7]):
        """Calculate short-term trends a new variable over given day windows"""

        for days in days_list:
            trend_name = f"{variable}_Trend_{days}d"

            # calculate today minus value from 'days' ago
            shifted_time = dataset.time - pd.Timedelta(days=days)
            da_shifted = dataset[variable].assign_coords(time=shifted_time)
            reference_past = da_shifted.interp(time = dataset.time, method = "nearest")
            diff = dataset[variable] - reference_past

            # add to dataset
            dataset[trend_name] = diff

        return dataset

