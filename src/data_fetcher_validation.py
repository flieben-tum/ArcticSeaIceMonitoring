# import libraries
import os
import requests
import zipfile
import shutil
import re
import gc
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from pathlib import Path
from rasterio import features
from config import TARGET_CRS

# Egg-Code variables
EGG_CODE_VARS = ['CT', 'SA', 'SB']

# Validation Data Fetcher class
class ValidationDataFetcher:
    def __init__(self, output_dir = "data/Validation/G02171"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Basis URL for NOAA@NSIDC (G02171)
        self.base_url = "https://noaadata.apps.nsidc.org/NOAA/G10013/north"

        # Chache for datalist per year
        self._year_file_chache = {}

    def _get_file_url_from_index(self, year, date):
        """Reads the fies for a year and finds the one matching the date."""

        year_str = str(year)

        # Get List for this year
        if year_str not in self._year_file_chache:
            year_url = f"{self.base_url}/{year_str}/"
            try:
                r = requests.get(year_url, timeout = 15)
                r.raise_for_status()
                self._year_file_chache[year_str] = r.text
            except Exception as e:
                print(f"Error fetching file list for year {year_str}: {e}")
                return None, None
            
        # define date format
        date_pattern = date.strftime("%Y%m%d")

        # Regex search for the matching file
        pattern = re.compile(f'href="([^"]*{date_pattern}[^"]*\.zip)"')
        match = pattern.search(self._year_file_chache[year_str])

        if match:
            filename = match.group(1)
            return f"{self.base_url}/{year_str}/{filename}", filename
        
        return None, None
    
    def download_daily_chart(self, date):
        """Load chart for a specific date"""
        year = date.year

        # Find URL
        url, filename = self._get_file_url_from_index(year, date)

        if url is None:
            return None
        
        target_path = self.output_dir / str(year) / filename
        target_path.parent.mkdir(parents=True, exist_ok=True)

        if target_path.exists():
            return target_path
        
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(target_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            return target_path
        except Exception as e:
            if target_path.exists():
                target_path.unlink()
            return None
        
    def rasterize_zipfile(self, zip_path, reference_ds, date):
        """Opens the shapefile in the zip, rasterizes the Egg-Codes"""
        temp_dir = zip_path.parent / f"temp_{zip_path.stem}"
        temp_nc_path = zip_path.parent / f"rasterized_{zip_path.stem}.nc"

        # if already rasterized, load and return
        if temp_nc_path.exists():
            return temp_nc_path
        
        try:
            # ensure temp dir
            temp_dir.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

            # Find the shapefile (assuming only one .shp in the zip)
            shp_files = list(temp_dir.rglob("*.shp"))
            if not shp_files:
                return None
            
            try:
                gdf = gpd.read_file(shp_files[0])
            except Exception as e:
                print(f"Error reading shapefile {shp_files[0]}: {e}")
                return None
            
            cols_upper = {c.upper(): c for c in gdf.columns}

            # Reproject to target CRS
            target_crs = reference_ds.rio.crs
            if target_crs is None:
                target_crs = TARGET_CRS

            gdf = gdf.to_crs(target_crs)

            # cut to AOI
            minx, miny, maxx, maxy = reference_ds.rio.bounds()
            gdf = gdf.cx[minx:maxx, miny:maxy]

            # Prepare raster
            height = reference_ds.sizes['y']
            width = reference_ds.sizes['x']
            transform = reference_ds.rio.transform()

            dataset_dict = {}

            # Normalize Column Names
            cols_upper = {c.upper(): c for c in gdf.columns}

            for var in EGG_CODE_VARS:
                col_name = cols_upper.get(var)
                if not col_name:
                    continue

                # Convert to numbers, misstakes to -1
                values = pd.to_numeric(gdf[col_name], errors='coerce').fillna(np.nan)
                shapes = ((geom, val) for geom, val in zip(gdf.geometry, values))

                burned = features.rasterize(
                    shapes=shapes,
                    out_shape=(height, width),
                    transform=transform,
                    fill=np.nan,
                    dtype='float32'
                )

                dataset_dict[f"val_{var}"] = (("y", "x"), burned)

                del burned

            shutil.rmtree(temp_dir, ignore_errors=True)

            if not dataset_dict:
                return None
            
            # create dataset
            ds = xr.Dataset(dataset_dict, coords=reference_ds.coords)
            ds = ds.expand_dims(time = [pd.to_datetime(date)])

            # save array to netCDF
            encoding = {var: {"zlib": True, "complevel": 1} for var in ds.data_vars}
            ds.to_netcdf(temp_nc_path, format = "NETCDF4", encoding = encoding)
            ds.close()

            del ds
            del dataset_dict
            gc.collect()
            
            return temp_nc_path
        
        except Exception as e:
            print(f"Error rasterizing zipfile {zip_path}: {e}")
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            return None
        
    def create_validation_cube(self, data_cube, start_date, end_date):
        """Creates cube for the whole period"""
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        valid_nc_paths = []
        reference_grid = data_cube.isel(time=0).copy(deep = False)
        print(f"Creating validation cube for period {start_date} to {end_date}. Total days: {len(date_range)}")

        for date in date_range:
            zip_path = self.download_daily_chart(date)
            print(f"Processing date {date.date()}: {'Found' if zip_path else 'Not Found'}")

            if zip_path is None:
                continue

            nc_path = self.rasterize_zipfile(zip_path, reference_grid, date)

            if nc_path is not None:
                valid_nc_paths.append(nc_path)

        if not valid_nc_paths:
            print("No valid datasets found -> returning original cube")
            return data_cube
        
        # Load all valid datasets and concatenate
        val_cube = xr.open_mfdataset(valid_nc_paths, chunks = {"x": 128, "y": 128}, combine = 'nested', concat_dim = "time")
        val_cube = val_cube.sortby("time")

        # reindex to match data cube time
        val_cube_filled = val_cube.reindex(time=data_cube.time, method="nearest", tolerance=pd.Timedelta('7D'))
        print(f"Total valid daily datasets created: {len(valid_nc_paths)}. After reindexing: {val_cube_filled.sizes['time']} timestamps")

        # assign validation data on grid of data cube
        val_cube_filled = val_cube_filled.assign_coords({"x": data_cube.x, "y": data_cube.y})
       
        # merge cubes
        final_merged_cube = xr.merge([data_cube, val_cube_filled]) 

        # clean up
        del val_cube
        del val_cube_filled
        del valid_nc_paths
        gc.collect()

        return final_merged_cube