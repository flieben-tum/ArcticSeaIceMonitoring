'This is the main script'

#import necessary libraries
import os
import gc
import shutil
import pandas as pd
import xarray as xr

# import necessary modules
from config import TIME_PERIOD, TARGET_CRS
from src.data_fetcher import DataFetcher
from src.datacube import ArcticDataCube
from src.data_fetcher_validation import ValidationDataFetcher

def main():
    # config
    output_dir = "data/NetCDF/"

    print("=== Starting Arctic Data Cube Pipeline ===")

    # setup dates
    t_start = pd.to_datetime(TIME_PERIOD[0])
    t_end = pd.to_datetime(TIME_PERIOD[1])
    """
    # start data fetching data in a loop for every month in the time period
    fetcher = DataFetcher()
    months = pd.date_range(start = t_start.replace(day=1), end = t_end, freq='MS')
    
    for current_month in months:
        # create YYYY-MM-DD strings for each month
        month_start = current_month.strftime("%Y-%m-%d")
        month_end = (current_month + pd.offsets.MonthEnd(1)).strftime("%Y-%m-%d")

        # fetch SAR data for this month
        try:
            fetcher.fetch_sar_data(start_date = month_start, end_date = month_end)
            fetcher.fetch_thermal_data(start_date = month_start, end_date = month_end)
        except Exception as e:
            print(f"Error fetching data for {month_start} to {month_end}: {e}")
            continue

    sar_path = "data/GeoTIFF/SAR/"
    
    # check if data download was successful
    if not os.path.exists(sar_path) or not any(os.scandir(sar_path)):
        print("No SAR data downloaded. Exiting.")
        return
    
    print(("\n=== Start: Data Cube Fusion ==="))

    # create cubes
    cube = ArcticDataCube()
    validation_fetcher = ValidationDataFetcher()

    start_year = t_start.year
    end_year = t_end.year

    # year by year loop to reduce memory usage
    for year in range(start_year, end_year + 1):
        print(f"\n=== Processing Year: {year} ===")

        # load data for this year
        sar_year = cube.load_sar_data(year = year)
        sar_year = sar_year.chunk({"time": -1, "x": 128, "y": 128})
        sar_year = cube.denoise_sar(sar_cube = sar_year)
        thermal_year = cube.load_thermal_data(target_sar_cube = sar_year, year = year)
        thermal_year = thermal_year.chunk({"time": -1, "x": 128, "y": 128})
        print(f"Loaded SAR and Thermal data for year {year}. SAR timestamps: {len(sar_year.time)}, Thermal timestamps: {len(thermal_year.time)}")

        # output filing
        output_filename = f"arctic_datacube_{year}.nc"
        os.makedirs(output_dir, exist_ok=True)
        full_path = os.path.join(output_dir, output_filename)
        print(f"Output path for year {year}: {full_path}")
        checkpoint_path = os.path.join(output_dir, f"checkpoint_{year}.nc")

        # fuse data cube
        final_dataset = cube.create_fused_dataset(sar_cube = sar_year, thermal_cube = thermal_year)
        print(f"Fused dataset created for year {year}. Dimensions: {final_dataset.dims}, Variables: {list(final_dataset.data_vars)}")
        for var in final_dataset.variables:
            final_dataset[var].encoding.clear()
            print(f"Cleared encoding for variable: {var}")
        if os.path.exists(full_path):
            print(f"Output file {full_path} already exists. It will be overwritten.")
            os.remove(full_path)
        final_dataset = final_dataset.compute()
        print(f"Dataset computed for year {year}.")
        final_dataset.to_netcdf(checkpoint_path)
        print(f"Checkpoint saved for year {year} at {checkpoint_path}")
        # clean up memory before trend calculation
        del sar_year
        del thermal_year
        del final_dataset
        gc.collect()

        # load the clean dataset from checkpoint for validation
        final_dataset = xr.open_dataset(checkpoint_path, chunks={"time": -1, "x": 128, "y": 128})

        # add validation data
        year_start = max(pd.to_datetime(f"{year}-01-01"), t_start)
        year_end = min(pd.to_datetime(f"{year}-12-31"), t_end)
        final_validation_cube = validation_fetcher.create_validation_cube(data_cube = final_dataset, start_date = year_start, end_date = year_end)
        print(f"Validation cube created for year {year}. Dimensions: {final_validation_cube.dims}, Variables: {list(final_validation_cube.data_vars)}")

        # clean up before final export
        for var in final_validation_cube.variables:
            final_validation_cube[var].encoding.clear()
            print(f"Cleared encoding for variable: {var}")
        del final_dataset
        gc.collect()
        print(f"Cleaned up intermediate datasets for year {year} before final export.")

        # compute and save final dataset
        final_validation_cube = final_validation_cube.compute()
        final_validation_cube.to_netcdf(full_path)
        print(f"Saved fused dataset for year {year} to {full_path}")

        # short protocol output
        epsg_code = TARGET_CRS.split(":")[-1]
        print("\n" + "-"*40)
        print(f"    Pipeline successful for year {year}")
        print("-"*40)
        
        # clean up memory
        del final_validation_cube
        gc.collect()

        if os.path.exists(checkpoint_path):
            shutil.rmtree(checkpoint_path, ignore_errors=True)

    print("\n" + "="*50)
    print("=== ALL YEARS PROCESSED SUCCESSFULLY ===")
    print("="*50 + "\n")
    """

    start_year = t_start.year
    end_year = t_end.year

    # merge yearly files into one cube for the whole period
    print("Merging yearly files into one cube for the whole period...")
    nc_files = [os.path.join(output_dir, f"arctic_datacube_{year}.nc") for year in range(start_year, end_year + 1)]
    valid_files = [f for f in nc_files if os.path.exists(f)]

    master_path = os.path.join(output_dir, "arctic_datacube_full_period.nc")

    if len(valid_files) > 1:
        master_cube = xr.open_mfdataset(valid_files, combine = 'nested', concat_dim = "time", chunks = {"time": 10})
        print(f"Master cube created for full period. Dimensions: {master_cube.dims}, Variables: {list(master_cube.data_vars)}. Writing to {master_path}...")
        master_cube.to_netcdf(master_path)
        print(f"Master cube saved to {master_path}")

        # clean up
        del master_cube
        gc.collect()

    else:
        print("Not enough valid yearly files to merge into master cube. Please check the output directory.")

if __name__ == "__main__":
    main()
    