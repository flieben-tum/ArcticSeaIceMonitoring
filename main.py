'This is the main script'

#import necessary libraries
import os

# import necessary modules
from src.data_fetcher import DataFetcher
from src.datacube import ArcticDataCube

def main():
    """print("=== Starting Arctic Data Cube Pipeline ===")

    # start data fetching
    fetcher = DataFetcher()
    sar_path = fetcher.fetch_sar_data()
    thermal_path = fetcher.fetch_thermal_data()
    
    # check if data download was successful
    if not os.listdir(sar_path):
        print("No SAR data downloaded. Exiting.")
        return
    """
    print(("\n=== Start: Data Cube Fusion ==="))

    # create data cube
    cube = ArcticDataCube()

    # load SAR data cube
    sar_data = cube.load_sar_data()
    sar_data = cube.denoise_sar(sar_cube = sar_data)
    if sar_data is None:
        print("Failed to create SAR data cube.")
        return
    
    # load Thermal data cube and resample to SAR grid
    thermal_data = cube.load_thermal_data(target_sar_cube = sar_data)
    if thermal_data is None:
        print("Failed to create Thermal data cube.")
        return
    
    # fuse data cubes
    print("\n=== Finalizing Data Cube ===")
    final_dataset = cube.create_fused_dataset(sar_cube = sar_data, thermal_cube = thermal_data)
    final_dataset = cube.calculate_trends(final_dataset, variable = "Thermal", days_list = [3,7])

    # save final dataset
    output_filename = "arctic_datacube_final.nc"
    output_dir = "data/NetCDF/"
    full_path = os.path.join(output_dir, output_filename)
    final_dataset.to_netcdf(full_path)
    
    # Final Protocol Output
    print("\n" + "="*40)
    print("       Pipeline Successful")
    print("="*40)

    # extract CRS info
    epsg_code = sar_data.rio.crs.to_epsg()

    print(f"projection:    EPSG:{epsg_code}")
    print(f"Grid Size:     {sar_data.sizes['x']} x {sar_data.sizes['y']}")
    print("-"*40)
    print(f"SAR Data:      {sar_data.sizes['time']} Timestamps")
    print(f"Thermal Data:  {thermal_data.sizes['time']} Timestamps")
    print("="*40 + "\n")
        

if __name__ == "__main__":
    main()
    