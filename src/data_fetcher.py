'This document uses the process api to get data from copernicus.'

# import necessary libraries
import openeo
import os
from config import AOI_arctic, TIME_PERIOD

class DataFetcher:
    def __init__(self):
        self.aoi = AOI_arctic
        self.time_period = TIME_PERIOD

        # Initialize the OpenEO connection
        self.connection = openeo.connect("https://openeo.dataspace.copernicus.eu")
        self.connection.authenticate_oidc()
        print("Authenticated with OpenEO Copernicus Dataspace.")

    def get_bbox(self):
        """The function to convert WKT polygon to bounding box dictionary."""
        coords = self.aoi.replace("POLYGON((", "").replace("))", "").split(", ")
        lons = [float(coord.split(" ")[0]) for coord in coords]
        lats = [float(coord.split(" ")[1]) for coord in coords]
        return {"west": min(lons), "south": min(lats), "east": max(lons), "north": max(lats)}
    
    def fetch_sar_data(self, start_date, end_date):
        """Function to fetch Sentinel-1 SAR data from Copernicus OpenEO."""
        # Get AOI and time period
        ArOfIn = self.get_bbox()
        time_period = [start_date, end_date]
        print(f"\n[SAR] Fetching data for time period: {start_date} to {end_date}")

        # Load the Sentinel-1 Data 
        s1_cube = self.connection.load_collection(
            "SENTINEL1_GRD",
            spatial_extent = ArOfIn,
            temporal_extent = time_period,
            bands = ["VV", "VH"]
        )

        # Add elevation model
        s1_cube = s1_cube.sar_backscatter(
            elevation_model = "COPERNICUS_30",
            coefficient = "sigma0-ellipsoid"
        )

        # resample to 40m resolution
        s1_cube = s1_cube.resample_spatial(
            resolution = 40,
            projection = "3995",
            method = "bilinear"
        )

        # Create a batch job instead of immediate download
        job = s1_cube.create_job(
            out_format = "GTIFF",
            title = f"SAR_{start_date[:7]}"
        )

        print(f"Job created with ID: {job.job_id} for Sentinel-1 SAR data.")
        job.start_and_wait(print = lambda *args: None)

        # save to a yearly folder
        year_folder = start_date[:4] #YYYY
        out_dir = f"data/GeoTIFF/SAR/{year_folder}/"
        os.makedirs(out_dir, exist_ok=True)

        job.get_results().download_files(out_dir)
        return out_dir
    
    def fetch_thermal_data(self, start_date, end_date):
        """Function to fetch Sentinel-3 Thermal data from Copernicus OpenEO."""
        # Get AOI and time period
        ArOfIn = self.get_bbox()
        time_period = [start_date, end_date]
        print(f"\n[Thermal] Fetching data for time period: {start_date} to {end_date}")

        # Load the Sentinel-3 Data 
        s3_cube = self.connection.load_collection(
            "SENTINEL3_SLSTR",
            spatial_extent = ArOfIn,
            temporal_extent = time_period,
            bands = ["S8", "S9"]
        )

        # resample to 40m resolution
        s3_cube = s3_cube.resample_spatial(
            resolution = 40,
            projection = "3995",
            method = "bilinear"
        )

        # Create a batch job instead of immediate download
        job = s3_cube.create_job(
            out_format = "GTIFF",
            title = f"Thermal_{start_date[:7]}",
            job_options = {
                "executor-memory": "16G",
                "executor-memoryOverhead": "4G"
            }
        )

        print(f"Job created with ID: {job.job_id}")
        job.start_and_wait(print = lambda *args: None)

        # save to a yearly folder
        year_folder = start_date[:4] #YYYY
        out_dir = f"data/GeoTIFF/Thermal/{year_folder}/"
        os.makedirs(out_dir, exist_ok=True)

        job.get_results().download_files(out_dir)
        return out_dir

         

if __name__ == "__main__":
    fetscher = DataFetcher()


