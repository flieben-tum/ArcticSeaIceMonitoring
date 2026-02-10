'This document uses the process api to get data from copernicus.'

# import necessary libraries
import openeo
from config import AOI_muc, AOI_arctic, TIME_PERIOD

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
    
    def fetch_sar_data(self):
        """Function to fetch Sentinel-1 SAR data from Copernicus OpenEO."""
        # Get AOI and time period
        ArOfIn = self.get_bbox()
        print(f"Fetching data for bounding box: {ArOfIn} and time period: {self.time_period}")

        # Load the Sentinel-1 Data 
        s1_cube = self.connection.load_collection(
            "SENTINEL1_GRD",
            spatial_extent = ArOfIn,
            temporal_extent = self.time_period,
            bands = ["HH", "HV"]
        )

        # Add elevation model
        s1_cube = s1_cube.sar_backscatter(
            elevation_model = "COPERNICUS_30",
            coefficient = "sigma0-ellipsoid",
        )

        # Create a batch job instead of immediate download
        job = s1_cube.create_job(
            out_format = "GTIFF",
            title = "Sentinel-1 Data Fetch Job"
        )

        print(f"Job created with ID: {job.job_id}")
        job.start_and_wait(print = lambda *args: None)

        # Download the result
        result = job.get_results()
        result.download_files("data/GeoTIFF/SAR/")

        return "data/GeoTIFF/SAR/"
    
    def fetch_thermal_data(self):
        """Function to fetch Sentinel-3 Thermal data from Copernicus OpenEO."""
        # Get AOI and time period
        ArOfIn = self.get_bbox()
        print(f"Fetching data for bounding box: {ArOfIn} and time period: {self.time_period}")

        # Load the Sentinel-3 Data 
        s3_cube = self.connection.load_collection(
            "SENTINEL3_SLSTR",
            spatial_extent = ArOfIn,
            temporal_extent = self.time_period,
            bands = ["S8", "S9"]
        )

        # Create a batch job instead of immediate download
        job = s3_cube.create_job(out_format = "GTIFF", title = "Sentinel-3 Data Fetch Job")
        print(f"Job created with ID: {job.job_id}")
        job.start_and_wait(print = lambda *args: None)

        # Download the result
        job.get_results().download_files("data/GeoTIFF/Thermal/")
        return "data/GeoTIFF/Thermal/"

         

if __name__ == "__main__":
    fetscher = DataFetcher()


