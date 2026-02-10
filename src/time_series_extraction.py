# import necessary libraries
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# mute generic warnings
import warnings
warnings.filterwarnings("ignore")

def drill_pixel(nc_filename="arctic_datacube_final.nc", x_coord=None, y_coord=None):
    """
    Extracts and visualizes the time series for a specific pixel from the NetCDF datacube.
    Plots Thermal data (Temperature) vs. SAR data (Backscatter Intensity).
    """

    # 1. Construct path relative to project root
    base_dir = Path(__file__).parent.parent 
    file_path = base_dir / "data" / "NetCDF" / nc_filename

    print(f"📂 Loading Datacube: {file_path} ...")
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        print("Tip: Did you run main.py to generate the .nc file?")
        return

    # Open the dataset
    ds = xr.open_dataset(file_path)

    # 2. Select Pixel Coordinates (Default: Center of the image)
    if x_coord is None:
        x_idx = ds.sizes['x'] // 2
        y_idx = ds.sizes['y'] // 2
    else:
        x_idx = int(x_coord)
        y_idx = int(y_coord)

    print(f"📍 Drilling pixel at position: X={x_idx}, Y={y_idx}")

    # 3. Extract Data for this Pixel
    # .isel selects by index (0 to width/height)
    point_data = ds.isel(x=x_idx, y=y_idx)
    times = point_data.time.values
    
    # --- Thermal Processing ---
    # Access variable 'Thermal' (case-sensitive!)
    thermal_vals = point_data.Thermal.values
    
    # Check if Kelvin (>200) and convert to Celsius
    if np.nanmean(thermal_vals) > 200:
        thermal_vals = thermal_vals - 273.15
        temp_unit = "°C"
    else:
        temp_unit = "°C"

    # --- SAR Processing ---
    # Access variable 'SAR' (case-sensitive!)
    sar_raw = point_data.SAR.values
    
    # Convert linear intensity to Decibels (dB) for plotting
    # Formula: 10 * log10(intensity). We use np.maximum to avoid log(0) errors.
    sar_vals_db = 10 * np.log10(np.maximum(sar_raw, 1e-10))
    
    # Get band names (HH, HV) if they exist
    sar_bands = point_data.band.values if 'band' in point_data.dims else ['Avg']

    # ---------------------------------------------------------
    # PLOTTING
    # ---------------------------------------------------------
    fig = plt.figure(figsize=(14, 6))
    
    # --- Left Panel: Locator Map ---
    ax_map = plt.subplot2grid((1, 3), (0, 0))
    
    # Create a background image (Temporal Mean)
    bg_img = ds.SAR.mean(dim='time')
    
    # FIX: If we have multiple bands (HH/HV), average them to get a 2D image for imshow
    if 'band' in bg_img.dims:
        bg_img = bg_img.mean(dim='band')
        
    # Convert background to dB for better contrast visualization
    bg_log = 10 * np.log10(bg_img.where(bg_img > 0))
    
    # Plot map
    ax_map.imshow(bg_log, cmap='gray', aspect='auto', vmin=-25, vmax=0)
    ax_map.plot(x_idx, y_idx, 'rx', markersize=12, markeredgewidth=2) # Red X marker
    ax_map.set_title(f"Pixel Location\n(X: {x_idx}, Y: {y_idx})")
    ax_map.axis('off')

    # --- Right Panel: Time Series Graph ---
    ax_temp = plt.subplot2grid((1, 3), (0, 1), colspan=2)
    
    # Curve 1: Thermal (Red)
    color_temp = 'tab:red'
    ax_temp.set_xlabel('Time')
    ax_temp.set_ylabel(f'Temperature ({temp_unit})', color=color_temp, fontweight='bold')
    
    line_t = ax_temp.plot(times, thermal_vals, color=color_temp, marker='o', 
                          linestyle='-', linewidth=2, label='Thermal')
    
    ax_temp.tick_params(axis='y', labelcolor=color_temp)
    ax_temp.grid(True, linestyle='--', alpha=0.5)

    # Curve 2: SAR (Blue) on secondary Y-axis
    ax_sar = ax_temp.twinx()
    ax_sar.set_ylabel('SAR Backscatter (dB)', color='tab:blue', fontweight='bold')
    
    # Loop over bands to plot distinct lines for HH and HV
    sar_lines = []
    styles = [('--', 'x'), (':', 'v')] # Different styles for bands
    
    # Check if data is 2D (Time, Band) or 1D (Time)
    if sar_raw.ndim > 1:
        for i, band_name in enumerate(sar_bands):
            style = styles[i % len(styles)]
            l = ax_sar.plot(times, sar_vals_db[:, i], color='tab:blue', 
                            linestyle=style[0], marker=style[1], 
                            label=f'SAR {band_name}')
            sar_lines.extend(l)
    else:
        # Fallback for single band
        l = ax_sar.plot(times, sar_vals_db, color='tab:blue', 
                        linestyle='--', marker='x', label='SAR')
        sar_lines.extend(l)

    ax_sar.tick_params(axis='y', labelcolor='tab:blue')
    
    # Set SAR Y-limits to typical range (Water is ~ -25dB, Ice/Land ~ -5dB)
    ax_sar.set_ylim(-45, 5)

    plt.title("Time Series Analysis: Radar vs. Temperature")
    fig.autofmt_xdate() # Rotate date labels

    # Combine legends
    lines = line_t + sar_lines
    labels = [l.get_label() for l in lines]
    ax_temp.legend(lines, labels, loc='upper left')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Test run (Center of the image)
    drill_pixel(x_coord=400, y_coord=400)
    
    # Example for testing specific coordinates (e.g., aiming for water/ice):
    # drill_pixel(x_coord=1200, y_coord=400)