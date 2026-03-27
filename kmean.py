import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings

# Final English-only Unsupervised Validation Script
warnings.filterwarnings("ignore")

def run_final_kmeans_4_groups(nc_path, out_png="final_kmeans_4_clusters.png"):
    print("=== STARTING FINAL K-MEANS VALIDATION (K=4) ===")
    
    # Load dataset
    ds = xr.open_dataset(nc_path, chunks={'time': 10})
    
    # 1. Find the best day for SAR/Thermal overlap
    print("Scanning for the best day...")
    sar_var = "SAR"
    therm_var = "Thermal"
    
    sar_check = ds[sar_var].isel(band=0) if "band" in ds[sar_var].dims else ds[sar_var]
    valid_mask_all = sar_check.notnull() & ds[therm_var].notnull()
    valid_counts = valid_mask_all.sum(dim=["x", "y"]).compute()
    best_idx = int(valid_counts.argmax())
    
    day_ds = ds.isel(time=best_idx).compute()
    date_str = pd.to_datetime(day_ds.time.values).strftime('%Y-%m-%d')
    print(f" -> Selected Day: {date_str}")

    # 2. Prepare Data (Squeeze dimensions)
    sar = np.squeeze(day_ds[sar_var].isel(band=0).values if "band" in day_ds[sar_var].dims else day_ds[sar_var].values)
    therm = np.squeeze(day_ds[therm_var].values)
    
    # Mask NaNs
    mask = ~np.isnan(sar) & ~np.isnan(therm)
    features = np.column_stack([sar[mask], therm[mask]])
    
    # 3. Scale Features (Crucial: Normalizes dB and Kelvin)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # 4. K-Means Clustering
    print("Clustering into 4 physical groups...")
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_scaled)

    # Map back to image
    kmeans_image = np.full(sar.shape, np.nan)
    kmeans_image[mask] = labels

    # 5. Plotting (3 Columns: SAR | Thermal | K-Means)
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    fig.patch.set_facecolor('white')

    # SAR Plot
    vmin_s, vmax_s = np.nanpercentile(sar, [2, 98])
    im0 = axes[0].imshow(sar, cmap='gray', vmin=vmin_s, vmax=vmax_s)
    axes[0].set_title(f"A: Sentinel-1 Radar (SAR)\nSurface Texture", fontsize=16, fontweight='bold')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04).set_label('Backscatter [dB]')
    axes[0].axis('off')

    # Thermal Plot
    vmin_t, vmax_t = np.nanpercentile(therm, [2, 98])
    im1 = axes[1].imshow(therm, cmap='inferno', vmin=vmin_t, vmax=vmax_t)
    axes[1].set_title(f"B: Sentinel-3 Thermal\nSurface Temperature", fontsize=16, fontweight='bold')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04).set_label('Temperature [Kelvin]')
    axes[1].axis('off')

    # K-Means Plot
    # Discrete colormap for 4 clusters
    im2 = axes[2].imshow(kmeans_image, cmap='viridis') 
    axes[2].set_title(f"C: K-Means Clustering (K=4)\nAutomated Data Fusion", fontsize=16, fontweight='bold')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04).set_label('Cluster ID (0-3)')
    axes[2].axis('off')

    plt.suptitle(f"Arctic DataCube: Unsupervised Feature Validation ({date_str})", 
                 fontsize=22, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f"=== SUCCESS! Final image saved: {out_png} ===")

if __name__ == "__main__":
    CUBE_PATH = "data/NetCDF/arctic_datacube_full_period.nc" 
    run_final_kmeans_4_groups(CUBE_PATH)