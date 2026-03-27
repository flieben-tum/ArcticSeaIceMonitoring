import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

def plot_four_valid_spatial_frames(nc_path, out_png="four_frames_showcase.png"):
    print("=== SUCHE 4 PERFEKTE TAGE (MIT SAR, THERMAL UND EIS-ACTION) ===")
    
    ds = xr.open_dataset(nc_path, chunks={'time': 10})
    total_days = len(ds.time)
    
    target_dates = []
    
    # Wir scannen jetzt viel feinmaschiger (prüft bis zu 150 Tage, um garantierte Treffer zu finden)
    step = max(1, total_days // 150) 
    
    print("Scanne Cube nach Eiskanten (garantierter Mix aus Wasser und Eis)...")
    for i in range(0, total_days, step):
        day_ds = ds.isel(time=i).compute()
        
        if "SAR" in day_ds and "Thermal" in day_ds and "val_CT" in day_ds:
            sar_data = np.squeeze(day_ds["SAR"].isel(band=0).values if "band" in day_ds["SAR"].dims else day_ds["SAR"].values)
            therm_data = np.squeeze(day_ds["Thermal"].values)
            val_data = np.squeeze(day_ds["val_CT"].values)
            
            valid_mask = ~np.isnan(sar_data) & ~np.isnan(therm_data) & ~np.isnan(val_data)
            
            if valid_mask.sum() > 5000:
                valid_vals = val_data[valid_mask]
                
                # Zwinge Werte für den Check auf eine 0-1 Skala (falls sie z.B. 0-100 sind)
                if np.nanmax(valid_vals) > 1.5:
                    valid_vals = valid_vals / 100.0
                
                # DER KUGELSICHERE FILTER: Wir wollen an diesem Tag mindestens 10% klares Wasser (<0.2) 
                # UND mindestens 10% klares Eis (>0.8) sehen. Das garantiert eine wunderschöne Eiskante!
                water_ratio = np.sum(valid_vals < 0.2) / len(valid_vals)
                ice_ratio = np.sum(valid_vals > 0.8) / len(valid_vals)
                
                if water_ratio > 0.10 and ice_ratio > 0.10:
                    date_str = pd.to_datetime(day_ds.time.values).strftime('%Y-%m-%d')
                    target_dates.append(date_str)
                    print(f" -> Treffer! Tag {date_str} ist perfekt (Eiskante gefunden!).")
                
        if len(target_dates) == 4:
            break
            
    if len(target_dates) < 4:
        print(f"\nWarnung: Habe nur {len(target_dates)} perfekte Tage gefunden. Render diese trotzdem...")
        if len(target_dates) == 0:
            print("Kein Tag erfüllte die harten Kriterien. Der Filter ist vielleicht zu streng.")
            return

    print(f"\n=== GENERIERE POSTER FÜR DIESE TAGE ===")
    
    fig, axes = plt.subplots(len(target_dates), 3, figsize=(22, 5 * len(target_dates)))
    fig.patch.set_facecolor('white')
    
    if len(target_dates) == 1:
        axes = np.array([axes])

    for i, target_date in enumerate(target_dates):
        print(f"Zeichne {target_date}...")
        
        day_ds = ds.sel(time=target_date).compute()
        
        sar_img = np.squeeze(day_ds["SAR"].isel(band=0).values if "band" in day_ds["SAR"].dims else day_ds["SAR"].values)
        therm_img = np.squeeze(day_ds["Thermal"].values)
        val_img = np.squeeze(day_ds["val_CT"].values)

        # Normiere Ground Truth für das finale Bild zwingend auf 0.0 bis 1.0!
        if np.nanmax(val_img) > 1.5:
            val_img = val_img / 100.0

        # --- 1. SAR (Links) ---
        ax_sar = axes[i, 0]
        vmin_s, vmax_s = np.nanpercentile(sar_img, [2, 98]) if not np.isnan(sar_img).all() else (-30, 0)
        im_s = ax_sar.imshow(sar_img, cmap='gray', vmin=vmin_s, vmax=vmax_s)
        cb_s = plt.colorbar(im_s, ax=ax_sar, fraction=0.046, pad=0.04)
        cb_s.set_label('Rückstreuung (dB)', fontsize=12)
        ax_sar.set_title(f"Radar (SAR) - {target_date}", fontsize=16, fontweight='bold')
        ax_sar.axis('off')

        # --- 2. THERMAL (Mitte) ---
        ax_therm = axes[i, 1]
        vmin_t, vmax_t = np.nanpercentile(therm_img, [2, 98]) if not np.isnan(therm_img).all() else (240, 273)
        im_t = ax_therm.imshow(therm_img, cmap='inferno', vmin=vmin_t, vmax=vmax_t)
        cb_t = plt.colorbar(im_t, ax=ax_therm, fraction=0.046, pad=0.04)
        cb_t.set_label('Temperatur (Kelvin)', fontsize=12)
        ax_therm.set_title(f"Thermal - {target_date}", fontsize=16, fontweight='bold')
        ax_therm.axis('off')

        # --- 3. GROUND TRUTH (Rechts) ---
        ax_val = axes[i, 2]
        vmin_v, vmax_v = 0.0, 1.0 # Jetzt klappt die feste Skala von 0 bis 1!
        im_v = ax_val.imshow(val_img, cmap='viridis', vmin=vmin_v, vmax=vmax_v)
        cb_v = plt.colorbar(im_v, ax=ax_val, fraction=0.046, pad=0.04)
        cb_v.set_label('Egg Code (val_CT)', fontsize=12)
        ax_val.set_title(f"Ground Truth - {target_date}", fontsize=16, fontweight='bold')
        ax_val.axis('off')

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n=== FERTIG! Perfektes Poster mit Eiskanten gespeichert unter: {out_png} ===")

if __name__ == "__main__":
    CUBE_PATH = "data/NetCDF/arctic_datacube_full_period.nc" 
    plot_four_valid_spatial_frames(CUBE_PATH)