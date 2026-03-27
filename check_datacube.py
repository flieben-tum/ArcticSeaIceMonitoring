import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

def plot_top_10_best_days(nc_path, out_png="best_10_days_showcase.png", min_gap_days=14):
    print("=== STARTE SMARTEN SHOWCASE (Die 10 besten Tage) ===")
    print(f"Lese Cube virtuell ein: {nc_path}")
    
    ds = xr.open_dataset(nc_path, chunks={'time': 10})
    
    # 1. SAR und Thermal referenzieren
    if "SAR" in ds:
        sar_da = ds["SAR"].isel(band=0) if "band" in ds["SAR"].dims else ds["SAR"]
    else:
        print("Fehler: Kein SAR im Cube!")
        return
        
    if "Thermal" not in ds:
        print("Fehler: Kein Thermal im Cube!")
        return
    therm_da = ds["Thermal"]

    # 2. Scanne nach den perfekten Tagen (Das kann jetzt 1-2 Minuten dauern, die Festplatte arbeitet!)
    print("Scanne 4 Jahre Daten nach Tagen mit den meisten GEMEINSAMEN (SAR + Thermal) Pixeln...")
    print("Bitte kurz warten, Dask rechnet...")
    
    valid_mask = sar_da.notnull() & therm_da.notnull()
    valid_counts = valid_mask.sum(dim=["x", "y"]).compute() # Hier passiert die Magie!
    
    # 3. In einen Pandas DataFrame packen zum einfachen Filtern
    df_counts = pd.DataFrame({
        'date': pd.to_datetime(ds.time.values),
        'count': valid_counts.values,
        'index': np.arange(len(ds.time))
    })
    
    # Sortieren: Die Tage mit den absolut meisten Pixeln nach ganz oben
    df_counts = df_counts.sort_values('count', ascending=False)
    
    # 4. Top 10 auswählen (mit Mindestabstand!)
    selected_indices = []
    selected_dates = []
    
    for _, row in df_counts.iterrows():
        if row['count'] == 0:
            continue
            
        current_date = row['date']
        
        # Prüfen, ob der Tag zu nah an einem bereits gewählten Tag liegt
        too_close = False
        for sel_date in selected_dates:
            if abs((current_date - sel_date).days) < min_gap_days:
                too_close = True
                break
                
        if not too_close:
            selected_indices.append(int(row['index']))
            selected_dates.append(current_date)
            
        if len(selected_indices) == 10:
            break
            
    if len(selected_indices) < 10:
        print(f"Warnung: Nur {len(selected_indices)} gute Tage mit Abstand gefunden!")
        
    # Chronologisch sortieren für das Finale Bild
    selected_indices.sort()
    
    print("\nFolgende Top-Tage wurden für das Showcase ausgewählt:")
    for idx in selected_indices:
        print(f" - {pd.to_datetime(ds.time.values[idx]).strftime('%Y-%m-%d')} (Gültige Pixel: {valid_counts.values[idx]})")

    # ==========================================
    # VISUALISIERUNG (Das Poster)
    # ==========================================
    fig, axes = plt.subplots(4, 5, figsize=(25, 20))
    fig.patch.set_facecolor('white')
    
    print("\nGeneriere das finale Bild...")
    for i, idx_in_time in enumerate(selected_indices):
        day_ds = ds.isel(time=idx_in_time).compute()
        date_str = pd.to_datetime(day_ds.time.values).strftime('%Y-%m-%d')
        
        sar_data = day_ds["SAR"].isel(band=0).values if "band" in day_ds["SAR"].dims else day_ds["SAR"].values
        therm_data = day_ds["Thermal"].values

        if i < 5:
            ax_sar = axes[0, i]
            ax_therm = axes[1, i]
        else:
            ax_sar = axes[2, i - 5]
            ax_therm = axes[3, i - 5]
            
        # --- SAR PLOTTEN ---
        vmin_s, vmax_s = np.nanpercentile(sar_data, [2, 98])
        ax_sar.imshow(sar_data, cmap='gray', vmin=vmin_s, vmax=vmax_s)
        ax_sar.set_title(f"SAR - {date_str}", fontsize=14, fontweight='bold')
        ax_sar.axis('off')

        # --- THERMAL PLOTTEN ---
        vmin_t, vmax_t = np.nanpercentile(therm_data, [2, 98])
        im_t = ax_therm.imshow(therm_data, cmap='inferno', vmin=vmin_t, vmax=vmax_t)
        plt.colorbar(im_t, ax=ax_therm, fraction=0.046, pad=0.04).set_label('Kelvin', fontsize=10)
        ax_therm.set_title(f"Thermal - {date_str}", fontsize=14, fontweight='bold')
        ax_therm.axis('off')

    # Leere Subplots ausblenden, falls wir weniger als 10 Tage gefunden haben
    for i in range(len(selected_indices), 10):
        if i < 5:
            axes[0, i].axis('off')
            axes[1, i].axis('off')
        else:
            axes[2, i - 5].axis('off')
            axes[3, i - 5].axis('off')

    plt.suptitle("Die 10 stärksten Aufnahmen im DataCube (SAR + Thermal)", 
                 fontsize=24, fontweight='bold', y=0.95)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n=== PERFEKT! Übersichtsbild gespeichert unter: {out_png} ===")

if __name__ == "__main__":
    CUBE_PATH = "data/NetCDF/arctic_datacube_full_period.nc" 
    plot_top_10_best_days(CUBE_PATH, min_gap_days=14)