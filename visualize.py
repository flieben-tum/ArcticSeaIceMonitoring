import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import warnings

# Warnungen unterdrücken
warnings.filterwarnings("ignore")

def visualize_event(target_date_str, nc_filename="arctic_datacube_final.nc"):
    script_dir = Path(__file__).parent.resolve()
    base_dir = script_dir.parent if script_dir.name == "src" else script_dir
    input_path = base_dir / "data" / "NetCDF" / nc_filename

    if not input_path.exists():
        print(f"❌ Datei fehlt: {input_path}")
        return

    print(f"📂 Lade Datacube: {input_path} ...")
    ds = xr.open_dataset(input_path)
    
    # Bänder reduzieren
    if 'band' in ds.dims:
        ds = ds.mean(dim='band', keep_attrs=True)

    try:
        target_date = pd.to_datetime(target_date_str)
    except:
        print("❌ Ungültiges Datum.")
        return

    sar_var = 'SAR_Raw' if 'SAR_Raw' in ds else 'SAR'
    therm_var = 'Thermal_Raw' if 'Thermal_Raw' in ds else 'Thermal'

    # --- DATEN HOLEN ---
    try:
        selection = ds.sel(time=target_date, method='nearest')
        actual_time = str(selection.time.values)[:10]
        print(f"📅 Zeige Daten für: {actual_time}")
    except KeyError:
        print("❌ Datum außerhalb des Zeitraums.")
        return

    # --- HELPER: Skalierung ---
    def prepare_data(da, name_hint=""):
        vals = da.values.copy()
        
        # SAR Log-Skalierung
        if "SAR" in name_hint and np.nanmin(vals) >= 0:
            vals = 10 * np.log10(np.maximum(vals, 1e-5))
            
        # Thermal Kelvin -> Celsius
        elif "Thermal" in name_hint and np.nanmean(vals) > 200:
            vals = vals - 273.15
            
        da_new = da.copy(data=vals)
        return da_new.squeeze()

    print("⚙️  Bereite Plots vor...")
    
    # 1. Daten aufbereiten
    sar_mean = prepare_data(ds[sar_var].mean(dim='time', keep_attrs=True), "SAR Mean")
    therm_mean = prepare_data(ds[therm_var].mean(dim='time', keep_attrs=True), "Thermal Mean")
    
    sar_event = prepare_data(selection['SAR'], "SAR Event")
    therm_event = prepare_data(selection['Thermal'], "Thermal Event")

    # 2. DYNAMISCHE SKALIERUNG BERECHNEN 🎨
    # Wir berechnen Min/Max für jedes Bild separat!
    
    # SAR Skala (Baseline vs Event kann gleich bleiben für Vergleichbarkeit)
    s_vmin, s_vmax = np.nanpercentile(sar_mean.values, 2), np.nanpercentile(sar_mean.values, 98)
    
    # Thermal Baseline Skala
    tm_vals = therm_mean.values
    tm_min, tm_max = np.nanpercentile(tm_vals, 2), np.nanpercentile(tm_vals, 98)

    # Thermal EVENT Skala (Das ist der Fix!)
    # Wir schauen uns NUR das aktuelle Bild an.
    te_vals = therm_event.values
    if np.all(np.isnan(te_vals)):
        te_min, te_max = -30, 0 # Fallback falls wirklich leer
    else:
        te_min = np.nanpercentile(te_vals, 2)
        te_max = np.nanpercentile(te_vals, 98)
        
        # Kontrast-Sicherung: Wenn alles genau -28.0 Grad ist, stürzt der Plotter ab.
        # Wir erzwingen mindestens 2 Grad Unterschied für die Farbskala.
        if te_max - te_min < 2.0:
            te_max = te_min + 2.0

    print(f"🎨 Skalierung für Event: {te_min:.2f}°C bis {te_max:.2f}°C")

    # --- PLOTTING ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Oben Links: SAR Mean
    sar_mean.plot.imshow(ax=axes[0,0], cmap='gray', vmin=s_vmin, vmax=s_vmax, cbar_kwargs={'label': 'dB'})
    axes[0,0].set_title("SAR Baseline (Mean)")

    # Oben Rechts: SAR Event
    sar_event.plot.imshow(ax=axes[0,1], cmap='gray', vmin=s_vmin, vmax=s_vmax, cbar_kwargs={'label': 'dB'})
    axes[0,1].set_title(f"SAR Event: {actual_time}")

    # Unten Links: Thermal Mean (Eigene Skala)
    therm_mean.plot.imshow(ax=axes[1,0], cmap='inferno', vmin=tm_min, vmax=tm_max, cbar_kwargs={'label': '°C'})
    axes[1,0].set_title(f"Thermal Baseline (Avg: {tm_min:.1f} bis {tm_max:.1f}°C)")

    # Unten Rechts: Thermal Event (Eigene Dynamische Skala!)
    therm_event.plot.imshow(ax=axes[1,1], cmap='inferno', vmin=te_min, vmax=te_max, cbar_kwargs={'label': '°C'})
    axes[1,1].set_title(f"Thermal Event: {actual_time}\n(Scale: {te_min:.1f} bis {te_max:.1f}°C)")

    plt.tight_layout()
    out = base_dir / f"analysis_{actual_time}.png"
    plt.savefig(out)
    print(f"✅ Bild gespeichert: {out}")
    plt.show()

if __name__ == "__main__":
    visualize_event("2025-02-27")