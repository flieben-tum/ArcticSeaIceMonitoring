import os
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

def main():
    print("=== Start Advanced Plotting Dashboard ===")
    
    # Pfad zum DataCube
    cube_path = "data/NetCDF/arctic_datacube_full_period.nc"
    
    if not os.path.exists(cube_path):
        print(f"Fehler: Datei {cube_path} nicht gefunden!")
        return

    # Ordner für Plots
    plot_dir = "plots/"
    os.makedirs(plot_dir, exist_ok=True)

    print(f"Lade DataCube: {cube_path}")
    ds = xr.open_dataset(cube_path)

    # ---------------------------------------------------------
    # 1. TAG AUSWÄHLEN
    # ---------------------------------------------------------
    target_date = "2020-03-15" 
    try:
        day_data = ds.sel(time=target_date, method="nearest")
        date_str = str(day_data.time.values)[:10]
        print(f"Daten für den {date_str} erfolgreich extrahiert.")
    except Exception as e:
        print("Datum nicht gefunden, nehme Tag 100...")
        day_data = ds.isel(time=100)
        date_str = str(day_data.time.values)[:10]

    # Canvas (Leinwand) vorbereiten
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f"Arctic Data Cube Analysis - {date_str}", fontsize=22, fontweight='bold')

    # ---------------------------------------------------------
    # Plot 1 (Oben Links): SAR Daten (Radar)
    # ---------------------------------------------------------
    ax1 = axes[0, 0]
    # Durchschnitt aus HH und HV bilden, falls 'band' existiert
    if 'band' in day_data['SAR'].dims:
        sar_plot_data = day_data['SAR'].mean(dim='band')
    else:
        sar_plot_data = day_data['SAR']
    
    # SCHUTZSCHALTER: Prüfen, ob das Bild 100% leer ist
    if int(sar_plot_data.notnull().sum()) == 0:
        ax1.text(0.5, 0.5, "No SAR Overpass\non this Date", ha='center', va='center', fontsize=22, color='red', weight='bold')
        ax1.set_title("SAR Backscatter (Radar)", fontsize=14)
        ax1.axis('off')
    else:
        vmin_sar = float(sar_plot_data.quantile(0.05))
        vmax_sar = float(sar_plot_data.quantile(0.95))
        sar_plot_data.plot(ax=ax1, cmap='gray', vmin=vmin_sar, vmax=vmax_sar, add_colorbar=True, cbar_kwargs={'label': 'Sigma0 Backscatter'})
        ax1.set_title("SAR Backscatter (Radar)", fontsize=14)
        ax1.axis('off')

    # ---------------------------------------------------------
    # Plot 2 (Oben Rechts): Thermal Daten (Temperatur)
    # ---------------------------------------------------------
    ax2 = axes[0, 1]
    if int(day_data['Thermal'].notnull().sum()) == 0:
        ax2.text(0.5, 0.5, "No Thermal Data", ha='center', va='center', fontsize=22, color='red', weight='bold')
        ax2.set_title("Thermal Surface Temp", fontsize=14)
        ax2.axis('off')
    else:
        day_data['Thermal'].plot(ax=ax2, cmap='inferno', add_colorbar=True, cbar_kwargs={'label': 'Temperature (Kelvin)'})
        ax2.set_title("Thermal Surface Temperature", fontsize=14)
        ax2.axis('off')

    # ---------------------------------------------------------
    # Plot 3 (Unten Links): Measurement Age
    # ---------------------------------------------------------
    ax3 = axes[1, 0]
    # Rot/Grün Skala: 0 Tage (frisch) = Grün, >5 Tage (alt/interpoliert) = Rot
    day_data['Time_To_Nearest_S1'].plot(ax=ax3, cmap='RdYlGn_r', vmin=0, vmax=5, add_colorbar=True, cbar_kwargs={'label': 'Age of Measurement (Days)'})
    ax3.set_title("SAR Measurement Age (Interpolation Check)", fontsize=14)
    ax3.axis('off')

    # ---------------------------------------------------------
    # Plot 4 (Unten Rechts): Trend Zeitreihe (Das ganze Jahr!)
    # ---------------------------------------------------------
    ax4 = axes[1, 1]
    print("Berechne räumlichen Durchschnitt für die Zeitreihen-Kurve (Das dauert ein paar Sekunden)...")
    
    # Mittelwert der Trends für das gesamte Raster für jeden Tag des Jahres berechnen
    trend_3d_mean = ds['Thermal_Trend_3d'].mean(dim=['x', 'y']).compute()
    trend_7d_mean = ds['Thermal_Trend_7d'].mean(dim=['x', 'y']).compute()

    # Die beiden Linien plotten
    ax4.plot(ds.time, trend_3d_mean, label="3-Day Trend", color="dodgerblue", linewidth=1.2, alpha=0.8)
    ax4.plot(ds.time, trend_7d_mean, label="7-Day Trend", color="darkorange", linewidth=2.0)
    
    # Optische Hilfen: Nulllinie und der aktuelle Tag
    ax4.axhline(0, color='black', linestyle='-', linewidth=1.2)
    ax4.axvline(day_data.time.values, color='red', linestyle='--', linewidth=1.5, label=f"Current View ({date_str})")
    
    # Achsen-Design
    ax4.set_title("Average Thermal Trends over the entire year", fontsize=14)
    ax4.set_ylabel("Trend (Δ Kelvin)")
    ax4.set_xlabel("Date")
    ax4.legend(loc="upper left")
    ax4.grid(True, linestyle='--', alpha=0.5)
    
    # Datumstexte unten rotieren, damit sie lesbar sind
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # ---------------------------------------------------------
    # Speichern
    # ---------------------------------------------------------
    plt.tight_layout()
    plt.subplots_adjust(top=0.92) 
    
    output_img = os.path.join(plot_dir, "datacube_dashboard_final.png")
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"\nERFOLG! Dein Plot wurde gespeichert unter: {output_img}")

    # RAM aufräumen
    ds.close()

if __name__ == "__main__":
    main()