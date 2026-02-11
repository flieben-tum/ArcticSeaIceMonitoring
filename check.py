import os
import glob
import rioxarray
import matplotlib.pyplot as plt

def main():
    print("=== Raw TIFF Inspector ===")
    
    # Sucht das allererste SAR-TIFF aus dem Jahr 2023
    search_path = "data/GeoTIFF/SAR/2023/*.tif"
    tiff_files = glob.glob(search_path)
    
    if not tiff_files:
        print("Keine TIFFs gefunden! Stimmt der Pfad?")
        return
        
    first_tiff = sorted(tiff_files)[5] # Wir nehmen einfach das erste Bild
    print(f"Lade rohes Bild: {first_tiff}")
    
    # Roh laden (ohne Dask, ohne Masken)
    raw_ds = rioxarray.open_rasterio(first_tiff)
    
    # 1. Band auswählen (meist HH)
    raw_image = raw_ds.isel(band=0)
    
    # Nullen ausblenden für den Plot
    plot_data = raw_image.where(raw_image > 0)
    
    plt.figure(figsize=(10, 10))
    plot_data.plot(cmap='gray', robust=True)
    plt.title(f"RAW TIFF: {os.path.basename(first_tiff)}\n(Achte auf die Ränder!)", fontsize=14)
    plt.axis('equal') # Verhindert Verzerrungen
    
    out_path = "plots/raw_tiff_check.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Beweisfoto gespeichert unter: {out_path}")

if __name__ == "__main__":
    main()