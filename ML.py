import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
import gc

warnings.filterwarnings("ignore")

def plot_ml_ablation_study(nc_path, target_var="val_CT", sample_size=40000, out_png="ml_ablation_chart.png"):
    print("=== STARTE ML ABLATION STUDY (NUR CHART) ===")
    
    features_list = [
        "SAR", "Thermal", "Time_To_Nearest_SAR", "Time_To_Nearest_Thermal", 
        "Thermal_Trend_3d", "Thermal_Trend_7d", "Thermal_Trend_14d", "Thermal_Trend_30d", 
        "SAR_Trend_3d", "SAR_Trend_7d", "SAR_Trend_14d", "SAR_Trend_30d"
    ]
    
    print(f"1. Sammle repräsentative Trainingsdaten ({sample_size} Pixel)...")
    ds = xr.open_dataset(nc_path, chunks={"time": 10})
    total_days = len(ds.time)
    
    collected_data = []
    total_valid = 0
    step = max(1, total_days // 50)
    
    for i in range(0, total_days, step):
        if total_valid >= sample_size:
            break
            
        day_ds = ds.isel(time=i).compute()
        if target_var not in day_ds or np.isnan(day_ds[target_var].values).all():
            continue
            
        target_data = np.squeeze(day_ds[target_var].values).flatten()
        valid_indices = np.where(~np.isnan(target_data))[0]
        
        if len(valid_indices) > 0:
            # Begrenze pro Tag für mehr Abwechslung
            if len(valid_indices) > 1000:
                valid_indices = np.random.choice(valid_indices, 1000, replace=False)
                
            day_features = []
            for feat in features_list:
                if feat in day_ds:
                    feat_data = np.squeeze(day_ds[feat].isel(band=0).values if "band" in day_ds[feat].dims else day_ds[feat].values).flatten()
                    day_features.append(feat_data[valid_indices].astype(np.float32))
                else:
                    # Fallback falls Feature ganz fehlt
                    day_features.append(np.full(len(valid_indices), np.nan, dtype=np.float32))
                
            day_df = pd.DataFrame(np.column_stack(day_features), columns=features_list)
            day_df[target_var] = target_data[valid_indices].astype(np.float32)
            collected_data.append(day_df)
            total_valid += len(valid_indices)

    df_master = pd.concat(collected_data, ignore_index=True)
    del collected_data
    gc.collect()
    
    print("2. Bereite saubere Klassifikation vor (Wasser vs. Eis)...")
    # Dynamischer Check: Gehen die Daten von 0-1 oder 0-100?
    max_val = df_master[target_var].max()
    threshold_water = 0.15 if max_val <= 1.5 else 15.0
    threshold_ice = 0.80 if max_val <= 1.5 else 80.0
    
    # Filtere den Matsch in der Mitte raus für ein klares KI-Training
    df_clean = df_master[(df_master[target_var] < threshold_water) | (df_master[target_var] > threshold_ice)].copy()
    
    # 0 = Wasser, 1 = Packeis
    y = np.where(df_clean[target_var] > threshold_ice, 1, 0)
    
    print(f" -> Nutze {len(df_clean)} klare Pixel für das Training.")

    # ==========================================
    # ML TRAINING (DIE 3 SCHRITTE)
    # ==========================================
    print("3. Trainiere Modelle (Ablation)...")
    
    feature_sets = {
        "1. Nur SAR (Baseline)": ["SAR"],
        "2. SAR + Thermal (Fusion)": ["SAR", "Thermal"],
        "3. SAR + Thermal + Trends (4D Cube)": features_list
    }
    
    ablation_results = {}
    for name, feats in feature_sets.items():
        X = df_clean[feats]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # HistGradientBoosting kommt perfekt mit NaNs (Datenlücken) klar!
        model = HistGradientBoostingClassifier(random_state=42, max_iter=100)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, preds) * 100
        ablation_results[name] = accuracy
        print(f" -> {name}: {accuracy:.1f}%")

    # ==========================================
    # CHART GENERIEREN
    # ==========================================
    print("\n4. Generiere Chart...")
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.set_theme(style="whitegrid")
    
    y_labels = list(ablation_results.keys())
    x_values = list(ablation_results.values())
    
    # Schöne Farbpalette (Dunkelblau -> Magenta -> Orange)
    palette = sns.color_palette("rocket", len(y_labels))
    
    bars = sns.barplot(x=x_values, y=y_labels, palette=palette, ax=ax)
    
    ax.set_title("Machine Learning Evaluierung: Der Mehrwert des 4D DataCubes", fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel("Klassifikations-Genauigkeit (%)", fontsize=14)
    ax.set_ylabel("")
    
    # Achse ab 50% starten lassen (da 50% bei 2 Klassen bloßes Raten ist)
    ax.set_xlim(50, 100)
    
    # Zahlen in die Balken schreiben
    for i, v in enumerate(x_values):
        ax.text(v - 1, i, f"{v:.1f}%", color='white', fontweight='bold', fontsize=16, va='center', ha='right')

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"=== PERFEKT! Einzel-Chart gespeichert unter: {out_png} ===")

if __name__ == "__main__":
    CUBE_PATH = "data/NetCDF/arctic_datacube_full_period.nc" 
    plot_ml_ablation_study(CUBE_PATH)