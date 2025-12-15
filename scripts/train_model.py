"""
Entrena modelo predictivo de rotura de stock (Stockout14d).
Genera dataset, entrena Regresión Logística, guarda modelo y métricas.
Uso: python scripts/train_model.py
"""
from pathlib import Path
import pandas as pd
from wms_pipeline import load_masters, build_dataset, train_or_load_model, quality_checks

DATA_DIR = Path("data")
MODELS_DIR = Path("models")

def main():
    """Pipeline: carga maestros, EDA, genera dataset, entrena modelo."""
    print("=== CARGANDO MAESTROS ===")
    masters = load_masters(DATA_DIR)
    
    print("\n=== ANÁLISIS EXPLORATORIO (EDA) ===")
    qc = quality_checks(masters)
    print("\nResumen de datos:")
    print(qc["resumen"].to_string(index=False))
    
    print("\n=== GENERANDO DATASET TRANSACCIONAL ===")
    dataset = build_dataset(masters, periods=12)
    print(f"Registros generados: {len(dataset)}")
    print(f"Distribución target Stockout14d:\n{dataset['Stockout14d'].value_counts().to_dict()}")
    
    print("\n=== ENTRENANDO MODELO ===")
    train_or_load_model(dataset, MODELS_DIR)
    print("\n✓ Modelo y métricas guardados en /models")
    print("  - stockout14d_logreg.joblib")
    print("  - metrics.json")

if __name__ == "__main__":
    main()
