from __future__ import annotations

import re
import json
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report, accuracy_score, precision_recall_fscore_support


def _normalize_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza strings: elimina espacios y estandariza valores nulos."""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({
                "": np.nan, "nan": np.nan, "NaN": np.nan, "None": np.nan,
                "NULL": np.nan, "N/A": np.nan, "NA": np.nan
            })
    return df

def _dedupe_best(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """Elimina duplicados por ID, manteniendo el registro con más datos completos."""
    df = df.copy()
    df["_nonnull"] = df.notna().sum(axis=1)
    df = df.sort_values([id_col, "_nonnull"], ascending=[True, False])
    df = df.drop_duplicates(subset=[id_col], keep="first").drop(columns=["_nonnull"])
    return df

def load_masters(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Carga y normaliza maestros desde Excel."""
    cli_path = data_dir / "maestro_clientes.xlsx"
    prov_path = data_dir / "maestro_proveedores.xlsx"
    srv_path = data_dir / "maestro_servicios.xlsx"

    if not (cli_path.exists() and prov_path.exists() and srv_path.exists()):
        raise FileNotFoundError(
            "Faltan archivos en /data. Se esperan: maestro_clientes.xlsx, maestro_proveedores.xlsx, maestro_servicios.xlsx"
        )

    cli_sheets = pd.read_excel(cli_path, sheet_name=None)
    prov_sheets = pd.read_excel(prov_path, sheet_name=None)
    srv_sheets = pd.read_excel(srv_path, sheet_name=None)

    cli_raw = cli_sheets["Maestro de Clientes"]
    prov_raw = prov_sheets["Proveedores_data"]
    srv_raw = srv_sheets["Servicios_data"]

    cli = _dedupe_best(_normalize_strings(cli_raw), "ClienteID")
    prov = _dedupe_best(_normalize_strings(prov_raw), "ProveedorID")
    srv = _dedupe_best(_normalize_strings(srv_raw), "ServicioID")

    for c in ["LimiteCredito"]:
        if c in cli.columns:
            cli[c] = pd.to_numeric(cli[c], errors="coerce")

    for c in ["LeadTimePromedioDias", "ToleranciaEntregaDias", "RatingDesempeno", "DiasPago", "LimiteCredito"]:
        if c in prov.columns:
            prov[c] = pd.to_numeric(prov[c], errors="coerce")

    for c in ["TarifaBase", "LeadTimeMinDias", "LeadTimeMaxDias", "TiempoEjecucionHoras", "CantidadPedidoEstandar", "CostoEstandar", "TarifaImpuesto"]:
        if c in srv.columns:
            srv[c] = pd.to_numeric(srv[c], errors="coerce")

    def parse_sla_hours(s):
        if pd.isna(s): return np.nan
        m = re.search(r"(\\d+)\\s*h", str(s).lower())
        return float(m.group(1)) if m else np.nan

    def parse_sla_pct(s):
        if pd.isna(s): return np.nan
        m = re.search(r"(\\d+)\\s*%", str(s))
        return float(m.group(1)) if m else np.nan

    if "SLA" in srv.columns:
        srv["SLA_horas"] = srv["SLA"].apply(parse_sla_hours).fillna(0)
        srv["SLA_pct"] = srv["SLA"].apply(parse_sla_pct).fillna(0)
    else:
        srv["SLA_horas"] = 0
        srv["SLA_pct"] = 0

    return {
        "clientes": cli,
        "proveedores": prov,
        "servicios": srv,
        "dicc_clientes": cli_sheets.get("DICCIONARIO", pd.DataFrame()),
        "dicc_proveedores": prov_sheets.get("DICCIONARIO", pd.DataFrame()),
        "dicc_servicios": srv_sheets.get("DICCIONARIO", pd.DataFrame()),
    }


def _top_missing(df: pd.DataFrame, topn: int = 10) -> pd.DataFrame:
    miss = df.isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0].head(topn)
    return pd.DataFrame({"Campo": miss.index, "% Nulos": (miss.values * 100).round(2)})

def _invalid_ruc_count(series: pd.Series) -> int:
    """Cuenta cuántos RUC no tienen exactamente 11 dígitos."""
    s = series.astype(str).str.strip().replace({"nan": np.nan, "None": np.nan, "": np.nan})
    s = s.dropna()
    return int((~s.str.fullmatch(r"\\d{11}")).sum())

def quality_checks(masters: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """EDA: registros, nulos, RUC inválidos."""
    cli = masters["clientes"]
    prov = masters["proveedores"]
    srv = masters["servicios"]

    resumen = pd.DataFrame([
        {
            "Maestro": "Clientes",
            "Registros": len(cli),
            "IDs únicos": cli["ClienteID"].nunique(dropna=True),
            "Duplicados por ID (raw deducido)": 0,
            "Nulos totales": int(cli.isna().sum().sum()),
        },
        {
            "Maestro": "Proveedores",
            "Registros": len(prov),
            "IDs únicos": prov["ProveedorID"].nunique(dropna=True),
            "Nulos totales": int(prov.isna().sum().sum()),
            "RUC inválidos (≠11 dígitos)": _invalid_ruc_count(prov["RUC"]) if "RUC" in prov.columns else 0,
        },
        {
            "Maestro": "Servicios",
            "Registros": len(srv),
            "IDs únicos": srv["ServicioID"].nunique(dropna=True),
            "Nulos totales": int(srv.isna().sum().sum()),
        }
    ])

    return {
        "resumen": resumen,
        "top_nulos_clientes": _top_missing(cli),
        "top_nulos_proveedores": _top_missing(prov),
        "top_nulos_servicios": _top_missing(srv),
    }


def _map_supplier_category(service_cat: str) -> str:
    """Mapea categoría de servicio a categoría objetivo de proveedor (LOGISTICA o SERVICIOS)."""
    if pd.isna(service_cat):
        return "SERVICIOS"
    sc = str(service_cat).strip().lower()
    if sc in ["almacenaje", "distribución", "transporte", "comercio exterior", "it/tracking", "valor agregado"]:
        return "LOGISTICA"
    if sc in ["tecnología", "consultoría", "administrativo", "servicios"]:
        return "SERVICIOS"
    return "SERVICIOS"

def _extract_num(x: str) -> int:
    """Extrae primer número encontrado en un string."""
    m = re.search(r"(\\d+)", str(x))
    return int(m.group(1)) if m else 0

def build_dataset(masters: dict[str, pd.DataFrame], periods: int = 12) -> pd.DataFrame:
    """Genera dataset transaccional con variables derivadas y target Stockout14d."""
    cli = masters["clientes"]
    prov = masters["proveedores"]
    srv = masters["servicios"]


    base = srv.merge(
        cli[["ClienteID", "Segmento", "CanalPreferido", "ZonaDespacho", "Departamento"]],
        how="left",
        left_on="ClientePropietario",
        right_on="ClienteID",
        suffixes=("", "_cli"),
    )

    base["ProveedorCategoriaObjetivo"] = base["Categoria"].apply(_map_supplier_category)
    prov_rank = prov.copy()
    prov_rank["RatingDesempeno_fill"] = prov_rank["RatingDesempeno"].fillna(prov_rank["RatingDesempeno"].median())
    prov_rank["LeadTimePromedioDias_fill"] = prov_rank["LeadTimePromedioDias"].fillna(prov_rank["LeadTimePromedioDias"].median())

    def pick_supplier(row) -> str:
        cat = row["ProveedorCategoriaObjetivo"]
        dept = row.get("Departamento", None)
        candidates = prov_rank[prov_rank["Categoria"].str.upper() == cat] if "Categoria" in prov_rank.columns else prov_rank
        if candidates.empty:
            candidates = prov_rank
        candidates = candidates.copy()
        candidates["dept_match"] = (candidates["Departamento"] == dept).astype(int) if "Departamento" in candidates.columns else 0
        candidates = candidates.sort_values(
            ["dept_match", "RatingDesempeno_fill", "LeadTimePromedioDias_fill", "ProveedorID"],
            ascending=[False, False, True, True],
        )
        return str(candidates.iloc[0]["ProveedorID"])

    base["ProveedorID"] = base.apply(pick_supplier, axis=1)

    base = base.merge(
        prov[["ProveedorID", "Categoria", "LeadTimePromedioDias", "ToleranciaEntregaDias", "RatingDesempeno", "CertificadoCalidad", "Estado"]],
        on="ProveedorID",
        how="left",
        suffixes=("", "_prov"),
    )

    periods_df = pd.DataFrame({"Periodo": list(range(1, periods + 1))})
    base["key"] = 1
    periods_df["key"] = 1
    ds = base.merge(periods_df, on="key").drop(columns=["key"])

    seg_factor = {"BASICO": 0.8, "ESTANDAR": 1.0, "PREFERENTE": 1.2}
    ds["Segmento"] = ds["Segmento"].fillna("SIN_DATO").astype(str).str.upper()
    ds["FactorSegmento"] = ds["Segmento"].map(seg_factor).fillna(1.0)

    ds["CantidadPedidoEstandar"] = ds["CantidadPedidoEstandar"].fillna(ds["CantidadPedidoEstandar"].median())
    ds["DemandaDiariaEst"] = (ds["CantidadPedidoEstandar"] / 14.0) * ds["FactorSegmento"]

    ds["ServicioNum"] = ds["ServicioID"].apply(_extract_num)
    lead = ds["LeadTimeMaxDias"].fillna(ds["LeadTimeMaxDias"].median())
    ds["FactorLead"] = (1.0 + (lead / 60.0)).clip(0.8, 2.0)

    ds["Ciclo"] = (ds["Periodo"] + (ds["ServicioNum"] % 4)) % 4
    ds["Dip"] = np.where(ds["Ciclo"] == 0, 0.55, 1.0)

    ds["StockActual"] = np.round(
        ds["CantidadPedidoEstandar"] * (1 + (ds["Periodo"] % 3)) * ds["FactorLead"] * ds["Dip"]
    ).astype(int)

    prov_lead = ds["LeadTimePromedioDias"].fillna(ds["LeadTimePromedioDias"].median())
    tol = ds["ToleranciaEntregaDias"].fillna(0)
    ds["DiasHastaRecepcion"] = np.round(np.minimum(prov_lead + (ds["Periodo"] % 2) * tol, 45)).astype(int)
    ds["RecepcionPendiente"] = np.where(ds["StockActual"] < (ds["DemandaDiariaEst"] * 10), ds["CantidadPedidoEstandar"], 0).astype(int)

    dias_cobertura = ds["StockActual"] / ds["DemandaDiariaEst"].replace(0, np.nan)
    ds["Stockout14d"] = ((dias_cobertura < 14) & (ds["DiasHastaRecepcion"] > dias_cobertura)).astype(int)

    keep = [
        "ServicioID", "NombreServicio", "Categoria", "Subcategoria",
        "UnidadTarifa", "TipoUnidad", "TarifaBase", "Moneda",
        "RequiereCertificacion", "Temperatura", "LeadTimeMinDias", "LeadTimeMaxDias",
        "TiempoEjecucionHoras", "ModalidadContrato", "Estado",
        "CantidadPedidoEstandar", "CostoEstandar", "TarifaImpuesto",
        "TemperaturaControlada", "CaducidadControlada", "SLA_horas", "SLA_pct",
        "ClientePropietario", "Segmento", "CanalPreferido", "ZonaDespacho", "Departamento",
        "ProveedorID", "Categoria_prov", "LeadTimePromedioDias", "ToleranciaEntregaDias", "RatingDesempeno",
        "CertificadoCalidad", "Estado_prov",
        "Periodo", "StockActual", "DemandaDiariaEst", "DiasHastaRecepcion", "RecepcionPendiente",
        "Stockout14d",
    ]
    return ds[keep].copy()


FEATURE_COLS = [
    "Categoria", "Subcategoria", "UnidadTarifa", "TipoUnidad", "Moneda", "RequiereCertificacion", "Temperatura",
    "LeadTimeMinDias", "LeadTimeMaxDias", "TiempoEjecucionHoras", "ModalidadContrato", "Estado",
    "CantidadPedidoEstandar", "CostoEstandar", "TarifaImpuesto", "TemperaturaControlada", "CaducidadControlada", "SLA_horas", "SLA_pct",
    "Segmento", "CanalPreferido", "ZonaDespacho", "Departamento",
    "Categoria_prov", "LeadTimePromedioDias", "ToleranciaEntregaDias", "RatingDesempeno", "CertificadoCalidad", "Estado_prov",
    "Periodo", "StockActual", "RecepcionPendiente", "DiasHastaRecepcion", "DemandaDiariaEst",
]

def _build_pipeline(X: pd.DataFrame) -> Pipeline:
    """Construye pipeline de preprocesamiento + LogisticRegression."""
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), numeric_features),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical_features),
        ]
    )

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5, solver="liblinear")
    return Pipeline(steps=[("preprocess", preprocess), ("clf", clf)])

def train_or_load_model(dataset: pd.DataFrame, models_dir: Path):
    """Entrena modelo con GroupShuffleSplit y guarda .joblib + metrics.json."""
    models_dir.mkdir(exist_ok=True, parents=True)
    model_path = models_dir / "stockout14d_logreg.joblib"
    metrics_path = models_dir / "metrics.json"

    X = dataset[FEATURE_COLS].copy()
    y = dataset["Stockout14d"].copy()
    groups = dataset["ServicioID"].copy()

    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    pipe = _build_pipeline(X)
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    acc = float(accuracy_score(y_test, pred))
    auc = float(roc_auc_score(y_test, proba))
    cm = confusion_matrix(y_test, pred)
    rep = classification_report(y_test, pred, output_dict=True)

    precision_pos, recall_pos, f1_pos, _ = precision_recall_fscore_support(y_test, pred, average="binary", zero_division=0)

    joblib.dump(pipe, model_path)
    metrics_obj = {
        "accuracy": acc,
        "roc_auc": auc,
        "precision_pos": float(precision_pos),
        "recall_pos": float(recall_pos),
        "f1_pos": float(f1_pos),
        "confusion_matrix": cm.tolist(),
        "classification_report": rep,
    }
    metrics_path.write_text(json.dumps(metrics_obj, indent=2), encoding="utf-8")

    cm_df = pd.DataFrame(cm, index=["Real_0", "Real_1"], columns=["Pred_0", "Pred_1"])
    report_df = pd.DataFrame(rep).T[["precision", "recall", "f1-score", "support"]].round(3)

    return pipe, {
        "accuracy": acc,
        "roc_auc": auc,
        "precision_pos": float(precision_pos),
        "recall_pos": float(recall_pos),
        "f1_pos": float(f1_pos),
        "confusion_matrix_df": cm_df,
        "report_df": report_df,
    }


def _risk_message(prob: float, horizonte: int) -> str:
    """Genera mensaje de riesgo según probabilidad (>=0.7 ALTO, >=0.4 MEDIO, <0.4 BAJO)."""
    if prob >= 0.70:
        return f"Riesgo ALTO de rotura en {horizonte} días. Acción sugerida: generar reabastecimiento inmediato y priorizar recepción."
    if prob >= 0.40:
        return f"Riesgo MEDIO de rotura en {horizonte} días. Acción sugerida: monitoreo diario y validar recepción pendiente."
    return f"Riesgo BAJO de rotura en {horizonte} días. Acción sugerida: operación normal y revisión periódica."

def predict_from_dataset_row(model: Pipeline, row: pd.Series) -> dict:
    """Predice probabilidad de stockout desde una fila del dataset."""
    X = pd.DataFrame([row[FEATURE_COLS].to_dict()])
    prob = float(model.predict_proba(X)[:, 1][0])
    return {"prob": prob, "mensaje": _risk_message(prob, 14)}

def predict_from_form(model: Pipeline, base_row: pd.Series, stock: int, demanda: float, dias_rec: int, rec_pend: int, horizonte: int) -> dict:
    """Predice probabilidad de stockout desde inputs de formulario (operador WMS)."""
    payload = base_row[FEATURE_COLS].to_dict()
    payload["StockActual"] = int(stock)
    payload["DemandaDiariaEst"] = float(demanda)
    payload["DiasHastaRecepcion"] = int(dias_rec)
    payload["RecepcionPendiente"] = int(rec_pend)

    X = pd.DataFrame([payload])
    prob = float(model.predict_proba(X)[:, 1][0])
    return {"prob": prob, "mensaje": _risk_message(prob, horizonte)}
