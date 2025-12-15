import streamlit as st
import pandas as pd
from pathlib import Path
from wms_pipeline import (
    load_masters, quality_checks, build_dataset,
    train_or_load_model, predict_from_form, predict_from_dataset_row
)

st.set_page_config(page_title="WMS – Alerta de Rotura de Stock (MDM v3)", layout="wide")

DATA_DIR = Path("data")
MODELS_DIR = Path("models")

st.title("WMS – Alerta de Rotura de Stock (horizonte 14 días)")

# Load masters from local data folder (no uploaders)
with st.spinner("Cargando maestros (MDM v3) desde /data ..."):
    masters = load_masters(DATA_DIR)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Maestros", "Diccionarios", "Calidad de datos", "Modelo", "Predicción"
])

with tab1:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Clientes")
        st.dataframe(masters["clientes"], use_container_width=True, height=420)
    with c2:
        st.subheader("Proveedores")
        st.dataframe(masters["proveedores"], use_container_width=True, height=420)
    with c3:
        st.subheader("Servicios")
        st.dataframe(masters["servicios"], use_container_width=True, height=420)

with tab2:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Diccionario – Clientes")
        st.dataframe(masters["dicc_clientes"], use_container_width=True, height=420)
    with c2:
        st.subheader("Diccionario – Proveedores")
        st.dataframe(masters["dicc_proveedores"], use_container_width=True, height=420)
    with c3:
        st.subheader("Diccionario – Servicios")
        st.dataframe(masters["dicc_servicios"], use_container_width=True, height=420)

with tab3:
    st.subheader("Resumen de calidad")
    dq = quality_checks(masters)
    st.dataframe(dq["resumen"], use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.caption("Top nulos – Clientes")
        st.dataframe(dq["top_nulos_clientes"], use_container_width=True, height=260)
    with c2:
        st.caption("Top nulos – Proveedores")
        st.dataframe(dq["top_nulos_proveedores"], use_container_width=True, height=260)
    with c3:
        st.caption("Top nulos – Servicios")
        st.dataframe(dq["top_nulos_servicios"], use_container_width=True, height=260)

with tab4:
    st.subheader("Dataset del modelo")

    periods = st.slider("Cantidad de periodos (snapshots) por servicio", min_value=6, max_value=24, value=12, step=1)
    dataset = build_dataset(masters, periods=periods)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Registros", f"{len(dataset):,}")
    with c2:
        st.metric("% Stockout14d = 1", f"{dataset['Stockout14d'].mean()*100:.1f}%")
    with c3:
        st.metric("Servicios únicos", dataset["ServicioID"].nunique())

    st.dataframe(dataset.head(50), use_container_width=True, height=360)

    st.divider()
    st.subheader("Entrenamiento / Evaluación")
    model, metrics = train_or_load_model(dataset, MODELS_DIR)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    with m2:
        st.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    with m3:
        st.metric("Recall (Stockout=1)", f"{metrics['recall_pos']:.3f}")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.caption("Matriz de confusión (test)")
        st.dataframe(metrics["confusion_matrix_df"], use_container_width=True)
    with c2:
        st.caption("Reporte (resumen)")
        st.dataframe(metrics["report_df"], use_container_width=True)

    st.info("El modelo se guarda en /models como archivo .joblib después del entrenamiento.")

with tab5:
    st.subheader("Predicción")
    # Ensure dataset and model are available
    periods = 12
    dataset = build_dataset(masters, periods=periods)
    model, _ = train_or_load_model(dataset, MODELS_DIR)

    mode = st.radio("Modo", ["Usar un caso del dataset", "Ingresar valores (formulario)"], horizontal=True)

    if mode == "Usar un caso del dataset":
        colA, colB = st.columns([2, 1])
        with colA:
            servicio_id = st.selectbox("ServicioID", sorted(dataset["ServicioID"].unique()))
        with colB:
            periodo = st.selectbox("Periodo", sorted(dataset["Periodo"].unique()))

        row = dataset[(dataset["ServicioID"] == servicio_id) & (dataset["Periodo"] == periodo)].iloc[0]
        st.caption("Caso seleccionado (snapshot)")
        st.dataframe(pd.DataFrame([row]), use_container_width=True)

        result = predict_from_dataset_row(model, row)
        st.subheader("Resultado")
        st.metric("Probabilidad de rotura en 14 días", f"{result['prob']*100:.1f}%")
        st.write(result["mensaje"])

    else:
        servicio_id = st.selectbox("ServicioID (para sugerencias)", sorted(dataset["ServicioID"].unique()))
        periodo = st.selectbox("Periodo (para sugerencias)", sorted(dataset["Periodo"].unique()))
        base_row = dataset[(dataset["ServicioID"] == servicio_id) & (dataset["Periodo"] == periodo)].iloc[0]

        with st.form("form_prediccion"):
            st.caption("Valores sugeridos (puedes editarlos)")
            c1, c2, c3 = st.columns(3)
            with c1:
                stock = st.number_input("StockActual", min_value=0, value=int(base_row["StockActual"]))
                demanda = st.number_input("DemandaDiariaEst", min_value=0.0, value=float(base_row["DemandaDiariaEst"]), format="%.3f")
            with c2:
                dias_rec = st.number_input("DiasHastaRecepcion", min_value=0, value=int(base_row["DiasHastaRecepcion"]))
                rec_pend = st.number_input("RecepcionPendiente", min_value=0, value=int(base_row["RecepcionPendiente"]))
            with c3:
                horizonte = st.selectbox("Horizonte (días)", [7, 14, 30], index=1)

            submitted = st.form_submit_button("Predecir", type="primary")
            
        if submitted:
            result = predict_from_form(model, base_row, stock, demanda, dias_rec, rec_pend, horizonte)
            st.subheader("Resultado")
            st.metric("Probabilidad de rotura", f"{result['prob']*100:.1f}%")
            st.write(result["mensaje"])
