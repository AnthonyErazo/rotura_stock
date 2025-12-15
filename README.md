# 📦 WMS - Predicción de Rotura de Stock

Sistema de **Business Analytics & IA** que predice roturas de stock en un WMS con **14 días de anticipación**, permitiendo tomar decisiones preventivas de reabastecimiento.

## 🎯 Descripción

Este proyecto implementa un modelo de **Regresión Logística** que analiza 35 variables (stock actual, demanda, lead times, características de clientes y proveedores) para predecir la probabilidad de quedarse sin inventario. Incluye un dashboard interactivo desarrollado con Streamlit para visualización de datos, análisis exploratorio y predicciones en tiempo real.

**Métricas del modelo:**
- ✅ Accuracy: 84.5%
- ✅ ROC-AUC: 96.2%
- ✅ Recall: 97.4% (detecta casi todas las roturas)

---

## 🚀 Instalación y Uso

### 1. Clonar el repositorio

```bash
git clone <URL_DEL_REPOSITORIO>
cd BAIA_WMS_Streamlit_FinalRepo_v2
```

### 2. Crear entorno virtual

```bash
python -m venv venv
```

### 3. Activar el entorno virtual

**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### 4. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 5. Entrenar el modelo (opcional)

El modelo ya está pre-entrenado en `/models`, pero puedes re-entrenarlo:

```bash
python scripts/train_model.py
```

Esto generará:
- `models/stockout14d_logreg.joblib` (modelo)
- `models/metrics.json` (métricas de evaluación)

### 6. Ejecutar el dashboard

```bash
streamlit run app.py
```

El dashboard se abrirá automáticamente en tu navegador en `http://localhost:8501`

> **Nota**: Asegúrate de tener el entorno virtual activado cada vez que ejecutes el proyecto.

---

## 📊 Características del Dashboard

El sistema incluye **5 pestañas interactivas**:

### 1. **Maestros**
Visualización de los 3 maestros de datos (MDM v3):
- Clientes (109 registros)
- Proveedores (204 registros)
- Servicios (200 registros)

### 2. **Diccionarios**
Definiciones y metadatos de cada campo de los maestros.

### 3. **Calidad de Datos**
Análisis exploratorio (EDA) con:
- Conteo de registros e IDs únicos
- Detección de valores nulos por campo
- Validación de RUC de proveedores

### 4. **Modelo**
Entrenamiento y evaluación:
- Generación de dataset transaccional (ajustable de 6 a 24 períodos)
- Métricas: Accuracy, ROC-AUC, Recall
- Matriz de confusión y reporte de clasificación

### 5. **Predicción**
Dos modos de predicción:
- **Modo Dataset**: Selecciona un caso histórico y predice
- **Modo Formulario**: Ingresa valores manualmente para simular escenarios

---

## 🗂️ Estructura del Proyecto

```
BAIA_WMS_Streamlit_FinalRepo_v2/
├── data/                              # Datos maestros (Excel)
│   ├── maestro_clientes.xlsx
│   ├── maestro_proveedores.xlsx
│   └── maestro_servicios.xlsx
├── models/                            # Modelos entrenados
│   ├── stockout14d_logreg.joblib
│   └── metrics.json
├── scripts/
│   └── train_model.py                 # Script de entrenamiento
├── wms_pipeline.py                    # Pipeline de datos y modelado
├── app.py                             # Dashboard Streamlit
├── requirements.txt                   # Dependencias
└── README.md
```

---

## 📚 Documentación

Para entender en profundidad el sistema, consulta los siguientes documentos:

### 📖 Guías Completas

- **[Documentación del Modelo](./DOCUMENTACION_MODELO.md)** - Visión general, arquitectura, métricas, variables y aplicación TO-BE
- **[Explicación de Predicción](./EXPLICACION_PREDICCION.md)** - Uso detallado de los modos de predicción, variables operacionales y toma de decisiones
- **[Modelo Matemático](./MODELO_MATEMATICO.md)** - Fundamentos matemáticos, fórmulas, proceso de entrenamiento y optimización

### 🎓 Contenido Académico

Estos documentos incluyen:
- ✅ Problema de negocio y relación con el proceso core
- ✅ Variables de entrada/salida y su origen (maestros y transacciones)
- ✅ Técnica y modelo utilizado (Regresión Logística)
- ✅ Resultados principales, interpretación y limitaciones
- ✅ Integración con el proceso TO-BE (decisiones operativas)

---

## 🛠️ Tecnologías Utilizadas

- **Python 3.8+**
- **Streamlit** - Dashboard interactivo
- **Pandas** - Manipulación de datos
- **Scikit-learn** - Machine Learning
- **NumPy** - Cálculos numéricos
- **Joblib** - Serialización de modelos
- **OpenPyXL** - Lectura de archivos Excel

---

## 📈 Resultados

El modelo genera alertas clasificadas por nivel de riesgo:

| Riesgo | Probabilidad | Acción Sugerida |
|--------|--------------|-----------------|
| 🔴 **ALTO** | ≥ 70% | Reabastecimiento inmediato, priorizar recepción |
| 🟡 **MEDIO** | 40-70% | Monitoreo diario, validar recepción pendiente |
| 🟢 **BAJO** | < 40% | Operación normal, revisión periódica |
