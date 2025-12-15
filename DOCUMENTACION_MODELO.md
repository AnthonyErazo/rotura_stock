# 📊 Documentación Completa del Sistema WMS - Predicción de Rotura de Stock

## 🎯 1. Visión General del Sistema

Este sistema implementa un **modelo predictivo de Machine Learning** para anticipar roturas de stock (stockout) en un Warehouse Management System (WMS). El objetivo es predecir si un servicio/producto se quedará sin stock en los próximos **14 días**, permitiendo tomar acciones preventivas de reabastecimiento.

### Problema de Negocio
En un WMS, las roturas de stock generan:
- Pérdida de ventas y clientes insatisfechos
- Incumplimiento de SLA (Service Level Agreement)
- Costos de urgencia en reabastecimiento
- Deterioro de relaciones con proveedores

**Solución**: Predecir con anticipación cuándo ocurrirá una rotura para tomar decisiones proactivas.

---

## 🗂️ 2. Arquitectura del Sistema

### Componentes Principales

```
data/
  ├── maestro_clientes.xlsx      → Datos maestros de clientes
  ├── maestro_proveedores.xlsx   → Datos maestros de proveedores
  └── maestro_servicios.xlsx     → Datos maestros de servicios

scripts/
  └── train_model.py             → Script de entrenamiento del modelo

models/
  ├── stockout14d_logreg.joblib  → Modelo entrenado (Regresión Logística)
  └── metrics.json               → Métricas de evaluación

wms_pipeline.py                  → Pipeline completo de procesamiento y modelado
app.py                          → Dashboard interactivo (Streamlit)
```

### Flujo de Trabajo

1. **Carga de Maestros** → Lee y normaliza los 3 archivos Excel
2. **EDA (Análisis Exploratorio)** → Analiza calidad de datos
3. **Generación de Dataset** → Crea dataset transaccional expandido por períodos
4. **Entrenamiento** → Entrena modelo de Regresión Logística
5. **Predicción** → Dashboard interactivo para predecir roturas

---

## 📋 3. Calidad de Datos (Pestaña "Calidad de datos")

### Columnas del Resumen de Calidad

| Columna | Significado |
|---------|-------------|
| **Maestro** | Nombre del archivo maestro analizado (Clientes, Proveedores, Servicios) |
| **Registros** | Cantidad total de filas en el maestro |
| **IDs únicos** | Cantidad de identificadores únicos (sin duplicados) |
| **Duplicados por ID** | Cantidad de registros duplicados que fueron detectados (0 significa que ya se limpiaron) |
| **Nulos totales** | Suma total de celdas vacías en todo el maestro |
| **RUC inválidos** | (Solo proveedores) Cantidad de RUC que no tienen exactamente 11 dígitos |

### Interpretación

**Ejemplo:**
```
Maestro: Clientes
Registros: 109
IDs únicos: 109
Nulos totales: 118
```

Esto significa:
- Hay **109 clientes** en el sistema
- Todos tienen ID único (no hay duplicados)
- Existen **118 celdas vacías** distribuidas en todas las columnas (ej: email, teléfono, límite crédito faltantes)

### Top Nulos por Maestro

Muestra los **10 campos con mayor porcentaje de valores nulos**. Esto ayuda a identificar:
- Campos que necesitan completarse
- Campos que quizás no son relevantes para el negocio
- Oportunidades de mejora en la captura de datos

---

## 🔢 4. Modelo Predictivo (Pestaña "Modelo")

### 4.1 Dataset del Modelo

#### ¿Qué es "Cantidad de períodos por servicio"?

**Períodos** son "snapshots" o fotografías del estado del inventario en diferentes momentos del tiempo.

- **Mínimo 6 períodos**: Garantiza variabilidad temporal mínima para que el modelo aprenda patrones estacionales
- **Máximo 24 períodos**: Evita dataset excesivamente grande (2 años de datos mensuales)
- **Por defecto 12 períodos**: Representa 1 año de operación (12 meses)

**Ejemplo:**
Si tienes 200 servicios y 12 períodos:
```
200 servicios × 12 períodos = 2,400 registros
```

Cada registro representa el estado de un servicio en un período específico (mes).

#### Explicación de Métricas Mostradas

| Métrica | Valor Ejemplo | Significado |
|---------|---------------|-------------|
| **Registros** | 2,400 | Total de filas en el dataset (servicios × períodos) |
| **% Stockout14d = 1** | 22.0% | Porcentaje de casos donde SÍ hubo rotura de stock en 14 días |
| **Servicios únicos** | 200 | Cantidad de servicios/productos diferentes en el maestro |

**¿Por qué hay 200 servicios?**
Porque el archivo `maestro_servicios.xlsx` contiene 200 filas (200 productos/servicios diferentes que maneja el WMS).

**¿Qué significa "% Stockout14d = 1"?**
- Es el **target** (variable objetivo) del modelo
- `1` = SÍ habrá rotura de stock en los próximos 14 días
- `0` = NO habrá rotura de stock
- Si es 22%, significa que 22% de los casos analizados tuvieron rotura

### 4.2 Variables del Modelo

#### Variables de Entrada (Features)

El modelo usa **35 variables** agrupadas en:

**A) Características del Servicio/Producto:**
- Categoría, Subcategoría
- Lead time mínimo/máximo (días)
- Costo estándar, Tarifa base
- Requiere certificación (Sí/No)
- Temperatura controlada (Sí/No)
- SLA (horas y porcentaje)

**B) Características del Cliente:**
- Segmento (BÁSICO, ESTÁNDAR, PREFERENTE)
- Canal preferido
- Zona de despacho
- Departamento

**C) Características del Proveedor:**
- Categoría del proveedor
- Lead time promedio del proveedor
- Rating de desempeño (1-5)
- Certificado de calidad (Sí/No)
- Tolerancia de entrega (días)

**D) Variables Operacionales (las más importantes):**
- **StockActual**: Cantidad disponible HOY
- **DemandaDiariaEst**: Consumo diario estimado
- **DiasHastaRecepcion**: Días hasta que llegue el pedido del proveedor
- **RecepcionPendiente**: Cantidad que está por llegar
- **Periodo**: Momento temporal (1-12)

#### Variable de Salida (Target)

**Stockout14d**: Binaria (0 o 1)
- `1` = Habrá rotura de stock en 14 días
- `0` = No habrá rotura de stock

**Cálculo del target:**
```python
dias_cobertura = StockActual / DemandaDiariaEst

Stockout14d = 1 si:
  - dias_cobertura < 14 (se acaba antes de 14 días)
  Y
  - DiasHastaRecepcion > dias_cobertura (el pedido llega después del agotamiento)
```

**Ejemplo:**
- Stock actual: 100 unidades
- Demanda diaria: 10 unidades/día
- Días de cobertura: 100/10 = 10 días
- Días hasta recepción: 15 días
- **Resultado**: Stockout14d = 1 (habrá rotura porque el stock dura 10 días pero el pedido llega en 15)

### 4.3 Técnica de Modelado

**Algoritmo**: Regresión Logística (Logistic Regression)

**¿Por qué Regresión Logística?**
- ✅ Interpretable (puedes ver qué variables influyen más)
- ✅ Rápida de entrenar
- ✅ Buena para problemas de clasificación binaria (Sí/No)
- ✅ Genera probabilidades (no solo Sí/No)

**Preprocesamiento:**
1. **Variables numéricas**: Imputación de nulos (mediana) + Escalado (StandardScaler)
2. **Variables categóricas**: Imputación de nulos (moda) + One-Hot Encoding
3. **Validación**: GroupShuffleSplit por ServicioID (evita data leakage)
   - 75% entrenamiento
   - 25% prueba
   - Los mismos servicios NO aparecen en ambos conjuntos

---

## 📊 5. Métricas de Evaluación

### 5.1 Métricas Principales

| Métrica | Valor Ejemplo | Rango | Interpretación |
|---------|---------------|-------|----------------|
| **Accuracy** | 0.845 | 0-1 | 84.5% de predicciones correctas en general |
| **ROC-AUC** | 0.962 | 0-1 | 96.2% de capacidad para discriminar entre clases (excelente) |
| **Recall (Stockout=1)** | 0.974 | 0-1 | 97.4% de roturas reales fueron detectadas por el modelo |

### 5.2 Explicación Detallada de Métricas

#### Accuracy (Exactitud)
**Fórmula**: `(VP + VN) / Total`

**Interpretación**: Porcentaje de predicciones correctas sobre el total.

**Ejemplo con 0.845**:
De cada 100 predicciones, el modelo acierta 84-85 veces.

**Limitación**: Puede ser engañosa si las clases están desbalanceadas (ej: 90% clase 0, 10% clase 1).

---

#### ROC-AUC (Area Under the Curve)
**Rango**: 0 a 1 (1 es perfecto)

**Interpretación**: Capacidad del modelo para distinguir entre las dos clases.

| Valor | Interpretación |
|-------|----------------|
| 0.90 - 1.00 | Excelente |
| 0.80 - 0.90 | Muy bueno |
| 0.70 - 0.80 | Bueno |
| 0.50 - 0.70 | Regular |
| < 0.50 | Malo (peor que azar) |

**Ejemplo con 0.962**:
El modelo tiene un 96.2% de probabilidad de clasificar correctamente un caso de rotura vs. un caso sin rotura. ¡Excelente!

---

#### Recall (Sensibilidad)
**Fórmula**: `VP / (VP + FN)`

**Interpretación**: De todas las roturas reales, ¿cuántas detectó el modelo?

**Ejemplo con 0.974**:
Si hubo 100 roturas reales, el modelo detectó 97 de ellas (solo se le escaparon 3).

**¿Por qué es importante?**
En este problema, es **crítico** detectar las roturas para evitar pérdidas. Es mejor tener una "falsa alarma" que perder una rotura real.

---

### 5.3 Matriz de Confusión

La matriz muestra cómo se distribuyen las predicciones vs. la realidad:

```
                 Predicho
                 No (0)    Sí (1)
Real  No (0)  │   359   │   89   │  = 448 casos sin rotura
      Sí (1)  │    4    │  148   │  = 152 casos con rotura
```

**Interpretación de cada celda:**

| Celda | Nombre | Valor | Significado |
|-------|--------|-------|-------------|
| **Real_0, Pred_0** | Verdadero Negativo (VN) | 359 | Correctamente predijo "No rotura" |
| **Real_0, Pred_1** | Falso Positivo (FP) | 89 | Predijo "Rotura" pero NO pasó (falsa alarma) |
| **Real_1, Pred_0** | Falso Negativo (FN) | 4 | Predijo "No rotura" pero SÍ pasó (❌ peligroso) |
| **Real_1, Pred_1** | Verdadero Positivo (VP) | 148 | Correctamente predijo "Rotura" |

**Análisis del ejemplo:**
- ✅ Solo 4 roturas se escaparon (Falsos Negativos) → **Excelente Recall**
- ⚠️ 89 falsas alarmas (Falsos Positivos) → Costo de ser precavido
- El modelo es **conservador**: Prefiere alertar de más que de menos

---

### 5.4 Reporte de Clasificación

Muestra métricas detalladas por clase:

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **0 (No rotura)** | 0.989 | 0.801 | 0.885 | 448 |
| **1 (Rotura)** | 0.624 | 0.974 | 0.761 | 152 |

#### Explicación de columnas:

**Precision (Precisión)**
- Fórmula: `VP / (VP + FP)`
- Interpretación: De todas las predicciones de "Rotura", ¿cuántas fueron correctas?
- Clase 1 = 0.624: De cada 10 alertas de rotura, 6 son correctas y 4 son falsas alarmas

**Recall (Exhaustividad)**
- Ya explicado arriba
- Clase 1 = 0.974: Detecta 97.4% de las roturas reales

**F1-Score**
- Fórmula: `2 × (Precision × Recall) / (Precision + Recall)`
- Interpretación: Media armónica entre Precision y Recall (balance)
- Clase 1 = 0.761: Buen equilibrio entre detectar roturas y evitar falsas alarmas

**Support**
- Cantidad de casos reales de esa clase en el conjunto de prueba
- Clase 0 = 448 casos sin rotura
- Clase 1 = 152 casos con rotura

---

## 🔮 6. Predicción (Pestaña "Predicción")

### Modo 1: Usar un caso del dataset

Seleccionas un ServicioID y Periodo específico, y el modelo predice la probabilidad de rotura basándose en los datos reales de ese snapshot.

**Uso**: Validar el modelo con casos conocidos.

---

### Modo 2: Ingresar valores (formulario)

Puedes ajustar manualmente los valores operacionales para simular diferentes escenarios:

| Campo | Descripción | Ejemplo |
|-------|-------------|---------|
| **ServicioID** | Selecciona servicio base para cargar valores sugeridos | SRV-001 |
| **Periodo** | Selecciona período base | 5 |
| **StockActual** | Cantidad disponible HOY | 50 unidades |
| **DemandaDiariaEst** | Consumo diario estimado | 8.5 unidades/día |
| **DiasHastaRecepcion** | Días hasta que llegue pedido | 12 días |
| **RecepcionPendiente** | Cantidad en camino | 100 unidades |
| **Horizonte** | Ventana de predicción | 14 días |

**Resultado**: El modelo devuelve:
1. **Probabilidad de rotura** (0-100%)
2. **Mensaje de riesgo**:
   - **ALTO** (≥70%): Generar reabastecimiento inmediato
   - **MEDIO** (40-70%): Monitoreo diario
   - **BAJO** (<40%): Operación normal

---

## 🎯 7. Aplicación en el TO-BE (Proceso Mejorado)

### Integración del Modelo en Operaciones WMS

1. **Dashboard de Alertas**: El modelo se ejecuta diariamente y genera alertas automáticas
2. **Priorización de Compras**: Los casos con probabilidad >70% se priorizan
3. **Optimización de Inventario**: Permite reducir stock de seguridad sin riesgo
4. **Negociación con Proveedores**: Evidencia cuantitativa para negociar mejores lead times
5. **KPIs Mejorados**:
   - Reducción de roturas de stock: objetivo -50%
   - Reducción de inventario excesivo: objetivo -20%
   - Mejora en cumplimiento de SLA: objetivo +15%

---

## 🔬 8. Limitaciones y Mejoras Futuras

### Limitaciones Actuales

1. **Dataset sintético**: Aunque alineado con MDM v3, los datos no son reales
2. **Variables temporales simples**: No considera estacionalidad compleja (Navidad, campañas)
3. **Proveedores determinísticos**: La asignación de proveedores es simplificada
4. **Sin factores externos**: No incluye eventos (huelgas, desastres naturales, cambios de precio)

### Mejoras Propuestas

1. ✨ **Datos reales**: Conectar con sistema WMS real vía API
2. 📈 **Series temporales**: Implementar LSTM o Prophet para capturar estacionalidad
3. 🔄 **Re-entrenamiento automático**: Pipeline diario con nuevos datos
4. 🌍 **Variables externas**: Integrar días feriados, clima, eventos especiales
5. 📊 **Optimización de hiperparámetros**: GridSearch para mejorar performance
6. 🤖 **Ensemble models**: Combinar XGBoost + RandomForest + LogReg

---

## 📚 9. Requisitos Técnicos

### Dependencias (requirements.txt)

```
streamlit       → Framework del dashboard
pandas          → Manipulación de datos
numpy           → Operaciones numéricas
scikit-learn    → Algoritmos ML y métricas
joblib          → Serialización de modelos
openpyxl        → Lectura de archivos Excel
```

### Ejecución

```bash
# Entrenar modelo
python scripts/train_model.py

# Ejecutar dashboard
streamlit run app.py
```

---

## ✅ 10. Conclusión

Este sistema cumple con los requisitos académicos:

- ✅ **Modelo BA/IA**: Regresión Logística para clasificación binaria
- ✅ **Problema concreto**: Predicción de rotura de stock en WMS
- ✅ **Dataset >500 registros**: 2,400 registros generados
- ✅ **Alineado con MDM v3**: Maestros consistentes con modelo de datos
- ✅ **Documentación completa**: Variables, técnica, resultados, limitaciones
- ✅ **Integración TO-BE**: Dashboard operativo para decisiones diarias

**Métricas destacadas:**
- Accuracy: 84.5%
- ROC-AUC: 96.2% (excelente discriminación)
- Recall: 97.4% (detecta casi todas las roturas)

El modelo está listo para ser utilizado en un entorno operativo real con ajustes menores.

