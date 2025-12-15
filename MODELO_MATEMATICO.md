# 🧮 Modelo Matemático: Regresión Logística para Predicción de Stockout

## 📖 Índice
1. [Nombre y Tipo del Modelo](#1-nombre-y-tipo-del-modelo)
2. [¿Qué Predice Exactamente?](#2-qué-predice-exactamente)
3. [Fundamento Matemático](#3-fundamento-matemático)
4. [Variables Utilizadas](#4-variables-utilizadas)
5. [Proceso de Entrenamiento](#5-proceso-de-entrenamiento)
6. [Por Qué Este Modelo](#6-por-qué-este-modelo)
7. [Cálculo de Predicción](#7-cálculo-de-predicción)

---

## 1. Nombre y Tipo del Modelo

### Nombre Técnico
**Regresión Logística Binaria (Binary Logistic Regression)**

### Clasificación
- **Familia**: Modelos lineales generalizados (GLM - Generalized Linear Models)
- **Tipo**: Clasificación supervisada binaria
- **Clase en scikit-learn**: `sklearn.linear_model.LogisticRegression`

### Especificaciones del Modelo Implementado

```python
LogisticRegression(
    max_iter=2000,           # Máximo de iteraciones para convergencia
    class_weight='balanced',  # Compensa desbalance de clases
    C=0.5,                   # Regularización L2 (inversa de lambda)
    solver='liblinear'       # Algoritmo de optimización
)
```

---

## 2. ¿Qué Predice Exactamente?

### Definición Precisa

El modelo predice la **probabilidad** de que ocurra un evento binario:

**Variable objetivo (Target)**: `Stockout14d`

```
Stockout14d = {
    1  si habrá rotura de stock en los próximos 14 días
    0  si NO habrá rotura de stock en los próximos 14 días
}
```

### Salida del Modelo

El modelo NO predice directamente 0 o 1, sino **dos probabilidades**:

```
P(Stockout14d = 0) = probabilidad de NO rotura    (ej: 0.23 = 23%)
P(Stockout14d = 1) = probabilidad de SÍ rotura    (ej: 0.77 = 77%)
                                                  └─────────┘
                                                   Suma = 100%
```

**Usamos la segunda**: P(Stockout14d = 1)

### Interpretación

**Ejemplo:**
```
Input: ServicioID = SRV-045, Periodo = 7
       StockActual = 120 unidades
       DemandaDiariaEst = 15.3 unidades/día
       DiasHastaRecepcion = 12 días
       ... (32 variables más)

Output del modelo:
       P(Stockout14d = 1) = 0.89 = 89%
```

**Significado**: Hay un 89% de probabilidad de que el servicio SRV-045 se quede sin stock antes de 14 días.

---

## 3. Fundamento Matemático

### 3.1 Fórmula de la Regresión Logística

La regresión logística modela la probabilidad mediante la **función logística (sigmoide)**:

```
P(y = 1 | X) = 1 / (1 + e^(-z))
```

Donde:
- `P(y = 1 | X)` = Probabilidad de rotura dado el vector de características X
- `e` = Número de Euler (≈ 2.71828)
- `z` = Combinación lineal de las variables

### 3.2 Combinación Lineal (z)

```
z = β₀ + β₁·x₁ + β₂·x₂ + β₃·x₃ + ... + β₃₅·x₃₅
```

Donde:
- `β₀` = Intercepto (sesgo base del modelo)
- `β₁, β₂, ..., β₃₅` = Coeficientes (pesos) de cada variable
- `x₁, x₂, ..., x₃₅` = Valores de las 35 variables

### 3.3 Ejemplo Numérico Simplificado

Supongamos un modelo simplificado con solo 4 variables:

```
z = 2.5 + (-0.03)·StockActual + (0.15)·DemandaDiariaEst 
        + (0.08)·DiasHastaRecepcion + (-0.02)·RecepcionPendiente

Caso específico:
  StockActual = 120
  DemandaDiariaEst = 15.3
  DiasHastaRecepcion = 12
  RecepcionPendiente = 200

z = 2.5 + (-0.03)·120 + (0.15)·15.3 + (0.08)·12 + (-0.02)·200
z = 2.5 - 3.6 + 2.295 + 0.96 - 4.0
z = -1.845

P(Stockout = 1) = 1 / (1 + e^(1.845))
P(Stockout = 1) = 1 / (1 + 6.33)
P(Stockout = 1) = 1 / 7.33
P(Stockout = 1) = 0.136 = 13.6%
```

**Interpretación**:
- Stock alto (120) → reduce z (coeficiente negativo)
- Demanda alta (15.3) → aumenta z (coeficiente positivo)
- Días hasta recepción largos (12) → aumenta z
- Recepción pendiente alta (200) → reduce z

Resultado: 13.6% de probabilidad de rotura (BAJO RIESGO)

### 3.4 Función Sigmoide Visualizada

```
Probabilidad
    1.0 ┤              ╭────────
        │            ╭─╯
    0.75┤         ╭──╯
        │       ╭─╯
    0.5 ┤    ╭──╯
        │  ╭─╯
    0.25┤╭─╯
        │╯
    0.0 ┴─────────────────────────> z
       -6  -3   0   3   6
```

**Propiedades importantes**:
- Si `z = 0` → P = 0.5 (50%)
- Si `z → +∞` → P → 1.0 (100%)
- Si `z → -∞` → P → 0.0 (0%)
- La curva es suave (no hay saltos bruscos)

---

## 4. Variables Utilizadas

### 4.1 Total de Variables

El modelo usa **35 variables de entrada** para predecir **1 variable de salida**.

```
X (entrada) = [x₁, x₂, x₃, ..., x₃₅]  →  MODELO  →  y (salida) = Stockout14d
```

### 4.2 Lista Completa de Variables de Entrada

#### A) Características del Servicio (15 variables)

| # | Variable | Tipo | Ejemplo | Influencia |
|---|----------|------|---------|------------|
| 1 | Categoria | Categórica | "Almacenaje" | Baja |
| 2 | Subcategoria | Categórica | "Refrigerado" | Baja |
| 3 | UnidadTarifa | Categórica | "Por pallet" | Baja |
| 4 | TipoUnidad | Categórica | "Pallet" | Baja |
| 5 | Moneda | Categórica | "PEN" | Baja |
| 6 | RequiereCertificacion | Categórica | "Sí" | Baja |
| 7 | Temperatura | Categórica | "Frio" | Baja |
| 8 | **LeadTimeMinDias** | Numérica | 5 | Media |
| 9 | **LeadTimeMaxDias** | Numérica | 10 | Media |
| 10 | TiempoEjecucionHoras | Numérica | 24 | Baja |
| 11 | ModalidadContrato | Categórica | "Mensual" | Baja |
| 12 | Estado | Categórica | "Activo" | Baja |
| 13 | CantidadPedidoEstandar | Numérica | 250 | Media |
| 14 | CostoEstandar | Numérica | 150.00 | Baja |
| 15 | TarifaImpuesto | Numérica | 0.18 | Baja |

#### B) Características del Cliente (4 variables)

| # | Variable | Tipo | Ejemplo | Influencia |
|---|----------|------|---------|------------|
| 16 | TemperaturaControlada | Categórica | "Sí" | Baja |
| 17 | CaducidadControlada | Categórica | "No" | Baja |
| 18 | SLA_horas | Numérica | 24 | Baja |
| 19 | SLA_pct | Numérica | 95 | Baja |

#### C) Datos del Cliente Propietario (4 variables)

| # | Variable | Tipo | Ejemplo | Influencia |
|---|----------|------|---------|------------|
| 20 | **Segmento** | Categórica | "PREFERENTE" | Media |
| 21 | CanalPreferido | Categórica | "Directo" | Baja |
| 22 | ZonaDespacho | Categórica | "Norte" | Baja |
| 23 | Departamento | Categórica | "Lima" | Baja |

#### D) Características del Proveedor (6 variables)

| # | Variable | Tipo | Ejemplo | Influencia |
|---|----------|------|---------|------------|
| 24 | Categoria_prov | Categórica | "LOGISTICA" | Baja |
| 25 | **LeadTimePromedioDias** | Numérica | 7 | Alta |
| 26 | ToleranciaEntregaDias | Numérica | 2 | Media |
| 27 | **RatingDesempeno** | Numérica | 4.5 | Media |
| 28 | CertificadoCalidad | Categórica | "ISO9001" | Baja |
| 29 | Estado_prov | Categórica | "Activo" | Baja |

#### E) Variables Operacionales (5 variables) ⭐

| # | Variable | Tipo | Ejemplo | Influencia |
|---|----------|------|---------|------------|
| 30 | Periodo | Numérica | 7 | Baja |
| 31 | **StockActual** | Numérica | 120 | **MUY ALTA** |
| 32 | **RecepcionPendiente** | Numérica | 200 | Alta |
| 33 | **DiasHastaRecepcion** | Numérica | 12 | **MUY ALTA** |
| 34 | **DemandaDiariaEst** | Numérica | 15.3 | **MUY ALTA** |

#### F) Variable NO usada como entrada (es el target)

| Variable | Tipo | Valores | Rol |
|----------|------|---------|-----|
| **Stockout14d** | Binaria | 0 o 1 | **Variable de salida (y)** |

### 4.3 Ranking de Importancia de Variables

**Top 10 variables más influyentes** (estimado basado en lógica del modelo):

| Ranking | Variable | Peso Estimado | Razón |
|---------|----------|---------------|-------|
| 1 🥇 | **StockActual** | 35% | Determina directamente los días de cobertura |
| 2 🥈 | **DemandaDiariaEst** | 30% | Determina velocidad de consumo |
| 3 🥉 | **DiasHastaRecepcion** | 25% | Determina cuándo llega el reabastecimiento |
| 4 | **RecepcionPendiente** | 8% | Modifica el stock futuro |
| 5 | **LeadTimePromedioDias** | 4% | Afecta planificación de pedidos |
| 6 | **Segmento** | 3% | Afecta patrones de demanda |
| 7 | **RatingDesempeno** | 2% | Indica confiabilidad del proveedor |
| 8 | **LeadTimeMaxDias** | 1.5% | Define límite superior de espera |
| 9 | **CantidadPedidoEstandar** | 1% | Afecta tamaño de reabastecimiento |
| 10 | **ToleranciaEntregaDias** | 0.5% | Variabilidad del proveedor |

**Resto de variables**: < 0.5% cada una (contexto marginal)

---

## 5. Proceso de Entrenamiento

### 5.1 Preparación de Datos

#### Paso 1: Carga de Maestros
```
Entrada:
  - maestro_clientes.xlsx    (109 registros)
  - maestro_proveedores.xlsx (204 registros)
  - maestro_servicios.xlsx   (200 registros)

Limpieza:
  - Normalización de strings (trim, lowercase)
  - Eliminación de duplicados por ID
  - Conversión de tipos (numéricos, fechas)
```

#### Paso 2: Generación del Dataset Transaccional
```
Proceso:
  1. Join: Servicios ← Clientes (por ClientePropietario)
  2. Join: Servicios ← Proveedores (por asignación determinística)
  3. Expansión: 200 servicios × 12 períodos = 2,400 registros
  4. Generación de variables derivadas:
     - StockActual (basado en fórmula estacional)
     - DemandaDiariaEst (basado en segmento y cantidad estándar)
     - DiasHastaRecepcion (basado en lead time + tolerancia)
     - RecepcionPendiente (basado en lógica de reorden)

Output:
  Dataset con 2,400 filas × 36 columnas (35 features + 1 target)
```

#### Paso 3: Cálculo del Target (Stockout14d)
```python
# Lógica del target
dias_cobertura = StockActual / DemandaDiariaEst

Stockout14d = 1 si:
  (dias_cobertura < 14) Y (DiasHastaRecepcion > dias_cobertura)
  
Stockout14d = 0 en caso contrario
```

**Ejemplo:**
```
Caso 1:
  StockActual = 100
  DemandaDiariaEst = 10
  DiasHastaRecepcion = 15
  
  dias_cobertura = 100 / 10 = 10 días
  10 < 14 → SÍ
  15 > 10 → SÍ
  → Stockout14d = 1 (habrá rotura)

Caso 2:
  StockActual = 200
  DemandaDiariaEst = 10
  DiasHastaRecepcion = 8
  
  dias_cobertura = 200 / 10 = 20 días
  20 < 14 → NO
  → Stockout14d = 0 (no habrá rotura)
```

### 5.2 Preprocesamiento

El modelo NO recibe los datos crudos, sino transformados:

#### A) Variables Numéricas (14 variables)
```
Paso 1: Imputación de nulos
  - Estrategia: Mediana
  - Ejemplo: Si LeadTimePromedioDias tiene nulos, rellena con la mediana (ej: 7 días)

Paso 2: Escalado (Standardization)
  - Fórmula: x_scaled = (x - μ) / σ
  - μ = media de la variable
  - σ = desviación estándar
  
  Ejemplo:
    StockActual original: [50, 100, 150, 200, 250]
    μ = 150, σ = 70.7
    StockActual escalado: [-1.41, -0.71, 0, 0.71, 1.41]
```

**¿Por qué escalar?**
- Variables en diferentes escalas (ej: StockActual en cientos, DemandaDiariaEst en decenas)
- El modelo converge más rápido
- Todos los coeficientes están en escala comparable

#### B) Variables Categóricas (21 variables)
```
Paso 1: Imputación de nulos
  - Estrategia: Moda (valor más frecuente)

Paso 2: One-Hot Encoding
  - Convierte categorías en columnas binarias (0 o 1)
  
  Ejemplo:
    Segmento original: ["BASICO", "ESTANDAR", "PREFERENTE", "BASICO"]
    
    One-Hot Encoding →
      Segmento_BASICO    [1, 0, 0, 1]
      Segmento_ESTANDAR  [0, 1, 0, 0]
      Segmento_PREFERENTE[0, 0, 1, 0]
```

**Resultado final del preprocesamiento:**
- Variables numéricas: 14 columnas escaladas
- Variables categóricas: ~80 columnas binarias (depende de categorías únicas)
- **Total de features después de preprocesamiento**: ~94 columnas

### 5.3 División de Datos (Train/Test Split)

```
Estrategia: GroupShuffleSplit
  - Grupo: ServicioID
  - Train: 75% de los servicios
  - Test: 25% de los servicios
  - Random state: 42 (para reproducibilidad)

Dataset total: 2,400 registros
  → Train: 1,800 registros (~150 servicios × 12 períodos)
  → Test:    600 registros (~50 servicios × 12 períodos)
```

**¿Por qué GroupShuffleSplit?**

Evita **data leakage** (fuga de información):
- Si el servicio SRV-045 está en train, TODOS sus 12 períodos están en train
- Si está en test, TODOS sus 12 períodos están en test
- Nunca un servicio tiene datos en ambos conjuntos

**Comparación:**

```
❌ Split normal (malo):
  Train: SRV-001 períodos [1,2,3,4,5,6,7,8,9]
  Test:  SRV-001 períodos [10,11,12]
  → El modelo "conoce" SRV-001 y solo predice períodos futuros

✅ GroupShuffleSplit (correcto):
  Train: SRV-001, SRV-002, SRV-003, ... (150 servicios completos)
  Test:  SRV-151, SRV-152, ... (50 servicios completos)
  → El modelo predice servicios completamente nuevos
```

### 5.4 Entrenamiento del Modelo

#### Algoritmo de Optimización: liblinear

El modelo busca los mejores coeficientes (β₀, β₁, ..., β₃₅) mediante:

```
Objetivo: Minimizar la función de costo (Log-Loss)

Log-Loss = -1/n Σ [y·log(ŷ) + (1-y)·log(1-ŷ)]

Donde:
  n = número de registros (1,800)
  y = valor real (0 o 1)
  ŷ = probabilidad predicha (0 a 1)
```

**Interpretación del Log-Loss:**
- Penaliza predicciones incorrectas
- Si y=1 y ŷ=0.1 → Log-Loss alto (mala predicción)
- Si y=1 y ŷ=0.9 → Log-Loss bajo (buena predicción)

#### Regularización L2 (C = 0.5)

```
Función de costo completa:
  J = Log-Loss + λ·Σ(βᵢ²)
  
λ = 1/C = 1/0.5 = 2 (parámetro de regularización)
```

**¿Qué hace la regularización?**
- Penaliza coeficientes muy grandes
- Evita overfitting (sobreajuste)
- Hace que el modelo generalice mejor a datos nuevos

#### Class Weight Balancing

```
Distribución desbalanceada:
  Clase 0 (No rotura): 1,873 registros (78%)
  Clase 1 (Sí rotura):   527 registros (22%)

class_weight='balanced' ajusta pesos:
  w₀ = n / (2 · n₀) = 2,400 / (2 · 1,873) = 0.64
  w₁ = n / (2 · n₁) = 2,400 / (2 · 527) = 2.28
```

**Efecto:**
- Errores en clase 1 (rotura) se penalizan 2.28 veces más
- Fuerza al modelo a prestar más atención a la clase minoritaria
- Mejora el Recall (detectar roturas)

### 5.5 Proceso Iterativo

```
Iteración 1:
  - Inicializar β con valores aleatorios
  - Calcular predicciones
  - Calcular Log-Loss
  - Ajustar β usando gradiente descendente

Iteración 2:
  - Calcular predicciones con β actualizados
  - Calcular Log-Loss (debería ser menor)
  - Ajustar β nuevamente

...

Iteración 487:
  - Log-Loss converge (cambios < 0.0001)
  - ¡Entrenamiento completo!

Total iteraciones: ~500 (max 2,000 permitidas)
Tiempo: ~2 segundos en CPU moderna
```

### 5.6 Guardado del Modelo

```python
import joblib

# Guarda TODA la pipeline (preprocesamiento + modelo)
joblib.dump(pipeline, "models/stockout14d_logreg.joblib")
```

**El archivo .joblib contiene:**
1. Imputadores (medianas y modas aprendidas)
2. Scaler (medias y desviaciones estándar)
3. One-Hot Encoder (categorías conocidas)
4. Modelo de Regresión Logística (coeficientes β)

**Tamaño del archivo**: ~220 KB (comprimido)

---

## 6. Por Qué Este Modelo

### 6.1 Ventajas de la Regresión Logística

#### ✅ 1. Interpretabilidad
```
Coeficientes tienen significado directo:
  β = 0.15 para DemandaDiariaEst
  → Por cada unidad adicional de demanda, el log-odds aumenta 0.15
  → Mayor demanda → Mayor riesgo (intuitivo)
```

#### ✅ 2. Probabilidades Calibradas
```
El modelo no solo dice "Sí/No", sino "89% de probabilidad"
  → Permite tomar decisiones basadas en riesgo
  → Operador puede priorizar casos críticos (>70%)
```

#### ✅ 3. Eficiencia Computacional
```
Entrenamiento: 2 segundos
Predicción: <1 milisegundo por registro
  → Puede correr en producción en tiempo real
  → No requiere GPU
```

#### ✅ 4. Estabilidad
```
Pocos hiperparámetros (C, solver, max_iter)
  → Menos propensión a overfitting
  → Resultados reproducibles
```

#### ✅ 5. Baseline Robusto
```
Es el modelo estándar para clasificación binaria
  → Si falla, otros modelos más complejos también fallarán
  → Punto de partida académicamente aceptado
```

### 6.2 Comparación con Otros Modelos

| Modelo | Accuracy | Interpretabilidad | Velocidad | Complejidad |
|--------|----------|-------------------|-----------|-------------|
| **Regresión Logística** | 84.5% | ⭐⭐⭐⭐⭐ | ⚡⚡⚡⚡⚡ | Baja |
| Random Forest | ~87% (est.) | ⭐⭐⭐ | ⚡⚡⚡ | Media |
| XGBoost | ~89% (est.) | ⭐⭐ | ⚡⚡ | Alta |
| Red Neuronal | ~85% (est.) | ⭐ | ⚡⚡ | Muy Alta |
| Naive Bayes | ~78% (est.) | ⭐⭐⭐⭐ | ⚡⚡⚡⚡⚡ | Muy Baja |

**Conclusión**: 
- Regresión Logística ofrece el mejor balance interpretabilidad/performance
- Para MVP académico es la elección correcta
- Si se requiere más accuracy, Random Forest o XGBoost son siguientes pasos

### 6.3 Desventajas y Limitaciones

#### ❌ 1. Asume Linealidad
```
El modelo asume que el log-odds es una función lineal de X
  → No captura interacciones complejas automáticamente
  → Ej: No capta que "Stock bajo + Proveedor lento" es peor que la suma
```

#### ❌ 2. Sensible a Outliers
```
Un stock de 10,000 unidades (outlier) puede distorsionar el coeficiente
  → Requiere limpieza de datos cuidadosa
```

#### ❌ 3. No Captura Estacionalidad Compleja
```
El modelo trata "Periodo" como un número (1, 2, 3, ...)
  → No entiende que Diciembre (12) es temporada alta
  → Requiere feature engineering manual
```

---

## 7. Cálculo de Predicción (Paso a Paso)

### Ejemplo Completo

#### Entrada del Usuario
```
ServicioID: SRV-089
Periodo: 7
StockActual: 180
DemandaDiariaEst: 25.0
DiasHastaRecepcion: 14
RecepcionPendiente: 250
... (30 variables estructurales del servicio)
```

#### Paso 1: Preprocesamiento Automático

**Variables numéricas escaladas:**
```
StockActual_scaled = (180 - 220) / 85 = -0.47
DemandaDiariaEst_scaled = (25.0 - 15.2) / 8.3 = 1.18
DiasHastaRecepcion_scaled = (14 - 10.5) / 6.2 = 0.56
RecepcionPendiente_scaled = (250 - 200) / 90 = 0.56
... (10 variables numéricas más)
```

**Variables categóricas codificadas:**
```
Categoria_Almacenaje = 1
Categoria_Transporte = 0
Segmento_PREFERENTE = 1
Segmento_BASICO = 0
... (80 variables binarias)
```

#### Paso 2: Multiplicación por Coeficientes

```
z = β₀ + Σ(βᵢ · xᵢ)

z = 0.45                           (intercepto)
  + (-2.1) · (-0.47)              (StockActual: +0.99)
  + (1.8) · (1.18)                (DemandaDiariaEst: +2.12)
  + (1.2) · (0.56)                (DiasHastaRecepcion: +0.67)
  + (-0.9) · (0.56)               (RecepcionPendiente: -0.50)
  + (0.3) · 1                     (Categoria_Almacenaje: +0.30)
  + (0.5) · 1                     (Segmento_PREFERENTE: +0.50)
  + ... (suma de otros 88 términos)

z = 2.35 (suma total)
```

#### Paso 3: Aplicar Función Sigmoide

```
P(Stockout = 1) = 1 / (1 + e^(-z))
P(Stockout = 1) = 1 / (1 + e^(-2.35))
P(Stockout = 1) = 1 / (1 + 0.095)
P(Stockout = 1) = 1 / 1.095
P(Stockout = 1) = 0.913

→ 91.3% de probabilidad de rotura
```

#### Paso 4: Clasificación (si se requiere)

```
Umbral de decisión: 0.5 (50%)

Si P(Stockout = 1) >= 0.5 → Clasificar como 1 (habrá rotura)
Si P(Stockout = 1) < 0.5  → Clasificar como 0 (no habrá rotura)

En este caso: 0.913 >= 0.5 → Predicción = 1 (SÍ ROTURA)
```

#### Paso 5: Mensaje al Usuario

```
Probabilidad: 91.3%
Nivel de Riesgo: ALTO

Acción sugerida:
  → Generar reabastecimiento inmediato
  → Priorizar recepción con proveedor
```

---

## 8. Fórmula Matemática Completa

### Notación Formal

```
Dado:
  X = [x₁, x₂, ..., x₃₅]ᵀ  (vector de 35 features)
  β = [β₀, β₁, ..., β₃₅]ᵀ  (vector de 36 parámetros)
  
Modelo:
  z = βᵀ · X = β₀ + Σᵢ₌₁³⁵ βᵢ·xᵢ
  
  P(y = 1 | X) = σ(z) = 1 / (1 + e^(-z))
  
Donde:
  σ(z) = función sigmoide
  y ∈ {0, 1} = variable objetivo (Stockout14d)
```

### Función de Pérdida (Loss Function)

```
Durante entrenamiento:
  
  L(β) = -1/n Σⁿⱼ₌₁ [yⱼ·log(P(y=1|Xⱼ)) + (1-yⱼ)·log(1-P(y=1|Xⱼ))]
         + λ·Σᵢ₌₁³⁵ βᵢ²
         └─────────────────────────────────────────────────────────┘   └────┘
                        Log-Loss                                     Regularización L2
                        
Donde:
  n = 1,800 (tamaño del conjunto de entrenamiento)
  λ = 2 (parámetro de regularización)
  yⱼ = valor real del registro j (0 o 1)
```

### Gradiente Descendente (Optimización)

```
Actualización iterativa:
  
  β ← β - α · ∇L(β)
  
Donde:
  α = learning rate (tasa de aprendizaje)
  ∇L(β) = gradiente de la función de pérdida
  
  ∇L(β) = 1/n · XᵀΣⁿⱼ₌₁(σ(βᵀXⱼ) - yⱼ) + 2λβ
```

---

## 9. Resumen Ejecutivo

### ¿Qué Modelo Es?
**Regresión Logística Binaria** con regularización L2 y balanceo de clases.

### ¿Qué Predice?
**Probabilidad de rotura de stock en 14 días** (0% a 100%).

### ¿Con Qué Variables?
**35 variables de entrada**:
- 5 operacionales (stock, demanda, recepción, días, periodo)
- 30 estructurales (características del servicio, cliente, proveedor)

### ¿En Base a Qué?
**Patrones aprendidos** de 1,800 registros históricos que relacionan:
```
Stock bajo + Demanda alta + Recepción tardía → Alta probabilidad de rotura
Stock alto + Demanda baja + Recepción pronta → Baja probabilidad de rotura
```

### ¿Por Qué Este Modelo?
- ✅ Interpretable
- ✅ Rápido
- ✅ Calibrado (probabilidades confiables)
- ✅ Baseline académico estándar
- ✅ 84.5% accuracy, 96.2% AUC

### Fórmula Simplificada
```
P(Rotura) = 1 / (1 + e^(-(a + b·Stock + c·Demanda + d·DiasRec + ...)))
```

Donde `a, b, c, d, ...` son coeficientes aprendidos durante el entrenamiento.

---

**Última actualización**: Diciembre 2024  
**Versión del modelo**: stockout14d_logreg v1.0  
**Framework**: scikit-learn 1.3+  
**Python**: 3.8+

