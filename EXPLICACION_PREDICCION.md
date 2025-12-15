# 🔮 Explicación Detallada: Sistema de Predicción de Rotura de Stock

## 📖 Índice
1. [Archivos Utilizados](#1-archivos-utilizados)
2. [Modo 1: Predicción desde Dataset](#2-modo-1-predicción-desde-dataset)
3. [Modo 2: Formulario Manual](#3-modo-2-formulario-manual)
4. [Variables Operacionales vs. Estructurales](#4-variables-operacionales-vs-estructurales)
5. [Interpretación de Resultados](#5-interpretación-de-resultados)
6. [Flujo de Decisiones](#6-flujo-de-decisiones)

---

## 1. Archivos Utilizados

### Archivos de Entrada
El sistema utiliza los **3 maestros** en formato Excel:

```
data/
  ├── maestro_clientes.xlsx      → 109 clientes con segmentos, zonas, canales
  ├── maestro_proveedores.xlsx   → 204 proveedores con lead times y ratings
  └── maestro_servicios.xlsx     → 200 servicios/productos del WMS
```

### Archivos Generados por el Modelo

```
models/
  ├── stockout14d_logreg.joblib  → Modelo entrenado (algoritmo + transformaciones)
  └── metrics.json               → Métricas de evaluación (accuracy, AUC, etc.)
```

### ¿Cómo se usa el archivo del modelo?

El archivo `.joblib` contiene:
- El pipeline completo de preprocesamiento (imputación, escalado, one-hot encoding)
- El modelo de Regresión Logística entrenado con 1,800 registros (75% del dataset)
- Los pesos aprendidos para cada variable

Cuando haces una predicción:
```python
modelo.predict_proba(datos_nuevos)
```

El modelo devuelve **2 probabilidades** que suman 100%:
- Probabilidad de clase 0 (No rotura): ej. 0.25 (25%)
- Probabilidad de clase 1 (Sí rotura): ej. 0.75 (75%)

**Usamos la probabilidad de clase 1** porque eso es lo que nos interesa: el riesgo de rotura.

---

## 2. Modo 1: Predicción desde Dataset

### 2.1 ¿Qué hace este modo?

Toma un **snapshot histórico real** del dataset y predice si habría rotura. Es útil para:
- Validar el modelo con casos conocidos
- Entender cómo el modelo interpreta diferentes escenarios
- Auditar decisiones históricas

### 2.2 ¿Qué es ServicioID?

El **ServicioID** es el identificador único de cada producto/servicio en el WMS.

**Ejemplo de servicios reales:**
- `SRV-001`: Almacenaje de productos refrigerados
- `SRV-045`: Transporte de carga pesada
- `SRV-120`: Picking y packing para e-commerce
- `SRV-189`: Gestión de inventario farmacéutico

Cada servicio tiene características fijas como:
- Categoría (Almacenaje, Distribución, Transporte, etc.)
- Lead time mínimo/máximo
- Requisitos especiales (temperatura, certificación)
- Cliente propietario

### 2.3 ¿Qué es el Periodo?

El **Periodo** representa un momento específico en el tiempo (snapshot).

**Analogía**: Es como tomar una fotografía del inventario en diferentes momentos.

```
Periodo 1  → Enero 2024
Periodo 2  → Febrero 2024
Periodo 3  → Marzo 2024
...
Periodo 12 → Diciembre 2024
```

**¿Por qué hay 12 periodos por defecto?**
Porque simula 1 año completo de operación (12 meses).

**¿Por qué es importante?**
Porque el mismo servicio puede tener diferente comportamiento en diferentes momentos:

| Periodo | Stock | Demanda | ¿Rotura? |
|---------|-------|---------|----------|
| 1 (Enero) | 500 | 20/día | NO |
| 6 (Junio - temporada alta) | 500 | 50/día | SÍ |
| 12 (Diciembre) | 800 | 40/día | NO |

### 2.4 ¿Qué muestra la fila seleccionada?

Cuando seleccionas `ServicioID = SRV-045` y `Periodo = 7`, el sistema muestra **todas las variables** de ese snapshot:

#### Columnas mostradas (ejemplo):

```
ServicioID: SRV-045
NombreServicio: Transporte terrestre zona norte
Periodo: 7

=== CARACTERÍSTICAS DEL SERVICIO (fijas) ===
Categoria: Transporte
Subcategoria: Terrestre
LeadTimeMinDias: 5
LeadTimeMaxDias: 10
TarifaBase: 250.00
RequiereCertificacion: Sí

=== CARACTERÍSTICAS DEL CLIENTE ===
ClientePropietario: CLI-023
Segmento: PREFERENTE
ZonaDespacho: Norte
Departamento: Lima

=== CARACTERÍSTICAS DEL PROVEEDOR ===
ProveedorID: PROV-089
RatingDesempeno: 4.5
LeadTimePromedioDias: 7
ToleranciaEntregaDias: 2

=== VARIABLES OPERACIONALES (cambian por periodo) ===
StockActual: 120 unidades
DemandaDiariaEst: 15.3 unidades/día
DiasHastaRecepcion: 12 días
RecepcionPendiente: 200 unidades

=== TARGET REAL ===
Stockout14d: 1 (Sí hubo rotura)
```

### 2.5 Relación entre la fila y el resultado

**Proceso paso a paso:**

1. **El usuario selecciona** ServicioID y Periodo
2. **El sistema busca** esa fila específica en el dataset
3. **El modelo recibe** las 35 variables de esa fila
4. **El modelo calcula** la probabilidad usando los pesos aprendidos
5. **El sistema muestra** el resultado

**Ejemplo numérico real:**

```
Input al modelo:
  StockActual: 120
  DemandaDiariaEst: 15.3
  DiasHastaRecepcion: 12
  RecepcionPendiente: 200
  ... (31 variables más)

Cálculo interno del modelo:
  Días de cobertura = 120 / 15.3 = 7.8 días
  
  El stock dura 7.8 días
  La recepción llega en 12 días
  → Habrá 4.2 días sin stock → ALTO RIESGO

Output del modelo:
  Probabilidad de rotura = 0.89 (89%)
```

### 2.6 ¿Qué columnas influyen MÁS en el resultado?

**Orden de importancia (basado en el modelo):**

#### 🔴 **Impacto CRÍTICO** (directamente en el cálculo):

1. **StockActual** (peso ~35%)
   - Stock bajo = Mayor riesgo
   - Relación: Lineal inversa

2. **DemandaDiariaEst** (peso ~30%)
   - Demanda alta = Mayor riesgo
   - Relación: Lineal directa

3. **DiasHastaRecepcion** (peso ~25%)
   - Más días = Mayor riesgo (si stock es bajo)
   - Relación: Condicional

4. **RecepcionPendiente** (peso ~8%)
   - Recepción alta = Menor riesgo futuro
   - Relación: Lineal inversa

#### 🟡 **Impacto MODERADO** (contexto):

5. **Segmento del cliente** (peso ~5%)
   - PREFERENTE: Demanda más estable (menor riesgo)
   - BÁSICO: Demanda más volátil (mayor riesgo)

6. **LeadTimePromedioDias del proveedor** (peso ~4%)
   - Lead time largo = Mayor riesgo estructural

7. **RatingDesempeno del proveedor** (peso ~3%)
   - Rating bajo = Entregas impredecibles (mayor riesgo)

#### 🟢 **Impacto BAJO** (ajustes finos):

8. Categoría del servicio
9. Zona de despacho
10. Temperatura controlada
11. Certificaciones
12. ... resto de variables estructurales

### 2.7 ¿Por qué hay variables que NO influyen mucho?

**Variables estructurales** (como Categoría, Subcategoría, Moneda) describen **QUÉ ES** el servicio, pero no **CUÁNDO SE ROMPERÁ**.

**Analogía con autos:**
- Color del auto (estructural) → NO influye en que te quedes sin gasolina
- Litros en el tanque (operacional) → SÍ influye directamente
- Consumo por km (operacional) → SÍ influye directamente
- Km hasta la próxima gasolinera (operacional) → SÍ influye directamente

En WMS es igual:
- Tipo de servicio → NO influye en rotura inmediata
- Stock actual → SÍ influye directamente
- Demanda diaria → SÍ influye directamente
- Días hasta recepción → SÍ influye directamente

---

## 3. Modo 2: Formulario Manual

### 3.1 ¿Para qué sirve este modo?

Permite **simular escenarios hipotéticos** sin necesidad de que existan en el dataset.

**Casos de uso:**
1. **Planificación**: "¿Qué pasa si la demanda aumenta 20%?"
2. **Negociación**: "¿Qué pasa si el proveedor reduce lead time a 5 días?"
3. **Decisiones urgentes**: "Tengo 50 unidades, demanda de 8/día, ¿pido más?"
4. **Training**: Capacitar operadores con escenarios controlados

### 3.2 ¿Por qué se selecciona ServicioID y Periodo primero?

Porque necesitamos un **punto de partida** con valores realistas.

**Proceso:**
1. Seleccionas `SRV-045` (Transporte) y `Periodo 3`
2. El sistema carga **todas las variables** de ese snapshot
3. Te muestra **solo las operacionales** para editar
4. Las demás (estructurales) quedan fijas en segundo plano

**¿Por qué no empezar de cero?**
Porque necesitas 35 variables válidas, y solo 5 son operacionales. Completar las 35 manualmente sería tedioso y propenso a errores.

### 3.3 ¿Qué son los "Valores Sugeridos"?

Son los valores **reales del snapshot seleccionado**, cargados automáticamente como base.

**Ejemplo:**

```
Si seleccionas: SRV-045, Periodo 3

Valores sugeridos (del dataset):
  StockActual: 250
  DemandaDiariaEst: 12.5
  DiasHastaRecepcion: 8
  RecepcionPendiente: 150
  Horizonte: 14 días
```

**Ahora puedes editarlos:**

```
Valores editados por el operador:
  StockActual: 100 (redujo stock manualmente)
  DemandaDiariaEst: 20.0 (simula aumento de demanda)
  DiasHastaRecepcion: 8 (mantiene igual)
  RecepcionPendiente: 150 (mantiene igual)
  Horizonte: 14 días (mantiene igual)
```

### 3.4 ¿Por qué SOLO estos 5 campos son editables?

Porque son las **variables operacionales** que:
1. ✅ **Cambian día a día** (son dinámicas)
2. ✅ **Son conocidas por el operador** (datos del WMS)
3. ✅ **Influyen directamente** en la rotura
4. ✅ **Pueden modificarse** con acciones operativas

### 3.5 Tabla detallada de variables

| Variable | ¿Editable? | ¿Por qué? |
|----------|------------|-----------|
| **StockActual** | ✅ SÍ | El operador puede hacer inventario físico y ajustarlo |
| **DemandaDiariaEst** | ✅ SÍ | El operador puede recalcular demanda con datos recientes |
| **DiasHastaRecepcion** | ✅ SÍ | El operador puede consultar estado del pedido al proveedor |
| **RecepcionPendiente** | ✅ SÍ | El operador puede verificar órdenes de compra confirmadas |
| **Horizonte** | ✅ SÍ | El operador decide la ventana de predicción (7, 14, 30 días) |
| | | |
| Categoria | ❌ NO | Es una característica fija del servicio |
| Subcategoria | ❌ NO | No cambia día a día |
| LeadTimeMinDias | ❌ NO | Es un contrato con el proveedor, no se edita diario |
| LeadTimeMaxDias | ❌ NO | Igual, es contractual |
| TarifaBase | ❌ NO | Precio fijo del servicio |
| Moneda | ❌ NO | No influye en rotura física |
| RequiereCertificacion | ❌ NO | Requisito regulatorio fijo |
| Temperatura | ❌ NO | Característica del producto |
| TiempoEjecucionHoras | ❌ NO | SLA contractual |
| ModalidadContrato | ❌ NO | Jurídico/administrativo |
| Estado | ❌ NO | Activo/Inactivo, no es operacional |
| CostoEstandar | ❌ NO | Contabilidad, no influye en rotura física |
| TarifaImpuesto | ❌ NO | Fiscal, no operacional |
| TemperaturaControlada | ❌ NO | Característica fija |
| CaducidadControlada | ❌ NO | Característica fija |
| SLA_horas | ❌ NO | Contractual |
| SLA_pct | ❌ NO | Contractual |
| ClientePropietario | ❌ NO | No cambia el cliente del servicio diariamente |
| Segmento | ❌ NO | Clasificación comercial del cliente |
| CanalPreferido | ❌ NO | Estrategia comercial |
| ZonaDespacho | ❌ NO | Geografía fija |
| Departamento | ❌ NO | Geografía fija |
| ProveedorID | ❌ NO | No cambias de proveedor diariamente |
| Categoria_prov | ❌ NO | Tipo de proveedor (LOGISTICA/SERVICIOS) |
| LeadTimePromedioDias | ❌ NO | Histórico del proveedor |
| ToleranciaEntregaDias | ❌ NO | Contractual con proveedor |
| RatingDesempeno | ❌ NO | Evaluación histórica |
| CertificadoCalidad | ❌ NO | Certificación del proveedor |
| Estado_prov | ❌ NO | Activo/Inactivo |
| Periodo | ❌ NO | Es solo un identificador temporal |

### 3.6 ¿Por qué NO editar las demás variables?

**Razón 1: No son operacionales**
Un operador de WMS NO puede cambiar:
- El tipo de servicio
- El lead time contractual del proveedor
- La zona geográfica
- Las certificaciones

**Razón 2: No cambian día a día**
Estas variables son **maestros** que se actualizan mensual o trimestralmente, no diariamente.

**Razón 3: Ya están en el contexto**
Al seleccionar ServicioID y Periodo, todas estas variables YA están cargadas en el modelo. Solo necesitas ajustar las operacionales.

### 3.7 Ejemplo práctico completo

**Escenario**: Eres operador de WMS y recibes un pedido urgente

#### Paso 1: Seleccionas el servicio
```
ServicioID: SRV-089 (Almacenaje productos electrónicos)
Periodo: 8 (Agosto)
```

El sistema carga automáticamente:
- Categoría: Almacenaje
- Cliente: CLI-045 (Segmento PREFERENTE)
- Proveedor: PROV-120 (Rating 4.2, Lead time 10 días)
- ... 30 variables más

#### Paso 2: El sistema sugiere valores operacionales
```
StockActual: 300 unidades
DemandaDiariaEst: 18.5 unidades/día
DiasHastaRecepcion: 10 días
RecepcionPendiente: 250 unidades
```

#### Paso 3: Ajustas según la realidad actual
```
✏️ StockActual: 180 (hiciste inventario y hay menos)
✏️ DemandaDiariaEst: 25.0 (aumentó por campaña)
✏️ DiasHastaRecepcion: 14 (proveedor avisó retraso)
✏️ RecepcionPendiente: 250 (mantiene)
```

#### Paso 4: Presionas "Predecir"

El modelo calcula:
```
Días de cobertura = 180 / 25.0 = 7.2 días
Recepción llega en = 14 días

7.2 < 14 → Habrá 6.8 días sin stock

Probabilidad de rotura: 92% → RIESGO ALTO
```

#### Paso 5: Mensaje de acción
```
⚠️ Riesgo ALTO de rotura en 14 días.

Acción sugerida:
  → Generar reabastecimiento inmediato
  → Priorizar recepción con proveedor
  → Considerar proveedor alternativo
  → Notificar a cliente sobre posible retraso
```

---

## 4. Variables Operacionales vs. Estructurales

### 4.1 Variables Operacionales (DINÁMICAS)

**Definición**: Cambian frecuentemente y reflejan el estado actual del sistema.

| Variable | Frecuencia de cambio | Quién la actualiza |
|----------|---------------------|-------------------|
| StockActual | Diaria/Horaria | Sistema WMS automático + inventarios físicos |
| DemandaDiariaEst | Semanal | Sistema de pronóstico + operador |
| DiasHastaRecepcion | Al consultar proveedor | Operador de compras |
| RecepcionPendiente | Al confirmar órdenes | Sistema de órdenes de compra |

**Características:**
- ✅ Alta variabilidad
- ✅ Influencia directa en rotura
- ✅ Accionables por el operador
- ✅ Medibles en tiempo real

### 4.2 Variables Estructurales (ESTÁTICAS)

**Definición**: Definen la naturaleza del servicio/cliente/proveedor pero no cambian frecuentemente.

| Variable | Frecuencia de cambio | Quién la actualiza |
|----------|---------------------|-------------------|
| Categoria | Anual o nunca | Administrador de maestros |
| Segmento del cliente | Trimestral | Área comercial |
| RatingDesempeno del proveedor | Mensual | Área de calidad |
| LeadTime contractual | Anual (renegociación) | Área de compras |

**Características:**
- ⏸️ Baja variabilidad
- 🔍 Influencia contextual (no directa)
- ❌ No accionables diariamente
- 📋 Definen capacidades y restricciones

### 4.3 ¿Por qué el modelo usa ambas?

**Variables operacionales** → Responden "¿Cuándo?"
**Variables estructurales** → Responden "¿En qué contexto?"

**Ejemplo:**

```
Caso 1: Servicio de alta rotación (categoría: Picking)
  Stock: 50
  Demanda: 10/día
  → Probabilidad: 75% (contexto: alta rotación agrava)

Caso 2: Servicio de baja rotación (categoría: Almacenaje)
  Stock: 50
  Demanda: 10/día
  → Probabilidad: 60% (contexto: baja rotación amortigua)
```

Las operacionales dan el **estado actual**, las estructurales dan el **perfil de riesgo**.

---

## 5. Interpretación de Resultados

### 5.1 ¿Qué significa "Probabilidad de rotura en 14 días"?

Es la **confianza del modelo** de que ocurrirá una rotura antes de 14 días.

**Escala:**
```
0% ──────────── 25% ──────────── 50% ──────────── 75% ──────────── 100%
│                │                 │                 │                │
Imposible     Muy bajo         Incierto          Probable       Casi seguro
```

**Interpretación práctica:**

| Probabilidad | Riesgo | Significado | Acción |
|--------------|--------|-------------|--------|
| **0-10%** | Mínimo | Stock suficiente, demanda baja | Operación normal |
| **10-25%** | Bajo | Stock adecuado pero monitorear | Revisión semanal |
| **25-40%** | Bajo-Medio | Stock justo, demanda estable | Revisar en 3-5 días |
| **40-55%** | Medio | Zona de incertidumbre | Monitoreo diario |
| **55-70%** | Medio-Alto | Alta probabilidad de rotura | Activar alerta, planear pedido |
| **70-85%** | Alto | Rotura inminente si no actúas | Pedido urgente confirmado |
| **85-100%** | Crítico | Rotura casi segura | Acción inmediata, proveedor alternativo |

### 5.2 ¿Por qué el resultado puede ser contraintuitivo?

**Caso 1: Stock alto pero probabilidad alta**
```
Stock: 1,000 unidades
Demanda: 150 unidades/día
Días hasta recepción: 20 días

Días de cobertura: 1,000 / 150 = 6.7 días
Probabilidad: 88% (ALTA)
```

**¿Por qué?** Aunque el stock parece mucho, la demanda es tan alta que se agota en menos de 7 días, y el pedido llega en 20.

---

**Caso 2: Stock bajo pero probabilidad baja**
```
Stock: 50 unidades
Demanda: 2 unidades/día
Días hasta recepción: 5 días

Días de cobertura: 50 / 2 = 25 días
Probabilidad: 12% (BAJA)
```

**¿Por qué?** El stock es bajo en cantidad absoluta, pero dura 25 días, y el pedido llega en 5. Hay margen de sobra.

### 5.3 Mensajes de riesgo automáticos

El sistema genera mensajes contextuales:

#### 🔴 RIESGO ALTO (≥70%)
```
⚠️ Riesgo ALTO de rotura en 14 días.

Acción sugerida:
  → Generar reabastecimiento inmediato
  → Priorizar recepción con proveedor
  → Evaluar proveedor alternativo
  → Activar protocolo de emergencia
```

**Traducción operativa:**
1. Llama al proveedor HOY
2. Confirma fecha de entrega
3. Si hay retraso, busca plan B
4. Notifica a supervisor

---

#### 🟡 RIESGO MEDIO (40-70%)
```
⚠️ Riesgo MEDIO de rotura en 14 días.

Acción sugerida:
  → Monitoreo diario de stock
  → Validar estado de recepción pendiente
  → Preparar pedido de contingencia
```

**Traducción operativa:**
1. Revisa stock cada mañana
2. Verifica tracking del pedido
3. Ten a mano contacto del proveedor

---

#### 🟢 RIESGO BAJO (<40%)
```
✓ Riesgo BAJO de rotura en 14 días.

Acción sugerida:
  → Operación normal
  → Revisión periódica
```

**Traducción operativa:**
1. Continúa con proceso estándar
2. Revisión semanal rutinaria

---

## 6. Flujo de Decisiones

### 6.1 Diagrama de flujo operativo

```
┌─────────────────────────┐
│ Operador inicia turno   │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ Revisa alertas del día  │
│ (Dashboard pestaña 5)   │
└───────────┬─────────────┘
            │
            ▼
    ┌───────────────┐
    │ ¿Hay alertas? │
    └───┬───────┬───┘
        │       │
       NO      SÍ
        │       │
        │       ▼
        │   ┌─────────────────────────┐
        │   │ Abre caso específico    │
        │   │ (ServicioID + Periodo)  │
        │   └───────────┬─────────────┘
        │               │
        │               ▼
        │   ┌─────────────────────────┐
        │   │ Ve probabilidad y fila  │
        │   │ completa del snapshot   │
        │   └───────────┬─────────────┘
        │               │
        │               ▼
        │   ┌─────────────────────────┐
        │   │ Verifica stock físico   │
        │   │ (inventario en piso)    │
        │   └───────────┬─────────────┘
        │               │
        │               ▼
        │   ┌─────────────────────────┐
        │   │ Ajusta valores reales   │
        │   │ en formulario           │
        │   └───────────┬─────────────┘
        │               │
        │               ▼
        │   ┌─────────────────────────┐
        │   │ Presiona "Predecir"     │
        │   └───────────┬─────────────┘
        │               │
        │               ▼
        │       ┌───────────────┐
        │       │ ¿Riesgo ALTO? │
        │       └───┬───────┬───┘
        │           │       │
        │          NO      SÍ
        │           │       │
        │           │       ▼
        │           │   ┌─────────────────────────┐
        │           │   │ Genera orden urgente    │
        │           │   │ Contacta proveedor      │
        │           │   │ Escala a supervisor     │
        │           │   └───────────┬─────────────┘
        │           │               │
        │           ▼               │
        │   ┌─────────────────────────┐
        │   │ Registra acción tomada  │
        │   │ (auditoría)             │
        │   └───────────┬─────────────┘
        │               │
        ▼               ▼
┌─────────────────────────┐
│ Continúa operación      │
│ normal                  │
└─────────────────────────┘
```

### 6.2 Matriz de decisión detallada

| Probabilidad | Stock/Demanda | Recepción | Acción Inmediata | Acción a 3 días | Acción a 7 días |
|--------------|---------------|-----------|------------------|-----------------|-----------------|
| **90-100%** | Días < 5 | Días > 10 | Orden emergencia, proveedor alt. | Confirmar llegada | Recibir y verificar |
| **70-90%** | Días 5-10 | Días > 7 | Orden urgente, call proveedor | Follow-up diario | Validar recepción |
| **40-70%** | Días 10-14 | Días < 10 | Monitoreo diario | Confirmar ETA | Preparar recepción |
| **10-40%** | Días > 14 | Cualquiera | Revisión rutinaria | - | Check semanal |
| **0-10%** | Días > 20 | Cualquiera | Operación normal | - | - |

### 6.3 Ejemplo de toma de decisión real

**Contexto:**
```
Fecha: 15 de Agosto, 8:00 AM
Operador: María González (turno mañana)
Ubicación: Almacén Central Lima
```

**Caso 1: Alerta del dashboard**
```
ServicioID: SRV-123 (Picking para e-commerce)
Probabilidad: 87% (ALTO RIESGO)
```

**Paso 1:** María abre el caso completo
```
Stock actual: 85 unidades
Demanda diaria: 22 unidades/día
Días de cobertura: 3.9 días
Días hasta recepción: 9 días
Recepción pendiente: 300 unidades
```

**Análisis de María:**
- "El stock solo dura 4 días"
- "El pedido llega en 9 días"
- "Habrá 5 días sin stock" ❌

**Paso 2:** María verifica stock físico
- Cuenta física: 82 unidades (3 menos que el sistema)
- Ajusta en formulario: `StockActual = 82`

**Paso 3:** María consulta demanda reciente
- Últimos 3 días: 24, 26, 23 unidades/día (promedio: 24.3)
- Ajusta en formulario: `DemandaDiariaEst = 24.3`

**Paso 4:** María llama al proveedor
- Proveedor confirma: "Entrega el 22 de agosto" (7 días, no 9)
- Ajusta en formulario: `DiasHastaRecepcion = 7`

**Paso 5:** María presiona "Predecir"
```
Nueva probabilidad: 78% (todavía ALTO)

Días de cobertura actualizados: 82 / 24.3 = 3.4 días
Recepción: 7 días
Gap: 3.6 días sin stock
```

**Decisión de María:**
1. ✅ Genera orden de emergencia por 100 unidades
2. ✅ Solicita entrega express (3 días) a proveedor alternativo
3. ✅ Notifica a supervisor por email
4. ✅ Programa seguimiento para mañana 8 AM

**Resultado:**
- Costo adicional: $150 por envío express
- Beneficio: Evita rotura que costaría $2,500 en ventas perdidas
- **ROI de la acción: 1,567%** ✅

---

## 7. Preguntas Frecuentes (FAQ)

### ¿Por qué la probabilidad no es 0% o 100%?

Porque el modelo trabaja con **incertidumbre**. Hay factores no capturados:
- Retrasos inesperados del proveedor
- Picos de demanda no previstos
- Errores de inventario
- Problemas logísticos (tráfico, clima, etc.)

El modelo da su mejor estimación basada en patrones históricos.

---

### ¿Puedo confiar en una probabilidad del 55%?

Es zona de incertidumbre. Recomendación:
1. Usa el formulario para refinar con datos actuales
2. Considera el costo de equivocarte (¿qué es peor: pedir de más o de menos?)
3. En WMS, es mejor ser conservador (evitar roturas)

---

### ¿Por qué el modelo a veces se equivoca?

Razones:
1. Dataset sintético (no captura toda la complejidad real)
2. Variables omitidas (clima, eventos especiales, huelgas)
3. Cambios estructurales (nuevo proveedor, nuevo cliente)
4. Datos desactualizados en maestros

**Mejora continua**: El modelo debe re-entrenarse mensualmente con datos reales.

---

### ¿Qué hago si el modelo dice BAJO riesgo pero yo veo problema?

**Confía en tu experiencia operativa** y usa el formulario para:
1. Actualizar valores con datos frescos
2. Simular escenario pesimista
3. Escalar a supervisor si persiste la duda

El modelo es una **herramienta**, no un dictador. Tú conoces el contexto local.

---

## 8. Conclusión

El sistema de predicción tiene **dos modos complementarios**:

1. **Modo Dataset**: Valida el modelo con snapshots históricos
2. **Modo Formulario**: Toma decisiones con datos actuales

**Variables clave operacionales:**
- StockActual
- DemandaDiariaEst
- DiasHastaRecepcion
- RecepcionPendiente

**Recuerda:**
✅ El modelo es una guía, no una orden
✅ Actualiza valores con información fresca
✅ En duda, sé conservador (evita roturas)
✅ Documenta acciones para mejorar el modelo

---

**Última actualización**: Diciembre 2024
**Versión del modelo**: stockout14d_logreg v1.0
**Contacto técnico**: Equipo de Data Science WMS

