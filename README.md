# AI Metrics Analyst

API REST desarrollada en Python para analizar resultados de modelos de Machine Learning mediante una combinación de reglas deterministas y un modelo de IA generativa.

El objetivo del proyecto es recibir métricas de distintos modelos, como `precision`, `recall`, `f1`, `dice` e `iou`, y generar un análisis técnico estructurado indicando cuál es el mejor modelo, cuál es el más débil, el nivel de mejora y las principales advertencias que deben tenerse en cuenta.

Este proyecto está orientado principalmente al análisis de modelos de segmentación médica, aunque puede adaptarse a otros problemas de Machine Learning.

---

## Objetivo del proyecto

El proyecto nace como una práctica para afianzar conceptos relacionados con:

- Python aplicado a IA.
- Desarrollo de APIs REST.
- Uso de modelos LLM mediante API.
- Análisis automático de métricas de modelos.
- Salidas estructuradas en JSON.
- Buenas prácticas básicas con Git y organización de proyectos.

La idea principal es que el LLM no tome todas las decisiones.  
Las decisiones numéricas importantes, como seleccionar el mejor modelo o calcular la diferencia entre modelos, se realizan mediante código Python. El modelo generativo se utiliza para redactar una interpretación técnica más clara y estructurada.

---

## Tecnologías utilizadas

- Python
- FastAPI
- Pydantic
- Uvicorn
- Gemini API
- python-dotenv

---

## Estructura del proyecto

```text
ai-metrics-analyst/
│
├── app/
│   ├── main.py
│   ├── schemas.py
│   ├── rule_engine.py
│   ├── llm_client.py
│   └── analyzer.py
│
├── examples/
│   └── segmentacion_example.json
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Funcionamiento general

El flujo del sistema es el siguiente:

```text
Entrada JSON
     ↓
Validación con Pydantic
     ↓
Análisis determinista con Python
     ↓
Generación de explicación con Gemini
     ↓
Respuesta estructurada en JSON
```

La parte determinista calcula:

- Mejor modelo según la métrica principal.
- Modelo más débil.
- Diferencia frente al segundo mejor modelo.
- Nivel de mejora: `no_evaluable`, `baja`, `moderada` o `alta`.

Después, Gemini genera una explicación técnica respetando esos resultados.

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/mairena8m/ai-metrics-analyst.git
cd ai-metrics-analyst
```

### 2. Crear entorno virtual

En Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

En Linux/Mac:

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## Configuración

Crea un archivo `.env` en la raíz del proyecto.

Puedes basarte en el archivo `.env.example`:

```env
GEMINI_API_KEY=tu_api_key_aqui
GEMINI_MODEL=gemini-2.5-flash-lite
```

Importante: el archivo `.env` no debe subirse nunca a GitHub porque contiene claves privadas.

---

## Ejecutar la API

Desde la carpeta raíz del proyecto:

```bash
uvicorn app.main:app --reload
```

Después, abre en el navegador:

```text
http://127.0.0.1:8000/docs
```

FastAPI mostrará una interfaz interactiva donde se pueden probar los endpoints.

---

## Endpoints disponibles

### Health check

```http
GET /health
```

Respuesta esperada:

```json
{
  "status": "ok"
}
```

---

### Analizar métricas

```http
POST /analyze-metrics
```

Ejemplo de entrada:

```json
{
  "task": "segmentacion medica de glomerulos",
  "main_metric": "dice",
  "models": [
    {
      "name": "U-Net++ Alberto",
      "precision": 0.5,
      "recall": 0.603,
      "f1": 0.546,
      "dice": 0.546,
      "iou": 0.376
    },
    {
      "name": "U-Net++ Replica",
      "precision": 0.472,
      "recall": 0.638,
      "f1": 0.543,
      "dice": 0.543,
      "iou": 0.373
    },
    {
      "name": "U-Net++ MONAI",
      "precision": 0.535,
      "recall": 0.612,
      "f1": 0.571,
      "dice": 0.571,
      "iou": 0.4
    },
    {
      "name": "U-Net++ Fine Tuning Hiperparametros",
      "precision": 0.623,
      "recall": 0.509,
      "f1": 0.56,
      "dice": 0.56,
      "iou": 0.389
    }
  ]
}
```

Ejemplo de respuesta:

```json
{
  "best_model": "U-Net++ MONAI",
  "weakest_model": "U-Net++ Replica",
  "main_reason": "U-Net++ MONAI obtiene el mejor resultado en la métrica principal Dice con un valor de 0.571, superando al segundo mejor modelo por 0.011.",
  "improvement_level": "baja",
  "overfitting_risk": "No se puede evaluar el riesgo de sobreajuste sin las curvas de entrenamiento y validación.",
  "recommended_metric": "dice",
  "tfm_conclusion": "El modelo U-Net++ MONAI presenta una ligera ventaja sobre los demás modelos evaluados en la tarea de segmentación médica de glomérulos según la métrica Dice. Sin embargo, las diferencias son reducidas, por lo que la elección final debería apoyarse también en el IoU, la revisión visual de las máscaras y el comportamiento en casos difíciles.",
  "warnings": [
    "La diferencia de rendimiento entre el mejor y el segundo mejor modelo es de 0.011, lo que se considera una mejora baja.",
    "No se dispone de información sobre el riesgo de sobreajuste, ya que no se proporcionaron curvas de entrenamiento y validación.",
    "Se recomienda un análisis visual de las segmentaciones para complementar la métrica Dice y evaluar la calidad de los contornos y la detección de falsos positivos/negativos en casos difíciles."
  ]
}
```

---

## Ejemplo usando curl

```bash
curl -X POST "http://127.0.0.1:8000/analyze-metrics" \
  -H "Content-Type: application/json" \
  -d @examples/segmentacion_example.json
```

En Windows PowerShell, puedes probarlo así:

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/analyze-metrics" `
  -Method POST `
  -ContentType "application/json" `
  -InFile "examples/segmentacion_example.json"
```

---

## Criterio de análisis

El sistema utiliza la métrica principal indicada en el campo `main_metric`.

Por ejemplo, si se indica:

```json
"main_metric": "dice"
```

El modelo con mayor Dice será considerado el mejor.

La diferencia frente al segundo mejor modelo se clasifica así:

```text
< 0.005  → no_evaluable
< 0.015  → baja
< 0.040  → moderada
>= 0.040 → alta
```

Estos umbrales son orientativos y pueden ajustarse según el problema, el tamaño del conjunto de datos y el nivel de exigencia del análisis.

---

## Modo fallback

Si Gemini no está disponible o devuelve un error temporal, la API puede generar una respuesta básica sin LLM utilizando únicamente el análisis determinista.

Esto evita que el sistema falle por completo cuando el proveedor externo no responde.

---

## Limitaciones

Este proyecto no sustituye una evaluación científica completa de modelos.

Limitaciones principales:

- No permite detectar sobreajuste si no se proporcionan curvas de entrenamiento y validación.
- No realiza validación estadística.
- No evalúa automáticamente la calidad visual de las segmentaciones.
- Las conclusiones dependen de la calidad y tamaño del conjunto de evaluación.
- Las diferencias pequeñas entre modelos deben interpretarse con cautela.
- El LLM puede generar redacciones variables, aunque se le fuerce a seguir una estructura JSON.

## Estado del proyecto

Versión inicial funcional.

El proyecto permite enviar métricas de modelos mediante una API REST y obtener un análisis técnico estructurado combinando reglas en Python y generación de texto mediante Gemini.
