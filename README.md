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
