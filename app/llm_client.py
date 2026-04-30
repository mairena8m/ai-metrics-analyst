import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

from app.schemas import AnalysisResponse

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("Falta GEMINI_API_KEY en el archivo .env")

client = genai.Client(api_key=GEMINI_API_KEY)


def is_temporary_gemini_error(error: Exception) -> bool:
    error_text = str(error).lower()

    temporary_signals = [
        "503",
        "unavailable",
        "high demand",
        "overloaded",
        "temporarily",
        "capacity"
    ]

    return any(signal in error_text for signal in temporary_signals)


def generate_llm_analysis(basic_analysis: dict) -> AnalysisResponse:
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

    prompt = f"""
Eres un analista técnico experto en machine learning aplicado a imágenes médicas.

Vas a analizar resultados de modelos a partir de datos ya calculados por código.

Reglas obligatorias:
- No cambies el mejor modelo.
- No cambies el peor modelo.
- No exageres las conclusiones.
- No afirmes sobreajuste si no hay curvas de entrenamiento y validación.
- Si las diferencias son pequeñas, dilo claramente.
- Si la diferencia frente al segundo mejor modelo es menor de 0.015, considera la mejora como baja.
- Evita expresiones fuertes como "demuestra ser claramente superior", "es la mejor opción definitiva" o "mejora significativa" salvo que la diferencia sea amplia.
- Usa expresiones prudentes como "obtiene el mejor resultado", "presenta una ligera mejora" o "muestra una ventaja moderada".
- Si las diferencias son pequeñas, prioriza la cautela sobre la afirmación de superioridad.
- Redacta de forma útil para un informe académico o profesional.
- Devuelve una respuesta crítica, prudente y realista.
- No recomiendes F1 como métrica adicional si ya se está usando Dice en una tarea de segmentación binaria, salvo que expliques que ambas métricas pueden ser equivalentes según cómo se calculen.
- Para segmentación, si recomiendas métricas adicionales, prioriza IoU, revisión visual, análisis por caso difícil y curvas de entrenamiento/validación.

Datos calculados:
{basic_analysis}
"""

    max_retries = 3

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=(
                        "Responde siempre como analista técnico de ML. "
                        "Sé crítico, preciso y no inventes datos."
                    ),
                    temperature=0.2,
                    response_mime_type="application/json",
                    response_json_schema=AnalysisResponse.model_json_schema(),
                ),
            )

            return AnalysisResponse.model_validate_json(response.text)

        except Exception as e:
            if is_temporary_gemini_error(e) and attempt < max_retries - 1:
                sleep_seconds = 2 ** attempt
                time.sleep(sleep_seconds)
                continue

            raise RuntimeError(f"Error llamando a Gemini: {str(e)}")