from fastapi import FastAPI, HTTPException
from app.schemas import MetricsRequest, AnalysisResponse
from app.analyzer import analyze_metrics

app = FastAPI(
    title="AI Metrics Analyst",
    description="API para analizar resultados de modelos de ML usando reglas + LLM.",
    version="0.1.0"
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze-metrics", response_model=AnalysisResponse)
def analyze_metrics_endpoint(request: MetricsRequest):
    try:
        return analyze_metrics(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")