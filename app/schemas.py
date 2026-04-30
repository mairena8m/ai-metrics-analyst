from pydantic import BaseModel, Field
from typing import Literal


MetricName = Literal["precision", "recall", "f1", "dice", "iou"]


class ModelMetrics(BaseModel):
    name: str = Field(..., description="Nombre del modelo")
    precision: float | None = Field(None, ge=0, le=1)
    recall: float | None = Field(None, ge=0, le=1)
    f1: float | None = Field(None, ge=0, le=1)
    dice: float | None = Field(None, ge=0, le=1)
    iou: float | None = Field(None, ge=0, le=1)


class MetricsRequest(BaseModel):
    task: str = Field(..., examples=["segmentacion medica"])
    main_metric: MetricName = Field(default="dice")
    models: list[ModelMetrics]


class AnalysisResponse(BaseModel):
    best_model: str
    weakest_model: str
    main_reason: str
    improvement_level: Literal["baja", "moderada", "alta", "no_evaluable"]
    overfitting_risk: str
    recommended_metric: str
    tfm_conclusion: str
    warnings: list[str]