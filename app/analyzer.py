from app.schemas import MetricsRequest, AnalysisResponse
from app.rule_engine import calculate_basic_analysis
from app.llm_client import generate_llm_analysis


def build_fallback_analysis(basic_analysis: dict) -> AnalysisResponse:
    best_model = basic_analysis["best_model"]
    weakest_model = basic_analysis["weakest_model"]
    main_metric = basic_analysis["main_metric"]
    best_value = basic_analysis["best_value"]
    second_model = basic_analysis["second_model"]
    second_value = basic_analysis["second_value"]
    difference = basic_analysis["difference_vs_second"]
    improvement_level = basic_analysis["improvement_level"]

    return AnalysisResponse(
        best_model=best_model,
        weakest_model=weakest_model,
        main_reason=(
            f"El modelo {best_model} obtiene el mejor valor en la métrica principal "
            f"'{main_metric}' con {best_value:.4f}, frente a {second_model} con "
            f"{second_value:.4f}. La diferencia frente al segundo mejor modelo es "
            f"de {difference:.4f}."
        ),
        improvement_level=improvement_level,
        overfitting_risk=(
            "No se puede evaluar el sobreajuste únicamente con métricas agregadas. "
            "Serían necesarias curvas de entrenamiento y validación, además de una "
            "evaluación en un conjunto de test independiente."
        ),
        recommended_metric=main_metric,
        tfm_conclusion=(
            f"Los resultados indican que {best_model} presenta el mejor rendimiento "
            f"según la métrica principal seleccionada. Aun así, la mejora debe "
            f"interpretarse con cautela si la diferencia entre modelos es reducida "
            f"o si el conjunto de evaluación es pequeño."
        ),
        warnings=[
            "El análisis ha sido generado en modo fallback porque el LLM no estaba disponible.",
            "No debe afirmarse superioridad absoluta sin validación adicional.",
            "Las métricas agregadas no permiten detectar por sí solas problemas de sobreajuste.",
            "Conviene revisar ejemplos visuales y casos difíciles antes de extraer conclusiones fuertes."
        ]
    )


def analyze_metrics(request: MetricsRequest) -> AnalysisResponse:
    basic_analysis = calculate_basic_analysis(request)

    try:
        return generate_llm_analysis(basic_analysis)
    except Exception:
        return build_fallback_analysis(basic_analysis)