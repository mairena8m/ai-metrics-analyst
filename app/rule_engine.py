from app.schemas import MetricsRequest


def get_metric_value(model, metric_name: str) -> float:
    value = getattr(model, metric_name)
    if value is None:
        raise ValueError(f"El modelo '{model.name}' no tiene la métrica '{metric_name}'.")
    return value


def calculate_basic_analysis(request: MetricsRequest) -> dict:
    if len(request.models) < 2:
        raise ValueError("Necesitas al menos dos modelos para comparar.")

    metric = request.main_metric

    sorted_models = sorted(
        request.models,
        key=lambda model: get_metric_value(model, metric),
        reverse=True
    )

    best_model = sorted_models[0]
    second_model = sorted_models[1]
    weakest_model = sorted_models[-1]

    best_value = get_metric_value(best_model, metric)
    second_value = get_metric_value(second_model, metric)
    weakest_value = get_metric_value(weakest_model, metric)

    difference_vs_second = best_value - second_value
    difference_vs_weakest = best_value - weakest_value

    if difference_vs_second < 0.005:
        improvement_level = "no_evaluable"
    elif difference_vs_second < 0.015:
        improvement_level = "baja"
    elif difference_vs_second < 0.04:
        improvement_level = "moderada"
    else:
        improvement_level = "alta"

    return {
        "task": request.task,
        "main_metric": metric,
        "best_model": best_model.name,
        "best_value": best_value,
        "second_model": second_model.name,
        "second_value": second_value,
        "weakest_model": weakest_model.name,
        "weakest_value": weakest_value,
        "difference_vs_second": round(difference_vs_second, 4),
        "difference_vs_weakest": round(difference_vs_weakest, 4),
        "improvement_level": improvement_level,
        "models": [model.model_dump() for model in request.models],
    }