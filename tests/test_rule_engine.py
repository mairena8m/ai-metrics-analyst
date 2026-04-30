import pytest

from app.schemas import MetricsRequest, ModelMetrics
from app.rule_engine import calculate_basic_analysis


def build_request(models, main_metric="dice"):
    return MetricsRequest(
        task="segmentacion medica de glomerulos",
        main_metric=main_metric,
        models=models
    )


def test_selects_best_and_weakest_model_by_main_metric():
    request = build_request([
        ModelMetrics(
            name="U-Net++ Alberto",
            precision=0.500,
            recall=0.603,
            f1=0.546,
            dice=0.546,
            iou=0.376
        ),
        ModelMetrics(
            name="U-Net++ Replica",
            precision=0.472,
            recall=0.638,
            f1=0.543,
            dice=0.543,
            iou=0.373
        ),
        ModelMetrics(
            name="U-Net++ MONAI",
            precision=0.535,
            recall=0.612,
            f1=0.571,
            dice=0.571,
            iou=0.400
        ),
        ModelMetrics(
            name="U-Net++ Fine Tuning Hiperparametros",
            precision=0.623,
            recall=0.509,
            f1=0.560,
            dice=0.560,
            iou=0.389
        )
    ])

    result = calculate_basic_analysis(request)

    assert result["best_model"] == "U-Net++ MONAI"
    assert result["second_model"] == "U-Net++ Fine Tuning Hiperparametros"
    assert result["weakest_model"] == "U-Net++ Replica"
    assert result["main_metric"] == "dice"
    assert result["best_value"] == 0.571
    assert result["second_value"] == 0.560
    assert result["weakest_value"] == 0.543


def test_classifies_low_improvement():
    request = build_request([
        ModelMetrics(
            name="Modelo A",
            precision=0.5,
            recall=0.6,
            f1=0.571,
            dice=0.571,
            iou=0.4
        ),
        ModelMetrics(
            name="Modelo B",
            precision=0.5,
            recall=0.6,
            f1=0.560,
            dice=0.560,
            iou=0.39
        )
    ])

    result = calculate_basic_analysis(request)

    assert result["difference_vs_second"] == 0.011
    assert result["improvement_level"] == "baja"


def test_classifies_moderate_improvement():
    request = build_request([
        ModelMetrics(
            name="Modelo A",
            precision=0.5,
            recall=0.6,
            f1=0.580,
            dice=0.580,
            iou=0.41
        ),
        ModelMetrics(
            name="Modelo B",
            precision=0.5,
            recall=0.6,
            f1=0.550,
            dice=0.550,
            iou=0.37
        )
    ])

    result = calculate_basic_analysis(request)

    assert result["difference_vs_second"] == 0.03
    assert result["improvement_level"] == "moderada"


def test_classifies_high_improvement():
    request = build_request([
        ModelMetrics(
            name="Modelo A",
            precision=0.5,
            recall=0.6,
            f1=0.620,
            dice=0.620,
            iou=0.45
        ),
        ModelMetrics(
            name="Modelo B",
            precision=0.5,
            recall=0.6,
            f1=0.560,
            dice=0.560,
            iou=0.39
        )
    ])

    result = calculate_basic_analysis(request)

    assert result["difference_vs_second"] == 0.06
    assert result["improvement_level"] == "alta"


def test_classifies_not_evaluable_improvement():
    request = build_request([
        ModelMetrics(
            name="Modelo A",
            precision=0.5,
            recall=0.6,
            f1=0.562,
            dice=0.562,
            iou=0.39
        ),
        ModelMetrics(
            name="Modelo B",
            precision=0.5,
            recall=0.6,
            f1=0.560,
            dice=0.560,
            iou=0.388
        )
    ])

    result = calculate_basic_analysis(request)

    assert result["difference_vs_second"] == 0.002
    assert result["improvement_level"] == "no_evaluable"


def test_raises_error_when_only_one_model_is_provided():
    request = build_request([
        ModelMetrics(
            name="Modelo A",
            precision=0.5,
            recall=0.6,
            f1=0.56,
            dice=0.56,
            iou=0.39
        )
    ])

    with pytest.raises(ValueError, match="Necesitas al menos dos modelos"):
        calculate_basic_analysis(request)


def test_raises_error_when_main_metric_is_missing():
    request = build_request(
        models=[
            ModelMetrics(
                name="Modelo A",
                precision=0.5,
                recall=0.6,
                f1=0.56,
                dice=None,
                iou=0.39
            ),
            ModelMetrics(
                name="Modelo B",
                precision=0.5,
                recall=0.6,
                f1=0.55,
                dice=0.55,
                iou=0.38
            )
        ],
        main_metric="dice"
    )

    with pytest.raises(ValueError, match="no tiene la métrica"):
        calculate_basic_analysis(request)