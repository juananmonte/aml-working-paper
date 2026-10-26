
import pytest

from aml_working_paper.causal.evaluation import evaluate_skeleton


def test_perfect_recovery():

    discovered = [
        "economic_context --> account_exposure",
        "economic_context --> transaction_volume",
        "account_exposure --> cross_border_ratio",
        "structuring_score --> transaction_volume",
    ]

    result = evaluate_skeleton(discovered)

    assert result["TP"] == 4
    assert result["FP"] == 0
    assert result["FN"] == 0
    assert result["f1"] == 1.0


def test_missing_and_extra_connections():

    discovered = [
        "economic_context --- account_exposure",
        "economic_context --> transaction_volume",
        "structuring_score o-o cross_border_ratio",
    ]

    result = evaluate_skeleton(discovered)

    assert result["TP"] == 2
    assert result["FP"] == 1
    assert result["FN"] == 2
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == 0.5
    assert result["f1"] == pytest.approx(4 / 7)


def test_direction_does_not_matter():

    discovered = [
        "account_exposure --> economic_context",
        "transaction_volume --> economic_context",
        "cross_border_ratio --> account_exposure",
        "transaction_volume --> structuring_score",
    ]

    result = evaluate_skeleton(discovered)

    assert result["TP"] == 4
    assert result["f1"] == 1.0


def test_no_connections():

    result = evaluate_skeleton([])

    assert result["TP"] == 0
    assert result["FP"] == 0
    assert result["FN"] == 4
    assert result["f1"] == 0.0

import pandas as pd

from aml_working_paper.causal.evaluation import (
    summarize_causal_results,
)


def test_causal_summary():

    evaluation = pd.DataFrame([
        {
            "method": "PC",
            "strategy": "uncertainty",
            "f1": 0.75,
        }
    ])

    random_results = pd.DataFrame({
        "f1": [0.50, 0.75, 1.00],
    })

    summary = summarize_causal_results(
        evaluation,
        random_results,
    )

    assert summary["uncertainty_f1"] == 0.75
    assert summary["random_mean_f1"] == 0.75
    assert summary["random_fraction_ge_uncertainty"] == pytest.approx(2 / 3)
    assert summary["random_repetitions"] == 3