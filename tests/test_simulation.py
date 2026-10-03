import numpy as np

from aml_working_paper.simulation.equations import (
    generate_cross_border_ratio,
    generate_network_exposure,
    generate_structuring_score,
    generate_transaction_volume,
)


def test_network_exposure_is_probability_like():
    rng = np.random.default_rng(42)

    value = generate_network_exposure(
        economic_context=0.0,
        rng=rng,
    )

    assert 0.0 < value < 1.0


def test_structuring_score_is_bounded():
    rng = np.random.default_rng(42)

    value = generate_structuring_score(
        aml_state=2,
        rng=rng,
    )

    assert 0.0 < value < 1.0


def test_cross_border_ratio_is_bounded():
    rng = np.random.default_rng(42)

    value = generate_cross_border_ratio(
        network_exposure=0.5,
        aml_state=1,
        rng=rng,
    )

    assert 0.0 < value < 1.0


def test_transaction_volume_is_positive():
    rng = np.random.default_rng(42)

    value = generate_transaction_volume(
        economic_context=0.0,
        aml_state=1,
        structuring_score=0.5,
        rng=rng,
    )

    assert value > 0


    import numpy as np
import pytest

from aml_working_paper.simulation.states import (
    BASE_TRANSITION,
    sample_next_state,
    validate_transition_matrix,
)


def test_base_transition_matrix_is_valid():
    validate_transition_matrix(BASE_TRANSITION)


def test_invalid_transition_matrix_raises_error():
    matrix = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.5, 0.5, 0.5],
            [0.1, 0.2, 0.7],
        ]
    )

    with pytest.raises(ValueError):
        validate_transition_matrix(matrix)


def test_sample_next_state_returns_valid_state():
    rng = np.random.default_rng(42)

    state = sample_next_state(
        previous_state=0,
        transition_matrix=BASE_TRANSITION,
        rng=rng,
    )
    assert state in {0, 1, 2}