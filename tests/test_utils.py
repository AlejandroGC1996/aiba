import numpy as np
import pandas as pd
import pytest

from aiba.utils import calculate_metric, generate_mask, preprocess_data


def test_preprocess_data_normalizes_numeric_columns():
    frame = pd.DataFrame({"a": [10.0, 20.0, 30.0], "b": [2.0, 4.0, 8.0]})
    normalized, scaler = preprocess_data(frame)

    assert np.allclose(normalized.min().to_numpy(), [0.0, 0.0])
    assert np.allclose(normalized.max().to_numpy(), [1.0, 1.0])
    assert np.allclose(scaler.inverse_transform(normalized), frame)


def test_generate_mask_requires_all_dimensions_inside_interval():
    points = np.array([[0.2, 0.3], [0.5, 0.5], [0.8, 0.4]])
    interval = np.array([[0.1, 0.6], [0.2, 0.6]])
    assert generate_mask(points, interval).tolist() == [True, True, False]


def test_metric_is_mean_plus_penalized_standard_error():
    points = np.array([[0.1], [0.2], [0.9]])
    objective = np.array([1.0, 3.0, 100.0])
    metric, mean = calculate_metric(
        np.array([[0.0, 0.5]]), points, objective, M_min=2, beta=2.0
    )

    assert mean == pytest.approx(2.0)
    assert metric == pytest.approx(4.0)


def test_metric_rejects_underpopulated_interval():
    metric, mean = calculate_metric(
        np.array([[0.0, 0.15]]),
        np.array([[0.1], [0.2]]),
        np.array([1.0, 2.0]),
        M_min=2,
    )
    assert np.isinf(metric)
    assert mean == 0.0

