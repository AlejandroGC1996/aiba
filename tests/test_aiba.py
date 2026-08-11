import numpy as np
import pandas as pd
import pytest

from aiba import optimize_intervals


def synthetic_evaluations(seed=7, count=40):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.uniform(size=(count, 2)), columns=["x1", "x2"])
    y = (X["x1"] - 0.25) ** 2 + (X["x2"] - 0.75) ** 2
    return X, y


def test_optimize_intervals_returns_rankable_regions():
    X, y = synthetic_evaluations()
    result = optimize_intervals(
        X,
        y,
        M_min=3,
        step_int=0.2,
        step_iter=0.05,
        max_steps_int=4,
        max_steps_iter=20,
    )

    assert not result.empty
    assert result["final_metric"].notna().all()
    assert (result["final_points"] >= 3).all()
    assert all(np.asarray(bounds).shape == (2, 2) for bounds in result["final_interval_denorm"])


def test_input_columns_must_match_dataframe_order():
    X, y = synthetic_evaluations(count=10)
    with pytest.raises(ValueError, match="input_columns"):
        optimize_intervals(X, y, input_columns=["x2", "x1"], M_min=2)

