"""Internal data-transformation and interval-scoring utilities."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def preprocess_data(X: pd.DataFrame) -> tuple[pd.DataFrame, MinMaxScaler]:
    """Copy and min-max normalize all hyperparameter columns."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")
    if X.empty:
        raise ValueError("X must contain at least one evaluated configuration.")
    if X.isna().any().any():
        raise ValueError("X must not contain missing values.")

    scaler = MinMaxScaler()
    processed = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index,
    )
    return processed, scaler


def inverse_transform(
    data: np.ndarray,
    scaler: MinMaxScaler,
    input_columns: list[str],
    is_point: bool = False,
) -> pd.DataFrame:
    """Transform normalized points or interval bounds to their original scale.

    ``is_point`` is retained for compatibility with the original implementation.
    Both points and interval bounds are represented as rows during inversion.
    """
    del is_point
    frame = pd.DataFrame(data, columns=input_columns)
    frame.loc[:, :] = scaler.inverse_transform(frame)
    return frame


def log_transform(X_array: np.ndarray, interval: np.ndarray) -> np.ndarray:
    """Represent a normalized interval on a normalized logarithmic scale."""
    min_values = np.min(X_array, axis=0)
    max_values = np.max(X_array, axis=0)
    log_min_values = np.log(min_values + 1e-6)
    log_max_values = np.log(max_values + 1e-6)

    log_normalized_intervals = []
    for dimension in range(interval.shape[0]):
        denominator = log_max_values[dimension] - log_min_values[dimension]
        low = (np.log(interval[dimension][0] + 1e-6) - log_min_values[dimension]) / denominator
        high = (np.log(interval[dimension][1] + 1e-6) - log_min_values[dimension]) / denominator
        log_normalized_intervals.append([low, high])

    return np.asarray(log_normalized_intervals)


def generate_mask(X: np.ndarray, interval: np.ndarray) -> np.ndarray:
    """Select rows of ``X`` lying inside every interval dimension."""
    return np.all((X >= interval[:, 0]) & (X <= interval[:, 1]), axis=1)


def calculate_metric(
    interval: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    M_min: float = 20,
    beta: float = 0.25,
) -> tuple[float, float]:
    """Score an interval by mean objective plus penalized standard error.

    Intervals containing fewer than ``M_min`` observations receive an infinite
    score. Lower finite scores are better.
    """
    subset = Y[generate_mask(X, interval)]
    sample_count = len(subset)
    if sample_count < M_min:
        return float("inf"), 0.0

    mean = float(np.mean(subset))
    if sample_count == 1:
        standard_error = 0.0
    else:
        standard_error = float(np.std(subset, ddof=1) / np.sqrt(sample_count))
    return mean + beta * standard_error, mean
