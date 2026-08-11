"""Core implementation of the AIBA iterative interval-search algorithm."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .utils import (
    calculate_metric,
    generate_mask,
    inverse_transform,
    log_transform,
    preprocess_data,
)

RESULT_COLUMNS = [
    "model",
    "point_interval_norm",
    "point_interval_denorm",
    "initial_interval_norm",
    "initial_interval_norm_log",
    "initial_interval_denorm",
    "optimal_interval_norm",
    "optimal_interval_norm_log",
    "optimal_interval_denorm",
    "suboptimal_interval_norm",
    "suboptimal_interval_norm_log",
    "suboptimal_interval_denorm",
    "final_interval_norm",
    "final_interval_norm_log",
    "final_interval_denorm",
    "initial_metric",
    "optimal_metric",
    "suboptimal_metric",
    "final_metric",
    "initial_points",
    "optimal_points",
    "suboptimal_points",
    "final_points",
    "initial_range",
    "optimal_range",
    "suboptimal_range",
    "final_range",
    "initial_range_log",
    "optimal_range_log",
    "suboptimal_range_log",
    "final_range_log",
    "initial_loss_mean",
    "optimal_loss_mean",
    "suboptimal_loss_mean",
    "final_loss_mean",
]


def optimize_intervals(
    X: pd.DataFrame,
    Y: pd.Series | pd.DataFrame,
    input_columns: list[str] | None = None,
    step_int: float = 0.2,
    step_iter: float = 0.01,
    max_steps_int: int = 5,
    max_steps_iter: int = 1000,
    M_min: float = 20,
    beta: float = 0.5,
    *,
    verbose: bool = False,
) -> pd.DataFrame:
    """Find promising rectangular regions in an evaluated search space.

    Parameters
    ----------
    X:
        Evaluated hyperparameter configurations. Rows must align with ``Y`` and
        all columns must be numeric.
    Y:
        Scalar loss or cost for every row of ``X``. Lower values are better.
    input_columns:
        Hyperparameter names and output order for denormalized intervals. When
        omitted, ``X.columns`` is used.
    step_int:
        Normalized expansion applied during the initial phase.
    step_iter:
        Base normalized displacement used to find the next observed boundary.
    max_steps_int:
        Maximum number of initial expansion steps.
    max_steps_iter:
        Maximum number of iterative boundary-adjustment rounds.
    M_min:
        Minimum number of prior evaluations inside a valid interval. A float is
        accepted to preserve compatibility with proportions such as ``0.01 * n``.
    beta:
        Weight of the standard-error penalty in the interval metric.
    verbose:
        Print progress while starting points are processed.

    Returns
    -------
    pandas.DataFrame
        One row per accepted starting point, including normalized and original-
        scale intervals, metrics, volumes and contained-point counts.

    Notes
    -----
    AIBA consumes completed evaluations; it does not train models. New model
    configurations must be sampled and evaluated by the calling experiment.
    """
    input_columns = list(X.columns) if input_columns is None else list(input_columns)
    _validate_inputs(
        X,
        Y,
        input_columns,
        step_int,
        step_iter,
        max_steps_int,
        max_steps_iter,
        M_min,
        beta,
    )

    started_at = time.time()
    X_processed, scaler = preprocess_data(X)
    X_array = X_processed.to_numpy(dtype=float)
    Y_array = np.asarray(Y).reshape(-1).astype(float)
    sorted_indices = np.argsort(Y_array)
    results: list[dict[str, object]] = []

    for position, idx in enumerate(sorted_indices, start=1):
        point = X_array[idx]
        interval = point.reshape(-1, 1).repeat(2, axis=1)
        point_interval_norm = interval.copy()
        point_interval_denorm = inverse_transform(
            point_interval_norm.T, scaler, input_columns, is_point=True
        )

        metric, _ = calculate_metric(interval, X_array, Y_array, M_min=M_min, beta=beta)
        temporary_interval = interval.copy()

        # Phase 1: expand around the observed starting configuration.
        for _ in range(max_steps_int):
            temporary_interval[:, 0] = np.maximum(0, temporary_interval[:, 0] - step_int)
            temporary_interval[:, 1] = np.minimum(1, temporary_interval[:, 1] + step_int)
            temporary_metric, _ = calculate_metric(
                temporary_interval, X_array, Y_array, M_min=M_min, beta=beta
            )
            if temporary_metric < metric:
                interval = temporary_interval.copy()
                metric = temporary_metric
            if np.all(temporary_interval[:, 0] <= 0) and np.all(
                temporary_interval[:, 1] >= 1
            ):
                break

        if np.all(interval[:, 0] <= 0) and np.all(interval[:, 1] >= 1):
            continue

        initial_interval_norm = interval.copy()
        initial_interval_denorm = inverse_transform(
            initial_interval_norm.T, scaler, input_columns
        )
        initial_metric, initial_loss_mean = calculate_metric(
            initial_interval_norm, X_array, Y_array, M_min=M_min, beta=beta
        )
        initial_points = int(np.sum(generate_mask(X_array, interval)))

        # Phase 2: adjust each boundary until no candidate improves the metric.
        for _ in range(max_steps_iter):
            best_metric = metric
            best_interval = interval.copy()
            step_mask = generate_mask(X_array, interval)
            metric_improved = False

            for dimension in range(interval.shape[0]):
                for side in (0, 1):
                    for direction in (-1, 1):
                        step_multiplier = 1
                        while step_iter * step_multiplier <= 1:
                            candidate = interval.copy()
                            candidate[dimension][side] += direction * step_iter * step_multiplier
                            candidate[dimension][side] = np.clip(
                                candidate[dimension][side], 0, 1
                            )
                            candidate_mask = generate_mask(X_array, candidate)
                            if not np.array_equal(step_mask, candidate_mask):
                                candidate_metric, _ = calculate_metric(
                                    candidate, X_array, Y_array, M_min=M_min, beta=beta
                                )
                                if candidate_metric <= best_metric:
                                    best_metric = candidate_metric
                                    best_interval = candidate.copy()
                                    metric_improved = True
                                break
                            step_multiplier += 1

            if not metric_improved:
                break
            interval = best_interval
            metric = best_metric
            if np.all(interval[:, 0] <= 0) and np.all(interval[:, 1] >= 1):
                break

        # Expand to a neighbouring boundary without changing the selected set;
        # the midpoint becomes the stable final interval.
        suboptimal_interval = interval.copy()
        last_mask = generate_mask(X_array, interval)
        for dimension in range(interval.shape[0]):
            for side in (0, 1):
                direction = -1 if side == 0 else 1
                sub_multiplier = 1
                while step_iter * sub_multiplier <= 1:
                    candidate = suboptimal_interval.copy()
                    candidate[dimension][side] += direction * step_iter * sub_multiplier
                    candidate[dimension][side] = np.clip(candidate[dimension][side], 0, 1)
                    candidate_mask = generate_mask(X_array, candidate)
                    midpoint = (interval + candidate) / 2
                    midpoint_mask = generate_mask(X_array, midpoint)
                    if not np.array_equal(last_mask, candidate_mask) and np.array_equal(
                        generate_mask(X_array, interval), midpoint_mask
                    ):
                        suboptimal_interval[dimension][side] = candidate[dimension][side]
                        last_mask = candidate_mask
                        break
                    sub_multiplier += 1

        optimal_interval_norm = interval.copy()
        suboptimal_interval_norm = suboptimal_interval.copy()
        final_interval_norm = (optimal_interval_norm + suboptimal_interval_norm) / 2

        optimal_metric, optimal_loss_mean = calculate_metric(
            optimal_interval_norm, X_array, Y_array, M_min=M_min, beta=beta
        )
        suboptimal_metric, suboptimal_loss_mean = calculate_metric(
            suboptimal_interval_norm, X_array, Y_array, M_min=M_min, beta=beta
        )
        final_metric, final_loss_mean = calculate_metric(
            final_interval_norm, X_array, Y_array, M_min=M_min, beta=beta
        )

        optimal_points = int(np.sum(generate_mask(X_array, optimal_interval_norm)))
        suboptimal_points = int(np.sum(generate_mask(X_array, suboptimal_interval_norm)))
        final_points = int(np.sum(generate_mask(X_array, final_interval_norm)))

        optimal_interval_denorm = inverse_transform(
            optimal_interval_norm.T, scaler, input_columns
        )
        suboptimal_interval_denorm = inverse_transform(
            suboptimal_interval_norm.T, scaler, input_columns
        )
        final_interval_denorm = inverse_transform(final_interval_norm.T, scaler, input_columns)

        initial_interval_norm_log = log_transform(X_array, initial_interval_norm)
        optimal_interval_norm_log = log_transform(X_array, optimal_interval_norm)
        suboptimal_interval_norm_log = log_transform(X_array, suboptimal_interval_norm)
        final_interval_norm_log = log_transform(X_array, final_interval_norm)

        if not np.isclose(optimal_metric, final_metric, equal_nan=True):
            raise ValueError("The final midpoint changed the selected interval metric.")

        def volume(bounds: np.ndarray) -> float:
            return float(np.prod(bounds[:, 1] - bounds[:, 0]))

        results.append(
            {
                "model": position,
                "point_interval_norm": point_interval_norm.tolist(),
                "point_interval_denorm": point_interval_denorm.values.tolist(),
                "initial_interval_norm": initial_interval_norm.tolist(),
                "initial_interval_norm_log": initial_interval_norm_log.tolist(),
                "initial_interval_denorm": initial_interval_denorm.values.tolist(),
                "optimal_interval_norm": optimal_interval_norm.tolist(),
                "optimal_interval_norm_log": optimal_interval_norm_log.tolist(),
                "optimal_interval_denorm": optimal_interval_denorm.values.tolist(),
                "suboptimal_interval_norm": suboptimal_interval_norm.tolist(),
                "suboptimal_interval_norm_log": suboptimal_interval_norm_log.tolist(),
                "suboptimal_interval_denorm": suboptimal_interval_denorm.values.tolist(),
                "final_interval_norm": final_interval_norm.tolist(),
                "final_interval_norm_log": final_interval_norm_log.tolist(),
                "final_interval_denorm": final_interval_denorm.values.tolist(),
                "initial_metric": initial_metric,
                "optimal_metric": optimal_metric,
                "suboptimal_metric": suboptimal_metric,
                "final_metric": final_metric,
                "initial_points": initial_points,
                "optimal_points": optimal_points,
                "suboptimal_points": suboptimal_points,
                "final_points": final_points,
                "initial_range": volume(initial_interval_norm),
                "optimal_range": volume(optimal_interval_norm),
                "suboptimal_range": volume(suboptimal_interval_norm),
                "final_range": volume(final_interval_norm),
                "initial_range_log": volume(initial_interval_norm_log),
                "optimal_range_log": volume(optimal_interval_norm_log),
                "suboptimal_range_log": volume(suboptimal_interval_norm_log),
                "final_range_log": volume(final_interval_norm_log),
                "initial_loss_mean": initial_loss_mean,
                "optimal_loss_mean": optimal_loss_mean,
                "suboptimal_loss_mean": suboptimal_loss_mean,
                "final_loss_mean": final_loss_mean,
            }
        )

        if verbose:
            elapsed = time.time() - started_at
            print(
                f"Processed starting point {position}/{len(X_array)} "
                f"(metric={final_metric:.6g}, elapsed={elapsed:.2f}s)",
                end="\r",
            )

    if verbose:
        print()
    return pd.DataFrame(results, columns=RESULT_COLUMNS)


def _validate_inputs(
    X: pd.DataFrame,
    Y: pd.Series | pd.DataFrame,
    input_columns: list[str],
    step_int: float,
    step_iter: float,
    max_steps_int: int,
    max_steps_iter: int,
    M_min: float,
    beta: float,
) -> None:
    """Validate public API inputs before beginning the expensive search."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")
    if X.empty:
        raise ValueError("X must contain at least one evaluated configuration.")
    if list(X.columns) != input_columns:
        raise ValueError("input_columns must match X.columns in the same order.")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes):
        raise TypeError("All hyperparameter columns in X must be numeric.")
    if X.isna().any().any():
        raise ValueError("X must not contain missing values.")

    objective = np.asarray(Y).reshape(-1)
    if len(objective) != len(X):
        raise ValueError("X and Y must contain the same number of rows.")
    if not np.issubdtype(objective.dtype, np.number):
        raise TypeError("Y must contain numeric objective values.")
    if not np.isfinite(objective.astype(float)).all():
        raise ValueError("Y must contain only finite objective values.")

    if not 0 < step_int <= 1:
        raise ValueError("step_int must lie in (0, 1].")
    if not 0 < step_iter <= 1:
        raise ValueError("step_iter must lie in (0, 1].")
    if max_steps_int < 1 or max_steps_iter < 1:
        raise ValueError("max_steps_int and max_steps_iter must be positive integers.")
    if M_min < 1 or M_min > len(X):
        raise ValueError("M_min must lie between 1 and the number of evaluations.")
    if beta < 0:
        raise ValueError("beta must be non-negative.")

