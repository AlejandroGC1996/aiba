"""Utilities for the replicated, budget-matched AIBA experiment.

This module belongs to the external experiment workspace, not to the public AIBA
package. The paired design shares the initial broad-search evaluations between
both logical strategies and compares only how the remaining budget is allocated.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import random
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras


HYPERPARAMETER_COLUMNS = [
    "batch_size",
    "max_epochs",
    "learning_rate",
    "neurons",
    "dropout_rate",
    "num_hidden_layers",
    "activation",
]
ACTIVATIONS = {0.0: "relu", 0.5: "sigmoid", 1.0: "tanh"}


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration shared by every independent replication."""

    n_replications: int = 1
    n_initial_aiba: int = 100
    n_guided_aiba: int = 20
    base_seed: int = 2026
    test_size: float = 0.20
    aiba_min_fraction: float = 0.02
    aiba_beta: float = 1.0
    aiba_initial_step: float = 0.20
    aiba_iterative_step: float = 0.01
    aiba_max_initial_steps: int = 5
    aiba_max_iterations: int = 500

    @property
    def total_budget(self) -> int:
        return self.n_initial_aiba + self.n_guided_aiba

    def validate(self) -> None:
        if self.n_replications < 1:
            raise ValueError("n_replications must be at least 1.")
        if self.n_initial_aiba < 2:
            raise ValueError("n_initial_aiba must be at least 2.")
        if self.n_guided_aiba < 1:
            raise ValueError("n_guided_aiba must be at least 1.")
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must lie in (0, 1).")


def configure_local_import(repository: Path) -> None:
    """Make the local AIBA source tree importable by the experiment."""
    source = str((repository / "src").resolve())
    if source not in sys.path:
        sys.path.insert(0, source)


def load_dataset(data_path: Path, config: ExperimentConfig, replication_seed: int):
    """Load and split Yacht Hydrodynamics for one replication."""
    column_names = [
        "Center of buoyancy",
        "Prismatic coefficient",
        "Length-displacement",
        "Beam-draught",
        "Length-beam",
        "Froude number",
        "Residuary resistance",
    ]
    dataset = pd.read_csv(data_path, sep=";", header=None, names=column_names)
    X_data = dataset.drop(columns=["Residuary resistance"])
    y_data = dataset["Residuary resistance"]
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_data,
        y_data,
        test_size=config.test_size,
        random_state=replication_seed,
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    return X_train, X_test, y_train, y_test


def log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def activation_name(value: float) -> str:
    key = min(ACTIVATIONS, key=lambda candidate: abs(candidate - float(value)))
    return ACTIVATIONS[key]


def sample_broad_configuration(rng: np.random.Generator) -> dict[str, float | int]:
    """Sample one configuration from the complete search domain."""
    return {
        "batch_size": int(np.clip(round(log_uniform(rng, 1, 256)), 1, 256)),
        "max_epochs": int(np.clip(round(log_uniform(rng, 1, 400)), 1, 400)),
        "learning_rate": log_uniform(rng, 1e-5, 1e-1),
        "neurons": int(np.clip(round(log_uniform(rng, 3, 600)), 3, 600)),
        "dropout_rate": log_uniform(rng, 1e-5, 0.7),
        "num_hidden_layers": int(rng.integers(1, 4)),
        "activation": float(rng.choice(list(ACTIVATIONS))),
    }


def sample_guided_configuration(
    rng: np.random.Generator,
    lower: list[float],
    upper: list[float],
) -> dict[str, float | int]:
    """Sample one configuration inside a denormalized AIBA interval."""
    configuration: dict[str, float | int] = {}
    log_scaled = {"batch_size", "max_epochs", "learning_rate", "neurons", "dropout_rate"}
    integer_scaled = {"batch_size", "max_epochs", "neurons", "num_hidden_layers"}
    for index, name in enumerate(HYPERPARAMETER_COLUMNS):
        low, high = sorted((float(lower[index]), float(upper[index])))
        if name in log_scaled:
            value = log_uniform(rng, max(low, 1e-8), max(high, low + 1e-8))
        elif name == "activation":
            allowed = [item for item in ACTIVATIONS if low <= item <= high]
            fallback = min(ACTIVATIONS, key=lambda item: abs(item - (low + high) / 2))
            value = float(rng.choice(allowed or [fallback]))
        else:
            value = float(rng.uniform(low, high)) if high > low else low
        configuration[name] = int(round(value)) if name in integer_scaled else value
    return configuration


def train_and_evaluate(
    configuration: dict[str, float | int],
    seed: int,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, float | int]:
    """Train one neural surrogate and return its test score and resource metadata."""
    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    model = keras.Sequential([keras.layers.Input(shape=(X_train.shape[1],))])
    for _ in range(int(configuration["num_hidden_layers"])):
        model.add(
            keras.layers.Dense(
                int(configuration["neurons"]),
                activation=activation_name(float(configuration["activation"])),
            )
        )
        model.add(keras.layers.Dropout(float(configuration["dropout_rate"])))
    model.add(keras.layers.Dense(1))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=float(configuration["learning_rate"])),
        loss="mse",
    )
    started = time.perf_counter()
    history = model.fit(
        X_train,
        y_train,
        epochs=int(configuration["max_epochs"]),
        batch_size=int(configuration["batch_size"]),
        verbose=0,
    )
    predictions = model.predict(X_test, verbose=0).ravel()
    elapsed = time.perf_counter() - started
    return {
        **configuration,
        "r2": float(r2_score(y_test, predictions)),
        "training_seconds": elapsed,
        "actual_epochs": len(history.history["loss"]),
        "num_params": int(model.count_params()),
        "seed": seed,
    }


def run_configurations(
    configurations: list[dict[str, float | int]],
    seed_offset: int,
    label: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
) -> pd.DataFrame:
    rows = []
    for position, configuration in enumerate(configurations, start=1):
        row = train_and_evaluate(
            configuration,
            seed_offset + position,
            X_train,
            X_test,
            y_train,
            y_test,
        )
        rows.append(row)
        print(f"{label}: {position}/{len(configurations)}; R²={row['r2']:.4f}", end="\r")
    print()
    return pd.DataFrame(rows)


def summarize(replication: int, strategy: str, frame: pd.DataFrame, search_seconds=0.0):
    values = frame["r2"].to_numpy()
    return {
        "replication": replication,
        "strategy": strategy,
        "models": len(frame),
        "best_r2": np.max(values),
        "mean_r2": np.mean(values),
        "median_r2": np.median(values),
        "p75_r2": np.percentile(values, 75),
        "p90_r2": np.percentile(values, 90),
        "above_090": np.sum(values >= 0.90),
        "above_095": np.sum(values >= 0.95),
        "training_seconds": frame["training_seconds"].sum(),
        "search_seconds": search_seconds,
        "total_seconds": frame["training_seconds"].sum() + search_seconds,
    }


def best_so_far(replication: int, strategy: str, values: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "replication": replication,
            "strategy": strategy,
            "budget": np.arange(1, len(values) + 1),
            "best_r2": np.maximum.accumulate(values.to_numpy()),
        }
    )


def run_replications(
    config: ExperimentConfig,
    repository: Path,
    output_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Run paired replications and checkpoint all results after each one."""
    config.validate()
    configure_local_import(repository)
    from aiba import optimize_intervals

    data_path = repository / "examples/yacht_hydrodynamics/yacht_hydrodynamics.txt"
    if not data_path.exists():
        raise FileNotFoundError(data_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "experiment_config.json").write_text(
        json.dumps({**asdict(config), "total_budget": config.total_budget}, indent=2),
        encoding="utf-8",
    )

    stage_frames = []
    logical_frames = []
    summaries = []
    curves = []
    interval_frames = []

    for replication in range(1, config.n_replications + 1):
        replication_seed = config.base_seed + replication * 100_000
        print(f"\n=== Replicación {replication}/{config.n_replications}; semilla {replication_seed} ===")
        random.seed(replication_seed)
        np.random.seed(replication_seed)
        keras.utils.set_random_seed(replication_seed)
        X_train, X_test, y_train, y_test = load_dataset(
            data_path, config, replication_seed
        )

        rng_initial = np.random.default_rng(replication_seed + 10)
        initial_configs = [
            sample_broad_configuration(rng_initial) for _ in range(config.n_initial_aiba)
        ]
        initial = run_configurations(
            initial_configs,
            replication_seed + 10_000,
            "Inicial compartido",
            X_train,
            X_test,
            y_train,
            y_test,
        )

        m_min = max(2, int(np.ceil(config.aiba_min_fraction * config.n_initial_aiba)))
        search_started = time.perf_counter()
        intervals = optimize_intervals(
            X=initial[HYPERPARAMETER_COLUMNS],
            Y=-initial["r2"],
            input_columns=HYPERPARAMETER_COLUMNS,
            step_int=config.aiba_initial_step,
            step_iter=config.aiba_iterative_step,
            max_steps_int=config.aiba_max_initial_steps,
            max_steps_iter=config.aiba_max_iterations,
            M_min=m_min,
            beta=config.aiba_beta,
        )
        search_seconds = time.perf_counter() - search_started
        if intervals.empty:
            raise RuntimeError(f"AIBA produced no interval in replication {replication}.")
        best_interval = intervals.nsmallest(1, "final_metric").iloc[0]
        lower, upper = best_interval["final_interval_denorm"]

        rng_guided = np.random.default_rng(replication_seed + 20)
        guided_configs = [
            sample_guided_configuration(rng_guided, lower, upper)
            for _ in range(config.n_guided_aiba)
        ]
        guided = run_configurations(
            guided_configs,
            replication_seed + 20_000,
            "Extensión guiada",
            X_train,
            X_test,
            y_train,
            y_test,
        )

        rng_broad = np.random.default_rng(replication_seed + 30)
        broad_extension_configs = [
            sample_broad_configuration(rng_broad) for _ in range(config.n_guided_aiba)
        ]
        broad_extension = run_configurations(
            broad_extension_configs,
            replication_seed + 30_000,
            "Extensión amplia",
            X_train,
            X_test,
            y_train,
            y_test,
        )

        initial_tagged = initial.assign(replication=replication, stage="initial_shared")
        guided_tagged = guided.assign(replication=replication, stage="guided_extension")
        broad_tagged = broad_extension.assign(
            replication=replication, stage="broad_extension"
        )
        stage_frames.extend([initial_tagged, guided_tagged, broad_tagged])

        aiba_strategy = pd.concat([initial, guided], ignore_index=True)
        broad_strategy = pd.concat([initial, broad_extension], ignore_index=True)
        assert len(aiba_strategy) == len(broad_strategy) == config.total_budget
        logical_frames.extend(
            [
                aiba_strategy.assign(replication=replication, strategy="AIBA"),
                broad_strategy.assign(replication=replication, strategy="Broad"),
            ]
        )
        summaries.extend(
            [
                summarize(replication, "AIBA", aiba_strategy, search_seconds),
                summarize(replication, "Broad", broad_strategy),
            ]
        )
        curves.extend(
            [
                best_so_far(replication, "AIBA", aiba_strategy["r2"]),
                best_so_far(replication, "Broad", broad_strategy["r2"]),
            ]
        )
        interval_frames.append(intervals.assign(replication=replication))

        _save_checkpoints(
            output_dir,
            stage_frames,
            logical_frames,
            summaries,
            curves,
            interval_frames,
        )
        print(
            f"Replicación {replication}: mejor AIBA={aiba_strategy.r2.max():.6f}; "
            f"mejor amplia={broad_strategy.r2.max():.6f}"
        )

    return _save_checkpoints(
        output_dir,
        stage_frames,
        logical_frames,
        summaries,
        curves,
        interval_frames,
    )


def _save_checkpoints(
    output_dir: Path,
    stage_frames: list[pd.DataFrame],
    logical_frames: list[pd.DataFrame],
    summaries: list[dict],
    curves: list[pd.DataFrame],
    interval_frames: list[pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Persist partial results so a long replicated run can be audited."""
    result = {
        "stage_results": pd.concat(stage_frames, ignore_index=True),
        "logical_results": pd.concat(logical_frames, ignore_index=True),
        "replication_summary": pd.DataFrame(summaries),
        "best_so_far": pd.concat(curves, ignore_index=True),
        "intervals": pd.concat(interval_frames, ignore_index=True),
    }
    for name, frame in result.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    return result


def paired_differences(summary: pd.DataFrame) -> pd.DataFrame:
    """Return AIBA minus broad-search metrics for every replication."""
    metrics = [
        "best_r2",
        "mean_r2",
        "median_r2",
        "p75_r2",
        "p90_r2",
        "above_090",
        "above_095",
        "total_seconds",
    ]
    pivot = summary.pivot(index="replication", columns="strategy", values=metrics)
    differences = pd.DataFrame(index=pivot.index)
    for metric in metrics:
        differences[f"delta_{metric}"] = pivot[(metric, "AIBA")] - pivot[(metric, "Broad")]
    return differences.reset_index()


def aggregate_differences(differences: pd.DataFrame) -> pd.DataFrame:
    """Aggregate paired differences with normal-approximation 95% intervals."""
    rows = []
    for column in differences.columns:
        if not column.startswith("delta_"):
            continue
        values = differences[column].to_numpy(dtype=float)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
        half_width = 1.96 * std / np.sqrt(len(values)) if len(values) > 1 else float("nan")
        rows.append(
            {
                "metric": column.removeprefix("delta_"),
                "replications": len(values),
                "mean_difference_aiba_minus_broad": mean,
                "std_difference": std,
                "ci95_low": mean - half_width,
                "ci95_high": mean + half_width,
                "aiba_wins": int(np.sum(values > 0)),
                "ties": int(np.sum(values == 0)),
                "broad_wins": int(np.sum(values < 0)),
            }
        )
    return pd.DataFrame(rows)

