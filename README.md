# AIBA

**Adaptive interval search for efficient hyperparameter tuning**

AIBA is an iterative search algorithm that identifies promising hyperparameter
regions from a table of previously evaluated machine-learning configurations.
Instead of spending the next training budget over the full search space, AIBA
recommends narrower intervals in which new configurations can be sampled.

The intended comparison is budget-matched: a guided search inside the intervals
recommended by AIBA is compared with an unguided search using the same number of
new model-training executions.

## How it works

Given:

- `X`: one row per evaluated model and one column per hyperparameter;
- `y`: the loss or cost obtained by each model (lower is better);
- `M_min`: the minimum number of evaluated models required inside a candidate
  interval; and
- `beta`: the uncertainty penalty;

AIBA expands an interval around each promising observed configuration, then
iteratively adjusts its bounds. Candidate regions are scored as

```text
metric = mean(loss) + beta * standard_error(loss)
```

Lower metric values indicate regions combining good average performance with
low uncertainty. See [the algorithm description](docs/algorithm.md) for details.

## Installation

From the repository root:

```bash
python -m pip install .
```

For development and tests:

```bash
python -m pip install -e ".[dev]"
pytest
```

The full Yacht Hydrodynamics demonstration has additional dependencies:

```bash
python -m pip install -e ".[demo]"
```

## Quick start

```python
import pandas as pd
from aiba import optimize_intervals

evaluations = pd.DataFrame(
    {
        "learning_rate": [1e-4, 3e-4, 1e-3, 3e-3],
        "batch_size": [16, 32, 32, 64],
    }
)
loss = pd.Series([0.42, 0.31, 0.29, 0.38], name="validation_loss")

regions = optimize_intervals(
    X=evaluations,
    Y=loss,
    input_columns=list(evaluations.columns),
    M_min=2,
    beta=0.5,
)

best_region = regions.sort_values("final_metric").iloc[0]
print(best_region["final_interval_denorm"])
```

The returned data frame contains the initial, optimized and final interval for
each accepted starting point, in normalized and original hyperparameter scales,
together with the corresponding metrics and point counts.

## Reproducible example

The [`examples/yacht_hydrodynamics`](examples/yacht_hydrodynamics) directory
contains a budget-matched comparison between AIBA-guided search and broad random
search. The experiment can be run with one replication for a quick functional
check or with ten replications for the full comparison. Its design and reporting
protocol are documented in
[`docs/reproducibility.md`](docs/reproducibility.md).

The Yacht Hydrodynamics dataset is distributed under CC BY 4.0. Its attribution
and DOI are recorded alongside the example.

### Replicated budget-matched result

The executed notebook
[`aiba_replicated_budget_comparison.ipynb`](examples/yacht_hydrodynamics/replicated_budget_comparison/aiba_replicated_budget_comparison.ipynb)
compares AIBA with broad search over 10 paired replications. Both strategies
share 100 initial evaluations and differ only in how they allocate the remaining
20 evaluations: AIBA samples inside its recommended interval, whereas the
baseline continues sampling the full domain.

Across the 10 replications, AIBA improved mean R², median R², the 75th percentile
and the number of models above R² thresholds of 0.90 and 0.95 in every
replication. The absolute best R² was effectively tied, showing that AIBA's main
benefit in this experiment is a higher density and reliability of good
configurations under a small remaining budget, rather than a higher attainable
maximum. Full CSV results and the executed outputs are included with the example.

## Scope

AIBA currently assumes:

- numeric hyperparameters;
- a scalar objective to minimize;
- a tabular set of completed evaluations; and
- rectangular search regions.

AIBA does not train models or choose a surrogate architecture by itself. Model
training, sampling inside the recommended intervals and final validation belong
to the surrounding experiment.

## Academic reference

The algorithm is described in the 2025 Master's thesis *Diseño y análisis de
modelos subrogados de gemelos digitales para optimización de procesos
industriales*, by José María Vecino Otero, supervised by Abraham Prieto García
and Alejandro González Casal at Universidade da Coruña.

The thesis is cited here as academic context but its PDF is not distributed in
this repository.

## License and attribution

AIBA is released under the [MIT License](LICENSE). The software authors are José
María Vecino Otero, Alejandro González Casal and Abraham Prieto García; see
[`AUTHORS.md`](AUTHORS.md) and [`CITATION.cff`](CITATION.cff).

The Yacht Hydrodynamics dataset is separately licensed under CC BY 4.0 and must
retain its own attribution when redistributed.
