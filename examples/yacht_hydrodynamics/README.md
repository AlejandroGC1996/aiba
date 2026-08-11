# Yacht Hydrodynamics example

This example evaluates AIBA in a budget-matched hyperparameter search for neural
network surrogate models. The two strategies share an initial set of 100 model
evaluations and allocate the remaining 20 evaluations differently:

- AIBA samples configurations inside the interval recommended from the initial
  evidence;
- the baseline continues sampling over the complete hyperparameter domain.

The comparison therefore assigns the same logical budget of 120 evaluated models
to each strategy. Because the initial evidence is shared, each replication
requires 140 unique model trainings rather than 240.

The complete experiment is in
[`replicated_budget_comparison`](replicated_budget_comparison). Its notebook can
be run with one replication for a quick functional check or with ten replications
for the full statistical comparison. The repository includes the executed
ten-replication notebook and its reference result tables.

## Methodological choices

AIBA receives only the controllable hyperparameters. Model quality (`R²`) and
quantities known only after training, such as the actual number of epochs and the
number of model parameters, are excluded from its input features. Since AIBA
minimizes its objective, the interval search uses `-R²` as its loss.

## Dataset

The Delft Yacht Hydrodynamics dataset contains 308 experiments and is used as a
regression problem for residuary resistance.

- Creators: J. Gerritsma, R. Onnink and A. Versluis
- Source: UCI Machine Learning Repository
- DOI: <https://doi.org/10.24432/C5XG7R>
- License: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)

Suggested citation:

> Gerritsma, J., Onnink, R., & Versluis, A. (1981). Yacht Hydrodynamics
> [Dataset]. UCI Machine Learning Repository. DOI: 10.24432/C5XG7R.

The dataset is stored in `yacht_hydrodynamics.txt`. Its CC BY 4.0 license and
attribution are independent of the MIT license covering the AIBA software.
