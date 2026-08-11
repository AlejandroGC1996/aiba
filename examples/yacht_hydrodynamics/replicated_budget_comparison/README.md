# Replicated budget-matched comparison

This directory contains the final executed AIBA proof of concept. It tests
whether AIBA uses a limited remaining model-training budget more effectively than
continued broad search.

## Experimental design

Each replication begins with 100 broad-search model evaluations shared by both
logical strategies. AIBA uses those observations to recommend an interval.
The remaining budget of 20 evaluations is allocated as follows:

- **AIBA:** 20 configurations sampled inside the recommended interval;
- **Broad:** 20 new configurations sampled over the complete domain.

Each strategy therefore consumes 120 logical model evaluations. Because the
first 100 are shared, only 140 unique models must be trained per replication.
The reference run uses 10 independent seeds, for 1,400 unique trainings.

## Files

- `aiba_replicated_budget_comparison.ipynb`: executed notebook with tables and
  plots. Set `N_REPLICATIONS = 1` for a quick check or `10` for the full run.
- `replicated_experiment.py`: training, sampling, checkpointing and aggregation
  functions used by the notebook.
- `results/reference_run_10_replicates/`: configuration and complete CSV output
  from the saved 10-replication run.

New runs are written to `generated_replicated/`, which is ignored by Git and does
not overwrite the reference results.

## Main result

Mean paired differences are reported as AIBA minus broad search:

| Metric | Mean difference | Approximate 95% interval | AIBA wins |
|---|---:|---:|---:|
| Best R² | -0.000003 | [-0.000382, 0.000375] | 2 wins, 5 ties, 3 losses |
| Mean R² | +0.110342 | [0.093392, 0.127291] | 10/10 |
| Median R² | +0.384441 | [0.306561, 0.462320] | 10/10 |
| 75th percentile R² | +0.065179 | [0.040605, 0.089753] | 10/10 |
| 90th percentile R² | +0.005625 | [0.003218, 0.008033] | 9 wins, 1 tie |
| Models with R² ≥ 0.90 | +8.8 | [6.56, 11.04] | 10/10 |
| Models with R² ≥ 0.95 | +8.2 | [5.90, 10.50] | 10/10 |

The absolute maximum is effectively unchanged. AIBA's observed advantage is a
higher concentration of strong configurations: the final 20 evaluations are
more reliable, rather than extending the apparent maximum R².

## Cost interpretation

AIBA required approximately 59.6 additional seconds per replication on average,
including interval search. Equal model counts do not imply equal wall-clock,
energy or FLOP budgets because guided configurations can select larger networks
or more epochs. The CSV files therefore retain training and search times.

## Limitations

- Results concern one dataset, surrogate family and search domain.
- Ten paired replications quantify seed variation but are not a universal
  benchmark.
- The reported 95% intervals use a normal approximation over 10 paired
  differences.
- R² can be strongly negative for failed configurations; medians, percentiles and
  threshold counts should be interpreted alongside means.

