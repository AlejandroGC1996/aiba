# Reproducibility protocol

Use this protocol when evaluating whether AIBA improves hyperparameter search.

1. Define the full hyperparameter domain and a deterministic sampling procedure.
2. Generate an initial set of configurations and record every seed.
3. Train and evaluate all initial configurations under the same conditions.
4. Run AIBA only on those completed evaluations.
5. Select an interval using a rule fixed before inspecting final test results.
6. Draw `N` new configurations from the selected interval.
7. Draw `N` baseline configurations from the full domain.
8. Train both groups with identical data splits, stopping rules and resources.
9. Report best, mean, dispersion and wall-clock/compute cost for both groups.
10. Repeat the comparison across multiple random seeds.

The number of model-training executions must be reported separately from AIBA's
own CPU time. This distinguishes savings in expensive training or simulation
from the comparatively small cost of interval analysis.

Generated outputs should be written below an example's `generated/` directory;
that directory is ignored by Git by default.

## Paired low-budget design used in the Yacht example

The final Yacht Hydrodynamics demonstration uses a paired variant of this
protocol. Both strategies share the same initial evaluations, then differ only
in allocation of the remaining budget. This reduces experimental variance and
training cost while preserving an equal logical budget:

```text
shared initial evaluations + guided extension = AIBA strategy
shared initial evaluations + broad extension  = broad-search strategy
```

Independent replications must change the data split, configuration-sampling and
training seeds together, while keeping those choices shared within each paired
comparison. Report paired differences per replication instead of treating all
trained models as independent observations.
