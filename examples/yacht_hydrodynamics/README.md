# Yacht Hydrodynamics demonstration

This example preserves and corrects the original AIBA proof of concept. It trains
surrogate models over a broad hyperparameter domain, runs AIBA on their observed
quality, then samples a second set of models inside the recommended interval.

For a quick review, the notebook defaults to the included precomputed model
configurations and corrected AIBA results. Recomputing the neural-network portion
requires the optional `demo` dependencies and can be computationally expensive.

## Corrections applied to the archived prototype

The working materials are preserved unchanged under `AIBA/info_base`. This public
copy corrects three methodological issues found during repository preparation:

1. `Quality` (the observed R²) is no longer included as an AIBA input feature.
2. `actual_epochs` and `num_params`, which are known only after training, are no
   longer treated as controllable hyperparameters.
3. The AIBA objective is `-Quality`, because AIBA minimizes whereas R² is
   maximized.

`results_aiba.csv` was regenerated from the 1,000 archived evaluations after
these corrections. It contains 527 valid candidate intervals over seven
controllable hyperparameters.

The notebook's final plot uses an equal-size retrospective sample from the broad
evaluations. This is useful as a smoke test but does not include the cost of
building AIBA's initial evidence base. For a publication-quality efficiency
claim, execute the prospective equal-budget protocol in
[`../../docs/reproducibility.md`](../../docs/reproducibility.md).

## Replicated budget-matched comparison

The directory [`replicated_budget_comparison`](replicated_budget_comparison)
contains the final prospective comparison. Its two logical strategies share 100
initial broad-search evaluations and then allocate 20 additional evaluations in
different ways:

- AIBA samples the 20 configurations inside the recommended interval;
- the baseline samples 20 additional configurations over the complete domain.

The included notebook was executed for 10 independent replications and retains
its tables and plots. Reference CSV files are provided under
`results/reference_run_10_replicates/`.

## Dataset

The Delft Yacht Hydrodynamics dataset contains 308 experiments and is used here
as a regression problem for residuary resistance.

- Creators: J. Gerritsma, R. Onnink and A. Versluis
- Source: UCI Machine Learning Repository
- DOI: <https://doi.org/10.24432/C5XG7R>
- License: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)

Suggested citation:

> Gerritsma, J., Onnink, R., & Versluis, A. (1981). Yacht Hydrodynamics
> [Dataset]. UCI Machine Learning Repository. DOI: 10.24432/C5XG7R.

## Files

- `yacht_hydrodynamics_modelling.ipynb`: end-to-end demonstration.
- `yacht_hydrodynamics.txt`: UCI dataset in semicolon-separated form.
- `yacht_model_configs_test.csv`: precomputed broad-search configurations.
- `results_aiba.csv`: precomputed intervals produced by the original experiment.

The two CSV files are example artifacts produced by the research workflow, not
part of the UCI dataset license. Their software-release attribution must be
confirmed together with the AIBA code before publication.
