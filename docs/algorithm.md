# Algorithm overview

## Problem setting

AIBA receives a set of completed model evaluations. Each row contains a
hyperparameter configuration and its scalar loss. Its purpose is to identify
rectangular regions that are promising for a subsequent, fixed-budget search.

It is therefore a two-stage experimental workflow:

1. evaluate an initial, broadly sampled population of configurations;
2. use AIBA to delimit a promising region and spend a new training budget there.

To make efficiency claims meaningful, the guided second stage should be compared
with a baseline using the same number of new model-training executions.

## Interval metric

For the observations contained in interval \(I\), AIBA minimizes

\[
J(I) = \overline{L}_I + \beta \frac{s_I}{\sqrt{M_I}},
\]

where \(\overline{L}_I\) is mean loss, \(s_I\) is sample standard deviation,
\(M_I\) is the number of observations and \(\beta\) controls the uncertainty
penalty. Intervals with fewer than `M_min` observations are invalid.

## Search phases

### 1. Initial expansion

Starting points are considered from lowest to highest observed loss. Each point
begins as a zero-volume interval. Bounds expand symmetrically in normalized space
until the metric stops improving or the complete domain is reached.

### 2. Iterative optimization

Each lower and upper bound is moved in both directions. A movement continues
until it changes which observed configurations are inside the interval. The best
metric-improving modification becomes the next interval. Iteration stops when no
boundary modification improves the metric.

### 3. Stable final bounds

Each optimized boundary is extended toward the nearest excluded observation. The
midpoint between the optimized and neighbouring boundary is reported as the
final bound, preserving the selected set while avoiding a boundary directly on
an observation.

## Interpretation

AIBA returns several candidate intervals, not a newly trained model. A typical
consumer sorts candidates by `final_metric`, chooses a region, samples new
hyperparameter configurations inside its denormalized bounds and performs the
budget-matched validation experiment.

## Current limitations

- The objective is minimized; maximization scores must be negated or converted
  to a loss.
- Hyperparameters must be numeric before calling AIBA.
- Regions are axis-aligned hyperrectangles.
- Computational cost grows with the number of prior evaluations and dimensions.
- Claims of reduced training cost depend on the surrounding experimental design,
  not solely on running AIBA.

