# Practical drift monitor

The clean reference combines 200 Construction-PPE validation images and 118 filtered
RF100 validation images. It stores scalar mean/standard deviation, mean normalized image
histograms, and the five-class output-frequency distribution in
`configs/drift_reference.json`. No RF100 test data contributes to it.

Original width and height are profiled for observability but excluded from the composite
score; aspect ratio is scored. This prevents heterogeneous native resolutions from
overwhelming content-shift signals.

For histogram bins, population stability index is:

`PSI = sum((observed - reference) * ln(observed / reference))`

Both distributions receive `1e-6` additive smoothing and are renormalized. Scalar signals
use absolute reference-standard-deviation shifts. The input score is the maximum histogram
PSI or the maximum scalar shift divided by 10 and capped at 1. Output drift is PSI over
the five class frequencies. Overall drift is the maximum input or output score.

The transparent engineering categories are normal `< 0.10`, warning `0.10 to < 0.25`,
and high `>= 0.25`. These familiar generic PSI ranges were predeclared; they were not
retuned after viewing outcomes. Scores can exceed 1 and are not probabilities.

Clean CPPE scores 0.0416 (normal). Clean filtered RF100 scores 0.2214 (warning), driven by
the different output-class distribution; its input-only score is 0.0855. This is an
important false-alarm result: legitimate domain variation can raise an alert even though
both domains helped form the combined reference. The policy should trigger investigation,
not automatically disable predictions or declare model failure.

Synthetic brightness decrease, contrast reduction, and saturation reduction create large
histogram PSI despite modest accuracy loss. Conversely, CPPE severe motion blur causes a
meaningful violation-recall loss while its aggregate drift score is only 0.077 (normal).
The monitor therefore has blind spots as well as false alarms.

