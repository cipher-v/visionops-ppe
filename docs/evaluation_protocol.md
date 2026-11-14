# Evaluation protocol

## Dataset roles

- Construction-PPE `train` is the model training set.
- Construction-PPE `val` is the primary in-domain development split.
- The RF100 Construction Safety official `test` split was used once in Checkpoint 5
  to establish the initial independent-domain baseline.
- From Checkpoint 6 onward, RF100 `test` is frozen and must not be used for iterative
  experiment selection.
- No further model tuning may inspect RF100 `test` metrics until a candidate has been
  selected using a predeclared procedure for final evaluation.
- RF100 `train` and `valid` may be used only for clearly documented domain
  characterization or out-of-domain development analysis unless a later, explicit
  experiment changes the project into a domain-adaptation setting.

No dataset split is altered by this policy. Any future exception must be declared before
results are inspected and must describe the resulting evaluation limitations.
