# Zone-engine benchmark

The repository-owned microbenchmark evaluated 5000
person observations. It measures only polygon membership and timestamp stabilization,
excluding detector, tracker, PPE association, temporal compliance, file I/O, and
rendering.

- Mean: 0.005563 ms/observation
- p50: 0.005200 ms/observation
- p95: 0.007000 ms/observation

These workstation-specific measurements are an engineering sanity check, not a
deployment SLA. Re-run `python scripts/run_zone_policy_diagnostics.py` on the target
system before capacity planning.
