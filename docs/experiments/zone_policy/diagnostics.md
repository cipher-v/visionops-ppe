# Deterministic zone-policy diagnostics

This is synthetic correctness evidence, not real-world zone accuracy.

| Time (s) | Person | Raw zones | Stable zones | Helmet evidence | Policy state | Events |
| ---: | --- | --- | --- | --- | --- | --- |
| 0.00 | Person A | none | none | present | not_required | - |
| 1.00 | Person A | ppe_zone_1 | none | present | not_required | - |
| 1.40 | Person A | ppe_zone_1 | ppe_zone_1 | present | compliant | - |
| 5.00 | Person B | none | none | violation | not_required | - |
| 9.00 | Person B | none | none | violation | not_required | - |
| 10.00 | Person B | ppe_zone_1 | none | violation | not_required | - |
| 10.40 | Person B | ppe_zone_1 | ppe_zone_1 | violation | suspected_violation | - |
| 11.00 | Person B | ppe_zone_1 | ppe_zone_1 | violation | suspected_violation | - |
| 11.90 | Person B | ppe_zone_1 | ppe_zone_1 | violation | confirmed_violation | incident_opened |
| 12.50 | Person B | none | ppe_zone_1 | violation | confirmed_violation | - |
| 12.90 | Person B | none | none | violation | not_required | incident_closed |
| 20.00 | Person C | none | none | violation | not_required | - |
| 20.10 | Person C | ppe_zone_1 | none | violation | not_required | - |
| 20.30 | Person C | none | none | violation | not_required | - |

Person A enters and remains compliant. Person B's outside-zone negative evidence is suppressed; its timer starts at stable entry (10.40 s), the incident opens at 11.90 s, and stable exit closes it at 12.90 s with `zone_exited`. Person C's 0.20-second crossing never becomes stable membership.

Expected values exactly matched actual values; see `metrics.json`.
