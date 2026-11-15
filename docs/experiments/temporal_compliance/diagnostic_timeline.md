# Deterministic temporal event timeline

| Time (s) | Evidence | Temporal state | Violation time | Incident active |
|---:|---|---|---:|---|
| 0.00 | violation | suspected_violation | 0.00 s | False |
| 0.60 | violation | suspected_violation | 0.60 s | False |
| 0.75 | violation | suspected_violation | 0.75 s | False |
| 1.00 | violation | suspected_violation | 1.00 s | False |
| 1.50 | violation | confirmed_violation | 1.50 s | True |
| 2.00 | unknown | confirmed_violation | 1.50 s | True |
| 2.20 | violation | confirmed_violation | 1.50 s | True |
| 2.50 | conflict | conflict | 1.50 s | True |
| 2.80 | violation | confirmed_violation | 1.50 s | True |
| 3.00 | present | confirmed_violation | 0.00 s | True |
| 4.00 | present | compliant | 0.00 s | False |

Expected and actual events match exactly:

- `1.50 s`: track 2 helmet incident opened (candidate began at 0.00 s).
- `4.00 s`: the incident closed after 1.00 s sustained present evidence.
- Track 1 stayed compliant; track 3's transient no-vest episode opened
  no incident.

This is deterministic synthetic correctness, not real-world accuracy.
