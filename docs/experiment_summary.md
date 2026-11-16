# Experiment summary

| Experiment | Question/change | Result | Decision |
|---|---|---|---|
| YOLO26n baseline @640 | Establish 12-class baseline | mAP50-95 0.5461 | Reproducible starting point |
| Resolution @960 | Does resolution help small PPE? | 0.6012; tiny recall +0.150; higher latency | Keep 960 |
| YOLO26s capacity | Does 3.9× capacity generalize better? | Source 0.6343, ~2× inference; external test 0.3012 vs n 0.3029 | Keep n; capacity did not close gap |
| RF100 evaluation | Measure external-domain behavior | n external-test 0.3029 vs five-class internal 0.6611 | Diagnose domain shift |
| Domain-shift analysis | Is blur/small size the main cause? | RF100 boxes were larger; class/context shifted materially | Data diversity intervention |
| Mixed-source training C | Add 997 RF100 training images | Filtered RF100 val 0.3349→0.4948, CPPE retained 0.6642 | Select C |
| Exposure-matched D | Separate diversity from exposure | RF100 0.4722 at matched exposure | Diversity effect confirmed |
| Threshold calibration | Balance two validation domains | F1 CPPE 0.9020, RF100 0.9073 | Freeze class-specific policy |
| Real-video validation | Does full pipeline behave coherently? | Incidents/unknown/conflicts surfaced; tracking failures observed | Qualitative only; preserve uncertainty |
| Tracker comparison | Address walking identity transfer | BoT-SORT avoided primary transfer; ReID no visible benefit | BoT-SORT without ReID |
| ONNX parity | Can export replace PyTorch? | Final/output behavior diverged and controlled speed was worse | PyTorch production |
| Robustness/drift | What degrades, and can PSI warn? | Noise/motion blur hurt; PSI correlation near zero | Drift diagnostic-only |
| Review queue | Bound human inspection workload | 1,137 raw events→28 candidates | Human-only export, no retraining |

Detailed protocols, raw metrics, plots, and failures remain under [experiments](experiments/).
