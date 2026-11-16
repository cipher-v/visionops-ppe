# Predeclared visual review

## Review method

The two windows were fixed before alternative runs. Every candidate uses the same source
frames, spatial crop, frame labels, and timestamps. Contact sheets and short clips are under
ignored `results/tracker_robustness/review_artifacts/`; structured findings are recorded in
`review_manifest.csv`.

## Helmet duplicate: frames 104-112 (4.160-4.480 s)

- ByteTrack: the primary Track 1 is present at frame 104; Track 2 appears alongside it at
  frame 106, persists alone through 108-109, and Track 1 returns at 110.
- BoT-SORT: the same IDs and transition are visible.
- BoT-SORT + ReID: behavior is again identical.

Verdict: the duplicate/fragmentation failure is **not avoided by either alternative**.

## Walking crossing: frames 252-300 (10.511-12.513 s)

- ByteTrack: Track 2, which began on the dark-clothed worker, is attached to the red-vest
  worker in this window. Track 4 appears on the dark-clothed worker at frame 269. This is a
  probable identity switch plus fragmentation.
- BoT-SORT: Track 1 remains attached to the red-vest worker. Track 4 appears on the
  dark-clothed worker at frame 269. Identity transfer is avoided, although the dark worker
  still fragments from its earlier ID.
- BoT-SORT + ReID: the same qualitative identity behavior as non-ReID BoT-SORT.

Verdict: both BoT-SORT variants provide a **clear but partial improvement**. Appearance ReID
does not add observable benefit.

## Secondary observations

- The compliant clip remains stable under every candidate.
- BoT-SORT does not remove the short helmet duplicate.
- No new confirmed incident or incident fragmentation is introduced.
- Track-loss counters vary slightly, but counters alone are not treated as identity quality.

## Boundaries of judgment

There is no identity ground truth, so MOTA, IDF1, HOTA, and an ID-switch rate are not
reported. The review establishes visible identity attachment only within the declared
frames. Video provenance remains unresolved; public-demo eligibility is false.
