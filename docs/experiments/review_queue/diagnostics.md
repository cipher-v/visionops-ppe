# Deterministic diagnostic

The injected test timeline contains low-confidence detection, association ambiguity,
conflict, two repeated unknown frames, confirmed incident, recovered track, high drift and
duplicate timing. Eight raw events become seven after temporal deduplication and six after
the configured budget. The incident ranks first, the higher-priority repeated unknown is
retained, all human fields start blank, and high drift remains merely a review reason with
no inferred model-error verdict.

Tests additionally cover schema/update validation, each required reason weight, visual
hash deduplication and a manual export that includes only explicitly reviewed candidates.

