# Candidate selection

Reasons and base weights are: confirmed incident 40, conflict 35, association ambiguity
28, incident close 28, high-confidence unassigned PPE 26, recovered track 24, suspected
violation 24, repeated unknown 20, lost track 20, low confidence 18, near-gate
association 18, short track 16, high drift 8, warning drift 4. Low-confidence events add
up to 12 points based on proximity to their frozen class threshold. Scores are capped at
100 and are not probabilities of error.

Temporal deduplication groups source/type/track/equipment in two-second timestamp windows
and retains the highest priority, then earliest timestamp and ID. Visual deduplication uses
a 64-bit 9x8 grayscale difference hash and removes a lower-ranked same-source candidate at
Hamming distance <=4. This is visual redundancy, not semantic novelty.

The deterministic budget is 40 candidates/video, 12/video-minute, and 12/type. Ordering is
priority descending, timestamp ascending, then candidate ID. These caps permit natural
class/failure imbalance and never fabricate missing categories.

Incident signals nominate confirmation, active mid-incident samples through temporal
deduplication, and closing context when it remains visually distinct. The goal is compact
review of tracking, association, violation reality and timing—not every incident frame.

