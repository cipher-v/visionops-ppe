# Spatial safety-zone policy design

Checkpoint 14 adds an optional, additive policy layer after Person tracking and
frame-level PPE association. It does not alter detector evidence: it determines whether
helmet or vest evidence is policy-applicable at a worker's stabilized location.

## Configuration and geometry

`configs/zones/example_zones.yaml` uses versioned YAML and normalized `[x, y]`
coordinates in `[0, 1]`, so a polygon is independent of image resolution. Every enabled
zone has a unique ID, display name, polygon, and set of required equipment. Validation
rejects malformed input, fewer than three vertices, non-finite or out-of-range
coordinates, duplicate IDs, unsupported PPE, zero-area polygons, and self-intersection.
No geometry library was added.

Membership uses the normalized bottom-center of the tracked Person box,
`((x1 + x2) / 2, y2)`. This foot-point proxy better represents where a standing worker
occupies the ground than box overlap or centroid. PPE boxes never determine membership.
The version-1 contract names the anchor explicitly; `bottom_center` is currently the only
supported strategy. Points on polygon edges or vertices count as inside.

When polygons overlap, all containing zone IDs are retained and their PPE requirements
are combined by set union. Zone order therefore cannot discard a requirement.

## Timestamp stabilization

Raw membership is evaluated on every active Person observation. A continuous raw-inside
interval must reach `enter_confirm_seconds` before the zone becomes stable. Once stable,
a continuous raw-outside interval must reach `exit_confirm_seconds` before exit. These
decisions use source timestamps, never frame counts. The example values (0.35 seconds
each) are initial engineering defaults and were not tuned on the demo videos.

Brief tracking jitter resets the relevant pending transition. A direct transition
between two zones may change zone IDs, but if both require the same equipment, policy
applicability remains continuous.

## Policy and incidents

For each track/equipment pair, stabilized membership produces `required=true/false`.
Negative evidence remains visible in structured evidence outside required zones, but its
temporal state is `not_required` (`N/R`) and it cannot open an incident. Present evidence
outside a required zone is also `N/R`, not compliant. Unknown and conflict retain their
existing meanings whenever the equipment is required.

On stable entry, any earlier outside-zone violation duration is discarded. A fresh
violation episode begins at stable entry and must satisfy the unchanged temporal
confirmation duration. An active incident remains continuous across zone-ID changes
while at least one stable zone still requires that equipment. Once stable membership no
longer requires it, the incident closes with `zone_exited`.

Closing on zone exit preserves the existing incident cooldown. Immediate stable re-entry
starts a fresh violation timer, but a confirmation falling inside the prior cooldown is
suppressed by the existing cooldown rule; time outside the zone is never backdated.
Incident records add start, confirmation, and last zone IDs plus a policy flag. Incident
uniqueness remains run/source/track/equipment based, not zone based.

## Pipeline contract

Zones are enabled only with `--zones PATH` and require tracking, PPE association, and
temporal compliance. Without this flag, the Checkpoint 13B code path and all-required
temporal semantics are unchanged; the system does not invent a full-frame zone.

Enabled runs emit `zone_membership.jsonl` with frame/time/source/track, anchor, raw and
stable zone IDs, and effective required equipment. Annotated output draws polygon names,
track membership, and `N/R` for non-required equipment. Run metadata reports zone
configuration, diagnostics, and measured per-observation zone latency.

## Limitations

- Polygons are image-plane approximations, not calibrated world-space regions; perspective
  and camera movement can make them misleading.
- The foot point depends on detector/tracker box quality and may be hidden or truncated.
- Zones must be authored per fixed camera; there is no graphical editor in this checkpoint.
- The example polygon is illustrative and is not an authoritative site safety policy.
- Demo videos have no zone ground truth, so integration output cannot establish real-world
  membership or policy accuracy.
- Track fragmentation can reset membership and policy state because track IDs are anonymous
  run-local identities.
