# halifax_3town_e0_buffer, buffer-alerting counterfactual

Same network, spawns, fires, destinations, routes, corridors, alert areas, instructions,
hazard text and RCMP door-to-door channel as `halifax_3town_e0`. The only difference is
`issue_time_s` on the three broadcasts.

Each community is alerted when the fire margin around it falls to a buffer sized so the
community can finish evacuating before the fire arrives.

## The policy

| Community | Evacuation time | Buffer | Alert fires | Historical | Moved earlier by |
|---|---|---|---|---|---|
| Westwood Hills | 1903 s | none fits | 0 s | 6300 s | 105 min |
| Highland Park | 2780 s | 800 m | 7440 s | 9660 s | 37 min |
| Outer extension | 3580 s | 450 m | 13890 s | 15180 s | 22 min |

Evacuation time is the span from a community being ordered to its last household arriving,
taken as the worst of the three E0 seeds, because a buffer that covers only the median
leaves half the runs short. It is the `area_evacuation_time` metric, recovered for the E0
runs from their timeline exports since those runs predate the metric.

Westwood Hills has no feasible buffer. The fire is 184 m away at ignition and reaches the
subdivision at 480 s, against an evacuation time of 1903 s. Its trigger is set to 0 s, which
is the earliest the policy can act, and the config records it as infeasible.

## Why it compiles to timed events

The fire layer is scripted and deterministic, so the margin from a community to the nearest
burning edge is exact arithmetic at any instant. The trigger times are therefore computed
offline and written as ordinary `issue_time_s` values. The alert resolver stays pure and
SUMO-free, replay stays bit-identical, and the arm needs no engine change.

Regenerate with `python scripts/derive_buffer_schedule.py --write configs/halifax_3town_e0_buffer`.

## Known limits

Evacuation time is measured at 182 households on a network with no background traffic, so
it is a lower bound. At realistic demand the evacuation time rises and every buffer here is
undersized. This is the largest limitation of the arm and it should be reported with any
result from it.

The policy is given the whole ignition schedule in advance. A real operations centre would
not have it, and the margin does not close smoothly, so it cannot be extrapolated. Highland
Park sat at 799 m from t=8400 and fell to 52 m by t=10800 when a new source ignited. A live
policy watching the observed fire would have been overtaken by that jump, so these triggers
are an upper bound on what a real buffer policy achieves.

The buffer is a distance around the households, not a travel-time isochrone. Two
communities the same distance from the fire but with different road capacity get the same
buffer, which the evacuation-time term corrects for only in aggregate.
