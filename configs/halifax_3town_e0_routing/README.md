# halifax_3town_e0_routing, E2 routing counterfactual

Same network, spawns, fires, destinations, routes and alert timing as `halifax_3town_e0`.
The only delta is `routing_text`, populated on all three broadcasts. That makes
`routing_visible` true for an ordered household, so its per-agent regime resolves to
`advice_guided` and the destination menu carries advisory labels, per-route-head risk and
the authored guidance text. E0 is the same schedule with `routing_text` null throughout.

This is the (1,1,1) corner of the information triad against E0's (1,1,0). It answers the
question the 2023 record cannot, which is what changes if residents are told the routes.

Run it with `SCENARIO_TONE=neutral` so the arm is tone-matched to E0 and the added guidance
is not confounded with directive exhortation. `scripts/run_experiments.py --arms e2` sets
this.

## Where the guidance comes from

| Area | Basis |
|---|---|
| `westwood_hills` | Documented. HRM 231017rci05, Westwood Hills Evacuation Plan appendix, Operations and Police sections |
| `highland_park` | Inferred. No route guidance for this community exists in any source |
| `outer_extension` | Inferred. No route guidance for these communities exists in any source |

The asymmetry is a property of the record. HRM holds an operational evacuation plan for
Westwood Hills alone. It divides the subdivision into six evacuation districts with muster
points, sets RCMP access control at Westwood Boulevard and Winslow Drive, and fixes the turn
direction at Hammonds Plains Road, right from Westwood Boulevard and left from Winslow
Drive, with the stated purpose of keeping the two exits off the same stretch of road. It
also names two westerly and two easterly pre-determined reception centres, with St
Margaret's Bay Arena as a possible fifth.

For the other Hammonds Plains Road communities, meaning Highland Park, Haliburton, Glen
Arbour, White Hills and Lucasville, the Community Wildfire Protection Plan appendix
recommends that a community evacuation plan with evacuation routes be put in place and
specifies none. It also records that these subdivisions have limited entrance and egress.
The report's own recommendations ask HRM to revisit the evacuation-route process and to
publish routes. So the inferred guidance for those two areas is authored from the plan's
stated principle of directing outbound traffic away from the hazard, combined with the
comfort centres HRM p.12 records as opened, Canada Games Centre as the main evacuation
centre and Beaver Bank Kinsac for the Lucasville area.

None of this guidance was broadcast on 28 May. Every alert that day carried an evacuate
directive, an area and, in EA-1 only, a named comfort centre. That is what makes E2 a
forward-looking practice question.

## The Westwood two-exit split

EA-1 carries `routing_branches`, so the two sides of Westwood Hills hear different
instructions. A household on the Westwood Boulevard side is told to turn right at Hammonds
Plains Road and register at Black Point. A household on the Winslow Drive side is told to
turn left and register at the Canada Games Centre. Each text names that household's own
road, so nothing has to be resolved from a street name the agent cannot see.

| Branch | Districts | Households | Edges | Reception centre |
|---|---|---|---|---|
| Westwood Boulevard | 407-03 | 17 | 6 | Black Point and Area Community Centre |
| Winslow Drive | 407-04, 407-06 | 43 | 10 | Canada Games Centre |

The subdivision's two-exit structure is confirmed against the network. Westwood Boulevard
meets Hammonds Plains Road at exactly one junction and Winslow Drive at exactly one, 316 m
apart. The plan states the two exits terminate within 200 metres of each other, so the
structure matches and the separation is somewhat wider than the plan's figure.

The binding itself is map-confirmed and not derived. Every one of the 16 sampled household
edges was placed against the evacuation-district map in the report appendix, and the plan
then sends 407-01, 407-03 and 407-05 down Westwood Boulevard and 407-02, 407-04 and 407-06
via Winslow Drive. 407-03 holds all four Westwood Boulevard edges and both Tattingstone
Court edges. 407-04 holds Wyndham Drive and Windbreak Run, and Hemlock Drive straddles
407-04 and 407-06. No sampled edge is unplaced, and each branch carries its per-edge
district in `edge_districts`. Hemlock Drive's district is ambiguous within the Winslow
side, which leaves the branch certain because both of its districts use the same exit.

`scripts/derive_westwood_districts.py` recomputes the binding from the network, as a
shortest path from each household edge out to Hammonds Plains Road, and reports where it
disagrees with the config. It currently disagrees on 3 of 16 edges, all on Wyndham Drive
and 14 households, where the network favours Westwood Boulevard by 83 m while the map
places them in 407-04 on the Winslow side. The map is authoritative. That disagreement is
how the error was caught, which is why the cross-check is kept rather than deleted.

The two sides go to different centres because the plan names two westerly pre-determined
centres for the Westwood Boulevard side and two easterly for the Winslow Drive side. Those
resolve to the two that were actually opened on 28 May.

EA-2 and EA-3 carry no branches, since no source specifies routing for those communities.

## Known limits

Destination control mode. `CONTROL_MODE` is `destination`, so guidance can only act through
which comfort centre an agent picks. Naming an egress corridor steers the destination by
construction, and the E2 against E0 delta reads as the effect of official guidance on
destination choice and the exposure that follows, not as road-level route compliance.

Advisory labels are live, not planned. The `advisory` field on each destination is computed
per round by `build_driver_briefing` from current fire positions and blocked edges, labelled
as coming from the Emergency Operations Center. It is a live assessment and the authored
`routing_text` is the static plan. Both reach the agent in this arm.
