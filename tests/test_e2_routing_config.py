"""Integrity tests for the E2 routing counterfactual config.

E2's whole claim is that it differs from the E0 reconstruction in route guidance and in
nothing else.  A stray edit to spawns, fires or alert timing would silently turn the arm
into a different experiment that still produces a full metrics file, so the difference is
asserted here rather than assumed.
"""

import copy
import json
from pathlib import Path

import pytest

from agentevac.agents.alert_schedule import (
    AlertSchedule,
    effective_mode,
    route_advisory,
)

REPO = Path(__file__).resolve().parents[1]
E0 = REPO / "configs" / "halifax_3town_e0"
E2 = REPO / "configs" / "halifax_3town_e0_routing"

# Everything except alerts.json has to match E0 byte for byte.
SHARED_FILES = ["map.json", "spawns.json", "fires.json", "destinations.json", "routes.json",
                "corridors.json"]


def _load(path: Path):
    with open(path) as f:
        return json.load(f)


def test_config_dir_is_complete():
    assert E2.is_dir()
    for name in SHARED_FILES + ["alerts.json", "README.md"]:
        assert (E2 / name).is_file(), name


@pytest.mark.parametrize("name", SHARED_FILES)
def test_shared_files_match_e0_byte_for_byte(name):
    assert (E2 / name).read_bytes() == (E0 / name).read_bytes()


def test_alerts_differ_only_in_routing_fields():
    """Areas, timing, instructions, hazard text and the door sweep are E0's."""
    def strip(cfg):
        cfg = copy.deepcopy(cfg)
        cfg.pop("_meta", None)
        for event in cfg["schedule"]:
            event.pop("routing_text", None)
            event.pop("routing_branches", None)
            event.pop("source", None)   # provenance gains ROUTING and BRANCHES clauses
        return cfg

    assert strip(_load(E2 / "alerts.json")) == strip(_load(E0 / "alerts.json"))


def test_every_e2_event_carries_routing_text():
    """An empty routing_text degenerates the arm into E0 without failing the run."""
    schedule = _load(E2 / "alerts.json")["schedule"]
    assert schedule
    for event in schedule:
        assert event["routing_text"], event["id"]


def test_e0_stays_free_of_routing_text():
    for event in _load(E0 / "alerts.json")["schedule"]:
        assert not event["routing_text"], event["id"]


def test_ordered_household_resolves_to_advice_guided_with_guidance():
    cfg = _load(E2 / "alerts.json")
    first = cfg["schedule"][0]
    edge = cfg["areas"][first["areas"][0]]["edges"][0]
    state = AlertSchedule.from_config(cfg).active_for_edge(first["issue_time_s"], edge)

    assert state.routing_visible
    assert effective_mode(state) == "advice_guided"
    # EA-1 splits Westwood by egress road, so the household hears its own branch and not
    # the area-wide fallback.
    branch = next(b for b in first["routing_branches"] if edge in b["edges"])
    assert route_advisory(state)["guidance"] == branch["text"]


def test_an_unbranched_order_still_delivers_its_area_guidance():
    """EA-2 and EA-3 carry no branches, since no plan exists for those communities."""
    cfg = _load(E2 / "alerts.json")
    second = cfg["schedule"][1]
    edge = cfg["areas"][second["areas"][0]]["edges"][0]
    state = AlertSchedule.from_config(cfg).active_for_edge(second["issue_time_s"], edge)

    assert "routing_branches" not in second
    assert route_advisory(state)["guidance"] == second["routing_text"]


def test_same_household_sees_no_guidance_under_e0():
    cfg = _load(E0 / "alerts.json")
    first = cfg["schedule"][0]
    edge = cfg["areas"][first["areas"][0]]["edges"][0]
    state = AlertSchedule.from_config(cfg).active_for_edge(first["issue_time_s"], edge)

    assert state.received
    assert not state.routing_visible
    assert effective_mode(state) == "alert_guided"
    assert route_advisory(state) is None


# --- the Westwood Hills two-exit split ------------------------------------------


def _ea1():
    return next(e for e in _load(E2 / "alerts.json")["schedule"] if e["id"] == "EA-1")


def test_ea1_splits_westwood_between_its_two_egress_roads():
    labels = {b["label"] for b in _ea1()["routing_branches"]}
    assert labels == {"Westwood Boulevard", "Winslow Drive"}


def test_branches_partition_every_westwood_household_edge():
    """Each household edge is bound to exactly one egress road, none left over."""
    area_edges = set(_load(E2 / "alerts.json")["areas"]["westwood_hills"]["edges"])
    branch_edges = [set(b["edges"]) for b in _ea1()["routing_branches"]]

    assert set().union(*branch_edges) == area_edges
    assert not branch_edges[0] & branch_edges[1]


def test_branch_household_counts_match_the_spawn_config():
    spawns = {g["edge"]: g.get("count", 1) for g in _load(E2 / "spawns.json")["groups"]}
    for branch in _ea1()["routing_branches"]:
        assert branch["households"] == sum(spawns[e] for e in branch["edges"]), branch["label"]


def test_the_two_sides_are_sent_to_different_reception_centres():
    centres = {b["label"]: b["comfort_centre"] for b in _ea1()["routing_branches"]}
    assert centres["Westwood Boulevard"] != centres["Winslow Drive"]


def test_every_westwood_household_resolves_to_a_branch_it_can_act_on():
    cfg = _load(E2 / "alerts.json")
    schedule = AlertSchedule.from_config(cfg)
    ea1 = _ea1()
    for edge in cfg["areas"]["westwood_hills"]["edges"]:
        advisory = route_advisory(schedule.active_for_edge(ea1["issue_time_s"], edge))
        assert advisory["applies_to"] in {"Westwood Boulevard", "Winslow Drive"}, edge
        # The text names the household's own road, so it needs no street-name lookup.
        assert advisory["applies_to"] in advisory["guidance"], edge


def test_e0_westwood_households_get_no_branch():
    cfg = _load(E0 / "alerts.json")
    ea1 = next(e for e in cfg["schedule"] if e["id"] == "EA-1")
    assert "routing_branches" not in ea1


def test_branches_match_the_evacuation_district_map():
    """Locked to the district map, which overrules the network shortest path.

    A shortest path out favours Westwood Boulevard for Wyndham Drive by 83 m, but the map
    places Wyndham in 407-04, which the plan evacuates via Winslow Drive. The map wins, and
    this test exists so that cannot be silently reverted to the derived answer.
    """
    branches = {b["label"]: b for b in _ea1()["routing_branches"]}
    names = json.load(open(REPO / "sumo" / "westwood_street_names.json"))["names"]

    def streets(label):
        return {names[e.lstrip("-").split("#")[0]] for e in branches[label]["edges"]}

    assert streets("Winslow Drive") == {"Wyndham Drive", "Windbreak Run", "Hemlock Drive"}
    assert streets("Westwood Boulevard") == {"Tattingstone Court", "Westwood Boulevard"}
    assert branches["Westwood Boulevard"]["households"] == 17
    assert branches["Winslow Drive"]["households"] == 43
    assert branches["Westwood Boulevard"]["districts"] == ["407-03"]
    assert branches["Winslow Drive"]["districts"] == ["407-04", "407-06"]


def test_every_household_edge_is_placed_in_a_district():
    """No sampled edge is left unplaced against the appendix map."""
    branches = _ea1()["routing_branches"]
    placed = {e: d for b in branches for e, d in b["edge_districts"].items()}
    area_edges = set(_load(E2 / "alerts.json")["areas"]["westwood_hills"]["edges"])

    assert set(placed) == area_edges
    assert all(d.startswith("407-") for d in placed.values())


def test_each_edge_district_implies_its_branch():
    """The plan sends 407-01/03/05 down Westwood Blvd and 407-02/04/06 via Winslow Dr."""
    westwood_side, winslow_side = {"407-01", "407-03", "407-05"}, {"407-02", "407-04", "407-06"}
    for branch in _ea1()["routing_branches"]:
        for edge, district in branch["edge_districts"].items():
            possible = set(district.split(" or "))
            side = westwood_side if branch["label"] == "Westwood Boulevard" else winslow_side
            # Ambiguity within a branch is allowed, ambiguity across branches is not.
            assert possible <= side, (edge, district, branch["label"])
