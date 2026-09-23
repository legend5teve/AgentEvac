"""Unit tests for the M1 alert-schedule resolver (``agentevac.agents.alert_schedule``)."""

import json
from pathlib import Path

import pytest

from agentevac.agents.alert_schedule import (
    AlertSchedule,
    NO_ALERT,
    effective_mode,
    route_advisory,
    route_advisory_policy,
)


def _toy_config():
    """A two-area, two-event schedule mirroring the E0 shape at small scale."""
    return {
        "areas": {
            "westwood": {"edges": ["w0", "w1", "w2"]},
            "highland": {"edges": ["h0", "h1"]},
        },
        "schedule": [
            {
                "id": "EA-1", "issue_time_s": 6300, "areas": ["westwood"],
                "instruction": "evacuate_now", "hazard_text": "Leave Westwood",
                "comfort_centre": "Black Point", "routing_text": None,
                "channel": "wireless_emergency_alert",
            },
            {
                "id": "EA-2", "issue_time_s": 9660, "areas": ["highland"],
                "instruction": "evacuate_now", "hazard_text": "Leave Highland",
                "comfort_centre": None, "routing_text": None,
                "channel": "wireless_emergency_alert",
            },
        ],
        "door_to_door": {
            "start_time_s": 840,
            "initial_areas": ["westwood"],
            "sweep": [{"area": "westwood", "begin_s": 840, "cleared_by_s": 3600}],
        },
    }


def test_areas_for_edge_reverse_index():
    sched = AlertSchedule.from_config(_toy_config())
    assert sched.areas_for_edge("w1") == {"westwood"}
    assert sched.areas_for_edge("h0") == {"highland"}
    assert sched.areas_for_edge("unknown") == set()


def test_no_alert_before_first_order():
    sched = AlertSchedule.from_config(_toy_config())
    state = sched.active_for_edge(6299, "w0")
    assert state == NO_ALERT
    assert effective_mode(state) == "no_notice"


def test_westwood_ordered_after_ea1():
    sched = AlertSchedule.from_config(_toy_config())
    state = sched.active_for_edge(6300, "w0")
    assert state.received
    assert state.received_t_s == 6300
    assert state.instruction == "evacuate_now"
    assert state.order_text is not None
    assert state.order_text["comfort_centre"] == "Black Point"
    # routing_text is None, so this is the (1,1,0) view: alert_guided filter plus order.
    assert effective_mode(state) == "alert_guided"


def test_cumulative_extension_keeps_earlier_order():
    sched = AlertSchedule.from_config(_toy_config())
    # Highland becomes ordered only at EA-2.
    assert sched.active_for_edge(9659, "h0") == NO_ALERT
    assert sched.active_for_edge(9660, "h0").received
    # Westwood is still ordered from EA-1 after EA-2 fires.
    westwood = sched.active_for_edge(9660, "w0")
    assert westwood.received and westwood.received_t_s == 6300


def test_routing_visible_flips_to_advice():
    cfg = _toy_config()
    cfg["schedule"][0]["routing_text"] = "Take Hammonds Plains Rd west"
    sched = AlertSchedule.from_config(cfg)
    state = sched.active_for_edge(6300, "w0")
    assert state.routing_visible
    assert effective_mode(state) == "advice_guided"
    assert state.order_text["routing_text"] == "Take Hammonds Plains Rd west"


def test_empty_schedule_always_no_alert():
    sched = AlertSchedule.empty()
    assert sched.active_for_edge(99999, "w0") == NO_ALERT
    assert effective_mode(sched.active_for_edge(99999, "w0")) == "no_notice"


def test_time_offset_shifts_issue_times():
    sched = AlertSchedule.from_config(_toy_config(), time_offset_s=-3600)
    # EA-1 now issues an hour earlier, at 2700 s.
    assert sched.active_for_edge(2699, "w0") == NO_ALERT
    assert sched.active_for_edge(2700, "w0").received


def test_door_knock_time_spread_across_window():
    sched = AlertSchedule.from_config(_toy_config())
    # Three Westwood edges spread linearly across [840, 3600].
    assert sched.door_knock_time("w0") == 840
    assert sched.door_knock_time("w2") == 3600
    assert sched.door_knock_time("w1") == pytest.approx(840 + (3600 - 840) * 0.5)
    # Highland is not swept.
    assert sched.door_knock_time("h0") is None


def test_door_sweep_scale_stretches_knock_window():
    # Doubling the sweep duration widens [840, 3600] to [840, 6360]; the first edge
    # still knocks at the fixed start and the last at the new clear-by time.
    sched = AlertSchedule.from_config(_toy_config(), door_sweep_scale=2.0)
    assert sched.door_knock_time("w0") == 840
    assert sched.door_knock_time("w1") == pytest.approx(3600.0)
    assert sched.door_knock_time("w2") == pytest.approx(6360.0)
    assert sched.door_sweeps() == [("westwood", 840.0, 6360.0)]


def test_door_sweep_scale_default_is_identity():
    scaled = AlertSchedule.from_config(_toy_config(), door_sweep_scale=1.0)
    base = AlertSchedule.from_config(_toy_config())
    assert scaled.door_sweeps() == base.door_sweeps()
    assert scaled.door_knock_time("w2") == base.door_knock_time("w2")


def test_instruction_none_is_hazard_only():
    cfg = _toy_config()
    cfg["schedule"][0]["instruction"] = "none"
    sched = AlertSchedule.from_config(cfg)
    state = sched.active_for_edge(6300, "w0")
    assert state.received and state.instruction == "none"
    assert state.order_text is None  # an alert exists but carries no departure order
    assert effective_mode(state) == "alert_guided"


def test_scheduled_events_offset_applied():
    sched = AlertSchedule.from_config(_toy_config(), time_offset_s=-600)
    events = sched.scheduled_events()
    assert [e.id for e in events] == ["EA-1", "EA-2"]
    assert events[0].issue_time_s == 6300 - 600  # the E1 offset is baked in
    assert events[0].areas == ("westwood",)


def test_door_sweeps_returns_tuples():
    sched = AlertSchedule.from_config(_toy_config())
    assert sched.door_sweeps() == [("westwood", 840.0, 3600.0)]


def test_ordered_areas_channels_and_edges():
    ordered = AlertSchedule.from_config(_toy_config()).ordered_areas()
    assert set(ordered) == {"westwood", "highland"}
    # Westwood is door-swept at 840, before the 6300 broadcast, so it is tagged door.
    assert ordered["westwood"]["channel"] == "door"
    assert ordered["westwood"]["order_t_s"] == 840
    assert ordered["westwood"]["edges"] == ["w0", "w1", "w2"]
    # Highland has no door sweep, so it is tagged broadcast at its alert time.
    assert ordered["highland"]["channel"] == "broadcast"
    assert ordered["highland"]["order_t_s"] == 9660


def test_ordered_areas_broadcast_before_door_stays_broadcast():
    cfg = _toy_config()
    # Push the door sweep after the broadcast; the earlier broadcast sets the channel.
    cfg["door_to_door"]["sweep"][0]["begin_s"] = 7000
    cfg["door_to_door"]["sweep"][0]["cleared_by_s"] = 9000
    ordered = AlertSchedule.from_config(cfg).ordered_areas()
    assert ordered["westwood"]["channel"] == "broadcast"
    assert ordered["westwood"]["order_t_s"] == 6300


def test_empty_schedule_has_no_ordered_areas():
    assert AlertSchedule.empty().ordered_areas() == {}


def test_parses_real_e0_schedule():
    path = (
        Path(__file__).resolve().parents[1]
        / "configs" / "halifax_3town" / "alerts.json"
    )
    cfg = json.loads(path.read_text())
    sched = AlertSchedule.from_config(cfg)
    westwood_edge = cfg["areas"]["westwood_hills"]["edges"][0]
    outer_edge = cfg["areas"]["outer_extension"]["edges"][0]
    # Westwood ordered at EA-1 (6300 s), not before.
    assert sched.active_for_edge(6299, westwood_edge) == NO_ALERT
    assert sched.active_for_edge(6300, westwood_edge).instruction == "evacuate_now"
    # Outer extension ordered only at EA-3 (15180 s).
    assert sched.active_for_edge(15179, outer_edge) == NO_ALERT
    assert sched.active_for_edge(15180, outer_edge).received


# --- E2 routing arm -------------------------------------------------------------


def test_route_advisory_is_none_without_routing_text():
    """E0 and E1 leave routing_text null, so no advisory block reaches any prompt."""
    sched = AlertSchedule.from_config(_toy_config())
    assert route_advisory(sched.active_for_edge(6300, "w0")) is None
    assert route_advisory(NO_ALERT) is None


def test_route_advisory_carries_text_and_provenance():
    cfg = _toy_config()
    cfg["schedule"][0]["routing_text"] = "Turn right at Hammonds Plains Rd"
    sched = AlertSchedule.from_config(cfg)
    adv = route_advisory(sched.active_for_edge(6300, "w0"))
    assert adv["guidance"] == "Turn right at Hammonds Plains Rd"
    assert adv["alert_id"] == "EA-1"
    assert adv["source"] == "official_alert"
    assert adv["channel"] == "wireless_emergency_alert"
    assert adv["received_t_s"] == 6300
    assert adv["comfort_centre"] == "Black Point"


def test_route_advisory_none_before_the_order_issues():
    cfg = _toy_config()
    cfg["schedule"][0]["routing_text"] = "Turn right at Hammonds Plains Rd"
    sched = AlertSchedule.from_config(cfg)
    assert route_advisory(sched.active_for_edge(6299, "w0")) is None


def test_route_advisory_none_when_instruction_is_none():
    """Hazard-only (E3) carries no order block, so no route advisory either."""
    cfg = _toy_config()
    cfg["schedule"][0]["instruction"] = "none"
    cfg["schedule"][0]["routing_text"] = "Turn right at Hammonds Plains Rd"
    sched = AlertSchedule.from_config(cfg)
    state = sched.active_for_edge(6300, "w0")
    assert state.order_text is None
    assert route_advisory(state) is None


def test_route_advisory_policy_empty_without_advisory():
    assert route_advisory_policy(None, "neutral", "option") == ""
    assert route_advisory_policy({}, "directive", "route") == ""


def test_route_advisory_policy_tone_and_unit():
    adv = {"guidance": "x"}
    neutral = route_advisory_policy(adv, "neutral", "option")
    directive = route_advisory_policy(adv, "directive", "route")
    assert "official_route_advisory" in neutral and "official_route_advisory" in directive
    assert "Weigh it alongside the visible option facts" in neutral
    assert "Follow it unless a visible route fact makes it unsafe" in directive
    assert "Follow" not in neutral


# --- per-household routing branches (the Westwood Hills two-exit split) ------------


def _branched_config():
    """Toy schedule whose first order splits its area between two egress roads."""
    cfg = _toy_config()
    cfg["schedule"][0]["routing_text"] = "Area-wide fallback guidance"
    cfg["schedule"][0]["routing_branches"] = [
        {"id": "north_rd", "label": "North Road", "edges": ["w0"],
         "comfort_centre": "North Centre", "text": "Leave by North Road, turn right."},
        {"id": "south_rd", "label": "South Road", "edges": ["w1"],
         "text": "Leave by South Road, turn left."},
    ]
    return cfg


def test_branch_text_wins_over_area_guidance():
    sched = AlertSchedule.from_config(_branched_config())
    adv = route_advisory(sched.active_for_edge(6300, "w0"))
    assert adv["guidance"] == "Leave by North Road, turn right."
    assert adv["applies_to"] == "North Road"


def test_each_side_hears_only_its_own_instruction():
    sched = AlertSchedule.from_config(_branched_config())
    north = route_advisory(sched.active_for_edge(6300, "w0"))
    south = route_advisory(sched.active_for_edge(6300, "w1"))
    assert north["guidance"] != south["guidance"]
    assert "North Road" in north["guidance"] and "North Road" not in south["guidance"]


def test_uncovered_edge_falls_back_to_area_guidance():
    """w2 is in the area but in neither branch, so it hears the area-wide text."""
    sched = AlertSchedule.from_config(_branched_config())
    adv = route_advisory(sched.active_for_edge(6300, "w2"))
    assert adv["guidance"] == "Area-wide fallback guidance"
    assert "applies_to" not in adv


def test_branch_comfort_centre_overrides_the_order_centre():
    sched = AlertSchedule.from_config(_branched_config())
    assert route_advisory(sched.active_for_edge(6300, "w0"))["comfort_centre"] == "North Centre"


def test_branch_without_a_centre_keeps_the_order_centre():
    sched = AlertSchedule.from_config(_branched_config())
    assert route_advisory(sched.active_for_edge(6300, "w1"))["comfort_centre"] == "Black Point"


def test_branches_alone_make_routing_visible():
    """An order whose guidance is entirely per-household still resolves to advice_guided."""
    cfg = _branched_config()
    cfg["schedule"][0]["routing_text"] = None
    sched = AlertSchedule.from_config(cfg)
    covered = sched.active_for_edge(6300, "w0")
    assert covered.routing_visible
    assert effective_mode(covered) == "advice_guided"
    # w2 is in no branch and the area text is gone, so nothing reaches it.
    assert route_advisory(sched.active_for_edge(6300, "w2")) is None


def test_branch_with_empty_text_is_dropped():
    cfg = _branched_config()
    cfg["schedule"][0]["routing_branches"][0]["text"] = ""
    sched = AlertSchedule.from_config(cfg)
    adv = route_advisory(sched.active_for_edge(6300, "w0"))
    assert adv["guidance"] == "Area-wide fallback guidance"
