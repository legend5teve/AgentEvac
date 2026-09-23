"""Integrity tests for the buffer-alerting counterfactual config.

The arm's claim is that it differs from the E0 reconstruction in when each community is
alerted and in nothing else. A stray edit to the areas, the instructions or the
door-to-door channel would turn it into a different experiment that still produces a full
metrics file, so the difference is asserted here.
"""

import copy
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
E0 = REPO / "configs" / "halifax_3town_e0"
BUF = REPO / "configs" / "halifax_3town_e0_buffer"

SHARED_FILES = ["map.json", "spawns.json", "fires.json", "destinations.json", "routes.json",
                "corridors.json"]


def _load(path: Path):
    with open(path) as f:
        return json.load(f)


def test_config_dir_is_complete():
    assert BUF.is_dir()
    for name in SHARED_FILES + ["alerts.json", "README.md"]:
        assert (BUF / name).is_file(), name


@pytest.mark.parametrize("name", SHARED_FILES)
def test_shared_files_match_e0_byte_for_byte(name):
    assert (BUF / name).read_bytes() == (E0 / name).read_bytes()


def test_only_the_issue_times_differ():
    """Areas, instructions, hazard text, channel and the door sweep stay E0's."""
    def strip(cfg):
        cfg = copy.deepcopy(cfg)
        cfg.pop("_meta", None)
        for event in cfg["schedule"]:
            for key in ("issue_time_s", "wall_clock", "source", "buffer_m", "buffer_feasible"):
                event.pop(key, None)
        return cfg

    assert strip(_load(BUF / "alerts.json")) == strip(_load(E0 / "alerts.json"))


def test_every_community_is_alerted_no_later_than_history():
    """A buffer policy that warned later than the real broadcast would be pointless."""
    buffered = {e["id"]: e["issue_time_s"] for e in _load(BUF / "alerts.json")["schedule"]}
    historical = {e["id"]: e["issue_time_s"] for e in _load(E0 / "alerts.json")["schedule"]}

    assert set(buffered) == set(historical)
    for event_id, t in buffered.items():
        assert t <= historical[event_id], event_id


def test_the_policy_records_its_own_sizing():
    policy = _load(BUF / "alerts.json")["_meta"]["buffer_policy"]
    for area in _load(E0 / "alerts.json")["areas"]:
        assert area in policy["t_evac_s"]
        assert area in policy["trigger_t_s"]


def test_westwood_is_recorded_as_infeasible():
    """The fire reaches Westwood long before it can clear, so no buffer fits.

    Recording it rather than silently emitting a trigger is the point, since an alert at
    ignition still leaves that community short.
    """
    policy = _load(BUF / "alerts.json")["_meta"]["buffer_policy"]
    assert policy["infeasible"] == ["westwood_hills"]
    assert policy["buffer_m"]["westwood_hills"] is None
    assert policy["trigger_t_s"]["westwood_hills"] == 0


def test_feasible_communities_carry_a_buffer_distance():
    policy = _load(BUF / "alerts.json")["_meta"]["buffer_policy"]
    for area, buffer_m in policy["buffer_m"].items():
        if area in policy["infeasible"]:
            continue
        assert buffer_m and buffer_m > 0, area
