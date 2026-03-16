"""Unit tests for agentevac.agents.agent_state."""

import json
import math

import pytest

from agentevac.agents.agent_state import (
    AGENT_STATES,
    AgentRuntimeState,
    append_decision_history,
    append_observation_history,
    append_signal_history,
    append_social_history,
    ensure_agent_state,
    sample_profile_params,
    snapshot_agent_state,
)


@pytest.fixture(autouse=True)
def clear_agent_states():
    """Wipe the global registry before and after each test to avoid cross-test pollution."""
    AGENT_STATES.clear()
    yield
    AGENT_STATES.clear()


class TestSampleProfileParams:
    _MEANS = {"theta_trust": 0.5, "theta_r": 0.45, "lambda_e": 1.0}
    _BOUNDS = {"theta_trust": (0.0, 1.0), "theta_r": (0.1, 0.9), "lambda_e": (0.0, 5.0)}

    def test_zero_spread_returns_exact_means(self):
        spreads = {"theta_trust": 0.0, "theta_r": 0.0, "lambda_e": 0.0}
        result = sample_profile_params("v1", self._MEANS, spreads, self._BOUNDS)
        assert result["theta_trust"] == 0.5
        assert result["theta_r"] == 0.45
        assert result["lambda_e"] == 1.0

    def test_nonzero_spread_produces_inter_agent_variation(self):
        spreads = {"theta_trust": 0.15, "theta_r": 0.1, "lambda_e": 0.3}
        r1 = sample_profile_params("v1", self._MEANS, spreads, self._BOUNDS)
        r2 = sample_profile_params("v2", self._MEANS, spreads, self._BOUNDS)
        # Different agent IDs should (almost certainly) get different profiles.
        assert r1 != r2

    def test_same_agent_id_gives_same_profile(self):
        spreads = {"theta_trust": 0.15, "theta_r": 0.1, "lambda_e": 0.3}
        r1 = sample_profile_params("v1", self._MEANS, spreads, self._BOUNDS)
        r2 = sample_profile_params("v1", self._MEANS, spreads, self._BOUNDS)
        assert r1 == r2

    def test_values_stay_within_bounds(self):
        spreads = {"theta_trust": 0.5, "theta_r": 0.5, "lambda_e": 2.0}
        for agent_id in [f"v{i}" for i in range(50)]:
            result = sample_profile_params(agent_id, self._MEANS, spreads, self._BOUNDS)
            assert 0.0 <= result["theta_trust"] <= 1.0
            assert 0.1 <= result["theta_r"] <= 0.9
            assert 0.0 <= result["lambda_e"] <= 5.0

    def test_missing_spread_key_uses_mean(self):
        spreads = {"theta_trust": 0.15}  # theta_r and lambda_e missing
        result = sample_profile_params("v1", self._MEANS, spreads, self._BOUNDS)
        assert result["theta_r"] == 0.45
        assert result["lambda_e"] == 1.0


class TestEnsureAgentState:
    def test_creates_new_state(self):
        state = ensure_agent_state("v1", 0.0)
        assert isinstance(state, AgentRuntimeState)
        assert state.agent_id == "v1"

    def test_returns_same_instance_on_second_call(self):
        s1 = ensure_agent_state("v2", 0.0)
        s2 = ensure_agent_state("v2", 5.0)
        assert s1 is s2

    def test_updates_last_sim_t_s_on_second_call(self):
        ensure_agent_state("v3", 0.0)
        state = ensure_agent_state("v3", 10.0)
        assert state.last_sim_t_s == pytest.approx(10.0)

    def test_initial_belief_is_uniform(self):
        state = ensure_agent_state("v4", 0.0)
        b = state.belief
        assert abs(b["p_safe"] - 1 / 3) < 1e-9
        assert abs(b["p_risky"] - 1 / 3) < 1e-9
        assert abs(b["p_danger"] - 1 / 3) < 1e-9

    def test_initial_entropy_is_max(self):
        state = ensure_agent_state("v5", 0.0)
        assert state.belief["entropy"] == pytest.approx(math.log(3), rel=1e-6)
        assert state.belief["entropy_norm"] == pytest.approx(1.0)
        assert state.belief["uncertainty_bucket"] == "High"

    def test_custom_kwargs_set_profile(self):
        state = ensure_agent_state("v6", 0.0, default_theta_r=0.8, default_lambda_e=2.5)
        assert state.profile["theta_r"] == pytest.approx(0.8)
        assert state.profile["lambda_e"] == pytest.approx(2.5)

    def test_all_profile_keys_present(self):
        state = ensure_agent_state("v7", 0.0)
        for key in (
            "theta_trust",
            "theta_r",
            "theta_u",
            "gamma",
            "lambda_e",
            "lambda_t",
            "neighbor_window_s",
            "social_recent_weight",
            "social_total_weight",
            "social_trigger",
            "social_min_danger",
        ):
            assert key in state.profile

    def test_state_stored_in_global_registry(self):
        state = ensure_agent_state("v8", 0.0)
        assert AGENT_STATES["v8"] is state


class TestAppendBoundedHistories:
    def _make_state(self, agent_id="test"):
        return ensure_agent_state(agent_id, 0.0)

    def test_signal_history_appends(self):
        state = self._make_state("sh1")
        append_signal_history(state, {"x": 1})
        assert len(state.signal_history) == 1
        assert state.signal_history[0]["x"] == 1

    def test_signal_history_respects_max_items(self):
        state = self._make_state("sh2")
        for i in range(20):
            append_signal_history(state, {"i": i}, max_items=5)
        assert len(state.signal_history) == 5
        # Oldest entries evicted; most recent retained.
        assert state.signal_history[-1]["i"] == 19

    def test_social_history_appends(self):
        state = self._make_state("soc1")
        append_social_history(state, {"msg": "fire"})
        assert len(state.social_history) == 1

    def test_social_history_respects_max_items(self):
        state = self._make_state("soc2")
        for i in range(10):
            append_social_history(state, {"i": i}, max_items=3)
        assert len(state.social_history) == 3

    def test_decision_history_appends(self):
        state = self._make_state("dh1")
        append_decision_history(state, {"choice": 0})
        assert len(state.decision_history) == 1

    def test_decision_history_default_max_is_32(self):
        state = self._make_state("dh2")
        for i in range(40):
            append_decision_history(state, {"i": i})
        assert len(state.decision_history) == 32

    def test_observation_history_appends(self):
        state = self._make_state("obs1")
        append_observation_history(state, {"summary": "neighbor departed"})
        assert len(state.observation_history) == 1
        assert state.observation_history[0]["summary"] == "neighbor departed"

    def test_observation_history_respects_max_items(self):
        state = self._make_state("obs2")
        for i in range(20):
            append_observation_history(state, {"i": i}, max_items=4)
        assert len(state.observation_history) == 4
        assert state.observation_history[-1]["i"] == 19

    def test_appended_item_is_a_copy(self):
        state = self._make_state("copy1")
        original = {"x": 1}
        append_signal_history(state, original)
        original["x"] = 999
        assert state.signal_history[0]["x"] == 1


class TestSnapshotAgentState:
    def test_snapshot_is_json_serializable(self):
        state = ensure_agent_state("snap1", 5.0)
        snap = snapshot_agent_state(state)
        json.dumps(snap)  # must not raise

    def test_snapshot_contains_required_keys(self):
        state = ensure_agent_state("snap2", 5.0)
        snap = snapshot_agent_state(state)
        for key in ("agent_id", "created_sim_t_s", "last_sim_t_s", "profile",
                    "belief", "psychology", "signal_history", "social_history",
                    "decision_history", "observation_history", "has_departed"):
            assert key in snap

    def test_snapshot_agent_id_matches(self):
        state = ensure_agent_state("snap3", 5.0)
        snap = snapshot_agent_state(state)
        assert snap["agent_id"] == "snap3"

    def test_snapshot_does_not_share_mutable_references(self):
        state = ensure_agent_state("snap4", 5.0)
        append_signal_history(state, {"x": 1})
        snap = snapshot_agent_state(state)
        snap["signal_history"].clear()
        assert len(state.signal_history) == 1
