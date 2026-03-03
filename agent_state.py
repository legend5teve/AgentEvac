import math
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class AgentRuntimeState:
    agent_id: str
    created_sim_t_s: float
    last_sim_t_s: float
    profile: Dict[str, float] = field(default_factory=dict)
    belief: Dict[str, Any] = field(default_factory=dict)
    psychology: Dict[str, Any] = field(default_factory=dict)
    signal_history: List[Dict[str, Any]] = field(default_factory=list)
    social_history: List[Dict[str, Any]] = field(default_factory=list)
    decision_history: List[Dict[str, Any]] = field(default_factory=list)
    has_departed: bool = True


AGENT_STATES: Dict[str, AgentRuntimeState] = {}


def ensure_agent_state(
    agent_id: str,
    sim_t_s: float,
    *,
    default_theta_trust: float = 0.5,
    default_theta_r: float = 0.45,
    default_theta_u: float = 0.30,
    default_gamma: float = 0.995,
    default_lambda_e: float = 1.0,
    default_lambda_t: float = 0.1,
) -> AgentRuntimeState:
    state = AGENT_STATES.get(agent_id)
    if state is None:
        uniform_entropy = math.log(3.0)
        state = AgentRuntimeState(
            agent_id=agent_id,
            created_sim_t_s=float(sim_t_s),
            last_sim_t_s=float(sim_t_s),
            profile={
                "theta_trust": float(default_theta_trust),
                "theta_r": float(default_theta_r),
                "theta_u": float(default_theta_u),
                "gamma": float(default_gamma),
                "lambda_e": float(default_lambda_e),
                "lambda_t": float(default_lambda_t),
            },
            belief={
                "p_safe": 1.0 / 3.0,
                "p_risky": 1.0 / 3.0,
                "p_danger": 1.0 / 3.0,
                "entropy": uniform_entropy,
                "entropy_norm": 1.0,
                "uncertainty_bucket": "High",
            },
            psychology={
                "perceived_risk": 0.5,
                "confidence": 0.0,
            },
        )
        AGENT_STATES[agent_id] = state
    state.profile.setdefault("theta_trust", float(default_theta_trust))
    state.profile.setdefault("theta_r", float(default_theta_r))
    state.profile.setdefault("theta_u", float(default_theta_u))
    state.profile.setdefault("gamma", float(default_gamma))
    state.profile.setdefault("lambda_e", float(default_lambda_e))
    state.profile.setdefault("lambda_t", float(default_lambda_t))
    state.last_sim_t_s = float(sim_t_s)
    return state


def _append_bounded(items: List[Dict[str, Any]], value: Dict[str, Any], max_items: int) -> None:
    items.append(dict(value))
    if len(items) > max(1, int(max_items)):
        del items[:-max(1, int(max_items))]


def append_signal_history(
    state: AgentRuntimeState,
    signal: Dict[str, Any],
    *,
    max_items: int = 16,
) -> None:
    _append_bounded(state.signal_history, signal, max_items)


def append_social_history(
    state: AgentRuntimeState,
    signal: Dict[str, Any],
    *,
    max_items: int = 16,
) -> None:
    _append_bounded(state.social_history, signal, max_items)


def append_decision_history(
    state: AgentRuntimeState,
    decision: Dict[str, Any],
    *,
    max_items: int = 32,
) -> None:
    _append_bounded(state.decision_history, decision, max_items)


def snapshot_agent_state(state: AgentRuntimeState) -> Dict[str, Any]:
    return {
        "agent_id": state.agent_id,
        "created_sim_t_s": float(state.created_sim_t_s),
        "last_sim_t_s": float(state.last_sim_t_s),
        "profile": dict(state.profile),
        "belief": dict(state.belief),
        "psychology": dict(state.psychology),
        "signal_history": [dict(item) for item in state.signal_history],
        "social_history": [dict(item) for item in state.social_history],
        "decision_history": [dict(item) for item in state.decision_history],
        "has_departed": bool(state.has_departed),
    }
