from typing import Any, Dict, Tuple

from agent_state import AgentRuntimeState


def should_depart_now(
    agent_state: AgentRuntimeState,
    belief: Dict[str, Any],
    psychology: Dict[str, Any],
    sim_t_s: float,
) -> Tuple[bool, str]:
    p_danger = float(belief.get("p_danger", 0.0))
    p_safe = float(belief.get("p_safe", 0.0))
    theta_r = float(agent_state.profile.get("theta_r", 0.45))
    theta_u = float(agent_state.profile.get("theta_u", 0.30))
    gamma = float(agent_state.profile.get("gamma", 0.995))
    created_t = float(agent_state.created_sim_t_s)
    elapsed_s = max(0.0, float(sim_t_s) - created_t)

    if p_danger > theta_r:
        return True, "risk_threshold"

    urgency_term = (gamma ** elapsed_s) * p_safe
    if urgency_term < theta_u:
        return True, "urgency_threshold"

    confidence = float(psychology.get("confidence", 0.0))
    if confidence < 0.15 and p_danger >= max(0.20, theta_r * 0.6):
        return True, "low_confidence_precaution"

    return False, "wait"
