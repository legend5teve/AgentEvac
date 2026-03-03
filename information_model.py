import random
from typing import Any, Dict, List, Optional


def _state_from_margin(margin_m: Optional[float]) -> str:
    if margin_m is None:
        return "unknown"
    if margin_m <= 100.0:
        return "danger"
    if margin_m <= 300.0:
        return "risky"
    return "safe"


def inject_signal_noise(
    signal: Dict[str, Any],
    sigma_info: float,
    rng: Optional[random.Random] = None,
) -> Dict[str, Any]:
    out = dict(signal)
    sigma = max(0.0, float(sigma_info))
    base_margin = out.get("base_margin_m")
    if base_margin is None:
        out["noise_delta_m"] = 0.0
        out["observed_margin_m"] = None
        out["observed_state"] = "unknown"
        return out

    src = rng if rng is not None else random
    noise_delta = float(src.gauss(0.0, sigma)) if sigma > 0.0 else 0.0
    observed_margin = float(base_margin) + noise_delta
    out["noise_delta_m"] = round(noise_delta, 2)
    out["observed_margin_m"] = round(observed_margin, 2)
    out["observed_state"] = _state_from_margin(observed_margin)
    return out


def apply_signal_delay(
    signal: Dict[str, Any],
    history: List[Dict[str, Any]],
    delay_rounds: int,
) -> Dict[str, Any]:
    delay = max(0, int(delay_rounds))
    if delay <= 0:
        out = dict(signal)
        out["is_delayed"] = False
        out["delay_rounds_applied"] = 0
        return out

    if delay <= len(history):
        source = dict(history[-delay])
        source["is_delayed"] = True
        source["delay_rounds_applied"] = delay
        source["delay_source_round"] = source.get("decision_round")
        return source

    out = dict(signal)
    out["is_delayed"] = False
    out["delay_rounds_applied"] = 0
    return out


def sample_environment_signal(
    agent_id: str,
    sim_t_s: float,
    current_edge: str,
    current_edge_margin_m: Optional[float],
    route_head_min_margin_m: Optional[float],
    decision_round: int,
    sigma_info: float,
    rng: Optional[random.Random] = None,
) -> Dict[str, Any]:
    base_margin = route_head_min_margin_m
    source_metric = "route_head_min_margin_m"
    if base_margin is None:
        base_margin = current_edge_margin_m
        source_metric = "current_edge_margin_m"

    signal = {
        "agent_id": agent_id,
        "sim_t_s": round(float(sim_t_s), 2),
        "decision_round": int(decision_round),
        "current_edge": current_edge,
        "source_metric": source_metric,
        "sigma_info": float(max(0.0, sigma_info)),
        "base_margin_m": None if base_margin is None else round(float(base_margin), 2),
        "observed_margin_m": None,
        "observed_state": "unknown",
    }
    return inject_signal_noise(signal, sigma_info, rng=rng)


def build_social_signal(
    agent_id: str,
    inbox: List[Dict[str, Any]],
    *,
    max_messages: int = 5,
) -> Dict[str, Any]:
    considered = list(inbox[-max(1, int(max_messages)):]) if inbox else []
    votes = {"safe": 0, "risky": 0, "danger": 0}
    danger_words = ("fire", "blocked", "danger", "smoke", "trapped")
    risky_words = ("slow", "crowded", "traffic", "risk", "near")
    safe_words = ("clear", "open", "safe", "passable")

    for item in considered:
        text = str(item.get("message", "")).lower()
        if any(word in text for word in danger_words):
            votes["danger"] += 1
        elif any(word in text for word in risky_words):
            votes["risky"] += 1
        elif any(word in text for word in safe_words):
            votes["safe"] += 1

    total_votes = votes["safe"] + votes["risky"] + votes["danger"]
    if total_votes <= 0:
        social_belief = {
            "p_safe": 1.0 / 3.0,
            "p_risky": 1.0 / 3.0,
            "p_danger": 1.0 / 3.0,
        }
        dominant_state = "none"
    else:
        social_belief = {
            "p_safe": votes["safe"] / float(total_votes),
            "p_risky": votes["risky"] / float(total_votes),
            "p_danger": votes["danger"] / float(total_votes),
        }
        dominant_state = max(votes, key=votes.get)

    return {
        "agent_id": agent_id,
        "message_count": len(considered),
        "votes": dict(votes),
        "social_belief": social_belief,
        "dominant_state": dominant_state,
    }
