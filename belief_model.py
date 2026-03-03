import math
from typing import Any, Dict


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def _normalize_triplet(belief: Dict[str, float]) -> Dict[str, float]:
    total = float(
        belief.get("p_safe", 0.0) +
        belief.get("p_risky", 0.0) +
        belief.get("p_danger", 0.0)
    )
    if total <= 0.0:
        return {"p_safe": 1.0 / 3.0, "p_risky": 1.0 / 3.0, "p_danger": 1.0 / 3.0}
    return {
        "p_safe": float(belief.get("p_safe", 0.0)) / total,
        "p_risky": float(belief.get("p_risky", 0.0)) / total,
        "p_danger": float(belief.get("p_danger", 0.0)) / total,
    }


def categorize_hazard_state(signal: Dict[str, Any]) -> Dict[str, float]:
    margin = signal.get("observed_margin_m")
    if margin is None:
        margin = signal.get("base_margin_m")
    if margin is None:
        return {"p_safe": 1.0 / 3.0, "p_risky": 1.0 / 3.0, "p_danger": 1.0 / 3.0}

    margin_f = float(margin)
    if margin_f <= 0.0:
        return {"p_safe": 0.02, "p_risky": 0.08, "p_danger": 0.90}
    if margin_f <= 100.0:
        return {"p_safe": 0.05, "p_risky": 0.20, "p_danger": 0.75}
    if margin_f <= 300.0:
        return {"p_safe": 0.15, "p_risky": 0.55, "p_danger": 0.30}
    if margin_f <= 700.0:
        return {"p_safe": 0.35, "p_risky": 0.50, "p_danger": 0.15}
    return {"p_safe": 0.75, "p_risky": 0.20, "p_danger": 0.05}


def fuse_env_and_social_beliefs(
    env_belief: Dict[str, float],
    social_belief: Dict[str, float],
    theta_trust: float,
) -> Dict[str, float]:
    social_weight = _clamp(theta_trust, 0.0, 1.0)
    env_weight = 1.0 - social_weight
    fused = {
        "p_safe": env_weight * float(env_belief.get("p_safe", 0.0)) + social_weight * float(social_belief.get("p_safe", 0.0)),
        "p_risky": env_weight * float(env_belief.get("p_risky", 0.0)) + social_weight * float(social_belief.get("p_risky", 0.0)),
        "p_danger": env_weight * float(env_belief.get("p_danger", 0.0)) + social_weight * float(social_belief.get("p_danger", 0.0)),
    }
    return _normalize_triplet(fused)


def smooth_belief(
    prev_belief: Dict[str, float],
    new_belief: Dict[str, float],
    inertia: float = 0.35,
) -> Dict[str, float]:
    smooth = _clamp(inertia, 0.0, 0.999)
    prev = _normalize_triplet(prev_belief)
    new = _normalize_triplet(new_belief)
    merged = {
        "p_safe": smooth * prev["p_safe"] + (1.0 - smooth) * new["p_safe"],
        "p_risky": smooth * prev["p_risky"] + (1.0 - smooth) * new["p_risky"],
        "p_danger": smooth * prev["p_danger"] + (1.0 - smooth) * new["p_danger"],
    }
    return _normalize_triplet(merged)


def compute_belief_entropy(belief: Dict[str, float]) -> float:
    norm = _normalize_triplet(belief)
    total = 0.0
    for key in ("p_safe", "p_risky", "p_danger"):
        p = max(1e-12, float(norm[key]))
        total -= p * math.log(p)
    return total


def normalize_entropy(entropy: float) -> float:
    max_entropy = math.log(3.0)
    if max_entropy <= 0.0:
        return 0.0
    return _clamp(float(entropy) / max_entropy, 0.0, 1.0)


def bucket_uncertainty(entropy_norm: float) -> str:
    val = _clamp(entropy_norm, 0.0, 1.0)
    if val <= 0.33:
        return "Low"
    if val <= 0.67:
        return "Medium"
    return "High"


def update_agent_belief(
    prev_belief: Dict[str, float],
    env_signal: Dict[str, Any],
    social_signal: Dict[str, Any],
    theta_trust: float,
    inertia: float = 0.35,
) -> Dict[str, Any]:
    env_belief = categorize_hazard_state(env_signal)

    social_count = int(social_signal.get("message_count", 0) or 0)
    social_belief_raw = social_signal.get("social_belief") or {}
    if social_count > 0:
        social_belief = _normalize_triplet(social_belief_raw)
        fused = fuse_env_and_social_beliefs(env_belief, social_belief, theta_trust)
        social_weight = _clamp(theta_trust, 0.0, 1.0)
        env_weight = 1.0 - social_weight
    else:
        social_belief = {"p_safe": 1.0 / 3.0, "p_risky": 1.0 / 3.0, "p_danger": 1.0 / 3.0}
        fused = dict(env_belief)
        social_weight = 0.0
        env_weight = 1.0

    smoothed = smooth_belief(prev_belief or env_belief, fused, inertia=inertia)
    entropy = compute_belief_entropy(smoothed)
    entropy_norm = normalize_entropy(entropy)

    return {
        "p_safe": smoothed["p_safe"],
        "p_risky": smoothed["p_risky"],
        "p_danger": smoothed["p_danger"],
        "entropy": round(entropy, 4),
        "entropy_norm": round(entropy_norm, 4),
        "uncertainty_bucket": bucket_uncertainty(entropy_norm),
        "env_weight": round(env_weight, 4),
        "social_weight": round(social_weight, 4),
        "env_belief": env_belief,
        "social_belief": social_belief,
    }
