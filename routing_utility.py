from typing import Any, Dict, List


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _effective_margin_penalty(min_margin_m: Any) -> float:
    margin = _num(min_margin_m, default=float("inf"))
    if margin == float("inf"):
        return 0.25
    if margin <= 0.0:
        return 5.0
    if margin <= 100.0:
        return 3.0
    if margin <= 300.0:
        return 1.5
    if margin <= 700.0:
        return 0.6
    return 0.15


def _travel_cost(menu_item: Dict[str, Any]) -> float:
    travel_time = menu_item.get("travel_time_s_fastest_path")
    if travel_time is not None:
        return _num(travel_time, 0.0) / 60.0
    edge_count = menu_item.get("len_edges")
    if edge_count is None:
        edge_count = menu_item.get("len_edges_fastest_path")
    return _num(edge_count, 0.0) * 0.25


def _expected_exposure(
    menu_item: Dict[str, Any],
    belief: Dict[str, Any],
    psychology: Dict[str, Any],
) -> float:
    risk_sum = _num(menu_item.get("risk_sum", menu_item.get("risk_sum_on_fastest_path")), 0.0)
    blocked_edges = _num(menu_item.get("blocked_edges", menu_item.get("blocked_edges_on_fastest_path")), 0.0)
    min_margin_m = menu_item.get("min_margin_m", menu_item.get("min_margin_m_on_fastest_path"))

    p_risky = _num(belief.get("p_risky"), 1.0 / 3.0)
    p_danger = _num(belief.get("p_danger"), 1.0 / 3.0)
    perceived_risk = _num(psychology.get("perceived_risk"), p_danger)
    confidence = _num(psychology.get("confidence"), 0.0)

    severity_scale = 1.0 + (0.8 * p_risky) + (1.6 * p_danger) + (0.6 * perceived_risk)
    uncertainty_penalty = max(0.0, 1.0 - confidence) * 0.75

    return (
        risk_sum * severity_scale
        + (blocked_edges * 8.0)
        + _effective_margin_penalty(min_margin_m)
        + uncertainty_penalty
    )


def score_destination_utility(
    menu_item: Dict[str, Any],
    belief: Dict[str, Any],
    psychology: Dict[str, Any],
    profile: Dict[str, Any],
) -> float:
    lambda_e = max(0.0, _num(profile.get("lambda_e"), 1.0))
    lambda_t = max(0.0, _num(profile.get("lambda_t"), 0.1))
    expected_exposure = _expected_exposure(menu_item, belief, psychology)
    travel_cost = _travel_cost(menu_item)
    return -((lambda_e * expected_exposure) + (lambda_t * travel_cost))


def score_route_utility(
    menu_item: Dict[str, Any],
    belief: Dict[str, Any],
    psychology: Dict[str, Any],
    profile: Dict[str, Any],
) -> float:
    lambda_e = max(0.0, _num(profile.get("lambda_e"), 1.0))
    lambda_t = max(0.0, _num(profile.get("lambda_t"), 0.1))
    expected_exposure = _expected_exposure(menu_item, belief, psychology)
    travel_cost = _travel_cost(menu_item)
    return -((lambda_e * expected_exposure) + (lambda_t * travel_cost))


def annotate_menu_with_expected_utility(
    menu: List[Dict[str, Any]],
    *,
    mode: str,
    belief: Dict[str, Any],
    psychology: Dict[str, Any],
    profile: Dict[str, Any],
) -> List[Dict[str, Any]]:
    for item in menu:
        if mode == "destination":
            if not item.get("reachable", False):
                item["expected_utility"] = None
                item["utility_components"] = {
                    "lambda_e": max(0.0, _num(profile.get("lambda_e"), 1.0)),
                    "lambda_t": max(0.0, _num(profile.get("lambda_t"), 0.1)),
                    "reachable": False,
                }
                continue
            expected_exposure = _expected_exposure(item, belief, psychology)
            travel_cost = _travel_cost(item)
            utility = score_destination_utility(item, belief, psychology, profile)
        else:
            expected_exposure = _expected_exposure(item, belief, psychology)
            travel_cost = _travel_cost(item)
            utility = score_route_utility(item, belief, psychology, profile)

        item["expected_utility"] = round(utility, 4)
        item["utility_components"] = {
            "lambda_e": round(max(0.0, _num(profile.get("lambda_e"), 1.0)), 4),
            "lambda_t": round(max(0.0, _num(profile.get("lambda_t"), 0.1)), 4),
            "expected_exposure": round(expected_exposure, 4),
            "travel_cost": round(travel_cost, 4),
        }
    return menu
