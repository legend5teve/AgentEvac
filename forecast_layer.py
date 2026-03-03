from typing import Any, Callable, Dict, List, Optional


def _round_or_none(value: Optional[float], digits: int = 2) -> Optional[float]:
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def _margin_band(margin_m: Optional[float]) -> str:
    if margin_m is None:
        return "unknown"
    if margin_m <= 0.0:
        return "inside_predicted_fire"
    if margin_m <= 100.0:
        return "very_close"
    if margin_m <= 300.0:
        return "near"
    if margin_m <= 700.0:
        return "buffered"
    return "clear"


def build_fire_forecast(
    sim_t_s: float,
    current_fires: List[Dict[str, Any]],
    projected_fires: List[Dict[str, Any]],
    horizon_s: float,
) -> Dict[str, Any]:
    current_by_id = {str(item.get("id")): item for item in current_fires}
    projected_by_id = {str(item.get("id")): item for item in projected_fires}

    growth_values: List[float] = []
    for fire_id, projected in projected_by_id.items():
        current = current_by_id.get(fire_id)
        if current is None:
            growth_values.append(float(projected.get("r", 0.0)))
            continue
        growth_values.append(float(projected.get("r", 0.0)) - float(current.get("r", 0.0)))

    max_growth = max(growth_values) if growth_values else 0.0
    avg_growth = (sum(growth_values) / float(len(growth_values))) if growth_values else 0.0
    new_fire_ids = sorted([fire_id for fire_id in projected_by_id.keys() if fire_id not in current_by_id])

    return {
        "horizon_s": _round_or_none(horizon_s, 2),
        "generated_at_s": _round_or_none(sim_t_s, 2),
        "current_fire_count": len(current_fires),
        "projected_fire_count": len(projected_fires),
        "new_projected_fire_ids": new_fire_ids,
        "max_projected_radius_m": _round_or_none(
            max((float(item.get("r", 0.0)) for item in projected_fires), default=0.0),
            2,
        ),
        "max_radius_growth_m": _round_or_none(max_growth, 2),
        "avg_radius_growth_m": _round_or_none(avg_growth, 2),
    }


def estimate_edge_forecast_risk(
    edge_id: str,
    edge_risk_fn: Callable[[str], Any],
) -> Dict[str, Any]:
    blocked, risk_score, margin_m = edge_risk_fn(edge_id)
    return {
        "edge_id": edge_id,
        "blocked": bool(blocked),
        "risk_score": _round_or_none(risk_score, 4),
        "margin_m": _round_or_none(margin_m, 2),
        "band": _margin_band(_round_or_none(margin_m, 2)),
    }


def summarize_route_forecast(
    route_edges: List[str],
    edge_risk_fn: Callable[[str], Any],
    max_edges: int,
) -> Dict[str, Any]:
    checked = 0
    blocked_edges = 0
    min_margin: Optional[float] = None
    max_risk = 0.0

    for edge_id in route_edges:
        if checked >= max(1, int(max_edges)):
            break
        if not edge_id or edge_id.startswith(":"):
            continue
        blocked, risk_score, margin_m = edge_risk_fn(edge_id)
        checked += 1
        if blocked:
            blocked_edges += 1
        if margin_m is not None:
            margin_val = float(margin_m)
            if min_margin is None or margin_val < min_margin:
                min_margin = margin_val
        if risk_score is not None:
            max_risk = max(max_risk, float(risk_score))

    return {
        "head_edges_evaluated": checked,
        "blocked_edges": blocked_edges,
        "min_margin_m": _round_or_none(min_margin, 2),
        "max_risk_score": _round_or_none(max_risk, 4),
        "band": _margin_band(_round_or_none(min_margin, 2)),
    }


def render_forecast_briefing(
    agent_id: str,
    forecast: Dict[str, Any],
    belief: Dict[str, Any],
    edge_forecast: Dict[str, Any],
    route_forecast: Dict[str, Any],
) -> str:
    horizon_s = int(round(float(forecast.get("horizon_s") or 0.0)))
    edge_band = edge_forecast.get("band", "unknown").replace("_", " ")
    route_band = route_forecast.get("band", "unknown").replace("_", " ")
    blocked_edges = int(route_forecast.get("blocked_edges") or 0)
    p_danger = float(belief.get("p_danger") or 0.0)
    uncertainty = str(belief.get("uncertainty_bucket") or "High")

    if blocked_edges > 0:
        route_clause = f"{blocked_edges} route-head segment(s) may be blocked"
    else:
        route_clause = f"route head looks {route_band}"

    if edge_forecast.get("blocked"):
        edge_clause = "your current edge may be overtaken"
    else:
        edge_clause = f"your current edge looks {edge_band}"

    if p_danger >= 0.5:
        tone = "Forecast suggests the threat is building"
    elif uncertainty == "High":
        tone = "Forecast is uncertain, but conditions may tighten"
    else:
        tone = "Forecast suggests a manageable window"

    return f"{tone} within {horizon_s}s: {edge_clause}, and {route_clause}."
