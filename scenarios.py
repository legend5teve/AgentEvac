from typing import Any, Dict, List, Tuple


SCENARIO_CHOICES: Tuple[str, ...] = (
    "no_notice",
    "alert_guided",
    "advice_guided",
)


def load_scenario_config(mode: str) -> Dict[str, Any]:
    name = str(mode).strip().lower()
    if name == "no_notice":
        return {
            "mode": name,
            "title": "No-Notice Wildfire",
            "description": (
                "No official warning is available yet. Agents rely on self-observation and neighbor messages."
            ),
            "forecast_visible": False,
            "route_head_forecast_visible": False,
            "official_route_guidance_visible": False,
            "expected_utility_visible": False,
        }
    if name == "alert_guided":
        return {
            "mode": name,
            "title": "Alert-Guided Evacuation",
            "description": (
                "Official alerts expose hazard location and projected spread, but do not prescribe a route."
            ),
            "forecast_visible": True,
            "route_head_forecast_visible": False,
            "official_route_guidance_visible": False,
            "expected_utility_visible": False,
        }
    return {
        "mode": "advice_guided",
        "title": "Advice-Guided Evacuation",
        "description": (
            "Official alerts include both hazard information and route-oriented guidance."
        ),
        "forecast_visible": True,
        "route_head_forecast_visible": True,
        "official_route_guidance_visible": True,
        "expected_utility_visible": True,
    }


def apply_scenario_to_signals(
    mode: str,
    env_signal: Dict[str, Any],
    forecast: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    cfg = load_scenario_config(mode)
    env_prompt = dict(env_signal or {})
    forecast_prompt = dict(forecast or {})

    if cfg["mode"] == "no_notice":
        env_prompt = {
            "observed_state": env_prompt.get("observed_state"),
            "is_delayed": bool(env_prompt.get("is_delayed", False)),
            "delay_rounds_applied": int(env_prompt.get("delay_rounds_applied", 0) or 0),
            "source": "self_observation_and_neighbors_only",
            "note": "No official warning bulletin is available in this scenario.",
        }
        forecast_prompt = {
            "available": False,
            "briefing": "No official forecast is available yet.",
        }
        return env_prompt, forecast_prompt

    if cfg["mode"] == "alert_guided":
        route_head = dict(forecast_prompt.get("route_head") or {})
        forecast_prompt = {
            "available": True,
            "summary": dict(forecast_prompt.get("summary") or {}),
            "current_edge": dict(forecast_prompt.get("current_edge") or {}),
            "route_head": {
                "available": False,
                "note": "Alert-only mode: no official route-specific advice.",
                "head_edges_evaluated": route_head.get("head_edges_evaluated"),
            },
            "briefing": str(forecast_prompt.get("briefing") or ""),
        }
        return env_prompt, forecast_prompt

    return env_prompt, forecast_prompt


def filter_menu_for_scenario(
    mode: str,
    menu: List[Dict[str, Any]],
    *,
    control_mode: str,
) -> List[Dict[str, Any]]:
    cfg = load_scenario_config(mode)
    prompt_menu: List[Dict[str, Any]] = []

    for item in menu:
        out = dict(item)
        if not cfg["official_route_guidance_visible"]:
            out.pop("advisory", None)
            out.pop("briefing", None)
            out.pop("reasons", None)

        if not cfg["expected_utility_visible"]:
            out.pop("expected_utility", None)
            out.pop("utility_components", None)

        if cfg["mode"] == "no_notice":
            if control_mode == "destination":
                keep_keys = {"idx", "name", "dest_edge", "reachable", "note"}
            else:
                keep_keys = {"idx", "name", "len_edges"}
            out = {k: v for k, v in out.items() if k in keep_keys}

        prompt_menu.append(out)

    return prompt_menu


def scenario_prompt_suffix(mode: str) -> str:
    cfg = load_scenario_config(mode)
    if cfg["mode"] == "no_notice":
        return (
            "This is a no-notice wildfire scenario: do not assume official route instructions exist. "
            "Rely mainly on subjective_information, inbox messages, and your own caution."
        )
    if cfg["mode"] == "alert_guided":
        return (
            "This is an alert-guided scenario: official alerts describe the fire, but they do not prescribe a route. "
            "Use forecast and hazard cues, but make your own navigation choice."
        )
    return (
        "This is an advice-guided scenario: official alerts include route-oriented guidance. "
        "You may use advisories, briefings, and expected utility as formal support."
    )
