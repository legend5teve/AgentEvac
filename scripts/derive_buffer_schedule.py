#!/usr/bin/env python3
"""Size a buffer-alerting policy per community and compile it into an alert schedule.

The proposal is to alert a community once the fire crosses a buffer around it, sized so
the community can finish evacuating before the fire arrives. That needs two inputs.

Evacuation time comes from completed runs. ``area_evacuation_time`` in the run metrics
reports, per area, the span from the order instant to the last household arriving, which
is the window a buffer has to cover. The worst seed is used, since a buffer that only
covers the median leaves half the runs short.

Fire approach comes from the map's ``fires.json``. The fire layer is scripted and
deterministic, so the margin from a community to the nearest burning edge is exact
arithmetic at any instant and needs no simulation. That determinism is also what lets the
policy compile into ordinary timed events, leaving the alert resolver pure and the replay
path untouched.

The margin does not close smoothly. It collapses whenever a new source ignites nearer the
community, so a buffer cannot be sized as a closing speed times a duration. Instead the
smallest buffer is searched for whose headroom, from the trigger to the fire arriving,
covers the evacuation time.

Usage
    python scripts/derive_buffer_schedule.py
    python scripts/derive_buffer_schedule.py --safety 1.5 --write configs/halifax_3town_e0_buffer
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOURCE_MAP = "halifax_3town_e0"
#: Buffer distances searched, in metres.
BUFFER_RANGE = range(50, 6001, 50)
#: Sampling step for the margin trace, in seconds.
STEP_S = 30


def edge_midpoints(net_path: Path, wanted: set) -> dict:
    """Return ``{edge_id: (x, y)}`` at the midpoint of each wanted edge's shape."""
    mids = {}
    for line in open(net_path, errors="ignore"):
        m = re.search(r'<edge id="([^"]+)"[^>]*shape="([^"]+)"', line)
        if m and m.group(1) in wanted:
            pts = [tuple(map(float, p.split(","))) for p in m.group(2).split()]
            mids[m.group(1)] = pts[len(pts) // 2]
    return mids


def margin_at(area_edges, mids, sources, t: float) -> float:
    """Smallest distance in metres from any household in the area to any burning edge."""
    best = math.inf
    for edge_id in area_edges:
        point = mids.get(edge_id)
        if point is None:
            continue
        for src in sources:
            if t < src["t0"]:
                continue
            radius = min(src["r0"] + src["growth_m_per_s"] * (t - src["t0"]),
                         src.get("max_r_m", math.inf))
            best = min(best, math.dist(point, (src["x"], src["y"])) - radius)
    return best


def t_evac_from_timelines(paths, alerts) -> tuple:
    """Recover per-area evacuation time from timeline exports.

    Used for runs made before ``area_evacuation_time`` existed. The timeline carries every
    arrival with its agent id, and the order instant per area comes from the schedule, with
    the door sweep standing in where it reached a community before any broadcast.
    """
    area_of = {e: a for a, spec in alerts["areas"].items() for e in spec["edges"]}
    order_t = {}
    for event in alerts["schedule"]:
        for area in event["areas"]:
            order_t.setdefault(area, float(event["issue_time_s"]))
    door = alerts.get("door_to_door") or {}
    for area in door.get("initial_areas") or ():
        order_t[area] = min(order_t.get(area, math.inf), float(door.get("start_time_s", 0.0)))

    worst, used = {}, 0
    for path in sorted(paths):
        last, count = {}, {}
        for line in open(path):
            row = json.loads(line)
            if row.get("type") != "arrive":
                continue
            area = area_of.get(str(row.get("agent_id", "")).rsplit("_", 1)[0])
            if area is None:
                continue
            last[area] = max(last.get(area, 0.0), float(row["t_s"]))
            count[area] = count.get(area, 0) + 1
        if not last:
            continue
        used += 1
        for area, arrival in last.items():
            if area in order_t:
                worst[area] = max(worst.get(area, 0.0), arrival - order_t[area])
    return worst, used


def size_buffer(trace, t_evac: float, safety: float):
    """Return ``(buffer_m, trigger_t_s, arrival_t_s, headroom_s)``, or ``None`` if none fits.

    ``trace`` is a list of ``(t, margin)``. The buffer has to be large enough that the fire
    takes at least ``t_evac * safety`` to cover the remaining ground after the alert.
    """
    arrival = next((t for t, m in trace if m <= 0.0), None)
    needed = t_evac * safety
    for buffer_m in BUFFER_RANGE:
        trigger = next((t for t, m in trace if m <= buffer_m), None)
        if trigger is None:
            continue
        headroom = math.inf if arrival is None else arrival - trigger
        if headroom >= needed:
            return buffer_m, trigger, arrival, headroom
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Size buffer alerting and compile the schedule.")
    ap.add_argument("--map", default=SOURCE_MAP, help=f"Source config (default {SOURCE_MAP}).")
    ap.add_argument("--runs", default="outputs/E0/e0_msgoff/*/run_metrics_*.json",
                    help="Glob of completed run metrics supplying the evacuation times.")
    ap.add_argument("--timelines", default="outputs/E0/e0_msgoff/*/run_timeline_*.jsonl",
                    help="Fallback glob, used when the runs predate area_evacuation_time.")
    ap.add_argument("--safety", type=float, default=1.0,
                    help="Multiplier on the evacuation time the buffer must cover.")
    ap.add_argument("--write", help="Write the compiled config to this directory.")
    args = ap.parse_args()

    import glob
    map_dir = REPO / "configs" / args.map
    alerts = json.load(open(map_dir / "alerts.json"))
    fires = json.load(open(map_dir / "fires.json"))
    net = REPO / json.load(open(map_dir / "map.json"))["net_file"]

    # Worst observed evacuation time per area, across the supplied runs.
    t_evac, seen = {}, 0
    for path in sorted(glob.glob(str(REPO / args.runs))):
        summary = json.load(open(path))
        rows = summary.get("area_evacuation_time")
        if not rows:
            continue
        seen += 1
        for area, row in rows.items():
            if row.get("t_evac_s") is not None:
                t_evac[area] = max(t_evac.get(area, 0.0), float(row["t_evac_s"]))
    source = f"{seen} run metrics"
    if not t_evac:
        # Runs made before area_evacuation_time existed still carry every arrival in their
        # timeline export, and the order instants are in the config, so the same quantity
        # is recoverable without rerunning anything.
        t_evac, seen = t_evac_from_timelines(glob.glob(str(REPO / args.timelines)), alerts)
        source = f"{seen} timeline export(s), since the run metrics predate the metric"
    if not t_evac:
        sys.exit(f"No evacuation times in {args.runs} or {args.timelines}.")
    print(f"Evacuation times from {source}, worst case per area")

    wanted = {e for spec in alerts["areas"].values() for e in spec["edges"]}
    mids = edge_midpoints(net, wanted)
    horizon = int(alerts["_meta"]["clock"].get("recommended_sim_end_time_s", 28800))
    times = list(range(0, horizon + 1, STEP_S))

    print(f"\n{'community':18} {'T_evac s':>9} {'buffer m':>9} {'trigger s':>10} "
          f"{'fire in s':>10} {'headroom s':>11}")
    schedule = {}
    for area, spec in alerts["areas"].items():
        if area not in t_evac:
            print(f"{area:18} no completed evacuation in the runs, skipped")
            continue
        trace = [(t, margin_at(spec["edges"], mids, fires["sources"], t)) for t in times]
        fit = size_buffer(trace, t_evac[area], args.safety)
        if fit is None:
            trigger = trace[0][0]
            arrival = next((t for t, m in trace if m <= 0.0), None)
            head = "n/a" if arrival is None else f"{arrival - trigger:.0f}"
            print(f"{area:18} {t_evac[area]:9.0f} {'none fits':>9} {trigger:10} "
                  f"{arrival if arrival is not None else -1:10} {head:>11}  "
                  f"alert at ignition is already too late")
            schedule[area] = {"buffer_m": None, "trigger_t_s": trigger, "feasible": False}
            continue
        buffer_m, trigger, arrival, headroom = fit
        print(f"{area:18} {t_evac[area]:9.0f} {buffer_m:9} {trigger:10} "
              f"{arrival if arrival is not None else -1:10} "
              f"{'inf' if headroom == math.inf else f'{headroom:.0f}':>11}")
        schedule[area] = {"buffer_m": buffer_m, "trigger_t_s": trigger, "feasible": True}

    if args.write:
        write_config(Path(args.write), map_dir, alerts, schedule, t_evac, args.safety)
    return 0


def write_config(out_dir: Path, map_dir: Path, alerts, schedule, t_evac, safety) -> None:
    """Compile the buffer policy into a map config whose events are ordinary timed alerts."""
    import shutil
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("map.json", "spawns.json", "fires.json", "destinations.json",
                 "routes.json", "corridors.json"):
        if (map_dir / name).is_file():
            shutil.copy(map_dir / name, out_dir / name)

    events = []
    for event in alerts["schedule"]:
        areas = [a for a in event["areas"] if a in schedule]
        if not areas:
            continue
        new = dict(event)
        new["issue_time_s"] = min(schedule[a]["trigger_t_s"] for a in areas)
        new["wall_clock"] = None
        new["source"] = (
            f"BUFFER POLICY. Issued when the fire margin fell to the buffer sized for "
            f"{', '.join(areas)}. Areas, instruction, hazard text and channel are the "
            f"historical event {event['id']}. Compiled by scripts/derive_buffer_schedule.py."
        )
        new["buffer_m"] = {a: schedule[a]["buffer_m"] for a in areas}
        new["buffer_feasible"] = {a: schedule[a]["feasible"] for a in areas}
        events.append(new)
    events.sort(key=lambda e: e["issue_time_s"])

    out = dict(alerts)
    out["schedule"] = events
    meta = dict(alerts["_meta"])
    meta["arm"] = (
        "Buffer-alerting counterfactual. Each community is alerted when the fire margin "
        "falls to a buffer sized so the community can finish evacuating first. Identical "
        "to the E0 config in network, spawns, fires, destinations, routes, areas, "
        "instructions and the door-to-door channel. Only issue_time_s differs."
    )
    meta["buffer_policy"] = {
        "safety_factor": safety,
        "t_evac_s": {a: round(v, 1) for a, v in sorted(t_evac.items())},
        "buffer_m": {a: s["buffer_m"] for a, s in sorted(schedule.items())},
        "trigger_t_s": {a: s["trigger_t_s"] for a, s in sorted(schedule.items())},
        "infeasible": sorted(a for a, s in schedule.items() if not s["feasible"]),
        "note": (
            "Compiled offline because the fire layer is deterministic, so the alert "
            "resolver stays pure and replay is unchanged. That also means the policy is "
            "given the whole ignition schedule in advance, which a real operations centre "
            "would not have, so these triggers are an upper bound on a live policy."
        ),
    }
    out["_meta"] = meta
    with open(out_dir / "alerts.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    raise SystemExit(main())
