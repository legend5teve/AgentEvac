#!/usr/bin/env python3
"""Cross-check the egress-road binding of an alert area against the network.

The authoritative binding is the evacuation-district map in the HRM report appendix, and
it lives in the config as ``routing_branches``. This script recomputes the binding from the
network and reports where the two disagree, which is how the Wyndham Drive error was found.
A shortest path out is a proxy for the district rule and it is not the rule itself, so the
map wins wherever they differ.

The Westwood Hills Evacuation Plan (HRM 231017rci05, Operations and Police sections)
splits the subdivision at its two exits. Traffic leaving by Westwood Boulevard turns right
at Hammonds Plains Road, traffic leaving by Winslow Drive turns left, and the stated
purpose is to keep both exits off the same stretch of road. Reproducing that needs each
household to know which of the two roads is its own.

``sumo/halifax.net.xml`` carries no name attribute, so the binding runs in two steps.
Street names come from ``sumo/westwood_street_names.json``, an OpenStreetMap extract keyed
by way id, which is the base of a SUMO edge id. Then a shortest path is computed from each
household edge out to the trunk road, and the household is assigned to whichever named
egress road that path uses.

The assignment is a property of the network, so it is derived and not authored. Rerunning
this script reproduces it exactly from the two checked-in inputs, with no network access.

Usage
    python scripts/derive_westwood_districts.py
    python scripts/derive_westwood_districts.py --map halifax_3town_e0_routing --json out.json
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NAMES_FILE = REPO / "sumo" / "westwood_street_names.json"
DEFAULT_MAP = "halifax_3town_e0"
DEFAULT_AREA = "westwood_hills"
#: The arterial every household reaches, and the roads the plan splits the subdivision by.
TRUNK_ROAD = "Hammonds Plains Road"
EGRESS_ROADS = ("Westwood Boulevard", "Winslow Drive")
#: Fallback length for an edge the network gives no shape, in metres.
DEFAULT_EDGE_LEN_M = 50.0


def load_street_names() -> dict:
    """Return the OSM way id to street name lookup."""
    if not NAMES_FILE.is_file():
        sys.exit(f"Missing {NAMES_FILE}. It is the checked-in OpenStreetMap name extract.")
    return json.load(open(NAMES_FILE))["names"]


def way_id(edge_id: str) -> str:
    """Return the OSM way id a SUMO edge id derives from."""
    return edge_id.lstrip("-").split("#")[0]


def load_network(net_path: Path, names: dict):
    """Return ``(edges, named, out_of)`` for the driving graph.

    ``edges`` maps edge id to ``(from_node, to_node, length_m)``, ``named`` maps edge id to
    street name for the edges the extract covers, and ``out_of`` maps a node to the edges
    leaving it.
    """
    edges, named = {}, {}
    for line in open(net_path, errors="ignore"):
        m = re.search(r'<edge id="([^"]+)" from="([^"]+)" to="([^"]+)"', line)
        if not m:
            continue
        edge_id, from_node, to_node = m.groups()
        shape = re.search(r'shape="([^"]+)"', line)
        if shape:
            pts = [tuple(map(float, p.split(","))) for p in shape.group(1).split()]
            length = sum(math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))
        else:
            length = 0.0
        edges[edge_id] = (from_node, to_node, length or DEFAULT_EDGE_LEN_M)
        name = names.get(way_id(edge_id))
        if name:
            named[edge_id] = name

    out_of = defaultdict(list)
    for edge_id, (from_node, _to, _len) in edges.items():
        out_of[from_node].append(edge_id)
    return edges, named, out_of


def trunk_junctions(edges, named) -> dict:
    """Return the node where each egress road meets the trunk road.

    The plan describes a subdivision with two roads out, both meeting Hammonds Plains Road
    within 200 metres of each other, and that structure is what the split depends on. A
    road with no junction, or with several, means the name extract or the area is not the
    subdivision the plan describes, so this is checked and not assumed.
    """
    trunk_nodes = {n for e, (f, t, _l) in edges.items() if named.get(e) == TRUNK_ROAD
                   for n in (f, t)}
    found = {}
    for road in EGRESS_ROADS:
        road_nodes = {n for e, (f, t, _l) in edges.items() if named.get(e) == road
                      for n in (f, t)}
        found[road] = sorted(road_nodes & trunk_nodes)
    return found


def junction_positions(net_path: Path, node_ids) -> dict:
    """Return ``{node_id: (x, y)}`` for the junctions named in ``node_ids``."""
    wanted, pos = set(node_ids), {}
    for line in open(net_path, errors="ignore"):
        m = re.search(r'<junction id="([^"]+)" type="[^"]*" x="([-\d.]+)" y="([-\d.]+)"', line)
        if m and m.group(1) in wanted:
            pos[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    return pos


def egress_road(start: str, edges, named, out_of) -> tuple:
    """Return ``(egress_road, metres, path)`` for the shortest way out of the subdivision.

    Dijkstra runs from ``start`` until it first reaches the trunk road, then walks the path
    backwards for the last named egress road it used. Returns ``(None, None, [])`` when no
    path reaches the trunk, which would mean the area or the name extract is incomplete.
    """
    trunk = {e for e, n in named.items() if n == TRUNK_ROAD}
    queue, seen = [(0.0, start, [start])], set()
    while queue:
        dist, edge_id, path = heapq.heappop(queue)
        if edge_id in seen:
            continue
        seen.add(edge_id)
        if edge_id in trunk and len(path) > 1:
            for step in reversed(path[:-1]):
                if named.get(step) in EGRESS_ROADS:
                    return named[step], dist, path
            return "other", dist, path
        for nxt in out_of[edges[edge_id][1]]:
            if nxt not in seen:
                queue.append((dist + edges[nxt][2], nxt, path + [nxt]))
                heapq.heapify(queue)
    return None, None, []


def main() -> int:
    ap = argparse.ArgumentParser(description="Bind alert-area household edges to their egress road.")
    ap.add_argument("--map", default=DEFAULT_MAP, help=f"Config dir under configs/ (default {DEFAULT_MAP}).")
    ap.add_argument("--area", default=DEFAULT_AREA, help=f"Alert area to bind (default {DEFAULT_AREA}).")
    ap.add_argument("--json", help="Write the assignment to this path as JSON.")
    args = ap.parse_args()

    map_dir = REPO / "configs" / args.map
    alerts = json.load(open(map_dir / "alerts.json"))
    area = alerts["areas"].get(args.area)
    if area is None:
        sys.exit(f"Area {args.area!r} not in {map_dir/'alerts.json'}. Have {list(alerts['areas'])}.")

    spawn_counts = {g["edge"]: g.get("count", 1)
                    for g in json.load(open(map_dir / "spawns.json"))["groups"]}
    names = load_street_names()
    net_path = REPO / json.load(open(map_dir / "map.json"))["net_file"]
    edges, named, out_of = load_network(net_path, names)

    junctions = trunk_junctions(edges, named)
    print(f"Junctions with {TRUNK_ROAD}")
    flat = [n for ns in junctions.values() for n in ns]
    pos = junction_positions(net_path, flat)
    for road, ns in junctions.items():
        print(f"   {road:22} {len(ns)} junction(s) {ns}")
    if len(flat) == 2 and len(pos) == 2:
        (ax, ay), (bx, by) = (pos[flat[0]], pos[flat[1]])
        gap = math.dist((ax, ay), (bx, by))
        print(f"   separation {gap:.0f} m. The plan states the two exits terminate "
              f"within 200 metres of each other.")
    print()

    assignment, unresolved = defaultdict(lambda: {"edges": [], "households": 0}), []
    rows = []
    for edge_id in area["edges"]:
        road, dist, path = egress_road(edge_id, edges, named, out_of)
        if road in (None, "other"):
            unresolved.append(edge_id)
        bucket = assignment[str(road)]
        bucket["edges"].append(edge_id)
        bucket["households"] += spawn_counts.get(edge_id, 0)
        rows.append((edge_id, named.get(edge_id, "?"), str(road), dist or 0.0,
                     spawn_counts.get(edge_id, 0)))

    print(f"{'household edge':16} {'street':20} {'evacuates by':20} {'m':>6} {'homes':>6}")
    for edge_id, street, road, dist, homes in rows:
        print(f"{edge_id:16} {street:20} {road:20} {dist:6.0f} {homes:6}")
    print()
    for road, bucket in sorted(assignment.items()):
        print(f"{road:22} {bucket['households']:3} households on {len(bucket['edges'])} edges")
    if unresolved:
        print(f"\nUNRESOLVED, no named egress road on the path out: {unresolved}")

    # Compare against the map-confirmed binding in the config, when it carries one.
    config_branch = {}
    for event in alerts.get("schedule") or []:
        for branch in event.get("routing_branches") or []:
            for edge_id in branch.get("edges") or ():
                config_branch[edge_id] = branch.get("label", "")
    disagree = []
    if config_branch:
        derived = {edge_id: str(road) for edge_id, _s, road, _d, _h in rows}
        disagree = [(e, derived[e], config_branch[e])
                    for e in sorted(config_branch)
                    if e in derived and derived[e] != config_branch[e]]
        print(f"\nAgainst the map-confirmed binding in the config, "
              f"{len(config_branch) - len(disagree)}/{len(config_branch)} edges agree.")
        for edge_id, got, want in disagree:
            homes = spawn_counts.get(edge_id, 0)
            print(f"   DISAGREES {edge_id:16} network says {got:20} map says {want:20} "
                  f"{homes} homes")
        if disagree:
            print("   The map is authoritative. A shortest path out is only a proxy for the "
                  "district rule.")

    if args.json:
        Path(args.json).write_text(json.dumps(dict(assignment), indent=2) + "\n")
        print(f"\nwrote {args.json}")
    return 1 if unresolved else 0


if __name__ == "__main__":
    raise SystemExit(main())
