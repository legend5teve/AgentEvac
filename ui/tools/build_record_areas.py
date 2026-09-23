"""Place a package's households in the community the 2023 alerts name.

The operator console lets an area be drawn by hand, which is the right tool for a
counterfactual. Reproducing the historical order boundaries that way is a different
problem, because the boundaries are not something an operator can see on a road map and
getting them wrong silently mistimes thousands of households.

This script does that placement once, offline. It reads the community boundaries in
``_communities.geojson``, tests every household in a package against them, and writes one
``ui/assets/record_areas/<package>.json`` holding the areas the console offers under
"Load record areas". Each area carries the alert that names it, that alert's issue time,
and a colour, so the console can fill the whole schedule in one action.

Run it after authoring a package whose households sit in the Halifax study area::

    python -m ui.tools.build_record_areas                     # every package with a bundle
    python -m ui.tools.build_record_areas thousand_agent_test # one package

A community that no alert names still gets an area, marked unordered. Those households
were inside the study area and outside every broadcast, which is a real part of the record
and is what makes them a control group.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = REPO_ROOT / "ui" / "assets" / "record_areas"
PREVIEWS_DIR = REPO_ROOT / "ui" / "assets" / "previews"
COMMUNITIES_FILE = ASSETS_DIR / "_communities.geojson"
ALERT_ROADS_FILE = ASSETS_DIR / "_alert_roads.geojson"
SCHEDULE_FILE = ASSETS_DIR / "_schedule.json"

#: How close a household has to sit to a road the alerts name to count as on it. Lots on
#: these roads are deep, so this is the distance from the centroid of a house to the road
#: it fronts, and 120 m keeps the second row of a subdivision behind it out.
ROAD_MATCH_M = 120.0


def _ring_area(ring: Sequence[Sequence[float]]) -> float:
    """Twice the signed area of a ring, used only to rank polygons by size."""
    total = 0.0
    for i in range(len(ring) - 1):
        total += ring[i][0] * ring[i + 1][1] - ring[i + 1][0] * ring[i][1]
    return abs(total) / 2.0


class CommunityIndex:
    """Point-in-polygon lookup over the community boundaries, smallest match winning.

    OpenStreetMap nests these, so Haliburton Hills sits inside Stillwater Lake and
    Yankeetown sits inside Hammonds Plains. The alerts name the innermost one they know
    about, so the smallest polygon containing a household is the community whose order
    applies to it.
    """

    def __init__(self, geojson: Dict[str, Any]):
        self._polys: List[Tuple[float, str, List[List[float]], Tuple[float, float, float, float]]] = []
        for feature in geojson.get("features", []):
            name = str((feature.get("properties") or {}).get("name") or "")
            geometry = feature.get("geometry") or {}
            if not name or geometry.get("type") != "Polygon":
                continue
            for ring in geometry.get("coordinates", []):
                if len(ring) < 4:
                    continue
                lons = [p[0] for p in ring]
                lats = [p[1] for p in ring]
                self._polys.append(
                    (_ring_area(ring), name, ring, (min(lons), min(lats), max(lons), max(lats)))
                )
        self._polys.sort(key=lambda entry: entry[0])

    @classmethod
    def load(cls, path: Path = COMMUNITIES_FILE) -> "CommunityIndex":
        with open(path, encoding="utf-8") as handle:
            return cls(json.load(handle))

    @staticmethod
    def _inside(lon: float, lat: float, ring: Sequence[Sequence[float]]) -> bool:
        """Ray casting, counting crossings of the ring to the left of the point."""
        inside = False
        for i in range(len(ring) - 1):
            x1, y1 = ring[i][0], ring[i][1]
            x2, y2 = ring[i + 1][0], ring[i + 1][1]
            if (y1 > lat) != (y2 > lat):
                crossing = (x2 - x1) * (lat - y1) / (y2 - y1) + x1
                if lon < crossing:
                    inside = not inside
        return inside

    def community_of(self, lon: float, lat: float) -> Optional[str]:
        """Name of the smallest community containing the point, or None."""
        for _area, name, ring, (min_lon, min_lat, max_lon, max_lat) in self._polys:
            if not (min_lon <= lon <= max_lon and min_lat <= lat <= max_lat):
                continue
            if self._inside(lon, lat, ring):
                return name
        return None


class AlertRoadIndex:
    """Nearest road that an alert names, for a point.

    The broadcasts name roads as well as subdivisions. EA-3 extended the order to
    "Pockwock Rd" and "Lucasville Rd", and a household fronting one of those is inside the
    order whichever community polygon happens to contain it. Placing by polygon alone
    missed 173 households on Pockwock Road in the thousand-agent selection, every one of
    them inside the area EA-3 named.

    Distance is measured to the road polyline in a local metric frame, which is accurate
    far below the metre that matters over the twenty-odd kilometres a package spans.
    """

    def __init__(self, geojson: Dict[str, Any], lat0: float = 44.72):
        self._m_per_lon = 111320.0 * math.cos(math.radians(lat0))
        self._m_per_lat = 111132.0
        self._cell_m = 500.0
        self._grid: Dict[Tuple[int, int], List[Tuple[str, str, float, float, float, float]]] = {}
        for feature in geojson.get("features", []):
            props = feature.get("properties") or {}
            name, wave = str(props.get("name") or ""), str(props.get("wave") or "")
            line = (feature.get("geometry") or {}).get("coordinates") or []
            if not name or not wave or len(line) < 2:
                continue
            for start, end in zip(line, line[1:]):
                x1, y1 = start[0] * self._m_per_lon, start[1] * self._m_per_lat
                x2, y2 = end[0] * self._m_per_lon, end[1] * self._m_per_lat
                seg = (name, wave, x1, y1, x2, y2)
                for cx in range(int(min(x1, x2) // self._cell_m), int(max(x1, x2) // self._cell_m) + 1):
                    for cy in range(int(min(y1, y2) // self._cell_m), int(max(y1, y2) // self._cell_m) + 1):
                        self._grid.setdefault((cx, cy), []).append(seg)

    @classmethod
    def load(cls, path: Path = ALERT_ROADS_FILE) -> "AlertRoadIndex":
        if not path.exists():
            return cls({"features": []})
        with open(path, encoding="utf-8") as handle:
            return cls(json.load(handle))

    @staticmethod
    def _distance(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
        dx, dy = x2 - x1, y2 - y1
        length = dx * dx + dy * dy
        t = 0.0 if length == 0 else max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / length))
        return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))

    def road_of(
        self, lon: float, lat: float, max_distance_m: float = ROAD_MATCH_M
    ) -> Optional[Tuple[str, str]]:
        """Return ``(road name, wave)`` for a point beside a named road, or None."""
        px, py = lon * self._m_per_lon, lat * self._m_per_lat
        cx, cy = int(px // self._cell_m), int(py // self._cell_m)
        best: Optional[Tuple[str, str]] = None
        best_distance = max_distance_m
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                for name, wave, x1, y1, x2, y2 in self._grid.get((cx + i, cy + j), ()):
                    distance = self._distance(px, py, x1, y1, x2, y2)
                    if distance < best_distance:
                        best_distance = distance
                        best = (name, wave)
        return best


def _slug(name: str) -> str:
    """An area name the package format and the resolver both accept."""
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in name)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def _load_schedule(path: Path = SCHEDULE_FILE) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _household_rows(package_id: str) -> List[Dict[str, Any]]:
    """Every household in a package, joined to the geometry its bundle carries.

    ``spawns.json`` records simulation coordinates and the console draws in longitude and
    latitude, so the join runs through ``building_id``, which both sides record.
    """
    bundle_path = PREVIEWS_DIR / f"{package_id}.json"
    spawns_path = REPO_ROOT / "configs" / package_id / "spawns.json"
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"{package_id} has no map bundle. Run python -m ui.tools.build_map_assets {package_id}"
        )
    if not spawns_path.exists():
        raise FileNotFoundError(f"{package_id} has no spawns.json")

    with open(bundle_path, encoding="utf-8") as handle:
        buildings = {str(b["id"]): b for b in json.load(handle).get("buildings", []) if b.get("id")}
    with open(spawns_path, encoding="utf-8") as handle:
        spawns = json.load(handle)
    if not isinstance(spawns, dict) or "groups" not in spawns:
        raise ValueError(f"{package_id} does not use the compact spawn format, so it has no building ids")

    rows: List[Dict[str, Any]] = []
    for group in spawns["groups"]:
        ids = group.get("building_id") or []
        for i in range(int(group.get("count", 0))):
            if i >= len(ids):
                continue
            building = buildings.get(str(ids[i]))
            if not building:
                continue
            rows.append({
                "building_id": str(ids[i]),
                "edge": str(group["edge"]),
                "lon": float(building["lon"]),
                "lat": float(building["lat"]),
            })
    return rows


def build_package(package_id: str, *, out_dir: Path = ASSETS_DIR) -> Optional[Path]:
    """Write one package's record-area file. Returns the path, or None on failure."""
    started = time.monotonic()
    try:
        rows = _household_rows(package_id)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[areas] {package_id}: skipped, {exc}", flush=True)
        return None
    if not rows:
        print(f"[areas] {package_id}: skipped, no household carries a building id", flush=True)
        return None

    index = CommunityIndex.load()
    roads = AlertRoadIndex.load()
    schedule = _load_schedule()

    # Which alert names each community, and what that alert says.
    wave_of: Dict[str, Dict[str, Any]] = {}
    for event in schedule["schedule"]:
        for community in event["communities"]:
            wave_of[community] = event
    by_id = {event["id"]: event for event in schedule["schedule"]}
    door = schedule.get("door_to_door") or {}
    door_communities = set(door.get("communities") or [])

    # A household is grouped by the thing the record names it under.  A road an alert
    # names wins over the community polygon, because the broadcast named the road, and
    # placing by polygon alone leaves those households outside an order they were inside.
    members: Dict[str, List[str]] = {}
    edges: Dict[str, set] = {}
    kind: Dict[str, Tuple[str, Optional[str]]] = {}
    for row in rows:
        community = index.community_of(row["lon"], row["lat"])
        road = roads.road_of(row["lon"], row["lat"])
        if road is not None and wave_of.get(community or "") is None:
            group = road[0]
            kind[group] = ("road", road[1])
        else:
            group = community or "(outside every community)"
            kind[group] = ("community", None)
        members.setdefault(group, []).append(row["building_id"])
        edges.setdefault(group, set()).add(row["edge"])

    spare = list(schedule.get("unordered_palette") or ["#7F8C99"])
    areas: List[Dict[str, Any]] = []
    unordered_seen = 0
    for group in sorted(members, key=lambda c: (-len(members[c]), c)):
        source, wave_id = kind[group]
        event = by_id.get(wave_id) if source == "road" else wave_of.get(group)
        area: Dict[str, Any] = {
            "name": _slug(group),
            "label": group,
            "community": group,
            "placed_by": source,
            "building_ids": members[group],
            "agents": len(members[group]),
            "edges": len(edges[group]),
            "ordered": event is not None,
        }
        if event is not None:
            area.update({
                "wave": event["id"],
                "issue_time_s": event["issue_time_s"],
                "wall_clock": event["wall_clock"],
                "color": event["color"],
                "instruction": event["instruction"],
                "channel": event["channel"],
                "hazard_text": event["hazard_text"],
                "comfort_centre": event.get("comfort_centre"),
                "routing_text": event.get("routing_text"),
            })
            if source == "road":
                area["note"] = f"{group} is named in {event['id']} directly."
        else:
            # Still an area, so the console can show it and an operator can give it an
            # order.  It carries no time, so it writes no broadcast until one is chosen.
            area.update({
                "wave": None,
                "issue_time_s": None,
                "color": spare[unordered_seen % len(spare)],
                "instruction": "evacuate_now",
                "channel": "wireless_emergency_alert",
                "hazard_text": "",
                "comfort_centre": None,
                "note": "No broadcast on 28 May named this community.",
            })
            unordered_seen += 1
        if group in door_communities:
            area["door_sweep"] = {
                "begin_s": door.get("start_time_s"),
                "cleared_by_s": door.get("cleared_by_s"),
            }
        areas.append(area)

    payload = {
        "package": package_id,
        "generated_wall": time.time(),
        "households": len(rows),
        "source": schedule["_meta"]["sources"],
        "note": schedule["_meta"]["note"],
        "areas": areas,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{package_id}.json"
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))

    ordered = sum(a["agents"] for a in areas if a["ordered"])
    print(
        f"[areas] {package_id}: {len(areas)} communities, {ordered} of {len(rows)} households "
        f"ordered, {target.stat().st_size / 1000:.0f} kB in {time.monotonic() - started:.1f} s",
        flush=True,
    )
    for area in areas:
        wave = area["wave"] or "unordered"
        print(f"           {area['label'][:32]:32s} {area['agents']:5d} agents  {wave}", flush=True)
    return target


def discover_packages() -> List[str]:
    if not PREVIEWS_DIR.is_dir():
        return []
    return sorted(p.stem for p in PREVIEWS_DIR.glob("*.json"))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ui.tools.build_record_areas",
        description="Place a package's households in the community the 2023 alerts name.",
    )
    parser.add_argument("packages", nargs="*", help="Package ids. Defaults to every package with a bundle.")
    parser.add_argument("--out-dir", default=str(ASSETS_DIR))
    args = parser.parse_args(argv)

    targets = args.packages or discover_packages()
    if not targets:
        print("no package has a map bundle, so there is nothing to place")
        return 1

    built = 0
    for package_id in targets:
        if build_package(package_id, out_dir=Path(args.out_dir)) is not None:
            built += 1
    print(f"[areas] wrote {built} record-area file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
