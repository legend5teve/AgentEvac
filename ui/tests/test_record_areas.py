"""Placing a household in the community the 2023 alerts name.

The consequence of getting this wrong is silent.  A household put in the wrong community
is ordered at the wrong time and the run still completes, so the tests here pin the two
behaviours that decide it: which polygon wins when they nest, and which alert a community
is read off.
"""

import json

import pytest

from ui.tools import build_record_areas as record


def _square(lon, lat, half):
    """A closed ring centred on a point, in the winding the parser accepts."""
    return [
        [lon - half, lat - half],
        [lon + half, lat - half],
        [lon + half, lat + half],
        [lon - half, lat + half],
        [lon - half, lat - half],
    ]


def _index(*named):
    return record.CommunityIndex({
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature",
             "properties": {"name": name},
             "geometry": {"type": "Polygon", "coordinates": [ring]}}
            for name, ring in named
        ],
    })


class TestCommunityIndex:
    def test_a_point_inside_one_community_finds_it(self):
        index = _index(("Westwood Hills", _square(-63.87, 44.72, 0.01)))
        assert index.community_of(-63.87, 44.72) == "Westwood Hills"

    def test_a_point_outside_every_community_finds_none(self):
        index = _index(("Westwood Hills", _square(-63.87, 44.72, 0.01)))
        assert index.community_of(-63.50, 44.72) is None

    def test_the_smaller_of_two_nested_communities_wins(self):
        # This is the Haliburton Hills case.  OpenStreetMap nests the subdivision inside
        # the Stillwater Lake community, and EA-3 names the subdivision, so the inner
        # polygon is the one whose order applies.
        index = _index(
            ("Stillwater Lake", _square(-63.83, 44.70, 0.02)),
            ("Halliburton Subdivision", _square(-63.83, 44.70, 0.005)),
        )
        assert index.community_of(-63.83, 44.70) == "Halliburton Subdivision"
        # Still inside the parent, outside the subdivision.
        assert index.community_of(-63.815, 44.70) == "Stillwater Lake"

    def test_a_point_on_the_far_side_is_not_caught_by_the_bounding_box(self):
        # Two communities sharing a latitude band.  A test that stopped at the bounding
        # box would put this point in the first one it looked at.
        index = _index(
            ("West", _square(-63.90, 44.72, 0.01)),
            ("East", _square(-63.80, 44.72, 0.01)),
        )
        assert index.community_of(-63.80, 44.72) == "East"


def _road(name, wave, coords):
    return {"type": "Feature",
            "properties": {"name": name, "wave": wave},
            "geometry": {"type": "LineString", "coordinates": coords}}


class TestAlertRoadIndex:
    """EA-3 named Pockwock Rd outright, so a household fronting it is inside that order
    whichever community polygon happens to contain it. Placing by polygon alone left 173
    such households outside every area."""

    @pytest.fixture
    def index(self):
        return record.AlertRoadIndex({"features": [
            _road("Pockwock Road", "EA-3", [[-63.85, 44.75], [-63.84, 44.76]]),
            _road("Voyageur Way", "EA-4", [[-63.77, 44.72], [-63.76, 44.72]]),
        ]})

    def test_a_point_on_the_road_finds_it(self, index):
        assert index.road_of(-63.845, 44.755) == ("Pockwock Road", "EA-3")

    def test_a_point_far_from_every_road_finds_none(self, index):
        assert index.road_of(-63.80, 44.70) is None

    def test_the_match_radius_is_honoured(self, index):
        # About 300 m north of the Voyageur Way line, so outside the default 120 m.
        assert index.road_of(-63.765, 44.7227) is None
        assert index.road_of(-63.765, 44.7227, max_distance_m=400)[0] == "Voyageur Way"

    def test_the_nearer_of_two_roads_wins(self, index):
        assert index.road_of(-63.7605, 44.7201) == ("Voyageur Way", "EA-4")

    def test_an_absent_asset_matches_nothing(self, tmp_path):
        assert record.AlertRoadIndex.load(tmp_path / "nope.geojson").road_of(0, 0) is None


class TestShippedAlertRoads:
    def test_every_road_carries_a_wave_the_schedule_defines(self):
        with open(record.ALERT_ROADS_FILE, encoding="utf-8") as handle:
            geojson = json.load(handle)
        waves = {e["id"] for e in record._load_schedule()["schedule"]}
        assert geojson["features"], "no alert roads shipped"
        for feature in geojson["features"]:
            assert feature["properties"]["wave"] in waves

    def test_hammonds_plains_road_is_not_an_ordering_road(self):
        # It is the artery through the whole study area and no broadcast names it, so
        # ordering by it would sweep in communities the record never covered.
        with open(record.ALERT_ROADS_FILE, encoding="utf-8") as handle:
            names = {f["properties"]["name"] for f in json.load(handle)["features"]}
        assert "Hammonds Plains Road" not in names
        assert "Pockwock Road" in names


class TestSlug:
    @pytest.mark.parametrize("name,expected", [
        ("Westwood Hills", "westwood_hills"),
        ("Halliburton Subdivision", "halliburton_subdivision"),
        ("Wallace Hill 14A", "wallace_hill_14a"),
        ("St. Margarets Village", "st_margarets_village"),
    ])
    def test_names_become_area_ids(self, name, expected):
        assert record._slug(name) == expected


class TestShippedSchedule:
    """The record schedule is data, so the checks that matter are on the file itself."""

    @pytest.fixture(scope="class")
    def schedule(self):
        return record._load_schedule()

    def test_the_four_broadcasts_are_present_in_order(self, schedule):
        rows = schedule["schedule"]
        assert [e["id"] for e in rows] == ["EA-1", "EA-2", "EA-3", "EA-4"]
        times = [e["issue_time_s"] for e in rows]
        assert times == sorted(times)
        assert times == [6300, 9660, 15180, 17460]

    def test_every_broadcast_names_at_least_one_community(self, schedule):
        for event in schedule["schedule"]:
            assert event["communities"], f"{event['id']} names no community"

    def test_no_community_is_ordered_by_two_broadcasts(self, schedule):
        # Orders are cumulative, so a community named twice would be ordered by the
        # earlier one and the later naming would be dead weight in the schedule.
        seen = set()
        for event in schedule["schedule"]:
            for community in event["communities"]:
                assert community not in seen, f"{community} is ordered twice"
                seen.add(community)

    def test_stillwater_lake_is_ordered_by_no_broadcast(self, schedule):
        # It was named in the internal requests at 16:41 and 17:20 and in neither
        # broadcast.  Its households reach an order through Haliburton Hills at EA-3.
        ordered = {c for e in schedule["schedule"] for c in e["communities"]}
        assert "Stillwater Lake" not in ordered
        assert "Halliburton Subdivision" in ordered
        requested = {c for r in schedule["internal_requests"] for c in r["communities"]}
        assert "Stillwater Lake" in requested

    def test_every_wave_colour_is_distinct_and_is_not_the_fire_colour(self, schedule):
        colors = [e["color"] for e in schedule["schedule"]]
        assert len(set(colors)) == len(colors)
        # Vermilion belongs to the fire front, and blue to an unordered household.
        assert "#D55E00" not in colors
        assert "#0072B2" not in colors

    def test_the_run_has_to_reach_the_last_broadcast(self, schedule):
        clock = schedule["_meta"]["clock"]
        last = max(e["issue_time_s"] for e in schedule["schedule"])
        assert clock["required_sim_end_time_s"] >= last
        assert clock["recommended_sim_end_time_s"] >= clock["required_sim_end_time_s"]


class TestShippedCommunities:
    def test_every_community_the_schedule_names_has_a_boundary(self):
        with open(record.COMMUNITIES_FILE, encoding="utf-8") as handle:
            geojson = json.load(handle)
        have = {f["properties"]["name"] for f in geojson["features"]}
        schedule = record._load_schedule()
        named = {c for e in schedule["schedule"] for c in e["communities"]}
        named |= {c for r in schedule["internal_requests"] for c in r["communities"]}
        named |= set(schedule["door_to_door"]["communities"])
        assert named <= have, f"no boundary for {sorted(named - have)}"
