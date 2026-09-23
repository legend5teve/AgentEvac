"""Timed, area-scoped alert-schedule resolver for the M1 alert-event engine.

AgentEvac's legacy scenarios treated hazard visibility, an evacuation directive, and
route guidance as one run-global mode.  The real 2023 Halifax orders were bundled
evacuation orders, issued at specific times and extended twice as the fire spread, plus
an RCMP door-to-door channel.  This module turns that into data-driven state.

The schedule is loaded from a per-map ``alerts.json`` (see ``configs/halifax_3town_e0/``).
This resolver is deliberately pure, meaning no ``traci`` and no RNG, so it is fully
unit-testable and does not perturb replay determinism.

Core objects:
    * ``AlertEvent``     -- one scheduled broadcast alert.
    * ``AlertState``     -- the resolved situation for one agent at one instant.
    * ``AlertSchedule``  -- the loaded schedule, answering per-edge, per-time queries.
    * ``effective_mode`` -- adapter mapping an ``AlertState`` onto the legacy mode string
                            (``no_notice`` / ``alert_guided`` / ``advice_guided``) that the
                            existing ``scenarios.py`` filters already implement.

Membership is an edge test.  An agent belongs to an area if its spawn edge is in that
area's ``edges`` list, which is exact for the pre-departure phase where every agent sits
on its spawn edge.  Orders are cumulative, so once an area is ordered it stays ordered.
"""

from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple


# Instruction strength ordering.  When several covering orders overlap an agent's area,
# the strongest instruction wins.
_INSTRUCTION_PRIORITY: Dict[str, int] = {
    "none": 0,
    "shelter_in_place": 1,
    "evacuate_now": 2,
}


@dataclass(frozen=True)
class AlertEvent:
    """One scheduled broadcast alert, parsed from the ``alerts.json`` schedule.

    Attributes:
        id: Event identifier, for example ``"EA-1"``.
        issue_time_s: Seconds after ignition when the alert is broadcast.
        areas: Newly ordered areas, cumulative across events.
        instruction: ``none`` | ``evacuate_now`` | ``shelter_in_place``.
        hazard_text: Near-verbatim bundled hazard and directive text.
        routing_text: Route guidance for the whole area, ``None`` for the historical E0
            orders.  A household covered by a routing branch gets the branch text instead.
        routing_branches: Per-household route guidance, as
            ``(label, edges, text, comfort_centre)`` tuples.
            The Westwood Hills plan splits a subdivision at its two exits and gives each
            side a different instruction, so guidance is not always uniform across an area.
            Empty for an order whose guidance is the same for everyone.
        comfort_centre: Named destination, present only when the order named one.
        channel: Delivery channel, for example ``"wireless_emergency_alert"``.
    """

    id: str
    issue_time_s: float
    areas: Tuple[str, ...]
    instruction: str
    hazard_text: str
    routing_text: Optional[str]
    comfort_centre: Optional[str]
    channel: str
    routing_branches: Tuple[Tuple[str, FrozenSet[str], str, Optional[str]], ...] = ()

    def routing_for_edge(
        self, edge_id: str
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Return ``(guidance_text, branch_label, comfort_centre)`` for ``edge_id``.

        A branch covering the edge wins, so each side of a split subdivision hears only its
        own instruction and, where the plan sends the two sides to different reception
        centres, only its own destination.  Otherwise the area-wide ``routing_text`` and the
        order's own ``comfort_centre`` apply, with no label.
        """
        for label, edges, text, centre in self.routing_branches:
            if edge_id in edges:
                return text, label, centre or self.comfort_centre
        return self.routing_text, None, self.comfort_centre


@dataclass(frozen=True)
class AlertState:
    """The resolved alert situation for one agent at one instant.

    Attributes:
        received: True once any covering order has been issued by ``sim_t``.
        received_t_s: Earliest covering issue time, the alert-channel awareness instant.
        instruction: Strongest active instruction across covering orders.
        hazard_visible: True once any covering alert has been received.
        routing_visible: True if any covering order carried route guidance.
        order_text: Bundled order block for the C.7 belief channel and metrics, or
            ``None`` when no departure order is active.
    """

    received: bool
    received_t_s: Optional[float]
    instruction: str
    hazard_visible: bool
    routing_visible: bool
    order_text: Optional[Dict[str, Any]]


# The degenerate state, returned before any covering order and by ``empty()`` schedules.
NO_ALERT = AlertState(
    received=False,
    received_t_s=None,
    instruction="none",
    hazard_visible=False,
    routing_visible=False,
    order_text=None,
)


def effective_mode(state: AlertState) -> str:
    """Map a resolved ``AlertState`` onto the legacy scenario mode its filters implement.

    The directive bit is carried separately by ``state.order_text``, so the historical
    hazard-plus-directive-without-routing cell is produced by the ``alert_guided`` filter
    view together with an active order block.

    Args:
        state: The resolved alert state for an agent.

    Returns:
        One of ``"no_notice"``, ``"alert_guided"``, or ``"advice_guided"``.
    """
    if not state.received:
        return "no_notice"
    if state.routing_visible:
        return "advice_guided"
    return "alert_guided"


def route_advisory(state: AlertState) -> Optional[Dict[str, Any]]:
    """Return the official route-guidance block reaching an agent, or ``None``.

    E0 and E1 leave ``routing_text`` null on every event, so this is ``None`` there and no
    routing block is added to any prompt.  The E2 arm populates it, which is also what
    makes :func:`effective_mode` resolve to ``advice_guided``.

    The text comes from the operative order, meaning the strongest covering instruction
    with ties broken to the earliest issue.  Under the cumulative area rule a later event
    that re-covers an already ordered area does not displace the earlier order, so a
    household keeps the guidance of the alert that first covered it.

    Args:
        state: The resolved alert state for an agent.

    Returns:
        A dict with the guidance text and its provenance, or ``None`` when the agent has
        no order carrying route guidance.
    """
    order_text = state.order_text or {}
    guidance = order_text.get("routing_text")
    if not guidance:
        return None
    advisory = {
        "source": "official_alert",
        "alert_id": order_text.get("id"),
        "channel": order_text.get("channel"),
        "received_t_s": order_text.get("received_t_s"),
        "guidance": str(guidance),
        "comfort_centre": order_text.get("comfort_centre"),
    }
    branch = order_text.get("routing_branch")
    if branch:
        # Which side of a split subdivision this household is on, so the guidance reads as
        # addressed to it and the realised choice can be scored against the branch.
        advisory["applies_to"] = branch
    return advisory


def route_advisory_policy(
    advisory: Optional[Dict[str, Any]], tone: str, unit: str
) -> str:
    """Build the prompt clause that points an agent at its official route guidance.

    Empty when no guidance reaches the agent, so E0, E1 and every legacy run keep their
    policy text unchanged.

    Args:
        advisory: The block from :func:`route_advisory`, or ``None``.
        tone: ``"directive"`` or ``"neutral"``, following the arm.
        unit: ``"option"`` or ``"route"``, matching the menu the decision is made over.

    Returns:
        The clause to append to the guidance policy, or an empty string.
    """
    if not advisory:
        return ""
    lead = (
        "official_route_advisory carries the route guidance the emergency alert broadcast "
        "for your area. "
    )
    if tone == "neutral":
        return lead + f"Weigh it alongside the visible {unit} facts when you choose. "
    return lead + f"Follow it unless a visible {unit} fact makes it unsafe. "


class AlertSchedule:
    """The loaded alert schedule, answering per-edge and per-time queries.

    Build with :meth:`from_config` from a parsed ``alerts.json`` dict, or with
    :meth:`empty` for the ``no_notice`` and E3-degenerate arms that carry no schedule.
    """

    def __init__(
        self,
        events: List[AlertEvent],
        area_edges: Dict[str, List[str]],
        door_sweeps: List[Tuple[str, float, float]],
    ) -> None:
        # Events sorted by issue time so the earliest covering order resolves first.
        self._events: List[AlertEvent] = sorted(events, key=lambda e: e.issue_time_s)
        # Ordered edge lists (for door-sweep indexing) and sets (for O(1) membership).
        self._area_edges_list: Dict[str, List[str]] = {
            area: list(edges) for area, edges in area_edges.items()
        }
        self._area_edges: Dict[str, Set[str]] = {
            area: set(edges) for area, edges in area_edges.items()
        }
        self._door_sweeps: List[Tuple[str, float, float]] = list(door_sweeps)
        # Reverse index edge -> {area} built once for O(1) lookups.
        reverse: Dict[str, Set[str]] = {}
        for area, edges in self._area_edges.items():
            for edge in edges:
                reverse.setdefault(edge, set()).add(area)
        self._edge_to_areas: Dict[str, Set[str]] = reverse

    @classmethod
    def empty(cls) -> "AlertSchedule":
        """Return a schedule whose every resolve is ``NO_ALERT``.

        Backs the ``no_notice`` regime and the E3-degenerate ablation on the same path.
        """
        return cls([], {}, [])

    @classmethod
    def from_config(
        cls,
        cfg: Optional[Dict[str, Any]],
        *,
        time_offset_s: float = 0.0,
        door_sweep_scale: float = 1.0,
    ) -> "AlertSchedule":
        """Parse an ``alerts.json`` dict into an ``AlertSchedule``.

        Args:
            cfg: Parsed ``alerts.json`` content, or a falsy value for an empty schedule.
            time_offset_s: Global shift applied to every broadcast ``issue_time_s`` for
                the E1 timing arm.  The door-to-door channel is a fixed historical
                channel and is not shifted.
            door_sweep_scale: Multiplier on each door sweep's duration, meaning the
                ``[begin_s, cleared_by_s]`` window width, an M3 sensitivity knob.  The
                sweep start is fixed and only the clear-by time moves.

        Returns:
            A populated ``AlertSchedule``, or an empty one when ``cfg`` is falsy.
        """
        if not cfg:
            return cls.empty()

        areas_cfg = cfg.get("areas") or {}
        area_edges = {
            name: list(spec.get("edges") or [])
            for name, spec in areas_cfg.items()
        }

        events: List[AlertEvent] = []
        for ev in cfg.get("schedule") or []:
            events.append(
                AlertEvent(
                    id=str(ev.get("id", "")),
                    issue_time_s=float(ev.get("issue_time_s", 0.0)) + float(time_offset_s),
                    areas=tuple(ev.get("areas") or ()),
                    instruction=str(ev.get("instruction", "none")),
                    hazard_text=str(ev.get("hazard_text", "")),
                    routing_text=ev.get("routing_text"),
                    comfort_centre=ev.get("comfort_centre"),
                    channel=str(ev.get("channel", "")),
                    routing_branches=tuple(
                        (
                            str(br.get("label", br.get("id", ""))),
                            frozenset(str(e) for e in (br.get("edges") or ())),
                            str(br.get("text", "")),
                            br.get("comfort_centre"),
                        )
                        for br in (ev.get("routing_branches") or ())
                        if br.get("text")
                    ),
                )
            )

        door_sweeps: List[Tuple[str, float, float]] = []
        door = cfg.get("door_to_door") or {}
        default_start = float(door.get("start_time_s", 0.0))
        for sweep in door.get("sweep") or []:
            begin_s = float(sweep.get("begin_s", default_start))
            cleared_by_s = float(sweep.get("cleared_by_s", begin_s))
            cleared_by_s = begin_s + (cleared_by_s - begin_s) * float(door_sweep_scale)
            door_sweeps.append(
                (
                    str(sweep.get("area", "")),
                    begin_s,
                    cleared_by_s,
                )
            )

        return cls(events, area_edges, door_sweeps)

    def areas_for_edge(self, edge_id: str) -> Set[str]:
        """Return the set of area names whose edge list contains ``edge_id``."""
        return set(self._edge_to_areas.get(edge_id, set()))

    def active_for_edge(self, sim_t_s: float, edge_id: str) -> AlertState:
        """Resolve the alert state for an agent on ``edge_id`` at time ``sim_t_s``.

        Collects every event already issued whose areas intersect the edge's areas, takes
        the earliest as the receipt instant, the strongest instruction as the active one,
        and ORs route visibility so an earlier order stays in effect when a later one
        extends the area.

        Args:
            sim_t_s: Current simulation time in seconds.
            edge_id: The agent's spawn edge.

        Returns:
            The resolved ``AlertState``, or ``NO_ALERT`` when no covering order is active.
        """
        edge_areas = self._edge_to_areas.get(edge_id)
        if not edge_areas:
            return NO_ALERT

        covering = [
            e for e in self._events
            if e.issue_time_s <= sim_t_s and (set(e.areas) & edge_areas)
        ]
        if not covering:
            return NO_ALERT

        received_t_s = min(e.issue_time_s for e in covering)
        # Strongest instruction wins; ties break to the earliest issuing event.
        operative = max(
            covering,
            key=lambda e: (_INSTRUCTION_PRIORITY.get(e.instruction, 0), -e.issue_time_s),
        )
        instruction = operative.instruction
        # Guidance resolves per household, since a split subdivision gives each side its
        # own instruction, so visibility is asked of this edge and not of the area.
        routing_visible = any(bool(e.routing_for_edge(edge_id)[0]) for e in covering)

        order_text: Optional[Dict[str, Any]] = None
        if _INSTRUCTION_PRIORITY.get(instruction, 0) > 0:
            # A departure-driving order is active; bundle it for the belief channel.
            guidance, branch, centre = operative.routing_for_edge(edge_id)
            order_text = {
                "id": operative.id,
                "instruction": instruction,
                "hazard_text": operative.hazard_text,
                "comfort_centre": centre,
                "routing_text": guidance,
                "routing_branch": branch,
                "channel": operative.channel,
                "received_t_s": received_t_s,
            }

        return AlertState(
            received=True,
            received_t_s=received_t_s,
            instruction=instruction,
            hazard_visible=True,
            routing_visible=bool(routing_visible),
            order_text=order_text,
        )

    def door_knock_time(self, edge_id: str) -> Optional[float]:
        """Return the RCMP door-knock time for ``edge_id``, or ``None`` if never swept.

        Per-edge knock times are spread linearly across each swept area's
        ``[begin_s, cleared_by_s]`` window by the edge's order in that area's edge list.
        When an edge falls in several swept areas the earliest knock wins.

        Args:
            edge_id: The agent's spawn edge.

        Returns:
            The door-knock time in seconds, or ``None`` when the edge is never swept.
        """
        best: Optional[float] = None
        for area, begin_s, cleared_by_s in self._door_sweeps:
            edges = self._area_edges_list.get(area) or []
            if edge_id not in edges:
                continue
            count = len(edges)
            index = edges.index(edge_id)
            if count <= 1:
                knock = begin_s
            else:
                knock = begin_s + (cleared_by_s - begin_s) * (index / (count - 1))
            best = knock if best is None else min(best, knock)
        return best

    def scheduled_events(self) -> List[AlertEvent]:
        """Return the broadcast alert events, offset-applied and sorted by issue time.

        The per-run timeline export reads these so the rows carry the E1 time offset
        that ``from_config`` already applied, rather than the raw config times.
        """
        return list(self._events)

    def door_sweeps(self) -> List[Tuple[str, float, float]]:
        """Return the door sweeps as ``(area, begin_s, cleared_by_s)`` tuples."""
        return list(self._door_sweeps)

    def ordered_areas(self) -> Dict[str, Dict[str, Any]]:
        """Return the areas that receive an evacuate order, each with its edge list,
        the earliest order time, and the channel that warned it first.

        An area is ordered when a broadcast ``evacuate_now`` event covers it or an RCMP
        door sweep is scheduled for it.  When both cover an area the earlier one sets the
        channel, so a door sweep that precedes the broadcast tags the area ``"door"`` and
        an area reached only by broadcast is tagged ``"broadcast"``.  The channel is the
        historical first-warning channel for the area on the schedule.  It is a record
        label, not a claim about which channel drove any household's departure, and it is
        meaningful as an effectiveness contrast only after the channel weights are
        calibrated to the record.
        """
        out: Dict[str, Dict[str, Any]] = {}

        def _consider(area_name: str, order_t_s: float, channel: str) -> None:
            current = out.get(area_name)
            if current is None or order_t_s < current["order_t_s"]:
                out[area_name] = {"order_t_s": float(order_t_s), "channel": channel}

        evac = _INSTRUCTION_PRIORITY["evacuate_now"]
        for event in self._events:
            if _INSTRUCTION_PRIORITY.get(event.instruction, 0) >= evac:
                for area_name in event.areas:
                    _consider(area_name, event.issue_time_s, "broadcast")
        for area_name, begin_s, _cleared_by_s in self._door_sweeps:
            _consider(area_name, begin_s, "door")

        for area_name, spec in out.items():
            spec["edges"] = list(self._area_edges_list.get(area_name, []))
        return out
