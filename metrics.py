import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


class RunMetricsCollector:
    def __init__(self, enabled: bool, base_path: str, run_mode: str):
        self.enabled = bool(enabled)
        self.run_mode = str(run_mode)
        self.path: Optional[str] = None
        if self.enabled:
            self.path = self._timestamped_path(base_path)
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)

        self._depart_times: Dict[str, float] = {}
        self._arrival_times: Dict[str, float] = {}
        self._last_seen_active: Set[str] = set()
        self._last_seen_time: Dict[str, float] = {}

        self._choice_counts: Dict[str, int] = {}
        self._decision_snapshot_count = 0
        self._decision_changes: Dict[str, int] = {}
        self._last_decision_state: Dict[str, str] = {}

        self._exposure_sum = 0.0
        self._exposure_count = 0
        self._exposure_by_agent_sum: Dict[str, float] = {}
        self._exposure_by_agent_count: Dict[str, int] = {}

    @staticmethod
    def _timestamped_path(base_path: str) -> str:
        base = Path(base_path)
        ext = base.suffix or ".json"
        stem = base.stem if base.suffix else base.name
        ts = time.strftime("%Y%m%d_%H%M%S")
        candidate = base.with_name(f"{stem}_{ts}{ext}")
        idx = 1
        while candidate.exists():
            candidate = base.with_name(f"{stem}_{ts}_{idx:02d}{ext}")
            idx += 1
        return str(candidate)

    def record_departure(self, agent_id: str, sim_t_s: float, reason: Optional[str] = None) -> None:
        if not self.enabled:
            return
        if agent_id not in self._depart_times:
            self._depart_times[agent_id] = float(sim_t_s)
        self._last_seen_time[agent_id] = float(sim_t_s)

    def observe_active_vehicles(self, active_vehicle_ids: List[str], sim_t_s: float) -> None:
        if not self.enabled:
            return

        now = float(sim_t_s)
        current = set(active_vehicle_ids)
        for vid in current:
            self._last_seen_time[vid] = now

        for vid in (self._last_seen_active - current):
            if vid in self._depart_times and vid not in self._arrival_times:
                self._arrival_times[vid] = now

        self._last_seen_active = current

    def record_decision_snapshot(
        self,
        agent_id: str,
        sim_t_s: float,
        decision_round: int,
        state: Dict[str, Any],
        choice_idx: Optional[int],
        action_status: str,
    ) -> None:
        if not self.enabled:
            return

        self._decision_snapshot_count += 1
        if choice_idx is None:
            return

        decision_state = f"{state.get('control_mode', 'unknown')}::{int(choice_idx)}"
        prev_state = self._last_decision_state.get(agent_id)
        if prev_state is not None and prev_state != decision_state:
            self._decision_changes[agent_id] = self._decision_changes.get(agent_id, 0) + 1
        self._last_decision_state[agent_id] = decision_state

        if int(choice_idx) < 0:
            return

        selected = state.get("selected_option") or {}
        choice_name = selected.get("name")
        if not choice_name:
            choice_name = f"choice_{int(choice_idx)}"
        label = f"{state.get('control_mode', 'unknown')}::{choice_name}"
        self._choice_counts[label] = self._choice_counts.get(label, 0) + 1

    def record_exposure_sample(
        self,
        agent_id: str,
        sim_t_s: float,
        current_edge: str,
        current_margin_m: Optional[float],
        risk_score: Optional[float] = None,
    ) -> None:
        if not self.enabled:
            return
        exposure = float(risk_score if risk_score is not None else 0.0)
        self._exposure_sum += exposure
        self._exposure_count += 1
        self._exposure_by_agent_sum[agent_id] = self._exposure_by_agent_sum.get(agent_id, 0.0) + exposure
        self._exposure_by_agent_count[agent_id] = self._exposure_by_agent_count.get(agent_id, 0) + 1
        self._last_seen_time[agent_id] = float(sim_t_s)

    def compute_departure_time_variability(self) -> float:
        times = list(self._depart_times.values())
        n = len(times)
        if n <= 1:
            return 0.0
        mean = sum(times) / float(n)
        return sum((t - mean) ** 2 for t in times) / float(n)

    def compute_route_choice_entropy(self) -> float:
        total = sum(self._choice_counts.values())
        if total <= 0:
            return 0.0
        entropy = 0.0
        for count in self._choice_counts.values():
            p = float(count) / float(total)
            if p > 0.0:
                entropy -= p * math.log(p)
        return entropy

    def compute_decision_instability(self) -> Dict[str, Any]:
        if not self._last_decision_state:
            return {
                "average_changes": 0.0,
                "max_changes": 0,
                "per_agent_changes": {},
            }
        per_agent = {
            agent_id: int(self._decision_changes.get(agent_id, 0))
            for agent_id in self._last_decision_state.keys()
        }
        counts = list(per_agent.values())
        return {
            "average_changes": (sum(counts) / float(len(counts))) if counts else 0.0,
            "max_changes": max(counts) if counts else 0,
            "per_agent_changes": per_agent,
        }

    def compute_average_hazard_exposure(self) -> Dict[str, Any]:
        global_avg = (self._exposure_sum / float(self._exposure_count)) if self._exposure_count > 0 else 0.0
        per_agent = {}
        for agent_id, total in self._exposure_by_agent_sum.items():
            cnt = self._exposure_by_agent_count.get(agent_id, 0)
            per_agent[agent_id] = (total / float(cnt)) if cnt > 0 else 0.0
        return {
            "global_average": global_avg,
            "sample_count": self._exposure_count,
            "per_agent_average": per_agent,
        }

    def compute_average_travel_time(self) -> Dict[str, Any]:
        durations = []
        per_agent = {}
        for agent_id, depart_t in self._depart_times.items():
            arrive_t = self._arrival_times.get(agent_id)
            if arrive_t is None:
                continue
            duration = max(0.0, float(arrive_t) - float(depart_t))
            per_agent[agent_id] = duration
            durations.append(duration)
        average = (sum(durations) / float(len(durations))) if durations else 0.0
        return {
            "average": average,
            "completed_agents": len(durations),
            "per_agent": per_agent,
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "run_mode": self.run_mode,
            "departed_agents": len(self._depart_times),
            "arrived_agents": len(self._arrival_times),
            "decision_snapshot_count": self._decision_snapshot_count,
            "departure_time_variability": round(self.compute_departure_time_variability(), 6),
            "route_choice_entropy": round(self.compute_route_choice_entropy(), 6),
            "decision_instability": self.compute_decision_instability(),
            "average_hazard_exposure": self.compute_average_hazard_exposure(),
            "average_travel_time": self.compute_average_travel_time(),
        }

    def export_run_metrics(self, path: Optional[str] = None) -> Optional[str]:
        if not self.enabled:
            return None
        target = path or self.path
        if not target:
            return None
        with open(target, "w", encoding="utf-8") as fh:
            json.dump(self.summary(), fh, ensure_ascii=False, indent=2, sort_keys=True)
            fh.write("\n")
        return target

    def close(self) -> Optional[str]:
        if not self.enabled:
            return None
        return self.export_run_metrics()
