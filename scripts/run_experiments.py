#!/usr/bin/env python3
"""Driver for the E0 to E5 experiment grid on the ``halifax_3town_e0`` config.

Runs every cell of the grid as a separate ``agentevac.simulation.main`` subprocess,
using the interpreter that runs this script (``sys.executable``), so launching it from
a PyCharm Python configuration with the project venv reuses that venv automatically.

PyCharm setup
    Script path        scripts/run_experiments.py
    Working directory   the repo root
    Interpreter         the project venv (has openai + traci)
    Environment         OPENAI_API_KEY=sk-...   (only for the llm agent)
                        SUMO_HOME defaults to /usr/share/sumo if unset

Examples
    python scripts/run_experiments.py --dry-run
    python scripts/run_experiments.py --arms e0 --agents rule_based      # free half first
    python scripts/run_experiments.py --arms e1,e4 --agents llm --skip-existing
    python scripts/run_experiments.py --arms e4early --agents llm   # trust sweep, early alert
    python scripts/run_experiments.py --arms e3 --agents rule_based  # ablation, no-notice + hazard-only
    python scripts/run_experiments.py --arms e2 --agents llm         # routing counterfactual
    python scripts/run_experiments.py --arms e5 --agents rule_based  # buffer alerting

Fixed knobs, matching docs/build_plan and the calibration
    map=halifax_3town_e0 (E2 and E3 override it with their own config dir),
    sim-end-time=28800, scenario=no_notice,
    FIRE_PERCEPTION_RANGE_M=200. E1 sweeps ALERT_TIME_OFFSET_S, E4 sweeps
    DEFAULT_THETA_AUTH, and E4early sweeps DEFAULT_THETA_AUTH at ALERT_TIME_OFFSET_S=-3600
    so the order precedes the fire. E2 adds route guidance and runs SCENARIO_TONE=neutral
    so tone is matched across the contrast. The counterfactual arms run messaging off,
    where the alert channel is visible. E0 runs messaging on and off.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SUMO_HOME = "/usr/share/sumo"
MAP_NAME = "halifax_3town_e0"

# --- grid axes ---
SEEDS_E0 = [47, 1024, 7] # [47, 1024, 7, 13, 91, 128, 256, 512, 777, 2024]  # 10 seeds
SEEDS_CF = SEEDS_E0[:3]                                       # 3 seeds for E1/E4
ALL_AGENTS = ["llm", "rule_based"]
E1_OFFSETS = [-3600, -1800, -900, 900, 1800, 3600]           # capped at -3600 in code
E4_AUTH = [0.1, 0.3, 0.5, 0.7, 0.9]
E4_EARLY_OFFSET = -3600  # e4early sweeps theta_auth at this early alert offset, where the
                         # order precedes the fire so authority trust is the deciding input
E2_MAP = "halifax_3town_e0_routing"  # E0 schedule plus routing_text on every event
E5_MAP = "halifax_3town_e0_buffer"   # E0 schedule re-timed by the buffer policy
E3_CONFIGS = [  # E3 ablations use degenerate map configs that differ from E0 only in alerts
    ("nonotice", "halifax_3town_e0_nonotice"),      # (0,0,0), no alert schedule at all
    ("hazardonly", "halifax_3town_e0_hazardonly"),  # (1,0,0), forecast visible, no directive
]

FIXED_FLAGS = [
    "--sim-end-time", "28800",
    "--scenario", "no_notice", "--metrics", "on", "--events", "on",
]
PERCEPTION_RANGE = "200"

MANIFEST_HDR = ["arm", "subdir", "agent", "seed", "messaging", "offset_s", "theta_auth", "tone",
                "outdir", "status", "elapsed_s", "departed", "arrived", "total", "usable"]

# stdout lines worth keeping in each cell's run.log; the rest is per-step spam
KEEP_PREFIXES = (
    "[ALERTS]", "[CLOCK]", "[SCENARIO]", "[MESSAGING]", "[AGENT_TYPE]",
    "[M2]", "[METRICS]", "[SUMO]", "[SEED", "[CLI_FLAGS]",
)
ERR_RE = re.compile(r"Traceback|Error|Exception|Failed|CRITICAL")


@dataclass
class Cell:
    arm: str            # e0 | e1 | e4
    subdir: str         # e.g. e0_msgoff, e1_off-3600, e4_auth0.1
    agent: str          # llm | rule_based
    seed: int
    messaging: str      # on | off
    offset_s: int       # ALERT_TIME_OFFSET_S
    theta_auth: float   # DEFAULT_THETA_AUTH
    map_name: str = MAP_NAME  # E3, E2 and E5 override this with their own config dir
    tag: str = ""             # optional suffix on the output subdir, for re-runs
    tone: str = "directive"   # E2 runs neutral so tone is matched across the contrast

    @property
    def outdir(self) -> Path:
        # Grouped under an E0/E1/E4 family folder so outputs/ stays tidy. A tag keeps a
        # re-run beside the original instead of overwriting it, which the buffer arm needs
        # because its sizing iterates.
        sub = f"{self.subdir}__{self.tag}" if self.tag else self.subdir
        return REPO / "outputs" / self.arm.upper() / sub / f"{self.agent}_seed{self.seed}"

    @property
    def name(self) -> str:
        return f"{self.arm.upper()}/{self.subdir}/{self.agent}_seed{self.seed}"

    @property
    def usable(self) -> str:
        # E0 messaging-on is the unbounded-broadcast flood, marked unusable in the folder name.
        return "FALSE" if "UNUSABLE" in self.subdir else "TRUE"


def build_cells(arms, agents) -> list[Cell]:
    cells: list[Cell] = []
    if "e0" in arms:
        # Messaging only affects LLM agents. rule_based decides heuristically and never
        # composes outbox text, so messaging on is a byte-identical no-op there, verified
        # empirically. So LLM runs both on and off, rule_based runs once (off).
        for agent in agents:
            msgs = ("on", "off") if agent == "llm" else ("off",)
            for msg in msgs:
                # msg-on floods the map via unbounded broadcast, so it is marked unusable.
                sub = "e0_msgon__UNUSABLE" if msg == "on" else "e0_msgoff"
                for seed in SEEDS_E0:
                    cells.append(Cell("e0", sub, agent, seed, msg, 0, 0.5))
    if "e1" in arms:
        for off in E1_OFFSETS:
            for agent in agents:
                for seed in SEEDS_CF:
                    cells.append(Cell("e1", f"e1_off{off:+d}", agent, seed, "off", off, 0.5))
    if "e4" in arms:
        for auth in E4_AUTH:
            for agent in agents:
                for seed in SEEDS_CF:
                    cells.append(Cell("e4", f"e4_auth{auth}", agent, seed, "off", 0, auth))
    if "e4early" in arms:
        # E1 x E4 interaction. Sweep theta_auth at an early alert offset, where the order
        # precedes the fire and trust is the deciding input, unmasking the effect the
        # historical offset 0 hides because the fire arrives with the alert.
        for auth in E4_AUTH:
            for agent in agents:
                for seed in SEEDS_CF:
                    cells.append(Cell("e4early", f"e4early_off{E4_EARLY_OFFSET:+d}_auth{auth}",
                                      agent, seed, "off", E4_EARLY_OFFSET, auth))
    if "e2" in arms:
        # Content arm. The historical schedule plus route guidance, on a config that differs
        # from E0 only in routing_text, run tone-matched so the added guidance is not
        # confounded with directive exhortation. The rule_based mirror is a null control:
        # that policy never reads a prompt and the utility basis is identical outside
        # no_notice, so it should reproduce E0 exactly. A difference there means guidance
        # leaked into the non-prompt path.
        for agent in agents:
            for seed in SEEDS_CF:
                cells.append(Cell("e2", "e2_routing", agent, seed, "off", 0, 0.5,
                                  map_name=E2_MAP, tone="neutral"))
    if "e5" in arms:
        # Buffer alerting. Each community is warned when the fire margin falls to a buffer
        # sized so it can finish evacuating first, which is a principled per-community
        # re-timing where E1 applies one global offset to everybody. The config differs
        # from E0 only in issue_time_s, so the contrast is the alerting policy alone.
        for agent in agents:
            for seed in SEEDS_CF:
                cells.append(Cell("e5", "e5_buffer", agent, seed, "off", 0, 0.5,
                                  map_name=E5_MAP))
    if "e3" in arms:
        # Ablation. Each variant is a degenerate map config with no offset and the default
        # trust, differing from E0 only in the alert channel it exposes.
        for tag, mapname in E3_CONFIGS:
            for agent in agents:
                for seed in SEEDS_CF:
                    cells.append(Cell("e3", f"e3_{tag}", agent, seed, "off", 0, 0.5, map_name=mapname))
    return cells


def cell_cmd(cell: Cell, sumo_binary: str) -> list[str]:
    o = cell.outdir
    return [
        sys.executable, "-m", "agentevac.simulation.main",
        "--sumo-binary", sumo_binary, "--map", cell.map_name, *FIXED_FLAGS,
        "--agent-type", cell.agent, "--messaging", cell.messaging, "--seed", str(cell.seed),
        "--metrics-log-path", str(o / "run_metrics.json"),
        "--events-log-path", str(o / "events.jsonl"),
        "--params-log-path", str(o / "run_params.json"),
        "--replay-log-path", str(o / "llm_routes.jsonl"),
        "--timeline-log-path", str(o / "run_timeline.jsonl"),
    ]


def cell_env(cell: Cell) -> dict:
    env = os.environ.copy()
    env["SUMO_HOME"] = os.environ.get("SUMO_HOME", DEFAULT_SUMO_HOME)
    env["FIRE_PERCEPTION_RANGE_M"] = PERCEPTION_RANGE
    env["ALERT_TIME_OFFSET_S"] = str(cell.offset_s)
    env["DEFAULT_THETA_AUTH"] = str(cell.theta_auth)
    env["SCENARIO_TONE"] = cell.tone
    return env


def has_result(cell: Cell) -> bool:
    """True when the cell already holds a usable result.

    An llm cell that recorded no API calls does not count, so --skip-existing reruns it
    once the quota is restored instead of treating the fallback run as done.
    """
    if not [f for f in glob.glob(str(cell.outdir / "run_metrics_*.json")) if "profiles" not in f]:
        return False
    return not (cell.agent == "llm" and llm_calls_made(cell) == 0)


def llm_calls_made(cell: Cell):
    """Return the API call count an llm cell recorded, or None when unknown.

    An llm run whose every call fails, for example on an exhausted API quota, still
    completes and writes a full metrics file. Its agents fall back to a non-LLM path, so
    the cell looks finished and is not an llm result at all. Checking the recorded call
    count is the only way the driver can tell the difference.
    """
    files = [f for f in sorted(glob.glob(str(cell.outdir / "run_metrics_*.json")))
             if "profiles" not in f]
    if not files:
        return None
    try:
        return int((json.load(open(files[-1])).get("token_usage") or {}).get("llm_calls", 0))
    except Exception:
        return None


def read_result(cell: Cell):
    files = [f for f in sorted(glob.glob(str(cell.outdir / "run_metrics_*.json")))
             if "profiles" not in f]
    if not files:
        return None
    try:
        d = json.load(open(files[-1]))
        return d.get("departed_agents"), d.get("arrived_agents"), d.get("total_agents")
    except Exception:
        return None


def merge_manifest(manifest: Path, rows) -> None:
    """Upsert this run's rows into the manifest keyed by outdir, keeping other cells' rows.

    The driver used to overwrite the manifest each invocation, dropping cells from other
    runs. Merging by outdir means an llm batch and a rule_based batch accumulate into one
    complete record, and a re-run updates its own row in place.
    """
    existing: dict = {}
    if manifest.exists():
        with open(manifest, newline="") as f:
            for r in csv.DictReader(f):
                existing[r.get("outdir", "")] = r
    for c, status, dt, res in rows:
        dep, arr, tot = res if res else (None, None, None)
        key = str(c.outdir.relative_to(REPO))
        existing[key] = {
            "arm": c.arm, "subdir": c.subdir, "agent": c.agent, "seed": c.seed,
            "messaging": c.messaging, "offset_s": c.offset_s, "theta_auth": c.theta_auth,
            "tone": c.tone,
            "outdir": key, "status": status, "elapsed_s": f"{dt:.0f}",
            "departed": dep, "arrived": arr, "total": tot, "usable": c.usable,
        }
    with open(manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_HDR)
        w.writeheader()
        for key in sorted(existing):
            w.writerow({h: existing[key].get(h, "") for h in MANIFEST_HDR})


def run_cell(cell: Cell, sumo_binary: str):
    """Run one cell, streaming a filtered log. Returns (returncode, seconds, m2_line, tail)."""
    cell.outdir.mkdir(parents=True, exist_ok=True)
    cmd, env = cell_cmd(cell, sumo_binary), cell_env(cell)
    tail: deque[str] = deque(maxlen=60)
    m2 = None
    t0 = time.time()
    with open(cell.outdir / "run.log", "w") as log, subprocess.Popen(
        cmd, cwd=str(REPO), env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1,
    ) as p:
        for line in p.stdout:
            tail.append(line.rstrip("\n"))
            if line.startswith(KEEP_PREFIXES) or ERR_RE.search(line):
                log.write(line)
                if line.startswith("[M2]"):
                    m2 = line.strip()
        rc = p.wait()
    return rc, time.time() - t0, m2, tail


def api_probe(key: str):
    """Send the cheapest possible chat call. Returns None when it succeeds, else why not.

    Checks that the key can actually spend, which merely reading it cannot.
    """
    import urllib.error
    import urllib.request
    body = json.dumps({"model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                       "messages": [{"role": "user", "content": "ok"}],
                       "max_tokens": 1}).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions", data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            json.loads(resp.read())
        return None
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = json.loads(exc.read()).get("error", {}).get("message", "")
        except Exception:
            pass
        return f"HTTP {exc.code}. {detail[:160]}"
    except Exception as exc:
        return f"{type(exc).__name__}. {exc}"


def preflight(cells, dry_run) -> bool:
    ok = True
    # Every map the grid references, since E2 and E3 bring their own config dirs.
    for name in sorted({c.map_name for c in cells}):
        map_dir = REPO / "configs" / name
        if not map_dir.is_dir():
            print(f"  MISSING map config {map_dir}")
            ok = False
            continue
        fires = map_dir / "fires.json"
        if fires.is_file():
            d = json.load(open(fires))
            caps = sorted({s.get("max_r_m") for s in d.get("sources", [])})
            print(f"  config {name}: {len(d.get('sources', []))} fire sources, max_r_m={caps}")
        alerts = map_dir / "alerts.json"
        sched = json.load(open(alerts)).get("schedule", []) if alerts.is_file() else []
        n_routing = sum(1 for ev in sched if ev.get("routing_text"))
        print(f"  config {name}: {len(sched)} alert events, {n_routing} carry routing_text")
        # An E2 cell on a config with no routing_text degenerates into E0 and still writes a
        # full metrics file, so the arm has to be checked and not assumed.
        if name == E2_MAP and n_routing == 0:
            print(f"  E2 config {name} carries no routing_text, the arm would reproduce E0")
            ok = False
    if any(c.agent == "llm" for c in cells):
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            print("  OPENAI_API_KEY not set, the llm cells will fail. Set it or use --agents rule_based.")
            ok = ok and dry_run
        elif not dry_run:
            # A key that authenticates is not a key that can spend. An exhausted quota
            # answers every call with a 429, so without this probe a batch burns hours
            # producing runs whose agents never reached the model.
            reason = api_probe(key)
            if reason:
                print(f"  OPENAI_API_KEY ...{key[-4:]} cannot complete a call. {reason}")
                ok = False
            else:
                print(f"  OPENAI_API_KEY ...{key[-4:]} answered a one-token probe")
    if not os.environ.get("SUMO_HOME"):
        print(f"  SUMO_HOME not set, defaulting to {DEFAULT_SUMO_HOME}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the E0/E1/E2/E3/E4 experiment grid.")
    ap.add_argument("--arms", default="e0,e1,e4", help="Comma list from e0,e1,e2,e3,e4,e4early,e5 (default e0,e1,e4).")
    ap.add_argument("--agents", default="llm,rule_based", help="Comma list, llm and/or rule_based.")
    ap.add_argument("--sumo-binary", default="sumo", help="sumo or sumo-gui (default sumo).")
    ap.add_argument("--skip-existing", action="store_true", help="Skip cells that already have metrics.")
    ap.add_argument("--continue-on-error", action="store_true", help="Keep going past a failed cell.")
    ap.add_argument("--dry-run", action="store_true", help="Print the plan and exit without running.")
    ap.add_argument("--limit", type=int, default=0, help="Run at most N cells (0 = all). Useful for a test.")
    ap.add_argument("--tag", default="", help="Suffix the output subdir, so a re-run sits beside the original.")
    ap.add_argument("--tone", choices=("directive", "neutral"), default="",
                    help="Override every cell's prompt framing. Use with --tag to keep the runs apart.")
    ap.add_argument("--messaging", choices=("on", "off"), default="",
                    help="Keep only the cells with this messaging setting, so that an e0 rerun "
                         "can leave out the unusable messaging-on cells.")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    agents = [a.strip() for a in args.agents.split(",") if a.strip()]
    cells = build_cells(arms, agents)
    if args.tag:
        cells = [replace(c, tag=args.tag) for c in cells]
    if args.tone:
        cells = [replace(c, tone=args.tone) for c in cells]
    if args.messaging:
        cells = [c for c in cells if c.messaging == args.messaging]
    if args.limit:
        cells = cells[:args.limit]
    if not cells:
        print("No cells selected.")
        return 1

    n_llm = sum(1 for c in cells if c.agent == "llm")
    print(f"Grid: {len(cells)} cells  (arms={arms} agents={agents})")
    for arm in ("e0", "e1", "e2", "e3", "e4", "e4early", "e5"):
        k = sum(1 for c in cells if c.arm == arm)
        if k:
            print(f"  {arm}: {k} cells")
    print(f"  llm cells {n_llm} (budget API time), rule_based cells {len(cells) - n_llm} (free)")
    if not preflight(cells, args.dry_run):
        print("Preflight failed.")
        return 2

    if args.dry_run:
        print("\n-- dry run, sample cell --")
        c = cells[0]
        print("  env:", {k: cell_env(c)[k] for k in ("SUMO_HOME", "FIRE_PERCEPTION_RANGE_M",
                                                       "ALERT_TIME_OFFSET_S", "DEFAULT_THETA_AUTH",
                                                       "SCENARIO_TONE")})
        print("  cmd:", " ".join(cell_cmd(c, args.sumo_binary)))
        print("\n-- all cells --")
        for c in cells:
            print(f"  {c.name:44} msg={c.messaging} off={c.offset_s:+d} auth={c.theta_auth} "
                  f"tone={c.tone} map={c.map_name}")
        return 0

    manifest = REPO / "outputs" / "experiments_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    rows, failed = [], 0
    for i, c in enumerate(cells, 1):
        if args.skip_existing and has_result(c):
            print(f"[{i}/{len(cells)}] skip (exists) {c.name}")
            res = read_result(c)
            rows.append((c, "skipped", 0.0, res))
            continue
        print(f"[{i}/{len(cells)}] run  {c.name}  msg={c.messaging} off={c.offset_s:+d} auth={c.theta_auth}")
        rc, dt, m2, tail = run_cell(c, args.sumo_binary)
        res = read_result(c)
        status = "ok" if rc == 0 else f"FAIL(rc={rc})"
        calls = llm_calls_made(c) if c.agent == "llm" else None
        if status == "ok" and c.agent == "llm" and calls == 0:
            # The run completed and wrote a full metrics file while every API call failed,
            # so its agents fell back to a non-LLM path. Treat it as a failure, because it
            # is indistinguishable from a finished cell in every other respect.
            status = "FAIL(no llm calls)"
            rc = rc or 1
        extra = f"  {m2}" if m2 else ""
        depinfo = f"  departed/arrived={res[0]}/{res[1]}" if res else ""
        callinfo = f"  llm_calls={calls}" if calls is not None else ""
        print(f"      {status}  {dt/60:.1f} min{depinfo}{callinfo}{extra}")
        rows.append((c, status, dt, res))
        if rc != 0:
            failed += 1
            print("      --- last output lines ---")
            for ln in list(tail)[-15:]:
                print("      | " + ln)
            if not args.continue_on_error:
                print("      stopping (use --continue-on-error to keep going)")
                break

    merge_manifest(manifest, rows)

    done = sum(1 for _, s, _, _ in rows if s in ("ok", "skipped"))
    print(f"\nDone. {done}/{len(cells)} cells accounted for, {failed} failed. Manifest: {manifest.relative_to(REPO)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
