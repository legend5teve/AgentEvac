#!/usr/bin/env python3
"""Regenerate a human-readable ``.dialogs.log`` transcript from a ``.dialogs.csv``.

During a simulation run only the machine-readable ``routes_<run_id>.dialogs.csv``
table is written (see ``agentevac/utils/replay.py``).  The human-readable
``.dialogs.log`` transcript carries no extra information, so it is produced on
demand by this optional script instead of on every run.

Usage:
    # Write routes_<id>.dialogs.log next to the CSV
    python scripts/generate_dialog_log.py path/to/routes_<id>.dialogs.csv

    # Choose an explicit output path (single input only)
    python scripts/generate_dialog_log.py routes_<id>.dialogs.csv -o transcript.log

    # Convert several CSVs at once (each writes its own .log alongside)
    python scripts/generate_dialog_log.py outputs/exp/*.dialogs.csv

The output format matches the transcript previously emitted inline by
``RouteReplay.record_llm_dialog``.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional

# LLM prompts routinely exceed the default 128 KiB CSV field cap, so raise it as
# high as the platform allows.
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)


def _default_log_path(csv_path: Path) -> Path:
    """Derive the ``.dialogs.log`` path from a ``.dialogs.csv`` path.

    ``routes_<id>.dialogs.csv`` becomes ``routes_<id>.dialogs.log``.  Any other
    input simply has its final suffix swapped to ``.log``.
    """
    name = csv_path.name
    if name.endswith(".dialogs.csv"):
        return csv_path.with_name(name[: -len(".csv")] + ".log")
    return csv_path.with_suffix(".log")


def _format_record(row: dict) -> str:
    """Render one CSV dialog row as a transcript block."""
    step = row.get("step", "")
    time_s = row.get("time_s", "")
    veh_id = row.get("veh_id", "")
    control_mode = row.get("control_mode", "")
    model = row.get("model", "")
    system_prompt = row.get("system_prompt", "") or ""
    user_prompt = row.get("user_prompt", "") or ""
    response_text = row.get("response_text", "") or ""
    parsed_json = row.get("parsed_json", "") or ""
    error = row.get("error", "") or ""

    try:
        time_str = f"{float(time_s):.2f}"
    except (TypeError, ValueError):
        time_str = str(time_s)

    lines = []
    lines.append("=" * 80 + "\n")
    lines.append(
        f"step={step} time_s={time_str} veh_id={veh_id} "
        f"mode={control_mode} model={model}\n"
    )
    lines.append("-" * 80 + "\n")
    lines.append("SYSTEM PROMPT:\n")
    lines.append(system_prompt.strip() + "\n\n")
    lines.append("USER PROMPT:\n")
    lines.append(user_prompt.strip() + "\n\n")
    lines.append("MODEL RESPONSE:\n")
    lines.append((response_text.strip() + "\n") if response_text else "<none>\n")
    lines.append("\nPARSED OUTPUT:\n")
    if parsed_json:
        try:
            parsed = json.loads(parsed_json)
            lines.append(json.dumps(parsed, ensure_ascii=False, indent=2) + "\n")
        except (json.JSONDecodeError, ValueError):
            # Fall back to the raw stored string if it is not valid JSON.
            lines.append(parsed_json.strip() + "\n")
    else:
        lines.append("<none>\n")
    if error:
        lines.append("\nERROR:\n")
        lines.append(error.strip() + "\n")
    lines.append("\n")
    return "".join(lines)


def generate_log(csv_path: Path, out_path: Optional[Path] = None) -> Path:
    """Convert one ``.dialogs.csv`` into a ``.dialogs.log`` transcript.

    Args:
        csv_path: Path to the machine-readable dialog CSV.
        out_path: Optional explicit output path.  Defaults to the CSV's
            ``.dialogs.log`` sibling.

    Returns:
        The path the transcript was written to.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Dialog CSV not found: {csv_path}")
    target = out_path or _default_log_path(csv_path)

    with csv_path.open("r", encoding="utf-8", newline="") as fin, \
            target.open("w", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        for row in reader:
            fout.write(_format_record(row))
    return target


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate a .dialogs.log transcript from a .dialogs.csv table.",
    )
    parser.add_argument(
        "csv_paths",
        nargs="+",
        help="One or more routes_<id>.dialogs.csv files.",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output .log path (only valid when a single CSV is given). "
             "Defaults to the CSV's .dialogs.log sibling.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.output is not None and len(args.csv_paths) != 1:
        print("[DIALOG_LOG] --output is only valid with a single input CSV.", file=sys.stderr)
        return 2

    out_override = Path(args.output) if args.output else None
    exit_code = 0
    for raw in args.csv_paths:
        csv_path = Path(raw)
        try:
            written = generate_log(csv_path, out_override)
            print(f"[DIALOG_LOG] {csv_path} -> {written}")
        except FileNotFoundError as exc:
            print(f"[DIALOG_LOG] skip: {exc}", file=sys.stderr)
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
