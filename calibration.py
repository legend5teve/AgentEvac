import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


METRIC_SPECS: Dict[str, Tuple[str, float]] = {
    "departure_time_variability": ("departure_time_variability", 1.0),
    "route_choice_entropy": ("route_choice_entropy", 0.5),
    "decision_instability_avg": ("decision_instability.average_changes", 1.0),
    "average_hazard_exposure": ("average_hazard_exposure.global_average", 0.25),
    "average_travel_time": ("average_travel_time.average", 60.0),
    "arrived_agents": ("arrived_agents", 1.0),
}


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _normalize_metrics_payload(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object for metrics payload.")
    if isinstance(payload.get("reference_metrics"), dict):
        return dict(payload["reference_metrics"])
    if isinstance(payload.get("metrics"), dict):
        return dict(payload["metrics"])
    if isinstance(payload.get("summary"), dict):
        return dict(payload["summary"])
    return dict(payload)


def _get_path_value(payload: Dict[str, Any], path: str) -> Optional[float]:
    node: Any = payload
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    if isinstance(node, (int, float)):
        return float(node)
    return None


def load_reference_scenario(path: str) -> Dict[str, Any]:
    return _normalize_metrics_payload(_read_json(path))


def load_run_metrics(path: str) -> Dict[str, Any]:
    return _normalize_metrics_payload(_read_json(path))


def score_run_against_reference(
    run_metrics: Dict[str, Any],
    reference: Dict[str, Any],
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    metric_details: Dict[str, Any] = {}
    total_loss = 0.0
    total_weight = 0.0

    for label, (metric_path, floor) in METRIC_SPECS.items():
        ref_value = _get_path_value(reference, metric_path)
        run_value = _get_path_value(run_metrics, metric_path)
        if ref_value is None or run_value is None:
            continue

        weight = float(weights.get(label, 1.0)) if weights else 1.0
        if weight <= 0.0:
            continue

        abs_error = abs(run_value - ref_value)
        norm_den = max(abs(ref_value), float(floor))
        rel_error = abs_error / norm_den if norm_den > 0.0 else abs_error
        weighted_loss = weight * rel_error

        metric_details[label] = {
            "path": metric_path,
            "reference": ref_value,
            "run": run_value,
            "absolute_error": abs_error,
            "relative_error": rel_error,
            "weight": weight,
            "weighted_loss": weighted_loss,
        }

        total_loss += weighted_loss
        total_weight += weight

    normalized_loss = (total_loss / total_weight) if total_weight > 0.0 else 0.0
    fit_score = 1.0 / (1.0 + normalized_loss)

    return {
        "fit_score": round(fit_score, 6),
        "normalized_loss": round(normalized_loss, 6),
        "metric_count": len(metric_details),
        "metric_details": metric_details,
    }


def fit_agent_parameters(
    search_space: Dict[str, Any],
    *,
    reference: Dict[str, Any],
    experiments_results: Optional[List[Dict[str, Any]]] = None,
    results_path: Optional[str] = None,
) -> Dict[str, Any]:
    candidates = experiments_results
    if candidates is None and results_path:
        raw = _read_json(results_path)
        if not isinstance(raw, list):
            raise ValueError("Experiment results JSON must be a list of case records.")
        candidates = raw
    if candidates is None:
        raise ValueError("Either experiments_results or results_path is required.")

    weights = search_space.get("weights")
    top_k = max(1, int(search_space.get("top_k", 5)))

    ranked: List[Dict[str, Any]] = []
    for row in candidates:
        if not isinstance(row, dict):
            continue
        metrics_path = row.get("metrics_path")
        if not metrics_path:
            continue
        metrics_file = Path(str(metrics_path))
        if not metrics_file.exists():
            continue

        run_metrics = load_run_metrics(str(metrics_file))
        score = score_run_against_reference(run_metrics, reference, weights=weights)
        ranked.append({
            "case_id": row.get("case_id"),
            "case": dict(row.get("case") or {}),
            "status": row.get("status"),
            "metrics_path": str(metrics_file),
            "score": score,
        })

    ranked.sort(
        key=lambda item: (
            float(item["score"].get("normalized_loss", 0.0)),
            -float(item["score"].get("fit_score", 0.0)),
        )
    )

    best = ranked[0] if ranked else None
    return {
        "candidate_count": len(ranked),
        "best_case": best,
        "ranked_cases": ranked[:top_k],
    }


def export_calibration_report(report: Dict[str, Any], output_path: str) -> str:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    return str(target)


def _parse_weights(raw: Optional[str]) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    if not raw:
        return weights
    for part in str(raw).split(","):
        item = part.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid weight assignment: {item}")
        key, value = item.split("=", 1)
        weights[key.strip()] = float(value.strip())
    return weights


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--reference", required=True, help="Reference metrics JSON.")
    parser.add_argument("--metrics", help="Single run metrics JSON to score.")
    parser.add_argument(
        "--results-json",
        help="Experiment results JSON from experiments.py for batch calibration.",
    )
    parser.add_argument(
        "--weights",
        help="Comma-separated metric weights, e.g. average_travel_time=2.0,route_choice_entropy=0.5",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--output-path",
        default="outputs/calibration/calibration_report.json",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    reference = load_reference_scenario(args.reference)
    weights = _parse_weights(args.weights)

    if args.metrics:
        run_metrics = load_run_metrics(args.metrics)
        report = {
            "mode": "single_run",
            "reference_path": args.reference,
            "metrics_path": args.metrics,
            "score": score_run_against_reference(run_metrics, reference, weights=weights),
        }
    elif args.results_json:
        fit = fit_agent_parameters(
            {
                "weights": weights,
                "top_k": args.top_k,
            },
            reference=reference,
            results_path=args.results_json,
        )
        report = {
            "mode": "batch_fit",
            "reference_path": args.reference,
            "results_json": args.results_json,
            "fit": fit,
        }
    else:
        raise SystemExit("Either --metrics or --results-json is required.")

    report_path = export_calibration_report(report, args.output_path)
    print(f"[CALIBRATION] mode={report['mode']}")
    print(f"[CALIBRATION] output={report_path}")
    if report["mode"] == "single_run":
        score = report["score"]
        print(
            f"[CALIBRATION] fit_score={score['fit_score']} "
            f"normalized_loss={score['normalized_loss']} "
            f"metrics={score['metric_count']}"
        )
    else:
        best = report["fit"].get("best_case")
        print(f"[CALIBRATION] candidates={report['fit']['candidate_count']}")
        if best:
            print(
                f"[CALIBRATION] best_case={best.get('case_id')} "
                f"fit_score={best['score']['fit_score']} "
                f"normalized_loss={best['score']['normalized_loss']}"
            )
        else:
            print("[CALIBRATION] best_case=")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
