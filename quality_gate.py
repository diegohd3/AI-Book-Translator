"""
CI quality gate for translation report metrics.

Fails with exit code 1 when report quality thresholds are exceeded.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate translation report metrics")
    parser.add_argument("--report", required=True, help="Path to report JSON file")
    parser.add_argument("--max-errors", type=int, default=0, help="Allowed report errors")
    parser.add_argument(
        "--max-risks",
        type=int,
        default=0,
        help="Allowed consistency risks in report",
    )
    parser.add_argument(
        "--max-uncertain",
        type=int,
        default=0,
        help="Allowed uncertain translation entries in report",
    )
    parser.add_argument(
        "--max-warning-chunks",
        type=int,
        default=0,
        help="Allowed number of chunks with warnings",
    )
    parser.add_argument(
        "--max-refinement-drift",
        type=float,
        default=0.55,
        help="Max allowed refinement drift when drift metrics are present",
    )
    parser.add_argument(
        "--require-refinement-drift",
        action="store_true",
        help="Fail if no refinement drift metrics are present",
    )
    return parser.parse_args()


def _load_report(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Report must be a JSON object")
    return payload


def _get_list(payload: dict, key: str) -> List[object]:
    value = payload.get(key, [])
    if isinstance(value, list):
        return value
    raise ValueError(f"Report field '{key}' must be a list")


def _extract_max_refinement_drift(chunks: List[object]) -> Optional[float]:
    drifts: List[float] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        diagnostics = chunk.get("diagnostics")
        if not isinstance(diagnostics, dict):
            continue
        drift = diagnostics.get("refinement_drift")
        if isinstance(drift, (int, float)):
            drifts.append(float(drift))
    if not drifts:
        return None
    return max(drifts)


def main() -> int:
    args = parse_args()
    report_path = Path(args.report)

    if not report_path.exists():
        print(f"[quality-gate] Missing report: {report_path}", file=sys.stderr)
        return 1

    try:
        report = _load_report(report_path)
        errors = _get_list(report, "errors")
        consistency_risks = _get_list(report, "consistency_risks")
        uncertain = _get_list(report, "uncertain_translations")
        chunks = _get_list(report, "chunks")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[quality-gate] Invalid report payload: {exc}", file=sys.stderr)
        return 1

    warning_chunks = 0
    for chunk in chunks:
        if isinstance(chunk, dict) and isinstance(chunk.get("warnings"), list):
            if len(chunk["warnings"]) > 0:
                warning_chunks += 1

    max_drift = _extract_max_refinement_drift(chunks)

    failures: List[str] = []
    if len(errors) > args.max_errors:
        failures.append(f"errors={len(errors)} > {args.max_errors}")
    if len(consistency_risks) > args.max_risks:
        failures.append(f"consistency_risks={len(consistency_risks)} > {args.max_risks}")
    if len(uncertain) > args.max_uncertain:
        failures.append(f"uncertain_translations={len(uncertain)} > {args.max_uncertain}")
    if warning_chunks > args.max_warning_chunks:
        failures.append(f"warning_chunks={warning_chunks} > {args.max_warning_chunks}")

    if max_drift is None:
        if args.require_refinement_drift:
            failures.append("no refinement_drift metrics found")
    elif max_drift > args.max_refinement_drift:
        failures.append(
            f"max_refinement_drift={max_drift:.4f} > {args.max_refinement_drift:.4f}"
        )

    print(
        "[quality-gate] "
        f"errors={len(errors)}, "
        f"risks={len(consistency_risks)}, "
        f"uncertain={len(uncertain)}, "
        f"warning_chunks={warning_chunks}, "
        f"max_refinement_drift={max_drift if max_drift is not None else 'n/a'}"
    )

    if failures:
        print("[quality-gate] FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("[quality-gate] PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
