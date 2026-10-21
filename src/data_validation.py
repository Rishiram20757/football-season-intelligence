"""Data-quality checks run before any metric is computed.

Returns readable validation summaries and uses ``assert`` for genuine
invariants plus ``warnings.warn`` for nonfatal issues. Nothing is silently
removed or coerced into misleading zeros.
"""

from __future__ import annotations

import warnings
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from data_loader import PITCH_LENGTH, PITCH_WIDTH

# Event fields we rely on; their presence is validated but never assumed.
PITCH_X_MIN, PITCH_X_MAX = 0.0, PITCH_LENGTH
PITCH_Y_MIN, PITCH_Y_MAX = 0.0, PITCH_WIDTH

# Shot outcomes that count as a goal / on target (StatsBomb open-data names).
SHOT_GOAL = "Goal"
SHOT_ON_TARGET_OUTCOMES = {SHOT_GOAL, "Saved"}
# Pass outcomes that indicate an incomplete pass.
PASS_INCOMPLETE_OUTCOMES = {"Out", "Incomplete", "Pass Offside"}


def _event_name(event: dict[str, Any]) -> str:
    typ = event.get("type")
    return typ.get("name") if isinstance(typ, dict) else str(typ)


def validate_match_metadata(matches: list[dict[str, Any]]) -> dict[str, Any]:
    """Check match metadata completeness and id uniqueness."""
    report: dict[str, Any] = {"n_matches": len(matches), "warnings": [], "errors": []}
    if not matches:
        report["errors"].append("No matches supplied.")
        return report

    match_ids = [m.get("match_id") for m in matches]
    missing_id = [i for i, m in enumerate(matches) if m.get("match_id") is None]
    if missing_id:
        report["errors"].append(f"{len(missing_id)} matches missing match_id")
    if len(set(match_ids)) != len(match_ids):
        dupes = [mid for mid, cnt in Counter(match_ids).items() if cnt > 1]
        report["errors"].append(f"Duplicate match_ids: {dupes[:10]}")

    for field in ("match_date", "home_team", "away_team", "home_score", "away_score"):
        n_missing = sum(1 for m in matches if field not in m or m.get(field) is None)
        if n_missing:
            report["warnings"].append(f"{n_missing} matches missing '{field}'")

    scores_missing = sum(
        1
        for m in matches
        if m.get("home_score") is None or m.get("away_score") is None
    )
    if scores_missing:
        report["warnings"].append(f"{scores_missing} matches missing score values")

    report["n_unique_matches"] = len(set(match_ids))
    report["valid"] = not report["errors"]
    return report


def validate_events(events: list[dict[str, Any]], match_id: int | str) -> dict[str, Any]:
    """Validate a single match's event stream."""
    report: dict[str, Any] = {
        "match_id": match_id,
        "n_events": len(events),
        "warnings": [],
        "errors": [],
        "event_types": {},
        "teams": set(),
        "players_without_id": 0,
        "events_outside_pitch": 0,
        "duplicate_event_ids": 0,
    }
    if not events:
        report["warnings"].append("Empty event list")
        report["valid"] = True
        return report

    event_ids: list[str] = []
    for event in events:
        typ = _event_name(event)
        report["event_types"][typ] = report["event_types"].get(typ, 0) + 1

        team = event.get("team")
        if isinstance(team, dict):
            report["teams"].add(team.get("name"))
        player = event.get("player")
        if player is None or not isinstance(player, dict) or player.get("id") is None:
            report["players_without_id"] += 1

        location = event.get("location")
        if isinstance(location, list) and len(location) >= 2:
            x, y = location[0], location[1]
            if (
                x is None
                or y is None
                or not (PITCH_X_MIN <= float(x) <= PITCH_X_MAX)
                or not (PITCH_Y_MIN <= float(y) <= PITCH_Y_MAX)
            ):
                report["events_outside_pitch"] += 1

        eid = event.get("id")
        if eid is not None:
            event_ids.append(eid)

    if report["players_without_id"]:
        report["warnings"].append(
            f"{report['players_without_id']} events missing a usable player id"
        )
    if report["events_outside_pitch"]:
        report["warnings"].append(
            f"{report['events_outside_pitch']} events with coordinates outside the "
            f"{PITCH_LENGTH}x{PITCH_WIDTH} pitch"
        )

    id_counts = Counter(event_ids)
    dupes = [eid for eid, cnt in id_counts.items() if cnt > 1]
    report["duplicate_event_ids"] = len(dupes)
    if dupes:
        report["errors"].append(f"Duplicate event ids within match {match_id}: {dupes[:5]}")

    n_teams = len(report["teams"])
    report["teams"] = sorted(report["teams"])
    if n_teams != 2:
        report["warnings"].append(f"Expected 2 teams, found {n_teams}: {report['teams']}")

    report["valid"] = not report["errors"]
    return report


def validate_shots(shots: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate shot events: outcomes, xG range, end locations."""
    report: dict[str, Any] = {
        "n_shots": len(shots),
        "warnings": [],
        "errors": [],
        "outcomes": {},
        "shots_without_xg": 0,
        "xg_out_of_range": 0,
    }
    if not shots:
        report["warnings"].append("No shot events")
        report["valid"] = True
        return report

    xg_values: list[float] = []
    for shot in shots:
        shot_body = shot.get("shot", {})
        outcome = shot_body.get("outcome", {})
        outcome_name = outcome.get("name") if isinstance(outcome, dict) else None
        report["outcomes"][outcome_name] = report["outcomes"].get(outcome_name, 0) + 1

        xg = shot_body.get("statsbomb_xg")
        if xg is None:
            report["shots_without_xg"] += 1
        else:
            try:
                xg_values.append(float(xg))
                if not 0.0 <= float(xg) <= 1.0:
                    report["xg_out_of_range"] += 1
            except (TypeError, ValueError):
                report["shots_without_xg"] += 1

    if report["shots_without_xg"]:
        report["warnings"].append(
            f"{report['shots_without_xg']} shots have no statsbomb_xg value"
        )
    if report["xg_out_of_range"]:
        report["warnings"].append(f"{report['xg_out_of_range']} shots have xG outside [0, 1]")

    if xg_values:
        report["xg_min"] = float(np.min(xg_values))
        report["xg_max"] = float(np.max(xg_values))
        report["xg_sum"] = float(np.sum(xg_values))

    report["valid"] = not report["errors"]
    return report


def validate_passes(passes: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate pass events and their outcome tags."""
    report: dict[str, Any] = {
        "n_passes": len(passes),
        "warnings": [],
        "errors": [],
        "outcomes": {},
        "passes_without_outcome": 0,
        "unknown_outcome_passes": 0,
    }
    if not passes:
        report["warnings"].append("No pass events")
        report["valid"] = True
        return report

    for p in passes:
        p_body = p.get("pass", {})
        outcome = p_body.get("outcome")
        outcome_name = outcome.get("name") if isinstance(outcome, dict) else None
        if outcome is None:
            report["passes_without_outcome"] += 1
        elif outcome_name == "Unknown":
            report["unknown_outcome_passes"] += 1
        report["outcomes"][outcome_name] = report["outcomes"].get(outcome_name, 0) + 1

    if report["passes_without_outcome"]:
        report["warnings"].append(
            f"{report['passes_without_outcome']} passes have no outcome tag "
            f"(treated as completed)"
        )
    if report["unknown_outcome_passes"]:
        report["warnings"].append(
            f"{report['unknown_outcome_passes']} passes have outcome 'Unknown' "
            f"(treated as completed; treat completion rate as approximate)"
        )

    report["valid"] = not report["errors"]
    return report


def validate_season_coverage(
    matches: list[dict[str, Any]],
    events_by_match: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Cross-check that every match has an event stream and consistent ids."""
    expected_ids = {int(m["match_id"]) for m in matches}
    loaded_ids = set(events_by_match.keys())
    missing = sorted(expected_ids - loaded_ids)
    extra = sorted(loaded_ids - expected_ids)

    report: dict[str, Any] = {
        "expected_matches": len(expected_ids),
        "matches_with_events": len(loaded_ids),
        "missing_event_files": missing,
        "unexpected_match_ids": extra,
    }
    if missing:
        report["warnings"] = [f"{len(missing)} matches have no event file: {missing[:10]}"]
    else:
        report["warnings"] = []
    report["complete_coverage"] = not missing and not extra
    report["valid"] = not missing and not extra
    return report


def summarize_validation(reports: list[dict[str, Any]]) -> str:
    """Flatten validation reports into a short readable text block."""
    lines = []
    for rep in reports:
        status = "OK" if rep.get("valid") else "ISSUES"
        lines.append(f"- {rep.get('match_id', rep.get('section', 'check'))}: {status}")
        for warning in rep.get("warnings", [])[:5]:
            lines.append(f"    * warning: {warning}")
        for error in rep.get("errors", [])[:5]:
            lines.append(f"    * ERROR: {error}")
    return "\n".join(lines)
