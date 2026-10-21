"""Football metrics built from StatsBomb Open Data events.

All definitions are explicit and documented here so team, player and shot
tables stay consistent. Key conventions:

* Goals are shots whose ``shot.outcome.name`` is "Goal" (this correctly
  attributes own goals: the shooter's team is credited, the opponent
  concedes). Penalty goals count as goals.
* Shots on target = outcome "Goal" or "Saved".
* A pass is complete when it has no ``pass.outcome`` tag, or when that tag is
  not one of "Out", "Incomplete", "Pass Offside". Passes tagged "Unknown" are
  treated as complete but flagged as approximate (see data validation).
* Match results use the authoritative ``home_score`` / ``away_score`` from the
  match metadata, not the sum of shot events; the two are cross-validated.
* Minutes played are derived from lineup position intervals (``from``/``to``),
  where available; they are NOT estimated from event counts. Per-90 rates are
  therefore not used as headline metrics.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from data_validation import (
    PASS_INCOMPLETE_OUTCOMES,
    SHOT_GOAL,
    SHOT_ON_TARGET_OUTCOMES,
    _event_name,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _team_of(event: dict[str, Any]) -> str | None:
    team = event.get("team")
    return team.get("name") if isinstance(team, dict) else None


def _player_of(event: dict[str, Any]) -> tuple[str | None, int | None]:
    player = event.get("player")
    if isinstance(player, dict):
        return player.get("name"), player.get("id")
    return None, None


def _location_xy(event: dict[str, Any]) -> tuple[float | None, float | None]:
    location = event.get("location")
    if isinstance(location, list) and len(location) >= 2:
        return float(location[0]), float(location[1])
    return None, None


def _shot_body(event: dict[str, Any]) -> dict[str, Any]:
    return event.get("shot", {}) or {}


def _outcome_name(obj: Any) -> str | None:
    return obj.get("name") if isinstance(obj, dict) else None


# ---------------------------------------------------------------------------
# Shot and pass row extraction
# ---------------------------------------------------------------------------


def shot_rows(events: list[dict[str, Any]], match_id: int) -> list[dict[str, Any]]:
    """Flatten shot events into one row per shot."""
    rows: list[dict[str, Any]] = []
    for event in events:
        if _event_name(event) != "Shot":
            continue
        shot = _shot_body(event)
        x, y = _location_xy(event)
        outcome = _outcome_name(shot.get("outcome"))
        xg = shot.get("statsbomb_xg")
        try:
            xg = float(xg) if xg is not None else np.nan
        except (TypeError, ValueError):
            xg = np.nan
        end_location = shot.get("end_location")
        end_x = float(end_location[0]) if isinstance(end_location, list) and end_location else np.nan
        player_name, player_id = _player_of(event)
        rows.append(
            {
                "match_id": match_id,
                "team": _team_of(event),
                "player": player_name,
                "player_id": player_id,
                "period": event.get("period"),
                "minute": event.get("minute"),
                "second": event.get("second"),
                "shot_x": x,
                "shot_y": y,
                "end_x": end_x,
                "outcome": outcome,
                "is_goal": outcome == SHOT_GOAL,
                "is_on_target": outcome in SHOT_ON_TARGET_OUTCOMES,
                "xg": xg,
                "shot_type": _outcome_name(shot.get("type")),
                "technique": _outcome_name(shot.get("technique")),
                "body_part": _outcome_name(shot.get("body_part")),
                "play_pattern": _outcome_name(event.get("play_pattern")),
            }
        )
    return rows


def pass_rows(events: list[dict[str, Any]], match_id: int) -> list[dict[str, Any]]:
    """Flatten pass events into one row per pass with a completion flag."""
    rows: list[dict[str, Any]] = []
    for event in events:
        if _event_name(event) != "Pass":
            continue
        p_body = event.get("pass", {}) or {}
        outcome = p_body.get("outcome")
        outcome_name = _outcome_name(outcome)
        is_complete = (outcome is None) or (outcome_name not in PASS_INCOMPLETE_OUTCOMES)
        end_location = p_body.get("end_location")
        end_x = float(end_location[0]) if isinstance(end_location, list) and end_location else np.nan
        player_name, player_id = _player_of(event)
        rows.append(
            {
                "match_id": match_id,
                "team": _team_of(event),
                "player": player_name,
                "player_id": player_id,
                "period": event.get("period"),
                "minute": event.get("minute"),
                "length": p_body.get("length"),
                "angle": p_body.get("angle"),
                "pass_type": _outcome_name(p_body.get("type")),
                "body_part": _outcome_name(p_body.get("body_part")),
                "height": _outcome_name(p_body.get("height")),
                "outcome": outcome_name,
                "is_complete": is_complete,
                "end_x": end_x,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------


def build_match_table(matches: list[dict[str, Any]]) -> pd.DataFrame:
    """Match-level table from competition metadata.

    Columns: match_id, date, kick_off, home_team, away_team, home_goals,
    away_goals, total_goals, outcome_home (W/D/L from home perspective),
    match_week, competition_stage, stadium, referee.
    """
    rows = []
    for m in matches:
        home = m.get("home_team", {}) or {}
        away = m.get("away_team", {}) or {}
        home_goals = m.get("home_score")
        away_goals = m.get("away_score")
        if home_goals is None or away_goals is None:
            outcome_home = None
        elif home_goals > away_goals:
            outcome_home = "W"
        elif home_goals == away_goals:
            outcome_home = "D"
        else:
            outcome_home = "L"
        rows.append(
            {
                "match_id": m.get("match_id"),
                "date": m.get("match_date"),
                "kick_off": m.get("kick_off"),
                "home_team": home.get("home_team_name"),
                "away_team": away.get("away_team_name"),
                "home_goals": home_goals,
                "away_goals": away_goals,
                "total_goals": (home_goals + away_goals)
                if (home_goals is not None and away_goals is not None)
                else None,
                "outcome_home": outcome_home,
                "match_week": m.get("match_week"),
                "competition_stage": m.get("competition_stage"),
                "stadium": m.get("stadium"),
                "referee": m.get("referee"),
            }
        )
    return pd.DataFrame(rows)


def build_shot_table(events_by_match: dict[int, list[dict[str, Any]]]) -> pd.DataFrame:
    """One row per shot across all matches."""
    all_rows: list[dict[str, Any]] = []
    for match_id, events in events_by_match.items():
        all_rows.extend(shot_rows(events, match_id))
    return pd.DataFrame(all_rows)


def build_pass_table(events_by_match: dict[int, list[dict[str, Any]]]) -> pd.DataFrame:
    """One row per pass across all matches."""
    all_rows: list[dict[str, Any]] = []
    for match_id, events in events_by_match.items():
        all_rows.extend(pass_rows(events, match_id))
    return pd.DataFrame(all_rows)


def build_team_table(
    match_table: pd.DataFrame,
    shot_table: pd.DataFrame,
    pass_table: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate team-level season metrics.

    Goals for/against come from shot events (goal shots), cross-validated
    against match_table scores. Shots and passing are event-based.
    """
    teams = sorted(
        set(match_table["home_team"].dropna()) | set(match_table["away_team"].dropna())
    )

    # Shot-derived goals: goals scored by a team, goals conceded (opponent's goals).
    team_goals_for: dict[str, int] = {t: 0 for t in teams}
    team_goals_against: dict[str, int] = {t: 0 for t in teams}
    if not shot_table.empty and "outcome" in shot_table.columns:
        goals = shot_table[shot_table["is_goal"]]
        for _, g in goals.iterrows():
            team_goals_for[g["team"]] = team_goals_for.get(g["team"], 0) + 1
        # conceded: in each match, the opponent of the scoring team concedes
        match_teams = dict(zip(match_table["match_id"], match_table[["home_team", "away_team"]].values.tolist()))
        for _, g in goals.iterrows():
            mid = g["match_id"]
            if mid in match_teams:
                home, away = match_teams[mid]
                opponent = away if g["team"] == home else home
                if opponent is not None:
                    team_goals_against[opponent] = team_goals_against.get(opponent, 0) + 1

    # Shot volume / accuracy
    team_shots: dict[str, int] = {t: 0 for t in teams}
    team_sot: dict[str, int] = {t: 0 for t in teams}
    if not shot_table.empty:
        for _, s in shot_table.iterrows():
            team_shots[s["team"]] = team_shots.get(s["team"], 0) + 1
            if s["is_on_target"]:
                team_sot[s["team"]] = team_sot.get(s["team"], 0) + 1

    # xG
    team_xg_for: dict[str, float] = {t: 0.0 for t in teams}
    team_xg_against: dict[str, float] = {t: 0.0 for t in teams}
    if not shot_table.empty and "xg" in shot_table.columns:
        for _, s in shot_table.iterrows():
            team_xg_for[s["team"]] = team_xg_for.get(s["team"], 0.0) + float(s["xg"])
        match_teams = dict(zip(match_table["match_id"], match_table[["home_team", "away_team"]].values.tolist()))
        for _, s in shot_table.iterrows():
            mid = s["match_id"]
            if mid in match_teams:
                home, away = match_teams[mid]
                opponent = away if s["team"] == home else home
                if opponent is not None:
                    team_xg_against[opponent] = team_xg_against.get(opponent, 0.0) + float(s["xg"])

    # Passing
    team_passes: dict[str, int] = {t: 0 for t in teams}
    team_passes_complete: dict[str, int] = {t: 0 for t in teams}
    if not pass_table.empty:
        for _, p in pass_table.iterrows():
            team_passes[p["team"]] = team_passes.get(p["team"], 0) + 1
            if p["is_complete"]:
                team_passes_complete[p["team"]] = team_passes_complete.get(p["team"], 0) + 1

    rows = []
    for t in teams:
        played = int(
            ((match_table["home_team"] == t) | (match_table["away_team"] == t)).sum()
        )
        home_wins = int(((match_table["home_team"] == t) & (match_table["outcome_home"] == "W")).sum())
        away_wins = int(((match_table["away_team"] == t) & (match_table["outcome_home"] == "L")).sum())
        draws = int(((match_table["outcome_home"] == "D") & ((match_table["home_team"] == t) | (match_table["away_team"] == t))).sum())
        losses = played - home_wins - away_wins - draws
        gf = team_goals_for.get(t, 0)
        ga = team_goals_against.get(t, 0)
        shots = team_shots.get(t, 0)
        sot = team_sot.get(t, 0)
        passes = team_passes.get(t, 0)
        passes_comp = team_passes_complete.get(t, 0)
        rows.append(
            {
                "team": t,
                "matches_played": played,
                "wins": home_wins + away_wins,
                "draws": draws,
                "losses": losses,
                "points": 3 * (home_wins + away_wins) + draws,
                "goals_for": gf,
                "goals_against": ga,
                "goal_difference": gf - ga,
                "shots": shots,
                "shots_on_target": sot,
                "shot_accuracy_pct": (100.0 * sot / shots) if shots else np.nan,
                "xg_for": round(team_xg_for.get(t, 0.0), 2),
                "xg_against": round(team_xg_against.get(t, 0.0), 2),
                "xg_difference": round(team_xg_for.get(t, 0.0) - team_xg_against.get(t, 0.0), 2),
                "shots_per_match": round(shots / played, 1) if played else np.nan,
                "goals_per_match": round(gf / played, 2) if played else np.nan,
                "passes_attempted": passes,
                "passes_completed": passes_comp,
                "pass_completion_pct": round(100.0 * passes_comp / passes, 1) if passes else np.nan,
                "passes_per_match": round(passes / played, 1) if played else np.nan,
            }
        )
    df = pd.DataFrame(rows).sort_values("points", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


def build_player_table(
    events_by_match: dict[int, list[dict[str, Any]]],
    minutes_by_player: dict[int, float] | None = None,
) -> pd.DataFrame:
    """Player-level season table: shots, goals, xG; optional minutes played."""
    player_stats: dict[int, dict[str, Any]] = {}
    for match_id, events in events_by_match.items():
        for event in events:
            if _event_name(event) != "Shot":
                continue
            shot = _shot_body(event)
            player_name, player_id = _player_of(event)
            if player_id is None:
                continue
            team = _team_of(event)
            stats = player_stats.setdefault(
                player_id,
                {"player": player_name, "team": team, "shots": 0, "goals": 0, "xg": 0.0},
            )
            stats["shots"] += 1
            if _outcome_name(shot.get("outcome")) == SHOT_GOAL:
                stats["goals"] += 1
            xg = shot.get("statsbomb_xg")
            if xg is not None:
                try:
                    stats["xg"] += float(xg)
                except (TypeError, ValueError):
                    pass

    rows = []
    for pid, stats in player_stats.items():
        minutes = minutes_by_player.get(pid) if minutes_by_player else None
        row = {
            "player_id": pid,
            "player": stats["player"],
            "team": stats["team"],
            "shots": stats["shots"],
            "goals": stats["goals"],
            "xg": round(stats["xg"], 2),
        }
        if minutes is not None:
            row["minutes_played"] = round(minutes, 1)
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["goals", "shots", "xg"], ascending=False).reset_index(drop=True)
    return df


def minutes_from_lineups(
    lineups_by_match: dict[int, list[dict[str, Any]]],
    match_end_minutes: dict[int, float] | None = None,
) -> dict[int, float]:
    """Derive minutes played from lineup position intervals.

    Each lineup entry carries position spells with ``from`` / ``to`` timestamps
    (``MM:SS``). A null ``to`` means the player remained until the end of the
    match (stoppage time included). This is a derived, approximate measure:
    it relies on the lineup files being complete and correctly timestamped.
    """
    minutes: dict[int, float] = {}
    for match_id, lineups in lineups_by_match.items():
        end_min = 90.0
        if match_end_minutes and match_id in match_end_minutes:
            end_min = match_end_minutes[match_id]
        for team_entry in lineups:
            for player in team_entry.get("lineup", []) or []:
                pid = player.get("player_id")
                if pid is None:
                    continue
                total_seconds = 0.0
                for spell in player.get("positions", []) or []:
                    start = _parse_ts(spell.get("from"))
                    end = _parse_ts(spell.get("to")) if spell.get("to") is not None else None
                    if start is None:
                        continue
                    end_sec = end if end is not None else end_min * 60.0
                    total_seconds += max(0.0, end_sec - start)
                minutes[pid] = minutes.get(pid, 0.0) + total_seconds / 60.0
    return minutes


def _parse_ts(value: Any) -> float | None:
    """Parse a ``MM:SS`` timestamp into seconds from match start."""
    if value is None or value == "":
        return None
    try:
        parts = str(value).split(":")
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except (ValueError, TypeError):
        return None
    return None
