"""Reusable visualisations for the football season analysis.

A restrained, colourblind-friendly theme is used throughout: deep navy as the
primary colour, orange for goals, grey for non-goals, and the mplsoccer pitch
in its classic green. Every figure is saved to ``outputs/figures/`` with a
descriptive filename.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from mplsoccer import Pitch

from data_loader import PITCH_LENGTH, PITCH_WIDTH
from data_validation import PASS_INCOMPLETE_OUTCOMES, SHOT_GOAL


def _type_name(event: dict[str, Any]) -> str:
    typ = event.get("type")
    return typ.get("name") if isinstance(typ, dict) else str(typ)


def _team_of(event: dict[str, Any]) -> str | None:
    team = event.get("team")
    return team.get("name") if isinstance(team, dict) else None

# ---------------------------------------------------------------------------
# Palette (colourblind-safe, restrained)
# ---------------------------------------------------------------------------
NAVY = "#14426C"
STEEL = "#2E75B6"
ORANGE = "#E69F00"
GREY = "#8C8C8C"
GREEN = "#1A9850"
LIGHT_GREY = "#D9D9D9"

TEAM_COLOURS = {
    "Tottenham Hotspur": NAVY,
    "default": STEEL,
}

FIGSIZE = (10, 6)
DPI = 150

PITCH_KWARGS = dict(pitch_type="statsbomb", pitch_color="#EEF3F7", line_color="#1E3F5A")


def setup_style() -> None:
    """Apply a consistent, professional plotting style."""
    rcParams.update(
        {
            "figure.figsize": FIGSIZE,
            "figure.dpi": DPI,
            "font.size": 11,
            "font.family": "DejaVu Sans",
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.edgecolor": "#4D4D4D",
            "axes.linewidth": 0.8,
            "axes.labelcolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 10,
            "legend.frameon": False,
            "grid.color": "#D3D3D3",
            "grid.linestyle": "--",
            "grid.alpha": 0.7,
            "axes.grid": True,
            "savefig.bbox": "tight",
            "savefig.dpi": DPI,
        }
    )


def save_fig(fig: plt.Figure, path: Path | str) -> Path:
    """Save a figure, creating the parent directory if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Section 8 charts
# ---------------------------------------------------------------------------


def plot_goals_per_match(match_table, path: Path | str) -> Path:
    """Histogram of total goals scored per match."""
    goals = match_table["total_goals"].dropna()
    fig, ax = plt.subplots()
    bins = np.arange(0, goals.max() + 2) - 0.5
    ax.hist(goals, bins=bins, color=NAVY, edgecolor="white", linewidth=0.8)
    ax.set_xticks(range(int(goals.max()) + 1))
    ax.set_xlabel("Total goals in a match")
    ax.set_ylabel("Number of matches")
    ax.set_title("Goals per match distribution", pad=12, loc="left", fontweight="bold")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, alpha=0.7)
    ax.set_axisbelow(True)
    return save_fig(fig, path)


def plot_home_away_goals(match_table, path: Path | str) -> Path:
    """Compare total home vs away goals."""
    home_goals = match_table["home_goals"].sum()
    away_goals = match_table["away_goals"].sum()
    fig, ax = plt.subplots()
    bars = ax.bar(
        ["Home teams", "Away teams"],
        [home_goals, away_goals],
        color=[STEEL, NAVY],
        width=0.55,
    )
    for bar, value in zip(bars, [home_goals, away_goals], strict=False):
        ax.annotate(
            f"{value:,}",
            (bar.get_x() + bar.get_width() / 2, value),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            fontweight="bold",
        )
    ax.set_ylabel("Goals")
    ax.set_title("Home vs away goals (all matches)", pad=12, loc="left", fontweight="bold")
    ax.set_axisbelow(True)
    return save_fig(fig, path)


def plot_match_outcomes(match_table, path: Path | str) -> Path:
    """Home win / draw / away win share."""
    counts = match_table["outcome_home"].value_counts().reindex(["W", "D", "A"]).fillna(0)
    labels = ["Home win", "Draw", "Away win"]
    colours = [STEEL, LIGHT_GREY, NAVY]
    fig, ax = plt.subplots()
    wedges, _, autotexts = ax.pie(
        counts.values,
        labels=labels,
        colors=colours,
        autopct=lambda p: f"{p:.1f}%\n({int(round(p * counts.sum() / 100))})",
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.42, edgecolor="white"),
    )
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")
        autotext.set_fontsize(9.5)
    ax.set_title("Match outcomes (home-team perspective)", pad=12, loc="left", fontweight="bold")
    return save_fig(fig, path)


def plot_highest_scoring_matches(match_table, path: Path | str, top_n: int = 10) -> Path:
    """Highest-scoring matches of the season."""
    top = match_table.sort_values("total_goals", ascending=False).head(top_n)
    top = top.iloc[::-1]
    labels = [
        f"{r.home_team} {r.home_goals}-{r.away_goals} {r.away_team}"
        for r in top.itertuples()
    ]
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(top) + 1)))
    ax.barh(labels, top["total_goals"].values, color=ORANGE, edgecolor="white")
    for i, (_, row) in enumerate(top.iterrows()):
        ax.annotate(
            f"{int(row.total_goals)} goals · {row.date}",
            (row.total_goals, i),
            textcoords="offset points",
            xytext=(5, 0),
            va="center",
            fontsize=9,
            color="#333333",
        )
    ax.set_xlabel("Total goals")
    ax.set_title("Highest-scoring matches of the season", pad=12, loc="left", fontweight="bold")
    ax.invert_yaxis()
    return save_fig(fig, path)


def plot_goals_by_period(shot_table, path: Path | str) -> Path:
    """First-half vs second-half goal share from shot events."""
    goals = shot_table[shot_table["is_goal"]]
    first = int((goals["period"] == 1).sum())
    second = int((goals["period"] == 2).sum())
    fig, ax = plt.subplots()
    bars = ax.bar(
        ["First half", "Second half"],
        [first, second],
        color=[STEEL, NAVY],
        width=0.5,
    )
    for bar, value in zip(bars, [first, second], strict=False):
        ax.annotate(
            f"{value}",
            (bar.get_x() + bar.get_width() / 2, value),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            fontweight="bold",
        )
    ax.set_ylabel("Goals")
    ax.set_title("Goals by match period", pad=12, loc="left", fontweight="bold")
    ax.set_axisbelow(True)
    return save_fig(fig, path)


# ---------------------------------------------------------------------------
# Section 9 charts
# ---------------------------------------------------------------------------


def plot_team_goals_for(team_table, path: Path | str) -> Path:
    df = team_table.sort_values("goals_for").iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 9))
    colours = [NAVY if t == "Tottenham Hotspur" else STEEL for t in df["team"]]
    bars = ax.barh(df["team"], df["goals_for"], color=colours, edgecolor="white")
    for bar, value in zip(bars, df["goals_for"], strict=False):
        ax.annotate(
            f"{int(value)}",
            (bar.get_width(), bar.get_y() + bar.get_height() / 2),
            textcoords="offset points",
            xytext=(3, 0),
            va="center",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_xlabel("Goals scored")
    ax.set_title("Goals scored by team (season ranking)", pad=12, loc="left", fontweight="bold")
    ax.set_axisbelow(True)
    return save_fig(fig, path)


def plot_team_goal_difference(team_table, path: Path | str) -> Path:
    df = team_table.sort_values("goal_difference").iloc[::-1]
    colours = [
        NAVY if team == "Tottenham Hotspur" else (GREEN if d >= 0 else "#B23A48")
        for team, d in zip(df["team"], df["goal_difference"])
    ]
    fig, ax = plt.subplots(figsize=(10, 9))
    bars = ax.barh(df["team"], df["goal_difference"], color=colours, edgecolor="white")
    ax.axvline(0, color="#4D4D4D", linewidth=0.8)
    for bar, value in zip(bars, df["goal_difference"], strict=False):
        ax.annotate(
            f"{int(value)}",
            (bar.get_width() + (0.8 if value >= 0 else -0.8), bar.get_y() + bar.get_height() / 2),
            textcoords="offset points",
            xytext=(3 if value >= 0 else -3, 0),
            ha="left" if value >= 0 else "right",
            va="center",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_xlabel("Goal difference (goals for − goals against)")
    ax.set_title("Goal difference by team", pad=12, loc="left", fontweight="bold")
    ax.set_axisbelow(True)
    return save_fig(fig, path)


def plot_shots_vs_goals(team_table, path: Path | str) -> Path:
    fig, ax = plt.subplots()
    colours = [NAVY if t == "Tottenham Hotspur" else STEEL for t in team_table["team"]]
    ax.scatter(team_table["shots"], team_table["goals_for"], c=colours, s=55, alpha=0.85, edgecolor="white", linewidth=0.5)
    for _, row in team_table.iterrows():
        if row["team"] == "Tottenham Hotspur":
            ax.annotate("Tottenham", (row.shots, row.goals_for), textcoords="offset points", xytext=(6, 6), fontsize=9, fontweight="bold", color=NAVY)
    ax.set_xlabel("Shots attempted")
    ax.set_ylabel("Goals scored")
    ax.set_title("Team shot volume vs goals scored", pad=12, loc="left", fontweight="bold")
    ax.set_axisbelow(True)
    return save_fig(fig, path)


def plot_xg_vs_goals(team_table, path: Path | str) -> Path:
    fig, ax = plt.subplots()
    colours = [NAVY if t == "Tottenham Hotspur" else STEEL for t in team_table["team"]]
    ax.scatter(team_table["xg_for"], team_table["goals_for"], c=colours, s=55, alpha=0.85, edgecolor="white", linewidth=0.5)
    lims = [
        min(team_table["xg_for"].min(), team_table["goals_for"].min()),
        max(team_table["xg_for"].max(), team_table["goals_for"].max()),
    ]
    ax.plot(lims, lims, ls="--", color="#4D4D4D", linewidth=1)
    ax.annotate("Goals = xG", xy=(lims[0] * 0.95 + lims[1] * 0.05, lims[1] * 0.95), fontsize=9, color="#4D4D4D")
    for _, row in team_table.iterrows():
        if row["team"] == "Tottenham Hotspur":
            ax.annotate("Tottenham", (row.xg_for, row.goals_for), textcoords="offset points", xytext=(6, 6), fontsize=9, fontweight="bold", color=NAVY)
    ax.set_xlabel("Expected goals (xG, StatsBomb)")
    ax.set_ylabel("Actual goals scored")
    ax.set_title("Team xG vs actual goals", pad=12, loc="left", fontweight="bold")
    ax.set_axisbelow(True)
    return save_fig(fig, path)


def plot_passing_comparison(team_table, path: Path | str) -> Path:
    fig, ax = plt.subplots()
    colours = [NAVY if t == "Tottenham Hotspur" else STEEL for t in team_table["team"]]
    ax.scatter(team_table["passes_attempted"], team_table["pass_completion_pct"], c=colours, s=55, alpha=0.85, edgecolor="white", linewidth=0.5)
    for _, row in team_table.iterrows():
        if row["team"] == "Tottenham Hotspur":
            ax.annotate("Tottenham", (row.passes_attempted, row.pass_completion_pct), textcoords="offset points", xytext=(6, 6), fontsize=9, fontweight="bold", color=NAVY)
    ax.set_xlabel("Passes attempted")
    ax.set_ylabel("Pass completion %")
    ax.set_title("Passing volume vs completion rate by team", pad=12, loc="left", fontweight="bold")
    ax.set_axisbelow(True)
    return save_fig(fig, path)


# ---------------------------------------------------------------------------
# Section 10 charts: shot maps
# ---------------------------------------------------------------------------


def plot_shot_map(
    shot_table,
    path: Path | str,
    team: str | None = None,
    title: str | None = None,
) -> Path:
    """Shot map distinguishing goals (orange) from non-goals (grey)."""
    data = shot_table.copy()
    if team:
        data = data[data["team"] == team]
    pitch = Pitch(**PITCH_KWARGS)
    fig, ax = pitch.draw(figsize=(11, 7))

    non_goals = data[~data["is_goal"]]
    goals = data[data["is_goal"]]

    if not non_goals.empty:
        pitch.scatter(non_goals["shot_x"], non_goals["shot_y"], ax=ax, s=45, color=GREY, alpha=0.75, edgecolor="white", linewidth=0.4)
    if not goals.empty:
        pitch.scatter(goals["shot_x"], goals["shot_y"], ax=ax, s=120, color=ORANGE, edgecolor="white", linewidth=1.2, zorder=5)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="Non-goal shot", markerfacecolor=GREY, markersize=8, alpha=0.8),
        Line2D([0], [0], marker="o", color="w", label="Goal", markerfacecolor=ORANGE, markersize=11, markeredgecolor="white"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", frameon=False, title="Shot outcome")
    t = title or ("Shot map — all shots" if not team else f"Shot map — {team}")
    ax.set_title(t, pad=10, loc="left", fontweight="bold")
    if team:
        ax.set_title(f"{t}  (attacking left → right)", fontsize=9, loc="right", color="#4D4D4D")
    else:
        ax.text(1.0, -0.12, "Attacking left → right (StatsBomb pitch 120×80, goal at x = 120)", transform=ax.transAxes, ha="right", fontsize=8.5, color="#666666")
    return save_fig(fig, path)


def plot_shots_by_team(shot_table, team_table, path: Path | str, top_n: int = 10) -> Path:
    top = team_table.sort_values("shots", ascending=False).head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(top) + 1)))
    colours = [NAVY if t == "Tottenham Hotspur" else STEEL for t in top["team"]]
    bars = ax.barh(top["team"], top["shots"], color=colours, edgecolor="white")
    for bar, value in zip(bars, top["shots"], strict=False):
        ax.annotate(
            f"{int(value)}",
            (bar.get_width(), bar.get_y() + bar.get_height() / 2),
            textcoords="offset points",
            xytext=(3, 0),
            va="center",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_xlabel("Shots attempted")
    ax.set_title(f"Shot volume — top {top_n} teams", pad=12, loc="left", fontweight="bold")
    ax.set_axisbelow(True)
    return save_fig(fig, path)


def plot_xg_distribution(shot_table, path: Path | str) -> Path:
    fig, ax = plt.subplots()
    ax.hist(shot_table["xg"].dropna(), bins=30, color=STEEL, edgecolor="white", alpha=0.85)
    ax.set_xlabel("xG per shot (StatsBomb statsbomb_xg)")
    ax.set_ylabel("Number of shots")
    ax.set_title("Distribution of shot xG values", pad=12, loc="left", fontweight="bold")
    ax.set_axisbelow(True)
    return save_fig(fig, path)


# ---------------------------------------------------------------------------
# Section 11 charts
# ---------------------------------------------------------------------------


def plot_player_goals(player_table, path: Path | str, top_n: int = 12) -> Path:
    top = player_table.sort_values("goals", ascending=False).head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(top) + 1)))
    colours = [NAVY if (t == "Tottenham Hotspur") else STEEL for t in top["team"]]
    bars = ax.barh(top["player"], top["goals"], color=colours, edgecolor="white")
    for bar, (_, row) in zip(bars, top.iterrows(), strict=False):
        ax.annotate(
            f"{int(row.goals)} goals  ·  {row.shots} shots  ·  xG {row.xg:.2f}",
            (bar.get_width(), bar.get_y() + bar.get_height() / 2),
            textcoords="offset points",
            xytext=(3, 0),
            va="center",
            fontsize=9,
        )
    ax.set_xlabel("Goals")
    ax.set_title(f"Top goalscorers (top {top_n})", pad=12, loc="left", fontweight="bold")
    ax.set_axisbelow(True)
    return save_fig(fig, path)


# ---------------------------------------------------------------------------
# Section 12 charts: passing map for a single match
# ---------------------------------------------------------------------------


def plot_pass_map(
    events: list[dict[str, Any]],
    match_id: int,
    path: Path | str,
    team: str | None = None,
) -> Path:
    """Pass map for a single match: completed passes (steel) vs incomplete (grey)."""
    pitch = Pitch(**PITCH_KWARGS)
    fig, ax = pitch.draw(figsize=(11, 7))

    completed_lines, incomplete_lines = [], []
    for event in events:
        if _type_name(event) != "Pass":
            continue
        p_body = event.get("pass", {}) or {}
        outcome = p_body.get("outcome")
        outcome_name = outcome.get("name") if isinstance(outcome, dict) else None
        team_name = _team_of(event)
        if team and team_name != team:
            continue
        start = event.get("location")
        end = p_body.get("end_location")
        if not (isinstance(start, list) and len(start) >= 2 and isinstance(end, list) and len(end) >= 2):
            continue
        line = [[start[0], start[1]], [end[0], end[1]]]
        if outcome is None or outcome_name not in PASS_INCOMPLETE_OUTCOMES:
            completed_lines.append(line)
        else:
            incomplete_lines.append(line)

    for line in completed_lines:
        ax.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], color=STEEL, alpha=0.35, linewidth=0.8, zorder=2)
    for line in incomplete_lines:
        ax.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], color=GREY, alpha=0.5, linewidth=1.0, zorder=3, linestyle="--")

    # starting locations of passes
    xs, ys = [], []
    for event in events:
        if _type_name(event) != "Pass":
            continue
        if team and _team_of(event) != team:
            continue
        start = event.get("location")
        if isinstance(start, list) and len(start) >= 2:
            xs.append(start[0]); ys.append(start[1])
    if xs:
        ax.scatter(xs, ys, s=16, color=NAVY, zorder=4, edgecolor="white", linewidth=0.3)

    title = f"Pass map — match {match_id}" + (f" ({team})" if team else "")
    ax.set_title(title, pad=10, loc="left", fontweight="bold")
    ax.text(1.0, -0.12, "Solid = completed · dashed = incomplete · attacking left → right", transform=ax.transAxes, ha="right", fontsize=8.5, color="#666666")
    return save_fig(fig, path)


# ---------------------------------------------------------------------------
# Player-specific charts
# ---------------------------------------------------------------------------


def plot_player_shot_map(
    shot_table,
    player_name: str,
    path: Path | str,
    title: str | None = None,
) -> Path:
    """Shot map for a single player: goals (orange) vs non-goals (grey)."""
    data = shot_table[shot_table["player"] == player_name].copy()
    pitch = Pitch(**PITCH_KWARGS)
    fig, ax = pitch.draw(figsize=(11, 7))

    non_goals = data[~data["is_goal"]]
    goals = data[data["is_goal"]]

    if not non_goals.empty:
        pitch.scatter(non_goals["shot_x"], non_goals["shot_y"], ax=ax, s=60, color=GREY, alpha=0.75, edgecolor="white", linewidth=0.4)
    if not goals.empty:
        pitch.scatter(goals["shot_x"], goals["shot_y"], ax=ax, s=150, color=ORANGE, edgecolor="white", linewidth=1.4, zorder=5)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="Non-goal shot", markerfacecolor=GREY, markersize=9, alpha=0.8),
        Line2D([0], [0], marker="o", color="w", label="Goal", markerfacecolor=ORANGE, markersize=13, markeredgecolor="white"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", frameon=False, title="Shot outcome")
    t = title or f"Shot map — {player_name}"
    ax.set_title(t, pad=10, loc="left", fontweight="bold")
    ax.text(1.0, -0.12, "Attacking left → right (StatsBomb pitch 120×80, goal at x = 120)", transform=ax.transAxes, ha="right", fontsize=8.5, color="#666666")
    return save_fig(fig, path)


def plot_player_shot_outcomes(
    shot_table,
    player_name: str,
    path: Path | str,
    title: str | None = None,
) -> Path:
    """Bar chart of shot outcomes for a single player."""
    data = shot_table[shot_table["player"] == player_name].copy()
    if data.empty:
        raise ValueError("No shots found for player " + player_name)
    counts = data["outcome"].value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    colours = [ORANGE if o == SHOT_GOAL else GREY for o in counts.index]
    bars = ax.barh(counts.index, counts.values, color=colours, edgecolor="white")
    for bar, value in zip(bars, counts.values, strict=False):
        ax.annotate(str(int(value)), (bar.get_width(), bar.get_y() + bar.get_height() / 2), textcoords="offset points", xytext=(3, 0), va="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("Number of shots")
    ax.set_title(title or f"Shot outcomes — {player_name}", pad=12, loc="left", fontweight="bold")
    ax.set_axisbelow(True)
    return save_fig(fig, path)


def plot_player_heatmap(
    events_by_match: dict[int, list[dict[str, Any]]],
    player_id: int,
    path: Path | str,
    title: str | None = None,
) -> Path:
    """Event-location heatmap for a single player (all event types)."""
    xs, ys = [], []
    for events in events_by_match.values():
        for event in events:
            p = event.get("player")
            if p and p.get("id") == player_id:
                loc = event.get("location")
                if isinstance(loc, list) and len(loc) >= 2:
                    xs.append(float(loc[0]))
                    ys.append(float(loc[1]))
    pitch = Pitch(**PITCH_KWARGS)
    fig, ax = pitch.draw(figsize=(11, 7))
    if xs:
        ax.hist2d(xs, ys, bins=15, cmap="YlOrRd", alpha=0.85, zorder=2)
    t = title or f"Event heatmap — player {player_id}"
    ax.set_title(t, pad=10, loc="left", fontweight="bold")
    ax.text(1.0, -0.12, "All event locations, not tracking data (StatsBomb pitch 120×80)", transform=ax.transAxes, ha="right", fontsize=8.5, color="#666666")
    return save_fig(fig, path)


def plot_player_match_by_match(
    shot_table,
    pass_table,
    player_id: int,
    player_name: str,
    path: Path | str,
    title: str | None = None,
) -> Path:
    """Match-by-match summary: shots, goals, xG, passes for one player."""
    rows = []
    for mid in sorted(shot_table["match_id"].unique()):
        s = shot_table[(shot_table["match_id"] == mid) & (shot_table["player_id"] == player_id)]
        p = pass_table[(pass_table["match_id"] == mid) & (pass_table["player_id"] == player_id)]
        goals = int(s["is_goal"].sum())
        shots = int(len(s))
        xg = float(s["xg"].sum()) if "xg" in s.columns else 0.0
        passes = int(len(p))
        passes_comp = int(p["is_complete"].sum()) if not p.empty else 0
        rows.append({"match_id": mid, "shots": shots, "goals": goals, "xg": xg, "passes": passes, "pass_comp": passes_comp})
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No match data found for player " + player_name)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title or f"Match-by-match performance — {player_name}", fontsize=13, fontweight="bold", y=1.02)

    axes[0, 0].bar(df["match_id"].astype(str), df["shots"], color=STEEL, edgecolor="white")
    axes[0, 0].set_title("Shots per match")
    axes[0, 0].set_xlabel("Match ID")
    axes[0, 0].set_ylabel("Shots")
    axes[0, 0].tick_params(axis="x", rotation=45, labelsize=7)

    axes[0, 1].bar(df["match_id"].astype(str), df["goals"], color=ORANGE, edgecolor="white")
    axes[0, 1].set_title("Goals per match")
    axes[0, 1].set_xlabel("Match ID")
    axes[0, 1].set_ylabel("Goals")
    axes[0, 1].tick_params(axis="x", rotation=45, labelsize=7)

    axes[1, 0].bar(df["match_id"].astype(str), df["xg"], color=NAVY, edgecolor="white")
    axes[1, 0].set_title("xG per match")
    axes[1, 0].set_xlabel("Match ID")
    axes[1, 0].set_ylabel("xG")
    axes[1, 0].tick_params(axis="x", rotation=45, labelsize=7)

    axes[1, 1].bar(df["match_id"].astype(str), df["passes"], color=GREY, edgecolor="white")
    axes[1, 1].set_title("Passes per match")
    axes[1, 1].set_xlabel("Match ID")
    axes[1, 1].set_ylabel("Passes")
    axes[1, 1].tick_params(axis="x", rotation=45, labelsize=7)

    fig.tight_layout()
    return save_fig(fig, path)


def plot_player_comparison(
    player_table,
    focal_player: str,
    path: Path | str,
    title: str | None = None,
    top_n: int = 10,
) -> Path:
    """Compare focal player against peers on goals, shots, xG (per 90 where minutes available)."""
    df = player_table.copy()
    focal = df[df["player"] == focal_player]
    if focal.empty:
        raise ValueError("Focal player not found: " + focal_player)

    # Select attackers: players with at least 5 shots as a reasonable filter
    attackers = df[df["shots"] >= 5].copy()
    if "minutes_played" in attackers.columns and attackers["minutes_played"].notna().any():
        attackers = attackers[attackers["minutes_played"].notna() & (attackers["minutes_played"] > 0)].copy()
        attackers["shots_per_90"] = attackers["shots"] / attackers["minutes_played"] * 90.0
        attackers["goals_per_90"] = attackers["goals"] / attackers["minutes_played"] * 90.0
        attackers["xg_per_90"] = attackers["xg"] / attackers["minutes_played"] * 90.0
        metric_col = "goals_per_90"
        sort_col = "goals_per_90"
    else:
        metric_col = "goals"
        sort_col = "goals"

    attackers = attackers.sort_values(sort_col, ascending=False).head(top_n)
    attackers = attackers.iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(attackers) + 1)))
    colours = [NAVY if p == focal_player else STEEL for p in attackers["player"]]
    bars = ax.barh(attackers["player"], attackers[metric_col], color=colours, edgecolor="white")
    for bar, (_, row) in zip(bars, attackers.iterrows(), strict=False):
        val = row[metric_col]
        ax.annotate(f"{val:.2f}", (bar.get_width(), bar.get_y() + bar.get_height() / 2), textcoords="offset points", xytext=(3, 0), va="center", fontsize=9, fontweight="bold")

    label = metric_col.replace("_", " ").title()
    ax.set_xlabel(label)
    ax.set_title(title or f"Goals per 90 — {focal_player} vs peers (min 5 shots)", pad=12, loc="left", fontweight="bold")
    ax.set_axisbelow(True)
    return save_fig(fig, path)

