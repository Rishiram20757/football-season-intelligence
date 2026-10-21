"""Data acquisition for the StatsBomb Open Data repository.

All files are fetched from the official public repository:

    https://github.com/statsbomb/open-data

Raw JSON files are cached under ``data/raw/`` so that re-running the notebook
does not re-download everything. Every network or parsing failure is surfaced
with a clear error message: a failed download is never treated as valid data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests

# Official StatsBomb Open Data layout (raw file exports from the StatsBomb API).
BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

# Repository paths
COMPETITIONS_PATH = "competitions.json"
MATCHES_TEMPLATE = "matches/{competition_id}/{season_id}.json"
EVENTS_TEMPLATE = "events/{match_id}.json"
LINEUPS_TEMPLATE = "lineups/{match_id}.json"

# StatsBomb pitch dimensions (metres): x = 0..120, y = 0..80.
# In this coordinate system the attacking goal is at x = 120, so teams in the
# event data attack left-to-right (increasing x).
PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0


class DataSourceError(RuntimeError):
    """Raised when the official data source cannot be reached or parsed."""


def _cache_path(cache_dir: Path, relative_path: str) -> Path:
    return cache_dir / relative_path


def fetch_json(url: str, cache_path: Path | None = None, force_download: bool = False) -> Any:
    """Download a JSON resource, optionally caching it to ``cache_path``.

    Parameters
    ----------
    url:
        Full URL to fetch.
    cache_path:
        If given, the response is written here after a successful fetch, and
        read from here on subsequent calls unless ``force_download=True``.
    force_download:
        Re-download even if a cached copy exists.

    Raises
    ------
    DataSourceError
        On network failure, non-200 responses, invalid JSON, or empty payloads.
    """
    if cache_path is not None and cache_path.exists() and not force_download:
        try:
            with cache_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            if data is not None and data != []:
                return data
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            # Corrupt cache: fall through and re-download.
            pass

    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise DataSourceError(f"Network error fetching {url}: {exc}") from exc

    try:
        data = response.json()
    except ValueError as exc:
        raise DataSourceError(f"Invalid JSON from {url}: {exc}") from exc

    if data is None or data == []:
        raise DataSourceError(f"Empty dataset returned from {url}")

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh)

    return data


def load_competitions(cache_dir: Path | str, force_download: bool = False) -> list[dict[str, Any]]:
    """Load the competitions index (every competition-season combination)."""
    cache_dir = Path(cache_dir)
    url = f"{BASE_URL}/competitions.json"
    data = fetch_json(url, _cache_path(cache_dir, COMPETITIONS_PATH), force_download)
    if not isinstance(data, list):
        raise DataSourceError("competitions.json did not return a list")
    return data


def find_competition_season(
    competitions: list[dict[str, Any]],
    competition_name: str,
    season_name: str,
) -> dict[str, Any]:
    """Return the competition-season record matching the given names.

    Raises
    ------
    DataSourceError
        If no matching competition-season exists in the index.
    """
    matches = [
        c
        for c in competitions
        if c.get("competition_name") == competition_name
        and c.get("season_name") == season_name
    ]
    if not matches:
        available = sorted(
            {f"{c.get('competition_name')} / {c.get('season_name')}" for c in competitions}
        )
        raise DataSourceError(
            f"Competition-season '{competition_name} / {season_name}' not found in "
            f"the competitions index. Available (sample): {available[:10]} ... "
            f"({len(available)} distinct in total)"
        )
    if len(matches) > 1:
        raise DataSourceError(
            f"Multiple records found for '{competition_name} / {season_name}': {matches}"
        )
    return matches[0]


def load_matches(
    competition_id: int | str,
    season_id: int | str,
    cache_dir: Path | str,
    force_download: bool = False,
) -> list[dict[str, Any]]:
    """Load match metadata for a competition-season."""
    cache_dir = Path(cache_dir)
    relative = MATCHES_TEMPLATE.format(competition_id=competition_id, season_id=season_id)
    url = f"{BASE_URL}/{relative}"
    data = fetch_json(url, _cache_path(cache_dir, relative), force_download)
    if not isinstance(data, list):
        raise DataSourceError(f"Matches file {url} did not return a list")
    return data


def load_match_events(
    match_id: int | str,
    cache_dir: Path | str,
    force_download: bool = False,
) -> list[dict[str, Any]]:
    """Load the event stream for a single match."""
    cache_dir = Path(cache_dir)
    relative = EVENTS_TEMPLATE.format(match_id=match_id)
    url = f"{BASE_URL}/{relative}"
    data = fetch_json(url, _cache_path(cache_dir, relative), force_download)
    if not isinstance(data, list):
        raise DataSourceError(f"Events file {url} did not return a list")
    return data


def load_season_events(
    match_ids: list[int | str],
    cache_dir: Path | str,
    force_download: bool = False,
    verbose: bool = True,
) -> dict[int, list[dict[str, Any]]]:
    """Load event streams for every match in ``match_ids``.

    Each match is fetched independently: a failure for one match is reported as
    a warning and skipped — it is never silently replaced with empty or guessed
    data. Returns a dict mapping match ID -> list of event dicts for matches
    that loaded successfully.
    """
    import warnings

    cache_dir = Path(cache_dir)
    events_by_match: dict[int, list[dict[str, Any]]] = {}
    failed: list[tuple[int, str]] = []

    for match_id in match_ids:
        key = int(match_id)
        try:
            events = load_match_events(key, cache_dir, force_download=force_download)
            events_by_match[key] = events
            if verbose:
                print(f"  loaded match {key}: {len(events)} events")
        except DataSourceError as exc:
            failed.append((key, str(exc)))
            print(f"  WARNING: match {key} could not be loaded: {exc}")

    if failed:
        warnings.warn(
            f"{len(failed)} of {len(match_ids)} matches could not be loaded: "
            f"{[fid for fid, _ in failed[:5]]}..."
        )

    if not events_by_match:
        raise DataSourceError("No match events were loaded successfully")

    return events_by_match
