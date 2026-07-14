#!/usr/bin/env python3
"""Independent, offline audit of the W1 historical market probe.

This script deliberately does not import probe_opening_markets.py. It rebuilds
the billing and coverage results from the saved raw responses, then joins the
sampled games to the existing NHL play-by-play and boxscore archives to test
player/team matching and listed-skater shot coverage.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_PROBE_DIR = Path("data/raw/betting_lines/probes/w1_market_coverage")
DEFAULT_PBP_DIR = Path("data/raw/play_by_play")
DEFAULT_BOXSCORE_DIR = Path("data/raw/boxscores")
STANDARD_SOG = "player_shots_on_goal"
MARKETS = (
    "player_total_saves",
    "player_total_saves_alternate",
    STANDARD_SOG,
    "player_shots_on_goal_alternate",
)
DFS_BOOKS = {"underdog", "prizepicks"}
TEAM_NAME_TO_ABBREV = {
    "Anaheim Ducks": "ANA",
    "Boston Bruins": "BOS",
    "Buffalo Sabres": "BUF",
    "Calgary Flames": "CGY",
    "Carolina Hurricanes": "CAR",
    "Chicago Blackhawks": "CHI",
    "Colorado Avalanche": "COL",
    "Columbus Blue Jackets": "CBJ",
    "Dallas Stars": "DAL",
    "Detroit Red Wings": "DET",
    "Edmonton Oilers": "EDM",
    "Florida Panthers": "FLA",
    "Los Angeles Kings": "LAK",
    "Minnesota Wild": "MIN",
    "Montreal Canadiens": "MTL",
    "Montréal Canadiens": "MTL",
    "Nashville Predators": "NSH",
    "New Jersey Devils": "NJD",
    "New York Islanders": "NYI",
    "New York Rangers": "NYR",
    "Ottawa Senators": "OTT",
    "Philadelphia Flyers": "PHI",
    "Pittsburgh Penguins": "PIT",
    "San Jose Sharks": "SJS",
    "Seattle Kraken": "SEA",
    "St Louis Blues": "STL",
    "St. Louis Blues": "STL",
    "Tampa Bay Lightning": "TBL",
    "Toronto Maple Leafs": "TOR",
    "Utah Hockey Club": "UTA",
    "Utah Mammoth": "UTA",
    "Vancouver Canucks": "VAN",
    "Vegas Golden Knights": "VGK",
    "Washington Capitals": "WSH",
    "Winnipeg Jets": "WPG",
}
SEASON_TO_NHL_CODE = {
    "2023-24": "20232024",
    "2024-25": "20242025",
    "2025-26": "20252026",
}


def normalize_name(value: str) -> str:
    value = value.split("(", 1)[0].strip().replace("'", "").replace("’", "")
    folded = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    cleaned = "".join(char.lower() if char.isalnum() else " " for char in folded)
    parts = [part for part in cleaned.split() if part not in {"jr", "sr", "ii", "iii", "iv"}]
    return " ".join(parts)


def person_key(value: str) -> tuple[str, str]:
    parts = normalize_name(value).split()
    return ((parts[0][0] if parts else ""), (parts[-1] if parts else ""))


def load_records(probe_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(probe_dir.glob("w1_event=*.json")):
        with open(path, "r", encoding="utf-8") as handle:
            record = json.load(handle)
        record["_path"] = str(path)
        record["_body"] = json.loads(record["raw_body"])
        records.append(record)
    if len(records) != 24:
        raise ValueError(f"expected 24 probe records, found {len(records)}")
    if len({record["signature"] for record in records}) != len(records):
        raise ValueError("duplicate request signatures in probe records")
    return records


def market_units(market: dict[str, Any]) -> tuple[set[tuple[str, Any]], set[tuple[str, Any]]]:
    sides: dict[tuple[str, Any], set[str]] = defaultdict(set)
    for outcome in market.get("outcomes") or []:
        player = outcome.get("description")
        if player:
            sides[(str(player), outcome.get("point"))].add(str(outcome.get("name") or "").lower())
    units = set(sides)
    complete = {unit for unit, names in sides.items() if {"over", "under"}.issubset(names)}
    return units, complete


def audit_billing(records: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for record in records:
        body = record["_body"]
        returned_markets = {
            market["key"]
            for book in (body.get("data") or {}).get("bookmakers") or []
            for market in book.get("markets") or []
        }
        headers = record["quota_headers"]
        last = int(headers["x-requests-last"])
        rows.append(
            {
                "season": record["event"]["season"],
                "used": int(headers["x-requests-used"]),
                "remaining": int(headers["x-requests-remaining"]),
                "last": last,
                "returned_markets": len(returned_markets),
                "named_book_grouping_matches": last == 10 * len(returned_markets),
                "status_code": record["status_code"],
            }
        )
    ordered = sorted(rows, key=lambda row: row["used"])
    increments_match = all(
        current["used"] - previous["used"] == current["last"]
        for previous, current in zip(ordered, ordered[1:])
    )
    by_season = []
    for season in sorted({row["season"] for row in rows}):
        group = [row for row in rows if row["season"] == season]
        by_season.append(
            {
                "season": season,
                "calls": len(group),
                "credits": sum(row["last"] for row in group),
                "per_call_costs": sorted({row["last"] for row in group}),
            }
        )
    return {
        "calls": len(rows),
        "all_http_200": all(row["status_code"] == 200 for row in rows),
        "actual_credits": sum(row["last"] for row in rows),
        "first_used_after_call": ordered[0]["used"],
        "last_used": ordered[-1]["used"],
        "last_remaining": ordered[-1]["remaining"],
        "quota_total": ordered[-1]["used"] + ordered[-1]["remaining"],
        "usage_increments_match_headers": increments_match,
        "named_book_grouping_matches_all_calls": all(
            row["named_book_grouping_matches"] for row in rows
        ),
        "by_season": by_season,
    }


def event_markets(record: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    output: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    data = record["_body"].get("data") or {}
    for book in data.get("bookmakers") or []:
        for market in book.get("markets") or []:
            if market.get("key") in MARKETS:
                output[book["key"]][market["key"]] = market
    return dict(output)


def audit_market_coverage(records: list[dict[str, Any]]) -> dict[str, Any]:
    coverage: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "events": set(),
            "outcomes": 0,
            "duplicate_outcomes": 0,
            "multipliers": 0,
            "units": 0,
            "complete": 0,
            "player_events": 0,
            "complete_player_events": 0,
        }
    )
    sog_multi_book_events: dict[str, int] = defaultdict(int)
    for record in records:
        season = record["event"]["season"]
        markets_by_book = event_markets(record)
        usable_sog_books = 0
        for book, markets in markets_by_book.items():
            for market_key, market in markets.items():
                key = (season, book, market_key)
                outcomes = market.get("outcomes") or []
                units, complete = market_units(market)
                outcome_keys = {
                    (outcome.get("description"), outcome.get("point"), outcome.get("name"))
                    for outcome in outcomes
                }
                coverage[key]["events"].add(record["event"]["event_id"])
                coverage[key]["outcomes"] += len(outcomes)
                coverage[key]["duplicate_outcomes"] += len(outcomes) - len(outcome_keys)
                coverage[key]["multipliers"] += sum(
                    outcome.get("multiplier") is not None for outcome in outcomes
                )
                coverage[key]["units"] += len(units)
                coverage[key]["complete"] += len(complete)
                coverage[key]["player_events"] += len({player for player, _ in units})
                coverage[key]["complete_player_events"] += len({player for player, _ in complete})
                if market_key == STANDARD_SOG and complete:
                    usable_sog_books += 1
        if usable_sog_books >= 2:
            sog_multi_book_events[season] += 1

    rows = []
    for (season, book, market), values in sorted(coverage.items()):
        rows.append(
            {
                "season": season,
                "bookmaker": book,
                "market": market,
                "events": len(values["events"]),
                "outcomes": values["outcomes"],
                "duplicate_outcomes": values["duplicate_outcomes"],
                "multiplier_outcomes": values["multipliers"],
                "player_line_units": values["units"],
                "complete_over_under_units": values["complete"],
                "both_side_rate": (
                    round(values["complete"] / values["units"], 6) if values["units"] else None
                ),
                "player_events": values["player_events"],
                "player_events_with_complete_line": values["complete_player_events"],
                "player_event_both_side_rate": (
                    round(values["complete_player_events"] / values["player_events"], 6)
                    if values["player_events"]
                    else None
                ),
            }
        )
    seasons = sorted({record["event"]["season"] for record in records})
    sog_player_event_gate = []
    for season in seasons:
        season_rows = [
            row for row in rows if row["season"] == season and row["market"] == STANDARD_SOG
        ]
        player_events = sum(row["player_events"] for row in season_rows)
        complete_player_events = sum(row["player_events_with_complete_line"] for row in season_rows)
        sog_player_event_gate.append(
            {
                "season": season,
                "player_events": player_events,
                "player_events_with_complete_line": complete_player_events,
                "rate": round(complete_player_events / player_events, 6) if player_events else None,
                "passes_95_percent": (
                    complete_player_events / player_events >= 0.95 if player_events else None
                ),
            }
        )
    return {
        "rows": rows,
        "sog_multiple_book_gate": [
            {
                "season": season,
                "events_with_at_least_two_usable_books": sog_multi_book_events[season],
                "sampled_events": sum(record["event"]["season"] == season for record in records),
                "rate": round(
                    sog_multi_book_events[season]
                    / sum(record["event"]["season"] == season for record in records),
                    6,
                ),
            }
            for season in seasons
        ],
        "sog_player_event_both_side_gate": sog_player_event_gate,
        "dfs_rows": [row for row in rows if row["bookmaker"] in DFS_BOOKS],
        "alternate_rows": [row for row in rows if row["market"].endswith("_alternate")],
        "multiplier_rows": [row for row in rows if row["multiplier_outcomes"] > 0],
    }


def event_key(record: dict[str, Any]) -> tuple[str, frozenset[str]]:
    event = record["event"]
    date_value = event["bettime_ts"][:10]
    teams = frozenset(
        (TEAM_NAME_TO_ABBREV[event["home_team"]], TEAM_NAME_TO_ABBREV[event["away_team"]])
    )
    return date_value, teams


def load_game_index(
    pbp_dir: Path,
    targets: set[tuple[str, frozenset[str]]],
) -> tuple[dict[tuple[str, frozenset[str]], Path], dict[tuple[str, str], set[str]]]:
    target_dates = {target[0] for target in targets}
    index = {}
    season_team_players: dict[tuple[str, str], set[str]] = defaultdict(set)
    season_by_code = {code: season for season, code in SEASON_TO_NHL_CODE.items()}
    for path in sorted(pbp_dir.glob("*.json")):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                game = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        season = season_by_code.get(str(game.get("season")))
        if season:
            team_by_id = {
                game.get("homeTeam", {}).get("id"): game.get("homeTeam", {}).get("abbrev"),
                game.get("awayTeam", {}).get("id"): game.get("awayTeam", {}).get("abbrev"),
            }
            for player in game.get("rosterSpots") or []:
                team = team_by_id.get(player.get("teamId"))
                if not team:
                    continue
                name = (
                    f"{player.get('firstName', {}).get('default', '')} "
                    f"{player.get('lastName', {}).get('default', '')}"
                )
                season_team_players[(season, team)].add(normalize_name(name))

        game_date = game.get("gameDate")
        if game_date not in target_dates:
            continue
        key = (game_date, frozenset((game.get("homeTeam", {}).get("abbrev"), game.get("awayTeam", {}).get("abbrev"))))
        if key in targets:
            if key in index:
                raise ValueError(f"multiple NHL games matched target {key}")
            index[key] = path
    return index, season_team_players


def roster_and_sog(pbp_path: Path, boxscore_dir: Path) -> tuple[dict[str, tuple[int, str]], dict[int, int], int]:
    with open(pbp_path, "r", encoding="utf-8") as handle:
        pbp = json.load(handle)
    team_by_id = {
        pbp["homeTeam"]["id"]: pbp["homeTeam"]["abbrev"],
        pbp["awayTeam"]["id"]: pbp["awayTeam"]["abbrev"],
    }
    roster = {}
    for player in pbp.get("rosterSpots") or []:
        name = f"{player.get('firstName', {}).get('default', '')} {player.get('lastName', {}).get('default', '')}"
        roster[normalize_name(name)] = (int(player["playerId"]), team_by_id[player["teamId"]])

    boxscore_path = boxscore_dir / pbp_path.name
    with open(boxscore_path, "r", encoding="utf-8") as handle:
        boxscore = json.load(handle)
    sog_by_id = {}
    for side in ("homeTeam", "awayTeam"):
        stats = (boxscore.get("playerByGameStats") or {}).get(side) or {}
        for group in ("forwards", "defense"):
            for player in stats.get(group) or []:
                sog_by_id[int(player["playerId"])] = int(player.get("sog") or 0)
    return roster, sog_by_id, sum(sog_by_id.values())


def audit_player_matching(records: list[dict[str, Any]], pbp_dir: Path, boxscore_dir: Path) -> dict[str, Any]:
    targets = {event_key(record) for record in records}
    game_index, season_team_players = load_game_index(pbp_dir, targets)
    if set(game_index) != targets:
        missing = sorted(str(key) for key in targets - set(game_index))
        raise ValueError(f"missing NHL game matches: {missing}")

    unique_players = set()
    matched_players = set()
    season_roster_matched_players = set()
    ambiguous_season_matches = set()
    player_details = {}
    book_event_rows = []
    for record in records:
        key = event_key(record)
        roster, sog_by_id, total_sog = roster_and_sog(game_index[key], boxscore_dir)
        for book, markets in event_markets(record).items():
            market = markets.get(STANDARD_SOG)
            if not market:
                continue
            units, _ = market_units(market)
            player_names = {player for player, _ in units}
            matched_ids = set()
            for player in player_names:
                unit = (record["event"]["event_id"], normalize_name(player))
                unique_players.add(unit)
                event_teams = {
                    TEAM_NAME_TO_ABBREV[record["event"]["home_team"]],
                    TEAM_NAME_TO_ABBREV[record["event"]["away_team"]],
                }
                season_candidates = sorted(
                    team
                    for team in event_teams
                    if (
                        unit[1] in season_team_players[(record["event"]["season"], team)]
                        or person_key(player)
                        in {
                            person_key(roster_name)
                            for roster_name in season_team_players[(record["event"]["season"], team)]
                        }
                    )
                )
                if len(season_candidates) == 1:
                    season_roster_matched_players.add(unit)
                elif len(season_candidates) > 1:
                    ambiguous_season_matches.add(unit)
                player_details[unit] = {
                    "event_id": record["event"]["event_id"],
                    "season": record["event"]["season"],
                    "player": player,
                    "event_teams": sorted(event_teams),
                    "season_team_candidates": season_candidates,
                }
                roster_hit = roster.get(unit[1])
                if roster_hit is None:
                    roster_candidates = {
                        roster_value
                        for roster_name, roster_value in roster.items()
                        if person_key(roster_name) == person_key(player)
                    }
                    if len(roster_candidates) == 1:
                        roster_hit = next(iter(roster_candidates))
                if roster_hit:
                    matched_players.add(unit)
                    matched_ids.add(roster_hit[0])
            book_event_rows.append(
                {
                    "season": record["event"]["season"],
                    "bookmaker": book,
                    "event_id": record["event"]["event_id"],
                    "listed_players": len(player_names),
                    "matched_players": len(matched_ids),
                    "actual_sog_coverage": (
                        sum(sog_by_id.get(player_id, 0) for player_id in matched_ids) / total_sog
                        if total_sog
                        else None
                    ),
                }
            )

    coverage_summary = []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in book_event_rows:
        groups[(row["season"], row["bookmaker"])].append(row)
    for (season, book), rows in sorted(groups.items()):
        shares = [row["actual_sog_coverage"] for row in rows if row["actual_sog_coverage"] is not None]
        coverage_summary.append(
            {
                "season": season,
                "bookmaker": book,
                "events": len(rows),
                "median_listed_players": statistics.median(row["listed_players"] for row in rows),
                "median_actual_team_sog_share": round(statistics.median(shares), 6) if shares else None,
                "min_actual_team_sog_share": round(min(shares), 6) if shares else None,
                "max_actual_team_sog_share": round(max(shares), 6) if shares else None,
            }
        )
    return {
        "nhl_games_matched": len(game_index),
        "sampled_games": len(records),
        "unique_event_players": len(unique_players),
        "matched_event_players": len(matched_players),
        "actual_game_roster_match_rate": (
            round(len(matched_players) / len(unique_players), 6) if unique_players else None
        ),
        "season_team_matched_event_players": len(season_roster_matched_players),
        "season_team_match_rate": (
            round(len(season_roster_matched_players) / len(unique_players), 6)
            if unique_players
            else None
        ),
        "ambiguous_season_team_matches": len(ambiguous_season_matches),
        "unmatched_on_actual_game_roster": [
            player_details[unit] for unit in sorted(unique_players - matched_players)
        ],
        "unmatched_on_season_team_rosters": [
            player_details[unit]
            for unit in sorted(unique_players - season_roster_matched_players - ambiguous_season_matches)
        ],
        "coverage_by_season_book": coverage_summary,
    }


def atomic_create(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite audit output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.parent / f".{path.name}.{os.getpid()}.tmp"
    with open(temp, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temp, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline independent audit of the W1 probe.")
    parser.add_argument("--probe-dir", type=Path, default=DEFAULT_PROBE_DIR)
    parser.add_argument("--pbp-dir", type=Path, default=DEFAULT_PBP_DIR)
    parser.add_argument("--boxscore-dir", type=Path, default=DEFAULT_BOXSCORE_DIR)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    records = load_records(args.probe_dir)
    audit = {
        "audit": "w1_probe_independent_raw_rebuild",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "billing": audit_billing(records),
        "market_coverage": audit_market_coverage(records),
        "player_matching": audit_player_matching(records, args.pbp_dir, args.boxscore_dir),
    }
    if args.output:
        atomic_create(args.output, audit)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
